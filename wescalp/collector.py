"""Coletor ao vivo do perpétuo BTC na KuCoin: livro de ordens (50 níveis) e negócios.

    python -m wescalp.collector

A KuCoin não guarda histórico do livro, então este processo precisa rodar sem parar desde o primeiro dia.
Reconecta sozinho; se ficar 30 s sem mensagem, derruba a conexão e abre outra.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

import websockets

from . import db
from .config import BOOK_SNAPSHOT_MS, DATABASE_URL, FUTURES_SYMBOL, HEARTBEAT_FILE
from .kucoin import Book, KucoinRest, Trade, parse_depth, parse_execution

log = logging.getLogger("wescalp.collector")

STALE_SECONDS = 30
FLUSH_SECONDS = 1.0


class Buffer:
    def __init__(self):
        self.trades: list[Trade] = []
        self.l1: list[Book] = []
        self.snapshots: list[Book] = []
        self.last_snapshot_ms = 0

    def add_book(self, book: Book, now_ms: int) -> None:
        self.l1.append(book)
        if now_ms - self.last_snapshot_ms >= BOOK_SNAPSHOT_MS:
            self.snapshots.append(book)
            self.last_snapshot_ms = now_ms

    def take(self):
        out = (self.trades, self.l1, self.snapshots)
        self.trades, self.l1, self.snapshots = [], [], []
        return out


async def writer(buf: Buffer, stop: asyncio.Event) -> None:
    conn = await db.connect(DATABASE_URL)
    await db.apply_schema(conn)
    counts = [0, 0, 0]
    last_log = time.monotonic()
    try:
        while not stop.is_set():
            await asyncio.sleep(FLUSH_SECONDS)
            trades, l1, snaps = buf.take()
            try:
                await db.insert_trades(conn, trades)
                await db.insert_book_l1(conn, l1)
                await db.insert_book_snapshots(conn, snaps)
            except Exception:
                log.exception("falha ao gravar; reconectando ao banco")
                # Devolve o lote para a próxima tentativa.
                buf.trades[:0], buf.l1[:0], buf.snapshots[:0] = trades, l1, snaps
                await conn.close()
                await asyncio.sleep(2)
                conn = await db.connect(DATABASE_URL)
                continue
            counts = [counts[0] + len(trades), counts[1] + len(l1), counts[2] + len(snaps)]
            if time.monotonic() - last_log >= 60:
                log.info("último minuto: %d negócios, %d topos de livro, %d fotos", *counts)
                counts, last_log = [0, 0, 0], time.monotonic()
    finally:
        await conn.close()


def touch_heartbeat() -> None:
    Path(HEARTBEAT_FILE).touch()


async def stream(buf: Buffer, rest: KucoinRest) -> None:
    url, ping_ms = await rest.futures_ws_endpoint()
    url += f"&connectId={uuid.uuid4().hex}"
    topics = [f"/contractMarket/level2Depth50:{FUTURES_SYMBOL}", f"/contractMarket/execution:{FUTURES_SYMBOL}"]

    async with websockets.connect(url, ping_interval=None, max_size=2**22) as ws:
        welcome = json.loads(await ws.recv())
        if welcome.get("type") != "welcome":
            raise RuntimeError(f"esperava welcome, veio {welcome}")
        for topic in topics:
            await ws.send(json.dumps({"id": uuid.uuid4().hex, "type": "subscribe", "topic": topic, "response": True}))
        log.info("conectado e inscrito em %s", ", ".join(topics))

        async def pinger():
            while True:
                await asyncio.sleep(ping_ms / 1000 * 0.8)
                await ws.send(json.dumps({"id": uuid.uuid4().hex, "type": "ping"}))

        ping_task = asyncio.create_task(pinger())
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=STALE_SECONDS)
                msg = json.loads(raw)
                if msg.get("type") != "message":
                    continue
                now_ms = int(time.time() * 1000)
                subject = msg.get("subject")
                if subject == "level2":
                    buf.add_book(parse_depth(FUTURES_SYMBOL, msg["data"], now_ms), now_ms)
                elif subject == "match":
                    buf.trades.append(parse_execution(msg["data"]))
                touch_heartbeat()
        finally:
            ping_task.cancel()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    buf, stop = Buffer(), asyncio.Event()
    write_task = asyncio.create_task(writer(buf, stop))
    rest = KucoinRest()
    backoff = 1
    try:
        while True:
            started = time.monotonic()
            try:
                await stream(buf, rest)
            except Exception as e:
                log.warning("conexão caiu: %r", e)
            if write_task.done():
                write_task.result()  # propaga erro fatal do gravador
            # Conexão que durou mais de 1 min zera o backoff.
            backoff = 1 if time.monotonic() - started > 60 else min(backoff * 2, 60)
            log.info("reconectando em %ss", backoff)
            await asyncio.sleep(backoff)
    finally:
        stop.set()
        await rest.aclose()
        await write_task


if __name__ == "__main__":
    asyncio.run(main())
