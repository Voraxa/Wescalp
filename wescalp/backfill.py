"""Baixa o histórico de candles de 5 min (spot e futuros) para o banco.

    python -m wescalp.backfill --weeks 100
    python -m wescalp.backfill --weeks 100 --follow   # depois continua buscando os candles novos

Retoma de onde parou: só baixa o que falta antes do primeiro e depois do último candle gravado de cada
fonte, a menos que --full seja passado. Com --follow fica rodando e a cada --every segundos grava os
candles que fecharam desde a última volta; é o que mantém o gráfico do painel em dia.
"""

import argparse
import asyncio
import logging
import time

from . import db
from .config import CANDLE_SECONDS, DATABASE_URL, FUTURES_SYMBOL, SPOT_SYMBOL
from .kucoin import FUTURES_MAX_CANDLES, SPOT_MAX_CANDLES, KucoinRest

log = logging.getLogger("wescalp.backfill")


def windows(start: int, end: int, size: int):
    """Janelas [a, b) de `size` segundos cobrindo [start, end)."""
    a = start
    while a < end:
        b = min(a + size, end)
        yield a, b
        a = b


def missing_ranges(start: int, now: int, first: int | None, last: int | None) -> list[tuple[int, int]]:
    """Intervalos [a, b) ainda sem candle, dado o primeiro e o último candle já gravados (abertura, em s)."""
    if first is None:
        return [(start, now)]
    return [(a, b) for a, b in ((start, first), (last + CANDLE_SECONDS, now)) if a < b]


async def backfill_source(conn, rest: KucoinRest, source: str, weeks: int, full: bool, quiet: bool = False) -> int:
    symbol, fetch, per_call = {
        "spot": (SPOT_SYMBOL, rest.spot_candles, SPOT_MAX_CANDLES),
        "futures": (FUTURES_SYMBOL, rest.futures_candles, FUTURES_MAX_CANDLES),
    }[source]

    now = int(time.time()) // CANDLE_SECONDS * CANDLE_SECONDS  # só candles fechados
    start = now - weeks * 7 * 24 * 3600
    ranges = [(start, now)]
    if not full:
        first, last = await db.candle_span(conn, source, symbol)
        if first:
            ranges = missing_ranges(start, now, int(first.timestamp()), int(last.timestamp()))

    total = empty = 0
    for a, b in (w for lo, hi in ranges for w in windows(lo, hi, per_call * CANDLE_SECONDS)):
        candles = await fetch(symbol, a, b)
        await db.insert_candles(conn, candles)
        total += len(candles)
        empty += not candles
        if total and total % 20000 < len(candles):
            log.info("%s %s: %d candles até %s", source, symbol, total, time.strftime("%Y-%m-%d", time.gmtime(b)))
    if total or not quiet:
        log.info("%s %s: %d candles gravados (%d janelas vazias)", source, symbol, total, empty)
    return total


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weeks", type=int, default=100)
    p.add_argument("--source", choices=["spot", "futures", "all"], default="all")
    p.add_argument("--full", action="store_true", help="ignora o que já está no banco e baixa a janela inteira")
    p.add_argument("--follow", action="store_true", help="depois do histórico, continua buscando os candles novos")
    p.add_argument("--every", type=int, default=60, help="segundos entre buscas no modo --follow")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    sources = ["spot", "futures"] if args.source == "all" else [args.source]
    conn = await db.connect(DATABASE_URL)
    await db.apply_schema(conn)
    rest = KucoinRest()
    try:
        for source in sources:
            await backfill_source(conn, rest, source, args.weeks, args.full)
        while args.follow:
            await asyncio.sleep(args.every)
            try:
                for source in sources:
                    await backfill_source(conn, rest, source, args.weeks, False, quiet=True)
            except Exception:
                # Rede ou banco fora: tenta de novo na próxima volta, com conexão nova.
                log.exception("falha ao buscar candles novos")
                await conn.close()
                try:
                    conn = await db.connect(DATABASE_URL)
                except Exception:
                    log.exception("banco indisponível")
    finally:
        await rest.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
