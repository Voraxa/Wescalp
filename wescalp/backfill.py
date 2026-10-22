"""Baixa o histórico de candles de 5 min (spot e futuros) para o banco.

    python -m wescalp.backfill --weeks 100

Retoma de onde parou: só baixa o que falta antes do primeiro e depois do último candle gravado de cada
fonte, a menos que --full seja passado.
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


async def backfill_source(conn, rest: KucoinRest, source: str, weeks: int, full: bool) -> int:
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
            ranges = [(start, int(first.timestamp())), (int(last.timestamp()) + CANDLE_SECONDS, now)]

    total = empty = 0
    for a, b in (w for lo, hi in ranges for w in windows(lo, hi, per_call * CANDLE_SECONDS)):
        candles = await fetch(symbol, a, b)
        await db.insert_candles(conn, candles)
        total += len(candles)
        empty += not candles
        if total and total % 20000 < len(candles):
            log.info("%s %s: %d candles até %s", source, symbol, total, time.strftime("%Y-%m-%d", time.gmtime(b)))
    log.info("%s %s: %d candles gravados (%d janelas vazias)", source, symbol, total, empty)
    return total


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weeks", type=int, default=100)
    p.add_argument("--source", choices=["spot", "futures", "all"], default="all")
    p.add_argument("--full", action="store_true", help="ignora o que já está no banco e baixa a janela inteira")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    conn = await db.connect(DATABASE_URL)
    await db.apply_schema(conn)
    rest = KucoinRest()
    try:
        for source in (["spot", "futures"] if args.source == "all" else [args.source]):
            await backfill_source(conn, rest, source, args.weeks, args.full)
    finally:
        await rest.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
