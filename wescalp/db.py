"""Gravação no Postgres/TimescaleDB. Tudo é idempotente (ON CONFLICT DO NOTHING)."""

from datetime import datetime
from pathlib import Path

import psycopg

from .kucoin import Book, Candle, Trade

SQL_DIR = Path(__file__).resolve().parent.parent / "db"


async def connect(url: str) -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(url, autocommit=True)


async def apply_schema(conn: psycopg.AsyncConnection) -> None:
    await conn.execute((SQL_DIR / "schema.sql").read_text())
    cur = await conn.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
    if await cur.fetchone():
        await conn.execute((SQL_DIR / "timescale.sql").read_text())


async def insert_candles(conn: psycopg.AsyncConnection, candles: list[Candle]) -> None:
    if not candles:
        return
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO candles (source, symbol, ts, open, high, low, close, volume, turnover)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [(c.source, c.symbol, c.ts, c.open, c.high, c.low, c.close, c.volume, c.turnover) for c in candles],
        )


async def candle_span(
    conn: psycopg.AsyncConnection, source: str, symbol: str
) -> tuple[datetime | None, datetime | None]:
    cur = await conn.execute("SELECT min(ts), max(ts) FROM candles WHERE source = %s AND symbol = %s", (source, symbol))
    return await cur.fetchone()


async def insert_trades(conn: psycopg.AsyncConnection, trades: list[Trade]) -> None:
    if not trades:
        return
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO trades (symbol, ts, trade_id, side, price, size)"
            " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [(t.symbol, t.ts, t.trade_id, t.side, t.price, t.size) for t in trades],
        )


def l1_row(b: Book) -> tuple:
    (bid_px, bid_sz), (ask_px, ask_sz) = b.bids[0], b.asks[0]
    return (
        b.symbol, b.ts, b.seq, bid_px, bid_sz, ask_px, ask_sz,
        sum(s for _, s in b.bids[:5]), sum(s for _, s in b.asks[:5]),
    )


async def insert_book_l1(conn: psycopg.AsyncConnection, books: list[Book]) -> None:
    rows = [l1_row(b) for b in books if b.bids and b.asks]
    if not rows:
        return
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO book_l1 (symbol, ts, seq, bid_px, bid_sz, ask_px, ask_sz, bid_sz5, ask_sz5)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            rows,
        )


async def insert_book_snapshots(conn: psycopg.AsyncConnection, books: list[Book]) -> None:
    if not books:
        return
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO book_snapshots (symbol, ts, seq, bid_px, bid_sz, ask_px, ask_sz)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            [
                (b.symbol, b.ts, b.seq,
                 [p for p, _ in b.bids], [s for _, s in b.bids],
                 [p for p, _ in b.asks], [s for _, s in b.asks])
                for b in books
            ],
        )
