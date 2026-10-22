"""Painel web: candles do perpétuo com VWAP e EMAs, marcas dos setups, buracos do livro, fluxo e estado do coletor.

    python -m wescalp.web        # 0.0.0.0:WESCALP_WEB_PORT (padrão 8090)

Só lê o banco (sessão read-only, tempo máximo por consulta) e não fala com a KuCoin. Sem login aqui: o
basic_auth fica no Caddy. Toda consulta tem janela limitada (candles, fluxo, buracos); o estado do coletor lê
as 24 h uma vez e depois só reconta os minutos novos.

A camada de banco é a classe Store; o resto são funções puras que montam o JSON (testadas com um Store falso).
"""

import asyncio
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import db, features, health, research
from .config import CANDLE_SECONDS, DATABASE_URL, FUTURES_SYMBOL, SPOT_SYMBOL
from .flow import Quote, Tick, aggregate
from .kucoin import Candle
from .labels import Barriers, Label, label_candles
from .setups import SETUPS

log = logging.getLogger("wescalp.web")

PORT = int(os.environ.get("WESCALP_WEB_PORT", "8090"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
UTC = timezone.utc
STEP = timedelta(seconds=CANDLE_SECONDS)
SOURCES = {"futures": FUTURES_SYMBOL, "spot": SPOT_SYMBOL}

MAX_CANDLES = 3000  # por requisição de candles ou de marcas
DEFAULT_CANDLES = 600  # ~2 dias
HOLDOUT_WEEKS = 16  # mesmo padrão de wescalp.research
LIVE_MAX = timedelta(hours=24)  # candles montados de negócios: só nas últimas 24 h
CONTRACT_BTC = 0.001
FLOW_MAX = timedelta(hours=6)
FLOW_MAX_BUCKETS = 10800
FLOW_BUCKETS = (1, 2, 5, 10, 15, 30, 60, 120, 300)
GAPS_MAX_HOURS = 72
HEALTH_FULL_EVERY_S = 600  # relê as 24 h inteiras de vez em quando; no meio, só os minutos novos
SIZES_EVERY_S = 60
OLD_GAPS_TTL_S = 600


class BadRequest(ValueError):
    pass


# --- funções puras ---

def parse_time(raw: str | None) -> datetime | None:
    """Época em segundos ou ISO 8601 (UTC se sem fuso). Vazio = None."""
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    try:
        return datetime.fromtimestamp(float(raw), tz=UTC)
    except ValueError:
        pass
    try:
        t = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise BadRequest(f"data inválida: {raw!r}") from None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def floor_candle(t: datetime) -> datetime:
    return datetime.fromtimestamp(int(t.timestamp()) // CANDLE_SECONDS * CANDLE_SECONDS, tz=UTC)


def warmup_start(start: datetime) -> datetime:
    """Início do dia UTC anterior a `start`: VWAP do dia, máxima e mínima de ontem e EMAs já assentadas."""
    day = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return day - timedelta(days=1)


def clip_window(start: datetime, end: datetime, limit: timedelta) -> tuple[datetime, bool]:
    """Encurta a janela pelo começo se passar do limite; devolve o novo início e se cortou."""
    if end - start > limit:
        return end - limit, True
    return start, False


def epoch(t: datetime | None) -> int | None:
    return None if t is None else int(t.timestamp())


def num(x: float | None, nd: int) -> float | None:
    """Arredonda; None, NaN e infinito viram null no JSON."""
    if x is None or not math.isfinite(x):
        return None
    return round(x, nd)


def trade_candles(rows, start: datetime, end: datetime, symbol: str = FUTURES_SYMBOL) -> list[Candle]:
    """Candles de 5 min montados de negócios (ts em segundos, preço, contratos), só em [start, end).

    Cobre o trecho depois do último candle gravado pelo backfill, inclusive o candle em formação. `rows` em
    ordem de ts. Balde sem negócio fica sem candle.
    """
    lo, hi = start.timestamp(), end.timestamp()
    out: list[Candle] = []
    cur = None  # [balde, abertura, máxima, mínima, fechamento, contratos, USDT]

    def close(b):
        out.append(Candle("futures", symbol, datetime.fromtimestamp(b[0], tz=UTC), *b[1:]))

    for ts, px, size in rows:
        if not lo <= ts < hi:
            continue
        bucket = int(ts // CANDLE_SECONDS) * CANDLE_SECONDS
        if cur is None or bucket != cur[0]:
            if cur is not None:
                close(cur)
            cur = [bucket, px, px, px, px, 0.0, 0.0]
        cur[2], cur[3], cur[4] = max(cur[2], px), min(cur[3], px), px
        cur[5] += size
        cur[6] += px * size * CONTRACT_BTC
    if cur is not None:
        close(cur)
    return out


def candles_payload(candles: list[Candle], start: datetime, end: datetime,
                    live_from: datetime | None, now: datetime) -> dict:
    """Candles em [start, end) com VWAP do dia (± 1 e 2 desvios) e EMAs 9 e 21, colunas alinhadas.

    `candles` traz o aquecimento antes de `start` (o dia UTC anterior inteiro). Os candles a partir de
    `live_from` vieram de negócios; o do balde de `now`, se houver, está em formação.
    """
    cols = features.compute(candles)
    out: dict = {k: [] for k in ("candles", "vwap", "vwap_up1", "vwap_dn1", "vwap_up2", "vwap_dn2", "ema9", "ema21")}
    for i, c in enumerate(candles):
        if not start <= c.ts < end:
            continue
        out["candles"].append([epoch(c.ts), num(c.open, 1), num(c.high, 1), num(c.low, 1), num(c.close, 1),
                               num(c.volume, 3)])
        vwap, sd = cols["vwap"][i], cols["vwap_sd"][i]
        out["vwap"].append(num(vwap, 2))
        for k in (1, 2):
            out[f"vwap_up{k}"].append(num(vwap + k * sd, 2) if vwap is not None else None)
            out[f"vwap_dn{k}"].append(num(vwap - k * sd, 2) if vwap is not None else None)
        out["ema9"].append(num(cols["ema9"][i], 2))
        out["ema21"].append(num(cols["ema21"][i], 2))
    kept = out["candles"]
    live = live_from is not None and bool(kept) and kept[-1][0] >= epoch(live_from)
    out["live_from"] = max(epoch(live_from), kept[0][0]) if live else None
    out["forming"] = kept[-1][0] if live and kept[-1][0] == epoch(floor_candle(now)) else None
    return out


def holdout_start(last: datetime | None, weeks: int = HOLDOUT_WEEKS) -> datetime | None:
    """Início do período intocado: as últimas `weeks` semanas antes do último candle, como em wescalp.research."""
    return last - timedelta(weeks=weeks) if last is not None and weeks else None


def mark_json(setup: str, lb: Label) -> dict:
    return {
        "ts": epoch(lb.ts),  # candle da entrada (abertura); a entrada é no fechamento dele
        "setup": setup,
        "side": lb.side,
        "entry_px": num(lb.entry_px, 1),
        "outcome": lb.outcome,
        "bars": lb.bars,
        "exit_ts": epoch(lb.ts + lb.bars * STEP),  # candle em que saiu (no fechamento dele se outcome = time)
        "exit_px": num(lb.exit_px, 1),
        "atr": num(lb.atr, 2),
        "ret_gross_bps": num(lb.ret_gross * 1e4, 2),
        "ret_net_bps": num(lb.ret_net * 1e4, 2),
    }


def marks_payload(candles: list[Candle], barriers: Barriers, start: datetime, end: datetime,
                  cutoff: datetime | None, include_holdout: bool) -> dict:
    """Entradas dos 3 setups com candle de entrada em [start, end) e o resultado da tripla barreira.

    Sem `include_holdout`, os candles a partir de `cutoff` (período intocado) nem entram no cálculo, como em
    wescalp.research: nenhuma entrada nem saída cai lá dentro. `candles` traz aquecimento antes de `start` e
    `barriers.max_bars` candles depois de `end`, para as últimas entradas terem resultado.
    """
    if cutoff is not None and not include_holdout:
        candles = [c for c in candles if c.ts < cutoff]
    marks = []
    if candles:
        cols = features.compute(candles, atr_period=barriers.atr_period)
        labels = {(lb.ts, lb.side): lb for lb in label_candles(candles, barriers)}
        for name, rule in SETUPS.items():
            marks += [mark_json(name, lb) for lb in research.trades_for(candles, cols, labels, rule)
                      if start <= lb.ts < end]
    marks.sort(key=lambda m: (m["ts"], m["setup"], m["side"]))
    return {
        "config": barriers.name,
        "marks": marks,
        "holdout": {"from": epoch(cutoff), "weeks": HOLDOUT_WEEKS, "included": include_holdout},
    }


def gap_intervals(per_minute: dict[int, int], before: bool, any_book: bool,
                  start_min: int, now_min: int) -> list[list[int]]:
    """Intervalos [início, fim) em segundos sem book_l1 entre start_min e now_min, com a regra do vigia:
    só conta a partir do primeiro livro gravado se o coletor ligou depois de start_min."""
    cover = health.coverage_start(per_minute, before, any_book, start_min)
    if cover is None:
        return []
    return [[m * 60, (m + n) * 60] for m, n in health.gap_runs(per_minute, cover, now_min)]


def health_payload(report: health.Report, m: health.Metrics | None) -> dict:
    """Estado de wescalp.health (o mesmo do vigia) e os números da barra de estado."""
    out: dict = {
        "state": report.state,
        "headline": report.headline(),
        "problems": list(report.problems),
        "details": list(report.details),
        "now": report.now.isoformat(),
    }
    if m is None:
        return out
    now_s = m.now.timestamp()
    now_min = int(now_s // 60)
    gaps = gap_intervals(m.book_per_minute, m.book_before_window, m.last_book is not None,
                         now_min - health.WINDOW_MIN, now_min)
    ages = {k: (now_s - t.timestamp() if t is not None else None)
            for k, t in (("book", m.last_book), ("trade", m.last_trade))}
    per_minute = {"book_l1": m.book_per_minute, "trades": m.trades_per_minute,
                  "book_snapshots": m.snapshots_per_minute}
    gap_min = sum(b - a for a, b in gaps) // 60
    out.update({
        "last_book": m.last_book.isoformat() if m.last_book else None,
        "last_trade": m.last_trade.isoformat() if m.last_trade else None,
        "book_age_s": num(ages["book"], 1),
        "trade_age_s": num(ages["trade"], 1),
        "gap_minutes_24h": gap_min,
        "gaps_24h": gaps,
        "last_hour": {t: health.last_hour(pm, now_min) for t, pm in per_minute.items()},
        "hourly": {t: sorted([h * 3600, n] for h, n in health.hourly(pm).items()) for t, pm in per_minute.items()},
        "db_bytes": m.db_bytes,
        "table_bytes": m.table_bytes,
        "disk_free_bytes": m.disk_free_bytes,
        "disk_total_bytes": m.disk_total_bytes,
        "text": {  # já formatado em pt-BR, igual ao vigia
            "book_age": health.fmt_age(ages["book"]) if ages["book"] is not None else None,
            "trade_age": health.fmt_age(ages["trade"]) if ages["trade"] is not None else None,
            "db": health.fmt_bytes(m.db_bytes),
            "disk_free": health.fmt_bytes(m.disk_free_bytes) if m.disk_free_bytes is not None else None,
        },
    })
    return out


def flow_payload(cols: dict[str, list], start: datetime, end: datetime, bucket: float) -> dict:
    """Série de wescalp.flow.aggregate com o fim do balde em (start, end]. null = balde sem dado (buraco)."""
    lo, hi = start.timestamp(), end.timestamp()
    keep = [i for i, t in enumerate(cols.get("t", [])) if lo < t <= hi]
    digits = {"mid": 2, "ofi": 3, "delta": 3, "imb5": 4, "micro_bps": 3, "spread_bps": 4}
    out: dict = {"bucket": bucket, "t": [cols["t"][i] for i in keep]}
    for k, nd in digits.items():
        out[k] = [num(cols[k][i], nd) for i in keep]
    return out


def pick_bucket(raw: str | None) -> float:
    try:
        b = float(raw) if raw else 10.0
    except ValueError:
        raise BadRequest("bucket precisa ser número de segundos") from None
    return min(FLOW_BUCKETS, key=lambda x: abs(x - b))


# --- banco ---

async def _configure(conn: psycopg.AsyncConnection) -> None:
    await conn.execute("SET default_transaction_read_only = on")
    await conn.execute("SET statement_timeout = '30s'")


class Store:
    """Leitura do banco, num pool pequeno de conexões só de leitura. Os testes trocam por um falso."""

    def __init__(self, url: str = DATABASE_URL, symbol: str = FUTURES_SYMBOL, disk_path: str | None = None):
        self.symbol = symbol
        self.disk_path = disk_path
        self.pool = AsyncConnectionPool(
            url, min_size=1, max_size=4, open=False, timeout=15, configure=_configure,
            kwargs={"autocommit": True, "connect_timeout": 10},
        )
        self.flow_lock = asyncio.Semaphore(1)  # fluxo carrega até 6 h de livro em memória: um por vez
        self.health_lock = asyncio.Lock()
        self._metrics: health.Metrics | None = None
        self._full_at = self._sizes_at = 0.0
        self._old_gaps: dict[int, tuple[float, dict[int, int], bool]] = {}

    async def open(self) -> None:
        await self.pool.open(wait=False)

    async def close(self) -> None:
        await self.pool.close()

    async def candle_span(self, source: str) -> tuple[datetime | None, datetime | None]:
        async with self.pool.connection() as conn:
            return await db.candle_span(conn, source, SOURCES[source])

    async def candles(self, source: str, since: datetime, until: datetime) -> list[Candle]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT source, symbol, ts, open, high, low, close, volume, turnover FROM candles"
                " WHERE source = %s AND symbol = %s AND ts >= %s AND ts < %s ORDER BY ts",
                (source, SOURCES[source], since, until),
            )
            return [Candle(*r) for r in await cur.fetchall()]

    async def trade_rows(self, since: datetime, until: datetime) -> list[tuple[float, float, float]]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT extract(epoch FROM ts)::float8, price, size FROM trades"
                " WHERE symbol = %s AND ts >= %s AND ts < %s ORDER BY ts, trade_id",
                (self.symbol, since, until),
            )
            return await cur.fetchall()

    async def quotes(self, since: datetime, until: datetime) -> list[Quote]:
        """Topo do livro em ordem, por cursor no servidor: não guarda as linhas cruas além dos Quote."""
        async with self.pool.connection() as conn, conn.transaction():
            async with conn.cursor(name="wescalp_web_flow") as cur:
                cur.itersize = 5000
                await cur.execute(
                    "SELECT extract(epoch FROM ts)::float8, bid_px, bid_sz, ask_px, ask_sz, bid_sz5, ask_sz5"
                    " FROM book_l1 WHERE symbol = %s AND ts >= %s AND ts < %s ORDER BY ts, seq",
                    (self.symbol, since, until),
                )
                return [Quote(*r) async for r in cur]

    async def ticks(self, since: datetime, until: datetime) -> list[Tick]:
        async with self.pool.connection() as conn:
            return [Tick(*r) for r in await db.load_trades(conn, self.symbol, since, until)]

    async def metrics(self) -> health.Metrics:
        """Métricas do vigia. A primeira leitura (e uma a cada 10 min) é a do próprio wescalp.health; nas
        outras, só os minutos desde a última leitura são recontados e a janela de 24 h anda."""
        async with self.health_lock, self.pool.connection() as conn:
            mono = time.monotonic()
            m = self._metrics
            if m is not None and mono - self._full_at < HEALTH_FULL_EVERY_S:
                try:
                    m = await self._refresh(conn, m, mono)
                except psycopg.Error as e:
                    log.warning("leitura parcial do estado falhou (%s); relendo tudo", str(e).splitlines()[0])
                    m = None
            if m is None or mono - self._full_at >= HEALTH_FULL_EVERY_S:
                m = await health.read_metrics(conn, self.symbol, self.disk_path)
                self._full_at = self._sizes_at = mono
            self._metrics = m
            return m

    async def _refresh(self, conn, m: health.Metrics, mono: float) -> health.Metrics:
        now = await health._one(conn, "SELECT now()")
        since_min = int(m.now.timestamp()) // 60  # o minuto da última leitura estava incompleto: reconta
        since = datetime.fromtimestamp(since_min * 60, tz=UTC)
        window_start = int(now.timestamp()) // 60 - health.WINDOW_MIN

        def merge(old: dict[int, int], new: dict[int, int]) -> dict[int, int]:
            out = {k: n for k, n in old.items() if window_start <= k < since_min}
            out.update(new)
            return out

        async def last(table: str, old: datetime | None) -> datetime | None:
            q = f"SELECT max(ts) FROM {table} WHERE symbol = %s AND ts >= %s"  # nome fixo, não vem do usuário
            return await health._one(conn, q, (self.symbol, since)) or old

        counts = {t: await health._per_minute(conn, t, self.symbol, since) for t in health.TABLES}
        kw = dict(
            now=now,
            last_book=await last("book_l1", m.last_book),
            last_trade=await last("trades", m.last_trade),
            book_per_minute=merge(m.book_per_minute, counts["book_l1"]),
            trades_per_minute=merge(m.trades_per_minute, counts["trades"]),
            snapshots_per_minute=merge(m.snapshots_per_minute, counts["book_snapshots"]),
            book_before_window=m.book_before_window or any(k < window_start for k in m.book_per_minute),
        )
        if mono - self._sizes_at >= SIZES_EVERY_S:
            free, total = health.disk_usage(self.disk_path)
            kw.update(db_bytes=await health._one(conn, "SELECT pg_database_size(current_database())"),
                      table_bytes=await health.table_sizes(conn), disk_free_bytes=free, disk_total_bytes=total)
            self._sizes_at = mono
        return replace(m, **kw)

    async def old_book_minutes(self, start_min: int, end_min: int) -> tuple[dict[int, int], bool]:
        """Livro por minuto em [start_min, end_min) e se havia livro antes de start_min. Esse trecho tem mais
        de 24 h (já comprimido no TimescaleDB) e muda pouco: guardado por 10 min."""
        key = start_min // 10
        hit = self._old_gaps.get(key)
        if hit and time.monotonic() - hit[0] < OLD_GAPS_TTL_S:
            return hit[1], hit[2]
        since, until = (datetime.fromtimestamp(x * 60, tz=UTC) for x in (start_min, end_min))
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT date_trunc('minute', ts), count(*) FROM book_l1"
                " WHERE symbol = %s AND ts >= %s AND ts < %s GROUP BY 1",
                (self.symbol, since, until),
            )
            per_minute = {int(t.timestamp()) // 60: n for t, n in await cur.fetchall()}
            before = bool(await health._one(
                conn, "SELECT EXISTS (SELECT 1 FROM book_l1 WHERE symbol = %s AND ts < %s)", (self.symbol, since)))
        self._old_gaps = {key: (time.monotonic(), per_minute, before)}
        return per_minute, before

    async def reports(self) -> list[tuple]:
        """Último relatório de cada nome na tabela reports; vazio se a tabela não existe."""
        async with self.pool.connection() as conn:
            if not await health._one(conn, "SELECT to_regclass('reports') IS NOT NULL"):
                return []
            cur = await conn.execute(
                "SELECT DISTINCT ON (name) name, created_at, body FROM reports ORDER BY name, created_at DESC")
            return await cur.fetchall()


# --- HTTP ---

def _default(x):
    if isinstance(x, datetime):
        return x.isoformat()
    return str(x)


class JSON(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"), default=_default).encode()

    def __init__(self, content, status_code: int = 200):
        super().__init__(content, status_code, headers={"Cache-Control": "no-store"})


class Static(StaticFiles):
    def file_response(self, *args, **kw):
        resp = super().file_response(*args, **kw)
        resp.headers["Cache-Control"] = "no-cache"  # revalida: deploy novo aparece sem limpar cache
        return resp


def _query_float(q, name: str, default: float, lo: float, hi: float) -> float:
    try:
        x = float(q.get(name, default))
    except ValueError:
        raise BadRequest(f"{name} precisa ser número") from None
    if not lo <= x <= hi:
        raise BadRequest(f"{name} fora de [{lo:g}, {hi:g}]")
    return x


def _window(q, default_end: datetime, default_len: timedelta, limit: timedelta) -> tuple[datetime, datetime, bool]:
    end = parse_time(q.get("to")) or default_end
    start = parse_time(q.get("from")) or end - default_len
    if start >= end:
        raise BadRequest("from precisa ser antes de to")
    start, clipped = clip_window(start, end, limit)
    return start, end, clipped


async def api_candles(request: Request) -> JSON:
    store, q = request.app.state.store, request.query_params
    source = q.get("source", "futures")
    if source not in SOURCES:
        raise BadRequest("source: futures ou spot")
    now = datetime.now(UTC)
    start, end, clipped = _window(q, now, DEFAULT_CANDLES * STEP, MAX_CANDLES * STEP)
    start = floor_candle(start)
    first, last = await store.candle_span(source)
    stored = await store.candles(source, warmup_start(start), end)
    live: list[Candle] = []
    if source == "futures":
        live_start = floor_candle(now - LIVE_MAX)
        if last is not None:
            live_start = max(live_start, last + STEP)
        if end > live_start:
            rows = await store.trade_rows(live_start, min(end, now))
            live = trade_candles(rows, live_start, end, SOURCES[source])
    payload = await asyncio.to_thread(candles_payload, stored + live, start, end, live[0].ts if live else None, now)
    payload.update({
        "source": source, "symbol": SOURCES[source], "from": epoch(start), "to": epoch(end), "clipped": clipped,
        "first": epoch(first), "last_stored": epoch(last), "more": first is not None and first < start,
        "now": epoch(now),
    })
    return JSON(payload)


async def api_marks(request: Request) -> JSON:
    store, q = request.app.state.store, request.query_params
    barriers = Barriers(
        _query_float(q, "tp", Barriers.tp_atr, 0.1, 20),
        _query_float(q, "sl", Barriers.sl_atr, 0.1, 20),
        int(_query_float(q, "bars", Barriers.max_bars, 1, 96)),
    )
    include = q.get("holdout", "") in ("1", "true", "sim")
    first, last = await store.candle_span("futures")
    if last is None:
        return JSON({"config": barriers.name, "marks": [], "holdout": {"from": None, "weeks": HOLDOUT_WEEKS,
                                                                      "included": include}})
    start, end, clipped = _window(q, last + STEP, DEFAULT_CANDLES * STEP, MAX_CANDLES * STEP)
    start = floor_candle(start)
    candles = await store.candles("futures", warmup_start(start), end + (barriers.max_bars + 1) * STEP)
    cutoff = holdout_start(last)
    payload = await asyncio.to_thread(marks_payload, candles, barriers, start, end, cutoff, include)
    payload.update({"from": epoch(start), "to": epoch(end), "clipped": clipped})
    return JSON(payload)


async def api_health(request: Request) -> JSON:
    store, lim = request.app.state.store, request.app.state.limits
    try:
        m = await store.metrics()
    except Exception as e:  # banco fora do ar também é um estado: parado
        return JSON(health_payload(health.unreachable(e, datetime.now(UTC)), None))
    return JSON(health_payload(health.evaluate(m, lim), m))


async def api_gaps(request: Request) -> JSON:
    store = request.app.state.store
    hours = int(_query_float(request.query_params, "hours", 48, 1, GAPS_MAX_HOURS))
    m = await store.metrics()
    now_min = int(m.now.timestamp()) // 60
    start_min = now_min - hours * 60
    window_start = now_min - health.WINDOW_MIN
    per_minute = {k: n for k, n in m.book_per_minute.items() if k >= start_min}
    if start_min < window_start:
        older, before = await store.old_book_minutes(start_min, window_start)
        per_minute.update(older)
    else:
        before = m.book_before_window or any(k < start_min for k in m.book_per_minute)
    gaps = gap_intervals(per_minute, before, m.last_book is not None, start_min, now_min)
    return JSON({"hours": hours, "from": start_min * 60, "to": now_min * 60, "gaps": gaps,
                 "minutes": sum(b - a for a, b in gaps) // 60})


async def api_flow(request: Request) -> JSON:
    store, q = request.app.state.store, request.query_params
    bucket = pick_bucket(q.get("bucket"))
    limit = min(FLOW_MAX, timedelta(seconds=bucket * FLOW_MAX_BUCKETS))
    start, end, clipped = _window(q, datetime.now(UTC), timedelta(hours=1), limit)
    async with store.flow_lock:
        quotes = await store.quotes(start, end)
        ticks = await store.ticks(start, end)
        cols = await asyncio.to_thread(aggregate, quotes, ticks, bucket)
        del quotes
    payload = flow_payload(cols, start, end, bucket)
    payload.update({"from": start.timestamp(), "to": end.timestamp(), "clipped": clipped,
                    "max_hours": limit.total_seconds() / 3600})
    return JSON(payload)


async def api_reports(request: Request) -> JSON:
    try:
        rows = await request.app.state.store.reports()
    except psycopg.errors.UndefinedTable:
        rows = []
    return JSON({"reports": [{"name": n, "created_at": c, "body": b} for n, c, b in rows]})


async def _bad_request(request: Request, exc: Exception) -> JSON:
    return JSON({"error": str(exc)}, 400)


async def _db_error(request: Request, exc: Exception) -> JSON:
    msg = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
    log.warning("%s: %s", request.url.path, msg)
    return JSON({"error": f"banco indisponível: {msg}"}, 503)


def create_app(store=None, limits: health.Limits | None = None) -> Starlette:
    store = store or Store(DATABASE_URL, FUTURES_SYMBOL, os.environ.get("WESCALP_HEALTH_DISK_PATH") or None)

    @asynccontextmanager
    async def lifespan(app):
        await store.open()
        try:
            yield
        finally:
            await store.close()

    app = Starlette(
        routes=[
            Route("/api/candles", api_candles),
            Route("/api/marks", api_marks),
            Route("/api/health", api_health),
            Route("/api/gaps", api_gaps),
            Route("/api/flow", api_flow),
            Route("/api/reports", api_reports),
            Mount("/", Static(directory=STATIC_DIR, html=True)),
        ],
        middleware=[Middleware(GZipMiddleware, minimum_size=1024)],
        exception_handlers={BadRequest: _bad_request, psycopg.Error: _db_error, PoolTimeout: _db_error},
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.limits = limits or health.Limits.from_env()
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("psycopg.pool").setLevel(logging.WARNING)
    # Sem log de acesso: a página consulta a cada 10 s.
    uvicorn.run(create_app(), host="0.0.0.0", port=PORT, access_log=False, proxy_headers=True)


if __name__ == "__main__":
    main()
