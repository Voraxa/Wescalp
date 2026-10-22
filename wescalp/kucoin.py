"""Cliente mínimo da API pública da KuCoin (candles, token de WebSocket) e parsers das mensagens."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from .config import CANDLE_SECONDS, FUTURES_REST, SPOT_REST

log = logging.getLogger(__name__)

SPOT_MAX_CANDLES = 1500
FUTURES_MAX_CANDLES = 200


@dataclass(frozen=True)
class Candle:
    source: str
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


def _utc(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def parse_spot_candle(symbol: str, row: list) -> Candle:
    # Spot: [time(s), open, close, high, low, volume, turnover] — atenção à ordem open, close, high, low.
    t, o, c, h, l, v, q = row
    return Candle("spot", symbol, _utc(int(t)), float(o), float(h), float(l), float(c), float(v), float(q))


def parse_futures_candle(symbol: str, row: list) -> Candle:
    # Futuros: [time(ms), open, high, low, close, volume, turnover]
    t, o, h, l, c, v, q = row
    return Candle("futures", symbol, _utc(int(t) / 1000), float(o), float(h), float(l), float(c), float(v), float(q))


class KucoinRest:
    def __init__(self, client: httpx.AsyncClient | None = None, pause: float = 0.35):
        self.client = client or httpx.AsyncClient(timeout=20)
        self.pause = pause

    async def _get(self, url: str, params: dict) -> list:
        for attempt in range(6):
            try:
                r = await self.client.get(url, params=params)
                if r.status_code == 429:
                    raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
                r.raise_for_status()
                body = r.json()
                if body.get("code") != "200000":
                    raise RuntimeError(f"KuCoin {body.get('code')}: {body.get('msg')}")
                await asyncio.sleep(self.pause)
                return body["data"] or []
            except (httpx.HTTPError, RuntimeError) as e:
                wait = 2 ** attempt
                log.warning("GET %s falhou (%s); nova tentativa em %ss", url, e, wait)
                await asyncio.sleep(wait)
        raise RuntimeError(f"GET {url} falhou depois de 6 tentativas")

    async def spot_candles(self, symbol: str, start: int, end: int) -> list[Candle]:
        """Candles de 5 min em [start, end) (segundos), até 1500 por chamada."""
        data = await self._get(
            f"{SPOT_REST}/api/v1/market/candles",
            {"type": "5min", "symbol": symbol, "startAt": start, "endAt": end},
        )
        return [c for c in (parse_spot_candle(symbol, row) for row in data) if c.ts.timestamp() < end]

    async def futures_candles(self, symbol: str, start: int, end: int) -> list[Candle]:
        """Candles de 5 min em [start, end) (segundos), até 200 por chamada."""
        data = await self._get(
            f"{FUTURES_REST}/api/v1/kline/query",
            {"symbol": symbol, "granularity": CANDLE_SECONDS // 60, "from": start * 1000, "to": end * 1000},
        )
        return [c for c in (parse_futures_candle(symbol, row) for row in data) if c.ts.timestamp() < end]

    async def futures_ws_endpoint(self) -> tuple[str, int]:
        """Endereço do WebSocket público de futuros (com token) e o intervalo de ping em ms."""
        r = await self.client.post(f"{FUTURES_REST}/api/v1/bullet-public")
        r.raise_for_status()
        data = r.json()["data"]
        server = data["instanceServers"][0]
        return f"{server['endpoint']}?token={data['token']}", int(server["pingInterval"])

    async def aclose(self) -> None:
        await self.client.aclose()


@dataclass(frozen=True)
class Trade:
    symbol: str
    ts: datetime
    trade_id: int
    side: str
    price: float
    size: float


@dataclass(frozen=True)
class Book:
    symbol: str
    ts: datetime
    seq: int
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


def parse_execution(data: dict) -> Trade:
    # ts vem em nanossegundos.
    return Trade(
        symbol=data["symbol"],
        ts=_utc(int(data["ts"]) / 1e9),
        trade_id=int(data["tradeId"]),
        side=data["side"],
        price=float(data["price"]),
        size=float(data["size"]),
    )


def parse_depth(symbol: str, data: dict, received_ms: int) -> Book:
    ts_ms = data.get("ts") or data.get("timestamp") or received_ms
    return Book(
        symbol=symbol,
        ts=_utc(int(ts_ms) / 1000),
        seq=int(data["sequence"]),
        bids=[(float(p), float(s)) for p, s in data["bids"]],
        asks=[(float(p), float(s)) for p, s in data["asks"]],
    )
