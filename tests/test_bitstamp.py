from datetime import datetime, timedelta, timezone

import httpx
import pytest

from wescalp import bitstamp, brain

UTC = timezone.utc
# Trecho real da resposta (01/01/2013, hora sem negócio, e 11/02/2013).
BODY = {"data": {"pair": "BTC/USD", "ohlc": [
    {"timestamp": "1356998400", "open": "13.24", "high": "13.24", "low": "13.24", "close": "13.24",
     "volume": "0.00000000"},
    {"timestamp": "1360594800", "open": "23.43", "high": "23.43", "low": "23.42", "close": "23.43",
     "volume": "238.36693561"},
]}}


def test_parse_named_fields():
    a, b = bitstamp.parse(BODY)
    assert a.ts == datetime(2013, 1, 1, tzinfo=UTC) and a.volume == 0.0
    assert (b.open, b.high, b.low, b.close) == (23.43, 23.43, 23.42, 23.43)
    assert b.volume == 238.36693561 and b.turnover == pytest.approx(238.36693561 * 23.43)


def fake_api(first_trade: datetime, last_trade: datetime, calls: list[int]) -> httpx.Client:
    """A API como ela é: até LIMIT candles a partir de `start`, só onde já há negócio."""
    def handler(req: httpx.Request) -> httpx.Response:
        start, limit = int(req.url.params["start"]), int(req.url.params["limit"])
        calls.append(start)
        ts = [t for t in range(start, start + limit * 3600, 3600) if first_trade.timestamp() <= t <= last_trade.timestamp()]
        ohlc = [{"timestamp": str(t), "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"} for t in ts]
        return httpx.Response(200, json={"data": {"pair": "BTC/USD", "ohlc": ohlc}})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_pages_skips_the_empty_start_and_stops_at_last(monkeypatch):
    monkeypatch.setattr(bitstamp, "LIMIT", 10)
    first = datetime(2011, 8, 1, tzinfo=UTC)
    trade = first + timedelta(hours=25)  # janelas vazias antes do primeiro negócio
    last = first + timedelta(hours=58)
    calls = []
    cs = bitstamp.fetch(first, last, fake_api(trade, last + timedelta(days=30), calls))
    assert cs[0].ts == trade and cs[-1].ts == last and len(cs) == 34
    assert all((b.ts - a.ts) == timedelta(hours=1) for a, b in zip(cs, cs[1:]))
    assert calls[:3] == [int((first + timedelta(hours=10 * k)).timestamp()) for k in range(3)]


def test_locked_files_need_the_unlock(tmp_path, monkeypatch):
    p = bitstamp.path(tmp_path / "locked")
    monkeypatch.delenv("WESCALP_UNLOCK", raising=False)
    brain.save_candles(p, bitstamp.parse(BODY))
    with pytest.raises(PermissionError):
        brain.load_candles(p, bitstamp.SYMBOL, "spot")
    monkeypatch.setenv("WESCALP_UNLOCK", "prova-unica")
    assert [c.close for c in brain.load_candles(p, bitstamp.SYMBOL, "spot")] == [13.24, 23.43]


def test_digest_ignores_gzip_time(tmp_path):
    cs = bitstamp.parse(BODY)
    a, b = tmp_path / "a" / "x.csv.gz", tmp_path / "b" / "x.csv.gz"
    brain.save_candles(a, cs)
    brain.save_candles(b, cs)
    assert bitstamp.digest(a) == bitstamp.digest(b)


def test_check_counts_gaps_empty_hours_and_jumps():
    t0 = datetime(2013, 1, 1, tzinfo=UTC)
    closes = [10.0, 10.0, 10.0, 12.0, 12.0, 6.0]
    vols = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    hours = [0, 1, 2, 3, 5, 6]  # falta a hora 4
    cs = [brain.Candle("spot", "BTCUSD", t0 + timedelta(hours=h), c, c, c, c, v, v * c)
          for h, c, v in zip(hours, closes, vols)]
    s = bitstamp.check(cs)
    assert s["gaps"] == 1 and s["empty"] == 2 and s["longest_empty"] == 2 and s["longest_empty_at"] == cs[1].ts
    assert s["jumps"][0] == (cs[5].ts, pytest.approx(-0.5)) and s["jumps"][1] == (cs[3].ts, pytest.approx(0.2))
