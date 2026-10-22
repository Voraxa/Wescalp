from wescalp.backfill import windows
from wescalp.collector import Buffer
from wescalp.db import l1_row
from wescalp.kucoin import parse_depth, parse_execution, parse_futures_candle, parse_spot_candle


def test_spot_candle_reorders_open_close_high_low():
    c = parse_spot_candle("BTC-USDT", ["1732293600", "98730.3", "98598.9", "98749.9", "98504.6", "14.2", "1401612.1"])
    assert (c.open, c.high, c.low, c.close) == (98730.3, 98749.9, 98504.6, 98598.9)
    assert c.ts.timestamp() == 1732293600


def test_futures_candle_uses_milliseconds():
    c = parse_futures_candle("XBTUSDTM", [1792428300000, 81985.9, 81987.5, 81906.4, 81971.2, 1057, 86632.19])
    assert (c.open, c.high, c.low, c.close) == (81985.9, 81987.5, 81906.4, 81971.2)
    assert c.ts.timestamp() == 1792428300


def test_execution_timestamp_in_nanoseconds():
    t = parse_execution({
        "symbol": "XBTUSDTM", "sequence": 1, "side": "buy", "size": 1, "price": "81446",
        "tradeId": "1948879082354", "ts": 1792687284162000000,
    })
    assert t.trade_id == 1948879082354 and t.side == "buy" and t.price == 81446.0
    assert abs(t.ts.timestamp() - 1792687284.162) < 1e-3


def test_depth_and_l1():
    b = parse_depth("XBTUSDTM", {
        "sequence": 5, "ts": 1792687281170,
        "bids": [["100", 1], ["99", 2], ["98", 3], ["97", 4], ["96", 5], ["95", 6]],
        "asks": [["101", 7], ["102", 8]],
    }, received_ms=0)
    row = l1_row(b)
    assert row[3:] == (100.0, 1.0, 101.0, 7.0, 15.0, 15.0)


def test_depth_falls_back_to_receive_time():
    b = parse_depth("XBTUSDTM", {"sequence": 1, "bids": [], "asks": []}, received_ms=1_000_000)
    assert b.ts.timestamp() == 1000


def test_windows_cover_range_without_gaps():
    w = list(windows(0, 1000, 300))
    assert w == [(0, 300), (300, 600), (600, 900), (900, 1000)]


def test_buffer_throttles_snapshots(monkeypatch):
    buf = Buffer()
    b = parse_depth("XBTUSDTM", {"sequence": 1, "bids": [["1", 1]], "asks": [["2", 1]]}, received_ms=0)
    for ms in (10_000, 10_100, 10_900, 11_000, 11_500):
        buf.add_book(b, ms)
    trades, l1, snaps = buf.take()
    assert len(l1) == 5 and len(snaps) == 2
