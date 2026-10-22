import csv
import gzip
import io
from datetime import date

import pytest

from wescalp import flow, tardis
from wescalp.flow import Quote, Tick

# Cabeçalho e primeiras linhas reais de book_snapshot_5/2025/01/01/XBTUSDTM: níveis intercalados ask, bid.
BOOK_HEADER = ("exchange,symbol,timestamp,local_timestamp,"
               + ",".join(f"asks[{i}].price,asks[{i}].amount,bids[{i}].price,bids[{i}].amount" for i in range(5)))
BOOK_ROWS = [
    "kucoin-futures,XBTUSDTM,1735689600980665,1735689600980665,93593.4,4810,93593.3,25,93595,332,93586.3,37,"
    "93595.4,18,93584,210,93595.7,10,93583.9,395,93597.2,1,93583.3,215",
    "kucoin-futures,XBTUSDTM,1735689601006000,1735689601009780,93593.4,4792,93593.3,25,93595,332,93586.3,37,"
    "93595.4,18,93584,210,93595.7,10,93583.9,395,93597.2,1,93583.3,215",
]
TRADE_HEADER = "exchange,symbol,timestamp,local_timestamp,id,side,price,amount"
TRADE_ROWS = [
    "kucoin-futures,XBTUSDTM,1735689601757000,1735689601761463,1809576860625,buy,93593.4,1",
    "kucoin-futures,XBTUSDTM,1735689602649000,1735689602653109,1809576863981,sell,93593.3,3",
]


def rows(header: str, lines: list[str]) -> list[dict]:
    return list(csv.DictReader(io.StringIO("\n".join([header, *lines]))))


def test_snapshot5_maps_interleaved_levels_and_microseconds():
    q = tardis.parse_snapshot5(rows(BOOK_HEADER, BOOK_ROWS)[0])
    assert q.ts == pytest.approx(1735689600.980665)  # local_timestamp em µs
    assert (q.bid_px, q.bid_sz, q.ask_px, q.ask_sz) == (93593.3, 25, 93593.4, 4810)
    assert q.bid_sz5 == 25 + 37 + 210 + 395 + 215
    assert q.ask_sz5 == 4810 + 332 + 18 + 10 + 1


def test_snapshot5_uses_local_timestamp():
    q = tardis.parse_snapshot5(rows(BOOK_HEADER, BOOK_ROWS)[1])
    assert q.ts == pytest.approx(1735689601.009780)  # não o timestamp da corretora (…006000)


def test_snapshot5_missing_levels():
    row = rows(BOOK_HEADER, BOOK_ROWS)[0]
    for i in (3, 4):
        row[f"bids[{i}].price"] = row[f"bids[{i}].amount"] = ""
    assert tardis.parse_snapshot5(row).bid_sz5 == 25 + 37 + 210
    row["asks[0].price"] = row["asks[0].amount"] = ""
    assert tardis.parse_snapshot5(row) is None


def test_trade_side_is_aggressor_and_size_in_contracts():
    a, b = (tardis.parse_trade(r) for r in rows(TRADE_HEADER, TRADE_ROWS))
    assert a == Tick(pytest.approx(1735689601.761463), "buy", 1.0)
    assert (b.side, b.size) == ("sell", 3.0)


def q(ts: float, bid: float = 100.0) -> Quote:
    return Quote(ts, bid, 1, bid + 0.1, 1, 5, 5)


def test_thin_keeps_last_quote_of_each_100ms():
    out = tardis.thin([q(10.00, 1), q(10.05, 2), q(10.09, 3), q(10.10, 4), q(10.35, 5), q(10.36, 6)])
    assert [x.bid_px for x in out] == [3, 4, 6]


def test_concat_pad_blocks_returns_across_days():
    day1 = {"t": [1, 2, 3], "mid": [100.0, 101.0, 102.0]}
    day2 = {"t": [9, 10, 11], "mid": [200.0, 201.0, 202.0]}
    cols = tardis.concat([day1, day2], pad=3)
    assert cols["mid"] == [100.0, 101.0, 102.0, None, None, None, 200.0, 201.0, 202.0]
    rets = flow.forward_returns(cols["mid"], 2)
    assert rets[0] == pytest.approx(0.02)
    assert rets[1:6] == [None] * 5  # nada de 101 → 200
    assert rets[6] == pytest.approx(0.01)


def test_free_days_first_of_month_until_yesterday():
    days = tardis.free_days(date(2026, 10, 22))
    assert days[0] == date(2024, 2, 1) and days[-1] == date(2026, 10, 1)
    assert len(days) == 33 and all(d.day == 1 for d in days)
    assert tardis.free_days(date(2026, 10, 1))[-1] == date(2026, 9, 1)  # o dia de hoje ainda não saiu


def test_fetch_day_converts_and_loads_back(tmp_path, monkeypatch):
    book = BOOK_ROWS + [  # mais duas fotos: uma na mesma janela de 100 ms (fica a última) e uma 3 s depois
        "kucoin-futures,XBTUSDTM,1,1735689601050000," + BOOK_ROWS[0].split(",", 4)[4].replace("4810", "4700"),
        "kucoin-futures,XBTUSDTM,1,1735689604000000," + BOOK_ROWS[0].split(",", 4)[4],
    ]

    def fake_download(client, src, dst):
        header, lines = (BOOK_HEADER, book) if "book_snapshot_5" in src else (TRADE_HEADER, TRADE_ROWS)
        with gzip.open(dst, "wt") as f:
            f.write("\n".join([header, *lines]) + "\n")

    monkeypatch.setattr(tardis, "_download", fake_download)
    line = tardis.fetch_day(date(2025, 1, 1), tmp_path)
    assert "4 linhas do livro → 3 fotos, 2 negócios, 0 buracos > 30 s, maior intervalo 3.0 s" in line
    quotes, ticks = tardis.load_day(date(2025, 1, 1), tmp_path)
    assert [x.ask_sz for x in quotes] == [4810, 4700, 4810]
    assert ticks[1] == Tick(pytest.approx(1735689602.653109), "sell", 3.0)
    assert tardis.available_days(tmp_path) == [date(2025, 1, 1)]
    assert tardis.fetch_day(date(2025, 1, 1), tmp_path).endswith("já convertido")
