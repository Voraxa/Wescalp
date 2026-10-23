import io
import zipfile
from datetime import date, datetime, timezone

import pytest

from wescalp import binance

# Primeiras linhas reais de BTCUSDT-5m-2024-12 (milissegundos) e de 2025-01 (microssegundos).
MS_ROW = ("1733011200000,96407.99000000,96524.00000000,96350.13000000,96508.29000000,35.87706000,1733011499999,"
          "3459760.39953790,10233,19.01492000,1833852.07539100,0")
US_ROW = ("1735689600000000,93576.00000000,93702.15000000,93537.50000000,93661.20000000,45.94160000,"
          "1735689899999999,4301907.48382050,7448,16.87967000,1580504.06031420,0")
HEADER = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
          "taker_buy_quote_volume,ignore")


def test_milliseconds_until_2024():
    c = binance.parse_row(MS_ROW.split(","))
    assert c.ts == datetime(2024, 12, 1, tzinfo=timezone.utc)
    assert (c.open, c.high, c.low, c.close) == (96407.99, 96524.0, 96350.13, 96508.29)  # ordem normal: O, H, L, C
    assert (c.volume, c.turnover) == (35.87706, 3459760.3995379)


def test_microseconds_from_2025():
    c = binance.parse_row(US_ROW.split(","))
    assert c.ts == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert c.close == 93661.2


def test_header_and_blank_rows_are_skipped():
    assert binance.parse_row(HEADER.split(",")) is None
    assert binance.parse_row([]) is None


def test_parse_zip_with_header():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("BTCUSDT-5m-2025-01.csv", "\n".join([HEADER, US_ROW]) + "\n")
    (c,) = binance.parse_zip(buf.getvalue())
    assert c.ts == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_months_inclusive_across_years():
    assert binance.months(date(2017, 11, 1), date(2018, 2, 1)) == [
        date(2017, 11, 1), date(2017, 12, 1), date(2018, 1, 1), date(2018, 2, 1)]


@pytest.mark.parametrize("t, s", [(1502942400000, 1502942400.0), (1735689600000000, 1735689600.0)])
def test_to_seconds(t, s):
    assert binance.to_seconds(t) == s
