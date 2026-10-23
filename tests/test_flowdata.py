import io
import zipfile
from datetime import date

import pytest

from wescalp import flowdata

# Linhas reais de cada arquivo (com e sem cabeçalho).
KLINE_HEADER = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
                "taker_buy_quote_volume,ignore")
PERP = "1748736000000,104544.60,104599.60,104476.70,104476.80,418.297,1748736299999,43722484.33930,8524,148.643,15538459.17940,0"
SPOT_US = ("1748736000000000,104591.88000000,104647.11000000,104530.42000000,104530.43000000,44.40977000,"
           "1748736299999999,4644728.54027490,8151,14.88668000,1557076.54914650,0")
PREMIUM = "1577836800000,-0.00077732,-0.00077581,-0.00111218,-0.00098597,0,1577837099999,0,5,0,0,0"
FUNDING = "calc_time,funding_interval_hours,last_funding_rate\n1748736000001,8,-0.00000582\n1748764800002,8,0.00002335"
METRICS = ("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
           "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
           "2021-12-01 00:05:00,BTCUSDT,59192.97400000,3365451460.62466000,4.128851729485537,1.1345591007,"
           "3.996236679335271,0.8262745659")


def zipped(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", text + "\n")
    return buf.getvalue()


def test_perp_kline_in_milliseconds_with_header():
    (row,) = flowdata.csv_rows(zipped(KLINE_HEADER + "\n" + PERP))
    assert flowdata.parse("perp", row) == (1748736000, 104544.6, 104599.6, 104476.7, 104476.8, 418.297, 148.643, 8524)


def test_spot_flow_in_microseconds():
    (row,) = flowdata.csv_rows(zipped(SPOT_US))
    assert flowdata.parse("spot-flow", row) == (1748736000, 44.40977, 14.88668, 8151)


def test_premium_keeps_the_premium_candle():
    (row,) = flowdata.csv_rows(zipped(PREMIUM))
    assert flowdata.parse("premium", row) == (1577836800, -0.00077732, -0.00077581, -0.00111218, -0.00098597)


def test_funding_drops_the_extra_milliseconds():
    rows = flowdata.csv_rows(zipped(FUNDING))
    assert [flowdata.parse("funding", r) for r in rows] == [(1748736000, -0.00000582), (1748764800, 0.00002335)]


def test_metrics_text_time_is_utc():
    (row,) = flowdata.csv_rows(zipped(METRICS))
    out = flowdata.parse("metrics", row)
    assert out[0] == 1638317100 and out[1] == 59192.974 and out[-1] == pytest.approx(0.8262745659)


def test_metrics_empty_fields_become_nan():
    row = "2022-03-10 00:00:00,BTCUSDT,60000.1,2400000000.5,,,,0.95".split(",")
    out = flowdata.parse("metrics", row)
    assert out[1] == 60000.1 and all(x != x for x in out[3:6]) and out[6] == 0.95


def test_files_cover_months_or_days_until_the_end_of_the_month():
    assert len(flowdata.files("perp", date(2020, 3, 1))) == 3
    days = flowdata.files("metrics", date(2021, 12, 1))
    assert len(days) == 31 and days[-1].endswith("BTCUSDT-metrics-2021-12-31.zip")


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(flowdata, "files", lambda kind, until: ["u1", "u2"])

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            body = zipped(FUNDING) if url == "u1" else zipped("calc_time,funding_interval_hours,last_funding_rate\n"
                                                                "1748736000001,8,-0.00000582")
            return type("R", (), {"status_code": 200, "content": body, "raise_for_status": lambda self: None})()

    monkeypatch.setattr(flowdata.httpx, "Client", Client)
    assert flowdata.fetch("funding", date(2025, 6, 1), tmp_path, jobs=2) == 2  # a repetida entra uma vez só
    s = flowdata.load("funding", tmp_path)
    assert list(s["ts"]) == [1748736000, 1748764800] and list(s["rate"]) == [-0.00000582, 0.00002335]
