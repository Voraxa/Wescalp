from datetime import date

import pytest

np = pytest.importorskip("numpy")

from wescalp import factors  # noqa: E402

# Trechos reais das respostas (alternative.me manda do mais novo para o mais velho; FRED deixa feriado vazio).
FNG = {"data": [
    {"value": "50", "value_classification": "Neutral", "timestamp": "1792713600", "time_until_update": "77793"},
    {"value": "30", "value_classification": "Fear", "timestamp": "1517443200"},
], "metadata": {"error": None}}
FRED = "observation_date,NASDAQCOM\n1971-02-05,100.000\n1971-02-08,100.840\n1971-02-15,\n1971-02-16,.\n1971-02-17,101.5\n"


def day(y: int, m: int, d: int) -> int:
    return (date(y, m, d) - factors.EPOCH).days


def test_parse_fng_is_sorted_by_day_utc():
    assert factors.parse_fng(FNG) == [(date(2018, 2, 1), 30.0), (date(2026, 10, 23), 50.0)]


def test_parse_fred_skips_holidays():
    assert factors.parse_fred(FRED) == [(date(1971, 2, 5), 100.0), (date(1971, 2, 8), 100.84), (date(1971, 2, 17), 101.5)]


def test_weekday():
    assert list(factors.weekday(np.array([day(2026, 10, 19), day(2026, 10, 23), day(2026, 10, 25)]))) == [0, 4, 6]


def test_daily_values_count_from_the_next_day():
    days = np.array([day(2026, 10, 15), day(2026, 10, 16), day(2026, 10, 19)])  # quinta, sexta, segunda
    vals = np.array([1.0, 2.0, 3.0])
    when = np.array([day(2026, 10, 15), day(2026, 10, 16), day(2026, 10, 19), day(2026, 10, 20)])
    out = factors.known("NASDAQCOM", days, vals, when)
    assert np.isnan(out[0]) and list(out[1:]) == [1.0, 2.0, 3.0]  # na segunda 00h, o fechamento de sexta


def test_weekly_release_waits_for_the_monday_report():
    weeks = [day(2026, 10, 5) + i for i in range(5)] + [day(2026, 10, 12) + i for i in range(5)]
    days = np.array(weeks)
    vals = np.arange(10.0)
    when = np.array([day(2026, 10, 12), day(2026, 10, 13), day(2026, 10, 19), day(2026, 10, 20)])
    out = factors.known("DTWEXBGS", days, vals, when)
    # segunda 12/10 00h: o relatório da semana de 05/10 ainda não saiu; terça 13/10: até a sexta 09/10
    assert np.isnan(out[0]) and list(out[1:]) == [4.0, 4.0, 9.0]


def test_save_and_load_round_trip(tmp_path):
    rows = [(date(2020, 1, 2), 1.5), (date(2020, 1, 3), 2.25)]
    factors.save(rows, factors.path("x", tmp_path))
    days, vals = factors.load("x", tmp_path)
    assert list(days) == [(d - date(1970, 1, 1)).days for d, _ in rows] and list(vals) == [1.5, 2.25]
