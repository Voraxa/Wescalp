import gzip
import json
import sys

import pytest

np = pytest.importorskip("numpy")

from wescalp import odds  # noqa: E402

# Com a média de 2 dias, "acima da média" é só "fechou acima de ontem": dá para conferir tudo à mão.
CLOSE = np.array([10.0, 11, 10, 12, 15, 9, 8, 8.5, 13, 12, 13])
DAY = np.arange(len(CLOSE)) + 20000
UP, DOWN = [3, 7, 10], [2, 5, 9]  # o dia 1 (primeiro com média, já acima) não é virada
FEE2 = (1 - odds.FEE) ** 2


def test_flips_with_a_real_average_skip_the_first_day_of_it():
    c = np.array([1.0, 2, 3, 4, 1, 1, 5, 5, 2])
    # média de 3: dia 2 = 2 (3 acima, primeiro dia: não conta), 3 → 3, 4 → 2,67 (1 abaixo: venda), 5 → 2,
    # 6 → 2,33 (5 acima: compra), 7 → 3,67, 8 → 4 (2 abaixo: venda)
    up, down = odds.flips(c, 3)
    assert list(up) == [6] and list(down) == [4, 8]
    up, down = odds.flips(CLOSE, 2)
    assert list(up) == UP and list(down) == DOWN


def test_flips_only_look_back():
    for k in range(1, len(CLOSE) + 1):
        up, down = odds.flips(CLOSE[:k], 2)
        assert list(up) == [i for i in UP if i < k] and list(down) == [i for i in DOWN if i < k]
    changed = CLOSE.copy()
    changed[6:] = [100.0, 1, 100, 1, 100]  # mudar o futuro não mexe nas viradas de antes
    up, down = odds.flips(changed, 2)
    assert [i for i in up if i < 6] == [3] and [i for i in down if i < 6] == [2, 5]


def test_after_uses_calendar_days_and_drops_events_without_data():
    day = np.array([0, 1, 2, 4, 5])  # falta o dia 3
    close = np.array([100.0, 110, 90, 120, 80])
    out = odds.after(day, close, np.array([0, 2]), horizons=(1, 3, 4))
    # 1 dia: 110/100 e, do dia 2, o dia 3 falta e vale o primeiro depois (dia 4): 120/90
    assert out["1"] == {"n": 2, "median": pytest.approx((0.1 + 1 / 3) / 2), "up": 1.0}
    # 3 dias: do dia 0 vai ao dia 4 (120/100); do dia 2, ao dia 5 (80/90)
    assert out["3"] == {"n": 2, "median": pytest.approx((0.2 - 1 / 9) / 2), "up": 0.5}
    # 4 dias: o dia 2 + 4 passa do fim dos dados e fica de fora
    assert out["4"] == {"n": 1, "median": pytest.approx(0.2), "up": 1.0}


def test_trade_of_each_buy_until_the_next_sell_with_fee():
    s = odds.stats(DAY, CLOSE, 2)
    buy, sell = s["M2 ↑"], s["M2 ↓"]
    assert buy["n"] == 3 and buy["first"] == "2024-10-07" and buy["last"] == "2024-10-14"  # dias 3 e 10
    # compras nos dias 3 (12) e 7 (8,5), vendidas nos dias 5 (9) e 9 (12); a do dia 10 ainda está aberta
    assert buy["trade"] == {"n": 2, "open": 1, "win": 0.5, "median_gain": pytest.approx(12 / 8.5 * FEE2 - 1),
                            "median_loss": pytest.approx(9 / 12 * FEE2 - 1), "median_days": 2.0}
    # vendas nos dias 2 (10), 5 (9) e 9 (12), recompradas nos dias 3 (12), 7 (8,5) e 10 (13)
    assert sell["n"] == 3 and sell["rebuy"] == {"n": 3, "out": 0, "cheaper": pytest.approx(1 / 3),
                                                "median": pytest.approx(13 / 12 - 1), "median_days": 1.0}
    # 7 dias depois: só a compra do dia 3 (dia 10: 13/12) e a venda do dia 2 (dia 9: 12/10) têm dados
    assert buy["after"]["7"] == {"n": 1, "median": pytest.approx(13 / 12 - 1), "up": 1.0}
    assert sell["after"]["7"] == {"n": 1, "median": pytest.approx(0.2), "up": 1.0}
    assert buy["after"]["30"] == {"n": 0, "median": None, "up": None}


def test_build_has_the_period_and_both_flips_of_each_average():
    body = odds.build(DAY, CLOSE, "2026-10-25", lookbacks=(2, 3))
    assert body["from"] == "2024-10-04" and body["to"] == "2024-10-14"
    assert body["generated"] == "2026-10-25" and body["fee"] == odds.FEE and body["horizons"] == [7, 30, 90]
    assert list(body["events"]) == ["M2 ↑", "M2 ↓", "M3 ↑", "M3 ↓"]
    json.dumps(body)  # só tipos do Python, sem numpy


def test_cli_writes_the_json(tmp_path, monkeypatch):
    days = 400
    close = 100 + 30 * np.sin(np.arange(days) * 2 * np.pi / 120) + np.arange(days) * 0.05
    src, out = tmp_path / "c.csv.gz", tmp_path / "odds.json"
    with gzip.open(src, "wt") as f:
        for i, c in enumerate(close):  # um candle por dia, às 00:00 UTC
            f.write(f"{(19000 + i) * 86400},{c},{c},{c},{c},1,{c}\n")
    monkeypatch.setattr(sys, "argv", ["odds", "--file", str(src), "--out", str(out)])
    odds.main()
    text = out.read_text(encoding="utf-8")
    body = json.loads(text)
    assert list(body["events"]) == ["M50 ↑", "M50 ↓", "M100 ↑", "M100 ↓", "M150 ↑", "M150 ↓"] and "M50 ↑" in text
    assert body["from"] == "2022-01-08" and body["to"] == "2023-02-11"
    assert all(body["events"][e]["n"] > 0 for e in body["events"])
    t = body["events"]["M50 ↑"]["trade"]
    assert t["n"] + t["open"] == body["events"]["M50 ↑"]["n"] and 0 <= t["win"] <= 1


def test_committed_json_covers_the_frozen_events():
    body = json.loads(odds.OUT.read_text(encoding="utf-8"))
    assert list(body["events"]) == ["M50 ↑", "M50 ↓", "M100 ↑", "M100 ↓", "M150 ↑", "M150 ↓"]
    for name, e in body["events"].items():
        assert e["n"] > 10 and set(e["after"]) == {"7", "30", "90"}
        assert ("trade" in e) == name.endswith("↑") and ("rebuy" in e) == name.endswith("↓")
