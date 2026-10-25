import math
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from wescalp import paper

UTC = timezone.utc
D0 = date(2026, 1, 1)


def bars(closes: list[float], start: date = D0, full: bool = True) -> list[paper.Bar]:
    out = []
    for i, c in enumerate(closes):
        d = start + timedelta(days=i)
        last = datetime.combine(d, time(23, 55), UTC) if full else datetime.combine(d, time(12, 0), UTC)
        out.append(paper.Bar(d, c, c * 1.01, c * 0.99, c, last))
    return out


def after(d: date, hours: float = 1) -> datetime:
    return datetime.combine(d, time(), UTC) + timedelta(days=1, hours=hours)


def wavy(n: int, seed: int = 0) -> list[float]:
    return [100 + 12 * math.sin(i / (9 + seed)) + 8 * math.sin(i / 37) + i * 0.15 for i in range(n)]


def test_day_is_closed_by_last_candle_or_after_grace():
    b = bars([100.0])[0]
    assert paper.closed(b, datetime.combine(D0, time(0), UTC) + timedelta(days=1))
    partial = bars([100.0], full=False)[0]
    assert not paper.closed(partial, after(D0, 0.25))  # 00:15 do dia seguinte, sem o candle das 23:55
    assert paper.closed(partial, after(D0, 0.6))  # passou a folga de 30 min


def test_frozen_sma_position_equity_and_orders():
    closes = [100.0] * 150 + [130.0, 143.0, 100.0]
    rows, orders = paper.advance(paper.FROZEN, None, bars(closes), D0, after(D0 + timedelta(days=200)))
    assert rows[48].comps[0].sma is None and rows[49].comps[0].sma == pytest.approx(100.0)
    a, b, c = rows[150], rows[151], rows[152]
    assert [x.on for x in a.comps] == [1, 1, 1] and a.position == 1.0 and a.held == 0.0
    assert b.held == 1.0 and b.equity == pytest.approx(a.equity * (1 + 0.1 - paper.FEE))  # comprou: paga a taxa
    assert b.hold == pytest.approx(1.43)
    assert c.position == 0.0 and c.equity == pytest.approx(b.equity * (100 / 143))
    assert [(o.day, o.action, o.size, o.position, o.reason) for o in orders] == [
        (a.day, "compra", 1.0, 1.0, "M50 ↑ M100 ↑ M150 ↑"), (c.day, "venda", 1.0, 0.0, "M50 ↓ M100 ↓ M150 ↓")]
    assert rows[149].position == 0.0  # fechamento igual à média não conta como acima


@pytest.mark.parametrize("spec", paper.STRATEGIES, ids=lambda s: s.name)
def test_incremental_equals_batch_and_stops_at_open_day(spec):
    closes = wavy(320)
    bs = bars(closes)
    start = D0 + timedelta(days=250)
    batch, batch_orders = paper.advance(spec, None, bs, start, after(bs[-1].day))
    rows, orders = paper.advance(spec, None, bs[:200], start, after(bs[199].day))
    more, more_orders = paper.advance(spec, rows[-1], bs, start, after(bs[-1].day))
    assert rows + more == batch and orders + more_orders == batch_orders
    assert [r.live for r in batch] == [r.day >= start for r in batch]
    open_day = bars(closes[:-1]) + bars([closes[-1]], start=bs[-1].day, full=False)
    cut, _ = paper.advance(spec, None, open_day, D0, datetime.combine(bs[-1].day, time(18), UTC))
    assert cut[-1].day == bs[-2].day  # o dia de hoje, ainda aberto, fica de fora


def test_frozen_matches_the_research():
    np = pytest.importorskip("numpy")
    from wescalp import refine, regime

    rng = np.random.default_rng(3)
    closes = list(100 * np.cumprod(1 + rng.normal(0.001, 0.03, 500)))
    rows, _ = paper.advance(paper.FROZEN, None, bars(closes), D0, after(D0 + timedelta(days=600)))
    c = np.array(closes)
    assert np.allclose([r.position for r in rows], refine.position(c, refine.BASE))
    research = regime.returns_from_held(regime.held_from(refine.position(c, refine.BASE), 1), c[1:] / c[:-1] - 1)
    engine = np.array([r.equity for r in rows])
    assert np.allclose(engine[1:] / engine[:-1] - 1, research)


def test_learner_matches_the_research():
    np = pytest.importorskip("numpy")
    from wescalp import learn, regime

    rng = np.random.default_rng(5)
    closes = list(100 * np.cumprod(1 + rng.normal(0.0008, 0.03, 700)))
    rows, _ = paper.advance(paper.LEARNER, None, bars(closes), D0, after(D0 + timedelta(days=800)))
    c = np.array(closes)
    pos, w = learn.learn_positions(learn.comp_positions(c), c)
    assert np.allclose([r.position for r in rows], pos)
    assert np.allclose([[x.weight for x in r.comps] for r in rows], w)
    research = regime.returns_from_held(regime.held_from(pos, 1), c[1:] / c[:-1] - 1)
    engine = np.array([r.equity for r in rows])
    assert np.allclose(engine[1:] / engine[:-1] - 1, research)
    assert len({round(x.weight, 6) for x in rows[-1].comps}) > 1  # aprendeu: pesos diferentes


def test_payload_marks_now_and_splits_history_from_live():
    closes = [100.0] * 150 + [110.0] * 5 + [121.0] * 5
    live_start = D0 + timedelta(days=157)
    bs = bars(closes)
    rows, orders = paper.advance(paper.FROZEN, None, bs, live_start, after(bs[-1].day))
    last = (datetime.combine(bs[-1].day + timedelta(days=1), time(10), UTC), 133.1)
    p = paper.payload(paper.FROZEN, live_start, rows, orders, last)
    assert len(p["days"]) == len(rows) and p["days"][-1]["live"] and not p["days"][0]["live"]
    assert p["days"][-1]["sma"][0] == pytest.approx(rows[-1].comps[0].sma) and p["days"][-1]["w"] is None
    end = rows[-1]
    assert p["now"]["equity"] == pytest.approx(end.equity * 1.1) and p["now"]["position"] == 1.0
    first_live = next(r for r in rows if r.live)
    assert p["live"]["ret"] == pytest.approx(end.equity * 1.1 / first_live.equity - 1)
    assert p["live"]["hold_ret"] == pytest.approx(133.1 / first_live.close - 1)
    assert p["live"]["from"] == paper.day_epoch(live_start) + 86400  # conta da primeira decisão ao vivo
    assert p["history"]["from"] == paper.day_epoch(D0 + timedelta(days=149)) and p["history"]["orders"] == 1
    assert [c["n"] for c in p["components"]] == [50, 100, 150] and p["components"][0]["weight"] == pytest.approx(1 / 3)


def test_learner_payload_has_weights_and_chart_smas():
    bs = bars(wavy(260))
    rows, orders = paper.advance(paper.LEARNER, None, bs, D0 + timedelta(days=240), after(bs[-1].day))
    p = paper.payload(paper.LEARNER, D0 + timedelta(days=240), rows, orders, None)
    assert len(p["days"][-1]["w"]) == 13 and sum(p["days"][-1]["w"]) == pytest.approx(1, abs=1e-3)
    assert p["days"][-1]["sma"][1] == pytest.approx(rows[-1].comps[5].sma)  # a de 100 dias é a 6ª do aprendiz
    assert len(p["components"]) == 13 and all("weight_30d" in c for c in p["components"])


def test_payload_without_rows_or_live_days():
    assert paper.payload(paper.FROZEN, None, [], [], None)["days"] == []
    rows, orders = paper.advance(paper.FROZEN, None, bars([100.0] * 200), D0 + timedelta(days=400),
                                 after(D0 + timedelta(days=300)))
    p = paper.payload(paper.FROZEN, D0 + timedelta(days=400), rows, orders, None)
    assert p["live"] is None and p["now"] is None and p["history"]["ret"] == pytest.approx(0.0)


def test_orders_carry_the_history_of_their_flips(tmp_path):
    odds = paper.load_odds()
    assert list(odds["events"]) == ["M50 ↑", "M50 ↓", "M100 ↑", "M100 ↓", "M150 ↑", "M150 ↓"]
    got = paper.order_odds("M50 ↑ M100 ↑", odds)
    assert [e["event"] for e in got] == ["M50 ↑", "M100 ↑"] and got[1] == {"event": "M100 ↑", **odds["events"]["M100 ↑"]}
    assert [e["event"] for e in paper.order_odds("M60 ↓ M150 ↓", odds)] == ["M150 ↓"]  # as outras do aprendiz não têm
    assert paper.order_odds("pesos", odds) == [] and paper.order_odds("M50 ↑", None) == []
    rows, orders = paper.advance(paper.FROZEN, None, bars([100.0] * 150 + [130.0, 143.0, 100.0]), D0,
                                 after(D0 + timedelta(days=200)))
    p = paper.payload(paper.FROZEN, D0, rows, orders, None, odds)
    assert [[e["event"] for e in o["odds"]] for o in p["orders"]] == [
        ["M50 ↑", "M100 ↑", "M150 ↑"], ["M50 ↓", "M100 ↓", "M150 ↓"]]
    assert p["orders"][0]["odds"][0]["trade"] == odds["events"]["M50 ↑"]["trade"]
    assert all(o["odds"] == [] for o in paper.payload(paper.FROZEN, D0, rows, orders, None)["orders"])
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert paper.load_odds(tmp_path / "bad.json") is None and paper.load_odds(tmp_path / "none.json") is None


def test_paper_reads_the_odds_without_numpy():
    code = "import sys; from wescalp import paper; assert paper.load_odds(); assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(paper.__file__).resolve().parents[1])


def test_detail_json_round_trip():
    rows, _ = paper.advance(paper.LEARNER, None, bars(wavy(200)), D0, after(D0 + timedelta(days=300)))
    r = rows[-1]
    back = paper._row((r.day, r.open, r.high, r.low, r.close, r.held, r.position, r.equity, r.hold, r.live,
                       paper.comps_json(r.comps)))
    assert back == r


def test_drawdown():
    assert paper.drawdown([1.0, 1.2, 0.9, 1.3, 1.04]) == pytest.approx(0.25)
    assert paper.drawdown([]) == 0.0
