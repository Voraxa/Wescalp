import asyncio
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from starlette.testclient import TestClient

from wescalp import health, paper, web
from wescalp.flow import Quote, Tick
from wescalp.kucoin import Candle
from wescalp.labels import Barriers

UTC = timezone.utc
T0 = datetime(2026, 6, 1, tzinfo=UTC)
STEP = timedelta(minutes=5)


def candle(t: datetime, o: float, h: float, l: float, c: float, v: float = 10.0) -> Candle:
    return Candle("futures", "XBTUSDTM", t, o, h, l, c, v, v * c * 0.001)


def wave(n: int, start: datetime = T0) -> list[Candle]:
    """Candles contínuos com subida e descida, para os setups dispararem e as barreiras serem tocadas."""
    out, px = [], 100_000.0
    for i in range(n):
        step = 60 if (i // 30) % 2 == 0 else -60
        o, c = px, px + step
        vol = 10 + (i % 7) * 5
        out.append(candle(start + i * STEP, o, max(o, c) + 40, min(o, c) - 40, c, vol))
        px = c
    return out


class FakeStore:
    """Mesmos métodos do web.Store, em memória."""

    def __init__(self, candles=(), trades=(), quotes=(), ticks=(), metrics=None, reports=None, paper=None):
        self._candles = list(candles)
        self._trades = list(trades)
        self._quotes = list(quotes)
        self._ticks = list(ticks)
        self._metrics = metrics
        self._reports = reports  # None = tabela não existe
        self._paper = paper  # None = tabelas da carteira simulada não existem
        self.flow_lock = asyncio.Semaphore(1)
        self.calls: list[tuple] = []

    async def open(self):
        pass

    async def close(self):
        pass

    async def candle_span(self, source):
        if not self._candles:
            return None, None
        return self._candles[0].ts, self._candles[-1].ts

    async def candles(self, source, since, until):
        self.calls.append(("candles", since, until))
        return [c for c in self._candles if since <= c.ts < until]

    async def trade_rows(self, since, until):
        self.calls.append(("trade_rows", since, until))
        return [r for r in self._trades if since.timestamp() <= r[0] < until.timestamp()]

    async def quotes(self, since, until):
        return [q for q in self._quotes if since.timestamp() <= q.ts < until.timestamp()]

    async def ticks(self, since, until):
        return [t for t in self._ticks if since.timestamp() <= t.ts < until.timestamp()]

    async def metrics(self):
        if self._metrics is None:
            raise psycopg.OperationalError("connection refused")
        return self._metrics

    async def old_book_minutes(self, start_min, end_min):
        return {}, False

    async def paper(self):
        return self._paper

    async def reports(self):
        if self._reports is None:
            raise psycopg.errors.UndefinedTable('relation "reports" does not exist')
        return self._reports


def client(store) -> TestClient:
    return TestClient(web.create_app(store, health.Limits()))


# --- candle em formação ---

def test_trade_candles_build_ohlc_per_bucket_and_ignore_outside():
    start = datetime(2026, 10, 22, 20, 0, tzinfo=UTC)
    s = start.timestamp()
    rows = [
        (s - 1, 1.0, 9.0),  # antes do início: fora
        (s + 10, 100.0, 1.0), (s + 20, 103.0, 2.0), (s + 200, 99.0, 1.0), (s + 299, 101.0, 3.0),
        (s + 300, 105.0, 1.0), (s + 400, 104.0, 1.0),
        (s + 600, 1.0, 1.0),  # depois do fim: fora
    ]
    cs = web.trade_candles(rows, start, start + 2 * STEP)
    assert [c.ts for c in cs] == [start, start + STEP]
    a, b = cs
    assert (a.open, a.high, a.low, a.close, a.volume) == (100.0, 103.0, 99.0, 101.0, 7.0)
    assert a.turnover == pytest.approx((100 + 206 + 99 + 303) * 0.001)
    assert (b.open, b.high, b.low, b.close, b.volume) == (105.0, 105.0, 104.0, 104.0, 2.0)
    assert web.trade_candles([], start, start + STEP) == []


def test_candles_endpoint_appends_forming_candle_from_trades():
    stored = wave(400, web.floor_candle(datetime.now(UTC)) - 600 * STEP)
    stored = [c for c in stored if c.ts < web.floor_candle(datetime.now(UTC)) - 2 * STEP]
    last = stored[-1].ts
    now_bucket = web.floor_candle(datetime.now(UTC))
    nb = now_bucket.timestamp()
    trades = [
        ((last + STEP).timestamp() + 5, 200_000.0, 1.0),  # candle fechado montado de negócios
        (nb + 0.5, 200_100.0, 2.0), (nb + 1, 200_300.0, 1.0), (nb + 1.5, 200_050.0, 4.0),  # em formação
    ]
    r = client(FakeStore(stored, trades)).get("/api/candles")
    assert r.status_code == 200
    body = r.json()
    cs = body["candles"]
    assert cs[-1] == [int(nb), 200_100.0, 200_300.0, 200_050.0, 200_050.0, 7.0]
    assert body["forming"] == int(nb)
    assert body["live_from"] == int((last + STEP).timestamp())
    assert body["last_stored"] == int(last.timestamp())
    assert len(body["vwap"]) == len(body["ema9"]) == len(cs)
    assert body["vwap"][-1] is not None and body["vwap_up2"][-1] >= body["vwap_up1"][-1] >= body["vwap"][-1]


def test_candles_without_trades_has_no_forming_candle():
    stored = wave(300, web.floor_candle(datetime.now(UTC)) - 400 * STEP)
    body = client(FakeStore(stored)).get("/api/candles").json()
    assert body["forming"] is None and body["live_from"] is None
    assert body["candles"][-1][0] == int(stored[-1].ts.timestamp())


def test_candles_window_is_clipped_and_loads_warmup_from_previous_day():
    stored = wave(4000)
    store = FakeStore(stored)
    to = stored[-1].ts + STEP
    body = client(store).get("/api/candles", params={"from": 0, "to": int(to.timestamp())}).json()
    assert body["clipped"] and len(body["candles"]) == web.MAX_CANDLES
    since = store.calls[0][1]
    assert since == web.warmup_start(datetime.fromtimestamp(body["from"], tz=UTC))
    assert since.hour == 0 and since < datetime.fromtimestamp(body["from"], tz=UTC) - timedelta(days=1) + STEP


def test_candles_bad_params():
    c = client(FakeStore(wave(10)))
    assert c.get("/api/candles", params={"source": "x"}).status_code == 400
    assert c.get("/api/candles", params={"from": "ontem"}).status_code == 400
    assert c.get("/api/candles", params={"from": 200, "to": 100}).status_code == 400


# --- marcas e período intocado ---

def test_holdout_start_is_16_weeks_before_last_candle():
    last = datetime(2026, 10, 22, 19, 45, tzinfo=UTC)
    assert web.holdout_start(last) == last - timedelta(weeks=16)
    assert web.holdout_start(None) is None


def test_marks_payload_cuts_holdout_unless_asked():
    cs = wave(2000)
    start, end = cs[0].ts, cs[-1].ts + STEP
    cutoff = cs[1000].ts
    b = Barriers()
    full = web.marks_payload(cs, b, start, end, cutoff, include_holdout=True)["marks"]
    cut = web.marks_payload(cs, b, start, end, cutoff, include_holdout=False)
    assert any(m["ts"] >= cutoff.timestamp() for m in full)
    assert cut["marks"] and all(m["exit_ts"] < cutoff.timestamp() for m in cut["marks"])
    assert cut["holdout"] == {"from": int(cutoff.timestamp()), "weeks": 16, "included": False}
    # Antes do corte, as marcas são as mesmas (a não ser as que só saíam dentro do período intocado).
    before = [m for m in full if m["exit_ts"] < cutoff.timestamp()]
    assert cut["marks"] == before


def test_marks_have_entry_exit_and_net_return():
    cs = wave(1500)
    marks = web.marks_payload(cs, Barriers(), cs[300].ts, cs[-1].ts, None, False)["marks"]
    assert {m["setup"] for m in marks} <= {"breakout", "vwap_reversion", "pullback"}
    for m in marks:
        assert m["ts"] >= cs[300].ts.timestamp()
        assert m["outcome"] in ("tp", "sl", "time") and m["side"] in ("long", "short")
        assert m["exit_ts"] == m["ts"] + m["bars"] * 300
        sign = 1 if m["side"] == "long" else -1
        assert m["ret_gross_bps"] == pytest.approx(sign * (m["exit_px"] / m["entry_px"] - 1) * 1e4, abs=0.02)
        assert m["ret_net_bps"] < m["ret_gross_bps"]


def test_marks_endpoint_hides_holdout_by_default():
    cs = wave(40000)  # ~139 dias
    last = cs[-1].ts
    cutoff = last - timedelta(weeks=16)
    c = client(FakeStore(cs))
    params = {"from": int((cutoff - 1500 * STEP).timestamp()), "to": int((cutoff + 1500 * STEP).timestamp())}
    hidden = c.get("/api/marks", params=params).json()
    shown = c.get("/api/marks", params=params | {"holdout": 1}).json()
    assert hidden["holdout"]["from"] == int(cutoff.timestamp()) and not hidden["holdout"]["included"]
    assert hidden["marks"] and all(m["ts"] < cutoff.timestamp() for m in hidden["marks"])
    assert any(m["ts"] >= cutoff.timestamp() for m in shown["marks"]) and shown["holdout"]["included"]


def test_marks_endpoint_validates_barriers():
    c = client(FakeStore(wave(100)))
    assert c.get("/api/marks", params={"tp": "0"}).status_code == 400
    assert c.get("/api/marks", params={"bars": "abc"}).status_code == 400


# --- relatórios ---

def test_reports_without_table_is_empty():
    r = client(FakeStore()).get("/api/reports")
    assert r.status_code == 200 and r.json() == {"reports": []}


def test_reports_returns_latest_rows():
    created = datetime(2026, 10, 22, 21, 0, tzinfo=UTC)
    body = {"rows": [{"setup": "pullback", "side": "long", "net_bps": -5.6}]}
    r = client(FakeStore(reports=[("research", created, body)])).get("/api/reports").json()
    assert r == {"reports": [{"name": "research", "created_at": created.isoformat(), "body": body}]}


# --- estado e buracos ---

NOW = datetime(2026, 10, 22, 12, 0, 30, tzinfo=UTC)
NOW_MIN = int(NOW.timestamp()) // 60


def metrics(**kw) -> health.Metrics:
    minutes = set(range(NOW_MIN - 24 * 60, NOW_MIN + 1)) - set(range(NOW_MIN - 100, NOW_MIN - 80))
    base = dict(now=NOW, last_book=NOW - timedelta(seconds=0.5), last_trade=NOW - timedelta(seconds=3),
                book_per_minute={m: 600 for m in minutes}, trades_per_minute={m: 5 for m in minutes},
                snapshots_per_minute={m: 60 for m in minutes}, book_before_window=True,
                db_bytes=2 * health.GB, table_bytes={"book_l1": health.GB}, disk_free_bytes=50 * health.GB,
                disk_total_bytes=100 * health.GB)
    base.update(kw)
    return health.Metrics(**base)


def test_health_endpoint_reuses_vigia_state():
    body = client(FakeStore(metrics=metrics())).get("/api/health").json()
    assert body["state"] == health.LATE  # 20 min sem livro passa do limite de 15
    assert body["gap_minutes_24h"] == 20
    assert body["gaps_24h"] == [[(NOW_MIN - 100) * 60, (NOW_MIN - 80) * 60]]
    assert body["book_age_s"] == 0.5 and body["text"]["db"] == "2,0 GB" and body["text"]["disk_free"] == "50 GB"
    assert body["last_hour"]["book_l1"] == 36000
    assert body["details"] == list(health.evaluate(metrics(), health.Limits()).details)


def test_health_endpoint_when_db_is_down():
    body = client(FakeStore(metrics=None)).get("/api/health").json()
    assert body["state"] == health.DOWN and body["problems"][0].startswith("sem acesso ao banco")


def test_gaps_endpoint():
    body = client(FakeStore(metrics=metrics())).get("/api/gaps", params={"hours": 24}).json()
    assert body["gaps"] == [[(NOW_MIN - 100) * 60, (NOW_MIN - 80) * 60]] and body["minutes"] == 20
    assert client(FakeStore(metrics=metrics())).get("/api/gaps", params={"hours": 500}).status_code == 400


def test_gap_intervals_start_at_first_book_when_collector_is_new():
    pm = {m: 1 for m in (10, 11, 15)}
    assert web.gap_intervals(pm, False, True, 0, 18) == [[12 * 60, 15 * 60], [16 * 60, 18 * 60]]
    assert web.gap_intervals({}, False, False, 0, 18) == []


# --- fluxo ---

def test_flow_endpoint_series():
    t0 = 1_800_000_000.0
    quotes = [Quote(t0 + i * 0.5, 100.0, 1.0 + i % 3, 100.1, 2.0, 10.0, 5.0) for i in range(200)]
    ticks = [Tick(t0 + 3, "buy", 2.0), Tick(t0 + 4, "sell", 0.5)]
    c = client(FakeStore(quotes=quotes, ticks=ticks))
    body = c.get("/api/flow", params={"from": t0, "to": t0 + 100, "bucket": 10}).json()
    assert body["bucket"] == 10 and body["t"][0] == t0 + 10 and len(body["t"]) == len(body["ofi"])
    assert body["imb5"][1] == pytest.approx(1 / 3, abs=1e-4) and body["delta"][0] is None
    assert body["mid"][1] == pytest.approx(100.05)


def test_flow_window_is_clipped():
    body = client(FakeStore()).get("/api/flow", params={"from": 0, "to": 100_000, "bucket": 1}).json()
    assert body["clipped"] and body["to"] - body["from"] == 3 * 3600  # 1 s × 10.800 baldes
    body = client(FakeStore()).get("/api/flow", params={"from": 0, "to": 100_000, "bucket": 10}).json()
    assert body["to"] - body["from"] == 6 * 3600


def test_static_index_is_served():
    r = client(FakeStore()).get("/")
    assert r.status_code == 200 and "Wescalp" in r.text and r.headers["cache-control"] == "no-cache"


# --- carteira simulada ---

def test_paper_without_tables_is_empty():
    body = client(FakeStore()).get("/api/paper").json()
    assert [x["id"] for x in body["strategies"]] == [paper.FROZEN.id, paper.LEARNER.id]
    assert all(x["days"] == [] and x["live"] is None for x in body["strategies"])
    assert body["odds"]["from"] == "2017-08-17" and "Binance" in body["odds"]["source"] and "events" not in body["odds"]


def test_paper_endpoint_has_both_portfolios_and_marks_now():
    from datetime import date, time as dtime

    d0 = date(2026, 3, 1)
    bars = [paper.Bar(d0 + timedelta(days=i), 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i,
                      datetime.combine(d0 + timedelta(days=i), dtime(23, 55), UTC)) for i in range(200)]
    live_start = d0 + timedelta(days=190)
    now = datetime.combine(d0 + timedelta(days=201), dtime(1), UTC)
    data = {s.id: paper.advance(s, None, bars, live_start, now) for s in paper.STRATEGIES}
    meta = {s.id: live_start for s in paper.STRATEGIES}
    last = (datetime.combine(d0 + timedelta(days=200), dtime(10), UTC), 330.0)
    body = client(FakeStore(paper=(meta, data, last))).get("/api/paper").json()
    frozen, learner = body["strategies"]
    assert frozen["name"] == "congelada" and learner["name"] == "aprendiz" and learner["learn"]
    assert len(frozen["days"]) == 200 and frozen["live_start"] == live_start.isoformat()
    assert frozen["now"]["price"] == 330.0 and frozen["now"]["position"] == 1.0
    assert frozen["live"]["days"] == 9
    assert frozen["history"]["orders"] == 1  # só a M150 liga dentro do histórico completo (dia 149)
    assert learner["history"]["from"] == paper.day_epoch(d0 + timedelta(days=169))  # quando a média de 170 existe
    # cada ordem leva o histórico das viradas dela; as médias do aprendiz fora de 50, 100 e 150 não têm
    assert [(o["reason"], [e["event"] for e in o["odds"]]) for o in frozen["orders"]] == [
        ("M50 ↑", ["M50 ↑"]), ("M100 ↑", ["M100 ↑"]), ("M150 ↑", ["M150 ↑"])]
    assert frozen["orders"][0]["odds"][0]["n"] > 10 and "trade" in frozen["orders"][0]["odds"][0]
    by_reason = {o["reason"]: [e["event"] for e in o["odds"]] for o in learner["orders"]}
    assert by_reason["M50 ↑"] == ["M50 ↑"] and by_reason["M60 ↑"] == []
