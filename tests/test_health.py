import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import httpx

from wescalp.health import (
    DOWN, GB, LATE, OK, Alert, Limits, Metrics, Notifier, Report, WatchState, alert_text, decide, evaluate,
    fmt_age, fmt_bytes, fmt_int, gap_runs, hourly, unreachable, watch,
)

NOW = datetime(2026, 10, 22, 12, 0, 30, tzinfo=timezone.utc)
NOW_MIN = int(NOW.timestamp()) // 60
LIM = Limits()


def healthy(**kw) -> Metrics:
    """Coletor rodando há mais de 24 h, sem buracos, ~10 topos de livro por segundo."""
    minutes = range(NOW_MIN - 24 * 60, NOW_MIN + 1)
    base = dict(
        now=NOW,
        last_book=NOW - timedelta(seconds=0.5),
        last_trade=NOW - timedelta(seconds=3),
        book_per_minute={m: 600 for m in minutes},
        trades_per_minute={m: 50 for m in minutes},
        snapshots_per_minute={m: 60 for m in minutes},
        book_before_window=True,
        db_bytes=2 * GB,
        table_bytes={"book_l1": GB, "candles": 30 * 1024**2},
        disk_free_bytes=50 * GB,
        disk_total_bytes=100 * GB,
    )
    base.update(kw)
    return Metrics(**base)


def test_healthy_is_ok():
    r = evaluate(healthy(), LIM)
    assert r.state == OK and r.problems == ()
    text = r.text()
    assert "OK" in text and "36.000 na última hora" in text and "0 min sem livro em 24 h" in text
    assert "50 GB livres de 100 GB (50%)" in text


def test_empty_tables_are_down_without_exception():
    r = evaluate(Metrics(now=NOW), LIM)
    assert r.state == DOWN
    assert "livro: nada gravado" in r.problems and "negócios: nada gravado" in r.problems
    assert "livro (book_l1): nada gravado" in r.text()


def test_book_age_thresholds():
    assert evaluate(healthy(last_book=NOW - timedelta(seconds=20)), LIM).state == LATE
    r = evaluate(healthy(last_book=NOW - timedelta(seconds=61)), LIM)
    assert r.state == DOWN and r.problems[0] == "livro parado: último há 1 min 01 s (limite 1 min 00 s)"


def test_trade_age_thresholds():
    assert evaluate(healthy(last_trade=NOW - timedelta(seconds=119)), LIM).state == OK
    assert evaluate(healthy(last_trade=NOW - timedelta(seconds=121)), LIM).state == LATE
    assert evaluate(healthy(last_trade=NOW - timedelta(minutes=11)), LIM).state == DOWN


def test_worst_state_wins_and_comes_first():
    r = evaluate(healthy(last_book=NOW - timedelta(seconds=20), last_trade=NOW - timedelta(hours=1)), LIM)
    assert r.state == DOWN and r.problems[0].startswith("negócios parado")


def test_limits_from_env():
    lim = Limits.from_env({"WESCALP_HEALTH_BOOK_STALE_S": "90", "WESCALP_HEALTH_GAP_MINUTES": "3",
                           "WESCALP_HEALTH_DB_WARN_GB": ""})
    assert lim.book_stale_s == 90.0 and lim.gap_minutes == 3 and lim.db_warn_gb == 40.0


def test_gap_runs():
    assert gap_runs({1, 2, 5, 9}, 0, 10) == [(0, 1), (3, 2), (6, 3)]
    assert gap_runs(set(range(10)), 0, 10) == []
    assert gap_runs(set(), 0, 3) == [(0, 3)]


def test_gaps_over_limit_are_late():
    book = healthy().book_per_minute.copy()
    for m in range(NOW_MIN - 120, NOW_MIN - 100):  # 20 min sem livro há 2 h
        del book[m]
    r = evaluate(healthy(book_per_minute=book), LIM)
    assert r.state == LATE
    assert r.problems == ("20 min sem livro nas últimas 24 h; maior buraco: 20 min a partir de 22/10 10:00 UTC",)


def test_small_gap_is_ok_but_reported():
    book = healthy().book_per_minute.copy()
    del book[NOW_MIN - 30]
    r = evaluate(healthy(book_per_minute=book), LIM)
    assert r.state == OK and "1 min sem livro em 24 h" in r.text()


def test_gaps_only_count_since_collector_started():
    # Coletor ligou há 30 min: as 23,5 h anteriores não são buraco.
    start = NOW_MIN - 30
    m = healthy(book_per_minute={x: 600 for x in range(start, NOW_MIN + 1)}, book_before_window=False)
    r = evaluate(m, LIM)
    assert r.state == OK and "0 min sem livro desde 22/10 11:30 UTC" in r.text()


def test_ongoing_outage_flagged():
    book = {m: 600 for m in range(NOW_MIN - 24 * 60, NOW_MIN - 20)}
    r = evaluate(healthy(book_per_minute=book, last_book=NOW - timedelta(minutes=20, seconds=30)), LIM)
    assert r.state == DOWN
    assert any("maior buraco: 20 min" in p and p.endswith("(em andamento)") for p in r.problems)


def test_thin_book_last_hour_is_late():
    book = {m: (10 if m >= NOW_MIN - 60 else 600) for m in range(NOW_MIN - 24 * 60, NOW_MIN + 1)}
    r = evaluate(healthy(book_per_minute=book), LIM)
    assert r.state == LATE and r.problems == ("livro ralo: 600 linhas na última hora (mínimo 3.600)",)


def test_disk_and_db_size():
    assert evaluate(healthy(disk_free_bytes=4 * GB), LIM).state == LATE
    r = evaluate(healthy(disk_free_bytes=GB // 2), LIM)
    assert r.state == DOWN and r.problems[0].startswith("disco quase cheio: 512 MB livres")
    assert evaluate(healthy(db_bytes=41 * GB), LIM).state == LATE
    # Sem caminho de disco configurado, não mede e não reclama.
    assert evaluate(healthy(disk_free_bytes=None, disk_total_bytes=None), LIM).state == OK


def test_hourly():
    assert hourly({60: 1, 61: 2, 120: 5}) == {1: 3, 2: 5}


def test_formatting():
    assert fmt_int(1234567) == "1.234.567"
    assert fmt_age(0.25) == "0,2 s" and fmt_age(42) == "42 s" and fmt_age(312) == "5 min 12 s"
    assert fmt_age(3 * 3600 + 5 * 60) == "3 h 05 min" and fmt_age(2 * 86400 + 4 * 3600) == "2 d 4 h"
    assert fmt_bytes(8192) == "8,0 KB" and fmt_bytes(126 * 1024**2) == "126 MB" and fmt_bytes(int(1.4 * GB)) == "1,4 GB"


def test_unreachable():
    r = unreachable(OSError("conexão recusada\ndetalhe"), NOW)
    assert r.state == DOWN and r.problems == ("sem acesso ao banco: conexão recusada",)


# --- transições ---

H = 3600.0


def test_first_check_ok_is_silent_and_bad_alerts():
    assert decide(WatchState(), OK, 0, 6 * H) == (WatchState(OK), None)
    ws, alert = decide(WatchState(), DOWN, 10, 6 * H)
    assert alert == Alert("mudou", None, DOWN, 0.0) and ws == WatchState(DOWN, 10, 10)


def test_transitions_and_repeat():
    ws, alert = decide(WatchState(OK), OK, 0, 6 * H)
    assert alert is None
    ws, alert = decide(ws, DOWN, 100, 6 * H)
    assert alert == Alert("mudou", OK, DOWN, 0.0)
    ws, alert = decide(ws, DOWN, 100 + 6 * H - 1, 6 * H)
    assert alert is None
    ws, alert = decide(ws, DOWN, 100 + 6 * H, 6 * H)
    assert alert == Alert("repete", DOWN, DOWN, 6 * H)
    ws, alert = decide(ws, DOWN, 100 + 7 * H, 6 * H)
    assert alert is None  # a repetição conta a partir do último aviso
    ws, alert = decide(ws, OK, 100 + 8 * H, 6 * H)
    assert alert == Alert("mudou", DOWN, OK, 8 * H) and ws.bad_since is None
    ws, alert = decide(ws, OK, 100 + 20 * H, 6 * H)
    assert alert is None


def test_bad_to_worse_keeps_bad_since():
    ws, _ = decide(WatchState(OK), LATE, 0, 6 * H)
    ws, alert = decide(ws, DOWN, 600, 6 * H)
    assert alert == Alert("mudou", LATE, DOWN, 600) and ws.bad_since == 0
    ws, alert = decide(ws, OK, 1200, 6 * H)
    assert alert.bad_for == 1200


def test_repeat_zero_never_repeats():
    ws, _ = decide(WatchState(OK), DOWN, 0, 0)
    assert decide(ws, DOWN, 100 * H, 0)[1] is None


def test_alert_text():
    report = Report(DOWN, ("livro parado: último há 5 min 00 s (limite 1 min 00 s)",), ("banco: 1 GB",), NOW)
    t = alert_text(Alert("mudou", OK, DOWN, 0), report)
    assert t.splitlines()[:4] == [
        "[Wescalp] PARADO (antes: ok)",
        "Wescalp XBTUSDTM: PARADO em 22/10 12:00 UTC",
        "Problemas:",
        "- livro parado: último há 5 min 00 s (limite 1 min 00 s)",
    ]
    assert alert_text(Alert("mudou", None, DOWN, 0), report).startswith("[Wescalp] PARADO (início do vigia)")
    assert alert_text(Alert("repete", DOWN, DOWN, 6 * H), report).startswith("[Wescalp] continua PARADO há 6 h 00 min")
    ok = Report(OK, (), ("banco: 1 GB",), NOW)
    assert alert_text(Alert("mudou", DOWN, OK, 4320), ok).startswith("[Wescalp] voltou ao normal (ficou 1 h 12 min parado)")
    long = Report(DOWN, tuple("x" * 100 for _ in range(50)), (), NOW)
    assert len(alert_text(Alert("mudou", OK, DOWN, 0), long)) == 1900


# --- envio, sem rede ---

def run(coro):
    return asyncio.run(coro)


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_webhook_and_telegram_payloads():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"ok": True})

    n = Notifier("https://hooks.example/abc", "123:TOKEN", "-100", client=mock_client(handler))
    assert run(n.send("oi")) == 2
    assert seen == [
        ("https://hooks.example/abc", {"text": "oi"}),
        ("https://api.telegram.org/bot123:TOKEN/sendMessage",
         {"chat_id": "-100", "text": "oi", "disable_web_page_preview": True}),
    ]


def test_telegram_needs_token_and_chat():
    n = Notifier(None, "123:TOKEN", None, client=mock_client(lambda r: httpx.Response(200)))
    assert n.targets("x") == []


def test_no_channel_only_logs(caplog):
    n = Notifier(client=mock_client(lambda r: (_ for _ in ()).throw(AssertionError("não devia chamar"))))
    with caplog.at_level(logging.WARNING, logger="wescalp.health"):
        assert run(n.send("socorro")) == 0
    assert "socorro" in caplog.text


def test_send_failures_do_not_raise_and_hide_secrets(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if "telegram" in request.url.host:
            raise httpx.ConnectError("sem rede", request=request)
        return httpx.Response(500)

    n = Notifier("https://hooks.example/SEGREDO", "123:TOKEN", "-100", client=mock_client(handler))
    with caplog.at_level(logging.ERROR, logger="wescalp.health"):
        assert run(n.send("oi")) == 0
    assert "falha ao avisar por webhook" in caplog.text and "falha ao avisar por telegram" in caplog.text
    assert "SEGREDO" not in caplog.text and "TOKEN" not in caplog.text


# --- loop do --watch ---

class FakeNotifier:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    async def send(self, text):
        self.sent.append(text)
        if self.fail:
            raise RuntimeError("canal fora do ar")
        return 1


def scripted(states, clock_step=60.0):
    """run_check que devolve os estados em sequência e um relógio que anda clock_step por chamada."""
    it = iter(states)
    t = [0.0]

    async def run_check():
        s = next(it)
        if s == "erro":
            raise RuntimeError("banco caiu no meio")
        return Report(s, () if s == OK else ("algo",), (), NOW)

    def clock():
        t[0] += clock_step
        return t[0]

    async def no_sleep(_):
        pass

    return run_check, clock, no_sleep


def test_watch_alerts_only_on_change():
    states = [OK, OK, DOWN, DOWN, DOWN, OK, OK]
    run_check, clock, sleep = scripted(states)
    fake = FakeNotifier()
    run(watch(run_check, fake, 60, 6 * H, clock=clock, sleep=sleep, iterations=len(states)))
    assert [t.splitlines()[0] for t in fake.sent] == [
        "[Wescalp] PARADO (antes: ok)", "[Wescalp] voltou ao normal (ficou 3 min 00 s parado)"]


def test_watch_repeats_while_bad():
    run_check, clock, sleep = scripted([DOWN] * 4, clock_step=3 * H)
    fake = FakeNotifier()
    run(watch(run_check, fake, 60, 6 * H, clock=clock, sleep=sleep, iterations=4))
    assert [t.splitlines()[0] for t in fake.sent] == [
        "[Wescalp] PARADO (início do vigia)", "[Wescalp] continua PARADO há 6 h 00 min"]


def test_watch_survives_check_error_and_send_failure():
    run_check, clock, sleep = scripted([OK, "erro", OK])
    fake = FakeNotifier(fail=True)
    ws = run(watch(run_check, fake, 60, 6 * H, clock=clock, sleep=sleep, iterations=3))
    assert ws.state == OK and len(fake.sent) == 2
    assert "sem acesso ao banco: banco caiu no meio" in fake.sent[0]
