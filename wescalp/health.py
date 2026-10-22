"""Vigia dos dados: o coletor está gravando, o banco aceita escrita, o disco aguenta?

    python -m wescalp.health                # resumo; sai 0 se ok, 1 se atrasado, 2 se parado
    python -m wescalp.health --horas        # idem, com as linhas por hora das últimas 24 h
    python -m wescalp.health --watch        # loop; avisa quando o estado muda e repete enquanto estiver ruim
    python -m wescalp.health --test-alert   # manda um aviso de teste pelos canais configurados

Avisos: WESCALP_ALERT_WEBHOOK (POST {"text": ...}; Slack, Discord com /slack no fim da URL, etc.) e/ou
WESCALP_TELEGRAM_TOKEN + WESCALP_TELEGRAM_CHAT_ID. Sem nenhum, o aviso só vai para o log.
Só lê o banco (sessão read-only) e não fala com a KuCoin. Limiares em Limits (WESCALP_HEALTH_*).
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone

import httpx
import psycopg
from psycopg import sql

from .config import DATABASE_URL, FUTURES_SYMBOL

log = logging.getLogger("wescalp.health")

OK, LATE, DOWN = "ok", "atrasado", "parado"
RANK = {OK: 0, LATE: 1, DOWN: 2}
EXIT_CODES = RANK
WINDOW_MIN = 24 * 60
MAX_ALERT_CHARS = 1900  # Discord corta em 2000 caracteres; Telegram em 4096.
GB = 1024**3
TABLES = ("book_l1", "trades", "book_snapshots")


@dataclass(frozen=True)
class Limits:
    """Limiares. Cada um pode vir do ambiente como WESCALP_HEALTH_<NOME>, ex.: WESCALP_HEALTH_BOOK_STALE_S=90."""

    book_late_s: float = 15.0  # topo do livro mais velho que isso = atrasado
    book_stale_s: float = 60.0  # ... e que isso = parado
    trade_late_s: float = 120.0  # negócios são mais espaçados que o livro
    trade_stale_s: float = 600.0
    gap_minutes: int = 15  # minutos sem livro nas últimas 24 h acima disso = atrasado
    book_min_per_hour: int = 3600  # livro na última hora abaixo disso = atrasado (normal: ~36 mil)
    db_warn_gb: float = 40.0  # banco acima disso = atrasado
    disk_warn_gb: float = 5.0  # disco livre abaixo disso = atrasado
    disk_crit_gb: float = 1.0  # ... e disso = parado (o Postgres para de gravar com o disco cheio)

    @classmethod
    def from_env(cls, env=os.environ) -> "Limits":
        kw = {}
        for f in fields(cls):
            raw = env.get(f"WESCALP_HEALTH_{f.name.upper()}", "")
            if raw.strip():
                kw[f.name] = type(f.default)(raw)
        return cls(**kw)


@dataclass(frozen=True)
class Metrics:
    """Números lidos do banco. Contagens por minuto: chave = época em minutos (segundos // 60), só 24 h."""

    now: datetime
    last_book: datetime | None = None
    last_trade: datetime | None = None
    book_per_minute: dict[int, int] = field(default_factory=dict)
    trades_per_minute: dict[int, int] = field(default_factory=dict)
    snapshots_per_minute: dict[int, int] = field(default_factory=dict)
    book_before_window: bool = False  # já havia livro antes da janela de 24 h
    db_bytes: int = 0
    table_bytes: dict[str, int] = field(default_factory=dict)
    disk_free_bytes: int | None = None
    disk_total_bytes: int | None = None


@dataclass(frozen=True)
class Report:
    state: str  # ok | atrasado | parado
    problems: tuple[str, ...]
    details: tuple[str, ...]
    now: datetime
    symbol: str = FUTURES_SYMBOL

    def headline(self) -> str:
        return f"Wescalp {self.symbol}: {self.state.upper()} em {self.now.astimezone(timezone.utc):%d/%m %H:%M} UTC"

    def text(self) -> str:
        lines = [self.headline()]
        if self.problems:
            lines += ["Problemas:"] + [f"- {p}" for p in self.problems]
        if self.details:
            lines += ["Resumo:"] + [f"- {d}" for d in self.details]
        return "\n".join(lines)


# --- formatação (pt-BR) ---

def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_dec(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def fmt_age(seconds: float) -> str:
    s = max(seconds, 0)
    if s < 10:
        return f"{fmt_dec(s)} s"
    if s < 60:
        return f"{int(s)} s"
    if s < 3600:
        return f"{int(s // 60)} min {int(s % 60):02d} s"
    if s < 86400:
        return f"{int(s // 3600)} h {int(s % 3600 // 60):02d} min"
    return f"{int(s // 86400)} d {int(s % 86400 // 3600)} h"


def fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024:
            return f"{x:.0f} {unit}" if unit == "B" or x >= 10 else f"{fmt_dec(x)} {unit}"
        x /= 1024
    return f"{fmt_dec(x)} TB"


def fmt_minute(m: int) -> str:
    return f"{datetime.fromtimestamp(m * 60, tz=timezone.utc):%d/%m %H:%M}"


# --- lógica pura ---

def gap_runs(present, start: int, end: int) -> list[tuple[int, int]]:
    """Sequências de minutos ausentes em [start, end): (primeiro minuto, duração em minutos)."""
    runs: list[tuple[int, int]] = []
    run_start = None
    for m in range(start, end):
        if m in present:
            if run_start is not None:
                runs.append((run_start, m - run_start))
                run_start = None
        elif run_start is None:
            run_start = m
    if run_start is not None:
        runs.append((run_start, end - run_start))
    return runs


def hourly(per_minute: dict[int, int]) -> dict[int, int]:
    """Soma as contagens por minuto em horas cheias (chave = época em horas, UTC)."""
    out: dict[int, int] = {}
    for m, n in per_minute.items():
        out[m // 60] = out.get(m // 60, 0) + n
    return out


def last_hour(per_minute: dict[int, int], now_min: int) -> int:
    """Linhas nos 60 minutos fechados antes de now_min."""
    return sum(n for m, n in per_minute.items() if now_min - 60 <= m < now_min)


def worst(states) -> str:
    return max(states, key=RANK.__getitem__, default=OK)


def evaluate(m: Metrics, lim: Limits, symbol: str = FUTURES_SYMBOL) -> Report:
    """Decide o estado a partir das métricas. Nunca levanta exceção, nem com o banco vazio."""
    found: list[tuple[str, str]] = []
    details: list[str] = []
    now_s = m.now.timestamp()
    now_min = int(now_s // 60)  # o minuto corrente ainda está incompleto e fica de fora

    def age_check(name: str, last: datetime | None, late: float, stale: float) -> float | None:
        if last is None:
            found.append((DOWN, f"{name}: nada gravado"))
            return None
        age = now_s - last.timestamp()
        if age > stale:
            found.append((DOWN, f"{name} parado: último há {fmt_age(age)} (limite {fmt_age(stale)})"))
        elif age > late:
            found.append((LATE, f"{name} atrasado: último há {fmt_age(age)} (limite {fmt_age(late)})"))
        return age

    book_age = age_check("livro", m.last_book, lim.book_late_s, lim.book_stale_s)
    trade_age = age_check("negócios", m.last_trade, lim.trade_late_s, lim.trade_stale_s)

    # Buracos: só conta a partir do primeiro livro gravado, se o coletor ligou há menos de 24 h.
    window_start = now_min - WINDOW_MIN
    if m.book_before_window or (m.last_book is not None and not m.book_per_minute):
        cover = window_start
    elif m.book_per_minute:
        cover = max(min(m.book_per_minute), window_start)
    else:
        cover = None
    runs = gap_runs(m.book_per_minute, cover, now_min) if cover is not None else []
    missing = sum(n for _, n in runs)
    if missing > lim.gap_minutes:
        start, size = max(runs, key=lambda r: r[1])
        ongoing = " (em andamento)" if start + size == now_min else ""
        found.append((LATE, f"{missing} min sem livro nas últimas 24 h; maior buraco: {size} min"
                            f" a partir de {fmt_minute(start)} UTC{ongoing}"))

    book_hour = last_hour(m.book_per_minute, now_min)
    covers_hour = cover is not None and cover <= now_min - 60
    if covers_hour and book_age is not None and book_age <= lim.book_stale_s and book_hour < lim.book_min_per_hour:
        found.append((LATE, f"livro ralo: {fmt_int(book_hour)} linhas na última hora"
                            f" (mínimo {fmt_int(lim.book_min_per_hour)})"))

    if m.db_bytes > lim.db_warn_gb * GB:
        found.append((LATE, f"banco com {fmt_bytes(m.db_bytes)} (aviso acima de {lim.db_warn_gb:g} GB)"))
    if m.disk_free_bytes is not None:
        free = m.disk_free_bytes
        if free < lim.disk_crit_gb * GB:
            found.append((DOWN, f"disco quase cheio: {fmt_bytes(free)} livres (limite {lim.disk_crit_gb:g} GB)"))
        elif free < lim.disk_warn_gb * GB:
            found.append((LATE, f"disco enchendo: {fmt_bytes(free)} livres (aviso abaixo de {lim.disk_warn_gb:g} GB)"))

    # Resumo legível, um item por fonte.
    def table_line(label: str, age: float | None, per_minute: dict[int, int], with_gaps: bool) -> str:
        if age is None and not per_minute:
            return f"{label}: nada gravado"
        parts = [label + ":"]
        if age is not None:
            parts.append(f"último há {fmt_age(age)} ·")
        parts.append(f"{fmt_int(last_hour(per_minute, now_min))} na última hora")
        full = [n for h, n in hourly(per_minute).items()
                if cover is not None and h * 60 >= cover and (h + 1) * 60 <= now_min]
        if full:
            parts.append(f"· por hora: mín {fmt_int(min(full))}, máx {fmt_int(max(full))}")
        if with_gaps and cover is not None:
            since = "em 24 h" if cover == window_start else f"desde {fmt_minute(cover)} UTC"
            parts.append(f"· {missing} min sem livro {since}")
        return " ".join(parts)

    details.append(table_line("livro (book_l1)", book_age, m.book_per_minute, True))
    details.append(table_line("negócios (trades)", trade_age, m.trades_per_minute, False))
    snaps = m.snapshots_per_minute
    details.append(f"fotos (book_snapshots): {fmt_int(last_hour(snaps, now_min))} na última hora"
                   if snaps else "fotos (book_snapshots): nada nas últimas 24 h")
    sizes = sorted(m.table_bytes.items(), key=lambda kv: -kv[1])
    details.append(f"banco: {fmt_bytes(m.db_bytes)}"
                   + "".join(f" · {name} {fmt_bytes(n)}" for name, n in sizes))
    if m.disk_free_bytes is not None and m.disk_total_bytes:
        pct = 100 * m.disk_free_bytes / m.disk_total_bytes
        details.append(f"disco: {fmt_bytes(m.disk_free_bytes)} livres de {fmt_bytes(m.disk_total_bytes)}"
                       f" ({pct:.0f}%)")

    ordered = sorted(found, key=lambda f: -RANK[f[0]])
    return Report(worst(s for s, _ in found), tuple(msg for _, msg in ordered), tuple(details), m.now, symbol)


def unreachable(err: BaseException, now: datetime, symbol: str = FUTURES_SYMBOL) -> Report:
    """Relatório para quando nem dá para ler o banco."""
    msg = str(err).strip().splitlines()[0] if str(err).strip() else type(err).__name__
    return Report(DOWN, (f"sem acesso ao banco: {msg}",), (), now, symbol)


def hourly_table(m: Metrics) -> str:
    """Linhas por hora das últimas 24 h, uma hora por linha (UTC)."""
    hours = {name: hourly(pm) for name, pm in (
        ("book_l1", m.book_per_minute), ("trades", m.trades_per_minute), ("book_snapshots", m.snapshots_per_minute))}
    now_h = int(m.now.timestamp() // 3600)
    lines = [f"{'hora (UTC)':<12}{'book_l1':>10}{'trades':>10}{'book_snapshots':>16}"]
    for h in range(now_h - 23, now_h + 1):
        row = [hours[t].get(h, 0) for t in TABLES]
        lines.append(f"{datetime.fromtimestamp(h * 3600, tz=timezone.utc):%d/%m %H}h    "
                     f"{fmt_int(row[0]):>10}{fmt_int(row[1]):>10}{fmt_int(row[2]):>16}")
    return "\n".join(lines)


# --- transições do --watch ---

@dataclass(frozen=True)
class WatchState:
    state: str | None = None  # último estado visto (None antes da primeira verificação)
    bad_since: float | None = None  # quando saiu de ok
    last_alert: float | None = None


@dataclass(frozen=True)
class Alert:
    kind: str  # mudou | repete
    old: str | None
    new: str
    bad_for: float  # segundos fora de ok (na volta ao ok, quanto tempo ficou ruim)


def decide(ws: WatchState, new: str, now: float, repeat_s: float) -> tuple[WatchState, Alert | None]:
    """Avisa quando o estado muda e, enquanto não for ok, de novo a cada repeat_s (0 = nunca repete)."""
    if ws.state is None:
        if new == OK:
            return WatchState(new), None
        return WatchState(new, now, now), Alert("mudou", None, new, 0.0)
    if new != ws.state:
        bad_for = now - ws.bad_since if ws.bad_since is not None else 0.0
        if new == OK:
            bad_since = None
        else:
            bad_since = ws.bad_since if ws.bad_since is not None else now
        return WatchState(new, bad_since, now), Alert("mudou", ws.state, new, bad_for)
    if new != OK and repeat_s > 0 and ws.last_alert is not None and now - ws.last_alert >= repeat_s:
        bad_for = now - ws.bad_since if ws.bad_since is not None else 0.0
        return WatchState(new, ws.bad_since, now), Alert("repete", new, new, bad_for)
    return ws, None


def clip(text: str, limit: int = MAX_ALERT_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def alert_text(alert: Alert, report: Report) -> str:
    """Texto do aviso: uma linha de título e o relatório."""
    if alert.kind == "repete":
        title = f"[Wescalp] continua {alert.new.upper()} há {fmt_age(alert.bad_for)}"
    elif alert.new == OK:
        title = f"[Wescalp] voltou ao normal (ficou {fmt_age(alert.bad_for)} {alert.old})"
    elif alert.old is None:
        title = f"[Wescalp] {alert.new.upper()} (início do vigia)"
    else:
        title = f"[Wescalp] {alert.new.upper()} (antes: {alert.old})"
    return clip(title + "\n" + report.text())


# --- envio dos avisos ---

class Notifier:
    """Manda o aviso para o webhook genérico e/ou o Telegram. Nunca levanta exceção."""

    def __init__(self, webhook: str | None = None, telegram_token: str | None = None,
                 telegram_chat_id: str | None = None, client: httpx.AsyncClient | None = None):
        self.webhook = (webhook or "").strip() or None
        self.telegram_token = (telegram_token or "").strip() or None
        self.telegram_chat_id = (telegram_chat_id or "").strip() or None
        self.client = client or httpx.AsyncClient(timeout=15)

    @classmethod
    def from_env(cls, env=os.environ, client: httpx.AsyncClient | None = None) -> "Notifier":
        return cls(env.get("WESCALP_ALERT_WEBHOOK"), env.get("WESCALP_TELEGRAM_TOKEN"),
                   env.get("WESCALP_TELEGRAM_CHAT_ID"), client)

    def targets(self, text: str) -> list[tuple[str, str, dict]]:
        out = []
        if self.webhook:
            out.append(("webhook", self.webhook, {"text": text}))
        if self.telegram_token and self.telegram_chat_id:
            out.append(("telegram", f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                        {"chat_id": self.telegram_chat_id, "text": text, "disable_web_page_preview": True}))
        return out

    def scrub(self, msg: str) -> str:
        """Tira URL do webhook e token do Telegram das mensagens de erro (são segredos)."""
        for secret in (self.webhook, self.telegram_token):
            if secret:
                msg = msg.replace(secret, "***")
        return msg

    async def send(self, text: str) -> int:
        """Envia para todos os canais; devolve quantos aceitaram."""
        text = clip(text)
        targets = self.targets(text)
        if not targets:
            log.warning("sem canal de aviso configurado; aviso só no log:\n%s", text)
            return 0
        sent = 0
        for name, url, payload in targets:
            try:
                r = await self.client.post(url, json=payload)
                r.raise_for_status()
                sent += 1
            except Exception as e:
                log.error("falha ao avisar por %s: %s", name, self.scrub(f"{type(e).__name__}: {e}"))
        return sent

    async def aclose(self) -> None:
        await self.client.aclose()


# --- leitura do banco ---

async def connect(url: str) -> psycopg.AsyncConnection:
    """Conexão só de leitura, com tempo máximo por consulta."""
    conn = await psycopg.AsyncConnection.connect(url, autocommit=True, connect_timeout=10)
    await conn.execute("SET default_transaction_read_only = on")
    await conn.execute("SET statement_timeout = '30s'")
    return conn


async def _one(conn, query, params=()):
    cur = await conn.execute(query, params)
    row = await cur.fetchone()
    return row[0] if row else None


async def _per_minute(conn, table: str, symbol: str, since: datetime) -> dict[int, int]:
    cur = await conn.execute(
        sql.SQL("SELECT date_trunc('minute', ts), count(*) FROM {} WHERE symbol = %s AND ts >= %s GROUP BY 1")
        .format(sql.Identifier(table)),
        (symbol, since),
    )
    return {int(m.timestamp()) // 60: n for m, n in await cur.fetchall()}


async def table_sizes(conn) -> dict[str, int]:
    """Tamanho de cada tabela do schema public; no TimescaleDB, das hypertables com todos os chunks."""
    cur = await conn.execute(
        "SELECT c.relname, pg_total_relation_size(c.oid) FROM pg_class c"
        " JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p')"
    )
    sizes = dict(await cur.fetchall())
    if await _one(conn, "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"):
        cur = await conn.execute(
            "SELECT hypertable_name, hypertable_size(format('%I.%I', hypertable_schema, hypertable_name)::regclass)"
            " FROM timescaledb_information.hypertables WHERE hypertable_schema = 'public'"
        )
        sizes.update({name: n for name, n in await cur.fetchall() if n is not None})
    return sizes


def disk_usage(path: str | None) -> tuple[int | None, int | None]:
    if not path:
        return None, None
    try:
        u = shutil.disk_usage(path)
        return u.free, u.total
    except OSError as e:
        log.warning("não deu para medir o disco em %s: %s", path, e)
        return None, None


async def read_metrics(conn, symbol: str = FUTURES_SYMBOL, disk_path: str | None = None) -> Metrics:
    now = await _one(conn, "SELECT now()")
    since = datetime.fromtimestamp((int(now.timestamp()) // 60 - WINDOW_MIN) * 60, tz=timezone.utc)
    exists = {t: bool(await _one(conn, "SELECT to_regclass(%s) IS NOT NULL", (t,))) for t in TABLES}

    async def last(table: str):
        if not exists[table]:
            return None
        return await _one(conn, sql.SQL("SELECT max(ts) FROM {} WHERE symbol = %s").format(sql.Identifier(table)),
                          (symbol,))

    async def per_minute(table: str) -> dict[int, int]:
        return await _per_minute(conn, table, symbol, since) if exists[table] else {}

    before = exists["book_l1"] and bool(await _one(
        conn, "SELECT EXISTS (SELECT 1 FROM book_l1 WHERE symbol = %s AND ts < %s)", (symbol, since)))
    free, total = disk_usage(disk_path)
    return Metrics(
        now=now,
        last_book=await last("book_l1"),
        last_trade=await last("trades"),
        book_per_minute=await per_minute("book_l1"),
        trades_per_minute=await per_minute("trades"),
        snapshots_per_minute=await per_minute("book_snapshots"),
        book_before_window=before,
        db_bytes=await _one(conn, "SELECT pg_database_size(current_database())"),
        table_bytes=await table_sizes(conn),
        disk_free_bytes=free,
        disk_total_bytes=total,
    )


async def collect(url: str, symbol: str, disk_path: str | None, tries: int = 3) -> Metrics:
    """Lê as métricas; tenta de novo algumas vezes antes de desistir (ex.: banco ainda subindo)."""
    for attempt in range(tries):
        try:
            conn = await connect(url)
            try:
                return await read_metrics(conn, symbol, disk_path)
            finally:
                await conn.close()
        except (psycopg.Error, OSError) as e:
            if attempt == tries - 1:
                raise
            log.warning("leitura do banco falhou (%s); nova tentativa em 5 s", (str(e).splitlines() or [""])[0])
            await asyncio.sleep(5)
    raise AssertionError("inalcançável")


async def check(url: str, lim: Limits, symbol: str, disk_path: str | None) -> tuple[Report, Metrics | None]:
    """Uma verificação completa. Banco inacessível vira relatório 'parado', não exceção."""
    try:
        m = await collect(url, symbol, disk_path)
    except Exception as e:
        return unreachable(e, datetime.now(timezone.utc), symbol), None
    return evaluate(m, lim, symbol), m


async def watch(run_check, notifier: Notifier, interval: float, repeat_s: float,
                clock=time.time, sleep=asyncio.sleep, iterations: int | None = None) -> WatchState:
    """Loop do --watch. run_check() devolve um Report; erro na verificação ou no envio não derruba o loop."""
    ws = WatchState()
    n = 0
    while iterations is None or n < iterations:
        n += 1
        try:
            report = await asyncio.wait_for(run_check(), timeout=max(interval, 120))
        except Exception as e:
            report = unreachable(e if str(e) else RuntimeError("verificação demorou demais"),
                                 datetime.now(timezone.utc))
        ws, alert = decide(ws, report.state, clock(), repeat_s)
        log.info("%s%s", report.headline(), "; " + "; ".join(report.problems) if report.problems else "")
        if alert:
            try:
                await notifier.send(alert_text(alert, report))
            except Exception:
                log.exception("falha inesperada ao avisar; o vigia segue")
        await sleep(interval)
    return ws


async def main() -> int:
    p = argparse.ArgumentParser(description="Vigia dos dados do Wescalp.")
    p.add_argument("--watch", action="store_true", help="roda em loop e avisa quando o estado muda")
    p.add_argument("--interval", type=float, default=float(os.environ.get("WESCALP_HEALTH_INTERVAL_S", "60")),
                   help="segundos entre verificações no --watch (padrão 60)")
    p.add_argument("--repeat-hours", type=float, default=float(os.environ.get("WESCALP_HEALTH_REPEAT_HOURS", "6")),
                   help="repete o aviso a cada N horas enquanto não estiver ok (0 = não repete; padrão 6)")
    p.add_argument("--horas", action="store_true", help="mostra as linhas por hora das últimas 24 h")
    p.add_argument("--test-alert", action="store_true", help="manda um aviso de teste e sai")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    lim = Limits.from_env()
    disk_path = os.environ.get("WESCALP_HEALTH_DISK_PATH") or None
    notifier = Notifier.from_env()
    try:
        if args.test_alert:
            sent = await notifier.send("[Wescalp] aviso de teste do vigia (wescalp.health)")
            print(f"aviso de teste aceito por {sent} de {len(notifier.targets('x'))} canal(is)")
            return 0 if sent or not notifier.targets("x") else 1
        if args.watch:
            channels = ", ".join(name for name, _, _ in notifier.targets("x")) or "nenhum (só log)"
            log.info("vigia ligado: a cada %gs, repete a cada %gh; canais: %s; %s",
                     args.interval, args.repeat_hours, channels, lim)

            async def run_check():
                return (await check(DATABASE_URL, lim, FUTURES_SYMBOL, disk_path))[0]

            await watch(run_check, notifier, args.interval, args.repeat_hours * 3600)
            return 0
        report, metrics = await check(DATABASE_URL, lim, FUTURES_SYMBOL, disk_path)
        print(report.text())
        if args.horas and metrics is not None:
            print()
            print(hourly_table(metrics))
        return EXIT_CODES[report.state]
    finally:
        await notifier.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
