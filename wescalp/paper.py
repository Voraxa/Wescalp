"""Carteiras simuladas andando no tempo: a regra congelada e o aprendiz, com ordens e retorno gravados dia a dia.

    python -m wescalp.paper            # completa os dias fechados que faltam e sai
    python -m wescalp.paper --follow   # fica rodando: confere a cada minuto se um dia novo fechou

Nada aqui envia ordem nem usa chave de API: é o registro de regras fixadas antes (wescalp.refine e wescalp.learn,
23/10/2026) aplicadas aos candles do BTC-USDT spot que o serviço candles grava.
- congelada: 1/3 do capital em cada média diária de 50, 100 e 150 dias; o terço fica comprado se o fechamento das
  00h UTC está acima da média dele.
- aprendiz (experimento): as 13 médias de 50 a 170 dias, pesadas pelo que os sinais de cada uma renderam (o Hedge de
  wescalp.learn: nota = log do retorno com taxa, descontado com meia-vida de 180 dias; peso = exp(η · nota) com
  η = ln 2 / 0,1), e a carteira só ajusta quando a posição desejada se afasta 5 pontos percentuais.
Nas duas, a posição decidida no fechamento vale para o dia seguinte; 0,1% em cada compra e venda; USDT parado não
rende. Cada ordem simulada é a mudança da fração comprada.

Os dias e as ordens são só gravados, nunca reescritos. Na primeira execução de cada carteira, o histórico disponível
é reconstruído (live = false) e o dia de hoje (UTC) vira o início do período ao vivo (paper_meta.live_start): a
partir do fechamento dele, cada dia é decidido quando fecha. Um dia conta como fechado quando o candle das 23:55 já
está no banco (o serviço candles só grava candle fechado) ou 30 minutos depois da virada, o que vier primeiro.
"""

import argparse
import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from psycopg.types.json import Jsonb

from . import db
from .config import DATABASE_URL, SPOT_SYMBOL

log = logging.getLogger("wescalp.paper")

FEE = 0.001  # spot, nível básico, por lado
GAMMA = 0.5 ** (1 / 180)  # meia-vida de 180 dias na nota do aprendiz
ETA = math.log(2) / 0.10  # 10% a mais de retorno descontado dobra o peso relativo
BAND = 0.05  # o aprendiz só ajusta quando a posição desejada se afasta 5 p.p.
GRACE = timedelta(minutes=30)
LAST_CANDLE = timedelta(hours=23, minutes=55)
CHART_SMAS = (50, 100, 150)  # as linhas do gráfico


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    lookbacks: tuple[int, ...]
    learn: bool
    rule: str


FROZEN = Spec(
    "media-50-100-150", "congelada", (50, 100, 150), False,
    "1/3 do capital em cada média diária de 50, 100 e 150 dias do BTC-USDT spot; o terço fica comprado se o fechamento"
    " das 00h UTC está acima da média dele; decide no fechamento e vale para o dia seguinte; 0,1% em cada compra e venda")
LEARNER = Spec(
    "aprendiz-50-170", "aprendiz", tuple(range(50, 171, 10)), True,
    "13 médias diárias de 50 a 170 dias do BTC-USDT spot; cada uma pesa exp(η · nota), com nota = log do retorno dos"
    " sinais dela (com taxa) descontado com meia-vida de 180 dias e η = ln 2 / 0,1; posição = soma dos pesos das médias"
    " com o fechamento acima; ajusta só quando a posição desejada se afasta 5 p.p.; 0,1% em cada compra e venda")
STRATEGIES = (FROZEN, LEARNER)


@dataclass(frozen=True)
class Bar:
    day: date
    open: float
    high: float
    low: float
    close: float
    last: datetime  # abertura do último candle de 5 min do dia no banco


@dataclass(frozen=True)
class Comp:
    n: int  # dias da média
    sma: float | None  # None enquanto não há histórico
    on: int  # 1 se o fechamento está acima da média
    held: int  # o `on` do dia anterior: como o componente passou este dia
    weight: float
    score: float  # log do retorno dos sinais do componente, descontado


@dataclass(frozen=True)
class Row:
    day: date
    open: float
    high: float
    low: float
    close: float
    comps: tuple[Comp, ...]
    held: float  # fração comprada durante o dia (decidida no fechamento anterior)
    position: float  # fração decidida no fechamento do dia, vale para o dia seguinte
    equity: float  # carteira simulada no fechamento (1 no primeiro dia)
    hold: float  # segurar o BTC desde o primeiro dia
    live: bool


@dataclass(frozen=True)
class Order:
    day: date  # dia em cujo fechamento a decisão foi tomada; a ordem simulada sai na abertura seguinte
    action: str  # compra | venda
    size: float  # fração do capital
    position: float  # fração comprada depois da ordem
    price: float  # fechamento do dia da decisão
    reason: str  # médias que viraram ("M50 ↑ M100 ↓") ou "pesos"
    live: bool


def closed(bar: Bar, now: datetime) -> bool:
    start = datetime.combine(bar.day, time(), timezone.utc)
    return bar.last >= start + LAST_CANDLE or now >= start + timedelta(days=1) + GRACE


def sma_at(closes: list[float], i: int, n: int) -> float | None:
    return sum(closes[i + 1 - n:i + 1]) / n if i + 1 >= n else None


def weights(spec: Spec, scores: list[float]) -> list[float]:
    if not spec.learn:
        return [1 / len(scores)] * len(scores)
    top = max(scores)
    e = [math.exp(ETA * (s - top)) for s in scores]
    total = sum(e)
    return [x / total for x in e]


def advance(spec: Spec, last: Row | None, bars: list[Bar], live_start: date,
            now: datetime) -> tuple[list[Row], list[Order]]:
    """Linhas e ordens dos dias fechados depois de `last` (todos, se None).

    `bars` vem em ordem e com o histórico inteiro, incluindo os dias já gravados: a média de um dia usa os
    fechamentos até ele. Para no primeiro dia ainda aberto. Igual a wescalp.refine (congelada) e a
    wescalp.learn.learn_positions (aprendiz).
    """
    closes = [b.close for b in bars]
    rows: list[Row] = []
    orders: list[Order] = []
    prev = last
    for i, b in enumerate(bars):
        if last is not None and b.day <= last.day:
            continue
        if not closed(b, now):
            break
        r = b.close / prev.close - 1 if prev is not None else 0.0
        parts = []
        for j, n in enumerate(spec.lookbacks):
            sma = sma_at(closes, i, n)
            on = int(sma is not None and b.close > sma)
            pc = prev.comps[j] if prev is not None else None
            held_c, before = (pc.on, pc.held) if pc is not None else (0, 0)
            score = (GAMMA * pc.score if pc is not None else 0.0) + math.log1p(held_c * r - FEE * abs(held_c - before))
            parts.append((n, sma, on, held_c, score))
        w = weights(spec, [p[4] for p in parts])
        comps = tuple(Comp(n, sma, on, h, wi, s) for (n, sma, on, h, s), wi in zip(parts, w))
        target = sum(c.weight * c.on for c in comps)
        current = prev.position if prev is not None else 0.0
        position = target
        if spec.learn and not (abs(target - current) >= BAND - 1e-12 or (target in (0.0, 1.0) and target != current)):
            position = current
        held = current
        equity = prev.equity * (1 + held * r - FEE * abs(held - prev.held)) if prev is not None else 1.0
        live = b.day >= live_start
        row = Row(b.day, b.open, b.high, b.low, b.close, comps, held, position, equity,
                  prev.hold * (1 + r) if prev is not None else 1.0, live)
        if abs(position - current) > 1e-12:
            flips = [f"M{c.n} {'↑' if c.on else '↓'}" for c, p in zip(comps, prev.comps if prev else [None] * len(comps))
                     if c.on != (p.on if p is not None else 0)]
            orders.append(Order(b.day, "compra" if position > current else "venda", abs(position - current), position,
                                b.close, " ".join(flips) or "pesos", live))
        rows.append(row)
        prev = row
    return rows, orders


def drawdown(values: list[float]) -> float:
    peak, worst = 0.0, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, 1 - v / peak)
    return worst


def day_epoch(d: date) -> int:
    return int(datetime.combine(d, time(), timezone.utc).timestamp())


def payload(spec: Spec, live_start: date | None, rows: list[Row], orders: list[Order],
            last: tuple[datetime, float] | None) -> dict:
    """O que o painel mostra de uma carteira: dias, ordens, componentes, a marcação de agora e os números.

    `last`: (abertura, fechamento) do último candle de 5 min do spot; entra como "agora" se é de depois do último
    dia gravado. O período ao vivo conta a partir do fechamento do primeiro dia ao vivo (a primeira decisão ao
    vivo); o histórico, do primeiro dia com todas as médias até o último dia simulado.
    """
    out = {"id": spec.id, "name": spec.name, "learn": spec.learn, "rule": spec.rule,
           "live_start": live_start.isoformat() if live_start else None, "lookbacks": list(spec.lookbacks),
           "chart_smas": list(CHART_SMAS), "days": [], "orders": [], "components": [], "now": None,
           "history": None, "live": None}
    if not rows:
        return out
    idx = {n: j for j, n in enumerate(spec.lookbacks)}
    out["days"] = [{
        "t": day_epoch(r.day), "o": r.open, "h": r.high, "l": r.low, "c": r.close,
        "sma": [r.comps[idx[n]].sma if n in idx else None for n in CHART_SMAS],
        "on": [c.on for c in r.comps], "w": [round(c.weight, 4) for c in r.comps] if spec.learn else None,
        "held": r.held, "pos": r.position, "eq": r.equity, "hold": r.hold, "live": r.live,
    } for r in rows]
    out["orders"] = [{"t": day_epoch(o.day), "action": o.action, "size": o.size, "position": o.position,
                      "price": o.price, "reason": o.reason, "live": o.live} for o in orders]
    end = rows[-1]
    month_ago = rows[max(0, len(rows) - 31)]
    out["components"] = [{"n": c.n, "sma": c.sma, "on": c.on, "weight": c.weight, "score": c.score,
                          "weight_30d": p.weight} for c, p in zip(end.comps, month_ago.comps)]
    eq_now, hold_now, t_now = end.equity, end.hold, day_epoch(end.day) + 86400
    if last is not None and last[0].timestamp() >= day_epoch(end.day) + 86400:
        px = last[1]
        eq_now = end.equity * (1 + end.position * (px / end.close - 1))
        hold_now = end.hold * px / end.close
        t_now = int(last[0].timestamp()) + 300
        out["now"] = {"t": t_now, "price": px, "position": end.position, "equity": eq_now, "hold": hold_now}

    full = next((i for i, r in enumerate(rows) if all(c.sma is not None for c in r.comps)), None)
    sim = [r for r in rows[full:] if not r.live] if full is not None else []
    if len(sim) > 1:
        a, b = sim[0], sim[-1]
        years = (b.day - a.day).days / 365.25
        out["history"] = {
            "from": day_epoch(a.day), "to": day_epoch(b.day), "ret": b.equity / a.equity - 1,
            "hold_ret": b.hold / a.hold - 1, "cagr": (b.equity / a.equity) ** (1 / years) - 1 if years > 0 else None,
            "dd": drawdown([r.equity for r in sim]), "hold_dd": drawdown([r.hold for r in sim]),
            "orders": sum(1 for o in orders if not o.live and a.day <= o.day <= b.day),
        }
    live = [r for r in rows if r.live]
    if live:
        a = live[0]
        series = [r.equity for r in live] + ([eq_now] if out["now"] else [])
        out["live"] = {
            "from": day_epoch(a.day) + 86400, "to": t_now, "ret": eq_now / a.equity - 1,
            "hold_ret": hold_now / a.hold - 1, "dd": drawdown(series),
            "hold_dd": drawdown([r.hold for r in live] + ([hold_now] if out["now"] else [])),
            "orders": sum(1 for o in orders if o.live), "days": len(live) - 1,
        }
    return out


# --- banco ---

COLS = "day, open, high, low, close, held, position, equity, hold, live, detail"


def comps_json(comps: tuple[Comp, ...]) -> dict:
    return {"components": [{"n": c.n, "sma": c.sma, "on": c.on, "held": c.held, "w": c.weight, "s": c.score}
                           for c in comps]}


def _row(r) -> Row:
    d, o, h, l, c, held, pos, eq, hold, live, detail = r
    comps = tuple(Comp(x["n"], x["sma"], x["on"], x["held"], x["w"], x["s"]) for x in detail["components"])
    return Row(d, o, h, l, c, comps, held, pos, eq, hold, live)


async def load_bars(conn, symbol: str = SPOT_SYMBOL) -> list[Bar]:
    cur = await conn.execute(
        "SELECT (ts AT TIME ZONE 'UTC')::date, (array_agg(open ORDER BY ts))[1], max(high), min(low),"
        " (array_agg(close ORDER BY ts DESC))[1], max(ts) FROM candles WHERE source = 'spot' AND symbol = %s"
        " GROUP BY 1 ORDER BY 1", (symbol,))
    return [Bar(*r) for r in await cur.fetchall()]


async def ensure_meta(conn, spec: Spec, today: date) -> date:
    await conn.execute(
        "INSERT INTO paper_meta (strategy, name, rule, live_start) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
        (spec.id, spec.name, spec.rule, today))
    cur = await conn.execute("SELECT live_start FROM paper_meta WHERE strategy = %s", (spec.id,))
    return (await cur.fetchone())[0]


async def load_meta(conn) -> dict[str, date]:
    cur = await conn.execute("SELECT strategy, live_start FROM paper_meta")
    return dict(await cur.fetchall())


async def load_rows(conn, spec: Spec, last_only: bool = False) -> list[Row]:
    order = "DESC LIMIT 1" if last_only else "ASC"
    cur = await conn.execute(f"SELECT {COLS} FROM paper_days WHERE strategy = %s ORDER BY day {order}", (spec.id,))
    return [_row(r) for r in await cur.fetchall()]


async def load_orders(conn, spec: Spec) -> list[Order]:
    cur = await conn.execute(
        "SELECT day, action, size, position, price, reason, live FROM paper_orders WHERE strategy = %s ORDER BY day",
        (spec.id,))
    return [Order(*r) for r in await cur.fetchall()]


async def save(conn, spec: Spec, rows: list[Row], orders: list[Order]) -> None:
    """Só insere: um dia ou uma ordem já gravados nunca mudam (ON CONFLICT DO NOTHING)."""
    async with conn.transaction():
        async with conn.cursor() as cur:
            await cur.executemany(
                f"INSERT INTO paper_days (strategy, {COLS}) VALUES ({', '.join(['%s'] * 12)}) ON CONFLICT DO NOTHING",
                [(spec.id, r.day, r.open, r.high, r.low, r.close, r.held, r.position, r.equity, r.hold, r.live,
                  Jsonb(comps_json(r.comps))) for r in rows])
            await cur.executemany(
                "INSERT INTO paper_orders (strategy, day, action, size, position, price, reason, live)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                [(spec.id, o.day, o.action, o.size, o.position, o.price, o.reason, o.live) for o in orders])


async def run_once(conn, now: datetime) -> dict[str, list[Row]]:
    """Grava, em cada carteira, os dias que fecharam desde a última volta; devolve as linhas novas de cada uma."""
    out: dict[str, list[Row]] = {}
    bars = None
    for spec in STRATEGIES:
        live_start = await ensure_meta(conn, spec, now.date())
        stored = await load_rows(conn, spec, last_only=True)
        last = stored[0] if stored else None
        if last is not None and last.day >= (now - timedelta(days=1)).date():
            out[spec.id] = []  # ontem já está gravado: nada pode ter fechado
            continue
        bars = bars if bars is not None else await load_bars(conn)
        rows, orders = advance(spec, last, bars, live_start, now)
        if rows:
            await save(conn, spec, rows, orders)
        out[spec.id] = rows
    return out


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--follow", action="store_true", help="fica rodando e grava cada dia quando ele fecha")
    p.add_argument("--every", type=int, default=60, help="segundos entre verificações no modo --follow")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    conn = await db.connect(DATABASE_URL)
    await db.apply_schema(conn)
    try:
        while True:
            try:
                for sid, rows in (await run_once(conn, datetime.now(timezone.utc))).items():
                    if len(rows) > 3:
                        log.info("%s: %d dias gravados (%s a %s)", sid, len(rows), rows[0].day, rows[-1].day)
                    for r in rows[-3:]:
                        log.info("%s %s %s: fechamento %.1f → %.0f%% comprado; carteira %.4f, segurar %.4f", sid,
                                 r.day, "ao vivo" if r.live else "histórico", r.close, r.position * 100, r.equity,
                                 r.hold)
            except Exception:
                log.exception("falha ao atualizar as carteiras simuladas; tenta de novo na próxima volta")
                await conn.close()
                try:
                    conn = await db.connect(DATABASE_URL)
                except Exception:
                    log.exception("banco indisponível")
            if not args.follow:
                break
            await asyncio.sleep(args.every)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
