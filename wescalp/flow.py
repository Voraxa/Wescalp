"""Sinais de fluxo de ordens (livro e negócios) e quanto eles preveem o mid nos segundos seguintes.

    python -m wescalp.flow --since 2026-10-22T20:00 --buckets 1 10 --horizons 10 30 60 300

Lê book_l1 e trades, agrega em baldes de tempo fixos e, para cada sinal e horizonte, mostra a correlação com
o retorno futuro do mid e o retorno médio por decil do sinal, em bps, ao lado do custo da ida e volta.

Sem vazamento: o balde [a, b) só usa mensagens com ts < b; o retorno vai do mid em b até o mid em b + h.

Buraco: intervalo de mais de `--max-gap` segundos sem mensagem do livro (o livro chega a cada 100 ms, mesmo
sem mudança, então isso é queda de conexão). Balde que encosta num buraco fica sem sinal e sem mid; o par de
fotos que atravessa o buraco não entra no OFI. O primeiro e o último balde da amostra sempre ficam de fora.
"""

import argparse
import asyncio
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone

from . import db
from .config import DATABASE_URL, FUTURES_SYMBOL, MAKER_FEE, TAKER_FEE

MAX_GAP = 2.0  # segundos sem mensagem do livro que contam como buraco
SIGNALS = ("ofi", "delta", "imb5", "micro_bps", "spread_bps")
UNDIRECTED = {"spread_bps"}  # não diz para que lado o preço vai


@dataclass(frozen=True, slots=True)
class Quote:
    """Topo do livro numa foto (linha de book_l1). ts em segundos desde a época."""

    ts: float
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float
    bid_sz5: float
    ask_sz5: float


@dataclass(frozen=True, slots=True)
class Tick:
    """Um negócio: lado do agressor (buy | sell) e tamanho em contratos."""

    ts: float
    side: str
    size: float


def ofi(prev: Quote, cur: Quote) -> float:
    """OFI de Cont, Kukanov e Stoikov (2014) entre duas fotos do topo, em contratos.

    Lado da compra: bid subiu conta o tamanho novo inteiro; bid caiu tira o tamanho antigo inteiro; mesmo
    preço conta a diferença de tamanho. Lado da venda é o espelho, com sinal trocado (ask que desce ou cresce
    é pressão vendedora).
    """
    e = 0.0
    if cur.bid_px >= prev.bid_px:
        e += cur.bid_sz
    if cur.bid_px <= prev.bid_px:
        e -= prev.bid_sz
    if cur.ask_px <= prev.ask_px:
        e -= cur.ask_sz
    if cur.ask_px >= prev.ask_px:
        e += prev.ask_sz
    return e


def depth_imbalance(bid5: float, ask5: float) -> float | None:
    """(bid5 − ask5) / (bid5 + ask5), entre −1 e 1."""
    total = bid5 + ask5
    return (bid5 - ask5) / total if total > 0 else None


def mid(q: Quote) -> float:
    return (q.bid_px + q.ask_px) / 2


def microprice(q: Quote) -> float:
    """Média do bid e do ask ponderada pelo tamanho do lado oposto: puxa para o lado com fila menor."""
    total = q.bid_sz + q.ask_sz
    if total <= 0:
        return mid(q)
    return (q.bid_px * q.ask_sz + q.ask_px * q.bid_sz) / total


def micro_dev_bps(q: Quote) -> float:
    m = mid(q)
    return (microprice(q) - m) / m * 1e4


def spread_bps(q: Quote) -> float:
    return (q.ask_px - q.bid_px) / mid(q) * 1e4


def trade_delta(ticks) -> float:
    """Compra agressora menos venda agressora, em contratos."""
    return sum(t.size if t.side == "buy" else -t.size for t in ticks)


def valid(q: Quote) -> bool:
    """Foto utilizável: preços positivos e livro não cruzado."""
    return 0 < q.bid_px < q.ask_px


def aggregate(quotes: list[Quote], ticks: list[Tick], width: float, max_gap: float = MAX_GAP) -> dict[str, list]:
    """Baldes [a, b) de `width` segundos alinhados à época, do balde da primeira foto ao da última.

    Colunas alinhadas, None onde o balde é inválido:
    - t: fim do balde (b), o instante em que o sinal fica pronto
    - mid: mid da última foto antes de b
    - ofi, delta: somas dentro do balde (balde sem mensagem e sem buraco vale 0)
    - imb5, micro_bps, spread_bps: estado do livro na última foto antes de b

    `quotes` precisa estar em ordem de ts; `ticks` pode vir em qualquer ordem.
    """
    quotes = [q for q in quotes if valid(q)]
    if not quotes:
        return {k: [] for k in ("t", "mid", *SIGNALS)}
    first = math.floor(quotes[0].ts / width)
    n = math.floor(quotes[-1].ts / width) - first + 1
    flow = [0.0] * n
    last: list[Quote | None] = [None] * n
    bad = [False] * n
    bad[0] = bad[-1] = True  # antes da primeira e depois da última foto não há dado

    prev, kprev = None, 0
    for q in quotes:
        k = math.floor(q.ts / width) - first
        if prev is not None:
            if q.ts - prev.ts > max_gap:
                for j in range(kprev, k + 1):
                    bad[j] = True
            else:
                flow[k] += ofi(prev, q)
        last[k] = q
        prev, kprev = q, k

    delta = [0.0] * n
    for t in ticks:
        k = math.floor(t.ts / width) - first
        if 0 <= k < n:
            delta[k] += t.size if t.side == "buy" else -t.size

    cols: dict[str, list] = {k: [] for k in ("t", "mid", *SIGNALS)}
    state = None
    for k in range(n):
        state = last[k] or state
        cols["t"].append((first + k + 1) * width)
        if bad[k]:
            for name in ("mid", *SIGNALS):
                cols[name].append(None)
            continue
        cols["mid"].append(mid(state))
        cols["ofi"].append(flow[k])
        cols["delta"].append(delta[k])
        cols["imb5"].append(depth_imbalance(state.bid_sz5, state.ask_sz5))
        cols["micro_bps"].append(micro_dev_bps(state))
        cols["spread_bps"].append(spread_bps(state))
    return cols


def forward_returns(mids: list[float | None], steps: int) -> list[float | None]:
    """Retorno do mid do fim do balde i até o fim do balde i + steps; None se faltar uma das pontas."""
    out: list[float | None] = []
    for i, m in enumerate(mids):
        j = i + steps
        later = mids[j] if j < len(mids) else None
        out.append(later / m - 1 if m and later else None)
    return out


def bin_means(signal: list, rets: list, bins: int) -> list[tuple[int, float] | None]:
    """Retorno médio por faixa de quantil do sinal (1 = menor). Valores iguais ficam na mesma faixa.

    Cada valor vai para a faixa do seu posto médio; com muitos empates (ex.: delta zero) algumas faixas ficam
    vazias (None).
    """
    pairs = sorted((s, r) for s, r in zip(signal, rets) if s is not None and r is not None)
    n = len(pairs)
    acc = [[0, 0.0] for _ in range(bins)]
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        b = min(int((i + j - 1) / 2 * bins / n), bins - 1)
        acc[b][0] += j - i
        acc[b][1] += sum(r for _, r in pairs[i:j])
        i = j
    return [(c, s / c) if c else None for c, s in acc]


def correlation(signal: list, rets: list) -> float | None:
    xs, ys = [], []
    for s, r in zip(signal, rets):
        if s is not None and r is not None:
            xs.append(s)
            ys.append(r)
    if len(xs) < 3:
        return None
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:  # sinal ou retorno constante
        return None


def costs_bps() -> dict[str, float]:
    """Custo da ida e volta em bps: as duas pontas maker, uma de cada, as duas taker."""
    return {
        "maker+maker": 2 * MAKER_FEE * 1e4,
        "maker+taker": (MAKER_FEE + TAKER_FEE) * 1e4,
        "taker+taker": 2 * TAKER_FEE * 1e4,
    }


def report(cols: dict[str, list], width: float, horizons: list[int], bins: int) -> list[dict]:
    """Uma linha por sinal e horizonte: n, correlação, retorno médio por faixa e o maior extremo, em bps."""
    rows = []
    for h in horizons:
        steps = round(h / width)
        if steps < 1 or abs(steps * width - h) > 1e-9:
            continue  # horizonte precisa ser múltiplo do balde
        rets = forward_returns(cols["mid"], steps)
        for name in SIGNALS:
            means = bin_means(cols[name], rets, bins)
            n = sum(m[0] for m in means if m)
            lo, hi = means[0], means[-1]
            extremes = [abs(m[1]) * 1e4 for m in (lo, hi) if m]
            rows.append({
                "signal": name, "h": h, "n": n,
                "n_ef": n * width / h,  # retornos que não se sobrepõem
                "corr": correlation(cols[name], rets),
                "bins_bps": [m[1] * 1e4 if m else None for m in means],
                "extreme_bps": max(extremes) if extremes else None,
            })
    return rows


def horizon_summary(cols: dict[str, list], width: float, horizons: list[int]) -> list[dict]:
    """Por horizonte: quantos retornos, a média e a média do valor absoluto, em bps."""
    out = []
    for h in horizons:
        steps = round(h / width)
        if steps < 1 or abs(steps * width - h) > 1e-9:
            continue
        rets = [r for r in forward_returns(cols["mid"], steps) if r is not None]
        out.append({
            "h": h, "n": len(rets),
            "mean_bps": statistics.fmean(rets) * 1e4 if rets else None,
            "abs_bps": statistics.fmean(abs(r) for r in rets) * 1e4 if rets else None,
        })
    return out


def _f(x: float | None, fmt: str) -> str:
    return "—" if x is None else format(x, fmt)


def format_report(rows: list[dict], bins: int) -> str:
    cost = costs_bps()
    head = (f"{'sinal':<11}{'h(s)':>5}{'n':>8}{'n_ef':>7}{'corr':>7}"
            + "".join(f"{'D' + str(i + 1):>7}" for i in range(bins)) + f"{'|ext|':>7}  passa")
    lines = [head]
    for r in rows:
        ext = r["extreme_bps"]
        passes = [k for k, v in cost.items() if ext is not None and ext > v]
        verdict = "sem lado" if r["signal"] in UNDIRECTED else passes[-1] if passes else "—"
        lines.append(
            f"{r['signal']:<11}{r['h']:>5}{r['n']:>8}{r['n_ef']:>7.0f}{_f(r['corr'], '.3f'):>7}"
            + "".join(f"{_f(b, '.2f'):>7}" for b in r["bins_bps"])
            + f"{_f(ext, '.2f'):>7}  {verdict}"
        )
    return "\n".join(lines)


def _parse_time(s: str) -> datetime:
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", type=_parse_time, help="início (ISO, UTC se sem fuso)")
    p.add_argument("--until", type=_parse_time, help="fim, exclusivo")
    p.add_argument("--buckets", type=float, nargs="+", default=[1, 10], help="largura dos baldes em segundos")
    p.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 60, 300], help="horizontes em segundos")
    p.add_argument("--bins", type=int, default=10, help="faixas de quantil do sinal (10 = decis)")
    p.add_argument("--max-gap", type=float, default=MAX_GAP, help="segundos sem livro que contam como buraco")
    args = p.parse_args()

    conn = await db.connect(DATABASE_URL)
    try:
        quotes = [Quote(*r) for r in await db.load_book_l1(conn, FUTURES_SYMBOL, args.since, args.until)]
        ticks = [Tick(*r) for r in await db.load_trades(conn, FUTURES_SYMBOL, args.since, args.until)]
    finally:
        await conn.close()
    if len(quotes) < 2:
        raise SystemExit("sem livro gravado no período")

    span = quotes[-1].ts - quotes[0].ts
    fmt = "%Y-%m-%d %H:%M:%S"
    print(f"{FUTURES_SYMBOL} {datetime.fromtimestamp(quotes[0].ts, timezone.utc):{fmt}} a "
          f"{datetime.fromtimestamp(quotes[-1].ts, timezone.utc):{fmt}} UTC ({span / 3600:.2f} h): "
          f"{len(quotes)} fotos do topo, {len(ticks)} negócios")
    print("custo da ida e volta (bps): " + ", ".join(f"{k} {v:.1f}" for k, v in costs_bps().items()))
    print("D1..D10: retorno médio futuro do mid (bps) por decil do sinal; |ext| = maior |D1|, |D10|;"
          " passa = maior custo que |ext| supera")
    print("cortes dos decis usam a amostra toda (só mede; uma regra precisa de cortes fixados antes)")
    print("spread não aponta direção: entra como referência e como filtro de custo, não como sinal de lado")
    print("os decis incluem a deriva da amostra (retorno médio abaixo): compare com ele antes de ver vantagem")
    if span < 7 * 24 * 3600:
        print("AVISO: amostra curta demais para concluir qualquer coisa; serve só para conferir o pipeline")

    for width in args.buckets:
        cols = aggregate(quotes, ticks, width, args.max_gap)
        ok = sum(m is not None for m in cols["mid"])
        spreads = [s for s in cols["spread_bps"] if s is not None]
        print(f"\nbaldes de {width:g} s: {len(cols['t'])}, válidos {ok} ({ok / max(len(cols['t']), 1):.1%}),"
              f" spread mediano {_f(statistics.median(spreads) if spreads else None, '.3f')} bps")
        for s in horizon_summary(cols, width, args.horizons):
            print(f"  h={s['h']:>4}s: {s['n']} retornos, médio {_f(s['mean_bps'], '.2f')} bps,"
                  f" |retorno| médio {_f(s['abs_bps'], '.2f')} bps")
        print(format_report(report(cols, width, args.horizons, args.bins), args.bins))


if __name__ == "__main__":
    asyncio.run(main())
