"""Day trade com várias operações por dia: confluência de sinais, DCA e mínimo de 2 ciclos por dia, no BTC spot.

    python -m wescalp.flowdata                                                  # antes: fluxo, prêmio, posições
    python -m wescalp.daytrade --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Fixado antes de rodar (23/10/2026), sem ajuste depois:
- camada do dia: cada parte do ciclo usa 1/3 do capital × (1/3 + 2/3 × a posição da congelada no fechamento de
  ontem), cheia na tendência de alta e um terço dela na baixa; o sistema nunca deixa de operar;
- confluência: um ridge com os 14 sinais de wescalp.intraday, três heurísticas de candle (RSI de 14 h, %B de
  Bollinger de 20 h e distância da VWAP de 24 h) e a hora do dia prevê o retorno das 4 h seguintes; walk-forward
  anual: treina com tudo desde 2020 antes do ano e testa o ano, de 2021 a 06/2026;
- entrada por sinal: de hora em hora, um ciclo por vez, quando a previsão passa do percentil 80 do treino, a mercado
  no fechamento. Entrada forçada para garantir 2 ciclos por dia, com metade do tamanho: a partir das 12h UTC sem
  nenhum ciclo no dia e a partir das 18h UTC com menos de 2;
- juiz que aprende com os próprios ciclos (variantes "com juiz"): ridge online do resultado líquido de cada ciclo
  por sinal sobre os sinais da entrada, esquecendo 1% a cada ciclo; depois de 100 ciclos, a entrada por sinal só
  sai se ele previr lucro (as forçadas não passam por ele nem entram no aprendizado);
- ciclo: DCA do mesmo tamanho com ordem limitada 1 e 2 passos abaixo da primeira compra, venda com ordem limitada
  no alvo sobre o preço médio, stop da posição inteira a mercado 3 passos abaixo da primeira compra e venda a
  mercado depois de 12 h; geometrias de passo e alvo de 0,6% e de 1,0%. Dentro de um candle de 5 min, o pior
  primeiro: DCA e stop antes do alvo, e o alvo não vale no candle em que o DCA comprou;
- taxa de 0,1% por lado (spot) e as mesmas entradas com 0,08%, 0,04%, 0,02% e 0%, para saber a partir de que custo
  dá lucro;
- teste da sorte: 1.000 sorteios com o mesmo número de ciclos em cada dia, em horas ao acaso, com os mesmos tamanhos
  e a mesma geometria; p = fração com retorno total igual ou maior;
- passa a variante que, com 0,1% por lado, de 2021 a 06/2026, tiver lucro, Sharpe maior que o de segurar o BTC,
  p ≤ 0,0125 (0,05 dividido pelas 4 variantes) e 2 ciclos ou mais em pelo menos 80% dos dias. Quem passar vai uma vez
  para as 16 semanas intocadas (07 a 10/2026) e depois para a carteira simulada.
"""

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import flowdata, intraday, refine, regime, spot, study

H, DAY = 3600, 86400
TRANCHE = 1 / 3
FORCED_SIZE = 0.5
FORCE_FIRST, FORCE_SECOND = 12, 18
QUANTILE = 0.8
HORIZON_H = 4
FIRST_TEST_YEAR = 2021
FEES = (0.001, 0.0008, 0.0004, 0.0002, 0.0)
FEE = FEES[0]
JUDGE_MIN, JUDGE_DECAY, JUDGE_SHRINK = 100, 0.99, 1.0
LUCK = 1000
MIN_DAYS_2 = 0.8
MAX_P = 0.05 / 4
HEURISTICS = ("RSI 14h", "%B Bollinger 20h", "distância da VWAP 24h")


@dataclass(frozen=True)
class Geometry:
    tp: float  # alvo sobre o preço médio
    step: float  # espaçamento do DCA, abaixo da primeira compra
    stop: float  # stop da posição inteira, abaixo da primeira compra
    bars: int  # tempo máximo, em candles de 5 min


GEOMETRIES = {"passo 0,6%": Geometry(0.006, 0.006, 0.018, 144), "passo 1,0%": Geometry(0.010, 0.010, 0.030, 144)}


@dataclass
class Cycle:
    entry: int  # candle em cujo fechamento saiu a primeira compra
    exit: int
    outcome: str  # alvo | stop | tempo
    size: float  # fração do capital em cada parte
    fills: list[tuple[int, float]] = field(default_factory=list)  # (candle, preço) de cada compra
    exit_px: float = 0.0
    forced: bool = False

    def pnl(self, fee: float) -> float:
        """Resultado do ciclo em fração do capital do início dele, com a taxa nas compras e na venda."""
        qty = sum(self.size * (1 - fee) / p for _, p in self.fills)
        return qty * self.exit_px * (1 - fee) - self.size * len(self.fills)

    def mark(self, k: int, close: float, fee: float) -> float:
        """Resultado marcado a mercado no fechamento do candle k (com a taxa de venda)."""
        filled = [(i, p) for i, p in self.fills if i <= k]
        qty = sum(self.size * (1 - fee) / p for _, p in filled)
        return qty * close * (1 - fee) - self.size * len(filled)


def cycle(o, h, l, c, up: spot.Crossing, dn: spot.Crossing, i: int, size: float, g: Geometry) -> Cycle:
    """Compra `size` no fechamento do candle i e deixa o ciclo correr (ver o cabeçalho do módulo)."""
    first = c[i]
    last = min(i + g.bars, len(c) - 1)
    cy = Cycle(i, last, "tempo", size, [(i, first)])
    levels = [first * (1 - g.step), first * (1 - 2 * g.step)]
    stop = first * (1 - g.stop)
    start = i + 1

    def hit(k):
        return k if k is not None and k <= last else None

    while start <= last:
        qty = sum(1 / p for _, p in cy.fills)
        target = len(cy.fills) / qty * (1 + g.tp)  # preço médio (cada parte gasta o mesmo) × (1 + alvo)
        kd = hit(dn.first(start, levels[0])) if levels else None
        ks, kt = hit(dn.first(start, stop)), hit(up.first(start, target))
        if kd is not None and (kt is None or kd <= kt):
            while levels and l[kd] <= levels[0]:  # o candle pode passar por mais de um nível
                cy.fills.append((kd, min(levels.pop(0), o[kd])))
            if l[kd] <= stop:
                cy.exit, cy.outcome, cy.exit_px = kd, "stop", min(stop, o[kd])
                return cy
            start = kd + 1  # o alvo não vale no candle em que o DCA comprou
            continue
        if ks is not None and (kt is None or ks <= kt):
            cy.exit, cy.outcome, cy.exit_px = ks, "stop", min(stop, o[ks])
            return cy
        if kt is not None:
            cy.exit, cy.outcome, cy.exit_px = kt, "alvo", max(target, o[kt])
            return cy
        break
    cy.exit, cy.exit_px = last, c[last]
    return cy


def heuristics(decision: np.ndarray, ts: np.ndarray, close: np.ndarray, volume: np.ndarray) -> dict[str, np.ndarray]:
    """RSI de 14 h e %B de Bollinger de 20 h nos fechamentos de hora em hora, e distância da VWAP das últimas 24 h."""
    hourly = np.column_stack([intraday.at(ts, close, decision - k * H - 300) for k in range(21)])  # coluna 0 = agora
    diff = hourly[:, :14] - hourly[:, 1:15]
    gain, loss = np.where(diff > 0, diff, 0).sum(axis=1), np.where(diff < 0, -diff, 0).sum(axis=1)
    window = hourly[:, :20]
    mean, sd = window.mean(axis=1), window.std(axis=1)
    pv = intraday.window_sum(ts, close * volume, decision - 24 * H, decision)
    v = intraday.window_sum(ts, volume, decision - 24 * H, decision)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {"RSI 14h": 100 * gain / (gain + loss), "%B Bollinger 20h": (hourly[:, 0] - mean + 2 * sd) / (4 * sd),
               "distância da VWAP 24h": np.log(hourly[:, 0] / (pv / v))}
    return {k: np.where(np.isfinite(x), x, np.nan) for k, x in out.items()}


def scaler(x: np.ndarray):
    mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    std = np.where(std > 0, std, 1.0)
    return lambda q: np.nan_to_num((q - mean) / std)


def walk_forward(x: np.ndarray, y: np.ndarray, decision: np.ndarray, years: np.ndarray):
    """Previsão do retorno de 4 h, corte do percentil 80 e padronização de cada ano, treinando com os anos de antes."""
    score, cut, z = np.full(len(decision), np.nan), np.full(len(decision), np.nan), np.zeros_like(x)
    for yr in range(FIRST_TEST_YEAR, int(years.max()) + 1):
        test = years == yr
        if not test.any():
            continue
        first = decision[test].min()
        train = (decision + HORIZON_H * H <= first) & np.isfinite(y)  # o alvo do treino acaba antes do ano
        model = intraday.ridge_fit(x[train], y[train])
        score[test], cut[test] = model(x[test]), np.quantile(model(x[train]), QUANTILE)
        z[test] = scaler(x[train])(x[test])
    return score, cut, z


@dataclass
class Market:
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    up: spot.Crossing
    dn: spot.Crossing
    regime: dict[int, float]  # dia → 1/3 + 2/3 × posição da congelada no fechamento da véspera

    def bar(self, t: int) -> int:
        """Índice do candle que fecha no instante t (o que abre em t - 5 min)."""
        return int(np.searchsorted(self.ts, t - 300))


class Judge:
    """Ridge online do resultado líquido dos próprios ciclos sobre os sinais da entrada, com esquecimento."""

    def __init__(self, n: int, fee: float):
        self.a, self.b, self.n, self.fee = np.zeros((n + 1, n + 1)), np.zeros(n + 1), 0, fee

    def allows(self, z: np.ndarray) -> bool:
        if self.n < JUDGE_MIN:
            return True
        x = np.concatenate([[1.0], z])
        return float(x @ np.linalg.solve(self.a + JUDGE_SHRINK * np.eye(len(x)), self.b)) > 0

    def learn(self, z: np.ndarray, pnl: float) -> None:
        x = np.concatenate([[1.0], z])
        self.a = JUDGE_DECAY * self.a + np.outer(x, x)
        self.b = JUDGE_DECAY * self.b + x * pnl
        self.n += 1


def run(m: Market, decision: np.ndarray, signal: np.ndarray, z: np.ndarray, g: Geometry, judge: bool,
        fee: float = FEE) -> list[Cycle]:
    """Uma passada de hora em hora, um ciclo por vez, com as entradas por sinal e as forçadas."""
    cycles: list[Cycle] = []
    judge_ = Judge(z.shape[1], fee) if judge else None
    free_at, today, count = -1, None, 0
    for k, t in enumerate(decision):
        day, hour = int(t // DAY), int(t // H) % 24
        if day != today:
            today, count = day, 0
        if t < free_at:
            continue
        forced = (hour >= FORCE_FIRST and count == 0) or (hour >= FORCE_SECOND and count < 2)
        by_signal = bool(signal[k]) and (judge_ is None or judge_.allows(z[k]))
        if not (by_signal or forced):
            continue
        i = m.bar(t)
        if i >= len(m.ts) or m.ts[i] != t - 300:
            continue  # falta o candle da decisão
        size = TRANCHE * m.regime.get(day, 1 / 3) * (1 if by_signal else FORCED_SIZE)
        cy = cycle(m.open, m.high, m.low, m.close, m.up, m.dn, i, size, g)
        cy.forced = not by_signal
        cycles.append(cy)
        count += 1
        free_at = int(m.ts[cy.exit]) + 300  # livre a partir do fechamento do candle da saída
        if judge_ is not None and by_signal:
            judge_.learn(z[k], cy.pnl(fee))
    return cycles


def daily_equity(m: Market, cycles: list[Cycle], days: np.ndarray, fee: float) -> np.ndarray:
    """Patrimônio (começa em 1) no fim de cada dia de `days`, com o ciclo aberto marcado a mercado."""
    last_bar = np.searchsorted(m.ts, (days + 1) * DAY) - 1  # último candle de cada dia
    out = np.empty(len(days))
    wealth, j = 1.0, 0
    for d, b in enumerate(last_bar):
        while j < len(cycles) and cycles[j].exit <= b:
            wealth *= 1 + cycles[j].pnl(fee)
            j += 1
        open_ = cycles[j] if j < len(cycles) and cycles[j].entry <= b else None
        out[d] = wealth * (1 + open_.mark(b, m.close[b], fee)) if open_ else wealth
    return out


def stats(m: Market, cycles: list[Cycle], days: np.ndarray, fee: float) -> dict:
    eq = daily_equity(m, cycles, days, fee)
    r = np.diff(np.concatenate([[1.0], eq]))  / np.concatenate([[1.0], eq[:-1]])
    per_day = np.bincount((np.array([int(m.ts[c.entry]) + 300 for c in cycles]) // DAY - days[0]).astype(int),
                          minlength=len(days)) if cycles else np.zeros(len(days), dtype=int)
    pnl = np.array([c.pnl(fee) for c in cycles]) if cycles else np.zeros(0)
    deployed = np.array([c.size * len(c.fills) for c in cycles]) if cycles else np.zeros(0)
    years = len(days) / 365.25
    return {
        "total": float(eq[-1] - 1), "cagr": float(eq[-1] ** (1 / years) - 1), "dd": study.max_dd(r),
        "sharpe": float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else 0.0,
        "cycles": len(cycles), "per_day": len(cycles) / len(days), "days_2": float((per_day >= 2).mean()),
        "hit": float((pnl > 0).mean()) if len(pnl) else math.nan,
        # Soma dos resultados sobre a soma do capital usado: a média simples por ciclo deixaria os muitos ganhos
        # pequenos pesarem tanto quanto os poucos prejuízos grandes (os ciclos com DCA usam até 3 partes).
        "net_per_cycle": float(pnl.sum() / deployed.sum()) if len(pnl) else math.nan,
        "mean_pnl": float(pnl.mean()) if len(pnl) else math.nan,
        "outcomes": {k: sum(1 for c in cycles if c.outcome == k) for k in ("alvo", "stop", "tempo")},
        "parts": float(np.mean([len(c.fills) for c in cycles])) if cycles else math.nan,
        "forced": float(np.mean([c.forced for c in cycles])) if cycles else math.nan,
    }


def luck_p(m: Market, cycles: list[Cycle], g: Geometry, days: np.ndarray, n: int = LUCK, seed: int = 0) -> float:
    """Fração dos sorteios (o mesmo número de ciclos em cada dia, em horas ao acaso, com os mesmos tamanhos) com
    retorno total igual ou maior que o do sistema."""
    real = math.prod(1 + c.pnl(FEE) for c in cycles)
    by_day: dict[int, list[float]] = {}
    for c in cycles:
        by_day.setdefault((int(m.ts[c.entry]) + 300) // DAY, []).append(c.size)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        wealth, free_at = 1.0, -1
        for day, sizes in by_day.items():
            hours = np.sort(rng.choice(24, size=min(len(sizes), 24), replace=False))
            for hour, size in zip(hours, sizes):
                t = day * DAY + int(hour) * H
                t = max(t, free_at + (-free_at) % H)  # próxima hora cheia em que está livre
                if t // DAY != day:
                    break
                i = m.bar(t)
                if i >= len(m.ts) or m.ts[i] != t - 300:
                    continue
                cy = cycle(m.open, m.high, m.low, m.close, m.up, m.dn, i, size, g)
                wealth *= 1 + cy.pnl(FEE)
                free_at = int(m.ts[cy.exit]) + 300
        hits += wealth >= real
    return hits / n


def benchmark(m: Market, day_close: regime.Daily, days: np.ndarray) -> dict[str, dict]:
    lo = int(np.searchsorted(day_close.day, days[0]))
    hi = lo + len(days)
    rets = day_close.close[lo:hi] / day_close.close[lo - 1:hi - 1] - 1
    out = {}
    for name, pos in (("segurar BTC", np.ones(len(day_close.close))),
                      ("congelada 50/100/150", refine.position(day_close.close, refine.BASE))):
        r = regime.returns_from_held(pos[lo - 1:hi - 1], rets)
        eq = np.cumprod(1 + r)
        out[name] = {"total": float(eq[-1] - 1), "cagr": study.cagr(r), "dd": study.max_dd(r),
                     "sharpe": float(r.mean() / r.std() * math.sqrt(365))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--data", type=Path, default=flowdata.DATA_DIR)
    p.add_argument("--luck", type=int, default=LUCK)
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    flow, perp, premium, funding, metrics = (flowdata.load(k, args.data) for k in
                                             ("spot-flow", "perp", "premium", "funding", "metrics"))
    d = regime.daily_bars(ts, o, h, l, c)
    pos = refine.position(d.close, refine.BASE)
    regime_of = {int(day) + 1: 1 / 3 + 2 / 3 * float(x) for day, x in zip(d.day, pos)}  # vale no dia seguinte
    m = Market(ts, o, h, l, c, spot.Crossing(h, True), spot.Crossing(l, False), regime_of)

    decision = np.arange(intraday.START, intraday.END - HORIZON_H * H, H, dtype=np.int64)
    sig = intraday.signals(decision, {"ts": ts, "close": c}, flow, perp, premium, funding, metrics)
    heu = heuristics(decision, ts, c, np.nan_to_num(intraday.at(flow["ts"], flow["volume"], ts)))
    x = np.column_stack([intraday.features(sig, decision)] + [heu[k] for k in HEURISTICS])
    y = intraday.forward({"ts": ts, "close": c}, decision, HORIZON_H)
    years = (decision * 1).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    score, cut, z = walk_forward(x, y, decision, years)
    test = years >= FIRST_TEST_YEAR
    decision, signal, z = decision[test], (score >= cut)[test], z[test]
    days = np.arange(decision[0] // DAY, decision[-1] // DAY + 1)

    fmt = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    print(f"day trade no BTC spot, de {fmt(decision[0])} a {fmt(decision[-1])} ({len(days)} dias), walk-forward anual;"
          f" sinal em {signal.mean():.0%} das horas")
    bench = benchmark(m, d, days)
    results = {}
    for gname, g in GEOMETRIES.items():
        for judge in (False, True):
            name = f"{gname}{' com juiz' if judge else ''}"
            cycles = run(m, decision, signal, z, g, judge)
            s = {fee: stats(m, cycles, days, fee) for fee in FEES}
            s[FEE]["p"] = luck_p(m, cycles, g, days, args.luck) if args.luck else math.nan
            results[name] = s
            print(f"  {name}: {len(cycles)} ciclos", flush=True)

    print(f"\ncom {FEE:.1%} por lado:")
    print(f"{'variante':<24}{'total':>9}{'ao ano':>9}{'queda':>7}{'Sharpe':>8}{'ciclos/dia':>12}{'dias ≥2':>9}"
          f"{'acerto':>8}{'sobre o usado':>15}{'por ciclo':>11}{'forçadas':>10}{'p sorte':>9}")
    for name, s in results.items():
        x0 = s[FEE]
        print(f"{name:<24}{x0['total']:>9.0%}{x0['cagr']:>9.1%}{x0['dd']:>7.0%}{x0['sharpe']:>8.2f}{x0['per_day']:>12.2f}"
              f"{x0['days_2']:>9.0%}{x0['hit']:>8.0%}{x0['net_per_cycle']:>15.3%}{x0['mean_pnl']:>11.3%}"
              f"{x0['forced']:>10.0%}{x0['p']:>9.3f}")
    for name, b in bench.items():
        print(f"{name:<24}{b['total']:>9.0%}{b['cagr']:>9.1%}{b['dd']:>7.0%}{b['sharpe']:>8.2f}")
    print("\nsaídas (alvo / stop / tempo) e partes compradas por ciclo, com 0,1% por lado:")
    for name, s in results.items():
        oc = s[FEE]["outcomes"]
        print(f"  {name:<24}{oc['alvo']:>6} / {oc['stop']:>5} / {oc['tempo']:>5}   {s[FEE]['parts']:.2f} partes")
    print("\nretorno ao ano conforme a taxa por lado (as mesmas entradas):")
    print(f"{'variante':<24}" + "".join(f"{f:>9.2%}" for f in FEES))
    for name, s in results.items():
        print(f"{name:<24}" + "".join(f"{s[f]['cagr']:>9.1%}" for f in FEES))

    hold = bench["segurar BTC"]
    print()
    for name, s in results.items():
        x0, fails = s[FEE], []
        if not x0["total"] > 0:
            fails.append("lucro")
        if not x0["sharpe"] > hold["sharpe"]:
            fails.append("Sharpe")
        if not x0["p"] <= MAX_P:
            fails.append("sorte")
        if not x0["days_2"] >= MIN_DAYS_2:
            fails.append("2 ciclos/dia")
        print(f"{name}: {'PASSA' if not fails else 'não passa: ' + ', '.join(fails)}")


if __name__ == "__main__":
    main()
