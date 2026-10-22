"""Varredura de barreiras × setups × lados, com walk-forward, Deflated Sharpe e PBO.

    python -m wescalp.sweep [--jobs 4] [--blocks 16]

Pergunta: ajustar barreiras e escolher setup acha algo real ou só sorte? Usa só candles anteriores às últimas
`--holdout` semanas (padrão 16, o período intocado), e as funções de wescalp.labels e wescalp.research como são.

Grade testada (todas contam como tentativas no DSR):
- barreiras: tp ∈ {1, 1.5, 2, 3} ATR × sl ∈ {0.5, 1, 1.5} ATR × tempo ∈ {6, 12, 24} candles, ATR(14) = 36;
- regras: breakout, vwap_reversion, pullback e "todos os candles" (entra em todo candle em que está fora,
  sem sobrepor, como trades_for faz com os setups) = 4;
- lados: long, short = 2.
Total: 36 × 4 × 2 = 288 configurações.

Retorno de uma configuração na semana k: soma do ret_net das entradas feitas nela (mesmo valor por entrada,
sem juros compostos; semana sem entrada vale 0). A semana conta a partir do primeiro candle; a semana final
incompleta fica de fora. Sharpe é semanal. Tudo roda duas vezes: líquido (o que importa) e bruto (ret_gross,
para ver se há vantagem antes das taxas).

- walk-forward: treina 60 semanas, testa as 4 seguintes, anda 4. Em cada split escolhe a configuração de maior
  Sharpe no treino e guarda as 4 semanas dela no teste. Uma entrada perto do fim do treino pode sair até 2 h
  depois do começo do teste; não purgamos isso.
- DSR: a melhor configuração no período todo, contra o Sharpe máximo esperado entre as 288. V[SR] vem de duas
  formas: a do paper (variância dos 288 Sharpes, inflada aqui porque as taxas separam muito as regras que
  operam muito das que operam pouco) e a da hipótese nula, 1/(T-1), que é o ruído de estimar um Sharpe 0.
- PBO: CSCV na matriz semanas × 288 configurações.

Tempo: os rótulos de uma barreira servem para os 3 setups e a linha de base; os sinais dos setups são
calculados uma vez só (não dependem das barreiras). `--jobs` rotula barreiras em paralelo.
"""

import argparse
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from itertools import product
from typing import NamedTuple

from . import db, features
from .config import DATABASE_URL, FUTURES_SYMBOL
from .kucoin import Candle
from .labels import Barriers, Label, label_candles
from .research import trades_for
from .setups import SETUPS
from .validation import (
    deflated_sharpe,
    expected_max_sharpe,
    moments,
    pbo,
    probabilistic_sharpe,
    sharpe,
    walk_forward,
    walk_forward_splits,
)

WEEK = timedelta(weeks=1)
TP = (1.0, 1.5, 2.0, 3.0)
SL = (0.5, 1.0, 1.5)
BARS = (6, 12, 24)
BASELINE = "todos os candles"
SIDES = ("long", "short")


def grid(atr_period: int = 14) -> list[Barriers]:
    return [Barriers(tp, sl, bars, atr_period) for tp, sl, bars in product(TP, SL, BARS)]


def weekly_returns(trades: list[Label], start: datetime, n_weeks: int, field: str = "ret_net") -> list[float]:
    """Soma do retorno por semana de entrada, contando de `start`; entradas depois de `n_weeks` ficam de fora."""
    out = [0.0] * n_weeks
    for t in trades:
        k = (t.ts - start) // WEEK
        if 0 <= k < n_weeks:
            out[k] += getattr(t, field)
    return out


def _signals_rule(sig: list[str | None]):
    return lambda _cs, _f, i: sig[i]


def _always(side: str):
    return lambda _cs, _f, _i: side


class Config(NamedTuple):
    barriers: str
    rule: str
    side: str
    n: int  # entradas
    net: list[float]  # retorno líquido por semana
    gross: list[float]  # antes das taxas

    @property
    def name(self) -> str:
        return f"{self.barriers} {self.rule} {self.side}"


def config_returns(candles: list[Candle], signals: dict[str, list], barriers: Barriers, n_weeks: int) -> list[Config]:
    """Todas as regras e lados de uma barreira, com os mesmos rótulos."""
    labels = {(lb.ts, lb.side): lb for lb in label_candles(candles, barriers)}
    start = candles[0].ts

    def make(rule: str, side: str, trades: list[Label]) -> Config:
        trades = [t for t in trades if (t.ts - start) // WEEK < n_weeks]
        return Config(barriers.name, rule, side, len(trades), weekly_returns(trades, start, n_weeks),
                      weekly_returns(trades, start, n_weeks, "ret_gross"))

    out = [make(BASELINE, side, trades_for(candles, {}, labels, _always(side))) for side in SIDES]
    for name, sig in signals.items():
        trades = trades_for(candles, {}, labels, _signals_rule(sig))
        out += [make(name, side, [t for t in trades if t.side == side]) for side in SIDES]
    return out


_W: dict = {}


def _init_worker(candles, signals, n_weeks):
    _W.update(candles=candles, signals=signals, n_weeks=n_weeks)


def _work(barriers: Barriers) -> list[tuple]:
    return config_returns(_W["candles"], _W["signals"], barriers, _W["n_weeks"])


def sweep(candles: list[Candle], barriers: list[Barriers], jobs: int = 1) -> tuple[list[Config], int]:
    n_weeks = int((candles[-1].ts - candles[0].ts) // WEEK)  # só semanas completas
    cols = features.compute(candles, atr_period=barriers[0].atr_period)
    signals = {name: [rule(candles, cols, i) for i in range(len(candles))] for name, rule in SETUPS.items()}
    if jobs <= 1:
        parts = [config_returns(candles, signals, b, n_weeks) for b in barriers]
    else:
        with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(candles, signals, n_weeks)) as ex:
            parts = list(ex.map(_work, barriers))
    return [row for part in parts for row in part], n_weeks


def bps(x: float) -> str:
    return f"{x * 1e4:.1f}"


def week_rows(splits, start: datetime) -> list[tuple[int, int, int, int]]:
    return [tuple((t - start) // WEEK for t in (s.train_start, s.train_end, s.test_start, s.test_end)) for s in splits]


def report(configs: list[Config], start: datetime, n_weeks: int, n_blocks: int, field: str) -> str:
    """Tabela, walk-forward, DSR e PBO sobre os retornos semanais `field` ("net" ou "gross")."""
    series = [getattr(c, field) for c in configs]
    matrix = [list(r) for r in zip(*series)]  # semanas × configurações
    srs = [sharpe(s) for s in series]
    lines = []

    lines.append(f"{'melhor barreira por regra e lado':<46}{'entradas':>9}{'bps/entr':>9}{'total(bps)':>12}"
                 f"{'bps/sem':>9}{'SR sem':>8}")
    for rule in (BASELINE, *SETUPS):
        for side in SIDES:
            j = max((j for j, c in enumerate(configs) if c.rule == rule and c.side == side), key=srs.__getitem__)
            tot = sum(series[j])
            per = bps(tot / configs[j].n) if configs[j].n else "-"
            lines.append(f"{configs[j].name:<46}{configs[j].n:>9}{per:>9}{bps(tot):>12}{bps(tot / n_weeks):>9}"
                         f"{srs[j]:>8.3f}")
    lines.append(f"configurações com total positivo: {sum(sum(s) > 0 for s in series)} de {len(configs)}")

    splits = walk_forward_splits(start, start + n_weeks * WEEK)
    steps = walk_forward(matrix, week_rows(splits, start))
    lines.append(f"\nWalk-forward (treino 60 sem, teste 4, passo 4): {len(steps)} splits")
    lines.append(f"{'teste':<25}{'escolhida no treino':<46}{'SR treino':>10}{'teste(bps)':>11}")
    oos: list[float] = []
    for sp, st in zip(splits, steps):
        oos += st.test_returns
        period = f"{sp.test_start:%Y-%m-%d} a {sp.test_end:%Y-%m-%d}"
        lines.append(f"{period:<25}{configs[st.best].name:<46}{st.train_sharpe:>10.3f}{bps(sum(st.test_returns)):>11}")
    if oos:
        lines.append(f"fora da amostra concatenado: {len(oos)} semanas, total {bps(sum(oos))} bps, "
                     f"{bps(sum(oos) / len(oos))} bps/sem, SR {sharpe(oos):.3f}, PSR(SR > 0) {probabilistic_sharpe(oos):.3f}, "
                     f"semanas positivas {sum(r > 0 for r in oos)}/{len(oos)}")

    best = max(range(len(configs)), key=srs.__getitem__)
    rets = series[best]
    n, mean = len(srs), sum(srs) / len(srs)
    var = sum((s - mean) ** 2 for s in srs) / (n - 1)
    var_null = 1 / (len(rets) - 1)  # variância do Sharpe estimado quando o verdadeiro é 0
    skew, kurt = moments(rets)
    lines.append(f"\nDeflated Sharpe da melhor no período todo ({configs[best].name}):")
    lines.append(f"SR semanal {srs[best]:.3f}, T {len(rets)}, assimetria {skew:.2f}, curtose {kurt:.2f}, "
                 f"PSR(SR > 0) {probabilistic_sharpe(rets):.3f}")
    lines.append(f"N {n}, V[SR] das tentativas {var:.4f}: SR esperado por sorte {expected_max_sharpe(var, n):.3f}, "
                 f"DSR {deflated_sharpe(rets, srs):.3f}")
    lines.append(f"N {n}, V[SR] sob a nula 1/(T-1) = {var_null:.4f}: SR esperado por sorte "
                 f"{expected_max_sharpe(var_null, n):.3f}, "
                 f"DSR {probabilistic_sharpe(rets, expected_max_sharpe(var_null, n)):.3f}")

    r = pbo(matrix, n_blocks)
    logits = sorted(r.logits)
    lines.append(f"\nPBO (CSCV, {n_blocks} blocos de ~{n_weeks / n_blocks:.1f} semanas, {len(r.logits)} combinações):")
    lines.append(f"PBO {r.pbo:.3f}, λ mediano {logits[len(logits) // 2]:.2f}, "
                 f"P(escolhida perde fora) {r.prob_oos_loss:.3f}, "
                 f"SR dentro médio {sum(r.is_sharpes) / len(r.is_sharpes):.3f}, "
                 f"fora médio {sum(r.oos_sharpes) / len(r.oos_sharpes):.3f}")
    return "\n".join(lines)


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--holdout", type=int, default=16, help="semanas finais intocadas (mínimo 16)")
    p.add_argument("--blocks", type=int, default=16, help="blocos do CSCV (par)")
    p.add_argument("--jobs", type=int, default=1, help="processos para rotular as barreiras")
    p.add_argument("--save", action="store_true", help="grava o resultado na tabela reports (para o painel)")
    args = p.parse_args()
    if args.holdout < 16:
        raise SystemExit("a varredura não olha as últimas 16 semanas")

    t0 = time.perf_counter()
    conn = await db.connect(DATABASE_URL)
    try:
        candles = await db.load_candles(conn, "futures", FUTURES_SYMBOL)
    finally:
        await conn.close()
    cutoff = candles[-1].ts - timedelta(weeks=args.holdout)
    candles = [c for c in candles if c.ts < cutoff]
    t1 = time.perf_counter()

    barriers = grid()
    configs, n_weeks = sweep(candles, barriers, args.jobs)
    t2 = time.perf_counter()

    header = [
        f"{FUTURES_SYMBOL} {candles[0].ts:%Y-%m-%d %H:%M} a {cutoff:%Y-%m-%d %H:%M} (intocado depois), "
        f"{len(candles)} candles, {n_weeks} semanas completas",
        f"grade: tp {list(TP)} × sl {list(SL)} × tempo {list(BARS)} (ATR 14) = {len(barriers)} barreiras; "
        f"regras {[BASELINE, *SETUPS]} × lados {list(SIDES)} = {len(configs)} configurações",
    ]
    print("\n".join(header))
    sections = []
    for field, title in (("net", "LÍQUIDO (com taxas)"), ("gross", "BRUTO (antes das taxas)")):
        text = report(configs, candles[0].ts, n_weeks, args.blocks, field)
        sections.append({"title": title, "text": text})
        print(f"\n=== {title} ===")
        print(text)
    t3 = time.perf_counter()
    if args.save:
        conn = await db.connect(DATABASE_URL)
        try:
            await db.apply_schema(conn)
            await db.save_report(conn, "sweep", {
                "symbol": FUTURES_SYMBOL, "from": candles[0].ts.isoformat(), "to": cutoff.isoformat(),
                "configs": len(configs), "header": header, "sections": sections,
            })
        finally:
            await conn.close()
    print(f"\ntempo: banco {t1 - t0:.1f} s, rótulos e entradas {t2 - t1:.1f} s ({args.jobs} processo(s)), "
          f"walk-forward/DSR/PBO {t3 - t2:.1f} s, total {t3 - t0:.1f} s")


if __name__ == "__main__":
    asyncio.run(main())
