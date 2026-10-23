"""Modelo bayesiano no estilo do Meridian para o BTC: linha de base com knots mais fatores com peso e incerteza.

    python -m wescalp.factors                                              # antes: baixa os fatores
    python -m wescalp.bayes --file data/candles/BINANCE-BTCUSDT-5m.csv.gz

Desenho fixado antes de rodar (23/10/2026, versão 1), sem ajuste depois de ver o resultado:
- decisão toda segunda às 00h UTC (domingo 21h em Brasília); alvo: log do retorno do BTC na semana que começa ali
  (do fechamento de domingo ao do domingo seguinte);
- linha de base: knots igualmente espaçados, um a cada ~26 semanas do treino, com interpolação linear entre eles e
  passeio aleatório entre knots vizinhos; para a frente vale o último knot, sem inventar tendência;
- fatores, cada um com seu β e padronizados com a média e o desvio do treino: a posição da combinação 50/100/150 no
  domingo (0, 1/3, 2/3 ou 1), Medo e Ganância (nível de domingo e variação de 7 dias), Nasdaq em 4 semanas, dólar em
  4 semanas, BTC em 4 semanas e volatilidade diária do BTC em 30 dias. Só entra o que já era público na decisão
  (wescalp.factors: bolsa da sexta, Medo e Ganância de domingo, dólar com a semana de atraso do relatório do Fed);
- erro Student-t. Priors: primeiro knot N(0; 0,02), passos entre knots N(0; τ) com τ ~ meia-normal(0,02), cada
  β ~ N(0; 0,01) (1 desvio do fator mexe ~1% no retorno esperado da semana), σ ~ meia-normal(0,1), ν ~ Gama(2; 0,1);
- NUTS com 4 cadeias de sementes 0 a 3 (1.000 passos de aquecimento e 1.000 amostras cada) e uma segunda rodada com
  as sementes 4 a 7, para conferir que o resultado não depende delas;
- divisão: as primeiras 87% das semanas (desde fev/2018) treinam e as 13% finais testam, até 30/06/2026 (as 16
  semanas seguintes já foram usadas na prova única e ficam de fora); depois, walk-forward anual: para cada ano de
  2020 a 2026, treina com todas as semanas anteriores e prevê o ano;
- carteira, a partir da chance de alta P(retorno da semana > 0) prevista: (a) 100% comprado se P > 50%, senão USDT;
  (b) comprado na fração P. 0,1% por lado sobre o que muda;
- régua no mesmo período: a congelada (50/100/150, diária), o aprendiz e segurar o BTC. Teste da sorte da carteira:
  os trechos de (a) em ordem sorteada (como em wescalp.regime) e as semanas de (b) embaralhadas, 2.000 sorteios.

Vira a terceira carteira simulada ao vivo só se uma das formas cumprir tudo (regra também fixada antes de rodar):
no walk-forward, Sharpe maior que o da congelada no mesmo período, queda máxima igual ou menor e p ≤ 0,025 no teste
da sorte (0,05 dividido pelas duas formas); no teste de 13%, Sharpe não menor que o da congelada; e ajuste válido
(R-hat ≤ 1,01 em todos os parâmetros e as duas rodadas de sementes com a mesma decisão (a) em 95% das semanas ou mais).
"""

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
from scipy import stats

from . import factors, learn, refine, regime, study

FACTORS = ("médias 50/100/150", "medo e ganância", "medo e ganância 7 dias", "Nasdaq 4 semanas", "dólar 4 semanas",
           "BTC 4 semanas", "volatilidade BTC 30 dias")
SHORT = ("médias", "M&G", "M&G 7d", "Nasdaq", "dólar", "BTC 4s", "vol 30d")
KNOT_WEEKS = 26
TRAIN_SHARE = 0.87
END = date(2026, 6, 30)  # último dia da pesquisa; o que vem depois é a prova única, já usada
FIRST_WF_YEAR = 2020
SEEDS = (0, 1, 2, 3)
CHECK_SEEDS = (4, 5, 6, 7)
WARMUP = SAMPLES = 1000
LUCK = 2000
MAX_R_HAT = 1.01
MIN_AGREE = 0.95
MAX_P = 0.025
BENCH = ("congelada 50/100/150", "aprendiz", "segurar BTC")
FORMS = ("bayes (a) P > 50%", "bayes (b) fração P")


@dataclass
class Weekly:
    monday: np.ndarray  # dia de cada decisão (desde 01/01/1970), segunda 00h UTC
    x: np.ndarray  # fatores sem padronizar (semanas × FACTORS)
    y: np.ndarray  # log do retorno do BTC na semana que começa na decisão
    first: np.ndarray  # índice, nas barras diárias, da segunda de cada semana


def weekly(d: regime.Daily, fac: dict[str, tuple[np.ndarray, np.ndarray]], end: int | None = None) -> Weekly:
    """Uma linha por segunda com todos os fatores já conhecidos e a semana seguinte inteira até o dia `end`."""
    if (np.diff(d.day) != 1).any():
        raise ValueError("faltam dias nas barras diárias")
    close = d.close
    last = len(close) - 1 if end is None else int(np.searchsorted(d.day, end, side="right")) - 1
    idx = np.arange(len(close))
    i = np.flatnonzero((factors.weekday(d.day) == 0) & (idx >= 150) & (idx + 6 <= last))  # 150: a média mais lenta
    sun, m = i - 1, d.day[i]
    trend = refine.position(close, refine.BASE)
    logr = np.concatenate([[np.nan], np.diff(np.log(close))])

    def k(name, when):
        return factors.known(name, *fac[name], when)

    fng = k("fng", m)
    x = np.column_stack([
        trend[sun],
        fng,
        fng - k("fng", m - 7),
        np.log(k("NASDAQCOM", m) / k("NASDAQCOM", m - 28)),
        np.log(k("DTWEXBGS", m) / k("DTWEXBGS", m - 28)),
        np.log(close[sun] / close[sun - 28]),
        np.array([np.std(logr[s - 29:s + 1]) for s in sun]),
    ])
    y = np.log(close[i + 6] / close[sun])
    ok = np.isfinite(x).all(axis=1)
    return Weekly(m[ok], x[ok], y[ok], i[ok])


def knot_weights(t_len: int, every: int = KNOT_WEEKS) -> np.ndarray:
    """(semanas × knots): interpolação linear entre knots igualmente espaçados do início ao fim do treino."""
    k = max(2, round((t_len - 1) / every) + 1)
    at = np.linspace(0, t_len - 1, k)
    t = np.arange(t_len)
    j = np.clip(np.searchsorted(at, t, side="right") - 1, 0, k - 2)
    frac = (t - at[j]) / (at[j + 1] - at[j])
    w = np.zeros((t_len, k))
    w[t, j] = 1 - frac
    w[t, j + 1] = frac
    return w


def model(x, w, y=None):
    b0 = numpyro.sample("b0", dist.Normal(0.0, 0.02))
    tau = numpyro.sample("tau", dist.HalfNormal(0.02))
    steps = numpyro.sample("steps", dist.Normal(0.0, 1.0).expand([w.shape[1] - 1]).to_event(1))
    knots = numpyro.deterministic("knots", b0 + jnp.concatenate([jnp.zeros(1), jnp.cumsum(tau * steps)]))
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.01).expand([x.shape[1]]).to_event(1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    with numpyro.plate("semanas", x.shape[0]):
        numpyro.sample("y", dist.StudentT(nu, w @ knots + x @ beta, sigma), obs=y)


@dataclass
class Fit:
    samples: dict[str, np.ndarray]  # as cadeias juntas
    mean: np.ndarray  # média e desvio do treino, usados para padronizar os fatores
    std: np.ndarray
    r_hat: float  # o pior entre todos os parâmetros
    n_eff: float  # o menor
    divergences: int


def fit(x: np.ndarray, y: np.ndarray, seeds=SEEDS, warmup: int = WARMUP, samples: int = SAMPLES,
        chain_method: str = "parallel") -> Fit:
    mean, std = x.mean(axis=0), x.std(axis=0)
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples, num_chains=len(seeds),
                chain_method=chain_method, progress_bar=False)
    keys = jnp.stack([random.PRNGKey(s) for s in seeds])  # uma semente por cadeia
    mcmc.run(keys, jnp.asarray((x - mean) / std), jnp.asarray(knot_weights(len(y))), jnp.asarray(y),
             extra_fields=("diverging",))
    by_chain = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    diag = summary(by_chain)
    return Fit({k: v.reshape(-1, *v.shape[2:]) for k, v in by_chain.items()}, mean, std,
               max(float(np.max(v["r_hat"])) for v in diag.values()),
               min(float(np.min(v["n_eff"])) for v in diag.values()),
               int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))


def p_up(f: Fit, x: np.ndarray) -> np.ndarray:
    """Chance de alta de cada semana: média, nas amostras, de P(Student-t(ν, μ, σ) > 0), com μ do último knot."""
    s = f.samples
    mu = s["knots"][:, -1:] + s["beta"] @ ((x - f.mean) / f.std).T  # (amostras × semanas)
    return stats.t.cdf(mu / s["sigma"][:, None], df=s["nu"][:, None]).mean(axis=0)


def years_of(days: np.ndarray) -> np.ndarray:
    return (days * 86400).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970


def walk_forward(wk: Weekly, seeds=SEEDS, first_year: int = FIRST_WF_YEAR, **kw):
    """Chance de alta fora da amostra das semanas de `first_year` em diante: um ajuste por ano, com tudo antes dele.

    Devolve (chance de alta, frequência de semanas de alta no treino de cada uma, [(ano, ajuste)])."""
    years = years_of(wk.monday)
    p, clim, fits = np.full(len(wk.y), np.nan), np.full(len(wk.y), np.nan), []
    for yr in range(first_year, int(years[-1]) + 1):
        test = np.flatnonzero(years == yr)
        if not len(test):
            continue
        train = np.arange(test[0])  # as semanas são seguidas: o alvo da última de treino acaba na primeira de teste
        f = fit(wk.x[train], wk.y[train], seeds, **kw)
        p[test] = p_up(f, wk.x[test])
        clim[test] = (wk.y[train] > 0).mean()
        fits.append((yr, f))
    return p, clim, fits


def luck_p(f: np.ndarray, rets: np.ndarray, n: int = LUCK, seed: int = 0) -> float:
    """Fração dos sorteios com Sharpe ≥ o da carteira semanal `f`: trechos de 0/1 em ordem sorteada (os mesmos
    trechos, com os mesmos comprimentos) ou, para frações, as semanas embaralhadas. `rets`: retornos diários."""
    rng = np.random.default_rng(seed)

    def sharpe_of(g):
        r = regime.returns_from_held(np.repeat(g, 7), rets)
        return r.mean() / r.std() if r.std() > 0 else np.nan

    real = sharpe_of(f)
    if not np.isfinite(real):
        return 1.0
    binary = bool(np.isin(f, (0.0, 1.0)).all())
    hits = sum(sharpe_of(regime.shuffled_timing(f, rng) if binary else rng.permutation(f)) >= real for _ in range(n))
    return hits / n


def score(r: np.ndarray, days: np.ndarray, held: np.ndarray) -> dict:
    years = years_of(days)
    m = study.metrics(r, (days * 86400).astype("datetime64[s]").astype("datetime64[M]").astype(int), years)
    changes = int((np.abs(np.diff(np.concatenate([[0.0], held]))) > 1e-9).sum())
    return {"total": float(np.prod(1 + r) - 1), "cagr": m["cagr"], "dd": m["dd"], "sharpe": m["sharpe"],
            "exposure": float(held.mean()), "orders_year": changes / (len(r) / 365.25),
            "by_year": {int(y): float(np.prod(1 + r[years == y]) - 1) for y in np.unique(years)}}


def evaluate(d: regime.Daily, wk: Weekly, rows: np.ndarray, p: np.ndarray, luck: int = LUCK) -> dict[str, dict]:
    """As formas (a) e (b) da chance de alta `p` (uma por semana de `rows`, seguidas) e a régua nos mesmos dias."""
    if (np.diff(wk.first[rows]) != 7).any():
        raise ValueError("as semanas precisam ser seguidas")
    lo, hi = int(wk.first[rows[0]]), int(wk.first[rows[-1]]) + 7
    rets = d.close[lo:hi] / d.close[lo - 1:hi - 1] - 1
    forms = dict(zip(FORMS, ((p > 0.5).astype(float), p)))
    held = {k: np.repeat(f, 7) for k, f in forms.items()}
    held[BENCH[0]] = refine.position(d.close, refine.BASE)[lo - 1:hi - 1]
    held[BENCH[1]] = learn.learn_positions(learn.comp_positions(d.close), d.close)[0][lo - 1:hi - 1]
    held[BENCH[2]] = np.ones(hi - lo)
    out = {k: score(regime.returns_from_held(h, rets), d.day[lo:hi], h) for k, h in held.items()}
    if luck:
        for k, f in forms.items():
            out[k]["p"] = luck_p(f, rets, luck)
    return out


def quality(p: np.ndarray, y: np.ndarray, clim: np.ndarray) -> dict:
    """Acerto da direção e nota de Brier (menor é melhor) contra prever sempre a frequência de alta do treino."""
    up = (y > 0).astype(float)
    return {"hit": float(((p > 0.5) == (up > 0)).mean()), "brier": float(((p - up) ** 2).mean()),
            "brier_clim": float(((clim - up) ** 2).mean()), "up": float(up.mean())}


def decide(wf: dict, test: dict, valid: bool) -> dict[str, list[str]]:
    """Aplica a regra fixada: para cada forma, os critérios que ela não cumpriu (lista vazia = cumpre tudo)."""
    fails = {}
    for form in FORMS:
        f = []
        if not wf[form]["sharpe"] > wf[BENCH[0]]["sharpe"]:
            f.append("Sharpe no walk-forward")
        if wf[form]["dd"] > wf[BENCH[0]]["dd"] + 1e-9:
            f.append("queda no walk-forward")
        if wf[form]["p"] > MAX_P:
            f.append("sorte")
        if test[form]["sharpe"] < test[BENCH[0]]["sharpe"] - 1e-9:
            f.append("Sharpe no teste de 13%")
        if not valid:
            f.append("ajuste")
        fails[form] = f
    return fails


def day_str(day) -> str:
    return str(np.datetime64(int(day), "D"))


def print_table(rows: dict[str, dict]) -> None:
    print(f"{'carteira':<24}{'total':>9}{'ao ano':>9}{'queda':>8}{'Sharpe':>8}{'exposição':>11}{'ordens/ano':>12}"
          f"{'p sorte':>9}")
    for k, s in rows.items():
        luck = f"{s['p']:>9.3f}" if "p" in s else ""
        print(f"{k:<24}{s['total']:>9.0%}{s['cagr']:>9.1%}{s['dd']:>8.0%}{s['sharpe']:>8.2f}{s['exposure']:>11.0%}"
              f"{s['orders_year']:>12.1f}{luck}")


def print_quality(q: dict) -> None:
    print(f"acerto da direção (P > 50% contra a semana real): {q['hit']:.1%} (semanas de alta: {q['up']:.1%});"
          f" Brier {q['brier']:.4f} contra {q['brier_clim']:.4f} de prever sempre a frequência do treino")


def print_fit(f: Fit) -> None:
    b = f.samples["beta"] * 100
    print(f"{'fator':<26}{'β médio':>9}{'faixa de 90%':>20}{'P(β > 0)':>10}")
    for i, name in enumerate(FACTORS):
        lo, hi = np.quantile(b[:, i], [0.05, 0.95])
        print(f"{name:<26}{b[:, i].mean():>8.2f}%{lo:>10.2f}% a {hi:>5.2f}%{(b[:, i] > 0).mean():>10.0%}")
    last = f.samples["knots"][:, -1] * 100
    lo, hi = np.quantile(last, [0.05, 0.95])
    print(f"linha de base no último knot: {last.mean():.2f}% por semana (90%: {lo:.2f}% a {hi:.2f}%);"
          f" τ {f.samples['tau'].mean():.4f}, σ {f.samples['sigma'].mean():.4f}, ν {f.samples['nu'].mean():.1f}")
    print(f"R-hat máximo {f.r_hat:.3f}, menor amostra efetiva {f.n_eff:.0f}, divergências {f.divergences}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True, help="candles de 5 min do BTC spot (ex.: o de wescalp.binance)")
    p.add_argument("--factors", type=Path, default=factors.DATA_DIR)
    p.add_argument("--luck", type=int, default=LUCK)
    args = p.parse_args()
    numpyro.set_host_device_count(len(SEEDS))  # uma cadeia por núcleo; precisa vir antes de o JAX iniciar
    numpyro.enable_x64()

    ts, o, h, l, c = study.load(args.file)
    d = regime.daily_bars(ts, o, h, l, c)
    fac = {name: factors.load(name, args.factors) for name in ("fng", *factors.FRED_SERIES)}
    wk = weekly(d, fac, (END - factors.EPOCH).days)
    n_train = round(TRAIN_SHARE * len(wk.y))
    train, test = np.arange(n_train), np.arange(n_train, len(wk.y))
    print(f"{len(wk.y)} semanas, decisões de {day_str(wk.monday[0])} a {day_str(wk.monday[-1])};"
          f" treino até a semana de {day_str(wk.monday[train[-1]])} ({len(train)}), teste de"
          f" {day_str(wk.monday[test[0]])} ({len(test)}); taxa {regime.FEE:.1%} por lado")

    main_fit = fit(wk.x[train], wk.y[train], SEEDS)
    check_fit = fit(wk.x[train], wk.y[train], CHECK_SEEDS)
    p_test, p_test2 = p_up(main_fit, wk.x[test]), p_up(check_fit, wk.x[test])
    print("\nβ de cada fator no treino de 87% (% do retorno esperado da semana por 1 desvio do fator):")
    print_fit(main_fit)
    print(f"\nteste de 13% ({len(test)} semanas):")
    test_rows = evaluate(d, wk, test, p_test, args.luck)
    print_table(test_rows)
    print_quality(quality(p_test, wk.y[test], np.full(len(test), (wk.y[train] > 0).mean())))
    print(f"chance de alta prevista: de {p_test.min():.1%} a {p_test.max():.1%}, média {p_test.mean():.1%}")

    p_wf, clim, fits = walk_forward(wk, SEEDS)
    p_wf2, _, fits2 = walk_forward(wk, CHECK_SEEDS)
    wf = np.flatnonzero(np.isfinite(p_wf))
    print(f"\nwalk-forward anual, fora da amostra ({day_str(wk.monday[wf[0]])} a {day_str(wk.monday[wf[-1]] + 6)},"
          f" {len(wf)} semanas):")
    wf_rows = evaluate(d, wk, wf, p_wf[wf], args.luck)
    print_table(wf_rows)
    print_quality(quality(p_wf[wf], wk.y[wf], clim[wf]))
    print("\npor ano (total):")
    print(f"{'carteira':<24}" + "".join(f"{y:>8}" for y in wf_rows[FORMS[0]]["by_year"]))
    for k, s in wf_rows.items():
        print(f"{k:<24}" + "".join(f"{x:>8.0%}" for x in s["by_year"].values()))
    print("\nβ médio de cada ajuste do walk-forward (% por desvio; treino com tudo antes do ano):")
    print(f"{'ano':<6}" + "".join(f"{s:>9}" for s in SHORT) + f"{'R-hat':>8}{'diverg.':>9}")
    for yr, f in fits:
        print(f"{yr:<6}" + "".join(f"{x:>9.2f}" for x in f.samples["beta"].mean(axis=0) * 100)
              + f"{f.r_hat:>8.3f}{f.divergences:>9}")

    all_fits = [main_fit, check_fit] + [f for _, f in fits + fits2]
    r_hat = max(f.r_hat for f in all_fits)
    agree_test = float(((p_test > 0.5) == (p_test2 > 0.5)).mean())
    agree_wf = float(((p_wf[wf] > 0.5) == (p_wf2[wf] > 0.5)).mean())
    test2, wf2 = evaluate(d, wk, test, p_test2, 0), evaluate(d, wk, wf, p_wf2[wf], 0)
    print(f"\nsegunda rodada (sementes {CHECK_SEEDS[0]} a {CHECK_SEEDS[-1]}): decisão (a) igual em {agree_test:.0%} das"
          f" semanas do teste e {agree_wf:.0%} do walk-forward; maior diferença na chance de alta"
          f" {max(np.abs(p_test - p_test2).max(), np.abs(p_wf[wf] - p_wf2[wf]).max()):.2%}")
    for k in FORMS:
        print(f"  {k}: Sharpe {test_rows[k]['sharpe']:.2f} → {test2[k]['sharpe']:.2f} no teste,"
              f" {wf_rows[k]['sharpe']:.2f} → {wf2[k]['sharpe']:.2f} no walk-forward")
    print(f"R-hat máximo entre os {len(all_fits)} ajustes: {r_hat:.3f};"
          f" divergências: {sum(f.divergences for f in all_fits)}")

    valid = r_hat <= MAX_R_HAT and min(agree_test, agree_wf) >= MIN_AGREE
    fails = decide(wf_rows, test_rows, valid)
    print()
    for k, f in fails.items():
        print(f"{k}: {'cumpre todos os critérios' if not f else 'não cumpre: ' + ', '.join(f)}")
    winners = [k for k, f in fails.items() if not f]
    print(f"pela regra fixada: {'entra como terceira carteira ao vivo: ' + winners[0] if winners else 'não entra'}")


if __name__ == "__main__":
    main()
