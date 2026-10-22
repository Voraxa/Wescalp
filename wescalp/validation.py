"""Validação contra sobreajuste: walk-forward, Sharpe, PSR, Deflated Sharpe e PBO. Só biblioteca padrão.

Sharpe por período (sem anualizar), com T retornos r_t:

    SR = média(r) / desvio(r)                        (desvio amostral, T - 1)

Probabilistic Sharpe Ratio (Bailey e López de Prado, "The Sharpe Ratio Efficient Frontier", Journal of Risk,
2012), probabilidade de o Sharpe verdadeiro passar de SR*, dados assimetria g3 e curtose g4 (não excesso,
normal = 3) dos retornos:

    PSR(SR*) = Φ( (SR - SR*) · √(T - 1) / √(1 - g3·SR + (g4 - 1)/4 · SR²) )

Deflated Sharpe Ratio (Bailey e López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias,
Backtest Overfitting and Non-Normality", Journal of Portfolio Management, 2014): o PSR com SR* igual ao maior
Sharpe que se espera achar por sorte entre N tentativas sem vantagem, cujos Sharpes têm variância V:

    SR* = √V · ( (1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ),   γ = 0,5772… (Euler-Mascheroni)
    DSR = PSR(SR*)

Probability of Backtest Overfitting por CSCV (Bailey, Borwein, López de Prado e Zhu, "The Probability of
Backtest Overfitting", Journal of Computational Finance, 2017): a matriz M (T períodos × N configurações) é
cortada em S blocos de linhas consecutivas (S par). Para cada uma das C(S, S/2) escolhas de metade dos blocos
como dentro da amostra (o resto é fora), pega a configuração n* de maior Sharpe dentro, acha o rank ω de n*
fora (ω = rank / (N + 1), rank 1 = pior) e λ = ln(ω / (1 - ω)). PBO = fração das combinações com λ ≤ 0, isto
é, em que a melhor de dentro ficou na metade de baixo fora.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from statistics import NormalDist
from typing import NamedTuple

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


class Split(NamedTuple):
    train_start: datetime
    train_end: datetime  # = test_start
    test_start: datetime
    test_end: datetime


def walk_forward_splits(
    start: datetime,
    end: datetime,
    train: timedelta = timedelta(weeks=60),
    test: timedelta = timedelta(weeks=4),
    step: timedelta = timedelta(weeks=4),
) -> list[Split]:
    """Janelas móveis [treino)[teste) que andam `step` por vez; o último teste termina até `end`."""
    out = []
    s = start
    while s + train + test <= end:
        out.append(Split(s, s + train, s + train, s + train + test))
        s += step
    return out


def sharpe(returns: list[float]) -> float:
    """Média sobre desvio-padrão amostral, por período. 0 se não há variação (ou menos de 2 períodos)."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    return mean / math.sqrt(var) if var > 0 else 0.0


def moments(returns: list[float]) -> tuple[float, float]:
    """Assimetria e curtose (não excesso: normal = 3) com momentos populacionais, como nos papers."""
    n = len(returns)
    mean = sum(returns) / n
    m2 = sum((r - mean) ** 2 for r in returns) / n
    if m2 == 0:
        return 0.0, 3.0
    m3 = sum((r - mean) ** 3 for r in returns) / n
    m4 = sum((r - mean) ** 4 for r in returns) / n
    return m3 / m2 ** 1.5, m4 / m2 ** 2


def psr(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(Sharpe verdadeiro > sr_benchmark) com `n` retornos."""
    den = 1 - skew * sr + (kurt - 1) / 4 * sr * sr  # ≥ 0 sempre, porque kurt ≥ 1 + skew²
    if n < 2 or den <= 0:
        return 0.5
    return _N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(den))


def probabilistic_sharpe(returns: list[float], sr_benchmark: float = 0.0) -> float:
    skew, kurt = moments(returns)
    return psr(sharpe(returns), len(returns), skew, kurt, sr_benchmark)


def expected_max_sharpe(variance: float, n_trials: int) -> float:
    """Maior Sharpe esperado entre `n_trials` tentativas sem vantagem cujos Sharpes têm `variance`."""
    if n_trials <= 1 or variance <= 0:
        return 0.0
    return math.sqrt(variance) * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(returns: list[float], trial_sharpes: list[float]) -> float:
    """DSR dos `returns` escolhidos entre as tentativas (N = len(trial_sharpes), V = variância amostral delas).

    Os Sharpes das tentativas têm de estar na mesma frequência dos `returns`.
    """
    n = len(trial_sharpes)
    mean = sum(trial_sharpes) / n if n else 0.0
    var = sum((s - mean) ** 2 for s in trial_sharpes) / (n - 1) if n > 1 else 0.0
    return probabilistic_sharpe(returns, expected_max_sharpe(var, n))


class WalkForwardStep(NamedTuple):
    train: tuple[int, int]  # linhas [lo, hi)
    test: tuple[int, int]
    best: int  # coluna escolhida no treino
    train_sharpe: float
    test_returns: list[float]


def walk_forward(matrix: list[list[float]], splits: list[tuple[int, int, int, int]]) -> list[WalkForwardStep]:
    """Em cada split (linhas treino_lo, treino_hi, teste_lo, teste_hi), escolhe a coluna de maior Sharpe no
    treino e guarda os retornos dela no teste. Empate fica com a primeira coluna."""
    cols = list(zip(*matrix))
    out = []
    for a, b, c, d in splits:
        scores = [sharpe(list(col[a:b])) for col in cols]
        best = max(range(len(cols)), key=scores.__getitem__)
        out.append(WalkForwardStep((a, b), (c, d), best, scores[best], list(cols[best][c:d])))
    return out


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    logits: list[float]  # λ de cada combinação
    is_sharpes: list[float]  # Sharpe dentro da amostra da escolhida, por combinação
    oos_sharpes: list[float]  # Sharpe fora da amostra da mesma configuração

    @property
    def prob_oos_loss(self) -> float:
        """Fração das combinações em que a escolhida perde dinheiro fora da amostra."""
        return sum(s < 0 for s in self.oos_sharpes) / len(self.oos_sharpes)


def _block_bounds(t: int, s: int) -> list[tuple[int, int]]:
    """S blocos consecutivos de tamanhos que diferem no máximo em 1 (T nem sempre é múltiplo de S)."""
    q, r = divmod(t, s)
    out, lo = [], 0
    for k in range(s):
        hi = lo + q + (1 if k < r else 0)
        out.append((lo, hi))
        lo = hi
    return out


def _sharpe_from_sums(n: int, s: float, ss: float) -> float:
    if n < 2:
        return 0.0
    var = (ss - s * s / n) / (n - 1)
    return (s / n) / math.sqrt(var) if var > 1e-300 else 0.0


def pbo(matrix: list[list[float]], n_blocks: int = 16) -> PBOResult:
    """PBO por CSCV. `matrix[t][n]` = retorno da configuração n no período t; `n_blocks` par.

    Cada combinação soma, por configuração, a soma e a soma dos quadrados dos blocos de dentro; fora é o total
    menos dentro. Rank fora com empates pela média.
    """
    t, n_cfg = len(matrix), len(matrix[0])
    if n_blocks % 2 or n_blocks < 2 or t < 2 * n_blocks:
        raise ValueError("n_blocks precisa ser par e ter ao menos 2 períodos por bloco")
    bounds = _block_bounds(t, n_blocks)
    sizes = [hi - lo for lo, hi in bounds]
    sums = [[sum(matrix[i][j] for i in range(lo, hi)) for j in range(n_cfg)] for lo, hi in bounds]
    sqs = [[sum(matrix[i][j] ** 2 for i in range(lo, hi)) for j in range(n_cfg)] for lo, hi in bounds]
    tot_s = [sum(col) for col in zip(*sums)]
    tot_ss = [sum(col) for col in zip(*sqs)]

    logits, is_sr, oos_sr = [], [], []
    for combo in combinations(range(n_blocks), n_blocks // 2):
        n_in = sum(sizes[k] for k in combo)
        n_out = t - n_in
        s_in = [sum(v) for v in zip(*(sums[k] for k in combo))]
        ss_in = [sum(v) for v in zip(*(sqs[k] for k in combo))]
        sr_in = [_sharpe_from_sums(n_in, a, b) for a, b in zip(s_in, ss_in)]
        best = max(range(n_cfg), key=sr_in.__getitem__)
        sr_out = [_sharpe_from_sums(n_out, ts - a, tss - b) for ts, tss, a, b in zip(tot_s, tot_ss, s_in, ss_in)]
        x = sr_out[best]
        below = sum(v < x for v in sr_out)
        ties = sum(v == x for v in sr_out)  # inclui a própria
        rank = below + (ties + 1) / 2
        w = rank / (n_cfg + 1)
        logits.append(math.log(w / (1 - w)))
        is_sr.append(sr_in[best])
        oos_sr.append(x)
    return PBOResult(sum(lam <= 0 for lam in logits) / len(logits), logits, is_sr, oos_sr)
