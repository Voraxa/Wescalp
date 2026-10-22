"""LightGBM sobre os sinais de fluxo dos dias da Tardis, contra regras simples, em walk-forward por dia.

    python -m wescalp.ml --horizons 60 300

Uma linha por segundo (balde de 1 s de wescalp.flow.aggregate), só com o que já aconteceu até o fim dele:
OFI e delta somados em janelas de 1 a 60 s e divididos pelo |OFI| (|delta|) médio por segundo dos 10 min
anteriores (o tamanho das filas muda muito em 2 anos), desequilíbrio dos 5 níveis, microprice, spread, retorno
passado do mid, volatilidade realizada e a hora UTC. O alvo é o retorno do mid nos `h` segundos seguintes, em bps.

Walk-forward por dia: treina nos primeiros 10 dias, testa nos 3 seguintes, anda 3 (dias de meses diferentes, então
nada vaza de um para o outro). Parâmetros fixados antes de rodar. Concorrentes, cada um com o corte tirado só do
treino:
- tail:<sinal>: o sinal sozinho, no lado da correlação do treino, só acima do quantil 99% / 99,9% do |sinal|;
- linear: regressão ridge com todas as variáveis;
- lgbm: LightGBM com as mesmas variáveis.
Para linear e lgbm também a regra "|previsão| > custo" (previsão em bps).

Simulação: a cada disparo entra no mid, sai no mid `h` segundos depois e só volta a olhar sinal depois de sair
(sem posições sobrepostas). Bruto em bps e líquido de 4 (maker+maker) e 8 (maker+taker). Entrar e sair no mid como
maker é otimista: ignora fila, ordem não executada e seleção adversa. As últimas `--holdout` datas ficam de fora.
"""

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import flow, tardis

OFI_WIN = (1, 5, 10, 30, 60)
DELTA_WIN = (5, 30, 60)
RET_WIN = (10, 60, 300)
VOL_WIN = (60, 300)
NORM_WIN = 600  # segundos da média de |OFI| e |delta| usada para normalizar
FEATURES = (
    [f"ofi_{w}" for w in OFI_WIN] + [f"delta_{w}" for w in DELTA_WIN]
    + ["imb5", "micro_bps", "spread_bps"]
    + [f"ret_{w}" for w in RET_WIN] + [f"vol_{w}" for w in VOL_WIN] + ["hour"]
)
TAILS = ("imb5", "micro_bps", "ofi_10")
QUANTILES = (0.99, 0.999)
COSTS = (4.0, 8.0)
TRAIN_DAYS, TEST_DAYS = 10, 3
TRAIN_STRIDE = 5  # usa 1 segundo a cada 5 no treino: alvos vizinhos são quase o mesmo número
CLIP_BPS = 50.0  # corta o alvo do treino para um pico não dominar o ajuste
LGBM_ROUNDS = 300
LGBM_PARAMS = dict(
    objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000,
    bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, lambda_l2=10.0, seed=0, verbose=-1,
)


def arr(xs) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in xs], dtype=np.float64)


def rsum(x: np.ndarray, w: int) -> np.ndarray:
    """Soma dos últimos w valores até t, inclusive; NaN antes de haver w valores. NaN na entrada conta como 0."""
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out = np.full(len(x), np.nan)
    if w <= len(x):
        out[w - 1:] = c[w:] - c[:len(x) - w + 1]
    return out


def past_ret_bps(mid: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(mid), np.nan)
    out[w:] = (mid[w:] / mid[:-w] - 1) * 1e4
    return out


def fwd_ret_bps(mid: np.ndarray, h: int) -> np.ndarray:
    out = np.full(len(mid), np.nan)
    out[:-h] = (mid[h:] / mid[:-h] - 1) * 1e4
    return out


def features(cols: dict[str, list]) -> np.ndarray:
    """Matriz (baldes × FEATURES) a partir das colunas de wescalp.flow.aggregate com balde de 1 s."""
    mid = arr(cols["mid"])
    ofi, delta = arr(cols["ofi"]), arr(cols["delta"])
    ofi_scale = rsum(np.abs(ofi), NORM_WIN) / NORM_WIN
    delta_scale = rsum(np.abs(delta), NORM_WIN) / NORM_WIN
    one = past_ret_bps(mid, 1)
    out = {}
    for w in OFI_WIN:
        out[f"ofi_{w}"] = rsum(ofi, w) / (ofi_scale * w)
    for w in DELTA_WIN:
        out[f"delta_{w}"] = rsum(delta, w) / (delta_scale * w)
    out["imb5"], out["micro_bps"], out["spread_bps"] = arr(cols["imb5"]), arr(cols["micro_bps"]), arr(cols["spread_bps"])
    for w in RET_WIN:
        out[f"ret_{w}"] = past_ret_bps(mid, w)
    for w in VOL_WIN:
        out[f"vol_{w}"] = rsum(np.abs(one), w)
    t = np.array(cols["t"], dtype=np.float64)
    out["hour"] = (t % 86400) // 3600
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.column_stack([out[f] for f in FEATURES])
    m[~np.isfinite(m)] = np.nan
    m[np.isnan(mid)] = np.nan  # balde inválido (buraco) fica inteiro de fora
    return m


@dataclass
class Day:
    day: str
    x: np.ndarray
    y: dict[int, np.ndarray]  # retorno futuro em bps por horizonte


def load(days, data_dir: Path, horizons: list[int], max_gap: float) -> list[Day]:
    out = []
    for d in days:
        quotes, ticks = tardis.load_day(d, data_dir)
        cols = flow.aggregate(quotes, ticks, 1.0, max_gap)
        mid = arr(cols["mid"])
        out.append(Day(str(d), features(cols), {h: fwd_ret_bps(mid, h) for h in horizons}))
    return out


def folds(n: int, train: int = TRAIN_DAYS, test: int = TEST_DAYS) -> list[tuple[range, range]]:
    """Janela crescente: treina em [0, k), testa em [k, k + test), k anda `test`."""
    return [(range(0, k), range(k, min(k + test, n))) for k in range(train, n, test)]


def stack(days: list[Day], h: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for d in days:
        ok = ~np.isnan(d.x).any(axis=1) & ~np.isnan(d.y[h])
        idx = np.flatnonzero(ok)[::stride]
        xs.append(d.x[idx])
        ys.append(d.y[h][idx])
    return np.concatenate(xs), np.concatenate(ys)


def simulate(pred: np.ndarray, fire: np.ndarray, y: np.ndarray, h: int) -> list[float]:
    """Retornos brutos (bps) das entradas sem sobreposição: entra no lado de `pred` onde `fire`, fica h segundos."""
    out, t, n = [], 0, len(y)
    while t < n:
        if fire[t] and not math.isnan(y[t]):
            out.append(math.copysign(1.0, pred[t]) * y[t])
            t += h
        else:
            t += 1
    return out


class Ridge:
    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Ridge":
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0) + 1e-12
        z = np.column_stack([np.ones(len(x)), (x - self.mu) / self.sd])
        reg = self.lam * len(x) * 1e-3 * np.eye(z.shape[1])
        reg[0, 0] = 0
        self.beta = np.linalg.solve(z.T @ z + reg, z.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(x)), (x - self.mu) / self.sd]) @ self.beta


class Lgbm:
    """API nativa do LightGBM (a de scikit-learn exige o scikit-learn instalado)."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Lgbm":
        import lightgbm

        self.booster = lightgbm.train(LGBM_PARAMS, lightgbm.Dataset(x, y), num_boost_round=LGBM_ROUNDS)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(x)


def nan_safe_predict(model, x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x).any(axis=1)
    out = np.full(len(x), np.nan)
    if ok.any():
        out[ok] = model.predict(x[ok])
    return out


def run(days: list[Day], h: int) -> tuple[dict[str, list[tuple[str, float]]], list[tuple[str, float]]]:
    """Walk-forward de um horizonte. Devolve {regra: [(dia, bruto_bps), ...]} e a importância do último lgbm."""
    trades: dict[str, list[tuple[str, float]]] = {}
    importance: list[tuple[str, float]] = []

    def add(name: str, day: Day, pred: np.ndarray, fire: np.ndarray):
        trades.setdefault(name, []).extend((day.day, r) for r in simulate(pred, fire, day.y[h], h))

    for train_idx, test_idx in folds(len(days)):
        xtr, ytr = stack([days[i] for i in train_idx], h, TRAIN_STRIDE)
        ytr_fit = np.clip(ytr, -CLIP_BPS, CLIP_BPS)
        models = {"linear": Ridge().fit(xtr, ytr_fit), "lgbm": Lgbm().fit(xtr, ytr_fit)}
        cuts = {}
        for name, m in models.items():
            p = np.abs(m.predict(xtr))
            cuts[name] = {q: np.quantile(p, q) for q in QUANTILES}
        tail_side, tail_cut = {}, {}
        for f in TAILS:
            j = FEATURES.index(f)
            tail_side[f] = math.copysign(1.0, np.corrcoef(xtr[:, j], ytr)[0, 1])
            tail_cut[f] = {q: np.quantile(np.abs(xtr[:, j]), q) for q in QUANTILES}
        importance = sorted(zip(FEATURES, models["lgbm"].booster.feature_importance("gain")), key=lambda r: -r[1])

        for i in test_idx:
            d = days[i]
            for name, m in models.items():
                p = nan_safe_predict(m, d.x)
                a = np.nan_to_num(np.abs(p))
                for q in QUANTILES:
                    add(f"{name} top {1 - q:.1%}", d, p, a > cuts[name][q])
                for c in COSTS:
                    add(f"{name} |prev|>{c:g}", d, p, a > c)
            for f in TAILS:
                s = d.x[:, FEATURES.index(f)] * tail_side[f]
                a = np.nan_to_num(np.abs(s))
                for q in QUANTILES:
                    add(f"tail:{f} top {1 - q:.1%}", d, s, a > tail_cut[f][q])
    return trades, importance


def summarize(trades: list[tuple[str, float]], test_days: list[str]) -> dict:
    r = np.array([x for _, x in trades])
    by_day = {d: 0.0 for d in test_days}
    for d, x in trades:
        by_day[d] += x - COSTS[0]
    n = len(r)
    return {
        "n": n,
        "gross": r.mean() if n else math.nan,
        "t": r.mean() / (r.std(ddof=1) / math.sqrt(n)) if n > 2 and r.std() > 0 else math.nan,
        "net4": r.mean() - COSTS[0] if n else math.nan,
        "net8": r.mean() - COSTS[1] if n else math.nan,
        "total4": (r - COSTS[0]).sum() if n else 0.0,
        "days_pos": sum(v > 0 for d, v in by_day.items() if any(x[0] == d for x in trades)),
        "days_traded": len({d for d, _ in trades}),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=tardis.DATA_DIR, help="pasta dos dias convertidos (wescalp.tardis)")
    p.add_argument("--horizons", type=int, nargs="+", default=[60, 300], help="segundos até a saída")
    p.add_argument("--holdout", type=int, default=tardis.HOLDOUT, help="últimas datas deixadas de fora")
    p.add_argument("--max-gap", type=float, default=tardis.MAX_GAP, help="segundos sem foto que contam como buraco")
    args = p.parse_args()

    all_days = tardis.available_days(args.data)
    days = all_days[:-args.holdout] if args.holdout else all_days
    if len(days) <= TRAIN_DAYS:
        raise SystemExit(f"precisa de mais de {TRAIN_DAYS} dias convertidos; rode: python -m wescalp.tardis fetch")
    print(f"{len(days)} dias de {days[0]} a {days[-1]}; fora da amostra (intocados): {len(all_days) - len(days)}")
    loaded = load(days, args.data, args.horizons, args.max_gap)
    fs = folds(len(loaded))
    test_days = [loaded[i].day for _, te in fs for i in te]
    print(f"walk-forward: {len(fs)} rodadas, treino a partir de {TRAIN_DAYS} dias, teste {TEST_DAYS};"
          f" {len(test_days)} dias de teste: {test_days[0]} a {test_days[-1]}")
    print("bruto/líq.: bps por entrada; t: estatística t do bruto; dias+: dias de teste com líquido(4) positivo")
    print(f"gerado em {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")

    for h in args.horizons:
        trades, importance = run(loaded, h)
        print(f"\nh = {h} s")
        print(f"{'regra':<26}{'entradas':>9}{'bruto':>8}{'t':>7}{'líq.4':>8}{'líq.8':>8}{'total4':>10}{'dias+':>8}")
        for name in sorted(trades, key=lambda k: (k.split()[0], k)):
            s = summarize(trades[name], test_days)
            print(f"{name:<26}{s['n']:>9}{s['gross']:>8.2f}{s['t']:>7.1f}{s['net4']:>8.2f}{s['net8']:>8.2f}"
                  f"{s['total4']:>10.0f}{s['days_pos']:>5}/{s['days_traded']:<2}")
        print("importância no último lgbm (gain): "
              + ", ".join(f"{f} {g / sum(x for _, x in importance):.0%}" for f, g in importance[:6]))


if __name__ == "__main__":
    main()
