"""Prova única no trecho trancado: BTC/USD da Bitstamp de 01/01/2013 a 16/08/2017, que nenhuma regra viu.

    WESCALP_UNLOCK=prova-unica python -m wescalp.proof

Critérios fixados antes de abrir os dados (23/10/2026), sem ajuste depois. Mesmas contas da pesquisa: decisão no
fechamento das 00h UTC valendo para o dia seguinte, 0,1% por lado, tudo reinvestido; o histórico desde 18/08/2011
só aquece as médias e o aprendiz.
- a congelada 50/100/150 passa se cumprir tudo: lucro no período, Sharpe maior que o de segurar o BTC, queda máxima
  menor que a de segurar e p ≤ 0,05 no teste da sorte (cada média com o momento sorteado por conta própria, 2.000
  sorteios, como em wescalp.refine);
- o aprendiz é julgado pela mesma régua (teste da sorte de wescalp.learn, 1.000 sorteios);
- as regras clássicas de wescalp.regime aparecem só como contexto: nenhuma substitui a congelada por ter ido bem
  aqui.
Antes de rodar, o conteúdo do arquivo precisa ter o SHA-256 registrado no README quando ele foi trancado.
"""

import math
from datetime import datetime, timezone

import numpy as np

from . import bitstamp, brain, learn, refine, regime, study

DIGEST = "5fb3f3543026448e83de75f0ec0b201429687dcccf6fdd33da65a60fe5e8d78e"
START = datetime.combine(bitstamp.EVAL_START, datetime.min.time(), timezone.utc)
HOLD, FROZEN, LEARNER = "segurar BTC", "congelada 50/100/150", learn.LEARNER
LUCK_FROZEN, LUCK_LEARNER = 2000, 1000
MAX_P = 0.05


def criteria(s: dict, hold: dict) -> list[str]:
    """Os critérios que a estratégia não cumpriu (lista vazia = passou)."""
    fails = []
    if not s["total"] > 0:
        fails.append("lucro")
    if not s["sharpe"] > hold["sharpe"]:
        fails.append("Sharpe")
    if not s["dd"] < hold["dd"]:
        fails.append("queda")
    if not s["p"] <= MAX_P:
        fails.append("sorte")
    return fails


def positions(close: np.ndarray) -> dict[str, np.ndarray]:
    return {HOLD: np.ones(len(close)), FROZEN: refine.position(close, refine.BASE),
            LEARNER: learn.learn_positions(learn.comp_positions(close), close)[0], **regime.rules(close)}


def check_digest(p) -> None:
    got = bitstamp.digest(p)
    if got != DIGEST:
        raise SystemExit(f"{p}: SHA-256 {got} diferente do registrado ({DIGEST}); os dados mudaram, não rodo")


def main() -> None:
    p = bitstamp.path()
    check_digest(p)
    try:
        cs = brain.load_candles(p, bitstamp.SYMBOL, "spot")
    except PermissionError as e:
        raise SystemExit(str(e))
    ts = np.array([int(c.ts.timestamp()) for c in cs], dtype=np.int64)
    o, h, l, c = (np.array([getattr(x, k) for x in cs]) for k in ("open", "high", "low", "close"))
    start_day = int(START.timestamp() // 86400)

    runs = {}
    for off in study.OFFSETS_H:
        d = regime.daily_bars(ts, o, h, l, c, off * 3600)
        lo = int(np.searchsorted(d.day, start_day))
        rets = d.close[lo:] / d.close[lo - 1:-1] - 1
        runs[off] = (d, lo, rets, {k: regime.held_from(v, lo) for k, v in positions(d.close).items()})
    d, lo, rets, held = runs[0]
    when = (d.day[lo:] * 86400).astype("datetime64[s]")
    months, years = when.astype("datetime64[M]").astype(int), when.astype("datetime64[Y]").astype(int) + 1970
    scores = {}
    for k, hd in held.items():
        r = regime.returns_from_held(hd, rets)
        m = study.metrics(r, months, years)
        changes = int((np.abs(np.diff(np.concatenate([[0.0], hd]))) > 1e-12).sum())
        scores[k] = {
            "total": float(np.prod(1 + r) - 1), "cagr": m["cagr"], "dd": m["dd"], "sharpe": m["sharpe"],
            "exposure": float(hd.mean()), "orders_year": changes / (len(r) / 365.25),
            "worst_hour": min(study.cagr(regime.returns_from_held(x[3][k], x[2])) for x in runs.values()),
            "fee5": study.cagr(regime.returns_from_held(hd, rets, 0.005)),
            "by_year": {int(y): float(np.prod(1 + r[years == y]) - 1) for y in np.unique(years)},
        }
    scores[FROZEN]["p"] = refine.luck_p(d.close, lo, refine.BASE, LUCK_FROZEN)
    scores[LEARNER]["p"] = learn.luck_p(d.close, lo, learn.LEARNER, LUCK_LEARNER)

    fmt = lambda k: datetime.fromtimestamp(int(d.day[k]) * 86400, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    print(f"prova única: Bitstamp BTC/USD, {fmt(lo)} a {fmt(len(d.day) - 1)} ({len(rets)} dias); SHA-256 conferido;"
          f" taxa {regime.FEE:.1%} por lado")
    print(f"{'estratégia':<22}{'total':>10}{'ao ano':>9}{'queda':>8}{'Sharpe':>8}{'pior hora':>11}{'0,5%/lado':>11}"
          f"{'exposição':>11}{'ordens/ano':>12}{'p sorte':>9}")
    for k, s in scores.items():
        luck = f"{s['p']:>9.3f}" if "p" in s else ""
        print(f"{k:<22}{s['total']:>10.0%}{s['cagr']:>9.1%}{s['dd']:>8.0%}{s['sharpe']:>8.2f}{s['worst_hour']:>11.1%}"
              f"{s['fee5']:>11.1%}{s['exposure']:>11.0%}{s['orders_year']:>12.1f}{luck}")
    ys = list(scores[HOLD]["by_year"])
    print("\npor ano (2017 até 16/08):")
    print(f"{'estratégia':<22}" + "".join(f"{y:>9}" for y in ys))
    for k in (HOLD, FROZEN, LEARNER):
        print(f"{k:<22}" + "".join(f"{scores[k]['by_year'][y]:>9.0%}" for y in ys))
    print()
    for k in (FROZEN, LEARNER):
        f = criteria(scores[k], scores[HOLD])
        print(f"{k}: {'PASSOU (lucro, Sharpe, queda e sorte)' if not f else 'não passou: ' + ', '.join(f)}")


if __name__ == "__main__":
    main()
