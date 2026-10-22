"""Roda os setups em walk-forward nas 100 semanas e grava o relatório da Fase 1.

    python -m wescalp.research.run [--parquet data/candles_futures_5m.parquet]

Saída: reports/fase1.md (para ler) e reports/fase1.json (para as próximas fases).
"""

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd

from .evaluate import beta_posterior, breakdown, deflated_sharpe, metrics, walk_forward
from .features import add_features
from .setups import SETUPS

RAIZ = Path(__file__).resolve().parents[2]


def _bps(x: float) -> float:
    return round(x * 1e4, 2)


def avaliar(df: pd.DataFrame, setup) -> dict:
    wf = walk_forward(df, setup)
    tr = wf.trades
    m = metrics(tr) if len(tr) else {"n": 0}
    out = {"setup": setup.name, "n_trials": wf.n_trials, "oos": m}
    if not len(tr):
        return out

    wins = int((tr["ret_net"] > 0).sum())
    out["win_rate_posterior"] = beta_posterior(wins, len(tr))
    r = tr["ret_net"]
    sr = r.mean() / r.std(ddof=1)
    out["dsr"] = deflated_sharpe(sr, len(r), wf.n_trials, float(r.skew()), float(r.kurt()) + 3,
                                 float(wf.trial_sharpes.var(ddof=1)))
    # Separa o que é falta de padrão do que é custo: média antes das taxas.
    out["avg_gross_bps"] = _bps(tr["ret_gross"].mean())
    out["avg_cost_bps"] = _bps(tr["cost"].mean())
    out["por_regime"] = breakdown(tr, "regime").reset_index().to_dict("records")
    out["por_hora"] = breakdown(tr, "hour").reset_index().to_dict("records")
    out["params_escolhidos"] = wf.folds["params"].value_counts().to_dict()
    out["dobras"] = wf.folds[["fold", "test_start", "status", "params"]
                             + [c for c in wf.folds if c in ("test_n", "test_avg_net_bps")]].astype(str).to_dict("records")
    return out


def _linha_setup(r: dict) -> str:
    m = r["oos"]
    if not m.get("n"):
        return f"| {r['setup']} | 0 | – | – | – | – | – | – |"
    wr, p05, p95 = r["win_rate_posterior"]
    return (f"| {r['setup']} | {m['n']} | {m['trades_por_dia']:.1f} | {wr:.1%} ({p05:.1%}–{p95:.1%}) "
            f"| {r['avg_gross_bps']:+.1f} | {r['avg_cost_bps']:.1f} | {m['avg_net_bps']:+.1f} "
            f"| {r['dsr']:.2f} |")


def relatorio(resultados: list[dict], periodo: str, segundos: float) -> str:
    out = [
        "# Fase 1 — setups em walk-forward",
        "",
        f"Perpétuo XBTUSDTM, candles de 5 min, {periodo}. Treino de 60 semanas, teste nas 4 seguintes, "
        "andando 4 semanas. Só trades fora da amostra entram nas métricas. Custos: entrada maker 0,02%; "
        "saída no alvo maker 0,02%; stop ou tempo taker 0,06% + 0,01% de derrapagem.",
        "",
        "| Setup | Trades | Por dia | Acerto (intervalo 90%) | Bruto (bps) | Custo (bps) | Líquido (bps) | DSR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *(_linha_setup(r) for r in resultados),
        "",
        "DSR = probabilidade de o Sharpe verdadeiro ser positivo, descontando as combinações testadas "
        "(Bailey & López de Prado, 2014). Abaixo de 0,95 não há evidência de vantagem.",
        "",
    ]
    for r in resultados:
        if not r["oos"].get("n"):
            continue
        out += [f"## {r['setup']}", "", "Por regime (líquido em bps):", "",
                "| Regime | Trades | Acerto | Líquido |", "| --- | --- | --- | --- |"]
        for b in r["por_regime"]:
            out.append(f"| {b['regime']} | {b['n']} | {b['win_rate']:.1%} | {b['avg_net_bps']:+.1f} |")
        horas = sorted(r["por_hora"], key=lambda b: b["avg_net_bps"], reverse=True)
        melhores = ", ".join(f"{b['hour']}h ({b['avg_net_bps']:+.1f}, n={b['n']})" for b in horas[:3])
        piores = ", ".join(f"{b['hour']}h ({b['avg_net_bps']:+.1f}, n={b['n']})" for b in horas[-3:])
        out += ["", f"Melhores horas UTC: {melhores}. Piores: {piores}.", "",
                "Parâmetros escolhidos por dobra: "
                + "; ".join(f"`{p}` ×{n}" for p, n in r["params_escolhidos"].items()), ""]
    out.append(f"_Gerado em {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, {segundos:.0f} s._")
    return "\n".join(out) + "\n"


def _json_default(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return str(o)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", default=str(RAIZ / "data" / "candles_futures_5m.parquet"))
    p.add_argument("--saida", default=str(RAIZ / "reports"))
    args = p.parse_args()

    t0 = time.monotonic()
    df = add_features(pd.read_parquet(args.parquet))
    periodo = f"{df.index.min():%Y-%m-%d} a {df.index.max():%Y-%m-%d}"
    resultados = [avaliar(df, s) for s in SETUPS.values()]
    segundos = time.monotonic() - t0

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "fase1.md").write_text(relatorio(resultados, periodo, segundos))
    (saida / "fase1.json").write_text(json.dumps(resultados, default=_json_default, indent=2, ensure_ascii=False))
    print((saida / "fase1.md").read_text())


if __name__ == "__main__":
    main()
