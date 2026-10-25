"""O que o histórico diz de cada sinal da congelada: quantas vezes cada média virou e o que aconteceu depois.

    python -m wescalp.odds --file data/candles/BINANCE-BTCUSDT-5m.csv.gz      # regrava wescalp/static/odds.json

A ideia vem da "Máquina do Tempo" do day-trader-skill (github.com/Jakeschincariol/day-trader-skill): antes de seguir
um sinal, olhar o que o mesmo sinal fez no passado. Aqui o sinal é a virada de uma das médias da congelada (50, 100 ou
150 dias) no fechamento das 00h UTC: o fechamento passa a ficar acima da média (compra daquele terço) ou abaixo
(venda). Para cada um dos 6 tipos de virada:
- quantas vezes aconteceu, a primeira e a última;
- o retorno do BTC 7, 30 e 90 dias depois do fechamento da virada: mediana e fração das vezes em que estava acima;
- nas compras, a operação daquele terço até a venda seguinte da mesma média, com 0,1% por lado: fração no lucro,
  mediana do ganho (das que ganharam), da perda (das que perderam) e da duração; as que ainda estavam abertas no fim
  dos dados ficam de fora e são contadas à parte;
- nas vendas, o que o BTC fez até a recompra seguinte da mesma média: fração das vezes em que recomprou mais barato,
  mediana da variação e da duração.
O primeiro dia em que a média passa a existir não conta como virada. É histórico, não previsão; e é o mesmo período
em que a congelada foi pesquisada, então não é uma prova nova dela (ver o README).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import regime, study

LOOKBACKS = (50, 100, 150)
HORIZONS = (7, 30, 90)
FEE = regime.FEE
OUT = Path(__file__).resolve().parent / "static" / "odds.json"
SOURCE = "Binance BTCUSDT spot, candles de 5 min agregados em dias UTC (wescalp.binance)"


def flips(close: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(dias em que o fechamento passou para cima da média de n dias, dias em que passou para baixo).

    Cada dia só olha os fechamentos até ele. O dia em que a média passa a existir não conta."""
    m = regime.sma(close, n)
    ok = ~np.isnan(m)
    on = close > np.where(ok, m, np.inf)  # igual a refine.above: fechamento igual à média não está acima
    both = ok[1:] & ok[:-1]  # a virada precisa da média no dia e no anterior
    up = np.flatnonzero(both & on[1:] & ~on[:-1]) + 1
    down = np.flatnonzero(both & ~on[1:] & on[:-1]) + 1
    return up, down


def iso(day: np.ndarray, k: int) -> str:
    return str(np.datetime64(int(day[k]), "D"))


def median(x) -> float | None:
    return float(np.median(x)) if len(x) else None


def share(mask) -> float | None:
    return float(np.mean(mask)) if len(mask) else None


def after(day: np.ndarray, close: np.ndarray, idx: np.ndarray, horizons=HORIZONS) -> dict[str, dict]:
    """Retorno do BTC k dias de calendário depois de cada dia de `idx` (se faltar o dia, o primeiro depois dele):
    quantos casos, mediana e fração acima. As viradas sem k dias de dados depois ficam de fora."""
    out = {}
    for k in horizons:
        j = np.searchsorted(day, day[idx] + k)
        ok = j < len(day)
        r = close[j[ok]] / close[idx[ok]] - 1
        out[str(k)] = {"n": len(r), "median": median(r), "up": share(r > 0)}
    return out


def pairs(starts: np.ndarray, ends: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    """Cada início com o primeiro fim depois dele; e quantos inícios ainda não tinham fim nos dados."""
    out, still = [], 0
    for s in starts:
        k = int(np.searchsorted(ends, s, side="right"))
        if k == len(ends):
            still += 1
        else:
            out.append((int(s), int(ends[k])))
    return out, still


def stats(day: np.ndarray, close: np.ndarray, n: int, fee: float = FEE) -> dict[str, dict]:
    """O histórico das duas viradas da média de n dias."""
    up, down = flips(close, n)
    out = {}
    for arrow, idx in (("↑", up), ("↓", down)):
        out[f"M{n} {arrow}"] = {"n": len(idx), "first": iso(day, idx[0]) if len(idx) else None,
                                "last": iso(day, idx[-1]) if len(idx) else None, "after": after(day, close, idx)}
    # compra no fechamento da virada para cima, venda no fechamento da virada para baixo seguinte, 0,1% em cada uma
    trades, still_open = pairs(up, down)
    r = np.array([close[b] / close[a] * (1 - fee) ** 2 - 1 for a, b in trades])
    out[f"M{n} ↑"]["trade"] = {
        "n": len(trades), "open": still_open, "win": share(r > 0), "median_gain": median(r[r > 0]),
        "median_loss": median(r[r <= 0]), "median_days": median([day[b] - day[a] for a, b in trades])}
    # depois da venda: o preço da recompra seguinte da mesma média contra o da venda (sem taxa: é só o caminho)
    waits, still_out = pairs(down, up)
    ch = np.array([close[b] / close[a] - 1 for a, b in waits])
    out[f"M{n} ↓"]["rebuy"] = {
        "n": len(waits), "out": still_out, "cheaper": share(ch < 0), "median": median(ch),
        "median_days": median([day[b] - day[a] for a, b in waits])}
    return out


def build(day: np.ndarray, close: np.ndarray, today: str, lookbacks=LOOKBACKS) -> dict:
    """O conteúdo de odds.json: de onde e de quando são os dados e o histórico de cada virada."""
    events = {}
    for n in lookbacks:
        events.update(stats(day, close, n))
    return {"source": SOURCE, "from": iso(day, 0), "to": iso(day, -1), "generated": today, "fee": FEE,
            "horizons": list(HORIZONS), "lookbacks": list(lookbacks), "events": events}


def rounded(x, digits: int = 4):
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v, digits) for v in x]
    return x


def pct(x, spec: str = ".0%") -> str:
    return "—" if x is None else format(x, spec)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", type=Path, required=True, help="candles de 5 min do BTC spot (ex.: o de wescalp.binance)")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    ts, o, h, l, c = study.load(args.file)
    d = regime.daily_bars(ts, o, h, l, c)
    body = rounded(build(d.day, d.close, datetime.now(timezone.utc).date().isoformat()))
    args.out.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {body['from']} a {body['to']}")
    print(f"{'virada':<9}{'vezes':>6}{'30 d: acima':>13}{'mediana':>9}   compra: até a venda / venda: até a recompra")
    for name, e in body["events"].items():
        a = e["after"]["30"]
        if "trade" in e:
            t = e["trade"]
            rest = (f"lucro em {pct(t['win'])} de {t['n']} ({pct(t['median_gain'], '+.1%')} /"
                    f" {pct(t['median_loss'], '+.1%')}), {pct(t['median_days'], '.0f')} dias; {t['open']} aberta(s)")
        else:
            b = e["rebuy"]
            rest = (f"recomprou mais barato em {pct(b['cheaper'])} de {b['n']} (mediana {pct(b['median'], '+.1%')}),"
                    f" {pct(b['median_days'], '.0f')} dias")
        print(f"{name:<9}{e['n']:>6}{pct(a['up']):>13}{pct(a['median'], '+.1%'):>9}   {rest}")


if __name__ == "__main__":
    main()
