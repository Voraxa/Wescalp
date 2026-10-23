"""Fatores de fora do preço do BTC para o modelo bayesiano (wescalp.bayes): sentimento, bolsa e dólar.

    python -m wescalp.factors        # baixa ou atualiza data/factors/

- Medo e Ganância (alternative.me, diário desde 01/02/2018): 0 = medo extremo, 100 = ganância extrema. O valor com
  carimbo do dia D sai por volta das 00h UTC de D; aqui só vale a partir do dia seguinte (na decisão de segunda
  00h UTC, o de domingo).
- FRED (Federal Reserve de St. Louis), dias úteis dos EUA. NASDAQCOM (Nasdaq Composite) e VIXCLS (VIX, a
  volatilidade esperada do S&P 500): o fechamento de D sai às ~21h UTC de D e vale a partir de D+1 (na segunda, o
  da sexta). DTWEXBGS (dólar contra uma cesta ampla de moedas,
  relatório H.10 do Fed): sai uma vez por semana, na segunda à tarde, com a semana anterior inteira; na decisão de
  segunda 00h UTC, a última semana publicada é a retrasada (valor da sexta de 10 dias antes). Dia sem pregão vem
  vazio no CSV e fica de fora.
"""

import argparse
import csv
import io
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

log = logging.getLogger("wescalp.factors")

FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_SERIES = ("NASDAQCOM", "DTWEXBGS", "VIXCLS")
WEEKLY_RELEASE = ("DTWEXBGS",)
DATA_DIR = Path("data/factors")
EPOCH = date(1970, 1, 1)


def parse_fng(body: dict) -> list[tuple[date, float]]:
    """Resposta da API → [(dia UTC, valor)] em ordem de tempo (a API manda do mais novo para o mais velho)."""
    rows = [(datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).date(), float(x["value"])) for x in body["data"]]
    return sorted(rows)


def parse_fred(text: str) -> list[tuple[date, float]]:
    """CSV do FRED (observation_date,SÉRIE) → [(dia, valor)]; linhas sem valor (feriado) ficam de fora."""
    out = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 2 or not row[0][:1].isdigit() or row[1].strip() in ("", "."):
            continue
        out.append((date.fromisoformat(row[0]), float(row[1])))
    return out


def path(name: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"{name}.csv"


def save(rows: list[tuple[date, float]], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows((d.isoformat(), v) for d, v in rows)


def load(name: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """(dias desde 01/01/1970, valores) de um fator gravado, em ordem de tempo."""
    with open(path(name, data_dir), newline="") as f:
        rows = [(date.fromisoformat(d), float(v)) for d, v in csv.reader(f)]
    days = np.array([(d - EPOCH).days for d, _ in rows], dtype=np.int64)
    return days, np.array([v for _, v in rows])


def weekday(days: np.ndarray) -> np.ndarray:
    """0 = segunda ... 6 = domingo, para dias contados desde 01/01/1970 (uma quinta)."""
    return (days + 3) % 7


def available(name: str, days: np.ndarray) -> np.ndarray:
    """Primeiro dia em que o valor de cada data já era público às 00h UTC."""
    if name in WEEKLY_RELEASE:
        return days - weekday(days) + 8  # a semana inteira sai na segunda seguinte à tarde; vale da terça em diante
    return days + 1


def known(name: str, days: np.ndarray, values: np.ndarray, when: np.ndarray) -> np.ndarray:
    """Para cada dia de `when` (decisão às 00h UTC), o último valor já público; NaN se ainda não havia nenhum."""
    idx = np.searchsorted(available(name, days), when, side="right") - 1
    out = np.full(len(when), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def fetch(data_dir: Path = DATA_DIR) -> dict[str, int]:
    counts = {}
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        r = client.get(FNG_URL)
        r.raise_for_status()
        rows = parse_fng(r.json())
        save(rows, path("fng", data_dir))
        counts["fng"] = len(rows)
        for s in FRED_SERIES:
            r = client.get(FRED_URL.format(s))
            r.raise_for_status()
            rows = parse_fred(r.text)
            save(rows, path(s, data_dir))
            counts[s] = len(rows)
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_DIR)
    args = p.parse_args()
    for name, n in fetch(args.data).items():
        days, _ = load(name, args.data)
        first, last = (EPOCH + timedelta(days=int(x)) for x in (days[0], days[-1]))
        log.info("%s: %d dias, de %s a %s", name, n, first, last)


if __name__ == "__main__":
    main()
