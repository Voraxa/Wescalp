"""Fluxo e derivativos do BTC na Binance (data.binance.vision), de graça e com anos de história, para o day trade.

    python -m wescalp.flowdata --until 2026-06

Grava em data/flow/, um arquivo por série, em barras de 5 min (tempo de abertura em segundos UTC):
- spot-flow: volume, compra agressora (taker buy) e número de negócios do BTCUSDT spot, desde 08/2017;
- perp: candles do perpétuo BTCUSDT (USDⓈ-M) com volume, compra agressora e negócios, desde 01/2020;
- premium: o prêmio do perpétuo sobre o índice spot (abertura, máxima, mínima e fechamento), desde 01/2020;
- funding: a taxa de funding de cada liquidação (a cada 8 h), desde 01/2020;
- metrics: posições abertas (BTC e USDT), relação compra/venda dos grandes traders (contas e posições), das contas
  em geral e do volume agressor, a cada 5 min, desde 01/12/2021 (arquivos diários; campo vazio vira NaN).

Armadilhas: os candles do spot vêm em milissegundos até 12/2024 e em microssegundos depois (wescalp.binance); os do
perpétuo seguem em milissegundos; vários arquivos têm linha de cabeçalho e outros não; o funding vem com 1 a 2 ms
a mais na hora; as métricas trazem a hora em texto ("2021-12-01 00:05:00", UTC). O padrão `--until 2026-06` deixa
de fora as 16 semanas intocadas, como em wescalp.binance.
"""

import argparse
import csv
import gzip
import io
import logging
import math
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from .binance import months, to_seconds

log = logging.getLogger("wescalp.flowdata")

BASE = "https://data.binance.vision/data"
SYMBOL = "BTCUSDT"
DATA_DIR = Path("data/flow")
FIRST = {"spot-flow": date(2017, 8, 1), "perp": date(2020, 1, 1), "premium": date(2020, 1, 1),
         "funding": date(2020, 1, 1), "metrics": date(2021, 12, 1)}
URLS = {
    "spot-flow": BASE + "/spot/monthly/klines/{s}/5m/{s}-5m-{y}-{m:02d}.zip",
    "perp": BASE + "/futures/um/monthly/klines/{s}/5m/{s}-5m-{y}-{m:02d}.zip",
    "premium": BASE + "/futures/um/monthly/premiumIndexKlines/{s}/5m/{s}-5m-{y}-{m:02d}.zip",
    "funding": BASE + "/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{y}-{m:02d}.zip",
    "metrics": BASE + "/futures/um/daily/metrics/{s}/{s}-metrics-{y}-{m:02d}-{d:02d}.zip",
}
COLUMNS = {
    "spot-flow": ("ts", "volume", "taker_buy", "trades"),
    "perp": ("ts", "open", "high", "low", "close", "volume", "taker_buy", "trades"),
    "premium": ("ts", "open", "high", "low", "close"),
    "funding": ("ts", "rate"),
    "metrics": ("ts", "oi", "oi_usdt", "top_accounts", "top_positions", "accounts", "taker_ratio"),
}


def csv_rows(data: bytes) -> list[list[str]]:
    """Linhas de todos os CSV de um zip, sem cabeçalho nem linha vazia."""
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            for row in csv.reader(io.TextIOWrapper(z.open(name), encoding="utf-8")):
                if row and row[0][:1].isdigit():
                    out.append(row)
    return out


def parse(kind: str, row: list[str]) -> tuple:
    """Uma linha do arquivo da Binance → a linha gravada (COLUMNS[kind]), com o tempo em segundos UTC."""
    if kind == "metrics":
        ts = int(datetime.fromisoformat(row[0]).replace(tzinfo=timezone.utc).timestamp())
        return (ts, *(float(x) if x.strip() else math.nan for x in row[2:8]))  # alguns dias têm campo vazio
    t = int(round(to_seconds(int(row[0]))))  # o funding vem com 1 a 2 ms de sobra
    if kind == "funding":
        return t, float(row[2])
    if kind == "spot-flow":
        return t, float(row[5]), float(row[9]), int(row[8])
    if kind == "perp":
        return (t, *(float(x) for x in row[1:6]), float(row[9]), int(row[8]))
    if kind == "premium":
        return (t, *(float(x) for x in row[1:5]))
    raise ValueError(kind)


def path(kind: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"BINANCE-{SYMBOL}-{kind}.csv.gz"


def files(kind: str, until: date) -> list[str]:
    """URLs dos arquivos da série, do primeiro mês até o fim do mês de `until`."""
    if kind != "metrics":
        return [URLS[kind].format(s=SYMBOL, y=m.year, m=m.month) for m in months(FIRST[kind], until)]
    last = (date(until.year + until.month // 12, until.month % 12 + 1, 1) - timedelta(days=1))
    days = [FIRST[kind] + timedelta(days=i) for i in range((last - FIRST[kind]).days + 1)]
    return [URLS[kind].format(s=SYMBOL, y=d.year, m=d.month, d=d.day) for d in days]


def fetch(kind: str, until: date, data_dir: Path = DATA_DIR, jobs: int = 8) -> int:
    """Baixa a série inteira e grava ordenada e sem repetição. Devolve o número de linhas."""
    def get(url: str) -> list[tuple]:
        r = client.get(url)
        if r.status_code == 404:
            log.warning("%s: sem arquivo", url.rsplit("/", 1)[-1])
            return []
        r.raise_for_status()
        return [parse(kind, row) for row in csv_rows(r.content)]

    with httpx.Client(timeout=120, follow_redirects=True) as client, ThreadPoolExecutor(jobs) as pool:
        have = {row[0]: row for rows in pool.map(get, files(kind, until)) for row in rows}
    p = path(kind, data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS[kind])
        w.writerows(have[k] for k in sorted(have))
    tmp.rename(p)
    return len(have)


def load(kind: str, data_dir: Path = DATA_DIR):
    """{coluna: array} da série gravada, em ordem de tempo."""
    import numpy as np

    with gzip.open(path(kind, data_dir), "rt", newline="") as f:
        r = csv.reader(f)
        head = next(r)
        rows = list(r)
    cols = list(zip(*rows)) if rows else [()] * len(head)
    return {k: np.array(v, dtype=np.int64 if k in ("ts", "trades") else float) for k, v in zip(head, cols)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--until", default="2026-06", help="último mês (AAAA-MM), inclusive")
    p.add_argument("--kinds", nargs="*", default=list(URLS))
    p.add_argument("--data", type=Path, default=DATA_DIR)
    args = p.parse_args()
    y, m = map(int, args.until.split("-"))
    for kind in args.kinds:
        n = fetch(kind, date(y, m, 1), args.data)
        s = load(kind, args.data)["ts"]
        first, last = (datetime.fromtimestamp(int(x), timezone.utc) for x in (s[0], s[-1]))
        log.info("%s: %d linhas, de %s a %s", kind, n, first, last)


if __name__ == "__main__":
    main()
