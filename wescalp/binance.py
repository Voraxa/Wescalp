"""Histórico de candles de 5 min do BTCUSDT spot da Binance (data.binance.vision), de graça desde 17/08/2017.

    python -m wescalp.binance --until 2026-06

A KuCoin entrega só ~2 anos de candles de 5 min; a Binance publica um arquivo por mês desde 2017, com várias altas
e quedas. O preço do BTC nas duas difere em centésimos de por cento, então serve para pesquisar regras que vão
operar na KuCoin. Grava em data/candles/BINANCE-BTCUSDT-5m.csv.gz no formato de wescalp.brain.

Armadilhas dos arquivos: o tempo de abertura vem em milissegundos até dez/2024 e em microssegundos a partir de
jan/2025; alguns arquivos trazem linha de cabeçalho. Colunas: abertura (tempo), abertura, máxima, mínima,
fechamento, volume (BTC), fechamento (tempo), volume em USDT, negócios, compra agressora em BTC e em USDT, ignorar.
O padrão `--until 2026-06` para antes das 16 semanas intocadas (desde 04/07/2026).
"""

import argparse
import csv
import io
import logging
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from . import brain
from .kucoin import Candle

log = logging.getLogger("wescalp.binance")

SYMBOL = "BTCUSDT"
FIRST_MONTH = date(2017, 8, 1)
URL = "https://data.binance.vision/data/spot/monthly/klines/{sym}/5m/{sym}-5m-{y}-{m:02d}.zip"


def to_seconds(t: int) -> float:
    """Milissegundos (até 2024) ou microssegundos (2025 em diante) → segundos."""
    return t / 1e6 if t > 10**14 else t / 1e3


def parse_row(row: list[str], symbol: str = SYMBOL) -> Candle | None:
    """Uma linha do CSV → Candle; None para cabeçalho ou linha vazia."""
    if not row or not row[0].strip().isdigit():
        return None
    return Candle("spot", symbol, datetime.fromtimestamp(to_seconds(int(row[0])), timezone.utc),
                  float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[7]))


def parse_zip(data: bytes, symbol: str = SYMBOL) -> list[Candle]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        out = []
        for name in z.namelist():
            with z.open(name) as f:
                out += [c for c in map(lambda r: parse_row(r, symbol), csv.reader(io.TextIOWrapper(f))) if c]
        return out


def months(first: date, last: date) -> list[date]:
    out, d = [], first
    while d <= last:
        out.append(d)
        d = date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)
    return out


def path(data_dir: Path = brain.CANDLE_DIR, symbol: str = SYMBOL) -> Path:
    return data_dir / f"BINANCE-{symbol}-5m.csv.gz"


def fetch(until: date, data_dir: Path = brain.CANDLE_DIR, symbol: str = SYMBOL) -> list[Candle]:
    """Baixa os meses de FIRST_MONTH até `until` (inclusive) e grava tudo, ordenado e sem repetição."""
    have: dict[datetime, Candle] = {}
    with httpx.Client(timeout=60) as client:
        for m in months(FIRST_MONTH, until):
            r = client.get(URL.format(sym=symbol, y=m.year, m=m.month))
            if r.status_code == 404:
                log.warning("%s: sem arquivo", m)
                continue
            r.raise_for_status()
            for c in parse_zip(r.content, symbol):
                have[c.ts] = c
    candles = [have[k] for k in sorted(have)]
    brain.save_candles(path(data_dir, symbol), candles)
    return candles


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--until", default="2026-06", help="último mês (AAAA-MM), inclusive")
    p.add_argument("--data", type=Path, default=brain.CANDLE_DIR)
    args = p.parse_args()
    y, m = map(int, args.until.split("-"))
    cs = fetch(date(y, m, 1), args.data)
    gaps = sum(1 for a, b in zip(cs, cs[1:]) if (b.ts - a.ts).total_seconds() != brain.BAR)
    log.info("%d candles de %s a %s, %d buracos", len(cs), cs[0].ts, cs[-1].ts, gaps)


if __name__ == "__main__":
    main()
