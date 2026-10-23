"""BTC/USD da Bitstamp antes da Binance, em candles de 1 h: o trecho trancado para a prova única.

    python -m wescalp.bitstamp        # baixa e grava data/locked/BITSTAMP-BTCUSD-1h.csv.gz

Toda a pesquisa até aqui usou a Binance, que começa em 17/08/2017. Este arquivo cobre os seis anos anteriores
(18/08/2011 a 16/08/2017), que nenhuma regra viu: a bolha de 2013, a queda de 2014-2015 e a alta até meados de
2017. Ele fica trancado até a prova única: wescalp.brain.load_candles recusa arquivo dentro de data/locked/ sem
WESCALP_UNLOCK=prova-unica. Aqui só sai a conferência dos dados (cobertura, horas sem negócio, maiores saltos e o
SHA-256 do conteúdo), nunca resultado de estratégia. Na prova única, a avaliação vai de 01/01/2013 a 16/08/2017;
o que vem antes só aquece as médias.

A API devolve até 1.000 candles por chamada, a partir de `start`, com os campos pelo nome (sem a armadilha de ordem
da KuCoin), tudo em texto, e preenche hora sem negócio com o fechamento anterior e volume zero. Ela não dá o volume
em USD: o giro gravado é volume × fechamento, aproximado.
"""

import argparse
import gzip
import hashlib
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from . import brain
from .kucoin import Candle

log = logging.getLogger("wescalp.bitstamp")

URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
SYMBOL = "BTCUSD"
STEP = 3600
LIMIT = 1000
FIRST = datetime(2011, 8, 18, tzinfo=timezone.utc)  # o primeiro candle da Bitstamp
LAST = datetime(2017, 8, 16, 23, tzinfo=timezone.utc)  # a última hora antes do primeiro candle da Binance
EVAL_START = date(2013, 1, 1)
LOCKED_DIR = Path("data/locked")


def parse(body: dict) -> list[Candle]:
    out = []
    for x in body["data"]["ohlc"]:
        close, volume = float(x["close"]), float(x["volume"])
        out.append(Candle("spot", SYMBOL, datetime.fromtimestamp(int(x["timestamp"]), timezone.utc),
                          float(x["open"]), float(x["high"]), float(x["low"]), close, volume, volume * close))
    return out


def fetch(first: datetime = FIRST, last: datetime = LAST, client: httpx.Client | None = None) -> list[Candle]:
    """Candles de 1 h de `first` a `last` (inclusive), em ordem e sem repetição."""
    own = client is None
    client = client or httpx.Client(timeout=60)
    have: dict[datetime, Candle] = {}
    start = first
    try:
        while start <= last:
            r = client.get(URL, params={"step": STEP, "limit": LIMIT, "start": int(start.timestamp())})
            r.raise_for_status()
            got = [c for c in parse(r.json()) if first <= c.ts <= last]
            for c in got:
                have[c.ts] = c
            # janela vazia: pula para a próxima em vez de parar (só deve acontecer antes do primeiro negócio)
            start = got[-1].ts + timedelta(seconds=STEP) if got else start + timedelta(seconds=STEP * LIMIT)
    finally:
        if own:
            client.close()
    return [have[k] for k in sorted(have)]


def path(data_dir: Path = LOCKED_DIR) -> Path:
    return data_dir / f"BITSTAMP-{SYMBOL}-1h.csv.gz"


def digest(p: Path) -> str:
    """SHA-256 do conteúdo descompactado (o gzip grava a hora em que foi feito; o conteúdo, não)."""
    with gzip.open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check(candles: list[Candle]) -> dict:
    """Conferência dos dados, sem estratégia: buracos, horas sem negócio e os maiores saltos de uma hora."""
    gaps = sum(1 for a, b in zip(candles, candles[1:]) if (b.ts - a.ts).total_seconds() != STEP)
    longest, run, run_start, where = 0, 0, None, None
    for c in candles:
        if c.volume == 0:
            run, run_start = run + 1, run_start or c.ts
            if run > longest:
                longest, where = run, run_start
        else:
            run, run_start = 0, None
    jumps = sorted(((abs(math.log(b.close / a.close)), b.ts, b.close / a.close - 1) for a, b in zip(candles, candles[1:])),
                   reverse=True)[:5]
    return {"n": len(candles), "first": candles[0].ts, "last": candles[-1].ts, "gaps": gaps,
            "empty": sum(1 for c in candles if c.volume == 0), "longest_empty": longest, "longest_empty_at": where,
            "jumps": [(ts, r) for _, ts, r in jumps]}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=LOCKED_DIR)
    args = p.parse_args()
    cs = fetch()
    brain.save_candles(path(args.data), cs)
    s = check(cs)
    log.info("%d candles de %s a %s, %d buracos", s["n"], s["first"], s["last"], s["gaps"])
    log.info("horas sem negócio: %d (%.1f%%); a maior sequência: %d h, desde %s", s["empty"], 100 * s["empty"] / s["n"],
             s["longest_empty"], s["longest_empty_at"])
    log.info("maiores saltos de uma hora: %s", ", ".join(f"{ts:%Y-%m-%d %H}h {r:+.0%}" for ts, r in s["jumps"]))
    log.info("SHA-256 do conteúdo: %s", digest(path(args.data)))


if __name__ == "__main__":
    main()
