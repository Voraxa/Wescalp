"""Livro e negócios históricos do perpétuo gravados pela Tardis.dev, para testar o fluxo sem esperar o coletor.

    python -m wescalp.tardis fetch --jobs 4
    python -m wescalp.tardis flow --buckets 1 10 --horizons 10 30 60 300

A Tardis grava o WebSocket da KuCoin Futures desde 25/01/2024 e libera de graça o dia 1º de cada mês. Usamos
`book_snapshot_5` (os 5 melhores níveis de cada lado a cada mudança, montados pela Tardis) e `trades`.

`fetch` baixa cada dia, reduz o livro a uma foto por 100 ms (a última de cada janela, como o level2Depth50 do
coletor) e grava em `data/tardis/AAAA-MM-DD.{quotes,trades}.csv.gz` no formato de wescalp.flow.Quote e Tick.
O arquivo bruto (~55 MB por dia) é apagado depois. Dia já convertido não é baixado de novo.

`flow` roda os sinais de wescalp.flow em cada dia e junta os dias, sem retorno que atravesse de um dia para o
outro. As últimas `--holdout` datas (padrão 8) ficam de fora até a regra estar congelada.

Buraco: o book_snapshot_5 só tem linha quando os 5 níveis mudam, e em mercado calmo isso leva até ~7 s sem nenhum
negócio no meio (01/01/2025). Então aqui buraco é mais de 30 s sem foto, não os 2 s do coletor, que recebe o
livro a cada 100 ms mesmo parado. Entre duas fotos o livro não mudou, então o OFI do par continua certo.

Armadilhas do CSV da Tardis: tempos em microssegundos; níveis intercalados ask, bid (asks[0], bids[0], asks[1]...);
nível que não existe vem vazio; `amount` em contratos; `side` do negócio é o do agressor. O tempo usado é o
`local_timestamp` (quando a Tardis recebeu), o momento em que a informação estaria disponível para nós.
"""

import argparse
import csv
import gzip
import logging
import math
import statistics
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from . import flow
from .config import FUTURES_SYMBOL
from .flow import Quote, Tick

log = logging.getLogger("wescalp.tardis")

BASE_URL = "https://datasets.tardis.dev/v1/kucoin-futures"
FIRST_DAY = date(2024, 2, 1)  # primeiro dia 1º depois do início da gravação (25/01/2024)
DATA_DIR = Path("data/tardis")
STEP = 0.1  # segundos entre fotos, como o level2Depth50
HOLDOUT = 8
MAX_GAP = 30.0  # segundos sem foto que contam como buraco (ver acima)


def free_days(today: date) -> list[date]:
    """Dias 1º de cada mês de FIRST_DAY até antes de `today` (o arquivo de um dia sai depois que ele acaba)."""
    out, d = [], FIRST_DAY
    while d < today:
        out.append(d)
        d = date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)
    return out


def url(dataset: str, day: date, symbol: str = FUTURES_SYMBOL) -> str:
    return f"{BASE_URL}/{dataset}/{day:%Y/%m/%d}/{symbol}.csv.gz"


def _num(s: str) -> float:
    return float(s) if s else 0.0


def parse_snapshot5(row: dict) -> Quote | None:
    """Linha do book_snapshot_5 → Quote; None se faltar o melhor bid ou o melhor ask."""
    if not row["bids[0].price"] or not row["asks[0].price"]:
        return None
    return Quote(
        ts=int(row["local_timestamp"]) / 1e6,
        bid_px=float(row["bids[0].price"]),
        bid_sz=float(row["bids[0].amount"]),
        ask_px=float(row["asks[0].price"]),
        ask_sz=float(row["asks[0].amount"]),
        bid_sz5=sum(_num(row[f"bids[{i}].amount"]) for i in range(5)),
        ask_sz5=sum(_num(row[f"asks[{i}].amount"]) for i in range(5)),
    )


def parse_trade(row: dict) -> Tick:
    return Tick(ts=int(row["local_timestamp"]) / 1e6, side=row["side"], size=float(row["amount"]))


def thin(quotes, step: float = STEP) -> list[Quote]:
    """Última foto de cada janela [k·step, (k+1)·step). `quotes` em ordem de ts."""
    out: list[Quote] = []
    cell = None
    width = round(step * 1e6)
    for q in quotes:
        k = round(q.ts * 1e6) // width  # em µs inteiros: 10.1 / 0.1 em float dá 100,999…
        if k == cell:
            out[-1] = q
        else:
            out.append(q)
            cell = k
    return out


def _rows(path: Path):
    with gzip.open(path, "rt", newline="") as f:
        yield from csv.DictReader(f)


def _download(client: httpx.Client, src: str, dst: Path) -> None:
    with client.stream("GET", src) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)


def _write(path: Path, rows) -> None:
    tmp = path.with_suffix(".part")
    with gzip.open(tmp, "wt", newline="") as f:
        csv.writer(f).writerows(rows)
    tmp.rename(path)


def day_paths(day: date, data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    return data_dir / f"{day}.quotes.csv.gz", data_dir / f"{day}.trades.csv.gz"


def fetch_day(day: date, data_dir: Path = DATA_DIR) -> str:
    """Baixa e converte um dia; devolve uma linha de resumo."""
    qpath, tpath = day_paths(day, data_dir)
    if qpath.exists() and tpath.exists():
        return f"{day}: já convertido"
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, httpx.Client(timeout=120) as client:
        book, trades = Path(tmp) / "book.csv.gz", Path(tmp) / "trades.csv.gz"
        _download(client, url("book_snapshot_5", day), book)
        _download(client, url("trades", day), trades)
        raw = 0

        def quotes():
            nonlocal raw
            for row in _rows(book):
                raw += 1
                q = parse_snapshot5(row)
                if q is not None:
                    yield q

        qs = thin(quotes())
        ts = [parse_trade(r) for r in _rows(trades)]
    gaps = [b.ts - a.ts for a, b in zip(qs, qs[1:])]
    _write(tpath, ((t.ts, t.side, t.size) for t in ts))
    _write(qpath, ((q.ts, q.bid_px, q.bid_sz, q.ask_px, q.ask_sz, q.bid_sz5, q.ask_sz5) for q in qs))
    return (f"{day}: {raw} linhas do livro → {len(qs)} fotos, {len(ts)} negócios,"
            f" {sum(g > MAX_GAP for g in gaps)} buracos > {MAX_GAP:g} s, maior intervalo {max(gaps, default=0):.1f} s")


def load_day(day: date, data_dir: Path = DATA_DIR) -> tuple[list[Quote], list[Tick]]:
    qpath, tpath = day_paths(day, data_dir)
    with gzip.open(qpath, "rt", newline="") as f:
        quotes = [Quote(*map(float, r)) for r in csv.reader(f)]
    with gzip.open(tpath, "rt", newline="") as f:
        ticks = [Tick(float(ts), side, float(size)) for ts, side, size in csv.reader(f)]
    return quotes, ticks


def concat(segments: list[dict[str, list]], pad: int) -> dict[str, list]:
    """Junta as colunas de wescalp.flow.aggregate de vários dias com `pad` baldes vazios entre eles.

    Com pad maior que o horizonte (em baldes), nenhum retorno futuro sai de um dia e termina no outro.
    """
    out: dict[str, list] = {k: [] for k in segments[0]} if segments else {}
    for i, seg in enumerate(segments):
        if i:
            for k in out:
                out[k].extend([None] * pad)
        for k in out:
            out[k].extend(seg[k])
    return out


def day_corr_signs(segments: list[dict[str, list]], width: float, horizons: list[int]) -> dict[tuple[str, int], tuple[int, int]]:
    """Por sinal e horizonte: em quantos dias a correlação com o retorno futuro foi positiva, e de quantos dias."""
    out = {}
    for h in horizons:
        steps = round(h / width)
        if steps < 1 or abs(steps * width - h) > 1e-9:
            continue
        for name in flow.SIGNALS:
            pos = tot = 0
            for seg in segments:
                c = flow.correlation(seg[name], flow.forward_returns(seg["mid"], steps))
                if c is not None:
                    tot += 1
                    pos += c > 0
            out[name, h] = (pos, tot)
    return out


def available_days(data_dir: Path = DATA_DIR) -> list[date]:
    return sorted(date.fromisoformat(p.name[:10]) for p in data_dir.glob("*.quotes.csv.gz")
                  if day_paths(date.fromisoformat(p.name[:10]), data_dir)[1].exists())


def run_flow(args) -> None:
    days = available_days(args.data)
    if args.holdout:
        held, days = days[-args.holdout:], days[:-args.holdout]
        print(f"fora da amostra (intocados): {', '.join(map(str, held))}")
    if not days:
        raise SystemExit("nenhum dia convertido; rode antes: python -m wescalp.tardis fetch")
    print(f"{FUTURES_SYMBOL}, Tardis book_snapshot_5 + trades, {len(days)} dias: {days[0]} a {days[-1]}")
    print("custo da ida e volta (bps): " + ", ".join(f"{k} {v:.1f}" for k, v in flow.costs_bps().items()))
    print("D1..D10: retorno médio futuro do mid (bps) por decil do sinal, cortes na amostra toda; |ext| = maior"
          " |D1|, |D10|; passa = maior custo que |ext| supera; dias+ = dias com correlação positiva")

    segs: dict[float, list] = {w: [] for w in args.buckets}
    n_quotes = n_ticks = 0
    for day in days:
        quotes, ticks = load_day(day, args.data)
        n_quotes, n_ticks = n_quotes + len(quotes), n_ticks + len(ticks)
        for w in args.buckets:
            segs[w].append(flow.aggregate(quotes, ticks, w, args.max_gap))
    print(f"{n_quotes} fotos do topo, {n_ticks} negócios")

    for w in args.buckets:
        pad = math.ceil(max(args.horizons) / w) + 1
        cols = concat(segs[w], pad)
        ok = sum(m is not None for m in cols["mid"])
        spreads = [s for s in cols["spread_bps"] if s is not None]
        print(f"\nbaldes de {w:g} s: {ok} válidos, spread mediano {statistics.median(spreads):.3f} bps")
        for s in flow.horizon_summary(cols, w, args.horizons):
            print(f"  h={s['h']:>4}s: {s['n']} retornos, médio {s['mean_bps']:.2f} bps,"
                  f" |retorno| médio {s['abs_bps']:.2f} bps")
        signs = day_corr_signs(segs[w], w, args.horizons)
        rows = flow.report(cols, w, args.horizons, args.bins)
        lines = flow.format_report(rows, args.bins).split("\n")
        print(lines[0] + "   dias+")
        for line, r in zip(lines[1:], rows):
            pos, tot = signs.get((r["signal"], r["h"]), (0, 0))
            print(f"{line}   {pos}/{tot}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_DIR, help="pasta dos dias convertidos")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="baixa e converte os dias grátis que faltam")
    f.add_argument("--jobs", type=int, default=2, help="dias em paralelo")
    f.add_argument("--days", type=date.fromisoformat, nargs="+", help="só estes dias (AAAA-MM-DD)")
    r = sub.add_parser("flow", help="sinais de fluxo nos dias convertidos")
    r.add_argument("--buckets", type=float, nargs="+", default=[1, 10], help="largura dos baldes em segundos")
    r.add_argument("--horizons", type=int, nargs="+", default=[10, 30, 60, 300], help="horizontes em segundos")
    r.add_argument("--bins", type=int, default=10, help="faixas de quantil do sinal (10 = decis)")
    r.add_argument("--max-gap", type=float, default=MAX_GAP, help="segundos sem foto que contam como buraco")
    r.add_argument("--holdout", type=int, default=HOLDOUT, help="últimas datas deixadas de fora")
    args = p.parse_args()

    if args.cmd == "fetch":
        days = args.days or free_days(datetime.now(timezone.utc).date())
        with ProcessPoolExecutor(args.jobs) as pool:
            for line in pool.map(fetch_day, days, [args.data] * len(days)):
                log.info(line)
    else:
        run_flow(args)


if __name__ == "__main__":
    main()
