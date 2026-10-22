"""Mede os setups contra os rótulos de tripla barreira, descontadas as taxas.

    python -m wescalp.research --tp 1.5 --sl 1 --bars 12

As últimas `--holdout` semanas (padrão 16) ficam intocadas: não entram no relatório até a regra estar
congelada. `--holdout 0` só quando for a avaliação final de uma regra que não muda mais.

Um setup só abre nova entrada no mesmo lado depois que a anterior saiu. Cada linha mostra o retorno bruto e o
líquido médio por entrada, o intervalo de 90% dessa média e em quantas janelas de 4 semanas ela ficou positiva.
"""

import argparse
import asyncio
import math
from datetime import timedelta

from . import db, features
from .config import DATABASE_URL, FUTURES_SYMBOL
from .kucoin import Candle
from .labels import Barriers, Label, label_candles
from .setups import SETUPS

WINDOW = timedelta(weeks=4)
MIN_TRADES_PER_WINDOW = 5


def trades_for(candles: list[Candle], cols: dict, labels: dict, rule) -> list[Label]:
    """Entradas do setup, sem sobrepor duas no mesmo lado."""
    out: list[Label] = []
    busy_until = {"long": -1, "short": -1}
    for i in range(len(candles)):
        side = rule(candles, cols, i)
        if side is None or i < busy_until[side]:  # pode reentrar no candle em que a anterior saiu
            continue
        lb = labels.get((candles[i].ts, side))
        if lb is None:  # sem rótulo: sem ATR ou janela incompleta
            continue
        out.append(lb)
        busy_until[side] = i + lb.bars
    return out


def stats(trades: list[Label], start) -> dict:
    n = len(trades)
    rets = [t.ret_net for t in trades]
    mean = sum(rets) / n if n else 0.0
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (n - 1)) if n > 1 else 0.0
    half = 1.645 * sd / math.sqrt(n) if n else 0.0
    windows: dict[int, list[float]] = {}
    for t in trades:
        windows.setdefault((t.ts - start) // WINDOW, []).append(t.ret_net)
    full = [w for w in windows.values() if len(w) >= MIN_TRADES_PER_WINDOW]
    return {
        "n": n,
        "tp": sum(t.outcome == "tp" for t in trades) / n if n else 0.0,
        "win": sum(r > 0 for r in rets) / n if n else 0.0,
        "gross_bps": sum(t.ret_gross for t in trades) / n * 1e4 if n else 0.0,
        "net_bps": mean * 1e4,
        "lo_bps": (mean - half) * 1e4,
        "hi_bps": (mean + half) * 1e4,
        "windows_pos": sum(sum(w) > 0 for w in full),
        "windows": len(full),
    }


def report(candles: list[Candle], barriers: Barriers) -> list[tuple[str, str, dict]]:
    cols = features.compute(candles, atr_period=barriers.atr_period)
    all_labels = label_candles(candles, barriers)
    labels = {(lb.ts, lb.side): lb for lb in all_labels}
    start = candles[0].ts
    rows = [("todos os candles", side, stats([lb for lb in all_labels if lb.side == side], start))
            for side in ("long", "short")]
    for name, rule in SETUPS.items():
        trades = trades_for(candles, cols, labels, rule)
        for side in ("long", "short"):
            rows.append((name, side, stats([t for t in trades if t.side == side], start)))
    return rows


def format_report(rows) -> str:
    lines = [f"{'setup':<18}{'lado':<7}{'entradas':>9}{'alvo':>7}{'ganha':>7}"
             f"{'bruto(bps)':>12}{'líquido(bps)':>14}{'IC 90%':>18}{'janelas +':>11}"]
    for name, side, s in rows:
        ic = f"[{s['lo_bps']:.1f}, {s['hi_bps']:.1f}]"
        lines.append(f"{name:<18}{side:<7}{s['n']:>9}{s['tp']:>7.1%}{s['win']:>7.1%}"
                     f"{s['gross_bps']:>12.2f}{s['net_bps']:>14.2f}{ic:>18}{s['windows_pos']:>6}/{s['windows']}")
    return "\n".join(lines)


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tp", type=float, default=Barriers.tp_atr, help="alvo em ATRs")
    p.add_argument("--sl", type=float, default=Barriers.sl_atr, help="stop em ATRs")
    p.add_argument("--bars", type=int, default=Barriers.max_bars, help="candles até sair por tempo")
    p.add_argument("--atr", type=int, default=Barriers.atr_period, help="período do ATR")
    p.add_argument("--holdout", type=int, default=16, help="semanas finais fora do relatório")
    args = p.parse_args()

    conn = await db.connect(DATABASE_URL)
    try:
        candles = await db.load_candles(conn, "futures", FUTURES_SYMBOL)
    finally:
        await conn.close()
    if args.holdout and candles:
        cutoff = candles[-1].ts - timedelta(weeks=args.holdout)
        candles = [c for c in candles if c.ts < cutoff]
    if not candles:
        raise SystemExit("sem candles no período")
    barriers = Barriers(args.tp, args.sl, args.bars, args.atr)
    print(f"{FUTURES_SYMBOL} {candles[0].ts:%Y-%m-%d} a {candles[-1].ts:%Y-%m-%d}, {len(candles)} candles, {barriers.name}")
    print(format_report(report(candles, barriers)))


if __name__ == "__main__":
    asyncio.run(main())
