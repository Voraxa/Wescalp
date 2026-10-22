"""Rótulos de tripla barreira sobre os candles de 5 min do perpétuo.

    python -m wescalp.labels --tp 1.5 --sl 1 --bars 12

Para cada candle fechado, simula uma compra e uma venda no fechamento dele e marca o que veio primeiro nos
candles seguintes: alvo (tp), stop (sl) ou tempo (time). Alvo e stop são múltiplos do ATR no momento da
entrada, que só usa candles já fechados. O retorno líquido desconta as taxas da KuCoin: entrada e alvo com
ordem limitada (maker), stop e saída por tempo a mercado (taker).

Quando o mesmo candle toca alvo e stop, não dá para saber a ordem dentro dele: conta como stop. Entradas
cuja janela cruza um buraco nos candles, ou passa do último candle, ficam sem rótulo.
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from . import db
from .config import CANDLE_SECONDS, DATABASE_URL, FUTURES_SYMBOL, MAKER_FEE, TAKER_FEE
from .kucoin import Candle

log = logging.getLogger("wescalp.labels")


@dataclass(frozen=True)
class Barriers:
    tp_atr: float = 1.5
    sl_atr: float = 1.0
    max_bars: int = 12
    atr_period: int = 14

    @property
    def name(self) -> str:
        return f"tp{self.tp_atr:g}-sl{self.sl_atr:g}-t{self.max_bars}-atr{self.atr_period}"


@dataclass(frozen=True, slots=True)
class Label:
    symbol: str
    ts: datetime  # abertura do candle em cujo fechamento a entrada acontece
    side: str  # long | short
    outcome: str  # tp | sl | time
    bars: int  # candles até a saída
    entry_px: float
    exit_px: float
    atr: float
    ret_gross: float
    ret_net: float


def contiguous(a: Candle, b: Candle) -> bool:
    return (b.ts - a.ts).total_seconds() == CANDLE_SECONDS


def atr_series(candles: list[Candle], period: int) -> list[float | None]:
    """ATR de Wilder no fechamento de cada candle; recomeça do zero depois de um buraco."""
    out: list[float | None] = []
    trs: list[float] = []
    atr = None
    for i, c in enumerate(candles):
        if i == 0 or not contiguous(candles[i - 1], c):
            trs, atr = [], None
            tr = c.high - c.low
        else:
            prev = candles[i - 1].close
            tr = max(c.high - c.low, abs(c.high - prev), abs(c.low - prev))
        if atr is None:
            trs.append(tr)
            if len(trs) == period:
                atr = sum(trs) / period
        else:
            atr = (atr * (period - 1) + tr) / period
        out.append(atr)
    return out


def run_ends(candles: list[Candle]) -> list[int]:
    """Para cada candle, o índice do último candle da sequência sem buraco em que ele está."""
    ends = [0] * len(candles)
    end = len(candles) - 1
    for i in range(len(candles) - 1, -1, -1):
        if i < len(candles) - 1 and not contiguous(candles[i], candles[i + 1]):
            end = i
        ends[i] = end
    return ends


def first_touch(candles: list[Candle], i: int, side: str, tp: float, sl: float, max_bars: int) -> tuple[str, int, float]:
    for k in range(1, max_bars + 1):
        c = candles[i + k]
        if side == "long":
            if c.low <= sl:  # stop primeiro, mesmo se o alvo foi tocado no mesmo candle
                return "sl", k, min(sl, c.open)  # abriu além do stop: sai na abertura
            if c.high >= tp:
                return "tp", k, tp
        else:
            if c.high >= sl:
                return "sl", k, max(sl, c.open)
            if c.low <= tp:
                return "tp", k, tp
    return "time", max_bars, candles[i + max_bars].close


def label_candles(candles: list[Candle], barriers: Barriers) -> list[Label]:
    """Rótulos de compra e venda para cada candle com ATR pronto e janela completa à frente."""
    atrs = atr_series(candles, barriers.atr_period)
    ends = run_ends(candles)
    out = []
    for i, c in enumerate(candles):
        atr = atrs[i]
        if atr is None or atr <= 0 or i + barriers.max_bars > ends[i]:
            continue
        entry = c.close
        for side, sign in (("long", 1), ("short", -1)):
            tp = entry + sign * barriers.tp_atr * atr
            sl = entry - sign * barriers.sl_atr * atr
            outcome, bars, exit_px = first_touch(candles, i, side, tp, sl, barriers.max_bars)
            gross = sign * (exit_px - entry) / entry
            fees = MAKER_FEE + (MAKER_FEE if outcome == "tp" else TAKER_FEE)
            out.append(Label(c.symbol, c.ts, side, outcome, bars, entry, exit_px, atr, gross, gross - fees))
    return out


def summarize(labels: list[Label]) -> list[dict]:
    rows = []
    for side in ("long", "short"):
        ls = [lb for lb in labels if lb.side == side]
        if not ls:
            continue
        n = len(ls)
        rows.append({
            "side": side,
            "n": n,
            "tp": sum(lb.outcome == "tp" for lb in ls) / n,
            "sl": sum(lb.outcome == "sl" for lb in ls) / n,
            "time": sum(lb.outcome == "time" for lb in ls) / n,
            "win": sum(lb.ret_net > 0 for lb in ls) / n,
            "gross_bps": sum(lb.ret_gross for lb in ls) / n * 1e4,
            "net_bps": sum(lb.ret_net for lb in ls) / n * 1e4,
        })
    return rows


def format_summary(rows: list[dict]) -> str:
    lines = ["lado   entradas    alvo   stop  tempo  ganha  bruto(bps)  líquido(bps)"]
    for r in rows:
        lines.append(
            f"{r['side']:<6} {r['n']:>8}  {r['tp']:>5.1%}  {r['sl']:>5.1%}  {r['time']:>5.1%}  {r['win']:>5.1%}"
            f"  {r['gross_bps']:>10.2f}  {r['net_bps']:>12.2f}"
        )
    return "\n".join(lines)


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tp", type=float, default=Barriers.tp_atr, help="alvo em ATRs")
    p.add_argument("--sl", type=float, default=Barriers.sl_atr, help="stop em ATRs")
    p.add_argument("--bars", type=int, default=Barriers.max_bars, help="candles até sair por tempo")
    p.add_argument("--atr", type=int, default=Barriers.atr_period, help="período do ATR")
    p.add_argument("--dry-run", action="store_true", help="só mostra o resumo, não grava")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    barriers = Barriers(args.tp, args.sl, args.bars, args.atr)
    conn = await db.connect(DATABASE_URL)
    try:
        await db.apply_schema(conn)
        candles = await db.load_candles(conn, "futures", FUTURES_SYMBOL)
        labels = label_candles(candles, barriers)
        log.info("%s: %d candles, %d rótulos", barriers.name, len(candles), len(labels))
        print(format_summary(summarize(labels)))
        if not args.dry_run:
            await db.replace_labels(conn, barriers.name, FUTURES_SYMBOL, labels)
            log.info("gravados em labels (config = %s)", barriers.name)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
