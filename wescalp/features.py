"""Sinais calculados só dos candles de 5 min, no fechamento de cada candle e sem olhar o futuro.

Fluxo de ordens (OFI, delta) fica de fora: o livro e os negócios só existem a partir do dia em que o coletor
ligou, e os candles de futuros não separam volume comprador de vendedor.
"""

import math
from datetime import timedelta

from .kucoin import Candle
from .labels import atr_series


def ema(values: list[float], period: int) -> list[float]:
    k = 2 / (period + 1)
    out, e = [], values[0] if values else 0.0
    for v in values:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def compute(candles: list[Candle], atr_period: int = 14, vol_period: int = 20) -> dict[str, list]:
    """Colunas alinhadas com `candles`. None onde o sinal ainda não tem histórico suficiente.

    - vwap, vwap_sd: VWAP do dia (UTC) com preço típico e o desvio-padrão ponderado por volume em volta dela
    - prev_high, prev_low: máxima e mínima do dia UTC anterior (None se o dia anterior não está nos dados)
    - vol_ratio: volume sobre a média dos `vol_period` candles anteriores
    """
    closes = [c.close for c in candles]
    cols: dict[str, list] = {
        "atr": atr_series(candles, atr_period),
        "ema9": ema(closes, 9),
        "ema21": ema(closes, 21),
        "vwap": [], "vwap_sd": [], "prev_high": [], "prev_low": [], "vol_ratio": [],
    }

    day = None
    pv = pv2 = vol = 0.0
    hi = lo = prev_hi = prev_lo = None
    vols: list[float] = []
    for c in candles:
        d = c.ts.date()
        if d != day:
            # Só vale como "dia anterior" se for mesmo o dia de ontem.
            if day is not None and d - day == timedelta(days=1):
                prev_hi, prev_lo = hi, lo
            else:
                prev_hi = prev_lo = None
            day, pv, pv2, vol, hi, lo = d, 0.0, 0.0, 0.0, c.high, c.low
        hi, lo = max(hi, c.high), min(lo, c.low)

        tp = (c.high + c.low + c.close) / 3
        pv, pv2, vol = pv + tp * c.volume, pv2 + tp * tp * c.volume, vol + c.volume
        if vol > 0:
            vwap = pv / vol
            cols["vwap"].append(vwap)
            cols["vwap_sd"].append(math.sqrt(max(pv2 / vol - vwap * vwap, 0.0)))
        else:
            cols["vwap"].append(None)
            cols["vwap_sd"].append(None)
        cols["prev_high"].append(prev_hi)
        cols["prev_low"].append(prev_lo)

        avg = sum(vols) / len(vols) if len(vols) == vol_period else None
        cols["vol_ratio"].append(c.volume / avg if avg else None)
        vols.append(c.volume)
        if len(vols) > vol_period:
            vols.pop(0)
    return cols
