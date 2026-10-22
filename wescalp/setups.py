"""Os três setups do documento de pesquisa, na versão que dá para medir só com candles.

Cada regra olha o candle i já fechado (e os sinais de wescalp.features) e devolve "long", "short" ou None.
A entrada é no fechamento do candle i, como nos rótulos. Os parâmetros foram fixados antes de olhar o
resultado; mexer neles depois de ver o resultado conta como mais uma variação testada (Deflated Sharpe).
"""

from collections.abc import Callable

from .kucoin import Candle

BREAKOUT_VOL = 1.5  # volume do candle de rompimento sobre a média
VWAP_BAND = 2.0  # banda da VWAP em desvios-padrão


def breakout(cs: list[Candle], f: dict, i: int) -> str | None:
    """Rompimento da máxima (mínima) do dia anterior, fechando além dela, com volume forte."""
    hi, lo, vr = f["prev_high"][i], f["prev_low"][i], f["vol_ratio"][i]
    if hi is None or vr is None or vr < BREAKOUT_VOL or i == 0:
        return None
    prev, c = cs[i - 1].close, cs[i].close
    if prev <= hi < c:
        return "long"
    if prev >= lo > c:
        return "short"
    return None


def vwap_reversion(cs: list[Candle], f: dict, i: int) -> str | None:
    """Pavio passa da 2ª banda da VWAP e o candle fecha de volta para dentro, na direção da VWAP."""
    vwap, sd = f["vwap"][i], f["vwap_sd"][i]
    if vwap is None or not sd:
        return None
    c = cs[i]
    lower, upper = vwap - VWAP_BAND * sd, vwap + VWAP_BAND * sd
    if c.low <= lower < c.close and c.close > c.open:
        return "long"
    if c.high >= upper > c.close and c.close < c.open:
        return "short"
    return None


def pullback(cs: list[Candle], f: dict, i: int) -> str | None:
    """Tendência curta (EMA 9 x 21), recuo até a EMA 9 e retomada com volume maior que o do candle anterior."""
    if i == 0 or f["atr"][i] is None:
        return None
    c, e9, e21 = cs[i], f["ema9"][i], f["ema21"][i]
    more_volume = c.volume > cs[i - 1].volume
    if e9 > e21 and c.low <= e9 < c.close and c.close > c.open and more_volume:
        return "long"
    if e9 < e21 and c.high >= e9 > c.close and c.close < c.open and more_volume:
        return "short"
    return None


SETUPS: dict[str, Callable[[list[Candle], dict, int], str | None]] = {
    "breakout": breakout,
    "vwap_reversion": vwap_reversion,
    "pullback": pullback,
}
