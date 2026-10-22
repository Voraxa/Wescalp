# Fase 1 — setups em walk-forward

Perpétuo XBTUSDTM, candles de 5 min, 2024-11-23 a 2026-10-22. Treino de 60 semanas, teste nas 4 seguintes, andando 4 semanas. Só trades fora da amostra entram nas métricas. Custos: entrada maker 0,02%; saída no alvo maker 0,02%; stop ou tempo taker 0,06% + 0,01% de derrapagem.

| Setup | Trades | Por dia | Acerto (intervalo 90%) | Bruto (bps) | Custo (bps) | Líquido (bps) | DSR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rompimento | 1492 | 5.3 | 33.3% (31.3%–35.4%) | +0.1 | 7.4 | -7.2 | 0.00 |
| vwap_reversao | 285 | 1.0 | 38.3% (33.7%–43.1%) | -2.0 | 7.1 | -9.0 | 0.00 |
| pullback_ema | 4245 | 15.0 | 52.4% (51.1%–53.6%) | -0.3 | 6.3 | -6.6 | 0.00 |

DSR = probabilidade de o Sharpe verdadeiro ser positivo, descontando as combinações testadas (Bailey & López de Prado, 2014). Abaixo de 0,95 não há evidência de vantagem.

## rompimento

Por regime (líquido em bps):

| Regime | Trades | Acerto | Líquido |
| --- | --- | --- | --- |
| alta | 611 | 33.2% | -7.4 |
| baixa | 604 | 33.4% | -7.4 |
| lateral | 277 | 33.2% | -6.7 |

Melhores horas UTC: 4h (+0.2, n=25), 20h (+0.1, n=48), 0h (-0.4, n=46). Piores: 2h (-15.8, n=36), 9h (-16.4, n=32), 17h (-17.4, n=71).

Parâmetros escolhidos por dobra: `{"max_bars": 18, "sl_atr": 1.0, "tp_atr": 2.0, "vol_z_min": 2.0}` ×11

## vwap_reversao

Por regime (líquido em bps):

| Regime | Trades | Acerto | Líquido |
| --- | --- | --- | --- |
| alta | 136 | 36.8% | -8.4 |
| baixa | 121 | 43.8% | -8.5 |
| lateral | 28 | 21.4% | -14.6 |

Melhores horas UTC: 21h (+13.9, n=3), 15h (+4.6, n=19), 19h (+2.7, n=3). Piores: 12h (-15.4, n=15), 22h (-21.9, n=5), 17h (-23.2, n=12).

Parâmetros escolhidos por dobra: `{"k": 2.5, "max_bars": 6, "sl_atr": 1.5, "tp_atr": 1.5}` ×3; `{"k": 3.0, "max_bars": 6, "sl_atr": 1.0, "tp_atr": 1.0}` ×2; `{"k": 3.0, "max_bars": 12, "sl_atr": 1.5, "tp_atr": 1.0}` ×2; `{"k": 2.5, "max_bars": 6, "sl_atr": 1.5, "tp_atr": 1.0}` ×2; `{"k": 3.0, "max_bars": 6, "sl_atr": 1.5, "tp_atr": 1.5}` ×1; `{"k": 2.5, "max_bars": 12, "sl_atr": 1.0, "tp_atr": 1.0}` ×1

## pullback_ema

Por regime (líquido em bps):

| Regime | Trades | Acerto | Líquido |
| --- | --- | --- | --- |
| alta | 1132 | 51.4% | -7.1 |
| baixa | 1103 | 53.4% | -6.3 |
| lateral | 2010 | 52.3% | -6.6 |

Melhores horas UTC: 4h (-2.0, n=157), 9h (-2.5, n=146), 17h (-2.7, n=156). Piores: 1h (-9.9, n=183), 20h (-10.0, n=167), 22h (-10.3, n=202).

Parâmetros escolhidos por dobra: `{"max_bars": 24, "sl_atr": 1.5, "toque": "ema21", "tp_atr": 1.0}` ×8; `{"max_bars": 24, "sl_atr": 1.5, "toque": "ema21", "tp_atr": 2.0}` ×3

_Gerado em 2026-10-22 19:34 UTC, 11 s._
