# Fase 1 — contrato entre os módulos de pesquisa

Pacote `wescalp/research/`. Entrada única: candles de 5 min do perpétuo `XBTUSDTM` em
`data/candles_futures_5m.parquet` (colunas `ts` (UTC, tz-aware), `open`, `high`, `low`, `close`,
`volume` (contratos), `turnover` (USDT)); ~201 mil linhas de 2024-11-23 a 2026-10-22, com 6 buracos
curtos. Não há histórico do livro de ordens: nesta fase o "fluxo" é aproximado por volume.

Regra geral: **nada olha para o futuro.** Toda feature na linha `t` usa só dados até o fechamento de `t`.
Sinal na linha `t` → entrada na abertura de `t+1`.

## features.py

`add_features(df: pd.DataFrame) -> pd.DataFrame` — recebe o parquet (índice qualquer), devolve cópia
com índice `ts` ordenado e as colunas abaixo (NaN no aquecimento é aceitável):

| coluna | definição |
| --- | --- |
| `ema9`, `ema21` | EMA do close |
| `atr14` | ATR de Wilder, 14 barras |
| `atr_pct` | `atr14 / close` |
| `vwap` | VWAP do dia UTC (preço típico × volume), zera às 00:00 UTC |
| `vwap_sd` | desvio-padrão ponderado por volume do preço típico em torno do `vwap`, no dia |
| `vwap_z` | `(close - vwap) / vwap_sd` (NaN se `vwap_sd` == 0) |
| `rsi14` | RSI de Wilder |
| `vol_z` | z-score do volume contra as 288 barras anteriores (sem incluir a atual na média) |
| `hh20`, `ll20` | máxima/mínima das 20 barras ANTERIORES (shift 1) |
| `pdh`, `pdl` | máxima/mínima do dia UTC anterior |
| `hour` | hora UTC (int) |
| `regime` | `"alta"`, `"baixa"` ou `"lateral"`: inclinação da `ema21` em 12 barras normalizada pelo `atr14`; > +0,5 alta, < −0,5 baixa, senão lateral |

## labels.py

`CostModel` (dataclass): `maker=0.0002`, `taker=0.0006`, `slippage=0.0001` (por lado, só em ordem a
mercado). Entrada é ordem limitada (maker) no open de `t+1`; saída no alvo é maker; saída no stop ou
por tempo é taker + slippage.

`triple_barrier(df, signals: pd.Series, tp_atr: float, sl_atr: float, max_bars: int,
cost: CostModel = CostModel()) -> pd.DataFrame`

- `df` é a saída de `add_features`; `signals` tem o mesmo índice, valores em {-1, 0, +1}.
- Para cada sinal ≠ 0 com `atr14` válido: entrada no `open` da barra seguinte; alvo e stop a
  `tp_atr`/`sl_atr` × `atr14` (da barra do sinal) do preço de entrada.
- Percorre as barras a partir da de entrada (inclusive). Se alvo e stop forem tocados na mesma barra,
  conta **stop** (conservador). Sem toque em `max_bars` barras: sai no `close` da última, por tempo.
- **Uma posição por vez:** sinais que caem enquanto há trade aberto são ignorados.
- Retorno colunas: `entry_ts`, `exit_ts`, `side`, `entry_px`, `exit_px`, `outcome` (`"tp"|"sl"|"time"`),
  `bars`, `ret_gross` (fração, com sinal do lado), `cost` (fração), `ret_net`, `regime`, `hour`
  (estes dois da barra do sinal).

## setups.py

Cada setup: função `fn(df, **params) -> pd.Series` (índice de `df`, valores {-1,0,+1}) que usa só as
colunas de `add_features`. Registro:

```python
SETUPS: dict[str, Setup]  # Setup = dataclass(name, fn, grid: list[dict])
```

`grid` inclui os parâmetros do sinal **e** os de saída (`tp_atr`, `sl_atr`, `max_bars`), no máximo
~30 combinações por setup. Setups:

1. `rompimento` — close rompe `hh20` (compra) / `ll20` (venda) com `vol_z` acima de um limiar.
2. `vwap_reversao` — `vwap_z` além de ±k (k em 2–3) e candle de rejeição (fecha de volta para dentro
   em relação à barra anterior); opera a favor da volta à VWAP.
3. `pullback_ema` — `ema9` acima da `ema21` (alta), low toca a `ema9`/`ema21` e close fecha acima da
   `ema9`; espelhado para venda.

Opcional por setup: filtro de hora (lista de horas permitidas) como parâmetro.

## evaluate.py

- `walk_forward(df, setup, train_weeks=60, test_weeks=4, step_weeks=4, min_trades=30)` → para cada
  dobra escolhe no treino a combinação do `grid` com maior `ret_net` médio (com ≥ `min_trades`), aplica
  no teste, e devolve os trades fora da amostra concatenados + uma tabela por dobra (params escolhidos,
  métricas de treino e teste).
- `metrics(trades) -> dict`: `n`, `win_rate`, `avg_net_bps`, `median_net_bps`, `profit_factor`,
  `sharpe_trade` (média/desvio do `ret_net`), `max_dd` (sobre a soma acumulada de `ret_net`),
  `trades_por_dia`.
- `beta_posterior(wins, n, prior_wins=1, prior_n=2) -> (media, p05, p95)`.
- `deflated_sharpe(sr, n_obs, n_trials, skew, kurt, sr_var_trials) -> float` (Bailey & López de Prado
  2014), probabilidade de o Sharpe verdadeiro ser > 0 dado quantas combinações foram testadas.
- `breakdown(trades, by)` para `regime` e `hour`.

## run.py (integração)

`python -m wescalp.research.run` lê o parquet, roda os 3 setups em walk-forward, grava
`reports/fase1.md` e `reports/fase1.json`.

Testes em `tests/research/` com dados sintéticos pequenos; `.venv/bin/pytest -q` precisa passar.
