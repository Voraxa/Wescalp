# Wescalp

Bot de scalping de BTC na KuCoin, publicado em `wescalp.avadata.com.br`. Pesquisa e arquitetura:
[Bot de scalping: pesquisa e arquitetura](https://claude.ai/artifact/MDFU7zTSg4VqsySbQJeZD1).

**Estado: Fase 0 (dados).** Nada aqui envia ordem. O código só lê a API pública da KuCoin e grava no banco.

## O que já existe

| Peça | Comando | O que faz |
| --- | --- | --- |
| Histórico | `python -m wescalp.backfill --weeks 100` | Candles de 5 min do spot `BTC-USDT` e do perpétuo `XBTUSDTM`, 100 semanas de cada (~201 mil candles por fonte, ~10 min). Retoma de onde parou. |
| Coletor ao vivo | `python -m wescalp.collector` | Livro de ordens (50 níveis por lado, ~10 atualizações/s) e cada negócio do perpétuo. Grava o topo do livro a cada atualização e a foto completa a cada 1 s. |

A KuCoin não entrega histórico do livro de ordens: ele só existe a partir do dia em que o coletor liga.
Por isso o coletor tem `restart: unless-stopped` e healthcheck.

## Tabelas

- `candles` — 5 min, `source` = `spot` ou `futures`.
- `trades` — negócios do perpétuo; `side` é o lado do agressor; `size` em contratos (1 contrato = 0,001 BTC).
- `book_l1` — melhor compra/venda e soma dos 5 primeiros níveis, a cada atualização (base do OFI).
- `book_snapshots` — 50 níveis de cada lado, a cada `WESCALP_BOOK_SNAPSHOT_MS` (padrão 1000).

No TimescaleDB viram hypertables com compressão depois de 1 dia (`db/timescale.sql`). Em Postgres puro
(desenvolvimento) só `db/schema.sql` é aplicado.

## Rodar

Local, com um Postgres qualquer:

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
export DATABASE_URL=postgresql://usuario@localhost:5432/wescalp
.venv/bin/pytest -q
.venv/bin/python -m wescalp.backfill --weeks 1
.venv/bin/python -m wescalp.collector
```

Na VPS (Docker):

```bash
cp .env.example .env   # e troque a senha
docker compose up -d db collector
docker compose run --rm backfill --weeks 100
```

## Pesquisa (Fase 1)

```bash
.venv/bin/pip install -e '.[dev,research]'
# exporta os candles do banco para data/candles_futures_5m.parquet, depois:
.venv/bin/python -m wescalp.research.run
```

Contrato entre os módulos em `docs/fase1-contrato.md`; resultado em `reports/fase1.md`.

## Próximas fases

1. **Pesquisa:** feita para setups de candle (sem vantagem); próxima rodada com o livro de ordens.
2. **Observação:** motor marca entrada e saída no gráfico a cada candle de 5 min, sem dinheiro; Gemini como segunda opinião.
3. **Livro de lições e rodadas de agentes:** revisão diária e semanal dos trades marcados.
4. **Painel** em `wescalp.avadata.com.br`, atrás do Caddy da VPS.
5. **Dinheiro real pequeno**, só depois de 2 a 3 meses de observação dentro do previsto.
