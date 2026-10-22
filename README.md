# Wescalp

Bot de scalping de BTC na KuCoin, publicado em `wescalp.avadata.com.br`. Pesquisa e arquitetura:
[Bot de scalping: pesquisa e arquitetura](https://claude.ai/artifact/MDFU7zTSg4VqsySbQJeZD1).

**Estado: Fase 0 (dados) no ar; Fase 1 (pesquisa) com rotulagem, validação e sinais de fluxo prontos.** Nada aqui envia ordem. O código só lê a API pública da KuCoin e grava no banco.

## O que já existe

| Peça | Comando | O que faz |
| --- | --- | --- |
| Histórico | `python -m wescalp.backfill --weeks 100` | Candles de 5 min do spot `BTC-USDT` e do perpétuo `XBTUSDTM`, 100 semanas de cada (~201 mil candles por fonte, ~10 min). Retoma de onde parou. |
| Coletor ao vivo | `python -m wescalp.collector` | Livro de ordens (50 níveis por lado, ~10 atualizações/s) e cada negócio do perpétuo. Grava o topo do livro a cada atualização e a foto completa a cada 1 s. |
| Rótulos | `python -m wescalp.labels` | Tripla barreira em cada candle do perpétuo, para compra e venda, com taxas (ver abaixo). |
| Pesquisa | `python -m wescalp.research` | Mede os 3 setups (rompimento, reversão na VWAP, pullback) contra os rótulos, sem tocar as últimas 16 semanas. |
| Validação | `python -m wescalp.sweep --jobs 4` | Grade de 288 configurações com walk-forward, Deflated Sharpe e PBO (~35 s). |
| Fluxo | `python -m wescalp.flow` | OFI, desequilíbrio do livro, microprice e delta contra o retorno futuro do mid. |
| Candles ao vivo | `python -m wescalp.backfill --follow` | Serviço `candles`: completa as 100 semanas e busca os candles novos a cada minuto. |
| Painel | `python -m wescalp.web` | Serviço `web` (porta 8090): gráfico, marcas dos setups, fluxo, estado do coletor e relatórios. |
| Estado | `python -m wescalp.health` | Estado do coletor e do banco no terminal; o painel mostra o mesmo. Alerta desligado por enquanto. |

A KuCoin não entrega histórico do livro de ordens: ele só existe a partir do dia em que o coletor liga.
Por isso o coletor tem `restart: unless-stopped` e healthcheck.

## Rótulos (Fase 1)

`python -m wescalp.labels --tp 1.5 --sl 1 --bars 12` marca, para cada candle de 5 min do perpétuo, o que teria
acontecido com uma compra e uma venda no fechamento dele (tripla barreira): alvo (`tp`), stop (`sl`) ou tempo
(`time`). Alvo e stop são múltiplos do ATR(14) de Wilder na entrada; a saída por tempo é no fechamento do
`--bars`-ésimo candle. O retorno líquido desconta as taxas: entrada e alvo como maker (0,02%), stop e tempo como
taker (0,06%). Candle que toca alvo e stop ao mesmo tempo conta como stop. Janela que cruza buraco nos candles
fica sem rótulo. Cada combinação de barreiras é gravada sob seu próprio `config` (ex.: `tp1.5-sl1-t12-atr14`) e
substitui a anterior de mesmo nome; `--dry-run` só mostra o resumo.

Rotular todo candle, sem filtro, é a linha de base: o retorno bruto médio fica perto de zero e o líquido perde
as taxas. Um setup só interessa se, nos candles que ele escolhe, o líquido ficar positivo fora da amostra.

`python -m wescalp.research` aplica os setups de `wescalp/setups.py` (só com candles: VWAP do dia, EMA 9/21,
máxima e mínima do dia anterior, volume relativo) e mostra, por setup e lado, entradas, taxa de acerto, retorno
bruto e líquido em bps, intervalo de 90% e em quantas janelas de 4 semanas ficou positivo. As últimas
`--holdout` semanas (padrão 16) ficam de fora até a regra estar congelada.

Primeira rodada (22/10, `tp1.5-sl1-t12-atr14`, nov/2024 a jul/2026): nenhum setup tem vantagem nem antes das
taxas (bruto entre -2,4 e +0,9 bps; ATR mediano de 13,8 bps) e todos perdem de 5,6 a 8,9 bps por entrada
depois delas. Bate com a literatura do documento de pesquisa: candle sozinho não basta, a aposta é o fluxo de
ordens que o coletor está gravando.

## Validação

`python -m wescalp.sweep` testa 36 barreiras (tp ∈ {1; 1,5; 2; 3} ATR, sl ∈ {0,5; 1; 1,5} ATR, tempo ∈ {6; 12; 24}
candles) × 4 regras (os 3 setups e "todos os candles") × 2 lados = 288 configurações, em retorno semanal, fora das
últimas 16 semanas. Sobre elas roda walk-forward (treina 60 semanas, testa 4, anda 4), Deflated Sharpe (Bailey e
López de Prado, 2014) e PBO por CSCV (Bailey, Borwein, López de Prado e Zhu, 2017); as funções ficam em
`wescalp/validation.py`.

Primeira rodada (22/10): nenhuma configuração fica positiva depois das taxas; a escolhida no walk-forward perdeu
42 bps por semana fora da amostra (DSR 0). Antes das taxas há um efeito estável de stop curto e alvo largo
(tp3/sl0.5, 18 de 20 semanas positivas fora da amostra), mas de 0,6 a 0,9 bps por entrada, uma ordem de grandeza
abaixo da taxa, e supondo stop sem derrapagem. Barreiras e setups só de candle estão esgotados.

## Fluxo de ordens

`python -m wescalp.flow [--since ... --until ...]` lê `book_l1` e `trades` e calcula, em baldes de 1 s e 10 s: OFI de
Cont, Kukanov e Stoikov entre fotos consecutivas do topo, desequilíbrio dos 5 primeiros níveis, desvio do microprice
sobre o mid, spread e delta agressor. Para horizontes de 10 a 300 s mostra a correlação com o retorno futuro do mid
e o retorno médio por decil, ao lado da ida e volta (4, 8 ou 12 bps). O sinal usa só mensagens anteriores ao fim do
balde; balde que encosta num buraco de mais de 2 s sem livro fica de fora. Carrega tudo em memória (~100 MB por dia
de livro), então use `--since`/`--until` para períodos longos.

A amostra de 21 min de 22/10 só conferiu o pipeline: o decil extremo andou menos de 0,6 bps em 10 s. O spread é
de 1 tick (~0,01 bps), então o custo é praticamente só taxa, e o perpétuo tem ~0,2 negócio por segundo. A resposta
(algum decil anda mais que 4 a 8 bps?) precisa de semanas de livro gravado.

### Histórico do livro pela Tardis

A Tardis.dev grava o WebSocket da KuCoin Futures desde 25/01/2024 e libera de graça o dia 1º de cada mês
(01/02/2024 a 01/10/2026 = 33 dias; os outros dias são pagos). `python -m wescalp.tardis fetch --jobs 4` baixa
`book_snapshot_5` (5 níveis de cada lado a cada mudança) e `trades` de cada dia, reduz o livro a uma foto por
100 ms como o coletor e grava em `data/tardis/` (~6 MB por dia; o bruto, ~55 MB, é apagado). Roda na máquina de
desenvolvimento, não na VPS. `python -m wescalp.tardis flow` passa os dias pelos mesmos sinais de `wescalp.flow`,
sem retorno que atravesse de um dia para o outro, e conta em quantos dias a correlação de cada sinal foi
positiva. As últimas 8 datas (`--holdout`) ficam de fora até a regra estar congelada. Como o `book_snapshot_5`
só tem linha quando o topo muda, buraco aqui é mais de 30 s sem foto (o maior intervalo de 01/01/2025 foi de 7 s,
sem nenhum negócio no meio).

Primeira rodada (22/10, 25 dias de 01/02/2024 a 01/02/2026; os 33 dias sem buraco > 30 s): o fluxo prevê o mid de
verdade, mas pouco. Desequilíbrio dos 5 níveis, microprice e OFI têm correlação positiva com o retorno de 10 s em
25 de 25 dias (imb5: 0,20 em 10 s, 0,14 em 30 s, 0,10 em 60 s); o delta agressor quase não prevê. O decil extremo
anda no máximo ~0,8 bps em 10 s e ~1,2 bps em 60 a 300 s, contra 4 bps da ida e volta mais barata (maker+maker).
Nenhum sinal sozinho, em decil, paga a taxa. As 8 datas de mar a out/2026 seguem intocadas.

### Modelo (LightGBM)

`python -m wescalp.ml` (precisa de `pip install -e '.[ml]'`) monta uma linha por segundo com OFI e delta em
janelas de 1 a 60 s (normalizados pelos 10 min anteriores), desequilíbrio, microprice, spread, retorno passado,
volatilidade e hora, e prevê o retorno do mid em 60 e 300 s. Walk-forward por dia (treina 10, testa 3, anda 3;
15 dias de teste de dez/2024 a fev/2026), parâmetros fixos, contra ridge e contra cada sinal sozinho nos extremos
de 1% e 0,1% do treino. Entradas sem sobreposição, no mid.

Primeira rodada (22/10): nenhuma regra fica positiva depois de 4 bps, em nenhum horizonte. O que mais prevê é o
desequilíbrio dos 5 níveis nos 1% extremos: +1,3 bps brutos por entrada em 60 s, com t de 15,6 (o efeito é real),
e perde 2,7 bps líquidos. Ir do extremo de 1% para o de 0,1% quase não aumenta o movimento (1,3 → 1,4 bps): a
previsão do fluxo satura em 1 a 2 bps. O LightGBM foi pior que as regras simples: aprendeu volatilidade, spread e
hora (o tamanho do movimento), não a direção. Fluxo de ordens para entrar na direção, com taxa de varejo, está
descartado nesta escala de tempo.

## Painel

`wescalp.avadata.com.br`, atrás do Caddy compartilhado da VPS com `basic_auth`. O serviço `web` entra na rede
`app_default` do Caddy com o nome `wescalp-web` e não publica porta no host. Mostra:

- gráfico de 5 min do perpétuo com VWAP e bandas, EMA 9/21 e o candle em formação (montado dos negócios);
- as entradas dos 3 setups, com cor pelo resultado e tracejado até a saída. No período intocado da pesquisa
  (as últimas 16 semanas) as marcas ficam escondidas até alguém pedir, com aviso;
- faixas nos minutos sem livro gravado e, embaixo, OFI, delta e desequilíbrio do livro da janela visível;
- barra de estado do coletor (`ok`, `atrasado`, `parado`), idade do último livro e negócio, banco e disco;
- aba Pesquisa com o último `research --save` e `sweep --save`.

## Estado do coletor

`wescalp.health` olha o banco: idade do último topo de livro e do último negócio,
minutos sem livro em 24 h, linhas por hora, tamanho do banco e disco livre. Classifica em `ok`, `atrasado` ou
`parado`; o painel mostra esse estado. O alerta (`--watch`, que avisa quando o estado muda) está desligado
por enquanto e não sobe no compose. O aviso vai para
`WESCALP_ALERT_WEBHOOK` (POST `{"text": ...}`; Slack, ou Discord com `/slack` no fim da URL) e/ou Telegram
(`WESCALP_TELEGRAM_TOKEN` + `WESCALP_TELEGRAM_CHAT_ID`); sem canal, só no log. `--test-alert` testa o canal e os
limiares ficam em `WESCALP_HEALTH_*` (padrão: livro parado há mais de 60 s ou negócios há mais de 10 min).
Se a VPS inteira cair, o vigia cai junto; isso pediria um ping externo.

## Tabelas

- `candles` — 5 min, `source` = `spot` ou `futures`.
- `trades` — negócios do perpétuo; `side` é o lado do agressor; `size` em contratos (1 contrato = 0,001 BTC).
- `book_l1` — melhor compra/venda e soma dos 5 primeiros níveis, a cada atualização (base do OFI).
- `book_snapshots` — 50 níveis de cada lado, a cada `WESCALP_BOOK_SNAPSHOT_MS` (padrão 1000).
- `labels` — rótulos de tripla barreira, um por candle, lado e `config` de barreiras.

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

Na VPS: `deploy/ssm-deploy.sh` empacota o commit atual e instala em `/opt/wescalp` via AWS SSM
(precisa de `AWS_EC2_ACCESS_KEY_ID`/`AWS_EC2_SECRET_ACCESS_KEY` e do AWS CLI). O SSM recusa comando acima de
97 KB, então o pacote sobe em partes de 60 KB e é conferido por sha256 antes de instalar. Na primeira vez cria o `.env`
com senha aleatória, sobe `db` + `collector` e dispara o backfill (log em `/var/log/wescalp-backfill.log`).
Na mão, dentro de `/opt/wescalp`:

```bash
docker compose up -d db collector
docker compose run --rm backfill --weeks 100
```

## Próximas fases

1. **Pesquisa:** rótulos, setups só com candles e validação (feitos, sem vantagem); sinais de fluxo (prontos, esperando semanas de livro) e confluências, validação walk-forward, Deflated Sharpe/PBO.
2. **Observação:** motor marca entrada e saída no gráfico a cada candle de 5 min, sem dinheiro; Gemini como segunda opinião.
3. **Livro de lições e rodadas de agentes:** revisão diária e semanal dos trades marcados.
4. **Painel** em `wescalp.avadata.com.br`, atrás do Caddy da VPS.
5. **Dinheiro real pequeno**, só depois de 2 a 3 meses de observação dentro do previsto.
