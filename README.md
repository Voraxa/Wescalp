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
| Carteiras simuladas | `python -m wescalp.paper --follow` | Serviço `paper`: a regra congelada e o aprendiz aplicados ao BTC-USDT spot, gravados dia a dia quando o dia fecha (sem ordem real). |
| Estado | `python -m wescalp.health` | Estado do coletor e do banco no terminal; o painel mostra o mesmo. Alerta desligado por enquanto. |

A KuCoin não entrega histórico do livro de ordens: ele só existe a partir do dia em que o coletor liga.
Por isso o coletor tem `restart: unless-stopped` e healthcheck.

## Como os testes foram feitos

Tudo foi testado em histórico real, sempre do mesmo jeito:

1. O modelo aprende com uma parte do passado.
2. Depois emite os sinais, um a um, num trecho seguinte que ele não viu. Cada sinal usa só o que já era conhecido
   naquele instante.
3. O resultado sai dos candles reais desse trecho, com taxa, DCA e stop simulados candle a candle. Dentro de um
   candle de 5 min não se sabe a ordem dos preços, então conta o pior caso.

É a prova real com o histórico: o sinal sai antes, e a verdade já está nos candles do trecho de teste.

Três cuidados a mais:

- **Walk-forward:** em vez de uma divisão só, o treino anda com o tempo. O modelo aprende até um ponto, testa o
  trecho seguinte e avança, para o resultado não depender de uma data de corte.
- **Desenho fixado antes:** regras, parâmetros e critério de aprovação foram decididos antes de rodar (os últimos
  testes, num commit antes da rodada), para não ajustar o modelo ao resultado.
- **Teste da sorte e dados intocados:** cada candidato foi comparado com milhares de sorteios das mesmas operações em
  momentos ao acaso. Os aprovados foram para trechos guardados que nenhum teste tinha usado: as 16 semanas de jul a
  out/2026 e o BTC de 2013 a 2017 da Bitstamp.

| Teste | Aprendeu com | Sinais testados em | Resultado |
| --- | --- | --- | --- |
| Setups de candle e 288 combinações de alvo e stop | 60 semanas, andando de 4 em 4 | as 4 semanas seguintes, de nov/2024 a jul/2026 | nenhuma lucra depois da taxa |
| Livro de ordens da Tardis com LightGBM | 10 dias, andando de 3 em 3 | os 3 dias seguintes, de dez/2024 a fev/2026 | prevê 0,01% a 0,02%, abaixo da taxa |
| Cérebro de compra com DCA | 60 semanas, andando de 4 em 4 | as 4 semanas seguintes, de jan a jul/2026 | perde em todas as geometrias |
| Spot com alvo fixo e DCA | o histórico da Binance desde 2017, andando | 10/2018 a 06/2026 | perde em todos os alvos |
| Tendência diária (médias de 50 a 150 dias) | regras clássicas fixadas antes | 10/2018 a 06/2026, depois as 16 semanas guardadas | passa no teste da sorte; lucro, mas atrás de segurar nas altas |
| Aprendiz (pesos pelos próprios sinais) | dia a dia, só com o passado | 10/2018 a 06/2026 | empata com a congelada |
| Modelo bayesiano com knots | 87% do histórico e walk-forward anual | os 13% finais e cada ano de 2020 a 2026 | perde para a congelada |
| Risco por cima da congelada | regras fixadas antes | 10/2018 a 06/2026 | nenhum melhora |
| Prova única da congelada | nada: dados nunca vistos | 2013 a 08/2017, Bitstamp | lucro e queda menor, mas não passa no teste da sorte |
| Fluxo, prêmio, funding e posições | até 2022 ou 2023 | 2023 ou 2024 a 06/2026 | nenhum sinal paga a taxa |
| Sistema de day trade em camadas | walk-forward anual desde 2020 | cada ano de 2021 a 06/2026 | perde de 19% a 49% ao ano |

**Limites:**

- É simulação sobre o passado, não operação real. Na prática a ordem limitada nem sempre é executada e o preço
  escorrega um pouco, então o resultado real tende a ser igual ou pior.
- Nenhum teste prova que nada funciona; prova que o que foi tentado não funcionou.
- Comparar com segurar numa janela só depende do dia de entrada e do de saída. Com cada dia como entrada, a
  congelada ganhou de segurar em pouco mais da metade das janelas de 1 e 2 anos, mas caiu menos em 79% a 89% delas (ver
  "Comparar com segurar depende do dia de entrada"). Os vereditos de day trade não dependem de segurar: a régua era
  ter lucro, e eles perdem dinheiro com a taxa.
- O único teste do futuro de verdade é a carteira simulada ao vivo, rodando desde 23/10/2026.

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

## Cérebro de compra com DCA

`python -m wescalp.brain fetch` copia as 100 semanas de candles de 5 min para `data/candles/` (a mesma API do
backfill; a pesquisa com modelo roda fora da VPS) e `python -m wescalp.brain run` testa o ciclo só de compra:
sete gatilhos gráficos (pullback na tendência de 1 h, banda inferior da VWAP, RSI sobrevendido, Bollinger
inferior, recuperação da mínima de ontem, rompimento da máxima de ontem, rejeição com tendência de 4 h), um juiz
LightGBM que dá a probabilidade de o ciclo fechar no lucro, entrada a mercado, duas ordens de DCA a 1 e 2 passos
de ATR de 1 h, alvo sobre o preço médio, um stop para a posição inteira e 12 h no máximo. Walk-forward de 60
semanas (o juiz aprende em 45, o corte sai das 15 seguintes, mirando acerto ≥ 85% e ≥ 2 ciclos/dia), testa 4,
anda 4; 12 geometrias de alvo/DCA/stop; últimas 16 semanas intocadas.

Primeira rodada (22/10, 24 semanas de teste de 17/01 a 04/07/2026, em que o BTC caiu 34%): nenhuma regra fica
positiva em nenhuma geometria. Acerto de 80% sai da geometria, não do juiz: entrar em qualquer candle com alvo de
0,5 ATR e stop de 4 ATR já acerta 86,9%, e o juiz nessa geometria acerta 88,5% e perde 6,6 bps por ciclo
(-36% do capital em 24 semanas). O melhor juiz (tp1-dca0.5-sl4) perdeu 2,2 bps por ciclo com 76% de acerto (DSR
0,04 entre 12). O juiz decora o treino (AUC 0,93) e quase não separa ciclos fora dele (AUC 0,43 a 0,65); sem taxa,
continua negativo (-1 a -3 bps por ciclo).

## Entrada no spot com alvo fixo e DCA por sinal

`python -m wescalp.spot fetch` copia 100 semanas de candles de 5 min do BTC-USDT e `python -m wescalp.spot run`
testa o pedido de 23/10: só compra no spot, o juiz escolhe a entrada, o DCA (até 2 aportes de 1/3) só compra com
um novo sinal e o preço pelo menos um alvo abaixo da última compra, venda no alvo fixo sobre o preço médio (2%,
3% ou 5%), stop definido na entrada a 3 alvos abaixo dela, 30 dias no máximo, taxa de 0,1% por lado. O juiz aprende
a "boa entrada" (subir o alvo antes de cair o alvo) com contexto de 15 min, 1 h, 4 h e 1 dia. Mesmo walk-forward;
o teste é uma simulação contínua das 24 semanas.

Primeira rodada (23/10, teste de 15/01 a 02/07/2026, BTC -35,7%): nenhum alvo dá lucro e nenhum chega a 80% de
ciclos no lucro. O melhor é o alvo de 2% com juiz: 32 ciclos (0,19 por dia), 78% de acerto, -2,7% no período
contra -21,5% dos gatilhos sem juiz, -31,7% do DCA cego e -35,7% de segurar o BTC; queda máxima de 18%. O juiz
ganhou porque ficou fora do mercado metade do tempo (as variáveis do diário dominam a importância), não por
escolher entradas melhores: as entradas dele subiram o alvo antes de cair em 34% das vezes, contra 42% ao acaso.
Com 32 ciclos e um só regime (queda), a amostra é pequena demais para concluir (DSR 0,21 entre 3 alvos).

Segunda rodada (23/10, histórico da Binance de `python -m wescalp.binance`, 931 mil candles de 08/2017 a
06/2026, `run --file ... --tps 0.01 0.015 0.02 --holdout 0`; teste contínuo de 11/10/2018 a 30/06/2026, 100
rodadas, em que segurar o BTC rendeu +825%): todas as regras perdem em todos os alvos. As entradas do juiz sobem
o alvo antes de cair o alvo em 49% a 51% das vezes, o mesmo que ao acaso (50%): não há boa entrada. O acerto dos
ciclos fica em 78% a 80% por causa do stop 3 alvos abaixo, e mesmo assim cada ciclo perde 0,05% a 0,10% do capital,
o que somado passa de todo o capital (juiz com alvo de 2%: -116%; com 1%: -263%). Nem nos anos de alta salva:
em 2020 (+302% do BTC) o melhor juiz fez +63%. O juiz ≥ 2/dia nunca achou corte com 85% de acerto e 2 ciclos/dia
na calibração. O modelo aprende o treino (AUC 0,82 a 0,92) e fora dele oscila de 0,30 a 0,74 conforme o regime,
média perto de 0,5 (DSR 0,000 entre 6 juízes).

## Tendência diária (regime)

`python -m wescalp.regime --file data/candles/BINANCE-BTCUSDT-5m.csv.gz` testa ficar comprado no BTC spot só
quando a tendência diária é de alta, com 9 regras clássicas fixadas antes (médias de 50, 100, 150 e 200 dias,
cruzamentos 20/50 e 50/200, momento de 30, 90 e 180 dias), decisão no fechamento valendo para o dia seguinte e
0,1% por lado. O teste da sorte embaralha o momento dos trechos comprado e fora, mantendo o número de operações, o
tempo comprado e a duração de cada uma (2.000 sorteios). `--cycles 0.02 0.03` roda os ciclos de `wescalp.spot`
(DCA com novo sinal, stop 3 alvos abaixo) só nos dias em que a regra está comprada, contra 200 sorteios.

Primeira rodada (23/10, de 11/10/2018 a 30/06/2026; segurar o BTC: +783%, 32,6% ao ano, queda máxima de 77%):

| Regra | Total | Ao ano | Queda máx. | Sharpe | Operações/ano | Acerto/operação | p da sorte (total) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| média de 50 dias | +3.305% | 57,9% | 58% | 1,28 | 9,7 | 25% | 0,004 |
| média de 100 dias | +2.474% | 52,3% | 38% | 1,18 | 6,3 | 22% | 0,029 |
| média de 150 dias | +2.278% | 50,7% | 45% | 1,15 | 4,0 | 26% | 0,040 |
| momento de 30 dias | +2.006% | 48,4% | 47% | 1,13 | 15,2 | 37% | 0,018 |
| média de 200 dias | +897% | 34,7% | 64% | 0,89 | 3,9 | 23% | 0,180 |

As regras de 50 a 150 dias passam no teste da sorte e o Sharpe continua acima de zero mesmo contando 1.000
tentativas (DSR de 0,96 a 0,98); as lentas (200 dias, 50/200, momento de 90 e 180) não se distinguem da sorte. O
ganho é ficar fora das quedas (2022: -64% do BTC, -21% da média de 100 dias); render mais que segurar o BTC ainda
não está provado (P(excesso > 0) de 0,58 a 0,71). Os ciclos dentro da regra viram positivos e passam no sorteio
(p de 0,000 a 0,020), com 82% a 84% de ciclos no lucro, mas rendem pouco: média de 100 dias + alvo de 3% fez +205%
composto (15,5% ao ano, queda máxima de 41%, 0,27 ciclo por dia), contra +2.474% de só seguir a regra. Ciclos com
alvo de 1% (2 por dia) perdem mesmo dentro da tendência.

### Estudo das aprovadas

`python -m wescalp.study --file data/candles/BINANCE-BTCUSDT-5m.csv.gz` compara as regras que passaram no teste da
sorte e a combinação 50/100/150 (1/3 do capital em cada média), tudo reinvestido, de 11/10/2018 a 30/06/2026:

| Estratégia | Ao ano | Queda máx. | Sharpe | Fator de lucro (meses) | Pior mês | Anos + | Ao ano com 0,5%/lado | P(bate segurar) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| segurar BTC | 32,6% | 77% | 0,77 | 1,75 | -37% | 5/9 | 33% | — |
| média 50 | 57,9% | 58% | 1,28 | 2,76 | -20% | 6/9 | 46% | 88% |
| média 100 | 52,3% | 38% | 1,18 | 2,77 | -18% | 6/9 | 45% | 80% |
| média 150 | 50,7% | 45% | 1,15 | 2,83 | -20% | 5/9 | 46% | 78% |
| combinação 50/100/150 | 54,9% | 44% | 1,25 | 2,89 | -15% | 6/9 | 47% | 85% |
| média 100 + ciclos de 3% | 15,5% | 41% | 0,68 | 1,77 | -15% | 5/9 | -26% | 20% |

As médias de 60 a 170 dias formam um platô (44% a 64% ao ano, Sharpe 1,05 a 1,34), sem pico isolado; mudar a hora
em que o dia vira (00, 06, 12, 18 h UTC) mexe pouco no retorno, mas a queda de 38% da média de 100 só aparece às
00 h (50% a 52% nas outras horas). A combinação passa no teste da sorte (p 0,000 no total, 0,002 no Sharpe, 0,037
na queda). O ganho sobre segurar veio da primeira metade (até 06/2022: 86% contra 34% ao ano); na segunda, o
retorno empata (30% contra 31%) e o ganho é cair menos (29% contra 53%). Os ciclos perdem quase tudo com 0,2% por
lado (4% ao ano). P(bate segurar) vem de um bootstrap estacionário pareado de 1.000 reamostras.

### Refinamentos e prova única

`python -m wescalp.refine --file ...` testa três refinamentos da combinação com a regra de decisão fixada antes
(Sharpe maior, queda igual ou menor, sem piorar a pior hora de virada do dia nem a segunda metade, p ≤ 0,05 no
teste da sorte): reduzir a exposição pela volatilidade de 30 dias, a média das 13 regras de 50 a 170 dias e a
mistura com o momento de 30 dias e o cruzamento 20/50. Nenhum cumpriu todos os critérios (23/10): a volatilidade
subiu o Sharpe (1,27 contra 1,25) mas caiu de 54% para 44% ao ano na pior hora; as outras empataram. A
combinação 50/100/150 ficou congelada como está.

Prova única nas 16 semanas intocadas (KuCoin BTC-USDT spot, 04/07 a 21/10/2026, 110 dias, rodada uma vez em
23/10): a combinação fez +22,2% contra +34,2% de segurar o BTC, as duas com queda máxima de 6,9%. Entrou aos
poucos depois do fundo de julho (33% em vaivém até agosto) e está 100% comprada desde 19/08. Lucro, mas atrás de
segurar numa alta, o mesmo padrão da segunda metade do estudo.

### Aprendiz: aprender com os próprios sinais

`python -m wescalp.learn --file ...` testa uma carteira que aprende com os sinais que ela mesma dá, com a forma de
aprender fixada antes de rodar: as 13 médias de 50 a 170 dias como componentes, cada uma com uma nota (log do
retorno dos sinais dela, com taxa, descontado com meia-vida de 180 dias) e peso exp(η · nota) (Hedge, η = ln 2 / 0,1:
10% a mais dobra o peso relativo), posição = soma dos pesos das médias compradas, ajuste só quando a posição desejada
se afasta 5 p.p. Controle: as mesmas 13 médias com pesos iguais, para isolar o efeito de aprender.

Primeira rodada (23/10, 11/10/2018 a 30/06/2026): aprendiz 52,9% ao ano, queda de 41%, Sharpe 1,22, p 0,004 no teste
da sorte; sem aprender, 53,9%, 43%, 1,23; congelada, 54,9%, 44%, 1,25. Ele aprende coisas sensatas (médias rápidas na
alta de 2020-2021, lentas na queda de 2022), mas não rende mais: as médias andam juntas, e escolher a melhor entre
elas vale pouco. Pela regra fixada, a congelada segue campeã; o aprendiz entra na carteira simulada ao vivo como
experimento, ao lado dela.

### Modelo bayesiano com knots e fatores

`python -m wescalp.factors` baixa os fatores de fora do preço: Medo e Ganância (alternative.me), Nasdaq e dólar
(FRED). Depois, `python -m wescalp.bayes --file ...` (precisa de `pip install -e '.[bayes]'`) roda um modelo no estilo
do Meridian. O desenho foi fixado antes de rodar:

- **Decisão:** semanal, toda segunda 00h UTC. O alvo é o retorno do BTC na semana seguinte.
- **Linha de base:** um knot a cada ~26 semanas, com passeio aleatório entre eles. Daí para a frente vale o último.
- **Fatores:** 7, cada um com peso β e incerteza:
  - a posição da combinação 50/100/150;
  - o Medo e Ganância (nível e variação de 7 dias);
  - Nasdaq, dólar e BTC em 4 semanas;
  - a volatilidade de 30 dias do BTC.
- **Sem olhar o futuro:** cada fator só usa o que já era público na hora da decisão. O dólar do Fed sai às segundas,
  com a semana anterior inteira, e por isso entra com uma semana a mais de atraso.
- **Estimação:** erro Student-t, NUTS com 4 cadeias (sementes 0 a 3) e uma segunda rodada com as sementes 4 a 7.
- **Divisão:** 87% das semanas para treino (fev/2018 a mai/2025) e 13% para teste (mai/2025 a jun/2026), e depois
  um walk-forward anual de 2020 a 2026.
- **Carteira:** (a) 100% comprado se a chance de alta passa de 50%; (b) comprado na fração dessa chance.

Primeira rodada (23/10):

| Carteira | Teste de 13% (57 semanas): total / queda / Sharpe | Walk-forward 2020-2026 (338 semanas): ao ano / queda / Sharpe | p da sorte (walk-forward) |
| --- | --- | --- | --- |
| bayes (a), chance > 50% | -35% / 43% / -0,77 | 17,0% / 79% / 0,57 | 0,845 |
| bayes (b), fração da chance | -26% / 31% / -1,10 | 24,0% / 53% / 0,80 | 0,717 |
| congelada 50/100/150 | -14% / 23% / -0,64 | 50,8% / 44% / 1,22 | — |
| aprendiz | -16% / 25% / -0,74 | 48,6% / 41% / 1,17 | — |
| segurar BTC | -45% / 52% / -1,11 | 38,1% / 77% / 0,84 | — |

Nenhum fator tem peso que se distinga de zero: todas as faixas de 90% passam pelo zero. A média 50/100/150, por
exemplo, ficou em +0,42% por semana por desvio, com faixa de -0,57% a +1,39%. Os sinais se repetem nos 7 ajustes do
walk-forward, mas os pesos são pequenos demais para mudar a decisão:

- sempre positivos: a média e a variação de 7 dias do Medo e Ganância;
- sempre negativos: o dólar e o BTC de 4 semanas.

A chance de alta prevista ficou entre 49% e 61%. O acerto da direção foi de 50,6% no walk-forward, abaixo de
apostar sempre na alta (52,4% das semanas subiram), e a nota de Brier ficou pior que a de prever sempre a frequência
do treino. A linha de base leva a tendência do passado para a frente. Depois da alta de 2021, o modelo ficou
comprado na queda de 2022: -65%, contra -28% da congelada.

As duas rodadas de sementes deram a mesma decisão em 99% a 100% das semanas. O R-hat ficou em no máximo 1,006, com 16
divergências em 64 mil passos. Pela regra fixada, o modelo não entra na carteira ao vivo: as duas formas falham em
todos os critérios (Sharpe e queda contra a congelada, teste da sorte e teste de 13%).

A lição é que a média móvel não acerta a direção da semana. Ela ganha porque sai das quedas grandes: acerta só 22% a
26% das operações, mas os ganhos são muito maiores que as perdas. Um modelo que prevê o retorno médio da semana não
pega essa assimetria.

### Trecho trancado: BTC de 2011 a 2017 (Bitstamp)

`python -m wescalp.bitstamp` grava em `data/locked/` os candles de 1 h do BTC/USD da Bitstamp de 18/08/2011 a
16/08/2017. São os seis anos antes do primeiro dado da Binance, que nenhuma regra viu. Esse trecho é a reserva para
uma prova única:

- **Quem roda:** a congelada, o aprendiz e quem passar nos próximos testes.
- **Quando:** uma vez só, no fim.
- **Período avaliado:** de 01/01/2013 a 16/08/2017. O que vem antes só aquece as médias.
- **Trava:** até lá, `brain.load_candles` se recusa a abrir a pasta sem `WESCALP_UNLOCK=prova-unica`.

Conferência dos dados, sem nenhuma estratégia (23/10):

- 52.572 horas, sem buraco.
- Horas sem negócio: 90% em 2011, 37% em 2012 e no máximo 1,4% por ano de 2013 em diante. Os 1,2% de 2015 são a
  parada da Bitstamp em janeiro.
- Saltos absurdos de uma hora (de -82% a +242%) só em 2011, fora do período avaliado.
- Nos 41 dias em que Bitstamp e Binance se sobrepõem (ago–set/2017), a diferença mediana do fechamento foi 0,00% e a
  correlação dos retornos diários, 0,97.

SHA-256 do conteúdo: `5fb3f3543026448e83de75f0ec0b201429687dcccf6fdd33da65a60fe5e8d78e`. Um segundo download deu o
mesmo hash; a prova única confere esse valor antes de rodar.

### Modelo de risco por cima da congelada

`python -m wescalp.risk --file ...` testa três candidatos que só mexem no tamanho da posição da congelada, entre 0% e
100%. Antes, `python -m wescalp.events` monta o calendário (reuniões marcadas do FOMC, das páginas do Fed, e
divulgações do CPI, das safras do índice no ALFRED) e `python -m wescalp.factors` baixa o VIX. O desenho foi
registrado num commit antes da primeira rodada (`decc6c2`):

- **volatilidade prevista:** a posição cresce quando o mercado está mais calmo que a mediana de um ano e encolhe
  quando está mais agitado (EWMA, λ = 0,94);
- **cautela em notícia:** metade da posição nos dias de comunicado do FOMC ou de divulgação do CPI;
- **VIX:** a mesma conta da volatilidade, com o VIX no lugar.

Primeira rodada (23/10, de 11/10/2018 a 30/06/2026):

| Carteira | Ao ano | Queda máx. | Sharpe | Pior hora | 2ª metade | p da sorte |
| --- | --- | --- | --- | --- | --- | --- |
| congelada 50/100/150 | 54,9% | 44% | 1,25 | 54% | 30% | — |
| volatilidade prevista | 46,6% | 48% | 1,22 | 43% | 27% | 0,475 |
| cautela FOMC/CPI | 48,0% | 41% | 1,17 | 46% | 27% | 0,905 |
| VIX | 51,1% | 42% | 1,23 | 51% | 27% | 0,406 |

Nenhum passa na régua. Todos rendem menos que a congelada, e o momento de cada um não é melhor que o mesmo ajuste
deslocado ao acaso no tempo (p de 0,41 a 0,91). A volatilidade prevista ainda caiu mais (48%): na queda de nov/2021
a jan/2023, ficou um pouco mais comprada que a congelada (16% contra 14% em média), porque aumentava a posição nos
trechos calmos. Nenhum vai para a prova única.

As notícias mexem com o BTC, mas sem direção:

- **FOMC:** nas 2 h depois do comunicado, o movimento mediano foi de 0,89%, contra 0,56% na mesma hora dos dias
  úteis vizinhos (1,6 vez).
- **CPI:** 0,87% contra 0,77% (1,1 vez).
- **Direção:** o preço subiu em 48% (FOMC) e 49% (CPI) das vezes.
- **Dia inteiro:** o retorno absoluto médio sobe de 2,15% para cerca de 2,65%.

Reduzir a posição nesses dias custa mais, em taxa e em alta perdida, do que evita em queda.

### Prova única em 2013–2017

`WESCALP_UNLOCK=prova-unica python -m wescalp.proof` rodou uma vez só, em 23/10:

- os critérios foram registrados antes, no commit `81737f3`;
- o SHA-256 dos dados foi conferido;
- período: 01/01/2013 a 16/08/2017 (1.689 dias), BTC/USD da Bitstamp, 0,1% por lado.

| Estratégia | Total | Ao ano | Queda máx. | Sharpe | p da sorte | Resultado |
| --- | --- | --- | --- | --- | --- | --- |
| segurar BTC | +32.940% | 250,5% | 85% | 1,84 | — | — |
| congelada 50/100/150 | +26.615% | 234,8% | 69% | 1,98 | 0,100 | não passou (sorte) |
| aprendiz | +40.421% | 266,4% | 70% | 2,06 | 0,059 | não passou (sorte) |

Por ano, a congelada contra segurar:

| Ano | Congelada | Segurar |
| --- | --- | --- |
| 2013 | +3.367% | +5.423% |
| 2014 | -21% | -56% |
| 2015 | +43% | +34% |
| 2016 | +91% | +124% |
| 2017 (até 16/08) | +255% | +353% |

- **A regra fez o que faz:** na queda lenta de 2014 (o BTC caiu 85% do topo de dez/2013 a jan/2015), ela perdeu 21%
  no ano, contra 56% de segurar. Teve lucro e Sharpe acima de segurar.
- **Mas não passou no teste da sorte:** 10% dos sorteios, com o mesmo tempo comprado em momentos ao acaso, tiveram
  Sharpe igual ou maior. A régua aceita no máximo 5%. Numa época em que o BTC se multiplicou por 330, quase qualquer
  exposição alta rende muito.
- **A queda máxima da congelada (69%) veio do crash de abril de 2013:** o BTC despencou em uma semana (09 a 16/04),
  com a regra 96% comprada. Uma regra de médias diárias não consegue sair de uma queda de poucos dias.
- **As regras clássicas trocam de lugar entre as épocas:** a média de 200 dias e o momento de 180 dias, que não
  passaram em 2018–2026, ficaram entre as melhores aqui. Todas as regras de tendência tiveram Sharpe acima de
  segurar (1,91 a 2,07, contra 1,84). O que se repete nas duas épocas é seguir tendência perder menos nas quedas
  lentas (2014, 2022), não uma média específica.

Pela regra fixada, a congelada não ganha nesta época a prova de que não é sorte. No painel ela segue como estava. O
trecho trancado já foi usado: ideias novas passam a ser julgadas pelo walk-forward e pelo período ao vivo.

### Comparar com segurar depende do dia de entrada

As contas acima comparam a regra com segurar numa janela só, de 11/10/2018 a 30/06/2026. O resultado de segurar
depende muito do dia em que se compra e do dia em que se mede. `python -m wescalp.entries` refaz a comparação com cada
dia como entrada. O desenho foi registrado antes da rodada (`98700fb`):

- **Entrada:** começa em USDT e paga 0,1% para comprar. A congelada compra os terços que a regra manda naquele dia;
  segurar compra tudo. Nenhuma das duas vende no fim.
- **Janelas:** de 1, 2 e 4 anos, e de cada entrada até 30/06/2026 (com pelo menos 1 ano).
- **Régua de mesma exposição:** no BTC, a congelada tem em média de 55% a 62% do capital comprado, então perde nas
  altas e ganha nas quedas só por isso. Essa régua compra no começo a mesma fração média e segura.

BTC, rodado em 25/10:

| Janela | Entradas | Congelada acima de segurar | Acima da mesma exposição | Caiu menos que segurar | Mediana ao ano: congelada / segurar | Pior janela ao ano: congelada / segurar | Janelas no prejuízo: congelada / segurar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ano | 2.456 | 51% | 60% | 89% | +48,7% / +54,7% | -44,5% / -76,3% | 22% / 31% |
| 2 anos | 2.091 | 56% | 80% | 79% | +60,8% / +67,5% | -12,8% / -40,5% | 5% / 22% |
| 4 anos | 1.360 | 97% | 100% | 100% | +54,8% / +34,2% | +18,6% / +7,0% | 0% / 0% |
| até 30/06/2026 | 2.456 | 98% | 100% | 100% | +30,8% / +24,9% | -14,8% / -45,9% | 7% / 21% |

- **Ganhar de segurar não se sustenta em 1 e 2 anos:** depende da entrada, e a congelada ganhou em 51% e 56% das
  janelas. Quem entrou em 2023, no começo da alta, perdeu para segurar em todas as janelas de 2 anos. No topo de
  10/11/2021, segurar fez -76% no ano seguinte e a congelada -44%; no fundo da Covid, em 13/03/2020, segurar fez
  +1.093% e a congelada +537%.
- **O que se repete em quase toda entrada é cair menos:** em 89% das janelas de 1 ano e em todas as de 4 anos. O
  pior ano da congelada foi -44,5%, contra -76,3% de segurar. A exceção foi o vaivém de 03 a 10/2024: para quem
  entrou em 2023, a congelada caiu 29%, contra 26% a 28% de segurar, porque vendeu nas quedas e recomprou mais caro.
- **Contra a régua de mesma exposição, ganha mais vezes:** em 60% das janelas de 1 ano, 80% das de 2 anos e todas as
  de 4 anos. A hora de entrar e sair vale alguma coisa além de ficar menos tempo comprado.
- **Em 4 anos, ganhou em 97% das entradas** porque toda janela de 4 anos destes dados pega ao menos uma queda grande
  (2018, 2022 ou 2025–2026). Isso depende de o BTC continuar com esses ciclos.
- **A janela usada no resto do README foi boa para a congelada:** de 11/10/2018 até o fim, ela ganhou 22,3 p.p. ao
  ano de segurar (54,9% contra 32,6%), mais do que em 90% das entradas; na entrada mediana, foram 14,9 p.p. O fim
  também ajuda: em 30/06/2026 o BTC estava em 58.625, 53% abaixo do topo de 06/10/2025.
- **As janelas se sobrepõem:** em 7,7 anos, janelas de 2 anos têm só 3 ou 4 independentes, e é o mesmo período em
  que a regra foi escolhida. A tabela mede o quanto a conclusão depende da entrada; não é prova nova.

Cesta das 6 moedas de "Moedas mais voláteis", de 01/07/2021 a 30/06/2026:

| Janela | Entradas | Congelada acima de segurar | Acima da mesma exposição | Caiu menos que segurar | Mediana ao ano: congelada / segurar | Pior janela ao ano: congelada / segurar | Janelas no prejuízo: congelada / segurar |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ano | 1.462 | 38% | 53% | 100% | +18,9% / +22,1% | -42,5% / -80,0% | 46% / 40% |
| 2 anos | 1.097 | 43% | 57% | 100% | +26,0% / +44,0% | -28,8% / -46,2% | 13% / 30% |
| até 30/06/2026 | 1.462 | 96% | 93% | 100% | +18,2% / -2,0% | -27,5% / -57,1% | 16% / 54% |

- **O "18% contra 2%" da cesta depende da data:** a entrada de 01/07/2021 foi a melhor de todas para a congelada em 1
  ano. Em janelas de 1 e 2 anos, ela perdeu de segurar na maioria das entradas, por 22 e 13 p.p. ao ano na mediana.
  Contra a régua de mesma exposição, dá cara ou coroa (53% e 57%).
- **Até 30/06/2026, ganha em 96% das entradas porque as altcoins estavam caídas no fim:** segurar a cesta até ali deu
  prejuízo em 54% das entradas. É o efeito da data de saída.
- **Caiu menos que segurar em todas as janelas.**

## Day trade: informação nova

O norte do projeto é o day trade: várias compras e vendas por dia, com heurísticas, confluências e modelos rodando
juntos, e DCA para corrigir a entrada. Os juízes anteriores só olhavam candles. `python -m wescalp.flowdata` baixa do
data.binance.vision o que eles não viam:

- o fluxo agressor de 5 min do spot (desde 08/2017) e do perpétuo (desde 01/2020);
- o prêmio do perpétuo sobre o índice e o funding (desde 01/2020);
- as posições abertas e a relação compradas/vendidas das contas e dos grandes traders (desde 12/2021).

Não há liquidações históricas grátis: a série "liquidação de comprados" usa a queda de preço junto com a queda das
posições.

`python -m wescalp.intraday --file ...` testa se algum desses sinais prevê o BTC spot nas 1, 4 e 12 h seguintes mais
do que a ida e volta de 0,2%. O desenho foi registrado antes da rodada (`ee2b6c4`):

- **Sinais:** 14 sinais isolados e duas confluências de todos eles com a hora do dia (ridge e LightGBM).
- **Decisão:** de hora em hora, de 2020 a 06/2026.
- **Treino:** o lado (decil de cima ou de baixo) e o corte saem do treino (até 2022, ou até 2023 para os sinais de
  posições).
- **Régua:** passa quem render, fora da amostra, mais que 0,2% por operação com t ≥ 3.

Primeira rodada (23/10): **nenhum passa**. Melhores decis fora da amostra:

| Sinal | Horizonte | Lado | Retorno médio fora da amostra | Base | t |
| --- | --- | --- | --- | --- | --- |
| retorno da última hora | 1 h | queda forte | 0,058% | 0,005% | 2,7 |
| variação do prêmio em 1 h | 4 h | queda forte | 0,350% (140 casos) | 0,022% | 2,5 |
| prêmio do perpétuo | 4 h | mais negativo | 0,151% | 0,022% | 2,8 |
| retorno de 24 h | 4 h | queda forte | 0,201% | 0,021% | 2,4 |
| fluxo do perpétuo em 24 h | 12 h | compra forte | 0,339% | 0,064% | 2,6 |
| confluência ridge | 12 h | previsão alta | 0,179% | 0,034% | 2,4 |

O LightGBM decorou o treino (1,74% dentro, 0,04% fora em 12 h).

O padrão que se repete, escolhido no treino e confirmado na direção fora dele, é o de **comprar depois de queda
forte**: retorno negativo, prêmio caindo ou negativo, funding baixo, contas mais vendidas. Ele rende mais que a
média, mas pouco por operação:

- em 1 h, de 0,05% a 0,07%, um terço da taxa do spot;
- em 4 e 12 h, de 0,15% a 0,35%, perto dos 0,2% da ida e volta e sem t ≥ 3.

Com a taxa do spot (0,1% por lado), esse efeito não paga várias operações por dia. Com ordem limitada nos futuros
(0,02% por lado, ida e volta de 0,04%), pagaria com folga, se for real.

### O sistema de day trade em camadas

`python -m wescalp.daytrade --file ...` monta o sistema pedido. O desenho foi registrado antes da rodada (`36e1993`):

- **Dia:** a tendência diária (a congelada) dimensiona as partes: cheias na alta, um terço na baixa. O sistema nunca
  deixa de operar.
- **Confluência:** um ridge prevê as 4 h seguintes com os 14 sinais de fluxo, prêmio e posições, três heurísticas de
  candle (RSI, Bollinger, VWAP) e a hora do dia. Walk-forward anual, testado de 2021 a 06/2026.
- **Entrada:** a cada hora, se a previsão estiver no quinto de cima. Entradas forçadas, com metade do tamanho,
  garantem 2 ciclos por dia.
- **Ciclo:** até 2 DCAs com ordem limitada, alvo sobre o preço médio, stop da posição e 12 h no máximo.
- **Variantes:** passo e alvo de 0,6% ou de 1,0%, cada um com e sem um juiz que aprende com o resultado dos próprios
  ciclos.

Primeira rodada (23/10, de 2021 a 06/2026, 2.007 dias, 0,1% por lado):

| Variante | Total | Ao ano | Queda máx. | Ciclos/dia | Dias com 2+ | Acerto | Líquido sobre o capital usado | p da sorte |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| passo 0,6% | -98% | -49,0% | 98% | 3,49 | 94% | 76% | -0,170% por ciclo | 0,998 |
| passo 0,6% com juiz | -76% | -23,2% | 77% | 1,95 | 88% | 75% | -0,167% | 0,252 |
| passo 1,0% | -91% | -35,6% | 92% | 2,51 | 83% | 66% | -0,165% | 0,945 |
| passo 1,0% com juiz | -69% | -19,3% | 70% | 1,77 | 71% | 66% | -0,154% | 0,150 |
| segurar BTC | +102% | 13,7% | 77% | — | — | — | — | — |
| congelada 50/100/150 | +347% | 31,3% | 44% | — | — | — | — | — |

Retorno ao ano com as mesmas entradas e taxas menores:

| Variante | 0,10% | 0,08% | 0,04% | 0,02% | 0% |
| --- | --- | --- | --- | --- | --- |
| passo 0,6% | -49,0% | -40,3% | -18,3% | -4,4% | +11,9% |
| passo 0,6% com juiz | -23,2% | -18,2% | -7,2% | -1,2% | +5,2% |
| passo 1,0% | -35,6% | -28,5% | -12,0% | -2,3% | +8,4% |
| passo 1,0% com juiz | -19,3% | -14,8% | -5,0% | +0,4% | +6,0% |

- **Ele opera como pedido:** de 1,8 a 3,5 ciclos por dia, 2 ou mais em 71% a 94% dos dias, e 66% a 76% dos ciclos no
  lucro.
- **Mas cada ciclo perde cerca de 0,17% do capital usado.** O ganho bruto é de uns 0,03% por ciclo; a taxa, 0,2%.
- **Mesmo com taxa zero, rende de 5% a 12% ao ano,** menos que segurar o BTC (13,7%) e que a congelada (31,3%).
- **O momento das entradas não é melhor que horas ao acaso:** p de 0,15 a 1,0.
- **O juiz aprendeu a barrar quase todas as entradas por sinal:** de 87% a 91% dos ciclos viraram forçados, com metade
  do tamanho. As variantes com juiz perdem menos porque operam menos, não porque escolhem melhor.

Pela regra fixada, nenhuma variante passa; nada vai para as 16 semanas intocadas nem para o painel.

**Sem DCA** (`--sem-dca`, fixado em `e21b71c`): uma compra só por ciclo, com o capital das 3 partes de uma vez. Fica
pior:

- perde de 31% a 68% ao ano com 0,1% por lado (acerto de 59% a 69%, p de 0,33 a 1,0);
- com taxa zero, rende de 5% a 17% ao ano.

A perda por real usado é a mesma (cerca de 0,17% por ciclo), mas cada ciclo usa o capital inteiro.

### Efeitos de horário publicados

`python -m wescalp.sessions --file ...` testa três regras de day trade publicadas, usando a data de publicação como
corte: o teste vai de 07/2022 a 06/2026, depois dos artigos, num trecho que os autores não viram. O desenho foi
registrado antes da rodada (`e21b71c`).

- **Fora do pregão americano** (Bespoke, citado pela Bloomberg em 06/2022): compra no fechamento de Nova York e vende
  na abertura seguinte.
- **21h às 23h UTC** (Quantpedia).
- **Momento intradiário** (Shen, Urquhart e Wang, *Financial Review*, 2022): se a primeira meia hora do dia subiu,
  compra às 23h30 UTC e vende à meia-noite.

| Regra (07/2022 a 06/2026) | Operações | Ao ano com 0,1% | Com 0,02% | Com 0% | p da sorte |
| --- | --- | --- | --- | --- | --- |
| fora do pregão americano | 1.000 | -5,1% | +41,6% | +56,5% | 0,041 |
| 21h às 23h UTC | 1.461 | -42,0% | +4,1% | +20,5% | 0,009 |
| momento intradiário | 728 | -31,6% | -8,5% | -1,6% | 0,943 |
| segurar BTC | — | +30,9% | — | — | — |

Nenhuma passa na regra fixada: todas perdem com a taxa do spot. Mesmo assim, é a primeira pista real do day trade:

- **Fora do pregão americano:** antes da taxa, ficar comprado só com a bolsa americana fechada (cerca de 80% do
  tempo) rendeu mais que segurar o tempo todo: 56,5% contra 30,9% ao ano. O efeito sobreviveu à publicação. Com
  0,02% por lado (ordem limitada nos futuros), renderia 41,6%. Ele empata com segurar com cerca de 0,035% por lado. O
  p de 0,041, porém, não chega ao limite de 0,0167 para três regras.
- **21h às 23h UTC:** o horário importa de verdade (p 0,009), mas o ganho bruto, de 20,5% ao ano em 2 h por dia, não
  paga a taxa.
- **Momento intradiário:** não se repetiu.

Prova única da regra fora do pregão americano nas 16 semanas guardadas (`sessions --holdout`), rodada uma vez em 23/10:

- **Desenho:** fixado antes no commit `4e287e3`.
- **Dados:** BTC-USDT spot da KuCoin, de 04/07 a 21/10/2026, 76 operações.

| Taxa por lado | Total | Queda máx. |
| --- | --- | --- |
| 0,1% (spot) | -4,8% | 10,4% |
| 0,02% | +7,5% | 7,9% |
| 0% | +10,8% | 7,3% |
| segurar BTC | +34,4% | 6,9% |

O efeito não se repetiu. Mesmo sem taxa, a regra rendeu um terço do que rendeu segurar, e o teste da sorte deu p de
0,898. Neste trecho, a maior parte da alta (cerca de +21%) veio com a bolsa americana aberta, o contrário do efeito
publicado. Com a taxa do spot, a regra perdeu dinheiro.

### Por que procurar um vale menor não resolve

A taxa é fixa por operação (0,2% na ida e volta do spot), e o movimento encolhe junto com o candle. Contas de 10/2018 a
06/2026, com candles da Binance:

| Candle | Movimento médio de um candle | Acerto de direção necessário para empatar com a taxa |
| --- | --- | --- |
| 5 min | 0,12% | impossível: mesmo acertando todos, perde |
| 15 min | 0,20% | 99% |
| 1 h | 0,40% | 75% |
| 4 h | 0,81% | 62% |
| 1 dia | 2,1% a 2,3% | 52% |

Os modelos testados acertam a direção de 50% a 52% das vezes, o que só basta no diário; por isso só o diário
funcionou. Esse acerto de direção é outra coisa que o acerto dos ciclos, que sai alto com alvo pequeno e stop longe.

A regra "comprar quando o candle fecha abaixo da abertura do anterior (ou de dois antes)" aparece em cerca de metade
dos candles. Depois dela, o preço sobe em média de 0,003% (5 min) a 0,03% (1 h e 4 h) nos candles seguintes: de 7 a 70
vezes menos que a taxa.

### O vale no candle diário com alvo

`python -m wescalp.dailydip --file ...` testa a mesma regra no candle diário. O desenho foi registrado antes da rodada
(`bbaecf3`):

- **Compra:** no fechamento do dia que fecha abaixo da abertura do anterior (ou de dois antes), com o capital inteiro.
- **Venda:** no alvo de 2%, 3% ou 5% sobre a compra, conferido nos candles de 5 min.
- **Stop:** duas versões, sem stop e com stop 3 alvos abaixo da compra.

Período de 10/2018 a 06/2026, com 0,1% por lado. Resultados com o dia anterior; com dois dias antes, ficam parecidos:

| Saída | Alvo | Operações/ano | Acerto | Ao ano | Queda máx. | Sharpe | p da sorte |
| --- | --- | --- | --- | --- | --- | --- | --- |
| só alvo | 2% | 10,9 | 99% | 9,9% | 75% | 0,46 | 0,972 |
| só alvo | 3% | 7,9 | 98% | 12,7% | 76% | 0,50 | 0,957 |
| só alvo | 5% | 5,6 | 98% | 17,4% | 76% | 0,57 | 0,924 |
| alvo e stop | 2% | 88,5 | 74% | -25,5% | 90% | -0,68 | 0,860 |
| alvo e stop | 3% | 57,1 | 74% | -24,4% | 89% | -0,46 | 0,920 |
| alvo e stop | 5% | 31,9 | 77% | -6,3% | 80% | 0,13 | 0,882 |
| segurar BTC | — | — | — | 32,6% | 77% | 0,77 | — |
| congelada 50/100/150 | — | — | — | 54,9% | 44% | 1,25 | — |

Nenhuma variante passa.

- **Só com alvo:** quase toda operação fecha no lucro (98% a 99%), mas rende de um terço a pouco mais da metade de
  segurar. Ela vende cedo nas altas e espera um novo vale, e nas quedas fica presa. A operação mais longa comprou em
  10/11/2021 a 64.882 (perto do topo) e só bateu os +5% em 04/03/2024, 845 dias depois. A queda máxima é a mesma de
  segurar.
- **Com stop:** as perdas no stop comem os ganhos, de -6% a -37% ao ano.
- **Dias de sinal:** não são melhores que dias ao acaso (p de 0,86 a 1,0).

## Moedas mais voláteis

`python -m wescalp.altcoins` testa ETH, SOL, XRP, DOGE e ADA contra USDT. Os candles vêm de
`wescalp.binance.fetch(..., symbol=...)`. As cinco estão na Classe A da KuCoin, com a mesma taxa do BTC (0,1% por lado,
conferida na API pública). O desenho foi registrado antes da rodada (`8a8e1cc`). Período de 01/2021 a 06/2026. Ficam
de fora as moedas que morreram no caminho, o que favorece quem só compra.

**1. Movimento médio de um candle e acerto de direção necessário para empatar com a ida e volta de 0,2%:**

| Moeda | 15 min | 1 h | 4 h | 1 dia |
| --- | --- | --- | --- | --- |
| BTC | 0,20% / impossível | 0,39% / 76% | 0,80% / 63% | 2,08% / 52% |
| ETH | 0,26% / 89% | 0,52% / 69% | 1,06% / 59% | 2,81% / 52% |
| SOL | 0,40% / 74% | 0,79% / 62% | 1,60% / 55% | 4,03% / 48% |
| XRP | 0,32% / 81% | 0,62% / 66% | 1,24% / 57% | 3,16% / 49% |
| DOGE | 0,39% / 75% | 0,75% / 63% | 1,50% / 55% | 4,07% / 45% |
| ADA | 0,36% / 77% | 0,70% / 64% | 1,37% / 56% | 3,45% / 50% |

A volatilidade melhora a conta: no candle de 1 h, a SOL pede 62% de acerto, contra 76% do BTC. Ainda assim, fica
acima dos 50% a 52% que os modelos acertam.

**2. Previsão no curto prazo:**

- **Sinais:** de preço e volume, de hora em hora.
- **Horas testadas:** todas e só as de volatilidade alta.
- **Treino e teste:** lado e corte saem de 2021–2023; a medida é fora da amostra, de 2024 a 06/2026.
- **Régua:** mais que 0,2% por operação com t ≥ 3,4.

**Nenhum passa.** As altcoins mostram de novo o padrão de comprar depois de queda forte, agora com movimento acima da
taxa: depois da pior queda de 24 h, de +0,5% a +2,0% nas 12 h seguintes. Os casos são poucos e o ruído é grande, com t
de 1,0 a 2,5. O mais perto da régua foi o volume alto na DOGE: +1,67% nas 12 h seguintes, em 122 casos, com t de 3,3.

**3. Tendência diária** (a congelada 50/100/150), de 07/2021 a 06/2026, 0,1% por lado:

| Moeda | Congelada: ao ano / queda / Sharpe | Segurar: ao ano / queda / Sharpe | p da sorte |
| --- | --- | --- | --- |
| BTC | 22,6% / 44% / 0,78 | 10,8% / 77% / 0,46 | 0,050 |
| ETH | 17,2% / 52% / 0,59 | -7,2% / 79% / 0,25 | 0,064 |
| SOL | 47,8% / 69% / 0,93 | 15,7% / 96% / 0,64 | 0,085 |
| XRP | -3,3% / 80% / 0,20 | 8,0% / 78% / 0,49 | 0,749 |
| DOGE | -5,7% / 75% / 0,21 | -22,3% / 85% / 0,18 | 0,356 |
| ADA | -2,9% / 71% / 0,19 | -36,4% / 95% / -0,08 | 0,170 |
| cesta das 6 | 18,0% / 51% / 0,63 | 2,0% / 82% / 0,38 | — |

- **A cesta passa na régua fixada:** com 1/6 do capital em cada moeda, a congelada rendeu 18% ao ano, contra 2% de
  segurar a cesta, e caiu 51% contra 82%. Mas isso depende da janela: em janelas de 1 e 2 anos, ela perdeu de segurar
  na maioria das entradas (ver "Comparar com segurar depende do dia de entrada").
- **Nenhuma moeda sozinha passa:** BTC, ETH e SOL ficaram com p de 0,05 a 0,085, acima do limite de 0,0083 para seis
  moedas.
- **Na XRP a regra perdeu para segurar.**

## Ideias do day-trader-skill

O [day-trader-skill](https://github.com/Jakeschincariol/day-trader-skill) (e o [trader-skill](https://github.com/Jakeschincariol/trader-skill),
do mesmo autor) é um pacote de skills do Claude para ações americanas. Não traz uma estratégia que ganha, e sim
ferramentas de disciplina: scanner, entrada, saída, simulação, tamanho, trava de risco e aprovação humana antes de
cada ordem. No exemplo do próprio autor, um rompimento da Tesla, a simulação e o histórico deram cara ou coroa. Três
ideias vieram para cá: o rompimento da faixa de abertura, que foi testado; o tamanho pela perda máxima com a trava
de risco; e as chances antes de cada sinal, no painel.

### Rompimento da faixa de abertura (ORB)

`python -m wescalp.orb --file ...` testa no BTC o setup principal do pacote, com a abertura da bolsa de Nova York no
lugar da abertura da ação. O desenho foi registrado antes da rodada (`8ce45c8`):

- **Faixa:** máxima e mínima dos primeiros 15 ou 30 min depois das 9h30 de Nova York, em cada dia de pregão.
- **Entrada:** só compra, com o capital inteiro, no primeiro candle de 5 min que fechar acima da faixa, até as 12h de
  Nova York.
- **Saída:** stop no meio da faixa e alvo de 1R ou 2R (R = entrada - stop); se nada bater, venda às 16h.

Primeira rodada (25/10, de 10/2018 a 06/2026):

| Variante | Operações | Acerto | Ao ano com 0,1% | Com 0,02% | Com 0% | Queda máx. | Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| faixa 15 min, alvo 1R | 1.239 | 43% | -28,1% | -7,1% | -0,9% | 92% | -3,85 |
| faixa 15 min, alvo 2R | 1.239 | 35% | -27,5% | -6,3% | -0,0% | 92% | -2,95 |
| faixa 30 min, alvo 1R | 1.039 | 46% | -24,7% | -6,6% | -1,4% | 89% | -3,23 |
| faixa 30 min, alvo 2R | 1.039 | 35% | -22,0% | -3,2% | +2,1% | 85% | -2,19 |
| segurar BTC | — | — | +32,6% | — | — | 77% | 0,77 |

Nenhuma variante passa, e a prova única não rodou.

- **Sem taxa, o rompimento rende perto de zero:** de -1,4% a +2,1% ao ano. Com a taxa do spot, perde de 22% a 28%.
- **O teste da sorte saiu viesado a favor dos sorteios.** Ele punha a entrada num horário sorteado da janela, mas só
  nos dias em que houve rompimento. Assim, muitos sorteios entravam antes da alta, num dia que já se sabia que ia
  romper. O p de 1,000 não mede o setup; o veredito vem do resultado em si.

### Tamanho pela perda máxima e trava de risco

`wescalp/guard.py` é a camada de execução de uma fase futura, inspirada no trader-budget e no trader-risk do pacote. É
só conta: nada envia ordem nem usa chave.

- **Tamanho (`size`):** a quantidade de BTC para que a perda no stop, com a taxa de compra e a de venda, não passe da
  perda máxima. Respeita um teto por posição e as regras do par BTC-USDT spot da KuCoin, lidas da API pública em
  25/10/2026:
  - mínimo de 0,00001 BTC e 0,1 USDT;
  - incremento de 0,00000001 BTC;
  - tick de 0,1 USDT.

  A quantidade é arredondada para baixo e o stop vai para o tick de baixo.
- **Trava (`check`):** passa ou bloqueia uma ordem, com os motivos, em sete regras:
  - perda por operação;
  - perda do dia, já com o stop da ordem nova;
  - operações por dia;
  - relação alvo/risco;
  - teto da posição;
  - caixa, sem alavancagem;
  - regras do par.

  Venda só é barrada se for a descoberto ou fora das regras do par.
- **Stop da congelada:** ela sai no fechamento abaixo da média, então o risco de cada parte é a distância até a média
  mais 1 ATR diário de 14 dias de folga.

Pesquisa com o desenho registrado antes da rodada (`7748ecf`), rodada uma vez em 25/10, de 10/2018 a 06/2026, 0,1% por
lado. Cada parte da congelada entra com o menor entre 1/3 do capital e o tamanho pela perda máxima, e fica assim até
sair:

| Variante | Ao ano | Queda máx. | Sharpe | Exposição | Saídas que passaram de 1% / 2% | Pior saída |
| --- | --- | --- | --- | --- | --- | --- |
| congelada (1/3 por parte) | 54,9% | 44% | 1,25 | 55% | 50 / 20 | -5,0% |
| perda máx. 1% por parte | 37,1% | 24% | 1,36 | 32% | 16 / 0 | -1,9% |
| perda máx. 2% por parte | 51,5% | 39% | 1,31 | 49% | 41 / 12 | -3,4% |

- **O que muda:** a perda máxima por parte troca retorno por queda. Com 1%, a carteira fica menos da metade do tempo
  comprada, cai 24% em vez de 44% e o Sharpe sobe um pouco, mas rende 37% ao ano em vez de 55%. Com 2%, quase não muda.
- **O limite não é garantido:** a congelada sai no fechamento, que pode cair mais de 1 ATR abaixo da média, e a média
  anda depois da entrada. No limite de 1%, 16 das 117 saídas com perda passaram dele, a pior em -1,9%.
- **Nada foi promovido:** não houve teste da sorte nem de robustez para isso. É a régua que a fase de enviar ordens
  usaria.

### Chances antes de cada sinal

A "máquina do tempo" do pacote mostra, antes de cada operação, o que o mesmo setup fez nas vezes anteriores. Aqui ela
olha os sinais da congelada: a virada de uma das médias no fechamento das 00h UTC, quando o fechamento passa para
cima da média (compra daquele terço) ou para baixo (venda). `python -m wescalp.odds --file ...` conta as viradas nos
candles diários da Binance de 17/08/2017 a 30/06/2026, com 0,1% por lado, e grava `wescalp/static/odds.json`. O
primeiro dia de cada média não conta.

| Compra | Vezes | 30 dias depois, acima | Variação mediana em 30 dias | Operações no lucro | Ganho mediano | Perda mediana | Duração mediana |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M50 ↑ | 84 | 44% | -1,4% | 23% | +19,2% | -2,6% | 4 dias |
| M100 ↑ | 57 | 53% | +2,1% | 19% | +28,4% | -2,6% | 6 dias |
| M150 ↑ | 35 | 46% | -3,7% | 23% | +36,5% | -3,6% | 7 dias |

A operação vai da compra até a venda seguinte da mesma média.

| Venda | Vezes | 30 dias depois, acima | Variação mediana em 30 dias | Recompra mais barata | Recompra mediana | Dias até a recompra |
| --- | --- | --- | --- | --- | --- | --- |
| M50 ↓ | 85 | 42% | -1,9% | 24% | 1,5% mais cara | 8 |
| M100 ↓ | 58 | 44% | -2,1% | 25% | 2,2% mais cara | 2 |
| M150 ↓ | 36 | 33% | -7,6% | 23% | 3,3% mais cara | 5 |

- **A virada não diz para onde o preço vai:** 30 dias depois de uma compra, o BTC estava acima em 44% a 53% das
  vezes; depois de uma venda, em 33% a 44%.
- **O ganho vem da assimetria:** só 19% a 23% das compras fecharam no lucro, mas o ganho mediano é de 7 a 11 vezes a
  perda mediana. A operação típica é um vai e volta de 4 a 7 dias.
- **Só 1 em cada 4 vendas deu recompra mais barata.** Em mediana, a recompra saiu de 1,5% a 3,3% mais cara. A venda
  serve para cortar a perda, que fica pequena nas compras que dão errado.
- **No painel:** a tabela de ordens da aba Estratégia tem a coluna "no histórico". Passar o mouse sobre o dia de uma
  ordem mostra o resumo na legenda do gráfico; clicar numa seta ou numa linha da tabela abre o texto completo no
  quadro abaixo do gráfico, que começa no último sinal. No aprendiz, só as viradas de 50, 100 e 150 dias têm
  histórico; as outras mostram um traço.
- **É histórico, não previsão:** é o mesmo período em que a congelada foi escolhida, então não é prova nova dela.

## Carteiras simuladas ao vivo

O serviço `paper` (`wescalp/paper.py`, Python puro, sem numpy) aplica duas regras fixadas antes aos candles do
BTC-USDT spot que o serviço `candles` grava, e registra tudo em `paper_meta`, `paper_days` e `paper_orders`:

- **congelada** (controle): 1/3 do capital em cada média diária de 50, 100 e 150 dias;
- **aprendiz** (experimento): as 13 médias de 50 a 170 dias pesadas pelo que os sinais de cada uma renderam (o
  Hedge de `wescalp.learn`), ajustando só quando a posição desejada se afasta 5 p.p.

Decisão no fechamento das 00h UTC (21h em Brasília), valendo para o dia seguinte, com 0,1% por compra e venda. Na
primeira execução, o histórico disponível no banco é reconstruído com a mesma regra (`live = false`) e o dia vira o
início do período ao vivo; daí em diante, cada dia é decidido quando fecha e gravado sem reescrita. Um teste confere
que o motor dá exatamente as posições e os pesos da pesquisa (`refine`, `learn`). A aba **Estratégia** do painel
mostra o gráfico diário com as médias e as ordens (simuladas claras, ao vivo fortes), a curva das duas carteiras
contra segurar o BTC, os pesos que o aprendiz está dando a cada média e a tabela de ordens, com o que o histórico diz
de cada sinal (ver "Chances antes de cada sinal").

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
