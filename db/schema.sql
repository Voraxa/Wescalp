-- Wescalp: dados de mercado (Fase 0).
-- Tabelas em Postgres puro; db/timescale.sql transforma em hypertables quando o TimescaleDB existe.

-- Candles de 5 min. source = 'spot' (BTC-USDT) ou 'futures' (XBTUSDTM); a API guarda ~2 anos de cada.
CREATE TABLE IF NOT EXISTS candles (
    source    text             NOT NULL,
    symbol    text             NOT NULL,
    ts        timestamptz      NOT NULL,  -- abertura do candle
    open      double precision NOT NULL,
    high      double precision NOT NULL,
    low       double precision NOT NULL,
    close     double precision NOT NULL,
    volume    double precision NOT NULL,  -- spot: BTC; futures: contratos
    turnover  double precision NOT NULL,  -- em USDT
    PRIMARY KEY (source, symbol, ts)
);

-- Cada negócio executado no perpétuo (topic /contractMarket/execution).
CREATE TABLE IF NOT EXISTS trades (
    symbol    text             NOT NULL,
    ts        timestamptz      NOT NULL,
    trade_id  bigint           NOT NULL,
    side      text             NOT NULL,  -- lado do agressor (taker): buy | sell
    price     double precision NOT NULL,
    size      double precision NOT NULL,  -- contratos
    PRIMARY KEY (symbol, trade_id, ts)
);

-- Topo do livro a cada atualização (~100 ms): base do OFI.
CREATE TABLE IF NOT EXISTS book_l1 (
    symbol    text             NOT NULL,
    ts        timestamptz      NOT NULL,
    seq       bigint           NOT NULL,
    bid_px    double precision NOT NULL,
    bid_sz    double precision NOT NULL,
    ask_px    double precision NOT NULL,
    ask_sz    double precision NOT NULL,
    bid_sz5   double precision NOT NULL,  -- soma dos 5 primeiros níveis
    ask_sz5   double precision NOT NULL,
    PRIMARY KEY (symbol, ts, seq)
);

-- Foto das 50 melhores ofertas de cada lado, no intervalo de BOOK_SNAPSHOT_MS (padrão 1 s).
CREATE TABLE IF NOT EXISTS book_snapshots (
    symbol    text               NOT NULL,
    ts        timestamptz        NOT NULL,
    seq       bigint             NOT NULL,
    bid_px    double precision[] NOT NULL,
    bid_sz    double precision[] NOT NULL,
    ask_px    double precision[] NOT NULL,
    ask_sz    double precision[] NOT NULL,
    PRIMARY KEY (symbol, ts, seq)
);

-- Rótulos de tripla barreira (wescalp.labels): o que teria acontecido com uma compra e uma venda no fechamento
-- de cada candle de 5 min do perpétuo. config descreve as barreiras, ex.: tp1.5-sl1-t12-atr14.
CREATE TABLE IF NOT EXISTS labels (
    config     text             NOT NULL,
    symbol     text             NOT NULL,
    ts         timestamptz      NOT NULL,  -- abertura do candle em cujo fechamento a entrada acontece
    side       text             NOT NULL,  -- long | short
    outcome    text             NOT NULL,  -- tp | sl | time
    bars       smallint         NOT NULL,  -- candles até a saída
    entry_px   double precision NOT NULL,
    exit_px    double precision NOT NULL,
    atr        double precision NOT NULL,
    ret_gross  double precision NOT NULL,  -- fração do preço de entrada, antes das taxas
    ret_net    double precision NOT NULL,  -- depois das taxas maker/taker
    PRIMARY KEY (config, symbol, side, ts)
);
