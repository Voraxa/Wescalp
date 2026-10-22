-- Aplicado só quando a extensão TimescaleDB está disponível (na VPS, sempre).
CREATE EXTENSION IF NOT EXISTS timescaledb;

SELECT create_hypertable('candles', 'ts', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '30 days');
SELECT create_hypertable('trades', 'ts', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('book_l1', 'ts', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '1 day');
SELECT create_hypertable('book_snapshots', 'ts', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '1 day');

-- Compressão: dado de mais de 1 dia é só leitura.
ALTER TABLE trades SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
ALTER TABLE book_l1 SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
ALTER TABLE book_snapshots SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
SELECT add_compression_policy('trades', INTERVAL '1 day', if_not_exists => TRUE);
SELECT add_compression_policy('book_l1', INTERVAL '1 day', if_not_exists => TRUE);
SELECT add_compression_policy('book_snapshots', INTERVAL '1 day', if_not_exists => TRUE);
