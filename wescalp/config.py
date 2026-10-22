import os

SPOT_REST = "https://api.kucoin.com"
FUTURES_REST = "https://api-futures.kucoin.com"

SPOT_SYMBOL = os.environ.get("WESCALP_SPOT_SYMBOL", "BTC-USDT")
FUTURES_SYMBOL = os.environ.get("WESCALP_FUTURES_SYMBOL", "XBTUSDTM")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://wescalp:wescalp@localhost:5432/wescalp")

# Intervalo entre fotos completas do livro (50 níveis). O topo do livro é gravado a cada atualização.
BOOK_SNAPSHOT_MS = int(os.environ.get("WESCALP_BOOK_SNAPSHOT_MS", "1000"))

# Arquivo tocado a cada mensagem recebida; o healthcheck do Docker olha a idade dele.
HEARTBEAT_FILE = os.environ.get("WESCALP_HEARTBEAT_FILE", "/tmp/wescalp-collector.heartbeat")

CANDLE_SECONDS = 300
