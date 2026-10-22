#!/usr/bin/env bash
# Roda NA VPS (root). Espera o código já extraído em /opt/wescalp.
# Sobe banco + coletor e dispara o backfill das 100 semanas em segundo plano.
set -euo pipefail
cd /opt/wescalp

if [ ! -f .env ]; then
  umask 077
  echo "WESCALP_DB_PASSWORD=$(openssl rand -hex 24)" > .env
  echo ".env criado com senha nova"
fi

docker compose build
docker compose up -d db collector

# Backfill idempotente: retoma de onde parou se rodar de novo.
mkdir -p /var/log
nohup docker compose run -T --rm backfill --weeks 100 < /dev/null > /var/log/wescalp-backfill.log 2>&1 &
echo "backfill disparado (log em /var/log/wescalp-backfill.log)"

sleep 20
docker compose ps
docker compose logs --tail 20 collector
