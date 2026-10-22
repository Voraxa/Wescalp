#!/usr/bin/env bash
# Roda NA VPS (root). Espera o código já extraído em /opt/wescalp.
# Sobe todos os serviços fora do perfil tools (banco, coletor, candles e painel). O serviço candles
# completa sozinho as 100 semanas de histórico e depois segue buscando os candles novos.
set -euo pipefail
cd /opt/wescalp

if [ ! -f .env ]; then
  umask 077
  echo "WESCALP_DB_PASSWORD=$(openssl rand -hex 24)" > .env
  echo ".env criado com senha nova"
fi

docker compose build
# --remove-orphans tira serviços que saíram do compose (ex.: o vigia com alerta).
docker compose up -d --remove-orphans

sleep 20
docker compose ps
docker compose logs --tail 20 collector
