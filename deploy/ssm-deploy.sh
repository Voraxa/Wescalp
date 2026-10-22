#!/usr/bin/env bash
# Roda na sessão de desenvolvimento: empacota o commit atual e instala na VPS via AWS SSM.
#   AWS_EC2_ACCESS_KEY_ID=... AWS_EC2_SECRET_ACCESS_KEY=... deploy/ssm-deploy.sh
# A VPS não tem acesso ao GitHub privado, então o código vai junto nos comandos. Cada comando do SSM aceita
# ~97 KB, por isso o pacote vai em partes, remontado e conferido por sha256 na VPS antes de instalar.
set -euo pipefail
INSTANCE=mi-0a0e856e4884e0559
CHUNK=60000  # caracteres de base64 por comando, com folga sob o limite do SSM
export AWS_DEFAULT_REGION=sa-east-1
# Os segredos das sessões vêm com aspas em volta; sem tirar, a AWS responde "token inválido".
strip_quotes() { local v="${1%\"}"; printf '%s' "${v#\"}"; }
export AWS_ACCESS_KEY_ID="$(strip_quotes "${AWS_EC2_ACCESS_KEY_ID:?falta AWS_EC2_ACCESS_KEY_ID}")"
export AWS_SECRET_ACCESS_KEY="$(strip_quotes "${AWS_EC2_SECRET_ACCESS_KEY:?falta AWS_EC2_SECRET_ACCESS_KEY}")"
unset AWS_SESSION_TOKEN

cd "$(git rev-parse --show-toplevel)"
SHA=$(git rev-parse --short HEAD)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git archive --format=tar.gz HEAD > "$WORK/pkg.tgz"
SUM=$(sha256sum "$WORK/pkg.tgz" | cut -d' ' -f1)
base64 -w0 "$WORK/pkg.tgz" | split -b "$CHUNK" - "$WORK/part-"
REMOTE=/tmp/wescalp-deploy-$SHA.b64

# Manda uma lista de comandos shell (JSON) e espera; falha se a VPS não responder Success.
run() {
  local params="$1" comment="$2" id status
  id=$(aws ssm send-command --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
    --comment "$comment" --parameters "$params" --query Command.CommandId --output text)
  # Sem "aws ssm wait": ele desiste em 100 s, menos que um docker compose build.
  while :; do
    status=$(aws ssm get-command-invocation --command-id "$id" --instance-id "$INSTANCE" \
      --query Status --output text 2>/dev/null || echo Pending)
    case $status in Pending|InProgress|Delayed) sleep 5 ;; *) break ;; esac
  done
  aws ssm get-command-invocation --command-id "$id" --instance-id "$INSTANCE" \
    --query '[StandardOutputContent,StandardErrorContent]' --output text
  [ "$status" = Success ] || { echo "comando '$comment' terminou em $status" >&2; exit 1; }
}

json() { python3 -c 'import json, sys; print(json.dumps({"commands": sys.argv[2:], "executionTimeout": [sys.argv[1]]}))' "$@"; }

parts=("$WORK"/part-*)
echo "pacote $SHA: $(wc -c < "$WORK/pkg.tgz") bytes em ${#parts[@]} parte(s)"
op='>'
for i in "${!parts[@]}"; do
  run "$(json 120 "set -euo pipefail" "printf '%s' '$(cat "${parts[$i]}")' $op $REMOTE")" "wescalp deploy $SHA parte $((i + 1))"
  op='>>'
done

run "$(json 1800 \
  "set -euo pipefail" \
  "base64 -d $REMOTE > /tmp/wescalp-deploy-$SHA.tgz" \
  "echo '$SUM  /tmp/wescalp-deploy-$SHA.tgz' | sha256sum -c -" \
  "mkdir -p /opt/wescalp" \
  "find /opt/wescalp -mindepth 1 -maxdepth 1 ! -name .env -exec rm -rf {} +" \
  "tar -xzf /tmp/wescalp-deploy-$SHA.tgz -C /opt/wescalp" \
  "rm -f $REMOTE /tmp/wescalp-deploy-$SHA.tgz" \
  "echo $SHA > /opt/wescalp/REVISION" \
  "bash /opt/wescalp/deploy/remote-install.sh")" "wescalp deploy $SHA instalar"
