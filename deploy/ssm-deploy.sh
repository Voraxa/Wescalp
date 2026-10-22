#!/usr/bin/env bash
# Roda na sessão de desenvolvimento: empacota o commit atual e instala na VPS via AWS SSM.
#   AWS_EC2_ACCESS_KEY_ID=... AWS_EC2_SECRET_ACCESS_KEY=... deploy/ssm-deploy.sh
# A VPS não tem acesso ao GitHub privado, então o código vai junto nos comandos. O SSM recusa comando
# acima de 97 KB, então o pacote sobe em partes para um arquivo na VPS, conferido por sha256 antes de instalar.
set -euo pipefail
INSTANCE=mi-0a0e856e4884e0559
export AWS_DEFAULT_REGION=sa-east-1
CHUNK=60000                         # bytes de base64 por comando, com folga para o limite de 97 KB
STAGE=/root/wescalp-deploy.b64      # pacote remontado na VPS
# Os segredos das sessões vêm com aspas em volta; sem tirar, a AWS responde "token inválido".
strip_quotes() { local v="${1%\"}"; printf '%s' "${v#\"}"; }
export AWS_ACCESS_KEY_ID="$(strip_quotes "${AWS_EC2_ACCESS_KEY_ID:?falta AWS_EC2_ACCESS_KEY_ID}")"
export AWS_SECRET_ACCESS_KEY="$(strip_quotes "${AWS_EC2_SECRET_ACCESS_KEY:?falta AWS_EC2_SECRET_ACCESS_KEY}")"
unset AWS_SESSION_TOKEN

cd "$(git rev-parse --show-toplevel)"
SHA=$(git rev-parse --short HEAD)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git archive --format=tar.gz HEAD | base64 -w0 > "$WORK/payload.b64"
SUM=$(sha256sum < "$WORK/payload.b64" | cut -d' ' -f1)
split -b "$CHUNK" -d -a 3 "$WORK/payload.b64" "$WORK/part."

# Envia um comando (linhas de shell em arquivo) e espera. O AWS-RunShellScript roda com sh, não bash.
# Não usa "aws ssm wait": ele desiste em 100 s, menos que o docker compose build.
ssm_run() {
  local file=$1 comment=$2 timeout=$3 params id status
  params=$(python3 - "$file" "$timeout" <<'PY'
import json, sys
cmds = open(sys.argv[1]).read().splitlines()
print(json.dumps({"commands": cmds, "executionTimeout": [sys.argv[2]]}))
PY
)
  id=$(aws ssm send-command --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
    --comment "$comment" --parameters "$params" --query Command.CommandId --output text)
  while :; do
    status=$(aws ssm get-command-invocation --command-id "$id" --instance-id "$INSTANCE" \
      --query Status --output text 2>/dev/null || echo Pending)
    case $status in Pending|InProgress|Delayed) sleep 5 ;; *) break ;; esac
  done
  aws ssm get-command-invocation --command-id "$id" --instance-id "$INSTANCE" \
    --query '[StandardOutputContent,StandardErrorContent]' --output text
  echo "$comment: $status"
  [ "$status" = Success ]
}

parts=("$WORK"/part.*)
for i in "${!parts[@]}"; do
  {
    echo "set -eu"
    [ "$i" -eq 0 ] && echo "rm -f $STAGE"
    # base64 não tem aspas simples, então a parte vai literal entre elas.
    echo "printf '%s' '$(cat "${parts[$i]}")' >> $STAGE"
    echo "wc -c < $STAGE"
  } > "$WORK/cmd.sh"
  if ! out=$(ssm_run "$WORK/cmd.sh" "wescalp deploy $SHA parte $((i + 1))/${#parts[@]}" 120); then
    echo "$out" >&2
    exit 1
  fi
  echo "parte $((i + 1))/${#parts[@]} enviada"
done

cat > "$WORK/cmd.sh" <<EOF
set -eu
echo "$SUM  $STAGE" | sha256sum -c -
mkdir -p /opt/wescalp
# Mantém o .env (senhas) e troca o resto do código.
find /opt/wescalp -mindepth 1 -maxdepth 1 ! -name .env -exec rm -rf {} +
base64 -d $STAGE | tar -xz -C /opt/wescalp
rm -f $STAGE
echo $SHA > /opt/wescalp/REVISION
bash /opt/wescalp/deploy/remote-install.sh
EOF
ssm_run "$WORK/cmd.sh" "wescalp deploy $SHA" 1800
