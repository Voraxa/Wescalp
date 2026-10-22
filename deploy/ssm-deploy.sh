#!/usr/bin/env bash
# Roda na sessão de desenvolvimento: empacota o commit atual e instala na VPS via AWS SSM.
#   AWS_EC2_ACCESS_KEY_ID=... AWS_EC2_SECRET_ACCESS_KEY=... deploy/ssm-deploy.sh
# A VPS não tem acesso ao GitHub privado, então o código vai junto no comando (o pacote tem poucos KB).
set -euo pipefail
INSTANCE=mi-0a0e856e4884e0559
export AWS_DEFAULT_REGION=sa-east-1
# Os segredos das sessões vêm com aspas em volta; sem tirar, a AWS responde "token inválido".
strip_quotes() { local v="${1%\"}"; printf '%s' "${v#\"}"; }
export AWS_ACCESS_KEY_ID="$(strip_quotes "${AWS_EC2_ACCESS_KEY_ID:?falta AWS_EC2_ACCESS_KEY_ID}")"
export AWS_SECRET_ACCESS_KEY="$(strip_quotes "${AWS_EC2_SECRET_ACCESS_KEY:?falta AWS_EC2_SECRET_ACCESS_KEY}")"
unset AWS_SESSION_TOKEN

cd "$(git rev-parse --show-toplevel)"
SHA=$(git rev-parse --short HEAD)
PAYLOAD=$(git archive --format=tar.gz HEAD | base64 -w0)

PARAMS=$(python3 - "$PAYLOAD" "$SHA" <<'PY'
import json, sys
payload, sha = sys.argv[1], sys.argv[2]
cmds = [
    "set -euo pipefail",
    "mkdir -p /opt/wescalp",
    # Mantém o .env (senha do banco) e troca o resto do código.
    "find /opt/wescalp -mindepth 1 -maxdepth 1 ! -name .env -exec rm -rf {} +",
    f"echo '{payload}' | base64 -d | tar -xz -C /opt/wescalp",
    f"echo {sha} > /opt/wescalp/REVISION",
    "bash /opt/wescalp/deploy/remote-install.sh",
]
print(json.dumps({"commands": cmds, "executionTimeout": ["1800"]}))
PY
)

CMD_ID=$(aws ssm send-command --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
  --comment "wescalp deploy $SHA" --parameters "$PARAMS" --query Command.CommandId --output text)
echo "comando $CMD_ID enviado; aguardando..."
aws ssm wait command-executed --command-id "$CMD_ID" --instance-id "$INSTANCE" || true
aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE" \
  --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
