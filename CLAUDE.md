# Wescalp — instruções para agentes

- Bot de scalping de BTC na KuCoin (perpétuo `XBTUSDTM`, 1 contrato = 0,001 BTC, tick 0,1, taxa 0,02% maker / 0,06% taker no nível básico).
- Fase atual: **0, só dados.** Nenhum código envia ordem. Qualquer código que use chave de API privada ou envie ordem precisa de aprovação explícita do Gui.
- Decisão de entrada/saída é sempre código determinístico; o Gemini só comenta, veta ou revisa (ver o documento de pesquisa no README).
- Hospedagem: VPS Hostinger `82.25.76.109`, junto com Livegenda/emqhero; domínio `wescalp.avadata.com.br` pelo Caddy compartilhado (`/opt/livegenda/app/Caddyfile`, nunca `sed -i` nele). Limite a memória dos containers.
- Testes: `.venv/bin/pytest -q`. Os parsers da KuCoin têm armadilhas (spot vem open, close, high, low; futuros em ms; negócios em ns) — cubra com teste qualquer parser novo.
