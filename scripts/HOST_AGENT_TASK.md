# Task for Host Agent: Fix aegis proxy for container access

## Problem
The aegis MITM proxy runs on port 8444 but only accepts connections from
127.0.0.1 (localhost). Docker containers connect via host.docker.internal
(192.168.65.254 on Docker Desktop for Mac), which is a different interface.

The container can TCP-connect to port 8444 but gets 401 — the proxy either
doesn't inject API keys for non-localhost clients, or doesn't bind to the
Docker bridge.

## What needs to happen

Look at the hermes-aegis codebase at ~/Projects/hermes-aegis to understand:

1. **How the MITM proxy binds** — does it bind to 127.0.0.1 or 0.0.0.0?
   If 127.0.0.1, Docker containers can't use it. It needs to bind to
   0.0.0.0 or at least to the Docker bridge interface.

2. **How API key injection works** — does it check the source IP? The vault
   injects x-api-key headers into outbound Anthropic requests. If it only
   does this for localhost clients, container requests get sent without keys.

3. **Fix one or both** so that container processes can:
   ```bash
   curl --proxy http://host.docker.internal:8444 \
        --cacert /certs/mitmproxy-ca-cert.pem \
        https://api.anthropic.com/v1/messages
   ```
   and have their API key injected by the vault.

## How to verify

From inside the container (or with docker exec):
```bash
python3 /workspace/Projects/hermes-agent/scripts/test_llm_contradiction.py
```

Should show "CONTRADICTS" responses instead of 401 errors.

## Context

- Aegis codebase: ~/Projects/hermes-aegis
- Proxy config: ~/.hermes-aegis/proxy-config.json
- Proxy PID/port: ~/.hermes-aegis/proxy.pid (currently port 8444)
- Proxy log: ~/.hermes-aegis/proxy.log
- Vault: ~/.hermes-aegis/vault.enc (has ANTHROPIC_API_KEY)
- Container cert: /certs/mitmproxy-ca-cert.pem (mounted from ~/.mitmproxy/)

## After fix

Once the proxy works from containers, the container agent will run:
```bash
cd /workspace/Projects/hermes-agent
.venv/bin/python3 scripts/test_llm_contradiction.py
.venv/bin/python3 -m benchmarks.runner --backend cognitive --suite all \
  --runs 1 --seeds 42 --contradiction-llm claude-haiku-4-5
```
