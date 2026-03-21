# Host-side setup for LLM benchmark runs

Run these on the Mac (outside Docker):

## 1. Start hermes-aegis proxy

```bash
cd ~/Projects/hermes-aegis
hermes-aegis run
```

Should show: `Proxy listening on 127.0.0.1:8443` and `Vault 1 API keys protected.`

If the vault doesn't have the Anthropic key yet:

```bash
hermes-aegis vault set ANTHROPIC_API_KEY <your-key>
```

## 2. Bump Docker memory (optional but recommended)

Docker Desktop → Settings → Resources → Memory → set to 16GB → Apply & Restart

This prevents OOM when running all benchmark suites together.

## 3. Verify from container

Once aegis is running, the container agent can verify with:

```bash
curl -s -o /dev/null -w "%{http_code}" \
  --proxy http://host.docker.internal:8443 \
  --cacert /certs/mitmproxy-ca-cert.pem \
  https://api.anthropic.com/v1/messages
```

Should return `401` (auth required) — NOT `000` (connection refused).

## 4. Then the container agent runs

```bash
cd /workspace/Projects/hermes-agent

# Test LLM contradiction detection
.venv/bin/python3 scripts/test_llm_contradiction.py

# Full benchmark with LLM contradictions
.venv/bin/python3 -m benchmarks.runner \
  --backend cognitive \
  --suite all \
  --runs 1 \
  --seeds 42 \
  --judge-model heuristic \
  --contradiction-llm claude-haiku-4-5
```
