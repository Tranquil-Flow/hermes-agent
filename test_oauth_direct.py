"""Test: manually send with the exact same headers hermes uses."""
import httpx
import os

# Read the vault token from the aegis config (mounted read-only)
# We can't read the vault, but let's try sending with known OAuth headers
# to see if the APPROACH works at all

proxy = "http://host.docker.internal:8444"

# The vault token is sk-ant-oat01-* (we know from host-side check)
# Let's see if we can make a request that would work if the token is valid

# Try: send request WITH the correct headers pre-set, to see if proxy 
# double-injects or if the headers themselves are the problem
print("=== TEST: Pre-set OAuth headers + proxy ===")
try:
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json={"model": "claude-sonnet-4-20250514", "max_tokens": 5,
              "messages": [{"role": "user", "content": "hi"}]},
        headers={
            "x-api-key": "placeholder",
            "anthropic-version": "2023-06-01",
            "user-agent": "claude-cli/2.1.74 (external, cli)",
            "x-app": "cli",
            "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14,oauth-2025-04-20",
        },
        proxy=proxy, verify=False, timeout=15,
    )
    print(f"Status: {r.status_code}")
    try:
        body = r.json()
        err = body.get("error", {})
        print(f"Error type: {err.get('type', 'N/A')}")
        print(f"Error msg: {err.get('message', 'N/A')}")
        if r.status_code == 400:
            print(">>> 400 means OAuth IS accepted but request body is wrong!")
        elif r.status_code == 200:
            print(">>> SUCCESS! OAuth works through proxy!")
    except:
        print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"ERROR: {e}")
