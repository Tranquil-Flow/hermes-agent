"""Test: does the proxy inject anything, or pass through as-is?"""
import httpx

proxy = "http://host.docker.internal:8444"

# Send WITHOUT x-api-key header to see if proxy adds auth at all
print("=== TEST: No x-api-key, see if proxy injects ===")
try:
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json={"model": "claude-sonnet-4-20250514", "max_tokens": 5,
              "messages": [{"role": "user", "content": "hi"}]},
        headers={"anthropic-version": "2023-06-01"},
        proxy=proxy, verify=False, timeout=15,
    )
    print(f"Status: {r.status_code}")
    body = r.json()
    err = body.get("error", {})
    print(f"Error: {err.get('message', 'N/A')}")
    if "missing" in err.get("message", "").lower():
        print(">>> Proxy did NOT inject auth!")
    elif "OAuth" in err.get("message", ""):
        print(">>> Proxy DID inject OAuth token")
except Exception as e:
    print(f"ERROR: {e}")

# Send WITH good headers already set (pre-test showed 400 = works)
print("\n=== TEST: Good headers pre-set ===")
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
    body = r.json()
    err = body.get("error", {})
    print(f"Error: {err.get('type', 'N/A')}: {err.get('message', 'N/A')}")
except Exception as e:
    print(f"ERROR: {e}")
