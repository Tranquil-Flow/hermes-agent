"""Debug: check what headers the proxy actually sends."""
import httpx
import os

proxy_url = os.environ.get("HTTPS_PROXY", os.environ.get("HTTP_PROXY", "http://host.docker.internal:8444"))
print(f"Using proxy: {proxy_url}")

# Use httpbin to echo back what headers the proxy sends
# This tells us if the OAuth headers are being injected
print("\n--- Test 1: What headers does Anthropic actually receive? ---")
try:
    with httpx.Client(proxy=proxy_url, verify=False, timeout=15) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": "placeholder-for-injection",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        print(f"Status: {resp.status_code}")
        body = resp.text[:500]
        print(f"Response: {body}")
        # Print response headers that might give clues
        print(f"Request ID: {resp.headers.get('request-id', 'N/A')}")
except Exception as e:
    print(f"Exception: {e}")

# Test 2: Try sending WITH the OAuth headers ourselves to see if the problem
# is the proxy not injecting them vs Anthropic rejecting them
print("\n--- Test 2: Manually add OAuth headers (bypass proxy injection) ---")
try:
    with httpx.Client(proxy=proxy_url, verify=False, timeout=15) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14,claude-code-20250219,oauth-2025-04-20",
                "user-agent": "claude-cli/2.1.74 (external, cli)",
                "x-app": "cli",
                "x-api-key": "placeholder-for-injection",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        print(f"Status: {resp.status_code}")
        body = resp.text[:500]
        print(f"Response: {body}")
except Exception as e:
    print(f"Exception: {e}")
