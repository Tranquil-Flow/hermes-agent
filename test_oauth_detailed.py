"""Detailed OAuth proxy test — shows exactly what headers are sent/received."""
import httpx
import json

proxy = "http://host.docker.internal:8444"

# Test 1: Through proxy (should get OAuth headers injected)
print("=== TEST 1: Through aegis proxy ===")
try:
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json={"model": "claude-sonnet-4-20250514", "max_tokens": 5,
              "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "placeholder", "anthropic-version": "2023-06-01"},
        proxy=proxy, verify=False, timeout=15,
    )
    print(f"Status: {r.status_code}")
    try:
        body = r.json()
        print(f"Error type: {body.get('error', {}).get('type', 'N/A')}")
        print(f"Error msg: {body.get('error', {}).get('message', 'N/A')}")
    except:
        print(f"Body: {r.text[:200]}")
    # Show response headers that might hint at what went wrong
    for h in ["x-request-id", "cf-ray"]:
        if h in r.headers:
            print(f"  {h}: {r.headers[h]}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Same but with manually correct OAuth headers (bypass proxy injection)
print("\n=== TEST 2: Manual OAuth headers (no proxy) ===")
print("(skipped - no direct internet from container)")

# Test 3: Check what proxy port we're hitting
print("\n=== TEST 3: Proxy connectivity check ===")
try:
    r = httpx.get("http://host.docker.internal:8444/", timeout=5)
    print(f"Proxy root: {r.status_code} - {r.text[:100]}")
except Exception as e:
    print(f"Proxy check: {type(e).__name__}: {e}")

# Test 4: Try port 8443 instead (seen in some sessions)
print("\n=== TEST 4: Try proxy on port 8443 ===")
try:
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json={"model": "claude-sonnet-4-20250514", "max_tokens": 5,
              "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "placeholder", "anthropic-version": "2023-06-01"},
        proxy="http://host.docker.internal:8443", verify=False, timeout=15,
    )
    print(f"Status: {r.status_code}")
    try:
        body = r.json()
        print(f"Error type: {body.get('error', {}).get('type', 'N/A')}")
        print(f"Error msg: {body.get('error', {}).get('message', 'N/A')}")
    except:
        print(f"Body: {r.text[:200]}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
