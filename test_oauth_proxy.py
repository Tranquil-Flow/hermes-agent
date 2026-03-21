import httpx

proxy = "http://host.docker.internal:8444"
try:
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Say hello"}],
        },
        headers={
            "x-api-key": "placeholder",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        proxy=proxy,
        verify=False,
        timeout=15,
    )
    print(f"STATUS: {r.status_code}")
    print(f"BODY: {r.text[:500]}")
except Exception as e:
    print(f"ERROR: {e}")
