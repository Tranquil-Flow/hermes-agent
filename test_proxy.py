"""Test if Anthropic API works through the aegis proxy."""
import httpx
import json
import os

proxy_url = os.environ.get("HTTP_PROXY", os.environ.get("HTTPS_PROXY", "http://host.docker.internal:8444"))
print(f"Using proxy: {proxy_url}")

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
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
            },
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"SUCCESS! Response: {data['content'][0]['text']}")
        else:
            print(f"Error body: {resp.text[:500]}")
except Exception as e:
    print(f"Exception: {e}")
