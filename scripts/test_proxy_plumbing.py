#!/usr/bin/env python3
"""Quick plumbing check: verify vault key injection and proxy reach from container."""
import os
import sys

proxy = os.environ.get('HTTP_PROXY', '')
token = os.environ.get('ANTHROPIC_TOKEN', '')
aegis = os.environ.get('AEGIS_ACTIVE', '')
ca_cert = os.environ.get('SSL_CERT_FILE', '/certs/mitmproxy-ca-cert.pem')

print(f"HTTP_PROXY: {proxy or 'NOT SET'}")
print(f"ANTHROPIC_TOKEN: {'set (' + str(len(token)) + ' chars)' if token else 'NOT SET'}")
print(f"AEGIS_ACTIVE: {aegis or 'NOT SET'}")
print(f"SSL_CERT_FILE: {ca_cert}")
print()

if not proxy:
    print("FAIL: no HTTP_PROXY — env forwarding not working")
    sys.exit(1)
if not token:
    print("FAIL: no ANTHROPIC_TOKEN — vault injection not working")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("httpx not available, installing...")
    os.system("pip install httpx -q")
    import httpx

print("Testing Anthropic API via aegis proxy...")
try:
    client = httpx.Client(
        proxy=proxy,
        verify=ca_cert if os.path.exists(ca_cert) else True,
        timeout=20
    )
    resp = client.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'x-api-key': token,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        },
        json={
            'model': 'claude-haiku-4-5',
            'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'Reply: ok'}]
        }
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        content = data.get('content', [{}])[0].get('text', '')
        print(f"Response: {content}")
        print("\nSUCCESS — vault key injection + proxy working end-to-end!")
    else:
        print(f"Error body: {resp.text[:300]}")
        sys.exit(1)
except Exception as e:
    print(f"Exception: {e}")
    sys.exit(1)
