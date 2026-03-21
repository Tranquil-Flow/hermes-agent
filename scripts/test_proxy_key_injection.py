#!/usr/bin/env python3
"""Test that the proxy injects the Anthropic API key even with a placeholder."""
import json
import sys
from pathlib import Path

# Read port from PID file
pid_file = Path("/Users/evinova/.hermes-aegis/proxy.pid")
pid_info = json.loads(pid_file.read_text())
port = pid_info["port"]
proxy_url = f"http://host.docker.internal:{port}"

print(f"Proxy: {proxy_url}")
print("Sending placeholder key — proxy should inject real key...")

ca_cert = "/certs/mitmproxy-ca-cert.pem"

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

import os

client = httpx.Client(
    proxy=proxy_url,
    verify=ca_cert,
    timeout=20
)

resp = client.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": "placeholder",  # proxy should replace this
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    },
    json={
        "model": "claude-haiku-4-5",
        "max_tokens": 20,
        "messages": [{"role": "user", "content": "Reply with exactly: injection works"}]
    }
)

print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    content = resp.json().get("content", [{}])[0].get("text", "")
    print(f"Response: {content}")
    print("\nSUCCESS — proxy key injection working!")
    sys.exit(0)
else:
    print(f"Body: {resp.text[:500]}")
    sys.exit(1)
