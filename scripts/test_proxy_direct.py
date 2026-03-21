#!/usr/bin/env python3
"""Test proxy reachability directly — bypass env forwarding by reading the PID file."""
import json
import os
import socket
import sys
from pathlib import Path

# Read PID file directly from the mounted aegis dir
pid_file = Path("/Users/evinova/.hermes-aegis/proxy.pid")

if not pid_file.exists():
    print("FAIL: no proxy.pid file")
    sys.exit(1)

try:
    pid_info = json.loads(pid_file.read_text())
    port = pid_info["port"]
    pid = pid_info["pid"]
    print(f"Proxy PID: {pid}, port: {port}")
except Exception as e:
    print(f"FAIL: can't read pid file: {e}")
    sys.exit(1)

# Try to reach proxy via host.docker.internal (Docker's way to reach the host)
for host in ["host.docker.internal", "172.17.0.1", "127.0.0.1"]:
    sock = socket.socket()
    sock.settimeout(2.0)
    try:
        sock.connect((host, port))
        sock.close()
        print(f"Proxy reachable at {host}:{port}")
        proxy_url = f"http://{host}:{port}"
        break
    except OSError as e:
        print(f"  {host}:{port} -> {e}")
        sock.close()
else:
    print("FAIL: proxy not reachable from container")
    sys.exit(1)

# Also check auth.json for the minted API key
auth_file = Path("/Users/evinova/.hermes/auth.json")
if auth_file.exists():
    try:
        auth = json.loads(auth_file.read_text())
        agent_key = auth.get("providers", {}).get("nous", {}).get("agent_key", "")
        if agent_key:
            print(f"Agent key found: {agent_key[:12]}... ({len(agent_key)} chars)")
        else:
            print("No agent_key in auth.json")
    except Exception as e:
        print(f"auth.json error: {e}")
else:
    print("No auth.json found")

# Now try a real API call
ca_cert = "/certs/mitmproxy-ca-cert.pem"
if not os.path.exists(ca_cert):
    print(f"No CA cert at {ca_cert}")
    # Try without cert verification
    ca_cert = None

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx

# Use the agent_key directly if available
if auth_file.exists():
    auth = json.loads(auth_file.read_text())
    api_key = auth.get("providers", {}).get("nous", {}).get("agent_key", "")
else:
    api_key = ""

if not api_key:
    print("No API key available — cannot test Anthropic call")
    sys.exit(0)

print(f"\nTesting API call via proxy {proxy_url}...")
try:
    client = httpx.Client(
        proxy=proxy_url,
        verify=ca_cert if ca_cert else False,
        timeout=20
    )
    resp = client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        },
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 15,
            "messages": [{"role": "user", "content": "Reply: proxy works"}]
        }
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        content = resp.json().get("content", [{}])[0].get("text", "")
        print(f"Response: {content}")
        print("\nSUCCESS — proxy + API key working from container!")
    else:
        print(f"Error: {resp.text[:300]}")
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
