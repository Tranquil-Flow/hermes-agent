import httpx
import sys

# Try both common aegis ports
for port in [8443, 8444]:
    try:
        resp = httpx.post(
            'https://api.anthropic.com/v1/messages',
            json={
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 10,
                'messages': [{'role': 'user', 'content': 'say hi'}]
            },
            headers={
                'content-type': 'application/json',
                'anthropic-version': '2023-06-01',
            },
            proxy=f'http://host.docker.internal:{port}',
            verify='/certs/mitmproxy-ca-cert.pem',
            timeout=15
        )
        print(f'Port {port}: Status={resp.status_code}')
        print(f'Body: {resp.text[:300]}')
        if resp.status_code == 200:
            print('SUCCESS!')
            sys.exit(0)
    except Exception as e:
        print(f'Port {port}: {e}')

