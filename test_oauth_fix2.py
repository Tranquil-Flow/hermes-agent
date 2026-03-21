import httpx

# Test with explicit debug - send request and check what headers come back in error
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
    proxy='http://host.docker.internal:8443',
    verify='/certs/mitmproxy-ca-cert.pem',
    timeout=30
)
print(f'Status: {resp.status_code}')
print(f'Headers: {dict(resp.headers)}')
print(f'Body: {resp.text[:500]}')

# Also test with explicit x-api-key placeholder to see if it gets replaced
print('\n--- Test 2: with x-api-key placeholder ---')
resp2 = httpx.post(
    'https://api.anthropic.com/v1/messages',
    json={
        'model': 'claude-sonnet-4-20250514',
        'max_tokens': 10,
        'messages': [{'role': 'user', 'content': 'say hi'}]
    },
    headers={
        'content-type': 'application/json',
        'anthropic-version': '2023-06-01',
        'x-api-key': 'placeholder',
    },
    proxy='http://host.docker.internal:8443',
    verify='/certs/mitmproxy-ca-cert.pem',
    timeout=30
)
print(f'Status: {resp2.status_code}')
print(f'Body: {resp2.text[:500]}')
