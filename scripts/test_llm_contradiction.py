#!/usr/bin/env python3
"""Test LLM contradiction detection end-to-end via aegis proxy."""
import sys
import os
import logging

# Enable debug logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')

sys.path.insert(0, '/workspace/Projects/hermes-agent')

from cognitive_memory.llm_contradiction import check_contradiction_llm, _detect_proxy_port

print(f"Proxy port: {_detect_proxy_port()}")
print()

test_cases = [
    # Should CONTRADICT
    (
        "We use a monolithic architecture where one service handles all API requests.",
        "Payments have been extracted into a separate microservice.",
        True,
        "monolith -> microservice"
    ),
    (
        "The project uses React for the frontend.",
        "We migrated the frontend from React to Next.js.",
        True,
        "React -> Next.js"
    ),
    # Should NOT contradict
    (
        "The API uses JSON for data exchange.",
        "The authentication service uses JWT tokens.",
        False,
        "unrelated facts"
    ),
    (
        "The database is PostgreSQL.",
        "We use Redis for caching.",
        False,
        "complementary facts"
    ),
]

passed = 0
failed = 0
for new_content, existing_content, expected, desc in test_cases:
    result = check_contradiction_llm(new_content, existing_content)
    status = "PASS" if result == expected else "FAIL"
    if result == expected:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] {desc}: expected={expected}, got={result}")

print(f"\n{passed}/{passed+failed} tests passed")
if failed == 0:
    print("LLM contradiction detection working via proxy!")
    sys.exit(0)
else:
    sys.exit(1)
