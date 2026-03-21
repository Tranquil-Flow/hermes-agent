"""Test LLM contradiction detection on known failure cases."""
import sys, os
sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/Projects/.huggingface_cache'

import logging
logging.basicConfig(level=logging.INFO)

from cognitive_memory.llm_contradiction import check_contradiction_llm

# Test cases — known failures that heuristic can't catch
cases = [
    # ct_07: monolith -> microservice extraction (emb_sim=0.189)
    ("The payments service was extracted into a separate microservice",
     "The monolith handles all API requests",
     True),
    # Complementary facts that should NOT be contradictions
    ("API keys are stored in environment variables, never in code",
     "Rotate API keys every 90 days or immediately after any suspected exposure",
     False),
    # Clear contradiction
    ("The database now runs on PostgreSQL 16",
     "We use PostgreSQL 14 for all services",
     True),
    # Complementary 
    ("The API uses REST with JSON payloads",
     "API authentication uses OAuth 2.0 with PKCE",
     False),
]

print("Testing LLM contradiction detection:\n")
for new, existing, expected in cases:
    result = check_contradiction_llm(new, existing)
    status = "✓" if result == expected else "✗"
    print(f"  [{status}] Expected={expected}, Got={result}")
    print(f"      New: {new[:60]}")
    print(f"      Old: {existing[:60]}")
    print()
