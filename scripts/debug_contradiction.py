"""Debug contradiction detection for ad_14 facts."""
import sys, os
sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/Projects/.huggingface_cache'

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('cognitive_memory.store')
logger.setLevel(logging.DEBUG)

from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter

adapter = CognitiveBenchmarkAdapter()

facts = [
    "API keys are stored in environment variables, never in code",
    "Use hermes-aegis vault to access API keys securely",
    "Rotate API keys every 90 days or immediately after any suspected exposure",
]

for fact in facts:
    print(f"\n--- Storing: {fact[:60]} ---")
    adapter.store(fact, category='factual')
