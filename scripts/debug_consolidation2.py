"""Debug consolidation: check access matching."""
import sys, os
sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/Projects/.huggingface_cache'

import json
from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter

adapter = CognitiveBenchmarkAdapter()

# Load consolidation fixtures
with open('benchmarks/suite_b/fixtures/consolidation.json') as f:
    scenarios = json.load(f)

# Debug cb_01 in detail
sc = [s for s in scenarios if s['id'] == 'cb_01'][0]
adapter.reset()

print(f"Facts:")
for fact in sc['facts']:
    adapter.store(fact, category='factual')
    print(f"  {fact[:60]}")

print(f"\nAccess sequence:")
for hint in sc['access_sequence']:
    # Check what matches
    active = adapter._store.backend.get_all_active()
    needle = hint.lower()
    matches = [m for m in active if needle in m.content.lower()]
    print(f"  Hint: '{hint}' -> {len(matches)} matches")
    for m in matches:
        print(f"    [{m.access_count}] {m.content[:60]}")
    adapter.simulate_access(hint)

print(f"\nPost-access states:")
active = adapter._store.backend.get_all_active()
for mem in active:
    print(f"  [count={mem.access_count}, imp={mem.importance}, layer={mem.layer}] {mem.content[:60]}")

print(f"\nCore promotion requires: access_count >= {adapter._store.config.core_promotion_count}, importance >= {adapter._store.config.core_promotion_importance}")
