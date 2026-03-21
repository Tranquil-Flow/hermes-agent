"""Debug consolidation failures."""
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

# Debug cb_01
for sc in scenarios:
    if sc['id'] not in ('cb_01', 'cb_06', 'cb_08', 'cb_11', 'cb_12'):
        continue
    
    adapter.reset()
    print(f"\n{'='*60}")
    print(f"Scenario: {sc['id']} (expects_layer: {sc.get('expects_layer')})")
    print(f"  Query: {sc['query']}")
    print(f"  Gold: {sc['gold_answer']}")
    
    # Store facts
    for fact in sc['facts']:
        adapter.store(fact, category='factual')
    
    # Simulate rehearsals
    access_seq = sc.get('access_sequence', [])
    print(f"  Access sequence ({len(access_seq)} items): {access_seq}")
    for access_hint in access_seq:
        adapter.simulate_access(access_hint)
    
    # Check access counts before consolidation
    active = adapter._store.backend.get_all_active()
    for mem in active:
        if mem.access_count > 1:
            print(f"  Memory (access_count={mem.access_count}, layer={mem.layer}): {mem.content[:60]}")
    
    # Advance time
    time_gap = sc.get('time_gap_days', 30)
    adapter.simulate_time(time_gap)
    print(f"  Time gap: {time_gap} days")
    
    # Consolidate
    report = adapter._store.consolidate()
    print(f"  Consolidation: +{report.promoted_to_core} core, -{report.demoted_to_archive} archive")
    for d in report.details:
        print(f"    {d}")
    
    # Recall
    results = adapter.recall(sc['query'], top_k=5)
    print(f"  Results:")
    for i, r in enumerate(results[:3]):
        print(f"    [{i}] {r[:80]}")
