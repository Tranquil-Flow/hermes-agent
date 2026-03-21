"""Debug adversarial failures."""
import sys, os, json
sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/Projects/.huggingface_cache'

from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter

adapter = CognitiveBenchmarkAdapter()

with open('benchmarks/suite_d/fixtures/adversarial.json') as f:
    scenarios = json.load(f)

for sc in scenarios:
    if sc['id'] not in ('ad_07', 'ad_09', 'ad_14', 'ad_15'):
        continue
    
    adapter.reset()
    print(f"\n{'='*60}")
    print(f"Scenario: {sc['id']} (type: {sc.get('type')})")
    print(f"  Query: {sc['query']}")
    print(f"  Gold: {sc['gold_answer']}")
    print(f"  Malicious: {sc['malicious_fact'][:80]}")
    
    for fact in sc.get('normal_facts', []):
        adapter.store(fact, category='factual')
    adapter.store(sc['malicious_fact'], category='factual')
    
    results = adapter.recall(sc['query'], top_k=5)
    print(f"  Results:")
    for i, r in enumerate(results[:5]):
        print(f"    [{i}] {r[:100]}")
