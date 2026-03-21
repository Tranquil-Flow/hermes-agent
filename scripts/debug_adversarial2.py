"""Debug adversarial: show all scored results."""
import sys, os, json
sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/Projects/.huggingface_cache'

from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter
from cognitive_memory.store import CognitiveMemoryStore

adapter = CognitiveBenchmarkAdapter()

with open('benchmarks/suite_d/fixtures/adversarial.json') as f:
    scenarios = json.load(f)

for sc in scenarios:
    if sc['id'] not in ('ad_14', 'ad_15'):
        continue
    
    adapter.reset()
    for fact in sc.get('normal_facts', []):
        adapter.store(fact, category='factual')
    adapter.store(sc['malicious_fact'], category='factual')
    
    # Access internal store for detailed scoring
    store = adapter._store
    query_emb = store._embedder.encode(sc['query'])
    now = store._now()
    candidates = store.backend.get_all_active()
    link_map, emb_cache = store._build_link_map_and_embeddings(candidates)
    
    print(f"\n{'='*60}")
    print(f"{sc['id']}: {sc['query']}")
    print(f"Gold: {sc['gold_answer']}")
    for mem in candidates:
        comps = store._compute_activation(mem, query_emb, now, link_map, None, emb_cache)
        total = sum(comps.values())
        print(f"  [{total:+.3f}] {mem.content[:80]}")
        print(f"    components: {comps}")
