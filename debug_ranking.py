"""Debug recall ranking for specific scenarios."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from cognitive_memory.store import CognitiveMemoryStore
from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.embeddings import cosine_similarity
import random

config = CognitiveMemoryConfig.balanced()
config.db_path = ":memory:"
store = CognitiveMemoryStore(config=config, db_path=":memory:")
store.enable_virtual_clock()

# Test Suite E sc_e03
with open("benchmarks/suite_e/fixtures/scale.json") as f:
    scenarios = json.load(f)

sc = scenarios[2]  # sc_e03
print(f"=== {sc['id']}: {sc.get('description', '')[:80]} ===")
print(f"Target: {sc['target_fact'][:80]}")
print(f"Gold: {sc['gold_answer']}")
print(f"Noise count: {sc.get('noise_count', len(sc.get('noise_facts', [])))}")

store.reset()
store.store(sc["target_fact"], category="factual", scope="global", importance=0.8)
store.advance_time(0.0001)

for nf in sc.get("noise_facts", []):
    store.store(nf, category="factual", scope="global", importance=0.3)
    store.advance_time(0.0001)

noise_count = sc.get("noise_count", 0)
template = sc.get("noise_template", "System fact {i}: value {val}")
existing = len(sc.get("noise_facts", [])) + 1
rng = random.Random(42)
for i in range(existing, noise_count):
    synth = template.format(i=i, val=rng.randint(1000, 9999))
    store.store(synth, category="factual", scope="global", importance=0.1)
    store.advance_time(0.0001)

store.advance_time(30 * 86400)

# Recall with full scoring details
results = store.recall(sc["query"], top_k=10)
print(f"\nQuery: {sc['query']}")
print(f"Top 10 results:")
for i, r in enumerate(results):
    is_target = sc["target_fact"] in r.entry.content
    marker = " <<<TARGET" if is_target else ""
    print(f"  [{i+1}] score={r.score:.3f} imp={r.entry.importance:.1f} {r.components}")
    print(f"      {r.entry.content[:100]}{marker}")

# Also check cosine similarity of target vs query
query_emb = store.embedder.encode(sc["query"])
target_emb = store.embedder.encode(sc["target_fact"])
print(f"\nTarget cosine sim to query: {cosine_similarity(query_emb, target_emb):.3f}")

# Check highest-sim noise fact
print("\nHighest-sim noise facts:")
all_active = store.backend.get_all_active()
sims = []
for mem in all_active:
    if mem.embedding is not None:
        sim = cosine_similarity(query_emb, mem.embedding)
        is_t = sc["target_fact"] in mem.content
        sims.append((sim, mem.importance, mem.content[:80], is_t))
sims.sort(reverse=True)
for sim, imp, content, is_t in sims[:10]:
    marker = " <<<" if is_t else ""
    print(f"  sim={sim:.3f} imp={imp:.1f} {content}{marker}")
