"""Debug ad_06 — hallucinated fact."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.INFO)

from cognitive_memory.store import CognitiveMemoryStore
from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.embeddings import cosine_similarity

config = CognitiveMemoryConfig.balanced()
config.db_path = ":memory:"
store = CognitiveMemoryStore(config=config, db_path=":memory:")
store.enable_virtual_clock()

with open("benchmarks/suite_d/fixtures/adversarial.json") as f:
    scenarios = json.load(f)
sc = scenarios[5]  # ad_06

print(f"=== {sc['id']}: {sc['description'][:80]} ===")
print(f"Malicious: {sc['malicious_fact'][:100]}")
print(f"Gold: {sc['gold_answer']}")

for fact in sc["normal_facts"]:
    store.store(fact, category="factual")
    store.advance_time(0.0001)
store.store(sc["malicious_fact"], category="factual")
store.advance_time(0.0001)

# Check supersessions
all_mems = store.backend.get_all_active()
print(f"\nActive memories: {len(all_mems)}")
for m in all_mems:
    print(f"  imp={m.importance:.2f} sup={m.superseded_by} content={m.content[:80]}")

# Recall
results = store.recall(sc["query"], top_k=5)
print(f"\nRecall top 5:")
for i, r in enumerate(results):
    print(f"  [{i+1}] score={r.score:.3f} {r.components}")
    print(f"      {r.entry.content[:100]}")
