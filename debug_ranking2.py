"""Debug the ranking issue — check why position 1 has imp=1.0"""
import json, sys, os, random
sys.path.insert(0, os.path.dirname(__file__))

from cognitive_memory.store import CognitiveMemoryStore
from cognitive_memory.config import CognitiveMemoryConfig

config = CognitiveMemoryConfig.balanced()
config.db_path = ":memory:"
store = CognitiveMemoryStore(config=config, db_path=":memory:")
store.enable_virtual_clock()

with open("benchmarks/suite_e/fixtures/scale.json") as f:
    scenarios = json.load(f)
sc = scenarios[2]  # sc_e03

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

# Check all for boosted importance
all_active = store.backend.get_all_active()
boosted = [(m.importance, m.content[:80]) for m in all_active if m.importance > 0.5]
print(f"Total active: {len(all_active)}")
print(f"Memories with imp > 0.5: {len(boosted)}")
for imp, c in sorted(boosted, reverse=True):
    print(f"  imp={imp:.2f}: {c}")
