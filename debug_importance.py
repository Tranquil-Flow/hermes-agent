"""Debug why facts get high importance."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from cognitive_memory.store import CognitiveMemoryStore
from cognitive_memory.config import CognitiveMemoryConfig

config = CognitiveMemoryConfig.balanced()
config.db_path = ":memory:"
store = CognitiveMemoryStore(config=config, db_path=":memory:")
store.enable_virtual_clock()

# Simulate sc_e03 storage sequence
with open("benchmarks/suite_e/fixtures/scale.json") as f:
    scenarios = json.load(f)
sc = scenarios[2]  # sc_e03

import logging
logging.basicConfig(level=logging.INFO)

store.store(sc["target_fact"], category="factual", scope="global", importance=0.8)
store.advance_time(0.0001)

for i, nf in enumerate(sc.get("noise_facts", [])[:10]):
    mem_id = store.store(nf, category="factual", scope="global", importance=0.3)
    store.advance_time(0.0001)
    mem = store.get(mem_id)
    if mem and mem.importance != 0.3:
        print(f"  Noise fact {i} imp changed: 0.3 -> {mem.importance}: {nf[:80]}")
    if mem and mem.metadata.get("supersedes"):
        print(f"  Noise fact {i} SUPERSEDED something: {mem.metadata}")

# Check all active for boosted importance
all_active = store.backend.get_all_active()
print(f"\nActive memories with importance > 0.5:")
for mem in all_active:
    if mem.importance > 0.5:
        print(f"  imp={mem.importance} superseded_by={mem.superseded_by} meta={mem.metadata} content={mem.content[:80]}")
