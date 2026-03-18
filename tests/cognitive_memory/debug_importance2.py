#!/usr/bin/env python3
import os, sys, time
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.store import CognitiveMemoryStore

config = CognitiveMemoryConfig.balanced()
config.db_path = ":memory:"
config.embedding_model = "auto"
store = CognitiveMemoryStore(config=config, db_path=":memory:")
print(f"Backend: {store.embedder.backend_name}")

store.store("Production database password must never be logged", importance=1.0)
store.store("Dave had coffee this morning", importance=0.1)
store.store("The weather was nice yesterday", importance=0.1)

# Check all memories' access_times
for mem in store.backend.get_all_active():
    now = store._now()
    elapsed = [now - t for t in mem.access_times]
    import math
    bl_terms = [e**(-0.5) for e in elapsed]
    bl = math.log(sum(bl_terms)) if sum(bl_terms) > 0 else -10
    print(f"  imp={mem.importance} bl={bl:.3f} elapsed_ms={[f'{e*1000:.1f}' for e in elapsed]} | {mem.content[:50]}")
