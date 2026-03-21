#!/usr/bin/env python3
"""Test LLM contradiction detection at the STORE level (not judge).
These are the 4 cases that needed LLM to catch (ct_05, ct_07, ct_13, ct_17).
"""
import sys
import os
sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'

from cognitive_memory.store import CognitiveMemoryStore
from cognitive_memory.config import CognitiveMemoryConfig

config = CognitiveMemoryConfig()
config.contradiction_llm_model = "claude-haiku-4-5"
config.db_path = ":memory:"

# ct_07: monolith -> microservice (the hard one)
store = CognitiveMemoryStore(config=config, db_path=":memory:")
store.store("We use a monolithic architecture where one service handles all API requests.")
store.store("The payments team extracted their service into a separate microservice.")

results = store.recall("What is the architecture of our system?", top_k=3)
print("ct_07 (monolith -> microservice):")
for r in results:
    print(f"  [{r['score']:.3f}] {r['content'][:80]}")
# Check contradictions were detected
print()

# ct_05: React -> Next.js
store2 = CognitiveMemoryStore(config=config, db_path=":memory:")
store2.store("The project uses React for the frontend.")
store2.store("We migrated the frontend from React to Next.js last quarter.")

results2 = store2.recall("What frontend technology do we use?", top_k=3)
print("ct_05 (React -> Next.js):")
for r in results2:
    print(f"  [{r['score']:.3f}] {r['content'][:80]}")
print()

# Check contradiction flags
print("Store2 memories with contradictions:")
for m in store2._store.values() if hasattr(store2, '_store') else []:
    if hasattr(m, 'contradicts') and m.contradicts:
        print(f"  CONTRADICTED: {m.content[:60]}")
