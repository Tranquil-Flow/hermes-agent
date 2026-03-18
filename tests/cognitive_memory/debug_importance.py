#!/usr/bin/env python3
"""Debug importance filtering with sentence-transformers."""
import os, sys, json
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.store import CognitiveMemoryStore

config = CognitiveMemoryConfig.balanced()
config.db_path = ":memory:"
# Test with both embedding backends
for emb in ["tfidf", "auto"]:
    config.embedding_model = emb
    store = CognitiveMemoryStore(config=config, db_path=":memory:")
    print(f"\n=== Embedding: {store.embedder.backend_name} ===")

    # Simulate importance_filtering scenario
    store.store("Production database password must never be logged", importance=1.0)
    store.store("Dave had coffee this morning", importance=0.1)
    store.store("The weather was nice yesterday", importance=0.1)

    results = store.recall("What's the critical rule about database credentials?", top_k=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r.score:.3f}] imp={r.entry.importance} "
              f"base={r.components['base_level']:.3f} "
              f"spread={r.components['spreading']:.3f} "
              f"imp_boost={r.components['importance_boost']:.3f} "
              f"| {r.entry.content[:50]}")
    store.reset()
