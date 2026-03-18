#!/usr/bin/env python3
"""
End-to-end smoke test for the cognitive memory system.
Tests with sentence-transformers (real embeddings) if available,
falls back to TF-IDF.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.store import CognitiveMemoryStore


def smoke_test():
    print("=== Cognitive Memory Smoke Test ===\n")

    # Create store with auto embedding (will use sentence-transformers if available)
    config = CognitiveMemoryConfig.balanced()
    config.db_path = ":memory:"
    store = CognitiveMemoryStore(config=config, db_path=":memory:")

    print(f"Embedding backend: {store.embedder.backend_name}")
    print()

    # 1. Store some memories
    print("--- Storing memories ---")
    memories = [
        ("User prefers dark mode for all applications", None, "global"),
        ("The API server runs on port 8443", None, "project:hermes"),
        ("Docker containers need the aegis proxy for network access", None, "project:hermes"),
        ("Python virtual environments isolate package dependencies", None, "global"),
        ("The database uses PostgreSQL 15 on the production server", None, "project:api"),
        ("User's name is Tranquil-Flow", None, "global"),
        ("To deploy, run make deploy from the project root", None, "project:hermes"),
        ("Last session we discussed the memory architecture redesign", None, "global"),
        ("ACT-R uses decay parameter d=0.5 as the standard value", None, "global"),
        ("The cat should be patted regularly", None, "global"),
    ]

    for content, cat, scope in memories:
        mem_id = store.store(content, category=cat, scope=scope)
        mem = store.get(mem_id)
        print(f"  [{mem.category:12}] imp={mem.importance:.2f} | {content[:60]}")

    print(f"\nTotal stored: {store.backend.count()}")

    # 2. Recall tests
    print("\n--- Recall tests ---")

    queries = [
        ("What theme does the user prefer?", None),
        ("How do I deploy the project?", "project:hermes"),
        ("What port does the API run on?", None),
        ("container networking", None),
        ("Tell me about the user", None),
    ]

    for query, scope in queries:
        print(f"\n  Query: \"{query}\"" + (f" (scope={scope})" if scope else ""))
        results = store.recall(query, scope=scope, top_k=3)
        for i, r in enumerate(results):
            print(f"    {i+1}. [{r.score:.3f}] {r.entry.content[:60]}...")
            print(f"       base={r.components['base_level']:.3f} "
                  f"spread={r.components['spreading']:.3f} "
                  f"imp={r.components['importance_boost']:.3f} "
                  f"scope={r.components['scope_boost']:.3f}")

    # 3. Contradiction test
    print("\n--- Contradiction test ---")
    old_id = store.store("The database uses MySQL on production", category="environment", scope="project:api")
    old_mem = store.get(old_id)
    # Check if the PostgreSQL memory was superseded
    stats = store.get_stats()
    print(f"  Superseded memories: {stats.get('superseded', 0)}")

    # 4. Consolidation
    print("\n--- Consolidation ---")
    # Access some memories multiple times to trigger promotion
    for _ in range(5):
        store.recall("dark mode preference")
        store.recall("deploy the project")

    report = store.consolidate()
    print(f"  Promoted to core: {report.promoted_to_core}")
    print(f"  Demoted to archive: {report.demoted_to_archive}")
    print(f"  Pruned: {report.pruned}")
    print(f"  Links pruned: {report.links_pruned}")

    # 5. Final stats
    print("\n--- Final Stats ---")
    stats = store.get_stats()
    print(f"  Active memories: {stats['active']}")
    print(f"  By layer: {stats['by_layer']}")
    print(f"  By category: {stats['by_category']}")
    print(f"  Links: {stats['links']}")
    print(f"  Embedding: {stats['embedding_backend']}")

    print("\n=== Smoke test PASSED ===")


if __name__ == "__main__":
    smoke_test()
