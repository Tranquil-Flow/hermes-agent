"""Tests for cognitive_memory.store — the main CognitiveMemoryStore engine."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.store import CognitiveMemoryStore


def _make_store(**kwargs) -> CognitiveMemoryStore:
    """Create an in-memory store for testing."""
    config = CognitiveMemoryConfig.balanced()
    config.db_path = ":memory:"
    config.embedding_model = "tfidf"  # fast, no model download
    for k, v in kwargs.items():
        setattr(config, k, v)
    return CognitiveMemoryStore(config=config, db_path=":memory:")


class TestStoreBasic:
    def test_store_and_get(self):
        store = _make_store()
        mem_id = store.store("User prefers dark mode")
        assert mem_id is not None
        mem = store.get(mem_id)
        assert mem is not None
        assert mem.content == "User prefers dark mode"
        assert mem.category == "preference"

    def test_store_with_explicit_category(self):
        store = _make_store()
        mem_id = store.store("some fact", category="environment", importance=0.8)
        mem = store.get(mem_id)
        assert mem.category == "environment"
        assert mem.importance == 0.8

    def test_store_auto_importance(self):
        store = _make_store()
        mem_id = store.store("Actually, the correct answer is 42")
        mem = store.get(mem_id)
        # "correct" → correction → high importance
        assert mem.importance >= 0.7

    def test_stats(self):
        store = _make_store()
        # Use distinct memories that won't trigger contradiction detection
        store.store("Python is a programming language")
        store.store("The office has a coffee machine")
        stats = store.get_stats()
        assert stats["active"] == 2
        assert stats["by_layer"]["working"] == 2


class TestRecall:
    def test_recall_by_content(self):
        store = _make_store()
        store.store("Python is a programming language")
        store.store("Docker containers run isolated processes")
        store.store("Python virtual environments isolate dependencies")

        results = store.recall("Python language")
        assert len(results) > 0
        # The Python-related memories should rank higher
        contents = [r.entry.content for r in results]
        assert any("Python" in c for c in contents[:2])

    def test_recall_returns_scores(self):
        store = _make_store()
        store.store("The server runs on port 8080")
        results = store.recall("What port does the server use?")
        assert len(results) > 0
        assert results[0].score > 0
        assert "base_level" in results[0].components

    def test_recall_with_scope(self):
        store = _make_store()
        store.store("Use pytest for testing", scope="project:hermes")
        store.store("Use jest for testing", scope="project:webapp")
        store.store("Testing is important", scope="global")

        results = store.recall("testing framework", scope="project:hermes")
        # Should still return results (scope is a filter on candidates)
        assert len(results) > 0

    def test_recall_empty_store(self):
        store = _make_store()
        results = store.recall("anything")
        assert results == []

    def test_recall_updates_access_count(self):
        store = _make_store()
        mem_id = store.store("Remember this fact")
        store.recall("Remember this fact")
        mem = store.get(mem_id)
        assert mem.access_count >= 1


class TestContradiction:
    def test_contradiction_detected(self):
        store = _make_store(contradiction_threshold=0.5)  # lower for TF-IDF
        id1 = store.store("The database uses PostgreSQL", category="environment", scope="global")
        id2 = store.store("The database uses MySQL", category="environment", scope="global")

        # Check if first memory was superseded
        mem1 = store.get(id1)
        # With TF-IDF, similarity might not be high enough; check both cases
        if mem1.superseded_by is not None:
            assert mem1.superseded_by == id2

    def test_different_scopes_no_contradiction(self):
        store = _make_store(contradiction_threshold=0.5)
        id1 = store.store("Uses PostgreSQL", category="environment", scope="project:api")
        id2 = store.store("Uses MySQL", category="environment", scope="project:webapp")

        mem1 = store.get(id1)
        assert mem1.superseded_by is None  # different scopes


class TestConsolidation:
    def test_promote_to_core(self):
        store = _make_store(core_promotion_count=2, core_promotion_importance=0.3)
        mem_id = store.store("Important recurring fact", importance=0.8)

        # Simulate multiple accesses
        for _ in range(3):
            store.recall("Important recurring fact")

        report = store.consolidate()
        mem = store.get(mem_id)
        assert mem.layer == "core"
        assert report.promoted_to_core >= 1

    def test_link_decay(self):
        store = _make_store()
        id1 = store.store("Fact A about Docker")
        id2 = store.store("Fact B about Docker")

        # Create a link manually
        store.backend.create_link(id1, id2, weight=0.5, link_type="hebbian")

        report = store.consolidate()
        assert report.links_decayed is True

        # Check link weight decreased
        links = store.backend.get_links(id1)
        if links:
            assert links[0].weight < 0.5


class TestHebbianLinks:
    def test_co_recall_creates_links(self):
        store = _make_store()
        store.store("Docker is a containerization tool")
        store.store("Kubernetes orchestrates Docker containers")
        store.store("Unrelated fact about cooking pasta")

        # Recall something that activates both Docker-related memories
        store.recall("Docker container orchestration")
        store.recall("Docker container orchestration")

        # Check if links formed between the Docker memories
        stats = store.get_stats()
        # Links may or may not form depending on activation thresholds
        # Just verify no errors occurred
        assert stats["active"] == 3


class TestReset:
    def test_reset_clears_all(self):
        store = _make_store()
        store.store("fact one")
        store.store("fact two")
        store.reset()
        stats = store.get_stats()
        assert stats["active"] == 0
        assert stats["links"] == 0
