"""Tests for cognitive_memory.benchmark_adapter — BenchmarkableStore wrapper."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter


class TestBenchmarkAdapter:
    def _make_adapter(self, **kwargs) -> CognitiveBenchmarkAdapter:
        return CognitiveBenchmarkAdapter(
            profile="balanced",
            embedding_model="tfidf",
            **kwargs,
        )

    def test_store_and_recall(self):
        adapter = self._make_adapter()
        adapter.store("Python is a programming language")
        adapter.store("Docker runs containers")
        results = adapter.recall("Python language")
        assert len(results) > 0
        assert isinstance(results[0], str)

    def test_simulate_time(self):
        adapter = self._make_adapter()
        adapter.store("fact one")
        adapter.simulate_time(7.0)  # 7 days
        # Should still be recallable
        results = adapter.recall("fact one")
        assert len(results) > 0

    def test_simulate_access(self):
        adapter = self._make_adapter()
        adapter.store("Remember this specific fact about cats")
        adapter.simulate_access("specific fact")
        stats = adapter.get_stats()
        assert stats["active"] == 1

    def test_consolidate(self):
        adapter = self._make_adapter()
        adapter.store("Important fact", importance=0.8)
        # Access several times to trigger promotion
        for _ in range(5):
            adapter.recall("Important fact")
        adapter.consolidate()
        stats = adapter.get_stats()
        assert stats["by_layer"]["core"] >= 1

    def test_reset(self):
        adapter = self._make_adapter()
        adapter.store("fact one")
        adapter.store("fact two")
        adapter.reset()
        stats = adapter.get_stats()
        assert stats["active"] == 0

    def test_get_stats(self):
        adapter = self._make_adapter()
        adapter.store("test memory")
        stats = adapter.get_stats()
        assert "active" in stats
        assert "by_layer" in stats
        assert "embedding_backend" in stats
        assert stats["active"] == 1

    def test_recall_with_scope(self):
        adapter = self._make_adapter()
        adapter.store("Use pytest", scope="project:hermes")
        adapter.store("Use jest", scope="project:webapp")
        results = adapter.recall("testing framework", scope="project:hermes")
        assert len(results) > 0

    def test_recall_top_k(self):
        adapter = self._make_adapter()
        for i in range(20):
            adapter.store(f"Memory number {i} about various topics")
        results = adapter.recall("various topics", top_k=5)
        assert len(results) <= 5
