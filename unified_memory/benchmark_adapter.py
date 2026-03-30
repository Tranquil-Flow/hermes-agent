"""
Benchmark adapter — wraps UnifiedMemoryStore as a BenchmarkableStore.

Allows the benchmark runner to swap in the unified memory system
alongside cognitive, structured, and flat baselines transparently.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple

from benchmarks.interface import BenchmarkableStore
from unified_memory.config import UnifiedMemoryConfig
from unified_memory.store import UnifiedMemoryStore


class UnifiedBenchmarkAdapter(BenchmarkableStore):
    """Adapter that makes UnifiedMemoryStore benchmarkable."""

    def __init__(
        self,
        profile: str = "balanced",
        embedding_model: str = "auto",
        **kwargs,
    ):
        config = UnifiedMemoryConfig.from_profile(profile)
        config.embedding_model = embedding_model
        config.db_path = ":memory:"

        # Apply any custom parameter overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self._store = UnifiedMemoryStore(config=config, db_path=":memory:")
        self._store.enable_virtual_clock()

    def store(
        self,
        content: str,
        category: str = "factual",
        scope: str = "global",
        importance: float = 0.5,
    ) -> None:
        self._store.store(
            content=content,
            category=category,
            scope=scope,
            importance=importance,
        )
        # Tiny virtual time step preserves insertion order
        self._store.advance_time(0.0001)

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> List[str]:
        results = self._store.recall(query=query, scope=scope, top_k=top_k)
        return [r.fact.content for r in results]

    def simulate_time(self, days: float) -> None:
        self._store.advance_time(days * 86400)

    def simulate_access(self, content_substring: str) -> None:
        self._store.simulate_access(content_substring)

    def consolidate(self) -> None:
        self._store.consolidate()

    def get_stats(self) -> Dict[str, Any]:
        return self._store.get_stats()

    def reset(self) -> None:
        self._store.reset()

    def reward_memory(self, memory_id: str, signal: float) -> None:
        self._store.reward_memory(memory_id, signal)

    def explore(
        self,
        query: str,
        top_k: int = 20,
        scope: Optional[str] = None,
    ) -> List[str]:
        results = self._store.explore(query=query, scope=scope, top_k=top_k)
        return [r.fact.content for r in results]

    def recall_with_ids(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        return self._store.recall_with_ids(query=query, top_k=top_k, scope=scope)
