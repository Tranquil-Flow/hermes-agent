"""
Benchmark adapter — wraps CognitiveMemoryStore as a BenchmarkableStore.

This allows the benchmark runner to swap in the cognitive memory system
alongside flat/FTS5 baselines transparently.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional

from benchmarks.interface import BenchmarkableStore
from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.store import CognitiveMemoryStore


class CognitiveBenchmarkAdapter(BenchmarkableStore):
    """Adapter that makes CognitiveMemoryStore benchmarkable."""

    def __init__(
        self,
        profile: str = "balanced",
        embedding_model: str = "auto",
        **kwargs,
    ):
        config = CognitiveMemoryConfig.from_profile(profile)
        config.embedding_model = embedding_model
        # Use in-memory SQLite for benchmarks
        config.db_path = ":memory:"
        self._store = CognitiveMemoryStore(config=config, db_path=":memory:")
        # Use virtual clock so encoding speed doesn't create timing artifacts
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
        # Tiny virtual time step preserves insertion order while keeping
        # base_level differences negligible between batch-stored memories.
        self._store.advance_time(0.0001)  # 0.1ms virtual gap

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> List[str]:
        results = self._store.recall(query=query, scope=scope, top_k=top_k)
        return [r.entry.content for r in results]

    def simulate_time(self, days: float) -> None:
        self._store.advance_time(days * 86400)  # convert days to seconds

    def simulate_access(self, content_substring: str) -> None:
        """Simulate accessing a memory by content substring match."""
        import time as _time
        active = self._store.backend.get_all_active()
        now = self._store._now()
        for mem in active:
            if content_substring in mem.content:
                access_times = mem.access_times + [now]
                if len(access_times) > self._store.config.max_access_times:
                    access_times = access_times[-self._store.config.max_access_times:]
                self._store.backend.update(
                    mem.id,
                    last_accessed=now,
                    access_count=mem.access_count + 1,
                    access_times=access_times,
                )
                break

    def consolidate(self) -> None:
        self._store.consolidate()

    def get_stats(self) -> Dict[str, Any]:
        return self._store.get_stats()

    def reset(self) -> None:
        self._store.reset()
