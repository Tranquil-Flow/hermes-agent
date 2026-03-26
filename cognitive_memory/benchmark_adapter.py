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

        # Apply any custom parameter overrides (for ablation/sweeps)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

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
        """Simulate accessing a memory by content substring or semantic match.

        First tries case-insensitive substring match; if no match, falls
        back to embedding similarity.  Picks the best-matching memory and
        advances the virtual clock between accesses so ACT-R base-level
        can discriminate multiple rehearsals.
        """
        active = self._store.backend.get_all_active()
        if not active:
            return
        now = self._store._now()
        needle = content_substring.lower()

        # Try substring match first
        matches = [
            mem for mem in active
            if needle in mem.content.lower()
        ]

        if matches:
            # Prefer shortest matching content (most specific fact)
            best = min(matches, key=lambda m: len(m.content))
        else:
            # Fall back to embedding similarity
            from cognitive_memory.embeddings import cosine_similarity
            query_emb = self._store._embedder.encode(content_substring)
            best = None
            best_sim = -1.0
            for mem in active:
                if mem.embedding is not None:
                    sim = cosine_similarity(query_emb, mem.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best = mem
            if best is None or best_sim < 0.3:
                return

        access_times = best.access_times + [now]
        if len(access_times) > self._store.config.max_access_times:
            access_times = access_times[-self._store.config.max_access_times:]
        self._store.backend.update(
            best.id,
            last_accessed=now,
            access_count=best.access_count + 1,
            access_times=access_times,
        )
        # Small time step between accesses so ACT-R sees distinct timestamps
        self._store.advance_time(3600)  # 1 hour between rehearsals

    def consolidate(self) -> None:
        self._store.consolidate()

    def get_stats(self) -> Dict[str, Any]:
        return self._store.get_stats()

    def reset(self) -> None:
        self._store.reset()
        # Also reset Q-value store so benchmark runs are independent
        if self._store._qvalue_store:
            self._store._qvalue_store.reset()

    def reward_memory(self, memory_id: str, signal: float) -> None:
        """Apply a reward signal to a memory's Q-value.
        Delegates to the underlying CognitiveMemoryStore.
        """
        self._store.reward_memory(memory_id, signal)

    def recall_with_ids(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> List[tuple]:
        """Like recall() but returns (content, memory_id) tuples.
        Used by Q-learning benchmark to apply rewards to specific memories.
        """
        results = self._store.recall(query=query, scope=scope, top_k=top_k)
        return [(r.entry.content, r.entry.id) for r in results]
