"""
CognitiveMemoryStore — the main engine.

Orchestrates:
- Encoding (classify + importance)
- Embedding generation
- ACT-R activation scoring (base level + spreading + importance + scope)
- Hebbian co-activation link strengthening
- Contradiction detection (embedding-based)
- Three-layer consolidation (working → core → archive)

Usage:
    from cognitive_memory.store import CognitiveMemoryStore
    from cognitive_memory.config import CognitiveMemoryConfig

    store = CognitiveMemoryStore(CognitiveMemoryConfig.balanced())
    mem_id = store.store("User prefers dark mode")
    results = store.recall("What theme does the user like?")
"""

from __future__ import annotations

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

from cognitive_memory.config import CognitiveMemoryConfig
from cognitive_memory.backends.base import MemoryEntry, MemoryLink, ScoredMemory
from cognitive_memory.backends.builtin import BuiltinSQLiteBackend
from cognitive_memory.embeddings import EmbeddingProvider, cosine_similarity
from cognitive_memory.encoding import encode as encode_content

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationReport:
    """Report from a consolidation cycle."""
    promoted_to_core: int = 0
    demoted_to_archive: int = 0
    pruned: int = 0
    links_pruned: int = 0
    links_decayed: bool = False
    details: List[str] = field(default_factory=list)


class CognitiveMemoryStore:
    """
    Main cognitive memory engine.

    Combines SQLite storage, embedding-based retrieval, ACT-R activation
    scoring, Hebbian link formation, contradiction detection, and
    three-layer consolidation.
    """

    def __init__(
        self,
        config: Optional[CognitiveMemoryConfig] = None,
        db_path: Optional[str] = None,
    ):
        self._config = config or CognitiveMemoryConfig.balanced()
        self._backend = BuiltinSQLiteBackend(
            db_path or self._config.db_path
        )
        self._embedder = EmbeddingProvider(model=self._config.embedding_model)
        self._simulated_time_offset: float = 0.0  # for benchmarking

        logger.info(
            f"CognitiveMemoryStore initialized: "
            f"embedding={self._embedder.backend_name}, "
            f"db={db_path or self._config.db_path}"
        )

    @property
    def config(self) -> CognitiveMemoryConfig:
        return self._config

    @property
    def backend(self) -> BuiltinSQLiteBackend:
        return self._backend

    @property
    def embedder(self) -> EmbeddingProvider:
        return self._embedder

    def _now(self) -> float:
        """Current time, accounting for simulated time offset."""
        return time.time() + self._simulated_time_offset

    def advance_time(self, seconds: float) -> None:
        """Advance simulated clock (for benchmarking)."""
        self._simulated_time_offset += seconds

    # ─── Store ───────────────────────────────────────────────────

    def store(
        self,
        content: str,
        category: Optional[str] = None,
        scope: str = "global",
        importance: Optional[float] = None,
        source: str = "agent",
        pinned: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a new memory. Auto-classifies category and importance
        if not provided. Generates embedding. Checks for contradictions.
        Creates semantic links.

        Returns the memory ID.
        """
        now = self._now()

        # Auto-classify if not provided
        if category is None or importance is None:
            auto_cat, auto_imp = encode_content(content)
            if category is None:
                category = auto_cat
            if importance is None:
                importance = auto_imp

        # Generate embedding
        embedding = self._embedder.encode(content)

        # Create entry
        mem_id = str(uuid.uuid4())
        entry = MemoryEntry(
            id=mem_id,
            content=content,
            category=category,
            scope=scope,
            importance=importance,
            embedding=embedding,
            created_at=now,
            last_accessed=now,
            access_count=0,
            access_times=[now],
            layer="working",
            superseded_by=None,
            source=source,
            pinned=pinned,
            metadata=metadata or {},
        )

        # Fetch existing memories once for both contradiction + link checks
        existing_memories = self._backend.get_all_active()

        # Contradiction check before storing
        self._check_contradictions(entry, existing_memories)

        # Persist
        self._backend.store(entry)

        # Create semantic links to existing memories (reuse same list)
        self._create_semantic_links(entry, existing_memories)

        logger.debug(
            f"Stored memory {mem_id[:8]}: cat={category}, imp={importance:.2f}, "
            f"scope={scope}"
        )
        return mem_id

    # ─── Recall ──────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        scope: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[ScoredMemory]:
        """
        Recall memories using ACT-R activation scoring.

        Returns top-K ScoredMemory objects sorted by activation score.
        """
        top_k = top_k or self._config.top_k
        now = self._now()

        # Encode query
        query_embedding = self._embedder.encode(query)

        # Get all active memories
        if scope:
            candidates = self._backend.get_by_scope(scope)
        else:
            candidates = self._backend.get_all_active()

        if not candidates:
            return []

        # Pre-compute link map + embedding cache in batch (avoids N+1 queries)
        link_map, embedding_cache = self._build_link_map_and_embeddings(candidates)

        # Score each memory
        scored = []
        for mem in candidates:
            components = self._compute_activation(
                mem, query_embedding, now, link_map, scope, embedding_cache
            )
            total = sum(components.values())
            scored.append(ScoredMemory(
                entry=mem,
                score=total,
                components=components,
            ))

        # Sort by score, take top-K
        scored.sort(key=lambda s: s.score, reverse=True)
        results = scored[:top_k]

        # Update access stats for recalled memories
        self._update_access_stats(results, now)

        # Hebbian link strengthening for co-recalled pairs
        self._strengthen_hebbian_links(results, now)

        return results

    # ─── ACT-R Activation ────────────────────────────────────────

    def _compute_activation(
        self,
        memory: MemoryEntry,
        query_embedding,
        now: float,
        link_map: Dict[str, List[MemoryLink]],
        scope: Optional[str] = None,
        embedding_cache: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute activation score with component breakdown.

        ACTIVATION = BASE_LEVEL + SPREADING + IMPORTANCE_BOOST + SCOPE_BOOST
        """
        cfg = self._config

        # 1. Base level: ACT-R ln(Σ tᵢ^(-d))
        base_level = self._actr_base_level(memory.access_times, now, cfg.d)

        # 2. Spreading activation: semantic + Hebbian
        semantic_sim = 0.0
        if memory.embedding is not None and query_embedding is not None:
            semantic_sim = cosine_similarity(query_embedding, memory.embedding)
        spreading = cfg.w_semantic * semantic_sim

        # Add Hebbian spreading (one hop)
        hebbian_spread = self._hebbian_spreading(
            memory.id, link_map, query_embedding,
            embedding_cache or {},
        )
        spreading += hebbian_spread

        # 3. Importance boost
        importance_boost = cfg.w_importance * memory.importance

        # 4. Scope boost — prefix match so project:hermes matches project:hermes/sub
        scope_boost = 0.0
        if scope and (memory.scope == scope or memory.scope.startswith(scope)):
            scope_boost = cfg.scope_multiplier * cfg.w_importance  # scaled to importance weight

        return {
            "base_level": base_level,
            "spreading": spreading,
            "importance_boost": importance_boost,
            "scope_boost": scope_boost,
        }

    def _actr_base_level(
        self, access_times: List[float], now: float, d: float
    ) -> float:
        """
        ACT-R base-level activation: ln(Σ tᵢ^(-d))

        tᵢ = time since access i (in seconds)
        d = decay parameter (typically 0.5)
        """
        if not access_times:
            return -10.0  # very low activation for never-accessed

        total = 0.0
        for t in access_times:
            elapsed = now - t
            if elapsed <= 0:
                elapsed = 0.001  # avoid division by zero / log issues
            total += elapsed ** (-d)

        if total <= 0:
            return -10.0

        return math.log(total)

    def _hebbian_spreading(
        self,
        memory_id: str,
        link_map: Dict[str, List[MemoryLink]],
        query_embedding,
        embedding_cache: Dict[str, Any],
    ) -> float:
        """
        One-hop Hebbian spreading activation.
        If linked memories are semantically similar to the query,
        this memory gets a boost proportional to link weight × similarity.

        Uses pre-fetched embedding_cache to avoid N+1 queries.
        """
        links = link_map.get(memory_id, [])
        if not links or query_embedding is None:
            return 0.0

        spread = 0.0
        for link in links:
            emb = embedding_cache.get(link.target_id)
            if emb is not None:
                sim = cosine_similarity(query_embedding, emb)
                spread += link.weight * sim

        # Cap spreading to avoid runaway values
        return min(spread, 0.5)

    def _build_link_map_and_embeddings(
        self, memories: List[MemoryEntry]
    ) -> tuple:
        """
        Build link map AND embedding cache in minimal SQL queries.
        Returns (link_map, embedding_cache).
        """
        # Single query for all links
        all_links = self._backend.get_all_links()
        link_map: Dict[str, List[MemoryLink]] = {}
        needed_ids: set = set()
        memory_ids = {m.id for m in memories}

        for link in all_links:
            if link.source_id in memory_ids:
                link_map.setdefault(link.source_id, []).append(link)
                needed_ids.add(link.target_id)

        # Build embedding cache from candidates + linked targets
        embedding_cache: Dict[str, Any] = {}
        for mem in memories:
            if mem.embedding is not None:
                embedding_cache[mem.id] = mem.embedding

        # Fetch any linked memories not in the candidate set
        missing_ids = list(needed_ids - set(embedding_cache))
        if missing_ids:
            linked_memories = self._backend.get_many(missing_ids)
            for mid, mem in linked_memories.items():
                if mem.embedding is not None:
                    embedding_cache[mid] = mem.embedding

        return link_map, embedding_cache

    # ─── Access Stats & Hebbian Strengthening ────────────────────

    def _update_access_stats(
        self, results: List[ScoredMemory], now: float
    ) -> None:
        """Update access stats for recalled memories."""
        max_times = self._config.max_access_times
        for scored in results:
            mem = scored.entry
            # Update access times (capped)
            access_times = mem.access_times + [now]
            if len(access_times) > max_times:
                access_times = access_times[-max_times:]

            self._backend.update(
                mem.id,
                last_accessed=now,
                access_count=mem.access_count + 1,
                access_times=access_times,
            )

    def _strengthen_hebbian_links(
        self, results: List[ScoredMemory], now: float
    ) -> None:
        """
        Strengthen Hebbian links between co-recalled memories.
        "Neurons that fire together wire together."
        """
        if len(results) < 2:
            return

        cfg = self._config
        # Only strengthen among top results (avoid quadratic explosion)
        top_results = results[:cfg.links_per_memory]

        for i, a in enumerate(top_results):
            for b in top_results[i + 1:]:
                # Compute activation product
                activation_product = a.score * b.score
                if activation_product <= 0:
                    continue

                delta_w = cfg.hebbian_learning_rate * activation_product

                # Check if link exists (in either direction)
                existing_links = self._backend.get_links(a.entry.id)
                existing = None
                for link in existing_links:
                    if link.target_id == b.entry.id:
                        existing = link
                        break

                if existing:
                    new_weight = min(existing.weight + delta_w, 1.0)
                    self._backend.update_link(
                        a.entry.id, b.entry.id,
                        weight=new_weight,
                        last_coactivated=now,
                    )
                elif activation_product > 0.1:  # threshold for new link
                    self._backend.create_link(
                        a.entry.id, b.entry.id,
                        weight=delta_w,
                        link_type="hebbian",
                    )
                    # Create reverse link too (bidirectional)
                    self._backend.create_link(
                        b.entry.id, a.entry.id,
                        weight=delta_w,
                        link_type="hebbian",
                    )

    # ─── Contradiction Detection ─────────────────────────────────

    def _check_contradictions(
        self,
        new_entry: MemoryEntry,
        active_memories: Optional[List[MemoryEntry]] = None,
    ) -> None:
        """
        Check for contradictions between new memory and existing ones.
        Stage 1 only (embedding similarity). No LLM calls.

        If high similarity + same category + same scope → mark old as superseded.
        """
        if new_entry.embedding is None:
            return

        threshold = self._config.contradiction_threshold
        if active_memories is None:
            active_memories = self._backend.get_all_active()

        for existing in active_memories:
            if existing.embedding is None:
                continue
            if existing.category != new_entry.category:
                continue
            if existing.scope != new_entry.scope:
                continue

            sim = cosine_similarity(new_entry.embedding, existing.embedding)
            if sim >= threshold:
                # High similarity + same category + same scope → likely contradiction
                # Newer supersedes older
                logger.info(
                    f"Contradiction detected: '{existing.content[:50]}...' "
                    f"superseded by '{new_entry.content[:50]}...' "
                    f"(similarity={sim:.3f})"
                )
                self._backend.update(
                    existing.id,
                    superseded_by=new_entry.id,
                )
                # Store contradiction metadata
                meta = new_entry.metadata.copy()
                meta["supersedes"] = existing.id
                meta["contradiction_similarity"] = float(sim)
                new_entry.metadata = meta

    # ─── Semantic Link Creation ──────────────────────────────────

    def _create_semantic_links(
        self,
        entry: MemoryEntry,
        active_memories: Optional[List[MemoryEntry]] = None,
    ) -> None:
        """Create semantic links to existing similar memories."""
        if entry.embedding is None:
            return

        cfg = self._config
        if active_memories is None:
            active_memories = self._backend.get_all_active()

        # Find similar memories
        similarities = []
        for existing in active_memories:
            if existing.id == entry.id:
                continue
            if existing.embedding is None:
                continue

            sim = cosine_similarity(entry.embedding, existing.embedding)

            # Link if same category and above threshold,
            # or if above high threshold regardless of category
            should_link = (
                (sim >= cfg.semantic_link_threshold and existing.category == entry.category)
                or sim >= cfg.high_link_threshold
            )
            if should_link:
                similarities.append((sim, existing))

        # Sort by similarity, take top links_per_memory
        similarities.sort(key=lambda x: x[0], reverse=True)
        for sim, existing in similarities[:cfg.links_per_memory]:
            self._backend.create_link(
                entry.id, existing.id,
                weight=sim * 0.5,  # initial weight proportional to similarity
                link_type="semantic",
            )
            # Bidirectional
            self._backend.create_link(
                existing.id, entry.id,
                weight=sim * 0.5,
                link_type="semantic",
            )

    # ─── Consolidation ───────────────────────────────────────────

    def consolidate(self) -> ConsolidationReport:
        """
        Run consolidation cycle:
        1. Promote working → core (high access + importance)
        2. Demote core → archive (low activation)
        3. Prune archive (below threshold)
        4. Decay + prune Hebbian links
        """
        report = ConsolidationReport()
        cfg = self._config
        now = self._now()

        # 1. Promote working → core
        working = self._backend.get_by_layer("working")
        for mem in working:
            if mem.pinned:
                continue
            if (
                mem.access_count >= cfg.core_promotion_count
                and mem.importance >= cfg.core_promotion_importance
            ):
                self._backend.update(mem.id, layer="core")
                report.promoted_to_core += 1
                report.details.append(
                    f"Promoted to core: {mem.content[:50]}..."
                )

        # 2. Demote core → archive
        # ACT-R base level is in log-space (typically -10 to +5).
        # Convert thresholds: ln(threshold) maps linear to log-space.
        # archive_threshold=0.1 → ln(0.1) ≈ -2.3 (demote if rarely accessed)
        # prune_threshold=0.01  → ln(0.01) ≈ -4.6 (prune if essentially forgotten)
        archive_bl_threshold = math.log(max(cfg.archive_threshold, 1e-10))
        prune_bl_threshold = math.log(max(cfg.prune_threshold, 1e-10))

        core = self._backend.get_by_layer("core")
        for mem in core:
            if mem.pinned:
                continue
            activation = self._actr_base_level(mem.access_times, now, cfg.d)
            if activation < archive_bl_threshold:
                self._backend.update(mem.id, layer="archive")
                report.demoted_to_archive += 1
                report.details.append(
                    f"Demoted to archive: {mem.content[:50]}..."
                )

        # 3. Prune archive
        archive = self._backend.get_by_layer("archive")
        for mem in archive:
            if mem.pinned:
                continue
            activation = self._actr_base_level(mem.access_times, now, cfg.d)
            if activation < prune_bl_threshold:
                self._backend.update(mem.id, superseded_by="pruned")
                report.pruned += 1
                report.details.append(
                    f"Pruned: {mem.content[:50]}..."
                )

        # 4. Decay and prune links
        self._backend.decay_links(cfg.link_decay_rate)
        report.links_decayed = True
        report.links_pruned = self._backend.prune_links(cfg.link_prune_threshold)

        logger.info(
            f"Consolidation: +{report.promoted_to_core} core, "
            f"-{report.demoted_to_archive} archive, "
            f"-{report.pruned} pruned, "
            f"-{report.links_pruned} links pruned"
        )

        return report

    # ─── Utilities ───────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        stats = self._backend.stats()
        stats["embedding_backend"] = self._embedder.backend_name
        stats["config_profile"] = self._config.to_dict()
        return stats

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        return self._backend.get(memory_id)

    def resolve_contradiction(
        self, memory_id: str, keep: bool = True
    ) -> None:
        """
        Manually resolve a contradiction.
        If keep=True, un-supersede this memory.
        If keep=False, confirm the supersession.
        """
        mem = self._backend.get(memory_id)
        if not mem:
            return

        if keep:
            self._backend.update(memory_id, superseded_by=None)
        # If not keeping, it stays superseded (no action needed)

    def reset(self) -> None:
        """Clear all data (for benchmarks)."""
        self._backend.reset()
        self._embedder.reset()
        self._simulated_time_offset = 0.0
