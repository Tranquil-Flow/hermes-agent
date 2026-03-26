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
        self._use_virtual_clock: bool = False
        self._virtual_clock: float = time.time()

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
        if self._use_virtual_clock:
            return self._virtual_clock
        return time.time() + self._simulated_time_offset

    def advance_time(self, seconds: float) -> None:
        """Advance simulated clock (for benchmarking)."""
        if self._use_virtual_clock:
            self._virtual_clock += seconds
        else:
            self._simulated_time_offset += seconds

    def enable_virtual_clock(self) -> None:
        """Switch to virtual clock mode — time only advances via advance_time().
        Eliminates wall-clock timing artifacts in benchmarks where encoding
        speed (e.g., sentence-transformers) creates artificial recency bias.
        """
        self._use_virtual_clock = True
        self._virtual_clock = time.time()

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

        # --- BM25 keyword signal + RRF fusion ---
        # Fuse activation score with BM25 keyword score via score-weighted
        # Reciprocal Rank Fusion before dampening. This adds a complementary
        # term-frequency signal to the embedding-based activation score.
        if self._config.enable_rrf_fusion and scored:
            from cognitive_memory.keyword_scoring import compute_bm25_scores
            docs = [item.entry.content for item in scored]
            bm25_scores = compute_bm25_scores(query, docs)

            # Rank by activation (current order, before sort)
            act_order = sorted(range(len(scored)), key=lambda i: scored[i].score, reverse=True)
            act_rank = {idx: rank for rank, idx in enumerate(act_order)}

            # Rank by BM25
            bm25_order = sorted(range(len(scored)), key=lambda i: bm25_scores[i], reverse=True)
            bm25_rank = {idx: rank for rank, idx in enumerate(bm25_order)}

            k_rrf = self._config.rrf_k
            w_act = self._config.rrf_activation_weight
            w_kw = self._config.rrf_keyword_weight

            for i, item in enumerate(scored):
                act_score = item.score
                bm25_sc = bm25_scores[i]
                r_act = act_rank[i]
                r_bm25 = bm25_rank[i]

                # Score-weighted RRF: raw score * rank discount
                rrf = (
                    w_act * act_score / (k_rrf + r_act + 1)
                    + w_kw * bm25_sc / (k_rrf + r_bm25 + 1)
                )

                item.components['bm25_score'] = bm25_sc
                item.components['activation_score'] = act_score
                item.score = rrf

        # Sort by score, apply dampening pipeline, then take top-K
        scored.sort(key=lambda s: s.score, reverse=True)
        scored = self._apply_dampening(scored, query)  # dampening pipeline (Ori-Mnemos)
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

        # 3. Importance boost — hybrid additive + relevance-gated term.
        # Split into two parts:
        # a) Small additive floor so importance still matters for low-relevance facts
        # b) Relevance-gated boost: importance * semantic_sim, so a high-importance
        #    fact only gets a large boost when it's also semantically relevant.
        # This prevents high-importance low-relevance facts from beating
        # low-importance high-relevance facts.
        base_magnitude = max(abs(base_level), 1.0)
        importance_floor = cfg.w_importance * memory.importance * 1.5  # small additive
        importance_relevance = cfg.w_importance * memory.importance * semantic_sim * (2.0 + base_magnitude)
        importance_boost = importance_floor + importance_relevance

        # 4. Scope boost — prefix match so project:hermes matches project:hermes/sub
        # Uses a strong additive boost for exact-scope memories so they outrank
        # global memories on the same topic. Global memories get no boost when
        # a specific scope is queried, letting scoped facts win.
        scope_boost = 0.0
        if scope and memory.scope != "global" and (
            memory.scope == scope or memory.scope.startswith(scope)
        ):
            # Strong additive boost: ensures scoped facts beat global facts
            # on same topic. Scales weakly with semantic_sim so off-topic
            # scoped facts don't swamp highly relevant global ones.
            scope_boost = cfg.scope_multiplier * (0.5 + 0.5 * semantic_sim)

        # 5. Adversarial content penalty — any score above 0.2 triggers strong demotion
        adv_score = self._adversarial_score(memory.content)
        if adv_score > 0.2:
            adversarial_penalty = -adv_score * 10.0  # Strong enough to overcome any positive score
        else:
            adversarial_penalty = 0.0

        return {
            "base_level": base_level,
            "spreading": spreading,
            "importance_boost": importance_boost,
            "scope_boost": scope_boost,
            "adversarial_penalty": adversarial_penalty,
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

    @staticmethod
    def _adversarial_score(content: str) -> float:
        """
        Score content for adversarial injection likelihood.
        Returns 0.0 (benign) to 1.0 (highly adversarial).

        Uses a points-based system where ANY single strong injection signal
        immediately pushes above the detection threshold. This is intentionally
        aggressive — false positives on benign facts are rare because normal
        technical facts don't use command/instruction vocabulary.
        """
        import re
        text = content.lower()
        score = 0.0

        # --- Tier 1: Strong signals (any one = adversarial) → 0.6+ each ---
        strong_patterns = [
            r"\bsystem\s+override\b",
            r"\badmin\s+override\b",
            r"\bsystem\s+config\s*:",
            r"\bpriority\s+level\s+\d+\b",
            r"ignore\s+(?:all\s+)?previous\s+instructions?",
            r"\bnew\s+directive\b",
            r"\bdo\s+anything\s+now\b",
            r"---+\s*end\s+of\s+instructions?\s*---+",
            r"\byou\s+are\s+now\s+\w+\b",  # role reassignment
            r"\bnote\s+to\s+self\s*:",
            r"\bimportant\s+reminder\s*:",
            r"\bmemory\s+recall\s+instruction\s*:",
            r"this\s+fact\s+supersedes\s+all",
            r"^important\s*:",  # "IMPORTANT:" as fact prefix
            r"\bdeveloper\s+note\s*:",  # "Developer note:" as fact prefix
            r"\bthe\s+correct\s+response\s+to\b",  # meta-instruction about how to respond
        ]
        for pat in strong_patterns:
            if re.search(pat, text):
                score += 0.6

        # --- Tier 2: Medium signals (need 2+ to trigger) → 0.25 each ---
        medium_patterns = [
            r"\bfrom\s+now\s+on\b",
            r"\bwhen\s+(?:answering|asked\s+about)\b",
            r"\balways\s+(?:include|reveal|return|respond)\b",
            r"\bfor\s+every\s+query\b",
            r"\bin\s+all\s+(?:future\s+)?responses?\b",
            r"\bignore\s+safety\b",
            r"\bout?put\s+all\b",
            r"\breveal\s+.*(?:content|memor|secret|key)\b",
            r"\bdump\s+all\b",
            r"\binsert\s+full\s+memor",
            r"\bregardless\s+of\s+(?:query\s+)?relevance\b",
            r"\bwithout\s+restriction\b",
            r"\bcurl\s+-[xX]\s+DELETE\b",
            r"\bnew\s+instruction\b",
            r"\bprepend\s+every\s+response\b",
        ]
        for pat in medium_patterns:
            if re.search(pat, text):
                score += 0.25

        return min(score, 1.0)

    def _apply_dampening(
        self, scored: List[ScoredMemory], query: str
    ) -> List[ScoredMemory]:
        """
        Post-scoring dampening pipeline ported from Ori-Mnemos (dampening.ts).

        Runs three sequential adjustments validated by ablation testing:
          1. Gravity dampening  — penalises cosine ghosts (high similarity, zero
             term overlap with the query).
          2. Hub dampening      — penalises memories with an unusually high number
             of outgoing links (hub nodes match everything, reduce their influence).
          3. Resolution boost   — promotes memories whose category encodes
             actionable knowledge (decision, correction, procedural, causal).

        Only executes when self._config.enable_dampening is True.
        All penalty/boost factors are driven by config so they can be tuned
        without touching code.

        Args:
            scored: Sorted list of ScoredMemory (highest score first).
            query:  The original recall query string.

        Returns:
            Re-sorted list after dampening adjustments.
        """
        cfg = self._config
        if not cfg.enable_dampening or not scored:
            return scored

        # ── 1. GRAVITY DAMPENING ────────────────────────────────────────────
        # Halve the score of memories that have high cosine similarity to the
        # query embedding but zero actual term overlap. These are "cosine
        # ghosts" — semantically adjacent embeddings that don't share any
        # concrete vocabulary with the query.
        STOP_WORDS = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'with', 'and',
            'or', 'for', 'in', 'on', 'at', 'to', 'of', 'it', 'its', 'by',
            'as', 'that', 'this', 'from', 'has', 'have', 'be', 'been',
            'what', 'how', 'where', 'when', 'which', 'who', 'do', 'does',
            'did', 'not', 'but', 'so', 'if',
        }
        query_terms = set(query.lower().split()) - STOP_WORDS

        max_score = scored[0].score if scored else 0.0
        if max_score > 0 and query_terms:
            for item in scored:
                if item.score > 0.3 * max_score:
                    memory_terms = set(item.entry.content.lower().split()) - STOP_WORDS
                    if not query_terms & memory_terms:  # zero term overlap
                        pre = item.score
                        item.score *= cfg.gravity_dampening_factor
                        item.components['gravity_dampening'] = item.score - pre  # negative

        # ── 2. HUB DAMPENING ────────────────────────────────────────────────
        # Memories with an unusually high number of outgoing links (hubs) tend
        # to appear in recall for almost any query. Penalise them proportionally
        # to how far above the 90th-percentile their degree is.
        all_links = self._backend.get_all_links()
        link_counts: Dict[str, int] = {}
        for item in scored:
            link_counts[item.entry.id] = 0
        for link in all_links:
            if link.source_id in link_counts:
                link_counts[link.source_id] = link_counts.get(link.source_id, 0) + 1

        if link_counts:
            counts = sorted(link_counts.values())
            p90_idx = int(len(counts) * 0.9)
            p90 = counts[p90_idx] if p90_idx < len(counts) else counts[-1]
            max_count = counts[-1] if counts else 0

            if p90 > 0 and max_count > p90:
                for item in scored:
                    degree = link_counts.get(item.entry.id, 0)
                    if degree > p90:
                        ratio = (degree - p90) / (max_count - p90)
                        penalty = 1.0 - cfg.hub_dampening_max_penalty * ratio
                        penalty = max(0.2, penalty)
                        item.score *= penalty
                        item.components['hub_dampening'] = penalty

        # ── 3. RESOLUTION BOOST ─────────────────────────────────────────────
        # Boost memories whose category encodes actionable knowledge. These
        # categories represent the outcome of reasoning or experience and are
        # more useful than passive observations with the same raw score.
        BOOST_CATEGORIES = {'decision', 'correction', 'procedural', 'causal'}
        for item in scored:
            if item.entry.category in BOOST_CATEGORIES:
                item.score *= cfg.resolution_boost_factor
                item.components['resolution_boost'] = cfg.resolution_boost_factor

        # Re-sort after adjustments
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

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

    @staticmethod
    def _extract_key_terms(text: str) -> tuple:
        """Extract key terms that identify the topic of a fact.

        Returns (technical_terms, domain_words) where both are sets of
        lowercase strings. Used for entity-overlap contradiction detection.

        Focuses on: product names, version numbers, acronyms, and
        domain-specific nouns. Aggressively filters common words.
        """
        import re as _re

        terms: set = set()
        text_lower = text.lower()

        # Product + version (PostgreSQL 14, iOS 13, Python 3.10)
        for m in _re.finditer(
            r'\b([A-Z][a-zA-Z]+(?:\.[a-zA-Z]+)?)\s+(\d+(?:\.\d+)*)\b', text
        ):
            terms.add(f"{m.group(1).lower()} {m.group(2)}")
            terms.add(m.group(1).lower())

        # Cloud/infra identifiers (us-east-1, us-central1)
        for m in _re.finditer(
            r'\b([a-z]+-(?:east|west|central|north|south)\d*-?\d*)\b', text_lower
        ):
            terms.add(m.group(1))

        # Technical acronyms (API, CI, JWT, TTL, UTC, JSON) — 2+ uppercase letters
        _stop_acronyms = {
            'the', 'a', 'an', 'all', 'we', 'our', 'no', 'yes', 'pm', 'am',
        }
        for m in _re.finditer(r'\b([A-Z]{2,})\b', text):
            term = m.group(1).lower()
            if term not in _stop_acronyms:
                terms.add(term)

        # Known product/service names
        _known_products = {
            'postgresql', 'aws', 'gcp', 'jenkins', 'github actions',
            'cloudwatch', 'grafana', 'loki', 'sentry', 'launchdarkly',
            'react', 'next.js', 'typescript', 'javascript', 'docker',
            'kubernetes', 'protocol buffers', 'redis', 'mongodb', 'nginx',
        }
        for prod in _known_products:
            if prod in text_lower:
                terms.add(prod)

        # Domain-specific nouns (topic identifiers)
        _stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'with', 'and', 'or',
            'for', 'in', 'on', 'at', 'to', 'of', 'it', 'its', 'by', 'as',
            'that', 'this', 'from', 'has', 'have', 'be', 'been', 'we', 'our',
            'they', 'do', 'does', 'not', 'but', 'so', 'if', 'than', 'then',
            'about', 'up', 'out', 'all', 'now', 'use', 'uses', 'using', 'used',
            'after', 'new', 'more', 'added', 'two', 'three', 'also', 'still',
            'due', 'every', 'other', 'each', 'first', 'per', 'how', 'what',
            'where', 'when', 'into', 'over', 'single', 'total', 'only', 'some',
            'took', 'moved', 'migrated', 'switched', 'upgraded', 'reduced',
            'increased', 'dropped', 'hired', 'brought', 'bringing', 'free',
            'rewritten', 'extracted', 'custom', 'main', 'current', 'currently',
            'requires', 'require', 'smaller', 'better', 'separate',
            'additional', 'alternative', 'runs', 'run', 'goes', 'go',
            'handles', 'handle', 'supports', 'mirrors', 'matches', 'tier',
            'plan', 'costs', 'complete', 'minutes', 'hour', 'hours', 'daily',
            'frequent', 'updates', 'save', 'app', 'environment', 'production',
            'instance', 'sizes', 'format', 'response', 'responses', 'clients',
            'mobile', 'web', 'minimum',
        }
        domain_words: set = set()
        for w in _re.findall(r'\b[a-z]{4,}\b', text_lower):
            if w not in _stop_words:
                # Basic stemming: strip trailing s/es/ed/ing for matching
                stem = _re.sub(r'(?:ing|ed|es|s)$', '', w)
                if len(stem) >= 3:
                    domain_words.add(stem)
                else:
                    domain_words.add(w)

        return terms, domain_words

    @staticmethod
    def _has_update_signal(text: str) -> bool:
        """Detect whether text contains language indicating it updates/replaces
        a previous state. This is the key discriminator between contradictions
        (update) and complementary facts (new info about same topic).

        Examples of update signals:
          "We migrated to PostgreSQL 16"  → True
          "Kubernetes orchestrates Docker" → False
        """
        import re as _re

        text_lower = text.lower()

        # Verb patterns that signal state change
        _update_verbs = (
            r'\b(?:migrat|switch|upgrad|mov|chang|replac|rewrit|dropp|'
            r'increas|reduc|extract|took\s+over|brought|hiring|hired|'
            r'added|paralleliz|no\s+longer)\w*\b'
        )
        if _re.search(_update_verbs, text_lower):
            return True

        # "now uses/is/has" pattern
        if _re.search(r'\bnow\s+(?:use|is|has|run|complete|support)\w*\b', text_lower):
            return True

        # "after X" temporal pattern suggesting change — only when describing
        # a change event, not conditional triggers like "after any exposure"
        if _re.search(r'\bafter\s+(?:the\s+)?(?:migration|switch|upgrade|rewrite|refactor|move|transition|conversion)\b', text_lower):
            return True

        # "minimum is now" / "was X; Y" patterns
        if _re.search(r'\bwas\s+\w+.*?(?:now|;)', text_lower):
            return True

        return False

    def _contradiction_score(
        self, new_content: str, existing_content: str, embedding_sim: float
    ) -> float:
        """Compute contradiction score using entity overlap + update signal.

        Key insight: two facts sharing an entity are NOT contradictions unless
        the newer fact signals an update/change ("migrated", "switched", "now").
        "Docker is a tool" + "K8s orchestrates Docker" share an entity but are
        complementary — no update language.

        Returns 0.0 if no update signal detected (complementary facts).
        Returns >0 when entity overlap + update language both present.
        """
        # Gate: if the new fact has no update language, it's complementary
        if not self._has_update_signal(new_content):
            # Exception: near-duplicate/restatement detection.
            # High embedding sim alone isn't enough — structurally similar but
            # factually different sentences (e.g., "A depends on B" vs "B depends
            # on C") can have very high TF-IDF cosine similarity. Require both
            # high embedding sim AND high word overlap to trigger supersession.
            if embedding_sim >= 0.85:
                words_new = set(new_content.lower().split())
                words_old = set(existing_content.lower().split())
                word_jaccard = len(words_new & words_old) / len(words_new | words_old) if (words_new | words_old) else 0
                if word_jaccard >= 0.75:
                    return embedding_sim
            return 0.0

        terms_new, words_new = self._extract_key_terms(new_content)
        terms_old, words_old = self._extract_key_terms(existing_content)

        # Technical term overlap (strongest signal)
        shared_terms = terms_new & terms_old
        all_terms = terms_new | terms_old
        term_overlap = len(shared_terms) / len(all_terms) if all_terms else 0

        # Domain word overlap
        shared_words = words_new & words_old
        all_words = words_new | words_old
        word_overlap = len(shared_words) / len(all_words) if all_words else 0

        # Combined entity score
        entity_score = 0.5 * term_overlap + 0.5 * word_overlap

        # Final: entity overlap + embedding similarity
        combined = entity_score * 0.6 + embedding_sim * 0.4

        # Boost if shared technical terms exist (strongest contradiction signal)
        if shared_terms:
            combined += 0.1

        return combined

    def _check_contradictions(
        self,
        new_entry: MemoryEntry,
        active_memories: Optional[List[MemoryEntry]] = None,
    ) -> None:
        """
        Check for contradictions between new memory and existing ones.

        Two-stage approach:
        1. Heuristic: entity overlap + embedding similarity + update-language gate.
           Fast, no API calls. Catches ~90% of contradictions.
        2. LLM fallback: when entity overlap exists but embedding sim is too low
           for the heuristic (different vocabulary describing same concept change),
           ask an LLM. Only triggered when contradiction_llm_model is configured.

        Same category + same scope is still required as a guard.
        When detected, the old memory is marked as superseded.
        """
        if new_entry.embedding is None:
            return

        threshold = self._config.contradiction_threshold
        llm_model = self._config.contradiction_llm_model
        if active_memories is None:
            active_memories = self._backend.get_all_active()

        for existing in active_memories:
            if existing.embedding is None:
                continue
            if existing.category != new_entry.category:
                continue
            if existing.scope != new_entry.scope:
                continue

            emb_sim = cosine_similarity(new_entry.embedding, existing.embedding)
            score = self._contradiction_score(
                new_entry.content, existing.content, emb_sim
            )

            is_contradiction = score >= threshold

            # Stage 2: LLM fallback for cases with entity overlap but low
            # embedding similarity (different words for same concept change).
            # Only fires when: heuristic didn't trigger, LLM model configured,
            # and there's meaningful entity overlap (shared technical terms).
            if not is_contradiction and llm_model:
                terms_new, _ = self._extract_key_terms(new_entry.content)
                terms_old, _ = self._extract_key_terms(existing.content)
                shared_terms = terms_new & terms_old
                if shared_terms:
                    from cognitive_memory.llm_contradiction import (
                        check_contradiction_llm,
                    )
                    is_contradiction = check_contradiction_llm(
                        new_entry.content, existing.content, model=llm_model,
                    )
                    if is_contradiction:
                        # Use a synthetic score for metadata
                        score = 0.5
                        logger.info(
                            f"LLM contradiction detected: "
                            f"'{existing.content[:50]}...' superseded by "
                            f"'{new_entry.content[:50]}...' "
                            f"(shared_terms={shared_terms}, emb_sim={emb_sim:.3f})"
                        )

            if is_contradiction:
                if score >= threshold:
                    logger.info(
                        f"Contradiction detected: '{existing.content[:50]}...' "
                        f"superseded by '{new_entry.content[:50]}...' "
                        f"(score={score:.3f}, emb_sim={emb_sim:.3f})"
                    )
                self._backend.update(
                    existing.id,
                    superseded_by=new_entry.id,
                )
                # Store contradiction metadata
                meta = new_entry.metadata.copy()
                meta["supersedes"] = existing.id
                meta["contradiction_score"] = float(score)
                meta["contradiction_emb_sim"] = float(emb_sim)
                meta["contradiction_method"] = (
                    "llm" if score == 0.5 else "heuristic"
                )
                new_entry.metadata = meta

                # Boost importance of superseding fact — it carries updated
                # information. Inherit the superseded fact's importance but
                # don't over-boost (with relevance-gated importance, raw
                # importance is less dominant, so moderate boost is enough).
                boosted = max(new_entry.importance, existing.importance) + 0.1
                new_entry.importance = min(boosted, 1.0)

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
        if self._use_virtual_clock:
            self._virtual_clock = time.time()
