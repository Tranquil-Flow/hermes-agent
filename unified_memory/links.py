"""
Hebbian Link Management for the Unified Memory System.

Handles:
- Semantic link creation (embedding similarity)
- Keyword-overlap link seeding (Jaccard similarity)
- Hebbian strengthening (GloVe + Ebbinghaus + homeostasis)
- Link queries and map building
"""

from __future__ import annotations

import math
import sqlite3
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from unified_memory.types import MemoryFact, MemoryLink

logger = logging.getLogger(__name__)

# Stop words for keyword link seeding
_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'with', 'and',
    'or', 'for', 'in', 'on', 'at', 'to', 'of', 'it', 'its', 'by',
    'as', 'that', 'this', 'from', 'has', 'have', 'be', 'been',
    'what', 'how', 'where', 'when', 'which', 'who', 'do', 'does',
    'did', 'not', 'but', 'so', 'if',
})


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two numpy arrays.

    Handles different-length vectors (e.g. from TF-IDF) by zero-padding
    the shorter one — same approach as cognitive_memory.embeddings.
    """
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    # Pad shorter vector with zeros (new vocab dimensions = 0 for older docs)
    if a.shape[0] != b.shape[0]:
        max_len = max(a.shape[0], b.shape[0])
        if a.shape[0] < max_len:
            a = np.pad(a, (0, max_len - a.shape[0]))
        else:
            b = np.pad(b, (0, max_len - b.shape[0]))
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercase content words."""
    return {w.lower() for w in text.split() if w.lower() not in _STOP_WORDS and len(w) > 1}


# ─── Link Queries ──────────────────────────────────────────────


def get_all_links(conn: sqlite3.Connection) -> List[MemoryLink]:
    """Fetch all links from the database."""
    rows = conn.execute(
        "SELECT source_id, target_id, strength, npmi, co_occurrence_count, "
        "link_type, last_updated FROM um_links"
    ).fetchall()
    return [MemoryLink(
        source_id=r["source_id"], target_id=r["target_id"],
        strength=r["strength"], npmi=r["npmi"] or 0.0,
        co_occurrence_count=r["co_occurrence_count"] or 0,
        link_type=r["link_type"] or "hebbian",
        last_updated=r["last_updated"] or 0.0,
    ) for r in rows]


def get_links_for(conn: sqlite3.Connection, fact_id: str) -> List[MemoryLink]:
    """Fetch outgoing links for a specific fact."""
    rows = conn.execute(
        "SELECT source_id, target_id, strength, npmi, co_occurrence_count, "
        "link_type, last_updated FROM um_links WHERE source_id = ?",
        (fact_id,)
    ).fetchall()
    return [MemoryLink(
        source_id=r["source_id"], target_id=r["target_id"],
        strength=r["strength"], npmi=r["npmi"] or 0.0,
        co_occurrence_count=r["co_occurrence_count"] or 0,
        link_type=r["link_type"] or "hebbian",
        last_updated=r["last_updated"] or 0.0,
    ) for r in rows]


def build_link_map_and_embeddings(
    conn: sqlite3.Connection,
    fact_ids: Set[str],
) -> Tuple[Dict[str, List[MemoryLink]], Dict[str, np.ndarray]]:
    """Build link map AND embedding cache in minimal SQL queries.

    Returns (link_map, embedding_cache).
    """
    all_links = get_all_links(conn)
    link_map: Dict[str, List[MemoryLink]] = {}
    needed_ids: Set[str] = set()

    for link in all_links:
        if link.source_id in fact_ids:
            link_map.setdefault(link.source_id, []).append(link)
            needed_ids.add(link.target_id)

    # Build embedding cache from candidates
    embedding_cache: Dict[str, np.ndarray] = {}
    all_needed = fact_ids | needed_ids
    if all_needed:
        placeholders = ",".join("?" * len(all_needed))
        rows = conn.execute(
            f"SELECT id, embedding FROM um_facts WHERE id IN ({placeholders})",
            list(all_needed)
        ).fetchall()
        for r in rows:
            if r["embedding"]:
                embedding_cache[r["id"]] = np.frombuffer(r["embedding"], dtype=np.float32)

    return link_map, embedding_cache


# ─── Link Creation ─────────────────────────────────────────────


def create_semantic_links(
    conn: sqlite3.Connection,
    fact_id: str,
    fact_embedding: np.ndarray,
    now: float,
    threshold: float = 0.70,
    high_threshold: float = 0.90,
    max_links: int = 5,
) -> int:
    """Create semantic links from a new fact to existing facts based on embedding similarity."""
    rows = conn.execute(
        "SELECT id, embedding, category FROM um_facts "
        "WHERE id != ? AND status IN ('active', 'cold') AND embedding IS NOT NULL",
        (fact_id,)
    ).fetchall()

    candidates = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32)
        sim = cosine_similarity(fact_embedding, emb)
        candidates.append((r["id"], sim))

    # Sort by similarity descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    created = 0
    for other_id, sim in candidates[:max_links * 2]:  # Check more than we need
        if created >= max_links:
            break
        if sim >= threshold:
            _upsert_link(conn, fact_id, other_id, sim * 0.5, now, "semantic")
            created += 1

    return created


def create_keyword_links(
    conn: sqlite3.Connection,
    fact_id: str,
    fact_content: str,
    now: float,
    threshold: float = 0.15,
    min_shared: int = 3,
    max_recent: int = 50,
) -> int:
    """Create keyword-overlap links to densify the graph for PPR walks."""
    fact_tokens = _tokenize(fact_content)
    if len(fact_tokens) < min_shared:
        return 0

    # Get recent active facts
    rows = conn.execute(
        "SELECT id, content FROM um_facts "
        "WHERE id != ? AND status IN ('active', 'cold') "
        "ORDER BY created_at DESC LIMIT ?",
        (fact_id, max_recent)
    ).fetchall()

    # Check existing links to avoid duplicates
    existing_targets = {
        r["target_id"] for r in conn.execute(
            "SELECT target_id FROM um_links WHERE source_id = ?", (fact_id,)
        ).fetchall()
    }

    created = 0
    for r in rows:
        if r["id"] in existing_targets:
            continue
        other_tokens = _tokenize(r["content"])
        shared = fact_tokens & other_tokens
        if len(shared) >= min_shared:
            union = fact_tokens | other_tokens
            jaccard = len(shared) / len(union) if union else 0.0
            if jaccard >= threshold:
                _upsert_link(conn, fact_id, r["id"], jaccard * 0.3, now, "keyword")
                created += 1

    return created


# ─── Link Strengthening ───────────────────────────────────────


def strengthen_hebbian_links(
    conn: sqlite3.Connection,
    co_recalled_ids: List[Tuple[str, float]],
    now: float,
    learning_rate: float = 0.12,
    glove_xmax: int = 100,
    strength_rate: float = 0.2,
    max_links: int = 5,
    enable_homeostasis: bool = True,
    homeostasis_target: float = 0.7,
) -> None:
    """Strengthen Hebbian links between co-recalled facts.

    Uses GloVe frequency weighting + Ebbinghaus decay + Turrigiano homeostasis.
    """
    if len(co_recalled_ids) < 2:
        return

    xmax_norm = glove_xmax ** 0.75
    top_results = co_recalled_ids[:max_links]
    updated: Dict[Tuple[str, str], float] = {}

    for i, (id_a, score_a) in enumerate(top_results):
        for id_b, score_b in top_results[i + 1:]:
            activation_product = score_a * score_b
            if activation_product <= 0:
                continue

            existing = _get_link(conn, id_a, id_b)
            if existing:
                co_count = existing.co_occurrence_count + 1
                co_signal = (min(co_count, glove_xmax) ** 0.75) / xmax_norm
                strength = 1.0 + strength_rate * math.log1p(co_count)
                delta_w = learning_rate * activation_product * co_signal * strength
                new_weight = min(existing.strength + delta_w, 1.0)

                conn.execute(
                    "UPDATE um_links SET strength=?, co_occurrence_count=?, last_updated=? "
                    "WHERE source_id=? AND target_id=?",
                    (new_weight, co_count, now, id_a, id_b)
                )
                updated[(id_a, id_b)] = new_weight
            else:
                initial_weight = min(learning_rate * activation_product, 0.3)
                _upsert_link(conn, id_a, id_b, initial_weight, now, "hebbian")
                updated[(id_a, id_b)] = initial_weight

    # Turrigiano homeostasis
    if enable_homeostasis and updated:
        _apply_homeostasis(conn, updated, homeostasis_target)

    conn.commit()


def decay_all_links(conn: sqlite3.Connection, decay_rate: float = 0.05) -> int:
    """Decay all link strengths by decay_rate. Prune links below 0.01."""
    conn.execute(
        "UPDATE um_links SET strength = strength * ?",
        (1.0 - decay_rate,)
    )
    result = conn.execute("DELETE FROM um_links WHERE strength < 0.01")
    conn.commit()
    return result.rowcount


# ─── Internal Helpers ──────────────────────────────────────────


def _upsert_link(
    conn: sqlite3.Connection,
    source_id: str, target_id: str,
    strength: float, now: float,
    link_type: str,
) -> None:
    """Insert or update a link. Creates bidirectional links."""
    conn.execute(
        "INSERT INTO um_links (source_id, target_id, strength, link_type, last_updated, co_occurrence_count) "
        "VALUES (?, ?, ?, ?, ?, 1) "
        "ON CONFLICT(source_id, target_id) DO UPDATE SET "
        "strength = MAX(um_links.strength, excluded.strength), "
        "last_updated = excluded.last_updated",
        (source_id, target_id, strength, link_type, now)
    )
    # Reverse direction
    conn.execute(
        "INSERT INTO um_links (source_id, target_id, strength, link_type, last_updated, co_occurrence_count) "
        "VALUES (?, ?, ?, ?, ?, 1) "
        "ON CONFLICT(source_id, target_id) DO UPDATE SET "
        "strength = MAX(um_links.strength, excluded.strength), "
        "last_updated = excluded.last_updated",
        (target_id, source_id, strength, link_type, now)
    )


def _get_link(conn: sqlite3.Connection, source_id: str, target_id: str) -> Optional[MemoryLink]:
    """Get a specific link if it exists."""
    row = conn.execute(
        "SELECT source_id, target_id, strength, npmi, co_occurrence_count, "
        "link_type, last_updated FROM um_links WHERE source_id=? AND target_id=?",
        (source_id, target_id)
    ).fetchone()
    if not row:
        return None
    return MemoryLink(
        source_id=row["source_id"], target_id=row["target_id"],
        strength=row["strength"], npmi=row["npmi"] or 0.0,
        co_occurrence_count=row["co_occurrence_count"] or 0,
        link_type=row["link_type"] or "hebbian",
        last_updated=row["last_updated"] or 0.0,
    )


def _apply_homeostasis(
    conn: sqlite3.Connection,
    updated: Dict[Tuple[str, str], float],
    target: float,
) -> None:
    """Turrigiano homeostatic scaling — prevent hub nodes from absorbing all weight."""
    node_weights: Dict[str, List[float]] = defaultdict(list)
    for (src, tgt), weight in updated.items():
        node_weights[src].append(weight)
        node_weights[tgt].append(weight)

    for node_id, weights in node_weights.items():
        mean_w = sum(weights) / len(weights) if weights else 0.0
        if mean_w > target:
            scale = target / mean_w
            conn.execute(
                "UPDATE um_links SET strength = strength * ? "
                "WHERE source_id = ? OR target_id = ?",
                (scale, node_id, node_id)
            )
