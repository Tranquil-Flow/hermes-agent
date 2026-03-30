"""
Tests for Inverse Propensity Scoring (IPS) debiasing in retrieval.py.

Covers:
- Under-retrieved facts (low access_count) receive a positive IPS boost.
- Over-retrieved facts (high access_count / propensity > 0.8) get a flat penalty.
- When fewer than 5 facts have access_count > 0, IPS is a no-op.
"""

from __future__ import annotations

import time

import pytest

from unified_memory.config import UnifiedMemoryConfig
from unified_memory.retrieval import apply_ips_debiasing
from unified_memory.schema import get_connection
from unified_memory.types import FactType, MemoryFact, ScoredFact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(enable_ips: bool = True) -> UnifiedMemoryConfig:
    cfg = UnifiedMemoryConfig()
    cfg.enable_ips = enable_ips
    return cfg


def _make_fact(fact_id: str) -> MemoryFact:
    return MemoryFact(
        id=fact_id,
        content=f"Fact content for {fact_id}",
        embedding=None,
    )


def _make_scored(fact_id: str, score: float = 1.0) -> ScoredFact:
    return ScoredFact(fact=_make_fact(fact_id), score=score, components={})


def _seed_db(conn, facts: list[tuple[str, int]]) -> None:
    """Insert (fact_id, access_count) rows into um_facts."""
    now = time.time()
    for fact_id, access_count in facts:
        conn.execute(
            """
            INSERT INTO um_facts
                (id, content, type, target, layer,
                 access_count, created_at, updated_at, last_accessed)
            VALUES (?, ?, 'V', 'general', 'working', ?, ?, ?, ?)
            """,
            (fact_id, f"Fact {fact_id}", access_count, now, now, now),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ips_boosts_under_retrieved():
    """A fact with 0 (or few) accesses should receive a positive IPS adjustment."""
    conn = get_connection(":memory:")

    # 6 facts: one with 0 accesses, five with moderate accesses (satisfies the >=5 gate)
    facts = [
        ("rare-001", 0),
        ("hot-001", 50),
        ("hot-002", 45),
        ("hot-003", 40),
        ("hot-004", 35),
        ("hot-005", 30),
    ]
    _seed_db(conn, facts)

    scored = [_make_scored(fid, score=1.0) for fid, _ in facts]
    cfg = _cfg(enable_ips=True)

    original_score = scored[0].score  # rare-001, access_count=0
    apply_ips_debiasing(scored, conn, cfg)

    rare_item = next(s for s in scored if s.fact.id == "rare-001")
    assert rare_item.score > original_score, (
        f"Expected score boost for under-retrieved fact; "
        f"before={original_score}, after={rare_item.score}"
    )
    assert "ips_adjustment" in rare_item.components
    assert rare_item.components["ips_adjustment"] > 0


def test_ips_penalizes_over_retrieved():
    """A fact with many accesses (propensity > 0.8) should receive a penalty."""
    conn = get_connection(":memory:")

    # hot-001 has max accesses (propensity=1.0 > 0.8)
    facts = [
        ("hot-001", 100),
        ("mid-001", 10),
        ("mid-002", 8),
        ("mid-003", 6),
        ("mid-004", 5),
    ]
    _seed_db(conn, facts)

    scored = [_make_scored(fid, score=1.0) for fid, _ in facts]
    cfg = _cfg(enable_ips=True)

    original_score = scored[0].score  # hot-001, access_count=100
    apply_ips_debiasing(scored, conn, cfg)

    hot_item = next(s for s in scored if s.fact.id == "hot-001")
    assert hot_item.score < original_score, (
        f"Expected score penalty for over-retrieved fact; "
        f"before={original_score}, after={hot_item.score}"
    )
    assert "ips_adjustment" in hot_item.components
    assert hot_item.components["ips_adjustment"] == -0.02


def test_ips_no_effect_few_facts():
    """With fewer than 5 facts having access_count > 0, IPS should be a no-op."""
    conn = get_connection(":memory:")

    # Only 4 facts have any access — below the gate threshold of 5
    facts = [
        ("fact-001", 0),
        ("fact-002", 10),
        ("fact-003", 8),
        ("fact-004", 6),
        ("fact-005", 4),
    ]
    _seed_db(conn, facts)

    # fact-001 has 0 accesses, so only 4 facts have access_count > 0 → no-op
    scored = [_make_scored(fid, score=1.0) for fid, _ in facts]
    original_scores = {s.fact.id: s.score for s in scored}
    cfg = _cfg(enable_ips=True)

    apply_ips_debiasing(scored, conn, cfg)

    for item in scored:
        assert item.score == original_scores[item.fact.id], (
            f"Score for {item.fact.id} changed when IPS should be disabled "
            f"(got {item.score}, expected {original_scores[item.fact.id]})"
        )
        assert "ips_adjustment" not in item.components


def test_ips_disabled_by_config():
    """When enable_ips=False, scores must not change regardless of access counts."""
    conn = get_connection(":memory:")

    facts = [(f"fact-{i:03d}", i * 10) for i in range(6)]
    _seed_db(conn, facts)

    scored = [_make_scored(fid, score=1.0) for fid, _ in facts]
    original_scores = {s.fact.id: s.score for s in scored}
    cfg = _cfg(enable_ips=False)

    apply_ips_debiasing(scored, conn, cfg)

    for item in scored:
        assert item.score == original_scores[item.fact.id], (
            f"Score for {item.fact.id} changed even though enable_ips=False"
        )
