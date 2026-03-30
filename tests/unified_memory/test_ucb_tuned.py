"""
Tests for UCB-Tuned variance-aware exploration bonus in retrieval.py.
"""
from __future__ import annotations

import math
import time
import types
import sqlite3
import pytest

from unified_memory.schema import get_connection
from unified_memory.types import MemoryFact, ScoredFact, FactType
from unified_memory.config import UnifiedMemoryConfig
from unified_memory.retrieval import apply_qvalue_reranking


def _make_fact(fact_id: str) -> MemoryFact:
    now = time.time()
    return MemoryFact(
        id=fact_id,
        content=f"Fact {fact_id}",
        embedding=None,
        fact_type=FactType.VALUE,
        target="test",
        layer="working",
        importance=0.5,
        activation=0.5,
        q_value=0.5,
        access_count=0,
        metabolic_rate=1.0,
        category=None,
        scope_id=None,
        status="active",
        pinned=False,
        created_at=now,
        updated_at=now,
        last_accessed=now,
        source_hash=None,
        superseded_by=None,
    )


def _make_scored_fact(fact_id: str, score: float = 1.0) -> ScoredFact:
    return ScoredFact(
        fact=_make_fact(fact_id),
        score=score,
        components={"base_level": score},
    )


def _make_qvalue_store(memory_qvalues_rows: list, um_qvalues_rows: list = None):
    """Create a minimal qvalue_store mock backed by sqlite.

    memory_qvalues_rows: list of (memory_id, q_value, update_count, total_retrievals)
    um_qvalues_rows: list of (memory_id, reward_variance)
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE memory_qvalues (
            memory_id TEXT PRIMARY KEY,
            q_value REAL DEFAULT 0.5,
            update_count INTEGER DEFAULT 0,
            total_retrievals INTEGER DEFAULT 0,
            last_updated REAL DEFAULT 0.0,
            last_retrieved REAL DEFAULT 0.0
        )
    """)
    conn.execute("""
        CREATE TABLE um_qvalues (
            memory_id TEXT PRIMARY KEY,
            q_value REAL DEFAULT 0.5,
            update_count INTEGER DEFAULT 0,
            total_retrievals INTEGER DEFAULT 0,
            last_updated REAL,
            last_retrieved REAL,
            reward_variance REAL DEFAULT 0.0
        )
    """)

    for row in memory_qvalues_rows:
        conn.execute(
            "INSERT INTO memory_qvalues VALUES (?,?,?,?,0,0)",
            row
        )

    if um_qvalues_rows:
        for row in um_qvalues_rows:
            conn.execute(
                "INSERT INTO um_qvalues (memory_id, reward_variance) VALUES (?,?)",
                row
            )

    conn.commit()

    store = types.SimpleNamespace()
    store._conn = conn

    def get_q_batch(memory_ids):
        return {mid: 0.5 for mid in memory_ids}

    def get_total_updates():
        row = conn.execute(
            "SELECT COALESCE(SUM(update_count), 0) FROM memory_qvalues"
        ).fetchone()
        return row[0] if row else 0

    def record_retrieval(memory_id):
        pass

    store.get_q_batch = get_q_batch
    store.get_total_updates = get_total_updates
    store.record_retrieval = record_retrieval

    return store


def test_ucb_never_retrieved():
    """Verify never-retrieved facts get a large UCB exploration bonus."""
    cfg = UnifiedMemoryConfig()
    c = cfg.qvalue_exploration_c

    # fact "new" has never been retrieved (no row in memory_qvalues)
    scored = [_make_scored_fact("new_fact", score=1.0)]
    store = _make_qvalue_store(
        memory_qvalues_rows=[],  # no entries => n_retrievals=0
    )

    initial_score = scored[0].score
    apply_qvalue_reranking(scored, store, cfg)

    ucb_bonus = scored[0].components.get("ucb_bonus", 0.0)
    expected_bonus = c * 2.5

    assert ucb_bonus == pytest.approx(expected_bonus, rel=1e-6), (
        f"Expected UCB bonus {expected_bonus} for never-retrieved, got {ucb_bonus}"
    )
    assert ucb_bonus > 0, "UCB bonus should be positive for never-retrieved facts"


def test_ucb_variance_aware():
    """Verify that variance from um_qvalues affects the UCB exploration bonus."""
    cfg = UnifiedMemoryConfig()
    c = cfg.qvalue_exploration_c

    n = 5  # total_retrievals
    # We need at least 1 update so T >= 1 so log(T) > 0
    total_updates = 10

    fact_id_low_var = "fact_low_var"
    fact_id_high_var = "fact_high_var"

    # Both facts have been retrieved 5 times
    memory_rows = [
        (fact_id_low_var, 0.5, total_updates, n),
        (fact_id_high_var, 0.5, total_updates, n),
    ]

    # Low variance fact: 0.01; high variance fact: 0.25 (maximum)
    um_rows = [
        (fact_id_low_var, 0.01),
        (fact_id_high_var, 0.25),
    ]

    scored_low = [_make_scored_fact(fact_id_low_var, score=1.0)]
    scored_high = [_make_scored_fact(fact_id_high_var, score=1.0)]

    store_low = _make_qvalue_store(memory_rows, um_rows)
    store_high = _make_qvalue_store(memory_rows, um_rows)

    apply_qvalue_reranking(scored_low, store_low, cfg)
    apply_qvalue_reranking(scored_high, store_high, cfg)

    ucb_low = scored_low[0].components["ucb_bonus"]
    ucb_high = scored_high[0].components["ucb_bonus"]

    # UCB-Tuned: V = variance + sqrt(2*ln(T)/n), ucb = c * sqrt(ln(T+1)/n * min(0.25, V))
    # High variance (0.25) should produce a larger or equal bonus vs low variance (0.01)
    assert ucb_high >= ucb_low, (
        f"High-variance UCB {ucb_high} should be >= low-variance UCB {ucb_low}"
    )

    # T = sum of all update_counts across both memory_rows (each has total_updates)
    # store has 2 rows each with update_count=total_updates
    T = max(total_updates * 2, 1)

    # Verify the formula manually for the high-variance fact
    V_high = 0.25 + math.sqrt(2.0 * math.log(T) / n)
    expected_high = c * math.sqrt(math.log(T + 1) / n * min(0.25, V_high))
    assert ucb_high == pytest.approx(expected_high, rel=1e-6), (
        f"UCB formula mismatch: expected {expected_high}, got {ucb_high}"
    )

    # Verify the formula manually for the low-variance fact
    V_low = 0.01 + math.sqrt(2.0 * math.log(T) / n)
    expected_low = c * math.sqrt(math.log(T + 1) / n * min(0.25, V_low))
    assert ucb_low == pytest.approx(expected_low, rel=1e-6), (
        f"UCB formula mismatch: expected {expected_low}, got {ucb_low}"
    )


def test_ucb_default_variance_when_no_um_qvalues_row():
    """When um_qvalues has no row, default variance=0.25 is used."""
    cfg = UnifiedMemoryConfig()
    c = cfg.qvalue_exploration_c

    n = 3
    total_updates = 5
    fact_id = "fact_no_variance_row"

    memory_rows = [(fact_id, 0.5, total_updates, n)]
    um_rows = []  # no row in um_qvalues

    scored = [_make_scored_fact(fact_id, score=1.0)]
    store = _make_qvalue_store(memory_rows, um_rows)

    apply_qvalue_reranking(scored, store, cfg)

    ucb = scored[0].components["ucb_bonus"]

    # Default variance = 0.25
    T = total_updates
    V = 0.25 + math.sqrt(2.0 * math.log(T) / n)
    expected = c * math.sqrt(math.log(T + 1) / n * min(0.25, V))
    assert ucb == pytest.approx(expected, rel=1e-6), (
        f"UCB with default variance mismatch: expected {expected}, got {ucb}"
    )
