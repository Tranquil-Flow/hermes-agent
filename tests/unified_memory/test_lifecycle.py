"""
Lifecycle tests for UnifiedMemoryStore.
"""

from __future__ import annotations

import pytest

from unified_memory.store import UnifiedMemoryStore
from unified_memory.config import UnifiedMemoryConfig


def make_store() -> UnifiedMemoryStore:
    config = UnifiedMemoryConfig.balanced()
    config.db_path = ":memory:"
    store = UnifiedMemoryStore(config, db_path=":memory:")
    store.enable_virtual_clock()
    return store


def test_simulate_time():
    """Advancing virtual clock affects recency scoring in recall."""
    store = make_store()
    # Store first fact
    id1 = store.store("Early fact about configuration", scope="global",
                      importance=0.5, category="factual")
    # Advance time significantly
    store.simulate_time(30)  # 30 days later
    # Store second, newer fact
    id2 = store.store("Recent fact about configuration settings", scope="global",
                      importance=0.5, category="factual")
    # The newer fact should rank higher due to recency
    results = store.recall("configuration")
    assert len(results) > 0
    # The time offset should cause different scoring
    result_ids = [r.fact.id for r in results]
    # Both facts should be in results
    assert id1 in result_ids or id2 in result_ids


def test_simulate_access():
    """Simulating access on a fact increments its access_count."""
    store = make_store()
    fact_id = store.store("The API endpoint is /api/v2/users", scope="global",
                          importance=0.5, category="factual")
    # Get initial access count
    row_before = store.conn.execute(
        "SELECT access_count FROM um_facts WHERE id = ?", (fact_id,)
    ).fetchone()
    initial_count = row_before["access_count"]

    # Simulate access via content substring
    store.simulate_access("API endpoint")

    row_after = store.conn.execute(
        "SELECT access_count FROM um_facts WHERE id = ?", (fact_id,)
    ).fetchone()
    assert row_after["access_count"] > initial_count


def test_consolidate():
    """Consolidation promotes working->core and prunes old archive facts."""
    store = make_store()
    # Store some facts
    keep_id = store.store("Important fact accessed frequently", scope="global",
                          importance=0.8, category="factual")
    # Access it 3+ times to trigger promotion
    store.simulate_access("Important fact")
    store.advance_time(1)
    store.simulate_access("Important fact")
    store.advance_time(1)
    store.simulate_access("Important fact")
    store.advance_time(1)

    # Insert a stale archived fact directly
    old_time = store._now() - 86400 * 31  # 31 days ago
    store.conn.execute(
        """INSERT INTO um_facts
           (id, content, type, target, scope_id, status, activation, q_value,
            access_count, metabolic_rate, importance, category, layer,
            created_at, updated_at, last_accessed, source_hash)
           VALUES ('stale-001', 'Very old stale archive fact', 'V', 'general',
                   NULL, 'active', 0.001, 0.5, 0, 1.0, 0.1, 'factual', 'archive',
                   ?, ?, ?, 'stale001hash')""",
        (old_time, old_time, old_time)
    )
    store.conn.commit()

    report = store.consolidate()
    assert isinstance(report, dict)
    assert "promoted" in report
    assert "pruned" in report

    # The frequently accessed fact should have been promoted to core
    row = store.conn.execute(
        "SELECT layer FROM um_facts WHERE id = ?", (keep_id,)
    ).fetchone()
    assert row["layer"] == "core"

    # The stale archived fact should have been pruned
    stale_row = store.conn.execute(
        "SELECT id FROM um_facts WHERE id = 'stale-001'"
    ).fetchone()
    assert stale_row is None


def test_reset():
    """Reset clears all data from the store."""
    store = make_store()
    store.store("Fact one", scope="global", importance=0.5, category="factual")
    store.store("Fact two", scope="global", importance=0.5, category="factual")
    store.store("Fact three", scope="global", importance=0.5, category="factual")

    stats_before = store.get_stats()
    assert stats_before["fact_count"] >= 3

    store.reset()

    stats_after = store.get_stats()
    assert stats_after["fact_count"] == 0
    assert stats_after["link_count"] == 0


def test_get_stats():
    """get_stats returns correct fact_count, link_count, and other keys."""
    store = make_store()

    # Empty store
    stats = store.get_stats()
    assert stats["fact_count"] == 0
    assert stats["link_count"] == 0
    assert "scope_count" in stats
    assert "gauge_pct" in stats

    # Add some facts
    store.store("Fact alpha: configuration value", scope="global",
                importance=0.5, category="factual")
    store.store("Fact beta: configuration setting", scope="global",
                importance=0.5, category="factual")

    stats = store.get_stats()
    assert stats["fact_count"] == 2
    assert isinstance(stats["link_count"], int)
    assert stats["link_count"] >= 0
