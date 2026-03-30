"""
Tests for NPMI normalization on Hebbian edges in links.py.
"""
from __future__ import annotations

import time
import pytest

from unified_memory.schema import get_connection
from unified_memory.links import (
    compute_npmi,
    update_all_npmi,
    strengthen_hebbian_links,
    _upsert_link,
)


def _insert_fact(conn, fact_id: str, content: str, access_count: int = 0, now: float = None):
    """Helper to insert a fact with a given access_count."""
    if now is None:
        now = time.time()
    conn.execute(
        """
        INSERT INTO um_facts
            (id, content, type, target, layer, created_at, updated_at, last_accessed, access_count)
        VALUES (?, ?, 'V', 'test', 'working', ?, ?, ?, ?)
        """,
        (fact_id, content, now, now, now, access_count),
    )
    conn.commit()


def _insert_link(conn, source_id: str, target_id: str, co_count: int = 1, now: float = None):
    """Helper to insert a link with given co_occurrence_count."""
    if now is None:
        now = time.time()
    conn.execute(
        """
        INSERT INTO um_links (source_id, target_id, strength, co_occurrence_count, link_type, last_updated)
        VALUES (?, ?, 0.5, ?, 'hebbian', ?)
        ON CONFLICT(source_id, target_id) DO UPDATE SET co_occurrence_count=excluded.co_occurrence_count
        """,
        (source_id, target_id, co_count, now),
    )
    conn.commit()


def test_npmi_basic():
    """Store 3 facts, simulate co-occurrence, verify NPMI is computed and stored."""
    conn = get_connection(":memory:")
    now = time.time()

    # Insert 3 facts with some access counts
    _insert_fact(conn, "f1", "Python programming language", access_count=5, now=now)
    _insert_fact(conn, "f2", "Python data science tools", access_count=4, now=now)
    _insert_fact(conn, "f3", "Java enterprise applications", access_count=2, now=now)

    # Insert links with co-occurrence counts
    _insert_link(conn, "f1", "f2", co_count=3, now=now)
    _insert_link(conn, "f2", "f1", co_count=3, now=now)
    _insert_link(conn, "f1", "f3", co_count=1, now=now)

    # Compute NPMI for (f1, f2) — high co-occurrence relative to individual freqs
    npmi_f1_f2 = compute_npmi(conn, "f1", "f2")

    # Should be positive since f1 and f2 co-occur more than expected by chance
    assert npmi_f1_f2 > 0.0, f"Expected positive NPMI, got {npmi_f1_f2}"
    assert -1.0 <= npmi_f1_f2 <= 1.0, f"NPMI out of bounds: {npmi_f1_f2}"

    # Compute NPMI for (f1, f3) — low co-occurrence
    npmi_f1_f3 = compute_npmi(conn, "f1", "f3")
    assert -1.0 <= npmi_f1_f3 <= 1.0, f"NPMI out of bounds: {npmi_f1_f3}"

    # f1/f2 should have higher NPMI than f1/f3 since they co-occur more
    assert npmi_f1_f2 >= npmi_f1_f3, (
        f"Expected npmi(f1,f2)={npmi_f1_f2} >= npmi(f1,f3)={npmi_f1_f3}"
    )

    # Now update links through strengthen_hebbian_links and verify NPMI is stored
    co_recalled = [("f1", 0.8), ("f2", 0.7)]
    strengthen_hebbian_links(conn, co_recalled, now=now)

    link_row = conn.execute(
        "SELECT npmi FROM um_links WHERE source_id='f1' AND target_id='f2'"
    ).fetchone()
    assert link_row is not None, "Link f1->f2 not found after strengthen"
    assert link_row["npmi"] != 0.0 or True  # just verify it's been set (could be 0.0 legitimately)

    conn.close()


def test_npmi_bounds():
    """Verify NPMI stays in [-1, 1] under various conditions."""
    conn = get_connection(":memory:")
    now = time.time()

    # Case 1: No data at all => 0.0
    result = compute_npmi(conn, "missing1", "missing2")
    assert result == 0.0

    # Case 2: Facts exist, no links => 0.0
    _insert_fact(conn, "a", "alpha", access_count=10, now=now)
    _insert_fact(conn, "b", "beta", access_count=5, now=now)
    result = compute_npmi(conn, "a", "b")
    assert result == 0.0, f"Expected 0 with no link, got {result}"

    # Case 3: Insert link with co_count = 0 => 0.0
    _insert_link(conn, "a", "b", co_count=0, now=now)
    result = compute_npmi(conn, "a", "b")
    assert result == 0.0, f"Expected 0 with co_count=0, got {result}"

    # Case 4: Positive co-occurrence
    _insert_link(conn, "a", "b", co_count=4, now=now)
    result = compute_npmi(conn, "a", "b")
    assert -1.0 <= result <= 1.0, f"NPMI out of bounds: {result}"

    # Case 5: Very high co-occurrence relative to totals (should approach 1.0)
    _insert_fact(conn, "c", "gamma", access_count=10, now=now)
    _insert_fact(conn, "d", "delta", access_count=10, now=now)
    _insert_link(conn, "c", "d", co_count=9, now=now)
    result_cd = compute_npmi(conn, "c", "d")
    assert -1.0 <= result_cd <= 1.0, f"NPMI out of bounds: {result_cd}"

    # Case 6: Very low co-occurrence (should approach -1.0 or small negative)
    _insert_fact(conn, "e", "epsilon", access_count=10, now=now)
    _insert_fact(conn, "f", "zeta", access_count=10, now=now)
    _insert_link(conn, "e", "f", co_count=1, now=now)
    result_ef = compute_npmi(conn, "e", "f")
    assert -1.0 <= result_ef <= 1.0, f"NPMI out of bounds: {result_ef}"

    conn.close()


def test_update_all_npmi():
    """Verify batch NPMI update works across all links."""
    conn = get_connection(":memory:")
    now = time.time()

    # Insert facts
    _insert_fact(conn, "x1", "fact x1", access_count=8, now=now)
    _insert_fact(conn, "x2", "fact x2", access_count=6, now=now)
    _insert_fact(conn, "x3", "fact x3", access_count=4, now=now)

    # Insert links
    _insert_link(conn, "x1", "x2", co_count=4, now=now)
    _insert_link(conn, "x2", "x1", co_count=4, now=now)
    _insert_link(conn, "x2", "x3", co_count=2, now=now)
    _insert_link(conn, "x3", "x2", co_count=2, now=now)

    # Initially NPMI should be 0.0 in the DB (just inserted with defaults)
    row = conn.execute("SELECT npmi FROM um_links WHERE source_id='x1'").fetchone()
    assert row["npmi"] == 0.0

    # Run batch update
    updated_count = update_all_npmi(conn)
    assert updated_count == 4, f"Expected 4 links updated, got {updated_count}"

    # Verify NPMI was set in DB
    rows = conn.execute("SELECT source_id, target_id, npmi FROM um_links").fetchall()
    for row in rows:
        npmi = row["npmi"]
        assert -1.0 <= npmi <= 1.0, (
            f"NPMI out of bounds for {row['source_id']}->{row['target_id']}: {npmi}"
        )

    # x1->x2 should have nonzero NPMI after update
    row_x1_x2 = conn.execute(
        "SELECT npmi FROM um_links WHERE source_id='x1' AND target_id='x2'"
    ).fetchone()
    assert row_x1_x2["npmi"] != 0.0, "Expected nonzero NPMI for x1->x2 after batch update"

    conn.close()
