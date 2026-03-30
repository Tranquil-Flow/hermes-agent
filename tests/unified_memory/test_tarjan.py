"""
Tests for Tarjan articulation point detection and bridge node protection.
"""

from __future__ import annotations

import time
import pytest

from unified_memory.schema import get_connection
from unified_memory.lifecycle import find_articulation_points, protect_bridge_nodes


def _make_db():
    conn = get_connection(":memory:")
    return conn


def _insert_fact(conn, fact_id, content="test fact"):
    now = time.time()
    conn.execute(
        """INSERT INTO um_facts
           (id, content, type, target, created_at, updated_at, last_accessed)
           VALUES (?, ?, 'V', 'general', ?, ?, ?)""",
        (fact_id, content, now, now, now),
    )


def _insert_link(conn, src, tgt):
    now = time.time()
    conn.execute(
        "INSERT OR IGNORE INTO um_links (source_id, target_id, strength, last_updated) VALUES (?, ?, 0.5, ?)",
        (src, tgt, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO um_links (source_id, target_id, strength, last_updated) VALUES (?, ?, 0.5, ?)",
        (tgt, src, now),
    )


def test_find_articulation_points_linear_chain():
    """
    Linear chain: A - B - C - D
    B and C are articulation points (removing either disconnects the graph).
    A and D are not articulation points.
    """
    conn = _make_db()
    for fid in ["A", "B", "C", "D"]:
        _insert_fact(conn, fid, f"fact {fid}")
    _insert_link(conn, "A", "B")
    _insert_link(conn, "B", "C")
    _insert_link(conn, "C", "D")
    conn.commit()

    aps = find_articulation_points(conn)
    # In a linear chain A-B-C-D, B and C are articulation points
    assert "B" in aps, f"B should be an articulation point, got {aps}"
    assert "C" in aps, f"C should be an articulation point, got {aps}"
    assert "A" not in aps, f"A should not be an articulation point, got {aps}"
    assert "D" not in aps, f"D should not be an articulation point, got {aps}"


def test_find_articulation_points_bridge():
    """
    Graph:
      1 - 2 - 3 - 4
          |       |
          5 ------
    Node 2 connects the left cluster (1) to the right cluster (3-4-5).
    Node 3 connects 2 to {4,5} which also connects back via 5-2.
    Actually: 1-2-3, 2-5, 3-4, 4-5 forms a cycle on 2-3-4-5.
    Only node 2 is an articulation point (removing it isolates node 1).
    """
    conn = _make_db()
    for fid in ["n1", "n2", "n3", "n4", "n5"]:
        _insert_fact(conn, fid, f"fact {fid}")
    # n1 connects only to n2 (n2 is the bridge)
    _insert_link(conn, "n1", "n2")
    # n2-n3-n4-n5 form a cycle
    _insert_link(conn, "n2", "n3")
    _insert_link(conn, "n3", "n4")
    _insert_link(conn, "n4", "n5")
    _insert_link(conn, "n5", "n2")
    conn.commit()

    aps = find_articulation_points(conn)
    assert "n2" in aps, f"n2 should be an articulation point (bridge to n1), got {aps}"
    assert "n1" not in aps, f"n1 is a leaf, not an AP, got {aps}"


def test_find_articulation_points_complete_graph():
    """
    Complete graph (K4): every pair connected. No articulation points.
    """
    conn = _make_db()
    nodes = ["p", "q", "r", "s"]
    for fid in nodes:
        _insert_fact(conn, fid, f"fact {fid}")
    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            _insert_link(conn, a, b)
    conn.commit()

    aps = find_articulation_points(conn)
    assert len(aps) == 0, f"Complete graph should have no APs, got {aps}"


def test_find_articulation_points_empty_graph():
    """Empty graph returns empty set."""
    conn = _make_db()
    aps = find_articulation_points(conn)
    assert aps == set()


def test_protect_bridge_nodes():
    """Bridge nodes should get pinned=1 after protect_bridge_nodes."""
    conn = _make_db()
    for fid in ["x1", "x2", "x3"]:
        _insert_fact(conn, fid, f"fact {fid}")
    # x1 - x2 - x3 linear: x2 is the bridge
    _insert_link(conn, "x1", "x2")
    _insert_link(conn, "x2", "x3")
    conn.commit()

    aps = find_articulation_points(conn)
    assert "x2" in aps

    updated = protect_bridge_nodes(conn, aps)
    assert updated >= 1

    # Verify x2 is pinned
    row = conn.execute("SELECT pinned FROM um_facts WHERE id = 'x2'").fetchone()
    assert row["pinned"] == 1, "x2 should be pinned"

    # Verify x1 and x3 are not pinned (not articulation points)
    row1 = conn.execute("SELECT pinned FROM um_facts WHERE id = 'x1'").fetchone()
    row3 = conn.execute("SELECT pinned FROM um_facts WHERE id = 'x3'").fetchone()
    assert row1["pinned"] == 0, "x1 should not be pinned"
    assert row3["pinned"] == 0, "x3 should not be pinned"


def test_protect_bridge_nodes_empty():
    """Calling protect_bridge_nodes with empty set returns 0."""
    conn = _make_db()
    updated = protect_bridge_nodes(conn, set())
    assert updated == 0
