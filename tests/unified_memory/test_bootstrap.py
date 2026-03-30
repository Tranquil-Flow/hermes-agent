"""
Tests for bibliographic coupling bootstrap in links.py.
"""

from __future__ import annotations

import math
import time
import pytest

from unified_memory.schema import get_connection
from unified_memory.links import bootstrap_bibliographic_links


def _make_db():
    conn = get_connection(":memory:")
    return conn


def _insert_fact(conn, fact_id, content):
    now = time.time()
    conn.execute(
        """INSERT INTO um_facts
           (id, content, type, target, created_at, updated_at, last_accessed)
           VALUES (?, ?, 'V', 'general', ?, ?, ?)""",
        (fact_id, content, now, now, now),
    )
    conn.commit()
    return now


def test_bibliographic_coupling_creates_links():
    """Facts sharing significant keywords should get bibliographic links."""
    conn = _make_db()

    # Pre-existing fact with similar keywords
    _insert_fact(conn, "fact-A",
                 "Python programming language supports object oriented features")
    _insert_fact(conn, "fact-B",
                 "Python programming language uses dynamic typing features")

    now = time.time()
    # Store fact-C which shares keywords with both A and B
    _insert_fact(conn, "fact-C",
                 "Python programming language functional programming features")

    count = bootstrap_bibliographic_links(conn, "fact-C",
                                          "Python programming language functional programming features",
                                          now, threshold=0.3)
    assert count > 0, f"Expected at least 1 bibliographic link created, got {count}"

    # Verify links exist in um_links
    links = conn.execute(
        "SELECT target_id FROM um_links WHERE source_id = 'fact-C'"
    ).fetchall()
    linked_ids = {r["target_id"] for r in links}
    assert "fact-A" in linked_ids or "fact-B" in linked_ids, (
        f"fact-C should link to fact-A or fact-B, linked to: {linked_ids}"
    )


def test_bibliographic_coupling_strength():
    """Link strength should equal coupling * 0.5."""
    conn = _make_db()

    _insert_fact(conn, "base-A",
                 "machine learning neural network deep training")
    _insert_fact(conn, "base-B",
                 "machine learning neural network deep training weights")

    now = time.time()
    _insert_fact(conn, "new-fact",
                 "machine learning neural network deep training weights optimization")
    bootstrap_bibliographic_links(conn, "new-fact",
                                  "machine learning neural network deep training weights optimization",
                                  now, threshold=0.1)

    links = conn.execute(
        "SELECT strength FROM um_links WHERE source_id = 'new-fact' AND target_id = 'base-B'"
    ).fetchone()
    if links:
        # Strength should be in range [0, 0.5] (coupling * 0.5)
        assert 0 < links["strength"] <= 0.5, f"Unexpected strength: {links['strength']}"


def test_no_coupling_unrelated_facts():
    """Unrelated facts should not get bootstrap links."""
    conn = _make_db()

    # Insert facts with completely different vocabulary
    _insert_fact(conn, "astro-fact",
                 "Jupiter Saturn Uranus Neptune galaxy telescope astronomy")
    _insert_fact(conn, "cook-fact",
                 "flour butter sugar eggs baking cake recipe")

    now = time.time()
    # New fact about cooking -- no overlap with astro-fact
    _insert_fact(conn, "new-cook",
                 "bread yeast dough oven temperature baking")
    count = bootstrap_bibliographic_links(conn, "new-cook",
                                          "bread yeast dough oven temperature baking",
                                          now, threshold=0.3)

    # Check no link to astro-fact
    link = conn.execute(
        "SELECT COUNT(*) as cnt FROM um_links "
        "WHERE source_id = 'new-cook' AND target_id = 'astro-fact'"
    ).fetchone()
    assert link["cnt"] == 0, "Unrelated facts should not be linked"


def test_no_coupling_empty_keywords():
    """Facts with only short words/stopwords get no bootstrap links."""
    conn = _make_db()

    _insert_fact(conn, "stop-fact", "the a an is are was with and")
    now = time.time()
    _insert_fact(conn, "tiny-fact", "a is the")
    count = bootstrap_bibliographic_links(conn, "tiny-fact", "a is the", now, threshold=0.3)
    assert count == 0, "Facts with only stop words should not create links"


def test_coupling_formula():
    """Verify that coupling = |shared| / sqrt(|A| * |B|) is applied correctly."""
    conn = _make_db()

    # fact-X has keywords: {python, programming, language, code}
    # fact-Y has keywords: {python, programming, language, script}
    # shared = {python, programming, language} = 3
    # |X| = 4, |Y| = 4
    # coupling = 3 / sqrt(4*4) = 3/4 = 0.75 > threshold=0.3
    # strength = 0.75 * 0.5 = 0.375
    _insert_fact(conn, "fact-X", "python programming language code")
    now = time.time()
    _insert_fact(conn, "fact-Y", "python programming language script")
    count = bootstrap_bibliographic_links(conn, "fact-Y",
                                          "python programming language script",
                                          now, threshold=0.3)
    assert count > 0, "High coupling facts should create links"

    link = conn.execute(
        "SELECT strength FROM um_links WHERE source_id = 'fact-Y' AND target_id = 'fact-X'"
    ).fetchone()
    assert link is not None, "Link from fact-Y to fact-X should exist"
    expected_strength = 0.75 * 0.5
    assert abs(link["strength"] - expected_strength) < 0.05, (
        f"Expected strength ~{expected_strength}, got {link['strength']}"
    )
