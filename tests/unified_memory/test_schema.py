"""
Tests for unified_memory.schema — DDL correctness.
"""

from __future__ import annotations

import time

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r["name"] for r in rows}


def _index_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    return {r["name"] for r in rows}


def _view_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view'"
    ).fetchall()
    return {r["name"] for r in rows}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tables_created(db_conn):
    """All expected tables must exist after init_db."""
    tables = _table_names(db_conn)
    expected = {
        "um_facts",
        "um_links",
        "um_scopes",
        "um_access_times",
        "um_qvalues",
    }
    missing = expected - tables
    assert not missing, f"Missing tables: {missing}"


def test_fts5_created(db_conn):
    """The FTS5 virtual table um_facts_fts must exist and be queryable."""
    tables = _table_names(db_conn)
    assert "um_facts_fts" in tables, "um_facts_fts virtual table not found"

    # Insert a row and make sure FTS search works end-to-end
    now = time.time()
    db_conn.execute(
        """
        INSERT INTO um_facts (id, content, type, target, created_at, updated_at, last_accessed)
        VALUES ('fts-test-1', 'hello world fts check', 'V', 'general', ?, ?, ?)
        """,
        (now, now, now),
    )
    db_conn.commit()

    results = db_conn.execute(
        "SELECT rowid FROM um_facts_fts WHERE um_facts_fts MATCH 'hello'"
    ).fetchall()
    assert len(results) == 1, "FTS5 search did not return the inserted row"


def test_indexes_created(db_conn):
    """Expected indexes must exist."""
    indexes = _index_names(db_conn)
    expected = {
        "idx_um_facts_type",
        "idx_um_facts_target",
        "idx_um_facts_scope_id",
        "idx_um_facts_status",
        "idx_um_facts_updated_at",
        "idx_um_facts_source_hash",
        "idx_um_facts_layer",
        "idx_um_scopes_active_label",
    }
    missing = expected - indexes
    assert not missing, f"Missing indexes: {missing}"


def test_views_created(db_conn):
    """Both views must exist."""
    views = _view_names(db_conn)
    assert "um_gauge" in views, "um_gauge view missing"
    assert "um_hot_facts" in views, "um_hot_facts view missing"


def test_gauge_view(db_conn):
    """um_gauge used_chars should reflect the chars from active/cold facts."""
    now = time.time()
    # Insert one active fact
    content = "track me"
    ftype   = "V"
    target  = "test"
    db_conn.execute(
        """
        INSERT INTO um_facts (id, content, type, target, status,
                              created_at, updated_at, last_accessed)
        VALUES ('gauge-1', ?, ?, ?, 'active', ?, ?, ?)
        """,
        (content, ftype, target, now, now, now),
    )
    # Insert one superseded fact (should NOT count)
    db_conn.execute(
        """
        INSERT INTO um_facts (id, content, type, target, status,
                              created_at, updated_at, last_accessed)
        VALUES ('gauge-2', 'ignore me', 'V', 'test', 'superseded', ?, ?, ?)
        """,
        (now, now, now),
    )
    db_conn.commit()

    row = db_conn.execute("SELECT used_chars, max_chars FROM um_gauge").fetchone()
    assert row is not None
    assert row["max_chars"] == 10000

    expected_chars = len(ftype) + len(target) + len(content) + 4
    assert row["used_chars"] == expected_chars, (
        f"Expected {expected_chars} used_chars, got {row['used_chars']}"
    )


def test_scope_unique_constraint(db_conn):
    """Two active scopes with the same label must be rejected."""
    import sqlite3

    now = time.time()
    db_conn.execute(
        "INSERT INTO um_scopes (id, label, status, created_at) VALUES ('s1', 'main', 'active', ?)",
        (now,),
    )
    db_conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "INSERT INTO um_scopes (id, label, status, created_at) VALUES ('s2', 'main', 'active', ?)",
            (now,),
        )
        db_conn.commit()


def test_scope_unique_constraint_allows_closed_duplicate(db_conn):
    """A closed scope with the same label as an active one is allowed."""
    now = time.time()
    db_conn.execute(
        "INSERT INTO um_scopes (id, label, status, created_at) VALUES ('s3', 'dup', 'active', ?)",
        (now,),
    )
    db_conn.execute(
        "INSERT INTO um_scopes (id, label, status, created_at, closed_at) "
        "VALUES ('s4', 'dup', 'closed', ?, ?)",
        (now, now),
    )
    db_conn.commit()  # should not raise


def test_populated_db_has_five_facts(populated_db):
    """The populated_db fixture should insert exactly 5 facts."""
    count = populated_db.execute(
        "SELECT COUNT(*) as n FROM um_facts"
    ).fetchone()["n"]
    assert count == 5


def test_fts_triggers_on_update(db_conn):
    """Updating a fact content should update the FTS index."""
    now = time.time()
    db_conn.execute(
        """
        INSERT INTO um_facts (id, content, type, target, created_at, updated_at, last_accessed)
        VALUES ('upd-1', 'original content', 'V', 'general', ?, ?, ?)
        """,
        (now, now, now),
    )
    db_conn.commit()

    db_conn.execute(
        "UPDATE um_facts SET content='updated content', updated_at=? WHERE id='upd-1'",
        (now + 1,),
    )
    db_conn.commit()

    # Old content should not match
    old_hits = db_conn.execute(
        "SELECT rowid FROM um_facts_fts WHERE um_facts_fts MATCH 'original'"
    ).fetchall()
    assert len(old_hits) == 0, "Old content still in FTS after update"

    # New content should match
    new_hits = db_conn.execute(
        "SELECT rowid FROM um_facts_fts WHERE um_facts_fts MATCH 'updated'"
    ).fetchall()
    assert len(new_hits) == 1, "New content not found in FTS after update"


def test_fts_triggers_on_delete(db_conn):
    """Deleting a fact should remove it from the FTS index."""
    now = time.time()
    db_conn.execute(
        """
        INSERT INTO um_facts (id, content, type, target, created_at, updated_at, last_accessed)
        VALUES ('del-1', 'delete me please', 'V', 'general', ?, ?, ?)
        """,
        (now, now, now),
    )
    db_conn.commit()

    db_conn.execute("DELETE FROM um_facts WHERE id='del-1'")
    db_conn.commit()

    hits = db_conn.execute(
        "SELECT rowid FROM um_facts_fts WHERE um_facts_fts MATCH 'delete'"
    ).fetchall()
    assert len(hits) == 0, "Deleted fact still found in FTS"
