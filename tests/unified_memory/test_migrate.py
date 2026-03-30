"""
Tests for unified_memory/migrate.py

Covers:
- migrate_markdown_files with MEMORY_SPEC notation lines
- migrate_markdown_files with USER.md content
- Plain lines becoming V[general]
- § separator lines being skipped
- migrate_structured_memory from a mock sm_facts SQLite DB
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from unified_memory.migrate import (
    migrate_cognitive_memory,
    migrate_markdown_files,
    migrate_structured_memory,
)
from unified_memory.types import FactType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_temp_file(content: str) -> str:
    """Write content to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


def _make_store():
    """Return a mock store that records .store() calls and returns a fact_id."""
    store = MagicMock()
    store.store.return_value = "mock-fact-id"
    return store


# ---------------------------------------------------------------------------
# test_migrate_memory_md
# ---------------------------------------------------------------------------

MEMORY_MD_CONTENT = """\
C[api]: no breaking changes allowed
§
D[storage]: use SQLite for persistence
V[environment]: macOS / AEST (UTC+10)
§
"""


def test_migrate_memory_md():
    """MEMORY_SPEC lines in MEMORY.md are stored with the correct type and target."""
    memory_path = _write_temp_file(MEMORY_MD_CONTENT)
    try:
        store = _make_store()
        count = migrate_markdown_files(store, memory_path=memory_path, user_path=None)

        assert count == 3, f"Expected 3 facts, got {count}"

        calls = store.store.call_args_list
        assert len(calls) == 3

        # First call: C[api]
        kwargs0 = calls[0].kwargs if calls[0].kwargs else {}
        args0 = calls[0].args
        assert "no breaking changes allowed" in (args0[0] if args0 else kwargs0.get("content", ""))
        assert kwargs0.get("fact_type") == FactType.CONSTRAINT.value or kwargs0.get("fact_type") == "C"
        assert kwargs0.get("target") == "api"
        assert kwargs0.get("scope") == "memory"

        # Second call: D[storage]
        kwargs1 = calls[1].kwargs if calls[1].kwargs else {}
        args1 = calls[1].args
        assert "use SQLite for persistence" in (args1[0] if args1 else kwargs1.get("content", ""))
        assert kwargs1.get("fact_type") == FactType.DECISION.value or kwargs1.get("fact_type") == "D"
        assert kwargs1.get("target") == "storage"
        assert kwargs1.get("scope") == "memory"

        # Third call: V[environment]
        kwargs2 = calls[2].kwargs if calls[2].kwargs else {}
        args2 = calls[2].args
        assert "macOS / AEST (UTC+10)" in (args2[0] if args2 else kwargs2.get("content", ""))
        assert kwargs2.get("fact_type") == FactType.VALUE.value or kwargs2.get("fact_type") == "V"
        assert kwargs2.get("target") == "environment"
        assert kwargs2.get("scope") == "memory"
    finally:
        os.unlink(memory_path)


# ---------------------------------------------------------------------------
# test_migrate_user_md
# ---------------------------------------------------------------------------

USER_MD_CONTENT = """\
User is Tranquil-Flow (tranquil_flow@protonmail.com).
§
Reminder: pat the cat!
§
C[privacy]: never share personal data
"""


def test_migrate_user_md():
    """USER.md content is stored with scope='user'."""
    user_path = _write_temp_file(USER_MD_CONTENT)
    try:
        store = _make_store()
        count = migrate_markdown_files(store, memory_path=None, user_path=user_path)

        assert count == 3, f"Expected 3 facts, got {count}"

        calls = store.store.call_args_list
        # All calls must use scope='user'
        for c in calls:
            assert c.kwargs.get("scope") == "user", f"Expected scope='user', got {c.kwargs}"

        # First plain line -> V[general]
        args0 = calls[0].args
        kwargs0 = calls[0].kwargs
        assert "Tranquil-Flow" in (args0[0] if args0 else kwargs0.get("content", ""))
        assert kwargs0.get("target") == "general"
        assert kwargs0.get("fact_type") == FactType.VALUE.value or kwargs0.get("fact_type") == "V"

        # Last MEMORY_SPEC line -> C[privacy]
        kwargs2 = calls[2].kwargs
        args2 = calls[2].args
        assert "never share personal data" in (args2[0] if args2 else kwargs2.get("content", ""))
        assert kwargs2.get("fact_type") == FactType.CONSTRAINT.value or kwargs2.get("fact_type") == "C"
        assert kwargs2.get("target") == "privacy"
    finally:
        os.unlink(user_path)


# ---------------------------------------------------------------------------
# test_migrate_plain_lines
# ---------------------------------------------------------------------------

PLAIN_CONTENT = """\
CRITICAL PRACTICE: Use subagents aggressively.
Cron delivery: discord:1483466105862488156.
User environment: macOS/AEST (UTC+10).
"""


def test_migrate_plain_lines():
    """Non-MEMORY_SPEC lines are stored as V[general]."""
    memory_path = _write_temp_file(PLAIN_CONTENT)
    try:
        store = _make_store()
        count = migrate_markdown_files(store, memory_path=memory_path, user_path=None)

        assert count == 3

        for c in store.store.call_args_list:
            assert c.kwargs.get("fact_type") == FactType.VALUE.value or c.kwargs.get("fact_type") == "V", \
                f"Expected V type, got {c.kwargs}"
            assert c.kwargs.get("target") == "general", \
                f"Expected target='general', got {c.kwargs}"
            assert c.kwargs.get("scope") == "memory"
    finally:
        os.unlink(memory_path)


# ---------------------------------------------------------------------------
# test_skip_separators
# ---------------------------------------------------------------------------

SEPARATOR_HEAVY_CONTENT = """\
§
§ Section One
§
Real fact here.
§
Another real fact.
§
"""


def test_skip_separators():
    """Lines starting with § are skipped; empty lines are skipped too."""
    memory_path = _write_temp_file(SEPARATOR_HEAVY_CONTENT)
    try:
        store = _make_store()
        count = migrate_markdown_files(store, memory_path=memory_path, user_path=None)

        # Only 2 non-separator, non-empty lines
        assert count == 2, f"Expected 2 facts, got {count}"
        assert store.store.call_count == 2

        contents = [
            c.args[0] if c.args else c.kwargs.get("content", "")
            for c in store.store.call_args_list
        ]
        assert "Real fact here." in contents
        assert "Another real fact." in contents
    finally:
        os.unlink(memory_path)


# ---------------------------------------------------------------------------
# test_migrate_structured_memory
# ---------------------------------------------------------------------------

def _create_sm_db(rows: list[tuple]) -> str:
    """Create a temp SQLite DB with an sm_facts table populated with rows."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    conn = sqlite3.connect(f.name)
    conn.execute(
        "CREATE TABLE sm_facts (type TEXT, target TEXT, content TEXT, scope TEXT)"
    )
    conn.executemany("INSERT INTO sm_facts VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return f.name


def test_migrate_structured_memory():
    """Facts from sm_facts table are stored with correct type, target, scope."""
    rows = [
        ("C", "database", "Use UUIDs for all primary keys", "project"),
        ("D", "api",      "REST over GraphQL",              "project"),
        ("V", "general",  "Team meets on Tuesdays",         "global"),
    ]
    db_path = _create_sm_db(rows)
    try:
        store = _make_store()
        count = migrate_structured_memory(store, db_path)

        assert count == 3, f"Expected 3 migrated facts, got {count}"

        calls = store.store.call_args_list
        assert len(calls) == 3

        for i, (exp_type, exp_target, exp_content, exp_scope) in enumerate(rows):
            kw = calls[i].kwargs
            args = calls[i].args
            actual_content = args[0] if args else kw.get("content", "")
            assert actual_content == exp_content, \
                f"Row {i}: expected content '{exp_content}', got '{actual_content}'"
            assert kw.get("fact_type") == exp_type, \
                f"Row {i}: expected fact_type '{exp_type}', got '{kw.get('fact_type')}'"
            assert kw.get("target") == exp_target, \
                f"Row {i}: expected target '{exp_target}', got '{kw.get('target')}'"
            assert kw.get("scope") == exp_scope, \
                f"Row {i}: expected scope '{exp_scope}', got '{kw.get('scope')}'"
    finally:
        os.unlink(db_path)


def test_migrate_structured_memory_missing_db():
    """migrate_structured_memory returns 0 when DB file is missing."""
    store = _make_store()
    count = migrate_structured_memory(store, "/nonexistent/path/sm.db")
    assert count == 0
    store.store.assert_not_called()


def test_migrate_structured_memory_skips_empty_content():
    """Rows with empty content are skipped."""
    rows = [
        ("V", "general", "Valid fact", "global"),
        ("V", "general", "",           "global"),
        ("V", "general", "   ",        "global"),
    ]
    db_path = _create_sm_db(rows)
    try:
        store = _make_store()
        count = migrate_structured_memory(store, db_path)
        assert count == 1
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# test_migrate_cognitive_memory
# ---------------------------------------------------------------------------

def _create_cm_db(rows: list[dict]) -> str:
    """Create a temp SQLite DB with a memories table."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    conn = sqlite3.connect(f.name)
    conn.execute(
        """CREATE TABLE memories (
            content TEXT,
            category TEXT,
            importance REAL,
            scope TEXT,
            access_count INTEGER,
            created_at REAL,
            updated_at REAL,
            last_accessed REAL
        )"""
    )
    conn.executemany(
        "INSERT INTO memories VALUES (:content, :category, :importance, :scope, "
        ":access_count, :created_at, :updated_at, :last_accessed)",
        rows,
    )
    conn.commit()
    conn.close()
    return f.name


def test_migrate_cognitive_memory():
    """Memories from cognitive_memory DB are stored with preserved metadata."""
    now = time.time()
    rows = [
        {
            "content": "Remember to back up data weekly",
            "category": "operational",
            "importance": 0.8,
            "scope": "global",
            "access_count": 5,
            "created_at": now - 3600,
            "updated_at": now - 1800,
            "last_accessed": now - 600,
        },
        {
            "content": "User prefers concise responses",
            "category": "preference",
            "importance": 0.6,
            "scope": "user",
            "access_count": 12,
            "created_at": now - 7200,
            "updated_at": now - 3600,
            "last_accessed": now - 300,
        },
    ]
    db_path = _create_cm_db(rows)

    # Build a mock store that also exposes a .conn mock for the UPDATE call
    store = MagicMock()
    store.store.return_value = "mock-fact-id"
    mock_conn = MagicMock()
    store.conn = mock_conn

    try:
        count = migrate_cognitive_memory(store, db_path)
        assert count == 2

        # store.store was called twice
        assert store.store.call_count == 2

        # Each call used the right category/importance/scope
        for i, row in enumerate(rows):
            kw = store.store.call_args_list[i].kwargs
            args = store.store.call_args_list[i].args
            actual_content = args[0] if args else kw.get("content", "")
            assert actual_content == row["content"]
            assert kw.get("category") == row["category"]
            assert kw.get("importance") == row["importance"]
            assert kw.get("scope") == row["scope"]

        # conn.execute was called to UPDATE timestamps
        assert mock_conn.execute.called
    finally:
        os.unlink(db_path)


def test_migrate_cognitive_memory_missing_db():
    """migrate_cognitive_memory returns 0 when DB file is missing."""
    store = _make_store()
    count = migrate_cognitive_memory(store, "/nonexistent/path/cm.db")
    assert count == 0
    store.store.assert_not_called()
