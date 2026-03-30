"""
Pytest fixtures for unified_memory tests.
"""

from __future__ import annotations

import time

import pytest

from unified_memory.schema import get_connection


@pytest.fixture
def db_conn():
    """Yield a fresh in-memory unified-memory connection."""
    conn = get_connection(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def populated_db(db_conn):
    """
    Return a db_conn pre-populated with 5 sample facts covering different
    types: V (value), C (claim), D (decision), obs (observation), done.
    """
    now = time.time()
    facts = [
        ("fact-001", "User prefers dark mode",           "V",   "user",    "working"),
        ("fact-002", "Project uses Python 3.11",         "C",   "project", "working"),
        ("fact-003", "Decided to use SQLite for storage","D",   "project", "long"),
        ("fact-004", "Observed high memory usage today", "obs", "system",  "working"),
        ("fact-005", "Onboarding task completed",        "done","task",    "long"),
    ]
    conn = db_conn
    for fid, content, ftype, target, layer in facts:
        conn.execute(
            """
            INSERT INTO um_facts
                (id, content, type, target, layer,
                 created_at, updated_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fid, content, ftype, target, layer, now, now, now),
        )
    conn.commit()
    return conn
