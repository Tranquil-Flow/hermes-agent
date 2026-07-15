"""Regression test for the duplicate-column race on the ``model_override``
migration (#42553).

Two concurrent migrators can both pass the ``"model_override" not in cols``
check in ``_migrate_add_optional_columns`` and then both issue ``ALTER TABLE
tasks ADD COLUMN model_override TEXT``. The second ``ALTER TABLE`` raises
``OperationalError: duplicate column name``. The dedicated
``_migrate_add_model_override_column`` helper guards against this by
re-checking ``PRAGMA table_info`` immediately before the add, and by
delegating to ``add_column_if_missing`` which swallows the error as a
last-resort backstop.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest import mock

from hermes_cli import kanban_db as kb
from hermes_cli.sqlite_util import add_column_if_missing


def _fresh_db(tmp_path: Path, monkeypatch) -> Path:
    """Create a kanban DB that lacks the ``model_override`` column."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="dupcol")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    # Initialise so all tables/columns exist, then strip model_override.
    with kb.connect(db_path) as conn:
        conn.execute("ALTER TABLE tasks DROP COLUMN model_override")
        conn.commit()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return db_path


def _has_model_override(conn: sqlite3.Connection) -> bool:
    return "model_override" in {
        row["name"] for row in conn.execute("PRAGMA table_info(tasks)")
    }


def test_migrate_add_model_override_column_adds_when_missing(tmp_path, monkeypatch):
    """The helper adds ``model_override`` when the column is absent."""
    db_path = _fresh_db(tmp_path, monkeypatch)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        assert not _has_model_override(conn)
        added = kb._migrate_add_model_override_column(conn)
        assert added is True
        assert _has_model_override(conn)


def test_migrate_add_model_override_column_skips_when_present(tmp_path, monkeypatch):
    """The helper is a no-op (returns ``False``) when the column already exists."""
    db_path = kb.kanban_db_path(board="present")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path) as conn:
        # Column already exists from the normal init migration.
        assert _has_model_override(conn)
        added = kb._migrate_add_model_override_column(conn)
        assert added is False


def test_migrate_add_model_override_column_survives_duplicate_column_race(
    tmp_path, monkeypatch
):
    """Regression test for the duplicate-column race (#42553).

    Simulates a concurrent migrator that has *just* added the column in the
    window between the helper's ``PRAGMA table_info`` snapshot and the
    ``ALTER TABLE``. The ``add_column_if_missing`` backstop must swallow the
    ``duplicate column name`` error instead of propagating it.
    """
    db_path = _fresh_db(tmp_path, monkeypatch)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        original = add_column_if_missing

        def racing_add_column(sqlite_conn, table, column, ddl):
            # Simulate a concurrent migrator adding the column first.
            sqlite_conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {ddl}"
            )
            # Now the real call fires the duplicate error — but the backstop
            # swallows it, so we must NOT re-raise.
            return original(sqlite_conn, table, column, ddl)

        with mock.patch(
            "hermes_cli.kanban_db._add_column_if_missing",
            side_effect=racing_add_column,
        ):
            added = kb._migrate_add_model_override_column(conn)

        assert added is False
        assert _has_model_override(conn)
        # Ensure there is exactly one model_override column (no phantom copy).
        col_count = sum(
            1
            for row in conn.execute("PRAGMA table_info(tasks)")
            if row["name"] == "model_override"
        )
        assert col_count == 1


def test_duplicate_column_error_is_swallowed(tmp_path, monkeypatch):
    """Directly verify that ``add_column_if_missing`` swallows the duplicate
    column error (the mechanism the helper relies on as its last-resort guard)."""
    db_path = kb.kanban_db_path(board="dupswallow")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path) as conn:
        # Column already exists.
        result = add_column_if_missing(
            conn, "tasks", "model_override", "model_override TEXT"
        )
        assert result is False
