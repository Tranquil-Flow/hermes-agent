"""
Migration tool — converts legacy MEMORY.md, USER.md, structured_memory DB,
and cognitive_memory DB into the unified_memory database.

Usage:
    from unified_memory.migrate import run_migration
    run_migration("/path/to/unified.db", "~/.hermes")
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from unified_memory.types import parse_notation, FactType


# ---------------------------------------------------------------------------
# Markdown file migration
# ---------------------------------------------------------------------------

def migrate_markdown_files(
    store,
    memory_path: Optional[str] = None,
    user_path: Optional[str] = None,
) -> int:
    """Parse MEMORY.md and USER.md and store facts into unified_memory.

    Lines starting with '§' are section separators and are skipped.
    Empty lines are skipped.
    Lines matching MEMORY_SPEC notation (C[x]: y, D[x]: y, V[x]: y, etc.)
    are stored with the correct type and target.
    All other lines are stored as V[general]: <content>.

    memory_path facts get scope='memory'.
    user_path facts get scope='user'.

    Returns the total count of migrated facts.
    """
    count = 0

    files = []
    if memory_path and os.path.isfile(memory_path):
        files.append((memory_path, "memory"))
    if user_path and os.path.isfile(user_path):
        files.append((user_path, "user"))

    for file_path, scope in files:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and section separators
            if not stripped or stripped.startswith("§"):
                continue

            parsed = parse_notation(stripped)
            if parsed is not None:
                fact_type_enum, target, content = parsed
                store.store(
                    content,
                    scope=scope,
                    fact_type=fact_type_enum.value,
                    target=target,
                )
            else:
                # Plain text — store as V[general]
                store.store(
                    stripped,
                    scope=scope,
                    fact_type=FactType.VALUE.value,
                    target="general",
                )
            count += 1

    return count


# ---------------------------------------------------------------------------
# Structured memory DB migration
# ---------------------------------------------------------------------------

def migrate_structured_memory(store, db_path: str) -> int:
    """Migrate facts from a legacy structured_memory SQLite database.

    Reads from the sm_facts table and transfers type, target, content, scope
    to the unified memory store.

    Returns the count of migrated facts.
    """
    if not os.path.isfile(db_path):
        print(f"  [structured_memory] DB not found: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    count = 0

    try:
        rows = conn.execute(
            "SELECT type, target, content, scope FROM sm_facts"
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"  [structured_memory] Error reading sm_facts: {e}")
        conn.close()
        return 0

    for row in rows:
        fact_type = row["type"] or "V"
        target = row["target"] or "general"
        content = row["content"] or ""
        scope = row["scope"] or "global"

        if not content.strip():
            continue

        store.store(
            content,
            scope=scope,
            fact_type=fact_type,
            target=target,
        )
        count += 1

    conn.close()
    return count


# ---------------------------------------------------------------------------
# Cognitive memory DB migration
# ---------------------------------------------------------------------------

def migrate_cognitive_memory(store, db_path: str) -> int:
    """Migrate memories from a legacy cognitive_memory SQLite database.

    Reads memories and transfers content, category, importance, scope.
    Preserves access_count and timestamps by directly updating the
    um_facts row after storing.

    Returns the count of migrated facts.
    """
    if not os.path.isfile(db_path):
        print(f"  [cognitive_memory] DB not found: {db_path}")
        return 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    count = 0

    # Try common schema variants
    try:
        rows = conn.execute(
            """SELECT content, category, importance, scope,
                      access_count, created_at, updated_at, last_accessed
               FROM memories"""
        ).fetchall()
    except sqlite3.OperationalError:
        try:
            rows = conn.execute(
                """SELECT content, category, importance, scope,
                          access_count, created_at, updated_at, last_accessed
                   FROM cm_memories"""
            ).fetchall()
        except sqlite3.OperationalError as e:
            print(f"  [cognitive_memory] Error reading memories table: {e}")
            conn.close()
            return 0

    for row in rows:
        content = row["content"] or ""
        category = row["category"] or None
        importance = row["importance"] if row["importance"] is not None else 0.5
        scope = row["scope"] or "global"
        access_count = row["access_count"] if row["access_count"] is not None else 0
        created_at = row["created_at"] or time.time()
        updated_at = row["updated_at"] or time.time()
        last_accessed = row["last_accessed"] or time.time()

        if not content.strip():
            continue

        fact_id = store.store(
            content,
            scope=scope,
            category=category,
            importance=importance,
        )

        # Preserve timestamps and access_count from the original record
        try:
            store.conn.execute(
                """UPDATE um_facts
                   SET access_count = ?,
                       created_at   = ?,
                       updated_at   = ?,
                       last_accessed = ?
                   WHERE id = ?""",
                (access_count, created_at, updated_at, last_accessed, fact_id),
            )
            store.conn.commit()
        except Exception as e:
            print(f"  [cognitive_memory] Warning: could not preserve timestamps for fact {fact_id}: {e}")

        count += 1

    conn.close()
    return count


# ---------------------------------------------------------------------------
# Auto-detection runner
# ---------------------------------------------------------------------------

def run_migration(
    unified_db_path: str,
    hermes_home: str = "~/.hermes",
) -> None:
    """Auto-detect legacy data in hermes_home and run all applicable migrations.

    Checks for:
    - MEMORY.md
    - USER.md
    - structured_memory.db (or structured_memory/structured_memory.db)
    - cognitive_memory.db (or cognitive_memory/cognitive_memory.db)

    Prints a summary of what was migrated.
    """
    from unified_memory.store import UnifiedMemoryStore
    from unified_memory.config import UnifiedMemoryConfig

    hermes_path = Path(hermes_home).expanduser()
    config = UnifiedMemoryConfig.balanced()
    config = UnifiedMemoryConfig.balanced().__class__(**{
        **{k: getattr(config, k) for k in config.__dataclass_fields__},
        "db_path": unified_db_path,
    })
    store = UnifiedMemoryStore(config=config, db_path=unified_db_path)

    print(f"Unified Memory Migration")
    print(f"  Source: {hermes_path}")
    print(f"  Target: {unified_db_path}")
    print()

    total = 0

    # --- Markdown files ---
    memory_md = hermes_path / "MEMORY.md"
    user_md = hermes_path / "USER.md"

    if memory_md.exists() or user_md.exists():
        count = migrate_markdown_files(
            store,
            memory_path=str(memory_md) if memory_md.exists() else None,
            user_path=str(user_md) if user_md.exists() else None,
        )
        print(f"  MEMORY.md / USER.md  -> {count} facts")
        total += count
    else:
        print("  MEMORY.md / USER.md  -> not found, skipped")

    # --- Structured memory DB ---
    sm_candidates = [
        hermes_path / "structured_memory.db",
        hermes_path / "structured_memory" / "structured_memory.db",
        hermes_path / "sm.db",
    ]
    sm_path = next((p for p in sm_candidates if p.exists()), None)
    if sm_path:
        count = migrate_structured_memory(store, str(sm_path))
        print(f"  structured_memory DB -> {count} facts  ({sm_path.name})")
        total += count
    else:
        print("  structured_memory DB -> not found, skipped")

    # --- Cognitive memory DB ---
    cm_candidates = [
        hermes_path / "cognitive_memory.db",
        hermes_path / "cognitive_memory" / "cognitive_memory.db",
        hermes_path / "cm.db",
    ]
    cm_path = next((p for p in cm_candidates if p.exists()), None)
    if cm_path:
        count = migrate_cognitive_memory(store, str(cm_path))
        print(f"  cognitive_memory DB  -> {count} facts  ({cm_path.name})")
        total += count
    else:
        print("  cognitive_memory DB  -> not found, skipped")

    print()
    print(f"  Total migrated: {total} facts")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate legacy Hermes memory files into unified_memory."
    )
    parser.add_argument(
        "--db",
        default=str(Path("~/.hermes/unified_memory.db").expanduser()),
        help="Path to the unified_memory SQLite database (default: ~/.hermes/unified_memory.db)",
    )
    parser.add_argument(
        "--hermes-home",
        default="~/.hermes",
        help="Path to the Hermes home directory (default: ~/.hermes)",
    )
    args = parser.parse_args()
    run_migration(args.db, args.hermes_home)
