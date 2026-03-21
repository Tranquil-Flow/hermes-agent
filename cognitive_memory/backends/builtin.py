"""
Built-in SQLite storage backend for cognitive memory.

Implements StorageBackend with:
- SQLite for structured data
- Embeddings stored as BLOB (float32)
- access_times stored as JSON array
- Full CRUD + link operations
"""

from __future__ import annotations

import json
import sqlite3
import time
import logging
from typing import Dict, List, Any, Optional

import numpy as np

from cognitive_memory.backends.base import (
    MemoryEntry, MemoryLink, StorageBackend
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    category        TEXT NOT NULL,
    scope           TEXT DEFAULT 'global',
    importance      REAL NOT NULL,
    embedding       BLOB,
    created_at      REAL NOT NULL,
    last_accessed   REAL NOT NULL,
    access_count    INTEGER DEFAULT 0,
    access_times    TEXT,
    layer           TEXT DEFAULT 'working',
    superseded_by   TEXT,
    source          TEXT DEFAULT 'agent',
    pinned          INTEGER DEFAULT 0,
    metadata        TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(superseded_by)
    WHERE superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS memory_links (
    source_id   TEXT NOT NULL REFERENCES memories(id),
    target_id   TEXT NOT NULL REFERENCES memories(id),
    weight      REAL NOT NULL DEFAULT 0.1,
    link_type   TEXT DEFAULT 'hebbian',
    created_at  REAL NOT NULL,
    last_coactivated REAL,
    PRIMARY KEY (source_id, target_id)
);

CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id);
"""


def _embedding_to_blob(embedding) -> Optional[bytes]:
    """Convert embedding (numpy array or list) to bytes for SQLite BLOB."""
    if embedding is None:
        return None
    arr = np.asarray(embedding, dtype=np.float32)
    return arr.tobytes()


def _blob_to_embedding(blob: Optional[bytes]) -> Optional[np.ndarray]:
    """Convert SQLite BLOB back to numpy array."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    """Convert a database row to a MemoryEntry."""
    access_times_json = row["access_times"]
    access_times = json.loads(access_times_json) if access_times_json else []

    metadata_json = row["metadata"]
    metadata = json.loads(metadata_json) if metadata_json else {}

    return MemoryEntry(
        id=row["id"],
        content=row["content"],
        category=row["category"],
        scope=row["scope"],
        importance=row["importance"],
        embedding=_blob_to_embedding(row["embedding"]),
        created_at=row["created_at"],
        last_accessed=row["last_accessed"],
        access_count=row["access_count"],
        access_times=access_times,
        layer=row["layer"],
        superseded_by=row["superseded_by"],
        source=row["source"],
        pinned=bool(row["pinned"]),
        metadata=metadata,
    )


class BuiltinSQLiteBackend(StorageBackend):
    """SQLite-backed storage for cognitive memory."""

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ─── Memory CRUD ─────────────────────────────────────────────

    def store(self, entry: MemoryEntry) -> str:
        """Persist a memory entry. Returns the memory id."""
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, category, scope, importance, embedding,
                created_at, last_accessed, access_count, access_times,
                layer, superseded_by, source, pinned, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.content,
                entry.category,
                entry.scope,
                entry.importance,
                _embedding_to_blob(entry.embedding),
                entry.created_at,
                entry.last_accessed,
                entry.access_count,
                json.dumps(entry.access_times),
                entry.layer,
                entry.superseded_by,
                entry.source,
                int(entry.pinned),
                json.dumps(entry.metadata),
            ),
        )
        self._conn.commit()
        return entry.id

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a single memory by id."""
        cur = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
        row = cur.fetchone()
        return _row_to_entry(row) if row else None

    def get_all_active(self) -> List[MemoryEntry]:
        """Get all memories where superseded_by is None."""
        cur = self._conn.execute(
            "SELECT * FROM memories WHERE superseded_by IS NULL"
        )
        return [_row_to_entry(row) for row in cur.fetchall()]

    def get_by_layer(self, layer: str) -> List[MemoryEntry]:
        """Get all active memories in a specific layer."""
        cur = self._conn.execute(
            "SELECT * FROM memories WHERE layer = ? AND superseded_by IS NULL",
            (layer,),
        )
        return [_row_to_entry(row) for row in cur.fetchall()]

    def get_by_scope(self, scope: str) -> List[MemoryEntry]:
        """Get all active memories matching a scope prefix."""
        cur = self._conn.execute(
            "SELECT * FROM memories WHERE (scope LIKE ? OR scope = 'global') AND superseded_by IS NULL",
            (scope + "%",),
        )
        return [_row_to_entry(row) for row in cur.fetchall()]

    _ALLOWED_UPDATE_FIELDS = frozenset({
        "content", "category", "scope", "importance", "embedding",
        "last_accessed", "access_count", "access_times", "layer",
        "superseded_by", "source", "pinned", "metadata",
    })

    def update(self, memory_id: str, **fields) -> None:
        """Update specific fields on a memory."""
        if not fields:
            return

        # Validate field names to prevent SQL injection via kwargs
        bad_fields = set(fields) - self._ALLOWED_UPDATE_FIELDS
        if bad_fields:
            raise ValueError(f"Cannot update fields: {bad_fields}")

        # Handle special serialization
        if "embedding" in fields:
            fields["embedding"] = _embedding_to_blob(fields["embedding"])
        if "access_times" in fields:
            fields["access_times"] = json.dumps(fields["access_times"])
        if "metadata" in fields:
            fields["metadata"] = json.dumps(fields["metadata"])
        if "pinned" in fields:
            fields["pinned"] = int(fields["pinned"])

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [memory_id]
        self._conn.execute(
            f"UPDATE memories SET {set_clause} WHERE id = ?", values
        )
        self._conn.commit()

    def delete(self, memory_id: str) -> None:
        """Hard delete a memory (tests/reset only)."""
        # Delete associated links first
        self._conn.execute(
            "DELETE FROM memory_links WHERE source_id = ? OR target_id = ?",
            (memory_id, memory_id),
        )
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

    # ─── Link Operations ─────────────────────────────────────────

    def get_links(self, memory_id: str) -> List[MemoryLink]:
        """Get all links where this memory is the source."""
        cur = self._conn.execute(
            "SELECT * FROM memory_links WHERE source_id = ?", (memory_id,)
        )
        return [
            MemoryLink(
                source_id=row["source_id"],
                target_id=row["target_id"],
                weight=row["weight"],
                link_type=row["link_type"],
                created_at=row["created_at"],
                last_coactivated=row["last_coactivated"],
            )
            for row in cur.fetchall()
        ]

    def get_backlinks(self, memory_id: str) -> List[MemoryLink]:
        """Get all links where this memory is the target."""
        cur = self._conn.execute(
            "SELECT * FROM memory_links WHERE target_id = ?", (memory_id,)
        )
        return [
            MemoryLink(
                source_id=row["source_id"],
                target_id=row["target_id"],
                weight=row["weight"],
                link_type=row["link_type"],
                created_at=row["created_at"],
                last_coactivated=row["last_coactivated"],
            )
            for row in cur.fetchall()
        ]

    def create_link(
        self,
        source_id: str,
        target_id: str,
        weight: float = 0.1,
        link_type: str = "hebbian",
    ) -> None:
        """Create or update a link between two memories."""
        now = time.time()
        self._conn.execute(
            """INSERT INTO memory_links (source_id, target_id, weight, link_type, created_at, last_coactivated)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_id, target_id) DO UPDATE SET
                 weight = excluded.weight,
                 link_type = excluded.link_type,
                 last_coactivated = excluded.last_coactivated""",
            (source_id, target_id, weight, link_type, now, now),
        )
        self._conn.commit()

    def update_link(self, source_id: str, target_id: str, **fields) -> None:
        """Update link fields."""
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [source_id, target_id]
        self._conn.execute(
            f"UPDATE memory_links SET {set_clause} WHERE source_id = ? AND target_id = ?",
            values,
        )
        self._conn.commit()

    def prune_links(self, min_weight: float = 0.01) -> int:
        """Remove all links with weight below threshold."""
        cur = self._conn.execute(
            "DELETE FROM memory_links WHERE weight < ?", (min_weight,)
        )
        self._conn.commit()
        return cur.rowcount

    def decay_links(self, decay_rate: float = 0.95) -> None:
        """Multiply all link weights by decay_rate."""
        self._conn.execute(
            "UPDATE memory_links SET weight = weight * ?", (decay_rate,)
        )
        self._conn.commit()

    # ─── Lifecycle ───────────────────────────────────────────────

    def reset(self) -> None:
        """Delete all data."""
        self._conn.execute("DELETE FROM memory_links")
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()

    def count(self) -> int:
        """Total number of active (non-superseded) memories."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL"
        )
        return cur.fetchone()[0]

    def stats(self) -> Dict[str, Any]:
        """Return storage statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        active = self.count()

        layers = {}
        for layer in ("working", "core", "archive"):
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE layer = ? AND superseded_by IS NULL",
                (layer,),
            )
            layers[layer] = cur.fetchone()[0]

        categories = {}
        cur = self._conn.execute(
            "SELECT category, COUNT(*) FROM memories WHERE superseded_by IS NULL GROUP BY category"
        )
        for row in cur.fetchall():
            categories[row[0]] = row[1]

        link_count = self._conn.execute(
            "SELECT COUNT(*) FROM memory_links"
        ).fetchone()[0]

        superseded = total - active

        return {
            "total": total,
            "active": active,
            "superseded": superseded,
            "by_layer": layers,
            "by_category": categories,
            "links": link_count,
            "db_path": self._db_path,
        }

    # ─── Batch Operations (performance) ────────────────────────────

    def get_all_links(self) -> List[MemoryLink]:
        """Get ALL links in one query. Used for batch link map building."""
        cur = self._conn.execute("SELECT * FROM memory_links")
        return [
            MemoryLink(
                source_id=row["source_id"],
                target_id=row["target_id"],
                weight=row["weight"],
                link_type=row["link_type"],
                created_at=row["created_at"],
                last_coactivated=row["last_coactivated"],
            )
            for row in cur.fetchall()
        ]

    def get_many(self, memory_ids: List[str]) -> Dict[str, MemoryEntry]:
        """Fetch multiple memories by ID in a single query."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        cur = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        )
        return {row["id"]: _row_to_entry(row) for row in cur.fetchall()}

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
