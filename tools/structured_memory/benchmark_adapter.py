"""
Benchmark adapter — wraps the structured memory system as a BenchmarkableStore.

This allows the benchmark runner to swap in structured memory alongside
the cognitive memory backend and flat/FTS5 baselines transparently.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Dict, List, Any, Optional

from benchmarks.interface import BenchmarkableStore
from tools.structured_memory.db import get_sm_connection, SCHEMA_SQL
from tools.structured_memory import facts, gauge, scopes
from tools.structured_memory.facts import MemoryFullError


class StructuredMemoryBenchmarkAdapter(BenchmarkableStore):
    """Adapter that makes the structured memory system benchmarkable."""

    def __init__(self, **kwargs):
        # Use in-memory SQLite for full isolation between benchmark runs
        self._conn: sqlite3.Connection = get_sm_connection(":memory:")
        self._fact_counter: int = 0

    # ------------------------------------------------------------------
    # BenchmarkableStore API
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        category: str = "factual",
        scope: str = "global",
        importance: float = 0.5,
    ) -> None:
        """Store content as a V[target]: content fact.

        The category/importance parameters have no direct equivalent in
        structured memory; we map them to the target label for
        mild organisational benefit.
        """
        # Resolve (or create) the scope
        scope_id: Optional[str] = None
        if scope and scope.lower() not in ("global", "none", ""):
            scope_id = scopes.get_or_create(self._conn, label=scope)

        # Each fact gets a unique target to prevent false supersession.
        # Structured memory supersedes facts with the same (type, target, scope),
        # so we need unique targets for distinct facts.
        self._fact_counter += 1
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        target = f"f{self._fact_counter}.{content_hash}"

        # Compose MEMORY_SPEC notation: V[target]: content
        raw = f"V[{target}]: {content}"

        try:
            facts.write(self._conn, raw=raw, scope_id=scope_id)
        except MemoryFullError:
            # Graceful degradation: memory is full, skip this fact
            pass

    # Common stop words that hurt FTS5 matching on natural-language queries
    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "what", "which", "who", "whom", "where", "when", "why", "how",
        "that", "this", "these", "those", "it", "its",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
        "him", "her", "his", "them", "their",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "into", "about", "between", "through", "during", "before", "after",
        "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    })

    def _fts5_or_query(self, query: str) -> str:
        """Build an OR-based FTS5 query with stop-word filtering.

        FTS5 default AND logic fails on natural-language questions because
        question words (what, does, how) don't appear in stored facts.
        Using OR with content-word filtering gives much better recall.
        """
        import re
        tokens = query.strip().split()
        content_tokens = [
            t for t in tokens
            if t.lower() not in self._STOP_WORDS and len(t) > 1
        ]
        if not content_tokens:
            content_tokens = tokens[:3]  # Fallback: use first 3 tokens

        def _escape(tok: str) -> str:
            escaped = tok.replace('"', '""')
            if re.search(r'[^A-Za-z0-9_]', tok):
                return f'"{escaped}"'
            return f"{escaped}*"

        return " OR ".join(_escape(t) for t in content_tokens)

    def recall(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
    ) -> List[str]:
        """Return content strings matching the query via FTS5 search.

        Uses OR-based matching with stop-word removal for better recall
        on natural-language queries (questions, paraphrases).
        """
        scope_id: Optional[str] = None
        if scope and scope.lower() not in ("global", "none", ""):
            row = self._conn.execute(
                "SELECT id FROM sm_scopes WHERE label=? AND status='active'",
                (scope,),
            ).fetchone()
            if row:
                scope_id = row["id"]

        fts_query = self._fts5_or_query(query)
        if not fts_query:
            return []

        scope_filter = ""
        params: list = [fts_query, top_k]
        if scope_id:
            scope_filter = "AND f.scope_id = ?"
            params.insert(1, scope_id)

        rows = self._conn.execute(
            f"""
            SELECT f.content, rank
            FROM sm_facts_fts
            JOIN sm_facts f ON sm_facts_fts.rowid = f.rowid
            WHERE sm_facts_fts MATCH ?
              AND f.status IN ('active', 'cold')
              {scope_filter}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [r["content"] for r in rows]

    def simulate_time(self, days: float) -> None:
        """No-op: structured memory has no temporal decay mechanism."""
        pass

    def simulate_access(self, content_substring: str) -> None:
        """Simulate accessing a memory by triggering an FTS search.

        facts.search() increments access_count on matched rows, so
        querying with the content substring is sufficient.
        """
        facts.search(
            self._conn,
            query=content_substring,
            scope_id=None,
            limit=5,
        )

    def consolidate(self) -> None:
        """Run gauge pressure relief: merge duplicates, archive cold facts."""
        gauge.check_and_act(self._conn, llm_call=None)

    def get_stats(self) -> Dict[str, Any]:
        """Return fact count, gauge percentage, and scope count."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM sm_facts WHERE status IN ('active','cold')"
        ).fetchone()
        fact_count = row["cnt"] if row else 0

        g = gauge.read(self._conn)

        scope_row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM sm_scopes WHERE status='active'"
        ).fetchone()
        scope_count = scope_row["cnt"] if scope_row else 0

        return {
            "fact_count": fact_count,
            "gauge_pct": g.get("pct", 0.0),
            "used_chars": g.get("used_chars", 0),
            "max_chars": g.get("max_chars", 0),
            "active_scope_count": scope_count,
        }

    def reset(self) -> None:
        """Clear all stored memories by deleting all rows from every table."""
        self._conn.executescript("""
            DELETE FROM sm_facts;
            DELETE FROM sm_scopes;
            DELETE FROM sm_sessions;
            -- Rebuild the FTS5 index so it stays in sync
            INSERT INTO sm_facts_fts(sm_facts_fts) VALUES('rebuild');
        """)
        self._conn.commit()
        self._fact_counter = 0
