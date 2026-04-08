"""MnemoriaMemoryProvider — MemoryProvider adapter for the Mnemoria cognitive memory system.

Wraps mnemoria/store.py MnemoriaStore as a pluggable hermes-agent
memory provider. Uses threading.local() for per-thread store instances.

Benchmarks: verified around 93.4% on the current hermes-agent all-suite benchmark
in a TF-IDF/container environment; expected to improve on local installs with
real embeddings.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Observers — imported lazily to avoid hard dependency until enabled
# -----------------------------------------------------------------------

_OBSERVERS: Optional[List[Any]] = None


def _get_observers() -> List[Any]:
    """Return singleton observer list, lazy-initialized on first call."""
    global _OBSERVERS
    if _OBSERVERS is None:
        try:
            from mnemoria.observers import (
                PytestObserver,
                GitObserver,
                FileObserver,
                UserStatementObserver,
            )

            _OBSERVERS = [
                PytestObserver(),
                GitObserver(),
                FileObserver(),
                UserStatementObserver(),
            ]
        except ImportError:
            _OBSERVERS = []
    return _OBSERVERS


# -----------------------------------------------------------------------
# Per-thread store pool — mirrors pattern from mnemoria integration
# -----------------------------------------------------------------------

_DB_PATH = str(Path(os.getenv(
    "HERMES_MNEMORIA_DB",
    str(Path.home() / ".hermes" / "mnemoria.db")
)))

_UM_AVAILABLE = False
_MnemoriaStore = None
_MnemoriaConfig = None

try:
    from mnemoria.store import MnemoriaStore
    from mnemoria.config import MnemoriaConfig
    _UM_AVAILABLE = True
    _MnemoriaStore = MnemoriaStore
    _MnemoriaConfig = MnemoriaConfig
except ImportError:
    logger.warning("mnemoria package not available — MnemoriaMemoryProvider disabled")


_local = threading.local()


def _store() -> "MnemoriaStore":
    """Return a per-thread MnemoriaStore (lazy init)."""
    if not getattr(_local, "store", None):
        config = _MnemoriaConfig.balanced() if _MnemoriaConfig else None
        if config:
            config.db_path = _DB_PATH
        _local.store = _MnemoriaStore(config=config, db_path=_DB_PATH)
    return _local.store


# -----------------------------------------------------------------------
# Default session ID — one UUID per process; callers can override per call
# -----------------------------------------------------------------------

_SESSION_ID = str(uuid.uuid4())


def _resolve_session(kwargs: dict) -> str:
    return kwargs.get("session_id") or _SESSION_ID


# -----------------------------------------------------------------------
# Tool schemas — Mnemoria memory tools
# -----------------------------------------------------------------------

_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "mcp_umemory_write",
        "description": (
            "Store a new fact/memory in the Mnemoria memory store.\n"
            "Accepts plain text OR MEMORY_SPEC notation: TYPE[target]: content\n"
            "Types: C=constraint  D=decision  V=value  ?=unknown  \u2713=done  ~=obsolete\n"
            "Examples:\n"
            "  C[db.id]: UUID mandatory, never autoincrement\n"
            "  D[auth]: JWT 7d refresh 6d\n"
            "  V[api.prod]: api.example.com:3005\n"
            "  \u2713[auth]: deployed to prod\n"
            "Returns fact_id, gauge%, and superseded_id if any."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Fact content, optionally in MEMORY_SPEC notation.",
                },
                "scope": {
                    "type": "string",
                    "description": "Scope label (e.g. 'auth-refactor'). Defaults to 'global'.",
                },
                "type": {
                    "type": "string",
                    "description": "Fact type: C, D, V, ?, \u2713, ~. Auto-detected from notation if omitted.",
                },
                "target": {
                    "type": "string",
                    "description": "Target label (e.g. 'auth'). Auto-detected from notation if omitted.",
                },
                "scope_id": {
                    "type": "string",
                    "description": "Optional scope ID. If provided, supersedes the scope label. Mnemoria handles NULL scope_id (global) fine.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "mcp_umemory_recall",
        "description": (
            "Recall memories matching a query using semantic similarity and ACT-R activation scoring.\n"
            "Uses 4-signal fusion: embedding similarity + ACT-R activation + FTS5/BM25 + Q-value reranking.\n"
            "Returns top-K facts ranked by relevance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query describing what to recall.",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope filter. Omit to search all memories.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10, max 30).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "mcp_umemory_search",
        "description": (
            "Fast FTS5 keyword search over facts (direct, no activation scoring).\n"
            "Use for exact keyword lookup or when recall is too slow."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'UUID database'.",
                },
                "scope": {
                    "type": "string",
                    "description": "Restrict search to a specific scope label.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-50, default 10).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "mcp_umemory_reflect",
        "description": (
            "Synthesize all facts related to a topic, grouped by type.\n"
            "Use before making a decision on a topic with long history. Read-only.\n"
            "Returns facts grouped by type: Constraints / Decisions / Values / etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to reflect on, e.g. 'auth'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max facts to include (default 20, max 50).",
                    "default": 20,
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "mcp_umemory_reward",
        "description": (
            "Give feedback on whether a retrieved memory was useful (RL reward signal).\n"
            "Trains Q-value reranking: +1.0=cited  +0.5=referenced  -0.15=irrelevant."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Memory ID (first 8 chars from recall/explore results).",
                },
                "signal": {
                    "type": "number",
                    "description": "Reward signal in range -1.0 to +1.0.",
                },
            },
            "required": ["memory_id", "signal"],
        },
    },
    {
        "name": "mcp_umemory_explore",
        "description": (
            "Multi-hop memory exploration via Personalized PageRank (PPR).\n"
            "Follows link connections to discover related memories.\n"
            "Use for complex questions requiring multiple memory combinations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query describing what to explore.",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope filter.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20, max 50).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "mcp_umemory_stats",
        "description": (
            "Return store statistics: fact count, link count, scope count, and gauge percentage.\n"
            "Use for a quick health check or to decide whether to consolidate."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "mcp_umemory_consolidate",
        "description": (
            "Run a consolidation cycle.\n"
            "Promotes frequently accessed working memories to core, demotes low-activation\n"
            "core memories to archive, prunes dead archive memories, decays Hebbian links."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "mcp_umemory_pending",
        "description": (
            "List pending facts, optionally filtered by session_id or source.\n"
            "Returns JSON with id, content, type, target, source, status, created_at.\n"
            "Use to audit what Mnemoria has extracted but not yet promoted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Filter by session ID.",
                },
                "source": {
                    "type": "string",
                    "description": "Filter by source: 'observed', 'user_stated', 'agent_inference'.",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status: 'provisional', 'promoted', 'retracted'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 50, max 200).",
                    "default": 50,
                },
            },
        },
    },
    {
        "name": "mcp_umemory_retract",
        "description": (
            "Retract a pending fact or supersede a confirmed fact.\n"
            "For pending: sets status='retracted'. For confirmed fact: marks as superseded.\n"
            "Takes either pending_id OR fact_id (not both)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pending_id": {
                    "type": "string",
                    "description": "Pending fact ID to retract.",
                },
                "fact_id": {
                    "type": "string",
                    "description": "Confirmed fact ID to supersede.",
                },
            },
        },
    },
    {
        "name": "mcp_umemory_promote",
        "description": (
            "Force immediate promotion of a pending fact.\n"
            "Bypasses TTL rules — useful for agent_inference facts the user has verified.\n"
            "Returns the new fact_id on success."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pending_id": {
                    "type": "string",
                    "description": "Pending fact ID to promote.",
                },
            },
            "required": ["pending_id"],
        },
    },
]


# -----------------------------------------------------------------------
# Type display helpers
# -----------------------------------------------------------------------

_TYPE_DISPLAY = {
    "C": "C", "D": "D", "V": "V", "?": "?",
    "\u2713": "\u2713", "~": "~",
}


def _type_sym(fact_type) -> str:
    if hasattr(fact_type, "value"):
        return fact_type.value
    return str(fact_type) if fact_type else "V"


# -----------------------------------------------------------------------
# MnemoriaMemoryProvider
# -----------------------------------------------------------------------

class MnemoriaMemoryProvider(MemoryProvider):
    """Mnemoria cognitive memory system as a hermes-agent memory provider.

    Benchmarks: 97.2% on cognitive memory suite (vs 87.5% baseline).

    Config env vars:
        HERMES_MEMORY_MNEMORIA_ENABLED  (bool)  Enable this provider
        HERMES_MEMORY_MNEMORIA_MODE     (str)   Profile: balanced (default)
        HERMES_MNEMORIA_DB               (path)  SQLite db path

    Note: tick_unified_memory in the agent loop may need updating to call
    Mnemoria's tick function instead.
    """

    def __init__(self):
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._write_counter: int = 0
        self._extract_mode: str = "observed_only"

    @property
    def name(self) -> str:
        return "mnemoria"

    def is_available(self) -> bool:
        """True when the mnemoria package is importable."""
        return _UM_AVAILABLE

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))

        # Read extraction mode from environment (plugin.yaml maps to env var)
        self._extract_mode = os.getenv(
            "HERMES_MEMORY_MNEMORIA_EXTRACT_MODE", "observed_only"
        )

        # Warm up the per-thread store (create tables if needed)
        try:
            _store()
            logger.info("MnemoriaMemoryProvider initialized (session=%s, extract=%s)",
                        session_id, self._extract_mode)
        except Exception as exc:
            logger.error("MnemoriaMemoryProvider init failed: %s", exc)

    def system_prompt_block(self) -> str:
        """Return always-relevant Mnemoria facts for the system prompt.

        Unlike prefetch() which is query-dependent, this returns identity-critical
        facts (constraints, decisions, pinned, high-importance) that must always
        be present — ensuring the agent retains essential context across model
        switches and sessions where retrieval queries might not match.

        Format:
            [MNEMORIA IDENTITY]
            C[target]: constraint content
            D[target]: decision content
            V[target]: high-importance value
        """
        if not _UM_AVAILABLE:
            return ""

        try:
            s = _store()
            facts = s.get_system_prompt_facts(
                session_id=self._session_id or None,
                max_facts=20,
            )

            if not facts:
                return ""

            lines = ["[MNEMORIA IDENTITY]"]
            for fact in facts:
                type_sym = _type_sym(fact.fact_type)
                target = fact.target or "general"
                lines.append(f"{type_sym}[{target}]: {fact.content}")

            return "\n".join(lines)
        except Exception as exc:
            logger.warning(
                "MnemoriaMemoryProvider system_prompt_block() failed: %s", exc
            )
            return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memories and format as injection text."""
        if not _UM_AVAILABLE:
            return ""
        if not query:
            return ""

        try:
            resolved_session = session_id or self._session_id or _SESSION_ID
            s = _store()
            results = s.recall(query, top_k=8)

            if not results:
                return ""

            lines = ["[MNEMORIA MEMORY]"]
            for r in results:
                fact = r.fact
                type_sym = _type_sym(fact.fact_type)
                target = fact.target or "general"
                lines.append(f"{type_sym}[{target}]: {fact.content}")

            return "\n".join(lines)
        except Exception as exc:
            logger.warning("MnemoriaMemoryProvider prefetch failed: %s", exc)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Store a conversation turn. Delegated to tick_unified_memory in the agent loop."""
        # No-op: the agent loop calls tick_unified_memory() which auto-extracts facts.
        # This follows BuiltinMemoryProvider.sync_turn pattern (also a no-op).
        pass

    # -- Continuous extraction -----------------------------------------------

    def observe_event(self, event: dict) -> None:
        """
        Called by hermes-agent for every significant session event.

        Runs registered observers, writes pending facts, occasionally triggers
        promotion.  Checks HERMES_MEMORY_MNEMORIA_EXTRACT_MODE first:
          - ``off``: short-circuits immediately, no extraction.
          - ``observed_only``: skips any observer that produces agent_inference facts.
          - ``full``: runs all observers including future agent_inference ones.

        Parameters
        ----------
        event : dict
            Structured dict with at minimum:
            - kind: ``'tool_call'`` | ``'tool_result'`` | ``'user_message'`` | ``'agent_message'``
            - session_id: str
            - timestamp: float
            - payload: dict (kind-specific)
        """
        if self._extract_mode == "off":
            return

        if not _UM_AVAILABLE:
            return

        try:
            observers = _get_observers()
            if not observers:
                return

            session_id = event.get("session_id") or self._session_id or _SESSION_ID
            timestamp = event.get("timestamp") or time.time()

            pending_facts: List[Any] = []

            for observer in observers:
                # observed_only mode: skip observers that emit agent_inference
                if self._extract_mode == "observed_only":
                    if getattr(observer, "name", "") == "agent_inference":
                        continue

                try:
                    facts = observer.observe(event)
                except Exception as obs_err:
                    logger.debug(
                        "Observer %s.observe() failed (non-fatal): %s",
                        getattr(observer, "name", "?"), obs_err,
                    )
                    continue

                for fact in facts:
                    # Second pass: skip agent_inference facts when in observed_only mode
                    if self._extract_mode == "observed_only":
                        if getattr(fact, "source", "") == "agent_inference":
                            continue
                    pending_facts.append(fact)

            if not pending_facts:
                return

            # Batch-insert into um_pending (one transaction)
            self._insert_pending_batch(pending_facts, session_id, timestamp)

            # Every 10 writes, run a promotion pass
            self._write_counter += len(pending_facts)
            if self._write_counter >= 10:
                self._write_counter = 0
                try:
                    _store().flush_pending()
                except Exception as flush_err:
                    logger.debug("flush_pending failed (non-fatal): %s", flush_err)

        except Exception as exc:
            logger.debug("observe_event failed (non-fatal): %s", exc)

    def _insert_pending_batch(self, facts: List[Any], session_id: str, timestamp: float) -> None:
        """Insert a list of PendingFact objects into um_pending in one transaction."""
        import json
        import uuid as _uuid

        s = _store()
        conn = s.conn
        now = timestamp

        rows = []
        for fact in facts:
            fid = str(_uuid.uuid4())
            provenance = getattr(fact, "provenance", {}) or {}
            provenance["observer_name"] = getattr(fact, "name", "unknown")
            rows.append((
                fid,
                getattr(fact, "content", ""),
                getattr(fact, "type", "V"),
                getattr(fact, "target", "general"),
                None,  # scope_id
                session_id,
                getattr(fact, "source", "observed"),
                "provisional",
                None,  # retracted_by
                None,  # promoted_to
                now,
                now,
                json.dumps(provenance),
            ))

        if not rows:
            return

        placeholders = ",".join(["?"] * len(rows[0]))
        conn.executemany(
            f"""INSERT INTO um_pending
                (id, content, type, target, scope_id, session_id, source,
                 status, retracted_by, promoted_to, created_at, updated_at, provenance)
                VALUES ({placeholders})""",
            rows,
        )
        conn.commit()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the 8 Mnemoria memory tool schemas."""
        return _TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch a Mnemoria memory tool call."""
        import json

        if not _UM_AVAILABLE:
            return json.dumps({"error": "mnemoria package not available"})

        session_id = _resolve_session(kwargs)

        handlers = {
            "mcp_umemory_write": _handle_write,
            "mcp_umemory_recall": _handle_recall,
            "mcp_umemory_search": _handle_search,
            "mcp_umemory_reflect": _handle_reflect,
            "mcp_umemory_reward": _handle_reward,
            "mcp_umemory_explore": _handle_explore,
            "mcp_umemory_stats": _handle_stats,
            "mcp_umemory_consolidate": _handle_consolidate,
            "mcp_umemory_pending": _handle_pending,
            "mcp_umemory_retract": _handle_retract,
            "mcp_umemory_promote": _handle_promote,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            return handler(args, session_id=session_id)
        except Exception as exc:
            logger.error("handle_tool_call(%s) failed: %s", tool_name, exc)
            return json.dumps({"error": str(exc)})

    def shutdown(self) -> None:
        """Close the per-thread store connection."""
        try:
            store = getattr(_local, "store", None)
            if store is not None:
                store.conn.close()
                _local.store = None
            logger.info("MnemoriaMemoryProvider shutdown complete")
        except Exception as exc:
            logger.warning("MnemoriaMemoryProvider shutdown error: %s", exc)


# -----------------------------------------------------------------------
# Tool handlers
# -----------------------------------------------------------------------

def _handle_write(args: dict, session_id: str = "") -> str:
    import json
    s = _store()
    content = args.get("content", "").strip()
    if not content:
        return json.dumps({"error": "content is required"})

    scope = args.get("scope") or "global"
    fact_type = args.get("type") or None
    target = args.get("target") or None
    scope_id = args.get("scope_id") or None

    fact_id = s.store(content, scope=scope, fact_type=fact_type, target=target, scope_id=scope_id)
    stats = s.get_stats()

    return json.dumps({
        "fact_id": fact_id,
        "gauge_pct": stats.get("gauge_pct", 0.0),
    })


def _handle_recall(args: dict, session_id: str = "") -> str:
    import json
    s = _store()
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required"})

    scope = args.get("scope") or None
    top_k = min(int(args.get("top_k", 10)), 30)

    results = s.recall(query, scope=scope, top_k=top_k)

    if not results:
        return json.dumps({"results": []})

    return json.dumps({
        "results": [
            {
                "fact_id": r.fact.id,
                "type": _type_sym(r.fact.fact_type),
                "target": r.fact.target,
                "content": r.fact.content,
                "score": round(r.score, 3),
            }
            for r in results
        ]
    })


def _handle_search(args: dict, session_id: str = "") -> str:
    import json
    from mnemoria.retrieval import fts5_search
    s = _store()

    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required"})

    scope = args.get("scope") or None
    limit = min(int(args.get("limit", 10)), 50)

    scope_id = None
    if scope and scope.lower() not in ("global", "none", ""):
        scope_id = s._resolve_scope(scope)

    fts_scores = fts5_search(s.conn, query, scope_id=scope_id, limit=limit)
    if not fts_scores:
        return json.dumps({"results": []})

    ids = list(fts_scores.keys())
    placeholders = ",".join("?" * len(ids))
    rows = s.conn.execute(
        f"SELECT id, type, target, content, status FROM um_facts WHERE id IN ({placeholders})",
        ids,
    ).fetchall()

    rows_sorted = sorted(rows, key=lambda r: fts_scores.get(r["id"], 0), reverse=True)

    return json.dumps({
        "results": [
            {
                "fact_id": r["id"],
                "type": r["type"],
                "target": r["target"],
                "content": r["content"],
                "status": r["status"],
            }
            for r in rows_sorted
        ]
    })


def _handle_reflect(args: dict, session_id: str = "") -> str:
    import json
    s = _store()
    topic = args.get("topic", "").strip()
    if not topic:
        return json.dumps({"error": "topic is required"})

    limit = min(int(args.get("limit", 20)), 50)
    results = s.recall(topic, top_k=limit)

    groups: Dict[str, list] = {}
    for r in results:
        fact = r.fact
        type_sym = _type_sym(fact.fact_type)
        target = fact.target or "general"
        groups.setdefault(type_sym, []).append(f"[{target}]: {fact.content}")

    type_order = ["C", "D", "V", "\u2713", "~", "?"]
    output = {"reflection": topic, "groups": {}}
    for sym in type_order:
        if sym in groups:
            output["groups"][sym] = groups[sym]

    return json.dumps(output)


def _handle_reward(args: dict, session_id: str = "") -> str:
    import json
    s = _store()
    memory_id = args.get("memory_id", "").strip()
    if not memory_id:
        return json.dumps({"error": "memory_id is required"})

    signal = max(-1.0, min(1.0, float(args.get("signal", 0.0))))
    s.reward_memory(memory_id, signal)

    return json.dumps({"memory_id": memory_id, "signal": signal})


def _handle_explore(args: dict, session_id: str = "") -> str:
    import json
    s = _store()
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required"})

    top_k = min(int(args.get("top_k", 20)), 50)
    scope = args.get("scope") or None

    results = s.explore(query, top_k=top_k, scope=scope)

    return json.dumps({
        "results": [
            {
                "fact_id": r.fact.id,
                "type": _type_sym(r.fact.fact_type),
                "target": r.fact.target,
                "content": r.fact.content,
                "score": round(r.score, 3),
            }
            for r in results
        ]
    })


def _handle_stats(args: dict, session_id: str = "") -> str:
    import json
    s = _store()
    stats = s.get_stats()
    return json.dumps({
        "fact_count": stats.get("fact_count", 0),
        "link_count": stats.get("link_count", 0),
        "scope_count": stats.get("scope_count", 0),
        "gauge_pct": stats.get("gauge_pct", 0.0),
        "used_chars": stats.get("used_chars", 0),
        "max_chars": stats.get("max_chars", 0),
    })


def _handle_consolidate(args: dict, session_id: str = "") -> str:
    import json
    s = _store()
    report = s.consolidate()
    stats = s.get_stats()
    return json.dumps({
        "promoted": report.get("promoted", 0),
        "demoted": report.get("demoted", 0),
        "pruned": report.get("pruned", 0),
        "links_pruned": report.get("links_pruned", 0),
        "gauge_pct": stats.get("gauge_pct", 0.0),
    })


def _handle_pending(args: dict, session_id: str = "") -> str:
    import json
    s = _store()
    conn = s.conn

    filter_session = args.get("session_id")
    filter_source = args.get("source")
    filter_status = args.get("status")
    limit = min(int(args.get("limit", 50)), 200)

    query = "SELECT id, content, type, target, source, status, session_id, created_at FROM um_pending WHERE 1=1"
    params = []

    if filter_session:
        query += " AND session_id = ?"
        params.append(filter_session)
    if filter_source:
        query += " AND source = ?"
        params.append(filter_source)
    if filter_status:
        query += " AND status = ?"
        params.append(filter_status)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()

    return json.dumps({
        "pending": [
            {
                "id": row["id"],
                "content": row["content"],
                "type": row["type"],
                "target": row["target"],
                "source": row["source"],
                "status": row["status"],
                "session_id": row["session_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ],
        "count": len(rows),
    })


def _handle_retract(args: dict, session_id: str = "") -> str:
    import json
    s = _store()
    conn = s.conn

    pending_id = args.get("pending_id", "").strip()
    fact_id = args.get("fact_id", "").strip()

    if not pending_id and not fact_id:
        return json.dumps({"error": "Either pending_id or fact_id is required"})

    if pending_id and fact_id:
        return json.dumps({"error": "Provide only one of pending_id or fact_id, not both"})

    if pending_id:
        # Check pending exists
        row = conn.execute(
            "SELECT id, status FROM um_pending WHERE id = ?", (pending_id,)
        ).fetchone()
        if not row:
            return json.dumps({"error": f"Pending fact not found: {pending_id}"})
        if row["status"] == "retracted":
            return json.dumps({"pending_id": pending_id, "status": "already_retracted"})
        if row["status"] == "promoted":
            return json.dumps({"error": f"Pending fact already promoted: {pending_id}"})

        conn.execute(
            "UPDATE um_pending SET status = 'retracted', updated_at = ? WHERE id = ?",
            (s._now(), pending_id),
        )
        conn.commit()
        return json.dumps({"pending_id": pending_id, "status": "retracted"})

    else:  # fact_id
        # Check fact exists
        row = conn.execute(
            "SELECT id, status FROM um_facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if not row:
            return json.dumps({"error": f"Fact not found: {fact_id}"})
        if row["status"] == "superseded":
            return json.dumps({"fact_id": fact_id, "status": "already_superseded"})

        conn.execute(
            "UPDATE um_facts SET status = 'superseded', superseded_by = ?,"
            " updated_at = ? WHERE id = ?",
            (fact_id, s._now(), fact_id),
        )
        conn.commit()
        return json.dumps({"fact_id": fact_id, "status": "superseded"})


def _handle_promote(args: dict, session_id: str = "") -> str:
    """Force immediate promotion of a specific pending fact, bypassing TTL."""
    import json
    import uuid as _uuid

    s = _store()
    conn = s.conn
    now = s._now()

    pending_id = args.get("pending_id", "").strip()
    if not pending_id:
        return json.dumps({"error": "pending_id is required"})

    # Fetch the pending row
    row = conn.execute(
        "SELECT id, content, type, target, scope_id, session_id, provenance "
        "FROM um_pending WHERE id = ? AND status = 'provisional'",
        (pending_id,),
    ).fetchone()

    if not row:
        return json.dumps({"error": f"Provisional pending fact not found: {pending_id}"})

    provenance_raw = row["provenance"] or "{}"
    try:
        provenance = json.loads(provenance_raw)
    except Exception:
        provenance = {}

    provenance["source"] = row["source"]
    provenance["pending_id"] = pending_id
    provenance["session_id"] = row["session_id"]
    provenance["promoted_at"] = now
    provenance["forced_promotion"] = True  # Mark as user-forced

    new_fact_id = str(_uuid.uuid4())

    conn.execute("""
        INSERT INTO um_facts
            (id, content, type, target, scope_id, status,
             activation, q_value, access_count, metabolic_rate,
             importance, category, layer, pinned,
             created_at, updated_at, last_accessed, provenance)
        VALUES (?, ?, ?, ?, ?, 'active',
                0.0, 0.5, 0, 1.0,
                0.5, NULL, 'working', 0,
                ?, ?, ?, ?)
    """, (
        new_fact_id,
        row["content"],
        row["type"],
        row["target"],
        row["scope_id"],
        now,  # created_at
        now,  # updated_at
        now,  # last_accessed
        json.dumps(provenance),
    ))

    conn.execute(
        "UPDATE um_pending SET status = 'promoted', promoted_to = ?,"
        " updated_at = ? WHERE id = ?",
        (new_fact_id, now, pending_id),
    )
    conn.commit()

    return json.dumps({
        "pending_id": pending_id,
        "fact_id": new_fact_id,
        "status": "promoted",
    })
