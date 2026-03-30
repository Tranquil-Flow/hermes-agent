#!/usr/bin/env python3
"""
Unified Memory Tool — agent-callable toolset wrapping UnifiedMemoryStore.

Exposes 8 tools backed by the unified_memory SQLite store:
    mcp_umemory_write       store a fact (plain text OR MEMORY_SPEC notation)
    mcp_umemory_recall      semantic + ACT-R ranked recall
    mcp_umemory_search      FTS5 keyword search (fast, no activation scoring)
    mcp_umemory_reflect     grouped reflection by type for a topic
    mcp_umemory_reward      apply RL reward signal to a memory's Q-value
    mcp_umemory_explore     multi-hop PPR exploration
    mcp_umemory_stats       store statistics (fact_count, gauge%, etc.)
    mcp_umemory_consolidate run consolidation cycle (promote/demote/prune)

Non-tool helpers (called by agent framework):
    get_unified_memory_injection(session_id?) -> str
        Build system prompt injection block with hot facts + gauge.
    tick_unified_memory(turn, message_text, session_id?)
        Advance turn counter and trigger scope auto-cooling.

Thread-safety: one UnifiedMemoryStore per OS thread via threading.local().
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Import UnifiedMemoryStore
# ---------------------------------------------------------------------------
try:
    from unified_memory.store import UnifiedMemoryStore
    from unified_memory.config import UnifiedMemoryConfig
    from unified_memory.retrieval import fts5_search

    _UM_AVAILABLE = True
except ImportError:
    _UM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Store pool — one UnifiedMemoryStore per thread
# ---------------------------------------------------------------------------

_DB_PATH = str(Path(os.getenv(
    "HERMES_UNIFIED_MEMORY_DB",
    str(Path.home() / ".hermes" / "unified_memory.db")
)))

_local = threading.local()


def _store() -> "UnifiedMemoryStore":
    """Return a per-thread UnifiedMemoryStore (lazy init)."""
    if not _UM_AVAILABLE:
        raise RuntimeError("unified_memory package is not installed.")
    if not getattr(_local, "store", None):
        config = UnifiedMemoryConfig.balanced()
        config.db_path = _DB_PATH
        _local.store = UnifiedMemoryStore(config=config, db_path=_DB_PATH)
    return _local.store


# ---------------------------------------------------------------------------
# Default session ID — one UUID per process; callers can override per call
# ---------------------------------------------------------------------------

_SESSION_ID = str(uuid.uuid4())


def _resolve_session(kwargs: dict) -> str:
    return kwargs.get("session_id") or _SESSION_ID


# ---------------------------------------------------------------------------
# TYPE display mapping (matches structured_memory conventions)
# ---------------------------------------------------------------------------

def _type_sym(fact_type) -> str:
    """Convert FactType enum or string to display symbol."""
    if hasattr(fact_type, 'value'):
        return fact_type.value  # FactType enum → "C", "D", "V", etc.
    return str(fact_type) if fact_type else "V"

_TYPE_DISPLAY = {
    "C": "C",    # Constraint
    "D": "D",    # Decision
    "V": "V",    # Value
    "?": "?",    # Open question
    "\u2713": "\u2713",  # Done/resolved
    "~": "~",    # Obsolete
    "factual": "V",
    "preference": "V",
    "procedural": "D",
    "episodic": "V",
}

_TYPE_LABELS = {
    "C": "Constraints",
    "D": "Decisions",
    "V": "Values",
    "\u2713": "Resolved",
    "~": "Obsolete",
    "?": "Open questions",
    "factual": "Facts",
    "preference": "Preferences",
    "procedural": "Procedures",
    "episodic": "Episodes",
}


# ---------------------------------------------------------------------------
# Tool handler functions
# ---------------------------------------------------------------------------

def _handle_mcp_umemory_write(args: dict, **kw) -> str:
    if not _UM_AVAILABLE:
        return "[UMEMORY ERROR] unified_memory package is not installed."

    try:
        s = _store()
        content = args.get("content", "").strip()
        if not content:
            return "[UMEMORY ERROR] content is required."

        scope = args.get("scope") or "global"
        fact_type = args.get("type") or None
        target = args.get("target") or None

        # store() accepts MEMORY_SPEC notation natively
        fact_id = s.store(
            content,
            scope=scope,
            fact_type=fact_type,
            target=target,
        )

        # Get gauge stats
        stats = s.get_stats()
        gauge_pct = stats.get("gauge_pct", 0.0)

        lines = [
            f"stored: {fact_id[:8]}",
            f"gauge:  {gauge_pct:.0f}%",
        ]

        if gauge_pct >= 85:
            lines.append(
                f"[UMEMORY WARNING] Store at {gauge_pct:.0f}% capacity. "
                "Consider calling mcp_umemory_consolidate to reclaim space."
            )

        return "\n".join(lines)

    except Exception as exc:
        return f"[UMEMORY ERROR] write failed: {exc}"


def _handle_mcp_umemory_recall(args: dict, **kw) -> str:
    if not _UM_AVAILABLE:
        return "[UMEMORY ERROR] unified_memory package is not installed."

    try:
        s = _store()
        query = args.get("query", "").strip()
        if not query:
            return "[UMEMORY ERROR] query is required."

        scope = args.get("scope") or None
        top_k = int(args.get("top_k", 10))
        top_k = min(top_k, 30)

        results = s.recall(query, scope=scope, top_k=top_k)

        if not results:
            return "no results"

        lines = []
        for r in results:
            fact = r.fact
            type_sym = _type_sym(fact.fact_type)
            target = fact.target or "general"
            score = round(r.score, 3)
            lines.append(f"{type_sym}[{target}]: {fact.content}  (score: {score})")

        return "\n".join(lines)

    except Exception as exc:
        return f"[UMEMORY ERROR] recall failed: {exc}"


def _handle_mcp_umemory_search(args: dict, **kw) -> str:
    if not _UM_AVAILABLE:
        return "[UMEMORY ERROR] unified_memory package is not installed."

    try:
        s = _store()
        query = args.get("query", "").strip()
        if not query:
            return "[UMEMORY ERROR] query is required."

        scope = args.get("scope") or None
        limit = int(args.get("limit", 10))
        limit = min(limit, 50)

        # Resolve scope to ID if provided
        scope_id = None
        if scope and scope.lower() not in ("global", "none", ""):
            scope_id = s._resolve_scope(scope)

        fts_scores = fts5_search(s.conn, query, scope_id=scope_id, limit=limit)

        if not fts_scores:
            return "no results"

        # Fetch fact details for matched IDs
        ids = list(fts_scores.keys())
        placeholders = ",".join("?" * len(ids))
        rows = s.conn.execute(
            f"SELECT id, type, target, content, status FROM um_facts "
            f"WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

        if not rows:
            return "no results"

        # Sort by FTS score (descending)
        rows_sorted = sorted(rows, key=lambda r: fts_scores.get(r["id"], 0), reverse=True)

        lines = []
        for r in rows_sorted:
            type_sym = _TYPE_DISPLAY.get(r["type"], r["type"] or "V")
            target = r["target"] or "general"
            status_tag = "" if r["status"] == "active" else f"  [{r['status']}]"
            lines.append(f"{type_sym}[{target}]: {r['content']}{status_tag}")

        return "\n".join(lines)

    except Exception as exc:
        return f"[UMEMORY ERROR] search failed: {exc}"


def _handle_mcp_umemory_reflect(args: dict, **kw) -> str:
    if not _UM_AVAILABLE:
        return "[UMEMORY ERROR] unified_memory package is not installed."

    try:
        s = _store()
        topic = args.get("topic", "").strip()
        if not topic:
            return "[UMEMORY ERROR] topic is required."

        limit = int(args.get("limit", 20))
        limit = min(limit, 50)

        results = s.recall(topic, top_k=limit)

        if not results:
            return f"no facts found for topic: {topic}"

        groups: dict[str, list[str]] = {}
        for r in results:
            fact = r.fact
            type_sym = _type_sym(fact.fact_type)
            target = fact.target or "general"
            groups.setdefault(type_sym, []).append(
                f"  [{target}]: {fact.content}"
            )

        # Ordered display: C, D, V, ✓, ~, ?  then anything else
        type_order = ["C", "D", "V", "\u2713", "~", "?"]
        other_types = [t for t in groups if t not in type_order]

        lines = [f"reflection: {topic}  ({len(results)} facts)", ""]
        for sym in type_order + other_types:
            if sym not in groups:
                continue
            label = _TYPE_LABELS.get(sym, sym)
            lines.append(f"{label}:")
            lines.extend(groups[sym])
            lines.append("")

        return "\n".join(lines).rstrip()

    except Exception as exc:
        return f"[UMEMORY ERROR] reflect failed: {exc}"


def _handle_mcp_umemory_reward(args: dict, **kw) -> str:
    if not _UM_AVAILABLE:
        return "[UMEMORY ERROR] unified_memory package is not installed."

    try:
        s = _store()
        memory_id = args.get("memory_id", "").strip()
        if not memory_id:
            return "[UMEMORY ERROR] memory_id is required."

        signal = float(args.get("signal", 0.0))
        signal = max(-1.0, min(1.0, signal))

        s.reward_memory(memory_id, signal)

        return f"rewarded: {memory_id[:8]}  signal: {signal:+.2f}"

    except Exception as exc:
        return f"[UMEMORY ERROR] reward failed: {exc}"


def _handle_mcp_umemory_explore(args: dict, **kw) -> str:
    if not _UM_AVAILABLE:
        return "[UMEMORY ERROR] unified_memory package is not installed."

    try:
        s = _store()
        query = args.get("query", "").strip()
        if not query:
            return "[UMEMORY ERROR] query is required."

        top_k = int(args.get("top_k", 20))
        top_k = min(top_k, 50)

        scope = args.get("scope") or None

        results = s.explore(query, top_k=top_k, scope=scope)

        if not results:
            return "no results"

        lines = []
        for r in results:
            fact = r.fact
            type_sym = _type_sym(fact.fact_type)
            target = fact.target or "general"
            score = round(r.score, 3)
            ppr_tag = "  [ppr]" if r.components.get("ppr_discovery") else ""
            lines.append(f"{type_sym}[{target}]: {fact.content}  (score: {score}){ppr_tag}")

        return "\n".join(lines)

    except Exception as exc:
        return f"[UMEMORY ERROR] explore failed: {exc}"


def _handle_mcp_umemory_stats(args: dict, **kw) -> str:
    if not _UM_AVAILABLE:
        return "[UMEMORY ERROR] unified_memory package is not installed."

    try:
        s = _store()
        stats = s.get_stats()

        lines = [
            f"fact_count:  {stats['fact_count']}",
            f"link_count:  {stats['link_count']}",
            f"scope_count: {stats['scope_count']}",
            f"gauge:       {stats['gauge_pct']:.1f}%  "
            f"({stats['used_chars']}/{stats['max_chars']} chars)",
        ]

        return "\n".join(lines)

    except Exception as exc:
        return f"[UMEMORY ERROR] stats failed: {exc}"


def _handle_mcp_umemory_consolidate(args: dict, **kw) -> str:
    if not _UM_AVAILABLE:
        return "[UMEMORY ERROR] unified_memory package is not installed."

    try:
        s = _store()
        report = s.consolidate()

        lines = [
            "consolidation complete:",
            f"  promoted:    {report.get('promoted', 0)}",
            f"  demoted:     {report.get('demoted', 0)}",
            f"  pruned:      {report.get('pruned', 0)}",
            f"  links_pruned:{report.get('links_pruned', 0)}",
        ]

        # Show updated gauge
        stats = s.get_stats()
        lines.append(f"gauge: {stats['gauge_pct']:.1f}%")

        return "\n".join(lines)

    except Exception as exc:
        return f"[UMEMORY ERROR] consolidate failed: {exc}"


# ---------------------------------------------------------------------------
# Public helpers called by the agent framework (not tool calls)
# ---------------------------------------------------------------------------

def get_unified_memory_injection(session_id: Optional[str] = None) -> str:
    """
    Build the system prompt injection block for unified memory.

    Called at agent startup. Returns a compact formatted string showing
    hot facts (active facts in active scopes) and gauge state.
    Returns empty string if the store is unavailable or there are no facts.
    """
    if not _UM_AVAILABLE:
        return ""

    try:
        s = _store()
        stats = s.get_stats()

        # Get hot facts: active facts ordered by activation (last_accessed desc)
        rows = s.conn.execute(
            "SELECT type, target, content FROM um_facts "
            "WHERE status='active' "
            "ORDER BY last_accessed DESC LIMIT 20"
        ).fetchall()

        if not rows:
            return ""

        gauge_pct = stats.get("gauge_pct", 0.0)
        used = stats.get("used_chars", 0)
        max_chars = stats.get("max_chars", 0)

        lines = [
            f"[UNIFIED MEMORY — {gauge_pct:.0f}% ({used}/{max_chars})]"
        ]

        for r in rows:
            type_sym = _TYPE_DISPLAY.get(r["type"], r["type"] or "V")
            target = r["target"] or "general"
            lines.append(f"{type_sym}[{target}]: {r['content']}")

        # Active scopes
        scope_rows = s.conn.execute(
            "SELECT label FROM um_scopes WHERE status='active' ORDER BY last_referenced DESC LIMIT 5"
        ).fetchall()
        if scope_rows:
            scope_labels = [r["label"] for r in scope_rows]
            lines.append(f"active scopes: {', '.join(scope_labels)}")

        return "\n".join(lines)

    except Exception:
        return ""


def tick_unified_memory(
    turn: int, message_text: str = "", session_id: Optional[str] = None
) -> None:
    """
    Advance the turn counter and trigger scope auto-cooling.

    Called by the agent loop on every user message. Never raises.
    """
    if not _UM_AVAILABLE:
        return

    try:
        s = _store()
        import time
        now = time.time()

        # Cool scopes that have been silent for too long
        if turn > 10:
            s.conn.execute(
                "UPDATE um_scopes SET status='cold' "
                "WHERE status='active' AND (? - last_referenced) > 3600",
                (now,)
            )
            s.conn.commit()

        # Auto-extract facts from assistant responses (every 3rd turn to avoid noise)
        if message_text and turn % 3 == 0 and len(message_text) > 50:
            try:
                from unified_memory.ingestion import extract_facts, compute_memorability
                facts = extract_facts(message_text)
                # Only store high-confidence facts to avoid noise
                for fact in facts:
                    if fact["confidence"] >= 0.6:
                        memorability = compute_memorability(
                            fact["content"], fact["fact_type"]
                        )
                        if memorability >= 0.5:
                            s.store(
                                content=fact["content"],
                                fact_type=fact["fact_type"].value,
                                target=fact["target"],
                                importance=memorability,
                            )
            except Exception:
                pass  # Ingestion failures are non-critical

    except Exception:
        pass  # Silent — never interrupt the agent loop


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_unified_memory_requirements() -> bool:
    """Returns True if the unified_memory package is installed and usable."""
    return _UM_AVAILABLE


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_MCP_UMEMORY_WRITE_SCHEMA = {
    "name": "mcp_umemory_write",
    "description": (
        "Store a new fact/memory in the unified memory store.\n"
        "Accepts plain text OR MEMORY_SPEC notation: TYPE[target]: content\n"
        "Types: C=constraint  D=decision  V=value  ?=unknown  \u2713=done  ~=obsolete\n"
        "Examples:\n"
        "  C[db.id]: UUID mandatory, never autoincrement\n"
        "  D[auth]: JWT 7d refresh 6d\n"
        "  V[api.prod]: api.example.com:3005\n"
        "  \u2713[auth]: deployed to prod\n"
        "Use this whenever a constraint, decision, fact, or value is established. "
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
                "description": "Scope label (e.g. 'auth-refactor', 'phase-b'). Defaults to 'global'.",
            },
            "type": {
                "type": "string",
                "description": "Fact type: C, D, V, ?, \u2713, ~. Auto-detected from notation if omitted.",
            },
            "target": {
                "type": "string",
                "description": "Target label (e.g. 'auth', 'db.id'). Auto-detected from notation if omitted.",
            },
        },
        "required": ["content"],
    },
}

_MCP_UMEMORY_RECALL_SCHEMA = {
    "name": "mcp_umemory_recall",
    "description": (
        "Recall memories matching a query using semantic similarity and ACT-R activation scoring.\n"
        "Uses 4-signal fusion: embedding similarity + ACT-R activation + FTS5/BM25 + Q-value reranking.\n"
        "Returns top-K facts ranked by relevance. Call this before answering questions "
        "that depend on past context, user preferences, or previously learned facts."
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
                "description": "Maximum number of results to return (default 10, max 30).",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}

_MCP_UMEMORY_SEARCH_SCHEMA = {
    "name": "mcp_umemory_search",
    "description": (
        "Fast FTS5 keyword search over facts (direct, no activation scoring).\n"
        "Use for exact keyword lookup or when recall is too slow.\n"
        "Returns matching facts sorted by relevance score."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'UUID database' or 'auth JWT'.",
            },
            "scope": {
                "type": "string",
                "description": "Restrict search to a specific scope label.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (1-50, default 10).",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}

_MCP_UMEMORY_REFLECT_SCHEMA = {
    "name": "mcp_umemory_reflect",
    "description": (
        "Synthesize all facts related to a topic, grouped by type.\n"
        "Use when the user asks 'what did we decide about X?' or before making\n"
        "a decision on a topic with long history. Read-only, no writes.\n"
        "Returns facts grouped by type: Constraints / Decisions / Values / etc."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Topic to reflect on, e.g. 'auth' or 'database schema'.",
            },
            "limit": {
                "type": "integer",
                "description": "Max facts to include in reflection (default 20, max 50).",
                "default": 20,
            },
        },
        "required": ["topic"],
    },
}

_MCP_UMEMORY_REWARD_SCHEMA = {
    "name": "mcp_umemory_reward",
    "description": (
        "Give feedback on whether a retrieved memory was useful (RL reward signal).\n"
        "This trains the Q-value reranking system — memories that receive positive rewards\n"
        "will rank higher in future recalls. Call after using a memory:\n"
        "  +1.0 = cited/directly used  +0.5 = referenced/updated\n"
        "  +0.6 = created new content after  +0.4 = re-recalled\n"
        "  -0.15 = irrelevant/dead end"
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
}

_MCP_UMEMORY_EXPLORE_SCHEMA = {
    "name": "mcp_umemory_explore",
    "description": (
        "Multi-hop memory exploration via Personalized PageRank (PPR).\n"
        "Unlike recall which finds directly matching memories, explore follows\n"
        "link connections to discover related memories a single query might miss.\n"
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
                "description": "Optional scope filter. Omit to search all memories.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return (default 20, max 50).",
                "default": 20,
            },
        },
        "required": ["query"],
    },
}

_MCP_UMEMORY_STATS_SCHEMA = {
    "name": "mcp_umemory_stats",
    "description": (
        "Return store statistics: fact count, link count, scope count, and gauge percentage.\n"
        "Use for a quick health check or to decide whether to consolidate."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}

_MCP_UMEMORY_CONSOLIDATE_SCHEMA = {
    "name": "mcp_umemory_consolidate",
    "description": (
        "Run a consolidation cycle on the unified memory store.\n"
        "Promotes frequently accessed working memories to core, demotes low-activation\n"
        "core memories to archive, prunes dead archive memories, and decays Hebbian links.\n"
        "Call at the end of long sessions or when mcp_umemory_stats shows high gauge%."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry  # noqa: E402 — must come after definitions

registry.register(
    name="mcp_umemory_write",
    toolset="unified_memory",
    schema=_MCP_UMEMORY_WRITE_SCHEMA,
    handler=lambda args, **kw: _handle_mcp_umemory_write(args, **kw),
    check_fn=check_unified_memory_requirements,
    emoji="\U0001f9e0",
)

registry.register(
    name="mcp_umemory_recall",
    toolset="unified_memory",
    schema=_MCP_UMEMORY_RECALL_SCHEMA,
    handler=lambda args, **kw: _handle_mcp_umemory_recall(args, **kw),
    check_fn=check_unified_memory_requirements,
    emoji="\U0001f4ad",
)

registry.register(
    name="mcp_umemory_search",
    toolset="unified_memory",
    schema=_MCP_UMEMORY_SEARCH_SCHEMA,
    handler=lambda args, **kw: _handle_mcp_umemory_search(args, **kw),
    check_fn=check_unified_memory_requirements,
    emoji="\U0001f50d",
)

registry.register(
    name="mcp_umemory_reflect",
    toolset="unified_memory",
    schema=_MCP_UMEMORY_REFLECT_SCHEMA,
    handler=lambda args, **kw: _handle_mcp_umemory_reflect(args, **kw),
    check_fn=check_unified_memory_requirements,
    emoji="\U0001f4a1",
)

registry.register(
    name="mcp_umemory_reward",
    toolset="unified_memory",
    schema=_MCP_UMEMORY_REWARD_SCHEMA,
    handler=lambda args, **kw: _handle_mcp_umemory_reward(args, **kw),
    check_fn=check_unified_memory_requirements,
    emoji="\u2b50",
)

registry.register(
    name="mcp_umemory_explore",
    toolset="unified_memory",
    schema=_MCP_UMEMORY_EXPLORE_SCHEMA,
    handler=lambda args, **kw: _handle_mcp_umemory_explore(args, **kw),
    check_fn=check_unified_memory_requirements,
    emoji="\U0001f30e",
)

registry.register(
    name="mcp_umemory_stats",
    toolset="unified_memory",
    schema=_MCP_UMEMORY_STATS_SCHEMA,
    handler=lambda args, **kw: _handle_mcp_umemory_stats(args, **kw),
    check_fn=check_unified_memory_requirements,
    emoji="\U0001f4ca",
)

registry.register(
    name="mcp_umemory_consolidate",
    toolset="unified_memory",
    schema=_MCP_UMEMORY_CONSOLIDATE_SCHEMA,
    handler=lambda args, **kw: _handle_mcp_umemory_consolidate(args, **kw),
    check_fn=check_unified_memory_requirements,
    emoji="\U0001f504",
)
