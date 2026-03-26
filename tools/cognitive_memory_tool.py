"""Cognitive Memory Tool — ACT-R-scored episodic/semantic memory for Hermes Agent.

Wraps CognitiveMemoryStore: embedding-based recall with ACT-R activation scoring,
Hebbian link formation, contradiction detection, and 3-layer consolidation
(working → core → archive).

Two tools:
  cognitive_recall    — retrieve relevant memories by semantic query
  cognitive_store     — store a new memory with auto-classification
  cognitive_consolidate — trigger a consolidation cycle (promote/archive/prune)

The store singleton is injected by AIAgent.__init__ via set_cognitive_store().
"""

import json
import logging
from typing import Any, Dict

from tools.registry import registry

logger = logging.getLogger(__name__)

# ── Module-level singleton (injected at init time by AIAgent) ──

_cognitive_store = None  # CognitiveMemoryStore | None


def set_cognitive_store(store) -> None:
    """Register the active CognitiveMemoryStore. Called by AIAgent.__init__."""
    global _cognitive_store
    _cognitive_store = store


def clear_cognitive_store() -> None:
    """Clear singleton (for testing)."""
    global _cognitive_store
    _cognitive_store = None


def _check_available() -> bool:
    return _cognitive_store is not None


# ── cognitive_recall ──────────────────────────────────────────────────────────

_RECALL_SCHEMA = {
    "name": "cognitive_recall",
    "description": (
        "Retrieve memories from the cognitive store using semantic similarity "
        "and ACT-R activation scoring. Returns the most relevant memories for a "
        "given query, ranked by recency × importance × semantic match. "
        "Use this before answering questions that depend on past context, user "
        "preferences, or previously learned facts."
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
                "description": (
                    "Optional scope filter: 'global', 'project:<name>', or 'topic:<name>'. "
                    "Omit to search all memories."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of memories to return (default 10).",
            },
        },
        "required": ["query"],
    },
}


def _handle_recall(args: Dict[str, Any], **kwargs) -> str:
    store = _cognitive_store
    if store is None:
        return json.dumps({"error": "Cognitive memory store not initialized."})

    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required."})

    scope = args.get("scope") or None
    top_k = args.get("top_k") or None

    try:
        results = store.recall(query, scope=scope, top_k=top_k)
    except Exception as e:
        logger.exception("cognitive_recall error")
        return json.dumps({"error": f"Recall failed: {e}"})

    memories = []
    for sm in results:
        e = sm.entry
        memories.append({
            "id": e.id[:8],
            "content": e.content,
            "category": e.category,
            "layer": e.layer,
            "importance": round(e.importance, 3),
            "score": round(sm.score, 4),
            "scope": e.scope,
            "access_count": e.access_count,
        })

    return json.dumps({"memories": memories, "count": len(memories)}, ensure_ascii=False)


# ── cognitive_store ───────────────────────────────────────────────────────────

_STORE_SCHEMA = {
    "name": "cognitive_store",
    "description": (
        "Store a new memory in the cognitive store. The content is automatically "
        "classified by category (factual, preference, procedural, episodic, etc.) "
        "and assigned an importance score. Embeddings and Hebbian links are created "
        "automatically. Contradictions with existing memories are detected and flagged. "
        "Use this to record important facts, user preferences, learned procedures, "
        "or any information worth preserving across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory content to store.",
            },
            "scope": {
                "type": "string",
                "description": (
                    "Optional scope: 'global' (default), 'project:<name>', or 'topic:<name>'. "
                    "Scoped memories are boosted when recalling within the same scope."
                ),
            },
            "importance": {
                "type": "number",
                "description": "Importance 0.0–1.0 (auto-assigned if omitted).",
            },
            "pinned": {
                "type": "boolean",
                "description": "Pin the memory so it is never archived or pruned.",
            },
        },
        "required": ["content"],
    },
}


def _handle_store(args: Dict[str, Any], **kwargs) -> str:
    store = _cognitive_store
    if store is None:
        return json.dumps({"error": "Cognitive memory store not initialized."})

    content = args.get("content", "").strip()
    if not content:
        return json.dumps({"error": "content is required."})

    scope = args.get("scope", "global")
    importance = args.get("importance")
    pinned = bool(args.get("pinned", False))

    try:
        mem_id = store.store(
            content,
            scope=scope,
            importance=importance,
            source="agent",
            pinned=pinned,
        )
    except Exception as e:
        logger.exception("cognitive_store error")
        return json.dumps({"error": f"Store failed: {e}"})

    return json.dumps({"stored": True, "id": mem_id[:8]}, ensure_ascii=False)


# ── cognitive_consolidate ─────────────────────────────────────────────────────

_CONSOLIDATE_SCHEMA = {
    "name": "cognitive_consolidate",
    "description": (
        "Trigger a cognitive memory consolidation cycle. Promotes frequently accessed "
        "working memories to core, demotes low-activation core memories to archive, "
        "prunes dead archive memories, and decays Hebbian link weights. "
        "Call at the end of long sessions or when memory feels cluttered."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def _handle_consolidate(args: Dict[str, Any], **kwargs) -> str:
    store = _cognitive_store
    if store is None:
        return json.dumps({"error": "Cognitive memory store not initialized."})

    try:
        report = store.consolidate()
    except Exception as e:
        logger.exception("cognitive_consolidate error")
        return json.dumps({"error": f"Consolidation failed: {e}"})

    return json.dumps({
        "promoted_to_core": report.promoted_to_core,
        "demoted_to_archive": report.demoted_to_archive,
        "pruned": report.pruned,
        "links_pruned": report.links_pruned,
        "links_decayed": report.links_decayed,
    }, ensure_ascii=False)


# ── cognitive_explore ─────────────────────────────────────────────────────

_EXPLORE_SCHEMA = {
    "name": "cognitive_explore",
    "description": (
        "Multi-hop memory exploration using Personalized PageRank graph walking. "
        "Unlike cognitive_recall which finds directly matching memories, explore "
        "follows link connections to discover related memories that a single query "
        "might miss. Use this for complex questions that require combining information "
        "from multiple memories, or when recall returns incomplete answers."
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
                "description": (
                    "Optional scope filter: 'global', 'project:<name>', or 'topic:<name>'. "
                    "Omit to search all memories."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of memories to return (default 20).",
            },
        },
        "required": ["query"],
    },
}


def _handle_explore(args: Dict[str, Any], **kwargs) -> str:
    store = _cognitive_store
    if store is None:
        return json.dumps({"error": "Cognitive memory store not initialized."})

    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required."})

    scope = args.get("scope") or None
    top_k = args.get("top_k") or 20

    try:
        result = store.explore(query, scope=scope, top_k=top_k)
    except Exception as e:
        logger.exception("cognitive_explore error")
        return json.dumps({"error": f"Explore failed: {e}"})

    memories = []
    for sm in result.results:
        e = sm.entry
        memories.append({
            "id": e.id[:8],
            "content": e.content,
            "category": e.category,
            "layer": e.layer,
            "importance": round(e.importance, 3),
            "score": round(sm.score, 4),
            "scope": e.scope,
            "ppr_discovery": sm.components.get("ppr_discovery", False),
        })

    return json.dumps({
        "memories": memories,
        "count": len(memories),
        "candidates_visited": result.total_candidates_visited,
        "rounds": result.rounds,
    }, ensure_ascii=False)


# ── cognitive_reward ─────────────────────────────────────────────────────

_REWARD_SCHEMA = {
    "name": "cognitive_reward",
    "description": (
        "Give feedback on whether a retrieved memory was useful. This trains "
        "the Q-value reranking system — memories that receive positive rewards "
        "will rank higher in future recalls, and penalized memories will sink. "
        "Call this after using a memory to confirm it helped (+1.0), after "
        "updating a memory (+0.5), or when a top-ranked memory was irrelevant (-0.15)."
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
                "description": (
                    "Reward signal: +1.0 (cited/used), +0.5 (updated), "
                    "+0.6 (created new content after), +0.4 (re-recalled), "
                    "-0.15 (irrelevant/dead end). Range: -1.0 to +1.0."
                ),
            },
        },
        "required": ["memory_id", "signal"],
    },
}


def _handle_reward(args: Dict[str, Any], **kwargs) -> str:
    store = _cognitive_store
    if store is None:
        return json.dumps({"error": "Cognitive memory store not initialized."})

    memory_id = args.get("memory_id", "").strip()
    signal = args.get("signal", 0.0)

    if not memory_id:
        return json.dumps({"error": "memory_id is required."})

    signal = max(-1.0, min(1.0, float(signal)))

    try:
        store.reward_memory(memory_id, signal)
    except Exception as e:
        logger.exception("cognitive_reward error")
        return json.dumps({"error": f"Reward failed: {e}"})

    return json.dumps({
        "rewarded": True,
        "memory_id": memory_id,
        "signal": signal,
    }, ensure_ascii=False)


# ── Registration ──────────────────────────────────────────────────────────────

registry.register(
    name="cognitive_recall",
    toolset="cognitive_memory",
    schema=_RECALL_SCHEMA,
    handler=_handle_recall,
    check_fn=_check_available,
    emoji="🧠",
)

registry.register(
    name="cognitive_store",
    toolset="cognitive_memory",
    schema=_STORE_SCHEMA,
    handler=_handle_store,
    check_fn=_check_available,
    emoji="💾",
)

registry.register(
    name="cognitive_consolidate",
    toolset="cognitive_memory",
    schema=_CONSOLIDATE_SCHEMA,
    handler=_handle_consolidate,
    check_fn=_check_available,
    emoji="🔄",
)

registry.register(
    name="cognitive_explore",
    toolset="cognitive_memory",
    schema=_EXPLORE_SCHEMA,
    handler=_handle_explore,
    check_fn=_check_available,
    emoji="🔍",
)

registry.register(
    name="cognitive_reward",
    toolset="cognitive_memory",
    schema=_REWARD_SCHEMA,
    handler=_handle_reward,
    check_fn=_check_available,
    emoji="⭐",
)
