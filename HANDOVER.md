# Handover — Mnemoria Memory Plugin (PR 2)

**Date:** 2026-04-04
**Worktree:** `/Users/evinova/Projects/pr2-unified-memory`
**Branch:** `pr/mnemoria`
**Upstream:** `upstream/main` (NousResearch/hermes-agent)
**Origin:** `origin/pr/mnemoria` (Tranquil-Flow/hermes-agent)

---

## PR 1 — `pr/mnemoria-plugin` — MERGED ✅

PR 1 added the benchmark infrastructure and the `agent/memory_provider.py` ABC,
`agent/memory_manager.py`, and `agent/builtin_memory_provider.py`.

### Key files added by PR 1
| File | Purpose |
|------|---------|
| `agent/memory_provider.py` | ABC with 7 required methods + 5 optional hooks |
| `agent/memory_manager.py` | Orchestrator — one builtin + one external provider |
| `agent/builtin_memory_provider.py` | MEMORY.md/USER.md wrapped as a MemoryProvider |
| `hermes_cli/memory_setup.py` | CLI for interactive memory provider setup |
| `benchmarks/` | Suites A–L, hotpotqa, locomo, longmemeval — 785-line metrics engine |
| `benchmarks/backends/` | 7 plugin adapters (mem0, honcho, holographic, byterover, hindsight, openviking, retaindb) |
| `tests/agent/test_memory_provider.py` | 549 lines — ABC tests, MemoryManager tests, provider lifecycle tests |
| `tools/structured_memory/` | Full structured memory implementation (fact store, gauge, scopes, optimize) |

PR 1 is **fully merged** into `upstream/main`.

---

## PR 2 — `pr/mnemoria` — READY FOR REVIEW (1 bug fixed today) ✅

PR 2 delivers the **MnemoriaMemoryProvider** — the production plugin backed by
UnifiedMemoryStore (97.2% benchmark score vs 87.5% flat baseline).

### Today's changes (commit `87b076f5`)

```
plugins/memory/mnemoria/__init__.py         — package + docstring
plugins/memory/mnemoria/provider.py         — MnemoriaMemoryProvider (577 lines)
plugins/memory/mnemoria/plugin.yaml         — plugin metadata
plugins/memory/mnemoria/SKILL.md            — skill reference
plugins/memory/mnemoria/benchmark_adapter.py — moved from unified_memory/
plugins/__init__.py                         — restored from upstream/main
plugins/memory/__init__.py                  — restored from upstream/main
plugins/memory/holographic/*                — restored from upstream/main (5 files)
unified_memory/store.py                     — dedup threshold fix
```

### Bug fixed today

**`unified_memory/store.py` — near-duplicate dedup threshold**

- **Problem:** `find_near_duplicates()` used `threshold=0.90`. Embedding similarity
  between "Python version is 3.10" and "Python version is 3.12" is **0.9129**
  (MiniLM-L6-v2 encodes these nearly identically). Both facts were collapsed
  into one, causing `test_recall_ranking` to see only 1 result.
- **Fix:** `threshold=0.90` → `threshold=0.95`. True paraphrases like
  "JWT tokens" ≈ "JWT authentication" score ≥ 0.97, so this threshold correctly
  separates them. The word-overlap secondary check (≥ 0.4) remains unchanged.
- **Test:** `test_recall_ranking` now passes.

### Current test status

```
tests/unified_memory/      — 17 test files
tests/cognitive_memory/    — 5 test files
tests/agent/test_memory_provider.py — 549 lines
tests/honcho_integration/  — 4 test files
tests/structured_memory/    — 8 test files

Total: 231 passed, 0 failed
```

### How UnifiedMemoryStore works in production today

The agent currently uses **three parallel memory systems** (no MemoryManager yet):

| Layer | File | Used via |
|-------|------|----------|
| Built-in | `tools/memory_tool.py` | `memory_tool` (MEMORY.md/USER.md) |
| Cognitive | `cognitive_memory/store.py` | `tools/cognitive_memory_tool.py` |
| Unified | `unified_memory/store.py` | `tools/unified_memory_tool.py` (8 MCP tools) |

The unified store is:
- Initialized per-session in `run_agent.py` lines 939–954
- Ticked each agent loop in `run_agent.py` lines 5714–5715
- Hot-fact injection in system prompt via `get_unified_memory_injection()` (line 2386)
- MCP tools: `mcp_umemory_write`, `mcp_umemory_recall`, `mcp_umemory_search`,
  `mcp_umemory_reflect`, `mcp_umemory_reward`, `mcp_umemory_explore`,
  `mcp_umemory_stats`, `mcp_umemory_consolidate`

---

## REMAINING — Phase 3: Agent Loop Integration ⚠️

**This is the critical remaining work.** The `MnemoriaMemoryProvider` plugin exists
and 231 tests pass, but the `MemoryManager` is **not wired into `run_agent.py`** —
it only exists as a test artifact.

### What needs to happen

**File: `run_agent.py`**

The `AIAgent.__init__` needs to:

1. Import the new infrastructure:
   ```python
   from agent.memory_manager import MemoryManager
   from agent.builtin_memory_provider import BuiltinMemoryProvider
   ```

2. Construct and populate a `MemoryManager`:
   ```python
   self._memory_manager = MemoryManager()
   self._memory_manager.add_provider(BuiltinMemoryProvider(
       memory_store=self._memory_store,
       memory_enabled=self._memory_enabled,
       user_profile_enabled=self._user_profile_enabled,
   ))
   # Then conditionally add the external provider (honcho OR mnemoria OR mem0, etc.)
   # based on config — see MemoryManager.add_provider() which enforces single external
   ```

3. Call lifecycle methods in the agent loop:
   - `self._memory_manager.prefetch_all(query, session_id=...)` — before each turn
   - `self._memory_manager.sync_all(user_msg, assistant_response)` — after each turn
   - `self._memory_manager.build_system_prompt()` — concatenated into system prompt

4. **Or**, alternatively, wire `MnemoriaMemoryProvider` via the existing config-driven
   `honcho` / `cognitive_memory` initialization pattern (lines 934–1001 of run_agent.py)
   rather than the MemoryManager path — both approaches are valid but must be consistent.

### Important constraints

- `MemoryManager` allows **one external provider at a time** — builtin is always first
- `BuiltinMemoryProvider.get_tool_schemas()` returns `[]` (memory tool is already wired
  at the agent level; this prevents double-exposure)
- The existing `cognitive_memory` and `unified_memory_tool` direct initialization
  (lines 934–954 and 2376–2387) is **parallel to** the MemoryManager, not through it.
  One of these two patterns must be chosen for mnemoria:
  - Pattern A: Wire through MemoryManager (architecturally clean)
  - Pattern B: Keep direct tool initialization + add mnemoria alongside (as cognitive
    memory currently works — initialized directly, not via MemoryManager)

### Test gap

`MnemoriaMemoryProvider` is not yet tested in `tests/agent/test_memory_provider.py`.
Add a `TestMnemoriaMemoryProvider` class following the `FakeMemoryProvider` pattern
from that file. Minimum coverage:
- `name == "mnemoria"`
- `is_available()` — True when `HERMES_MEMORY_MNEMORIA_ENABLED=true`, False otherwise
- `initialize()` — creates store, initializes schema
- `get_tool_schemas()` — returns 8 tool schemas
- `handle_tool_call("mcp_umemory_write", ...)` — stores and returns fact ID
- `handle_tool_call("mcp_umemory_recall", ...)` — recalls facts
- `shutdown()` — closes store

### Optional: `consolidate` not in ABC

`MnemoriaMemoryProvider._handle_consolidate()` exists but `consolidate` is not
required by the `MemoryProvider` ABC. The tool is still exposed to the model.
No action needed — just be aware it can't be lifecycle-managed by the ABC.

---

## Config / Env Vars

```
HERMES_MEMORY_MNEMORIA_ENABLED=false    # Set true to activate mnemoria
HERMES_MEMORY_MNEMORIA_MODE=balanced     # balanced | recall | write | disabled
HERMES_UNIFIED_MEMORY_DB=~/.hermes/unified_memory.db
```

---

## Key People / Context

- **Owner:** Tranquil-Flow (tranquil_flow@protonmail.com)
- **Upstream:** NousResearch/hermes-agent (upstream/main)
- **Discord:** tranquil_flow / 385694377655271424
- **This worktree:** `/Users/evinova/Projects/pr2-unified-memory`
- **PR 2 pushed to:** `origin/pr/mnemoria`

---

*Generated 2026-04-04 by Moon (Moonsong) — good luck with Phase 3* 🌙
