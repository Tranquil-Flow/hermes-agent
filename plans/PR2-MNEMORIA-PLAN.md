# PR 2 - Mnemoria Memory Plugin

## Overview

PR 2 (worktree: /Users/evinova/Projects/pr2-unified-memory, branch: pr/mnemoria) delivers the Mnemoria memory plugin backed by the Unified Memory System (97.2% benchmark score).

**Current state:** 92 commits ahead of main. Core unified_memory/ module is complete with 17 test files. However, this PR deleted the memory plugin infrastructure (agent/memory_provider.py, agent/memory_manager.py, all plugins/memory/* plugins) without replacing them.

**Goal:** Package unified_memory/ as plugins/memory/mnemoria/ plugin, restore the memory plugin infrastructure, and integrate into the agent loop.

**HEAD commit:** eea6ec20 (fix(unified): add missing unified_memory/export.py module)

---

## What Exists vs What's Missing

### Already in PR 2 (do NOT recreate)
  unified_memory/ - 14 files: store, retrieval, links, bandit, ingestion, etc.
  cognitive_memory/ - ACT-R dependency: embeddings, encoding, qvalue_store, etc.
  tools/unified_memory_tool.py - MCP tool wrapper (8 tools)
  tests/unified_memory/ - 17 test files
  benchmarks/ - Suite A-H fixtures + runners

### Deleted and NOT Restored (Phase 1 - CRITICAL)
  agent/memory_provider.py - MemoryProvider ABC (REQUIRED)
  agent/memory_manager.py - MemoryManager orchestrator (REQUIRED)
  agent/builtin_memory_provider.py - BuiltinMemoryProvider (REQUIRED)
  hermes_cli/memory_setup.py - hermes memory setup CLI (REQUIRED)
  plugins/memory/__init__.py - Plugin package init
  plugins/memory/honcho/ etc. - All 7 plugin stubs (deleted)
  tests/agent/test_memory_provider.py - (deleted)
  tests/agent/test_memory_plugin_e2e.py - (deleted)

### PR 1 (separate) - Benchmark Backend Adapters
  benchmarks/backends/ (mem0_adapter, honcho_adapter, etc.) lives in PR 1 (pr/benchmark-suite). Do NOT recreate in PR 2.

---

## Phase 1 - Restore Memory Plugin Infrastructure

These files are prerequisites for ANY memory plugin. Restore verbatim from main.

Step 1.1: Restore agent/memory_provider.py
  git show main:agent/memory_provider.py > agent/memory_provider.py

Step 1.2: Restore agent/memory_manager.py
  git show main:agent/memory_manager.py > agent/memory_manager.py

Step 1.3: Restore agent/builtin_memory_provider.py
  git show main:agent/builtin_memory_provider.py > agent/builtin_memory_provider.py

Step 1.4: Restore hermes_cli/memory_setup.py
  git show main:hermes_cli/memory_setup.py > hermes_cli/memory_setup.py

Step 1.5: Create plugins/memory/__init__.py
  echo '# Memory plugins package' > plugins/memory/__init__.py

Step 1.6: Restore test files
  git show main:tests/agent/test_memory_provider.py > tests/agent/test_memory_provider.py
  git show main:tests/agent/test_memory_plugin_e2e.py > tests/agent/test_memory_plugin_e2e.py

Step 1.7: Create plugin stubs
  For each: honcho mem0 byterover hindsight holographic openviking retaindb
    mkdir -p plugins/memory/<name>
    echo '# Stub - actual impl in benchmarks/backends/ (PR 1)' > plugins/memory/<name>/__init__.py

Verify: python -c 'from agent.memory_provider import MemoryProvider; from agent.memory_manager import MemoryManager; print(Phase 1 OK)'

---

## Phase 2 - Create plugins/memory/mnemoria/ Plugin

Step 2.1: mkdir -p plugins/memory/mnemoria

Step 2.2: Create plugins/memory/mnemoria/__init__.py
  Import MnemoriaMemoryProvider from .provider
  Set __all__ = ['MnemoriaMemoryProvider']
  Add docstring (97.2% benchmark, config env vars: HERMES_MEMORY_MNEMORIA_ENABLED, HERMES_MEMORY_MNEMORIA_MODE, HERMES_UNIFIED_MEMORY_DB)

Step 2.3: Create plugins/memory/mnemoria/provider.py - MOST CRITICAL FILE
  Implement MnemoriaMemoryProvider(MemoryProvider):
  Required methods:
    name = 'mnemoria'
    is_available() - check unified_memory imports work
    initialize(session_id, **kwargs) - create UnifiedMemoryStore, init DB
    system_prompt_block() - return static intro text
    prefetch(query, session_id='') - call store.recall(), format as injection
    sync_turn(user_content, assistant_content, session_id='') - call ingest_turn()
    get_tool_schemas() - return 9 MCP tool schemas (mirror from tools/unified_memory_tool.py):
      mcp_umemory_write, mcp_umemory_recall, mcp_umemory_search, mcp_umemory_reflect,
      mcp_umemory_reward, mcp_umemory_explore, mcp_umemory_stats, mcp_umemory_consolidate
    handle_tool_call(tool_name, args, **kwargs) - dispatch to store methods, return JSON string
    shutdown() - close store connections
  Optional hooks: on_turn_start(), on_session_end(), on_pre_compress(), get_config_schema(), save_config()
  Key patterns (from tools/unified_memory_tool.py):
    Use threading.local() for per-thread store instances
    _DB_PATH = str(Path.home() / '.hermes' / 'unified_memory.db')
    _resolve_session() pattern for session_id resolution

Step 2.4: Create plugins/memory/mnemoria/plugin.yaml
  name: mnemoria, version: 0.1.0, provider: mnemoria, memory_provider: true
  config fields: HERMES_MEMORY_MNEMORIA_ENABLED (bool), HERMES_MEMORY_MNEMORIA_MODE (balanced/recall/write/disabled), HERMES_UNIFIED_MEMORY_DB (path)

Step 2.5: Create plugins/memory/mnemoria/SKILL.md
  Document MEMORY_SPEC notation: C[t]: Constraint, D[t]: Decision, V[t]: Value, ?[t]: Unknown, check[t]: Done, tilde[t]: Obsolete
  Document 8 MCP tools with usage examples and output formats
  Document config options, architecture, benchmark results (97.2%), comparison to other providers

Step 2.6: mv unified_memory/benchmark_adapter.py plugins/memory/mnemoria/benchmark_adapter.py

Verify: python -c 'from plugins.memory.mnemoria import MnemoriaMemoryProvider; p = MnemoriaMemoryProvider(); print(p.name, p.is_available())'

---

## Phase 3 - Agent Loop Integration

Step 3.1: Check agent/__init__.py - add MemoryProvider and MemoryManager imports

Step 3.2: Wire run_agent.py
  IMPORTANT: FIRST cat run_agent.py to see its current state (may have been modified in this PR)
  Add carefully without breaking existing changes:
    from agent.memory_manager import MemoryManager
    from agent.builtin_memory_provider import BuiltinMemoryProvider
    from plugins.memory.mnemoria import MnemoriaMemoryProvider
    self._memory_manager = MemoryManager()
    self._memory_manager.add_provider(BuiltinMemoryProvider(...))
    if MnemoriaMemoryProvider().is_available():
      self._memory_manager.add_provider(MnemoriaMemoryProvider())

Step 3.3: Wire memory context injection
  Check agent/anthropic_adapter.py and prompt_builder.py for system prompt assembly
  Ensure MemoryManager.build_system_prompt() result is included

---

## Phase 4 - Fix tools/memory_tool.py

Check tools/memory_tool.py was modified in this PR. Verify:
  1. Imports from valid locations
  2. BuiltinMemoryProvider import works (restored in Phase 1)
  3. Fix any broken imports

---

## Phase 5 - Handle cognitive_memory/

Keep cognitive_memory/ as TOP-LEVEL package (dependency of unified_memory). Do NOT move it.

Verify imports:
  python -c 'from cognitive_memory.qvalue_store import QValueStore; print(OK)'
  python -c 'from cognitive_memory.embeddings import EmbeddingProvider; print(OK)'
  python -c 'from cognitive_memory.encoding import encode; print(OK)'

---

## Phase 6 - Tests

cd /Users/evinova/Projects/pr2-unified-memory
  python -m pytest tests/unified_memory/ -v --tb=short -q
  python -m pytest tests/cognitive_memory/ -v --tb=short -q
  python -m pytest tests/agent/test_memory_provider.py -v --tb=short -q
  python -m pytest tests/ -k memory -v --tb=short -q

---

## Phase 7 - Benchmark Validation

python -m benchmarks.runner --backend mnemoria --suite all --runs 1
Or smoke test: python -c 'from unified_memory.store import UnifiedMemoryStore; from unified_memory.config import UnifiedMemoryConfig; store = UnifiedMemoryStore(UnifiedMemoryConfig.balanced()); store.store("V[test]: hello"); results = store.recall("test"); print(len(results))'

---

## Phase 8 - Cleanup

1. honcho_integration/ - DELETE it (out of scope, benchmark adapters are in PR 1)
2. Verify no references to plugins/memory/honcho/ remain
3. Check website/docs/user-guide/features/memory.md was updated
4. Check AGENTS.md for stale memory provider references

---

## Phase 9 - PR Readiness Verification

cd /Users/evinova/Projects/pr2-unified-memory

1. No gateway fix commits: git log --oneline main..HEAD | grep -iE 'gateway.*fix' | head -5
   Expected: empty

2. Imports: python -c 'from agent.memory_provider import MemoryProvider; from agent.memory_manager import MemoryManager; from agent.builtin_memory_provider import BuiltinMemoryProvider; from plugins.memory.mnemoria import MnemoriaMemoryProvider; print(All imports OK)'

3. Tests: python -m pytest tests/unified_memory/ tests/cognitive_memory/ -q --tb=short

4. Provider: python -c 'from plugins.memory.mnemoria import MnemoriaMemoryProvider; p = MnemoriaMemoryProvider(); print(p.name, p.is_available())'

5. Benchmark smoke: python -c 'from unified_memory.store import UnifiedMemoryStore; from unified_memory.config import UnifiedMemoryConfig; store = UnifiedMemoryStore(UnifiedMemoryConfig.balanced()); store.store("V[test]: hello"); results = store.recall("test"); print(len(results))'

All checks must pass before pushing.

---

## Key Files Summary

Restore from main (Phase 1):
  agent/memory_provider.py, agent/memory_manager.py, agent/builtin_memory_provider.py
  hermes_cli/memory_setup.py
  tests/agent/test_memory_provider.py, tests/agent/test_memory_plugin_e2e.py

Create new (Phase 2):
  plugins/memory/__init__.py
  plugins/memory/mnemoria/__init__.py
  plugins/memory/mnemoria/provider.py (MOST CRITICAL)
  plugins/memory/mnemoria/plugin.yaml
  plugins/memory/mnemoria/SKILL.md
  plugins/memory/mnemoria/benchmark_adapter.py

Move (Phase 2):
  unified_memory/benchmark_adapter.py -> plugins/memory/mnemoria/benchmark_adapter.py

Modify (Phase 3-4):
  run_agent.py (wire MemoryManager), agent/__init__.py, agent/anthropic_adapter.py, tools/memory_tool.py

Delete (Phase 8):
  honcho_integration/ (out of scope for PR 2)

---

## Key Risks

1. agent/memory_provider.py is pure ABC with no external deps. Restores cleanly.
2. Keep cognitive_memory/ at repo root. If moved, unified_memory/store.py imports break.
3. plugins/memory/honcho/ -> honcho_integration/. Check old references in run_agent.py and memory_setup.py.
4. If run_agent.py was modified to remove MemoryManager, re-add carefully without breaking other changes.
5. Must restore test files for full test suite.

---

## Verification Checklist

- [ ] agent/memory_provider.py restored
- [ ] agent/memory_manager.py restored
- [ ] agent/builtin_memory_provider.py restored
- [ ] hermes_cli/memory_setup.py restored
- [ ] plugins/memory/__init__.py exists
- [ ] plugins/memory/mnemoria/__init__.py exists
- [ ] plugins/memory/mnemoria/provider.py exists with full MnemoriaMemoryProvider
- [ ] plugins/memory/mnemoria/plugin.yaml exists
- [ ] plugins/memory/mnemoria/SKILL.md exists
- [ ] unified_memory/benchmark_adapter.py moved
- [ ] MnemoriaMemoryProvider.is_available() returns correct bool
- [ ] All 9 MCP tool schemas returned by get_tool_schemas()
- [ ] run_agent.py wires MemoryManager with MnemoriaMemoryProvider
- [ ] pytest tests/unified_memory/ -q passes
- [ ] pytest tests/cognitive_memory/ -q passes
- [ ] MnemoriaMemoryProvider imports successfully
- [ ] No gateway security fix commits in main..HEAD
- [ ] honcho_integration/ deleted

---

## Starting Point

Current HEAD: eea6ec20 (fix(unified): add missing unified_memory/export.py module)
Worktree: /Users/evinova/Projects/pr2-unified-memory
Branch: pr/mnemoria

First command: cd /Users/evinova/Projects/pr2-unified-memory && git status