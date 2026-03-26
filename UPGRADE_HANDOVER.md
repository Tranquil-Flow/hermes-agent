# Hermes-Agent Upgrade Handover: v0.3.0 → v0.4.0

**Date:** 2026-03-26
**For:** Claude Code (host-side agent)
**Author:** Moonsong (container-side agent)

---

## MISSION

Two-phase upgrade of hermes-agent from v0.3.0 to v0.4.0:
- **Phase 1:** Clean upgrade to vanilla v0.4.0 + verify hermes-aegis works
- **Phase 2:** Replay our 45 custom cognitive memory commits on the known-good base

This separation ensures we never debug platform upgrade issues and custom code
conflicts at the same time. Phase 2 should only happen AFTER Phase 1 is fully
verified.

**IMPORTANT:** Hermes is installed on the HOST via `pip install -e .` from
`~/Projects/hermes-agent`. The Docker container has a bind-mount to this directory.
All git operations modify the host filesystem directly. Stop hermes before starting.

---

## CURRENT STATE

### hermes-agent
- **Location:** `~/Projects/hermes-agent`
- **Current version:** 0.3.0 (pyproject.toml)
- **Current HEAD:** `4f632ffe` (docs: handover prompt for host-side benchmark run)
- **Base tag:** `v2026.3.17` = commit `e2c2a187fe8295d84a84401735777888422caff5`
- **Local commits:** 45 ahead of origin/main (17,016 lines added across 101 files)
- **Uncommitted changes:** 3 files (hermes_planner/cli.py, registry.py, session_prompt.md)
- **Remotes:**
  - `origin` → `git@github.com:NousResearch/hermes-agent.git` (SSH)
  - `0xbyt4` → `https://github.com/0xbyt4/hermes-agent.git` (HTTPS, writable)
- **Python:** >=3.11 required, macOS with PEP 668 (use `--break-system-packages`)

### hermes-aegis
- **Location:** `~/Projects/hermes-aegis`
- **Version:** 0.1.6
- **HEAD:** `e53e446` (24 commits ahead of origin, clean working tree)
- **Remote:** `https://github.com/Tranquil-Flow/hermes-aegis.git`
- **Test suite:** 834 passing / 17 skipped

### Target
- **v0.4.0 tag:** `v2026.3.23` = commit `8416bc21` on 0xbyt4/main
- **0xbyt4/main HEAD:** `e4033b2baf` (518 commits ahead of our local HEAD)
- **Note:** 0xbyt4/main has commits BEYOND v0.4.0 (post-release fixes). Use the v0.4.0 tag specifically, not 0xbyt4/main HEAD.

---

## PHASE 1: CLEAN UPGRADE + AEGIS VERIFICATION

Goal: Get a vanilla v0.4.0 running with hermes-aegis confirmed working.
No custom code yet.

### Step 1.0: Pre-flight
```bash
# Stop hermes gateway if running
hermes gateway stop

# Verify we're in the right place
cd ~/Projects/hermes-agent
git status
# Should show 3 modified files in hermes_planner/
```

### Step 1.1: Preserve everything
```bash
# Commit the 3 uncommitted planner files
git add hermes_planner/cli.py hermes_planner/registry.py hermes_planner/session_prompt.md
git commit -m "feat(planner): dashboard UI, task counting, and session prompt updates"

# Create backup branch (our 46 commits preserved here no matter what)
git branch backup/pre-v0.4.0-upgrade

# Export patches as safety net
mkdir -p /tmp/hermes-patches
git format-patch origin/main..HEAD -o /tmp/hermes-patches/
# Should create 46 patch files

# Push to fork for remote backup
git push 0xbyt4 HEAD:refs/heads/backup/pre-v0.4.0-upgrade
```

### Step 1.2: Fetch the v0.4.0 tag
```bash
# Fetch from 0xbyt4 (has the release)
git fetch 0xbyt4 --tags

# If v2026.3.23 tag doesn't appear, create it manually:
git tag v2026.3.23 8416bc21

# Verify
git log --oneline v2026.3.23 -1
# Should show: 8416bc21 chore: release v0.4.0 (v2026.3.23)
```

### Step 1.3: Switch to clean v0.4.0
```bash
# Move main to vanilla v0.4.0 (our commits are safe on backup branch)
git checkout main
git reset --hard v2026.3.23

# Reinstall with new deps
pip3 install -e ".[all]" --break-system-packages
```

### Step 1.4: Verify hermes boots
```bash
hermes --version
# Should show 0.4.0 or similar

hermes  # Start CLI, verify it boots cleanly, then exit
```

### Step 1.5: Verify hermes-aegis compatibility
```bash
cd ~/Projects/hermes-aegis
python -m pytest tests/ -q
# Expect 834 passing / 17 skipped
```

**If aegis tests fail:** Fix them NOW, before Phase 2. Failures here are
likely due to changed imports, API signatures, or new provider endpoints
in v0.4.0. Common areas to check:
- `hook.py` — env var names may have changed
- `proxy/addon.py` — new provider endpoints to intercept
- Docker env forwarding — v0.4.0 may have changed container setup
- OAuth handling — v0.4.0 adds MCP OAuth 2.1

Commit aegis fixes to hermes-aegis repo.

### Step 1.6: Verify gateway (if used)
```bash
cd ~/Projects/hermes-agent
hermes gateway start
# Verify Discord/Telegram connect, send a test message, then stop
hermes gateway stop
```

### Step 1.7: Checkpoint
```bash
# If everything works, tag this known-good state
cd ~/Projects/hermes-agent
git tag local/v0.4.0-aegis-verified
```

**STOP HERE if anything is broken. Fix it before proceeding to Phase 2.**

---

## PHASE 2: REPLAY COGNITIVE MEMORY WORK

Goal: Rebase our 46 custom commits onto the verified v0.4.0 base.
Only start this after Phase 1 is fully green.

### Step 2.1: Create upgrade branch and rebase
```bash
cd ~/Projects/hermes-agent

# Create branch from our backup (has all 46 custom commits)
git checkout -b upgrade/cognitive-memory backup/pre-v0.4.0-upgrade

# Rebase our commits onto the verified v0.4.0 base
# This replays our 46 commits on top of v2026.3.23
git rebase --onto v2026.3.23 v2026.3.17 upgrade/cognitive-memory
```

**EXPECT CONFLICTS.** See the conflict resolution guide below.

### Step 2.2: Verify after rebase
```bash
# Check we're clean
git status
# Should say "nothing to commit"

# Verify our files exist
ls cognitive_memory/store.py
ls benchmarks/runner.py
ls tools/cognitive_memory_tool.py
ls honcho_integration/client.py

# Run cognitive memory tests
cd ~/Projects/hermes-agent
python -m pytest tests/cognitive_memory/ -v
# Should pass all 37 tests

# Run honcho tests
python -m pytest tests/honcho_integration/ -q
# Should pass all 103 tests

# Quick sanity on benchmarks
python -m benchmarks.runner --backend cognitive --suite a --runs 1 --seeds 42
# Suite A should score ~95%
```

### Step 2.3: Re-verify aegis (with our code on top)
```bash
cd ~/Projects/hermes-aegis
python -m pytest tests/ -q
# Should still pass 834 tests — if not, our rebase broke something
```

### Step 2.4: Merge to main
```bash
cd ~/Projects/hermes-agent

# If ALL tests pass:
git checkout main
git reset --hard upgrade/cognitive-memory

# Reinstall (in case we added deps like sentence-transformers)
pip3 install -e ".[all]" --break-system-packages
```

### Step 2.5: Final verification
```bash
hermes  # Boot CLI, verify it starts
hermes gateway start  # If using gateway
```

---

## CONFLICT RESOLUTION GUIDE

### Expected conflict files and how to resolve each:

#### 1. `tools/memory_tool.py` — TAKE UPSTREAM
**Why:** Upstream added file locking for concurrent writes (PR #1726).
Our cognitive memory uses its own SQLite backend via `tools/cognitive_memory_tool.py`
(a separate file). We don't need custom changes in memory_tool.py.
```bash
git checkout --theirs tools/memory_tool.py
git add tools/memory_tool.py
```

#### 2. `honcho_integration/client.py` — TAKE UPSTREAM, VERIFY
**Why:** Upstream (PR #2343) added:
- Instance-local config via $HERMES_HOME/honcho.json
- Per-directory session strategy (was per-session)
- base_url support for self-hosted Honcho
- explicitly_configured flag

Our changes were bug fixes (honcho-ai install, flush race fix).
Check if our fixes are included in upstream. If not, re-apply on top.
```bash
git checkout --theirs honcho_integration/client.py
git add honcho_integration/client.py
# Then verify our fix from commit 285182c1 is present or re-apply
```

#### 3. `honcho_integration/cli.py` — TAKE UPSTREAM
**Why:** Upstream replaced hardcoded paths with resolve_config_path().
Our changes were minor. Take upstream's cleaner version.
```bash
git checkout --theirs honcho_integration/cli.py
git add honcho_integration/cli.py
```

#### 4. `run_agent.py` — MERGE CAREFULLY
**Why:** We modified this to wire cognitive_memory_tool into the agent.
Upstream added background memory review (PR #2235) and other changes.
KEEP BOTH: upstream's background review + our cognitive memory wiring.
- Look for our additions (cognitive_memory imports, tool registration)
- Ensure they coexist with upstream's _spawn_background_review()
- Consider adding cognitive_memory_tool to the background reviewer's toolset

#### 5. `agent/prompt_builder.py` — MERGE CAREFULLY
**Why:** We added cognitive memory context injection. Upstream may have
changed the prompt building pipeline. Keep our injection logic, adapt
to upstream's new structure.

#### 6. `gateway/run.py` — TAKE UPSTREAM + RE-APPLY
**Why:** Upstream added flush memory stale guard (PR #2687) and other fixes.
We added discord platform changes. Take upstream, check if our discord
changes conflict, re-apply if needed.

#### 7. `pyproject.toml` — TAKE UPSTREAM + ADD OUR DEPS
**Why:** Version bump to 0.4.0, new dependencies. Take upstream's version,
then add any dependencies we introduced:
- `sentence-transformers` (for cognitive memory embeddings)
- `honcho-ai>=2.0.1` (if not already in upstream)

#### 8. `.gitignore` — MERGE BOTH
Simple — combine both sides' ignore patterns.

#### 9. `model_tools.py` / `tools/terminal_tool.py` / `tools/environments/docker.py`
We made small changes (proxy URL rewrite, env forwarding). Upstream likely
changed these too. Compare carefully, keep our proxy/docker fixes if
upstream doesn't include them.

### Files that should NOT conflict (new directories):
These are entirely our additions with no upstream equivalent:
- `cognitive_memory/` (entire directory) — NO CONFLICT
- `benchmarks/` (entire directory) — NO CONFLICT
- `tests/cognitive_memory/` — NO CONFLICT
- `tests/benchmarks/` — NO CONFLICT
- `hermes_planner/` — NO CONFLICT (new module)
- `skills/hermes-neurovision-theme-design/` — NO CONFLICT
- `docs/COGNITIVE_MEMORY_*.md` — NO CONFLICT
- `scripts/` (our additions) — NO CONFLICT

---

## POST-UPGRADE INTEGRATION OPPORTUNITY

### Wire cognitive memory into background review agent (PR #2235)

After the upgrade, the background review agent in `run_agent.py` spawns
a daemon thread every ~10 turns to review conversation and save memories.
Currently it only has `memory` and `skill_manage` tools.

**Proposed integration:**
1. Add `cognitive_memory_tool` to `_spawn_background_review()`'s tool list
2. The background reviewer would then automatically:
   - Extract and store new facts in the cognitive memory
   - Consolidate related facts (improving Suite B scores)
   - Update temporal access patterns (improving ACT-R activation)
3. This requires cognitive_memory_tool to be importable in the review context
4. Test with: run a conversation, check cognitive memory DB after ~10 turns

This is a natural extension that aligns our work with upstream architecture.
Implement AFTER the upgrade is stable and tests pass.

---

## OUR 46 COMMITS (for reference)

```
4f632ffe docs: handover prompt for host-side benchmark run with LLM judge
447e6181 docs: session 9 — aegis proxy plumbing verified, LLM contradiction confirmed
1504d0c5 test: verify aegis proxy + LLM contradiction detection working end-to-end
7ea1c353 chore: clean up stale debug/test scripts from previous sessions
04cd61c1 fix: detect aegis proxy port from pid file, prep host agent task
dfc0a78c docs: update TASKS.md and handover with Session 8 results
195460b3 feat: LLM contradiction detection fallback for semantic mismatches
b8b533f3 feat(benchmark): improve Suite B-E scores — access matching, top-k judges, false-positive fix
f47beaba test(docker): add tests for proxy URL rewrite in container env forwarding
c1a18747 fix(docker): rewrite localhost proxy URLs to host.docker.internal in containers
02210a66 fix(terminal): forward docker_forward_env to container_config
9b3ada74 docs: update TASKS.md and COGNITIVE_MEMORY_HANDOVER.md with session 7 results
285182c1 fix(honcho): install honcho-ai and fix race in flush_all test
f1158663 feat(benchmark): LongMemEval Suite B adapter — 500Q external benchmark
1b4e7799 feat(benchmark): sentence-transformers upgrade achieves 95.5% accuracy
e26bf299 docs: update TASKS.md with session 5 progress and current benchmark state
d91f527f feat: pass top-2 results for hard temporal_decay scenarios to improve LLM judge accuracy
2c005f56 feat: improve LLM judge prompt for memory retrieval accuracy + fix multi-fact scenario evaluation
33841296 fix: update LLM judge model name to claude-haiku-4-5 (verified working via aegis proxy)
d5415d7b fix: suite E scale runner equalizes fact age to isolate semantic discrimination
a6f9dc1f fix: remove container-incompatible sys.path hack from cognitive memory init
bfc8ee82 feat: complete benchmark suites E/F + fix Suite C scope boost
8f78aeba feat: benchmark suites B/C/D + suite A baseline results
f4c2309c feat: wire cognitive memory engine into hermes-agent
62dfa49a docs: update handover with session 4 results (sentence-transformers)
aa96c5b2 feat: enable sentence-transformers embeddings with Docker/aegis SSL fix
e103696e docs: cognitive memory handover document
6e0170f5 fix: prevent false supersession of structurally similar facts
ba558add feat: improve heuristic judge for reasoning answers — overall 84% → 93%
89d37265 feat: contradiction detection overhaul — 0% → 85%, overall 76% → 84%
b1dca0a0 data: honest benchmark results with harder contradiction test
721e8530 fix: harder contradiction benchmark — no longer just tests recency
f7ce907a feat: empirically optimized defaults from parameter sweep
0cc82d73 feat: config comparison CLI + ablation study reveals optimal parameters
e9923204 feat: improved heuristic judge + LLM judge infrastructure + seed randomization
283746f4 fix: temporal decay runner properly simulates time gaps between stores
c31e5273 feat: token usage tracking in benchmark framework
57d79840 fix: virtual clock + importance scaling — 71% with sentence-transformers
5b625b24 fix: persist HuggingFace model cache across container restarts
fb1e2eb3 feat: benchmark framework + first results — cognitive vs baseline
20d66c89 fix: production hardening for cognitive memory
77f0233e feat: core cognitive memory system — Phase 2 implementation
ae041806 skill: add slider-architecture + sound-system references to neurovision-theme-design
46a68f70 feat(skills/neurovision): add complexity guide + small-screen principles
e5ace622 feat(skills): add hermes-neurovision-theme-design skill v2
[+1 uncommitted planner commit after Step 1]
```

---

## BENCHMARK BASELINES (pre-upgrade, for regression check)

```
Suite A (200 scenarios): 95.5% overall
  semantic_recall:    1.000
  contradictions:     0.950
  cross_reference:    0.911
  temporal_decay:     0.911
  importance_filter:  varies

Suite B: consolidation 0.85, compression 1.0
Suite C: scopes 1.00
Suite D: adversarial 0.80
Suite E: scale 1.00
Suite F: integration 1.00

Tests: 161 passing (37 cognitive + 103 honcho + 21 longmemeval)
```

---

## HERMES-AEGIS POST-UPGRADE

After hermes-agent upgrade, run aegis tests:
```bash
cd ~/Projects/hermes-aegis
python -m pytest tests/ -q
# Expect 834 passing / 17 skipped
```

Known areas that might break:
- `hook.py` sets proxy env vars — check new v0.4.0 env var names
- `proxy/addon.py` intercepts API calls — check for new provider endpoints
- Docker env forwarding — v0.4.0 may have changed container setup
- OAuth token handling — v0.4.0 adds MCP OAuth 2.1, may interact with aegis

If aegis tests fail, the fixes should be made in hermes-aegis, not hermes-agent.
Commit fixes to the hermes-aegis repo.

---

## ROLLBACK PLAN

If anything goes catastrophically wrong:
```bash
cd ~/Projects/hermes-agent
git checkout main
git reset --hard backup/pre-v0.4.0-upgrade
pip3 install -e ".[all]" --break-system-packages
```

Patches are also saved at `/tmp/hermes-patches/` for manual re-application.

---

## GIT CONFIG

Commits should use:
- **Name:** Tranquil-Flow
- **Email:** tranquil_flow@protonmail.com
