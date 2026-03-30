# Unified Memory System Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Merge cognitive memory (ACT-R activation, Hebbian links, Q-value RL, embeddings) with structured memory (typed facts, FTS5, scopes, pressure management) into a single unified system that exceeds both.

**Architecture:** Single SQLite store with unified schema. 4-signal retrieval fusion (embedding + FTS5/BM25 + ACT-R activation + warmth/PPR). Type-aware metabolic decay. Self-optimizing pipeline via LinUCB contextual bandits.

**Reference Benchmarks:**
- Cognitive: 96.8%
- Structured (FTS5): 73.6%
- Baseline-flat: 83.1%

**Benchmark command:** `python -m benchmarks.runner --backend unified --suite a,b,c,d,e,f --runs 3`

---

## Phase 1: Unified Storage Layer (Target: ≥96.8%)

### Task 1.1: Schema & Types Module
**Objective:** Define the data model — fact types, metabolic rates, config dataclass.

**Files:**
- Create: `unified_memory/__init__.py`
- Create: `unified_memory/types.py`
- Create: `unified_memory/config.py`

**Details:**
- FactType enum: CONSTRAINT='C', DECISION='D', VALUE='V', UNKNOWN='?', DONE='done', OBSOLETE='obs'
- METABOLIC_RATES dict: C=0.3, D=0.7, V=1.0, ?=2.0, done=2.5, obs=5.0
- UnifiedMemoryConfig dataclass with all cognitive config fields + new fields (enable_typed_decay, enable_supersession, enable_pressure, gauge_max_chars=10000)
- MemoryFact dataclass: id, content, embedding, fact_type, target, scope_id, status, activation, q_value, access_count, metabolic_rate, created_at, updated_at, last_accessed, source_hash, superseded_by
- MemoryLink dataclass: source_id, target_id, strength, npmi, co_occurrence_count, link_type, last_updated

### Task 1.2: Schema SQL Module
**Objective:** SQLite schema with unified tables, FTS5, indexes, views.

**Files:**
- Create: `unified_memory/schema.py`
- Test: `tests/unified_memory/test_schema.py`

**Tables:**
- `um_facts`: id TEXT PK, content TEXT, embedding BLOB, type TEXT, target TEXT, scope_id TEXT, status TEXT DEFAULT 'active', activation REAL DEFAULT 0.0, q_value REAL DEFAULT 0.5, access_count INT DEFAULT 0, metabolic_rate REAL DEFAULT 1.0, created_at INT, updated_at INT, last_accessed INT, source_hash TEXT, superseded_by TEXT
- `um_links`: source_id TEXT, target_id TEXT, strength REAL, npmi REAL DEFAULT 0.0, co_occurrence_count INT DEFAULT 0, link_type TEXT DEFAULT 'hebbian', last_updated INT, PRIMARY KEY (source_id, target_id)
- `um_scopes`: id TEXT PK, label TEXT, status TEXT DEFAULT 'active', last_referenced INT, current_turn INT, created_at INT, closed_at INT
- `um_qvalues`: memory_id TEXT PK, q_value REAL, update_count INT, total_retrievals INT, last_updated INT, last_retrieved INT, reward_variance REAL DEFAULT 0.0
- `um_facts_fts`: FTS5 virtual table on content, target (with triggers)
- `um_gauge` view: SELECT sum(length(type)+1+length(target)+3+length(content)) as used_chars, 10000 as max_chars FROM um_facts WHERE status IN ('active','cold')
- Indexes on type, target, scope_id, status, updated_at, source_hash

### Task 1.3: Core Store — Write Path
**Objective:** UnifiedMemoryStore.store() that handles MEMORY_SPEC parsing, embedding, supersession, Hebbian link seeding.

**Files:**
- Create: `unified_memory/store.py`
- Test: `tests/unified_memory/test_store_write.py`

**Write path:**
1. Parse MEMORY_SPEC notation (regex: `^(C|D|V|\?|✓|~)\[([^\]]+)\]:\s*(.+)$`) OR accept plain text (default to V[general])
2. Generate embedding via sentence-transformers (same model as cognitive)
3. Compute source_hash = sha256(content)[:16]
4. Dedup check (same source_hash + active)
5. Supersession check (same type + target + scope → mark old as superseded, transfer activation)
6. Set metabolic_rate from METABOLIC_RATES[type]
7. INSERT fact
8. Seed Hebbian links (keyword overlap with recent facts, threshold=0.15)
9. Check gauge pressure (same cascade as structured memory)
10. Commit and return fact_id

### Task 1.4: Core Store — Read Path (4-Signal Retrieval)
**Objective:** UnifiedMemoryStore.recall() with 4-signal RRF fusion.

**Files:**
- Modify: `unified_memory/store.py`
- Create: `unified_memory/retrieval.py`
- Test: `tests/unified_memory/test_store_recall.py`

**Retrieval pipeline:**
1. Generate query embedding
2. Signal 1: Embedding cosine similarity (top 50 candidates)
3. Signal 2: FTS5/BM25 keyword match (OR-based with stop-word filtering)
4. Signal 3: ACT-R activation = ln(sum(t_j^-d)) where d = base_decay * metabolic_rate
   - Plus spreading activation via Hebbian links
5. Signal 4: Warmth — PPR seeded from top semantic matches
6. Score-weighted RRF fusion: score = sum(weight_i * 1/(k + rank_i))
   - Default weights: embedding=0.40, bm25=0.15, activation=0.30, warmth=0.15
7. Dampening pipeline: gravity (cosine ghosts), hub (P90 degree), resolution boost
8. Q-value reranking (lambda blend, cold-start protection)
9. Scope filtering (if scope specified)
10. Return top_k results

### Task 1.5: Lifecycle Methods
**Objective:** simulate_time, simulate_access, consolidate, get_stats, reset.

**Files:**
- Modify: `unified_memory/store.py`
- Create: `unified_memory/lifecycle.py`
- Test: `tests/unified_memory/test_lifecycle.py`

**Details:**
- simulate_time(days): Decay all activations using ACT-R formula with metabolic rates. C-type facts barely decay, ?-type facts decay fast.
- simulate_access(content): Search + increment access_count + boost activation
- consolidate(): Run gauge pressure cascade + merge duplicate links + prune dead links
- Scope management: get_or_create_scope, close_scope, auto-cool scopes
- Gauge: read_gauge(), check_and_act() with 5-tier cascade

### Task 1.6: Benchmark Adapter & Registration
**Objective:** Wire unified system into benchmark runner.

**Files:**
- Create: `unified_memory/benchmark_adapter.py`
- Modify: `benchmarks/runner.py` (add registration)
- Test: `tests/unified_memory/test_benchmark_adapter.py`

**BENCHMARK CHECKPOINT:** Run `python -m benchmarks.runner --backend unified --suite a,b,c,d,e,f --runs 3`
Target: ≥96.8% (match cognitive)

---

## Phase 2: Type-Aware Intelligence (Target: ≥97.5%)

### Task 2.1: Metabolic Decay Integration
**Objective:** ACT-R decay rates modulated by fact type.

**Files:**
- Modify: `unified_memory/store.py` (activation calculation)
- Test: `tests/unified_memory/test_metabolic_decay.py`

**Details:**
- effective_decay = base_decay * metabolic_rate
- C-facts (0.3x) persist ~3x longer than V-facts
- ?-facts (2.0x) decay ~2x faster — urgency drives resolution
- obs-facts (5.0x) rapidly clear themselves

### Task 2.2: Supersession with Activation Transfer
**Objective:** When fact B supersedes fact A, transfer A's activation energy.

**Files:**
- Modify: `unified_memory/store.py`
- Test: `tests/unified_memory/test_supersession.py`

**Details:**
- transfer_ratio = 0.7 (configurable)
- new_fact.activation += old_fact.activation * transfer_ratio
- Also transfer Hebbian links (repoint to new fact)
- Also transfer Q-value (weighted average)

### Task 2.3: Type-Aware Retrieval Boosting
**Objective:** Boost fact types that match query intent.

**Files:**
- Create: `unified_memory/intent.py`
- Modify: `unified_memory/retrieval.py`
- Test: `tests/unified_memory/test_intent.py`

**Intent classification (regex patterns):**
- "what value/url/port/address" → boost V-type
- "what rule/constraint/requirement" → boost C-type
- "what did we decide/choose" → boost D-type
- "what's unknown/unresolved" → boost ?-type
- Default: no boost

### Task 2.4: Hot-Facts Injection
**Objective:** Generate system prompt block from active hot facts.

**Files:**
- Modify: `unified_memory/store.py`
- Test: `tests/unified_memory/test_hot_facts.py`

**Details:**
- get_hot_facts_injection(): all active facts in active scopes
- Format as MEMORY_SPEC notation grouped by type
- Include gauge percentage

**BENCHMARK CHECKPOINT:** Target: ≥97.5%

---

## Phase 3: Advanced Features (Target: ≥98%)

### Task 3.1: NPMI-Normalized Co-occurrence Edges
**Objective:** Normalize Hebbian link weights against base retrieval rates.

**Files:**
- Create: `unified_memory/links.py`
- Modify: `unified_memory/store.py`
- Test: `tests/unified_memory/test_npmi.py`

**Formula:** NPMI = PMI(A,B) / -log(P(A,B))
- P(A) = retrievals_of_A / total_retrievals
- P(B) = retrievals_of_B / total_retrievals  
- P(A,B) = co_retrievals_AB / total_retrievals
- PMI = log(P(A,B) / (P(A) * P(B)))
- Bounded [-1, 1]

### Task 3.2: UCB-Tuned Variance-Aware Exploration
**Objective:** Replace basic UCB with variance-aware version.

**Files:**
- Modify: `unified_memory/store.py` (Q-value reranking section)
- Test: `tests/unified_memory/test_ucb_tuned.py`

**Formula:** V = variance + sqrt(2*ln(t)/n), bonus = c * sqrt(ln(t)/n * min(0.25, V))

### Task 3.3: Bibliographic Coupling Bootstrap
**Objective:** Pre-seed Hebbian links for new facts based on shared targets.

**Files:**
- Modify: `unified_memory/links.py`
- Test: `tests/unified_memory/test_bootstrap.py`

**Formula:** coupling(A,B) = |shared_targets| / sqrt(|targets_A| * |targets_B|)
If coupling > 0.3, create link with initial strength = coupling * 0.5

### Task 3.4: Tarjan Articulation Point Protection
**Objective:** Protect bridge nodes from pruning during pressure management.

**Files:**
- Modify: `unified_memory/lifecycle.py`
- Test: `tests/unified_memory/test_tarjan.py`

**Details:**
- Build adjacency graph from um_links
- Find articulation points (Tarjan's algorithm)
- Set vitality floor = 0.5 for bridge nodes during gauge cascade
- Skip these nodes in archive/cold-push actions

### Task 3.5: Revival Spike & Access Saturation
**Objective:** Boost dormant facts on new links; diminishing returns on access.

**Files:**
- Modify: `unified_memory/store.py`
- Test: `tests/unified_memory/test_revival.py`

**Revival:** 0.2 * exp(-0.2 * days_since_new_link)
**Saturation:** effective_access = 1 - exp(-access_count / 10)

**BENCHMARK CHECKPOINT:** Target: ≥98%

---

## Phase 4: Self-Optimizing Pipeline (Target: ≥98.5%+)

### Task 4.1: LinUCB Contextual Bandit
**Objective:** Per-stage bandit that learns run/skip/abstain decisions.

**Files:**
- Create: `unified_memory/bandit.py`
- Test: `tests/unified_memory/test_bandit.py`

**8-dim feature vector:**
1. query_length (normalized)
2. unique_terms (normalized)
3. has_question_mark (0/1)
4. temporal_markers (0/1) — "when", "yesterday", "last week"
5. named_entities (count, normalized)
6. embedding_entropy (Shannon entropy of embedding)
7. store_size (log-normalized fact count)
8. query_depth (0=surface, 1=deep — from intent classifier)

**Per-stage decisions:** run (1.0), skip (0.0), abstain (-1.0 = stop pipeline)
**Essential stages (never skipped):** embedding similarity, RRF fusion

### Task 4.2: ACQO Two-Phase Curriculum
**Objective:** Run all stages for first 50 queries, then optimize.

**Files:**
- Modify: `unified_memory/bandit.py`
- Modify: `unified_memory/retrieval.py`
- Test: `tests/unified_memory/test_acqo.py`

### Task 4.3: Heuristic Session Reward Tracking
**Objective:** Auto-infer rewards from store-after-recall patterns.

**Files:**
- Modify: `unified_memory/store.py`
- Test: `tests/unified_memory/test_session_rewards.py`

**Details:**
- Track last recall() results with timestamps
- If store() happens within 5 minutes of recall(), credit recalled memories +0.5
- If same content is recalled again, credit +0.4 (re-recall signal)

### Task 4.4: IPS Debiasing
**Objective:** Correct popularity bias in retrieval.

**Files:**
- Modify: `unified_memory/retrieval.py`
- Test: `tests/unified_memory/test_ips.py`

**Details:**
- Track propensity score per fact (frequency in top-k results)
- Weight = 1 / propensity_score (IPS correction)
- Inject under-retrieved facts as exploration candidates

**FINAL BENCHMARK:** Target: ≥98.5%

---

## File Summary

```
unified_memory/
├── __init__.py
├── types.py          # FactType enum, metabolic rates, dataclasses
├── config.py         # UnifiedMemoryConfig
├── schema.py         # SQL DDL, init_db(), migrate()
├── store.py          # UnifiedMemoryStore (main class)
├── retrieval.py      # 4-signal fusion, RRF, dampening
├── links.py          # Hebbian links, NPMI, bibliographic coupling
├── lifecycle.py      # Scopes, gauge, pressure, Tarjan
├── intent.py         # Query intent classification
├── bandit.py         # LinUCB contextual bandit
├── benchmark_adapter.py
tests/unified_memory/
├── __init__.py
├── conftest.py
├── test_schema.py
├── test_store_write.py
├── test_store_recall.py
├── test_lifecycle.py
├── test_benchmark_adapter.py
├── test_metabolic_decay.py
├── test_supersession.py
├── test_intent.py
├── test_hot_facts.py
├── test_npmi.py
├── test_ucb_tuned.py
├── test_bootstrap.py
├── test_tarjan.py
├── test_revival.py
├── test_bandit.py
├── test_acqo.py
├── test_session_rewards.py
├── test_ips.py
```
