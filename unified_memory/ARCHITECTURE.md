# Unified Memory System — Architecture

## Overview

The Unified Memory System merges three memory architectures into a single
engine that combines the best of cognitive science, structured knowledge
management, and self-optimizing retrieval.

**Sources:**
- **Cognitive Memory** — ACT-R activation, Hebbian links, Q-value RL
- **Structured Memory** (PR #3093) — typed facts, FTS5, scopes, gauge
- **Ori-Mnemos** — NPMI, UCB-Tuned, Tarjan, LinUCB bandits

**Benchmark:** 95.9% overall (284 scenarios, 11 categories)
- Exceeds cognitive on contradictions (95% vs 90%)
- Matches cognitive on 8 of 11 categories

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│              UNIFIED MEMORY SYSTEM              │
│                                                 │
│  WRITE PATH                                     │
│  ┌──────────────────────────────────────┐       │
│  │ 1. Parse MEMORY_SPEC notation         │       │
│  │ 2. Auto-classify (category+importance)│       │
│  │ 3. Generate embedding                 │       │
│  │ 4. Dedup check (source_hash)          │       │
│  │ 5. Supersession check (type+target)   │       │
│  │ 6. Contradiction detection            │       │
│  │ 7. Set metabolic decay rate           │       │
│  │ 8. Store in SQLite                    │       │
│  │ 9. Seed Hebbian links                 │       │
│  │ 10. Gauge pressure check              │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  RETRIEVAL PATH (4-signal fusion)               │
│  ┌──────────────────────────────────────┐       │
│  │ Signal 1: Embedding similarity        │       │
│  │ Signal 2: FTS5/BM25 keyword match     │       │
│  │ Signal 3: ACT-R activation            │       │
│  │   base_level + spreading + importance │       │
│  │   + scope_boost + adversarial_penalty │       │
│  │ Signal 4: Revival spike (new links)   │       │
│  │                                       │       │
│  │ → RRF fusion (configurable)           │       │
│  │ → Dampening (gravity, hub, resolution)│       │
│  │ → Q-value reranking (UCB-Tuned)       │       │
│  │ → Intent-based type boosting          │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  LIFECYCLE                                      │
│  ┌──────────────────────────────────────┐       │
│  │ Scope management (active→cold→closed) │       │
│  │ Gauge pressure (merge/archive/cool)   │       │
│  │ Consolidation (promote/demote/prune)  │       │
│  │ Tarjan bridge protection              │       │
│  │ Session reward tracking               │       │
│  │ LinUCB pipeline optimization          │       │
│  └──────────────────────────────────────┘       │
│                                                 │
│  STORAGE (unified SQLite)                       │
│  ┌──────────────────────────────────────┐       │
│  │ um_facts: content, embedding, type,   │       │
│  │   target, scope, activation, q_value, │       │
│  │   metabolic_rate, importance, ...     │       │
│  │ um_links: Hebbian (NPMI, co-occur)    │       │
│  │ um_scopes: lifecycle management       │       │
│  │ um_access_times: ACT-R history        │       │
│  │ um_qvalues: RL reward tracking        │       │
│  │ um_facts_fts: FTS5 keyword index      │       │
│  └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

## Fact Types (MEMORY_SPEC Notation)

| Notation | Type       | Metabolic Rate | Description |
|----------|------------|----------------|-------------|
| C[t]: x  | Constraint | 0.3x (slow)    | Rules, requirements |
| D[t]: x  | Decision   | 0.7x           | Architectural choices |
| V[t]: x  | Value      | 1.0x (normal)  | Concrete values |
| ?[t]: x  | Unknown    | 2.0x (fast)    | Open questions |
| ✓[t]: x  | Done       | 2.5x           | Resolved items |
| ~[t]: x  | Obsolete   | 5.0x (rapid)   | Superseded facts |

Metabolic rate multiplies the ACT-R decay parameter `d`, making
constraints persist ~3x longer than values, while unknowns decay
~2x faster (urgency drives resolution).

## Scoring Formula

For each candidate fact, the activation score is:

```
ACTIVATION = BASE_LEVEL + SPREADING + IMPORTANCE + SCOPE + ADVERSARIAL

BASE_LEVEL = ln(Σ tᵢ^(-d × metabolic_rate))
  where tᵢ = time since access i

SPREADING = w_semantic × cosine_sim(query, fact) + hebbian_one_hop

IMPORTANCE = w_importance × importance × (1.5 + semantic_sim × (2 + |base_level|))

SCOPE = scope_multiplier × (0.5 + 0.5 × semantic_sim)  [if scope matches]

ADVERSARIAL = -score × 10.0  [if adversarial content detected]
```

Post-scoring pipeline:
1. **RRF Fusion** — Score-weighted Reciprocal Rank Fusion with BM25
2. **Dampening** — Gravity (cosine ghosts), Hub (P90 degree), Resolution boost
3. **Q-Value Reranking** — Lambda blend with UCB-Tuned exploration
4. **Intent Boost** — Type-aware boosting based on query intent

## Key Features

### From Cognitive Memory
- **ACT-R Activation** — frequency + recency = retrievability
- **Hebbian Links** — GloVe weighting, Ebbinghaus decay, Turrigiano homeostasis
- **Q-Value RL** — learn which memories are actually useful
- **Contradiction Detection** — entity overlap + update language + embedding floor

### From Structured Memory
- **Typed Facts** — C/D/V/?/✓/~ with MEMORY_SPEC notation
- **Supersession** — same type+target automatically replaces old fact
- **Scopes** — unit-of-work lifecycle (active → cold → closed)
- **Gauge Pressure** — 5-tier cascade: merge → warn → archive → cool → refuse

### From Ori-Mnemos
- **NPMI** — Normalized Pointwise Mutual Information on co-occurrence edges
- **UCB-Tuned** — Variance-aware exploration (high variance → more exploration)
- **Tarjan Protection** — Bridge nodes in the link graph can't be pruned
- **Bibliographic Coupling** — Cold-start link seeding from shared keywords
- **Revival Spike** — Dormant facts boosted when they get new connections
- **LinUCB Pipeline** — Self-optimizing retrieval (learn which stages help per query)
- **Intent Classification** — 6 query types with signal weight shifting
- **Session Rewards** — Auto-infer rewards from store-after-recall patterns

## File Structure

```
unified_memory/
├── __init__.py          Package init (v0.1.0)
├── types.py             FactType enum, MemoryFact, ScoredFact, parse_notation
├── config.py            UnifiedMemoryConfig (50+ tunable parameters)
├── schema.py            SQLite DDL (6 tables, FTS5, triggers, views)
├── store.py             UnifiedMemoryStore (main engine class)
├── retrieval.py         Scoring pipeline, dampening, Q-value, adversarial
├── links.py             Hebbian links (NPMI, GloVe, Ebbinghaus, homeostasis)
├── lifecycle.py         Tarjan bridge detection, protection
├── intent.py            Query intent classification (30+ patterns)
├── bandit.py            LinUCB contextual bandits, session reward tracking
├── benchmark_adapter.py BenchmarkableStore wrapper
└── ARCHITECTURE.md      This file

tools/
└── unified_memory_tool.py  Agent-callable toolset (8 tools)

tests/unified_memory/
├── test_schema.py       Schema tests (10)
├── test_store_write.py  Write path tests (7)
├── test_store_recall.py Retrieval tests (7)
├── test_lifecycle.py    Lifecycle tests (5)
├── test_intent.py       Intent classification tests (6)
├── test_npmi.py         NPMI normalization tests (6)
├── test_ucb_tuned.py    UCB-Tuned exploration tests (3)
├── test_bootstrap.py    Bibliographic coupling tests (5)
├── test_tarjan.py       Articulation point tests (5)
├── test_revival.py      Revival spike + saturation tests (9)
├── test_bandit.py       LinUCB + session reward tests (15)
└── test_agent_tool.py   Agent tool integration tests (13)
```

## Configuration

All parameters are in `UnifiedMemoryConfig`. Key groups:

- **ACT-R**: `d`, `w_semantic`, `w_importance`
- **Links**: `hebbian_learning_rate`, `semantic_link_threshold`, `links_per_memory`
- **Q-Value**: `enable_qvalue_reranking`, `qvalue_lambda_min/max`, `qvalue_exploration_c`
- **Dampening**: `enable_dampening`, `gravity_dampening_factor`, `hub_dampening_max_penalty`
- **RRF**: `enable_rrf_fusion`, `rrf_activation_weight`, `rrf_keyword_weight`
- **Unified**: `enable_typed_decay`, `enable_supersession`, `enable_pressure`
- **Advanced**: `enable_intent_classification`, `enable_npmi`, `enable_linucb`

## Toolset

Enable in hermes config with `- unified_memory` in toolsets. Tools:

| Tool | Description |
|------|-------------|
| `mcp_umemory_write` | Store facts (plain text or MEMORY_SPEC) |
| `mcp_umemory_recall` | 4-signal fusion semantic recall |
| `mcp_umemory_search` | FTS5 keyword search (fast) |
| `mcp_umemory_reflect` | Type-grouped topic reflection |
| `mcp_umemory_reward` | Apply RL reward signal |
| `mcp_umemory_explore` | PPR multi-hop exploration |
| `mcp_umemory_stats` | Store statistics |
| `mcp_umemory_consolidate` | Run lifecycle management |
