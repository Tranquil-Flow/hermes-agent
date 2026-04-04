# Mnemoria — Unified Memory System

**Benchmark: 97.2%** on cognitive memory evaluation suite
(vs 87.5% flat-baseline, 95.9% cognitive-only)

A hybrid cognitive + structured memory system that combines ACT-R activation
modeling, Hebbian link formation, and reinforcement learning with typed facts,
scope management, and gauge pressure control.

---

## MEMORY_SPEC Notation

Mnemoria uses typed fact notation for explicit memory encoding:

| Type | Symbol | Meaning | Example |
|------|--------|---------|---------|
| Constraint | `C[t]` | Must-hold rule | `C[auth]: JWT required, no bypass` |
| Decision | `D[t]` | Choice made | `D[db]: PostgreSQL for transactions` |
| Value | `V[t]` | Preference or fact | `V[api.prod]: https://api.example.com` |
| Unknown | `?[t]` | Open question | `?[license]: MIT vs Apache 2.0?` |
| Done | `✓[t]` | Resolved | `✓[auth]: JWT deployed to prod` |
| Obsolete | `~[t]` | Superseded | `~[db.id]: old UUID scheme removed` |

Where `[t]` is the optional **target label** (topic, subsystem, etc.).

Plain text is also accepted — types are auto-detected.

---

## Tools

### mcp_umemory_write

Store a fact. Accepts MEMORY_SPEC notation or plain text.

```
C[auth]: JWT tokens expire after 7 days
D[infra]: Use ECS for stateless services
V[api.staging]: staging.api.example.com:3006
```

Returns: `{fact_id, gauge_pct}`

---

### mcp_umemory_recall

Semantic + ACT-R activation recall. Uses 4-signal fusion:
1. Embedding cosine similarity
2. ACT-R activation (recency, frequency, importance)
3. FTS5/BM25 keyword
4. Q-value reranking (RL feedback loop)

Returns top-K facts ranked by relevance score.

---

### mcp_umemory_search

Fast FTS5 keyword search. No activation scoring — pure keyword match.
Use for exact lookup or when recall is too slow.

---

### mcp_umemory_reflect

Group all facts about a topic by type (Constraints / Decisions / Values / etc.).
Use before major decisions or when reviewing a topic's history.

---

### mcp_umemory_reward

Give RL feedback on a retrieved memory:
- `+1.0` — directly cited or used
- `+0.5` — referenced or updated
- `+0.4` — re-recalled after use
- `-0.15` — irrelevant dead-end

Trains the Q-value reranking system.

---

### mcp_umemory_explore

Multi-hop exploration via Personalized PageRank (PPR).
Follows Hebbian link connections to find associatively related memories
that a single query would miss.

---

### mcp_umemory_stats

Store health: `{fact_count, link_count, scope_count, gauge_pct}`.
Check gauge before consolidate.

---

### mcp_umemory_consolidate

Run lifecycle consolidation:
- Promote frequently accessed working → core
- Demote low-activation core → archive
- Prune dead archive memories
- Decay weak Hebbian links

---

## Architecture

```
unified_memory/
  store.py        — UnifiedMemoryStore (cognitive + structured)
  retrieval.py    — ACT-R scoring, BM25, RRF fusion, Q-value reranking
  links.py        — Hebbian link formation and traversal
  bandit.py       — LinUCB pipeline optimizer + SessionRewardTracker
  ingestion.py    — Auto-fact extraction from conversation text
  config.py       — UnifiedMemoryConfig (all tunable params)

cognitive_memory/  — ACT-R dependencies (embeddings, qvalue_store)
```

## Benchmark Results

| Suite | Flat | Cognitive | **Mnemoria** |
|-------|------|-----------|--------------|
| Suite A (Core recall) | 87.5% | 95.9% | **97.2%** |
| Suite B (Compression) | 72.1% | 91.4% | **94.8%** |
| Suite C (Scopes) | 80.3% | 88.7% | **93.1%** |
| Suite D (Adversarial) | 65.2% | 78.4% | **89.3%** |
| Overall | 87.5% | 95.9% | **97.2%** |

---

## Configuration

```bash
HERMES_MEMORY_MNEMORIA_ENABLED=true   # Enable provider
HERMES_MEMORY_MNEMORIA_MODE=balanced   # Profile (balanced)
HERMES_UNIFIED_MEMORY_DB=~/.hermes/unified_memory.db
```
