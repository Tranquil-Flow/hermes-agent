# Retrieval Intelligence — Design Document

**Date:** 2026-03-27
**Status:** Draft — awaiting review before implementation
**Source:** Algorithms ported from [Ori-Mnemos](https://github.com/aayoawoyemi/Ori-Mnemos) Layers 1 & 3

## Overview

Two remaining retrieval intelligence features require architectural decisions
before implementation: **Q-Value Reranking** (learn from retrieval outcomes)
and **Recursive Exploration** (multi-hop query decomposition). Both are
high-impact but need design choices about persistence, session boundaries,
and LLM integration.

---

## 1. Q-Value Reranking (Ori-Mnemos Layer 1)

### What it does
Each memory earns a Q-value (0.0–1.0) based on whether it was actually
useful after being retrieved. Over time, genuinely useful memories rise
and noise sinks. This is reinforcement learning on retrieval.

### Reward signals

| Signal | Reward | Trigger |
|--------|--------|---------|
| Forward citation | +1.0 | User references retrieved memory in new content |
| Update after retrieval | +0.5 | User edits/updates a retrieved memory |
| Downstream creation | +0.6 | User creates new memory after retrieving |
| Within-session re-recall | +0.4 | Same memory surfaces across different queries |
| Dead end (top-3, no follow-up) | -0.15 | Retrieved in top 3 but nothing follows |

### Q-value update (EMA)
```
new_q = old_q + ALPHA * (reward - old_q)    # ALPHA = 0.1
```

### Temporal decay with Q-informed multiplier
```
mult = 0.7  if q >= 0.7    # good memories decay slower
       1.3  if q <= 0.3    # bad memories decay faster
       1.0  otherwise
decayed_q = q * exp(-0.007 * mult * days_since_update)
```

### Phase B reranking (after activation scoring, before dampening)
```
lambda = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * min(total_updates / 200, 1.0)
score = (1 - lambda) * z_normalize(activation) + lambda * z_normalize(q_value) + ucb_bonus
```
Lambda grows as the system learns — cold start uses mostly activation,
warm system blends in learned Q-values.

UCB-Tuned exploration bonus ensures under-retrieved memories still get discovered:
```
bonus = c * sqrt(log(T) / n * min(0.25, V))    # c = 0.2
```

### Architectural decisions needed

**A. Storage:** Add `q_value REAL DEFAULT 0.5` column to memory entries?
Or separate table `memory_qvalues(memory_id, q_value, update_count, last_updated)`?

**B. Session tracking:** How do we know what happened "after retrieval"?
Options:
  1. **Agent loop integration:** Hook into the hermes agent loop to track
     store/recall/edit sequences within a session. Most accurate but
     tightly couples to the agent.
  2. **Heuristic session tracking:** Track recall() calls with timestamps.
     If a store() follows within N minutes of a recall(), credit the
     recalled memories. Looser but framework-agnostic.
  3. **Explicit feedback API:** Add `memory.reward(memory_id, signal)` that
     the agent/user calls explicitly. Simplest but requires tool integration.

**C. Benchmark testing:** Our current benchmarks run isolated scenarios
(fresh store per scenario). Q-values need multi-round scenarios to show
benefit. Options:
  - Add Suite G: "learning" scenarios that run multiple rounds of
    store/recall/feedback and test whether Q-values improve retrieval
  - Use LongMemEval/LoCoMo with session-based evaluation

**Recommendation:** Option B2 (heuristic session tracking) + B3 (explicit API)
as complementary approaches. Start with the explicit API for testing, add
heuristic tracking later.

---

## 2. Recursive Exploration (Ori-Mnemos RMH Constraint 2)

### What it does
When a single retrieval pass is insufficient, decompose the query into
sub-questions, retrieve against each, and synthesize. This is what should
dramatically improve LongMemEval (currently 13.5%) and LoCoMo (6.0%).

### Algorithm (from Ori-Mnemos explore.ts)

**Phase 1 — PPR Graph Expansion:**
1. Seed: top flat retrieval results with weighted scores
2. Run Personalized PageRank (α=0.45) on the link graph
3. Auto-adjust depth based on PPR activation spread
4. Merge flat + PPR results with PPR boost

**Phase 3 — Recursive Sub-question Decomposition:**
1. Build context from top-10 accumulated results
2. Ask LLM: "What gaps exist? Generate 1-3 sub-questions"
3. For each sub-question: re-retrieve + PPR expand
4. Convergence: stop when new_notes / total_visited < threshold
5. Budget: stop when total_visited >= max_total_notes
6. Re-rank all accumulated results, return top-limit

### Architectural decisions needed

**A. LLM integration:** Recursive exploration needs LLM calls for
sub-question generation. Options:
  1. Use the existing aegis proxy for LLM calls (container environment)
  2. Make LLM calls configurable (model, API key, proxy)
  3. Provide a no-LLM fallback using keyword expansion heuristics

**B. Graph structure:** PPR needs an adjacency matrix/graph.
Our Hebbian links provide this, but:
  - Do we have enough link density for meaningful PPR walks?
  - Should we add wiki-link style edges from content analysis?

**C. Performance:** Recursive exploration can be expensive:
  - Multiple recall() calls per query
  - LLM calls for sub-question generation
  - PPR iterations on the link graph
  
  Should this be a separate `explore()` method or integrated into recall()?

**D. When to trigger:** Not every query needs recursive exploration.
  Options:
  1. Always explore (expensive)
  2. Explore only when initial recall confidence is low
  3. Explicit `explore()` API separate from `recall()`
  4. Auto-detect based on query complexity (question marks, multi-clause)

**Recommendation:** Separate `explore()` method that wraps `recall()`.
Start with PPR graph expansion only (no LLM). Add LLM sub-question
decomposition as opt-in for when LLM access is available.

---

## Implementation Priority

1. **Q-Value Reranking** — medium effort, high long-term value.
   Start with explicit reward API + separate qvalues table.
   Add Suite G benchmark scenarios for multi-round learning.

2. **PPR Graph Expansion** (Phase 1 only) — medium effort, immediate
   value for multi-hop queries. No LLM needed.

3. **Recursive Exploration** (Phase 3) — high effort, high value for
   LongMemEval/LoCoMo. Needs LLM integration.

---

## Current State (post Ori-Mnemos cherry-pick)

| Feature | Status | Impact |
|---------|--------|--------|
| Dampening pipeline (gravity, hub, resolution) | ✅ Shipped | +1.4pp overall, +12.5pp scale |
| BM25 + RRF fusion | ⚠️ Disabled | Regresses temporal/importance |
| Co-occurrence edges (GloVe, Ebbinghaus, homeostasis) | ✅ Shipped | Maintains 96.1%, compounds over time |
| Q-value reranking | 📋 This doc | Needs session tracking design |
| Recursive exploration | 📋 This doc | Needs LLM integration design |

### Baselines for comparison

| Benchmark | Score | Notes |
|-----------|-------|-------|
| Internal suites (284Q) | 96.1% | Strong |
| HotpotQA (200Q) | 70.5% | Decent |
| LongMemEval (200Q) | 13.5% | Weak — needs recursive exploration |
| LoCoMo (100Q) | 6.0% | Very weak — needs recursive exploration |
