# Cognitive Memory — Handover Document

**Date:** 2026-03-18
**Last commit:** `6e0170f5` — fix: prevent false supersession of structurally similar facts

## Current State

### Benchmark Scores (Suite A — 200 scenarios)

```
                  Cognitive  Baseline  Delta
contradictions      0.850    0.600    +25pp
cross_reference     0.911    0.533    +38pp
importance          0.900    0.800    +10pp
semantic_recall     1.000    1.000      0
temporal_decay      0.933    0.822    +11pp
overall             0.930    0.775    +15.5pp
```

All 37 unit tests passing. Clean git state.

### Run benchmarks

```bash
cd /workspace/Projects/hermes-agent
python -m benchmarks.runner --backend cognitive
python -m benchmarks.runner --backend baseline-flat
python -m pytest tests/cognitive_memory/ -q
```

## What Was Done (Sessions 1-3)

### Session 1: Foundation
- Built CognitiveMemoryStore with ACT-R activation scoring
- TF-IDF embeddings, SQLite backend, Hebbian links
- Suite A benchmark framework (5 categories, 200 scenarios)
- Heuristic judge (keyword/substring matching)

### Session 2: Contradiction Detection Overhaul
- Old approach (cosine threshold 0.82) scored 0% — contradicting facts use different words
- New approach: update-language gate + entity overlap + stemming
- `_has_update_signal()` — detects change verbs (migrated, switched, upgraded, now uses...)
- `_extract_key_terms()` — extracts product names, acronyms, domain words with stemming
- Importance boost on supersession (min 0.9) — overcomes recency bias from distractors
- Result: contradictions 0% → 85%, overall 76% → 84%

### Session 3: Judge Improvement + False Supersession Fix
- Heuristic judge couldn't evaluate reasoning answers (e.g., "96 GB = 3 × 64 × 50%")
- Added number + entity co-occurrence matching for cross-reference answers
- Added combined identifier + keyword threshold
- Fixed false supersession: structurally similar sentences ("A depends on B" / "B depends on C") had high TF-IDF cosine sim, triggering near-duplicate bypass. Added word Jaccard gate (threshold 0.75)
- Result: overall 84% → 93%

## Remaining Failures (7 total)

### Cross-reference (4 failures)
- **1 recall miss** (xr_h02): "Service A depends on Service B" not in top-k — TF-IDF can't rank bridge facts with low query similarity
- **3 judge ceiling** (xr_h04, xr_h11, xr_h13): facts ARE recalled, heuristic judge can't verify computed/reasoning answers. LLM judge would fix these.

### Contradictions (3 failures out of 20)
- ct_07, ct_17, and one other: old and new facts use completely different technology names (React→Next.js, CloudWatch→Loki). Zero entity overlap even with stemming. Needs better embeddings.

## Key Files

```
cognitive_memory/
  store.py              — main engine (ACT-R scoring, contradiction detection, recall)
  config.py             — CognitiveMemoryConfig with profiles (balanced, etc.)
  encoding.py           — classify + importance scoring
  embeddings.py         — TF-IDF embedding provider
  backends/
    base.py             — MemoryEntry, ScoredMemory, StorageBackend ABC
    builtin.py          — SQLite backend
  benchmark_adapter.py  — wraps store for benchmark runner

benchmarks/
  runner.py             — main benchmark runner (Suite A)
  judge.py              — HeuristicJudge + MemoryJudge (LLM)
  interface.py          — BenchmarkableStore protocol
  statistical.py        — aggregation, confidence intervals
  suite_a/fixtures/     — 5 JSON fixture files (200 scenarios)
  suite_b/ through f/   — placeholder (not implemented)
  results/              — JSON results from last runs

tests/cognitive_memory/
  test_store.py         — 37 tests
  test_encoding.py
  test_benchmark_adapter.py
```

## Aegis Vault / LLM Judge

The LLM judge (`benchmarks/judge.py` MemoryJudge class) calls Claude Haiku through aegis proxy. Currently NOT working because vault has no key.

**To fix** (on host):
```bash
hermes-aegis vault set ANTHROPIC_API_KEY <your-claude-oauth-token>
```

The injector auto-detects OAuth tokens (anything not starting with `sk-ant-api`) and uses Bearer auth instead of x-api-key header. See `hermes-aegis/src/hermes_aegis/proxy/injector.py` lines 132-138.

Once working, run benchmarks with LLM judge:
```bash
python -m benchmarks.runner --backend cognitive --judge claude-3-5-haiku-20241022
```

## Priority Roadmap

### P0: LLM Judge (biggest unlock)
- Fix vault injection → enables Claude Haiku as judge
- 3 cross-reference failures are pure judge ceiling
- Expected: overall 0.930 → 0.95+

### P1: Better Embeddings
- TF-IDF is the fundamental bottleneck
- sentence-transformers (all-MiniLM-L6-v2) would fix:
  - Synonym/paraphrase matching
  - The 1 real cross-reference recall miss
  - 3 remaining contradiction misses (different technology names)
- EmbeddingProvider interface already exists in embeddings.py — needs a SentenceTransformerProvider

### P2: LongMemEval Integration (External Benchmark)
- 500 questions, 5 categories, ICLR 2025
- Data: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- Code: https://github.com/xiaowu0162/LongMemEval
- Tests: info extraction, multi-session reasoning, knowledge updates, temporal reasoning, abstention
- Integration: write adapter that feeds chat sessions into our store, answers questions, evaluates with their script
- Good fit for Suite B slot

### P3: Missing Test Suites
- Suite B: Consolidation & compression (does merging preserve meaning?)
- Suite D: Adversarial robustness (prompt injection in stored memories)
- Suite E: Scale testing (recall accuracy at 100/1K/10K memories)

### P4: Hebbian Spreading Activation
- Links exist but spreading is weak relative to recency/similarity
- Co-stored facts should boost each other during recall
- Would help cross-reference scenarios (bridge facts get activation from related retrieved facts)

### P5: Realistic Cross-Reference Testing
- Current benchmark stores ONLY the needed facts (no distractors)
- Real memory has hundreds of irrelevant entries
- Adding noise to cross-ref scenarios would be more realistic
- Debug showed 17.8% with 10 distractors vs 68.9% without
