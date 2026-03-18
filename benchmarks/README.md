# Hermes Cognitive Memory Benchmark Suite

Quantitative evaluation framework for the Hermes cognitive memory system
described in `docs/COGNITIVE_MEMORY_DESIGN.md`. Compares recall accuracy
against a flat baseline and optionally against Engram or other backends.


## Overview

The current Hermes memory system stores facts as flat text entries in a
YAML file (`~/.hermes/memory/`), injected into the system prompt each turn.
The proposed cognitive memory system adds:

- Semantic embeddings for recall (not just substring match)
- Importance weighting and decay over time
- Contradiction detection and resolution
- Cross-reference linking between related memories
- Consolidation cycles that compress and merge memories

This benchmark suite measures whether those features actually improve
recall accuracy in realistic scenarios.


## Test Suites

### Suite A: Core Memory Operations (implemented)

200 test scenarios across 5 categories:

| Category             | Scenarios | Tests                                            |
|----------------------|-----------|--------------------------------------------------|
| Semantic Recall      | 50        | Synonym/paraphrase matching at easy/medium/hard  |
| Contradictions       | 20        | Clear overrides and subtle partial updates        |
| Temporal Decay       | 45        | Recency bias, rehearsal persistence, narrative    |
| Cross-Reference      | 45        | 2-4 fact chaining with inference                  |
| Importance Filtering | 40        | Signal vs. noise with varying importance scores   |

### Suite B: Consolidation & Compression (placeholder)
Tests whether consolidation preserves meaning while reducing token count.

### Suite C: Multi-scope Memory (placeholder)
Tests user-scoped vs. project-scoped vs. global memory isolation.

### Suite D: Adversarial Robustness (placeholder)
Tests resistance to prompt injection in stored memories, hallucinated
facts, and conflicting instructions.

### Suite E: Scale & Performance (placeholder)
Tests recall accuracy as memory count scales from 10 to 10,000 entries.
Measures token usage and latency.

### Suite F: Integration Tests (placeholder)
End-to-end tests with the actual Hermes agent loop.


## Quick Start

```bash
# Run baseline benchmark (Suite A)
python -m benchmarks.runner --backend baseline-flat --suite a

# Run with fewer iterations for quick testing
python -m benchmarks.runner --backend baseline-flat --suite a --runs 1

# Compare two backends
python -m benchmarks.runner --backend baseline-flat --compare cognitive --suite a

# Output JSON results
python -m benchmarks.runner --backend baseline-flat --suite a --json
```


## Backends

### baseline-flat (implemented)
Python list of strings with word-overlap scoring. No decay, importance,
or semantic understanding. This is the floor — cognitive memory must beat it.

### cognitive (TODO)
The full cognitive memory system from COGNITIVE_MEMORY_DESIGN.md:
- Sentence embeddings for semantic recall
- Exponential decay with rehearsal strengthening
- Contradiction detection via embedding similarity
- Importance-weighted storage and retrieval
- Periodic consolidation

### engram (TODO)
Optional comparison against the Engram memory system for external baseline.


## Scoring

### LLM-as-Judge
Each recalled answer is evaluated by an LLM judge (default: claude-haiku-4.5)
that determines semantic correctness. The judge is strict but allows
paraphrasing — the key information must be present.

A `HeuristicJudge` using keyword matching is available for testing without
API calls.

### Statistical Rigor
- 5 runs per benchmark with different random seeds (42-46)
- Mean ± standard deviation reported
- 95% confidence intervals via t-distribution
- Paired t-test + Wilcoxon signed-rank for backend comparisons
- Cohen's d effect size for practical significance
- Results are only "significant" if p < 0.05


## Fixture Format

Each fixture is a JSON array of scenario objects. Format varies by category:

```json
// semantic_recall.json
{
  "id": "sr_e01",
  "fact": "The project uses PostgreSQL 15",
  "query": "What database version?",
  "gold_answer": "PostgreSQL 15",
  "difficulty": "easy"
}

// temporal_decay.json
{
  "id": "td_m01",
  "facts": [
    {"content": "...", "stored_days_ago": 365, "rehearsed_days_ago": [30, 7]},
    {"content": "...", "stored_days_ago": 30}
  ],
  "query": "...",
  "gold_answer": "...",
  "expected_recency_bias": false,
  "difficulty": "medium"
}
```

See `suite_a/fixtures/` for all formats.


## Adding New Scenarios

1. Add JSON scenarios to the appropriate `suite_X/fixtures/` directory
2. If adding a new category, create a runner function in `runner.py` and
   register it in `CATEGORY_RUNNERS`
3. Follow the difficulty distribution: ~30% easy, ~40% medium, ~30% hard
4. Use realistic software engineering facts (not toy examples)
5. Ensure gold answers are unambiguous


## Results Format

Output JSON structure:

```json
{
  "backend": "cognitive",
  "num_runs": 5,
  "mean_score": 0.847,
  "std_score": 0.023,
  "ci_95": [0.818, 0.876],
  "per_category": {
    "semantic_recall": {"mean": 0.92, "std": 0.01},
    "contradictions": {"mean": 0.85, "std": 0.03},
    "temporal_decay": {"mean": 0.78, "std": 0.04},
    "cross_reference": {"mean": 0.72, "std": 0.05},
    "importance_filtering": {"mean": 0.88, "std": 0.02}
  }
}
```


## Directory Structure

```
benchmarks/
├── __init__.py            # Package init, version
├── __main__.py            # python -m benchmarks entry point
├── interface.py           # BenchmarkableStore ABC + result dataclasses
├── runner.py              # Main benchmark runner with CLI
├── judge.py               # LLM-as-judge + heuristic fallback
├── statistical.py         # Aggregation, CI, significance tests
├── README.md              # This file
├── baseline/
│   ├── __init__.py
│   └── flat_store.py      # Baseline flat memory (fully implemented)
├── suite_a/
│   ├── __init__.py
│   └── fixtures/
│       ├── semantic_recall.json      # 50 scenarios
│       ├── contradictions.json       # 20 scenarios
│       ├── temporal_decay.json       # 45 scenarios
│       ├── cross_reference.json      # 45 scenarios
│       └── importance_filtering.json # 40 scenarios
├── suite_b/               # Placeholder
├── suite_c/               # Placeholder
├── suite_d/               # Placeholder
├── suite_e/               # Placeholder
├── suite_f/               # Placeholder
├── visualize/             # Placeholder for result visualization
└── results/               # Output directory (gitignored)
```


## Requirements

- Python 3.10+
- scipy (for significance testing; degrades gracefully without it)
- anthropic SDK (for LLM judge; use HeuristicJudge without it)
- No other external dependencies for core benchmarking
