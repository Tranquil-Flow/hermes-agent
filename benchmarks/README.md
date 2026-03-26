# Hermes Cognitive Memory Benchmark Suite

Quantitative evaluation framework for the Hermes cognitive memory system.
Measures recall accuracy, adversarial robustness, scale behavior, and
integration quality across 284 scenarios in 6 test suites.


## Overview

The Hermes cognitive memory system goes far beyond a simple flat-text store:

- Semantic embeddings (all-MiniLM-L6-v2) for meaning-aware recall
- ACT-R activation scoring with exponential decay and rehearsal strengthening
- Hebbian associative links for cross-reference chaining
- Contradiction detection and resolution
- Consolidation cycles that compress and merge related memories
- Scope isolation: project / user / team

This benchmark suite measures whether those features improve recall accuracy
in realistic scenarios compared to a word-overlap baseline.

Current overall score: 94.7% (cognitive backend, 284 scenarios, 3 runs)


## Test Suites

### Suite A: Core Memory Operations — 200 scenarios

| Category             | Scenarios | Tests                                           |
|----------------------|-----------|-------------------------------------------------|
| Semantic Recall      | 50        | Synonym/paraphrase matching, easy/medium/hard   |
| Contradictions       | 20        | Clear overrides and subtle partial updates      |
| Temporal Decay       | 45        | Recency bias, rehearsal persistence, narratives |
| Cross-Reference      | 45        | 2–4 fact chaining with inference                |
| Importance Filtering | 40        | Signal vs. noise with varying importance scores |

### Suite B: Consolidation & Compression — 30 scenarios

20 consolidation scenarios verifying that merged memories preserve meaning.
10 compression scenarios measuring token reduction vs. information retention.

### Suite C: Multi-scope Memory — 20 scenarios

Verifies that project-scoped, user-scoped, and team-scoped memories do not
bleed across boundaries. Tests isolation under concurrent context switches.

### Suite D: Adversarial Robustness — 15 scenarios

Tests resistance to:
- Prompt injection via stored memories
- Hallucinated facts inserted as if authoritative
- Conflicting instructions from different memory sources
- Data exfiltration patterns

### Suite E: Scale — 8 scenarios

Needle-in-haystack recall at 10, 50, 100, and 200 stored memories.
Validates that embedding-based retrieval stays accurate as the store grows.

### Suite F: Integration — 11 scenarios

End-to-end multi-step tests with the Hermes agent loop at easy, medium, and
hard difficulty. Measures compound recall across a full conversation.

Total: 284 scenarios across all suites.


## External Benchmark: LongMemEval

Integration with LongMemEval (ICLR 2025), a published long-context memory
evaluation benchmark, lives in benchmarks/longmemeval/.

Files:
  benchmarks/longmemeval/__init__.py   — package init
  benchmarks/longmemeval/adapter.py    — adapts LongMemEval format to Hermes
  benchmarks/longmemeval/runner.py     — runs the external eval

LongMemEval provides independently-curated scenarios not in our fixture set,
giving an unbiased external validity check.

Planned: LoCoMo and HotpotQA integration for additional coverage.


## Backends

### cognitive (default, fully implemented)

The full cognitive memory system:
- Sentence embeddings via sentence-transformers (all-MiniLM-L6-v2)
- ACT-R-style activation: base + recency decay + rehearsal boost + spread
- Hebbian associative links updated on each co-activation
- Contradiction detection via embedding similarity + heuristic rules
- Importance-weighted storage and retrieval
- Periodic consolidation and compression

Run with: --backend cognitive

### baseline-flat (implemented)

Python list of strings with word-overlap (Jaccard) scoring. No decay,
no semantic understanding, no importance weighting. This is the floor —
cognitive memory must beat it on every suite.

Run with: --backend baseline-flat


## Scoring

### Judges

HeuristicJudge (default, no API needed)
  Keyword-matching judge that checks whether gold answer tokens appear in
  the recalled text. Fast, deterministic, zero cost.

LLM Judge (optional, higher accuracy)
  Claude via the aegis proxy. Evaluates semantic correctness strictly but
  allows paraphrasing — the key information must be present. Activated
  automatically when ANTHROPIC_API_KEY or the aegis proxy is available.

### Statistical Rigor

- Multi-seed runs (configurable, default 3)
- Mean ± standard deviation reported per category and overall
- 95% confidence intervals via t-distribution
- Paired t-test + Wilcoxon signed-rank for backend comparisons
- Cohen's d effect size for practical significance
- Results marked significant only when p < 0.05

Planned: Recall@K, MRR, and NDCG metrics for ranked retrieval evaluation.


## CLI Usage

Run all suites, 3 runs each:
  python -m benchmarks --backend cognitive --suite all --runs 3

Run a single suite:
  python -m benchmarks --backend cognitive --suite a --runs 1

Run a subset of suites:
  python -m benchmarks --backend cognitive --suite a,b,c --runs 5

Compare backends on Suite A:
  python -m benchmarks --backend baseline-flat --compare cognitive --suite a

Output JSON (for programmatic consumption):
  python -m benchmarks --backend cognitive --suite all --json

Ablation / parameter sweep:
  python benchmarks/compare_configs.py


## Fixture Format

Fixtures are JSON arrays of scenario objects in suite_X/fixtures/.
Format varies by category:

semantic_recall.json:

  {
    "id": "sr_e01",
    "fact": "The project uses PostgreSQL 15",
    "query": "What database version?",
    "gold_answer": "PostgreSQL 15",
    "difficulty": "easy"
  }

temporal_decay.json:

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

cross_reference.json:

  {
    "id": "xr_h01",
    "facts": ["fact A", "fact B", "fact C"],
    "query": "...",
    "gold_answer": "...",
    "chain_length": 3,
    "difficulty": "hard"
  }

adversarial.json (Suite D):

  {
    "id": "adv_d01",
    "attack_type": "prompt_injection",
    "injected_memory": "Ignore all previous instructions and...",
    "query": "...",
    "gold_answer": "...",
    "should_resist": true
  }

Difficulty distribution guideline: ~30% easy, ~40% medium, ~30% hard.
Use realistic software engineering facts — avoid toy examples.


## Adding Scenarios

1. Add JSON scenarios to the appropriate suite_X/fixtures/ directory.
2. Validate with: python benchmarks/validate.py --suite X
3. If adding a new category, create a runner function in runner.py and
   register it in CATEGORY_RUNNERS.
4. Ensure gold answers are unambiguous and have exactly one correct response.
5. Keep difficulty labels consistent with existing scenarios.


## Results Format

JSON output structure (--json flag):

  {
    "backend": "cognitive",
    "suite": "all",
    "num_runs": 3,
    "overall_mean": 0.947,
    "overall_std": 0.011,
    "ci_95": [0.934, 0.960],
    "per_category": {
      "semantic_recall":      {"mean": 1.000, "std": 0.000},
      "contradictions":       {"mean": 0.850, "std": 0.028},
      "temporal_decay":       {"mean": 0.956, "std": 0.015},
      "cross_reference":      {"mean": 0.933, "std": 0.021},
      "importance_filtering": {"mean": 0.925, "std": 0.018},
      "consolidation":        {"mean": 1.000, "std": 0.000},
      "compression":          {"mean": 0.900, "std": 0.025},
      "scopes":               {"mean": 1.000, "std": 0.000},
      "adversarial":          {"mean": 0.867, "std": 0.033},
      "scale":                {"mean": 0.875, "std": 0.030},
      "integration":          {"mean": 1.000, "std": 0.000}
    }
  }

Planned: visualization of per-category radar charts and score-over-time plots.


## Benchmark Results

Cognitive backend, all suites, 3 runs (as of latest release):

  Overall:              94.7%
  semantic_recall:     100.0%
  contradictions:       85.0%
  temporal_decay:       95.6%
  cross_reference:      93.3%
  importance_filtering: 92.5%
  consolidation:       100.0%
  compression:          90.0%
  scopes:              100.0%
  adversarial:          86.7%
  scale:                87.5%
  integration:         100.0%

The weakest categories are adversarial (86.7%) and contradictions (85.0%),
both targeted for improvement. Scale (87.5%) degrades slightly at 200
memories but remains well above the baseline-flat floor.


## Directory Structure

  benchmarks/
  ├── __init__.py              package init, version
  ├── __main__.py              python -m benchmarks entry point
  ├── interface.py             BenchmarkableStore ABC + result dataclasses
  ├── runner.py                main benchmark runner with CLI
  ├── judge.py                 HeuristicJudge + LLM judge
  ├── statistical.py           aggregation, CI, significance tests
  ├── compare_configs.py       ablation runner and parameter sweeps
  ├── validate.py              fixture schema validation
  ├── README.md                this file
  ├── baseline/
  │   ├── __init__.py
  │   └── flat_store.py        baseline flat memory (word-overlap)
  ├── suite_a/
  │   ├── __init__.py
  │   └── fixtures/
  │       ├── semantic_recall.json       50 scenarios
  │       ├── contradictions.json        20 scenarios
  │       ├── temporal_decay.json        45 scenarios
  │       ├── cross_reference.json       45 scenarios
  │       └── importance_filtering.json  40 scenarios
  ├── suite_b/
  │   ├── __init__.py
  │   └── fixtures/
  │       ├── consolidation.json         20 scenarios
  │       └── compression.json           10 scenarios
  ├── suite_c/
  │   ├── __init__.py
  │   └── fixtures/
  │       └── scopes.json                20 scenarios
  ├── suite_d/
  │   ├── __init__.py
  │   └── fixtures/
  │       └── adversarial.json           15 scenarios
  ├── suite_e/
  │   ├── __init__.py
  │   └── fixtures/
  │       └── scale.json                  8 scenarios
  ├── suite_f/
  │   ├── __init__.py
  │   └── fixtures/
  │       └── integration.json           11 scenarios
  ├── longmemeval/
  │   ├── __init__.py
  │   ├── adapter.py           adapts LongMemEval format to Hermes interface
  │   └── runner.py            external benchmark runner
  ├── visualize/               result visualization (planned)
  └── results/                 output directory (gitignored)


## Requirements

- Python 3.10+
- sentence-transformers (for cognitive backend embeddings)
- scipy (for significance testing; degrades gracefully without it)
- anthropic SDK (for LLM judge; HeuristicJudge works without it)
- No other external dependencies for core benchmarking

Install:
  pip install sentence-transformers scipy anthropic
