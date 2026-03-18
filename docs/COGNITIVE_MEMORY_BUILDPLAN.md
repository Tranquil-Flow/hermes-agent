# Cognitive Memory System — Build Plan

**For:** Autonomous building agent  
**Context:** Read `docs/COGNITIVE_MEMORY_DESIGN.md` first. This file tells you what to build and in what order.  
**Repo:** hermes-agent (you're already in it)  
**Rule:** Do NOT modify existing files (`memory_tool.py`, `prompt_builder.py`, `run_agent.py`) until Phase 5.

---

## Pre-Build Checklist

Before writing any code, verify:

```bash
# You're in the right repo
cd /workspace/Projects/hermes-agent
git log --oneline -3

# Key existing files you must NOT touch (yet)
ls tools/memory_tool.py
ls tools/session_search_tool.py  
ls agent/prompt_builder.py

# Python environment
python3 --version  # need 3.9+
pip3 install numpy  # if not already installed
```

---

## Phase 1: Benchmark Framework + Baseline

**Goal:** Working benchmark suite that produces baseline numbers for the current system.  
**Why first:** We need numbers BEFORE we change anything. Can't prove improvement without a before.

### Step 1.1: Create directory structure

```bash
mkdir -p cognitive_memory/backends
mkdir -p tools
mkdir -p tests/cognitive_memory
mkdir -p benchmarks/{suite_a/fixtures,suite_b/data,suite_c,suite_d,suite_e,suite_f,baseline,results,visualize}
```

All directories and `__init__.py` files are pre-created for you (see skeleton files).

### Step 1.2: Build fixture files

Fixture files are pre-created in `benchmarks/suite_a/fixtures/`. Verify they exist:

```bash
ls benchmarks/suite_a/fixtures/
# Should see: semantic_recall.json, contradictions.json, decay_scenarios.json,
#             linking_scenarios.json, preferences.json, scoped_memories.json,
#             adversarial.json
```

Each fixture is a JSON array of test scenarios. Format per file:

**semantic_recall.json:** `[{id, fact, query, gold_answer, difficulty}]`
**contradictions.json:** `[{id, fact_a, fact_b, query, gold_answer, type}]`
**decay_scenarios.json:** `[{id, memories[], rehearsal_pattern{}, time_days, query, expected_retained[], expected_pruned[]}]`
**linking_scenarios.json:** `[{id, facts[], coretrieval_pattern[], query, gold_answer}]`
**preferences.json:** `[{id, statements[], query, gold_answer, evolution_type}]`
**scoped_memories.json:** `[{id, memories[{content,scope}], query_scope, query, gold_answer}]`
**adversarial.json:** `[{id, memories[], query, gold_answer, adversarial_type, expected_behavior}]`

### Step 1.3: Build the judge

`benchmarks/judge.py` evaluates whether a system's answer matches the gold label.
Uses LLM (Haiku) with per-question-type prompt templates matching LongMemEval protocol.

Key functions:
```python
async def judge_answer(question_type: str, question: str, gold_answer: str, 
                       hypothesis: str, model: str = "claude-haiku-4.5") -> JudgeResult
```

Returns `JudgeResult(correct: bool, confidence: float, raw_response: str)`

### Step 1.4: Build statistical module

`benchmarks/statistical.py` provides:
```python
def aggregate_runs(runs: list[RunResult]) -> AggregateResult
  # mean, std, 95% CI per category and overall

def significance_test(baseline_runs: list, experiment_runs: list) -> SignificanceResult
  # paired t-test or Wilcoxon, returns p-value and effect size

def format_results_table(results: dict) -> str
  # Markdown table for PR description
```

### Step 1.5: Build baseline adapters

`benchmarks/baseline/flat_file_baseline.py`:
- Simulates current MEMORY.md system
- store() = append to list (with 2200 char limit)
- recall() = substring search across stored entries
- No embeddings, no scoring

`benchmarks/baseline/fts5_baseline.py`:
- Simulates current session_search
- SQLite FTS5 indexing of stored content
- recall() = FTS5 MATCH query, rank by BM25

Both implement the same interface as the cognitive memory store so the
benchmark runner can swap them transparently.

### Step 1.6: Build Suite A runner

`benchmarks/suite_a/runner.py`:
```python
def run_suite_a(store: MemoryStoreInterface, seed: int = 42, 
                judge_model: str = "claude-haiku-4.5") -> SuiteAResult
```

Loads all fixtures, runs each category, evaluates with judge, returns per-category scores.

### Step 1.7: Build benchmark CLI

`benchmarks/run_benchmarks.py`:
```bash
# Run Suite A baseline
python benchmarks/run_benchmarks.py suite-a --backend baseline-flat --runs 5

# Run Suite A with cognitive memory  
python benchmarks/run_benchmarks.py suite-a --backend builtin --profile balanced --runs 5

# Run ablation
python benchmarks/run_benchmarks.py ablation --runs 3

# Run sensitivity analysis
python benchmarks/run_benchmarks.py sensitivity --parameter d --runs 3

# Compare results
python benchmarks/run_benchmarks.py compare results/baseline.json results/cognitive.json
```

### Step 1.8: Run baselines

```bash
python benchmarks/run_benchmarks.py suite-a --backend baseline-flat --runs 5 --output results/baseline_flat.json
python benchmarks/run_benchmarks.py suite-a --backend baseline-fts5 --runs 5 --output results/baseline_fts5.json
```

**CHECKPOINT:** You should now have baseline numbers. Commit:
```bash
git add benchmarks/ cognitive_memory/ tests/
git commit -m "feat: benchmark framework + baseline measurements for cognitive memory"
```

---

## Phase 2: Core Implementation

**Goal:** Working cognitive memory with semantic search and ACT-R activation.

### Step 2.1: Config dataclass

`cognitive_memory/config.py` — Pre-created. Contains `CognitiveMemoryConfig` dataclass
with all parameters from the design doc Section 6, plus profile presets.

### Step 2.2: Embeddings

`cognitive_memory/embeddings.py` — Implement the fallback chain:
1. Try sentence-transformers (`all-MiniLM-L6-v2`)
2. Try ollama
3. Try openai  
4. Fall back to TF-IDF

Key class: `EmbeddingProvider` with `encode(text: str) -> np.ndarray`

**Test:** `tests/cognitive_memory/test_embeddings.py` — verify fallback chain works,
verify cosine similarity of "dog" and "puppy" > "dog" and "airplane".

### Step 2.3: Encoding

`cognitive_memory/encoding.py` — Category classification and importance estimation.
Pure heuristic, no LLM calls. Pattern matching on content.

**Test:** `tests/cognitive_memory/test_encoding.py` — verify "user prefers X" → preference,
"run command Y" → procedural, importance estimates are reasonable.

### Step 2.4: Storage backend

`cognitive_memory/backends/builtin.py` — SQLite implementation.
- Schema creation (memories table, indexes)
- CRUD operations
- Embedding storage as BLOB
- `access_times` as JSON array

**Test:** `tests/cognitive_memory/test_store.py` — CRUD, persistence, embedding round-trip.

### Step 2.5: ACT-R activation

`cognitive_memory/activation.py`:
```python
def compute_activation(memory: MemoryEntry, query_embedding: np.ndarray,
                       config: CognitiveMemoryConfig) -> float:
    base_level = actr_base_level(memory.access_times, config.d)
    spreading = config.w_semantic * cosine_similarity(query_embedding, memory.embedding)
    importance = config.w_importance * memory.importance
    return base_level + spreading + importance
```

**Test:** `tests/cognitive_memory/test_activation.py`:
- Frequently accessed memory > rarely accessed (base level)
- Recently accessed > old (base level)
- Semantically similar > dissimilar (spreading)
- High importance > low importance

### Step 2.6: Main store orchestrator

`cognitive_memory/store.py` — `CognitiveMemoryStore` class:
- `store(content, category=None, scope='global', importance=None) -> str`
- `recall(query, scope=None, top_k=None) -> list[ScoredMemory]`
- `get_stats() -> dict`

**Test:** Store 10 facts, recall by paraphrase, verify top results are correct.

### Step 2.7: First benchmark run

```bash
python benchmarks/run_benchmarks.py suite-a --backend builtin --profile balanced --runs 5
python benchmarks/run_benchmarks.py compare results/baseline_flat.json results/cognitive_v1.json
```

**CHECKPOINT:** First improvement numbers. Commit:
```bash
git commit -m "feat: core cognitive memory — embeddings, ACT-R activation, semantic recall"
```

---

## Phase 3: Advanced Features

**Goal:** Contradiction detection, Hebbian links, consolidation.

### Step 3.1: Contradiction detection

`cognitive_memory/contradiction.py`:
- Stage 1: embedding similarity > threshold + same category → flag
- Stage 2: LLM judge call (if available) → confirm/deny → set superseded_by

**Test:** Store "uses PostgreSQL", then "migrated to MySQL". Verify old superseded.
Also test near-contradiction that ISN'T one (different scopes).

### Step 3.2: Hebbian links

`cognitive_memory/hebbian.py`:
- On recall: for each pair of co-recalled memories, strengthen link
- Link weight += η × activation_product
- New links created when both highly activated but no link exists
- One-hop spreading in activation calculation

**Test:** Co-recall "Docker" and "deployment" 5 times. Verify link formed.
Query "deployment" and verify Docker-related memories get boosted.

### Step 3.3: Consolidation

`cognitive_memory/consolidation.py`:
- Working → core promotion (access_count + importance thresholds)
- Core → archive demotion (low activation)
- Archive pruning (below threshold)
- Link weight decay

**Test:** Create memories, simulate time + access patterns, run consolidation,
verify correct layer assignments.

### Step 3.4: Engram backend (optional)

`cognitive_memory/backends/engram_backend.py`:
- Wraps `engramai` if installed
- Same interface as builtin
- Skip if `engramai` not available

### Step 3.5: Full ablation

```bash
python benchmarks/run_benchmarks.py ablation --runs 3
```

This runs Suite D (all 11 configurations). Takes a while.

**CHECKPOINT:** Feature attribution data. Commit:
```bash
git commit -m "feat: contradiction detection, Hebbian links, consolidation, ablation results"
```

---

## Phase 4: Optimization

**Goal:** Find optimal parameters through systematic search.

### Step 4.1: Cross-embedding comparison (Suite E)

```bash
python benchmarks/run_benchmarks.py embedding-comparison --runs 3
```

### Step 4.2: Sensitivity analysis (Suite C1)

```bash
python benchmarks/run_benchmarks.py sensitivity --runs 3
```

### Step 4.3: Focused grid search (Suite C2)

After C1, identify the 3-4 most sensitive parameters. Update grid in config.

```bash
python benchmarks/run_benchmarks.py grid-search --runs 3
```

### Step 4.4: Temporal stability (Suite F)

```bash
python benchmarks/run_benchmarks.py temporal-stability
```

### Step 4.5: Final validation (Suite C3)

Top-3 configs from C2 on full benchmarks:

```bash
python benchmarks/run_benchmarks.py final-validation --runs 5
```

### Step 4.6: Update defaults

Based on results, update `cognitive_memory/config.py` profile defaults
to empirically optimal values.

**CHECKPOINT:** Optimization complete. Commit:
```bash
git commit -m "feat: optimized parameters — grid search results, sensitivity curves, temporal stability"
```

---

## Phase 5: Integration + PR

**Goal:** Wire cognitive memory into hermes-agent, prepare PR.

### Step 5.1: Tool registration

`tools/cognitive_memory_tool.py` — register as a new tool alongside `memory`.
New tool name: `cognitive_memory` (NOT replacing `memory`).

Actions: `store`, `recall`, `consolidate`, `stats`, `resolve` (contradiction).

### Step 5.2: Prompt builder hook

Small, non-breaking addition to `agent/prompt_builder.py`:
- If cognitive_memory module is importable AND has stored memories:
  - Call `recall(current_turn_summary)` with top-K=5
  - Append results after curated memory blocks in system prompt
- If not importable: skip entirely (no error)

**This is the ONLY existing file modification.**

### Step 5.3: Honcho bridge (optional)

`cognitive_memory/honcho_bridge.py`:
- If Honcho is configured, augment recall with `get_context()`
- Feed Honcho conclusions back as cognitive memories
- Skip entirely if not configured

### Step 5.4: Integration tests

Test the full flow: store via tool → recall via tool → appears in prompt → 
agent uses it correctly.

### Step 5.5: Generate PR artifacts

```bash
python benchmarks/run_benchmarks.py generate-report --output results/FINAL_REPORT.md
```

Generates:
- Before/after comparison tables
- Ablation waterfall chart description
- Parameter sensitivity findings
- Pareto accuracy-vs-cost summary
- Temporal stability findings
- Per-profile optimal configurations

### Step 5.6: Submit

```bash
git add .
git commit -m "feat: cognitive memory system with benchmarked improvements

- ACT-R activation scoring (frequency × recency × importance)
- Hebbian co-activation links
- Three-layer consolidation (working → core → archive)
- LLM-assisted contradiction detection
- Scope filtering (global / project / topic)

Benchmark results:
- Suite A: XX% improvement over baseline (p < 0.05)
- LongMemEval: XX% → XX% 
- LoCoMo: XX% → XX%
- See benchmarks/results/ for full data"
```

---

## Testing Strategy

### Unit Tests

Every module gets tests. Run with:
```bash
python -m pytest tests/cognitive_memory/ -v
```

Minimum test coverage targets:
- `activation.py`: 95% (core algorithm, must be correct)
- `hebbian.py`: 90%
- `contradiction.py`: 90%
- `consolidation.py`: 85%
- `encoding.py`: 80%
- `embeddings.py`: 75% (hard to test fallback chain fully)
- `store.py`: 90%
- `config.py`: 100% (simple dataclass)

### Integration Tests

After Phase 5:
- Store memory via tool → recall via tool → verify correct
- Store contradicting memory → verify old superseded
- Multiple sessions → verify consolidation runs
- Missing deps → verify graceful degradation

### Benchmark Tests

The benchmarks ARE the tests for the system as a whole.
Suite A is the fast iteration test. Suite B is the acceptance test.

---

## Key Interfaces

These are defined in the skeleton code. Implement against them.

```python
# cognitive_memory/backends/base.py
class StorageBackend(ABC):
    def store(self, entry: MemoryEntry) -> str: ...
    def get(self, memory_id: str) -> Optional[MemoryEntry]: ...
    def get_all_active(self) -> list[MemoryEntry]: ...
    def update(self, memory_id: str, **fields) -> None: ...
    def get_links(self, memory_id: str) -> list[MemoryLink]: ...
    def create_link(self, source_id: str, target_id: str, weight: float, link_type: str) -> None: ...
    def update_link(self, source_id: str, target_id: str, **fields) -> None: ...

# cognitive_memory/store.py
class CognitiveMemoryStore:
    def store(self, content: str, **kwargs) -> str: ...
    def recall(self, query: str, **kwargs) -> list[ScoredMemory]: ...
    def consolidate(self) -> ConsolidationReport: ...
    def get_stats(self) -> dict: ...
    def resolve_contradiction(self, memory_id: str, keep: bool) -> None: ...

# benchmarks interface
class BenchmarkableStore(ABC):
    def store(self, content: str, **kwargs) -> None: ...
    def recall(self, query: str, top_k: int = 10) -> list[str]: ...
    def reset(self) -> None: ...
```

---

## Pitfalls & Warnings

1. **Do NOT modify `memory_tool.py` until Phase 5.** The existing tool must keep working unchanged.

2. **Embedding model download:** `all-MiniLM-L6-v2` is ~80MB. First import will download it. Account for this in container environments. If download fails (no internet), fall back to TF-IDF.

3. **ACT-R `access_times` grows unbounded.** Cap at last 100 timestamps to prevent memory bloat. The ACT-R sum converges quickly — contributions from accesses older than ~100 are negligible.

4. **Hebbian links can explode.** If 10 memories are co-recalled, that's 45 pairs. With `LINKS_PER_MEMORY=5`, enforce the cap strictly.

5. **LLM judge calls in contradiction detection add latency.** Make them async where possible. If the LLM is unavailable, flag but don't block.

6. **Suite B datasets are large.** LongMemEval is ~500MB. Download once, cache in `benchmarks/suite_b/data/`.

7. **Benchmark reproducibility requires fixed seeds.** Use `random.seed(seed)` and `np.random.seed(seed)` at the start of every run.

8. **The TF-IDF fallback produces worse embeddings.** That's expected. Suite E will quantify exactly how much worse. Don't optimize for TF-IDF — optimize for sentence-transformers and let TF-IDF be the graceful degradation.

9. **AGPL-3.0 (Engram):** Do NOT copy any Engram source code. The backend adapter should only call their public API (`from engram import Memory`). This is safe as a dependency. Copying code would make our module AGPL.

10. **Tests must work without network access.** The container may not have internet. Mock the LLM judge calls in unit tests. Benchmark runs DO need LLM access for the judge.
