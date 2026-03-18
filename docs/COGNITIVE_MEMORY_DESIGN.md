# Cognitive Memory System — Design Document

**Author:** Tranquil-Flow  
**Date:** 2026-03-18  
**Status:** DRAFT — Ready for review  
**Base:** hermes-agent @ ae041806  
**References:** PR #727 (0xbyt4), Honcho v3.0.3, A-MEM (AGIResearch), Engram-AI (tonitangpotato)

---

## 1. Problem Statement

Hermes-agent's current memory system has hard limits that cap agent effectiveness:

| Limitation | Current State | Impact |
|-----------|--------------|--------|
| Curated memory size | 2,200 + 1,375 chars (two flat files) | ~2 paragraphs of notes total |
| Search method | FTS5 keyword matching | Can't find "Docker networking fix" if transcript says "container bridge problem" |
| Recall fidelity | Gemini Flash summarization of transcripts | Lossy — details evaporate |
| Temporal awareness | None | Can't distinguish last Tuesday from three weeks ago |
| Contradiction handling | None | Stale facts persist until manually replaced |
| Consolidation | None | No automatic promotion of recurring patterns |
| Mid-session learning | Frozen snapshot pattern | New memories invisible until next session |
| Memory relationships | None | Facts stored in isolation, no linking |
| Forgetting model | Manual delete only | No principled decay; agent never forgets gracefully |

### Goal

Build a cognitive memory layer that:
1. Stores memories with vector embeddings for semantic retrieval
2. Scores recall using ACT-R-inspired activation (frequency × recency × importance)
3. Detects and resolves contradictions
4. Implements Ebbinghaus-inspired forgetting with rehearsal strengthening
5. Forms Hebbian associations between co-recalled memories
6. Consolidates memories across layers (working → core → archive)
7. Scopes memories to projects/topics for precision
8. Ships with rigorous, reproducible benchmarks proving improvement over baseline
9. Finds optimal parameter settings through systematic grid search
10. Defers all architectural decisions (e.g., built-in vs Engram backend) to benchmark evidence

Honcho integration is an optional enhancement layer, not a requirement.

---

## 2. Prior Art Analysis

Four existing systems informed this design:

### PR #727 (0xbyt4) — Cognitive Memory System for hermes-agent
- SQLite + vector embeddings, composite scoring: `0.5×similarity + 0.3×recency + 0.2×importance`
- Cognitive encoding: auto-classifies memories by type, estimates importance
- Contradiction detection via embedding similarity
- Forgetting: `importance × 0.5^(days/half_life)`, prunes below 0.05
- **Strengths:** Built specifically for hermes-agent, clean module structure
- **Weaknesses:** Arbitrary scoring weights, fragile contradiction detection (embedding distance can't distinguish "similar and complementary" from "similar and contradictory"), simplistic decay model
- **Status:** 4,169 lines, unreviewed hackathon PR

### Engram-AI (tonitangpotato) — Neuroscience-grounded memory
- ACT-R activation: `base_level = ln(Σ tᵢ^(-d))` — proper frequency × recency model
- Hebbian learning: co-recalled memories auto-link (`ΔW = η × aᵢ × aⱼ`)
- Ebbinghaus forgetting curves with consolidation cycles
- Three-layer memory: working → core → archive
- Multiple embedding backends (sentence-transformers, ollama, openai, FTS5-only fallback)
- Available as pip package (`engramai`), also has Rust and TypeScript runtimes
- **Strengths:** More principled neuroscience models, production-tested (~230K recalls), zero inference cost, pip-installable
- **Weaknesses:** AGPL-3.0 license (copyleft), their hermes integration patches core files, no published benchmark results (LongMemEval/LoCoMo/BEAM)
- **Status:** PyPI package, hermes-engram wrapper is ~250 lines

### Honcho v3.0.3 (Plastic Labs) — AI-native reasoning layer
- External service with background reasoning using fine-tuned models
- Formal logic: deduction, induction, abduction over conversation history
- Builds evolving "representations" of users/agents
- Already partially integrated in hermes-agent (`honcho_integration/`)
- Published benchmarks: 89.9% LoCoMo, 90.4% LongMemEval S, 0.630 BEAM 100K
- **Strengths:** Cross-session reasoning, published SOTA numbers, token-efficient
- **Weaknesses:** External dependency, inference cost, privacy unless self-hosted
- **Status:** v3.0.3, funded company, integration already exists in hermes-agent

### A-MEM (AGIResearch) — Zettelkasten-inspired memory
- Each memory becomes a structured note with tags, context, connections
- ChromaDB for vector storage
- Agent-driven organization and memory evolution
- **Strengths:** Beautiful linking/evolution concepts
- **Weaknesses:** Stale repo (3mo), academic prototype, no hermes integration
- **Status:** Research project, 900 GitHub stars

### What We Take From Each

| Concept | Source | Why |
|---------|--------|-----|
| ACT-R activation scoring | Engram | More principled than arbitrary weighted sum |
| Hebbian co-activation links | Engram | Captures usage patterns, not just similarity |
| Ebbinghaus forgetting + consolidation | Engram | Three-layer model (working→core→archive) is structurally sound |
| Standalone module pattern | PR #727 | Don't patch core files, avoid merge conflicts |
| Contradiction detection | PR #727 + our improvement | High-similarity trigger + LLM judge |
| Memory scope | Original design | project/topic scoping for precision recall |
| Knowledge graph links | A-MEM | Bidirectional links with type labels |
| Honcho as optional reasoning layer | Honcho | Cross-session inference when configured |
| Embedding fallback chain | Engram | sentence-transformers → ollama → openai → FTS5 |

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    SYSTEM PROMPT INJECTION                    │
│  prompt_builder.py queries cognitive memory at context-build  │
│  time. Top-K recalled memories + curated snapshot injected.  │
└────────────────────────────┬─────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐  ┌────────▼────────┐  ┌────────▼────────┐
│  LAYER 1      │  │  LAYER 2        │  │  LAYER 3        │
│  Curated      │  │  Cognitive      │  │  Honcho         │
│  (existing)   │  │  (new)          │  │  (optional)     │
│               │  │                 │  │                 │
│  MEMORY.md    │  │  SQLite +       │  │  External API   │
│  USER.md      │  │  Embeddings     │  │  Reasoning      │
│  Flat files   │  │  Local, fast    │  │  Cross-session  │
│  Always works │  │  Zero ext deps  │  │  inference      │
└───────────────┘  └────────┬────────┘  └─────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼────┐ ┌─────▼────┐ ┌──────▼─────┐
        │ Store    │ │ Recall   │ │ Maintain   │
        │ encode   │ │ activate │ │ consolidate│
        │ link     │ │ spread   │ │ decay      │
        │ classify │ │ rank     │ │ prune      │
        └──────────┘ └──────────┘ └────────────┘
```

### Key Principle: Standalone Module

The cognitive memory system lives in `cognitive_memory/` as a self-contained
package. It does NOT patch `memory_tool.py`, `prompt_builder.py`, or
`run_agent.py` directly. Instead, it provides:

- A `CognitiveMemoryStore` class with clean API
- A `cognitive_memory_tool` that registers alongside (not replacing) the existing memory tool
- A thin integration hook that prompt_builder calls IF the module is present

This means:
- If the PR never merges, the module still works as a local install
- Upstream hermes-agent updates won't cause merge conflicts
- Users who don't want it aren't affected

### Backend Options

The cognitive memory module defines a storage interface. Backends:

1. **Built-in (default)** — Our own SQLite + embeddings implementation. Zero external deps beyond numpy.
2. **Engram backend (optional)** — If `engramai` is installed, delegate storage/recall/consolidation to Engram's engine. We wrap it with our scope/contradiction/tool-integration layer.
3. **Honcho sidecar (optional)** — If Honcho is configured, augment local recall with reasoned conclusions.

**Which backend to recommend is deferred to benchmark results.** We build both,
test both, and let the numbers decide.

---

## 4. Data Model

### 4.1 Memory Entry

```sql
CREATE TABLE memories (
    id              TEXT PRIMARY KEY,     -- uuid4
    content         TEXT NOT NULL,        -- the actual memory text
    category        TEXT NOT NULL,        -- factual | preference | procedural | environment | episodic | semantic | causal
    scope           TEXT DEFAULT 'global',-- global | project:<name> | topic:<name>
    importance      REAL NOT NULL,        -- 0.0-1.0, set at encoding time
    embedding       BLOB,                 -- float32 vector, model-dependent dimensionality
    created_at      REAL NOT NULL,        -- unix timestamp
    last_accessed   REAL NOT NULL,        -- updated on every recall hit
    access_count    INTEGER DEFAULT 0,    -- total recall hits (for ACT-R base level)
    access_times    TEXT,                 -- JSON array of unix timestamps (for ACT-R trace sum)
    layer           TEXT DEFAULT 'working', -- working | core | archive
    superseded_by   TEXT,                 -- FK to memories.id (null = active)
    source          TEXT DEFAULT 'agent', -- agent | user | consolidation | honcho
    pinned          INTEGER DEFAULT 0,   -- 1 = protected from decay
    metadata        TEXT                  -- JSON blob for extensibility
);

CREATE INDEX idx_memories_scope ON memories(scope);
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_layer ON memories(layer);
CREATE INDEX idx_memories_active ON memories(superseded_by) WHERE superseded_by IS NULL;
```

### 4.2 Memory Links (Hebbian + Semantic)

```sql
CREATE TABLE memory_links (
    source_id   TEXT NOT NULL REFERENCES memories(id),
    target_id   TEXT NOT NULL REFERENCES memories(id),
    weight      REAL NOT NULL DEFAULT 0.1,  -- Hebbian weight, strengthened by co-activation
    link_type   TEXT DEFAULT 'hebbian',     -- hebbian | semantic | contradiction | causal
    created_at  REAL NOT NULL,
    last_coactivated REAL,                  -- last time both were recalled together
    PRIMARY KEY (source_id, target_id)
);
```

### 4.3 Embedding Storage

Embeddings stored as raw BLOB in SQLite. On recall, load all active embeddings
into memory and compute cosine similarity in NumPy. For typical agent usage
(hundreds to low thousands of memories), this is fast enough — no vector DB needed.

**Embedding fallback chain** (inspired by Engram):
1. `sentence-transformers` — `all-MiniLM-L6-v2` (384d, ~80MB, <10ms, free, local)
2. `ollama` — any embedding model the user has running
3. `openai` — `text-embedding-3-small` (~$0.0001/query)
4. **TF-IDF fallback** — sklearn or custom. Works without any ML deps. Worse quality but functional.

Configured via `COGNITIVE_MEMORY_EMBEDDING` env var (default: `auto` = try in order above).

---

## 5. Core Operations

### 5.1 Encoding (Store)

When a memory is added:

```
1. Classify category using heuristic patterns (fast, no LLM call):
   - "user prefers..." / "I like..." → preference
   - "to do X, run..." / "the command is..." → procedural
   - "the server is at..." / "version is..." → environment
   - "we discussed..." / "last time..." → episodic
   - "X causes Y" / "because of X" → causal
   - default → factual

2. Estimate importance (0.0-1.0) using heuristics:
   - Contains correction/update → 0.85
   - Contains preference → 0.75
   - Contains technical detail / procedural → 0.65
   - General observation → 0.45
   - (Optionally refined by cheap LLM call if available)

3. Generate embedding vector (via fallback chain)

4. Contradiction check:
   - Find top-3 most similar active memories (cosine > CONTRADICTION_THRESHOLD)
   - If any found AND same category:
     - If cheap LLM available: send pair with prompt
       "Do these statements contradict? Which is more recent/authoritative?"
     - If no LLM: flag as potential contradiction (don't auto-supersede)
     - If contradiction confirmed: set old.superseded_by = new.id

5. Semantic link creation (A-MEM inspired):
   - Find top-K memories above SEMANTIC_LINK_THRESHOLD
   - Exclude if same content (dedup)
   - Only link if similarity > threshold AND (shared category OR similarity > HIGH_LINK_THRESHOLD)
   - Create bidirectional links with link_type='semantic'
   - Max LINKS_PER_MEMORY outgoing links

6. Set layer = 'working', write to SQLite
```

### 5.2 Recall (Retrieve)

ACT-R inspired activation scoring:

```
1. Encode the query into an embedding vector

2. Load all active embeddings (cached in memory between calls within a session)

3. For each active memory, compute activation:

   ACTIVATION = BASE_LEVEL + SPREADING + IMPORTANCE_BOOST + SCOPE_BOOST

   Where:
   
   BASE_LEVEL (ACT-R):
     = ln(Σ tᵢ^(-d))
     tᵢ = seconds since each access (from access_times array)
     d  = DECAY_PARAMETER (default 0.5, standard ACT-R value)
     This naturally handles both recency and frequency.
     More accesses = higher sum. Recent accesses = larger terms.
   
   SPREADING (semantic + Hebbian):
     = W_SEMANTIC × cosine_similarity(query_embedding, memory_embedding)
     + Σ (hebbian_weight × linked_memory_activation)  [one hop]
     Hebbian spreading means if memory A is linked to memory B,
     and B is highly activated by the query, A gets a boost too.
   
   IMPORTANCE_BOOST:
     = W_IMPORTANCE × importance
   
   SCOPE_BOOST:
     = SCOPE_MULTIPLIER if memory.scope matches current scope, else 0

4. Rank by ACTIVATION, return top-K

5. Update recalled memories:
   - Append current timestamp to access_times
   - Increment access_count
   - Update last_accessed

6. Hebbian link strengthening:
   - For every pair of memories recalled together in this query:
     - If link exists: weight += HEBBIAN_LEARNING_RATE × activation_product
     - If no link AND both highly activated: create new hebbian link
   - This is the "neurons that fire together wire together" mechanism
```

### 5.3 Consolidation

Runs at session start (or on explicit `consolidate` action):

```
Three-layer model inspired by Engram:

WORKING MEMORY (recent, frequently accessed):
  - New memories start here
  - High activation threshold to stay
  
CORE MEMORY (important, well-established):
  - Promoted from working when: access_count >= CORE_PROMOTION_COUNT
    AND importance >= CORE_PROMOTION_IMPORTANCE
  - Slower decay rate than working
  
ARCHIVE (low-activity, but retained):
  - Demoted from core when: activation drops below ARCHIVE_THRESHOLD
  - Not injected into system prompt by default
  - Still searchable via explicit recall
  - Eventually pruned if activation hits zero

Consolidation steps:
1. Compute activation for all memories
2. Promote working → core if criteria met
3. Demote core → archive if activation dropped
4. Prune archive memories below PRUNE_THRESHOLD
5. Decay Hebbian link weights by LINK_DECAY_RATE
6. Prune links below LINK_PRUNE_THRESHOLD
7. (Optional, LLM-assisted) Merge clusters of similar memories
   (>0.85 cosine, cluster_size >= 3) into consolidated summaries
```

### 5.4 Contradiction Resolution

Two-stage approach:

```
Stage 1 (cheap, always runs):
  - On every store(), find memories with cosine > CONTRADICTION_THRESHOLD
  - Same category required
  - If found, mark as "potential_contradiction" in metadata

Stage 2 (LLM judge, optional):
  - If a cheap LLM is available (Haiku-class):
    - Send both memories with system prompt:
      "Memory A: {old}. Memory B: {new}. Do these contradict each other?
       If yes, which should be kept as the current truth? Reply JSON:
       {contradicts: bool, keep: 'A'|'B', reason: string}"
    - ~200 tokens per check, only triggers on high-similarity pairs
  - If no LLM available:
    - Keep both, flag for user review
    - User can resolve via `memory(action='resolve', memory_id=...)`
```

---

## 6. Tunable Parameters

These are the parameters we'll grid-search in benchmarking:

### 6.1 ACT-R Parameters

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Decay parameter | d | 0.5 | 0.3–0.8 | ACT-R standard decay rate in `tᵢ^(-d)` |
| Semantic weight | W_SEMANTIC | 0.4 | 0.2–0.7 | Weight of cosine similarity in activation |
| Importance weight | W_IMPORTANCE | 0.2 | 0.1–0.4 | Weight of importance score in activation |
| Hebbian learning rate | η | 0.05 | 0.01–0.15 | How fast co-activation strengthens links |

### 6.2 Thresholds

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| CONTRADICTION_THRESHOLD | 0.82 | 0.75–0.90 | Cosine above which to check for contradictions |
| SEMANTIC_LINK_THRESHOLD | 0.70 | 0.60–0.85 | Minimum cosine to create a semantic link |
| HIGH_LINK_THRESHOLD | 0.90 | 0.85–0.95 | Cosine above which to link even across categories |
| LINKS_PER_MEMORY | 5 | 3–10 | Maximum outgoing links per memory |
| PRUNE_THRESHOLD | 0.01 | 0.005–0.05 | Activation below which to prune from archive |
| SCOPE_MULTIPLIER | 1.5 | 1.0–3.0 | Boost for in-scope memories |
| TOP_K | 10 | 5–20 | Number of memories returned per recall |

### 6.3 Consolidation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| CORE_PROMOTION_COUNT | 3 | 2–10 | Minimum accesses to promote working → core |
| CORE_PROMOTION_IMPORTANCE | 0.5 | 0.3–0.7 | Minimum importance to promote |
| ARCHIVE_THRESHOLD | 0.1 | 0.05–0.3 | Activation below which core → archive |
| LINK_DECAY_RATE | 0.95 | 0.85–0.99 | Hebbian weight multiplier per consolidation |
| LINK_PRUNE_THRESHOLD | 0.01 | 0.005–0.05 | Link weight below which to remove |

### 6.4 Profiles (presets for different use cases)

| Profile | d | W_SEM | W_IMP | η | Notes |
|---------|---|-------|-------|---|-------|
| developer | 0.5 | 0.45 | 0.15 | 0.06 | Recent project context matters; fast link formation |
| researcher | 0.4 | 0.55 | 0.25 | 0.03 | Older memories stay relevant; importance matters |
| personal | 0.5 | 0.35 | 0.35 | 0.05 | Preferences/corrections heavily weighted |
| balanced | 0.5 | 0.40 | 0.20 | 0.05 | Default / general purpose |

**Profile defaults are initial guesses. Final values determined by benchmark grid search.**

---

## 7. Integration Points

### 7.1 With Existing Memory Tool

The cognitive memory tool runs ALONGSIDE `memory_tool.py`, not replacing it:

- `memory_tool.py` continues to manage the curated flat files (MEMORY.md / USER.md)
- `cognitive_memory_tool` manages the SQLite-backed cognitive store
- `prompt_builder.py` gets a small addition: if cognitive_memory is available,
  call `recall(current_context_summary)` and append top-K results to system prompt
  after the curated memory blocks

The curated memory remains the "always-on, guaranteed" layer.
Cognitive memory is the "richer, searchable" layer.

### 7.2 With Session Search

Session search (FTS5 + Gemini Flash summarization) continues unchanged.
They solve different problems:

- Session search: "what happened in our conversation about X?" (transcript recall)
- Cognitive memory: "what do I know about X?" (extracted facts/preferences)

### 7.3 With Honcho (Optional)

If Honcho is configured (`HONCHO_APP_ID` + `HONCHO_API_KEY` set):

1. After cognitive memory encodes a fact, also send it to Honcho as a message
2. Before cognitive recall, also call `honcho.get_context(query)` for
   reasoned conclusions
3. Honcho's "dream processing" can feed back consolidated insights as
   new cognitive memories with source='honcho'

If Honcho is NOT configured, everything works locally. Zero degradation.

### 7.4 With Engram Backend (Optional)

If `engramai` is installed, the storage layer can delegate to Engram's engine.
Our module adds on top: scope filtering, contradiction detection with LLM judge,
tool registration, prompt injection, and the benchmark framework.

**Whether to recommend the Engram backend or our built-in as default is
determined entirely by benchmark results (see Section 8.5).**

---

## 8. Benchmark Plan

### 8.1 Design Principles

This benchmark suite is designed to produce **publishable-quality evidence**.
Every claim about improvement must be backed by:

- **Multiple runs** with variance reporting (mean ± std)
- **Ablation testing** to attribute improvement to specific features
- **Cross-embedding comparison** to isolate model effects from algorithm effects
- **Adversarial cases** alongside happy-path scenarios
- **Established benchmark protocols** (LongMemEval, LoCoMo, BEAM) for external comparability
- **Token cost as a first-class metric** alongside accuracy
- **Temporal stability testing** to ensure the system doesn't degrade over time

### 8.2 Statistical Protocol

**Every configuration is tested N=5 times** with different random seeds that
control:
- Memory insertion order (shuffled)
- Query presentation order (shuffled)
- Hebbian link formation order (path-dependent)
- Consolidation timing jitter

**Reported metrics:**
- Mean score ± standard deviation per category
- Overall mean ± std
- 95% confidence intervals where sample size permits
- Statistical significance testing (paired t-test or Wilcoxon signed-rank)
  between baseline and each configuration

**A result is only reported as an improvement if p < 0.05.**

### 8.3 Evaluation Protocol

**LLM Judge:** All answer correctness evaluations use the same judge model
and prompt templates throughout the entire benchmark suite for consistency.

- **Primary judge:** Claude Haiku 4.5 (cheap, fast, reliable)
- **Validation judge:** GPT-4o on a 10% random subset to check for judge bias
- **Judge agreement rate** is reported as a meta-metric

**Per-question-type prompts** follow LongMemEval's established templates:
- Factual recall: "Does the response contain the correct answer? yes/no"
- Preference: "Does the response satisfy the rubric? yes/no"
- Temporal: "Is the temporal information correct? (off-by-one tolerance) yes/no"
- Knowledge update: "Does the response reflect the updated information? yes/no"
- Abstention: "Does the model correctly identify the question as unanswerable? yes/no"
- Contradiction: "Does the response reflect the current/corrected state? yes/no"

### 8.4 Test Suites

#### Suite A: Custom Eval (fast iteration, ~150K tokens per run)

Seven test categories designed for agent memory use cases.
**160 core scenarios + 40 adversarial = 200 total.**

**A1 — Semantic Recall (50 questions)**
- Store 50 facts using exact wording
- Query using paraphrased questions (different words, same meaning)
- Score: % of facts correctly retrieved in top-K
- Difficulty tiers:
  - Easy (15): simple synonym substitution ("PostgreSQL" → "Postgres database")
  - Medium (20): structural rephrasing ("project uses X" → "what does the project use?")
  - Hard (15): abstract/indirect ("the persistent store" when stored as "PostgreSQL 15")
- Tests: embedding quality, semantic generalization

**A2 — Contradiction Resolution (20 scenarios)**
- Store fact A, then contradicting fact B at a later timestamp
- Query should return B (the update), not A
- Score: % where latest/correct fact is returned AND old fact marked superseded
- Includes both:
  - Clear contradictions ("uses PostgreSQL" → "migrated to MySQL")
  - Subtle updates ("Python 3.10" → "upgraded ML service to 3.12" — partial update)

**A3 — Decay & Rehearsal (30 scenarios)**
- Store 30 memories with varying importance levels
- Simulate time passing (configurable: days to weeks)
- Access some memories during the simulated period (rehearsal pattern)
- Query all at "end of time"
- Score matrix:
  - Rehearsed + important: should be retained (target: >95%)
  - Rehearsed + unimportant: should be retained (rehearsal overrides importance)
  - Not rehearsed + important: should be retained but lower activation
  - Not rehearsed + unimportant: should be pruned (target: >80% pruned)

**A4 — Hebbian Link Following (20 questions)**
- Store related facts that don't share keywords
- Simulate co-retrieval patterns to form Hebbian links
- Query via indirect path (requires link traversal)
- Score: does Hebbian spreading surface the chain?
- Example chain:
  - Store: "Alice leads the backend team"
  - Store: "The backend uses FastAPI and SQLAlchemy"
  - Store: "SQLAlchemy connects to our PostgreSQL cluster"
  - Co-retrieve "Alice" and "backend" 3 times (forms link)
  - Co-retrieve "backend" and "FastAPI" 3 times (forms link)
  - Query: "What database technology is relevant to Alice's work?"
  - Expected: PostgreSQL surfaces via link chain

**A5 — Preference Tracking (20 scenarios)**
- Across simulated sessions, express preferences that evolve:
  - "I prefer dark mode"
  - "Actually, light mode is easier on my eyes"
  - "Dark mode with warm colors is the sweet spot"
- Query: "What's my display preference?"
- Score: does it return the latest evolved preference?
- Includes:
  - Simple override (A → B)
  - Refinement (A → A with nuance)
  - Reversal (A → B → A again)

**A6 — Scope Filtering (20 scenarios)**
- Store memories in different scopes (project:aegis, project:neurovision, global)
- Query with scope=project:aegis
- Score: are in-scope results ranked higher? Are out-of-scope deprioritized?
- Includes cross-scope facts that should appear regardless (global scope)

**A7 — Adversarial / Edge Cases (40 scenarios)**

Designed to break naive implementations:

- **Near-contradictions that aren't contradictions (10):**
  "Server is in us-east-1" + "Staging server is in eu-west-1" — different scopes, not a contradiction.
  "Project uses Python 3.10" + "CI runs Python 3.11" — different contexts.
  Score: false positive rate for contradiction detection (target: <10%)

- **Partial updates (10):**
  "We use PostgreSQL for everything" → "We migrated analytics to ClickHouse"
  The first fact is still partially true. Score: does the system handle partial supersession?

- **High-similarity different facts (10):**
  "Project A uses PostgreSQL" + "Project B uses PostgreSQL" — should NOT be merged/linked as contradictions.
  "Alice prefers dark mode" + "Bob prefers dark mode" — different subjects.
  Score: false positive rate for merging/contradiction (target: <5%)

- **Ambiguous/noisy input (10):**
  Sarcastic preferences, hypothetical statements ("if we were to use Rust..."),
  negations ("we decided NOT to use MongoDB"). Score: correct classification rate.

#### Suite B: Established Benchmarks (external comparability)

Three established benchmarks, used with their published evaluation protocols
so our numbers are directly comparable to published results.

**B1 — LongMemEval S (500 questions, 6 categories, ~115K token conversations)**

| Category | # Questions | What It Measures |
|----------|------------|-----------------|
| Single-session (user) | ~80 | Recall facts user stated |
| Single-session (assistant) | ~80 | Recall facts agent stated |
| Single-session (preference) | ~80 | Track user preferences |
| Temporal reasoning | ~80 | "When did we discuss X?" / time math |
| Knowledge update | ~80 | Facts that changed over time |
| Multi-session | ~80 | Facts spanning multiple conversations |
| Abstention | ~20 | Correctly refusing unanswerable questions |

**Evaluation:** Use LongMemEval's exact `evaluate_qa.py` protocol.
Per-question-type prompt templates, LLM judge (yes/no), per-category accuracy.

**Execution plan:**
- Phase 1: 100-question stratified subset (20 per non-abstention category) for fast iteration
- Phase 2: Full 500 questions for final numbers
- Compare against published baselines:
  - Claude Haiku 4.5 baseline: 62.6%
  - Honcho: 90.4%
  - Our target: meaningful improvement over baseline (exact target set after baseline measurement)

**B2 — LoCoMo (1,540 questions, 5 categories)**

| Category | # Questions | What It Measures |
|----------|------------|-----------------|
| Single-hop | ~400 | Direct fact recall |
| Multi-hop | ~400 | Reasoning across multiple facts |
| Commonsense | ~300 | Inference requiring world knowledge |
| Temporal | ~240 | Time-related reasoning |
| Open-domain | ~200 | General knowledge grounding |

**Evaluation:** Standard LoCoMo protocol with LLM judge.

**Published baselines:**
- Claude Haiku 4.5: 75.6%
- Honcho: 89.9%

**B3 — BEAM 100K (400 questions, 10 abilities, 20 conversations)**

The most challenging benchmark. Tests abilities our system specifically targets:

| Ability | What It Measures | Our Feature |
|---------|-----------------|-------------|
| Preference following | Track user preferences | Preference category + decay |
| Instruction adherence | Follow user instructions | Procedural category |
| Information extraction | Extract specific facts | Semantic recall |
| Contradiction resolution | Handle updated info | Contradiction detection |
| Temporal reasoning | Time-aware queries | Temporal metadata |
| Multi-session reasoning | Cross-session facts | Consolidation |
| Event ordering | Sequence awareness | access_times timestamps |
| Summarization | Compress information | Consolidation merging |
| Knowledge update | Superseded facts | Superseded_by field |
| Abstention | Know what you don't know | Activation thresholds |

**Evaluation:** BEAM's published rubric-based scoring (not just pass/fail —
graded 0.0-1.0 per question). Reports overall score and per-ability breakdown.

**Published baselines:**
- Claude Haiku 4.5: 0.529
- Honcho: 0.630

#### Suite C: Parameter Optimization

**C1 — Sensitivity Analysis (one-at-a-time)**

For each parameter in the grid, sweep its range while holding all others
at profile defaults. Run Suite A (full 200 questions, N=3 runs each).

```python
SENSITIVITY_GRID = {
    "d":                        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "W_SEMANTIC":               [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "W_IMPORTANCE":             [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    "HEBBIAN_LEARNING_RATE":    [0.01, 0.03, 0.05, 0.08, 0.12, 0.15],
    "CONTRADICTION_THRESHOLD":  [0.75, 0.78, 0.80, 0.82, 0.85, 0.90],
    "SEMANTIC_LINK_THRESHOLD":  [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "TOP_K":                    [3, 5, 7, 10, 15, 20],
    "CORE_PROMOTION_COUNT":     [1, 2, 3, 5, 8, 10],
}
```

~48 configurations × N=3 × Suite A = ~144 runs of Suite A.
At ~150K tokens per run = ~21.6M tokens total ≈ $5-10 at Haiku pricing.

Output: **sensitivity curves** — score vs parameter value, per category.
Identifies which parameters actually matter (high sensitivity) vs which
are safe to leave at defaults (low sensitivity).

**C2 — Focused Grid Search (multi-parameter)**

Take the 3-4 most sensitive parameters from C1. Build a focused grid:

```python
# Example after C1 identifies W_SEMANTIC, d, and TOP_K as most sensitive
FOCUSED_GRID = {
    "W_SEMANTIC": [best-1, best, best+1],  # 3 values around optimum
    "d":          [best-1, best, best+1],
    "TOP_K":      [best-1, best, best+1],
}
# 27 combinations × N=3 × Suite A = 81 runs
```

Output: optimal configuration per profile, with confidence intervals.

**C3 — Final Validation**

Run the top-3 configurations from C2 on:
- Full Suite A (N=5, for tight confidence intervals)
- Suite B1 100-question subset (N=3)
- Suite B3 BEAM 100K (N=1, expensive)

This is the definitive result set.

#### Suite D: Ablation Matrix

Test every feature in isolation to attribute improvement:

| Configuration | Features Enabled |
|--------------|-----------------|
| D0: baseline | Flat files + FTS5 (current system) |
| D1: embeddings only | Semantic search, no ACT-R, no Hebbian |
| D2: + ACT-R | Semantic + ACT-R activation (no Hebbian) |
| D3: + Hebbian | Semantic + Hebbian links (no ACT-R) |
| D4: + ACT-R + Hebbian | Full scoring model |
| D5: + contradiction (embedding) | D4 + contradiction detection (embedding only) |
| D6: + contradiction (LLM) | D4 + contradiction detection (+ LLM judge) |
| D7: + consolidation | D4 + three-layer consolidation |
| D8: + scope | D4 + scope filtering |
| D9: full system | All features |
| D10: Engram backend | Same features via engramai |

Run each on Suite A (N=3). Report per-category improvement delta vs baseline.

**Key questions ablation answers:**
- Does ACT-R actually beat a simple weighted sum? (D2 vs D1)
- Do Hebbian links help, or just add noise? (D3 vs D1, D4 vs D2)
- Is LLM contradiction detection worth the token cost? (D6 vs D5)
- Does consolidation improve or hurt? (D7 vs D4)
- Is our built-in competitive with Engram? (D9 vs D10)

#### Suite E: Cross-Embedding Comparison

The embedding model choice could dominate all other effects.
Test at least three models across Suite A (N=3):

| Model | Dimensions | Size | Cost |
|-------|-----------|------|------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Free (local) |
| `all-mpnet-base-v2` | 768 | 420MB | Free (local) |
| `text-embedding-3-small` (OpenAI) | 1536 | API | ~$0.02/1M tokens |
| TF-IDF (no ML) | varies | 0 | Free |

Run each on Suite A with the full system (D9 config).

**Key question:** Does the embedding model produce a larger accuracy swing
than the algorithm choice (ACT-R vs weighted sum, Hebbian vs none)?
If embedding choice produces 20% swing and ACT-R produces 2%, that's
critical information for users.

#### Suite F: Temporal Stability Test

Does the system degrade as memories accumulate?

```
1. Store 200 memories (batch 1). Run Suite A queries. Record scores.
2. Store 200 more memories (batch 2). Run Suite A queries AGAIN on batch 1 facts.
3. Store 200 more (batch 3). Run batch 1 queries again.
4. Store 200 more (batch 4). Run batch 1 queries again.
5. Run consolidation. Run batch 1 queries again.
```

Score at each stage: does accuracy on batch 1 queries degrade as memory fills?
Does consolidation recover performance?

This tests:
- Embedding search degradation at scale (200 vs 400 vs 600 vs 800 memories)
- Whether consolidation/pruning actually helps or just deletes useful memories
- Whether ACT-R decay correctly deprioritizes old unrehearsed memories
  while preserving old rehearsed ones

**800 memories is a realistic 6-12 month usage scenario for an active agent.**

### 8.5 Baseline Measurements

Before building anything, establish baselines on Suite A + Suite B (100-question subset):

| System | Suite A | Suite B1 | Suite B2 | Suite B3 | Tokens | Notes |
|--------|---------|----------|----------|----------|--------|-------|
| Current (flat files + FTS5) | ? | ? | ? | ? | ~0 | Starting point |
| Cognitive (built-in, defaults) | ? | ? | ? | ? | ? | Post-implementation |
| Cognitive (built-in, optimized) | ? | ? | ? | ? | ? | Post-grid-search |
| Cognitive (Engram backend) | ? | ? | ? | ? | ? | If engramai installed |
| Cognitive + Honcho | ? | ? | ? | ? | ? | Optional enhancement |

Published reference points:
| System | LoCoMo | LongMemEval S | BEAM 100K |
|--------|--------|--------------|-----------|
| Haiku 4.5 (no memory) | 75.6% | 62.6% | 0.529 |
| Honcho | 89.9% | 90.4% | 0.630 |

### 8.6 Token Cost as First-Class Metric

Every benchmark run records:

**Per-operation costs:**
| Operation | Input tokens | Output tokens | Wall time |
|-----------|-------------|--------------|-----------|
| Embedding generation | measured | measured | measured |
| Store (encode + classify) | measured | measured | measured |
| Recall (search + score) | measured | measured | measured |
| Contradiction check (LLM) | measured | measured | measured |
| Consolidation | measured | measured | measured |
| Judge evaluation | measured | measured | measured |

**Aggregate cost metrics:**
- Total tokens per benchmark run
- Cost per question (encode + recall + judge)
- Cost per percentage point of improvement over baseline
- **Accuracy vs cost Pareto curves** across all configurations
  (plotted as scatter: x=total cost, y=accuracy)

A system that gets 85% at $2 total vs 87% at $50 tells a different story
than raw accuracy alone. The Pareto curve identifies the sweet spot.

**Token overhead per session (real usage):**
- Estimated tokens for a typical session (10 store + 20 recall operations)
- Compared to current system (zero token overhead)
- Target: <50K tokens overhead per session at Haiku pricing (<$0.05)

### 8.7 Benchmark Output Format

All results are saved as structured JSON for reproducibility:

```json
{
  "benchmark_id": "20260320_suite_a_builtin_balanced",
  "timestamp": "2026-03-20T14:30:00Z",
  "system": {
    "backend": "builtin",
    "profile": "balanced",
    "embedding_model": "all-MiniLM-L6-v2",
    "parameters": { "d": 0.5, "W_SEMANTIC": 0.4, ... }
  },
  "runs": [
    {
      "seed": 42,
      "results_by_category": {
        "A1_semantic_recall": { "score": 0.82, "easy": 0.93, "medium": 0.80, "hard": 0.67 },
        "A2_contradiction": { "score": 0.85, "clear": 0.90, "subtle": 0.75 },
        ...
      },
      "overall_score": 0.79,
      "token_usage": { "total_input": 95000, "total_output": 12000 },
      "wall_time_seconds": 180
    },
    { "seed": 43, ... },
    ...
  ],
  "aggregate": {
    "mean": 0.79, "std": 0.02, "ci_95": [0.77, 0.81],
    "per_category_mean": { ... },
    "total_cost_usd": 0.15
  }
}
```

Human-readable summary tables are also generated (markdown) for the PR description.

---

## 9. File Structure

```
cognitive_memory/
├── __init__.py
├── store.py              # CognitiveMemoryStore — main class, orchestrates everything
├── backends/
│   ├── __init__.py
│   ├── base.py           # StorageBackend abstract interface
│   ├── builtin.py        # SQLite + embeddings built-in backend
│   └── engram_backend.py # Optional engramai delegation
├── embeddings.py         # Embedding generation (fallback chain)
├── encoding.py           # Category classification, importance estimation
├── activation.py         # ACT-R activation scoring
├── hebbian.py            # Hebbian co-activation links
├── contradiction.py      # Contradiction detection (embedding + optional LLM judge)
├── consolidation.py      # Three-layer consolidation pipeline
├── config.py             # Parameter dataclass with profile presets
├── migrations.py         # Schema versioning for the SQLite DB
└── honcho_bridge.py      # Optional Honcho integration (only loads if configured)

tools/
├── memory_tool.py            # UNCHANGED — existing curated memory
└── cognitive_memory_tool.py  # New tool registration

tests/
└── cognitive_memory/
    ├── test_store.py
    ├── test_activation.py
    ├── test_hebbian.py
    ├── test_contradiction.py
    ├── test_consolidation.py
    ├── test_encoding.py
    ├── test_embeddings.py
    └── test_config.py

benchmarks/
├── README.md                 # How to run, interpret results, reproduce
├── run_benchmarks.py         # Main CLI: run, compare, visualize
├── judge.py                  # LLM judge with per-question-type prompts
├── statistical.py            # Mean, std, CI, significance tests
├── suite_a/
│   ├── __init__.py
│   ├── runner.py             # Suite A orchestrator
│   ├── semantic_recall.py
│   ├── contradiction.py
│   ├── decay_rehearsal.py
│   ├── hebbian_linking.py
│   ├── preference_tracking.py
│   ├── scope_filtering.py
│   ├── adversarial.py
│   └── fixtures/
│       ├── semantic_recall.json    # 50 fact/query/gold-label triples
│       ├── contradictions.json     # 20 contradiction scenarios
│       ├── decay_scenarios.json    # 30 decay/rehearsal scenarios
│       ├── linking_scenarios.json  # 20 Hebbian chain scenarios
│       ├── preferences.json        # 20 preference evolution scenarios
│       ├── scoped_memories.json    # 20 scope filtering scenarios
│       └── adversarial.json        # 40 adversarial edge cases
├── suite_b/
│   ├── longmemeval_adapter.py      # Feeds LongMemEval data through our system
│   ├── locomo_adapter.py           # Feeds LoCoMo data through our system
│   ├── beam_adapter.py             # Feeds BEAM data through our system
│   ├── evaluate_qa.py              # LongMemEval's judge (adapted)
│   └── data/                       # Downloaded benchmark datasets
│       └── .gitkeep
├── suite_c/
│   ├── sensitivity.py              # One-at-a-time parameter sweep
│   ├── grid_search.py              # Focused multi-parameter grid
│   └── final_validation.py         # Top configs on full benchmarks
├── suite_d/
│   └── ablation.py                 # Feature ablation matrix
├── suite_e/
│   └── embedding_comparison.py     # Cross-embedding model test
├── suite_f/
│   └── temporal_stability.py       # Memory accumulation degradation test
├── baseline/
│   ├── flat_file_baseline.py       # Current MEMORY.md system
│   └── fts5_baseline.py            # Current session_search system
├── results/
│   └── .gitkeep
└── visualize/
    ├── pareto_curves.py            # Accuracy vs cost plots
    ├── sensitivity_plots.py        # Parameter sensitivity curves
    ├── ablation_chart.py           # Feature contribution waterfall
    └── stability_chart.py          # Temporal degradation plots
```

---

## 10. Implementation Plan

### Phase 1: Benchmark Framework + Baseline (2-3 sessions)

Build the benchmark framework FIRST so we have baseline numbers before
writing any cognitive memory code.

1. Write all Suite A fixtures (200 test scenarios as JSON)
2. Write Suite A runners (each category as a module)
3. Write `judge.py` (LLM evaluation with per-question-type prompts)
4. Write `statistical.py` (mean, std, CI, significance tests)
5. Write baseline adapters (flat file + FTS5)
6. Download LongMemEval, LoCoMo, BEAM datasets
7. Write Suite B adapters
8. Run Suite A against current system (N=5) → **baseline numbers**
9. Run Suite B (100-question subsets) against current system → **baseline numbers**
10. Write `run_benchmarks.py` CLI

### Phase 2: Core Implementation (2-3 sessions)

1. `config.py` — parameter dataclass with profiles
2. `embeddings.py` — fallback chain
3. `encoding.py` — category classification, importance estimation
4. `backends/builtin.py` — SQLite schema, basic CRUD
5. `activation.py` — ACT-R scoring
6. `store.py` — main orchestrator (encode, store, recall)
7. Unit tests for all core modules
8. Run Suite A (N=5) → **first improvement numbers**
9. Run Suite D ablation (D0-D2) → **embeddings + ACT-R contribution**

### Phase 3: Advanced Features (2-3 sessions)

1. `contradiction.py` — two-stage detection
2. `hebbian.py` — co-activation links, Hebbian learning
3. `consolidation.py` — three-layer pipeline
4. `backends/engram_backend.py` — optional Engram delegation
5. Unit tests for advanced features
6. Run Suite A (N=5) → **measure each feature's contribution**
7. Run full Suite D ablation (D0-D10) → **complete feature attribution**
8. Run Suite E cross-embedding comparison → **embedding model recommendation**

### Phase 4: Optimization (2-3 sessions)

1. Run Suite C1 sensitivity analysis → **identify sensitive parameters**
2. Run Suite C2 focused grid → **optimal configurations per profile**
3. Run Suite C3 final validation on full benchmarks
4. Run Suite F temporal stability → **degradation characteristics**
5. Generate all visualization plots (Pareto, sensitivity, ablation, stability)
6. Update `config.py` defaults to empirically optimal values
7. Write `benchmarks/results/` summary

### Phase 5: Integration + PR (1-2 sessions)

1. `cognitive_memory_tool.py` — tool registration
2. `prompt_builder` hook (thin, non-breaking)
3. `honcho_bridge.py` — optional Honcho sidecar
4. Integration tests
5. Documentation with benchmark results
6. Format PR description:
   - Before/after comparison tables
   - Ablation waterfall chart
   - Parameter sensitivity curves
   - Pareto accuracy-vs-cost curve
   - Temporal stability plot
   - Per-profile optimal configurations
7. Submit PR to NousResearch/hermes-agent

---

## 11. Dependencies

### Required (stdlib + lightweight)
- `sqlite3` (stdlib)
- `numpy` (already in hermes-agent deps)
- `json`, `uuid`, `math`, `logging`, `statistics` (stdlib)

### Recommended (for full embeddings)
- `sentence-transformers` — for `all-MiniLM-L6-v2` embeddings (free, local)
- Falls back to TF-IDF if not installed

### Optional
- `engramai` — battle-tested ACT-R/Hebbian engine as backend (AGPL-3.0)
- `honcho` Python SDK — for Honcho sidecar integration
- `matplotlib` — benchmark visualization only
- `scipy` — statistical significance tests only

No new hard dependencies. Graceful degradation:
- No sentence-transformers → TF-IDF embeddings (worse but functional)
- No LLM available → skip contradiction LLM judge, use embedding-only flagging
- No engramai → use built-in backend
- No Honcho → fully local operation

---

## 12. Open Questions

1. **Embedding model choice:** Deferred to Suite E benchmark results.

2. **Proactive encoding:** Should the agent auto-store important facts from
   conversations without being asked? Test with and without in ablation.

3. **Migration from existing memories:** Should we auto-import current
   MEMORY.md/USER.md entries into the cognitive store on first run?

4. **Access time array size:** ACT-R `access_times` grows unbounded. Cap at
   last N timestamps? Or use a summary statistic? Test impact in Suite F.

5. **Consolidation frequency:** Every session start? Every N messages?
   Test impact in Suite A with different frequencies.

6. **Engram recommendation:** Deferred entirely to Suite D (D9 vs D10) benchmark results.

7. **Judge model reliability:** Is Haiku a reliable judge? Validated by
   running GPT-4o on 10% subset and reporting agreement rate.

---

## 13. Success Criteria

The PR is ready to submit when:

**Accuracy (all results at p < 0.05 vs baseline):**
- [ ] Suite A overall score improves by ≥20% over baseline (mean ± std reported)
- [ ] Suite A semantic recall (A1) improves by ≥30% over baseline
- [ ] Suite A contradiction resolution (A2) achieves ≥80% accuracy
- [ ] Suite A Hebbian linking (A4) achieves ≥60% accuracy
- [ ] Suite A adversarial (A7) false positive rate <10% for contradictions
- [ ] Suite B1 (LongMemEval 100q subset) improves by ≥15% over baseline
- [ ] Suite B2 (LoCoMo subset) improves by ≥10% over baseline

**Optimization:**
- [ ] Grid search identifies optimal parameters with clear sensitivity curves
- [ ] Each profile preset is empirically validated on Suite A
- [ ] Ablation matrix shows which features contribute (with significance)

**Efficiency:**
- [ ] Token cost per session <50K tokens overhead
- [ ] Pareto curve shows our system on or near the efficiency frontier

**Robustness:**
- [ ] Suite F temporal stability: <5% accuracy degradation at 800 memories vs 200
- [ ] Suite E: best embedding model identified with clear margin
- [ ] All unit tests pass
- [ ] System degrades gracefully when optional deps are missing
- [ ] Judge agreement rate >90% (Haiku vs GPT-4o on 10% subset)

**Documentation:**
- [ ] PR description includes all visualization plots
- [ ] Benchmark reproduction instructions in benchmarks/README.md
- [ ] Parameter guide with per-profile recommendations backed by data
