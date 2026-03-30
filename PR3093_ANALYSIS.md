# PR #3093 Architecture Analysis & Integration Candidates

## Architecture Comparison

### Our Cognitive Memory Store
- **Scoring**: ACT-R activation (base-level decay + spreading activation)
- **Recall**: Embedding similarity (sentence-transformers) + Hebbian link spreading + BM25/RRF fusion
- **Contradiction**: Heuristic entity overlap + update-language gate + LLM fallback + embedding sim floor
- **Consolidation**: Working → Core → Archive layers, access-frequency-based promotion
- **Links**: Hebbian co-recall links, semantic links, keyword-overlap links
- **Search**: Embedding cosine similarity + activation scoring (no keyword/FTS)
- **Storage**: SQLite with embeddings, per-memory metadata
- **Pressure**: None (no limit on fact count)

### PR #3093 Structured Memory
- **Scoring**: FTS5 rank (BM25-based)
- **Recall**: FTS5 keyword matching only (no embeddings, no semantic search)
- **Contradiction**: Simple target+type+scope match → supersede (no semantic analysis)
- **Consolidation**: Gauge pressure tiers (70% merge, 85% archive cold scopes, 95% LLM synthesis)
- **Links**: None (flat fact store, no linking)
- **Search**: FTS5 full-text search (sub-millisecond)
- **Storage**: SQLite with typed facts (C/D/V/?/✓/~), scoped, with FTS5 virtual table
- **Pressure**: Yes — character-based gauge (10,000 char budget, tiered responses)

## Key Differences

| Feature | Our System | PR #3093 |
|---------|-----------|----------|
| Recall method | Semantic (embeddings + ACT-R) | Keyword (FTS5) |
| Fact typing | category (factual/preference/etc) | MEMORY_SPEC (C/D/V/?/✓/~) |
| Scope model | Simple string prefix matching | Full lifecycle (active→cold→closed) with auto-cooling |
| Pressure mgmt | None | 4-tier gauge with automatic actions |
| Contradiction | Deep (embedding + entity + LLM) | Shallow (same target+type = supersede) |
| Memory links | Hebbian + semantic + keyword | None |
| System prompt injection | None (tool-call based) | Automatic gauge + hot facts injection |
| Turn-based ops | None | Scope tick on every user message |

## Integration Candidates (ranked by value)

### 1. FTS5 Keyword Search (HIGH VALUE)
**What**: Add FTS5 virtual table to our SQLite backend for sub-millisecond keyword search.
**Why**: Our current recall is 100% embedding-based. BM25/RRF fusion already exists but computes BM25 in-memory. FTS5 gives us persistent, indexed keyword search that survives restarts and handles exact matches (numbers, identifiers) better than embeddings.
**Impact**: Could improve Suite D (AD_07 port number case) and exact-match recall scenarios.
**Risk**: Low — additive, doesn't replace existing embedding recall.

### 2. Gauge / Pressure Management (HIGH VALUE)
**What**: Character-based budget with tiered automatic responses.
**Why**: Our system has no memory pressure management. In production, unbounded memory growth will degrade recall (more noise facts) and consume embedding compute.
**Impact**: Production readiness. Prevents degradation at scale.
**Risk**: Low — additive feature, doesn't change scoring.

### 3. MEMORY_SPEC Typed Facts (MEDIUM VALUE)
**What**: Classify facts as Constraints, Decisions, Values, Unknowns, Done, Obsolete.
**Why**: Type information adds a strong discrimination signal. A "Decision" about auth and a "Value" about auth are different things. Could improve contradiction detection (same target+type is a stronger signal than same topic).
**Impact**: Could improve contradiction precision and recall organization.
**Risk**: Medium — requires schema changes, migration path.

### 4. Scope Lifecycle (MEDIUM VALUE)
**What**: Full active→cold→closed lifecycle with auto-cooling (silence detection, closing signals).
**Why**: Our scope model is simple prefix matching. Auto-cooling would help in long sessions where earlier project contexts should naturally fade without being deleted.
**Impact**: Production behavior improvement, reduced noise in long sessions.
**Risk**: Medium — behavioral change, needs careful testing.

### 5. System Prompt Injection (MEDIUM VALUE)
**What**: Inject gauge + hot facts into system prompt at startup (no tool call consumed).
**Why**: Currently recalling memory costs a tool call. Auto-injection makes context available immediately.
**Impact**: Faster first-turn responses, fewer tool calls.
**Risk**: Low — additive.

### 6. Abbreviation/Compression (LOW VALUE)
**What**: Dictionary-based text compression for stored facts.
**Why**: Reduces storage footprint. Only valuable under pressure management.
**Impact**: Marginal unless combined with gauge system.
**Risk**: Low but adds complexity.

### 7. Optimize (MEMORY.md migration) (LOW VALUE for us)
**What**: Extract C/D/V facts from flat files into structured store.
**Why**: We already use structured SQLite storage. This is for migrating from flat-file memory.
**Impact**: Not applicable to our architecture.

## Recommended Integration Order

1. **FTS5 keyword search** — highest benchmark impact potential, lowest risk
2. **Gauge / pressure management** — production readiness
3. **Typed facts (MEMORY_SPEC)** — improved contradiction signal
4. **Scope lifecycle** — production behavior

Items 1-2 can be done independently. Items 3-4 are more invasive and should be separate phases.
