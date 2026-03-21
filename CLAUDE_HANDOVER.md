# Memory Upgrade — Handover for Host-Side Work

**Date:** 2026-03-21 Session 9
**Prepared by:** Moonsong (running inside Docker container)
**For:** Claude Code running on host macOS (outside container)

---

## Context

We've been doing a multi-session cognitive memory upgrade for hermes-agent. Phase 1 (LLM judge, sentence-transformers, LongMemEval, Honcho fixes) is complete. Session 8 improved Suite B–E. Session 9 (this one) verified the aegis proxy plumbing is finally working end-to-end from inside containers.

**The reason this needs to run on the host:** The full benchmark suite with LLM judge hits timeouts/OOM inside Docker containers. The sentence-transformers model load takes ~15-20s, and each LLM judge call adds ~1s. Suite A alone has 200 scenarios across 5 categories — that's 200+ LLM calls just for judging, plus contradiction detection calls. The container's 10GB RAM and network latency through the proxy make this consistently timeout at 300s.

---

## What to Do

### 1. Run Full Suite A Benchmark with LLM Judge + LLM Contradiction

```bash
cd ~/Projects/hermes-agent
source .venv/bin/activate

# Ensure sentence-transformers and anthropic are installed
pip install sentence-transformers anthropic httpx

# Run Suite A with LLM judge AND LLM contradiction detection
python -m benchmarks.runner \
  --backend cognitive \
  --suite a \
  --runs 3 \
  --seeds 42 43 44 \
  --judge-model claude-haiku-4-5 \
  --contradiction-llm claude-haiku-4-5 \
  --output-dir benchmarks/results
```

This will save results to `benchmarks/results/cognitive.json`.

**Expected improvement areas:**
- contradictions: should go from 95% → 100% (ct_07 monolith→microservice fixed by LLM contradiction)
- cross_reference: should go from 91.1% → ~97% (xr_h04, xr_h11, xr_h13 fixed by LLM judge)
- importance_filtering: may improve (LLM judge understands nuance better than heuristic)
- Overall target: 95%+ → ideally 97%+

### 2. Run Full Suite B–E with LLM Judge

```bash
python -m benchmarks.runner \
  --backend cognitive \
  --suite b,c,d,e,f \
  --runs 1 \
  --seeds 42 \
  --judge-model claude-haiku-4-5 \
  --contradiction-llm claude-haiku-4-5 \
  --output-dir benchmarks/results
```

Previous scores (heuristic judge, from container):
- consolidation: 0.850 (Suite B)
- compression: 1.000 (Suite B)  
- scopes: 1.000 (Suite C)
- adversarial: 0.800 (Suite D)
- scale: 1.000 (Suite E)
- integration: 1.000 (Suite F)

With LLM judge, adversarial (ad_15 JSON format) may improve.

### 3. Update COGNITIVE_MEMORY_HANDOVER.md

After benchmarks run, update the handover doc with:
- New benchmark scores (LLM judge + sentence-transformers + LLM contradiction)
- Mark Phase 1 as fully complete with final numbers
- Note any remaining failures

### 4. Run Unit Tests (Sanity Check)

```bash
python -m pytest tests/cognitive_memory/ -v
# Should be 37 passing

python -m pytest tests/benchmarks/ -v  
# Should be 21 passing (LongMemEval adapter tests)
```

### 5. Commit Results

```bash
git add benchmarks/results/ COGNITIVE_MEMORY_HANDOVER.md TASKS.md
git commit -m "benchmark: final Phase 1 scores with LLM judge + LLM contradiction (host run)

Suite A: XX.X% overall (LLM judge + sentence-transformers + LLM contradiction)
  semantic_recall: X%, importance: X%, contradictions: X%, cross_ref: X%, temporal: X%
Suite B-E: consolidation X%, scopes X%, adversarial X%, scale X%"
```

---

## Current Benchmark Baseline (for comparison)

**cognitive-sbert.json** (heuristic judge, sentence-transformers, 3 runs, from container):
```
overall: 83.6%
semantic_recall: 100%
temporal_decay: 91.1%
importance_filtering: 77.5%
contradictions: 95.0%
cross_reference: 91.1%
consolidation: 60.0%
compression: 100%
scopes: 35.0%
scale: 37.5%
integration: 100%
```

Note: The low consolidation/scopes/scale scores above are from an OLDER run. Session 8 improved them significantly but that was a separate results file. The key comparison point is Suite A's overall score with LLM judge.

**Previous LLM judge run** (from Session 5, before sentence-transformers):
```
overall: 92.5% (LLM judge, TF-IDF embeddings)
```

---

## Key Architecture Notes

- **Aegis proxy** runs on host at port 8443 (read from `~/.hermes-aegis/proxy.pid`)
- **API key injection** happens at the proxy level — no keys in env or code
- **LLM contradiction detection** (`cognitive_memory/llm_contradiction.py`): 
  - Fires when heuristic score < threshold but entity overlap detected
  - Asks claude-haiku-4-5 "does fact B update/replace fact A?"
  - Routes through aegis proxy with placeholder key
  - From host, it should just use ANTHROPIC_API_KEY from env directly (no proxy needed)
- **Sentence-transformers** model cached at `~/Projects/.huggingface_cache`
- **Judge** (`benchmarks/judge.py`): MemoryJudge class uses Anthropic API
  - From host, it should pick up ANTHROPIC_API_KEY from env or .env file
  - May need `export ANTHROPIC_API_KEY=...` if not set

### If API Key Issues

The LLM judge and LLM contradiction both need an Anthropic API key. From the host:

```bash
# Check if key is available
echo $ANTHROPIC_API_KEY

# If not, check aegis vault
hermes-aegis vault list

# Or check .env
cat ~/.hermes/.env
```

The judge code (`benchmarks/judge.py`) creates an `anthropic.Anthropic()` client which reads `ANTHROPIC_API_KEY` from env by default. If running through aegis (`hermes-aegis run`), the key is injected automatically.

---

## Files Modified in Sessions 1–9

Key files for this work:
```
cognitive_memory/store.py           — main store, contradiction detection
cognitive_memory/llm_contradiction.py — LLM fallback for semantic contradictions
cognitive_memory/config.py          — contradiction_llm_model config
cognitive_memory/embeddings.py      — sentence-transformers integration
benchmarks/runner.py                — benchmark runner, --contradiction-llm flag
benchmarks/judge.py                 — HeuristicJudge + MemoryJudge (LLM)
benchmarks/results/                 — JSON result files
TASKS.md                            — session-by-session progress log
COGNITIVE_MEMORY_HANDOVER.md        — comprehensive handover document
```

---

## Success Criteria

1. Suite A overall ≥ 95% with LLM judge + sentence-transformers + LLM contradiction
2. Contradictions = 100% (ct_07 now handled by LLM contradiction)
3. Cross-reference ≥ 95% (LLM judge handles reasoning answers)
4. All unit tests still passing (37 cognitive + 21 longmemeval)
5. Results committed and handover doc updated
