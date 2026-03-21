#!/usr/bin/env python3
"""Run only contradiction scenarios from Suite A, with LLM judge + LLM contradiction detection."""
import sys
import os
import json
import gc
import random

sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'

from benchmarks.runner import (
    run_contradictions, get_backend, BenchmarkConfig, load_fixtures,
    MemoryJudge
)

print("Loading fixtures...")
fixtures = load_fixtures("a")
scenarios = fixtures.get("contradictions", [])
print(f"Found {len(scenarios)} contradiction scenarios")

config = BenchmarkConfig(
    backend_name="cognitive",
    num_runs=1,
    judge_model="claude-haiku-4-5",
    seeds=[42],
    parameters={
        "suites": ["a"],
        "contradiction_llm_model": "claude-haiku-4-5",
    }
)

print("Initializing judge...")
judge = MemoryJudge(model="claude-haiku-4-5")

print("Running contradictions benchmark...")
random.seed(42)
shuffled = list(scenarios)
random.shuffle(shuffled)

backend = get_backend("cognitive", config)
result = run_contradictions(backend, shuffled, judge)

total = result.total
correct = result.correct
score = correct / total if total > 0 else 0

print(f"\nContradictions: {correct}/{total} = {score:.3f} ({100*score:.1f}%)")
print()

# Show per-scenario detail
if hasattr(result, 'details') and result.details:
    for d in result.details:
        status = "PASS" if d.get('correct') else "FAIL"
        print(f"  [{status}] {d.get('scenario', '?')}: {d.get('type', '?')}")
