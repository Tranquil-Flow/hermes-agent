#!/usr/bin/env python3
"""Debug judge differences for importance_filtering and contradictions."""
import json, sys, random
sys.path.insert(0, '/workspace/Projects/hermes-agent')
from benchmarks.judge import MemoryJudge, HeuristicJudge
from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter
from benchmarks.runner import run_importance_filtering, run_contradictions
from benchmarks.interface import BenchmarkConfig

random.seed(42)

heur_judge = HeuristicJudge()
llm_judge = MemoryJudge(model='claude-haiku-4-5')

# Load fixtures
with open('/workspace/Projects/hermes-agent/benchmarks/suite_a/fixtures/importance_filtering.json') as f:
    import_scenarios = json.load(f)
with open('/workspace/Projects/hermes-agent/benchmarks/suite_a/fixtures/contradictions.json') as f:
    contr_scenarios = json.load(f)

backend = CognitiveBenchmarkAdapter()

print('=== Importance Filtering ===')
heur_result = run_importance_filtering(backend, import_scenarios, heur_judge)
print(f'Heuristic: {heur_result.score:.3f} ({heur_result.correct}/{heur_result.total})')

backend.reset()
llm_result = run_importance_filtering(backend, import_scenarios, llm_judge)
print(f'LLM: {llm_result.score:.3f} ({llm_result.correct}/{llm_result.total})')

# Find disagreements
heur_ids = {d['id'] for d in heur_result.details if d['correct']}
llm_ids = {d['id'] for d in llm_result.details if d['correct']}
print(f'Heur correct, LLM wrong: {heur_ids - llm_ids}')
print(f'LLM correct, Heur wrong: {llm_ids - heur_ids}')

# Show what LLM gets wrong
for d in llm_result.details:
    if not d['correct']:
        print(f'  FAIL {d["id"]}: gold={d["gold"][:60]}, actual={d["actual"][:60]}')

print()
print('=== Contradictions ===')
backend.reset()
heur_c = run_contradictions(backend, contr_scenarios, heur_judge)
print(f'Heuristic: {heur_c.score:.3f} ({heur_c.correct}/{heur_c.total})')

backend.reset()
llm_c = run_contradictions(backend, contr_scenarios, llm_judge)
print(f'LLM: {llm_c.score:.3f} ({llm_c.correct}/{llm_c.total})')

heur_ids_c = {d['id'] for d in heur_c.details if d['correct']}
llm_ids_c = {d['id'] for d in llm_c.details if d['correct']}
print(f'Heur correct, LLM wrong: {heur_ids_c - llm_ids_c}')

for d in llm_c.details:
    if not d['correct']:
        print(f'  FAIL {d["id"]}: gold={d["gold"][:60]}, actual={d["actual"][:80]}')
