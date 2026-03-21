"""Run all benchmark suites (a-f) with a single seed for quick verification."""
import sys, os, json, gc
sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/Projects/.huggingface_cache'

from benchmarks.runner import load_fixtures, CATEGORY_RUNNERS
from benchmarks.interface import BenchmarkConfig
from benchmarks.judge import HeuristicJudge

# Late import to trigger model loading only once
from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter

config = BenchmarkConfig(
    backend_name="cognitive",
    num_runs=1,
    seeds=[42],
    judge_model="heuristic",
    parameters={},
)

adapter = CognitiveBenchmarkAdapter()
judge = HeuristicJudge(model="heuristic")

results = {}
for suite_letter in ["a", "b", "c", "d", "e", "f"]:
    try:
        fixtures = load_fixtures(suite_letter)
    except FileNotFoundError:
        continue
    
    for category_name, scenarios in fixtures.items():
        runner = CATEGORY_RUNNERS.get(category_name)
        if not runner:
            continue
        
        result = runner(adapter, scenarios, judge)
        results[category_name] = {
            "score": result.score,
            "correct": result.correct,
            "total": result.total,
        }
        print(f"  {category_name}: {result.score:.3f} ({result.correct}/{result.total})")
        gc.collect()

# Summary
print(f"\n{'='*50}")
total_correct = sum(r["correct"] for r in results.values())
total_items = sum(r["total"] for r in results.values())
overall = total_correct / total_items if total_items else 0
print(f"OVERALL: {overall:.3f} ({total_correct}/{total_items})")
for cat, r in sorted(results.items()):
    print(f"  {cat:<25} {r['score']:.3f} ({r['correct']}/{r['total']})")
