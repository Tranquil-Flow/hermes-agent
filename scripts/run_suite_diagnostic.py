"""Run a single benchmark suite with detailed output."""
import json
import sys
import os

suite_letter = sys.argv[1] if len(sys.argv) > 1 else "b"

sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/Projects/.huggingface_cache'

from benchmarks.runner import (
    load_fixtures, get_backend, register_backend, CATEGORY_RUNNERS,
)
from benchmarks.interface import BenchmarkConfig
from benchmarks.judge import HeuristicJudge

from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter
register_backend("cognitive", CognitiveBenchmarkAdapter)

config = BenchmarkConfig(
    backend_name="cognitive",
    num_runs=1,
    seeds=[42],
    judge_model="heuristic",
    parameters={"suites": [suite_letter]},
)

backend = get_backend("cognitive", config)
judge = HeuristicJudge(model="heuristic")

fixtures = load_fixtures(suite_letter)
for category_name, scenarios in fixtures.items():
    runner = CATEGORY_RUNNERS.get(category_name)
    if not runner:
        print(f"  No runner for {category_name}")
        continue
    
    result = runner(backend, scenarios, judge)
    print(f"\n{'='*60}")
    print(f"Suite {suite_letter.upper()} - {category_name}: {result.score:.3f} ({result.correct}/{result.total})")
    print(f"{'='*60}")
    
    if hasattr(result, 'sub_scores') and result.sub_scores:
        print(f"  Sub-scores: {result.sub_scores}")
    
    for d in result.details:
        status = "PASS" if d["correct"] else "FAIL"
        print(f"  [{status}] {d['id']}")
        if not d["correct"]:
            print(f"    Gold: {d.get('gold', 'N/A')}")
            print(f"    Actual: {d.get('actual', 'N/A')[:150]}")
            for key in ['answer_correct', 'preserved', 'must_preserve', 'no_leak',
                       'should_not_contain', 'query_scope', 'type', 'should_be_blocked',
                       'top1_is_malicious', 'blocked_ok', 'expects_layer', 'difficulty']:
                if key in d:
                    print(f"    {key}: {d[key]}")
