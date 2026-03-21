"""Run suites B-E one at a time and print failure details."""
import sys
import json
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'

sys.path.insert(0, '.')

from benchmarks.runner import (
    get_backend, load_fixtures, CATEGORY_RUNNERS,
    BenchmarkConfig
)
from benchmarks.judge import HeuristicJudge
from benchmarks.interface import BenchmarkConfig

import random
random.seed(42)

config = BenchmarkConfig(
    backend_name="cognitive",
    parameters={"embedding_model": "auto"},
    seeds=[42],
    num_runs=1,
    judge_model="heuristic",
)

backend = get_backend("cognitive", config)
judge = HeuristicJudge(model="heuristic")

for suite_letter in ['b', 'c', 'd', 'e']:
    try:
        fixtures = load_fixtures(suite_letter)
    except FileNotFoundError:
        continue
    
    for cat_name, scenarios in fixtures.items():
        runner = CATEGORY_RUNNERS.get(cat_name)
        if not runner:
            continue
        
        shuffled = list(scenarios)
        random.shuffle(shuffled)
        result = runner(backend, shuffled, judge)
        
        failures = [d for d in result.details if not d.get('correct')]
        if failures:
            print(f"\n{'='*60}")
            print(f"FAILURES: {cat_name} ({result.correct}/{result.total})")
            print(f"{'='*60}")
            for f_item in failures:
                print(f"\n  ID: {f_item.get('id', '?')}")
                for k, v in f_item.items():
                    if k != 'id':
                        val_str = str(v)[:150]
                        print(f"    {k}: {val_str}")
        else:
            print(f"\n✓ {cat_name}: {result.correct}/{result.total} (all pass)")
