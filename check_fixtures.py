import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from benchmarks.runner import load_fixtures

for suite in ['b', 'c', 'd', 'e']:
    fixtures = load_fixtures(suite)
    print(f'Suite {suite}: categories={list(fixtures.keys())}, counts={[(k, len(v)) for k, v in fixtures.items()]}')
