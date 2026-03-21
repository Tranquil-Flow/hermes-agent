"""Debug Suite D adversarial results."""
import json
from benchmarks.runner import load_fixtures, BACKENDS
from benchmarks.judge import HeuristicJudge
from benchmarks.interface import BenchmarkConfig

config = BenchmarkConfig(
    backend_name='cognitive',
    parameters={'profile': 'balanced', 'embedding_model': 'tfidf', 'suites': ['d']}
)
backend = BACKENDS['cognitive'](**config.parameters)
judge = HeuristicJudge()
fixtures = load_fixtures('d')
scenarios = fixtures['adversarial']

for sc in scenarios:
    backend.reset()
    for fact in sc.get('normal_facts', []):
        backend.store(fact, category='factual')
    backend.store(sc['malicious_fact'], category='factual')
    results = backend.recall(sc['query'], top_k=5)
    actual = results[0] if results else ''
    jr = judge.judge_answer(sc['query'], sc['gold_answer'], actual)
    malicious = sc['malicious_fact'].lower()
    top1_is_malicious = actual.lower().strip() == malicious.strip()
    blocked_ok = (not sc.get('should_be_blocked', False)) or (not top1_is_malicious)
    scenario_ok = jr.correct and blocked_ok
    print(f"{sc['id']} ({sc['type'][:20]:20}): answer={str(jr.correct):5} blocked_ok={str(blocked_ok):5} PASS={scenario_ok}")
    if not jr.correct:
        print(f"  gold:  {sc['gold_answer'][:60]!r}")
        print(f"  top-1: {actual[:60]!r}")
