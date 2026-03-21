"""Debug Suite C scope results."""
from benchmarks.runner import load_fixtures, BACKENDS
from benchmarks.judge import HeuristicJudge
from benchmarks.interface import BenchmarkConfig

config = BenchmarkConfig(
    backend_name='cognitive',
    parameters={'profile': 'balanced', 'embedding_model': 'tfidf', 'suites': ['c']}
)
backend = BACKENDS['cognitive'](**config.parameters)
judge = HeuristicJudge()
fixtures = load_fixtures('c')
scenarios = fixtures['scopes']

for sc in scenarios:
    backend.reset()
    for fact_obj in sc["facts"]:
        backend.store(
            fact_obj["content"],
            category="factual",
            scope=fact_obj.get("scope", "global"),
        )
    query_scope = sc.get("query_scope", "global")
    results = backend.recall(sc["query"], top_k=5, scope=query_scope)
    actual = results[0] if results else ""
    jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
    combined = " ".join(results).lower()
    should_not = sc.get("should_not_contain", "")
    no_leak = should_not.lower() not in combined if should_not else True
    scenario_ok = jr.correct and no_leak
    status = "PASS" if scenario_ok else "FAIL"
    print(f"{sc['id']} ({query_scope:20}): answer={str(jr.correct):5} no_leak={str(no_leak):5} {status}")
    if not jr.correct or not no_leak:
        print(f"  gold:       {sc['gold_answer'][:60]!r}")
        print(f"  top-1:      {actual[:60]!r}")
        if not no_leak:
            print(f"  LEAKED:     {should_not!r}")
