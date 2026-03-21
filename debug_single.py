"""Quick debug — run one suite at a time."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

suite = sys.argv[1] if len(sys.argv) > 1 else "c"

from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter
from benchmarks.judge import HeuristicJudge

judge = HeuristicJudge(model="heuristic")
backend = CognitiveBenchmarkAdapter()

if suite == "c":
    with open("benchmarks/suite_c/fixtures/scopes.json") as f:
        scenarios = json.load(f)
    passed = 0
    for sc in scenarios:
        backend.reset()
        for fact_obj in sc["facts"]:
            if isinstance(fact_obj, dict):
                backend.store(fact_obj["content"], category="factual", scope=fact_obj["scope"])
            else:
                backend.store(fact_obj, category="factual")
        recalled = backend.recall(sc["query"], top_k=5, scope=sc.get("query_scope"))
        actual = recalled[0] if recalled else ""
        all_text = " ".join(recalled)
        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        leak = sc.get("should_not_contain", "")
        no_leak = leak.lower() not in all_text.lower() if leak else True
        ok = jr.correct and no_leak
        if ok: passed += 1
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] {sc['id']} gold={sc['gold_answer'][:50]} got={actual[:80]}")
    print(f"\nSuite C: {passed}/{len(scenarios)}")

elif suite == "d":
    with open("benchmarks/suite_d/fixtures/adversarial.json") as f:
        scenarios = json.load(f)
    passed = 0
    for sc in scenarios:
        backend.reset()
        for fact in sc["normal_facts"]:
            backend.store(fact, category="factual")
        backend.store(sc["malicious_fact"], category="factual")
        recalled = backend.recall(sc["query"], top_k=5)
        actual = recalled[0] if recalled else ""
        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        blocked_ok = True
        if sc.get("should_be_blocked", False):
            blocked_ok = actual.strip() != sc["malicious_fact"].strip()
        ok = jr.correct and blocked_ok
        if ok: passed += 1
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] {sc['id']} ({sc['type']}) gold={sc['gold_answer'][:40]} got={actual[:80]}")
    print(f"\nSuite D: {passed}/{len(scenarios)}")

elif suite == "e":
    import random
    with open("benchmarks/suite_e/fixtures/scale.json") as f:
        scenarios = json.load(f)
    passed = 0
    for sc in scenarios:
        backend.reset()
        if "time_series" in sc:
            entries = sorted(sc["time_series"], key=lambda e: e["stored_days_ago"], reverse=True)
            prev_days = entries[0]["stored_days_ago"]
            for entry in entries:
                gap = prev_days - entry["stored_days_ago"]
                if gap > 0: backend.simulate_time(gap)
                backend.store(entry["content"], category="factual")
                prev_days = entry["stored_days_ago"]
            for nf in sc.get("noise_facts", []):
                backend.store(nf, category="factual")
        else:
            backend.store(sc["target_fact"], category="factual", importance=0.8)
            for nf in sc.get("noise_facts", []):
                backend.store(nf, category="factual", importance=0.3)
            noise_count = sc.get("noise_count", 0)
            template = sc.get("noise_template", "System fact {i}: value {val}")
            existing = len(sc.get("noise_facts", [])) + 1
            rng = random.Random(42)
            for i in range(existing, noise_count):
                synth = template.format(i=i, val=rng.randint(1000, 9999))
                backend.store(synth, category="factual", importance=0.1)
            backend.simulate_time(30)
        recalled = backend.recall(sc["query"], top_k=5)
        actual = recalled[0] if recalled else ""
        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct: passed += 1
        status = "PASS" if jr.correct else "FAIL"
        if not jr.correct:
            print(f"  [{status}] {sc['id']} (n={sc.get('memory_count', sc.get('noise_count','?'))}) gold={sc['gold_answer'][:50]} got={actual[:80]}")
    print(f"\nSuite E: {passed}/{len(scenarios)}")

elif suite == "b":
    with open("benchmarks/suite_b/fixtures/consolidation.json") as f:
        scenarios = json.load(f)
    passed = 0
    for sc in scenarios:
        backend.reset()
        for fact in sc["facts"]:
            backend.store(fact, category="factual")
        for hint in sc.get("access_sequence", []):
            backend.simulate_access(hint)
        backend.simulate_time(sc.get("time_gap_days", 30))
        backend.consolidate()
        recalled = backend.recall(sc["query"], top_k=5)
        actual = recalled[0] if recalled else ""
        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct: passed += 1
        status = "PASS" if jr.correct else "FAIL"
        if not jr.correct:
            print(f"  [{status}] {sc['id']} gold={sc['gold_answer'][:50]} got={actual[:80]}")
    print(f"\nSuite B consolidation: {passed}/{len(scenarios)}")
