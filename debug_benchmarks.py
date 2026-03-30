"""
Diagnostic script to identify which specific scenarios fail in Suite B and D.
"""
import json
import sys
import os
import math
sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter
from benchmarks.judge import HeuristicJudge


def run_consolidation_diagnostic():
    """Diagnose Suite B consolidation failures."""
    print("\n" + "="*60)
    print("SUITE B: CONSOLIDATION DIAGNOSTIC")
    print("="*60)

    with open('/workspace/Projects/hermes-agent/benchmarks/suite_b/fixtures/consolidation.json') as f:
        scenarios = json.load(f)

    judge = HeuristicJudge()
    results = []

    for sc in scenarios:
        backend = CognitiveBenchmarkAdapter()
        backend.reset()

        for fact in sc["facts"]:
            backend.store(fact, category="factual")

        for access_hint in sc.get("access_sequence", []):
            backend.simulate_access(access_hint)

        backend.simulate_time(sc.get("time_gap_days", 30))
        backend.consolidate()

        # Inspect layer of each memory
        all_mems = backend._store.backend.get_all_active()
        layers = {m.content[:40]: m.layer for m in all_mems}
        layer_counts = {}
        for layer in layers.values():
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        query_results = backend.recall(sc["query"], top_k=5)
        actual = " | ".join(query_results[:3]) if query_results else ""

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)

        status = "✓" if jr.correct else "✗"
        results.append({
            "id": sc["id"],
            "correct": jr.correct,
            "expects_layer": sc.get("expects_layer"),
            "layer_counts": layer_counts,
            "actual": actual[:80],
            "gold": sc["gold_answer"],
        })

        print(f"\n{status} {sc['id']} [{sc.get('expects_layer','?')}] - {sc['description'][:50]}")
        print(f"  Query: {sc['query'][:60]}")
        print(f"  Gold:  {sc['gold_answer'][:60]}")
        print(f"  Got:   {actual[:60]}")
        print(f"  Layers: {layer_counts}")

        # For archive scenarios, check if item was actually archived
        if sc.get('expects_layer') == 'archive':
            print(f"  Access seq: {sc.get('access_sequence', [])} (len={len(sc.get('access_sequence', []))})")
            print(f"  Time gap: {sc.get('time_gap_days')} days")
            # Check if facts got archived
            archived = [m.content[:40] for m in all_mems if m.layer == 'archive']
            working = [m.content[:40] for m in all_mems if m.layer == 'working']
            core = [m.content[:40] for m in all_mems if m.layer == 'core']
            print(f"  Working({len(working)}): {working[:2]}")
            print(f"  Core({len(core)}): {core[:2]}")
            print(f"  Archive({len(archived)}): {archived[:2]}")
            # Check if specific gold fact was archived
            gold_facts = [m for m in all_mems if sc["gold_answer"].lower() in m.content.lower()]
            if gold_facts:
                print(f"  GOLD FACT layer: {gold_facts[0].layer} | access_count={gold_facts[0].access_count}")
                # Check ACT-R activation
                now = backend._store._now()
                activation = backend._store._actr_base_level(gold_facts[0].access_times, now, backend._store.config.d)
                archive_bl_threshold = math.log(max(backend._store.config.archive_threshold, 1e-10))
                print(f"  ACT-R activation: {activation:.3f}, archive threshold: {archive_bl_threshold:.3f}")
                print(f"  Would archive: {activation < archive_bl_threshold}")

    # Summary
    print("\n\nSUMMARY:")
    for r in results:
        print(f"  {r['id']}: {'PASS' if r['correct'] else 'FAIL'} (expects={r['expects_layer']}) layers={r['layer_counts']}")
    
    by_layer = {'core': [], 'archive': []}
    for r in results:
        layer = r.get('expects_layer')
        if layer in by_layer:
            by_layer[layer].append(r['correct'])
    
    for layer, outcomes in by_layer.items():
        if outcomes:
            score = sum(outcomes)/len(outcomes)
            print(f"  {layer} score: {score:.0%} ({sum(outcomes)}/{len(outcomes)})")


def run_adversarial_diagnostic():
    """Diagnose Suite D adversarial failures."""
    print("\n" + "="*60)
    print("SUITE D: ADVERSARIAL DIAGNOSTIC")
    print("="*60)

    with open('/workspace/Projects/hermes-agent/benchmarks/suite_d/fixtures/adversarial.json') as f:
        scenarios = json.load(f)

    judge = HeuristicJudge()
    
    from cognitive_memory.store import CognitiveMemoryStore

    for sc in scenarios:
        backend = CognitiveBenchmarkAdapter()
        backend.reset()

        for fact in sc.get("normal_facts", []):
            backend.store(fact, category="factual")
        
        backend.store(sc["malicious_fact"], category="factual")

        # Check adversarial score
        adv_score = CognitiveMemoryStore._adversarial_score(sc["malicious_fact"])

        query_results = backend.recall(sc["query"], top_k=5)
        actual = " | ".join(query_results[:5]) if query_results else ""
        top1 = query_results[0] if query_results else ""

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)

        should_be_blocked = sc.get("should_be_blocked", False)
        malicious = sc["malicious_fact"].lower()
        top1_is_malicious = top1.lower().strip() == malicious.strip()
        blocked_ok = (not should_be_blocked) or (not top1_is_malicious)
        scenario_correct = jr.correct and blocked_ok

        status = "✓" if scenario_correct else "✗"
        print(f"\n{status} {sc['id']} [{sc['type']}] - {sc['description'][:50]}")
        print(f"  Query: {sc['query'][:60]}")
        print(f"  Gold:  {sc['gold_answer'][:60]}")
        print(f"  Top1:  {top1[:60]}")
        print(f"  Top2:  {query_results[1][:60] if len(query_results)>1 else 'N/A'}")
        print(f"  Adv score: {adv_score:.3f} (threshold: 0.2)")
        print(f"  Should block: {should_be_blocked}, Top1 is malicious: {top1_is_malicious}, blocked_ok: {blocked_ok}")
        print(f"  Answer correct: {jr.correct}, Overall correct: {scenario_correct}")
        if not jr.correct:
            print(f"  FAIL REASON: wrong answer (actual: {actual[:60]})")
        if not blocked_ok:
            print(f"  FAIL REASON: malicious content in top1")


if __name__ == "__main__":
    run_consolidation_diagnostic()
    run_adversarial_diagnostic()
