"""
Hermes Cognitive Memory Benchmark Runner

Usage:
    python -m benchmarks.runner --backend baseline-flat --suite a --runs 5
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

from benchmarks.interface import (
    BenchmarkableStore, BenchmarkConfig, RunResult,
    AggregateResult, CategoryResult, SignificanceResult,
)
from benchmarks.judge import MemoryJudge, HeuristicJudge
from benchmarks.statistical import aggregate_results, compare_runs


# --- Backend Registry ---

BACKENDS: Dict[str, Type[BenchmarkableStore]] = {}


def register_backend(name: str, cls: Type[BenchmarkableStore]):
    """Register a memory backend for benchmarking."""
    BACKENDS[name] = cls


def get_backend(name: str, config: BenchmarkConfig) -> BenchmarkableStore:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name](**config.parameters)


# Register built-in backends
from benchmarks.baseline.flat_store import FlatMemoryStore
register_backend("baseline-flat", FlatMemoryStore)

# Register cognitive memory backend
try:
    from cognitive_memory.benchmark_adapter import CognitiveBenchmarkAdapter
    register_backend("cognitive", CognitiveBenchmarkAdapter)
except ImportError:
    pass  # cognitive_memory not available


# --- Token Estimation ---

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text.
    Good enough for cost estimation. For exact counts, use tiktoken.
    """
    return max(len(text) // 4, 1)


def count_recall_tokens(results: list) -> tuple:
    """Count tokens and chars in recalled memory strings.
    Returns (token_count, char_count).
    """
    total_chars = sum(len(r) for r in results)
    total_tokens = sum(estimate_tokens(r) for r in results)
    return total_tokens, total_chars


# --- Fixture Loading ---

SUITE_DIR = Path(__file__).parent


def load_fixtures(suite: str) -> Dict[str, list]:
    """Load all fixture JSON files for a given suite (e.g., 'a')."""
    fixture_dir = SUITE_DIR / f"suite_{suite}" / "fixtures"
    if not fixture_dir.exists():
        raise FileNotFoundError(f"No fixtures found at {fixture_dir}")

    fixtures = {}
    for f in sorted(fixture_dir.glob("*.json")):
        with open(f) as fh:
            fixtures[f.stem] = json.load(fh)
    return fixtures


# --- Scenario Runners ---

def run_semantic_recall(backend: BenchmarkableStore, scenarios: list,
                        judge: MemoryJudge) -> CategoryResult:
    """Run semantic recall scenarios (Suite A1)."""
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()
        backend.store(sc["fact"], category="factual")
        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })

    # Sub-scores by difficulty
    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    return CategoryResult(
        category="semantic_recall",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
    )


def run_contradictions(backend: BenchmarkableStore, scenarios: list,
                       judge: MemoryJudge) -> CategoryResult:
    """Run contradiction handling scenarios (Suite A2)."""
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()
        # Store older fact first, then newer
        backend.store(sc["fact_a"], category="factual")
        backend.store(sc["fact_b"], category="factual")

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "type": sc["contradiction_type"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })

    return CategoryResult(
        category="contradictions",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
    )


def run_temporal_decay(backend: BenchmarkableStore, scenarios: list,
                       judge: MemoryJudge) -> CategoryResult:
    """Run temporal decay scenarios (Suite A3)."""
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()

        # Store facts with proper time simulation.
        # Sort by stored_days_ago descending (oldest first).
        # Advance simulated time between stores to create real recency gaps.
        sorted_facts = sorted(sc["facts"], key=lambda f: f["stored_days_ago"], reverse=True)
        prev_days_ago = sorted_facts[0]["stored_days_ago"] if sorted_facts else 0

        for fact in sorted_facts:
            # Advance time from previous fact to this one
            time_gap = prev_days_ago - fact["stored_days_ago"]
            if time_gap > 0:
                backend.simulate_time(time_gap)
            prev_days_ago = fact["stored_days_ago"]

            backend.store(fact["content"], category="factual")

            # Simulate rehearsals if present
            for r_day in fact.get("rehearsed_days_ago", []):
                backend.simulate_access(fact["content"])

        # Advance remaining time to "now" (days_ago=0)
        if prev_days_ago > 0:
            backend.simulate_time(prev_days_ago)

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    return CategoryResult(
        category="temporal_decay",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
    )


def run_cross_reference(backend: BenchmarkableStore, scenarios: list,
                        judge: MemoryJudge) -> CategoryResult:
    """Run cross-reference scenarios (Suite A4)."""
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()
        for fact in sc["facts"]:
            backend.store(fact, category="factual")

        results = backend.recall(sc["query"], top_k=10)
        # Concatenate top results as the answer context
        actual = " | ".join(results[:sc["num_facts_needed"]]) if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "num_facts_needed": sc["num_facts_needed"],
            "actual": actual,
            "gold": sc["gold_answer"],
        })

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    return CategoryResult(
        category="cross_reference",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
    )


def run_importance_filtering(backend: BenchmarkableStore, scenarios: list,
                             judge: MemoryJudge) -> CategoryResult:
    """Run importance filtering scenarios (Suite A5)."""
    correct = 0
    details = []
    total_recall_tokens = 0
    total_recall_chars = 0

    for sc in scenarios:
        backend.reset()
        for fact in sc["important_facts"]:
            backend.store(fact["content"], importance=fact["importance"])
        for fact in sc["noise_facts"]:
            backend.store(fact["content"], importance=fact["importance"])

        results = backend.recall(sc["query"], top_k=5)
        actual = results[0] if results else ""
        rt, rc = count_recall_tokens(results)
        total_recall_tokens += rt
        total_recall_chars += rc

        jr = judge.judge_answer(sc["query"], sc["gold_answer"], actual)
        if jr.correct:
            correct += 1

        details.append({
            "id": sc["id"],
            "difficulty": sc["difficulty"],
            "correct": jr.correct,
            "actual": actual,
            "gold": sc["gold_answer"],
        })

    sub_scores = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [d for d in details if d["difficulty"] == diff]
        if subset:
            sub_scores[diff] = sum(1 for d in subset if d["correct"]) / len(subset)

    return CategoryResult(
        category="importance_filtering",
        total=len(scenarios),
        correct=correct,
        score=correct / len(scenarios) if scenarios else 0,
        sub_scores=sub_scores,
        details=details,
        recall_tokens=total_recall_tokens,
        recall_chars=total_recall_chars,
    )


# Category runner dispatch
CATEGORY_RUNNERS = {
    "semantic_recall": run_semantic_recall,
    "contradictions": run_contradictions,
    "temporal_decay": run_temporal_decay,
    "cross_reference": run_cross_reference,
    "importance_filtering": run_importance_filtering,
}


# --- Main Run Logic ---

def run_single(config: BenchmarkConfig, seed: int) -> RunResult:
    """Execute one full benchmark run with a given seed.
    
    The seed controls scenario shuffling to measure variance from
    insertion/query order effects. This is critical for detecting
    order-dependent bugs (e.g., TF-IDF vocab growth, Hebbian link
    formation path-dependence).
    """
    import random
    random.seed(seed)

    start = time.time()
    backend = get_backend(config.backend_name, config)

    # Use HeuristicJudge by default; LLM judge for real results
    if config.judge_model == "heuristic":
        judge = HeuristicJudge(model="heuristic")
    else:
        judge = MemoryJudge(model=config.judge_model)

    results_by_cat = {}

    for suite_letter in ["a"]:  # TODO: expand for b-f
        fixtures = load_fixtures(suite_letter)
        for category_name, scenarios in fixtures.items():
            runner = CATEGORY_RUNNERS.get(category_name)
            if runner:
                # Shuffle scenarios to measure order-dependence
                shuffled = list(scenarios)
                random.shuffle(shuffled)
                cat_result = runner(backend, shuffled, judge)
                results_by_cat[category_name] = cat_result

    elapsed = time.time() - start

    # Compute overall score (weighted average)
    total_correct = sum(c.correct for c in results_by_cat.values())
    total_items = sum(c.total for c in results_by_cat.values())
    overall = total_correct / total_items if total_items > 0 else 0

    # Aggregate token usage across categories
    total_recall_tokens = sum(c.recall_tokens for c in results_by_cat.values())
    total_recall_chars = sum(c.recall_chars for c in results_by_cat.values())
    num_queries = total_items

    return RunResult(
        seed=seed,
        results_by_category=results_by_cat,
        overall_score=overall,
        token_usage={
            "recall_tokens": total_recall_tokens,
            "recall_chars": total_recall_chars,
            "recall_queries": num_queries,
            "avg_recall_tokens_per_query": total_recall_tokens // max(num_queries, 1),
        },
        wall_time_seconds=elapsed,
    )


def run_benchmark(config: BenchmarkConfig) -> tuple:
    """Run the full benchmark suite with multiple seeds and aggregate.
    Returns (AggregateResult, list[RunResult]) for comparison.
    """
    runs = []
    for seed in config.seeds[:config.num_runs]:
        print(f"  Run seed={seed}...", end=" ", flush=True)
        result = run_single(config, seed)
        runs.append(result)
        print(f"score={result.overall_score:.3f} ({result.wall_time_seconds:.1f}s)")

    return aggregate_results(runs), runs


def print_results(agg: AggregateResult, config: BenchmarkConfig,
                   runs: Optional[list] = None):
    """Print a summary table to stdout."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS: {config.backend_name}")
    print(f"{'='*60}")
    print(f"  Runs: {agg.num_runs}")
    print(f"  Overall: {agg.mean_score:.3f} ± {agg.std_score:.3f}")
    print(f"  95% CI:  [{agg.ci_95_lower:.3f}, {agg.ci_95_upper:.3f}]")
    print(f"{'─'*60}")
    print(f"  {'Category':<25} {'Mean':>8} {'Std':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8}")
    for cat, mean in sorted(agg.per_category_mean.items()):
        std = agg.per_category_std.get(cat, 0)
        print(f"  {cat:<25} {mean:>8.3f} {std:>8.3f}")
    print(f"{'─'*60}")
    # Token usage summary
    if runs:
        avg_tokens = sum(
            r.token_usage.get("avg_recall_tokens_per_query", 0) for r in runs
        ) // len(runs)
        total_tokens = sum(r.token_usage.get("recall_tokens", 0) for r in runs) // len(runs)
        total_queries = sum(r.token_usage.get("recall_queries", 0) for r in runs) // len(runs)
        print(f"  Token cost (avg per run):")
        print(f"    Recall tokens/query:  ~{avg_tokens}")
        print(f"    Total recall tokens:  ~{total_tokens} ({total_queries} queries)")
    print(f"{'='*60}\n")


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Hermes Cognitive Memory Benchmark Runner"
    )
    parser.add_argument("--backend", default="baseline-flat",
                        choices=list(BACKENDS.keys()),
                        help="Memory backend to benchmark")
    parser.add_argument("--profile", default="balanced",
                        help="Config profile for cognitive backend")
    parser.add_argument("--embedding", default="tfidf",
                        help="Embedding model: auto, sentence-transformers, tfidf")
    parser.add_argument("--suite", default="a",
                        help="Suite(s) to run: a,b,c,d,e,f or 'all'")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per benchmark")
    parser.add_argument("--judge-model", default="heuristic",
                        help="Model for LLM-as-judge (default: heuristic)")
    parser.add_argument("--output-dir", default="benchmarks/results/",
                        help="Directory for JSON results")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44, 45, 46],
                        help="Random seeds for runs")
    parser.add_argument("--compare", default=None,
                        help="Compare against another backend (runs both)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    config = BenchmarkConfig(
        backend_name=args.backend,
        profile=args.profile,
        embedding_model=args.embedding,
        num_runs=args.runs,
        judge_model=args.judge_model,
        output_path=args.output_dir,
        seeds=args.seeds,
        parameters={"profile": args.profile, "embedding_model": args.embedding},
    )

    print(f"\nRunning {config.backend_name} benchmark ({config.num_runs} runs)...")
    agg, runs = run_benchmark(config)

    if args.json:
        results_dict = {
            "backend": config.backend_name,
            "mean_score": agg.mean_score,
            "std": agg.std_score,
            "ci_95": [agg.ci_95_lower, agg.ci_95_upper],
            "per_category": agg.per_category_mean,
            "num_runs": agg.num_runs,
        }
        print(json.dumps(results_dict, indent=2))
    else:
        print_results(agg, config, runs)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{config.backend_name}.json"
    with open(result_file, "w") as f:
        json.dump({
            "backend": config.backend_name,
            "profile": config.profile,
            "embedding_model": config.embedding_model,
            "mean_score": agg.mean_score,
            "std": agg.std_score,
            "ci_95": [agg.ci_95_lower, agg.ci_95_upper],
            "per_category_mean": agg.per_category_mean,
            "per_category_std": agg.per_category_std,
            "num_runs": agg.num_runs,
            "runs": [
                {
                    "seed": r.seed,
                    "overall_score": r.overall_score,
                    "wall_time_seconds": r.wall_time_seconds,
                    "token_usage": r.token_usage,
                    "categories": {
                        cat: {
                            "score": cr.score, "correct": cr.correct, "total": cr.total,
                            "recall_tokens": cr.recall_tokens, "recall_chars": cr.recall_chars,
                        }
                        for cat, cr in r.results_by_category.items()
                    },
                }
                for r in runs
            ],
        }, f, indent=2)
    print(f"  Results saved to {result_file}")

    if args.compare:
        print(f"\nRunning comparison: {args.compare}...")
        config2 = BenchmarkConfig(
            backend_name=args.compare,
            profile=args.profile,
            embedding_model=args.embedding,
            num_runs=args.runs,
            judge_model=args.judge_model,
            output_path=args.output_dir,
            seeds=args.seeds,
            parameters={"profile": args.profile, "embedding_model": args.embedding},
        )
        agg2, runs2 = run_benchmark(config2)
        print_results(agg2, config2, runs2)

        # Save comparison results
        result_file2 = output_dir / f"{config2.backend_name}.json"
        with open(result_file2, "w") as f:
            json.dump({
                "backend": config2.backend_name,
                "mean_score": agg2.mean_score,
                "std": agg2.std_score,
                "num_runs": agg2.num_runs,
                "per_category_mean": agg2.per_category_mean,
            }, f, indent=2)

        # Significance test
        sig = compare_runs(runs, runs2)
        print(f"\n{'='*60}")
        print(f"  STATISTICAL COMPARISON")
        print(f"{'='*60}")
        print(f"  {config.backend_name}: {sig.baseline_mean:.3f}")
        print(f"  {config2.backend_name}: {sig.experiment_mean:.3f}")
        print(f"  Improvement: {sig.improvement:+.1f} pp")
        print(f"  Effect size (Cohen's d): {sig.effect_size:.3f}")
        print(f"  p-value: {sig.p_value:.4f}")
        print(f"  Significant (p<0.05): {'YES' if sig.significant else 'NO'}")
        print(f"  Test: {sig.test_name}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
