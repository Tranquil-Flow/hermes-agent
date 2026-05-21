#!/usr/bin/env python3
"""Tier 1: LLM generation + judge pass for LLMLingua agent-task benchmark.

Extends the existing fact-survival benchmark with:
  1. Generation pass: feed compressed context + question to an LLM, collect answers.
  2. Judge pass: score answers against ground-truth expected answers.
  3. Token consumption and wall-clock timing per task per lane.

Usage:
  python scripts/benchmark_llmlingua_generation.py
  python scripts/benchmark_llmlingua_generation.py --judge-model gpt-4o
  python scripts/benchmark_llmlingua_generation.py --gen-model qwen3:14b --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Reuse the existing benchmark harness for task fixtures and compression lanes.
# We import from the sibling script by path so the contrib checkout stays clean.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from benchmark_llmlingua_agent_tasks import (
    TaskCase,
    build_compressor,
    evaluate_text,
    is_default_llmlingua_candidate,
    task_fixtures,
)

from openai import OpenAI


# ── Expected answers (ground truth) ──────────────────────────────────────────
# Each task has a set of key facts that a correct answer must contain.
# We express them as loose regexes — the judge checks all of them.

EXPECTED_ANSWERS: dict[str, dict] = {
    "web_security_advisory": {
        "summary": "Affected versions 4.8.0–4.8.6, CVE-2026-4242, fixed in 4.8.7, rotate webhook secrets",
        "required_patterns": [
            r"4\.8\.[0-6]",                   # affected range
            r"CVE-2026-4242",
            r"4\.8\.7",                        # fixed version
            r"rotat\w+.*webhook.*secret",      # mitigation
        ],
    },
    "api_migration_docs": {
        "summary": "POST /v3/jobs/export, Idempotency-Key UUIDv7, webhook.destination_url, X-Export-Dry-Run",
        "required_patterns": [
            r"/v3/jobs/export",
            r"UUIDv7",
            r"webhook\.destination_url|destination_url.*replac",
            r"X-Export-Dry-Run|dry.run",
        ],
    },
    "pricing_policy_research": {
        "summary": "2M included tokens, $0.42 overage per 10K, monthly spend ceiling",
        "required_patterns": [
            r"2.?000.?000",
            r"0\.42",
            r"spend\s+ceiling|monthly\s+cap",
        ],
    },
    "browser_checkout_snapshot": {
        "summary": "Payment failed, 3-D Secure required",
        "required_patterns": [
            r"Payment\s+failed|payment.*failed",
            r"3.D\s+Secure|3DS",
        ],
    },
    "web_search_release_notes": {
        "summary": "AstraDB 1.14, XChaCha20-Poly1305, encrypted local-first sync",
        "required_patterns": [
            r"1\.14",
            r"XChaCha20",
            r"local.first",
        ],
    },
    "incident_feed_json": {
        "summary": "inc_9af31, Webhook delivery delays, eu-west-3, 45 min ETA",
        "required_patterns": [
            r"inc_9af31|9af31",
            r"Webhook\s+delivery\s+delays|delivery\s+delays",
            r"eu-west-3",
            r"45",
        ],
    },
    "terminal_failure_log_excluded": {
        "summary": "test_signature_rejects_replayed_timestamp, expected 401 got 204, replay window 900s vs 300s",
        "required_patterns": [
            r"test_signature_rejects_replayed_timestamp",
            r"401",
            r"204",
            r"900.*300|300.*900",
        ],
    },
    "code_search_result_excluded": {
        "summary": "tool_result_compressor.py line 768, trust_external_endpoint",
        "required_patterns": [
            r"tool_result_compressor",
            r"768",
            r"trust_external_endpoint",
        ],
    },
    "browser_dashboard_status": {
        "summary": "ingest-normal priority queue degraded, 37 min age, 6.2% error rate",
        "required_patterns": [
            r"ingest.normal|normal.priority",
            r"degraded",
            r"37",
            r"6\.2\s*%",
        ],
    },
    "oidc_docs": {
        "summary": "Issuer identity.example.dev/oauth2/default, hermes-agent-prod audience, 120s clock skew",
        "required_patterns": [
            r"identity\.example\.dev|oauth2/default",
            r"hermes.agent.prod",
            r"120\s*(?:second|sec|s\b)",
        ],
    },
    "regulatory_policy_article": {
        "summary": "7 years retention, prompt templates + model version + reviewer + appeal outcome, redacted after 30 days, audit hashes",
        "required_patterns": [
            r"7\s+year",
            r"prompt\s+template|model\s+version|reviewer",
            r"30\s+day",
            r"audit\s+hash|hash.*verif",
        ],
    },
    "package_changelog": {
        "summary": "Version 0.9.12, streaming cancellation leak, aclose() with 2s timeout",
        "required_patterns": [
            r"0\.9\.12",
            r"streaming\s+cancellation|cancellation\s+leak",
            r"aclose|2\s*(?:second|sec|s\b)\s*(?:timeout|time)",
        ],
    },
}


@dataclass
class GenerationResult:
    """Result of one task × lane generation pass."""
    task_id: str
    mode: str
    question: str
    context: str  # the compressed text shown to the model
    answer: str
    input_tokens: int = 0
    output_tokens: int = 0
    gen_latency_ms: float = 0.0
    judge_score: float = 0.0  # 0.0–1.0 fraction of required patterns matched
    judge_details: dict = field(default_factory=dict)


def make_client(base_url: str | None, api_key: str | None) -> OpenAI:
    """Create an OpenAI client from args or environment."""
    return OpenAI(
        base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8081/v1"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "benchmark"),
    )


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on the provided context. "
    "Answer concisely and factually using ONLY information from the context. "
    "If the context does not contain enough information, say so."
)


def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    context: str,
    tool_name: str = "",
) -> dict:
    """Call the LLM with compressed context and return answer + metrics."""
    user_msg = (
        f"Context (from tool '{tool_name}'):\n"
        f"---\n{context}\n---\n\n"
        f"Question: {question}"
    )

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    answer = resp.choices[0].message.content or ""
    usage = resp.usage

    return {
        "answer": answer,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "gen_latency_ms": round(latency_ms, 1),
    }


def judge_answer(answer: str, expected: dict) -> dict:
    """Score an answer against expected patterns.

    Returns dict with:
      score: fraction of required patterns matched (0.0–1.0)
      matched: list of matched pattern indices
      missed: list of missed pattern indices
      correct: bool (all required matched)
    """
    patterns = expected.get("required_patterns", [])
    matched = []
    missed = []

    for i, pat in enumerate(patterns):
        if re.search(pat, answer, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL):
            matched.append(i)
        else:
            missed.append(i)

    score = len(matched) / len(patterns) if patterns else 1.0
    return {
        "score": score,
        "matched_count": len(matched),
        "missed_count": len(missed),
        "total_patterns": len(patterns),
        "correct": len(missed) == 0,
        "matched_indices": matched,
        "missed_indices": missed,
    }


def run_tier1(
    tasks: Sequence[TaskCase],
    modes: Sequence[str],
    gen_client: OpenAI,
    gen_model: str,
) -> dict:
    """Run the full Tier 1 benchmark: compress → generate → judge."""

    all_tool_names = tuple(t.tool_name for t in tasks)
    results: dict[str, list[dict]] = {}
    started = time.perf_counter()

    for mode in modes:
        compressor = build_compressor(mode, force_all_tools=all_tool_names)
        lane_results = []

        for task in tasks:
            # Step 1: Compress
            comp_result = compressor.compress(
                task.tool_name, task.tool_args, task.content, task.question
            )
            context = comp_result.compressed_text
            comp_latency_ms = comp_result.latency_ms

            # Step 2: Generate answer
            gen = generate_answer(
                gen_client, gen_model, task.question, context, task.tool_name
            )

            # Step 3: Judge answer
            expected = EXPECTED_ANSWERS.get(task.id)
            if expected:
                judge = judge_answer(gen["answer"], expected)
            else:
                judge = {"score": -1, "correct": False, "matched_count": 0,
                         "missed_count": 0, "total_patterns": 0,
                         "matched_indices": [], "missed_indices": []}

            # Step 4: Fact survival (reuse existing scorer)
            eval_result = evaluate_text(task, context)

            total_wall_ms = comp_latency_ms + gen["gen_latency_ms"]

            lane_results.append({
                "task_id": task.id,
                "category": task.category,
                "tool_name": task.tool_name,
                "default_llmlingua_candidate": is_default_llmlingua_candidate(task),
                "mode": mode,
                # Compression metrics
                "input_chars": len(task.content),
                "compressed_chars": len(context),
                "compression_ratio": len(task.content) / max(len(context), 1),
                "input_tokens_est": comp_result.original_tokens,
                "compressed_tokens_est": comp_result.compressed_tokens,
                "token_compression_ratio": comp_result.original_tokens / max(comp_result.compressed_tokens, 1),
                "comp_latency_ms": round(comp_latency_ms, 1),
                "fell_back": comp_result.fell_back,
                # Generation metrics
                "gen_input_tokens": gen["input_tokens"],
                "gen_output_tokens": gen["output_tokens"],
                "gen_total_tokens": gen["input_tokens"] + gen["output_tokens"],
                "gen_latency_ms": gen["gen_latency_ms"],
                "answer": gen["answer"],
                # Total wall-clock
                "total_wall_ms": round(total_wall_ms, 1),
                # Judge results
                "judge_score": judge["score"],
                "judge_correct": judge["correct"],
                "judge_matched": judge["matched_count"],
                "judge_missed": judge["missed_count"],
                "judge_total_patterns": judge["total_patterns"],
                # Fact survival (regex)
                "fact_answerable": eval_result["answerable"],
                "required_recall": eval_result["required_recall"],
                "fact_recall": eval_result["fact_recall"],
            })

        results[mode] = lane_results

    return {
        "schema_version": 2,
        "tier": "generation_judge",
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "duration_s": round(time.perf_counter() - started, 1),
        "gen_model": gen_model,
        "modes": list(modes),
        "task_count": len(tasks),
        "results": results,
        "summary": {},
    }


def aggregate_lane(rows: list[dict]) -> dict:
    """Aggregate per-lane results."""
    if not rows:
        return {}
    n = len(rows)
    return {
        "tasks": n,
        "judge_correct": sum(1 for r in rows if r["judge_correct"]),
        "judge_accuracy": sum(1 for r in rows if r["judge_correct"]) / n,
        "judge_score_mean": statistics.fmean(r["judge_score"] for r in rows),
        "fact_answerable": sum(1 for r in rows if r["fact_answerable"]),
        "fact_answerability": sum(1 for r in rows if r["fact_answerable"]) / n,
        "required_recall_mean": statistics.fmean(r["required_recall"] for r in rows),
        "gen_input_tokens_mean": round(statistics.fmean(r["gen_input_tokens"] for r in rows), 1),
        "gen_output_tokens_mean": round(statistics.fmean(r["gen_output_tokens"] for r in rows), 1),
        "gen_total_tokens_mean": round(statistics.fmean(r["gen_total_tokens"] for r in rows), 1),
        "gen_total_tokens_sum": sum(r["gen_total_tokens"] for r in rows),
        "comp_latency_ms_mean": round(statistics.fmean(r["comp_latency_ms"] for r in rows), 1),
        "gen_latency_ms_mean": round(statistics.fmean(r["gen_latency_ms"] for r in rows), 1),
        "total_wall_ms_mean": round(statistics.fmean(r["total_wall_ms"] for r in rows), 1),
        "total_wall_ms_sum": round(sum(r["total_wall_ms"] for r in rows), 1),
        "token_compression_ratio": (
            sum(r["input_tokens_est"] for r in rows)
            / max(sum(r["compressed_tokens_est"] for r in rows), 1)
        ),
        "fell_back_count": sum(1 for r in rows if r["fell_back"]),
    }


def print_tier1_report(payload: dict) -> None:
    """Print a readable summary table."""
    print("\n" + "=" * 90)
    print("Tier 1: LLM Generation + Judge Benchmark")
    print("=" * 90)
    print(f"Model: {payload['gen_model']}   Tasks: {payload['task_count']}   Duration: {payload['duration_s']}s")
    print()

    # Overall
    print(f"{'Mode':22s} {'Judge OK':>10s} {'Accuracy':>10s} {'Score':>8s} {'Facts OK':>10s} "
          f"{'Gen Tok':>10s} {'Wall(ms)':>10s} {'Comp(ms)':>10s}")
    print("-" * 100)
    for mode in payload["modes"]:
        s = payload["summary"].get(mode, {})
        print(
            f"{mode:22s} "
            f"{s.get('judge_correct', 0):>4}/{s.get('tasks', 0):<5} "
            f"{s.get('judge_accuracy', 0):>9.1%} "
            f"{s.get('judge_score_mean', 0):>7.2f} "
            f"{s.get('fact_answerable', 0):>4}/{s.get('tasks', 0):<5} "
            f"{s.get('gen_total_tokens_mean', 0):>10.1f} "
            f"{s.get('total_wall_ms_mean', 0):>10.1f} "
            f"{s.get('comp_latency_ms_mean', 0):>10.1f}"
        )

    # Default-allowlist subset
    print("\nDefault-allowlist subset (web/browser only):")
    print("-" * 100)
    print(f"{'Mode':22s} {'Judge OK':>10s} {'Accuracy':>10s} {'Score':>8s} {'Facts OK':>10s} "
          f"{'Gen Tok':>10s} {'Wall(ms)':>10s} {'Comp(ms)':>10s}")
    print("-" * 100)
    for mode in payload["modes"]:
        rows = [r for r in payload["results"][mode] if r.get("default_llmlingua_candidate")]
        s = aggregate_lane(rows)
        print(
            f"{mode:22s} "
            f"{s.get('judge_correct', 0):>4}/{s.get('tasks', 0):<5} "
            f"{s.get('judge_accuracy', 0):>9.1%} "
            f"{s.get('judge_score_mean', 0):>7.2f} "
            f"{s.get('fact_answerable', 0):>4}/{s.get('tasks', 0):<5} "
            f"{s.get('gen_total_tokens_mean', 0):>10.1f} "
            f"{s.get('total_wall_ms_mean', 0):>10.1f} "
            f"{s.get('comp_latency_ms_mean', 0):>10.1f}"
        )

    # Per-task detail
    print("\nPer-task results:")
    print("-" * 120)
    print(f"{'Task':36s} {'Mode':20s} {'Judge':>6s} {'Score':>7s} {'Facts':>6s} "
          f"{'InTok':>8s} {'OutTok':>7s} {'TotalT':>8s} {'Wall':>8s}")
    print("-" * 120)
    for mode in payload["modes"]:
        for r in payload["results"][mode]:
            judge_mark = "✓" if r["judge_correct"] else "✗"
            fact_mark = "✓" if r["fact_answerable"] else "✗"
            print(
                f"{r['task_id']:36s} {mode:20s} {judge_mark:>6s} "
                f"{r['judge_score']:>6.2f} {fact_mark:>6s} "
                f"{r['gen_input_tokens']:>8d} {r['gen_output_tokens']:>7d} "
                f"{r['gen_total_tokens']:>8d} {r['total_wall_ms']:>7.0f}ms"
            )
        print()


def write_tier1_markdown(payload: dict, path: Path) -> None:
    """Write a PR-ready markdown report."""
    lines = [
        "# Tier 1: LLM Generation + Judge Benchmark",
        "",
        f"Generated: {payload['timestamp']}",
        f"Generation model: {payload['gen_model']}",
        f"Tasks: {payload['task_count']}",
        f"Duration: {payload['duration_s']}s",
        "",
        "## Overall Results",
        "",
        "| Mode | Judge correct | Accuracy | Score | Facts answerable | Avg gen tokens | Avg wall-clock | Avg comp latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in payload["modes"]:
        s = payload["summary"][mode]
        lines.append(
            f"| {mode} | {s['judge_correct']}/{s['tasks']} ({s['judge_accuracy']:.1%}) "
            f"| {s['judge_score_mean']:.2f} "
            f"| {s['fact_answerable']}/{s['tasks']} ({s['fact_answerability']:.1%}) "
            f"| {s['gen_total_tokens_mean']:.0f} "
            f"| {s['total_wall_ms_mean']:.0f} ms "
            f"| {s['comp_latency_ms_mean']:.0f} ms |"
        )

    # Default subset
    lines.extend([
        "",
        "## Default-allowlist subset (web/browser)",
        "",
        "| Mode | Judge correct | Accuracy | Score | Facts answerable | Avg gen tokens | Avg wall-clock | Avg comp latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in payload["modes"]:
        rows = [r for r in payload["results"][mode] if r.get("default_llmlingua_candidate")]
        s = aggregate_lane(rows)
        lines.append(
            f"| {mode} | {s['judge_correct']}/{s['tasks']} ({s['judge_accuracy']:.1%}) "
            f"| {s['judge_score_mean']:.2f} "
            f"| {s['fact_answerable']}/{s['tasks']} ({s['fact_answerability']:.1%}) "
            f"| {s['gen_total_tokens_mean']:.0f} "
            f"| {s['total_wall_ms_mean']:.0f} ms "
            f"| {s['comp_latency_ms_mean']:.0f} ms |"
        )

    # Per-task
    lines.extend([
        "",
        "## Per-task detail",
        "",
        "| Task | Tool | Mode | Judge ✓ | Score | Facts ✓ | Gen tokens | Wall-clock |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for mode in payload["modes"]:
        for r in payload["results"][mode]:
            jm = "✓" if r["judge_correct"] else "✗"
            fm = "✓" if r["fact_answerable"] else "✗"
            lines.append(
                f"| {r['task_id']} | {r['tool_name']} | {mode} | {jm} "
                f"| {r['judge_score']:.2f} | {fm} "
                f"| {r['gen_total_tokens']} | {r['total_wall_ms']:.0f} ms |"
            )

    # Delta analysis
    if "full" in payload["results"] and "llmlingua_default" in payload["results"]:
        full_map = {r["task_id"]: r for r in payload["results"]["full"]}
        lng_map = {r["task_id"]: r for r in payload["results"]["llmlingua_default"]}

        lines.extend([
            "",
            "## Delta: full vs llmlingua_default",
            "",
            "| Task | Full judge | LLMLingua judge | Full tokens | LLMLingua tokens | Token delta | Full wall | LLMLingua wall | Wall delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for tid in full_map:
            f = full_map[tid]
            l = lng_map[tid]
            tok_delta = l["gen_total_tokens"] - f["gen_total_tokens"]
            wall_delta = l["total_wall_ms"] - f["total_wall_ms"]
            fm = "✓" if f["judge_correct"] else "✗"
            lm = "✓" if l["judge_correct"] else "✗"
            lines.append(
                f"| {tid} | {fm} | {lm} "
                f"| {f['gen_total_tokens']} | {l['gen_total_tokens']} | {tok_delta:+d} "
                f"| {f['total_wall_ms']:.0f}ms | {l['total_wall_ms']:.0f}ms | {wall_delta:+.0f}ms |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", default="full,drop,llmlingua_default",
                   help="Comma-separated modes")
    p.add_argument("--gen-model", default="gpt-4o",
                   help="Model for generation pass")
    p.add_argument("--gen-base-url", default=None,
                   help="OpenAI-compatible base URL (default: localhost:8081/v1)")
    p.add_argument("--gen-api-key", default=None,
                   help="API key (default: benchmark)")
    p.add_argument("--json-out", default="benchmarks/results/llmlingua_tier1_generation.json")
    p.add_argument("--markdown-out", default="benchmarks/results/llmlingua_tier1_generation.md")
    p.add_argument("--no-report", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    tasks = task_fixtures()

    client = make_client(args.gen_base_url, args.gen_api_key)

    print(f"Tier 1 generation benchmark: {len(tasks)} tasks × {len(modes)} modes × {args.gen_model}")
    print(f"Modes: {modes}")
    print()

    payload = run_tier1(tasks, modes, client, args.gen_model)

    # Compute summaries
    payload["summary"] = {mode: aggregate_lane(payload["results"][mode]) for mode in modes}

    print_tier1_report(payload)

    if not args.no_report:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path = Path(args.markdown_out)
        write_tier1_markdown(payload, md_path)
        print(f"\nWrote JSON: {json_path}")
        print(f"Wrote report: {md_path}")


if __name__ == "__main__":
    main()
