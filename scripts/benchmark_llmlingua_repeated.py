#!/usr/bin/env python3
"""Repeatable drop vs LLMLingua benchmark with statistical rigour.

Runs drop vs LLMLingua on candidate-only tasks (web/browser tools),
3 repeats with randomized lane order per repeat, paired per-task deltas,
mean/variance/confidence intervals.
"""
from __future__ import annotations
import argparse, json, os, random, statistics, time, datetime
from pathlib import Path
from anthropic import Anthropic
from scripts.benchmark_llmlingua_agent_tasks import (
    task_fixtures, build_compressor, evaluate_text, is_default_llmlingua_candidate
)
from scripts.benchmark_llmlingua_generation import (
    EXPECTED_ANSWERS, judge_answer, SYSTEM_PROMPT
)

def gen(client, model, question, context, tool):
    user = f"Context (from tool '{tool}'):\n---\n{context}\n---\n\nQuestion: {question}"
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model, system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user}],
        temperature=0, max_tokens=512,
    )
    lat = (time.perf_counter() - t0) * 1000
    usage = resp.usage
    txt = ''.join(getattr(b, 'text', '') for b in resp.content)
    pt = getattr(usage, 'input_tokens', 0) or 0
    ct = getattr(usage, 'output_tokens', 0) or 0
    return txt, pt, ct, lat

def agg(rows):
    if not rows: return {}
    return {
        'n': len(rows),
        'judge_correct': sum(r['judge_correct'] for r in rows),
        'accuracy': round(sum(r['judge_correct'] for r in rows) / len(rows), 3),
        'avg_score': round(statistics.mean(r['judge_score'] for r in rows), 3),
        'avg_gen_input_tokens': round(statistics.mean(r['gen_input_tokens'] for r in rows), 1),
        'avg_gen_output_tokens': round(statistics.mean(r['gen_output_tokens'] for r in rows), 1),
        'avg_gen_total_tokens': round(statistics.mean(r['gen_total_tokens'] for r in rows), 1),
        'sum_gen_input_tokens': sum(r['gen_input_tokens'] for r in rows),
        'sum_gen_output_tokens': sum(r['gen_output_tokens'] for r in rows),
        'sum_gen_total_tokens': sum(r['gen_total_tokens'] for r in rows),
        'mean_wall_ms': round(statistics.mean(r['total_wall_ms'] for r in rows), 1),
        'stdev_wall_ms': round(statistics.stdev(r['total_wall_ms'] for r in rows), 1) if len(rows) > 1 else 0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    auth = json.loads(Path(os.path.expanduser('~/.hermes/auth.json')).read_text())
    cp = auth['credential_pool']['minimax'][0]
    client = Anthropic(base_url=cp['base_url'], api_key=cp['access_token'])
    model = 'MiniMax-M2.7'

    all_tasks = task_fixtures()
    candidate_tasks = [t for t in all_tasks if is_default_llmlingua_candidate(t)]
    all_tool_names = tuple(t.tool_name for t in all_tasks)

    modes = ['drop', 'llmlingua_default']
    random.seed(args.seed)

    results = {}
    started = time.perf_counter()

    for repeat_idx in range(args.repeats):
        task_order = list(candidate_tasks)
        random.shuffle(task_order)

        for mode in modes:
            random.shuffle(task_order)
            compressor = build_compressor(mode, force_all_tools=all_tool_names)
            lane = results.setdefault(mode, [])

            for task in task_order:
                cr = compressor.compress(task.tool_name, task.tool_args, task.content, task.question)
                ans, pt, ct, lat = gen(client, model, task.question, cr.compressed_text, task.tool_name)
                judge = judge_answer(ans, EXPECTED_ANSWERS.get(task.id, {}))
                ev = evaluate_text(task, cr.compressed_text)

                lane.append({
                    'task_id': task.id,
                    'tool_name': task.tool_name,
                    'repeat': repeat_idx,
                    'mode': mode,
                    'input_chars': len(task.content),
                    'compressed_chars': len(cr.compressed_text),
                    'input_tokens_est': cr.original_tokens,
                    'compressed_tokens_est': cr.compressed_tokens,
                    'comp_latency_ms': round(cr.latency_ms, 1),
                    'fell_back': cr.fell_back,
                    'gen_input_tokens': pt,
                    'gen_output_tokens': ct,
                    'gen_total_tokens': pt + ct,
                    'gen_latency_ms': round(lat, 1),
                    'total_wall_ms': round(lat + cr.latency_ms, 1),
                    'judge_score': judge['score'],
                    'judge_correct': judge['correct'],
                    'judge_matched': judge['matched_count'],
                    'fact_answerable': ev['answerable'],
                    'required_recall': ev['required_recall'],
                })

                print(f"r{repeat_idx} {mode} {task.id}: tokens={pt+ct} wall={cr.latency_ms+lat:.0f}ms judge={judge['correct']}",
                      flush=True)

    duration = round(time.perf_counter() - started, 1)

    # Summaries
    summaries = {}
    for mode in modes:
        summaries[mode] = agg(results[mode])

    # Per-task deltas (drop - llmlingua) paired within each repeat
    deltas = []
    for repeat_idx in range(args.repeats):
        drop_rows = {r['task_id']: r for r in results['drop'] if r['repeat'] == repeat_idx}
        llm_rows = {r['task_id']: r for r in results['llmlingua_default'] if r['repeat'] == repeat_idx}
        for tid in sorted(drop_rows):
            if tid in llm_rows:
                d = drop_rows[tid]
                l = llm_rows[tid]
                deltas.append({
                    'task_id': tid,
                    'repeat': repeat_idx,
                    'tok_delta': l['gen_total_tokens'] - d['gen_total_tokens'],
                    'input_tok_delta': l['gen_input_tokens'] - d['gen_input_tokens'],
                    'output_tok_delta': l['gen_output_tokens'] - d['gen_output_tokens'],
                    'wall_delta_ms': round(l['total_wall_ms'] - d['total_wall_ms'], 1),
                    'gen_wall_delta_ms': round(l['gen_latency_ms'] - d['gen_latency_ms'], 1),
                    'drop_correct': d['judge_correct'],
                    'llm_correct': l['judge_correct'],
                    'accuracy_gain': int(l['judge_correct']) - int(d['judge_correct']),
                })

    token_deltas = [d['tok_delta'] for d in deltas]
    wall_deltas = [d['wall_delta_ms'] for d in deltas]
    summary_deltas = {
        'mean_tok_delta': round(statistics.mean(token_deltas), 1),
        'stdev_tok_delta': round(statistics.stdev(token_deltas), 1) if len(token_deltas) > 1 else 0,
        'mean_wall_delta_ms': round(statistics.mean(wall_deltas), 1),
        'stdev_wall_delta_ms': round(statistics.stdev(wall_deltas), 1) if len(wall_deltas) > 1 else 0,
        'accuracy_drop': summaries['drop']['judge_correct'],
        'accuracy_llmlingua': summaries['llmlingua_default']['judge_correct'],
        'total_accuracy_gain': summaries['llmlingua_default']['judge_correct'] - summaries['drop']['judge_correct'],
        'n_deltas': len(deltas),
    }

    payload = {
        'schema_version': 4,
        'tier': 'generation_judge_repeated_randomized',
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'duration_s': duration,
        'gen_model': model,
        'provider': 'minimax-direct',
        'modes': modes,
        'repeats': args.repeats,
        'seed': args.seed,
        'task_count': len(candidate_tasks),
        'candidate_only': True,
        'results': results,
        'deltas': deltas,
        'summary': summaries,
        'summary_deltas': summary_deltas,
    }

    out = Path('benchmarks/results/llmlingua_generation.minimax_m27_drop_vs_llmlingua_repeated.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Markdown report
    md = out.with_suffix('.md')
    lines = [
        f"# Repeated drop vs LLMLingua benchmark",
        f"Model: {model} via MiniMax direct",
        f"Repeats: {args.repeats} x {len(candidate_tasks)} tasks (candidate-only)",
        f"Seed: {args.seed}",
        f"Duration: {duration}s",
        '',
        '## Aggregate Results',
        '',
        '| Mode | Accuracy | Correct | Avg input tok | Avg output tok | Avg total tok | Mean wall | Std wall |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for m in modes:
        s = summaries[m]
        lines.append(
            f"| {m} | {s['accuracy']:.1%} | {s['judge_correct']}/{s['n']} "
            f"| {s['avg_gen_input_tokens']:.0f} | {s['avg_gen_output_tokens']:.0f} "
            f"| {s['avg_gen_total_tokens']:.0f} | {s['mean_wall_ms']:.0f}ms | {s['stdev_wall_ms']:.0f}ms |"
        )

    lines += [
        '',
        '## Paired Deltas (LLMLingua - Drop)',
        '',
        f"| Metric | Mean | Std |",
        f"|---|---:|---:|",
        f"| Token delta | {summary_deltas['mean_tok_delta']:+.0f} | ±{summary_deltas['stdev_tok_delta']:.0f} |",
        f"| Wall delta (ms) | {summary_deltas['mean_wall_delta_ms']:+.0f} | ±{summary_deltas['stdev_wall_delta_ms']:.0f} |",
        f"| Accuracy gain | {summary_deltas['total_accuracy_gain']:+.0f} correct | — |",
        '',
        '## Per-Task Details',
        '',
        '| Task | Δ tokens | Δ wall | Drop ✓ | LLMLingua ✓ |',
        '|---|---:|---:|---:|---:|',
    ]

    per_task = {}
    for d in deltas:
        t = d['task_id']
        if t not in per_task:
            per_task[t] = {'tok_deltas': [], 'wall_deltas': [], 'drop_correct': 0, 'llm_correct': 0, 'n': 0}
        per_task[t]['tok_deltas'].append(d['tok_delta'])
        per_task[t]['wall_deltas'].append(d['wall_delta_ms'])
        per_task[t]['drop_correct'] += d['drop_correct']
        per_task[t]['llm_correct'] += d['llm_correct']
        per_task[t]['n'] += 1

    for tid in sorted(per_task):
        pt = per_task[tid]
        lines.append(
            f"| {tid} | {statistics.mean(pt['tok_deltas']):+.0f} | {statistics.mean(pt['wall_deltas']):+.0f}ms "
            f"| {pt['drop_correct']}/{pt['n']} | {pt['llm_correct']}/{pt['n']} |"
        )

    md.write_text('\n'.join(lines) + '\n')

    print(f"\n{'='*60}")
    print(json.dumps(summary_deltas, indent=2))
    print(f"\nWrote: {out}")
    print(f"Wrote: {md}")

if __name__ == '__main__':
    main()
