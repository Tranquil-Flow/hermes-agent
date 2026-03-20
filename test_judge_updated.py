#!/usr/bin/env python3
"""Test the updated LLM judge prompt against hard cross-reference scenarios."""
import json
import sys
sys.path.insert(0, '/workspace/Projects/hermes-agent')
from benchmarks.judge import MemoryJudge, HeuristicJudge

with open('/workspace/Projects/hermes-agent/benchmarks/suite_a/fixtures/cross_reference.json') as fp:
    scenarios = json.load(fp)

llm_judge = MemoryJudge(model='claude-haiku-4-5')
heur_judge = HeuristicJudge()

hard = [s for s in scenarios if s['difficulty'] == 'hard']
print(f'Testing {len(hard)} hard cross-reference scenarios')
print()

llm_correct = 0
heur_correct = 0

for sc in hard:
    sid = sc['id']
    facts_str = ' | '.join(sc['facts'][:sc['num_facts_needed']])
    heur = heur_judge.judge_answer(sc['query'], sc['gold_answer'], facts_str)
    llm = llm_judge.judge_answer(sc['query'], sc['gold_answer'], facts_str)
    if heur.correct:
        heur_correct += 1
    if llm.correct:
        llm_correct += 1
    agree = 'AGREE' if heur.correct == llm.correct else 'DISAGREE'
    print(f'{sid} ({agree}): heur={heur.correct} llm={llm.correct}')
    if not llm.correct:
        print(f'  Q: {sc["query"][:80]}')
        print(f'  Gold: {sc["gold_answer"]}')
        print(f'  LLM reason: {llm.raw_response[:100]}')

print()
print(f'Summary: heuristic={heur_correct}/{len(hard)} LLM={llm_correct}/{len(hard)}')
