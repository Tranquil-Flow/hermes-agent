#!/usr/bin/env python3
"""Run only contradiction tests with LLM judge enabled."""
import sys
import os
import json
import time

sys.path.insert(0, '/workspace/Projects/hermes-agent')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/workspace/Projects/.huggingface_cache'

from cognitive_memory.store import CognitiveMemoryStore
from cognitive_memory.config import CognitiveMemoryConfig
from benchmarks.judge import judge_answer

# Load contradiction fixtures
with open('/workspace/Projects/hermes-agent/benchmarks/suite_a/fixtures/contradictions.json') as f:
    fixtures = json.load(f)

print(f"Running {len(fixtures)} contradiction tests with LLM judge (claude-haiku-4-5)")
print()

config = CognitiveMemoryConfig()
config.contradiction_llm_model = "claude-haiku-4-5"

passed = 0
failed = 0
results = []

for i, fix in enumerate(fixtures):
    store = CognitiveMemoryStore(config=config)

    # Ingest memories
    for mem in fix.get('memories', []):
        store.store(mem['content'], metadata=mem.get('metadata', {}))

    # Query
    query = fix['query']
    expected = fix['expected_answer']
    scenario = fix.get('scenario', fix.get('id', f'test_{i}'))

    recalled = store.recall(query, top_k=3)
    recalled_text = [m['content'] for m in recalled]

    verdict = judge_answer(
        question=query,
        expected=expected,
        recalled=recalled_text,
        model="claude-haiku-4-5"
    )

    status = "PASS" if verdict else "FAIL"
    if verdict:
        passed += 1
    else:
        failed += 1
        results.append({
            'scenario': scenario,
            'query': query,
            'expected': expected,
            'recalled': recalled_text[:2]
        })

    print(f"[{status}] {scenario}")

print(f"\n{passed}/{passed+failed} passed ({100*passed/(passed+failed):.1f}%)")
if failed > 0:
    print("\nFailed cases:")
    for r in results:
        print(f"  {r['scenario']}: expected={r['expected'][:60]}")
        for rc in r['recalled']:
            print(f"    recalled: {rc[:80]}")
