#!/bin/bash
# Run Suite A benchmark with LLM judge and LLM contradiction detection
# Logs output so we can check results even if terminal times out

cd /workspace/Projects/hermes-agent
source .venv/bin/activate

echo "Starting Suite A benchmark at $(date)"
python3 -m benchmarks.runner \
  --backend cognitive \
  --suite a \
  --runs 1 \
  --seeds 42 \
  --judge-model claude-haiku-4-5 \
  --contradiction-llm claude-haiku-4-5 \
  2>&1 | tee /tmp/benchmark_suite_a.log

echo "Done at $(date)"
echo "Exit code: $?"
