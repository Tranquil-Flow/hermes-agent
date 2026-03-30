#!/bin/bash
# Capture full baseline benchmark snapshot
cd /workspace/Projects/hermes-agent
source .venv/bin/activate 2>/dev/null
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== BASELINE BENCHMARK SNAPSHOT $(date '+%Y-%m-%d %H:%M') ==="
echo ""

for suite in a b c d e f; do
    echo "--- Suite ${suite^^} ---"
    python3 -m benchmarks.runner --backend cognitive --suite $suite --runs 1 --seeds 42 2>&1 | grep -E "Overall:|^\s+\w+\s+[0-9]\." 
    echo ""
done

echo "=== Tests ==="
python3 -m pytest tests/cognitive_memory/ -q --tb=no 2>&1 | tail -3
