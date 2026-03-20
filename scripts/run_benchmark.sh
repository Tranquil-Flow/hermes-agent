#!/usr/bin/env bash
# Benchmark runner with sentence-transformers + LLM judge support.
# Run from hermes-agent/ directory.
#
# Usage:
#   ./scripts/run_benchmark.sh              # heuristic judge, all seeds
#   ./scripts/run_benchmark.sh --llm-judge  # LLM judge (claude-haiku-4-5)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Venv with sentence-transformers and anthropic
VENV="$PROJECT_DIR/.venv"
if [ ! -d "$VENV" ]; then
    echo "Creating venv..."
    python3 -m venv "$VENV"
fi

# Install deps if not present
PYTHON="$VENV/bin/python3"
PIP="$VENV/bin/pip3"

if ! "$PYTHON" -c "import sentence_transformers" 2>/dev/null; then
    echo "Installing sentence-transformers..."
    "$PIP" install sentence-transformers anthropic httpx pytest pytest-xdist \
        --quiet --root-user-action=ignore --trusted-host pypi.org --trusted-host files.pythonhosted.org
fi

# HuggingFace cache (persisted in workspace)
export HF_HOME="${PROJECT_DIR}/../.huggingface_cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_SSL_VERIFICATION=1

# Aegis proxy (injects API keys)
export HTTP_PROXY="http://host.docker.internal:8443"
export HTTPS_PROXY="http://host.docker.internal:8443"

JUDGE="heuristic"
for arg in "$@"; do
    if [ "$arg" = "--llm-judge" ]; then
        JUDGE="claude-haiku-4-5"
    fi
done

echo "Running benchmark: judge=$JUDGE embedding=auto"
"$PYTHON" -m benchmarks.runner \
    --backend cognitive \
    --suite a \
    --runs 3 \
    --seeds 42 99 123 \
    --judge-model "$JUDGE" \
    "$@"
