#!/usr/bin/env python3
"""Standalone benchmark for LLMLingua-2 tool-result compression.

Run from repo root:
    python scripts/benchmark_llmlingua_tool_compression.py

Requires the ``[llmlingua]`` extra and a local model cache.  Set
``HF_HUB_OFFLINE=1`` if the model is already cached to avoid network
round-trips.

Compares three compression modes against representative Hermes tool outputs:
  * ``drop``   – the historical DropBodyCompressor baseline
  * ``default``– LLMLingua-2 with the PR default allowlist
  * ``all``    – LLMLingua-2 forced on every tool

Reports per-sample compression ratio, throughput, and wall-clock latency.
"""

from __future__ import annotations

import json
import time
from typing import Dict

# ---------------------------------------------------------------------------
# Representative tool outputs (synthetic but realistic)
# ---------------------------------------------------------------------------

SAMPLES: Dict[str, Dict[str, str]] = {
    "web_prose": {
        "tool_name": "web_extract",
        "tool_args": '{"urls": ["https://example.com/article"]}',
        "content": (
            "The quick brown fox jumps over the lazy dog. " * 200  # ~10K chars
        ),
    },
    "json_api_payload": {
        "tool_name": "web_extract",
        "tool_args": '{"urls": ["https://api.example.com/v2/data"]}',
        "content": json.dumps(
            [{"id": i, "name": f"item_{i}", "description": "x" * 100} for i in range(200)]
        ),
    },
    "terminal_log": {
        "tool_name": "terminal",
        "tool_args": '{"command": "make -j8"}',
        "content": "\n".join(
            f"[{i:04d}] CC src/module_{i % 20}.c" for i in range(500)
        ),
    },
    "browser_snapshot": {
        "tool_name": "browser_snapshot",
        "tool_args": "{}",
        "content": "\n".join(
            f"[@e{i}] <div class='row'>Link {i}: https://example.com/page/{i}</div>"
            for i in range(300)
        ),
    },
    "code_search": {
        "tool_name": "search_files",
        "tool_args": '{"pattern": "compress", "path": "/repo/src"}',
        "content": "\n".join(
            f"{i * 10 + 1}|    def compress_{i}(data: bytes) -> bytes:\n"
            f"{i * 10 + 2}|        return zlib.compress(data, level={i % 9})\n"
            f"{i * 10 + 3}|\n"
            for i in range(100)
        ),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_compressor(mode: str):
    """Return a ToolResultCompressor for the given mode."""
    from agent.tool_result_compressor import (
        LLMLinguaConfig,
        ToolResultCompressor,
    )

    if mode == "drop":
        return ToolResultCompressor(method="drop")

    cfg = LLMLinguaConfig()
    if mode == "all":
        # Force all tools through LLMLingua-2
        cfg = LLMLinguaConfig(only_tools=())

    return ToolResultCompressor(
        method="llmlingua2_local",
        llmlingua_config=cfg,
    )


def _run_sample(comp, sample: Dict[str, str]) -> Dict:
    content = sample["content"]
    tool_name = sample["tool_name"]
    tool_args = sample["tool_args"]
    input_chars = len(content)

    t0 = time.perf_counter()
    result = comp.compress(
        tool_name=tool_name,
        tool_args=tool_args,
        content=content,
        question="What information is available?",
    )
    elapsed = time.perf_counter() - t0

    return {
        "input_chars": input_chars,
        "output_chars": len(result.compressed),
        "ratio": input_chars / max(len(result.compressed), 1),
        "latency_s": round(elapsed, 3),
        "method_used": result.method,
    }


def main() -> None:
    modes = ["drop", "default", "all"]
    results: Dict[str, Dict[str, Dict]] = {}

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"Mode: {mode}")
        print(f"{'=' * 60}")
        try:
            comp = _build_compressor(mode)
        except Exception as exc:
            print(f"  SKIP (init failed): {exc}")
            continue

        mode_results: Dict[str, Dict] = {}
        for name, sample in SAMPLES.items():
            print(f"  {name:25s} ... ", end="", flush=True)
            try:
                r = _run_sample(comp, sample)
                mode_results[name] = r
                print(
                    f"{r['input_chars']:>8d} → {r['output_chars']:>8d}  "
                    f"({r['ratio']:.2f}x)  {r['latency_s']:.3f}s  [{r['method_used']}]"
                )
            except Exception as exc:
                print(f"ERROR: {exc}")
                mode_results[name] = {"error": str(exc)}

        results[mode] = mode_results

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Sample':25s}  {'Drop':>10s}  {'Default':>10s}  {'All':>10s}")
    print("-" * 60)
    for name in SAMPLES:
        row = []
        for mode in modes:
            r = results.get(mode, {}).get(name, {})
            if "ratio" in r:
                row.append(f"{r['ratio']:.2f}x")
            else:
                row.append("ERR")
        print(f"{name:25s}  {row[0]:>10s}  {row[1]:>10s}  {row[2]:>10s}")


if __name__ == "__main__":
    main()
