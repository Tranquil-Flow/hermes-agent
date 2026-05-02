"""Pluggable compressor for tool result message bodies.

This module is the seam point introduced by PR 1. It currently exposes
only the abstraction + default ("drop body") implementation; LLMLingua-2
backends land in PR 2.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.model_metadata import estimate_tokens_rough


def summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """Create an informative 1-line summary of a tool call + result.

    Used during the pre-compression pruning pass to replace large tool
    outputs with a short but useful description of what the tool did,
    rather than a generic placeholder that carries zero information.

    Returns strings like::

        [terminal] ran `npm test` -> exit 0, 47 lines output
        [read_file] read config.py from line 1 (1,200 chars)
        [search_files] content search for 'compress' in agent/ -> 12 matches
    """
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        target = args.get("target", "content")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] {target} search for '{pattern}' in {path} -> {count} matches"

    if tool_name == "patch":
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({content_len:,} chars result)"

    if tool_name in ("browser_navigate", "browser_click", "browser_snapshot",
                     "browser_type", "browser_scroll", "browser_vision"):
        url = args.get("url", "")
        ref = args.get("ref", "")
        detail = f" {url}" if url else (f" ref={ref}" if ref else "")
        return f"[{tool_name}]{detail} ({content_len:,} chars)"

    if tool_name == "web_search":
        query = args.get("query", "?")
        return f"[web_search] query='{query}' ({content_len:,} chars result)"

    if tool_name == "web_extract":
        urls = args.get("urls", [])
        url_desc = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(urls, list) and len(urls) > 1:
            url_desc += f" (+{len(urls) - 1} more)"
        return f"[web_extract] {url_desc} ({content_len:,} chars)"

    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({content_len:,} chars result)"

    if tool_name == "execute_code":
        code_preview = (args.get("code") or "")[:60].replace("\n", " ")
        if len(args.get("code", "")) > 60:
            code_preview += "..."
        return f"[execute_code] `{code_preview}` ({line_count} lines output)"

    if tool_name in ("skill_view", "skills_list", "skill_manage"):
        name = args.get("name", "?")
        return f"[{tool_name}] name={name} ({content_len:,} chars)"

    if tool_name == "vision_analyze":
        question = args.get("question", "")[:50]
        return f"[vision_analyze] '{question}' ({content_len:,} chars)"

    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "?")
        return f"[memory] {action} on {target}"

    if tool_name == "todo":
        return "[todo] updated task list"

    if tool_name == "clarify":
        return "[clarify] asked user a question"

    if tool_name == "text_to_speech":
        return f"[text_to_speech] generated audio ({content_len:,} chars)"

    if tool_name == "cronjob":
        action = args.get("action", "?")
        return f"[cronjob] {action}"

    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "?")
        return f"[process] {action} session={sid}"

    # Generic fallback
    first_arg = ""
    for k, v in list(args.items())[:2]:
        sv = str(v)[:40]
        first_arg += f" {k}={sv}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)"


def count_tokens(text: str) -> int:
    """Token count for compression telemetry.

    Wraps the existing char-based rough estimator (~4 chars/token).
    Exact tokenization isn't required for compression-ratio reporting;
    PR 2's LLMLingua backends use their own model-specific tokenizer.
    """
    return estimate_tokens_rough(text)


@dataclass
class CompressionResult:
    """Result of compressing one tool result body."""
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    latency_ms: float
    cache_hit: bool
    fell_back: bool

    @property
    def compression_ratio(self) -> float:
        if self.compressed_tokens == 0:
            return float("inf")
        return self.original_tokens / self.compressed_tokens


class ToolResultCompressor(ABC):
    """Pluggable compressor for tool result message bodies.

    Subclasses implement `compress()`. The default `compress_many()` runs
    compress() in a threadpool via asyncio.gather; subclasses with native
    batch protocols (e.g. an HTTP /compress_batch endpoint) override it.

    `tool_args` is included in the signature so the default-behavior
    DropBodyCompressor can produce summaries that reference the call's
    args (command, path, query, etc.), matching today's
    _summarize_tool_result output. LLMLingua-style backends are free
    to ignore tool_args.
    """

    @abstractmethod
    def compress(
        self,
        tool_name: str,
        tool_args: str,
        content: str,
        question: Optional[str] = None,
    ) -> CompressionResult:
        """Compress one tool result body. Synchronous. Must not raise on
        bad input — fall back to a degraded result and set fell_back=True."""

    async def compress_many(
        self,
        items: List[Tuple[str, str, str]],
        question: Optional[str] = None,
    ) -> List[CompressionResult]:
        """Concurrent compression. Default: thread-pool over compress().

        items: list of (tool_name, tool_args, content) tuples.
        """
        loop = asyncio.get_running_loop()
        return list(await asyncio.gather(*[
            loop.run_in_executor(
                None, self.compress, tool_name, tool_args, content, question
            )
            for tool_name, tool_args, content in items
        ]))


class DropBodyCompressor(ToolResultCompressor):
    """Default compressor: replaces tool body with a 1-line summary.

    This is the historical hermes-agent behavior, wrapped in the new
    ToolResultCompressor interface. No information is preserved beyond the
    summary line — see LLMLinguaLocalCompressor (PR 2) for an alternative
    that preserves a high-information subset of the original content.
    """

    def compress(
        self,
        tool_name: str,
        tool_args: str,
        content: str,
        question: Optional[str] = None,
    ) -> CompressionResult:
        t0 = time.perf_counter()
        summary = summarize_tool_result(tool_name, tool_args, content)
        return CompressionResult(
            compressed_text=summary,
            original_tokens=count_tokens(content),
            compressed_tokens=count_tokens(summary),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cache_hit=False,
            fell_back=False,
        )


def make_tool_result_compressor(
    config: Optional[dict],
) -> ToolResultCompressor:
    """Construct a ToolResultCompressor from a config dict.

    Config schema (full schema documented in spec §11):
      method: 'drop' | 'llmlingua2_local' | 'llmlingua2_remote'    # default: 'drop'
      ... (additional keys consumed by specific implementations in PR 2)

    PR 1 only supports 'drop'. The llmlingua2_* methods raise
    NotImplementedError so users opting in before PR 2 lands get a
    clear error instead of a silent fallback.
    """
    config = config or {}
    method = config.get("method", "drop")

    if method == "drop":
        return DropBodyCompressor()
    if method in ("llmlingua2_local", "llmlingua2_remote"):
        raise NotImplementedError(
            f"tool_compression.method={method!r} requires PR 2 "
            "(LLMLingua-2 backends are not yet available)"
        )
    raise ValueError(f"unknown tool_compression method: {method!r}")
