"""Pluggable compressor for tool result message bodies.

This module is the seam point used by ``ContextCompressor._prune_old_tool_results``.
It exposes:

- ``ToolResultCompressor`` — the ABC.
- ``DropBodyCompressor`` — the default; replaces a long tool body with a
  one-line summary (the historical hermes-agent behaviour).
- ``LLMLinguaLocalCompressor`` — opt-in, in-process LLMLingua-2 backend
  that preserves a high-information subset of the tool body. Requires
  ``pip install hermes-agent[llmlingua]``.
- ``LLMLinguaRemoteCompressor`` — opt-in, sidecar HTTP backend.

Default behaviour is unchanged from before this module existed.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.model_metadata import estimate_tokens_rough

logger = logging.getLogger(__name__)


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
        """Async concurrent compression. Default: threadpool over compress().

        items: list of (tool_name, tool_args, content) tuples. This is the
        forward-compatible API for async callers. Sync callers should use
        ``compress_batch`` instead — it doesn't require a running event loop.
        """
        loop = asyncio.get_running_loop()
        return list(await asyncio.gather(*[
            loop.run_in_executor(
                None, self.compress, tool_name, tool_args, content, question
            )
            for tool_name, tool_args, content in items
        ]))

    def compress_batch(
        self,
        items: List[Tuple[str, str, str]],
        question: Optional[str] = None,
    ) -> List[CompressionResult]:
        """Synchronous batch compression. Default impl is sequential.

        Subclasses with a parallel-safe transport (e.g. HTTP) should
        override this with concurrent dispatch — see
        ``LLMLinguaRemoteCompressor.compress_batch``.

        Stays sync so callers like ``_prune_old_tool_results`` (which run
        in synchronous code paths) don't need the
        ``asyncio.run``-inside-running-loop dance.
        """
        return [
            self.compress(tn, ta, c, question)
            for tn, ta, c in items
        ]


class DropBodyCompressor(ToolResultCompressor):
    """Default compressor: replaces tool body with a 1-line summary.

    This is the historical hermes-agent behavior, wrapped in the new
    ToolResultCompressor interface. No information is preserved beyond the
    summary line — see LLMLinguaLocalCompressor for an alternative that
    preserves a high-information subset of the original content.
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


# ---------------------------------------------------------------------------
# LLMLingua-2 backends
# ---------------------------------------------------------------------------

# Tools whose output is rich natural-language text that compresses well.
# Other tools (terminal, read_file, etc.) keep the drop-body behaviour
# because their output is already information-dense or highly structured.
_DEFAULT_LLMLINGUA_TOOLS: Tuple[str, ...] = (
    "web_extract",
    "web_search",
    "browser_navigate",
    "browser_snapshot",
    "browser_vision",
    "firecrawl_scrape",
    "firecrawl_extract",
)

# Size-adaptive rate ladder. Each entry is (max_chars_inclusive, rate). The
# last entry is the catch-all (max_chars=None). Smaller outputs get gentler
# compression to preserve quality; larger outputs get more aggressive
# compression because they have more boilerplate to strip.
_DEFAULT_RATE_LADDER: Tuple[Tuple[Optional[int], float], ...] = (
    (10_000, 0.50),
    (30_000, 0.33),
    (None,   0.25),
)

# Default LLMLingua-2 model — bert-base multilingual, ~280MB, runs at
# ~250µs/token on Apple Silicon CPU. The xlm-roberta-large variant is
# ~3-5x slower for ~5-15% better compression — not worth it for this
# workload, see _llmlingua_bench/.
_DEFAULT_LLMLINGUA_MODEL = (
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)

# Force-tokens passed to compress_prompt(). Preserves sentence boundaries,
# markdown table separators, headers, and list bullets. Without these the
# output collapses to plain prose and loses structure that downstream LLMs
# use for reasoning.
_FORCE_TOKENS: Tuple[str, ...] = ("\n", "|", "#", "-", ".", "?", "!")

# URL pattern. Matches http(s) URLs up to a trailing whitespace or common
# closing bracket. Imperfect for URLs containing parens, but good enough for
# the placeholder-and-restore pattern we use in compress_with_wrapper().
_URL_RE = re.compile(r'https?://[^\s\)\]\}<>"\']+')


@dataclass
class LLMLinguaConfig:
    """Configuration for the LLMLingua backends.

    Constructed from the user's ``tool_compression`` config dict by the
    factory below; full schema is documented in the design spec.
    """
    method: str = "llmlingua2_local"
    model: str = _DEFAULT_LLMLINGUA_MODEL
    device: str = "cpu"
    only_tools: Tuple[str, ...] = _DEFAULT_LLMLINGUA_TOOLS
    min_output_chars: int = 2_000
    rate: Optional[float] = None  # None → use rate_ladder
    rate_ladder: Tuple[Tuple[Optional[int], float], ...] = _DEFAULT_RATE_LADDER
    use_question: bool = True
    cache_size: int = 256
    # Remote-only:
    endpoint: str = ""
    timeout_secs: float = 15.0
    fallback_method: str = "drop"


def compress_with_wrapper(
    pc: Any,
    content: str,
    rate: float,
    question: Optional[str],
    *,
    force_tokens: Tuple[str, ...] = _FORCE_TOKENS,
) -> str:
    """Compress ``content`` via LLMLingua-2 with structure-preserving wrapper.

    Steps:
      1. Pre-extract URLs to ``URLREF000``-style placeholders so the
         classifier (trained on MeetingBank, treats URLs as noise) can't
         drop them.
      2. Run ``compress_prompt`` with ``force_reserve_digit=True`` and
         markdown-structure ``force_tokens``.
      3. Restore URL placeholders, then append any URLs that the
         compressor still dropped as a ``[REFS: ...]`` block to guarantee
         100% URL preservation.

    Honours the ``question is None`` semantics (spec §7): when ``question``
    is None or empty, the kwarg is omitted entirely rather than coerced to
    an empty string, so LLMLingua-2's question-aware path is fully bypassed
    instead of running on empty input.

    ``pc`` is an instantiated ``llmlingua.PromptCompressor`` — passed in so
    callers can share a single model load and so unit tests can inject a
    fake.
    """
    # 1) URL placeholders
    urls = _URL_RE.findall(content)
    mapping: Dict[str, str] = {}
    pre = content
    for i, u in enumerate(urls):
        ph = f"URLREF{i:03d}"
        mapping[ph] = u
        pre = pre.replace(u, ph, 1)

    # 2) Compress
    kwargs: Dict[str, Any] = dict(
        rate=rate,
        force_reserve_digit=True,
        force_tokens=list(force_tokens),
    )
    if question:  # omit kwarg entirely when None or empty
        kwargs["question"] = question
    out = pc.compress_prompt(pre, **kwargs)["compressed_prompt"]

    # 3) Restore URLs
    for ph, u in mapping.items():
        out = out.replace(ph, u)

    # 4) Append any URLs the compressor still dropped
    surviving = set(_URL_RE.findall(out))
    missing = [u for u in urls if u not in surviving]
    if missing:
        out += "\n\n[REFS: " + " ".join(missing) + "]"

    return out


class _ContentLRU:
    """Tiny thread-safe LRU cache for compressor results. Per-process, bounded.

    Stores ``CompressionResult`` values. Lookups update recency. Eviction
    is least-recently-used.

    The cache key is derived from every input that affects compressor
    output: content, question (when use_question is on), tool_name (gating),
    and rate (per-size selection). Keying on content alone would cause
    a session that re-uses a previously-extracted page under a different
    user question to receive the prior session's compressed output.

    All read/write operations hold an internal lock so the cache is safe
    to share across threads — required by ``LLMLinguaRemoteCompressor.compress_batch``,
    which dispatches HTTP calls concurrently via ThreadPoolExecutor.
    """
    def __init__(self, max_size: int):
        self._max_size = max(1, int(max_size))
        self._store: "OrderedDict[str, CompressionResult]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key_for(
        content: str,
        *,
        question: Optional[str] = None,
        tool_name: str = "",
        rate: Optional[float] = None,
    ) -> str:
        """Hash all inputs that affect compressor output.

        ``question`` of None and "" are treated identically (both mean
        "no question conditioning"). ``rate`` is rounded to 4 decimals
        before hashing to avoid float-precision cache misses.
        """
        h = hashlib.sha256()
        h.update(content.encode("utf-8", errors="replace"))
        h.update(b"\x00question:")
        if question:
            h.update(question.encode("utf-8", errors="replace"))
        h.update(b"\x00tool:")
        h.update(tool_name.encode("utf-8", errors="replace"))
        h.update(b"\x00rate:")
        if rate is not None:
            h.update(f"{rate:.4f}".encode("ascii"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[CompressionResult]:
        with self._lock:
            value = self._store.get(key)
            if value is None:
                return None
            self._store.move_to_end(key)
            return value

    def put(self, key: str, value: CompressionResult) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
                return
            self._store[key] = value
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Drop all entries. Useful on session reset or test setup."""
        with self._lock:
            self._store.clear()


def _validate_compression(original: str, compressed: str) -> None:
    """Raise ValueError on pathological compressor output.

    Cheap sanity checks: empty output, output longer than input, or output
    containing model control sentinels. On failure the caller falls back
    to DropBodyCompressor for that one message.
    """
    if not compressed:
        raise ValueError("compressor returned empty output")
    if len(compressed) > len(original) * 1.05:
        # Allow a small overhead for the [REFS: ...] appendix, but a
        # genuinely-longer output means the compressor failed.
        raise ValueError(
            f"compressor returned more content than input: {len(compressed)} > {len(original)}"
        )
    if "<|endoftext|>" in compressed and "<|endoftext|>" not in original:
        raise ValueError("compressor leaked control sentinel <|endoftext|>")


class LLMLinguaLocalCompressor(ToolResultCompressor):
    """In-process LLMLingua-2 compressor.

    Loads a small token-classifier (~280MB) on first ``compress()`` call,
    then keeps it resident for the process lifetime. Compresses tool
    results from ``config.only_tools`` whose body is at least
    ``config.min_output_chars``; everything else is delegated to
    DropBodyCompressor.

    The backing model is loaded lazily so the module imports cleanly even
    when the ``[llmlingua]`` extra is not installed — the import only
    fails on first compress(), and is caught and falls back to drop-body.
    """

    def __init__(self, config: LLMLinguaConfig):
        self.config = config
        self._pc: Any = None  # PromptCompressor — lazy
        self._fallback = DropBodyCompressor()
        self._cache = _ContentLRU(config.cache_size)
        # If we ever fail to load the model, remember and fall back forever
        # rather than retrying on every call.
        self._model_load_failed = False

    def _ensure_loaded(self) -> None:
        if self._pc is not None or self._model_load_failed:
            return
        try:
            from llmlingua import PromptCompressor  # type: ignore
        except ImportError as e:
            self._model_load_failed = True
            logger.warning(
                "LLMLingua-2 not installed; falling back to drop-body. "
                "Run `pip install hermes-agent[llmlingua]` to enable. (%s)", e,
            )
            raise
        self._pc = PromptCompressor(
            model_name=self.config.model,
            use_llmlingua2=True,
            device_map=self.config.device,
        )

    def _rate_for_size(self, char_count: int) -> float:
        if self.config.rate is not None:
            return self.config.rate
        for max_chars, rate in self.config.rate_ladder:
            if max_chars is None or char_count <= max_chars:
                return rate
        return 0.33

    def _delegate_to_fallback(self, *args, **kwargs) -> CompressionResult:
        return self._fallback.compress(*args, **kwargs)

    def compress(
        self,
        tool_name: str,
        tool_args: str,
        content: str,
        question: Optional[str] = None,
    ) -> CompressionResult:
        # 1. Tool allowlist
        if tool_name not in self.config.only_tools:
            return self._delegate_to_fallback(tool_name, tool_args, content, question)

        # 2. Size gate
        if len(content) < self.config.min_output_chars:
            return self._delegate_to_fallback(tool_name, tool_args, content, question)

        # 3. Cache lookup — keyed on every input that affects output
        rate = self._rate_for_size(len(content))
        q = question if (self.config.use_question and question) else None
        key = self._cache.key_for(content, question=q, tool_name=tool_name, rate=rate)
        cached = self._cache.get(key)
        if cached is not None:
            return dataclasses.replace(cached, cache_hit=True)

        # 4. Compress with fallback on any failure
        t0 = time.perf_counter()
        try:
            self._ensure_loaded()
            compressed = compress_with_wrapper(self._pc, content, rate, q)
            _validate_compression(content, compressed)
        except Exception as e:
            logger.warning(
                "LLMLingua compression failed (%s: %s); falling back to drop-body",
                type(e).__name__, e,
            )
            fallback = self._delegate_to_fallback(tool_name, tool_args, content, question)
            return dataclasses.replace(fallback, fell_back=True)

        result = CompressionResult(
            compressed_text=compressed,
            original_tokens=count_tokens(content),
            compressed_tokens=count_tokens(compressed),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cache_hit=False,
            fell_back=False,
        )
        self._cache.put(key, result)
        return result


_REMOTE_BACKOFF_SECS = 60.0  # cooldown after a remote failure before retrying
_REMOTE_BATCH_MAX_WORKERS = 8  # concurrent HTTP calls to the compressor service


class LLMLinguaRemoteCompressor(ToolResultCompressor):
    """Sidecar HTTP backend.

    Posts ``{text, rate, question, force_urls}`` to a remote compressor
    service that runs LLMLingua-2 + the wrapper internally. Falls back to
    DropBodyCompressor on any network failure, malformed response, or
    HTTP error, and applies a process-local cooldown so a dead service
    doesn't add ``timeout_secs`` of latency to every prune for the next
    several minutes.

    Uses ``urllib`` only — no aiohttp / httpx dep — so this stays an
    optional opt-in with zero new heavy imports.
    """

    def __init__(self, config: LLMLinguaConfig):
        if not config.endpoint:
            raise ValueError(
                "LLMLinguaRemoteCompressor requires "
                "tool_compression.endpoint to be set"
            )
        self.config = config
        self._fallback = DropBodyCompressor()
        self._cache = _ContentLRU(config.cache_size)
        self._service_down_until: float = 0.0

    def _rate_for_size(self, char_count: int) -> float:
        if self.config.rate is not None:
            return self.config.rate
        for max_chars, rate in self.config.rate_ladder:
            if max_chars is None or char_count <= max_chars:
                return rate
        return 0.33

    def _delegate_to_fallback(self, *args, **kwargs) -> CompressionResult:
        return self._fallback.compress(*args, **kwargs)

    def _post(
        self, text: str, rate: float, question: Optional[str]
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "text": text,
            "rate": rate,
            "force_urls": True,
        }
        if question:
            body["question"] = question
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.config.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_secs) as resp:
            payload = resp.read()
        return json.loads(payload)

    def compress(
        self,
        tool_name: str,
        tool_args: str,
        content: str,
        question: Optional[str] = None,
    ) -> CompressionResult:
        # 1. Tool allowlist
        if tool_name not in self.config.only_tools:
            return self._delegate_to_fallback(tool_name, tool_args, content, question)

        # 2. Size gate
        if len(content) < self.config.min_output_chars:
            return self._delegate_to_fallback(tool_name, tool_args, content, question)

        # 3. Cooldown (service was recently down)
        if time.time() < self._service_down_until:
            return dataclasses.replace(
                self._delegate_to_fallback(tool_name, tool_args, content, question),
                fell_back=True,
            )

        # 4. Cache lookup — keyed on every input that affects output
        rate = self._rate_for_size(len(content))
        q = question if (self.config.use_question and question) else None
        key = self._cache.key_for(content, question=q, tool_name=tool_name, rate=rate)
        cached = self._cache.get(key)
        if cached is not None:
            return dataclasses.replace(cached, cache_hit=True)

        # 5. Remote call with fallback on any failure
        t0 = time.perf_counter()
        try:
            response = self._post(content, rate, q)
            compressed = response.get("compressed")
            if not isinstance(compressed, str):
                raise ValueError("missing 'compressed' field in response")
            _validate_compression(content, compressed)
        except (urllib.error.URLError, socket.timeout,
                json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning(
                "remote compressor failed (%s: %s); falling back for %.0fs",
                type(e).__name__, e, _REMOTE_BACKOFF_SECS,
            )
            self._service_down_until = time.time() + _REMOTE_BACKOFF_SECS
            fallback = self._delegate_to_fallback(tool_name, tool_args, content, question)
            return dataclasses.replace(fallback, fell_back=True)

        result = CompressionResult(
            compressed_text=compressed,
            original_tokens=count_tokens(content),
            compressed_tokens=count_tokens(compressed),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cache_hit=False,
            fell_back=False,
        )
        self._cache.put(key, result)
        return result

    def compress_batch(
        self,
        items: List[Tuple[str, str, str]],
        question: Optional[str] = None,
    ) -> List[CompressionResult]:
        """Concurrent HTTP batch — the remote service handles parallel calls fine.

        Falls back to a sequential loop for trivial inputs (1 item) so we
        don't pay threadpool setup cost on the common short prune.
        """
        if not items:
            return []
        if len(items) == 1:
            tn, ta, c = items[0]
            return [self.compress(tn, ta, c, question)]
        from concurrent.futures import ThreadPoolExecutor
        n_workers = min(_REMOTE_BATCH_MAX_WORKERS, len(items))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(self.compress, tn, ta, c, question)
                for tn, ta, c in items
            ]
            return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _build_llmlingua_config(method: str, raw: Dict[str, Any]) -> LLMLinguaConfig:
    """Translate the raw user config dict into a typed LLMLinguaConfig.

    Unknown keys are ignored (forward-compatibility). Only the
    ``rate_ladder`` field is reshaped — users write a list of dicts
    ``[{"max_chars": 10000, "rate": 0.5}, ...]`` and we convert it to the
    tuple-of-tuples form the runtime uses.
    """
    cfg = LLMLinguaConfig(method=method)
    for fld in dataclasses.fields(LLMLinguaConfig):
        if fld.name == "rate_ladder":
            continue
        if fld.name in raw:
            setattr(cfg, fld.name, raw[fld.name])
    if "only_tools" in raw and isinstance(raw["only_tools"], list):
        cfg.only_tools = tuple(raw["only_tools"])
    if "rate_ladder" in raw and isinstance(raw["rate_ladder"], list):
        ladder: List[Tuple[Optional[int], float]] = []
        for item in raw["rate_ladder"]:
            if not isinstance(item, dict):
                continue
            mc = item.get("max_chars")
            r = item.get("rate", 0.33)
            ladder.append((mc, float(r)))
        if ladder:
            cfg.rate_ladder = tuple(ladder)
    return cfg


def make_tool_result_compressor(
    config: Optional[dict],
) -> ToolResultCompressor:
    """Construct a ToolResultCompressor from a user config dict.

    Config schema (full schema documented in the design spec)::

        tool_compression:
          method: drop                    # default — current behaviour
          # method: llmlingua2_local      # opt-in, in-process
          # method: llmlingua2_remote     # opt-in, sidecar HTTP
          model: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
          device: cpu
          only_tools: [web_extract, web_search, browser_snapshot, ...]
          min_output_chars: 2000
          rate_ladder:
            - {max_chars: 10000, rate: 0.5}
            - {max_chars: 30000, rate: 0.33}
            - {max_chars: null,  rate: 0.25}
          # rate: 0.33                    # explicit — overrides rate_ladder
          use_question: true
          cache_size: 256
          endpoint: http://your-compressor-host:8080/compress    # remote only
          timeout_secs: 15
          fallback_method: drop
    """
    config = config or {}
    method = config.get("method", "drop")

    if method == "drop":
        return DropBodyCompressor()

    if method == "llmlingua2_local":
        cfg = _build_llmlingua_config(method, config)
        return LLMLinguaLocalCompressor(cfg)

    if method == "llmlingua2_remote":
        cfg = _build_llmlingua_config(method, config)
        return LLMLinguaRemoteCompressor(cfg)

    raise ValueError(f"unknown tool_compression method: {method!r}")
