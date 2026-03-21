"""
LLM-based contradiction detection for cognitive memory.

Asks an LLM whether two facts about the same topic contradict each other.
Falls back gracefully if anthropic is unavailable or the call fails.
"""

import os
import logging
from functools import lru_cache
from typing import Tuple

logger = logging.getLogger(__name__)

# Simple LRU cache using a dict (bounded to avoid unbounded growth)
_CACHE_MAX = 512
_cache: dict[Tuple[str, str], bool] = {}
_cache_order: list[Tuple[str, str]] = []

CONTRADICTION_PROMPT = """\
Fact A (existing): {existing}
Fact B (newer): {new}

Does Fact B update, replace, or invalidate Fact A?
Answer with exactly one word: CONTRADICTS or COMPATIBLE"""


def _cache_get(key: Tuple[str, str]) -> bool | None:
    return _cache.get(key)


def _cache_set(key: Tuple[str, str], value: bool) -> None:
    if key in _cache:
        return
    if len(_cache_order) >= _CACHE_MAX:
        oldest = _cache_order.pop(0)
        _cache.pop(oldest, None)
    _cache[key] = value
    _cache_order.append(key)


def _detect_proxy_port() -> int:
    """Read the aegis proxy port from its pid file, fall back to 8444."""
    import json as _json
    pid_file = os.path.expanduser("~/.hermes-aegis/proxy.pid")
    # Also check container-mounted path
    for path in [pid_file, "/Users/evinova/.hermes-aegis/proxy.pid"]:
        try:
            with open(path) as f:
                data = _json.load(f)
                return data.get("port", 8444)
        except (FileNotFoundError, ValueError, KeyError):
            continue
    return 8444


def _build_client(api_key: str | None = None):
    """Build an Anthropic client, routing through aegis proxy if available."""
    import anthropic
    import httpx

    ca_cert = "/certs/mitmproxy-ca-cert.pem"

    if os.path.exists(ca_cert):
        port = _detect_proxy_port()
        proxy_url = f"http://host.docker.internal:{port}"
        http_client = httpx.Client(proxy=proxy_url, verify=ca_cert)
        client = anthropic.Anthropic(
            api_key=api_key or "placeholder-aegis-injects",
            http_client=http_client,
        )
        logger.debug("llm_contradiction: using aegis proxy at port %d", port)
    else:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=resolved_key)
        logger.debug("llm_contradiction: using direct API connection")

    return client


def check_contradiction_llm(
    new_content: str,
    existing_content: str,
    model: str = "claude-haiku-4-5",
) -> bool:
    """Ask an LLM whether new_content contradicts existing_content.

    Returns True if the LLM determines the new fact contradicts/supersedes
    the existing one. Returns False on any error or if compatible.

    Results are cached by (new_content, existing_content) pair to avoid
    redundant API calls.
    """
    cache_key: Tuple[str, str] = (new_content, existing_content)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug("llm_contradiction: cache hit")
        return cached

    try:
        import anthropic  # noqa: F401 — check availability
    except ImportError:
        logger.warning("llm_contradiction: anthropic not installed, returning False")
        return False

    try:
        client = _build_client()
        prompt = CONTRADICTION_PROMPT.format(
            existing=existing_content.strip(),
            new=new_content.strip(),
        )
        msg = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        response = msg.content[0].text.strip().upper()
        result = response.startswith("CONTRADICTS")
        logger.debug(
            "llm_contradiction: model=%s response=%r result=%s",
            model,
            response,
            result,
        )
    except Exception as exc:
        logger.warning("llm_contradiction: LLM call failed (%s), returning False", exc)
        return False

    _cache_set(cache_key, result)
    return result
