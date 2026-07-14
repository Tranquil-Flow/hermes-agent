"""Regression tests for streaming token estimation fallback.

Tests the fix for providers that don't send usage data in stream chunks
(MiniMax, Kimi, etc.).  Verifies the `is None` check (not falsiness) is
used so that valid zero values (e.g. cache hits with prompt_tokens=0)
are preserved instead of being overwritten with estimated values.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_tokens_rough,
)


# ---------------------------------------------------------------------------
# Helper: extracts and runs the token-accounting conditional block from
# _call_chat_completions (run_agent.py ~line 5965-5974).
# ---------------------------------------------------------------------------
def _apply_token_fallback(usage_obj, api_messages, content_parts):
    """Mirror the token-fallback logic from the streaming response builder."""
    _raw = usage_obj
    _prompt = getattr(_raw, "prompt_tokens", None) if _raw else None
    _completion = getattr(_raw, "completion_tokens", None) if _raw else None
    if _prompt is None and _completion is None:
        _est = estimate_messages_tokens_rough(api_messages)
        _comp_text = "".join(content_parts) if content_parts else ""
        usage_obj = SimpleNamespace(
            prompt_tokens=_est,
            completion_tokens=estimate_tokens_rough(_comp_text),
        )
    return usage_obj


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStreamingTokenEstimation:
    def test_fallback_estimates_when_usage_none(self):
        """When usage_obj is None the block must estimate both token fields."""
        api_messages = [{"role": "user", "content": "hello world"}]
        content_parts = ["Hello! How can I help you today?"]

        result = _apply_token_fallback(None, api_messages, content_parts)

        # Estimation must produce non-zero prompt tokens
        assert result.prompt_tokens > 0, "prompt_tokens should be estimated from messages"
        # Completion should be estimated from the response text, not a char count
        assert result.completion_tokens > 0, "completion_tokens should be estimated from text"
        # Smoke-check: completion_tokens should be in the same rough ballpark as
        # the actual response (a few words), not proportional to a 500-char string
        assert result.completion_tokens < 10, (
            "completion_tokens looks like a char-count, not an estimate"
        )

    def test_preserves_valid_zero_usage(self):
        """When usage has prompt_tokens=0 (cache hit) the 0 must be preserved.

        The fix uses `is None` checks instead of truthiness so that a real
        prompt_tokens=0 value (common with cache hits on MiniMax/Kimi) is NOT
        overwritten by the estimation path.
        """
        api_messages = [{"role": "user", "content": "hello world"}]
        content_parts = ["Hello! How can I help you today?"]

        # Simulate a cache hit: provider reports 0 prompt tokens, 5 completion
        usage_with_zero = SimpleNamespace(prompt_tokens=0, completion_tokens=5)
        result = _apply_token_fallback(usage_with_zero, api_messages, content_parts)

        assert result.prompt_tokens == 0, (
            "prompt_tokens=0 (cache hit) must NOT be overwritten with an estimate"
        )
        assert result.completion_tokens == 5, (
            "completion_tokens should pass through unchanged"
        )

    def test_preserves_valid_nonzero_usage(self):
        """When both token fields are non-None they must pass through unchanged."""
        api_messages = [{"role": "user", "content": "hello world"}]
        content_parts = ["Hello! How can I help you today?"]

        usage_with_values = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        result = _apply_token_fallback(usage_with_values, api_messages, content_parts)

        assert result.prompt_tokens == 100, (
            "prompt_tokens should pass through unchanged"
        )
        assert result.completion_tokens == 50, (
            "completion_tokens should pass through unchanged"
        )

    def test_empty_content_parts_still_estimates_completion(self):
        """Even with empty content, completion_tokens must be estimated (not 0 or error)."""
        api_messages = [{"role": "user", "content": "hello world"}]
        content_parts = []

        result = _apply_token_fallback(None, api_messages, content_parts)

        assert result.prompt_tokens > 0, "prompt_tokens must still be estimated"
        # Empty text → 0 tokens from estimate_tokens_rough
        assert result.completion_tokens == 0, (
            "completion_tokens for empty content should be 0"
        )

    def test_is_none_check_discriminates_from_zero(self):
        """Explicitly verify the fix: None vs 0 are not the same check.

        The critical distinction is:
        - usage=None → both getattr calls return None → fallback branch entered
        - usage=SimpleNamespace(prompt_tokens=0, ...) → getattr returns 0 (not None)
          → fallback NOT entered, 0 is preserved
        """
        # usage=None with real messages → fallback triggered, estimate > 0
        result_none = _apply_token_fallback(None, [{"role": "user", "content": "hello"}], ["hi"])
        assert result_none.prompt_tokens > 0, "fallback should estimate when usage is None"

        # usage with prompt_tokens=0 (not None) → preserved, not overwritten
        result_zero = _apply_token_fallback(
            SimpleNamespace(prompt_tokens=0, completion_tokens=1),
            [{"role": "user", "content": "hello"}],
            ["hi"],
        )
        assert result_zero.prompt_tokens == 0, (
            "prompt_tokens=0 must NOT be overwritten — is None check catches this"
        )
