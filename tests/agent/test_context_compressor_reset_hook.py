"""Tests for on_session_reset() context-length re-probe behaviour.

Covers PR #31492 / #31043: the reset hook re-probes ``context_length`` so a
reloaded LM Studio model is picked up on ``/new``, but it must:

1. Preserve an explicitly configured override (``model.context_length`` or
   custom-provider per-model context_length) — never overwrite it with a
   probed value.
2. Fall back gracefully when the provider is unavailable — keep the old
   ``context_length`` rather than clobbering it with ``None``/default.
3. Adopt a *higher* probed value (the original use-case: user reloaded LM
   Studio with a larger context_length).
4. Ignore a *lower* probed value (transient endpoint underreport).
"""

import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Stub out optional heavy dependencies not installed in the test environment
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent.context_compressor import ContextCompressor


def _make_compressor(**kwargs):
    """Build a minimal ContextCompressor via __new__ + manual init for the
    reset-hook path (avoids the full ``__init__`` which calls
    ``get_model_context_length`` and sets up many unrelated fields)."""
    c = ContextCompressor.__new__(ContextCompressor)
    c.model = kwargs.get("model", "test/model")
    c.base_url = kwargs.get("base_url", "http://127.0.0.1:1234/v1")
    c.api_key = kwargs.get("api_key", "test-key")
    c.provider = kwargs.get("provider", "lmstudio")
    c.api_mode = kwargs.get("api_mode", "")
    c.context_length = kwargs.get("context_length", 8192)
    c._config_context_length = kwargs.get("_config_context_length", None)
    c.max_tokens = kwargs.get("max_tokens", None)
    c.threshold_percent = kwargs.get("threshold_percent", 0.50)
    c.summary_target_ratio = 0.20
    c.protect_first_n = 3
    c.protect_last_n = 20
    c.quiet_mode = True
    c.compression_count = 0
    # State that ContextEngine.on_session_reset() super() clears:
    c.last_prompt_tokens = 0
    c.last_completion_tokens = 0
    c.last_total_tokens = 0
    # State that ContextCompressor.on_session_reset() clears:
    c._previous_summary = None
    c._last_summary_error = None
    c._last_summary_dropped_count = 0
    c._last_summary_fallback_used = False
    c._last_aux_model_failure_error = None
    c._last_aux_model_failure_model = None
    c._last_compression_savings_pct = 100.0
    c._ineffective_compression_count = 0
    c._verify_compaction_cleared_threshold = False
    c._last_compression_made_progress = False
    c._summary_failure_cooldown_until = 0.0
    c._last_compress_aborted = False
    c._context_probed = False
    c._context_probe_persistable = False
    c.last_real_prompt_tokens = 0
    c.last_compression_rough_tokens = 0
    c.last_rough_tokens_when_real_prompt_fit = 0
    c.awaiting_real_usage_after_compression = False
    return c


# ---------------------------------------------------------------------------
# 1. Explicit override preservation
# ---------------------------------------------------------------------------

def test_explicit_config_override_preserved_on_reset():
    """When _config_context_length is set, the reset hook must NOT call
    get_model_context_length at all — the override is authoritative and
    re-probing can only risk overwriting it."""
    c = _make_compressor(
        context_length=500_000,
        _config_context_length=500_000,
    )

    with patch("agent.context_compressor.get_model_context_length") as mock_probe, \
         patch("agent.context_compressor.invalidate_endpoint_model_metadata_cache") as mock_inval:
        c.on_session_reset()

        # The resolver must not be called — the override short-circuits.
        mock_probe.assert_not_called()
        # Cache invalidation is inside the probe branch, so also skipped.
        mock_inval.assert_not_called()

    # The context_length is untouched.
    assert c.context_length == 500_000


def test_explicit_override_preserved_even_if_probe_returns_higher():
    """Even if a buggy/mocked probe returned a higher value, the explicit
    override must win — the hook skips probing entirely."""
    c = _make_compressor(
        context_length=100_000,
        _config_context_length=100_000,
    )

    with patch("agent.context_compressor.get_model_context_length", return_value=999_999):
        c.on_session_reset()

    assert c.context_length == 100_000


# ---------------------------------------------------------------------------
# 2. Provider-unavailable fallback
# ---------------------------------------------------------------------------

def test_provider_unavailable_preserves_old_context_length():
    """If the provider is down on /new (probe raises), the old context_length
    must be preserved — not clobbered with None or a default."""
    c = _make_compressor(
        context_length=32_768,
        _config_context_length=None,
    )

    with patch("agent.context_compressor.get_model_context_length", side_effect=ConnectionError("refused")), \
         patch("agent.context_compressor.invalidate_endpoint_model_metadata_cache"):
        c.on_session_reset()

    assert c.context_length == 32_768


def test_provider_unavailable_does_not_call_update_model():
    """The exception path must not invoke update_model (which would overwrite
    context_length with a bogus value)."""
    c = _make_compressor(context_length=16_384, _config_context_length=None)

    with patch("agent.context_compressor.get_model_context_length", side_effect=RuntimeError("timeout")), \
         patch.object(c, "update_model") as mock_update, \
         patch("agent.context_compressor.invalidate_endpoint_model_metadata_cache"):
        c.on_session_reset()

    mock_update.assert_not_called()
    assert c.context_length == 16_384


# ---------------------------------------------------------------------------
# 3. Probe-up adoption (the original use-case)
# ---------------------------------------------------------------------------

def test_probe_higher_adopts_new_context_length():
    """When no explicit override is set and the probe returns a higher value
    (e.g. LM Studio reloaded with a bigger context_length), the new value is
    adopted via update_model."""
    c = _make_compressor(
        context_length=8_192,
        _config_context_length=None,
        provider="lmstudio",
    )

    with patch("agent.context_compressor.get_model_context_length", return_value=32_768), \
         patch.object(c, "update_model") as mock_update, \
         patch("agent.context_compressor.invalidate_endpoint_model_metadata_cache") as mock_inval:
        c.on_session_reset()

    mock_inval.assert_called_once_with("http://127.0.0.1:1234/v1")
    mock_update.assert_called_once()
    # update_model receives the new context_length as second positional arg
    args, kwargs = mock_update.call_args
    assert args[0] == "test/model"
    assert args[1] == 32_768


def test_probe_lower_does_not_shrink_context_length():
    """A probe returning a lower value must NOT shrink the resolved budget."""
    c = _make_compressor(
        context_length=131_072,
        _config_context_length=None,
    )

    with patch("agent.context_compressor.get_model_context_length", return_value=8_192), \
         patch.object(c, "update_model") as mock_update, \
         patch("agent.context_compressor.invalidate_endpoint_model_metadata_cache"):
        c.on_session_reset()

    mock_update.assert_not_called()
    assert c.context_length == 131_072


def test_probe_equal_does_not_call_update_model():
    """A probe returning the exact same value is a no-op (no update_model call)."""
    c = _make_compressor(
        context_length=32_768,
        _config_context_length=None,
    )

    with patch("agent.context_compressor.get_model_context_length", return_value=32_768), \
         patch.object(c, "update_model") as mock_update, \
         patch("agent.context_compressor.invalidate_endpoint_model_metadata_cache"):
        c.on_session_reset()

    mock_update.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Non-LM-Studio providers: cache invalidation skipped
# ---------------------------------------------------------------------------

def test_non_lmstudio_provider_skips_cache_invalidation():
    """Cache invalidation is LM-Studio-specific; other providers must not
    trigger it."""
    c = _make_compressor(
        context_length=8_192,
        _config_context_length=None,
        provider="openai",
    )

    with patch("agent.context_compressor.get_model_context_length", return_value=16_384), \
         patch.object(c, "update_model"), \
         patch("agent.context_compressor.invalidate_endpoint_model_metadata_cache") as mock_inval:
        c.on_session_reset()

    mock_inval.assert_not_called()


# ---------------------------------------------------------------------------
# 5. _config_context_length = 0 or None treated as "not set"
# ---------------------------------------------------------------------------

def test_config_context_length_zero_treated_as_unset():
    """_config_context_length=0 means 'not set' — the hook should probe."""
    c = _make_compressor(
        context_length=8_192,
        _config_context_length=0,
    )

    with patch("agent.context_compressor.get_model_context_length", return_value=16_384) as mock_probe, \
         patch.object(c, "update_model"), \
         patch("agent.context_compressor.invalidate_endpoint_model_metadata_cache"):
        c.on_session_reset()

    mock_probe.assert_called_once()


def test_config_context_length_none_treated_as_unset():
    """_config_context_length=None means 'not set' — the hook should probe."""
    c = _make_compressor(
        context_length=8_192,
        _config_context_length=None,
    )

    with patch("agent.context_compressor.get_model_context_length", return_value=16_384) as mock_probe, \
         patch.object(c, "update_model"), \
         patch("agent.context_compressor.invalidate_endpoint_model_metadata_cache"):
        c.on_session_reset()

    mock_probe.assert_called_once()
