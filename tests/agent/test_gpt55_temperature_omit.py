"""Tests for gpt-5.5 family temperature handling in auxiliary_client.

gpt-5.5 family models (across ALL providers — openai-api direct, openai-codex
OAuth, openrouter, custom proxies) only accept the provider's default
``temperature=1``. Passing any other value is rejected with HTTP 400
"Unsupported value: 'temperature' does not support X with this model. Only
the default (1) value is supported." — see #51083.

Fix: ``_fixed_temperature_for_model`` returns ``OMIT_TEMPERATURE`` for any
gpt-5.5 family slug so ``_build_call_kwargs`` strips the ``temperature`` key
entirely and lets the provider pick its default. Eliminates the wasted 400
round-trip on the first call (instead of relying on the
``provider rejected temperature; retrying once without it`` path that
sometimes races with image-size retries / fallback chain timeouts).

Family detection is provider-agnostic on purpose: gpt-5.5 on openai-api,
openai-codex, openrouter, and any custom proxy all share the strict-default
contract. Sibling gpt-5.4 / gpt-5 / gpt-5-mini do NOT (they accept custom
temperature freely) and must NOT be matched.
"""

from __future__ import annotations

import pytest

from agent.auxiliary_client import (
    OMIT_TEMPERATURE,
    _build_call_kwargs,
    _fixed_temperature_for_model,
    _is_gpt55_family,
)


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.5-2026-04-23",  # dated snapshot
        "gpt-5.5-codex-mini",  # Codex variant of the 5.5 family
        "openai/gpt-5.5",       # aggregator-prefixed
        "openai/gpt-5.5-pro",   # aggregator-prefixed variant
        "GPT-5.5",              # case-insensitive
        "  gpt-5.5  ",          # whitespace tolerant
        "openai/GPT-5.5-Pro",   # case + aggregator
    ],
)
def test_is_gpt55_family_matches(model: str) -> None:
    assert _is_gpt55_family(model) is True


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "gpt-5.4",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.4-mini",
        "gpt-5.55",     # NOT gpt-5.5 family — false positive guard
        "gpt-5.50",     # NOT gpt-5.5 family — false positive guard
        "gpt-55",       # NOT gpt-5.5 family — prefix-only is too greedy
        "claude-sonnet-4.6",
        "kimi-k2",
        "trinity-large-thinking",
        "gpt-4o",
    ],
)
def test_is_gpt55_family_rejects_non_matches(model) -> None:
    assert _is_gpt55_family(model) is False


# ---------------------------------------------------------------------------
# _fixed_temperature_for_model directive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.5-2026-04-23",
        "gpt-5.5-codex-mini",
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "GPT-5.5",
    ],
)
def test_fixed_temperature_omits_for_gpt55_family(model: str) -> None:
    """All gpt-5.5 family slugs must return OMIT_TEMPERATURE.

    The directive is provider-agnostic — same answer regardless of base_url
    because the strict-default contract is on the model itself, not the
    route.
    """
    assert _fixed_temperature_for_model(model) is OMIT_TEMPERATURE
    # base_url must NOT change the answer.
    assert _fixed_temperature_for_model(
        model, base_url="https://api.openai.com/v1"
    ) is OMIT_TEMPERATURE
    assert _fixed_temperature_for_model(
        model, base_url="https://chatgpt.com/backend-api/codex"
    ) is OMIT_TEMPERATURE


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.4",
        "gpt-5",
        "gpt-5-mini",
        "gpt-4o",
        "claude-sonnet-4.6",
        "trinity-large-preview",
    ],
)
def test_fixed_temperature_preserved_for_non_gpt55(model: str) -> None:
    """Sibling gpt-5.x / gpt-4o / claude / arcee-preview must NOT be touched.

    Those models accept custom temperature freely. Forcing OMIT_TEMPERATURE
    on them would silently change user-facing sampling behavior.
    """
    assert _fixed_temperature_for_model(model) is None


# ---------------------------------------------------------------------------
# _build_call_kwargs end-to-end (real production path)
# ---------------------------------------------------------------------------


def test_build_call_kwargs_strips_temperature_for_gpt55() -> None:
    """Real production path: gpt-5.5 + vision_temperature=0.1 must drop the key.

    This is the exact scenario from #51083: auxiliary vision_analyze routes
    through openai-api/gpt-5.5 with the default vision_temperature=0.1, hits
    _build_call_kwargs, and the result MUST NOT contain a 'temperature' key
    so the provider's HTTP 400 "does not support 0.1" never fires.
    """
    kwargs = _build_call_kwargs(
        provider="openai-api",
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
        max_tokens=2000,
        timeout=120.0,
    )
    assert "temperature" not in kwargs, (
        f"gpt-5.5 must have temperature stripped (HTTP 400 otherwise), "
        f"got kwargs={kwargs!r}"
    )
    # Sanity: model and other kwargs survive.
    assert kwargs["model"] == "gpt-5.5"
    # _build_call_kwargs intentionally omits max_tokens for non-Anthropic
    # providers (see docstring on lines ~5245) — the OpenAI SDK accepts the
    # omission and uses the model's own max output. Verify it is NOT present
    # so a future regression that re-introduces it for openai-api is caught
    # here too.
    assert "max_tokens" not in kwargs, (
        "openai-api gpt-5.5 should not carry max_tokens (intentional omission); "
        f"got kwargs={kwargs!r}"
    )
    assert kwargs["timeout"] == 120.0


def test_build_call_kwargs_strips_temperature_for_gpt55_pro_via_openrouter() -> None:
    """Aggregator-prefixed gpt-5.5-pro via OpenRouter must also drop temperature."""
    kwargs = _build_call_kwargs(
        provider="openrouter",
        model="openai/gpt-5.5-pro",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        timeout=60.0,
    )
    assert "temperature" not in kwargs


def test_build_call_kwargs_preserves_temperature_for_gpt54() -> None:
    """gpt-5.4 must keep custom temperature — sibling model, different contract."""
    kwargs = _build_call_kwargs(
        provider="openai-api",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        timeout=60.0,
    )
    assert kwargs.get("temperature") == 0.7


def test_build_call_kwargs_preserves_temperature_for_gpt4o() -> None:
    """gpt-4o must keep custom temperature."""
    kwargs = _build_call_kwargs(
        provider="openai-api",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.3,
        timeout=60.0,
    )
    assert kwargs.get("temperature") == 0.3


def test_build_call_kwargs_strips_temperature_for_codex_codex_oauth_route() -> None:
    """Codex OAuth route to gpt-5.5 (provider=openai-codex) must drop temperature too.

    The existing _is_codex_gpt55() only handles compaction-threshold for this
    route. Temperature is the OTHER contract that gpt-5.5 enforces, and the
    fix must cover BOTH routes.
    """
    kwargs = _build_call_kwargs(
        provider="openai-codex",
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.1,
        timeout=120.0,
    )
    assert "temperature" not in kwargs
