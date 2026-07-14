"""Focused tests for named custom-provider timeout resolution (#34001).

Tests the ``get_effective_provider_for_timeout`` helper and verifies that
timeout lookups honor the original named-provider identity instead of the
generic ``"custom"`` that ``runtime_provider.py`` resolves it to.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.timeouts import (
    get_effective_provider_for_timeout,
    get_provider_request_timeout,
    get_provider_stale_timeout,
)


# ─── get_effective_provider_for_timeout ────────────────────────────────────


class TestGetEffectiveProviderForTimeout:
    """The helper must recover the original named-provider identity."""

    def test_plain_provider_string_passthrough(self):
        assert get_effective_provider_for_timeout("openrouter") == "openrouter"
        assert get_effective_provider_for_timeout("anthropic") == "anthropic"

    def test_custom_with_requested_provider(self):
        agent = SimpleNamespace(provider="custom", requested_provider="minimax")
        assert get_effective_provider_for_timeout(agent) == "minimax"

    def test_custom_without_requested_provider_falls_back(self):
        agent = SimpleNamespace(provider="custom", requested_provider=None)
        assert get_effective_provider_for_timeout(agent) == "custom"

    def test_custom_without_requested_provider_attr(self):
        """Agent-like object with .provider but no .requested_provider."""
        agent = SimpleNamespace(provider="custom")
        assert get_effective_provider_for_timeout(agent) == "custom"

    def test_non_custom_provider_ignores_requested(self):
        """For non-custom providers, the canonical provider is used."""
        agent = SimpleNamespace(provider="openrouter", requested_provider="minimax")
        assert get_effective_provider_for_timeout(agent) == "openrouter"


# ─── Integration: timeout resolution with named custom provider ─────────────


class TestNamedCustomProviderTimeoutResolution:
    """Verify get_provider_request_timeout / get_provider_stale_timeout find
    config under the original provider name, not ``"custom"``.
    """

    @pytest.fixture
    def mock_config(self):
        return {
            "providers": {
                "minimax": {
                    "request_timeout_seconds": 42,
                    "stale_timeout_seconds": 300,
                    "models": {
                        "MiniMax-M1": {"timeout_seconds": 77},
                    },
                },
            },
        }

    def test_request_timeout_found_under_named_provider(self, mock_config):
        with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
            assert get_provider_request_timeout("minimax") == 42.0

    def test_request_timeout_not_found_under_generic_custom(self, mock_config):
        with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
            assert get_provider_request_timeout("custom") is None

    def test_stale_timeout_found_under_named_provider(self, mock_config):
        with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
            assert get_provider_stale_timeout("minimax") == 300.0

    def test_stale_timeout_not_found_under_generic_custom(self, mock_config):
        with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
            assert get_provider_stale_timeout("custom") is None

    def test_per_model_override_found_under_named_provider(self, mock_config):
        with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
            assert get_provider_request_timeout("minimax", "MiniMax-M1") == 77.0

    def test_end_to_end_helper_resolves_named_custom(self, mock_config):
        """Simulate the real flow: agent has provider='custom' +
        requested_provider='minimax', helper recovers the name, and the
        timeout lookup finds the config."""
        agent = SimpleNamespace(provider="custom", requested_provider="minimax")
        effective = get_effective_provider_for_timeout(agent)
        assert effective == "minimax"
        with patch("hermes_cli.config.load_config_readonly", return_value=mock_config):
            assert get_provider_request_timeout(effective) == 42.0
            assert get_provider_stale_timeout(effective) == 300.0
