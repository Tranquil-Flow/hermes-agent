from __future__ import annotations

from typing import Any

# Sentinel for agents that don't set requested_provider.
_UNSET = object()


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def get_effective_provider_for_timeout(
    agent_or_provider: Any, model: str | None = None
) -> str:
    """Return the provider id that timeout config should be looked up under.

    When a named custom provider (e.g. ``providers.minimax``) is configured,
    ``runtime_provider.py`` resolves it to ``provider="custom"`` but stores the
    original name on ``agent.requested_provider``.  Timeout lookups must use the
    original name so per-provider ``request_timeout_seconds`` /
    ``stale_timeout_seconds`` settings are found.

    Accepts either:
      * An agent-like object with ``.provider`` and ``.requested_provider``
        attributes (preferred — used everywhere the agent is in scope).
      * A bare provider string (in which case it is returned unchanged — the
        caller already has the canonical id).
    """
    if isinstance(agent_or_provider, str):
        return agent_or_provider

    provider = getattr(agent_or_provider, "provider", None) or ""
    if provider != "custom":
        return provider

    # Named-custom: fall back to requested_provider, then to "custom" itself.
    requested = getattr(agent_or_provider, "requested_provider", _UNSET)
    if requested is _UNSET or not requested:
        return provider
    return requested


def get_provider_request_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("request_timeout_seconds"))


def get_provider_stale_timeout(
    provider_id: str, model: str | None = None
) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = (
        providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    )
    if not isinstance(provider_config, dict):
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None

    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    if isinstance(model_config, dict):
        return model_config
    return None
