"""Regression tests for Codex refresh_token self-heal (cross-store rotation).

Hermes keeps its OWN copy of the Codex OAuth token (per profile + top-level),
separate from the Codex CLI's ``~/.codex/auth.json``. OAuth refresh_tokens are
single-use, so when the Codex CLI (or another Hermes process) rotates the shared
token, the frozen copy's refresh_token goes stale and ``refresh_codex_oauth_pure``
fails with a relogin-required error. ``_refresh_codex_auth_tokens`` must then
recover by re-importing the canonical token from ``~/.codex/auth.json`` instead of
surfacing a hard 401 — but ONLY for relogin-required failures, never for transient
ones (e.g. 429 quota, where the stored token is still valid).
"""

import json

import pytest

import hermes_cli.auth as auth
from hermes_cli.auth import AuthError, _refresh_codex_auth_tokens, resolve_codex_runtime_credentials

STALE = {"access_token": "stale-access", "refresh_token": "stale-refresh"}


def test_self_heals_on_stale_refresh_token(monkeypatch):
    """invalid_grant (relogin-required) → reimport from ~/.codex and persist it."""
    saved = {}
    fresh = {
        "access_token": "fresh-access",
        "refresh_token": "fresh-refresh",
        "last_refresh": "2026-06-12T00:00:00Z",
    }

    def _rejected(*_a, **_k):
        raise AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rejected)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: dict(fresh))
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda t, *a, **k: saved.update(t))

    out = _refresh_codex_auth_tokens(STALE, 20.0)

    assert out["access_token"] == "fresh-access"
    assert out["refresh_token"] == "fresh-refresh"
    # the recovered token was persisted to the Hermes auth store
    assert saved["access_token"] == "fresh-access"










def test_self_heals_missing_singleton_access_token_from_codex_cli(tmp_path, monkeypatch):
    """Exact cron failure path: Hermes auth has refresh_token but missing access_token."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {"refresh_token": "stale-refresh"},
                "last_refresh": "2026-06-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    resolved = resolve_codex_runtime_credentials()

    assert resolved["api_key"] == "fresh-access"
    assert resolved["source"] == "hermes-auth-store"
    stored = json.loads((hermes_home / "auth.json").read_text())
    tokens = stored["providers"]["openai-codex"]["tokens"]
    assert tokens["access_token"] == "fresh-access"
    assert tokens["refresh_token"] == "fresh-refresh"


# ---------------------------------------------------------------------------
# Deterministic race regression tests — locked CAS for cross-workspace recovery
# ---------------------------------------------------------------------------


def test_recovery_cas_missing_token_path(tmp_path, monkeypatch):
    """Caller observes no token (None) → CAS allows recovery when store is also empty."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    # Auth store has no Codex tokens at all
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
    }))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    # caller_observed_access_token=None: store has no token → match → recover
    recovered = auth._recover_codex_tokens_from_cli(
        "test-missing", caller_observed_access_token=None,
    )
    assert recovered is not None
    assert recovered["access_token"] == "fresh-access"
    stored = json.loads((hermes_home / "auth.json").read_text())
    assert stored["providers"]["openai-codex"]["tokens"]["access_token"] == "fresh-access"


def test_recovery_cas_same_workspace_reauth(tmp_path, monkeypatch):
    """Caller-observed token matches stored → recovery succeeds (same workspace)."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "token-A",
                    "refresh_token": "refresh-A",
                },
                "last_refresh": "2026-06-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    # caller_observed_access_token="token-A" matches stored → CAS succeeds
    recovered = auth._recover_codex_tokens_from_cli(
        "test-same-workspace", caller_observed_access_token="token-A",
    )
    assert recovered is not None
    assert recovered["access_token"] == "fresh-access"
    stored = json.loads((hermes_home / "auth.json").read_text())
    assert stored["providers"]["openai-codex"]["tokens"]["access_token"] == "fresh-access"


def test_recovery_cas_cross_workspace_refused(tmp_path, monkeypatch):
    """Caller-observed token differs from stored → recovery REFUSED (cross-workspace)."""
    hermes_home = tmp_path / "hermes"
    codex_home = tmp_path / "codex"
    hermes_home.mkdir()
    codex_home.mkdir()
    (hermes_home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": "token-B",  # different workspace already stored
                    "refresh_token": "refresh-B",
                },
                "last_refresh": "2026-06-01T00:00:00Z",
                "auth_mode": "chatgpt",
            },
        },
    }))
    (codex_home / "auth.json").write_text(json.dumps({
        "tokens": {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    # caller observed "token-A" but store has "token-B" → cross-workspace → refuse
    recovered = auth._recover_codex_tokens_from_cli(
        "test-cross-workspace", caller_observed_access_token="token-A",
    )
    assert recovered is None
    # Store must remain unchanged
    stored = json.loads((hermes_home / "auth.json").read_text())
    assert stored["providers"]["openai-codex"]["tokens"]["access_token"] == "token-B"


def test_refresh_recovery_cas_match(monkeypatch):
    """Through _refresh_codex_auth_tokens: CAS match → recovery succeeds."""
    saved = {}
    fresh = {
        "access_token": "fresh-access",
        "refresh_token": "fresh-refresh",
        "last_refresh": "2026-06-12T00:00:00Z",
    }

    def _rejected(*_a, **_k):
        raise AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rejected)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: dict(fresh))
    monkeypatch.setattr(auth, "_save_codex_tokens", lambda t, *a, **k: saved.update(t))
    # Mock _read_codex_tokens to return a token that matches the caller-observed
    monkeypatch.setattr(auth, "_read_codex_tokens", lambda **kw: {
        "tokens": {"access_token": "token-A", "refresh_token": "refresh-A"},
        "last_refresh": "2026-06-01T00:00:00Z",
    })

    out = _refresh_codex_auth_tokens(
        STALE, 20.0,
        caller_observed_access_token="token-A",  # matches stored → CAS succeeds
    )

    assert out["access_token"] == "fresh-access"
    assert saved["access_token"] == "fresh-access"


def test_refresh_recovery_cas_mismatch_rejected(monkeypatch):
    """Through _refresh_codex_auth_tokens: CAS mismatch → recovery refused, re-raises."""
    fresh = {
        "access_token": "fresh-access",
        "refresh_token": "fresh-refresh",
        "last_refresh": "2026-06-12T00:00:00Z",
    }

    def _rejected(*_a, **_k):
        raise AuthError(
            "refresh token rejected",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth, "refresh_codex_oauth_pure", _rejected)
    monkeypatch.setattr(auth, "_import_codex_cli_tokens", lambda: dict(fresh))
    # Mock _read_codex_tokens to return "token-B" — different from caller-observed "token-A"
    monkeypatch.setattr(auth, "_read_codex_tokens", lambda **kw: {
        "tokens": {"access_token": "token-B", "refresh_token": "refresh-B"},
        "last_refresh": "2026-06-01T00:00:00Z",
    })

    with pytest.raises(AuthError, match="refresh token rejected"):
        _refresh_codex_auth_tokens(
            STALE, 20.0,
            caller_observed_access_token="token-A",  # mismatches stored "token-B" → CAS fails
        )


