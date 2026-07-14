"""Regression tests for Copilot ACP command resolution.

Covers the Windows ``.cmd`` / ``.exe`` launcher-resolution fallback added
in PR #42538 while preserving the existing bare-command behaviour elsewhere.
"""

from __future__ import annotations

import os

import pytest

from agent.copilot_acp_client import _resolve_command, _resolve_windows_launcher


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every env var ``_resolve_command`` consults so each test owns it."""
    monkeypatch.delenv("HERMES_COPILOT_ACP_COMMAND", raising=False)
    monkeypatch.delenv("COPILOT_CLI_PATH", raising=False)


def test_bare_command_fallback_is_default(clean_env):
    """With no env override, return the literal ``"copilot"`` on non-Windows."""
    assert _resolve_command() == "copilot"


def test_hermes_env_var_wins(clean_env, monkeypatch):
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "devin")
    assert _resolve_command() == "devin"


def test_legacy_env_var_used_when_primary_missing(clean_env, monkeypatch):
    monkeypatch.setenv("COPILOT_CLI_PATH", "/opt/copilot/bin/copilot")
    assert _resolve_command() == "/opt/copilot/bin/copilot"


def test_primary_env_var_wins_over_legacy(clean_env, monkeypatch):
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "primary")
    monkeypatch.setenv("COPILOT_CLI_PATH", "legacy")
    assert _resolve_command() == "primary"


def test_already_has_extension_is_returned_verbatim(tmp_path):
    """A command that already has an extension must not be rewritten."""
    real = tmp_path / "copilot.exe"
    real.write_text("")
    assert _resolve_windows_launcher(str(real)) == str(real)


def test_windows_launcher_prefers_literal_cmd_over_exe(tmp_path, monkeypatch):
    """When both ``copilot.cmd`` and ``copilot.exe`` exist, ``.cmd`` wins."""
    (tmp_path / "copilot.cmd").write_text("@echo off\r\n")
    (tmp_path / "copilot.exe").write_text("")
    monkeypatch.chdir(tmp_path)

    assert _resolve_windows_launcher("copilot") == "copilot.cmd"


def test_windows_launcher_falls_back_to_exe_when_cmd_missing(tmp_path, monkeypatch):
    (tmp_path / "copilot.exe").write_text("")
    monkeypatch.chdir(tmp_path)

    assert _resolve_windows_launcher("copilot") == "copilot.exe"


def test_windows_launcher_uses_path_only_discovery_via_shutil_which(monkeypatch):
    """Launcher not in cwd but resolvable via PATH (shutil.which)."""

    monkeypatch.setattr(
        "agent.copilot_acp_client.os.path.isfile",
        lambda candidate: False,
    )
    monkeypatch.setattr(
        "agent.copilot_acp_client.shutil.which",
        lambda candidate: f"/mock/bin/{candidate}" if candidate == "copilot.cmd" else None,
    )

    assert _resolve_windows_launcher("copilot") == "copilot.cmd"


def test_windows_launcher_keeps_bare_command_when_no_launcher_found(
    tmp_path,
    monkeypatch,
):
    """If neither .cmd nor .exe resolves, fall back to the bare command name."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)
    monkeypatch.setenv("PATH", str(empty_dir), prepend=os.pathsep)

    assert _resolve_windows_launcher("copilot") == "copilot"