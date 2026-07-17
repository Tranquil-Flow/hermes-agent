"""Regression tests for #53301: scoped Windows Terminal truecolor launch."""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.main


def test_injects_for_windows_terminal_256color():
    env = {"TERM": "xterm-256color", "WT_SESSION": "session-guid"}

    hermes_cli.main._apply_tui_truecolor_env(env)

    assert env["COLORTERM"] == "truecolor"


def test_noop_for_generic_256color_terminal_without_wt_session():
    """Do not change xterm/screen/tmux behavior outside Windows Terminal."""
    env = {"TERM": "xterm-256color"}

    hermes_cli.main._apply_tui_truecolor_env(env)

    assert "COLORTERM" not in env


@pytest.mark.parametrize("term", ["xterm", "screen", "tmux", ""])
def test_noop_when_windows_terminal_lacks_supported_term(term):
    env = {"TERM": term, "WT_SESSION": "session-guid"}

    hermes_cli.main._apply_tui_truecolor_env(env)

    assert "COLORTERM" not in env


@pytest.mark.parametrize("value", ["truecolor", "24bit", "TrueColor"])
def test_preserves_existing_truecolor_signal(value):
    env = {
        "TERM": "xterm-256color",
        "WT_SESSION": "session-guid",
        "COLORTERM": value,
    }

    hermes_cli.main._apply_tui_truecolor_env(env)

    assert env["COLORTERM"] == value
    assert "FORCE_COLOR" not in env


def test_launch_tui_passes_scoped_truecolor_to_subprocess(monkeypatch):
    captured = {}
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("WT_SESSION", "session-guid")
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setattr(
        hermes_cli.main,
        "_make_tui_argv",
        lambda _tui_dir, _tui_dev: (["node", "dist/entry.js"], Path(".")),
    )

    def fake_call(argv, cwd=None, env=None):
        captured.update(argv=argv, cwd=cwd, env=env)
        return 1

    monkeypatch.setattr(hermes_cli.main.subprocess, "call", fake_call)

    with pytest.raises(SystemExit) as exc:
        hermes_cli.main._launch_tui()

    assert exc.value.code == 1
    assert captured["env"]["WT_SESSION"] == "session-guid"
    assert captured["env"]["COLORTERM"] == "truecolor"
