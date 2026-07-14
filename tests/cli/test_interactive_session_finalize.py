from types import SimpleNamespace

import pytest

import cli


def test_finalize_interactive_session_runs_cleanup_summary_then_release(monkeypatch):
    calls = []
    fake_cli = SimpleNamespace(
        _print_exit_summary=lambda: calls.append("summary"),
        _release_active_session=lambda: calls.append("release"),
    )

    monkeypatch.setattr(cli, "_run_cleanup", lambda: calls.append("cleanup"))

    cli._finalize_interactive_session(fake_cli)

    assert calls == ["cleanup", "summary", "release"]


def test_finalize_interactive_session_releases_session_when_cleanup_fails(monkeypatch):
    calls = []
    fake_cli = SimpleNamespace(
        _print_exit_summary=lambda: calls.append("summary"),
        _release_active_session=lambda: calls.append("release"),
    )

    def cleanup():
        calls.append("cleanup")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(cli, "_run_cleanup", cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        cli._finalize_interactive_session(fake_cli)

    assert calls == ["cleanup", "release"]


def test_finalize_interactive_session_releases_session_when_summary_fails(monkeypatch):
    calls = []

    def summary():
        calls.append("summary")
        raise RuntimeError("summary failed")

    fake_cli = SimpleNamespace(
        _print_exit_summary=summary,
        _release_active_session=lambda: calls.append("release"),
    )

    monkeypatch.setattr(cli, "_run_cleanup", lambda: calls.append("cleanup"))

    with pytest.raises(RuntimeError, match="summary failed"):
        cli._finalize_interactive_session(fake_cli)

    assert calls == ["cleanup", "summary", "release"]
