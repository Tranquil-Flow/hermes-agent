"""Tests for oneshot (-z) mode error handling (#30623).

Covers:
- Exception in _run_agent → exit 1, error on stderr
- Empty response → exit 0 (legal, not an error)
- Broken pipe / closed stdout write → exit 1, error on stderr
"""

import os
import sys
import io
import pytest

# Force quiet agent defaults so we don't hit file I/O during import.
os.environ.setdefault("HERMES_QUIET", "1")


class TestOneshotErrorHandling:
    """Test that run_oneshot returns proper exit codes and error messages."""

    def test_exception_in_run_agent_returns_1(self, monkeypatch):
        """When _run_agent raises, exit code should be 1 with error on stderr."""
        from hermes_cli.oneshot import run_oneshot
        import hermes_cli.oneshot as oneshot_mod

        original_run_agent = oneshot_mod._run_agent

        def _failing_run_agent(*args, **kwargs):
            raise RuntimeError("simulated credential failure")

        monkeypatch.setattr(oneshot_mod, "_run_agent", _failing_run_agent)

        captured_stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", captured_stderr)
        monkeypatch.setattr(sys, "stdout", io.StringIO())

        exit_code = run_oneshot("test prompt")

        assert exit_code == 1, f"Expected exit 1 on exception, got {exit_code}"
        stderr_output = captured_stderr.getvalue()
        assert "hermes -z:" in stderr_output
        assert "simulated credential failure" in stderr_output

    def test_empty_response_exits_0(self, monkeypatch):
        """Empty response is valid — agent may have nothing to say (exit 0)."""
        from hermes_cli.oneshot import run_oneshot
        import hermes_cli.oneshot as oneshot_mod

        def _empty_run_agent(*args, **kwargs):
            return ""

        monkeypatch.setattr(oneshot_mod, "_run_agent", _empty_run_agent)
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        monkeypatch.setattr(sys, "stdout", io.StringIO())

        exit_code = run_oneshot("test prompt")
        assert exit_code == 0

    def test_successful_response_prints_to_stdout(self, monkeypatch):
        """A normal response should be written to real_stdout."""
        from hermes_cli.oneshot import run_oneshot
        import hermes_cli.oneshot as oneshot_mod

        def _ok_run_agent(*args, **kwargs):
            return "OK response"

        monkeypatch.setattr(oneshot_mod, "_run_agent", _ok_run_agent)

        captured_stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured_stdout)
        monkeypatch.setattr(sys, "stderr", io.StringIO())

        exit_code = run_oneshot("test prompt")
        assert exit_code == 0
        assert "OK response" in captured_stdout.getvalue()

    def test_broken_pipe_on_write_returns_1(self, monkeypatch):
        """When writing the response to stdout fails, exit 1 with error."""
        from hermes_cli.oneshot import run_oneshot
        import hermes_cli.oneshot as oneshot_mod

        def _ok_run_agent(*args, **kwargs):
            return "some response"

        monkeypatch.setattr(oneshot_mod, "_run_agent", _ok_run_agent)

        captured_stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", captured_stderr)

        class BrokenPipeStdout(io.StringIO):
            def write(self, s):
                raise BrokenPipeError("broken pipe")

        monkeypatch.setattr(sys, "stdout", BrokenPipeStdout())

        exit_code = run_oneshot("test prompt")
        assert exit_code == 1
        stderr_output = captured_stderr.getvalue()
        assert "output write failed" in stderr_output


class TestRunOneshotDetectsNonTtyEdgeCases:
    """Sanity checks for pipe / non-TTY edge cases the bug report mentions."""

    def test_oserror_on_stdout_write_returns_1(self, monkeypatch):
        """OSError during stdout write should return 1, not crash."""
        from hermes_cli.oneshot import run_oneshot
        import hermes_cli.oneshot as oneshot_mod

        def _ok_run_agent(*args, **kwargs):
            return "response"

        monkeypatch.setattr(oneshot_mod, "_run_agent", _ok_run_agent)

        captured_stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", captured_stderr)

        class OSErrorStdout(io.StringIO):
            def write(self, s):
                raise OSError(5, "Input/output error")

        monkeypatch.setattr(sys, "stdout", OSErrorStdout())

        exit_code = run_oneshot("test")
        assert exit_code == 1
        assert "output write failed" in captured_stderr.getvalue()

    def test_value_error_on_write_message(self, monkeypatch):
        """Encoding errors during write should return 1, not crash."""
        from hermes_cli.oneshot import run_oneshot
        import hermes_cli.oneshot as oneshot_mod

        def _ok_run_agent(*args, **kwargs):
            return "response"

        monkeypatch.setattr(oneshot_mod, "_run_agent", _ok_run_agent)

        captured_stderr = io.StringIO()
        monkeypatch.setattr(sys, "stderr", captured_stderr)

        class ValueErrorStdout(io.StringIO):
            def write(self, s):
                raise ValueError("encoding error")

        monkeypatch.setattr(sys, "stdout", ValueErrorStdout())

        exit_code = run_oneshot("test")
        assert exit_code == 1
