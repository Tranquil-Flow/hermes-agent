"""Regression for #117276: ``hermes doctor``'s exit status must agree with its unresolved findings."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("issues,manual,fixed,fix,expected", [
    (["needs repair"], [], 0, False, 1),
    ([], ["manual repair"], 0, False, 1),
    ([], [], 0, False, 0),
    ([], [], 1, True, 0),
    ([], ["remaining repair"], 1, True, 1),
])
def test_doctor_command_reports_remaining_findings(monkeypatch, capsys, issues, manual, fixed, fix, expected):
    import hermes_cli.doctor as doctor
    from hermes_cli.main import cmd_doctor
    from hermes_cli.doctor_report import Finding

    def check(should_fix):
        assert should_fix is fix
        return Finding(issues=issues, manual_issues=manual, fixed=fixed)

    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, check),))
    result = cmd_doctor(SimpleNamespace(fix=fix, ack=None, live=False))
    output = capsys.readouterr().out
    assert result == expected
    for issue in issues + manual:
        assert issue in output


@pytest.mark.parametrize("unresolved", [False, True])
def test_doctor_cli_process_status_matches_summary(unresolved):
    import subprocess
    import sys
    from pathlib import Path

    program = f"""
import sys
import hermes_cli.doctor as doctor
from hermes_cli.doctor_report import Finding
from hermes_cli.main import main
issues = ['fixture unresolved problem'] if {unresolved!r} else []
doctor.DOCTOR_CHECKS = ((None, lambda fix: Finding(issues=issues)),)
sys.argv = ['hermes', 'doctor']
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == int(unresolved), result.stdout + result.stderr
    assert ("fixture unresolved problem" if unresolved else "All checks passed") in result.stdout


class TestDoctorCheckRaises:
    """Regression for #132335: a check that raises must never fold into "All checks passed!" + exit 0.

    ``doctor_check(on_error=...)`` swallows the exception (best-effort contract stays), but the
    Finding must record that the check did not complete — for the ``"text"`` and the silent ``""``
    decorator alike — and both the summary and the exit status must account for it."""

    def _run(self, monkeypatch, capsys, checks, live=False):
        import hermes_cli.doctor as doctor

        monkeypatch.setattr(doctor, "DOCTOR_CHECKS", checks)
        code = doctor.run_doctor(SimpleNamespace(fix=False, ack=None, live=live))
        return code, capsys.readouterr().out

    def test_check_raise_with_error_text_is_incomplete(self, monkeypatch, capsys):
        from hermes_cli.doctor_report import doctor_check

        @doctor_check("Security advisory check failed: {e}")
        def raising_check(should_fix, f):
            raise RuntimeError("stand-in exception")

        code, out = self._run(monkeypatch, capsys, (("Security Advisories", raising_check),))
        assert code == 1
        assert "All checks passed" not in out
        assert "could not complete" in out
        assert "raising_check" in out and "stand-in exception" in out

    def test_check_raise_with_silent_decorator_is_incomplete(self, monkeypatch, capsys):
        from hermes_cli.doctor_report import doctor_check

        @doctor_check("")
        def silent_raising_check(should_fix, f):
            raise RuntimeError("silent boom")

        code, out = self._run(monkeypatch, capsys, ((None, silent_raising_check),))
        assert code == 1
        assert "All checks passed" not in out
        assert "could not complete" in out
        assert "silent boom" in out

    def test_partial_findings_survive_a_raise(self, monkeypatch, capsys):
        from hermes_cli.doctor_report import doctor_check

        @doctor_check("skipped: {e}")
        def partial_check(should_fix, f):
            f.manual_issues.append("recorded before the crash")
            raise RuntimeError("late boom")

        code, out = self._run(monkeypatch, capsys, ((None, partial_check),))
        assert code == 1
        assert "recorded before the crash" in out
        assert "could not complete" in out

    def test_live_import_failure_with_live_flag_is_incomplete(self, monkeypatch, capsys):
        import sys

        from hermes_cli.doctor_report import doctor_check

        @doctor_check()
        def healthy_check(should_fix, f):
            pass

        monkeypatch.setitem(sys.modules, "hermes_cli.doctor_live", None)
        code, out = self._run(monkeypatch, capsys, ((None, healthy_check),), live=True)
        assert code == 1
        assert "All checks passed" not in out
        assert "could not complete" in out

    def test_plain_run_does_not_depend_on_live_import(self, monkeypatch, capsys):
        import sys

        from hermes_cli.doctor_report import doctor_check

        @doctor_check()
        def healthy_check(should_fix, f):
            pass

        monkeypatch.setitem(sys.modules, "hermes_cli.doctor_live", None)
        code, out = self._run(monkeypatch, capsys, ((None, healthy_check),), live=False)
        assert code == 0
        assert "All checks passed" in out

    def test_healthy_decorated_check_still_passes(self, monkeypatch, capsys):
        from hermes_cli.doctor_report import check_ok, doctor_check

        @doctor_check()
        def healthy_check(should_fix, f):
            check_ok("stand-in check that runs and passes")

        code, out = self._run(monkeypatch, capsys, ((None, healthy_check),))
        assert code == 0
        assert "All checks passed" in out

    def test_finding_merge_carries_incomplete(self):
        from hermes_cli.doctor_report import Finding

        total, part = Finding(), Finding()
        part.incomplete.append("some check did not finish")
        total.merge(part)
        assert total.incomplete == ["some check did not finish"]
        assert total.issues == [] and total.manual_issues == [] and total.fixed == 0


@pytest.mark.parametrize("incomplete", [False, True])
def test_doctor_cli_process_status_matches_incomplete_checks(incomplete):
    """E2E for #132335: a raising decorated check drives the real process exit status — 1, not 0."""
    import subprocess
    import sys
    from pathlib import Path

    program = f"""
import sys
import hermes_cli.doctor as doctor
from hermes_cli.doctor_report import doctor_check

@doctor_check("Security advisory check failed: {{e}}")
def broken_check(should_fix, f):
    raise RuntimeError("fixture raising check")

healthy = lambda should_fix: doctor.Finding()
doctor.DOCTOR_CHECKS = ((None, broken_check),) if {incomplete!r} else ((None, healthy),)
sys.argv = ['hermes', 'doctor']
from hermes_cli.main import main
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == int(incomplete), result.stdout + result.stderr
    if incomplete:
        assert "could not complete" in result.stdout
    assert ("All checks passed" in result.stdout) != incomplete
