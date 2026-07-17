"""Tests for scheduler-owned non-blocking manual cron dispatch (#52705)."""

import json
import threading
from unittest.mock import MagicMock, patch

from tools.cronjob_tools import cronjob


_JOB = {
    "id": "job-run-1",
    "name": "manual run",
    "prompt": "hi",
    "enabled": True,
    "state": "scheduled",
    "schedule": {"kind": "cron", "expr": "0 9 * * *"},
}


def _shutdown_pools():
    from cron import scheduler

    scheduler._shutdown_parallel_pool()


class TestCronjobRunNonBlocking:
    def test_tool_queues_through_scheduler_owned_dispatch(self):
        with (
            patch("tools.cronjob_tools.resolve_job_ref", return_value=dict(_JOB)),
            patch("tools.cronjob_tools.get_job", return_value=dict(_JOB)),
            patch(
                "cron.scheduler.dispatch_job_now",
                return_value={
                    "mode": "scheduler",
                    "claimed": True,
                    "error": None,
                },
            ) as dispatch,
        ):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["success"] is True
        assert out["job"]["triggered"] is True
        assert out["job"]["trigger_mode"] == "scheduler"
        assert "execution_success" not in out["job"]
        dispatch.assert_called_once_with(dict(_JOB), source="manual")

    def test_tool_reports_scheduler_rejection(self):
        paused = dict(_JOB, enabled=False, state="paused")
        with (
            patch("tools.cronjob_tools.resolve_job_ref", return_value=paused),
            patch("tools.cronjob_tools.get_job", return_value=paused),
            patch(
                "cron.scheduler.dispatch_job_now",
                return_value={
                    "mode": "scheduler",
                    "claimed": False,
                    "error": "Job is paused/disabled; resume it before running.",
                },
            ),
        ):
            out = json.loads(cronjob(action="run", job_id="job-run-1"))

        assert out["job"]["triggered"] is False
        assert out["job"]["executed"] is False
        assert "paused/disabled" in out["job"]["execution_skipped"]


class TestSchedulerManualDispatch:
    def teardown_method(self):
        _shutdown_pools()

    def test_paused_job_is_rejected_before_claim(self):
        from cron.scheduler import dispatch_job_now

        paused = dict(_JOB, enabled=False, state="paused")
        with patch("cron.jobs.claim_job_for_fire") as claim:
            result = dispatch_job_now(paused, source="manual")

        assert result["claimed"] is False
        assert "paused/disabled" in result["error"]
        claim.assert_not_called()

    def test_claim_lost_is_not_dispatched(self):
        from cron.scheduler import dispatch_job_now

        with (
            patch("cron.jobs.claim_job_for_fire", return_value=False),
            patch("cron.jobs.get_job", return_value=dict(_JOB)),
            patch("cron.scheduler._get_parallel_pool") as pool,
        ):
            result = dispatch_job_now(dict(_JOB), source="manual")

        assert result["claimed"] is False
        assert "already being fired" in result["error"]
        pool.assert_not_called()

    def test_submission_returns_before_worker_runs(self):
        from cron.scheduler import dispatch_job_now

        pool = MagicMock()
        with (
            patch("cron.jobs.claim_job_for_fire", return_value=True),
            patch("cron.jobs.get_job", return_value=dict(_JOB)),
            patch("cron.scheduler.create_execution", return_value={"id": "exec-1"}),
            patch("cron.scheduler._get_parallel_pool", return_value=pool),
            patch("cron.scheduler.run_one_job") as run,
            patch("cron.scheduler._notify_provider_jobs_changed"),
        ):
            result = dispatch_job_now(dict(_JOB), source="manual")

        assert result == {"mode": "scheduler", "claimed": True, "error": None}
        pool.submit.assert_called_once()
        run.assert_not_called()

    def test_submit_failure_unwinds_claim_and_execution(self):
        from cron import scheduler

        pool = MagicMock()
        pool.submit.side_effect = RuntimeError("executor closed")
        with (
            patch("cron.jobs.claim_job_for_fire", return_value=True),
            patch("cron.jobs.get_job", return_value=dict(_JOB)),
            patch("cron.scheduler.create_execution", return_value={"id": "exec-1"}),
            patch("cron.scheduler._get_parallel_pool", return_value=pool),
            patch("cron.scheduler.mark_job_run") as mark_run,
            patch("cron.scheduler.finish_execution") as finish,
            patch("cron.scheduler._notify_provider_jobs_changed"),
        ):
            result = scheduler.dispatch_job_now(dict(_JOB), source="manual")

        assert result == {
            "mode": "scheduler",
            "claimed": False,
            "error": "executor closed",
        }
        mark_run.assert_called_once_with("job-run-1", False, "executor closed")
        finish.assert_called_once()

    def test_scheduler_shutdown_waits_for_manual_worker(self):
        """Standalone CLI shutdown must not discard a queued manual fire."""
        from cron import scheduler

        started = threading.Event()
        release = threading.Event()
        shutdown_done = threading.Event()

        def run_job(_job):
            started.set()
            assert release.wait(5)
            return True

        with (
            patch("cron.jobs.claim_job_for_fire", return_value=True),
            patch("cron.jobs.get_job", return_value=dict(_JOB)),
            patch("cron.scheduler.create_execution", return_value={"id": "exec-1"}),
            patch("cron.scheduler.run_one_job", side_effect=run_job),
            patch("cron.scheduler._notify_provider_jobs_changed"),
        ):
            result = scheduler.dispatch_job_now(dict(_JOB), source="manual")
            assert result["claimed"] is True
            assert started.wait(5)

            waiter = threading.Thread(
                target=lambda: (
                    scheduler._shutdown_parallel_pool(),
                    shutdown_done.set(),
                )
            )
            waiter.start()
            assert not shutdown_done.wait(0.05)
            release.set()
            waiter.join(5)

        assert shutdown_done.is_set()

    def test_external_provider_reconciles_after_claim_and_completion(self):
        from cron import scheduler

        notifications = MagicMock()
        with (
            patch("cron.jobs.claim_job_for_fire", return_value=True),
            patch("cron.jobs.get_job", return_value=dict(_JOB)),
            patch("cron.scheduler.create_execution", return_value={"id": "exec-1"}),
            patch("cron.scheduler.run_one_job", return_value=True),
            patch(
                "cron.scheduler._notify_provider_jobs_changed",
                notifications,
            ),
        ):
            result = scheduler.dispatch_job_now(dict(_JOB), source="manual")
            assert result["claimed"] is True
            scheduler._shutdown_parallel_pool()

        assert notifications.call_count == 2
