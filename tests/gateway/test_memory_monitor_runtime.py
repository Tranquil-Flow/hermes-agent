"""Regression tests for gateway runtime memory-monitor wiring.

Issue #49773: ``start_memory_monitoring()`` was defined in
``gateway/memory_monitor.py`` and its own unit tests passed, but the gateway
runtime (``gateway/run.py``) never imported or invoked it.  Result: no
``[MEMORY]`` heartbeat in ``gateway.log``, so external log-freshness watchdogs
false-triggered and killed healthy idle gateways.

These tests exercise the **real production path**: they write a real
``config.yaml``, call the real gateway-runtime helpers, and verify the real
background monitor thread actually starts/stops with the configured interval.
No inline recomputation of the expected behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from gateway import memory_monitor as mm


@pytest.fixture(autouse=True)
def _clean_monitor_state():
    """Every test starts and ends with the monitor stopped."""
    mm.stop_memory_monitoring(timeout=1.0)
    yield
    mm.stop_memory_monitoring(timeout=1.0)


# ---------------------------------------------------------------------------
# Helpers — these live in gateway.run and are the production wiring that was
# missing (issue #49773).  They read config.yaml and drive the real monitor.
# ---------------------------------------------------------------------------


def test_start_helper_starts_real_monitor_with_config_interval(monkeypatch, tmp_path):
    """_start_runtime_memory_monitoring() starts the background thread using
    the interval from config.yaml."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "logging:\n  memory_monitor:\n    enabled: true\n    interval_seconds: 99\n",
        encoding="utf-8",
    )

    gr._start_runtime_memory_monitoring()

    assert mm.is_running() is True
    assert mm._interval_seconds == 99.0


def test_start_helper_defaults_interval_when_unset(monkeypatch, tmp_path):
    """When interval_seconds is omitted, the default 300s applies."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "logging:\n  memory_monitor:\n    enabled: true\n",
        encoding="utf-8",
    )

    gr._start_runtime_memory_monitoring()

    assert mm.is_running() is True
    assert mm._interval_seconds == 300.0


def test_start_helper_skips_when_disabled(monkeypatch, tmp_path):
    """enabled: false must NOT start the monitor."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "logging:\n  memory_monitor:\n    enabled: false\n",
        encoding="utf-8",
    )

    assert gr._start_runtime_memory_monitoring() is False
    assert mm.is_running() is False


@pytest.mark.parametrize("yaml_value", ["0", "-1", "not-a-number", ".nan", ".inf"])
def test_invalid_or_nonpositive_interval_disables_monitor(
    monkeypatch, tmp_path, yaml_value
):
    """Explicit invalid intervals disable; they never fall back and start."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "logging:\n  memory_monitor:\n    enabled: true\n"
        f"    interval_seconds: {yaml_value}\n",
        encoding="utf-8",
    )

    assert gr._start_runtime_memory_monitoring() is False
    assert mm.is_running() is False


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), "bad", None])
def test_monitor_api_rejects_invalid_interval(interval):
    """Direct callers cannot create a zero-delay/busy-loop monitor either."""
    assert mm.start_memory_monitoring(interval_seconds=interval) is False
    assert mm.is_running() is False


def test_start_helper_starts_with_defaults_when_no_config(monkeypatch, tmp_path):
    """No config.yaml at all → monitoring starts enabled with the 300s default
    (the heartbeat is the safe default; users opt out via config)."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    # Deliberately do NOT write a config.yaml.

    gr._start_runtime_memory_monitoring()

    assert mm.is_running() is True
    assert mm._interval_seconds == 300.0


def test_stop_helper_stops_real_monitor():
    """_stop_runtime_memory_monitoring() stops the background thread."""
    import gateway.run as gr

    mm.start_memory_monitoring(interval_seconds=3600.0)
    assert mm.is_running() is True

    gr._stop_runtime_memory_monitoring()

    assert mm.is_running() is False


def test_stop_helper_is_noop_when_not_started():
    """Calling stop without a prior start must not raise."""
    import gateway.run as gr

    # _clean_monitor_state fixture already stopped it; call again.
    gr._stop_runtime_memory_monitoring()
    assert mm.is_running() is False


# ---------------------------------------------------------------------------
# Integration: the helpers are actually wired into the start_gateway() lifecycle.
# We mock the heavyweight external dependencies (PID locks, MCP discovery, the
# platform adapters) so we can drive start_gateway() to completion and observe
# that start/stop_memory_monitoring fire through the real code path.
# ---------------------------------------------------------------------------


class _StubRunner:
    """Lifecycle-accurate stand-in for GatewayRunner.

    ``start_gateway`` checks ``_running`` before launching scheduler-owned
    background tasks, then waits for shutdown and tears the runner down. Model
    those state transitions rather than returning from inert methods.
    """

    def __init__(self):
        self.should_exit_cleanly = False
        self.should_exit_with_failure = False
        self.exit_code = None
        self.exit_reason = None
        self.adapters = {}
        self._signal_initiated_shutdown = False
        self._restart_requested = False
        self._restart_via_service = False
        self._draining = False
        self._external_drain_active = False
        self._running = False
        self.started = asyncio.Event()
        self.shutdown_requested = asyncio.Event()
        self.stopped = asyncio.Event()

    async def start(self) -> bool:
        self._running = True
        self.started.set()
        return True

    async def wait_for_shutdown(self) -> None:
        self.shutdown_requested.set()
        return None

    async def stop(self) -> None:
        self._running = False
        self.stopped.set()


@pytest.mark.asyncio
async def test_start_gateway_lifecycle_invokes_memory_monitor(monkeypatch, tmp_path):
    """A successful start_gateway() run must start the memory monitor on boot
    and stop it on teardown."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "logging:\n  memory_monitor:\n    enabled: true\n    interval_seconds: 42\n",
        encoding="utf-8",
    )

    start_calls: list[float] = []
    stop_calls: list[bool] = []

    def _fake_start(interval_seconds: float = 300.0) -> bool:
        start_calls.append(interval_seconds)
        return True

    def _fake_stop(timeout: float = 2.0) -> None:
        stop_calls.append(True)

    # Patch the memory_monitor functions at their source so the lazy imports
    # inside the helpers resolve to our fakes.
    monkeypatch.setattr(mm, "start_memory_monitoring", _fake_start)
    monkeypatch.setattr(mm, "stop_memory_monitoring", _fake_stop)

    # Stub out the heavyweight external dependencies of start_gateway().
    # These are imported *inside* start_gateway() (function-local), so we
    # patch them at their source modules.
    import gateway.status as gstatus

    monkeypatch.setattr(gstatus, "get_running_pid", lambda: None)
    monkeypatch.setattr(gstatus, "acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr(gstatus, "write_pid_file", lambda *a, **k: None)
    monkeypatch.setattr(gstatus, "remove_pid_file", lambda *a, **k: None)
    monkeypatch.setattr(gstatus, "release_gateway_runtime_lock", lambda *a, **k: None)

    runner = _StubRunner()
    monkeypatch.setattr(gr, "GatewayRunner", lambda config=None: runner)

    import tools.skills_sync as ssync

    monkeypatch.setattr(ssync, "sync_skills", lambda *a, **k: None)

    import hermes_logging as hlog

    monkeypatch.setattr(hlog, "setup_logging", lambda *a, **k: None)

    import tools.mcp_tool as mcp

    monkeypatch.setattr(mcp, "discover_mcp_tools", lambda *a, **k: None)

    # The cron provider models a real blocking scheduler lifecycle: start owns
    # the thread until the shared stop event is signalled, and stop records the
    # explicit provider teardown call.
    class _FakeCronProvider:
        def __init__(self):
            self.started = threading.Event()
            self.exited = threading.Event()
            self.stop_called = threading.Event()
            self._stop_event = None

        def start(self, stop_event, adapters=None, loop=None):
            self._stop_event = stop_event
            self.started.set()
            stop_event.wait()
            self.exited.set()

        def stop(self):
            self.stop_called.set()
            if self._stop_event is not None:
                self._stop_event.set()

    import cron.scheduler_provider as csp

    scheduler = _FakeCronProvider()
    monkeypatch.setattr(csp, "resolve_cron_scheduler", lambda: scheduler)

    result = await gr.start_gateway(verbosity=None)

    assert result is True
    assert runner.started.is_set()
    assert runner.shutdown_requested.is_set()
    assert scheduler.started.is_set()
    assert scheduler.stop_called.is_set()
    assert scheduler.exited.is_set()
    assert start_calls == [42.0], f"expected start_memory_monitoring(42.0), got {start_calls}"
    assert stop_calls == [True], f"expected stop_memory_monitoring called once, got {stop_calls}"
