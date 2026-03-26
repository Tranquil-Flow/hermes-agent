import logging
from io import StringIO
import subprocess
import sys
import types

import pytest

from tools.environments import docker as docker_env


def _mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to intercept docker run -d and docker version calls.

    Returns a list of captured (cmd, kwargs) tuples for inspection.
    """
    calls = []

    def _run(cmd, **kwargs):
        calls.append((list(cmd) if isinstance(cmd, list) else cmd, kwargs))
        if isinstance(cmd, list) and len(cmd) >= 2:
            if cmd[1] == "version":
                return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")
            if cmd[1] == "run":
                return subprocess.CompletedProcess(cmd, 0, stdout="fake-container-id\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(docker_env.subprocess, "run", _run)
    return calls


def _make_dummy_env(**kwargs):
    """Helper to construct DockerEnvironment with minimal required args."""
    return docker_env.DockerEnvironment(
        image=kwargs.get("image", "python:3.11"),
        cwd=kwargs.get("cwd", "/root"),
        timeout=kwargs.get("timeout", 60),
        cpu=kwargs.get("cpu", 0),
        memory=kwargs.get("memory", 0),
        disk=kwargs.get("disk", 0),
        persistent_filesystem=kwargs.get("persistent_filesystem", False),
        task_id=kwargs.get("task_id", "test-task"),
        volumes=kwargs.get("volumes", []),
        network=kwargs.get("network", True),
        host_cwd=kwargs.get("host_cwd"),
        auto_mount_cwd=kwargs.get("auto_mount_cwd", False),
    )


def test_ensure_docker_available_logs_and_raises_when_not_found(monkeypatch, caplog):
    """When docker cannot be found, raise a clear error before container setup."""

    monkeypatch.setattr(docker_env, "find_docker", lambda: None)
    monkeypatch.setattr(
        docker_env.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("subprocess.run should not be called when docker is missing"),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker executable not found in PATH or known install locations" in str(excinfo.value)
    assert any(
        "no docker executable was found in PATH or known install locations"
        in record.getMessage()
        for record in caplog.records
    )


def test_ensure_docker_available_logs_and_raises_on_timeout(monkeypatch, caplog):
    """When docker version times out, surface a helpful error instead of hanging."""

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["/custom/docker", "version"], timeout=5)

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/custom/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _raise_timeout)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError) as excinfo:
            _make_dummy_env()

    assert "Docker daemon is not responding" in str(excinfo.value)
    assert any(
        "/custom/docker version' timed out" in record.getMessage()
        for record in caplog.records
    )


def test_ensure_docker_available_uses_resolved_executable(monkeypatch):
    """When docker is found outside PATH, preflight should use that resolved path."""

    calls = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="Docker version", stderr="")

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/opt/homebrew/bin/docker")
    monkeypatch.setattr(docker_env.subprocess, "run", _run)

    docker_env._ensure_docker_available()

    assert calls == [
        (["/opt/homebrew/bin/docker", "version"], {
            "capture_output": True,
            "text": True,
            "timeout": 5,
        })
    ]


def test_auto_mount_host_cwd_adds_volume(monkeypatch, tmp_path):
    """Opt-in docker cwd mounting should bind the host cwd to /workspace."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
    )

    # Find the docker run call and check its args
    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" in run_args_str


def test_auto_mount_disabled_by_default(monkeypatch, tmp_path):
    """Host cwd should not be mounted unless the caller explicitly opts in."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/root",
        host_cwd=str(project_dir),
        auto_mount_cwd=False,
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" not in run_args_str


def test_auto_mount_skipped_when_workspace_already_mounted(monkeypatch, tmp_path):
    """Explicit user volumes for /workspace should take precedence over cwd mount."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        volumes=[f"{other_dir}:/workspace"],
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{other_dir}:/workspace" in run_args_str
    assert run_args_str.count(":/workspace") == 1


def test_auto_mount_replaces_persistent_workspace_bind(monkeypatch, tmp_path):
    """Persistent mode should still prefer the configured host cwd at /workspace."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()

    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    _make_dummy_env(
        cwd="/workspace",
        persistent_filesystem=True,
        host_cwd=str(project_dir),
        auto_mount_cwd=True,
        task_id="test-persistent-auto-mount",
    )

    run_calls = [c for c in calls if isinstance(c[0], list) and len(c[0]) >= 2 and c[0][1] == "run"]
    assert run_calls, "docker run should have been called"
    run_args_str = " ".join(run_calls[0][0])
    assert f"{project_dir}:/workspace" in run_args_str
    assert "/sandboxes/docker/test-persistent-auto-mount/workspace:/workspace" not in run_args_str


def test_non_persistent_cleanup_removes_container(monkeypatch):
    """When persistent=false, cleanup() must schedule docker stop + rm."""
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    calls = _mock_subprocess_run(monkeypatch)

    popen_cmds = []
    monkeypatch.setattr(
        docker_env.subprocess, "Popen",
        lambda cmd, **kw: (popen_cmds.append(cmd), type("P", (), {"poll": lambda s: 0, "wait": lambda s, **k: None, "returncode": 0, "stdout": iter([]), "stdin": None})())[1],
    )

    env = _make_dummy_env(persistent_filesystem=False, task_id="ephemeral-task")
    assert env._container_id
    container_id = env._container_id

    env.cleanup()

    # Should have stop and rm calls via Popen
    stop_cmds = [c for c in popen_cmds if container_id in str(c) and "stop" in str(c)]
    assert len(stop_cmds) >= 1, f"cleanup() should schedule docker stop for {container_id}"


class _FakePopen:
    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = StringIO("")
        self.stdin = None
        self.returncode = 0

    def poll(self):
        return self.returncode


def _make_execute_only_env(forward_env=None):
    env = docker_env.DockerEnvironment.__new__(docker_env.DockerEnvironment)
    env.cwd = "/root"
    env.timeout = 60
    env._forward_env = forward_env or []
    env._prepare_command = lambda command: (command, None)
    env._timeout_result = lambda timeout: {"output": f"timed out after {timeout}", "returncode": 124}
    env._container_id = "test-container"
    env._docker_exe = "/usr/bin/docker"
    return env


def test_execute_uses_hermes_dotenv_for_allowlisted_env(monkeypatch):
    env = _make_execute_only_env(["GITHUB_TOKEN"])
    popen_calls = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"GITHUB_TOKEN": "value_from_dotenv"})
    monkeypatch.setattr(docker_env.subprocess, "Popen", _fake_popen)

    result = env.execute("echo hi")

    assert result["returncode"] == 0
    assert "GITHUB_TOKEN=value_from_dotenv" in popen_calls[0]


def test_execute_prefers_shell_env_over_hermes_dotenv(monkeypatch):
    env = _make_execute_only_env(["GITHUB_TOKEN"])
    popen_calls = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setenv("GITHUB_TOKEN", "value_from_shell")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {"GITHUB_TOKEN": "value_from_dotenv"})
    monkeypatch.setattr(docker_env.subprocess, "Popen", _fake_popen)

    env.execute("echo hi")

    assert "GITHUB_TOKEN=value_from_shell" in popen_calls[0]
    assert "GITHUB_TOKEN=value_from_dotenv" not in popen_calls[0]


def test_execute_rewrites_localhost_proxy_for_docker(monkeypatch):
    """HTTP_PROXY/HTTPS_PROXY 127.0.0.1 should become host.docker.internal."""
    env = _make_execute_only_env(["HTTP_PROXY", "HTTPS_PROXY"])
    popen_calls = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8444")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8444")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})
    monkeypatch.setattr(docker_env.subprocess, "Popen", _fake_popen)

    env.execute("echo hi")

    cmd = popen_calls[0]
    assert "HTTP_PROXY=http://host.docker.internal:8444" in cmd
    assert "HTTPS_PROXY=http://host.docker.internal:8444" in cmd


def test_execute_rewrites_localhost_proxy_preserves_non_proxy(monkeypatch):
    """Non-proxy vars should NOT have 127.0.0.1 rewritten."""
    env = _make_execute_only_env(["HTTP_PROXY", "AEGIS_ACTIVE"])
    popen_calls = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setenv("HTTP_PROXY", "http://localhost:8444")
    monkeypatch.setenv("AEGIS_ACTIVE", "1")
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})
    monkeypatch.setattr(docker_env.subprocess, "Popen", _fake_popen)

    env.execute("echo hi")

    cmd = popen_calls[0]
    assert "HTTP_PROXY=http://host.docker.internal:8444" in cmd
    assert "AEGIS_ACTIVE=1" in cmd


def test_execute_translates_cert_host_path_to_container_path(monkeypatch):
    """SSL_CERT_FILE and friends must point to /certs/ inside the container, not the host path."""
    env = _make_execute_only_env(["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
                                   "CURL_CA_BUNDLE", "GIT_SSL_CAINFO", "PIP_CERT"])
    popen_calls = []

    def _fake_popen(cmd, **kwargs):
        popen_calls.append(cmd)
        return _FakePopen(cmd, **kwargs)

    host_cert = "/Users/someone/.mitmproxy/mitmproxy-ca-cert.pem"
    monkeypatch.setenv("SSL_CERT_FILE", host_cert)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", host_cert)
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", host_cert)
    monkeypatch.setenv("CURL_CA_BUNDLE", host_cert)
    monkeypatch.setenv("GIT_SSL_CAINFO", host_cert)
    monkeypatch.setenv("PIP_CERT", host_cert)
    monkeypatch.setattr(docker_env, "_load_hermes_env_vars", lambda: {})
    monkeypatch.setattr(docker_env.subprocess, "Popen", _fake_popen)

    env.execute("echo hi")

    cmd = popen_calls[0]
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
                "CURL_CA_BUNDLE", "GIT_SSL_CAINFO", "PIP_CERT"):
        assert f"{var}=/certs/mitmproxy-ca-cert.pem" in cmd, f"{var} not translated"
        assert host_cert not in " ".join(
            v for v in cmd if v.startswith(f"{var}=")
        ), f"{var} still contains host path"


def test_aegis_cert_mounted_when_aegis_active(monkeypatch, tmp_path):
    """When AEGIS_ACTIVE=1 and the mitmproxy cert exists, it should be volume-mounted."""
    cert_file = tmp_path / ".mitmproxy" / "mitmproxy-ca-cert.pem"
    cert_file.parent.mkdir()
    cert_file.write_text("fake cert")

    monkeypatch.setenv("AEGIS_ACTIVE", "1")
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")
    # Patch Path.home() to point at tmp_path so the cert is "found"
    import pathlib
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

    calls = _mock_subprocess_run(monkeypatch)
    _make_dummy_env()

    run_call = next(c for c in calls if isinstance(c[0], list) and "run" in c[0])
    run_args = " ".join(run_call[0])
    assert "/certs/mitmproxy-ca-cert.pem" in run_args
    assert "mitmproxy-ca-cert.pem" in run_args


def test_aegis_cert_not_mounted_when_aegis_inactive(monkeypatch, tmp_path):
    """When AEGIS_ACTIVE is not set, no cert volume mount should be added."""
    cert_file = tmp_path / ".mitmproxy" / "mitmproxy-ca-cert.pem"
    cert_file.parent.mkdir()
    cert_file.write_text("fake cert")

    monkeypatch.delenv("AEGIS_ACTIVE", raising=False)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "/usr/bin/docker")

    calls = _mock_subprocess_run(monkeypatch)
    _make_dummy_env()

    run_call = next(c for c in calls if isinstance(c[0], list) and "run" in c[0])
    run_args = " ".join(run_call[0])
    assert "/certs/mitmproxy-ca-cert.pem" not in run_args
