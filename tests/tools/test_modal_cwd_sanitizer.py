"""Regression tests for Windows host paths leaking into Modal CWDs."""

from unittest.mock import patch

import pytest

from tools.environments.base import BaseEnvironment
from tools.environments.modal import ModalEnvironment
from tools.environments.modal_utils import (
    BaseModalExecutionEnvironment,
    ModalExecStart,
    sanitize_modal_cwd,
)


@pytest.mark.parametrize(
    "path",
    [
        "D:\\Users\\eve\\repo",
        "E:/dev/repo",
        "Z:\\data\\repo",
        "a:/lowercase-drive",
        "/Users/eve/repo",
        "/home/eve/repo",
        "relative/repo",
    ],
)
def test_shared_sanitizer_rejects_host_and_relative_paths(path):
    assert sanitize_modal_cwd(path) == "/root"


def test_shared_sanitizer_preserves_sandbox_absolute_path():
    assert sanitize_modal_cwd("/workspace/project") == "/workspace/project"


def test_terminal_modal_config_uses_shared_sanitizer(monkeypatch):
    from tools import terminal_tool

    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_CWD", "D:\\Users\\eve\\repo")

    assert terminal_tool._get_env_config()["cwd"] == "/root"


def test_direct_modal_constructor_sanitizes_cwd_before_base_init():
    class StopAfterBaseInit(RuntimeError):
        pass

    with patch.object(
        BaseEnvironment,
        "__init__",
        side_effect=StopAfterBaseInit,
    ) as base_init:
        with pytest.raises(StopAfterBaseInit):
            ModalEnvironment(image="debian:bookworm", cwd="D:\\Users\\eve\\repo")

    base_init.assert_called_once_with(cwd="/root", timeout=60)


def test_direct_modal_execute_sanitizes_per_call_override():
    env = object.__new__(ModalEnvironment)
    env.cwd = "/workspace/default"

    with patch.object(BaseEnvironment, "execute", return_value={"returncode": 0}) as execute:
        result = ModalEnvironment.execute(env, "pwd", cwd="E:/host/project")

    assert result == {"returncode": 0}
    assert execute.call_args.kwargs["cwd"] == "/workspace/default"


class _StubManagedModal(BaseModalExecutionEnvironment):
    def _start_modal_exec(self, prepared):
        return ModalExecStart(immediate_result={"output": "", "returncode": 0})

    def _poll_modal_exec(self, handle):
        return None

    def _cancel_modal_exec(self, handle):
        return None

    def _run_bash(self, *args, **kwargs):  # pragma: no cover - abstract base contract
        raise NotImplementedError

    def cleanup(self):
        return None


def test_managed_modal_prepare_sanitizes_per_call_override():
    env = object.__new__(_StubManagedModal)
    env.cwd = "/workspace/default"
    env.timeout = 60
    env._prepare_command = lambda command: (command, None)

    prepared = env._prepare_modal_exec("pwd", cwd="Z:\\host\\repo")

    assert prepared.cwd == "/root"
