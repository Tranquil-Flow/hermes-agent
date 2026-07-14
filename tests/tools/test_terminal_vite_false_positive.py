"""Tests for long-lived foreground process detection in terminal_tool.

Ensures that package manager commands like ``npm update vite`` are not
incorrectly flagged as long-lived server processes (false positive fix
for issue #42620), while chained commands like ``npm install vite && vite``
remain blocked because the trailing ``vite`` is a standalone server invocation.

The exemption is scoped to the matched keyword within its own shell-command
segment, so package-manager arguments exempt the match but a subsequent
foreground invocation in a chained command does not.
"""
from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

from tools.terminal_tool import (
    _is_pm_package_argument,
    _matched_keyword,
    _shell_segment_containing,
)


# ---------------------------------------------------------------------------
# Pure unit tests for the new helpers
# ---------------------------------------------------------------------------


class TestShellSegmentContaining:
    """``_shell_segment_containing`` must isolate the matched keyword to the
    segment delimited by ``&&``, ``||``, ``;``, ``|``, parens, and newlines."""

    def test_no_separators_returns_whole_string(self):
        assert _shell_segment_containing("npm install vite", 12) == "npm install vite"

    def test_chained_and_returns_left_segment_for_left_match(self):
        # First 'vite' lives in the left segment.
        text = "npm install vite && vite"
        left = _shell_segment_containing(text, text.index("vite"))  # first occurrence
        assert left.strip() == "npm install vite"

    def test_chained_and_returns_right_segment_for_right_match(self):
        text = "npm install vite && vite"
        right = _shell_segment_containing(text, text.rindex("vite"))  # last occurrence
        assert right.strip() == "vite"

    def test_chained_semicolon(self):
        text = "npm install vite; vite build"
        right = _shell_segment_containing(text, text.rindex("vite"))
        assert right.strip() == "vite build"

    def test_chained_or(self):
        text = "npm install vite || vite preview"
        right = _shell_segment_containing(text, text.rindex("vite"))
        assert right.strip() == "vite preview"

    def test_newline_is_separator(self):
        text = "npm install vite\nvite build"
        right = _shell_segment_containing(text, text.rindex("vite"))
        assert right.strip() == "vite build"

    def test_pipe_is_separator(self):
        text = "echo hi | vite build"
        right = _shell_segment_containing(text, text.index("vite"))
        assert right.strip() == "vite build"

    def test_parens_isolate_segments(self):
        text = "(npm install vite) && vite"
        # First vite lives in the parens segment.
        left = _shell_segment_containing(text, text.index("vite"))
        assert "npm install" in left
        assert "&&" not in left


class TestIsPmPackageArgument:
    """``_is_pm_package_argument`` must only return True when the keyword is
    a package name argument to a pm install verb in the SAME shell segment."""

    def test_simple_npm_install_vite(self):
        text = "npm install vite"
        # Find the 'vite' match.
        m = re.search(r"\bvite(?:\s|$)", text)
        assert _is_pm_package_argument(text, "vite", m) is True

    def test_pnpm_add_nodemon(self):
        text = "pnpm add nodemon"
        m = re.search(r"\bnodemon\b", text)
        assert _is_pm_package_argument(text, "nodemon", m) is True

    def test_chained_command_trailing_vite_is_NOT_package_arg(self):
        text = "npm install vite && vite"
        # The trailing standalone vite should NOT qualify.
        m = list(re.finditer(r"\bvite(?:\s|$)", text))[-1]
        assert _is_pm_package_argument(text, "vite", m) is False

    def test_chained_semicolon_trailing_vite_is_NOT_package_arg(self):
        text = "npm install vite; vite"
        m = list(re.finditer(r"\bvite(?:\s|$)", text))[-1]
        assert _is_pm_package_argument(text, "vite", m) is False

    def test_chained_or_trailing_vite_is_NOT_package_arg(self):
        text = "npm install vite || vite build"
        # The trailing 'vite build' should NOT qualify as a package arg.
        m = list(re.finditer(r"\bvite\b", text))[-1]
        assert _is_pm_package_argument(text, "vite", m) is False

    def test_chained_command_pm_arg_vite_DOES_qualify(self):
        text = "npm install vite && vite"
        # The first 'vite' (the install argument) DOES qualify.
        m = re.search(r"\bvite(?:\s|$)", text)
        assert _is_pm_package_argument(text, "vite", m) is True

    def test_standalone_vite_not_pm_argument(self):
        text = "vite"
        m = re.search(r"\bvite(?:\s|$)", text)
        assert _is_pm_package_argument(text, "vite", m) is False

    def test_npm_run_dev_not_pm_install(self):
        # 'dev' is not a package-manager install verb.
        text = "npm run dev"
        m = re.search(r"\bdev\b", text)
        assert _is_pm_package_argument(text, "dev", m) is False

    def test_no_pm_verb_returns_false(self):
        text = "echo vite && vite"
        m = re.search(r"\bvite(?:\s|$)", text)
        assert _is_pm_package_argument(text, "vite", m) is False


class TestMatchedKeyword:
    def test_returns_first_token(self):
        m = re.search(r"\bvite(?:\s|$)", "vite build --port 3000")
        assert _matched_keyword(m) == "vite"

    def test_lowercases(self):
        m = re.search(r"\bNodemon\b", "Nodemon")
        assert _matched_keyword(m) == "nodemon"


# ---------------------------------------------------------------------------
# End-to-end tests using the full terminal_tool path
# ---------------------------------------------------------------------------


def _make_env_config(**overrides):
    """Return a minimal _get_env_config()-shaped dict with optional overrides."""
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


def _run_terminal(command: str) -> dict:
    """Run terminal_tool with mocked env and return the parsed result."""
    from tools.terminal_tool import terminal_tool

    with patch("tools.terminal_tool._get_env_config", return_value=_make_env_config()), \
         patch("tools.terminal_tool._start_cleanup_thread"):

        mock_env = MagicMock()
        mock_env.execute.return_value = {"output": "done", "returncode": 0}

        with patch("tools.terminal_tool._active_environments", {"default": mock_env}), \
             patch("tools.terminal_tool._last_activity", {"default": 0}), \
             patch("tools.terminal_tool._check_all_guards", return_value={"approved": True}):
            return json.loads(terminal_tool(command=command))


def _assert_blocked_as_long_lived(result: dict) -> None:
    """Assert the terminal_tool rejected the command as long-lived foreground."""
    assert result.get("exit_code") == -1, f"expected block, got: {result}"
    assert "long-lived" in result["error"].lower()


class TestViteFalsePositive:
    """npm/pnpm/yarn/bun package operations with 'vite' as a package name
    should NOT be blocked as long-lived processes."""

    def test_npm_update_vite_not_blocked(self):
        result = _run_terminal("npm update vite")
        assert result.get("error") is None

    def test_npm_install_vite_not_blocked(self):
        result = _run_terminal("npm install vite")
        assert result.get("error") is None

    def test_npm_install_vite_save_dev_not_blocked(self):
        result = _run_terminal("npm install vite --save-dev")
        assert result.get("error") is None

    def test_npm_remove_vite_not_blocked(self):
        result = _run_terminal("npm remove vite")
        assert result.get("error") is None

    def test_pnpm_add_vite_not_blocked(self):
        result = _run_terminal("pnpm add vite")
        assert result.get("error") is None

    def test_yarn_add_vite_not_blocked(self):
        result = _run_terminal("yarn add vite")
        assert result.get("error") is None

    def test_bun_add_vite_not_blocked(self):
        result = _run_terminal("bun add vite")
        assert result.get("error") is None

    def test_npm_uninstall_vite_not_blocked(self):
        result = _run_terminal("npm uninstall vite")
        assert result.get("error") is None

    def test_npm_update_vite_with_other_packages_not_blocked(self):
        result = _run_terminal("npm update vite lodash axios")
        assert result.get("error") is None


class TestViteStillBlocked:
    """vite as a standalone dev server command should still be blocked."""

    def test_vite_standalone_blocked(self):
        result = _run_terminal("vite")
        _assert_blocked_as_long_lived(result)

    def test_vite_dev_blocked(self):
        result = _run_terminal("vite dev")
        _assert_blocked_as_long_lived(result)

    def test_vite_build_blocked(self):
        result = _run_terminal("vite build")
        _assert_blocked_as_long_lived(result)

    def test_vite_preview_blocked(self):
        result = _run_terminal("vite preview --port 3000")
        _assert_blocked_as_long_lived(result)


class TestOtherPmPackageArgsNotBlocked:
    """Other package manager commands with server-tool package names should
    also not be falsely blocked."""

    def test_npm_install_nodemon_not_blocked(self):
        result = _run_terminal("npm install nodemon")
        assert result.get("error") is None

    def test_npm_update_nodemon_not_blocked(self):
        result = _run_terminal("npm update nodemon")
        assert result.get("error") is None


class TestNonPmViteCommandsStillBlocked:
    """Commands that use vite as a server (not package manager arg) should
    still be blocked."""

    def test_npx_vite_blocked(self):
        result = _run_terminal("npx vite")
        _assert_blocked_as_long_lived(result)

    def test_npm_run_dev_still_blocked(self):
        result = _run_terminal("npm run dev")
        _assert_blocked_as_long_lived(result)


class TestChainedCommandsRemainBlocked:
    """Regression: the pm-argument exemption must NOT leak across shell-command
    boundaries. A chained foreground invocation like ``npm install vite && vite``
    must still be blocked, because the trailing ``vite`` is a standalone server
    invocation, not a package argument.

    This is the sweeper-flagged defect in the original PR: the unscoped matcher
    would find ``vite`` anywhere after the first pm verb and exempt the entire
    command, including a later foreground vite dev server.
    """

    def test_chained_and_trailing_vite_blocked(self):
        result = _run_terminal("npm install vite && vite")
        _assert_blocked_as_long_lived(result)

    def test_chained_semicolon_trailing_vite_blocked(self):
        result = _run_terminal("npm install vite; vite")
        _assert_blocked_as_long_lived(result)

    def test_chained_and_trailing_vite_build_blocked(self):
        result = _run_terminal("npm install vite && vite build")
        _assert_blocked_as_long_lived(result)

    def test_chained_or_trailing_vite_preview_blocked(self):
        result = _run_terminal("npm install vite || vite preview --port 3000")
        _assert_blocked_as_long_lived(result)

    def test_chained_and_trailing_nodemon_blocked(self):
        # Symmetric: same defect class for the nodemon pattern.
        result = _run_terminal("npm install nodemon && nodemon")
        _assert_blocked_as_long_lived(result)

    def test_chained_semicolon_trailing_nodemon_blocked(self):
        result = _run_terminal("npm install nodemon; nodemon server.js")
        _assert_blocked_as_long_lived(result)

    def test_triple_chain_trailing_vite_blocked(self):
        result = _run_terminal("npm install vite && echo done && vite")
        _assert_blocked_as_long_lived(result)

    def test_newline_separator_trailing_vite_blocked(self):
        result = _run_terminal("npm install vite\nvite")
        _assert_blocked_as_long_lived(result)