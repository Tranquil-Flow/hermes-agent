"""Regression tests for #52943 — resume provider mismatch warning.

When a session is resumed but its original provider differs from the current
default, the user must be warned rather than silently switching providers.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock


def _make_mixin(provider="custom:mydeepseek", session_id="current-session-123"):
    """Create a CLICommandsMixin with minimal attributes for _handle_resume_command."""
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    mixin = CLICommandsMixin()
    mixin.provider = provider
    mixin.session_id = session_id
    mixin._pending_resume_sessions = None
    mixin.conversation_history = []
    mixin.agent = None  # No agent yet — the check at L758 will skip
    mixin._display_resumed_history = mock.MagicMock()  # Skip UI rendering
    return mixin


def _make_mock_db(model_config=None):
    """Create a mock SessionDB with the given model_config JSON."""
    db = mock.MagicMock()
    meta = {"id": "target-456", "title": "Test Session"}
    if model_config is not None:
        meta["model_config"] = model_config
    db.get_session.return_value = meta
    db.resolve_resume_session_id.return_value = "target-456"
    db.get_messages_as_conversation.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    return db


class TestResumeProviderMismatch:
    """Verify that _handle_resume_command warns when the session's stored
    provider differs from the current default."""

    def test_warns_on_provider_mismatch(self):
        """Resuming a session created with a different provider prints a warning."""
        mixin = _make_mixin(provider="custom:mydeepseek")
        mixin._session_db = _make_mock_db(
            model_config=json.dumps({
                "provider": "zai",
                "model": "glm-5.1",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            })
        )

        with mock.patch("cli._cprint") as mock_cprint, \
             mock.patch("cli._sync_process_session_id"):
            mixin._handle_resume_command("/resume target-456")

        # Verify the warning was printed about provider mismatch
        warning_calls = [
            str(call_args[0][0]) for call_args in mock_cprint.call_args_list
            if "Provider changed" in str(call_args[0][0])
        ]
        assert len(warning_calls) == 1, (
            f"Expected 1 provider-change warning, got {len(warning_calls)}. "
            f"All _cprint calls: {mock_cprint.call_args_list}"
        )
        warning = warning_calls[0]
        assert "zai" in warning
        assert "custom:mydeepseek" in warning
        assert "Resuming with the current provider" in warning

        # Verify the session was still switched (resume proceeds)
        assert mixin.session_id == "target-456"
        assert mixin._resumed is True

    def test_no_warning_when_provider_matches(self):
        """No warning when session provider matches current default."""
        mixin = _make_mixin(provider="zai")
        mixin._session_db = _make_mock_db(
            model_config=json.dumps({
                "provider": "zai",  # same provider
                "model": "glm-5.1",
            })
        )

        with mock.patch("cli._cprint") as mock_cprint, \
             mock.patch("cli._sync_process_session_id"):
            mixin._handle_resume_command("/resume target-456")

        warning_calls = [
            str(call_args[0][0]) for call_args in mock_cprint.call_args_list
            if "Provider changed" in str(call_args[0][0])
        ]
        assert len(warning_calls) == 0

        assert mixin.session_id == "target-456"
        assert mixin._resumed is True

    def test_no_warning_when_no_model_config(self):
        """No warning when session_meta has no model_config field."""
        mixin = _make_mixin(provider="openai")
        mixin._session_db = _make_mock_db(model_config=None)

        with mock.patch("cli._cprint") as mock_cprint, \
             mock.patch("cli._sync_process_session_id"):
            mixin._handle_resume_command("/resume target-456")

        warning_calls = [
            str(call_args[0][0]) for call_args in mock_cprint.call_args_list
            if "Provider changed" in str(call_args[0][0])
        ]
        assert len(warning_calls) == 0
        assert mixin.session_id == "target-456"


def test_normal_agent_persists_resolved_provider_in_real_session_db(tmp_path):
    """Normal CLI-created rows must retain provider identity for later resume."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=Path(tmp_path) / "state.db")
    try:
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            provider="openrouter",
            quiet_mode=True,
            session_db=db,
            session_id="provider-persistence",
            skip_context_files=True,
            skip_memory=True,
        )
        agent._ensure_db_session()

        meta = db.get_session("provider-persistence")
        stored = meta["model_config"]
        config = json.loads(stored) if isinstance(stored, str) else stored
        assert config["provider"] == "openrouter"
        assert config["model"] == "test/model"
    finally:
        db.close()


def test_preload_warns_with_nonempty_startup_transcript():
    """Preload must warn before non-empty history bypasses _init_agent loading."""
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    mixin = CLIAgentSetupMixin()
    mixin.provider = "openrouter"
    mixin.session_id = "startup-resume"
    mixin._resumed = True
    mixin.conversation_history = []
    mixin._console_print = mock.MagicMock()
    mixin._restore_session_cwd = mock.MagicMock()
    mixin._session_db = _make_mock_db(
        model_config=json.dumps({"provider": "zai", "model": "glm-5.1"})
    )
    mixin._session_db.resolve_resume_session_id.return_value = "startup-resume"

    assert mixin._preload_resumed_session() is True
    assert mixin.conversation_history
    output = "\n".join(str(call.args[0]) for call in mixin._console_print.call_args_list)
    assert "Provider changed" in output
    assert "zai" in output
    assert "openrouter" in output
