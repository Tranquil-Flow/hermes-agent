"""Regression tests for ACP Copilot conversation_history normalization.

Verifies that non-iterable types (e.g. SimpleNamespace) passed as
conversation_history are handled gracefully instead of crashing. The
entrypoint regression exercises ``agent.turn_context.build_turn_context()``
so the live prologue path is covered — the helper-only tests on
``normalize_conversation_history`` are kept as fast unit checks but the
PR reviewer's guidance was to test the live boundary, not just the helper.

See: issue #11732
"""

import logging
from types import SimpleNamespace

import pytest

from run_agent import normalize_conversation_history


class TestConversationHistoryNormalization:
    """Tests for the helper ``normalize_conversation_history()``."""

    def test_normalizes_simplenamespace(self, caplog):
        """ACP Copilot can pass a SimpleNamespace instead of a list."""
        ns = SimpleNamespace(foo="bar", baz=123)
        with caplog.at_level(logging.WARNING):
            result = normalize_conversation_history(ns)

        assert result == []
        assert any(
            "SimpleNamespace" in record.message and "not iterable" in record.message
            for record in caplog.records
        )

    def test_normalizes_none(self, caplog):
        """Passing None should be treated as an empty list."""
        result = normalize_conversation_history(None)
        assert result == []
        assert not any(
            "not iterable" in record.message for record in caplog.records
        )

    def test_preserves_valid_list(self):
        """A normal list of message dicts passes through unchanged."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = normalize_conversation_history(messages)
        assert result == messages

    def test_preserves_empty_list(self):
        """An empty list remains an empty list."""
        result = normalize_conversation_history([])
        assert result == []

    def test_normalizes_non_iterable_int(self, caplog):
        """An int is not iterable and should be treated as empty with a warning."""
        with caplog.at_level(logging.WARNING):
            result = normalize_conversation_history(42)
        assert result == []
        assert any(
            "int" in record.message and "not iterable" in record.message
            for record in caplog.records
        )

    def test_string_becomes_list_of_chars(self):
        """A bare string is iterable (chars) — list() works, documenting current behavior."""
        result = normalize_conversation_history("hello")
        assert result == list("hello")


class TestBuildTurnContextNormalizesHistory:
    """Entrypoint-level coverage: ``build_turn_context`` must normalize before
    reaching the ``len(conversation_history)`` log site and the
    ``list(conversation_history)`` initialiser used downstream.

    This is the regression Teknium requested — without it, the patch's
    helper change can pass while the live path still crashes on a
    non-iterable SimpleNamespace passed through the ACP adapter.
    """

    @staticmethod
    def _build_agent_like():
        return SimpleNamespace(
            session_id=None,
            model="gpt-5.4",
            provider="custom",
            platform="acp",
            api_mode="openai_chat",
            quiet_mode=True,
            max_iterations=4,
            _compression_warning=None,
            _user_turn_count=0,
            _memory_store=None,
            _memory_nudge_interval=0,
            _turns_since_memory=0,
            _todo_store=SimpleNamespace(has_items=lambda: False),
            _stream_context_scrubber=None,
            _stream_think_scrubber=None,
            _persist_user_message_idx=None,
            _persist_user_message_override=None,
            _persist_user_message_timestamp=None,
            _is_user_initiated_turn=False,
            _anthropic_image_fallback_cache={},
            _hydrate_todo_store=lambda _h: None,
            _global_ext_prefetch_cache={},
            _last_unified_callback=None,
            _stream_passthrough_active=False,
            _skip_mcp_refresh=True,
            _restore_primary_runtime=lambda: None,
            _tool_guardrails=SimpleNamespace(reset_for_turn=lambda: None),
            _cached_system_prompt="system prompt",
            _ensure_db_session=lambda: None,
            _persist_session=lambda _messages, _history: None,
            compression_enabled=False,
            context_compressor=SimpleNamespace(
                protect_first_n=0,
                protect_last_n=0,
                threshold_tokens=999999,
            ),
            tools=[],
            _interrupt_requested=False,
            _interrupt_thread_signal_pending=False,
            _interrupt_message=None,
            _memory_manager=None,
        )

    @staticmethod
    def _build_kwargs(agent, history):
        return dict(
            agent=agent,
            user_message="hi",
            system_message=None,
            conversation_history=history,
            task_id=None,
            stream_callback=None,
            persist_user_message=None,
            restore_or_build_system_prompt=lambda a, s, h: None,
            install_safe_stdio=lambda: None,
            sanitize_surrogates=lambda x: x,
            summarize_user_message_for_log=lambda x: x,
            set_session_context=lambda *a, **k: None,
            set_current_write_origin=lambda *a, **k: None,
            ra=lambda: SimpleNamespace(
                estimate_request_tokens_rough=lambda *a, **k: 0,
                _set_interrupt=lambda *a, **k: None,
            ),
        )

    def test_build_turn_context_does_not_crash_on_namespace(self):
        """Live prologue: a SimpleNamespace history must NOT raise TypeError."""
        from agent.turn_context import build_turn_context
        agent = self._build_agent_like()
        bad_history = SimpleNamespace(legacy_field="not a list")
        build_turn_context(**self._build_kwargs(agent, bad_history))

    def test_build_turn_context_logs_simplenamespace_warning(self, caplog):
        """Live prologue: the non-iterable warning fires when handed a SimpleNamespace."""
        from agent.turn_context import build_turn_context
        agent = self._build_agent_like()
        bad_history = SimpleNamespace(legacy_field="not a list")
        with caplog.at_level(logging.WARNING):
            build_turn_context(**self._build_kwargs(agent, bad_history))
        assert any(
            "SimpleNamespace" in record.message and "not iterable" in record.message
            for record in caplog.records
        ), (
            "build_turn_context prologue must emit a non-iterable warning for "
            "SimpleNamespace conversation_history"
        )

    def test_build_turn_context_preserves_normal_list(self):
        """Live prologue: a real list of message dicts passes through unchanged."""
        from agent.turn_context import build_turn_context
        agent = self._build_agent_like()
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        ctx = build_turn_context(**self._build_kwargs(agent, list(history)))
        assert isinstance(ctx.conversation_history, list)
        assert list(ctx.messages)[:2] == history
