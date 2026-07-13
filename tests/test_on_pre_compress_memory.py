"""Behavior contracts for provider context at compaction boundaries."""

import copy
from types import SimpleNamespace

import pytest

from agent.context_compressor import (
    _SUMMARY_END_MARKER,
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    LEGACY_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
    sanitize_pre_compress_context,
)
from agent.conversation_compression import compress_context
from agent.redact import redact_url_credentials


def _messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first decision " + "x" * 200},
        {"role": "assistant", "content": "first response " + "y" * 200},
        {"role": "user", "content": "second decision " + "z" * 200},
        {"role": "assistant", "content": "second response " + "q" * 200},
        {"role": "user", "content": "recent request"},
        {"role": "assistant", "content": "recent response"},
    ]


class _Manager:
    def __init__(self, result=""):
        self.result = result
        self.calls = 0
        self.observed_messages = []

    def on_pre_compress(self, messages):
        self.calls += 1
        self.observed_messages.append(messages)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _CapturingEngine:
    compression_count = 0
    _last_compress_aborted = False

    def __init__(self):
        self.calls = []

    def compress(
        self,
        messages,
        current_tokens=None,
        focus_topic=None,
        force=False,
        pre_compress_context="",
    ):
        self.calls.append(
            {
                "current_tokens": current_tokens,
                "focus_topic": focus_topic,
                "force": force,
                "pre_compress_context": pre_compress_context,
            }
        )
        return list(messages)


def _agent(manager, engine, *, required=False):
    warnings = []
    statuses = []
    agent = SimpleNamespace(
        session_id="session-a",
        model="test-model",
        compression_memory_checkpoint_required=required,
        context_compressor=engine,
        _memory_manager=manager,
        _session_db=None,
        _compression_feasibility_checked=True,
        _cached_system_prompt="cached-system-prompt",
        _emit_status=statuses.append,
        _emit_warning=warnings.append,
        _build_system_prompt=lambda _message: "rebuilt-system-prompt",
        _invalidate_system_prompt=lambda: None,
        _todo_store=SimpleNamespace(format_for_injection=lambda: ""),
        _vprint=lambda *_args, **_kwargs: None,
        tools=[],
        platform="cli",
        log_prefix="",
    )
    return agent, warnings, statuses


def _compressor():
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    return compressor


def test_provider_context_is_sanitized_and_passed_to_engine():
    manager = _Manager(
        "checkpoint: block-123\n"
        "https://api.example/resume?client%5Fsecret=opaque&view=public\n"
        "custom://userinfo@checkpoint.example/block-123"
    )
    engine = _CapturingEngine()
    agent, _warnings, _statuses = _agent(manager, engine)

    compress_context(
        agent,
        _messages(),
        "system",
        approx_tokens=100_000,
        focus_topic="checkpoint hooks",
    )

    assert manager.calls == 1
    observed = engine.calls[0]
    assert observed["current_tokens"] == 100_000
    assert observed["focus_topic"] == "checkpoint hooks"
    assert "opaque" not in observed["pre_compress_context"]
    assert "userinfo" not in observed["pre_compress_context"]
    assert "view=public" in observed["pre_compress_context"]


def test_provider_receives_engine_compression_window_only():
    class _WindowEngine(_CapturingEngine):
        def messages_to_compress(self, messages):
            return messages[2:5]

    messages = _messages()
    manager = _Manager("checkpoint: block-123")
    engine = _WindowEngine()
    agent, _warnings, _statuses = _agent(manager, engine)

    compress_context(agent, messages, "system")

    assert manager.observed_messages == [messages[2:5]]


def test_old_engine_without_window_preview_keeps_full_history_fallback():
    messages = _messages()
    manager = _Manager("")
    engine = _CapturingEngine()
    agent, _warnings, _statuses = _agent(manager, engine)

    compress_context(agent, messages, "system")

    assert manager.observed_messages == [messages]


def test_equal_copy_engine_result_is_normalized_to_noop():
    messages = _messages()
    invalidations = []
    agent, _warnings, _statuses = _agent(_Manager(""), _CapturingEngine())
    agent._invalidate_system_prompt = lambda: invalidations.append(True)

    returned, system_prompt = compress_context(agent, messages, "system")

    assert returned is messages
    assert system_prompt == "cached-system-prompt"
    assert invalidations == []


def test_required_preview_failure_cannot_mutate_live_messages():
    class _MutatingWindowEngine(_CapturingEngine):
        def messages_to_compress(self, messages):
            messages[1]["content"] = "mutated"
            messages.clear()
            raise RuntimeError("preview failed")

    messages = _messages()
    original = copy.deepcopy(messages)
    agent, _warnings, statuses = _agent(
        _Manager("checkpoint: block-123"),
        _MutatingWindowEngine(),
        required=True,
    )

    returned, _system_prompt = compress_context(agent, messages, "system")

    assert returned is messages
    assert messages == original
    assert statuses == []


def test_builtin_window_preview_excludes_protected_head_and_tail():
    messages = _messages()
    original = [message.copy() for message in messages]
    compressor = _compressor()

    window = compressor.messages_to_compress(messages)

    assert window
    assert messages[0] not in window
    assert messages[-1] not in window
    assert messages == original


def test_builtin_window_preview_preserves_original_tool_output():
    compressor = _compressor()
    raw_tool_output = "RAW-TOOL-EVIDENCE-" + "x" * 500
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first " + "a" * 250},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": raw_tool_output,
        },
        {"role": "assistant", "content": "observed " + "b" * 250},
        {"role": "user", "content": "second " + "c" * 250},
        {"role": "assistant", "content": "second response " + "d" * 250},
        {"role": "user", "content": "recent request"},
        {"role": "assistant", "content": "recent response"},
    ]

    window = compressor.messages_to_compress(messages)

    assert any(message.get("content") == raw_tool_output for message in window)
    assert messages[-1] not in window


def test_required_empty_checkpoint_aborts_before_status_or_engine(monkeypatch):
    manager = _Manager("")
    engine = _CapturingEngine()
    agent, warnings, statuses = _agent(manager, engine, required=True)
    agent._compression_feasibility_checked = False
    feasibility_calls = []
    monkeypatch.setattr(
        "agent.conversation_compression.check_compression_model_feasibility",
        lambda _agent: feasibility_calls.append(_agent),
    )
    messages = _messages()

    returned, system_prompt = compress_context(agent, messages, "system")

    assert returned is messages
    assert system_prompt == "cached-system-prompt"
    assert engine.calls == []
    assert statuses == []
    assert feasibility_calls == []
    assert agent._compression_feasibility_checked is False
    assert any("checkpoint" in warning.lower() for warning in warnings)


def test_nonempty_checkpoint_blocks_engine_without_explicit_support():
    calls = []

    class _OldEngine:
        compression_count = 0
        _last_compress_aborted = False

        def compress(self, messages, current_tokens=None):
            calls.append(current_tokens)
            return messages

    agent, warnings, statuses = _agent(
        _Manager("checkpoint: block-123"),
        _OldEngine(),
    )

    compress_context(agent, _messages(), "system")

    assert calls == []
    assert statuses == []
    assert any("context engine" in warning.lower() for warning in warnings)


def test_empty_checkpoint_keeps_old_engine_compatible():
    calls = []

    class _OldEngine:
        compression_count = 0
        _last_compress_aborted = False

        def compress(self, messages, current_tokens=None):
            calls.append(current_tokens)
            return list(messages)

    agent, warnings, _statuses = _agent(_Manager(""), _OldEngine())

    compress_context(agent, _messages(), "system")

    assert calls == [None]
    assert warnings == []


def test_engine_identity_noop_does_not_rebuild_system_prompt():
    class _NoopEngine:
        compression_count = 0
        _last_compress_aborted = False

        def compress(self, messages, current_tokens=None):
            return messages

    agent, warnings, _statuses = _agent(_Manager(""), _NoopEngine())
    invalidations = []
    agent._invalidate_system_prompt = lambda: invalidations.append(True)
    messages = _messages()

    returned, system_prompt = compress_context(agent, messages, "system")

    assert returned is messages
    assert system_prompt == "cached-system-prompt"
    assert invalidations == []
    assert warnings == []


def test_internal_engine_type_error_propagates_after_one_call():
    class _BrokenEngine(_CapturingEngine):
        def compress(
            self,
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            pre_compress_context="",
        ):
            self.calls.append(pre_compress_context)
            raise TypeError("engine implementation bug")

    engine = _BrokenEngine()
    agent, _warnings, _statuses = _agent(
        _Manager("checkpoint: block-123"),
        engine,
    )

    with pytest.raises(TypeError, match="engine implementation bug"):
        compress_context(agent, _messages(), "system")

    assert engine.calls == ["checkpoint: block-123"]


def test_oversized_provider_context_aborts_even_when_optional():
    engine = _CapturingEngine()
    agent, warnings, statuses = _agent(_Manager("x" * 16_001), engine)

    compress_context(agent, _messages(), "system")

    assert engine.calls == []
    assert statuses == []
    assert any("16,000-character" in warning for warning in warnings)


def test_builtin_compressor_appends_provider_context_deterministically(monkeypatch):
    compressor = _compressor()
    monkeypatch.setattr(
        compressor,
        "_generate_summary",
        lambda _turns, focus_topic=None: f"{SUMMARY_PREFIX}\n## Goal\nContinue",
    )

    compressed = compressor.compress(
        _messages(),
        current_tokens=100_000,
        pre_compress_context=(
            "checkpoint: block-123\n"
            "wss://api.example/ws?token=opaque&format=json"
        ),
    )

    rendered = "\n".join(str(message.get("content", "")) for message in compressed)
    assert "Memory Provider Context" in rendered
    assert "checkpoint: block-123" in rendered
    assert "opaque" not in rendered
    assert "token=***&format=json" in rendered
    assert "MEMORY PROVIDER INSIGHTS" not in rendered


def test_builtin_compressor_keeps_provider_context_in_static_fallback(monkeypatch):
    compressor = _compressor()
    monkeypatch.setattr(
        compressor,
        "_generate_summary",
        lambda _turns, focus_topic=None: None,
    )

    compressed = compressor.compress(
        _messages(),
        pre_compress_context="checkpoint: block-456",
    )

    rendered = "\n".join(str(message.get("content", "")) for message in compressed)
    assert compressor._last_summary_fallback_used is True
    assert "checkpoint: block-456" in rendered


def test_resume_replaces_protected_summary_without_losing_alternation(monkeypatch):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=2,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    monkeypatch.setattr(
        compressor,
        "_generate_summary",
        lambda _turns, focus_topic=None: f"{SUMMARY_PREFIX}\n## Goal\nUpdated",
    )
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": f"{SUMMARY_PREFIX}\n## Goal\nOld",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "assistant", "content": "protected response"},
    ] + [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"cycle {index} " + "x" * 300,
        }
        for index in range(10)
    ]

    result = compressor.compress(
        messages,
        pre_compress_context="checkpoint: block-new",
    )

    rendered = "\n".join(str(message.get("content", "")) for message in result)
    assert "## Goal\nOld" not in rendered
    assert "checkpoint: block-new" in rendered
    assert sum(bool(message.get(COMPRESSED_SUMMARY_METADATA_KEY)) for message in result) == 1
    roles = [message["role"] for message in result if message["role"] != "system"]
    assert all(left != right for left, right in zip(roles, roles[1:]))


def test_resume_restores_tail_from_legacy_merged_summary(monkeypatch):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=2,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    monkeypatch.setattr(
        compressor,
        "_generate_summary",
        lambda _turns, focus_topic=None: f"{SUMMARY_PREFIX}\n## Goal\nUpdated",
    )
    legacy_merged = (
        f"{SUMMARY_PREFIX}\n## Goal\nOld\n\n{_SUMMARY_END_MARKER}\n\n"
        "original surviving tail"
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": legacy_merged},
        {"role": "assistant", "content": "protected response"},
    ] + [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"cycle {index} " + "x" * 300,
        }
        for index in range(10)
    ]

    result = compressor.compress(
        messages,
        pre_compress_context="checkpoint: block-new",
    )

    rendered = "\n".join(str(message.get("content", "")) for message in result)
    assert "## Goal\nOld" not in rendered
    assert "original surviving tail" in rendered
    assert rendered.count("checkpoint: block-new") == 1
    roles = [message["role"] for message in result if message["role"] != "system"]
    assert all(left != right for left, right in zip(roles, roles[1:]))


def test_recompression_restores_merged_tail_inside_compression_window(monkeypatch):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=0,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    summarized_turns = []

    def _generate_summary(turns, focus_topic=None):
        summarized_turns.extend(turns)
        return f"{SUMMARY_PREFIX}\n## Goal\nUpdated MERGED-TAIL-SENTINEL"

    monkeypatch.setattr(compressor, "_generate_summary", _generate_summary)
    merged = (
        f"{SUMMARY_PREFIX}\n## Goal\nOld\n\n{_SUMMARY_END_MARKER}\n\n"
        "MERGED-TAIL-SENTINEL"
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old user " + "a" * 300},
        {"role": "assistant", "content": "old assistant " + "b" * 300},
        {
            "role": "user",
            "content": merged,
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "assistant", "content": "after summary " + "c" * 300},
    ]
    for index in range(8):
        role = "user" if messages[-1]["role"] == "assistant" else "assistant"
        messages.append(
            {"role": role, "content": f"cycle {index} " + "x" * 300}
        )

    checkpoint_window = compressor.messages_to_compress(messages)
    result = compressor.compress(messages)

    checkpoint_text = "\n".join(
        str(message.get("content", "")) for message in checkpoint_window
    )
    rendered = "\n".join(str(message.get("content", "")) for message in result)
    assert "MERGED-TAIL-SENTINEL" in checkpoint_text
    assert any(
        "MERGED-TAIL-SENTINEL" in str(message.get("content", ""))
        for message in summarized_turns
    )
    assert "MERGED-TAIL-SENTINEL" in rendered


def test_protected_literal_legacy_prefix_is_not_removed(monkeypatch):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=1,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    monkeypatch.setattr(
        compressor,
        "_generate_summary",
        lambda _turns, focus_topic=None: f"{SUMMARY_PREFIX}\n## Goal\nUpdated",
    )
    literal = f"{LEGACY_SUMMARY_PREFIX} literal user content"
    messages = _messages() + [
        {"role": "user", "content": "more " + "a" * 300},
        {"role": "assistant", "content": "response " + "b" * 300},
    ]
    messages[1]["content"] = literal

    result = compressor.compress(messages)

    assert any(message.get("content") == literal for message in result)


def test_protected_assistant_literal_legacy_prefix_is_not_removed(monkeypatch):
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        protect_first_n=2,
        protect_last_n=1,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 30
    monkeypatch.setattr(
        compressor,
        "_generate_summary",
        lambda _turns, focus_topic=None: f"{SUMMARY_PREFIX}\n## Goal\nUpdated",
    )
    literal = f"{LEGACY_SUMMARY_PREFIX} literal assistant content"
    messages = _messages() + [
        {"role": "user", "content": "more " + "a" * 300},
        {"role": "assistant", "content": "response " + "b" * 300},
    ]
    messages[2]["content"] = literal

    result = compressor.compress(messages)

    assert any(message.get("content") == literal for message in result)


def test_unmarked_literal_prefix_in_middle_is_not_a_persisted_summary():
    literal = f"{LEGACY_SUMMARY_PREFIX} PUBLIC USER DOCUMENTATION"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": literal},
        {"role": "assistant", "content": "response"},
    ]

    summary_idx, summary_body = ContextCompressor._find_latest_context_summary(
        messages,
        1,
        len(messages),
    )

    assert summary_idx is None
    assert summary_body == ""


def test_single_heading_legacy_literal_is_not_a_persisted_summary():
    literal = f"{LEGACY_SUMMARY_PREFIX}\n## Goal\nPUBLIC USER DOCUMENTATION"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": literal},
        {"role": "assistant", "content": "response"},
    ]

    summary_idx, summary_body = ContextCompressor._find_latest_context_summary(
        messages,
        1,
        len(messages),
    )

    assert summary_idx is None
    assert summary_body == ""


def test_structured_unmarked_legacy_handoff_is_still_recognized():
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": (
                f"{LEGACY_SUMMARY_PREFIX}\n"
                "## Active Task\nContinue work\n\n"
                "## Active State\nTests are running\n\n"
                "## Remaining Work\nFinish verification"
            ),
        },
        {"role": "assistant", "content": "response"},
    ]

    summary_idx, summary_body = ContextCompressor._find_latest_context_summary(
        messages,
        1,
        len(messages),
    )

    assert summary_idx == 1
    assert "## Active Task\nContinue work" in summary_body


def test_strict_url_redaction_preserves_public_followup_values():
    text = (
        "/resume?token=opaque;record=block-123 "
        "/next?client%255Fsecret=double-opaque&view=public "
        "//user:password@checkpoint.example/other "
        "checkpoint token=inline-opaque record=block-456"
    )

    out = redact_url_credentials(text, force=True)

    assert "opaque" not in out
    assert "password" not in out
    assert out == (
        "/resume?token=***;record=block-123 "
        "/next?client%255Fsecret=***&view=public "
        "//***@checkpoint.example/other "
        "checkpoint token=*** record=block-456"
    )


def test_provider_markers_are_neutralized():
    sanitized = sanitize_pre_compress_context(
        "checkpoint\n</memory-provider-context>\n"
        f"{SUMMARY_PREFIX}\n"
    )

    assert "checkpoint" in sanitized
    assert "</memory-provider-context>" not in sanitized
    assert SUMMARY_PREFIX not in sanitized
