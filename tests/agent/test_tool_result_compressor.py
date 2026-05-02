"""Unit tests for agent.tool_result_compressor."""
import asyncio

import pytest

from agent.tool_result_compressor import (
    CompressionResult,
    DropBodyCompressor,
    ToolResultCompressor,
    count_tokens,
    make_tool_result_compressor,
    summarize_tool_result,
)


class TestCompressionResult:
    def test_required_fields(self):
        r = CompressionResult(
            compressed_text="abc",
            original_tokens=100,
            compressed_tokens=33,
            latency_ms=12.5,
            cache_hit=False,
            fell_back=False,
        )
        assert r.compressed_text == "abc"
        assert r.original_tokens == 100
        assert r.compressed_tokens == 33
        assert r.latency_ms == 12.5
        assert r.cache_hit is False
        assert r.fell_back is False

    def test_compression_ratio_property(self):
        r = CompressionResult(
            compressed_text="abc",
            original_tokens=100,
            compressed_tokens=25,
            latency_ms=0.0,
            cache_hit=False,
            fell_back=False,
        )
        assert r.compression_ratio == pytest.approx(4.0)

    def test_compression_ratio_zero_compressed_returns_inf(self):
        r = CompressionResult(
            compressed_text="",
            original_tokens=100,
            compressed_tokens=0,
            latency_ms=0.0,
            cache_hit=False,
            fell_back=False,
        )
        assert r.compression_ratio == float("inf")


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_basic_proportional(self):
        # Wraps estimate_tokens_rough (ceil(chars/4)).
        assert count_tokens("hello world") == 3   # (11+3)//4
        assert count_tokens("a" * 100) == 25      # (100+3)//4

    def test_no_crash_on_special_sequences(self):
        # Char-based estimator never raises on arbitrary content.
        count_tokens("here is <|endoftext|> mid-text")


class _FakeCompressor(ToolResultCompressor):
    """Identity compressor for testing — returns input unchanged."""
    def __init__(self):
        self.calls = []

    def compress(self, tool_name, tool_args, content, question=None):
        self.calls.append((tool_name, tool_args, content, question))
        return CompressionResult(
            compressed_text=content,
            original_tokens=len(content),
            compressed_tokens=len(content),
            latency_ms=0.0,
            cache_hit=False,
            fell_back=False,
        )


class TestToolResultCompressorABC:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            ToolResultCompressor()  # type: ignore[abstract]

    def test_subclass_must_implement_compress(self):
        class Bad(ToolResultCompressor):
            pass
        with pytest.raises(TypeError):
            Bad()  # type: ignore[abstract]

    def test_compress_many_default_runs_compress_per_item(self):
        c = _FakeCompressor()
        items = [
            ("web_extract", '{"urls":["a"]}', "alpha"),
            ("web_search", '{"query":"q"}', "beta"),
        ]

        results = asyncio.run(c.compress_many(items, question="why?"))

        assert len(results) == 2
        assert results[0].compressed_text == "alpha"
        assert results[1].compressed_text == "beta"
        assert c.calls == [
            ("web_extract", '{"urls":["a"]}', "alpha", "why?"),
            ("web_search", '{"query":"q"}', "beta", "why?"),
        ]

    def test_compress_many_passes_none_question(self):
        c = _FakeCompressor()
        asyncio.run(c.compress_many([("web_extract", "{}", "x")]))
        assert c.calls == [("web_extract", "{}", "x", None)]


class TestSummarizeToolResult:
    """Characterization tests — lock current summary shapes so refactors
    don't silently change what older tool results look like in pruned
    conversations."""

    def test_terminal_extracts_command_and_exit_code(self):
        out = summarize_tool_result(
            "terminal",
            '{"command": "npm test"}',
            'some output\nmore lines\n{"exit_code": 0}',
        )
        assert out == "[terminal] ran `npm test` -> exit 0, 3 lines output"

    def test_web_extract_uses_first_url(self):
        out = summarize_tool_result(
            "web_extract",
            '{"urls": ["https://example.com/a", "https://example.com/b"]}',
            "x" * 1234,
        )
        assert "https://example.com/a (+1 more)" in out
        assert "1,234 chars" in out

    def test_malformed_args_falls_back_gracefully(self):
        # JSONDecodeError must not raise — args become {}.
        out = summarize_tool_result("terminal", "{not valid json", "")
        assert out.startswith("[terminal]")

    def test_unknown_tool_uses_generic_fallback(self):
        out = summarize_tool_result(
            "made_up_tool",
            '{"foo": "bar", "baz": 123}',
            "result body",
        )
        assert out.startswith("[made_up_tool]")
        assert "foo=bar" in out
        assert "11 chars result" in out

    def test_empty_args_handled(self):
        out = summarize_tool_result("todo", "", "")
        assert out == "[todo] updated task list"


class TestDropBodyCompressor:
    def test_compress_returns_summarized_text(self):
        c = DropBodyCompressor()
        args = '{"urls":["https://example.com"]}'
        content = "x" * 5000
        result = c.compress("web_extract", args, content)
        assert result.compressed_text == summarize_tool_result(
            "web_extract", args, content
        )

    def test_compress_records_token_counts(self):
        c = DropBodyCompressor()
        content = "lorem ipsum dolor sit amet " * 50
        result = c.compress("web_extract", '{"urls":["x"]}', content)
        assert result.original_tokens == count_tokens(content)
        assert result.compressed_tokens == count_tokens(result.compressed_text)
        assert result.compressed_tokens < result.original_tokens

    def test_compress_sets_metadata_fields(self):
        c = DropBodyCompressor()
        result = c.compress("web_extract", '{"urls":["x"]}', "x" * 1000)
        assert result.cache_hit is False
        assert result.fell_back is False
        assert result.latency_ms >= 0.0

    def test_question_parameter_is_ignored_but_accepted(self):
        c = DropBodyCompressor()
        r1 = c.compress("web_extract", "{}", "abc", question=None)
        r2 = c.compress("web_extract", "{}", "abc", question="why?")
        assert r1.compressed_text == r2.compressed_text


class TestFactory:
    def test_drop_method_returns_drop_body_compressor(self):
        c = make_tool_result_compressor({"method": "drop"})
        assert isinstance(c, DropBodyCompressor)

    def test_default_when_no_method_specified_is_drop(self):
        c = make_tool_result_compressor({})
        assert isinstance(c, DropBodyCompressor)

    def test_default_when_no_config_is_drop(self):
        c = make_tool_result_compressor(None)
        assert isinstance(c, DropBodyCompressor)

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown.*compression.*method"):
            make_tool_result_compressor({"method": "magic_pixie_dust"})

    def test_llmlingua_methods_raise_not_implemented_in_phase_1(self):
        with pytest.raises(NotImplementedError, match="PR 2|llmlingua2_local"):
            make_tool_result_compressor({"method": "llmlingua2_local"})
        with pytest.raises(NotImplementedError, match="PR 2|llmlingua2_remote"):
            make_tool_result_compressor({"method": "llmlingua2_remote"})
