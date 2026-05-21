"""Unit tests for agent.tool_result_compressor."""
import asyncio
import socket
import time
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_result_compressor import (
    CompressionResult,
    DropBodyCompressor,
    LLMLinguaConfig,
    LLMLinguaLocalCompressor,
    LLMLinguaRemoteCompressor,
    ToolResultCompressor,
    _ContentLRU,
    _REMOTE_BACKOFF_SECS,
    _validate_compression,
    compress_with_wrapper,
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

    def test_compress_batch_sync_runs_compress_per_item(self):
        c = _FakeCompressor()
        items = [
            ("web_extract", "{}", "alpha"),
            ("web_search", "{}", "beta"),
            ("web_extract", "{}", "gamma"),
        ]
        results = c.compress_batch(items, question="why?")
        assert [r.compressed_text for r in results] == ["alpha", "beta", "gamma"]
        assert c.calls == [
            ("web_extract", "{}", "alpha", "why?"),
            ("web_search", "{}", "beta", "why?"),
            ("web_extract", "{}", "gamma", "why?"),
        ]

    def test_compress_batch_empty_items(self):
        c = _FakeCompressor()
        assert c.compress_batch([], question="x") == []
        assert c.calls == []


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

    def test_default_method_is_drop_even_when_llmlingua_installed(self):
        # No method specified must preserve historical Hermes behaviour.
        # Installing an optional extra must not silently alter production
        # context-pruning semantics.
        with patch(
            "agent.tool_result_compressor._llmlingua_extra_installed",
            return_value=True,
        ):
            c = make_tool_result_compressor({})
        assert isinstance(c, DropBodyCompressor)

    def test_default_when_no_config_is_drop_even_when_llmlingua_installed(self):
        with patch(
            "agent.tool_result_compressor._llmlingua_extra_installed",
            return_value=True,
        ):
            c = make_tool_result_compressor(None)
        assert isinstance(c, DropBodyCompressor)

    def test_auto_resolves_to_local_when_llmlingua_installed(self):
        with patch(
            "agent.tool_result_compressor._llmlingua_extra_installed",
            return_value=True,
        ):
            c = make_tool_result_compressor({"method": "auto"})
        assert isinstance(c, LLMLinguaLocalCompressor)

    def test_auto_resolves_to_drop_when_llmlingua_missing(self):
        # The most common case for upstream users: extra not installed,
        # auto must resolve to the byte-identical drop-body behaviour.
        with patch(
            "agent.tool_result_compressor._llmlingua_extra_installed",
            return_value=False,
        ):
            c = make_tool_result_compressor({"method": "auto"})
        assert isinstance(c, DropBodyCompressor)

    def test_explicit_drop_works_regardless_of_llmlingua_availability(self):
        # Users who explicitly disable compression keep that behaviour
        # even when the extra is installed.
        with patch(
            "agent.tool_result_compressor._llmlingua_extra_installed",
            return_value=True,
        ):
            c = make_tool_result_compressor({"method": "drop"})
        assert isinstance(c, DropBodyCompressor)

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown.*compression.*method"):
            make_tool_result_compressor({"method": "magic_pixie_dust"})

    def test_llmlingua_local_returns_local_compressor(self):
        c = make_tool_result_compressor({"method": "llmlingua2_local"})
        assert isinstance(c, LLMLinguaLocalCompressor)

    def test_llmlingua_remote_returns_remote_compressor(self):
        c = make_tool_result_compressor({
            "method": "llmlingua2_remote",
            "endpoint": "http://example.com/compress",
        })
        assert isinstance(c, LLMLinguaRemoteCompressor)

    def test_llmlingua_remote_requires_endpoint(self):
        with pytest.raises(ValueError, match="endpoint"):
            make_tool_result_compressor({"method": "llmlingua2_remote"})

    def test_llmlingua_local_passes_user_config_through(self):
        c = make_tool_result_compressor({
            "method": "llmlingua2_local",
            "min_output_chars": 5000,
            "rate": 0.5,
            "use_question": False,
            "only_tools": ["web_extract"],
            "rate_ladder": [
                {"max_chars": 8000, "rate": 0.6},
                {"max_chars": None, "rate": 0.2},
            ],
        })
        assert c.config.min_output_chars == 5000
        assert c.config.rate == 0.5
        assert c.config.use_question is False
        assert c.config.only_tools == ("web_extract",)
        assert c.config.rate_ladder == ((8000, 0.6), (None, 0.2))


class TestContentLRU:
    def test_insert_and_get(self):
        lru = _ContentLRU(max_size=4)
        r = CompressionResult("x", 10, 5, 1.0, False, False)
        key = _ContentLRU.key_for("hello")
        lru.put(key, r)
        assert lru.get(key) is r

    def test_eviction_least_recently_used(self):
        lru = _ContentLRU(max_size=2)
        for i in range(3):
            lru.put(_ContentLRU.key_for(f"c{i}"),
                    CompressionResult(f"x{i}", 1, 1, 0.0, False, False))
        # c0 should have been evicted; c1 + c2 remain
        assert lru.get(_ContentLRU.key_for("c0")) is None
        assert lru.get(_ContentLRU.key_for("c1")).compressed_text == "x1"
        assert lru.get(_ContentLRU.key_for("c2")).compressed_text == "x2"

    def test_get_updates_recency(self):
        lru = _ContentLRU(max_size=2)
        lru.put(_ContentLRU.key_for("a"), CompressionResult("A", 1, 1, 0.0, False, False))
        lru.put(_ContentLRU.key_for("b"), CompressionResult("B", 1, 1, 0.0, False, False))
        # Access "a" to make it most recent
        lru.get(_ContentLRU.key_for("a"))
        # Add "c" — "b" should be evicted (least recently used), not "a"
        lru.put(_ContentLRU.key_for("c"), CompressionResult("C", 1, 1, 0.0, False, False))
        assert lru.get(_ContentLRU.key_for("a")).compressed_text == "A"
        assert lru.get(_ContentLRU.key_for("b")) is None
        assert lru.get(_ContentLRU.key_for("c")).compressed_text == "C"

    def test_key_is_deterministic(self):
        assert _ContentLRU.key_for("same") == _ContentLRU.key_for("same")
        assert _ContentLRU.key_for("a") != _ContentLRU.key_for("b")

    def test_key_distinguishes_question(self):
        # Same content, different question → different cache key. Otherwise
        # a session reset (or any reuse of the same page under a new user
        # question) would return the stale prior compression.
        a = _ContentLRU.key_for("body", question="why?")
        b = _ContentLRU.key_for("body", question="how?")
        c = _ContentLRU.key_for("body", question=None)
        assert a != b
        assert a != c
        assert b != c

    def test_key_question_none_and_empty_match(self):
        # None and "" both mean "no question conditioning" — should hash the same.
        assert (_ContentLRU.key_for("body", question=None)
                == _ContentLRU.key_for("body", question=""))

    def test_key_distinguishes_tool_name_and_rate(self):
        a = _ContentLRU.key_for("body", tool_name="web_extract", rate=0.33)
        b = _ContentLRU.key_for("body", tool_name="web_search",  rate=0.33)
        c = _ContentLRU.key_for("body", tool_name="web_extract", rate=0.50)
        assert a != b and a != c

    def test_concurrent_get_put_does_not_crash(self):
        # Regression: the LRU is shared across threads in
        # LLMLinguaRemoteCompressor.compress_batch. Without locking, a
        # check-then-move race can raise KeyError outside the compressor's
        # try block, violating the "must not crash pruning" contract.
        import threading
        lru = _ContentLRU(max_size=8)
        errors: list = []
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                lru.put(_ContentLRU.key_for(f"k{i % 16}"),
                        CompressionResult(f"v{i}", 1, 1, 0.0, False, False))
                i += 1

        def reader():
            i = 0
            while not stop.is_set():
                try:
                    lru.get(_ContentLRU.key_for(f"k{i % 16}"))
                except Exception as e:
                    errors.append(e)
                i += 1

        threads = [threading.Thread(target=writer) for _ in range(2)] + \
                  [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.2)
        stop.set()
        for t in threads:
            t.join()
        assert errors == [], f"races raised: {errors[:3]}"


class TestValidateCompression:
    def test_valid_passes(self):
        _validate_compression("original" * 100, "compressed" * 10)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_compression("original", "")

    def test_longer_than_input_raises(self):
        with pytest.raises(ValueError, match="more content"):
            _validate_compression("short", "much longer than input")

    def test_5_percent_overhead_allowed_for_appendix(self):
        # The [REFS:] appendix can push compressed slightly past input length.
        original = "a" * 100
        compressed = "a" * 104
        _validate_compression(original, compressed)  # no raise

    def test_endoftext_sentinel_raises(self):
        with pytest.raises(ValueError, match="control sentinel"):
            _validate_compression("a" * 100, "shorter <|endoftext|>")

    def test_endoftext_in_original_is_fine(self):
        # If the input legitimately contained the sentinel, output may too.
        _validate_compression("a <|endoftext|> b", "a <|endoftext|>")


class TestCompressWithWrapper:
    def _fake_pc(self, return_text: str = None) -> MagicMock:
        pc = MagicMock()
        if return_text is not None:
            pc.compress_prompt.return_value = {"compressed_prompt": return_text}
        return pc

    def test_calls_compress_prompt_with_force_tokens_and_digits(self):
        pc = self._fake_pc("compressed body")
        compress_with_wrapper(pc, "input body with no urls", rate=0.33, question=None)
        kwargs = pc.compress_prompt.call_args.kwargs
        assert kwargs["rate"] == 0.33
        assert kwargs["force_reserve_digit"] is True
        assert "\n" in kwargs["force_tokens"]
        assert "|" in kwargs["force_tokens"]
        assert "#" in kwargs["force_tokens"]
        assert "question" not in kwargs   # omitted when None

    def test_question_passed_when_truthy(self):
        pc = self._fake_pc("out")
        compress_with_wrapper(pc, "input", rate=0.5, question="why?")
        assert pc.compress_prompt.call_args.kwargs["question"] == "why?"

    def test_question_omitted_when_empty_string(self):
        pc = self._fake_pc("out")
        compress_with_wrapper(pc, "input", rate=0.5, question="")
        assert "question" not in pc.compress_prompt.call_args.kwargs

    def test_url_placeholder_round_trip(self):
        # Compressor "preserves" the placeholder; wrapper restores it.
        pc = self._fake_pc("see URLREF000 for details")
        out = compress_with_wrapper(
            pc, "Background see https://example.com/a for details",
            rate=0.33, question=None,
        )
        assert "https://example.com/a" in out
        assert "URLREF" not in out

    def test_dropped_url_appended_in_refs_block(self):
        # Compressor drops the placeholder entirely; wrapper appends [REFS: ...]
        pc = self._fake_pc("compressed text without placeholder")
        out = compress_with_wrapper(
            pc, "see https://example.com/lost",
            rate=0.33, question=None,
        )
        assert "https://example.com/lost" in out
        assert "[REFS:" in out

    def test_input_without_urls_skips_appendix(self):
        pc = self._fake_pc("compressed body")
        out = compress_with_wrapper(pc, "no urls here at all", rate=0.33, question=None)
        assert "[REFS:" not in out
        assert out == "compressed body"

    def test_multiple_urls_each_get_unique_placeholder(self):
        pc = self._fake_pc("URLREF000 URLREF001 URLREF002 something")
        out = compress_with_wrapper(
            pc,
            "https://a.com https://b.com https://c.com something",
            rate=0.33, question=None,
        )
        assert "https://a.com" in out
        assert "https://b.com" in out
        assert "https://c.com" in out

    def test_url_with_balanced_parens_preserved(self):
        # Regression: the prior URL regex stopped at any ')' so
        # Wikipedia-style URLs lost their closing paren.
        pc = self._fake_pc("see URLREF000 for context")
        out = compress_with_wrapper(
            pc,
            "Background: see https://en.wikipedia.org/wiki/Foo_(bar) for context",
            rate=0.33, question=None,
        )
        assert "https://en.wikipedia.org/wiki/Foo_(bar)" in out

    def test_markdown_link_does_not_eat_trailing_paren(self):
        # The opposite case: ``(https://x.com)`` in Markdown link syntax
        # should NOT include the trailing close-paren in the URL.
        pc = self._fake_pc("see URLREF000 directly")
        out = compress_with_wrapper(
            pc, "see (https://example.com) directly",
            rate=0.33, question=None,
        )
        assert "https://example.com" in out
        # Bare URL without trailing paren survived
        assert "https://example.com)" not in out

    def test_markdown_autolink_does_not_eat_trailing_angle_bracket(self):
        # Regression: ``<https://x.com>`` autolink syntax was extracting
        # ``https://x.com>`` because the ``>`` wasn't tracked as a
        # bracket. The corrupted URL flowed through placeholder/restore.
        pc = self._fake_pc("see URLREF000 in autolink")
        out = compress_with_wrapper(
            pc, "see <https://example.com> in autolink",
            rate=0.33, question=None,
        )
        assert "https://example.com" in out
        assert "https://example.com>" not in out

    def test_url_extraction_robust_to_collisions(self):
        # Regression: deterministic URLREF000 placeholder collided with
        # literal "URLREF000" in the source. With the nonce-prefix fix,
        # content containing the literal token doesn't get rewritten as
        # a URL during the restore step.
        pc = self._fake_pc("the URLREF000 placeholder format and example.com link")
        out = compress_with_wrapper(
            pc,
            "the URLREF000 placeholder format and https://example.com link",
            rate=0.33, question=None,
        )
        # Literal "URLREF000" stays verbatim — was NOT rewritten as URL
        assert "URLREF000" in out
        # The actual URL is still in the output (inline or [REFS:])
        assert "https://example.com" in out


class TestLLMLinguaLocalCompressor:
    def _make(self, **overrides):
        cfg = LLMLinguaConfig(**overrides)
        return LLMLinguaLocalCompressor(cfg)

    def test_module_imports_without_llmlingua_installed(self):
        # Sanity: the module already imported in this test file. If [llmlingua]
        # extras were required, this whole file would have failed to collect.
        # The actual lazy-load behaviour is covered in
        # test_import_failure_falls_back_to_drop_body below.
        c = self._make()
        assert c._pc is None  # not loaded yet

    def test_non_allowlisted_tool_falls_through_to_drop_body(self):
        c = self._make()
        big = "X" * 5000
        result = c.compress("terminal", '{"command":"ls"}', big)
        # Output should equal what DropBody would produce
        expected = summarize_tool_result("terminal", '{"command":"ls"}', big)
        assert result.compressed_text == expected
        # Lazy: PromptCompressor was never instantiated
        assert c._pc is None

    def test_undersize_content_falls_through_to_drop_body(self):
        c = self._make(min_output_chars=2000)
        result = c.compress("web_extract", '{"urls":["x"]}', "x" * 500)
        expected = summarize_tool_result("web_extract", '{"urls":["x"]}', "x" * 500)
        assert result.compressed_text == expected
        assert c._pc is None  # never loaded

    def test_compresses_when_allowlisted_and_large(self):
        c = self._make(min_output_chars=100)
        body = "alpha beta gamma " * 100
        # Inject a fake PromptCompressor that returns something valid
        c._pc = MagicMock()
        c._pc.compress_prompt.return_value = {"compressed_prompt": "alpha gamma"}
        result = c.compress("web_extract", '{"urls":["x"]}', body)
        assert result.compressed_text == "alpha gamma"
        assert result.fell_back is False
        assert result.cache_hit is False
        assert result.original_tokens > result.compressed_tokens

    def test_cache_hit_returns_cached_result_with_flag_set(self):
        c = self._make(min_output_chars=100)
        c._pc = MagicMock()
        c._pc.compress_prompt.return_value = {"compressed_prompt": "compressed"}
        body = "y" * 500
        r1 = c.compress("web_extract", "{}", body)
        r2 = c.compress("web_extract", "{}", body)
        assert r1.cache_hit is False
        assert r2.cache_hit is True
        assert r2.compressed_text == r1.compressed_text
        # PromptCompressor only called once despite two compress() calls
        assert c._pc.compress_prompt.call_count == 1

    def test_cache_distinguishes_questions(self):
        # Regression: caching only on content meant a second session asking
        # a different question would receive the prior session's compressed
        # output. The cache key must include the question.
        c = self._make(min_output_chars=100, use_question=True)
        c._pc = MagicMock()
        # Different return values per call, so we can detect cache hits by
        # which value comes back.
        outputs = ["compressed-for-alpha", "compressed-for-beta"]
        c._pc.compress_prompt.side_effect = [
            {"compressed_prompt": outputs[0]},
            {"compressed_prompt": outputs[1]},
        ]
        body = "z" * 500
        r1 = c.compress("web_extract", "{}", body, question="alpha")
        r2 = c.compress("web_extract", "{}", body, question="beta")
        assert r1.compressed_text == "compressed-for-alpha"
        assert r2.compressed_text == "compressed-for-beta"
        assert r1.cache_hit is False
        assert r2.cache_hit is False
        assert c._pc.compress_prompt.call_count == 2

    def test_cache_distinguishes_rate(self):
        # Same content + question, different rate (because content size
        # bucket differs in the rate_ladder) → distinct cache entries.
        c = self._make(min_output_chars=100, rate=None)
        c._pc = MagicMock()
        c._pc.compress_prompt.return_value = {"compressed_prompt": "out"}
        # Build two contents of same hash-distinguishable bytes but different
        # sizes so the rate_ladder picks different rates.
        small = "y" * 5_000   # rate=0.5
        large = "y" * 20_000  # rate=0.33
        c.compress("web_extract", "{}", small)
        c.compress("web_extract", "{}", large)
        # Both went through compress_prompt — neither is a cache hit of the other
        assert c._pc.compress_prompt.call_count == 2

    def test_invalid_compression_output_triggers_fallback(self):
        c = self._make(min_output_chars=100)
        c._pc = MagicMock()
        c._pc.compress_prompt.return_value = {"compressed_prompt": ""}  # empty → fail
        body = "x" * 500
        result = c.compress("web_extract", "{}", body)
        assert result.fell_back is True
        # Got the drop-body summary instead
        assert result.compressed_text == summarize_tool_result("web_extract", "{}", body)

    def test_compress_prompt_exception_triggers_fallback(self):
        c = self._make(min_output_chars=100)
        c._pc = MagicMock()
        c._pc.compress_prompt.side_effect = RuntimeError("boom")
        body = "x" * 500
        result = c.compress("web_extract", "{}", body)
        assert result.fell_back is True
        assert "[web_extract]" in result.compressed_text

    def test_import_failure_falls_back_to_drop_body(self):
        c = self._make(min_output_chars=100)
        body = "y" * 500
        # Simulate `import llmlingua` raising ImportError on first call
        with patch.dict("sys.modules", {"llmlingua": None}):
            result = c.compress("web_extract", "{}", body)
        assert result.fell_back is True
        # And subsequent calls don't keep retrying — model_load_failed sticks
        assert c._model_load_failed is True

    def test_sticky_import_failure_falls_back_without_none_compressor_noise(self, caplog):
        c = self._make(min_output_chars=100)
        body = "y" * 500

        with patch.dict("sys.modules", {"llmlingua": None}):
            r1 = c.compress("web_extract", "{}", body)
        assert r1.fell_back is True
        assert c._model_load_failed is True

        caplog.clear()
        r2 = c.compress("web_extract", "{}", body + "second")

        assert r2.fell_back is True
        assert "NoneType" not in caplog.text
        assert "compress_prompt" not in caplog.text

    def test_model_construction_failure_marks_load_failed_sticky(self):
        # Regression: PromptCompressor() construction can fail even after
        # a successful import (HF download, bad model name, runtime
        # mismatch). Earlier versions only set _model_load_failed on
        # ImportError, so subsequent prunes paid the slow-failure path.
        c = self._make(min_output_chars=100)
        body = "z" * 500

        fake_module = MagicMock()
        fake_module.PromptCompressor.side_effect = RuntimeError(
            "could not download model from HuggingFace"
        )
        with patch.dict("sys.modules", {"llmlingua": fake_module}):
            r1 = c.compress("web_extract", "{}", body)
        assert r1.fell_back is True
        assert c._model_load_failed is True

        # A second call must NOT retry the construction — the sticky
        # flag short-circuits _ensure_loaded.
        r2 = c.compress("web_extract", "{}", body + "second")
        assert r2.fell_back is True
        assert fake_module.PromptCompressor.call_count == 1

    def test_rate_ladder_picks_per_size(self):
        c = self._make()  # default ladder
        assert c._rate_for_size(5_000) == 0.50
        assert c._rate_for_size(20_000) == 0.33
        assert c._rate_for_size(50_000) == 0.25
        assert c._rate_for_size(10_000) == 0.50      # boundary inclusive
        assert c._rate_for_size(10_001) == 0.33

    def test_explicit_rate_overrides_ladder(self):
        c = self._make(rate=0.4)
        assert c._rate_for_size(1_000_000) == 0.4
        assert c._rate_for_size(100) == 0.4

    def test_question_threading(self):
        c = self._make(min_output_chars=100, use_question=True)
        c._pc = MagicMock()
        c._pc.compress_prompt.return_value = {"compressed_prompt": "out"}
        body = "x" * 500
        c.compress("web_extract", "{}", body, question="why does this happen?")
        kwargs = c._pc.compress_prompt.call_args.kwargs
        assert kwargs.get("question") == "why does this happen?"

    def test_question_disabled_omits_kwarg(self):
        c = self._make(min_output_chars=100, use_question=False)
        c._pc = MagicMock()
        c._pc.compress_prompt.return_value = {"compressed_prompt": "out"}
        c.compress("web_extract", "{}", "x" * 500, question="should be ignored")
        assert "question" not in c._pc.compress_prompt.call_args.kwargs

    def test_question_none_omits_kwarg(self):
        c = self._make(min_output_chars=100, use_question=True)
        c._pc = MagicMock()
        c._pc.compress_prompt.return_value = {"compressed_prompt": "out"}
        c.compress("web_extract", "{}", "x" * 500, question=None)
        assert "question" not in c._pc.compress_prompt.call_args.kwargs


class TestLLMLinguaRemoteCompressor:
    def _make(self, **overrides):
        overrides.setdefault("endpoint", "http://example.com/compress")
        cfg = LLMLinguaConfig(**overrides)
        return LLMLinguaRemoteCompressor(cfg)

    def test_constructor_requires_endpoint(self):
        with pytest.raises(ValueError, match="endpoint"):
            LLMLinguaRemoteCompressor(LLMLinguaConfig(method="llmlingua2_remote"))

    def test_non_allowlisted_tool_falls_through(self):
        c = self._make()
        # Should never call _post because terminal isn't allowlisted
        with patch.object(c, "_post") as post:
            result = c.compress("terminal", '{"command":"ls"}', "x" * 5000)
        assert post.call_count == 0
        assert result.compressed_text == summarize_tool_result(
            "terminal", '{"command":"ls"}', "x" * 5000
        )

    def test_undersize_content_falls_through(self):
        c = self._make(min_output_chars=2000)
        with patch.object(c, "_post") as post:
            c.compress("web_extract", "{}", "x" * 500)
        assert post.call_count == 0

    def test_successful_post_returns_remote_result(self):
        c = self._make(min_output_chars=100)
        with patch.object(c, "_post", return_value={"compressed": "tiny output"}) as post:
            result = c.compress("web_extract", "{}", "y" * 500)
        assert post.call_count == 1
        assert result.compressed_text == "tiny output"
        assert result.fell_back is False
        assert result.cache_hit is False

    def test_timeout_falls_back_and_arms_cooldown(self):
        c = self._make(min_output_chars=100, timeout_secs=0.1)
        with patch.object(c, "_post", side_effect=socket.timeout("no answer")):
            r1 = c.compress("web_extract", "{}", "y" * 500)
        assert r1.fell_back is True
        assert c._last_remote_failure_time > 0
        assert time.time() < c._last_remote_failure_time + _REMOTE_BACKOFF_SECS
        # Subsequent call within cooldown window doesn't even try the post
        with patch.object(c, "_post") as post:
            r2 = c.compress("web_extract", "{}", "z" * 500)
        assert post.call_count == 0
        assert r2.fell_back is True

    def test_url_error_falls_back(self):
        c = self._make(min_output_chars=100)
        with patch.object(c, "_post", side_effect=urllib.error.URLError("conn refused")):
            r = c.compress("web_extract", "{}", "y" * 500)
        assert r.fell_back is True

    def test_invalid_response_falls_back(self):
        c = self._make(min_output_chars=100)
        with patch.object(c, "_post", return_value={"unexpected_key": 42}):
            r = c.compress("web_extract", "{}", "y" * 500)
        assert r.fell_back is True

    def test_oversized_response_falls_back(self):
        c = self._make(min_output_chars=100)
        # Response longer than input → validation failure
        with patch.object(c, "_post", return_value={"compressed": "z" * 10_000}):
            r = c.compress("web_extract", "{}", "y" * 500)
        assert r.fell_back is True

    @pytest.mark.parametrize("bad_response", [
        [],                                  # JSON array
        None,                                # JSON null
        "ok",                                # JSON string
        42,                                  # JSON number
        True,                                # JSON bool
    ])
    def test_non_dict_json_response_falls_back(self, bad_response):
        # Regression: _post is typed -> dict but JSON allows arrays,
        # null, strings, etc. A non-dict reply from a misbehaving
        # sidecar would raise AttributeError on .get(), bypassing the
        # fallback. compress() must catch this and fall back cleanly.
        c = self._make(min_output_chars=100)
        with patch.object(c, "_post", return_value=bad_response):
            r = c.compress("web_extract", "{}", "y" * 500)
        assert r.fell_back is True
        assert "[web_extract]" in r.compressed_text

    def test_cache_hit_skips_post(self):
        c = self._make(min_output_chars=100)
        with patch.object(c, "_post", return_value={"compressed": "out"}) as post:
            r1 = c.compress("web_extract", "{}", "y" * 500)
            r2 = c.compress("web_extract", "{}", "y" * 500)
        assert post.call_count == 1
        assert r1.cache_hit is False
        assert r2.cache_hit is True

    def test_post_includes_question_when_set(self):
        c = self._make(min_output_chars=100, use_question=True)
        captured = {}
        def fake_post(text, rate, question):
            captured["question"] = question
            return {"compressed": "out"}
        with patch.object(c, "_post", side_effect=fake_post):
            c.compress("web_extract", "{}", "y" * 500, question="why?")
        assert captured["question"] == "why?"

    def test_post_omits_question_when_disabled(self):
        c = self._make(min_output_chars=100, use_question=False)
        captured = {}
        def fake_post(text, rate, question):
            captured["question"] = question
            return {"compressed": "out"}
        with patch.object(c, "_post", side_effect=fake_post):
            c.compress("web_extract", "{}", "y" * 500, question="ignored")
        assert captured["question"] is None


class TestLLMLinguaConfigDefaults:
    def test_defaults_are_sensible(self):
        cfg = LLMLinguaConfig()
        assert cfg.method == "llmlingua2_local"
        assert "bert-base" in cfg.model
        assert cfg.device == "cpu"
        assert "web_extract" in cfg.only_tools
        assert "web_search" in cfg.only_tools
        assert cfg.min_output_chars == 2000
        assert cfg.rate is None  # use ladder by default
        assert cfg.use_question is True
        assert cfg.cache_size == 256
