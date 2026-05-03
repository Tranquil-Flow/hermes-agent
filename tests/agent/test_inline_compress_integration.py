"""Integration tests for ContextCompressor._maybe_inline_compress.

These tests verify the Pass-1 inline compression seam that compresses tool
results immediately after they are produced, before they enter the conversation.
"""
import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.tool_result_compressor import CompressionResult


@pytest.fixture()
def compressor():
    """Create a ContextCompressor with mocked model metadata."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
            tool_compression_config={"method": "drop"},
        )
        return c


class TestMaybeInlineCompress:
    """Tests for ContextCompressor._maybe_inline_compress."""

    def test_returns_compressed_content_when_compress_inline_succeeds(
        self, compressor
    ):
        """When compress_inline returns a CompressionResult, its text is returned."""
        fake_result = CompressionResult(
            compressed_text="[terminal] ran `npm test` -> exit 0, 47 lines output",
            original_tokens=500,
            compressed_tokens=20,
            latency_ms=3.1,
            cache_hit=False,
            fell_back=False,
        )

        with patch.object(
            compressor._tool_compressor,
            "compress_inline",
            return_value=fake_result,
        ) as mock_compress:
            content = compressor._maybe_inline_compress(
                tool_name="terminal",
                tool_args='{"command": "npm test"}',
                content="THIS IS A VERY LONG TOOL OUTPUT WITH LOTS OF LINES\n" * 50,
            )

        assert content == fake_result.compressed_text
        mock_compress.assert_called_once_with(
            tool_name="terminal",
            tool_args='{"command": "npm test"}',
            content="THIS IS A VERY LONG TOOL OUTPUT WITH LOTS OF LINES\n" * 50,
        )

    def test_passes_original_content_through_when_compress_inline_raises(
        self, compressor
    ):
        """When compress_inline raises, the original content is returned (fail-open)."""
        original = "THE UNCHANGED TOOL RESULT CONTENT"

        with patch.object(
            compressor._tool_compressor,
            "compress_inline",
            side_effect=RuntimeError("LLMLingua model unavailable"),
        ):
            content = compressor._maybe_inline_compress(
                tool_name="read_file",
                tool_args='{"path": "config.py"}',
                content=original,
            )

        assert content == original

    def test_passes_original_content_through_when_no_context_compressor(self):
        """When there is no _tool_compressor attribute, original content is returned."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test/model",
                quiet_mode=True,
                tool_compression_config={"method": "drop"},
            )
        # Remove the compressor to simulate a missing attribute
        with patch.object(c, "_tool_compressor", None):
            original = "raw tool result"
            result = c._maybe_inline_compress(
                tool_name="search_files",
                tool_args='{"pattern": "compress"}',
                content=original,
            )
        assert result == original

    def test_serializes_tool_args_and_content_passed_to_compress_inline(
        self, compressor
    ):
        """Tool args and content are forwarded exactly to compress_inline."""
        fake_result = CompressionResult(
            compressed_text="[search_files] content search for 'foo' -> 12 matches",
            original_tokens=300,
            compressed_tokens=15,
            latency_ms=2.0,
            cache_hit=False,
            fell_back=False,
        )

        tool_args = '{"path": "/tmp", "pattern": "hello", "target": "content"}'
        content = "line one\nline two\nline three\n" * 100

        with patch.object(
            compressor._tool_compressor,
            "compress_inline",
            return_value=fake_result,
        ) as mock_inline:
            compressor._maybe_inline_compress(
                tool_name="search_files",
                tool_args=tool_args,
                content=content,
            )

        mock_inline.assert_called_once_with(
            tool_name="search_files",
            tool_args=tool_args,
            content=content,
        )

    def test_returns_compressed_text_from_compression_result(self, compressor):
        """The compressed_text field of CompressionResult is what gets returned."""
        fake_result = CompressionResult(
            compressed_text="[patch] replace in setup.py (1,200 chars result)",
            original_tokens=800,
            compressed_tokens=18,
            latency_ms=4.5,
            cache_hit=True,
            fell_back=False,
        )

        with patch.object(
            compressor._tool_compressor,
            "compress_inline",
            return_value=fake_result,
        ):
            result = compressor._maybe_inline_compress(
                tool_name="patch",
                tool_args='{"path": "setup.py", "mode": "replace"}',
                content="x" * 1200,
            )

        assert result == "[patch] replace in setup.py (1,200 chars result)"

    def test_fails_open_on_non_compressionresult_return(self, compressor):
        """If compress_inline returns something unexpected, fail-open returns original."""
        original = "original content here"

        with patch.object(
            compressor._tool_compressor,
            "compress_inline",
            return_value="not a CompressionResult",
        ):
            result = compressor._maybe_inline_compress(
                tool_name="terminal",
                tool_args='{"command": "ls"}',
                content=original,
            )

        # The method tries to access .compressed_text on the return value,
        # which will raise AttributeError → fail-open → original returned
        assert result == original

    def test_cache_hit_is_passed_through_correctly(self, compressor):
        """Cache hit info in the result is not directly used, but no error occurs."""
        fake_result = CompressionResult(
            compressed_text="[read_file] read config.py from line 1 (3,400 chars)",
            original_tokens=400,
            compressed_tokens=12,
            latency_ms=0.5,
            cache_hit=True,
            fell_back=False,
        )

        with patch.object(
            compressor._tool_compressor,
            "compress_inline",
            return_value=fake_result,
        ):
            result = compressor._maybe_inline_compress(
                tool_name="read_file",
                tool_args='{"path": "config.py"}',
                content="x" * 3400,
            )

        assert result == fake_result.compressed_text
        assert result.startswith("[read_file]")
