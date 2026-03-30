"""Tests for the unified memory agent tool (tools/unified_memory_tool.py)."""

import os
import pytest
import tempfile

# Override the DB path BEFORE importing the tool
_tmp_dir = tempfile.mkdtemp()
os.environ["HERMES_UNIFIED_MEMORY_DB"] = os.path.join(_tmp_dir, "test_um.db")

from tools.unified_memory_tool import (
    _handle_mcp_umemory_write,
    _handle_mcp_umemory_recall,
    _handle_mcp_umemory_search,
    _handle_mcp_umemory_reflect,
    _handle_mcp_umemory_reward,
    _handle_mcp_umemory_stats,
    _handle_mcp_umemory_consolidate,
    _handle_mcp_umemory_explore,
    get_unified_memory_injection,
    tick_unified_memory,
)


class TestWriteTool:
    def test_write_plain_text(self):
        result = _handle_mcp_umemory_write({"content": "Python uses whitespace"})
        assert "stored:" in result
        assert "gauge:" in result

    def test_write_notation(self):
        result = _handle_mcp_umemory_write({"content": "V[api.url]: https://example.com"})
        assert "stored:" in result

    def test_write_with_scope(self):
        result = _handle_mcp_umemory_write({
            "content": "D[auth]: Use OAuth2",
            "scope": "project:hermes"
        })
        assert "stored:" in result

    def test_write_missing_content(self):
        result = _handle_mcp_umemory_write({})
        assert "UMEMORY ERROR" in result or "content" in result.lower()


class TestRecallTool:
    def test_recall_basic(self):
        _handle_mcp_umemory_write({"content": "The API uses JWT tokens for auth"})
        result = _handle_mcp_umemory_recall({"query": "API authentication"})
        assert "JWT" in result or "score" in result

    def test_recall_no_results(self):
        result = _handle_mcp_umemory_recall({"query": "xyznonexistentqueryzyx"})
        # Should return something (even low-scoring results) or "no results"
        assert isinstance(result, str)


class TestSearchTool:
    def test_search_keyword(self):
        _handle_mcp_umemory_write({"content": "PostgreSQL database version 15"})
        result = _handle_mcp_umemory_search({"query": "PostgreSQL"})
        assert "PostgreSQL" in result or "no results" in result


class TestReflectTool:
    def test_reflect_groups_by_type(self):
        _handle_mcp_umemory_write({"content": "V[db]: PostgreSQL 15"})
        _handle_mcp_umemory_write({"content": "C[auth]: JWT required"})
        result = _handle_mcp_umemory_reflect({"topic": "database auth"})
        assert isinstance(result, str)
        # Should have some grouping structure
        assert ":" in result


class TestStatsTool:
    def test_stats_returns_counts(self):
        result = _handle_mcp_umemory_stats({})
        assert "fact_count" in result
        assert "gauge" in result


class TestConsolidateTool:
    def test_consolidate_runs(self):
        result = _handle_mcp_umemory_consolidate({})
        assert "promoted" in result or "consolidation" in result.lower()


class TestInjection:
    def test_injection_returns_string(self):
        result = get_unified_memory_injection()
        assert isinstance(result, str)

    def test_tick_runs_silently(self):
        # Should not raise
        tick_unified_memory(turn=1, message_text="hello")


class TestRewardTool:
    def test_reward_without_id(self):
        result = _handle_mcp_umemory_reward({})
        assert "UMEMORY ERROR" in result or "memory_id" in result.lower()
