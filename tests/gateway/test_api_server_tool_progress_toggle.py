"""Tests for tool_progress_events config toggle.

Covers the new request-level gate that lets frontends disable custom
``event: hermes.tool.progress`` SSE events via
platforms.api_server.tool_progress_events: false."""

import asyncio
import pathlib

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_stream_app(adapter):
    """Create a minimal app with just the stream endpoint."""
    app = web.Application()
    app.router.add_post(
        "/v1/chat/completions", adapter._handle_chat_completions,
    )
    app.router.add_post(
        "/api/sessions/{session_id}/chat/stream",
        adapter._handle_session_chat_stream,
    )
    return app


@pytest.fixture
def session_db():
    from hermes_state import SessionDB
    import tempfile

    tmp = tempfile.mkdtemp(prefix="session-db-")
    db = SessionDB(pathlib.Path(tmp) / "sessions.sqlite")
    yield db
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# load_gateway_config integration — top-level vs nested config
# ---------------------------------------------------------------------------


def test_config_bridges_top_level_key_api_server_tool_progress_events_to_extra(monkeypatch, tmp_path):
    """api_server.tool_progress_events: false at top-level must reach extra."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "api_server:\n  tool_progress_events: false\n"
    )
    monkeypatch.setattr(
        "gateway.config.get_hermes_home", lambda: hermes_home,
    )
    monkeypatch.setenv("API_SERVER_ENABLED", "true")
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)

    from gateway.config import load_gateway_config

    config = load_gateway_config()
    api_config = config.platforms.get(Platform.API_SERVER)
    assert api_config is not None
    assert api_config.extra.get("tool_progress_events") is False


def test_config_bridges_nested_platforms_key_to_extra(monkeypatch, tmp_path):
    """platforms.api_server.tool_progress_events under the nested block must also work."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platforms:\n  api_server:\n    tool_progress_events: false\n"
    )
    monkeypatch.setattr(
        "gateway.config.get_hermes_home", lambda: hermes_home,
    )
    monkeypatch.setenv("API_SERVER_ENABLED", "true")
    monkeypatch.delenv("API_SERVER_PORT", raising=False)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)

    from gateway.config import load_gateway_config

    config = load_gateway_config()
    api_config = config.platforms.get(Platform.API_SERVER)
    assert api_config is not None
    assert api_config.extra.get("tool_progress_events") is False


# ---------------------------------------------------------------------------
# Adapter-level toggle — all three emission sites gated
# ---------------------------------------------------------------------------


class TestToolProgressEventsWhenDisabled:
    """When _tool_progress_events is False, no hermes.tool.progress events appear."""

    @pytest.mark.asyncio
    async def test_session_chat_stream_suppresses_tool_progress_events(
        self, session_db,
    ):
        adapter = APIServerAdapter(PlatformConfig(
            enabled=True,
            extra={"tool_progress_events": False},
        ))
        adapter._session_db = session_db
        sid = session_db.create_session("test-session", "api_server")

        async def fake_run(**kwargs):
            kwargs["stream_delta_callback"]("Hello")
            kwargs["tool_progress_callback"](
                "reasoning.available", tool_name="_thinking", preview="thinking",
            )
            return {"final_response": "Hello"}, {"total_tokens": 1}

        app = _make_stream_app(adapter)
        import unittest.mock as mock

        with mock.patch.object(adapter, "_run_agent", side_effect=fake_run):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    f"/api/sessions/{sid}/chat/stream",
                    json={"message": "hi"},
                )
                assert resp.status == 200
                body = await resp.text()

        assert "event: assistant.delta" in body
        assert "event: tool.progress" not in body

    @pytest.mark.asyncio
    async def test_chat_completions_suppresses_hermes_tool_progress_sse(
        self, session_db,
    ):
        adapter = APIServerAdapter(PlatformConfig(
            enabled=True,
            extra={"tool_progress_events": False},
        ))
        adapter._session_db = session_db

        async def fake_run(**kwargs):
            kwargs["stream_delta_callback"]("hello")
            return {"final_response": "hello"}, {"total_tokens": 1}

        app = _make_stream_app(adapter)
        import unittest.mock as mock

        with mock.patch.object(adapter, "_run_agent", side_effect=fake_run):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert "event: hermes.tool.progress" not in body
        assert "hello" in body.lower()

    @pytest.mark.asyncio
    async def test_runs_callback_suppresses_tool_events(self):
        adapter = APIServerAdapter(PlatformConfig(
            enabled=True,
            extra={"tool_progress_events": False},
        ))
        run_id = "run_disabled_progress"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._set_run_status(run_id, "running")

        callback = adapter._make_run_event_callback(run_id, asyncio.get_running_loop())
        callback("tool.started", tool_name="terminal", preview="ls")
        await asyncio.sleep(0)

        assert adapter._run_streams[run_id].empty()
        assert "last_event" not in adapter._run_statuses[run_id]
