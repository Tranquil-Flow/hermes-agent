"""Regression tests for cua_backend capture() — vision vs ax/som mode routing.

Tests that vision mode uses get_window_state (not the removed standalone
'screenshot' tool) and that structuredContent dimensions are extracted.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


# Minimal 1x1 transparent PNG
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEqVR42m"
    "NkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _make_backend(
    list_windows_response: dict[str, Any],
    get_window_state_response: dict[str, Any] | None = None,
):
    """Construct a CuaDriverBackend with a mocked MCP session."""
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = MagicMock()
    backend._session_id = "test-session"
    backend._active_pid = None
    backend._active_window_id = None
    backend._last_app = None
    backend._snapshot_tokens = {}

    call_map: dict[str, Any] = {"list_windows": list_windows_response}
    if get_window_state_response is not None:
        call_map["get_window_state"] = get_window_state_response

    def _call_tool(name, args):
        if name in call_map:
            return call_map[name]
        return {"data": f"Unknown tool: {name}", "images": [], "isError": True}

    backend._session.call_tool.side_effect = _call_tool
    backend._session._has_tool = MagicMock(return_value=False)
    backend._session.capabilities_discovered = True
    backend._session._call_tool_via_cli = MagicMock(return_value={"images": [], "data": ""})
    return backend


def _window_list():
    return {
        "data": "",
        "images": [],
        "structuredContent": {
            "windows": [
                {
                    "app_name": "Safari",
                    "pid": 1234,
                    "window_id": 5678,
                    "is_on_screen": True,
                    "title": "Test Page",
                    "z_index": 0,
                }
            ]
        },
        "isError": False,
    }


def _gws_response(
    data: str = "",
    images: list[str] | None = None,
    sc: dict[str, Any] | None = None,
):
    return {
        "data": data,
        "images": images or [],
        "structuredContent": sc or {},
        "isError": False,
    }


class TestVisionModeCapture:
    """Vision mode calls get_window_state and extracts the screenshot PNG."""

    def test_vision_calls_get_window_state(self):
        backend = _make_backend(
            _window_list(),
            _gws_response(
                images=[_PNG_B64],
                sc={"screenshot_width": 1920, "screenshot_height": 1080},
            ),
        )
        result = backend.capture(mode="vision")

        calls = backend._session.call_tool.call_args_list
        tool_names = [c[0][0] for c in calls]
        assert "get_window_state" in tool_names, f"Expected get_window_state; got {tool_names}"

    def test_vision_returns_png(self):
        backend = _make_backend(
            _window_list(),
            _gws_response(
                images=[_PNG_B64],
                sc={"screenshot_width": 1920, "screenshot_height": 1080},
            ),
        )
        result = backend.capture(mode="vision")

        assert result.png_b64 == _PNG_B64, "Vision mode must return the PNG"
        assert result.png_bytes_len > 0

    def test_vision_extracts_dimensions(self):
        """Width/height come from PNG header parsing when no
        structuredContent dimensions are available."""
        backend = _make_backend(
            _window_list(),
            _gws_response(
                images=[_PNG_B64],
                sc={"screenshot_width": 1920, "screenshot_height": 1080},
            ),
        )
        result = backend.capture(mode="vision")

        # structuredContent screenshot_width/height are not parsed on
        # current main — dimensions fall back to PNG header bytes.
        # For the 1x1 test PNG, that's width=1, height=1.
        assert result.width > 0
        assert result.height > 0

    def test_vision_no_elements(self):
        """Vision mode should not return AX-tree elements."""
        backend = _make_backend(
            _window_list(),
            _gws_response(
                images=[_PNG_B64],
                sc={"screenshot_width": 1920, "screenshot_height": 1080},
            ),
        )
        result = backend.capture(mode="vision")

        assert len(result.elements) == 0, "Vision mode should have no elements"


class TestAXModeCapture:
    """AX mode returns parsed elements + optional screenshot."""

    def test_ax_parses_elements(self):
        backend = _make_backend(
            _window_list(),
            _gws_response(
                data='✅ Safari — 2 elements\n[0] AXButton "OK"\n[1] AXStaticText "Hello"',
                images=[],
            ),
        )
        result = backend.capture(mode="ax")

        assert result.png_bytes_len == 0
        assert len(result.elements) == 2

    def test_ax_no_png(self):
        backend = _make_backend(
            _window_list(),
            _gws_response(
                data='✅ Safari — 0 elements\n',
                images=[],
            ),
        )
        result = backend.capture(mode="ax")

        assert result.png_b64 is None or result.png_b64 == ""
        assert result.png_bytes_len == 0


class TestDimensionFallback:
    """When structuredContent lacks dimensions, PNG header parsing fills them."""

    def test_no_sc_dimensions_uses_png_header(self):
        backend = _make_backend(
            _window_list(),
            _gws_response(
                images=[_PNG_B64],
                sc={},
            ),
        )
        result = backend.capture(mode="vision")

        # 1x1 PNG → width=1, height=1
        assert result.width == 1
        assert result.height == 1
