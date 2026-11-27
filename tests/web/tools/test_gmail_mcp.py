"""Tests for the Gmail MCP server.

Verifies that tool failures are raised as ``ToolError`` (``isError`` in MCP)
instead of being returned as JSON strings with ``status: error``.
"""

from unittest.mock import Mock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from xagent.web.tools.mcp import gmail


def _mock_service():
    """Create a mock Gmail service."""
    service = Mock()
    return service


def test_search_messages_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(
        gmail, "build", Mock(side_effect=RuntimeError("gmail API down"))
    )

    with pytest.raises(ToolError) as exc_info:
        gmail.gmail_search_messages("test query")
    assert "gmail API down" in str(exc_info.value)


def test_read_threads_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(gmail, "build", Mock(side_effect=RuntimeError("token expired")))

    with pytest.raises(ToolError) as exc_info:
        gmail.gmail_read_threads(["thread-1"])
    assert "token expired" in str(exc_info.value)


def test_send_messages_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(gmail, "build", Mock(side_effect=RuntimeError("send failed")))

    with pytest.raises(ToolError) as exc_info:
        gmail.gmail_send_messages([{"to": "test@example.com"}])
    assert "send failed" in str(exc_info.value)


def test_manage_labels_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(gmail, "build", Mock(side_effect=RuntimeError("label error")))

    with pytest.raises(ToolError) as exc_info:
        gmail.gmail_manage_labels(action="list")
    assert "label error" in str(exc_info.value)


def test_manage_labels_unknown_action_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    service = _mock_service()
    monkeypatch.setattr(gmail, "build", Mock(return_value=service))

    with pytest.raises(ToolError) as exc_info:
        gmail.gmail_manage_labels(action="unknown")
    assert "Unknown action" in str(exc_info.value)


def test_missing_token_raises_tool_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)
    with pytest.raises(ToolError) as exc_info:
        gmail.gmail_search_messages("test")
    assert "GOOGLE_ACCESS_TOKEN" in str(exc_info.value)
