"""Tests for the Teams MCP server (issue #3985).

Verifies that tool failures are raised as ``ToolError`` (``isError`` in MCP)
instead of being returned as JSON strings with ``status: error``.
"""

import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import teams


def _mock_graph_success(data):
    return Mock(return_value=data)


def _mock_graph_error(msg):
    return Mock(side_effect=RuntimeError(msg))


# ---------------------------------------------------------------------------
# Success-path tests
# ---------------------------------------------------------------------------


def test_auth_status_success(monkeypatch):
    mock = _mock_graph_success({"id": "u1", "displayName": "Test"})
    monkeypatch.setattr(teams, "_graph_request", mock)
    result = json.loads(teams.auth_status())
    assert result["status"] == "success"
    assert result["authenticated"] is True


def test_list_teams_success(monkeypatch):
    mock = _mock_graph_success({"value": [{"id": "t1"}]})
    monkeypatch.setattr(teams, "_graph_request", mock)
    result = json.loads(teams.list_teams())
    assert result["status"] == "success"
    assert result["teams"] == [{"id": "t1"}]


def test_list_channels_success(monkeypatch):
    mock = _mock_graph_success({"value": [{"id": "c1"}]})
    monkeypatch.setattr(teams, "_graph_request", mock)
    result = json.loads(teams.list_channels("team-1"))
    assert result["status"] == "success"


def test_send_channel_message_success(monkeypatch):
    mock = _mock_graph_success({"id": "msg1"})
    monkeypatch.setattr(teams, "_graph_request", mock)
    result = json.loads(teams.send_channel_message("team-1", "ch-1", "hello"))
    assert result["status"] == "success"


def test_send_chat_message_success(monkeypatch):
    mock = _mock_graph_success({"id": "msg2"})
    monkeypatch.setattr(teams, "_graph_request", mock)
    result = json.loads(teams.send_chat_message("chat-1", "hi"))
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Failure-path tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,args",
    [
        ("auth_status", {}),
        ("get_current_user", {}),
        ("list_teams", {}),
        ("list_channels", {"team_id": "t1"}),
        ("list_team_members", {"team_id": "t1"}),
        ("get_channel_messages", {"team_id": "t1", "channel_id": "c1"}),
        (
            "send_channel_message",
            {"team_id": "t1", "channel_id": "c1", "message": "hi"},
        ),
        ("list_chats", {}),
        ("get_chat_messages", {"chat_id": "ch1"}),
        ("send_chat_message", {"chat_id": "ch1", "message": "hi"}),
    ],
)
def test_teams_tool_failure_raises_tool_error(monkeypatch, tool_name, args):
    monkeypatch.setattr(teams, "_graph_request", _mock_graph_error("Server error"))
    with pytest.raises(teams.ToolError, match="Server error"):
        getattr(teams, tool_name)(**args)


# ---------------------------------------------------------------------------
# Validation-error tests
# ---------------------------------------------------------------------------


def test_send_channel_message_invalid_content_type_raises_tool_error(monkeypatch):
    mock = Mock(return_value={})
    monkeypatch.setattr(teams, "_graph_request", mock)
    with pytest.raises(teams.ToolError, match="content_type must be"):
        teams.send_channel_message("t1", "c1", "hi", content_type="markdown")


def test_send_chat_message_invalid_content_type_raises_tool_error(monkeypatch):
    mock = Mock(return_value={})
    monkeypatch.setattr(teams, "_graph_request", mock)
    with pytest.raises(teams.ToolError, match="content_type must be"):
        teams.send_chat_message("ch1", "hi", content_type="markdown")
