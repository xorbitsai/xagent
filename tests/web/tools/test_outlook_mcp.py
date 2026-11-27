"""Tests for the Outlook MCP server (issue #3311).

Verifies that tool failures are raised as ``ToolError`` (``isError`` in MCP)
instead of being returned as JSON strings with ``status: error``.
"""

import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import outlook


class MockResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json_data = json_data or {}
        self.text = text
        self.status_code = status_code
        self.content = text.encode() if text else b""

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests as req

            raise req.HTTPError(f"{self.status_code} Error", response=self)


def _mock_graph_request_success(data):
    """Return a Mock that replaces _graph_request and returns *data*."""
    return Mock(return_value=data)


def _mock_graph_request_error(msg):
    """Return a Mock that raises RuntimeError(*msg*)."""
    return Mock(side_effect=RuntimeError(msg))


# ---------------------------------------------------------------------------
# Success-path tests (verify behavior is unchanged)
# ---------------------------------------------------------------------------


def test_get_profile_success(monkeypatch):
    mock = _mock_graph_request_success({"id": "u1", "displayName": "Test"})
    monkeypatch.setattr(outlook, "_graph_request", mock)

    result = json.loads(outlook.outlook_get_profile())
    assert result["status"] == "success"
    assert result["user"]["displayName"] == "Test"


def test_list_messages_success(monkeypatch):
    mock = _mock_graph_request_success(
        {"value": [{"id": "m1"}], "@odata.nextLink": "next"}
    )
    monkeypatch.setattr(outlook, "_graph_request", mock)

    result = json.loads(outlook.outlook_list_messages(top=5))
    assert result["status"] == "success"
    assert result["messages"] == [{"id": "m1"}]
    assert result["next_link"] == "next"


def test_send_message_success(monkeypatch):
    mock = Mock(return_value={})
    monkeypatch.setattr(outlook, "_graph_request", mock)

    result = json.loads(
        outlook.outlook_send_message(to="a@b.com", subject="hi", body="hello")
    )
    assert result["status"] == "success"


def test_create_event_success(monkeypatch):
    mock = _mock_graph_request_success({"id": "e1"})
    monkeypatch.setattr(outlook, "_graph_request", mock)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Meeting",
            start_datetime="2026-12-01T10:00",
            end_datetime="2026-12-01T11:00",
        )
    )
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Failure-path tests (verify ToolError is raised)
# ---------------------------------------------------------------------------


def test_get_profile_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(
        outlook, "_graph_request", _mock_graph_request_error("Unauthorized")
    )
    with pytest.raises(outlook.ToolError, match="Unauthorized"):
        outlook.outlook_get_profile()


def test_list_messages_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(
        outlook, "_graph_request", _mock_graph_request_error("Bad request")
    )
    with pytest.raises(outlook.ToolError, match="Bad request"):
        outlook.outlook_list_messages()


def test_get_message_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(
        outlook, "_graph_request", _mock_graph_request_error("Not found")
    )
    with pytest.raises(outlook.ToolError, match="Not found"):
        outlook.outlook_get_message("missing-id")


def test_send_message_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(
        outlook, "_graph_request", _mock_graph_request_error("Recipient invalid")
    )
    with pytest.raises(outlook.ToolError, match="Recipient invalid"):
        outlook.outlook_send_message(to="a@b.com", subject="hi", body="hello")


def test_list_events_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(
        outlook, "_graph_request", _mock_graph_request_error("Server error")
    )
    with pytest.raises(outlook.ToolError, match="Server error"):
        outlook.outlook_list_events()


def test_create_event_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(
        outlook, "_graph_request", _mock_graph_request_error("Conflict")
    )
    with pytest.raises(outlook.ToolError, match="Conflict"):
        outlook.outlook_create_event(
            subject="Meeting",
            start_datetime="2026-12-01T10:00",
            end_datetime="2026-12-01T11:00",
        )


def test_update_event_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(
        outlook, "_graph_request", _mock_graph_request_error("Not found")
    )
    with pytest.raises(outlook.ToolError, match="Not found"):
        outlook.outlook_update_event("missing-id", subject="Updated")


def test_delete_event_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(
        outlook, "_graph_request", _mock_graph_request_error("Forbidden")
    )
    with pytest.raises(outlook.ToolError, match="Forbidden"):
        outlook.outlook_delete_event("missing-id")


def test_list_contacts_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(outlook, "_graph_request", _mock_graph_request_error("Timeout"))
    with pytest.raises(outlook.ToolError, match="Timeout"):
        outlook.outlook_list_contacts()


# ---------------------------------------------------------------------------
# Validation-error tests (bad input should also be ToolError)
# ---------------------------------------------------------------------------


def test_get_message_invalid_body_type_raises_tool_error(monkeypatch):
    mock = Mock(return_value={})
    monkeypatch.setattr(outlook, "_graph_request", mock)
    with pytest.raises(outlook.ToolError, match="body_type must be"):
        outlook.outlook_get_message("m1", body_type="markdown")


def test_send_message_no_recipients_raises_tool_error(monkeypatch):
    mock = Mock(return_value={})
    monkeypatch.setattr(outlook, "_graph_request", mock)
    with pytest.raises(outlook.ToolError, match="at least one recipient"):
        outlook.outlook_send_message(to="", subject="hi", body="hello")


def test_create_event_invalid_content_type_raises_tool_error(monkeypatch):
    """body validation through _message_body should surface as ToolError."""
    mock = Mock(return_value={})
    monkeypatch.setattr(outlook, "_graph_request", mock)
    with pytest.raises(outlook.ToolError, match="content_type must be"):
        outlook.outlook_send_message(
            to="a@b.com", subject="hi", body="hello", content_type="markdown"
        )


def test_update_event_no_fields_raises_tool_error(monkeypatch):
    mock = Mock(return_value={})
    monkeypatch.setattr(outlook, "_graph_request", mock)
    with pytest.raises(outlook.ToolError, match="at least one field"):
        outlook.outlook_update_event("e1")
