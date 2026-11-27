"""Tests for the Google Calendar MCP server (issue #3985).

Verifies that tool failures are raised as ``ToolError`` (``isError`` in MCP)
instead of being returned as JSON strings with ``status: error``.
"""

import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import calendar


def _mock_service_success(data):
    """Return a mock calendar service whose chain ends with .execute() -> data."""
    mock_request = Mock()
    mock_request.execute.return_value = data
    mock_resource = Mock()
    mock_resource.list.return_value = mock_request
    mock_resource.insert.return_value = mock_request
    mock_resource.get.return_value = mock_request
    mock_resource.update.return_value = mock_request
    mock_resource.delete.return_value = mock_request
    mock_service = Mock()
    mock_service.events.return_value = mock_resource
    return mock_service


def _mock_service_error(msg):
    """Return a mock calendar service that raises when .execute() is called."""
    mock_request = Mock()
    mock_request.execute.side_effect = RuntimeError(msg)
    mock_resource = Mock()
    mock_resource.list.return_value = mock_request
    mock_resource.insert.return_value = mock_request
    mock_resource.get.return_value = mock_request
    mock_resource.update.return_value = mock_request
    mock_resource.delete.return_value = mock_request
    mock_service = Mock()
    mock_service.events.return_value = mock_resource
    return mock_service


# ---------------------------------------------------------------------------
# Success-path tests
# ---------------------------------------------------------------------------


def test_search_events_success(monkeypatch):
    mock_service = _mock_service_success({"items": [{"id": "e1"}]})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: mock_service)
    result = json.loads(calendar.google_calendar_search_events())
    assert result["status"] == "success"
    assert result["events"] == [{"id": "e1"}]


def test_create_event_success(monkeypatch):
    mock_service = _mock_service_success({"id": "e1", "summary": "Meeting"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: mock_service)
    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Meeting",
            start_time="2026-12-01T10:00:00Z",
            end_time="2026-12-01T11:00:00Z",
        )
    )
    assert result["status"] == "success"


def test_get_event_success(monkeypatch):
    mock_service = _mock_service_success({"id": "e1"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: mock_service)
    result = json.loads(calendar.google_calendar_get_event("e1"))
    assert result["status"] == "success"


def test_delete_event_success(monkeypatch):
    mock_service = _mock_service_success(None)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: mock_service)
    result = json.loads(calendar.google_calendar_delete_events("e1"))
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Failure-path tests
# ---------------------------------------------------------------------------


def test_search_events_failure_raises_tool_error(monkeypatch):
    mock_service = _mock_service_error("Invalid credentials")
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: mock_service)
    with pytest.raises(calendar.ToolError, match="Invalid credentials"):
        calendar.google_calendar_search_events()


def test_create_event_failure_raises_tool_error(monkeypatch):
    mock_service = _mock_service_error("Time range invalid")
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: mock_service)
    with pytest.raises(calendar.ToolError, match="Time range invalid"):
        calendar.google_calendar_create_events(
            summary="M", start_time="bad", end_time="bad"
        )


def test_get_event_failure_raises_tool_error(monkeypatch):
    mock_service = _mock_service_error("Event not found")
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: mock_service)
    with pytest.raises(calendar.ToolError, match="Event not found"):
        calendar.google_calendar_get_event("missing")


def test_update_event_failure_raises_tool_error(monkeypatch):
    mock_service = _mock_service_error("Not found")
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: mock_service)
    with pytest.raises(calendar.ToolError, match="Not found"):
        calendar.google_calendar_update_events("missing", summary="Updated")


def test_delete_event_failure_raises_tool_error(monkeypatch):
    mock_service = _mock_service_error("Forbidden")
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: mock_service)
    with pytest.raises(calendar.ToolError, match="Forbidden"):
        calendar.google_calendar_delete_events("missing")


# ---------------------------------------------------------------------------
# Missing-token test
# ---------------------------------------------------------------------------


def test_missing_token_raises_tool_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)
    with pytest.raises(calendar.ToolError, match="GOOGLE_ACCESS_TOKEN"):
        calendar.google_calendar_search_events()
