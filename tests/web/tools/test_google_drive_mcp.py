"""Tests for the Google Drive MCP server.

Verifies that tool failures are raised as ``ToolError`` (``isError`` in MCP)
instead of being returned as JSON strings with ``status: error``.
"""

import json
from unittest.mock import Mock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from xagent.web.tools.mcp import google_drive


def _mock_service_success(data):
    """Return a mock Drive service whose chain ends with .execute() -> data."""
    mock_request = Mock()
    mock_request.execute.return_value = data
    mock_resource = Mock()
    mock_resource.list.return_value = mock_request
    service = Mock()
    service.files.return_value = mock_resource
    return service


def test_search_returns_success(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    service = _mock_service_success({"files": [{"id": "f1", "name": "doc"}]})
    monkeypatch.setattr(google_drive, "build", Mock(return_value=service))

    result = json.loads(google_drive.google_drive_search("doc"))
    assert result["status"] == "success"
    assert result["files"] == [{"id": "f1", "name": "doc"}]


def test_search_raises_tool_error_on_failure(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(
        google_drive, "build", Mock(side_effect=RuntimeError("drive API down"))
    )

    with pytest.raises(ToolError) as exc_info:
        google_drive.google_drive_search()
    assert "drive API down" in str(exc_info.value)


def test_get_file_content_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(
        google_drive, "build", Mock(side_effect=RuntimeError("token expired"))
    )

    with pytest.raises(ToolError) as exc_info:
        google_drive.google_drive_get_file_content("file-123")
    assert "token expired" in str(exc_info.value)


def test_create_file_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(
        google_drive, "build", Mock(side_effect=RuntimeError("quota exceeded"))
    )

    with pytest.raises(ToolError) as exc_info:
        google_drive.google_drive_create_file("test.txt", "hello")
    assert "quota exceeded" in str(exc_info.value)


def test_create_folder_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(
        google_drive, "build", Mock(side_effect=RuntimeError("permission denied"))
    )

    with pytest.raises(ToolError) as exc_info:
        google_drive.google_drive_create_folder("New Folder")
    assert "permission denied" in str(exc_info.value)


def test_rename_file_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(
        google_drive, "build", Mock(side_effect=RuntimeError("file not found"))
    )

    with pytest.raises(ToolError) as exc_info:
        google_drive.google_drive_rename_file("file-123", "renamed.txt")
    assert "file not found" in str(exc_info.value)


def test_delete_file_raises_tool_error(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(
        google_drive, "build", Mock(side_effect=RuntimeError("network error"))
    )

    with pytest.raises(ToolError) as exc_info:
        google_drive.google_drive_delete_file("file-123")
    assert "network error" in str(exc_info.value)


def test_missing_token_raises_tool_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)
    with pytest.raises(ToolError) as exc_info:
        google_drive.google_drive_search()
    assert "GOOGLE_ACCESS_TOKEN" in str(exc_info.value)
