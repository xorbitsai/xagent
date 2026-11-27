"""Tests for the OneDrive MCP server.

Verifies that tool failures are raised as ``ToolError`` (``isError`` in MCP)
instead of being returned as JSON strings with ``status: error``.
"""

import json
from unittest.mock import Mock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from xagent.web.tools.mcp import onedrive


class MockResponse:
    def __init__(self, json_data=None, status_code=200, text=""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.text = text
        self.content = text.encode() if text else b"{}"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def test_get_profile_returns_success(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    mock_resp = MockResponse({"id": "user-1", "displayName": "Test User"})
    monkeypatch.setattr(onedrive.requests, "request", Mock(return_value=mock_resp))

    result = json.loads(onedrive.onedrive_get_profile())
    assert result["status"] == "success"
    assert result["user"]["displayName"] == "Test User"


def test_get_profile_raises_tool_error(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(side_effect=RuntimeError("network down"))
    )

    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_get_profile()
    assert "network down" in str(exc_info.value)


def test_list_items_raises_tool_error(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(side_effect=RuntimeError("timeout"))
    )

    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_list_items()
    assert "timeout" in str(exc_info.value)


def test_search_files_raises_tool_error(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(side_effect=RuntimeError("search failed"))
    )

    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_search_files("query")
    assert "search failed" in str(exc_info.value)


def test_get_item_raises_tool_error(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(side_effect=RuntimeError("not found"))
    )

    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_get_item(path="/test.txt")
    assert "not found" in str(exc_info.value)


def test_get_file_content_raises_tool_error(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(side_effect=RuntimeError("download error"))
    )

    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_get_file_content("/test.txt")
    assert "download error" in str(exc_info.value)


def test_upload_text_file_raises_tool_error(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(side_effect=RuntimeError("upload failed")),
    )

    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_upload_text_file("/test.txt", "content")
    assert "upload failed" in str(exc_info.value)


def test_create_folder_raises_tool_error(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(side_effect=RuntimeError("create failed"))
    )

    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_create_folder("NewFolder")
    assert "create failed" in str(exc_info.value)


def test_rename_item_raises_tool_error(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(side_effect=RuntimeError("rename failed")),
    )

    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_rename_item("item-1", "newname.txt")
    assert "rename failed" in str(exc_info.value)


def test_delete_item_raises_tool_error(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(side_effect=RuntimeError("delete failed"))
    )

    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_delete_item("item-1")
    assert "delete failed" in str(exc_info.value)


def test_missing_token_raises_tool_error(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    with pytest.raises(ToolError) as exc_info:
        onedrive.onedrive_get_profile()
    assert "AUTH_TOKEN" in str(exc_info.value)
