"""Tests for naming the failing field in durable-upload errors (#3680).

Tools with more than one durable-upload field (e.g. jira_ticket_create with
attachment_path, attachment_path_2, attachment_path_3) should name the field
that failed in the not-found message.  Single-field tools keep today's text.
"""

from __future__ import annotations

import sys
import types

# Stub the MCP modules that fail to import in some local environments
# due to a pydantic/FastMCP version mismatch.
_mcp = types.ModuleType("mcp")
_mcp.ClientSession = type("ClientSession", (), {})
_mcp.StdioServerParameters = type("StdioServerParameters", (), {})
_mcp_types = types.ModuleType("mcp.types")
_mcp_types.CallToolResult = type("CallToolResult", (), {})
_mcp_types.Tool = type("Tool", (), {})
_mcp_types.TextContent = type("TextContent", (), {})
_mcp_types.ImageContent = type("ImageContent", (), {})
_mcp_types.EmbeddedResource = type("EmbeddedResource", (), {})
_mcp_types.ClientRequest = type("ClientRequest", (), {})
_mcp_types.ListToolsRequest = type("ListToolsRequest", (), {})
_mcp_types.ListToolsResult = type("ListToolsResult", (), {})
_mcp_types.PaginatedRequestParams = type("PaginatedRequestParams", (), {})
sys.modules.setdefault("mcp", _mcp)
sys.modules.setdefault("mcp.types", _mcp_types)
sys.modules.setdefault("mcp.client", types.ModuleType("mcp.client"))
sys.modules.setdefault("mcp.client.sse", types.ModuleType("mcp.client.sse"))
sys.modules.setdefault("mcp.client.stdio", types.ModuleType("mcp.client.stdio"))
sys.modules.setdefault(
    "mcp.client.streamable_http", types.ModuleType("mcp.client.streamable_http")
)
sys.modules.setdefault("mcp.server", types.ModuleType("mcp.server"))
sys.modules.setdefault("mcp.server.fastmcp", types.ModuleType("mcp.server.fastmcp"))

# Provide stub functions for the client factory functions.
sys.modules["mcp.client.sse"].sse_client = lambda *a, **kw: None
sys.modules["mcp.client.stdio"].stdio_client = lambda *a, **kw: None
sys.modules["mcp.client.streamable_http"].streamablehttp_client = lambda *a, **kw: None

from xagent.core.tools.adapters.vibe.mcp_adapter import (  # noqa: E402
    _durable_upload_not_found_message,
    _durable_upload_note,
    _DurableUploadFileNotFoundError,
)

# ---------------------------------------------------------------------------
# Single-field tools: messages must be unchanged
# ---------------------------------------------------------------------------


def test_single_field_not_found_message_unchanged():
    """Single-field tools keep today's generic message."""
    msg = _durable_upload_not_found_message(("local_file_path",), own_file_id=False)
    assert msg == "file_id not found or not accessible."


def test_single_field_not_found_with_failed_field_unchanged():
    """Even with failed_field, single-field tools keep today's message."""
    msg = _durable_upload_not_found_message(
        ("local_file_path",), own_file_id=False, failed_field="local_file_path"
    )
    assert msg == "file_id not found or not accessible."


def test_single_field_own_file_id_not_found_message_unchanged():
    msg = _durable_upload_not_found_message(
        ("file_path",), own_file_id=True, failed_field="file_path"
    )
    assert "file_path: the registered workspace file" in msg
    assert "not the file_id argument" in msg


# ---------------------------------------------------------------------------
# Multi-field tools: name the failing field
# ---------------------------------------------------------------------------

_MULTI_FIELDS = ("attachment_path", "attachment_path_2", "attachment_path_3")


def test_multi_field_not_found_names_failed_field():
    """When a specific field fails, the message names it."""
    msg = _durable_upload_not_found_message(
        _MULTI_FIELDS, own_file_id=False, failed_field="attachment_path_2"
    )
    assert "attachment_path_2" in msg
    assert "not found or not accessible" in msg
    assert "attachment_path_3" not in msg  # doesn't list all fields


def test_multi_field_not_found_without_failed_field_lists_all():
    """Without a specific failed field, lists all fields (backward compat)."""
    msg = _durable_upload_not_found_message(_MULTI_FIELDS, own_file_id=False)
    assert msg == "file_id not found or not accessible."


def test_multi_field_own_file_id_names_failed_field():
    msg = _durable_upload_not_found_message(
        _MULTI_FIELDS, own_file_id=True, failed_field="attachment_path_2"
    )
    assert "attachment_path_2" in msg
    assert "not the file_id argument" in msg


# ---------------------------------------------------------------------------
# Note wording for several fields
# ---------------------------------------------------------------------------


def test_multi_field_note_lists_all_fields():
    note = _durable_upload_note(_MULTI_FIELDS, own_file_id=False)
    assert "attachment_path" in note
    assert "attachment_path_2" in note
    assert "attachment_path_3" in note
    assert " or " in note


def test_single_field_note_unchanged():
    note = _durable_upload_note(("local_file_path",), own_file_id=False)
    assert "the local upload argument" in note


# ---------------------------------------------------------------------------
# Exception carries the field name
# ---------------------------------------------------------------------------


def test_durable_upload_error_carries_field_name():
    exc = _DurableUploadFileNotFoundError("attachment_path_2")
    assert exc.field_name == "attachment_path_2"
    assert isinstance(exc, FileNotFoundError)
