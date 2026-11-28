"""Read-only annotations on the Google Docs, Sheets, and Slides MCP tools."""

import json
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from xagent.core.agent.result import tool_result_succeeded
from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import MCPToolAdapter
from xagent.core.tools.core.mcp.tools import load_mcp_tools, raw_annotations_for
from xagent.web.tools.mcp import google_docs, google_sheets, google_slides


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server", "unannotated_writes"),
    [
        (
            google_docs.mcp,
            {
                "google_docs_append_text",
                "google_docs_batch_update",
                "google_docs_create_document",
                "google_docs_replace_text",
            },
        ),
        (
            google_sheets.mcp,
            {
                "google_sheets_add_sheet",
                "google_sheets_append_rows",
                "google_sheets_clear_range",
                "google_sheets_create_spreadsheet",
                "google_sheets_delete_sheet",
                "google_sheets_update_range",
            },
        ),
        (
            google_slides.mcp,
            {
                "google_slides_add_slide",
                "google_slides_batch_update",
                "google_slides_create_presentation",
                "google_slides_delete_slide",
                # The thumbnail tool creates a PNG in the task workspace.
                "google_slides_get_slide_thumbnail",
                "google_slides_import_pptx",
                "google_slides_update_slide",
            },
        ),
    ],
)
async def test_google_workspace_tool_annotations(server, unannotated_writes):
    tools = await server.list_tools()

    assert {tool.name for tool in tools if tool.annotations is None} == (
        unannotated_writes
    )
    for tool in tools:
        if tool.name not in unannotated_writes:
            assert tool.annotations is not None, tool.name
            assert tool.annotations.readOnlyHint is True, tool.name
            assert tool.annotations.destructiveHint is not True, tool.name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "vendor_method", "extra_arguments", "succeeded"),
    [
        ("google_sheets_read_range", "get", {}, False),
        ("google_sheets_append_rows", "append", {"values": [["value"]]}, True),
    ],
)
async def test_sheets_error_envelope_uses_wire_annotations(
    monkeypatch, tool_name, vendor_method, extra_arguments, succeeded
):
    service = Mock()
    values_resource = service.spreadsheets.return_value.values.return_value
    request = getattr(values_resource, vendor_method).return_value
    request.execute.side_effect = TimeoutError("Google Sheets request timed out")
    monkeypatch.setattr(google_sheets, "get_sheets_service", lambda: service)

    @asynccontextmanager
    async def _session(_connection):
        async with create_connected_server_and_client_session(
            google_sheets.mcp._mcp_server
        ) as session:
            yield session

    connection = {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", google_sheets.__name__],
    }
    monkeypatch.setattr(mcp_adapter_module, "create_session", _session)
    async with _session(connection) as session:
        tools = await load_mcp_tools(session)
    (tool,) = [tool for tool in tools if tool.name == tool_name]
    adapter = MCPToolAdapter(
        mcp_tool=tool,
        connection=connection,
        name_prefix="mcp_google_sheets_",
        source_server="google_sheets",
        raw_annotations=raw_annotations_for(tool),
    )

    result = await adapter.run_json_async(
        {
            "spreadsheet_id": "spreadsheet-id",
            "range_name": "Sheet1!A1",
            **extra_arguments,
        }
    )

    request.execute.assert_called_once_with()
    assert result["is_error"] is False
    assert json.loads(result["content"][0]["text"])["status"] == "error"
    assert tool_result_succeeded(result) is succeeded
