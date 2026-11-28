"""Durable FileRefs work through the Gmail and Slides MCP tool schemas."""

from contextlib import asynccontextmanager

import pytest
from mcp.types import CallToolResult, TextContent

from xagent.core.tools.adapters.vibe import mcp_adapter
from xagent.web.tools.mcp import gmail, google_slides

_FIRST_ID = "11111111-1111-4111-8111-111111111111"
_SECOND_ID = "22222222-2222-4222-8222-222222222222"
_FIRST_REF = f"file:{_FIRST_ID}"
_SECOND_REF = f"file:{_SECOND_ID}"


class _StagingWorkspace:
    def __init__(self, files: dict[str, str]):
        self.files = files
        self.discarded: list[str] = []

    def stage_file_for_external_upload(self, file_id: str) -> str:
        if file_id not in self.files:
            raise FileNotFoundError(file_id)
        return self.files[file_id]

    def discard_staged_external_upload(self, path: str) -> None:
        self.discarded.append(path)


async def _build_adapter(server_name, server, tool_name, workspace):
    tool = next(tool for tool in await server.list_tools() if tool.name == tool_name)
    return mcp_adapter._build_mcp_tool_adapter(
        server_name,
        {"transport": "stdio", "command": "python", "args": []},
        tool,
        workspace=workspace,
    )


@pytest.fixture
def calls(monkeypatch):
    calls = []

    class FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            calls.append((name, arguments))
            return CallToolResult(
                content=[TextContent(type="text", text=str(arguments))],
                isError=False,
            )

    @asynccontextmanager
    async def fake_create_session(_connection):
        yield FakeSession()

    monkeypatch.setattr(mcp_adapter, "create_session", fake_create_session)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("second_file_available", [True, False])
async def test_gmail_stages_attachments_across_messages(calls, second_file_available):
    first_path = "/task/temp/staged/report.pdf"
    second_path = "/task/temp/staged/chart.png"
    local_path = "/task/output/local.txt"
    workspace = _StagingWorkspace({_FIRST_ID: first_path})
    if second_file_available:
        workspace.files[_SECOND_ID] = second_path
    adapter = await _build_adapter("Gmail", gmail.mcp, "gmail_send_messages", workspace)
    first_message = {"body": _SECOND_REF, "attachments": [_FIRST_REF, local_path]}
    second_message = {"body": "See chart", "attachments": [_SECOND_REF]}
    messages = [first_message, {"body": "No attachment"}, second_message]

    result = await adapter.run_json_async({"messages": messages, "action": "send"})

    assert "messages[].attachments[]" in adapter.description
    assert first_message["attachments"] == [_FIRST_REF, local_path]
    assert second_message["attachments"] == [_SECOND_REF]
    if not second_file_available:
        assert result["is_error"] is True
        assert "messages[].attachments[]" in result["content"][0]["text"]
        assert calls == []
        assert workspace.discarded == [first_path]
        return

    assert result["is_error"] is False
    assert calls == [
        (
            "gmail_send_messages",
            {
                "messages": [
                    {"body": _SECOND_REF, "attachments": [first_path, local_path]},
                    {"body": "No attachment"},
                    {"body": "See chart", "attachments": [second_path]},
                ],
                "action": "send",
            },
        )
    ]
    assert workspace.discarded == [first_path, second_path]
    text = result["content"][0]["text"]
    assert _FIRST_REF in text and _SECOND_REF in text
    assert first_path not in text and second_path not in text


@pytest.mark.asyncio
async def test_slides_stages_a_pptx_file_ref_for_import(calls):
    path = "/task/temp/staged/deck.pptx"
    workspace = _StagingWorkspace({_FIRST_ID: path})
    adapter = await _build_adapter(
        "Google Slides", google_slides.mcp, "google_slides_import_pptx", workspace
    )

    result = await adapter.run_json_async({"file_path": _FIRST_REF, "title": "Review"})

    assert "file:<id>" in adapter.description
    assert result["is_error"] is False
    assert calls == [
        ("google_slides_import_pptx", {"file_path": path, "title": "Review"})
    ]
    assert workspace.discarded == [path]
