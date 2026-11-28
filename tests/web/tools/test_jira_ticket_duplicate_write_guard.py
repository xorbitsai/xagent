"""Exercise Jira's create outcome through the MCP wire and ReAct write guard."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from requests import Response

from xagent.core.agent import ExecutionContext, ReActPattern
from xagent.core.agent.pattern.react.duplicate_write_guard import (
    DUPLICATE_WRITE_SUPPRESSED_KEY,
)
from xagent.core.agent.result import tool_result_succeeded
from xagent.core.tools.adapters.vibe import mcp_adapter
from xagent.core.tools.core.mcp.tools import _list_all_tools
from xagent.web.tools.mcp import jira_ticket


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "can_retry"),
    [
        pytest.param(429, "{}", True, id="definite-rejection"),
        pytest.param(503, "{}", False, id="uncertain-server-error"),
        pytest.param(201, "<html>", False, id="non-json-create-response"),
        pytest.param(201, "{}", False, id="missing-issue-key"),
    ],
)
async def test_same_turn_jira_retry_depends_on_create_outcome(
    monkeypatch: pytest.MonkeyPatch, status: int, body: str, can_retry: bool
) -> None:
    monkeypatch.setenv("JIRA_TICKET_SITE", "acme")
    monkeypatch.setenv("JIRA_TICKET_EMAIL", "intake@example.com")
    monkeypatch.setenv("JIRA_TICKET_API_TOKEN", "mock-token")
    monkeypatch.setenv("JIRA_TICKET_PROJECT_KEY", "SUP")
    monkeypatch.setattr(jira_ticket, "reject_private_network_host", Mock())

    first_response = Response()
    first_response.status_code = status
    first_response._content = body.encode()
    created_response = Response()
    created_response.status_code = 201
    created_response._content = b'{"key":"SUP-7"}'
    request = Mock(side_effect=[first_response, created_response])
    monkeypatch.setattr(jira_ticket._session, "request", request)

    @asynccontextmanager
    async def in_memory_session(_connection):
        async with create_connected_server_and_client_session(
            jira_ticket.mcp._mcp_server
        ) as session:
            yield session

    monkeypatch.setattr(mcp_adapter, "create_session", in_memory_session)
    connection = {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "xagent.web.tools.mcp.jira_ticket"],
    }
    async with in_memory_session(connection) as session:
        (wire_tool,) = await _list_all_tools(session)
    tool = mcp_adapter._build_mcp_tool_adapter("jira_ticket", connection, wire_tool)
    arguments = json.dumps(
        {"summary": "Login broken", "description": "Cannot sign in."}
    )
    responses = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {"name": tool.metadata.name, "arguments": arguments},
                }
            ],
            "done": False,
        }
        for call_id in ("first", "retry")
    ]
    responses.append({"content": "Done.", "done": True})
    llm = SimpleNamespace(chat=AsyncMock(side_effect=responses))
    context = ExecutionContext()
    context.add_user_message("File a Jira ticket", metadata={"turn_id": "turn-1"})

    await ReActPattern(max_iterations=4).run(context=context, tools=[tool], llm=llm)

    first, second = [
        msg.metadata["raw_result"] for msg in context.messages if msg.role == "tool"
    ]
    assert first["is_error"] is can_retry
    assert tool_result_succeeded(first) is (not can_retry)
    if can_retry:
        assert request.call_count == 2
        assert "rate limit" in first["content"][0]["text"]
        assert DUPLICATE_WRITE_SUPPRESSED_KEY not in second
        assert json.loads(second["content"][0]["text"])["key"] == "SUP-7"
    else:
        assert request.call_count == 1
        assert second[DUPLICATE_WRITE_SUPPRESSED_KEY] is True
        assert second["suppressed_duplicate_of"] == "first"
        assert second["result"] == first
        assert "The ticket may already have been created" in first["content"][0]["text"]
