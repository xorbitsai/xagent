from __future__ import annotations

from typing import Any

import pytest

from xagent.core.tools.adapters.vibe import a2a_agent_tool
from xagent.core.tools.adapters.vibe.a2a_agent_tool import (
    A2AAgentTool,
    create_a2a_agent_tools,
)
from xagent.core.tools.adapters.vibe.config import ToolConfig


class _FakeResponse:
    def __init__(self, payload: Any):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeAsyncClient:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(("GET", url, kwargs))
        if url == "https://remote.example/.well-known/agent-card.json":
            return _FakeResponse(
                {
                    "name": "Remote Agent",
                    "description": "Remote A2A worker",
                    "supportedInterfaces": [
                        {
                            "url": "https://remote.example/a2a",
                            "protocolBinding": "HTTP+JSON",
                            "protocolVersion": "1.0",
                        }
                    ],
                }
            )
        if url == "https://remote.example/a2a/tasks/task-1":
            return _FakeResponse(
                {
                    "id": "task-1",
                    "contextId": "ctx-1",
                    "status": {
                        "state": "TASK_STATE_COMPLETED",
                        "message": {
                            "role": "ROLE_AGENT",
                            "parts": [{"text": "done"}],
                        },
                    },
                    "artifacts": [
                        {
                            "artifactId": "artifact-1",
                            "parts": [{"text": "artifact text"}],
                        }
                    ],
                }
            )
        raise AssertionError(f"Unexpected GET {url}")

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(("POST", url, kwargs))
        assert url == "https://remote.example/a2a/message:send"
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        assert kwargs["headers"]["A2A-Version"] == "1.0"
        assert kwargs["json"]["message"]["parts"] == [{"text": "do work"}]
        assert kwargs["json"]["configuration"]["returnImmediately"] is True
        return _FakeResponse(
            {
                "task": {
                    "id": "task-1",
                    "contextId": "ctx-1",
                    "status": {"state": "TASK_STATE_WORKING"},
                }
            }
        )


class _InterruptedAsyncClient(_FakeAsyncClient):
    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return _FakeResponse(
            {
                "task": {
                    "id": "task-input",
                    "contextId": "ctx-input",
                    "status": {
                        "state": "TASK_STATE_INPUT_REQUIRED",
                        "message": {
                            "role": "ROLE_AGENT",
                            "parts": [{"text": "Which region?"}],
                        },
                    },
                }
            }
        )

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        raise AssertionError(f"Interrupted tasks must not be polled: {url}")


class _DirectMessageAsyncClient(_FakeAsyncClient):
    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return _FakeResponse(
            {
                "message": {
                    "messageId": "response-1",
                    "role": "ROLE_AGENT",
                    "parts": [{"text": "direct answer"}],
                }
            }
        )


class _WorkingAsyncClient(_FakeAsyncClient):
    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "task": {
                    "id": "task-working",
                    "contextId": "ctx-working",
                    "status": {"state": "TASK_STATE_WORKING"},
                }
            }
        )


@pytest.mark.asyncio
async def test_a2a_agent_tool_fetches_card_sends_and_polls(monkeypatch) -> None:
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(a2a_agent_tool.httpx, "AsyncClient", _FakeAsyncClient)

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(a2a_agent_tool, "sleep", _no_sleep)
    tool = A2AAgentTool(
        name="Remote Agent",
        agent_card_url="https://remote.example/.well-known/agent-card.json",
        auth_token="secret",
    )

    result = await tool.run_json_async({"task": "do work"})

    assert result["success"] is True
    assert result["task_id"] == "task-1"
    assert result["context_id"] == "ctx-1"
    assert result["state"] == "TASK_STATE_COMPLETED"
    assert result["response"] == "done\n\nartifact text"
    assert [call[0] for call in _FakeAsyncClient.calls] == ["GET", "POST", "GET"]


@pytest.mark.asyncio
async def test_create_a2a_agent_tools_from_tool_config() -> None:
    config = ToolConfig(
        {
            "a2a_agent_configs": [
                {
                    "name": "Remote Researcher",
                    "description": "Use for remote research tasks.",
                    "endpoint_url": "https://remote.example/a2a",
                    "headers": {"X-Remote": "1"},
                }
            ]
        }
    )

    tools = await create_a2a_agent_tools(config)

    assert len(tools) == 1
    assert tools[0].name == "a2a_remote_researcher"
    assert tools[0].metadata.category == "agent"
    assert tools[0].description == "Use for remote research tasks."


@pytest.mark.asyncio
async def test_a2a_agent_tool_returns_interrupted_task_without_polling(
    monkeypatch,
) -> None:
    _InterruptedAsyncClient.calls = []
    monkeypatch.setattr(a2a_agent_tool.httpx, "AsyncClient", _InterruptedAsyncClient)
    tool = A2AAgentTool(
        name="Remote Agent",
        endpoint_url="https://remote.example/a2a",
    )

    result = await tool.run_json_async({"task": "research", "task_id": "task-1"})

    assert result["success"] is False
    assert result["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert result["task_id"] == "task-input"
    assert result["response"] == "Which region?"
    assert result["error"] == "Which region?"
    sent_message = _InterruptedAsyncClient.calls[0][2]["json"]["message"]
    assert sent_message["taskId"] == "task-1"


@pytest.mark.asyncio
async def test_a2a_agent_tool_extracts_direct_message_response(monkeypatch) -> None:
    _DirectMessageAsyncClient.calls = []
    monkeypatch.setattr(a2a_agent_tool.httpx, "AsyncClient", _DirectMessageAsyncClient)
    tool = A2AAgentTool(
        name="Remote Agent",
        endpoint_url="https://remote.example/a2a",
    )

    result = await tool.run_json_async({"task": "quick answer"})

    assert result["success"] is True
    assert result["response"] == "direct answer"
    assert result["task_id"] is None


def test_a2a_agent_tool_rejects_url_with_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="embedded credentials"):
        A2AAgentTool(
            name="Remote Agent",
            endpoint_url="https://user:secret@remote.example/a2a",
        )


@pytest.mark.asyncio
async def test_a2a_agent_tool_reports_timeout_as_failure(monkeypatch) -> None:
    monkeypatch.setattr(a2a_agent_tool.httpx, "AsyncClient", _WorkingAsyncClient)

    async def _time_out(_self: Any, **kwargs: Any) -> tuple[dict[str, Any], Any]:
        raise TimeoutError

    monkeypatch.setattr(A2AAgentTool, "_poll_task_until_terminal", _time_out)
    tool = A2AAgentTool(
        name="Remote Agent",
        endpoint_url="https://remote.example/a2a",
        timeout_seconds=1,
    )

    result = await tool.run_json_async({"task": "slow task"})

    assert result["success"] is False
    assert result["error"] == "A2A call timed out after 1s."
