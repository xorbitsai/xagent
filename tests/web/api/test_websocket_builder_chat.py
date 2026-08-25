"""Test builder chat WebSocket endpoint with agent-based implementation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect

from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import (
    handle_builder_chat,
    websocket_builder_chat_endpoint,
)
from xagent.web.models.user import User
from xagent.web.services.builder_chat_runtime import BuilderChatRuntimeInputs


@pytest.mark.asyncio
async def test_builder_endpoint_drains_replaced_and_disconnected_chat_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement never overlaps cleanup owned by the previous handler."""

    first_started = asyncio.Event()
    first_cleanup_started = asyncio.Event()
    release_first_cleanup = asyncio.Event()
    second_started = asyncio.Event()
    second_cleanup_started = asyncio.Event()
    release_second_cleanup = asyncio.Event()
    allow_disconnect = asyncio.Event()
    handler_calls = 0

    class FakeWebSocket:
        def __init__(self) -> None:
            self.state = SimpleNamespace()
            self.receive_count = 0

        async def accept(self) -> None:
            return None

        async def close(self, **_kwargs: object) -> None:
            return None

        async def receive_text(self) -> str:
            self.receive_count += 1
            if self.receive_count == 1:
                return '{"messages":[{"role":"user","content":"first"}]}'
            if self.receive_count == 2:
                await first_started.wait()
                return '{"messages":[{"role":"user","content":"second"}]}'
            await allow_disconnect.wait()
            raise WebSocketDisconnect()

    async def controlled_handler(
        _websocket: object,
        _message_data: dict,
        _user: object,
    ) -> None:
        nonlocal handler_calls
        handler_calls += 1
        if handler_calls == 1:
            first_started.set()
            try:
                await asyncio.Future()
            finally:
                first_cleanup_started.set()
                await release_first_cleanup.wait()
        else:
            second_started.set()
            try:
                await asyncio.Future()
            finally:
                second_cleanup_started.set()
                await release_second_cleanup.wait()

    principal = SimpleNamespace(id=1, is_admin=False)
    monkeypatch.setattr(
        websocket_api,
        "get_authenticated_user",
        AsyncMock(return_value=principal),
    )
    monkeypatch.setattr(websocket_api, "handle_builder_chat", controlled_handler)

    endpoint_task = asyncio.create_task(
        websocket_builder_chat_endpoint(FakeWebSocket(), token="token")  # type: ignore[arg-type]
    )
    try:
        await first_cleanup_started.wait()
        await asyncio.sleep(0)
        assert not second_started.is_set()

        release_first_cleanup.set()
        await second_started.wait()
        allow_disconnect.set()
        await second_cleanup_started.wait()
        await asyncio.sleep(0)
        assert not endpoint_task.done()

        release_second_cleanup.set()
        await endpoint_task
    finally:
        release_first_cleanup.set()
        release_second_cleanup.set()
        allow_disconnect.set()
        if not endpoint_task.done():
            endpoint_task.cancel()
        await asyncio.gather(endpoint_task, return_exceptions=True)
    assert handler_calls == 2


@pytest.mark.asyncio
async def test_handle_builder_chat_basic() -> None:
    """Test that handle_builder_chat creates an agent with only create_agent tool."""
    # Arrange
    mock_websocket = AsyncMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False

    message_data = {
        "messages": [
            {
                "role": "user",
                "content": "Create an agent for data analysis",
            }
        ],
        "current_config": {
            "name": "TestAgent",
            "description": "A test agent",
        },
        "available_options": {
            "models": [{"id": 1, "name": "gpt-4"}],
            "knowledgeBases": [],
            "skills": [],
            "toolCategories": [],
        },
        "files": [
            {"file_id": "owned-file"},
            {"file_id": "another-user-file"},
        ],
    }

    mock_llm = AsyncMock()
    mock_compact_llm = AsyncMock()
    runtime_loader = AsyncMock(
        return_value=BuilderChatRuntimeInputs(
            authorized_file_ids=("owned-file",),
            llm=mock_llm,
            compact_llm=mock_compact_llm,
        )
    )

    # Mock dependencies
    with (
        patch(
            "xagent.web.services.builder_chat_runtime.load_builder_chat_runtime_inputs",
            runtime_loader,
        ),
        patch("xagent.web.api.websocket.get_session_local", return_value=MagicMock()),
        patch("xagent.core.agent.service.AgentService") as MockAgentService,
        patch("xagent.core.agent.trace.Tracer"),
        patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
        patch("xagent.web.user_isolated_memory.UserContext"),
        patch(
            "xagent.core.tools.adapters.vibe.agent_tool.CreateAgentTool"
        ) as MockCreateAgentTool,
        patch(
            "xagent.core.tools.adapters.vibe.agent_tool.UpdateAgentTool"
        ) as MockUpdateAgentTool,
    ):
        mock_llm.stream_chat = AsyncMock()

        # Mock agent service
        mock_agent_service = MockAgentService.return_value
        mock_agent_service.execute_task = AsyncMock(
            return_value={"output": "Agent created successfully", "status": "completed"}
        )

        # Mock websocket state
        mock_websocket.state = MagicMock()
        mock_memory = MagicMock()
        mock_websocket.state.builder_memory = mock_memory
        # Don't set builder_task_id, so the function will create a new one
        del mock_websocket.state.builder_task_id
        # Don't set builder_agent_service, so the function will create a new one
        del mock_websocket.state.builder_agent_service

        # Act
        await handle_builder_chat(mock_websocket, message_data, mock_user)

        # Assert
        # Verify AgentService was created with v2 ReAct so builder chat can use
        # native ask_user_question/send_message control tools without Auto's
        # extra pattern-selection tool calls.
        assert MockAgentService.called
        call_kwargs = MockAgentService.call_args[1]
        assert call_kwargs["pattern"] == "react"
        assert call_kwargs["name"] == "builder_chat_agent"
        assert call_kwargs["compact_llm"] is mock_compact_llm
        mock_agent_service.set_allowed_skills.assert_called_once_with(["agent-builder"])
        mock_agent_service.set_recovered_skill_context.assert_called_once()
        mock_agent_service.set_outbound_message_handler.assert_called_once()
        skill_context = mock_agent_service.set_recovered_skill_context.call_args.args[0]
        assert "## Available Skill: agent-builder" in skill_context

        # Verify CreateAgentTool was created (direct tool creation, not via WebToolConfig)
        assert MockCreateAgentTool.called
        assert MockUpdateAgentTool.called
        runtime_loader.assert_awaited_once_with(
            user_id=1,
            requested_file_ids=["owned-file", "another-user-file"],
            model_name=None,
            compact_model_name=None,
        )

        # Verify agent service execute_task was called
        mock_agent_service.execute_task.assert_awaited_once()
        executed_message = mock_agent_service.execute_task.await_args.kwargs["task"]
        assert "owned-file" in executed_message
        assert "another-user-file" not in executed_message


@pytest.mark.asyncio
async def test_handle_builder_chat_uses_payload_compact_model() -> None:
    mock_websocket = AsyncMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False

    message_data = {
        "messages": [
            {
                "role": "user",
                "content": "Create an agent for data analysis",
            }
        ],
        "models": {"general": 10, "compact": 20},
        "selectedSkills": [],
        "selectedKbs": [],
        "tool_categories": [],
        "executionMode": "balanced",
    }
    mock_llm = AsyncMock()
    mock_compact_llm = AsyncMock()
    runtime_loader = AsyncMock(
        return_value=BuilderChatRuntimeInputs(
            authorized_file_ids=(),
            llm=mock_llm,
            compact_llm=mock_compact_llm,
        )
    )

    with (
        patch(
            "xagent.web.services.builder_chat_runtime.load_builder_chat_runtime_inputs",
            runtime_loader,
        ),
        patch("xagent.web.api.websocket.get_session_local", return_value=MagicMock()),
        patch("xagent.core.agent.service.AgentService") as MockAgentService,
        patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
        patch("xagent.web.user_isolated_memory.UserContext"),
        patch("xagent.core.tools.adapters.vibe.agent_tool.CreateAgentTool"),
        patch("xagent.core.tools.adapters.vibe.agent_tool.UpdateAgentTool"),
    ):
        mock_agent_service = MockAgentService.return_value
        mock_agent_service.execute_task = AsyncMock(
            return_value={"output": "Agent created successfully", "status": "completed"}
        )

        mock_websocket.state = MagicMock()
        mock_websocket.state.builder_memory = MagicMock()
        del mock_websocket.state.builder_task_id
        del mock_websocket.state.builder_agent_service

        await handle_builder_chat(mock_websocket, message_data, mock_user)

    call_kwargs = MockAgentService.call_args[1]
    assert call_kwargs["llm"] is mock_llm
    assert call_kwargs["compact_llm"] is mock_compact_llm
    runtime_loader.assert_awaited_once_with(
        user_id=1,
        requested_file_ids=[],
        model_name=10,
        compact_model_name=20,
    )


@pytest.mark.asyncio
async def test_handle_builder_chat_waiting_for_user_sends_chat_response() -> None:
    """Builder chat should surface v2 ask_user_question as structured UI."""
    mock_websocket = AsyncMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False

    message_data = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Create an agent grounded to the connected KB Velvet Enterprise FAQ"
                ),
            }
        ],
        "models": {"general": 1},
        "selectedSkills": [],
        "selectedKbs": [],
        "tool_categories": [],
        "executionMode": "balanced",
    }

    mock_llm = AsyncMock()
    runtime_loader = AsyncMock(
        return_value=BuilderChatRuntimeInputs(
            authorized_file_ids=(),
            llm=mock_llm,
            compact_llm=None,
        )
    )

    with (
        patch(
            "xagent.web.services.builder_chat_runtime.load_builder_chat_runtime_inputs",
            runtime_loader,
        ),
        patch("xagent.web.api.websocket.get_session_local", return_value=MagicMock()),
        patch("xagent.core.agent.service.AgentService") as MockAgentService,
        patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
        patch("xagent.web.user_isolated_memory.UserContext"),
    ):
        mock_agent_service = MockAgentService.return_value
        mock_agent_service.execute_task = AsyncMock(
            return_value={
                "success": False,
                "status": "waiting_for_user",
                "message": "Choose a method to provide FAQ content:",
                "interactions": [
                    {
                        "type": "action_cards",
                        "field": "kb_source",
                        "label": "Choose a method",
                        "options": [
                            {
                                "label": "Upload FAQ Documents",
                                "value": "upload",
                                "action_type": "upload",
                            }
                        ],
                    }
                ],
            }
        )

        mock_websocket.state = MagicMock()
        mock_websocket.state.builder_memory = MagicMock()
        del mock_websocket.state.builder_task_id
        del mock_websocket.state.builder_agent_service

        await handle_builder_chat(mock_websocket, message_data, mock_user)

    sent_events = [
        json.loads(call.args[0]) for call in mock_websocket.send_text.call_args_list
    ]
    task_completed = next(
        event for event in sent_events if event.get("type") == "task_completed"
    )
    chat_response = task_completed["result"]["chat_response"]
    assert chat_response["message"] == "Choose a method to provide FAQ content:"
    assert chat_response["interactions"][0]["type"] == "action_cards"
    mock_agent_service.set_allowed_skills.assert_called_once_with(["agent-builder"])
    mock_agent_service.set_recovered_skill_context.assert_called_once()
    mock_agent_service.set_outbound_message_handler.assert_called_once()


@pytest.mark.asyncio
async def test_handle_builder_chat_applies_voice_preference() -> None:
    """The builder chat's own meta-assistant must honor the user's onboarding
    voice preference the same way a saved agent's real conversation does
    (see apply_user_voice's call sites in chat.py) - before this fix, only
    agents the user created got voice injection, not this platform-owned
    assistant itself."""
    mock_websocket = AsyncMock()
    mock_user = SimpleNamespace(id=1, is_admin=False, voice="concise")

    message_data = {
        "messages": [{"role": "user", "content": "Create an agent"}],
        "models": {"general": 1},
        "selectedSkills": [],
        "selectedKbs": [],
        "tool_categories": [],
        "executionMode": "balanced",
    }

    mock_llm = AsyncMock()
    runtime_loader = AsyncMock(
        return_value=BuilderChatRuntimeInputs(
            authorized_file_ids=(),
            llm=mock_llm,
            compact_llm=None,
        )
    )

    with (
        patch(
            "xagent.web.services.builder_chat_runtime.load_builder_chat_runtime_inputs",
            runtime_loader,
        ),
        patch("xagent.web.api.websocket.get_session_local", return_value=MagicMock()),
        patch("xagent.core.agent.service.AgentService") as MockAgentService,
        patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
        patch("xagent.web.user_isolated_memory.UserContext"),
    ):
        mock_agent_service = MockAgentService.return_value
        mock_agent_service.execute_task = AsyncMock(
            return_value={"output": "Agent created successfully", "status": "completed"}
        )

        mock_websocket.state = MagicMock()
        mock_websocket.state.builder_memory = MagicMock()
        del mock_websocket.state.builder_task_id
        del mock_websocket.state.builder_agent_service

        await handle_builder_chat(mock_websocket, message_data, mock_user)

    execution_context = mock_agent_service.execute_task.await_args.kwargs["context"]
    system_prompt = execution_context["system_prompt"]
    assert "\n\n## OUTPUT VOICE\n" in system_prompt
    assert "As short as possible" in system_prompt
    # The voice tone must not bleed into create_agent/update_agent's
    # persisted name/description/instructions arguments - only the final
    # reply, mirroring the equivalent scoping already required of the
    # Workforce Prompt Builder's identically-shaped system prompt.
    assert "persisted configuration" in system_prompt
    assert "create_agent/update_agent" in system_prompt


@pytest.mark.asyncio
async def test_handle_builder_chat_no_llm() -> None:
    """
    Test that handle_builder_chat handles missing LLM gracefully.
    """
    # Arrange
    mock_websocket = AsyncMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False

    message_data = {
        "messages": [{"role": "user", "content": "Create an agent"}],
        "current_config": {},
        "available_options": {},
    }

    runtime_loader = AsyncMock(
        return_value=BuilderChatRuntimeInputs(
            authorized_file_ids=(),
            llm=None,
            compact_llm=None,
        )
    )

    # Mock dependencies
    with (
        patch(
            "xagent.web.services.builder_chat_runtime.load_builder_chat_runtime_inputs",
            runtime_loader,
        ),
    ):
        # Act
        await handle_builder_chat(mock_websocket, message_data, mock_user)

        # Assert
        # Verify error message was sent
        mock_websocket.send_text.assert_called()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "error"
        assert "No LLM configured" in sent_data["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
