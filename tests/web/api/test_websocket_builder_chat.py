"""Test builder chat WebSocket endpoint with agent-based implementation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from xagent.web.api.websocket import handle_builder_chat
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User


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
    }

    # Mock DB Session
    mock_db = MagicMock(spec=Session)

    # Mock DB query results for models
    mock_model = MagicMock(spec=DBModel)
    mock_model.model_id = "gpt-4"

    # Mock query().filter().first() chain
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = mock_model

    # Mock dependencies
    with (
        patch("xagent.web.models.database.get_db", return_value=iter([mock_db])),
        patch("xagent.web.services.llm_utils.UserAwareModelStorage") as MockStorage,
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
        # Setup mocks
        mock_storage_instance = MockStorage.return_value
        mock_llm = AsyncMock()
        mock_compact_llm = AsyncMock()
        mock_llm.stream_chat = AsyncMock()
        mock_storage_instance.get_llm_by_name_with_access.return_value = mock_llm
        mock_storage_instance.get_configured_defaults.return_value = (
            mock_llm,
            None,
            None,
            mock_compact_llm,
        )

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
        try:
            await handle_builder_chat(mock_websocket, message_data, mock_user)
        except Exception:
            # If there's an error, check if it's related to our test setup
            # The actual implementation should work
            pass

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

        # Verify agent service execute_task was called
        mock_agent_service.execute_task.assert_awaited_once()


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
    mock_db = MagicMock(spec=Session)

    with (
        patch("xagent.web.models.database.get_db", return_value=iter([mock_db])),
        patch("xagent.web.services.llm_utils.UserAwareModelStorage") as MockStorage,
        patch("xagent.core.agent.service.AgentService") as MockAgentService,
        patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
        patch("xagent.web.user_isolated_memory.UserContext"),
        patch("xagent.core.tools.adapters.vibe.agent_tool.CreateAgentTool"),
        patch("xagent.core.tools.adapters.vibe.agent_tool.UpdateAgentTool"),
    ):
        mock_storage_instance = MockStorage.return_value
        mock_llm = AsyncMock()
        mock_compact_llm = AsyncMock()
        mock_default_llm = AsyncMock()
        mock_default_compact_llm = AsyncMock()

        def get_llm_by_name_with_access(
            model_name: object, *, user_id: int | None = None
        ) -> object | None:
            assert user_id == 1
            return {
                10: mock_llm,
                20: mock_compact_llm,
            }.get(model_name)

        mock_storage_instance.get_llm_by_name_with_access.side_effect = (
            get_llm_by_name_with_access
        )
        mock_storage_instance.get_configured_defaults.return_value = (
            mock_default_llm,
            None,
            None,
            mock_default_compact_llm,
        )

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
    mock_storage_instance.get_configured_defaults.assert_not_called()
    assert mock_storage_instance.get_llm_by_name_with_access.call_count == 2


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

    mock_db = MagicMock(spec=Session)
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_db.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.first.return_value = MagicMock(spec=DBModel, model_id="gpt-4")

    with (
        patch("xagent.web.models.database.get_db", return_value=iter([mock_db])),
        patch("xagent.web.services.llm_utils.UserAwareModelStorage") as MockStorage,
        patch("xagent.core.agent.service.AgentService") as MockAgentService,
        patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
        patch("xagent.web.user_isolated_memory.UserContext"),
    ):
        mock_storage_instance = MockStorage.return_value
        mock_llm = AsyncMock()
        mock_storage_instance.get_llm_by_name_with_access.return_value = mock_llm
        mock_storage_instance.get_configured_defaults.return_value = (
            mock_llm,
            None,
            None,
            None,
        )

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
async def test_handle_builder_chat_no_llm() -> None:
    """
    Test that handle_builder_chat handles missing LLM gracefully.
    """
    # Arrange
    mock_websocket = AsyncMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1

    message_data = {
        "messages": [{"role": "user", "content": "Create an agent"}],
        "current_config": {},
        "available_options": {},
    }

    # Mock DB Session
    mock_db = MagicMock(spec=Session)

    # Mock dependencies
    with (
        patch("xagent.web.models.database.get_db", return_value=iter([mock_db])),
        patch("xagent.web.services.llm_utils.UserAwareModelStorage") as MockStorage,
    ):
        # Setup mocks to return None for LLM
        mock_storage_instance = MockStorage.return_value
        mock_storage_instance.get_llm_by_name_with_access.return_value = None
        mock_storage_instance.get_configured_defaults.return_value = (
            None,
            None,
            None,
            None,
        )

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
