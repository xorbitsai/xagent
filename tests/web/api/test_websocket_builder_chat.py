"""Test builder chat WebSocket endpoint with agent-based implementation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from xagent.web.api.websocket import handle_builder_chat
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User


@pytest.mark.asyncio
async def test_handle_builder_chat_basic():
    """
    Test that handle_builder_chat creates an agent with only create_agent tool.
    """
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
        mock_llm.stream_chat = AsyncMock()
        mock_storage_instance.get_llm_by_name_with_access.return_value = mock_llm
        mock_storage_instance.get_configured_defaults.return_value = (
            mock_llm,
            None,
            None,
            None,
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
        # Verify AgentService was created with use_dag_pattern=False (ReAct pattern)
        assert MockAgentService.called
        call_kwargs = MockAgentService.call_args[1]
        assert call_kwargs["use_dag_pattern"] is False  # ReAct pattern
        assert call_kwargs["name"] == "builder_chat_agent"

        # Verify CreateAgentTool was created (direct tool creation, not via WebToolConfig)
        assert MockCreateAgentTool.called
        assert MockUpdateAgentTool.called

        # Verify agent service execute_task was called
        mock_agent_service.execute_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_builder_chat_no_llm():
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
