from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from xagent.web.api.websocket import handle_build_preview_execution
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User


@pytest.mark.asyncio
async def test_handle_build_preview_execution_empty_tool_categories():
    """
    Test that handle_build_preview_execution does not raise UnboundLocalError
    when tool_categories is empty.
    """
    # Arrange
    mock_websocket = AsyncMock()
    mock_user = MagicMock(spec=User)
    mock_user.id = 1
    mock_user.is_admin = False

    message_data = {
        "instructions": "test instructions",
        "execution_mode": "graph",
        "models": {
            "general": 1,
        },
        "tool_categories": [],  # Empty list to trigger the potential issue
        "message": "test message",
    }

    # Mock DB Session
    mock_db = MagicMock(spec=Session)

    # Mock DB query results for models
    mock_model = MagicMock(spec=DBModel)
    mock_model.model_id = "test-model-id"

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
        patch("xagent.web.api.websocket.WebToolConfig") as MockWebToolConfig,
        patch("xagent.core.agent.trace.Tracer"),
        patch("xagent.core.memory.in_memory.InMemoryMemoryStore"),
    ):
        mock_storage_instance = MockStorage.return_value
        mock_storage_instance.get_llm_by_name_with_access.return_value = MagicMock()

        mock_agent_service = MockAgentService.return_value
        mock_agent_service.execute_task = AsyncMock(
            return_value={"output": "success", "status": "completed"}
        )

        # Act
        try:
            await handle_build_preview_execution(
                mock_websocket, message_data, mock_user
            )
        except UnboundLocalError as e:
            pytest.fail(f"UnboundLocalError raised: {e}")
        except Exception as e:
            # If other errors occur, we should check if they are related to our test setup
            # but getting past the UnboundLocalError is the main goal.
            # However, for a good test, it should run successfully.
            # Let's see if we can make it run successfully.
            # If we mock everything, it should be fine.
            pytest.fail(f"Unexpected error raised: {e}")

        # Assert
        # Verify WebToolConfig was called (this is where MinimalRequest is used)
        assert MockWebToolConfig.called


@pytest.mark.asyncio
async def test_websocket_build_preview_endpoint_clear_context():
    """
    Test that websocket_build_preview_endpoint handles 'clear_context' message correctly.
    """
    import json
    from unittest.mock import MagicMock, patch

    from xagent.web.api.websocket import websocket_build_preview_endpoint

    mock_websocket = AsyncMock()
    # Setup websocket state
    mock_websocket.state = MagicMock()
    mock_memory = MagicMock()
    mock_websocket.state.preview_memory = mock_memory
    mock_websocket.state.preview_history = [{"role": "user", "content": "hello"}]

    # Mock user
    mock_user = MagicMock(spec=User)
    mock_user.id = 1

    # Setup sequence of events: receive 'clear_context', then raise WebSocketDisconnect to exit loop
    from fastapi import WebSocketDisconnect

    mock_websocket.receive_text.side_effect = [
        json.dumps({"type": "clear_context"}),
        WebSocketDisconnect(),
    ]

    with patch(
        "xagent.web.api.websocket.get_authenticated_user", return_value=mock_user
    ):
        await websocket_build_preview_endpoint(mock_websocket)

    # Verify accept was called
    mock_websocket.accept.assert_called_once()

    # Verify memory was cleared
    mock_memory.clear.assert_called_once()

    # Verify history was cleared
    assert mock_websocket.state.preview_history == []

    # Verify a response was sent
    send_text_calls = mock_websocket.send_text.call_args_list
    assert len(send_text_calls) == 1
    sent_data = json.loads(send_text_calls[0][0][0])
    assert sent_data["type"] == "context_cleared"
    assert "timestamp" in sent_data


@pytest.mark.asyncio
async def test_websocket_build_preview_endpoint_pause_resume():
    """
    Test that websocket_build_preview_endpoint handles 'pause' and 'resume' messages correctly.
    """
    import json
    from unittest.mock import MagicMock, patch

    from xagent.web.api.websocket import websocket_build_preview_endpoint

    mock_websocket = AsyncMock()
    mock_websocket.state = MagicMock()

    mock_agent_service = AsyncMock()
    mock_websocket.state.preview_agent_service = mock_agent_service

    mock_user = MagicMock(spec=User)
    mock_user.id = 1

    from fastapi import WebSocketDisconnect

    mock_websocket.receive_text.side_effect = [
        json.dumps({"type": "pause"}),
        json.dumps({"type": "resume"}),
        WebSocketDisconnect(),
    ]

    with patch(
        "xagent.web.api.websocket.get_authenticated_user", return_value=mock_user
    ):
        await websocket_build_preview_endpoint(mock_websocket)

    # Verify pause and resume were called
    mock_agent_service.pause_execution.assert_awaited_once()
    mock_agent_service.resume_execution.assert_awaited_once()

    # Verify responses were sent
    send_text_calls = mock_websocket.send_text.call_args_list
    assert len(send_text_calls) == 2

    pause_data = json.loads(send_text_calls[0][0][0])
    assert pause_data["type"] == "task_paused"
    assert "timestamp" in pause_data

    resume_data = json.loads(send_text_calls[1][0][0])
    assert resume_data["type"] == "task_resumed"
    assert "timestamp" in resume_data
