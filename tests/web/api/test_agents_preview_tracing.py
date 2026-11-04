from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.utils.langfuse_execution_fakes import (
    CalculatorTool,
    DeterministicReActLLM,
    FakeLangfuseClient,
    find_trace_update,
)
from tests.utils.mock_helpers import create_langfuse_mock
from xagent.core.tracing.langfuse.handler import LangfuseTraceHandler
from xagent.web.api.agents import AgentPreviewRequest, preview_agent
from xagent.web.models.user import User
from xagent.web.services.execution_budget import ExecutionBudgetDefaults


@pytest.mark.asyncio
async def test_preview_agent_injects_langfuse_tracer(
    mocker, monkeypatch, langfuse_client_reset
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    create_langfuse_mock(mocker)

    current_user = User()
    current_user.id = 7
    current_user.is_admin = False

    db = MagicMock()
    model_record = MagicMock()
    model_record.model_id = "test-model"
    db.query.return_value.filter.return_value.first.return_value = model_record

    request = AgentPreviewRequest(
        instructions="preview instructions",
        execution_mode="balanced",
        models={"general": 1},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        message="hello",
    )

    with (
        patch("xagent.web.api.agents.UserAwareModelStorage") as mock_storage_class,
        patch("xagent.web.api.agents.InMemoryMemoryStore"),
        patch("xagent.web.api.agents.AgentService") as mock_agent_service_class,
    ):
        mock_storage = MagicMock()
        mock_llm = MagicMock()
        mock_storage.get_llm_by_name_with_access.return_value = mock_llm
        mock_storage_class.return_value = mock_storage

        mock_agent_service = mock_agent_service_class.return_value
        mock_agent_service.execute_task = AsyncMock(
            return_value={"output": "preview response", "status": "completed"}
        )

        response = await preview_agent(
            request=request, current_user=current_user, db=db
        )

    assert response.response == "preview response"
    tracer = mock_agent_service_class.call_args.kwargs["tracer"]
    assert any(isinstance(handler, LangfuseTraceHandler) for handler in tracer.handlers)


@pytest.mark.asyncio
async def test_preview_agent_rest_executes_with_langfuse_trace(
    mocker, monkeypatch, langfuse_client_reset, tmp_path: Path
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")

    fake_client = FakeLangfuseClient()
    mocker.patch(
        "xagent.core.tracing.langfuse.client.Langfuse", return_value=fake_client
    )
    # This tracing test uses a mocked database. Keep the real policy provider
    # and runtime wiring, but stub its independent database read as well.
    load_budget = mocker.patch(
        "xagent.web.services.execution_budget._load_user_budget",
        return_value=(ExecutionBudgetDefaults(), {}),
    )

    current_user = User()
    current_user.id = 7
    current_user.is_admin = False

    db = MagicMock()
    model_record = MagicMock()
    model_record.model_id = "test-model"
    db.query.return_value.filter.return_value.first.return_value = model_record

    request = AgentPreviewRequest(
        instructions="preview instructions",
        execution_mode="flash",
        models={"general": 1},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        message="calculate 2 + 2",
    )

    with (
        patch("xagent.web.api.agents.UserAwareModelStorage") as mock_storage_class,
        patch(
            "xagent.web.api.agents.get_uploads_dir",
            return_value=tmp_path / "uploads",
        ),
        patch(
            "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
            new=AsyncMock(return_value=[CalculatorTool()]),
        ),
    ):
        mock_storage = MagicMock()
        mock_storage.get_llm_by_name_with_access.return_value = DeterministicReActLLM()
        mock_storage_class.return_value = mock_storage

        response = await preview_agent(
            request=request, current_user=current_user, db=db
        )

    assert response.status == "completed"
    assert response.response == "The result is 4"
    load_budget.assert_called_once_with(7)

    agent_observations = [
        observation
        for observation in fake_client.observations
        if observation.start_kwargs.get("as_type") == "agent"
    ]
    assert len(agent_observations) == 1
    root = agent_observations[0]
    assert root.ended is True
    root_trace_update = find_trace_update(root, "user_id", "7")
    assert root_trace_update["user_id"] == "7"
    assert root_trace_update["session_id"].startswith("preview_")
    assert root_trace_update["tags"] == [
        "xagent",
        "web",
        "preview",
        "agent-builder",
    ]

    root_metadata = root.start_kwargs["metadata"]
    assert root_metadata["source"] == "xagent-web"
    assert root_metadata["is_preview"] is True
    assert root_metadata["preview_transport"] == "rest"

    generation_observations = [
        observation
        for observation in fake_client.observations
        if observation.start_kwargs.get("as_type") == "generation"
    ]
    assert len(generation_observations) >= 1
    assert all(observation.ended for observation in generation_observations)

    tool_observations = [
        observation
        for observation in fake_client.observations
        if observation.start_kwargs.get("as_type") == "tool"
    ]
    assert len(tool_observations) == 1
    assert tool_observations[0].ended is True
