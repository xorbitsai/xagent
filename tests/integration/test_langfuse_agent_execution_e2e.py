"""Integration tests for Langfuse tracing across real agent execution paths."""

from __future__ import annotations

import pytest

from tests.utils.langfuse_execution_fakes import (
    CalculatorTool,
    DeterministicReActLLM,
    DeterministicSingleCallLLM,
    DummyMemoryStore,
    FailingTool,
    FakeLangfuseClient,
    assert_handler_closed,
    find_trace_update,
    get_langfuse_handler,
    observation_names,
    observations_by_type,
    update_data_values,
)
from xagent.core.agent.service import AgentService
from xagent.core.tracing import create_agent_tracer


@pytest.fixture
def fake_langfuse_client(
    mocker, monkeypatch, langfuse_client_reset
) -> FakeLangfuseClient:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    client = FakeLangfuseClient()
    mocker.patch("xagent.core.tracing.langfuse.client.Langfuse", return_value=client)
    return client


@pytest.mark.asyncio
async def test_single_call_tool_success_exports_complete_langfuse_trace(
    fake_langfuse_client: FakeLangfuseClient,
) -> None:
    tracer = create_agent_tracer(
        task_id="single-call-success",
        user_id=7,
        trace_name="single-call-trace",
        session_id="session-single-call",
        tags=["xagent", "test"],
        metadata={"source": "test-suite"},
    )
    agent_service = AgentService(
        name="single-call-agent",
        id="single-call-agent",
        llm=DeterministicSingleCallLLM(),
        tools=[CalculatorTool()],
        memory=DummyMemoryStore(),
        pattern="single_call",
        tracer=tracer,
        enable_workspace=True,
        task_id="single-call-success",
    )
    agent_service.set_allowed_skills([])

    result = await agent_service.execute_task(
        task="calculate 2 + 2", task_id="single-call-success"
    )

    assert result["success"] is True
    handler = get_langfuse_handler(tracer.handlers)

    agent_observations = observations_by_type(fake_langfuse_client, "agent")
    assert len(agent_observations) == 1
    root = agent_observations[0]
    assert root.ended is True
    assert root.end_count == 1
    initial_trace_update = find_trace_update(root, "user_id", "7")
    assert initial_trace_update["session_id"] == "session-single-call"
    assert initial_trace_update["tags"] == ["xagent", "test"]

    generation_observations = observations_by_type(fake_langfuse_client, "generation")
    assert len(generation_observations) == 2
    assert all(observation.ended for observation in generation_observations)
    assert all(observation.end_count == 1 for observation in generation_observations)

    tool_observations = observations_by_type(fake_langfuse_client, "tool")
    assert len(tool_observations) == 1
    assert tool_observations[0].ended is True
    assert tool_observations[0].end_count == 1
    assert update_data_values(tool_observations[0], "success") == [True]

    assert_handler_closed(handler)


@pytest.mark.asyncio
async def test_single_call_tool_failure_closes_open_tool_observation_as_error(
    fake_langfuse_client: FakeLangfuseClient,
) -> None:
    tracer = create_agent_tracer(
        task_id="single-call-failure",
        user_id=7,
        trace_name="single-call-failure",
        session_id="session-single-call-failure",
    )
    agent_service = AgentService(
        name="single-call-failing-agent",
        id="single-call-failing-agent",
        llm=DeterministicSingleCallLLM(
            tool_name="failing_tool", final_answer="unreachable"
        ),
        tools=[FailingTool()],
        memory=DummyMemoryStore(),
        pattern="single_call",
        tracer=tracer,
        enable_workspace=True,
        task_id="single-call-failure",
    )
    agent_service.set_allowed_skills([])

    result = await agent_service.execute_task(
        task="trigger failing tool", task_id="single-call-failure"
    )

    assert result["success"] is True
    assert result["output"] == "unreachable"
    handler = get_langfuse_handler(tracer.handlers)

    tool_observations = observations_by_type(fake_langfuse_client, "tool")
    assert len(tool_observations) == 1
    tool_observation = tool_observations[0]
    assert tool_observation.ended is True
    assert tool_observation.end_count == 1
    error_update = tool_observation.updates[-1]
    assert error_update["level"] == "ERROR"
    assert "boom" in str(error_update["status_message"])

    root = observations_by_type(fake_langfuse_client, "agent")[0]
    assert root.ended is True
    assert_handler_closed(handler)


@pytest.mark.asyncio
async def test_react_tool_success_exports_langfuse_trace(
    fake_langfuse_client: FakeLangfuseClient,
) -> None:
    tracer = create_agent_tracer(
        task_id="react-success",
        user_id=11,
        trace_name="react-trace",
        session_id="session-react",
    )
    agent_service = AgentService(
        name="react-agent",
        id="react-agent",
        llm=DeterministicReActLLM(),
        tools=[CalculatorTool()],
        memory=DummyMemoryStore(),
        pattern="react",
        tracer=tracer,
        enable_workspace=True,
        task_id="react-success",
    )
    agent_service.set_allowed_skills([])

    result = await agent_service.execute_task(
        "calculate 2 + 2", task_id="react-success"
    )

    assert result["success"] is True
    handler = get_langfuse_handler(tracer.handlers)

    assert len(observations_by_type(fake_langfuse_client, "agent")) == 1
    assert len(observations_by_type(fake_langfuse_client, "generation")) >= 1
    assert len(observations_by_type(fake_langfuse_client, "tool")) >= 1
    assert any(
        "calculator" in name for name in observation_names(fake_langfuse_client, "tool")
    )
    assert observations_by_type(fake_langfuse_client, "agent")[0].ended is True
    assert_handler_closed(handler)
