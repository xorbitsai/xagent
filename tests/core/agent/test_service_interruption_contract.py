"""The interruption reason reaches AgentService callers, where runs are settled."""

from typing import Any, cast

import pytest

from tests.core.agent.test_execution_adapter import FakeLLM, FakeTool
from tests.core.agent.test_runner import FakePattern
from xagent.core.agent.execution_adapter import AgentExecutionAdapter
from xagent.core.agent.service import AgentService
from xagent.core.model.chat.exceptions import LLMTimeoutError


class RaisingLLM(FakeLLM):
    def __init__(self, error: Exception) -> None:
        super().__init__([])
        self.error = error

    async def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise self.error


def _service(llm: Any, name: str) -> AgentService:
    service = AgentService(
        name=name,
        id=name,
        pattern="react",
        llm=cast(Any, llm),
        tools=cast(Any, [FakeTool()]),
        tool_config=None,
    )
    service.allowed_skills = []
    return service


@pytest.mark.asyncio
async def test_service_result_carries_reason_for_raised_provider_failure() -> None:
    service = _service(RaisingLLM(LLMTimeoutError("timed out")), "reason-raised")

    result = await service.execute_task("Answer", task_id="reason-raised-task")

    assert result["success"] is False
    assert result["interruption_reason"] == "llm_unavailable"
    assert type(result["interruption_reason"]) is str


@pytest.mark.asyncio
async def test_service_result_carries_reason_for_unsuccessful_pattern_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ReAct's own path to this status is covered in test_react; this pins that
    # the result shape crosses runner -> registry -> adapter -> service.
    monkeypatch.setattr(
        AgentExecutionAdapter,
        "_build_pattern",
        lambda self: (
            FakePattern({"success": False, "status": "invalid_tool_protocol"}),
            "agent_react",
        ),
    )
    service = _service(FakeLLM([]), "reason-result")

    result = await service.execute_task("Answer", task_id="reason-result-task")

    assert result["success"] is False
    assert result["interruption_reason"] == "model_output_invalid"
    assert type(result["interruption_reason"]) is str
