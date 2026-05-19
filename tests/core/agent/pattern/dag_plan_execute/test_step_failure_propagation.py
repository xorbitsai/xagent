"""Tests for honoring tool-reported failures in DAG step execution.

Background: a ReAct step whose final_answer was emitted with ``success=False``
(typically because a tool inside the step reported a failure that the LLM
chose to forward instead of swallow) used to still be marked
``StepStatus.COMPLETED``. The result_analyzer's summary only surfaced
content from "completed" steps and only looked at the ``output`` field, so
the failure disappeared and the final-answer LLM concluded the task had
succeeded. These tests cover the fix at both layers — plan_executor marks
the step FAILED, and result_analyzer surfaces the error in its summary.
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.agent.pattern.dag_plan_execute.models import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
)
from xagent.core.agent.pattern.dag_plan_execute.plan_executor import PlanExecutor
from xagent.core.agent.pattern.dag_plan_execute.result_analyzer import (
    ResultAnalyzer,
)


class _StubLLM:
    """Minimal LLM stub — these tests never reach the LLM call path."""

    @property
    def model_name(self) -> str:
        return "stub"


def _make_executor() -> PlanExecutor:
    tracer = MagicMock()
    tracer.trace_event = AsyncMock(return_value="trace-id")
    return PlanExecutor(
        llm=_StubLLM(),
        tracer=tracer,
        workspace=MagicMock(),
    )


def _make_step() -> PlanStep:
    return PlanStep(
        id="s1",
        name="run script",
        description="run a JS script",
        tool_names=["execute_javascript_code"],
        dependencies=[],
    )


def _make_tool_map() -> Dict[str, Any]:
    tool = MagicMock()
    tool.metadata = MagicMock()
    tool.metadata.name = "execute_javascript_code"
    return {"execute_javascript_code": tool}


def _patch_react_with_result(monkeypatch, result: Dict[str, Any]) -> None:
    """Make every ReAct step return `result` immediately."""

    async def fake_run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return result

    fake_pattern = MagicMock()
    fake_pattern.set_step_context = MagicMock()
    fake_pattern.run_with_context = AsyncMock(side_effect=fake_run)

    monkeypatch.setattr(
        "xagent.core.agent.pattern.react.ReActPattern",
        lambda *a, **k: fake_pattern,
    )


@pytest.mark.asyncio
async def test_execute_plan_marks_step_failed_when_react_reports_success_false(
    monkeypatch,
):
    """End-to-end: a ReAct step returning success=False ⇒ plan.step is FAILED.

    Goes through `execute_plan` rather than the internal method directly so
    the unconditional `step.status = COMPLETED` at the call site is exercised
    — the fix raises DAGStepError before that assignment runs.
    """
    executor = _make_executor()
    step = _make_step()
    plan = ExecutionPlan(id="p1", goal="g", steps=[step], created_at=datetime.now())

    _patch_react_with_result(
        monkeypatch,
        {
            "type": "final_answer",
            "content": "I tried to write a pptx but addTable rejected the rows.",
            "success": False,
            "error": "addTable: 'rows' should be an array of cells!",
        },
    )

    execution_results = await executor.execute_plan(plan, _make_tool_map())

    assert step.status == StepStatus.FAILED
    assert "addTable" in (step.error or "")
    # The exception path records the failure in execution_results so the
    # result_analyzer's summarizer sees the FAILED status downstream.
    failed_entries = [r for r in execution_results if r.get("status") == "failed"]
    assert failed_entries, "execute_plan must record the failure in execution_results"
    assert "addTable" in str(failed_entries[0])


@pytest.mark.asyncio
async def test_execute_plan_keeps_step_completed_when_react_reports_success_true(
    monkeypatch,
):
    """Sanity: existing success path through execute_plan unchanged."""
    executor = _make_executor()
    step = _make_step()
    plan = ExecutionPlan(id="p1", goal="g", steps=[step], created_at=datetime.now())

    _patch_react_with_result(
        monkeypatch,
        {
            "type": "final_answer",
            "content": "done",
            "success": True,
            "error": None,
        },
    )

    execution_results = await executor.execute_plan(plan, _make_tool_map())

    assert step.status == StepStatus.COMPLETED
    completed_entries = [r for r in execution_results if r.get("status") == "completed"]
    assert len(completed_entries) == 1


def test_summary_surfaces_failed_steps_with_error():
    """_summarize_execution_history must list failed steps and their error message."""
    analyzer = ResultAnalyzer(llm=_StubLLM(), tracer=MagicMock())
    history = [
        {
            "plan": {"steps": [{"id": "s1"}, {"id": "s2"}]},
            "results": [
                {
                    "step_name": "s1 read",
                    "status": "completed",
                    "result": {"output": "report loaded"},
                },
                {
                    "step_name": "s2 generate pptx",
                    "status": "failed",
                    "error": "addTable: 'rows' should be an array of cells!",
                    "result": {
                        "success": False,
                        "output": "",
                        "error": "addTable: 'rows' should be an array of cells!",
                    },
                },
            ],
        }
    ]

    summary = analyzer._summarize_execution_history(history)

    assert "Failed Step Results" in summary
    assert "s2 generate pptx" in summary
    assert "addTable" in summary


def test_extract_content_prepends_error_when_output_empty():
    """A failed step with empty stdout must still surface its error in the summary."""
    analyzer = ResultAnalyzer(llm=_StubLLM(), tracer=MagicMock())

    content = analyzer._extract_content_from_result(
        {"success": False, "output": "", "error": "boom: bad shape"}
    )
    assert "boom: bad shape" in content
    assert content.startswith("[error:")


def test_extract_content_keeps_output_when_present_but_still_shows_error():
    """When a step has both partial output and an error, both are visible."""
    analyzer = ResultAnalyzer(llm=_StubLLM(), tracer=MagicMock())

    content = analyzer._extract_content_from_result(
        {"success": False, "output": "step started", "error": "halted midway"}
    )
    assert "halted midway" in content
    assert "step started" in content


def test_extract_content_unchanged_on_success():
    """No error ⇒ no prefix; existing behavior preserved."""
    analyzer = ResultAnalyzer(llm=_StubLLM(), tracer=MagicMock())

    content = analyzer._extract_content_from_result(
        {"success": True, "output": "all good"}
    )
    assert content == "all good"
