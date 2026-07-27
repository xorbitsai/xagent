"""Runtime capabilities that must survive the output-filter wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)


def _wrap(target: Any) -> OutputFilteredToolWrapper:
    return OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=1_000,
        max_fields=50,
        max_recursion=5,
    )


def test_optional_capabilities_reach_through_the_wrapper() -> None:
    class ResumableTool:
        name = "resumable"
        description = "Accepts a user response."
        tags: list[str] = []
        decision_group = "interactive"

        def __init__(self) -> None:
            self.responses: list[dict[str, str]] = []

        def resume_user_interaction(
            self,
            *,
            interaction_id: str,
            response: str,
        ) -> None:
            self.responses.append(
                {"interaction_id": interaction_id, "response": response}
            )

    target = ResumableTool()
    wrapper = _wrap(target)

    resume = getattr(wrapper, "resume_user_interaction", None)
    assert callable(resume)
    resume(interaction_id="interaction-1", response="Continue")

    assert target.responses == [
        {"interaction_id": "interaction-1", "response": "Continue"}
    ]
    assert wrapper.decision_group == "interactive"


def test_absent_capabilities_stay_absent() -> None:
    class PlainTool:
        name = "plain"
        description = "No optional capabilities."
        tags: list[str] = []

    wrapper = _wrap(PlainTool())

    assert getattr(wrapper, "resume_user_interaction", None) is None
    with pytest.raises(AttributeError):
        wrapper.resume_user_interaction  # noqa: B018


@pytest.mark.asyncio
async def test_waiting_control_envelope_survives_field_filtering() -> None:
    class WaitingTool:
        name = "waiting"
        description = "Returns an interaction after unrelated output."
        tags: list[str] = []

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "large_unrelated_field": "x" * 100,
                "status": "waiting_for_user",
                "interaction_id": "14ee67d1-d18e-47e0-a8f8-b28ef31262f5",
                "message": "Provide a value.",
                "message_type": "question",
                "interactions": [
                    {
                        "type": "file_upload",
                        "field": "exact-routing-field",
                        "label": "Choose the desired operation",
                        "accept": ["video/mp4", "audio/mpeg"],
                        "multiple": False,
                        "multiline": False,
                        "options": [
                            {
                                "label": "Approve this operation",
                                "value": "exact-routing-option-value",
                            },
                            {
                                "label": "Reject this operation",
                                "value": "second-routing-option-value",
                            },
                        ],
                    },
                    {
                        "type": "confirm",
                        "field": "second-routing-field",
                        "label": "Continue to the next step?",
                        "default": False,
                    },
                ],
            }

    wrapper = OutputFilteredToolWrapper(
        target_tool=WaitingTool(),
        max_chars=8,
        max_fields=1,
        max_recursion=3,
    )

    result = await wrapper.run_json_async({})

    assert result["status"] == "waiting_for_user"
    assert result["interaction_id"] == "14ee67d1-d18e-47e0-a8f8-b28ef31262f5"
    assert result["message"].startswith("Provide ")
    assert result["message_type"] == "question"
    assert len(result["interactions"]) == 2
    assert result["interactions"][0]["type"] == "file_upload"
    assert result["interactions"][0]["field"] == "exact-routing-field"
    assert result["interactions"][0]["label"].startswith("Choose t")
    assert result["interactions"][0]["accept"] == ["video/mp4", "audio/mpeg"]
    assert result["interactions"][0]["multiple"] is False
    assert result["interactions"][0]["multiline"] is False
    assert len(result["interactions"][0]["options"]) == 2
    assert result["interactions"][0]["options"][0]["value"] == (
        "exact-routing-option-value"
    )
    assert result["interactions"][0]["options"][0]["label"].startswith("Approve ")
    assert result["interactions"][0]["options"][1]["value"] == (
        "second-routing-option-value"
    )
    assert result["interactions"][1]["field"] == "second-routing-field"
    assert result["interactions"][1]["default"] is False


@pytest.mark.asyncio
async def test_teardown_forwards_execution_status_when_supported() -> None:
    class StatusAwareTool:
        name = "status-aware"
        description = "Records teardown."
        tags: list[str] = []

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def teardown(
            self,
            task_id: str | None = None,
            execution_status: str | None = None,
        ) -> None:
            self.calls.append(
                {"task_id": task_id, "execution_status": execution_status}
            )

    target = StatusAwareTool()

    await _wrap(target).teardown(
        task_id="task-1",
        execution_status="waiting_for_user",
    )

    assert target.calls == [
        {"task_id": "task-1", "execution_status": "waiting_for_user"}
    ]


@pytest.mark.asyncio
async def test_teardown_omits_status_for_legacy_tool() -> None:
    class LegacyTool:
        name = "legacy"
        description = "Legacy teardown signature."
        tags: list[str] = []

        def __init__(self) -> None:
            self.calls: list[str | None] = []

        async def teardown(self, task_id: str | None = None) -> None:
            self.calls.append(task_id)

    target = LegacyTool()

    await _wrap(target).teardown(
        task_id="task-2",
        execution_status="completed",
    )

    assert target.calls == ["task-2"]
