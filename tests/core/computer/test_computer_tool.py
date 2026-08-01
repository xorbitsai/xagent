from __future__ import annotations

from typing import Any

import pytest

from xagent.core.agent import ExecutionContext
from xagent.core.computer.environment import ComputerEnvironment
from xagent.core.computer.schema import (
    COMPUTER_FRAME_ID_METADATA_KEY,
    COMPUTER_SESSION_ID_METADATA_KEY,
    ComputerActionBatch,
    ComputerEnvironmentType,
    ComputerObservation,
    Viewport,
)
from xagent.core.context_ref import CONTEXT_REFS_KEY, ContextReference
from xagent.core.tools.adapters.vibe.computer import ComputerTool


def make_observation(session_id: str, index: int) -> ComputerObservation:
    frame_id = f"frame-{index}"
    return ComputerObservation(
        session_id=session_id,
        frame_id=frame_id,
        environment=ComputerEnvironmentType.BROWSER,
        viewport=Viewport(width=1280, height=720),
        screenshot=ContextReference(
            file_ref={
                "file_id": f"image-{index}",
                "filename": f"{frame_id}.png",
                "mime_type": "image/png",
            },
            metadata={
                COMPUTER_SESSION_ID_METADATA_KEY: session_id,
                COMPUTER_FRAME_ID_METADATA_KEY: frame_id,
            },
        ),
        active_url="about:blank",
    )


class FakeComputerEnvironment(ComputerEnvironment):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.observe_count = 0
        self.executed: list[ComputerActionBatch] = []
        self.close_count = 0

    async def _observe(self) -> ComputerObservation:
        self.observe_count += 1
        return make_observation(self.session_id, self.observe_count)

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        self.executed.append(batch)
        self.observe_count += 1
        return make_observation(self.session_id, self.observe_count)

    async def _close(self) -> None:
        self.close_count += 1


class EnvironmentFactory:
    def __init__(self) -> None:
        self.environments: list[FakeComputerEnvironment] = []
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeComputerEnvironment:
        self.calls.append(kwargs)
        environment = FakeComputerEnvironment(kwargs["session_id"])
        self.environments.append(environment)
        return environment


@pytest.mark.asyncio
async def test_computer_tool_requires_initial_screenshot_then_expected_frame() -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )

    initial = await tool.run_json_async({})

    assert initial["success"] is True
    assert initial["session_id"] == "task-1"
    assert initial["frame_id"] == "frame-1"
    assert initial[CONTEXT_REFS_KEY][0]["file_ref"]["file_id"] == "image-1"

    missing_frame = await tool.run_json_async(
        {
            "actions": [
                {
                    "type": "click",
                    "target": {"point": {"x": 0.5, "y": 0.5}},
                }
            ]
        }
    )
    assert missing_frame["success"] is False
    assert "expected_frame_id" in missing_frame["error"]

    acted = await tool.run_json_async(
        {
            "expected_frame_id": "frame-1",
            "actions": [
                {
                    "type": "click",
                    "target": {"point": {"x": 0.5, "y": 0.5}},
                }
            ],
        }
    )

    assert acted["success"] is True
    assert acted["frame_id"] == "frame-2"
    assert factory.environments[0].executed[0].expected_frame_id == "frame-1"


@pytest.mark.asyncio
async def test_computer_tool_derives_session_from_task_and_step() -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )

    result = await tool.run_json_async(
        {
            "session_id": "invented-by-model",
            "_xagent_step_id": "inspect page",
        }
    )

    assert result["session_id"] == "task-1:inspect_page"
    assert factory.calls[0]["session_id"] == "task-1:inspect_page"


@pytest.mark.asyncio
async def test_computer_tool_lifts_action_scoped_expected_frame_id() -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    await tool.run_json_async({})

    result = await tool.run_json_async(
        {
            "actions": [
                {
                    "type": "click",
                    "expected_frame_id": "frame-1",
                    "target": {"point": {"x": 0.5, "y": 0.5}},
                }
            ]
        }
    )

    assert result["success"] is True
    assert factory.environments[0].executed[0].expected_frame_id == "frame-1"


@pytest.mark.asyncio
async def test_computer_tool_rejects_action_before_first_observation() -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )

    result = await tool.run_json_async(
        {"actions": [{"type": "navigate", "url": "https://example.com"}]}
    )

    assert result["success"] is False
    assert "screenshot" in result["error"]
    assert factory.environments[0].current_observation is None


@pytest.mark.asyncio
async def test_computer_tool_returns_validation_error_with_current_frame() -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    await tool.run_json_async({})

    result = await tool.run_json_async(
        {"actions": [{"type": "click", "target": {"point": {"x": 2, "y": 0}}}]}
    )

    assert result["success"] is False
    assert result["session_id"] == "task-1"
    assert result["frame_id"] == "frame-1"
    assert "Invalid computer action" in result["error"]
    assert len(factory.environments) == 1


def test_computer_tool_accepts_only_one_action_per_observation() -> None:
    with pytest.raises(ValueError, match="at most 1"):
        ComputerTool(
            task_id="task-1",
            workspace=object(),  # type: ignore[arg-type]
            environment_factory=EnvironmentFactory(),
        ).args_type().model_validate(
            {"actions": [{"type": "screenshot"}, {"type": "screenshot"}]}
        )


@pytest.mark.asyncio
async def test_computer_tool_teardown_closes_created_environments(
    monkeypatch,
) -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    await tool.run_json_async({})

    async def no_browser_sessions(_task_id: str | None = None) -> None:
        return None

    monkeypatch.setattr(tool, "_close_task_sessions", no_browser_sessions)
    await tool.teardown()

    assert factory.environments[0].close_count == 1
    assert tool._environments == {}


def test_new_computer_observation_supersedes_stale_live_context() -> None:
    context = ExecutionContext()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=EnvironmentFactory(),
    )
    first = make_observation("task-1", 1)
    second = make_observation("task-1", 2)

    for observation in (first, second):
        result = {
            "success": True,
            "frame_id": observation.frame_id,
            "observation": observation.model_dump(mode="json"),
            CONTEXT_REFS_KEY: [observation.screenshot.durable_dict()],
            "_xagent_supersedes_scope": f"{tool.name}:task-1",
        }
        context.add_tool_result("computer", result)

    old_message, current_message = context.messages
    assert old_message.metadata["superseded"] is True
    assert old_message.metadata["raw_result"] == {
        "success": True,
        "superseded": True,
    }
    assert old_message.context_refs
    assert CONTEXT_REFS_KEY not in old_message.to_dict()
    assert old_message.context_refs_token_estimate() == 0
    assert "superseded by a later observation" in old_message.content
    assert CONTEXT_REFS_KEY in current_message.to_dict()
