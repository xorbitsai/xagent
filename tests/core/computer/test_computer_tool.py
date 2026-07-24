from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.computer.environment import ComputerEnvironment
from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerEnvironmentType,
    ComputerObservation,
    ComputerTarget,
    NormalizedPoint,
    Viewport,
)
from xagent.core.context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    ContextReferencePurpose,
)
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
            purpose=ContextReferencePurpose.OBSERVATION,
            frame_id=frame_id,
        ),
        active_url="about:blank",
    )


class FakeComputerEnvironment(ComputerEnvironment):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.observe_count = 0
        self.executed: list[ComputerActionBatch] = []
        self.closed = False

    async def _observe(self) -> ComputerObservation:
        self.observe_count += 1
        return make_observation(self.session_id, self.observe_count)

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        self.executed.append(batch)
        self.observe_count += 1
        return make_observation(self.session_id, self.observe_count)

    async def close(self) -> None:
        self.closed = True


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
    assert initial["browser_runtime_kind"] == "ephemeral_playwright"
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
async def test_computer_tool_rejects_action_before_first_observation() -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )

    result = await tool.run_json_async(
        {
            "actions": [
                {
                    "type": "navigate",
                    "url": "https://example.com",
                }
            ]
        }
    )

    assert result["success"] is False
    assert "screenshot action" in result["error"]
    assert factory.environments[0].current_observation is None


def test_computer_tool_accepts_only_one_action_per_observation() -> None:
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=EnvironmentFactory(),
    )

    with pytest.raises(ValueError, match="at most 1"):
        tool.args_type().model_validate(
            {
                "actions": [
                    {"type": "screenshot"},
                    {"type": "screenshot"},
                ]
            }
        )


@pytest.mark.asyncio
async def test_computer_tool_teardown_closes_created_environments() -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    await tool.run_json_async({})

    await tool.teardown()

    assert factory.environments[0].closed is True
    assert tool._environments == {}


@pytest.mark.asyncio
async def test_persistent_computer_tool_preserves_browser_while_waiting(
    tmp_path,
) -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
        browser_runtime_kind="persistent_playwright",
        user_id=9,
        browser_profile_root=tmp_path,
    )
    result = await tool.run_json_async({"session_id": "model-selected-session"})

    await tool.teardown(execution_status="waiting_for_user")

    binding = factory.calls[0]["session_binding"]
    assert result["session_id"] == "task-1"
    assert factory.calls[0]["session_id"] == "task-1"
    assert binding.manager_session_id("ignored") == ("computer-profile:user_9:default")
    assert factory.environments[0].closed is False
    assert "do not ask for credentials" in tool.description


def test_persistent_computer_tool_requires_authenticated_user(tmp_path) -> None:
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=EnvironmentFactory(),
        browser_runtime_kind="persistent_playwright",
        browser_profile_root=tmp_path,
    )

    with pytest.raises(ValueError, match="authenticated user_id"):
        tool._session_binding("task-1").persistent_profile_dir()


class VisionToolCallingLLM:
    abilities = ["chat", "vision"]

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "computer-1",
                        "function": {
                            "name": "computer",
                            "arguments": "{}",
                        },
                    }
                ],
                "done": False,
            }
        return {"content": "The blank browser page is visible.", "done": True}


class Resolver:
    async def resolve_image(self, reference: ContextReference) -> str:
        return "data:image/png;base64,c2NyZWVuc2hvdA=="


@pytest.mark.asyncio
async def test_react_computer_tool_feeds_observation_to_next_vision_call() -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    llm = VisionToolCallingLLM()
    context = ExecutionContext()
    context.add_user_message("Inspect the browser.")
    runtime = PatternRuntime(
        execution_id="task-1",
        context_ref_resolver=Resolver(),
    )

    result = await ReActPattern(max_iterations=2).run(
        context=context,
        tools=[tool],
        llm=llm,
        runtime=runtime,
    )

    assert result["success"] is True
    tool_message = next(
        message for message in context.messages if message.role == "tool"
    )
    assert tool_message.context_refs[0].file_id == "image-1"
    assert CONTEXT_REFS_KEY not in tool_message.metadata["raw_result"]
    assert "base64" not in json.dumps(context.to_dict())

    image_messages = [
        message
        for message in llm.calls[1]["messages"]
        if isinstance(message.get("content"), list)
    ]
    assert len(image_messages) == 1
    assert image_messages[0]["content"][-1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_computer_action_objects_remain_supported_by_tool_schema() -> None:
    action = ComputerAction(
        type=ComputerActionType.CLICK,
        target=ComputerTarget(point=NormalizedPoint(x=0.25, y=0.75)),
    )

    assert action.model_dump(mode="json")["type"] == "click"
