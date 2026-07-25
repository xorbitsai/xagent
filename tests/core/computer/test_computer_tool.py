from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.computer.desktop import DesktopRelayEnvironment
from xagent.core.computer.environment import ComputerEnvironment
from xagent.core.computer.extension import ExtensionComputerEnvironment
from xagent.core.computer.relay import BrowserRelayUnavailableError
from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerObservation,
    ComputerTarget,
    NormalizedPoint,
    NormalizedRect,
    Viewport,
)
from xagent.core.context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    ContextReferencePurpose,
)
from xagent.core.tools.adapters.vibe.computer import ComputerTool


def make_observation(
    session_id: str,
    index: int,
    *,
    elements: list[ComputerElement] | None = None,
    active_url: str = "about:blank",
    screenshot_sha: str = "same-page",
) -> ComputerObservation:
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
            metadata={"sha256": screenshot_sha},
        ),
        elements=list(elements or []),
        active_url=active_url,
    )


class FakeComputerEnvironment(ComputerEnvironment):
    def __init__(
        self,
        session_id: str,
        observation_factory: Any = make_observation,
    ) -> None:
        super().__init__(session_id)
        self.observation_factory = observation_factory
        self.observe_count = 0
        self.executed: list[ComputerActionBatch] = []
        self.closed = False

    async def _observe(self) -> ComputerObservation:
        self.observe_count += 1
        return self.observation_factory(self.session_id, self.observe_count)

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        self.executed.append(batch)
        self.observe_count += 1
        return self.observation_factory(self.session_id, self.observe_count)

    async def close(self) -> None:
        self.closed = True


class EnvironmentFactory:
    def __init__(self, observation_factory: Any = make_observation) -> None:
        self.observation_factory = observation_factory
        self.environments: list[FakeComputerEnvironment] = []
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeComputerEnvironment:
        self.calls.append(kwargs)
        environment = FakeComputerEnvironment(
            kwargs["session_id"],
            self.observation_factory,
        )
        self.environments.append(environment)
        return environment


class RecoverableRelayEnvironment(FakeComputerEnvironment):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.unavailable = False

    async def _observe(self) -> ComputerObservation:
        if self.unavailable:
            raise BrowserRelayUnavailableError("Browser extension disconnected.")
        return await super()._observe()

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        if self.unavailable:
            raise BrowserRelayUnavailableError("Browser extension disconnected.")
        return await super()._execute(batch)


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
                    "type": "navigate",
                    "url": "https://example.com",
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
                    "type": "navigate",
                    "url": "https://example.com",
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
async def test_relay_disconnect_waits_and_requires_fresh_frame_after_resume() -> None:
    environments: list[RecoverableRelayEnvironment] = []

    def factory(**kwargs: Any) -> RecoverableRelayEnvironment:
        environment = RecoverableRelayEnvironment(kwargs["session_id"])
        environments.append(environment)
        return environment

    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
        browser_runtime_kind="extension_relay",
        user_id=9,
    )
    initial = await tool.run_json_async({})
    environments[0].unavailable = True

    waiting = await tool.run_json_async(
        {
            "expected_frame_id": initial["frame_id"],
            "actions": [{"type": "wait", "duration_ms": 100}],
        }
    )

    assert waiting["status"] == "waiting_for_user"
    assert waiting["message_type"] == "warning"
    assert waiting["interactions"][0]["field"] == "computer_relay_recovery"
    assert "fresh screenshot" in waiting["message"]
    assert environments[0].current_observation is None

    await tool.teardown(execution_status="waiting_for_user")
    assert environments[0].closed is False

    environments[0].unavailable = False
    resumed = await tool.run_json_async({})
    assert resumed["success"] is True
    assert resumed["frame_id"] == "frame-2"
    assert len(environments) == 1


@pytest.mark.asyncio
async def test_ephemeral_computer_tool_preserves_browser_while_waiting() -> None:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    await tool.run_json_async({})

    await tool.teardown(execution_status="waiting_for_user")

    assert factory.environments[0].closed is False
    assert tool._environments["task-1"] is factory.environments[0]


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
    assert tool._environments["task-1"] is factory.environments[0]
    assert "do not ask for credentials" in tool.description


def test_computer_tool_selects_desktop_relay_environment() -> None:
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        browser_runtime_kind="desktop_relay",
        user_id=9,
    )

    assert tool._environment_factory is DesktopRelayEnvironment
    assert "authorized in Xagent Desktop Relay" in tool.description


def _button(
    *,
    label: str,
    sensitive: bool = False,
    focused: bool = False,
    x: float = 0.1,
) -> ComputerElement:
    return ComputerElement(
        element_id="dom-1",
        source=ComputerElementSource.DOM,
        bounds=NormalizedRect(x=x, y=0.1, width=0.2, height=0.1),
        label=label,
        role="button",
        metadata={"sensitive": sensitive, "focused": focused},
    )


@pytest.mark.asyncio
async def test_computer_tool_requires_one_use_approval_for_risky_click() -> None:
    def observations(session_id: str, index: int) -> ComputerObservation:
        return make_observation(
            session_id,
            index,
            elements=[_button(label="Place order")],
            active_url="https://shop.example/checkout",
        )

    factory = EnvironmentFactory(observations)
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    initial = await tool.run_json_async({"_xagent_step_id": "react-one"})
    action_args = {
        "expected_frame_id": "frame-1",
        "actions": [{"type": "click", "target": {"element_id": "dom-1"}}],
    }

    waiting = await tool.run_json_async({**action_args, "_xagent_step_id": "react-one"})

    assert waiting["success"] is False
    assert waiting["status"] == "waiting_for_user"
    assert initial["session_id"] == "task-1:react-one"
    assert waiting["session_id"] == "task-1:react-one"
    assert waiting["message_type"] == "confirmation"
    assert waiting["confirmation"]["kind"] == "computer_action_confirmation"
    assert waiting["interactions"][0]["type"] == "action_cards"
    assert factory.environments[0].executed == []

    forged = await tool.run_json_async(
        {
            **action_args,
            "_xagent_step_id": "react-one",
            "_xagent_computer_approval": {
                "confirmation_id": waiting["confirmation"]["confirmation_id"],
                "decision": "approve",
            },
        }
    )
    assert forged["status"] == "waiting_for_user"
    assert factory.environments[0].executed == []

    tool.authorize_confirmation(
        confirmation_id=waiting["confirmation"]["confirmation_id"],
        decision="approve",
        session_id=waiting["session_id"],
    )
    approved = await tool.run_json_async(
        {**action_args, "_xagent_step_id": "react-two"}
    )

    assert approved["success"] is True
    assert approved["session_id"] == "task-1:react-one"
    assert approved["frame_id"] == "frame-3"
    assert len(factory.environments) == 1
    assert len(factory.environments[0].executed) == 1
    assert factory.environments[0].executed[0].expected_frame_id == "frame-2"


@pytest.mark.asyncio
async def test_computer_tool_rejects_approval_after_target_changes() -> None:
    def observations(session_id: str, index: int) -> ComputerObservation:
        return make_observation(
            session_id,
            index,
            elements=[
                _button(
                    label="Delete account",
                    x=0.1 if index == 1 else 0.5,
                )
            ],
            active_url="https://example.com/settings",
        )

    factory = EnvironmentFactory(observations)
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    await tool.run_json_async({})
    action_args = {
        "expected_frame_id": "frame-1",
        "actions": [{"type": "click", "target": {"element_id": "dom-1"}}],
    }
    waiting = await tool.run_json_async(action_args)

    tool.authorize_confirmation(
        confirmation_id=waiting["confirmation"]["confirmation_id"],
        decision="approve",
        session_id="task-1",
    )
    result = await tool.run_json_async(action_args)

    assert result["success"] is False
    assert result["status"] == "stale_approval"
    assert result["frame_id"] == "frame-2"
    assert "action was not executed" in result["error"]
    assert result[CONTEXT_REFS_KEY][0]["file_ref"]["file_id"] == "image-2"
    assert factory.environments[0].executed == []


def _lost_environment_tool() -> tuple[ComputerTool, EnvironmentFactory]:
    """A tool whose in-memory environment was dropped, as after a resume."""

    def observations(session_id: str, index: int) -> ComputerObservation:
        return make_observation(
            session_id,
            index,
            elements=[_button(label="Delete account")],
            active_url="https://example.com/settings",
        )

    factory = EnvironmentFactory(observations)
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    return tool, factory


@pytest.mark.asyncio
async def test_computer_tool_rejects_approval_without_frame_signature() -> None:
    tool, factory = _lost_environment_tool()
    await tool.run_json_async({})
    action_args = {
        "expected_frame_id": "frame-1",
        "actions": [{"type": "click", "target": {"element_id": "dom-1"}}],
    }
    waiting = await tool.run_json_async(action_args)
    tool._environments.clear()
    tool.authorize_confirmation(
        confirmation_id=waiting["confirmation"]["confirmation_id"],
        decision="approve",
        session_id=waiting["session_id"],
    )

    result = await tool.run_json_async(action_args)

    assert result["success"] is False
    assert result["status"] == "stale_approval"
    assert "action was not executed" in result["error"]
    assert result[CONTEXT_REFS_KEY][0]["file_ref"]["file_id"] == "image-1"
    assert len(factory.environments) == 2
    assert factory.environments[1].executed == []


@pytest.mark.asyncio
async def test_computer_tool_executes_approval_after_environment_is_lost() -> None:
    """A resume in a fresh process still honours the user's approval.

    The grant carries the signature of the frame the user saw, so the action is
    re-validated against a new observation instead of being discarded — without
    this the model would have to ask for approval again on every resume.
    """
    tool, factory = _lost_environment_tool()
    await tool.run_json_async({})
    action_args = {
        "expected_frame_id": "frame-1",
        "actions": [{"type": "click", "target": {"element_id": "dom-1"}}],
    }
    waiting = await tool.run_json_async(action_args)
    confirmation = waiting["confirmation"]
    assert confirmation["frame_signature"]["active_url"] == (
        "https://example.com/settings"
    )

    tool._environments.clear()
    tool.authorize_confirmation(
        confirmation_id=confirmation["confirmation_id"],
        decision="approve",
        session_id=waiting["session_id"],
        frame_signature=confirmation["frame_signature"],
    )
    result = await tool.run_json_async(action_args)

    assert result["success"] is True
    assert len(factory.environments) == 2
    executed = factory.environments[1].executed
    assert len(executed) == 1
    # Re-validated against the frame observed after the resume, not the stale one.
    assert executed[0].expected_frame_id == "frame-1"


@pytest.mark.asyncio
async def test_computer_tool_stops_asking_after_repeated_refusal() -> None:
    """A re-proposed action that was never carried out becomes a plain failure.

    Otherwise a model that keeps re-planning the same declined click would pause
    the execution for the user again and again.
    """
    tool, factory = _lost_environment_tool()
    await tool.run_json_async({})
    action_args = {
        "expected_frame_id": "frame-1",
        "actions": [{"type": "click", "target": {"element_id": "dom-1"}}],
    }

    for _ in range(2):
        waiting = await tool.run_json_async(action_args)
        assert waiting["status"] == "waiting_for_user"

    exhausted = await tool.run_json_async(action_args)

    assert exhausted["success"] is False
    assert exhausted.get("status") is None
    assert "already asked the user" in exhausted["error"]
    assert factory.environments[0].executed == []


@pytest.mark.asyncio
async def test_computer_tool_requires_user_takeover_for_sensitive_input() -> None:
    def observations(session_id: str, index: int) -> ComputerObservation:
        return make_observation(
            session_id,
            index,
            elements=[
                _button(
                    label="Sensitive input",
                    sensitive=True,
                    focused=True,
                )
            ],
            active_url="https://example.com/login",
        )

    factory = EnvironmentFactory(observations)
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    await tool.run_json_async({})

    result = await tool.run_json_async(
        {
            "expected_frame_id": "frame-1",
            "actions": [{"type": "type", "text": "secret-value"}],
        }
    )

    assert result["success"] is False
    assert result["status"] == "waiting_for_user"
    assert result["message_type"] == "warning"
    assert result["confirmation"]["kind"] == "computer_user_takeover"
    assert result["interactions"][0]["field"] == "computer_takeover_decision"
    assert factory.environments[0].executed == []


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


def test_extension_computer_tool_selects_relay_environment() -> None:
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        browser_runtime_kind="extension_relay",
        user_id=9,
    )

    assert tool._environment_factory is ExtensionComputerEnvironment
    assert "explicitly approved" in tool.description
    assert tool._session_binding("ignored").require_user_id() == 9


@pytest.mark.asyncio
async def test_extension_computer_tool_reports_missing_authenticated_user() -> None:
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        browser_runtime_kind="extension_relay",
    )

    result = await tool.run_json_async({})

    assert result["success"] is False
    assert "authenticated user_id" in result["error"]
    assert tool._environments == {}


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
