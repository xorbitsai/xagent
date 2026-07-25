"""The computer tool as agents actually receive it: behind tool wrappers.

``ToolFactory`` wraps every tool in ``OutputFilteredToolWrapper``, so anything
the runtime reaches for through ``getattr`` — the one-use confirmation grant,
the teardown status that keeps a browser alive across a pause — has to survive
that wrapper. Constructing ``ComputerTool`` directly hides exactly these bugs.
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.computer.environment import ComputerEnvironment
from xagent.core.computer.schema import (
    ComputerActionBatch,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerObservation,
    NormalizedRect,
    Viewport,
)
from xagent.core.context_ref import ContextReference, ContextReferencePurpose
from xagent.core.tools.adapters.vibe.browser_use import create_browser_tools
from xagent.core.tools.adapters.vibe.computer import ComputerTool
from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)
from xagent.core.tools.confirmation import confirmation_grant_callable


def _button_observation(session_id: str, index: int) -> ComputerObservation:
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
        elements=[
            ComputerElement(
                element_id="dom-1",
                source=ComputerElementSource.DOM,
                bounds=NormalizedRect(x=0.1, y=0.1, width=0.2, height=0.1),
                label="Place order",
                role="button",
            )
        ],
        active_url="https://shop.example/checkout",
    )


class FakeEnvironment(ComputerEnvironment):
    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.observe_count = 0
        self.executed: list[ComputerActionBatch] = []

    async def _observe(self) -> ComputerObservation:
        self.observe_count += 1
        return _button_observation(self.session_id, self.observe_count)

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        self.executed.append(batch)
        self.observe_count += 1
        return _button_observation(self.session_id, self.observe_count)


class EnvironmentFactory:
    def __init__(self) -> None:
        self.environments: list[FakeEnvironment] = []

    def __call__(self, **kwargs: Any) -> FakeEnvironment:
        environment = FakeEnvironment(kwargs["session_id"])
        self.environments.append(environment)
        return environment


def _wrapped_computer_tool() -> tuple[Any, EnvironmentFactory]:
    factory = EnvironmentFactory()
    tool = ComputerTool(
        task_id="task-1",
        workspace=object(),  # type: ignore[arg-type]
        environment_factory=factory,
    )
    wrapper = OutputFilteredToolWrapper(
        target_tool=tool,
        max_chars=50_000,
        max_fields=1_000,
        max_recursion=20,
    )
    return wrapper, factory


@pytest.mark.asyncio
async def test_wrapper_exposes_the_confirmation_grant() -> None:
    wrapper, factory = _wrapped_computer_tool()
    await wrapper.run_json_async({})
    action_args = {
        "expected_frame_id": "frame-1",
        "actions": [{"type": "click", "target": {"element_id": "dom-1"}}],
    }

    waiting = await wrapper.run_json_async(action_args)
    assert waiting["status"] == "waiting_for_user"

    grant = confirmation_grant_callable(wrapper)
    assert grant is not None, "approval can never reach a wrapped tool"
    grant(
        confirmation_id=waiting["confirmation"]["confirmation_id"],
        decision="approve",
        session_id=waiting["session_id"],
        frame_signature=waiting["confirmation"]["frame_signature"],
    )
    approved = await wrapper.run_json_async(action_args)

    assert approved["success"] is True
    assert len(factory.environments[0].executed) == 1


def test_computer_capabilities_survive_the_wrapper() -> None:
    tool = ComputerTool(task_id="task-1", workspace=object())  # type: ignore[arg-type]
    wrapper = OutputFilteredToolWrapper(
        target_tool=tool,
        max_chars=100,
        max_fields=10,
        max_recursion=5,
    )

    assert wrapper.uses_browser_session is True
    assert wrapper.decision_group == "computer"


def test_mutating_browser_tools_are_not_exposed_by_default() -> None:
    """The selector tools would sidestep the computer action policy entirely."""
    names = {
        tool.name for tool in create_browser_tools(task_id="task-1", workspace=None)
    }

    assert "computer" in names
    assert names.isdisjoint(
        {
            "browser_click",
            "browser_fill",
            "browser_evaluate",
            "browser_navigate",
            "browser_select_option",
        }
    )


def test_mutating_browser_tools_can_be_opted_into() -> None:
    names = {
        tool.name
        for tool in create_browser_tools(
            task_id="task-1",
            workspace=None,
            include_legacy_dom_tools=True,
        )
    }

    assert {"browser_click", "browser_fill", "browser_evaluate"} <= names
