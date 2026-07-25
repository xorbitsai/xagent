"""What the output-filter wrapper must not hide from the runtime.

Every tool reaches a pattern wrapped by ``OutputFilteredToolWrapper``, so any
capability the runtime discovers with ``getattr`` — or any teardown argument it
forwards — has to survive the wrapper. When it does not, the failure is silent:
the capability simply appears to be absent.
"""

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


class CapabilityTool:
    name = "capable"
    description = "exposes optional capabilities"
    tags: list[str] = []
    decision_group = "capable-group"
    uses_browser_session = True

    def __init__(self) -> None:
        self.grants: list[dict[str, Any]] = []

    def authorize_confirmation(self, **kwargs: Any) -> None:
        self.grants.append(kwargs)


def test_optional_capabilities_reach_through_the_wrapper() -> None:
    target = CapabilityTool()
    wrapper = _wrap(target)

    grant = getattr(wrapper, "authorize_confirmation", None)
    assert callable(grant)
    grant(confirmation_id="c-1", decision="approve", session_id="s-1")

    assert target.grants == [
        {"confirmation_id": "c-1", "decision": "approve", "session_id": "s-1"}
    ]
    assert wrapper.decision_group == "capable-group"
    assert wrapper.uses_browser_session is True


def test_absent_capabilities_stay_absent() -> None:
    """Probing must not invent a capability the wrapped tool does not have."""

    class PlainTool:
        name = "plain"
        description = "no optional capabilities"
        tags: list[str] = []

    wrapper = _wrap(PlainTool())

    assert getattr(wrapper, "authorize_confirmation", None) is None
    with pytest.raises(AttributeError):
        wrapper.authorize_confirmation  # noqa: B018


@pytest.mark.asyncio
async def test_teardown_forwards_the_execution_status() -> None:
    """A stateful tool needs the status to keep resources alive across a pause."""

    class StatusAwareTool:
        name = "status-aware"
        description = "records teardown"
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

    await _wrap(target).teardown(task_id="task-1", execution_status="waiting_for_user")

    assert target.calls == [
        {"task_id": "task-1", "execution_status": "waiting_for_user"}
    ]


@pytest.mark.asyncio
async def test_teardown_omits_arguments_a_legacy_tool_cannot_accept() -> None:
    class LegacyTool:
        name = "legacy"
        description = "legacy teardown"
        tags: list[str] = []

        def __init__(self) -> None:
            self.calls: list[str | None] = []

        async def teardown(self, task_id: str | None = None) -> None:
            self.calls.append(task_id)

    target = LegacyTool()

    await _wrap(target).teardown(task_id="task-2", execution_status="completed")

    assert target.calls == ["task-2"]
