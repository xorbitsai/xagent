from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from ....computer.browser import BrowserComputerEnvironment
from ....computer.environment import (
    ComputerEnvironment,
    ComputerFrameMismatchError,
    ComputerSessionMismatchError,
    ComputerTargetNotFoundError,
)
from ....computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerObservation,
)
from ....context_ref import CONTEXT_REFS_KEY
from ....workspace import TaskWorkspace
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .browser_use import BrowserTaskSessionMixin

logger = logging.getLogger(__name__)

ComputerEnvironmentFactory = Callable[..., ComputerEnvironment]


def _initial_screenshot_actions() -> list[ComputerAction]:
    return [ComputerAction(type=ComputerActionType.SCREENSHOT)]


class ComputerToolArgs(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Browser session ID. Omit to use the current task session.",
    )
    expected_frame_id: str | None = Field(
        default=None,
        description=(
            "Frame ID that the actions were planned against. Required for every "
            "state-changing call; omit only when requesting a fresh screenshot."
        ),
    )
    actions: list[ComputerAction] = Field(
        default_factory=_initial_screenshot_actions,
        min_length=1,
        max_length=1,
        description=(
            "One browser action. Coordinates are normalized from 0 to 1. Every "
            "call returns a new browser observation and screenshot before another "
            "state-changing action may be planned."
        ),
    )


class ComputerToolResult(BaseModel):
    success: bool
    session_id: str
    frame_id: str | None = None
    observation: ComputerObservation | None = None
    message: str = ""
    error: str = ""


class ComputerTool(BrowserTaskSessionMixin, AbstractBaseTool):
    """Unified screenshot-and-action tool for ordinary vision models."""

    category = ToolCategory.BROWSER
    decision_group = "computer"

    def __init__(
        self,
        *,
        task_id: str | None = None,
        workspace: TaskWorkspace | None = None,
        environment_factory: ComputerEnvironmentFactory = BrowserComputerEnvironment,
        headless: bool = True,
    ) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id
        self._workspace = workspace
        self._environment_factory = environment_factory
        self._headless = headless
        self._environments: dict[str, ComputerEnvironment] = {}

    @property
    def name(self) -> str:
        return "computer"

    @property
    def description(self) -> str:
        return """Inspect and control one browser through screenshots.

        Workflow:
        1. First call: request only a screenshot and omit expected_frame_id.
        2. Inspect the returned screenshot.
        3. For click/type/scroll/keypress/drag/move/navigate, copy the returned
           frame_id into expected_frame_id.
        4. Every successful call automatically returns a fresh screenshot and frame_id.

        Coordinates are normalized: x=0/y=0 is the viewport top-left and
        x=1/y=1 is the bottom-right. Element IDs are valid only for the frame that
        returned them. `keys` represents one keyboard chord, e.g. ["CTRL", "A"].
        Do not request a separate screenshot after an action; the action result
        already contains the new observation.
        """

    @property
    def tags(self) -> list[str]:
        return ["browser", "computer-use", "vision", "automation"]

    def args_type(self) -> type[BaseModel]:
        return ComputerToolArgs

    def return_type(self) -> type[BaseModel]:
        return ComputerToolResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("computer is async-only")

    async def run_json_async(self, args: Mapping[str, Any]) -> dict[str, Any]:
        parsed = ComputerToolArgs.model_validate(self._with_default_session(args))
        session_id = str(parsed.session_id or "").strip()
        if not session_id:
            return self._error_result(
                session_id="",
                error="Computer tool requires a task or explicit session_id.",
            )
        if self._workspace is None:
            return self._error_result(
                session_id=session_id,
                error="Computer tool requires a task workspace for observations.",
            )

        environment = self._environments.get(session_id)
        if environment is None:
            environment = self._environment_factory(
                session_id=session_id,
                workspace=self._workspace,
                headless=self._headless,
            )
            self._environments[session_id] = environment

        screenshot_only = all(
            action.type == ComputerActionType.SCREENSHOT for action in parsed.actions
        )
        try:
            if environment.current_observation is None:
                if not screenshot_only:
                    return self._error_result(
                        session_id=session_id,
                        error=(
                            "No browser frame exists yet. Call computer with only a "
                            "screenshot action before planning other actions."
                        ),
                    )
                observation = await environment.observe()
            elif parsed.expected_frame_id is None:
                if not screenshot_only:
                    return self._error_result(
                        session_id=session_id,
                        frame_id=environment.current_observation.frame_id,
                        error=(
                            "expected_frame_id is required for state-changing "
                            "computer actions."
                        ),
                    )
                observation = await environment.observe()
            else:
                observation = await environment.execute(
                    ComputerActionBatch(
                        session_id=session_id,
                        expected_frame_id=parsed.expected_frame_id,
                        actions=parsed.actions,
                    )
                )
        except (
            ComputerFrameMismatchError,
            ComputerSessionMismatchError,
            ComputerTargetNotFoundError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exc:
            current = environment.current_observation
            return self._error_result(
                session_id=session_id,
                frame_id=current.frame_id if current else None,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - provider errors become tool failures.
            current = environment.current_observation
            return self._error_result(
                session_id=session_id,
                frame_id=current.frame_id if current else None,
                error=f"Browser computer action failed: {exc}",
            )

        result = ComputerToolResult(
            success=True,
            session_id=session_id,
            frame_id=observation.frame_id,
            observation=observation,
            message=(
                f"Browser observation captured for frame {observation.frame_id}. "
                "Use this exact frame_id for the next state-changing action."
            ),
        ).model_dump(mode="json", exclude_none=True)
        result[CONTEXT_REFS_KEY] = [observation.screenshot.durable_dict()]
        return result

    async def teardown(self, task_id: str | None = None) -> None:
        environments = list(self._environments.values())
        self._environments.clear()
        for environment in environments:
            try:
                await environment.close()
            except Exception:  # noqa: BLE001 - teardown must continue.
                logger.warning(
                    "Failed to close computer environment %s",
                    environment.session_id,
                    exc_info=True,
                )
        await self._close_task_sessions(task_id)

    @staticmethod
    def _error_result(
        *,
        session_id: str,
        error: str,
        frame_id: str | None = None,
    ) -> dict[str, Any]:
        return ComputerToolResult(
            success=False,
            session_id=session_id,
            frame_id=frame_id,
            error=error,
        ).model_dump(mode="json", exclude_none=True)
