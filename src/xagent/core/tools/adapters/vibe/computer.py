from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from ....computer.browser import BrowserComputerEnvironment
from ....computer.environment import (
    ComputerEnvironment,
    ComputerEnvironmentError,
)
from ....computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerObservation,
)
from ....context_ref import (
    CONTEXT_REFS_KEY,
    SUPERSEDES_SCOPE_KEY,
)
from ....workspace import TaskWorkspace
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .browser_use import BrowserTaskSessionMixin

logger = logging.getLogger(__name__)

ComputerEnvironmentFactory = Callable[..., ComputerEnvironment]
_STEP_SESSION_ARG = "_xagent_step_id"


def _initial_screenshot_actions() -> list[ComputerAction]:
    return [ComputerAction(type=ComputerActionType.SCREENSHOT)]


class ComputerToolArgs(BaseModel):
    expected_frame_id: str | None = Field(
        default=None,
        description=(
            "Frame ID that the action was planned against. Required for every "
            "state-changing call; omit only when requesting a fresh screenshot."
        ),
    )
    actions: list[ComputerAction] = Field(
        default_factory=_initial_screenshot_actions,
        min_length=1,
        max_length=1,
        description=(
            "One browser action. Coordinates are normalized from 0 to 1. Every "
            "call returns a fresh observation before another action is planned."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _lift_action_scoped_frame_id(cls, value: Any) -> Any:
        """Normalize a common tool-call shape without weakening frame checks."""

        if not isinstance(value, Mapping):
            return value
        raw_actions = value.get("actions")
        if (
            not isinstance(raw_actions, list)
            or len(raw_actions) != 1
            or not isinstance(raw_actions[0], Mapping)
            or "expected_frame_id" not in raw_actions[0]
        ):
            return value

        action = dict(raw_actions[0])
        nested_frame_id = action.pop("expected_frame_id")
        normalized = dict(value)
        normalized["actions"] = [action]
        if nested_frame_id is None:
            return normalized
        top_level_frame_id = normalized.get("expected_frame_id")
        if top_level_frame_id is None:
            normalized["expected_frame_id"] = nested_frame_id
        elif top_level_frame_id != nested_frame_id:
            raise ValueError(
                "action expected_frame_id conflicts with the top-level value"
            )
        return normalized


class ComputerToolResult(BaseModel):
    success: bool
    session_id: str
    frame_id: str | None = None
    observation: ComputerObservation | None = None
    message: str = ""
    error: str = ""


class ComputerTool(BrowserTaskSessionMixin, AbstractBaseTool):
    """Screenshot/action loop for ordinary tool-calling vision models."""

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
        return """Inspect and control an isolated temporary browser through screenshots.

        First request a screenshot without expected_frame_id. Inspect the returned
        image, then send exactly one action with that frame_id as expected_frame_id.
        Every successful call returns a fresh screenshot and frame_id; do not request
        a second screenshot after an action.

        Coordinates are normalized: (0, 0) is the viewport top-left and (1, 1)
        is the bottom-right. Prefer element_id when the observation exposes one.
        Use `type` to insert at the current caret and `replace_text` with a target
        to replace a field. Keyboard chords use keys such as ["CTRL", "A"].

        This is a new ephemeral browser. It does not inherit the user's existing
        browser profile or signed-in sessions.
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
        raw_args = dict(args)
        step_id = raw_args.pop(_STEP_SESSION_ARG, None)
        # The session is an execution-scoped resource. Never let model output
        # select another task's browser, even if it invents a session_id field.
        raw_args.pop("session_id", None)
        session_id = self._default_session_id(step_id)
        if not session_id:
            return self._error_result(
                session_id="",
                error="Computer tool requires a task-scoped browser session.",
            )
        if self._workspace is None:
            return self._error_result(
                session_id=session_id,
                error="Computer tool requires a task workspace for observations.",
            )
        try:
            parsed = ComputerToolArgs.model_validate(raw_args)
        except ValidationError as exc:
            environment = self._environments.get(session_id)
            current = environment.current_observation if environment else None
            return self._error_result(
                session_id=session_id,
                frame_id=current.frame_id if current else None,
                error=f"Invalid computer action: {exc}",
            )

        environment = self._environments.get(session_id)
        if environment is None:
            environment = self._environment_factory(
                session_id=session_id,
                workspace=self._workspace,
                headless=self._headless,
            )
            self._environments[session_id] = environment

        action = parsed.actions[0]
        screenshot_only = action.type is ComputerActionType.SCREENSHOT
        try:
            current = environment.current_observation
            if current is None:
                if not screenshot_only:
                    return self._error_result(
                        session_id=session_id,
                        error=(
                            "No browser frame exists yet. Request a screenshot "
                            "before planning another action."
                        ),
                    )
                observation = await environment.observe()
            elif screenshot_only:
                observation = await environment.observe()
            elif parsed.expected_frame_id is None:
                return self._error_result(
                    session_id=session_id,
                    frame_id=current.frame_id,
                    error=(
                        "expected_frame_id is required for state-changing "
                        "computer actions."
                    ),
                )
            else:
                observation = await environment.execute(
                    ComputerActionBatch(
                        session_id=session_id,
                        expected_frame_id=parsed.expected_frame_id,
                        actions=parsed.actions,
                    )
                )
        except (
            ComputerEnvironmentError,
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
        except Exception as exc:  # noqa: BLE001 - provider failures become tool results.
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
        result[SUPERSEDES_SCOPE_KEY] = f"{self.name}:{session_id}"
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
