from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ....computer.browser import BrowserComputerEnvironment
from ....computer.environment import (
    ComputerEnvironment,
    ComputerFrameMismatchError,
    ComputerSessionMismatchError,
    ComputerTargetNotFoundError,
)
from ....computer.extension import ExtensionComputerEnvironment
from ....computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerObservation,
)
from ....computer.session import BrowserRuntimeKind, ComputerSessionBinding
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
    browser_runtime_kind: BrowserRuntimeKind
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
        environment_factory: ComputerEnvironmentFactory | None = None,
        headless: bool = True,
        browser_runtime_kind: BrowserRuntimeKind | str = (
            BrowserRuntimeKind.EPHEMERAL_PLAYWRIGHT
        ),
        user_id: int | None = None,
        browser_profile_id: str = "default",
        browser_profile_root: Path | None = None,
    ) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id
        self._workspace = workspace
        self._headless = headless
        self._browser_runtime_kind = BrowserRuntimeKind(browser_runtime_kind)
        self._environment_factory = environment_factory or (
            ExtensionComputerEnvironment
            if self._browser_runtime_kind is BrowserRuntimeKind.EXTENSION_RELAY
            else BrowserComputerEnvironment
        )
        self._user_id = user_id
        self._browser_profile_id = browser_profile_id
        self._browser_profile_root = browser_profile_root
        self._environments: dict[str, ComputerEnvironment] = {}

    @property
    def name(self) -> str:
        return "computer"

    @property
    def description(self) -> str:
        description = """Inspect and control one browser through screenshots.

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
        if self._browser_runtime_kind is BrowserRuntimeKind.PERSISTENT_PLAYWRIGHT:
            description += """

        This task uses a visible persistent browser profile. If login, CAPTCHA,
        passkey, or two-factor authentication requires the user, do not ask for credentials.
        Ask the user to take control of the visible browser and
        complete the step, wait for their response, then request a fresh
        screenshot before continuing.
        """
        elif self._browser_runtime_kind is BrowserRuntimeKind.EXTENSION_RELAY:
            description += """

        This task controls only the browser tab that the user explicitly approved
        through the Xagent Chrome extension. If the extension is disconnected,
        no tab is attached, or login/CAPTCHA/passkey/two-factor authentication
        needs the user, do not ask for credentials. Ask the user to connect the
        extension or take control of that tab, wait for their response, then
        request a fresh screenshot before continuing.
        """
        return description

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
        prepared_args = self._with_default_session(args)
        if self._browser_runtime_kind is not BrowserRuntimeKind.EPHEMERAL_PLAYWRIGHT:
            # User-controlled browsers are authenticated task resources, not
            # model-selected browser namespaces.
            prepared_args["session_id"] = self._default_session_id()
        parsed = ComputerToolArgs.model_validate(prepared_args)
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
        try:
            if environment is None:
                environment = self._environment_factory(
                    session_id=session_id,
                    workspace=self._workspace,
                    headless=self._headless,
                    session_binding=self._session_binding(session_id),
                )
                self._environments[session_id] = environment

            screenshot_only = all(
                action.type == ComputerActionType.SCREENSHOT
                for action in parsed.actions
            )
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
            current = environment.current_observation if environment else None
            return self._error_result(
                session_id=session_id,
                frame_id=current.frame_id if current else None,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - provider errors become tool failures.
            current = environment.current_observation if environment else None
            return self._error_result(
                session_id=session_id,
                frame_id=current.frame_id if current else None,
                error=f"Browser computer action failed: {exc}",
            )

        result = ComputerToolResult(
            success=True,
            session_id=session_id,
            browser_runtime_kind=self._browser_runtime_kind,
            frame_id=observation.frame_id,
            observation=observation,
            message=(
                f"Browser observation captured for frame {observation.frame_id}. "
                "Use this exact frame_id for the next state-changing action."
            ),
        ).model_dump(mode="json", exclude_none=True)
        result[CONTEXT_REFS_KEY] = [observation.screenshot.durable_dict()]
        return result

    async def teardown(
        self,
        task_id: str | None = None,
        execution_status: str | None = None,
    ) -> None:
        preserve_for_user = self._browser_runtime_kind in {
            BrowserRuntimeKind.PERSISTENT_PLAYWRIGHT,
            BrowserRuntimeKind.EXTENSION_RELAY,
        } and execution_status in {"interrupted", "waiting_for_user"}
        environments = list(self._environments.values())
        self._environments.clear()
        if not preserve_for_user:
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

    def _session_binding(self, logical_session_id: str) -> ComputerSessionBinding:
        workspace_user_id = getattr(self._workspace, "owner_user_id", None)
        user_id = self._user_id
        if user_id is None and isinstance(workspace_user_id, int):
            user_id = workspace_user_id
        return ComputerSessionBinding.from_values(
            runtime_kind=self._browser_runtime_kind,
            owner_task_id=self._task_id or logical_session_id,
            user_id=user_id,
            profile_id=self._browser_profile_id,
            profile_root=self._browser_profile_root,
        )

    def _error_result(
        self,
        *,
        session_id: str,
        error: str,
        frame_id: str | None = None,
    ) -> dict[str, Any]:
        return ComputerToolResult(
            success=False,
            session_id=session_id,
            browser_runtime_kind=self._browser_runtime_kind,
            frame_id=frame_id,
            error=error,
        ).model_dump(mode="json", exclude_none=True)
