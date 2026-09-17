from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
    ComputerPerceptionMode,
    ComputerTarget,
    NormalizedPoint,
)
from ....context_ref import (
    CONTEXT_REFS_KEY,
    SUPERSEDES_SCOPE_KEY,
)
from ....file_ref import (
    build_file_id_ref,
    build_workspace_file_ref,
    sanitize_file_ref_for_context,
)
from ....workspace import TaskWorkspace
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .browser_use import BrowserTaskSessionMixin

logger = logging.getLogger(__name__)

ComputerEnvironmentFactory = Callable[..., ComputerEnvironment]
_STEP_SESSION_ARG = "_xagent_step_id"


class ComputerToolArgs(BaseModel):
    """One flat computer action.

    The public schema intentionally avoids a one-element ``actions`` array.
    Several otherwise capable tool-calling models serialize that nested shape
    as a string. The pre-validator still accepts the preview shape so existing
    traces and clients remain replayable.
    """

    model_config = ConfigDict(extra="forbid")

    expected_frame_id: str | None = Field(
        default=None,
        description=(
            "Frame ID that the action was planned against. Required for every "
            "state-changing call; omit for observe or screenshot."
        ),
    )
    action: ComputerActionType = Field(
        default=ComputerActionType.OBSERVE,
        description=(
            "One action to perform. Omit or use observe for internal visual "
            "context. Use screenshot to create a user-visible image artifact."
        ),
    )
    target: ComputerTarget | None = Field(
        default=None,
        description=(
            "Target for click, double_click, move, or replace_text. Use "
            '{"element_id": "<exact id>"} for a semantic element, or '
            '{"point": {"x": <0..1>, "y": <0..1>}} for normalized coordinates.'
        ),
    )
    url: str | None = None
    text: str | None = Field(default=None, max_length=65_536)
    keys: list[str] = Field(default_factory=list, max_length=16)
    delta_x: float = Field(default=0, ge=-1, le=1)
    delta_y: float = Field(default=0, ge=-1, le=1)
    start: NormalizedPoint | None = None
    end: NormalizedPoint | None = None
    duration_ms: int = Field(default=0, ge=0, le=30_000)
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=128)

    @model_validator(mode="before")
    @classmethod
    def _normalize_preview_shapes(cls, value: Any) -> Any:
        """Lift legacy nested shapes without exposing them in the JSON schema."""

        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)

        def merge_nested_field(key: str, item: Any) -> None:
            normalized_key = "action" if key == "type" else key
            if normalized_key in normalized and normalized[normalized_key] != item:
                raise ValueError(
                    f"{normalized_key} conflicts between top-level and nested action"
                )
            normalized.setdefault(normalized_key, item)

        if "actions" in normalized:
            raw_actions = normalized.pop("actions")
            if (
                not isinstance(raw_actions, list)
                or len(raw_actions) != 1
                or not isinstance(raw_actions[0], Mapping)
            ):
                raise ValueError("actions must contain exactly one action object")
            nested = dict(raw_actions[0])
            nested_frame_id = nested.pop("expected_frame_id", None)
            if nested_frame_id is not None:
                top_level_frame_id = normalized.get("expected_frame_id")
                if top_level_frame_id is None:
                    normalized["expected_frame_id"] = nested_frame_id
                elif top_level_frame_id != nested_frame_id:
                    raise ValueError(
                        "action expected_frame_id conflicts with the top-level value"
                    )
            for key, item in nested.items():
                merge_nested_field(key, item)
        raw_action = normalized.get("action")
        if isinstance(raw_action, Mapping):
            nested = dict(raw_action)
            normalized.pop("action")
            for key, item in nested.items():
                merge_nested_field(key, item)
        if "type" in normalized:
            merge_nested_field("type", normalized.pop("type"))
        raw_target = normalized.get("target")
        if isinstance(raw_target, str):
            # Tool-calling models commonly either copy an exposed element_id
            # directly or JSON-encode the nested target object as a string. Keep
            # the canonical action protocol structured while accepting both
            # unambiguous model-facing representations at this boundary.
            try:
                decoded_target = json.loads(raw_target)
            except json.JSONDecodeError:
                normalized["target"] = {"element_id": raw_target}
            else:
                if isinstance(decoded_target, Mapping):
                    normalized["target"] = dict(decoded_target)
                elif isinstance(decoded_target, str):
                    normalized["target"] = {"element_id": decoded_target}
                else:
                    raise ValueError(
                        "target JSON must decode to an object or element ID string"
                    )
        return normalized

    def to_action(self) -> ComputerAction:
        return ComputerAction(
            type=self.action,
            target=self.target,
            url=self.url,
            text=self.text,
            keys=self.keys,
            delta_x=self.delta_x,
            delta_y=self.delta_y,
            start=self.start,
            end=self.end,
            duration_ms=self.duration_ms,
            metadata=self.metadata,
        )


class ComputerToolResult(BaseModel):
    success: bool
    session_id: str
    frame_id: str | None = None
    observation: ComputerObservation | None = None
    file_ref: dict[str, Any] | None = None
    inline_markdown: str | None = None
    delivery: Literal[
        "none",
        "private_observation",
        "user_visible_artifact",
    ] = "none"
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
        locale: str | None = None,
        environment_factory: ComputerEnvironmentFactory = BrowserComputerEnvironment,
        environment_instructions: str | None = None,
        environment_label: str = "browser",
        perception_mode: ComputerPerceptionMode | str = ComputerPerceptionMode.AUTO,
        headless: bool = True,
        environment_scope: Literal["step", "task"] = "step",
    ) -> None:
        if environment_scope not in {"step", "task"}:
            raise ValueError("environment_scope must be 'step' or 'task'")
        self._visibility = ToolVisibility.PUBLIC
        self._task_id = task_id
        self._workspace = workspace
        self._locale = locale
        self._environment_factory = environment_factory
        self._environment_label = environment_label.strip() or "computer"
        self._perception_mode = ComputerPerceptionMode(perception_mode)
        self._environment_instructions = (
            environment_instructions.strip()
            if isinstance(environment_instructions, str)
            and environment_instructions.strip()
            else (
                "This is a new ephemeral browser. It does not inherit the user's "
                "existing browser profile or signed-in sessions."
            )
        )
        self._headless = headless
        self._environment_scope = environment_scope
        self._environments: dict[str, ComputerEnvironment] = {}

    @property
    def name(self) -> str:
        return "computer"

    @property
    def description(self) -> str:
        perception_instructions = {
            ComputerPerceptionMode.VISION: (
                "Plan from the screenshot and use normalized coordinates. "
                "Ignore semantic element IDs even if an adapter includes them "
                "as auxiliary observation data."
            ),
            ComputerPerceptionMode.SEMANTIC: (
                "Prefer an exact element_id from the current observation. Use "
                "the screenshot to verify results and use coordinates only when "
                "no matching semantic element is exposed."
            ),
            ComputerPerceptionMode.AUTO: (
                "Use an exact element_id when the current observation exposes a "
                "matching semantic element; otherwise inspect the screenshot and "
                "use normalized coordinates. Never invent an element_id."
            ),
        }[self._perception_mode]
        return f"""Inspect and control {self._environment_label} through fresh observations.

        First request an observation by omitting action or using action=observe.
        Inspect the returned image, then send exactly one action with that frame_id
        as expected_frame_id. Every successful call returns a fresh internal
        observation and frame_id, so do not call observe again after an action.

        observe and automatic post-action observations are private model context.
        screenshot is different: it captures the current state as a user-visible
        image artifact and returns a trusted file_ref plus exact inline_markdown.
        Use screenshot when an image must be delivered, and include the returned
        inline_markdown verbatim in final_answer rather than file_ref.markdown_link.
        Do not claim an image was delivered unless screenshot returned that public
        artifact.

        For state-changing actions, the latest observation's
        `metadata.supported_actions` list is authoritative. Never call a
        state-changing action that is absent. For browser URL changes, call the
        atomic `navigate` action when it is present; never simulate navigation by
        clicking, typing, or pressing keys in the address bar. If `navigate` is
        absent, explain that this exact window cannot navigate safely.

        Perception mode is {self._perception_mode.value}. {perception_instructions}

        Coordinates are normalized: (0, 0) is the viewport top-left and (1, 1)
        is the bottom-right.
        For a semantic target, send
        `target={{"element_id": "<exact element_id>"}}`.
        Semantic elements may expose a `surface` provenance field.
        `application_chrome` means browser/application controls, not website
        content. If a document goal has no matching `document` element, use the
        screenshot and a coordinate instead of a similarly named application
        control. For a coordinate target,
        send `target={{"point": {{"x": 0.5, "y": 0.5}}}}`. Models should emit
        these structured forms; the tool boundary only normalizes unambiguous
        bare or JSON-encoded element IDs for compatibility. Do not send null as
        the target of a target-requiring action.
        Use `type` to insert at the current caret and `replace_text` with a target
        to replace a field. Keyboard chords use keys such as ["CTRL", "A"].

        {self._environment_instructions}
        """

    @property
    def tags(self) -> list[str]:
        return ["computer-use", "vision", "automation"]

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
        session_id = self._default_session_id(
            step_id if self._environment_scope == "step" else None
        )
        if not session_id:
            return self._error_result(
                session_id="",
                error="Computer tool requires a task-scoped session.",
            )
        if self._workspace is None:
            return self._error_result(
                session_id=session_id,
                error="Computer tool requires a task workspace for observations.",
            )
        try:
            parsed = ComputerToolArgs.model_validate(raw_args)
            action = parsed.to_action()
        except ValidationError as exc:
            environment = self._environments.get(session_id)
            current = environment.current_observation if environment else None
            return self._error_result(
                session_id=session_id,
                frame_id=current.frame_id if current else None,
                error=f"Invalid computer action: {exc}",
            )

        observation_only = action.type in {
            ComputerActionType.OBSERVE,
            ComputerActionType.SCREENSHOT,
        }
        environment = self._environments.get(session_id)
        try:
            if environment is None:
                factory_kwargs: dict[str, Any] = {
                    "session_id": session_id,
                    "workspace": self._workspace,
                    "headless": self._headless,
                }
                # Only the default Playwright-backed factory understands
                # `locale` -- the native-local-browser factory (a functools.partial
                # around _authorized_native_browser_environment /
                # NativeBrowserEnvironment) drives the user's own already-running
                # browser and has no `locale`/`**kwargs` parameter, so passing it
                # there would raise TypeError instead of silently doing nothing.
                if self._locale:
                    if self._environment_factory is BrowserComputerEnvironment:
                        factory_kwargs["locale"] = self._locale
                    else:
                        logger.debug(
                            "Dropping resolved locale %r for computer tool "
                            "session %r: environment factory %r doesn't accept "
                            "a locale kwarg",
                            self._locale,
                            session_id,
                            self._environment_factory,
                        )
                environment = self._environment_factory(**factory_kwargs)
                self._environments[session_id] = environment
            current = environment.current_observation
            if current is None:
                if not observation_only:
                    return self._error_result(
                        session_id=session_id,
                        error=(
                            "No computer frame exists yet. Request an observation "
                            "before planning another action."
                        ),
                    )
                observation = await environment.observe()
            elif observation_only:
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
                supported_actions = current.metadata.get("supported_actions")
                if (
                    isinstance(supported_actions, list)
                    and action.type.value not in supported_actions
                ):
                    unsupported_actions = current.metadata.get("unsupported_actions")
                    reason = (
                        unsupported_actions.get(action.type.value)
                        if isinstance(unsupported_actions, Mapping)
                        else None
                    )
                    detail = f": {reason}" if reason else ""
                    return self._error_result(
                        session_id=session_id,
                        frame_id=current.frame_id,
                        error=(
                            f"{action.type.value} is not supported by the current "
                            f"computer observation{detail}. Do not retry or work "
                            "around this capability through another tool."
                        ),
                    )
                observation = await environment.execute(
                    ComputerActionBatch(
                        session_id=session_id,
                        expected_frame_id=parsed.expected_frame_id,
                        actions=[action],
                    )
                )
        except (
            ComputerEnvironmentError,
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
        except Exception as exc:  # noqa: BLE001 - provider failures become tool results.
            current = environment.current_observation if environment else None
            return self._error_result(
                session_id=session_id,
                frame_id=current.frame_id if current else None,
                error=f"Computer action failed: {exc}",
            )

        public_file_ref: dict[str, Any] | None = None
        inline_markdown: str | None = None
        if action.type is ComputerActionType.SCREENSHOT:
            try:
                public_file_ref = await asyncio.to_thread(
                    self._publish_screenshot,
                    observation,
                )
            except Exception as exc:  # noqa: BLE001 - tool boundary returns errors.
                return self._error_result(
                    session_id=session_id,
                    frame_id=observation.frame_id,
                    error=f"Could not publish computer screenshot: {exc}",
                )
            filename = str(public_file_ref["filename"])
            inline_markdown = (
                f"![{filename}]({build_file_id_ref(str(public_file_ref['file_id']))})"
            )

        result = ComputerToolResult(
            success=True,
            session_id=session_id,
            frame_id=observation.frame_id,
            observation=observation,
            file_ref=public_file_ref,
            inline_markdown=inline_markdown,
            delivery=(
                "user_visible_artifact" if inline_markdown else "private_observation"
            ),
            message=(
                (
                    "Screenshot captured as a user-visible artifact. Include "
                    "inline_markdown rather than file_ref.markdown_link in the "
                    f"final answer: {inline_markdown}"
                )
                if inline_markdown
                else (
                    "Private computer observation captured for frame "
                    f"{observation.frame_id}. It is model context only, not a "
                    "user-visible image or delivered artifact. Use this exact "
                    "frame_id for the next state-changing action. If the user "
                    "needs the current view as an output, call screenshot next."
                )
            ),
        ).model_dump(mode="json", exclude_none=True)
        result[CONTEXT_REFS_KEY] = (
            []
            if observation.metadata.get("screenshot_fresh") is False
            else [observation.screenshot.durable_dict()]
        )
        result[SUPERSEDES_SCOPE_KEY] = f"{self.name}:{session_id}"
        return result

    def _publish_screenshot(
        self,
        observation: ComputerObservation,
    ) -> dict[str, Any]:
        """Copy one internal observation into the user-visible output space."""

        if observation.metadata.get("screenshot_fresh") is False:
            raise RuntimeError(
                "a fresh screenshot is temporarily unavailable; "
                "the observation contains fresh semantic state but only a reused "
                "image. Wait for a fresh observation before publishing a screenshot."
            )
        workspace = self._workspace
        if workspace is None:
            raise ValueError("Computer screenshot requires a task workspace.")
        resolve_file_id = getattr(workspace, "resolve_file_id", None)
        if not callable(resolve_file_id):
            raise TypeError("workspace cannot resolve computer observation files")
        source = resolve_file_id(observation.screenshot.file_id)
        if source is None:
            raise FileNotFoundError("computer observation screenshot is unavailable")
        source_path = Path(source).resolve()
        output_dir = Path(workspace.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_key = hashlib.sha256(observation.frame_id.encode("utf-8")).hexdigest()[
            :12
        ]
        suffix = source_path.suffix or ".png"
        destination = output_dir / f"computer-screenshot-{frame_key}{suffix}"
        shutil.copy2(source_path, destination)
        file_ref = build_workspace_file_ref(
            workspace=workspace,
            file_path=destination,
            mime_type=str(
                observation.screenshot.file_ref.get("mime_type") or "image/png"
            ),
        )
        return sanitize_file_ref_for_context(file_ref)

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
