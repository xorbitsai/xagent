from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from ....computer.browser import BrowserComputerEnvironment
from ....computer.environment import (
    ComputerEnvironment,
    ComputerFrameMismatchError,
    ComputerSessionMismatchError,
    ComputerTargetNotFoundError,
)
from ....computer.extension import ExtensionComputerEnvironment
from ....computer.policy import (
    ComputerActionPolicy,
    ComputerPolicyDecision,
    ComputerPolicyOutcome,
    ComputerRiskLevel,
    DefaultComputerActionPolicy,
    find_computer_target_element,
)
from ....computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerObservation,
)
from ....computer.session import BrowserRuntimeKind, ComputerSessionBinding
from ....context_ref import CONTEXT_REFS_KEY
from ....workspace import TaskWorkspace
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .browser_use import BrowserTaskSessionMixin

logger = logging.getLogger(__name__)

ComputerEnvironmentFactory = Callable[..., ComputerEnvironment]
_COMPUTER_APPROVAL_ARG = "_xagent_computer_approval"


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


class ComputerConfirmationRequest(BaseModel):
    confirmation_id: str
    kind: Literal["computer_action_confirmation", "computer_user_takeover"]
    risk: ComputerRiskLevel
    reason: str
    action_indexes: list[int] = Field(default_factory=list)
    action_summary: str


class ComputerToolResult(BaseModel):
    success: bool
    session_id: str
    browser_runtime_kind: BrowserRuntimeKind
    status: str | None = None
    message_type: str | None = None
    frame_id: str | None = None
    observation: ComputerObservation | None = None
    policy_decision: ComputerPolicyDecision | None = None
    confirmation: ComputerConfirmationRequest | None = None
    interactions: list[dict[str, Any]] | None = None
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
        action_policy: ComputerActionPolicy | None = None,
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
        self._action_policy = action_policy or DefaultComputerActionPolicy()
        self._environments: dict[str, ComputerEnvironment] = {}
        self._approved_confirmation: dict[str, str] | None = None

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

        Some actions require user approval. If computer returns
        status=waiting_for_user, execution pauses automatically. After the user
        approves, call computer again with exactly the same expected_frame_id and
        action. The runtime supplies a one-use approval and refreshes the browser
        state before executing. Never alter or invent approval fields.
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

    def authorize_confirmation(
        self,
        *,
        confirmation_id: str,
        decision: str,
        session_id: str,
    ) -> None:
        """Accept a trusted runtime grant for one subsequent tool call."""
        if not confirmation_id or decision != "approve" or not session_id:
            self._approved_confirmation = None
            return
        self._approved_confirmation = {
            "confirmation_id": confirmation_id,
            "decision": decision,
            "session_id": session_id,
        }

    async def run_json_async(self, args: Mapping[str, Any]) -> dict[str, Any]:
        approval = self._approved_confirmation
        self._approved_confirmation = None
        prepared_args = self._with_default_session(args)
        prepared_args.pop(_COMPUTER_APPROVAL_ARG, None)
        if approval is not None:
            prepared_args["session_id"] = approval["session_id"]
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
                    if approval is not None and parsed.expected_frame_id is not None:
                        policy_result = await self._execute_with_policy(
                            environment=environment,
                            batch=ComputerActionBatch(
                                session_id=session_id,
                                expected_frame_id=parsed.expected_frame_id,
                                actions=parsed.actions,
                            ),
                            approval=approval,
                        )
                        if isinstance(policy_result, dict):
                            return policy_result
                        observation = policy_result
                    else:
                        return self._error_result(
                            session_id=session_id,
                            error=(
                                "No browser frame exists yet. Call computer with only "
                                "a screenshot action before planning other actions."
                            ),
                        )
                else:
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
                batch = ComputerActionBatch(
                    session_id=session_id,
                    expected_frame_id=parsed.expected_frame_id,
                    actions=parsed.actions,
                )
                policy_result = await self._execute_with_policy(
                    environment=environment,
                    batch=batch,
                    approval=approval,
                )
                if isinstance(policy_result, dict):
                    return policy_result
                observation = policy_result
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
        preserve_for_user = execution_status == "waiting_for_user" or (
            self._browser_runtime_kind
            in {
                BrowserRuntimeKind.PERSISTENT_PLAYWRIGHT,
                BrowserRuntimeKind.EXTENSION_RELAY,
            }
            and execution_status == "interrupted"
        )
        if preserve_for_user:
            return
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

    async def _execute_with_policy(
        self,
        *,
        environment: ComputerEnvironment,
        batch: ComputerActionBatch,
        approval: dict[str, str] | None,
    ) -> ComputerObservation | dict[str, Any]:
        current = environment.current_observation
        if current is None:
            if approval is None:
                return self._error_result(
                    session_id=batch.session_id,
                    error=(
                        "No browser frame exists yet. Call computer with only a "
                        "screenshot action before planning other actions."
                    ),
                )
            fresh = await environment.observe()
            return self._stale_approval_result(
                observation=fresh,
                error=(
                    "The approved browser frame is no longer available after "
                    "resume. Re-plan from this fresh screenshot."
                ),
            )

        decision = await self._action_policy.evaluate(batch, current)
        confirmation_id = self._confirmation_id(batch)
        if decision.outcome is ComputerPolicyOutcome.BLOCK:
            return self._takeover_result(
                observation=current,
                batch=batch,
                decision=decision,
                confirmation_id=confirmation_id,
            )
        if decision.outcome is ComputerPolicyOutcome.REQUIRE_CONFIRMATION:
            if (
                approval is None
                or approval.get("confirmation_id") != confirmation_id
                or approval.get("decision") != "approve"
            ):
                return self._confirmation_result(
                    observation=current,
                    batch=batch,
                    decision=decision,
                    confirmation_id=confirmation_id,
                )

            fresh = await environment.observe()
            if not self._approval_frame_matches(
                previous=current,
                fresh=fresh,
                actions=batch.actions,
            ):
                return self._stale_approval_result(
                    observation=fresh,
                    error=(
                        "The browser page or approved target changed while waiting "
                        "for confirmation. The action was not executed; re-plan "
                        "from this fresh screenshot."
                    ),
                )
            batch = batch.model_copy(update={"expected_frame_id": fresh.frame_id})

        return await environment.execute(batch)

    def _confirmation_result(
        self,
        *,
        observation: ComputerObservation,
        batch: ComputerActionBatch,
        decision: ComputerPolicyDecision,
        confirmation_id: str,
    ) -> dict[str, Any]:
        summary = self._action_summary(batch.actions, observation)
        page = self._page_label(observation.active_url)
        message = (
            f"Xagent wants to {summary}{f' on {page}' if page else ''}. "
            f"{decision.reason} Approve this exact action once, or deny it."
        )
        confirmation = ComputerConfirmationRequest(
            confirmation_id=confirmation_id,
            kind="computer_action_confirmation",
            risk=decision.risk,
            reason=decision.reason,
            action_indexes=decision.action_indexes,
            action_summary=summary,
        )
        return ComputerToolResult(
            success=False,
            status="waiting_for_user",
            message_type="confirmation",
            session_id=batch.session_id,
            browser_runtime_kind=self._browser_runtime_kind,
            frame_id=observation.frame_id,
            policy_decision=decision,
            confirmation=confirmation,
            interactions=[
                {
                    "type": "action_cards",
                    "field": "computer_action_decision",
                    "label": "Computer action decision",
                    "options": [
                        {
                            "label": "Approve this action",
                            "value": "approve",
                            "description": "Authorize this exact action once.",
                        },
                        {
                            "label": "Deny this action",
                            "value": "deny",
                            "description": "Do not perform the proposed action.",
                        },
                    ],
                }
            ],
            message=message,
        ).model_dump(mode="json", exclude_none=True)

    def _takeover_result(
        self,
        *,
        observation: ComputerObservation,
        batch: ComputerActionBatch,
        decision: ComputerPolicyDecision,
        confirmation_id: str,
    ) -> dict[str, Any]:
        summary = self._action_summary(batch.actions, observation)
        message = (
            f"Xagent will not automate this action: {decision.reason} "
            "Take control of the approved browser tab, complete the sensitive "
            "step yourself, then tell Xagent to continue. A fresh screenshot "
            "will be required."
        )
        confirmation = ComputerConfirmationRequest(
            confirmation_id=confirmation_id,
            kind="computer_user_takeover",
            risk=decision.risk,
            reason=decision.reason,
            action_indexes=decision.action_indexes,
            action_summary=summary,
        )
        return ComputerToolResult(
            success=False,
            status="waiting_for_user",
            message_type="warning",
            session_id=batch.session_id,
            browser_runtime_kind=self._browser_runtime_kind,
            frame_id=observation.frame_id,
            policy_decision=decision,
            confirmation=confirmation,
            interactions=[
                {
                    "type": "action_cards",
                    "field": "computer_takeover_decision",
                    "label": "User takeover",
                    "options": [
                        {
                            "label": "I completed the sensitive step",
                            "value": "completed",
                            "description": "Continue from a fresh screenshot.",
                        },
                        {
                            "label": "Cancel this action",
                            "value": "cancel",
                            "description": "Continue without this action.",
                        },
                    ],
                }
            ],
            message=message,
        ).model_dump(mode="json", exclude_none=True)

    def _stale_approval_result(
        self,
        *,
        observation: ComputerObservation,
        error: str,
    ) -> dict[str, Any]:
        result = ComputerToolResult(
            success=False,
            status="stale_approval",
            session_id=observation.session_id,
            browser_runtime_kind=self._browser_runtime_kind,
            frame_id=observation.frame_id,
            observation=observation,
            message=error,
            error=error,
        ).model_dump(mode="json", exclude_none=True)
        result[CONTEXT_REFS_KEY] = [observation.screenshot.durable_dict()]
        return result

    @staticmethod
    def _confirmation_id(batch: ComputerActionBatch) -> str:
        payload = json.dumps(
            batch.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _approval_frame_matches(
        cls,
        *,
        previous: ComputerObservation,
        fresh: ComputerObservation,
        actions: list[ComputerAction],
    ) -> bool:
        if (
            previous.active_url != fresh.active_url
            or previous.viewport != fresh.viewport
        ):
            return False
        for action in actions:
            previous_element = find_computer_target_element(action, previous)
            fresh_element = find_computer_target_element(action, fresh)
            if previous_element is not None or fresh_element is not None:
                if previous_element is None or fresh_element is None:
                    return False
                if cls._element_signature(previous_element) != cls._element_signature(
                    fresh_element
                ):
                    return False
                continue
            previous_hash = previous.screenshot.metadata.get("sha256")
            fresh_hash = fresh.screenshot.metadata.get("sha256")
            if (
                not isinstance(previous_hash, str)
                or not previous_hash
                or previous_hash != fresh_hash
            ):
                return False
        return True

    @staticmethod
    def _element_signature(element: ComputerElement) -> tuple[Any, ...]:
        bounds = element.bounds
        return (
            element.element_id,
            element.label,
            element.role,
            element.text,
            round(bounds.x, 4),
            round(bounds.y, 4),
            round(bounds.width, 4),
            round(bounds.height, 4),
            bool(element.metadata.get("sensitive")),
            bool(element.metadata.get("focused")),
            str(element.metadata.get("input_type") or ""),
        )

    @staticmethod
    def _action_summary(
        actions: list[ComputerAction],
        observation: ComputerObservation,
    ) -> str:
        action = actions[0]
        element = find_computer_target_element(action, observation)
        target = ""
        if element is not None:
            label = element.label or element.text or element.role
            if label:
                target = f" “{label[:120]}”"
        if action.type is ComputerActionType.KEYPRESS:
            keys = "+".join(key.upper() for key in action.keys)
            return f"press {keys or 'a key'}"
        if action.type is ComputerActionType.DRAG:
            return "drag content"
        if action.type is ComputerActionType.TYPE:
            return f"type text into{target or ' the current field'}"
        return f"{action.type.value.replace('_', ' ')}{target}"

    @staticmethod
    def _page_label(active_url: str | None) -> str:
        if not active_url:
            return ""
        parsed = urlsplit(active_url)
        if parsed.hostname:
            return parsed.hostname
        return parsed.scheme or ""

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
