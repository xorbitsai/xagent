from __future__ import annotations

import asyncio
import math
import re
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ...config import (
    SUPPORTED_NATIVE_BROWSER_APP_NAMES,
    get_browser_cua_driver_max_elements,
    get_native_browser_app_name,
    get_native_browser_enabled,
)
from .cua_driver import (
    CuaDriverClientProtocol,
    CuaDriverError,
    CuaDriverMCPClient,
    CuaDriverResult,
)
from .environment import (
    ComputerEnvironment,
    ComputerFrameMismatchError,
    ComputerTargetNotFoundError,
)
from .input_platform import (
    computer_input_metadata,
    host_computer_input_platform,
)
from .native_navigation import (
    NativeBrowserNavigator,
    default_native_browser_navigator,
)
from .schema import (
    COMPUTER_CONTROL_METADATA_KEY,
    COMPUTER_FRAME_ID_METADATA_KEY,
    COMPUTER_PERCEPTION_METADATA_KEY,
    COMPUTER_SESSION_ID_METADATA_KEY,
    ELEMENT_EXTRACTION_FAILED_KEY,
    ELEMENT_EXTRACTION_INCOMPLETE_KEY,
    ELEMENTS_TRUNCATED_KEY,
    MAX_OBSERVATION_ELEMENTS,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerControlTransport,
    ComputerElement,
    ComputerElementSource,
    ComputerElementSurface,
    ComputerEnvironmentType,
    ComputerObservation,
    ComputerPerceptionMode,
    NormalizedPoint,
    NormalizedRect,
    Viewport,
)
from .store import ObservationStore

_BASE_SUPPORTED_ACTIONS = tuple(
    action
    for action in ComputerActionType
    if action not in {ComputerActionType.MOVE, ComputerActionType.NAVIGATE}
)
_UNSCOPED_KEYBOARD_ACTIONS = frozenset(
    {
        ComputerActionType.TYPE,
        ComputerActionType.KEYPRESS,
    }
)
_UNSCOPED_KEYBOARD_REFUSAL_REASON = "unscoped_keyboard_input_disabled"
_ACTION_RESULT_FIELDS = (
    "path",
    "effect",
    "verified",
    "escalation",
    "status",
    "code",
)
_SUPPORTED_BROWSER_APP_NAMES = frozenset(
    name.casefold() for name in SUPPORTED_NATIVE_BROWSER_APP_NAMES
)
_AX_DOCUMENT_ROOT_ROLES = frozenset({"axwebarea", "webarea"})
_AX_OVERLAY_ROLES = frozenset(
    {
        "axdialog",
        "axdrawer",
        "axmenu",
        "axpopover",
        "axsheet",
        "dialog",
        "drawer",
        "menu",
        "popover",
        "sheet",
    }
)
_AX_WINDOW_ROLES = frozenset({"axwindow", "window"})
_ACTIONABLE_ROLE_MARKERS = (
    "button",
    "checkbox",
    "combobox",
    "field",
    "link",
    "menuitem",
    "popup",
    "radio",
    "row",
    "slider",
    "switch",
    "tab",
)
_PIXEL_FRAME_ERROR_MARKERS = (
    "not one coherent 1x/2x frame",
    "lies outside window",
    "px_frame_mismatch",
)
_POST_ACTION_SETTLE_SECONDS = 0.25
_POST_ACTION_SETTLE_RETRIES = 3
_SEMANTIC_SETTLE_RADIUS = 0.06
_SENSITIVE_AUTOCOMPLETE_VALUES = frozenset(
    {
        "cc-csc",
        "current-password",
        "new-password",
        "one-time-code",
    }
)
_SENSITIVE_LABEL_MARKERS = (
    "password",
    "passcode",
    "security code",
    "verification code",
    "one-time code",
    "密码",
    "口令",
    "验证码",
    "安全码",
    "动态码",
)
_SENSITIVE_LABEL_TOKEN_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:otp|2fa|mfa|cvv|cvc|pin|ssn)(?:$|[^a-z0-9])"
)

CuaDriverClientFactory = Callable[[], CuaDriverClientProtocol]
LOCAL_BROWSER_TASK_EXTENSION = "local_browser"


@dataclass(frozen=True)
class NativeBrowserWindow:
    pid: int
    window_id: int
    app_name: str
    title: str | None
    x: float
    y: float
    width: float
    height: float
    z_index: int | None
    is_on_screen: bool
    on_current_space: bool | None


class NativeBrowserEnvironment(ComputerEnvironment):
    """Control one configured browser window through cua-driver's native tools.

    A caller may provide one exact ``(pid, window_id)`` selected by the user.
    Otherwise the first observation chooses the frontmost visible window of the
    configured browser. The resulting binding is sticky: if the window closes,
    the environment fails instead of silently taking over another window.
    """

    def __init__(
        self,
        *,
        session_id: str,
        workspace: Any,
        driver: CuaDriverClientProtocol | None = None,
        driver_factory: CuaDriverClientFactory | None = None,
        observation_store: ObservationStore | None = None,
        target_pid: int | None = None,
        target_window_id: int | None = None,
        browser_app_name: str | None = None,
        perception_mode: ComputerPerceptionMode | str = ComputerPerceptionMode.AUTO,
        max_elements: int | None = None,
        headless: bool = False,
        native_browser_navigator: NativeBrowserNavigator | None = None,
    ) -> None:
        del headless
        super().__init__(session_id)
        if driver is not None and driver_factory is not None:
            raise ValueError("provide either driver or driver_factory, not both")
        if driver is None and not get_native_browser_enabled():
            raise RuntimeError(
                "Local browser access is disabled. Set "
                "XAGENT_NATIVE_BROWSER_ENABLED=true only on a trusted "
                "interactive Xagent host."
            )
        self.workspace = workspace
        self.observation_store = observation_store or ObservationStore(workspace)
        if (target_pid is None) != (target_window_id is None):
            raise ValueError(
                "target_pid and target_window_id must be provided together"
            )
        self.target_pid = target_pid
        self.target_window_id = target_window_id
        self.browser_app_name = (
            browser_app_name or get_native_browser_app_name()
        ).strip()
        if not self.browser_app_name:
            raise ValueError("local browser app name must not be empty")
        self.perception_mode = ComputerPerceptionMode(perception_mode)
        self.max_elements = (
            get_browser_cua_driver_max_elements()
            if max_elements is None
            else max_elements
        )
        if self.max_elements <= 0:
            raise ValueError("local browser max_elements must be positive")
        self._driver = driver
        self._driver_factory = driver_factory or CuaDriverMCPClient
        self._target: NativeBrowserWindow | None = None
        self._on_screen_windows: list[NativeBrowserWindow] = []
        self._session_started = False
        self._last_action_result: dict[str, Any] | None = None
        self._unsupported_action_reasons: dict[str, str] = {}
        self._known_element_signatures: set[tuple[Any, ...]] = set()
        self._native_browser_navigator = (
            native_browser_navigator or default_native_browser_navigator()
        )

    async def _close(self) -> None:
        driver = self._driver
        if driver is None:
            return
        if self._session_started:
            try:
                await driver.call_tool("end_session", {"session": self.session_id})
            except Exception:
                # Closing stdin still tears down process-owned driver state.
                pass
        await driver.close()
        self._driver = None
        self._session_started = False

    async def _observe(self) -> ComputerObservation:
        await self._ensure_session()
        if self._target is None:
            self._target = await self._select_target()
        else:
            await self._refresh_bound_target()
        return await self._capture_observation()

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        if len(batch.actions) != 1:
            raise ValueError("local browser executes exactly one action per frame")
        action = batch.actions[0]
        supported_actions = (
            self.current_observation.metadata.get("supported_actions")
            if self.current_observation is not None
            else None
        )
        if self.current_observation is not None and (
            not isinstance(supported_actions, list)
            or action.type.value not in supported_actions
        ):
            reason = self._unsupported_action_reasons.get(action.type.value)
            detail = f": {reason}" if reason else ""
            raise ValueError(
                f"{action.type.value} is not supported by the local browser runtime"
                f"{detail}"
            )
        await self._refresh_bound_target()
        self._validate_pixel_frame(action)
        try:
            await self._execute_action(action)
        except CuaDriverError as exc:
            if any(marker in str(exc) for marker in _PIXEL_FRAME_ERROR_MARKERS):
                raise ComputerFrameMismatchError(
                    "The selected window's pixel geometry changed after this "
                    "frame was captured. Request a fresh observation and plan "
                    "from its new frame_id; do not retry coordinates from the "
                    "invalidated frame."
                ) from exc
            raise
        if action.type not in {
            ComputerActionType.OBSERVE,
            ComputerActionType.SCREENSHOT,
            ComputerActionType.WAIT,
        }:
            await asyncio.sleep(_POST_ACTION_SETTLE_SECONDS)
        await self._refresh_bound_target()
        observation = await self._capture_observation()
        return await self._settle_unverified_action(
            action=action,
            previous=self.current_observation,
            observation=observation,
        )

    async def _settle_unverified_action(
        self,
        *,
        action: ComputerAction,
        previous: ComputerObservation | None,
        observation: ComputerObservation,
    ) -> ComputerObservation:
        """Wait briefly for asynchronous native UI state to become observable."""

        if previous is None or action.type in {
            ComputerActionType.OBSERVE,
            ComputerActionType.SCREENSHOT,
            ComputerActionType.WAIT,
        }:
            return observation
        action_result = observation.metadata.get("last_action_result")
        semantic_target_point = self._semantic_target_point(action, previous)
        include_screenshot = semantic_target_point is None
        previous_signature = self._observation_state_signature(
            previous,
            include_screenshot=include_screenshot,
            target_point=semantic_target_point,
        )
        state_changed = self._post_action_state_changed(
            action=action,
            previous=previous,
            observation=observation,
            previous_signature=previous_signature,
            include_screenshot=include_screenshot,
            semantic_target_point=semantic_target_point,
        )
        if (
            not isinstance(action_result, Mapping)
            or action_result.get("effect") != "unverifiable"
            or state_changed
        ):
            return observation

        for _attempt in range(_POST_ACTION_SETTLE_RETRIES):
            await asyncio.sleep(_POST_ACTION_SETTLE_SECONDS)
            await self._refresh_bound_target()
            refreshed = await self._capture_observation()
            refreshed = refreshed.model_copy(
                update={
                    "metadata": {
                        **refreshed.metadata,
                        "last_action_result": dict(action_result),
                    }
                }
            )
            observation = refreshed
            state_changed = self._post_action_state_changed(
                action=action,
                previous=previous,
                observation=observation,
                previous_signature=previous_signature,
                include_screenshot=include_screenshot,
                semantic_target_point=semantic_target_point,
            )
            if state_changed:
                break
        if (
            action.type is not ComputerActionType.NAVIGATE
            and str(action.metadata.get("delivery_mode") or "").strip().lower()
            != "foreground"
            and not state_changed
        ):
            updated_result = {
                **dict(action_result),
                "verified": False,
                "code": "no_observable_state_change_after_background_delivery",
            }
            observation = observation.model_copy(
                update={
                    "metadata": {
                        **observation.metadata,
                        "last_action_result": updated_result,
                    }
                }
            )
        return observation

    @classmethod
    def _post_action_state_changed(
        cls,
        *,
        action: ComputerAction,
        previous: ComputerObservation,
        observation: ComputerObservation,
        previous_signature: tuple[Any, ...],
        include_screenshot: bool,
        semantic_target_point: NormalizedPoint | None,
    ) -> bool:
        if observation.title != previous.title:
            return True
        if action.type is ComputerActionType.NAVIGATE:
            return False
        current_signature = cls._observation_state_signature(
            observation,
            include_screenshot=include_screenshot,
            target_point=semantic_target_point,
        )
        if semantic_target_point is None:
            return current_signature != previous_signature
        current_local_elements = current_signature[2]
        return bool(current_local_elements) and current_signature != previous_signature

    @staticmethod
    def _observation_state_signature(
        observation: ComputerObservation,
        *,
        include_screenshot: bool,
        target_point: NormalizedPoint | None,
    ) -> tuple[Any, ...]:
        elements = observation.elements
        if target_point is not None:
            elements = [
                element
                for element in elements
                if (
                    element.metadata.get("enabled") is not False
                    and any(
                        marker in (element.role or "").casefold()
                        for marker in _ACTIONABLE_ROLE_MARKERS
                    )
                    and (
                        element.surface is ComputerElementSurface.OVERLAY
                        or (
                            (
                                element.bounds.x
                                + element.bounds.width / 2
                                - target_point.x
                            )
                            ** 2
                            + (
                                element.bounds.y
                                + element.bounds.height / 2
                                - target_point.y
                            )
                            ** 2
                            <= _SEMANTIC_SETTLE_RADIUS**2
                        )
                    )
                )
            ]
        return (
            observation.title,
            (
                observation.screenshot.metadata.get("sha256")
                if include_screenshot
                else None
            ),
            tuple(
                sorted(
                    (
                        element.source.value,
                        (element.surface.value if element.surface is not None else ""),
                        element.role or "",
                        element.label or "",
                        element.text or "",
                        str(element.metadata.get("selected")),
                        str(element.metadata.get("enabled")),
                        round(element.bounds.x, 4),
                        round(element.bounds.y, 4),
                        round(element.bounds.width, 4),
                        round(element.bounds.height, 4),
                    )
                    for element in elements
                )
            ),
        )

    @staticmethod
    def _semantic_target_point(
        action: ComputerAction,
        observation: ComputerObservation,
    ) -> NormalizedPoint | None:
        target = action.target
        if target is None or target.element_id is None:
            return None
        element = next(
            (
                candidate
                for candidate in observation.elements
                if candidate.element_id == target.element_id
            ),
            None,
        )
        if element is None:
            return None
        return NormalizedPoint(
            x=element.bounds.x + element.bounds.width / 2,
            y=element.bounds.y + element.bounds.height / 2,
        )

    async def _ensure_session(self) -> None:
        if self._session_started:
            return
        await self._get_driver().call_tool(
            "start_session",
            {
                "session": self.session_id,
                "capture_scope": "window",
            },
        )
        self._session_started = True

    def _get_driver(self) -> CuaDriverClientProtocol:
        if self._driver is None:
            self._driver = self._driver_factory()
        return self._driver

    async def _select_target(self) -> NativeBrowserWindow:
        result = await self._get_driver().call_tool(
            "list_windows",
            {"on_screen_only": True},
        )
        raw_windows = result.structured.get("windows")
        if not isinstance(raw_windows, list):
            raise CuaDriverError("cua-driver list_windows returned no window list")
        windows = [
            parsed
            for raw in raw_windows
            if isinstance(raw, Mapping)
            and (parsed := self._parse_window(raw)) is not None
        ]
        visible = [
            window
            for window in windows
            if window.is_on_screen and window.on_current_space is not False
        ]
        self._on_screen_windows = visible
        browser_windows = [
            window
            for window in visible
            if window.app_name.casefold() == self.browser_app_name.casefold()
        ]
        if self.target_pid is not None and self.target_window_id is not None:
            exact = next(
                (
                    window
                    for window in browser_windows
                    if window.pid == self.target_pid
                    and window.window_id == self.target_window_id
                ),
                None,
            )
            if exact is None:
                raise ComputerTargetNotFoundError(
                    f"The selected {self.browser_app_name} window is no longer "
                    "visible. Choose the browser window again before starting "
                    "a new task."
                )
            return exact
        if not browser_windows:
            raise ComputerTargetNotFoundError(
                f"No visible {self.browser_app_name} window is available on the "
                "Xagent host."
            )
        return max(
            browser_windows,
            key=lambda window: (
                window.z_index is not None,
                window.z_index if window.z_index is not None else 0,
            ),
        )

    async def _refresh_bound_target(self) -> NativeBrowserWindow:
        target = self._require_target()
        result = await self._get_driver().call_tool(
            "list_windows",
            {"on_screen_only": True},
        )
        raw_windows = result.structured.get("windows")
        if not isinstance(raw_windows, list):
            raise CuaDriverError("cua-driver list_windows returned no window list")
        visible = [
            parsed
            for raw in raw_windows
            if isinstance(raw, Mapping)
            and (parsed := self._parse_window(raw)) is not None
            and parsed.is_on_screen
            and parsed.on_current_space is not False
        ]
        self._on_screen_windows = visible
        browser_windows = [
            window
            for window in visible
            if window.app_name.casefold() == self.browser_app_name.casefold()
        ]
        refreshed = next(
            (
                window
                for window in browser_windows
                if window.pid == target.pid and window.window_id == target.window_id
            ),
            None,
        )
        if refreshed is None:
            raise ComputerTargetNotFoundError(
                f"The selected {self.browser_app_name} window disappeared during "
                "navigation."
            )
        self._target = refreshed
        return refreshed

    @staticmethod
    def _parse_window(raw: Mapping[str, Any]) -> NativeBrowserWindow | None:
        bounds = raw.get("bounds")
        if not isinstance(bounds, Mapping):
            return None
        try:
            width = float(bounds["width"])
            height = float(bounds["height"])
            if width <= 0 or height <= 0:
                return None
            return NativeBrowserWindow(
                pid=int(raw["pid"]),
                window_id=int(raw["window_id"]),
                app_name=str(raw["app_name"]),
                title=(
                    str(raw["title"]).strip()
                    if raw.get("title") is not None and str(raw["title"]).strip()
                    else None
                ),
                x=float(bounds["x"]),
                y=float(bounds["y"]),
                width=width,
                height=height,
                z_index=NativeBrowserEnvironment._optional_z_index(raw.get("z_index")),
                is_on_screen=raw.get("is_on_screen") is True,
                on_current_space=(
                    raw.get("on_current_space")
                    if isinstance(raw.get("on_current_space"), bool)
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None

    async def _capture_observation(self) -> ComputerObservation:
        target = self._require_target()
        requested_max_elements = self.max_elements
        result = await self._get_driver().call_tool(
            "get_window_state",
            {
                "session": self.session_id,
                "pid": target.pid,
                "window_id": target.window_id,
                "include_screenshot": True,
                "max_elements": requested_max_elements,
            },
        )
        raw_elements = result.structured.get("elements")
        raw_element_list = raw_elements if isinstance(raw_elements, list) else []
        frame_id = f"frame-{uuid4().hex}"
        image_bytes = result.image_bytes
        screenshot_fresh = bool(image_bytes)
        if not screenshot_fresh and (
            self.current_observation is None
            or not raw_element_list
            or self.perception_mode is ComputerPerceptionMode.VISION
        ):
            raise CuaDriverError(
                "cua-driver could not capture the bound browser window. Check "
                "Screen Recording permission with `cua-driver health_report`."
            )
        if image_bytes:
            mime_type = result.image_mime_type or str(
                result.structured.get("screenshot_mime_type") or "image/png"
            )
            if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise CuaDriverError(
                    f"cua-driver returned unsupported screenshot type {mime_type!r}"
                )
            width, height = self._screenshot_size(result)
            viewport = Viewport(width=width, height=height, device_pixel_ratio=1.0)
            screenshot = await self.observation_store.save_screenshot(
                session_id=self.session_id,
                frame_id=frame_id,
                image_bytes=image_bytes,
                mime_type=mime_type,
                viewport=viewport,
                text_fallback=(f"Current local {target.app_name} window screenshot."),
                metadata={
                    "computer_runtime_kind": "local_browser",
                    "pid": target.pid,
                    "window_id": target.window_id,
                },
            )
        else:
            previous = self.current_observation
            if previous is None:
                raise CuaDriverError(
                    "cua-driver returned no screenshot before an observation existed"
                )
            viewport = previous.viewport
            width = viewport.width
            height = viewport.height
            screenshot = previous.screenshot.model_copy(
                update={
                    "text_fallback": (
                        "Previous exact-window screenshot reused because the "
                        "current transient UI could only be observed through "
                        "fresh accessibility elements."
                    ),
                    "metadata": {
                        **previous.screenshot.metadata,
                        COMPUTER_SESSION_ID_METADATA_KEY: self.session_id,
                        COMPUTER_FRAME_ID_METADATA_KEY: frame_id,
                        "screenshot_fresh": False,
                        "reused_from_frame_id": previous.frame_id,
                    },
                }
            )
        semantic_elements, semantic_candidate_count = self._build_elements(
            raw_element_list,
            target=target,
        )
        elements = (
            []
            if self.perception_mode is ComputerPerceptionMode.VISION
            else semantic_elements
        )
        raw_element_count = self._optional_int(result.structured.get("element_count"))
        metadata: dict[str, Any] = {
            "computer_runtime_kind": "local_browser",
            "native_driver": "cua-driver",
            "application": target.app_name,
            "pid": target.pid,
            "window_id": target.window_id,
            "screenshot_fresh": screenshot_fresh,
            "coordinate_frame": {
                "screenshot_width": width,
                "screenshot_height": height,
                "screenshot_fresh": screenshot_fresh,
                "window_bounds": self._window_bounds(target),
            },
            "user_takeover_available": True,
            "delivery_mode": "background",
            COMPUTER_PERCEPTION_METADATA_KEY: {
                "mode": self.perception_mode.value,
                "available": [
                    *(["vision"] if screenshot_fresh else []),
                    *(["semantic"] if semantic_elements else []),
                ],
                "semantic_source": "accessibility" if semantic_elements else None,
            },
            COMPUTER_CONTROL_METADATA_KEY: {
                "transport": ComputerControlTransport.NATIVE_ACCESSIBILITY.value,
                "scope": "window",
                "browser_debugging": False,
            },
            **computer_input_metadata(host_computer_input_platform()),
        }
        if result.structured.get("degraded") is True:
            metadata[ELEMENT_EXTRACTION_FAILED_KEY] = True
            metadata["driver_degraded_reason"] = str(
                result.structured.get("degraded_reason") or ""
            )
        if (
            self.perception_mode is not ComputerPerceptionMode.VISION
            and raw_element_count is not None
            and raw_element_count > len(elements)
        ):
            metadata[ELEMENT_EXTRACTION_INCOMPLETE_KEY] = True
        if (
            self.perception_mode is not ComputerPerceptionMode.VISION
            and isinstance(raw_elements, list)
            and len(elements) < len(raw_elements)
        ):
            metadata[ELEMENT_EXTRACTION_INCOMPLETE_KEY] = True
        if (
            semantic_candidate_count > MAX_OBSERVATION_ELEMENTS
            or (
                raw_element_count is not None
                and raw_element_count > requested_max_elements
            )
            or (
                isinstance(raw_elements, list)
                and len(raw_elements) >= requested_max_elements
            )
        ):
            metadata[ELEMENT_EXTRACTION_INCOMPLETE_KEY] = True
            metadata[ELEMENTS_TRUNCATED_KEY] = True
        escalation = result.structured.get("escalation")
        if isinstance(escalation, Mapping):
            metadata["driver_escalation"] = dict(escalation)
        background_input = result.structured.get("background_input")
        if isinstance(background_input, Mapping):
            metadata["background_input"] = dict(background_input)
        supported_actions, unsupported_reasons = self._supported_actions(
            background_input,
            elements=semantic_elements,
            target=target,
        )
        self._unsupported_action_reasons = unsupported_reasons
        metadata["supported_actions"] = supported_actions
        if unsupported_reasons:
            metadata["unsupported_actions"] = dict(unsupported_reasons)
        last_action_result = self._last_action_result
        if last_action_result is not None:
            metadata["last_action_result"] = last_action_result
            self._last_action_result = None
        return ComputerObservation(
            session_id=self.session_id,
            frame_id=frame_id,
            environment=ComputerEnvironmentType.DESKTOP,
            viewport=viewport,
            screenshot=screenshot,
            elements=elements,
            active_url=(
                _optional_active_url(result.structured)
                or _optional_active_url(last_action_result or {})
            ),
            title=target.title,
            metadata=metadata,
        )

    def _supported_actions(
        self,
        background_input: Any,
        *,
        elements: list[ComputerElement],
        target: NativeBrowserWindow,
    ) -> tuple[list[str], dict[str, str]]:
        keyboard_reason = self._pid_keyboard_refusal_reason(background_input)

        unsupported_reasons: dict[str, str] = {}
        supported: list[str] = []
        for action in _BASE_SUPPORTED_ACTIONS:
            if action in _UNSCOPED_KEYBOARD_ACTIONS:
                unsupported_reasons[action.value] = _UNSCOPED_KEYBOARD_REFUSAL_REASON
            else:
                supported.append(action.value)
        address_available = (
            self._browser_address_element(elements, target=target) is not None
        )
        native_navigation_available = (
            self._native_browser_navigator is not None
            and self._native_browser_navigator.supports(target)
        )
        if native_navigation_available or (
            address_available and keyboard_reason is None
        ):
            supported.append(ComputerActionType.NAVIGATE.value)
        elif address_available and keyboard_reason is not None:
            unsupported_reasons[ComputerActionType.NAVIGATE.value] = keyboard_reason
        return supported, unsupported_reasons

    @staticmethod
    def _pid_keyboard_refusal_reason(background_input: Any) -> str | None:
        if not isinstance(background_input, Mapping):
            return "pid_keyboard_capability_unknown"
        routes = background_input.get("routes")
        if not isinstance(routes, list):
            return "pid_keyboard_capability_unknown"
        for route in routes:
            if not isinstance(route, Mapping) or route.get("route") != "pid_keyboard":
                continue
            if str(route.get("status") or "").strip().lower() == "available":
                return None
            reason = str(route.get("reason") or "").strip()
            return reason or "pid_keyboard_unavailable"
        return "pid_keyboard_capability_unknown"

    def _screenshot_size(self, result: CuaDriverResult) -> tuple[int, int]:
        width = self._optional_int(result.structured.get("screenshot_width"))
        height = self._optional_int(result.structured.get("screenshot_height"))
        if width and height:
            return width, height
        image = result.image_bytes or b""
        if image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) >= 24:
            parsed_width, parsed_height = struct.unpack(">II", image[16:24])
            if parsed_width > 0 and parsed_height > 0:
                return parsed_width, parsed_height
        raise CuaDriverError("cua-driver screenshot dimensions are missing")

    def _build_elements(
        self,
        raw_elements: list[Any],
        *,
        target: NativeBrowserWindow,
    ) -> tuple[list[ComputerElement], int]:
        hierarchy = self._element_hierarchy(raw_elements)
        is_browser = target.app_name.casefold() in _SUPPORTED_BROWSER_APP_NAMES
        elements: list[ComputerElement] = []
        for raw in raw_elements:
            if not isinstance(raw, Mapping):
                continue
            frame = raw.get("frame")
            if not isinstance(frame, Mapping):
                continue
            bounds = self._normalize_element_bounds(frame, target=target)
            if bounds is None:
                continue
            token = str(raw.get("element_token") or "").strip()
            index = self._optional_int(raw.get("element_index"))
            # element_index is scoped to one driver snapshot and can be reused
            # for a different element after a worker restart. Only expose the
            # opaque token that cua-driver can validate against its snapshot.
            if not token:
                continue
            element_id = token
            role = str(raw.get("role") or "").strip() or None
            label = str(raw.get("label") or "").strip() or None
            sensitive = self._is_sensitive_element(
                raw,
                role=role,
                label=label,
            )
            value = str(raw.get("value") or "").strip() or None
            surface, surface_root_index = self._element_surface(
                raw,
                hierarchy=hierarchy,
                is_browser=is_browser,
            )
            metadata = {
                "element_index": index,
                "element_token": token or None,
                "depth": self._optional_int(raw.get("depth")),
                "parent_index": self._optional_int(raw.get("parent_index")),
                "enabled": raw.get("enabled"),
                "selected": raw.get("selected"),
                "sensitive": sensitive,
                "surface": surface.value,
                "surface_root_index": surface_root_index,
            }
            return_metadata = {
                key: value for key, value in metadata.items() if value is not None
            }
            elements.append(
                ComputerElement(
                    element_id=element_id,
                    source=ComputerElementSource.ACCESSIBILITY,
                    bounds=bounds,
                    surface=surface,
                    label="Sensitive input" if sensitive else label,
                    role=role,
                    text=None if sensitive else value,
                    metadata=return_metadata,
                )
            )
        prioritized = sorted(
            enumerate(elements),
            key=lambda item: self._element_priority(item[1], item[0]),
        )
        selected = [
            element for _index, element in prioritized[:MAX_OBSERVATION_ELEMENTS]
        ]
        self._known_element_signatures = {
            self._element_signature(element) for element in elements
        }
        return selected, len(elements)

    def _element_priority(
        self,
        element: ComputerElement,
        original_index: int,
    ) -> tuple[int, int, int]:
        role = (element.role or "").casefold()
        actionable = element.metadata.get("enabled") is not False and any(
            marker in role for marker in _ACTIONABLE_ROLE_MARKERS
        )
        is_new = self._element_signature(element) not in self._known_element_signatures
        if element.surface is ComputerElementSurface.OVERLAY:
            group = 0
        elif is_new and actionable:
            group = 1
        elif actionable and bool(element.label):
            group = 2
        elif element.label or element.text:
            group = 3
        else:
            group = 4
        return (
            group,
            0 if element.metadata.get("selected") is True else 1,
            original_index,
        )

    @staticmethod
    def _element_signature(element: ComputerElement) -> tuple[Any, ...]:
        return (
            (element.role or "").casefold(),
            element.label or "",
            element.text or "",
            element.surface.value if element.surface is not None else "",
            round(element.bounds.x, 3),
            round(element.bounds.y, 3),
            round(element.bounds.width, 3),
            round(element.bounds.height, 3),
        )

    @classmethod
    def _element_hierarchy(
        cls,
        raw_elements: list[Any],
    ) -> dict[int, tuple[int | None, int | None, str]]:
        hierarchy: dict[int, tuple[int | None, int | None, str]] = {}
        for raw in raw_elements:
            if not isinstance(raw, Mapping):
                continue
            index = cls._optional_int(raw.get("element_index"))
            if index is None:
                continue
            hierarchy[index] = (
                cls._optional_int(raw.get("parent_index")),
                cls._optional_int(raw.get("depth")),
                str(raw.get("role") or "").strip().casefold(),
            )
        return hierarchy

    @classmethod
    def _element_surface(
        cls,
        raw: Mapping[str, Any],
        *,
        hierarchy: Mapping[int, tuple[int | None, int | None, str]],
        is_browser: bool,
    ) -> tuple[ComputerElementSurface, int | None]:
        index = cls._optional_int(raw.get("element_index"))
        if index is None or index not in hierarchy:
            return ComputerElementSurface.UNKNOWN, None

        path: list[tuple[int, int | None, str]] = []
        visited: set[int] = set()
        current_index: int | None = index
        complete = False
        while current_index is not None:
            if current_index in visited:
                return ComputerElementSurface.UNKNOWN, None
            visited.add(current_index)
            node = hierarchy.get(current_index)
            if node is None:
                return ComputerElementSurface.UNKNOWN, None
            parent_index, depth, role = node
            path.append((current_index, depth, role))
            if parent_index is None:
                complete = depth == 0 or role in _AX_WINDOW_ROLES
                break
            current_index = parent_index

        document_root = next(
            (
                node_index
                for node_index, _depth, role in path
                if role in _AX_DOCUMENT_ROOT_ROLES
            ),
            None,
        )
        if document_root is not None:
            return ComputerElementSurface.DOCUMENT, document_root

        overlay_root = next(
            (
                node_index
                for node_index, _depth, role in path
                if role in _AX_OVERLAY_ROLES
            ),
            None,
        )
        if overlay_root is not None:
            return ComputerElementSurface.OVERLAY, overlay_root

        if not complete:
            return ComputerElementSurface.UNKNOWN, None
        root_index = path[-1][0]
        if is_browser:
            return ComputerElementSurface.APPLICATION_CHROME, root_index
        return ComputerElementSurface.NATIVE_APP, root_index

    @staticmethod
    def _is_sensitive_element(
        raw: Mapping[str, Any],
        *,
        role: str | None,
        label: str | None,
    ) -> bool:
        role_text = " ".join(
            str(value or "").casefold()
            for value in (role, raw.get("subrole"), raw.get("input_type"))
        )
        if any(marker in role_text for marker in ("secure", "password")):
            return True
        if any(
            raw.get(flag) is True for flag in ("sensitive", "protected", "is_password")
        ):
            return True
        autocomplete = str(raw.get("autocomplete") or "").strip().casefold()
        if autocomplete in _SENSITIVE_AUTOCOMPLETE_VALUES:
            return True
        is_text_input = any(
            marker in role_text for marker in ("text", "input", "field", "edit")
        )
        label_text = " ".join(
            str(value or "").casefold()
            for value in (
                label,
                raw.get("placeholder"),
                raw.get("description"),
            )
        )
        return is_text_input and (
            any(marker in label_text for marker in _SENSITIVE_LABEL_MARKERS)
            or _SENSITIVE_LABEL_TOKEN_RE.search(label_text) is not None
        )

    @staticmethod
    def _normalize_element_bounds(
        raw: Mapping[str, Any],
        *,
        target: NativeBrowserWindow,
    ) -> NormalizedRect | None:
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            width = float(raw["w"])
            height = float(raw["h"])
        except (KeyError, TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None

        # AX frames are normally screen-relative. Some platform backends emit
        # window-local frames, so accept that shape when subtracting the native
        # window origin would put the entire element outside the screenshot.
        local_x = x - target.x
        local_y = y - target.y
        if (
            (
                local_x + width <= 0
                or local_y + height <= 0
                or local_x >= target.width
                or local_y >= target.height
            )
            and 0 <= x < target.width
            and 0 <= y < target.height
        ):
            local_x = x
            local_y = y

        left = max(0.0, local_x)
        top = max(0.0, local_y)
        right = min(target.width, local_x + width)
        bottom = min(target.height, local_y + height)
        if right <= left or bottom <= top:
            return None
        return NormalizedRect(
            x=left / target.width,
            y=top / target.height,
            width=(right - left) / target.width,
            height=(bottom - top) / target.height,
        )

    async def _execute_action(self, action: ComputerAction) -> None:
        if action.type in {
            ComputerActionType.OBSERVE,
            ComputerActionType.SCREENSHOT,
        }:
            return
        if action.type is ComputerActionType.WAIT:
            duration_ms = action.duration_ms or 1_000
            await asyncio.sleep(duration_ms / 1_000)
            self._last_action_result = {
                "effect": "confirmed",
                "verified": True,
            }
            return
        target = self._require_target()
        common: dict[str, Any] = {
            "session": self.session_id,
            "pid": target.pid,
            "window_id": target.window_id,
            "delivery_mode": self._delivery_mode(action),
        }
        if action.type is ComputerActionType.NAVIGATE:
            await self._navigate_to_url(target, str(action.url or ""), common=common)
            return
        if action.type in _UNSCOPED_KEYBOARD_ACTIONS:
            raise ValueError(
                "Free type and keypress actions are disabled in the local browser. "
                "Use replace_text on an exact document element, or the atomic "
                "navigate action for an http/https URL."
            )
        if action.type in {
            ComputerActionType.CLICK,
            ComputerActionType.DOUBLE_CLICK,
        }:
            arguments = {**common, **self._action_target_arguments(action)}
            tool_name = (
                "double_click"
                if action.type is ComputerActionType.DOUBLE_CLICK
                else "click"
            )
            # A rejected accessibility target must fail closed. In particular,
            # element_outside_target_window means the driver could not prove
            # that the element belongs to the user-selected window. Silently
            # converting its center to a CGEvent pixel click can cross the
            # authorization boundary on mixed-display coordinate systems.
            await self._call_action(tool_name, arguments)
            return
        if action.type is ComputerActionType.REPLACE_TEXT:
            element = self._document_text_target(action)
            await self._call_action(
                "set_value",
                {
                    "session": self.session_id,
                    "pid": target.pid,
                    "window_id": target.window_id,
                    **self._element_arguments(element),
                    "value": action.text or "",
                },
            )
            return
        if action.type is ComputerActionType.SCROLL:
            if action.target is not None:
                raise ValueError(
                    "targeted scroll is not supported by the local browser runtime"
                )
            horizontal = abs(action.delta_x) > abs(action.delta_y)
            delta = action.delta_x if horizontal else action.delta_y
            direction = (
                ("right" if delta > 0 else "left")
                if horizontal
                else ("down" if delta > 0 else "up")
            )
            arguments = {
                **common,
                "direction": direction,
                "amount": max(1, min(20, math.ceil(abs(delta) * 10))),
                "by": "line",
            }
            await self._call_action("scroll", arguments)
            return
        if action.type is ComputerActionType.DRAG:
            if action.start is None or action.end is None:
                raise ValueError("drag requires start and end points")
            from_x, from_y = self._point_pixels(action.start)
            to_x, to_y = self._point_pixels(action.end)
            await self._call_action(
                "drag",
                {
                    **common,
                    "from_x": from_x,
                    "from_y": from_y,
                    "to_x": to_x,
                    "to_y": to_y,
                    "duration_ms": action.duration_ms or 500,
                    "steps": max(
                        1,
                        min(50, (action.duration_ms or 500) // 25),
                    ),
                },
            )
            return
        raise ValueError(f"unsupported local browser action: {action.type.value}")

    async def _call_action(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        remember: bool = True,
    ) -> CuaDriverResult:
        result = await self._get_driver().call_tool(name, arguments)
        status = str(result.structured.get("status") or "").strip().lower()
        if status in {"refused", "failed", "error"}:
            refusal = result.structured.get("refusal")
            detail = (
                refusal.get("message")
                if isinstance(refusal, Mapping)
                else result.structured.get("message")
            )
            raise CuaDriverError(
                str(detail or result.text or f"cua-driver {name} was {status}")
            )
        if remember:
            metadata = {
                field: result.structured[field]
                for field in _ACTION_RESULT_FIELDS
                if field in result.structured
            }
            if result.text:
                metadata["summary"] = result.text[:500]
            self._last_action_result = metadata or {
                "effect": "unverifiable",
                "verified": False,
            }
        return result

    def _action_target_arguments(self, action: ComputerAction) -> dict[str, Any]:
        target = action.target
        if target is None:
            return {}
        if target.element_id is not None:
            element = self._find_element(target.element_id)
            arguments = self._element_arguments(element)
            if arguments:
                return arguments
            raise ValueError(
                "local browser semantic target has no driver element identity"
            )
        x, y = self._action_point_pixels(action)
        return {"x": x, "y": y}

    async def _navigate_to_url(
        self,
        target: NativeBrowserWindow,
        url: str,
        *,
        common: Mapping[str, Any],
    ) -> None:
        navigator = self._native_browser_navigator
        if navigator is not None and navigator.supports(target):
            result = await navigator.navigate(target, url)
            self._last_action_result = {
                "path": result.route,
                "effect": "confirmed",
                "verified": True,
                "browser_window_id": result.browser_window_id,
                "actual_url": result.actual_url,
            }
            return
        observation = self.current_observation
        address = self._browser_address_element(
            observation.elements if observation is not None else [],
            target=target,
        )
        if address is None:
            raise ValueError("navigate requires a native browser address field")
        await self._call_action(
            "set_value",
            {
                "session": self.session_id,
                "pid": target.pid,
                "window_id": target.window_id,
                **self._element_arguments(address),
                "value": url,
            },
            remember=False,
        )

        # set_value may refresh the driver's element-token generation. Obtain a
        # fresh exact-window snapshot before directing Return to the address bar.
        refreshed = await self._get_driver().call_tool(
            "get_window_state",
            {
                "session": self.session_id,
                "pid": target.pid,
                "window_id": target.window_id,
                "include_screenshot": False,
                "max_elements": min(self.max_elements, MAX_OBSERVATION_ELEMENTS),
            },
        )
        raw_elements = refreshed.structured.get("elements")
        elements, _candidate_count = self._build_elements(
            raw_elements if isinstance(raw_elements, list) else [],
            target=target,
        )
        address = self._browser_address_element(elements, target=target)
        if address is None:
            raise ValueError("browser address field disappeared before navigation")
        await self._call_action(
            "press_key",
            {
                **common,
                **self._element_arguments(address),
                "key": "return",
            },
        )

    @staticmethod
    def _element_arguments(element: ComputerElement) -> dict[str, Any]:
        token = str(element.metadata.get("element_token") or "").strip()
        if token:
            return {"element_token": token}
        return {}

    @staticmethod
    def _browser_address_element(
        elements: list[ComputerElement],
        *,
        target: NativeBrowserWindow,
    ) -> ComputerElement | None:
        if target.app_name.casefold() not in _SUPPORTED_BROWSER_APP_NAMES:
            return None
        candidates = [
            element
            for element in elements
            if (element.role or "").casefold() in {"axtextfield", "textfield"}
            and element.bounds.y < 0.15
            and element.bounds.width >= 0.25
            and element.metadata.get("sensitive") is not True
            and bool(NativeBrowserEnvironment._element_arguments(element))
        ]
        return max(candidates, key=lambda element: element.bounds.width, default=None)

    def _action_point_pixels(self, action: ComputerAction) -> tuple[float, float]:
        target = action.target
        if target is None:
            raise ValueError(f"{action.type.value} requires a target")
        if target.point is not None:
            return self._point_pixels(target.point)
        element = self._find_element(target.element_id or "")
        return self._point_pixels(
            NormalizedPoint(
                x=element.bounds.x + element.bounds.width / 2,
                y=element.bounds.y + element.bounds.height / 2,
            )
        )

    def _point_pixels(self, point: NormalizedPoint) -> tuple[float, float]:
        observation = self.current_observation
        if observation is None:
            raise RuntimeError("local browser action requires a current observation")
        return (
            point.x * observation.viewport.width,
            point.y * observation.viewport.height,
        )

    def _validate_pixel_frame(self, action: ComputerAction) -> None:
        target = action.target
        uses_pixels = action.type in {
            ComputerActionType.DRAG,
        } or (
            action.type
            in {
                ComputerActionType.CLICK,
                ComputerActionType.DOUBLE_CLICK,
            }
            and target is not None
            and target.point is not None
        )
        if uses_pixels:
            self._validate_current_coordinate_frame()
            self._validate_unambiguous_pixel_target(action)

    def _validate_unambiguous_pixel_target(self, action: ComputerAction) -> None:
        target_window = self._require_target()
        points = self._pixel_action_points(action)
        other_windows = [
            window
            for window in self._on_screen_windows
            if (window.pid, window.window_id)
            != (target_window.pid, target_window.window_id)
        ]
        for point in points:
            screen_x = target_window.x + point.x * target_window.width
            screen_y = target_window.y + point.y * target_window.height
            if any(
                window.x <= screen_x <= window.x + window.width
                and window.y <= screen_y <= window.y + window.height
                and (
                    target_window.z_index is None
                    or window.z_index is None
                    or window.z_index > target_window.z_index
                )
                for window in other_windows
            ):
                raise ComputerTargetNotFoundError(
                    "Pixel action is ambiguous because another on-screen window "
                    "occludes the selected window at that point. Use an "
                    "accessibility element target or "
                    "select a non-overlapping window; do not retry the pixel "
                    "coordinate."
                )

    def _document_text_target(self, action: ComputerAction) -> ComputerElement:
        target = action.target
        if target is None or target.element_id is None:
            raise ValueError(
                "replace_text in the local browser requires an exact semantic "
                "document element target; coordinate and implicit keyboard input "
                "are disabled"
            )
        element = self._find_element(target.element_id)
        if element.surface is not ComputerElementSurface.DOCUMENT:
            raise ValueError(
                "replace_text is limited to document elements in the selected "
                "browser window; browser chrome and unknown surfaces are not "
                "authorized"
            )
        if element.metadata.get("sensitive") is True:
            raise ValueError(
                "replace_text is disabled for sensitive input elements in the "
                "local browser"
            )
        return element

    @staticmethod
    def _optional_z_index(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _pixel_action_points(self, action: ComputerAction) -> list[NormalizedPoint]:
        if action.type is ComputerActionType.DRAG:
            return [point for point in (action.start, action.end) if point is not None]
        target = action.target
        if target is None:
            return []
        if target.point is not None:
            return [target.point]
        return []

    def _validate_current_coordinate_frame(self) -> None:
        observation = self.current_observation
        target = self._require_target()
        if observation is None:
            raise ComputerFrameMismatchError(
                "pixel action requires a current computer observation"
            )
        coordinate_frame = observation.metadata.get("coordinate_frame")
        if not isinstance(coordinate_frame, Mapping):
            raise ComputerFrameMismatchError(
                "the current observation has no verified pixel coordinate frame; "
                "request a fresh observation"
            )
        captured_bounds = coordinate_frame.get("window_bounds")
        if not isinstance(captured_bounds, Mapping):
            raise ComputerFrameMismatchError(
                "the current observation has no verified pixel coordinate frame; "
                "request a fresh observation"
            )
        if coordinate_frame.get("screenshot_fresh") is not True:
            raise ComputerFrameMismatchError(
                "the current observation contains fresh semantic elements but "
                "no fresh exact-window screenshot. Use an accessibility element "
                "target or request a later observation; do not use pixel "
                "coordinates from the reused image."
            )
        current_bounds = self._window_bounds(target)
        for field, current_value in current_bounds.items():
            captured_value = captured_bounds.get(field)
            if isinstance(captured_value, bool) or not isinstance(
                captured_value,
                (str, int, float),
            ):
                unchanged = False
                captured_value = None
            try:
                if captured_value is not None:
                    unchanged = math.isclose(
                        float(captured_value),
                        current_value,
                        rel_tol=0.0,
                        abs_tol=0.5,
                    )
            except (TypeError, ValueError):
                unchanged = False
            if not unchanged:
                raise ComputerFrameMismatchError(
                    "The selected window moved or resized after frame "
                    f"{observation.frame_id!r} was captured. Request a fresh "
                    "observation before using pixel coordinates."
                )

    @staticmethod
    def _window_bounds(target: NativeBrowserWindow) -> dict[str, float]:
        return {
            "x": target.x,
            "y": target.y,
            "width": target.width,
            "height": target.height,
        }

    def _find_element(self, element_id: str) -> ComputerElement:
        observation = self.current_observation
        if observation is None:
            raise RuntimeError("element target requires a current observation")
        element = next(
            (item for item in observation.elements if item.element_id == element_id),
            None,
        )
        if element is None:
            raise ComputerTargetNotFoundError(
                f"element {element_id!r} is not present in frame "
                f"{observation.frame_id!r}"
            )
        return element

    def _require_target(self) -> NativeBrowserWindow:
        if self._target is None:
            raise RuntimeError("local browser window has not been selected")
        return self._target

    def _delivery_mode(self, action: ComputerAction) -> str:
        requested = str(action.metadata.get("delivery_mode") or "").strip().lower()
        if requested != "foreground":
            return "background"
        observation = self.current_observation
        metadata = observation.metadata if observation is not None else {}
        escalation = metadata.get("driver_escalation")
        if (
            isinstance(escalation, Mapping)
            and str(escalation.get("recommended") or "").strip().lower() == "foreground"
        ):
            return "foreground"
        raise ValueError(
            "foreground delivery requires a current cua-driver escalation recommendation "
            "from the current observation"
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None


def _optional_active_url(structured: Mapping[str, Any]) -> str | None:
    for key in ("active_url", "url", "actual_url"):
        value = structured.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized.startswith(("http://", "https://", "about:")):
            return normalized
    return None
