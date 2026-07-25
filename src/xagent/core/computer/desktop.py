from __future__ import annotations

import base64
import binascii
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .desktop_relay import (
    DesktopRelayRegistryProtocol,
    get_desktop_relay_registry,
)
from .environment import ComputerEnvironment, ComputerTargetNotFoundError
from .relay import (
    BROWSER_RELAY_MAX_MESSAGE_BYTES,
    BrowserRelayCommandConnection,
)
from .schema import (
    ELEMENT_EXTRACTION_FAILED_KEY,
    ELEMENT_EXTRACTION_INCOMPLETE_KEY,
    ELEMENTS_TRUNCATED_KEY,
    MAX_OBSERVATION_ELEMENTS,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerObservation,
    NormalizedPoint,
    NormalizedRect,
    Viewport,
)
from .session import BrowserRuntimeKind, ComputerSessionBinding
from .store import ObservationStore


class _DesktopRelayElement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    element_id: str = Field(min_length=1, max_length=128)
    bounds: NormalizedRect
    label: str | None = Field(default=None, max_length=500)
    role: str | None = Field(default=None, max_length=100)
    text: str | None = Field(default=None, max_length=1_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class _DesktopRelayObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    screenshot_base64: str = Field(
        min_length=1,
        max_length=BROWSER_RELAY_MAX_MESSAGE_BYTES,
    )
    viewport: Viewport
    elements: list[_DesktopRelayElement] = Field(
        default_factory=list,
        max_length=MAX_OBSERVATION_ELEMENTS,
    )
    elements_truncated: bool = False
    element_extraction_failed: bool = False
    # Old or minimal companions cannot prove that every native control was
    # enumerated. Defaulting to incomplete keeps coordinate actions fail-closed.
    element_extraction_incomplete: bool = True
    window_id: int = Field(ge=0)
    title: str | None = Field(default=None, max_length=500)
    application: str | None = Field(default=None, max_length=500)
    paused: bool = False
    emergency_stopped: bool = False


class DesktopRelayEnvironment(ComputerEnvironment):
    """Computer environment backed by one user-authorized desktop window."""

    def __init__(
        self,
        *,
        session_id: str,
        workspace: Any,
        session_binding: ComputerSessionBinding,
        registry: DesktopRelayRegistryProtocol | None = None,
        observation_store: ObservationStore | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__(session_id)
        if session_binding.runtime_kind is not BrowserRuntimeKind.DESKTOP_RELAY:
            raise ValueError("desktop environment requires desktop_relay binding")
        self.workspace = workspace
        self.session_binding = session_binding
        self.user_id = session_binding.require_user_id()
        self.owner_task_id = session_binding.require_owner_task_id()
        self.registry = registry or get_desktop_relay_registry()
        self.observation_store = observation_store or ObservationStore(workspace)

    async def close(self) -> None:
        await self.registry.release(
            user_id=self.user_id,
            owner_task_id=self.owner_task_id,
        )

    async def _observe(self) -> ComputerObservation:
        frame_id = self._new_frame_id()
        result = await (await self._connection()).request(
            "observe",
            {"frame_id": frame_id},
        )
        return self._build_observation(result, frame_id=frame_id)

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        if len(batch.actions) != 1:
            raise ValueError("desktop relay executes exactly one action per frame")
        action = batch.actions[0]
        if action.type is ComputerActionType.SCREENSHOT:
            return await self._observe()
        if action.type is ComputerActionType.NAVIGATE:
            raise ValueError("navigate is not supported by a desktop environment")
        if (
            self.current_observation is not None
            and self.current_observation.metadata.get("paused") is True
        ):
            raise RuntimeError(
                "Desktop relay is paused by the user. Ask the user to resume it, "
                "then request a fresh screenshot."
            )

        frame_id = self._new_frame_id()
        result = await (await self._connection()).request(
            "act",
            {
                "expected_frame_id": batch.expected_frame_id,
                "frame_id": frame_id,
                "action": self._serialize_action(action),
            },
            timeout=max(30.0, (action.duration_ms / 1_000) + 10.0),
        )
        return self._build_observation(result, frame_id=frame_id)

    async def _connection(self) -> BrowserRelayCommandConnection:
        connection = await self.registry.acquire(
            user_id=self.user_id,
            owner_task_id=self.owner_task_id,
        )
        await self.registry.touch_claim(
            user_id=self.user_id,
            owner_task_id=self.owner_task_id,
        )
        return connection

    def _serialize_action(self, action: ComputerAction) -> dict[str, Any]:
        payload = action.model_dump(mode="json", exclude_none=True)
        if action.target is None:
            return payload
        point = (
            action.target.point
            if action.target.point is not None
            else self._element_center(action.target.element_id or "")
        )
        payload["target"] = point.model_dump(mode="json")
        if action.target.element_id:
            payload["target_element_id"] = action.target.element_id
        return payload

    def _element_center(self, element_id: str) -> NormalizedPoint:
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
        return NormalizedPoint(
            x=element.bounds.x + element.bounds.width / 2,
            y=element.bounds.y + element.bounds.height / 2,
        )

    def _build_observation(
        self,
        raw_result: dict[str, Any],
        *,
        frame_id: str,
    ) -> ComputerObservation:
        raw_observation = raw_result.get("observation", raw_result)
        parsed = _DesktopRelayObservation.model_validate(raw_observation)
        if parsed.emergency_stopped:
            raise RuntimeError(
                "Desktop relay emergency stop is active. The user must explicitly "
                "re-authorize a window before automation can continue."
            )
        try:
            image_bytes = base64.b64decode(
                parsed.screenshot_base64,
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise ValueError("desktop relay returned an invalid screenshot") from exc
        if not image_bytes or len(image_bytes) > BROWSER_RELAY_MAX_MESSAGE_BYTES:
            raise ValueError("desktop screenshot is empty or too large")
        screenshot = self.observation_store.save_screenshot(
            session_id=self.session_id,
            frame_id=frame_id,
            image_bytes=image_bytes,
            mime_type="image/png",
            viewport=parsed.viewport,
            text_fallback="Current user-authorized desktop window screenshot.",
            metadata={
                "computer_runtime_kind": BrowserRuntimeKind.DESKTOP_RELAY.value,
                "window_id": parsed.window_id,
            },
        )
        metadata: dict[str, Any] = {
            "computer_runtime_kind": BrowserRuntimeKind.DESKTOP_RELAY.value,
            "user_takeover_available": True,
            "window_id": parsed.window_id,
            "application": parsed.application,
            "paused": parsed.paused,
        }
        if parsed.element_extraction_failed:
            metadata[ELEMENT_EXTRACTION_FAILED_KEY] = True
        if parsed.element_extraction_incomplete:
            metadata[ELEMENT_EXTRACTION_INCOMPLETE_KEY] = True
        if parsed.elements_truncated:
            metadata[ELEMENTS_TRUNCATED_KEY] = True
        return ComputerObservation(
            session_id=self.session_id,
            frame_id=frame_id,
            environment=ComputerEnvironmentType.DESKTOP,
            viewport=parsed.viewport,
            screenshot=screenshot,
            elements=[self._build_element(element) for element in parsed.elements],
            title=parsed.title,
            metadata=metadata,
        )

    @classmethod
    def _build_element(cls, element: _DesktopRelayElement) -> ComputerElement:
        metadata = element.metadata
        sensitive = cls._is_sensitive(metadata)
        safe_metadata = {
            key: metadata[key]
            for key in ("role", "subrole", "enabled", "focused")
            if key in metadata
        }
        return ComputerElement(
            element_id=element.element_id,
            source=ComputerElementSource.UI_AUTOMATION,
            bounds=element.bounds,
            label="Sensitive input" if sensitive else element.label,
            role=element.role,
            text=None if sensitive else element.text,
            metadata={**safe_metadata, "sensitive": sensitive},
        )

    @staticmethod
    def _is_sensitive(metadata: dict[str, Any]) -> bool:
        if metadata.get("sensitive") is True:
            return True
        role = str(metadata.get("role") or "").lower()
        subrole = str(metadata.get("subrole") or "").lower()
        return "secure" in role or "secure" in subrole

    @staticmethod
    def _new_frame_id() -> str:
        return f"frame-{uuid4().hex}"
