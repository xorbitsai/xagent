from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .environment import ComputerEnvironment, ComputerTargetNotFoundError
from .media_store import MediaArtifactStore, RelayMediaArtifact
from .policy import (
    find_computer_target_element,
    navigation_block_reason,
    normalize_host_patterns,
)
from .relay import (
    BROWSER_RELAY_MAX_MESSAGE_BYTES,
    BrowserRelayCommandConnection,
    BrowserRelayRegistryProtocol,
    get_browser_relay_registry,
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


class _RelayElement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    element_id: str = Field(min_length=1, max_length=128)
    bounds: NormalizedRect
    label: str | None = Field(default=None, max_length=500)
    role: str | None = Field(default=None, max_length=100)
    text: str | None = Field(default=None, max_length=1_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class _RelayObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    screenshot_base64: str = Field(
        min_length=1,
        max_length=BROWSER_RELAY_MAX_MESSAGE_BYTES,
    )
    viewport: Viewport
    elements: list[_RelayElement] = Field(
        default_factory=list,
        max_length=MAX_OBSERVATION_ELEMENTS,
    )
    elements_truncated: bool = False
    element_extraction_failed: bool = False
    element_extraction_incomplete: bool = False
    active_url: str | None = Field(default=None, max_length=4_096)
    title: str | None = Field(default=None, max_length=500)


class ExtensionComputerEnvironment(ComputerEnvironment):
    """Computer environment backed by a user-approved Chrome extension tab."""

    def __init__(
        self,
        *,
        session_id: str,
        workspace: Any,
        session_binding: ComputerSessionBinding,
        registry: BrowserRelayRegistryProtocol | None = None,
        observation_store: ObservationStore | None = None,
        media_store: MediaArtifactStore | None = None,
        navigation_allowlist: Sequence[str] | None = None,
        navigation_denylist: Sequence[str] | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__(session_id)
        if session_binding.runtime_kind is not BrowserRuntimeKind.EXTENSION_RELAY:
            raise ValueError("extension environment requires extension_relay binding")
        self.workspace = workspace
        self.session_binding = session_binding
        self.user_id = session_binding.require_user_id()
        self.owner_task_id = session_binding.require_owner_task_id()
        self.registry = registry or get_browser_relay_registry()
        self.observation_store = observation_store or ObservationStore(workspace)
        self.media_store = media_store
        self.navigation_allowlist = normalize_host_patterns(navigation_allowlist)
        self.navigation_denylist = normalize_host_patterns(navigation_denylist)

    async def close(self) -> None:
        await self.registry.release(
            user_id=self.user_id,
            owner_task_id=self.owner_task_id,
        )

    async def _observe(self) -> ComputerObservation:
        frame_id = self._new_frame_id()
        connection = await self._connection()
        result = await connection.request(
            "observe",
            {"frame_id": frame_id},
        )
        return self._build_observation(result, frame_id=frame_id)

    async def _execute(self, batch: ComputerActionBatch) -> ComputerObservation:
        if len(batch.actions) != 1:
            raise ValueError("extension relay executes exactly one action per frame")
        action = batch.actions[0]
        if action.type is ComputerActionType.SCREENSHOT:
            return await self._observe()
        if action.type is not ComputerActionType.NAVIGATE:
            current_url = (
                self.current_observation.active_url
                if self.current_observation is not None
                else ""
            )
            self._assert_navigation_allowed(current_url or "")

        frame_id = self._new_frame_id()
        connection = await self._connection()
        if action.type is ComputerActionType.CAPTURE_MEDIA:
            media_store = self.media_store or MediaArtifactStore(self.workspace)
            if action.media_kind is None:
                raise ValueError("capture_media requires media_kind")
            transfer = media_store.begin(
                media_kind=action.media_kind,
                output_filename=action.output_filename,
            )
            try:
                media_result = await connection.request(
                    "capture_media",
                    {
                        "expected_frame_id": batch.expected_frame_id,
                        "transfer_id": transfer.transfer_id,
                        "action": self._serialize_action(action),
                    },
                    timeout=max(30.0, (action.duration_ms / 1_000) + 10.0),
                    on_media_chunk=transfer.accept,
                )
                artifact = RelayMediaArtifact.model_validate(
                    media_result.get("artifact", media_result)
                )
                self.record_action_artifacts([transfer.finish(artifact)])
            except Exception:
                transfer.abort()
                raise
            result = await connection.request(
                "observe",
                {"frame_id": frame_id},
            )
            observation = self._build_observation(result, frame_id=frame_id)
            self._assert_navigation_allowed(observation.active_url or "")
            return observation
        result = await connection.request(
            "act",
            {
                "expected_frame_id": batch.expected_frame_id,
                "frame_id": frame_id,
                "action": self._serialize_action(action),
                "navigation_policy": {
                    "allowlist": list(self.navigation_allowlist),
                    "denylist": list(self.navigation_denylist),
                },
            },
            timeout=max(30.0, (action.duration_ms / 1_000) + 10.0),
        )
        observation = self._build_observation(result, frame_id=frame_id)
        self._assert_navigation_allowed(observation.active_url or "")
        return observation

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
        target = action.target
        if target is not None:
            point = (
                target.point
                if target.point is not None
                else self._element_center(target.element_id or "")
            )
            payload["target"] = point.model_dump(mode="json")
            target_element_id = target.element_id
            if target_element_id is None and self.current_observation is not None:
                element = find_computer_target_element(
                    action,
                    self.current_observation,
                )
                target_element_id = element.element_id if element is not None else None
            if target_element_id:
                # Lets the extension hit-test the coordinate against the very
                # element the model chose before dispatching the click.
                payload["target_element_id"] = target_element_id
        if action.type is ComputerActionType.NAVIGATE:
            payload["url"] = self._validate_navigation_url(action.url or "")
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
        parsed = _RelayObservation.model_validate(raw_observation)
        try:
            image_bytes = base64.b64decode(
                parsed.screenshot_base64,
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise ValueError("extension returned an invalid screenshot") from exc
        if not image_bytes or len(image_bytes) > BROWSER_RELAY_MAX_MESSAGE_BYTES:
            raise ValueError("extension screenshot is empty or too large")
        screenshot = self.observation_store.save_screenshot(
            session_id=self.session_id,
            frame_id=frame_id,
            image_bytes=image_bytes,
            mime_type="image/png",
            viewport=parsed.viewport,
            text_fallback="Current user-approved browser tab screenshot.",
            metadata={"browser_runtime_kind": BrowserRuntimeKind.EXTENSION_RELAY.value},
        )
        metadata: dict[str, Any] = {
            "browser_runtime_kind": BrowserRuntimeKind.EXTENSION_RELAY.value,
            "user_takeover_available": True,
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
            environment=ComputerEnvironmentType.BROWSER,
            viewport=parsed.viewport,
            screenshot=screenshot,
            elements=[self._build_element(element) for element in parsed.elements],
            active_url=parsed.active_url,
            title=parsed.title,
            metadata=metadata,
        )

    @classmethod
    def _build_element(cls, element: _RelayElement) -> ComputerElement:
        metadata = element.metadata
        sensitive = cls._is_sensitive(metadata)
        safe_metadata = {
            key: metadata[key]
            for key in (
                "tag",
                "input_type",
                "autocomplete",
                "disabled",
                "focused",
            )
            if key in metadata
        }
        return ComputerElement(
            element_id=element.element_id,
            source=ComputerElementSource.DOM,
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
        input_type = str(metadata.get("input_type") or "").strip().lower()
        autocomplete = str(metadata.get("autocomplete") or "").strip().lower()
        return (
            input_type in {"password", "hidden"}
            or autocomplete
            in {
                "current-password",
                "new-password",
                "one-time-code",
                "webauthn",
            }
            or autocomplete.startswith("cc-")
        )

    def _validate_navigation_url(self, raw_url: str) -> str:
        url = raw_url.strip()
        parsed = urlsplit(url)
        if parsed.scheme in {"http", "https"}:
            self._assert_navigation_allowed(url)
            return url
        if parsed.scheme == "about" and url == "about:blank":
            return url
        raise ValueError(
            "user-browser navigation requires http://, https://, or about:blank"
        )

    def _assert_navigation_allowed(self, raw_url: str) -> None:
        reason = navigation_block_reason(
            raw_url,
            allowlist=self.navigation_allowlist,
            denylist=self.navigation_denylist,
        )
        if reason is not None:
            raise ValueError(reason)

    @staticmethod
    def _new_frame_id() -> str:
        return f"frame-{uuid4().hex}"
