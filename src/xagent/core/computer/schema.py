from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..context_ref import ContextReference, ContextReferencePurpose

#: Observation metadata flag set when interactive elements could not be read.
#: Missing structure is diagnostic context, not evidence that an action is risky.
ELEMENT_EXTRACTION_FAILED_KEY = "element_extraction_failed"

#: Observation metadata flag set when some page surfaces (for example an
#: inaccessible frame or a closed shadow tree) could not be enumerated.
ELEMENT_EXTRACTION_INCOMPLETE_KEY = "element_extraction_incomplete"

#: Observation metadata flag set when the element list hit its cap, so the
#: model knows the list is incomplete rather than exhaustive.
ELEMENTS_TRUNCATED_KEY = "elements_truncated"

#: Maximum interactive elements reported per observation.
MAX_OBSERVATION_ELEMENTS = 100


class ComputerEnvironmentType(str, Enum):
    BROWSER = "browser"
    DESKTOP = "desktop"
    MOBILE = "mobile"


class ComputerActionType(str, Enum):
    SCREENSHOT = "screenshot"
    CAPTURE_MEDIA = "capture_media"
    NAVIGATE = "navigate"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    MOVE = "move"
    SCROLL = "scroll"
    TYPE = "type"
    REPLACE_TEXT = "replace_text"
    KEYPRESS = "keypress"
    DRAG = "drag"
    WAIT = "wait"


class ComputerMediaKind(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"


class ComputerElementSource(str, Enum):
    DOM = "dom"
    ACCESSIBILITY = "accessibility"
    UI_AUTOMATION = "ui_automation"
    OMNIPARSER = "omniparser"
    VISION = "vision"


class _ComputerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NormalizedPoint(_ComputerModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class NormalizedRect(_ComputerModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _within_viewport(self) -> "NormalizedRect":
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("normalized rectangle must fit within the viewport")
        return self


class Viewport(_ComputerModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    device_pixel_ratio: float = Field(default=1.0, gt=0)


class ComputerElement(_ComputerModel):
    element_id: str = Field(min_length=1)
    source: ComputerElementSource
    bounds: NormalizedRect
    label: str | None = None
    role: str | None = None
    text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComputerTarget(_ComputerModel):
    element_id: str | None = None
    point: NormalizedPoint | None = None

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "ComputerTarget":
        if (self.element_id is None) == (self.point is None):
            raise ValueError("target requires exactly one of element_id or point")
        if self.element_id is not None and not self.element_id.strip():
            raise ValueError("element_id must not be empty")
        return self


class ComputerAction(_ComputerModel):
    type: ComputerActionType
    target: ComputerTarget | None = None
    url: str | None = None
    text: str | None = None
    media_kind: ComputerMediaKind | None = None
    output_filename: str | None = Field(default=None, min_length=1, max_length=255)
    keys: list[str] = Field(default_factory=list, max_length=16)
    delta_x: float = Field(default=0, ge=-1, le=1)
    delta_y: float = Field(default=0, ge=-1, le=1)
    start: NormalizedPoint | None = None
    end: NormalizedPoint | None = None
    duration_ms: int = Field(default=0, ge=0, le=30_000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> "ComputerAction":
        target_actions = {
            ComputerActionType.CLICK,
            ComputerActionType.DOUBLE_CLICK,
            ComputerActionType.MOVE,
            ComputerActionType.REPLACE_TEXT,
        }
        if self.type in target_actions and self.target is None:
            raise ValueError(f"{self.type.value} requires a target")
        if self.type == ComputerActionType.NAVIGATE and not (
            self.url and self.url.strip()
        ):
            raise ValueError("navigate requires a URL")
        if self.type == ComputerActionType.TYPE and self.text is None:
            raise ValueError("type requires text")
        if self.type == ComputerActionType.REPLACE_TEXT and self.text is None:
            raise ValueError("replace_text requires text")
        if self.type == ComputerActionType.KEYPRESS and not self.keys:
            raise ValueError("keypress requires at least one key")
        if self.type == ComputerActionType.KEYPRESS and any(
            not key.strip() for key in self.keys
        ):
            raise ValueError("keypress keys must not be empty")
        if (
            self.type == ComputerActionType.SCROLL
            and self.delta_x == 0
            and self.delta_y == 0
        ):
            raise ValueError("scroll requires a non-zero delta")
        if self.type == ComputerActionType.DRAG and (
            self.start is None or self.end is None
        ):
            raise ValueError("drag requires start and end points")
        if self.type == ComputerActionType.CAPTURE_MEDIA:
            if self.media_kind is None:
                raise ValueError("capture_media requires media_kind")
            if not 1_000 <= self.duration_ms <= 30_000:
                raise ValueError(
                    "capture_media duration_ms must be between 1000 and 30000"
                )
            if self.target is not None:
                raise ValueError("capture_media does not accept a target")
        elif self.media_kind is not None or self.output_filename is not None:
            raise ValueError(
                "media_kind and output_filename are only valid for capture_media"
            )
        return self


class ComputerActionBatch(_ComputerModel):
    session_id: str = Field(min_length=1)
    expected_frame_id: str = Field(min_length=1)
    actions: list[ComputerAction] = Field(min_length=1, max_length=20)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComputerObservation(_ComputerModel):
    session_id: str = Field(min_length=1)
    frame_id: str = Field(min_length=1)
    environment: ComputerEnvironmentType
    viewport: Viewport
    screenshot: ContextReference
    elements: list[ComputerElement] = Field(default_factory=list)
    active_url: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_screenshot(self) -> "ComputerObservation":
        if self.screenshot.purpose != ContextReferencePurpose.OBSERVATION:
            raise ValueError("observation screenshot must use purpose=observation")
        if self.screenshot.frame_id != self.frame_id:
            raise ValueError("screenshot frame_id must match observation frame_id")
        return self
