from __future__ import annotations

import json
from enum import Enum
from typing import Annotated
from urllib.parse import SplitResult, urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from ..context_ref import ContextReference

COMPUTER_FRAME_ID_METADATA_KEY = "computer_frame_id"
COMPUTER_SESSION_ID_METADATA_KEY = "computer_session_id"

# Observation metadata flags set by adapters when their structural view is
# incomplete. Missing structure is diagnostic context, not proof of risk.
ELEMENT_EXTRACTION_FAILED_KEY = "element_extraction_failed"
ELEMENT_EXTRACTION_INCOMPLETE_KEY = "element_extraction_incomplete"
ELEMENTS_TRUNCATED_KEY = "elements_truncated"
MAX_OBSERVATION_ELEMENTS = 100
MAX_COMPUTER_METADATA_FIELD_BYTES = 32_768
_NORMALIZED_COORDINATE_EPSILON = 1e-9
_OBSERVED_URL_SCHEMES = frozenset(
    {
        "about",
        "blob",
        "chrome",
        "chrome-extension",
        "devtools",
        "edge",
        "http",
        "https",
        "moz-extension",
    }
)

Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
ShortText = Annotated[str, StringConstraints(max_length=4_096)]
ActionText = Annotated[str, StringConstraints(max_length=65_536)]


def _parse_safe_url(value: str, *, label: str) -> SplitResult:
    for character in value:
        if character.isspace():
            raise ValueError(f"{label} must not contain whitespace")
        if ord(character) < 0x20 or ord(character) == 0x7F:
            raise ValueError(f"{label} must not contain control characters")
    try:
        parsed = urlsplit(value)
        # Accessing ``port`` applies urllib's numeric range validation.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain embedded credentials")
    return parsed


def _validate_navigation_url(value: str) -> str:
    parsed = _parse_safe_url(value, label="navigation URL")
    if parsed.scheme.lower() not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("navigation URL must be absolute HTTP or HTTPS")
    return value


def _validate_observed_url(value: str) -> str:
    parsed = _parse_safe_url(value, label="active URL")
    if parsed.scheme.lower() not in _OBSERVED_URL_SCHEMES:
        raise ValueError("active URL uses an unsupported scheme")
    return value


NavigationUrl = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_192),
    AfterValidator(_validate_navigation_url),
]
ObservedUrl = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_192),
    AfterValidator(_validate_observed_url),
]
KeyName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64),
]
Metadata = dict[str, JsonValue]


class ComputerEnvironmentType(str, Enum):
    BROWSER = "browser"
    DESKTOP = "desktop"
    MOBILE = "mobile"


class ComputerActionType(str, Enum):
    SCREENSHOT = "screenshot"
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


class ComputerElementSource(str, Enum):
    DOM = "dom"
    ACCESSIBILITY = "accessibility"
    UI_AUTOMATION = "ui_automation"
    OMNIPARSER = "omniparser"
    VISION = "vision"


class _ComputerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("metadata", check_fields=False)
    @classmethod
    def _bound_metadata(cls, value: Metadata) -> Metadata:
        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON serializable") from exc
        if len(serialized.encode("utf-8")) > MAX_COMPUTER_METADATA_FIELD_BYTES:
            raise ValueError(
                f"metadata must be at most {MAX_COMPUTER_METADATA_FIELD_BYTES} bytes"
            )
        return value


class NormalizedPoint(_ComputerModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class NormalizedRect(_ComputerModel):
    """Viewport-clipped bounds for one visible element.

    Adapters must clip partially visible elements to the viewport and omit
    elements whose clipped bounds have zero area.
    """

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _within_viewport(self) -> "NormalizedRect":
        if (
            self.x + self.width > 1 + _NORMALIZED_COORDINATE_EPSILON
            or self.y + self.height > 1 + _NORMALIZED_COORDINATE_EPSILON
        ):
            raise ValueError("normalized rectangle must fit within the viewport")
        return self


class Viewport(_ComputerModel):
    width: int = Field(gt=0, le=100_000)
    height: int = Field(gt=0, le=100_000)
    device_pixel_ratio: float = Field(default=1.0, gt=0, le=16)


class ComputerElement(_ComputerModel):
    element_id: Identifier
    source: ComputerElementSource
    bounds: NormalizedRect
    label: ShortText | None = None
    role: ShortText | None = None
    text: ShortText | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: Metadata = Field(default_factory=dict, max_length=128)


class ComputerTarget(_ComputerModel):
    element_id: Identifier | None = None
    point: NormalizedPoint | None = None

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "ComputerTarget":
        if (self.element_id is None) == (self.point is None):
            raise ValueError("target requires exactly one of element_id or point")
        return self


class ComputerAction(_ComputerModel):
    type: ComputerActionType
    target: ComputerTarget | None = None
    url: NavigationUrl | None = None
    text: ActionText | None = None
    keys: list[KeyName] = Field(default_factory=list, max_length=16)
    delta_x: float = Field(default=0, ge=-1, le=1)
    delta_y: float = Field(default=0, ge=-1, le=1)
    start: NormalizedPoint | None = None
    end: NormalizedPoint | None = None
    duration_ms: int = Field(default=0, ge=0, le=30_000)
    metadata: Metadata = Field(default_factory=dict, max_length=128)

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
        if self.type not in target_actions and self.target is not None:
            raise ValueError(f"{self.type.value} does not accept a target")

        if self.type is ComputerActionType.NAVIGATE:
            if self.url is None:
                raise ValueError("navigate requires a URL")
        elif self.url is not None:
            raise ValueError(f"{self.type.value} does not accept a URL")

        text_actions = {ComputerActionType.TYPE, ComputerActionType.REPLACE_TEXT}
        if self.type in text_actions:
            if self.text is None:
                raise ValueError(f"{self.type.value} requires text")
            if self.type is ComputerActionType.TYPE and not self.text:
                raise ValueError("type requires non-empty text")
        elif self.text is not None:
            raise ValueError(f"{self.type.value} does not accept text")

        if self.type is ComputerActionType.KEYPRESS:
            if not self.keys:
                raise ValueError("keypress requires at least one key")
            if len(self.keys) != len({key.casefold() for key in self.keys}):
                raise ValueError("keypress keys must be unique")
        elif self.keys:
            raise ValueError(f"{self.type.value} does not accept keys")

        if self.type is ComputerActionType.SCROLL:
            if self.delta_x == 0 and self.delta_y == 0:
                raise ValueError("scroll requires a non-zero delta")
        elif self.delta_x != 0 or self.delta_y != 0:
            raise ValueError(f"{self.type.value} does not accept scroll deltas")

        if self.type is ComputerActionType.DRAG:
            if self.start is None or self.end is None:
                raise ValueError("drag requires start and end points")
            if (
                abs(self.start.x - self.end.x) <= _NORMALIZED_COORDINATE_EPSILON
                and abs(self.start.y - self.end.y) <= _NORMALIZED_COORDINATE_EPSILON
            ):
                raise ValueError("drag requires distinct start and end points")
        elif self.start is not None or self.end is not None:
            raise ValueError(f"{self.type.value} does not accept drag points")

        if self.type is ComputerActionType.WAIT:
            if self.duration_ms == 0:
                raise ValueError("wait requires a positive duration_ms")
        elif self.duration_ms != 0:
            raise ValueError(f"{self.type.value} does not accept duration_ms")
        return self


class ComputerActionBatch(_ComputerModel):
    session_id: Identifier
    expected_frame_id: Identifier
    # Every action must be followed by a new observation before another action
    # is planned. This keeps coordinates and element tokens tied to one frame.
    actions: list[ComputerAction] = Field(min_length=1, max_length=1)
    metadata: Metadata = Field(default_factory=dict, max_length=128)


class ComputerObservation(_ComputerModel):
    session_id: Identifier
    frame_id: Identifier
    environment: ComputerEnvironmentType
    viewport: Viewport
    screenshot: ContextReference
    elements: list[ComputerElement] = Field(
        default_factory=list,
        max_length=MAX_OBSERVATION_ELEMENTS,
    )
    active_url: ObservedUrl | None = None
    title: ShortText | None = None
    metadata: Metadata = Field(default_factory=dict, max_length=128)

    @field_validator("elements")
    @classmethod
    def _unique_element_ids(
        cls,
        elements: list[ComputerElement],
    ) -> list[ComputerElement]:
        element_ids = [element.element_id for element in elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("observation element_id values must be unique")
        return elements

    @model_validator(mode="after")
    def _validate_screenshot(self) -> "ComputerObservation":
        screenshot_session = self.screenshot.metadata.get(
            COMPUTER_SESSION_ID_METADATA_KEY
        )
        if screenshot_session != self.session_id:
            raise ValueError(
                "screenshot computer_session_id must match observation session_id"
            )
        screenshot_frame = self.screenshot.metadata.get(COMPUTER_FRAME_ID_METADATA_KEY)
        if screenshot_frame != self.frame_id:
            raise ValueError(
                "screenshot computer_frame_id must match observation frame_id"
            )
        return self
