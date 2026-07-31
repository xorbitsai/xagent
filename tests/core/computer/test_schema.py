from __future__ import annotations

import pytest
from pydantic import ValidationError

from xagent.core.computer.schema import (
    COMPUTER_FRAME_ID_METADATA_KEY,
    COMPUTER_SESSION_ID_METADATA_KEY,
    MAX_COMPUTER_METADATA_FIELD_BYTES,
    MAX_OBSERVATION_ELEMENTS,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerObservation,
    ComputerTarget,
    NormalizedPoint,
    NormalizedRect,
    Viewport,
)
from xagent.core.context_ref import ContextReference


def image_reference(
    *,
    session_id: str = "session-1",
    frame_id: str = "frame-1",
) -> ContextReference:
    return ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
        },
        metadata={
            COMPUTER_SESSION_ID_METADATA_KEY: session_id,
            COMPUTER_FRAME_ID_METADATA_KEY: frame_id,
        },
    )


def observation_element(element_id: str) -> ComputerElement:
    return ComputerElement(
        element_id=element_id,
        source=ComputerElementSource.ACCESSIBILITY,
        bounds=NormalizedRect(x=0.1, y=0.1, width=0.2, height=0.1),
    )


def test_normalized_rect_must_fit_viewport() -> None:
    with pytest.raises(ValidationError, match="fit within the viewport"):
        NormalizedRect(x=0.8, y=0.1, width=0.3, height=0.2)


def test_normalized_rect_tolerates_floating_point_boundary_noise() -> None:
    rect = NormalizedRect(
        x=0.5000000000000002,
        y=0.5000000000000002,
        width=0.5,
        height=0.5,
    )

    assert rect.x + rect.width > 1
    assert rect.y + rect.height > 1


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"type": ComputerActionType.CLICK}, "requires a target"),
        ({"type": ComputerActionType.NAVIGATE}, "requires a URL"),
        ({"type": ComputerActionType.TYPE}, "requires text"),
        ({"type": ComputerActionType.REPLACE_TEXT}, "requires a target"),
        ({"type": ComputerActionType.KEYPRESS}, "requires at least one key"),
        ({"type": ComputerActionType.SCROLL}, "requires a non-zero delta"),
        (
            {
                "type": ComputerActionType.DRAG,
                "start": {"x": 0.1, "y": 0.1},
            },
            "requires start and end points",
        ),
        ({"type": ComputerActionType.WAIT}, "requires a positive duration"),
    ],
)
def test_action_requires_type_specific_payload(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ComputerAction.model_validate(payload)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "data:text/html,hello",
        "//example.com/path",
    ],
)
def test_navigate_rejects_non_http_urls(url: str) -> None:
    with pytest.raises(ValidationError, match="absolute HTTP or HTTPS"):
        ComputerAction(type=ComputerActionType.NAVIGATE, url=url)


def test_navigate_allows_local_http_urls() -> None:
    action = ComputerAction(
        type=ComputerActionType.NAVIGATE,
        url="http://127.0.0.1:3999/launch-model/llm",
    )

    assert action.url == "http://127.0.0.1:3999/launch-model/llm"


def test_navigate_rejects_embedded_credentials() -> None:
    with pytest.raises(ValidationError, match="embedded credentials"):
        ComputerAction(
            type=ComputerActionType.NAVIGATE,
            url="https://user:password@example.com/",
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://exam ple.com/",
        "https://example.com/a\tb",
        "https://example.com/a\nb",
    ],
)
def test_navigate_rejects_interior_whitespace(url: str) -> None:
    with pytest.raises(ValidationError, match="must not contain whitespace"):
        ComputerAction(type=ComputerActionType.NAVIGATE, url=url)


@pytest.mark.parametrize(
    "url",
    ["about:blank", "chrome://newtab", "blob:https://example.com/frame-id"],
)
def test_observation_allows_non_http_active_url(url: str) -> None:
    observation = ComputerObservation(
        session_id="session-1",
        frame_id="frame-1",
        environment=ComputerEnvironmentType.BROWSER,
        viewport=Viewport(width=1280, height=720),
        screenshot=image_reference(),
        active_url=url,
    )

    assert observation.active_url == url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/\x00page",
        "https://example.com/\x7fpage",
        "https://example.com:99999/",
    ],
)
def test_urls_reject_control_characters_and_invalid_ports(url: str) -> None:
    with pytest.raises(ValidationError, match="control characters|invalid"):
        ComputerAction(type=ComputerActionType.NAVIGATE, url=url)


def test_observation_active_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ValidationError, match="embedded credentials"):
        ComputerObservation(
            session_id="session-1",
            frame_id="frame-1",
            environment=ComputerEnvironmentType.BROWSER,
            viewport=Viewport(width=1280, height=720),
            screenshot=image_reference(),
            active_url="https://user:password@example.com/",
        )


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
        "file:///Users/alice/secret.html",
        "C:/Users/alice/secret.html",
        "https://example.com/\x00page",
        "https://example.com:99999/",
    ],
)
def test_observation_rejects_unsafe_or_invalid_active_urls(url: str) -> None:
    with pytest.raises(
        ValidationError,
        match="unsupported scheme|control characters|invalid",
    ):
        ComputerObservation(
            session_id="session-1",
            frame_id="frame-1",
            environment=ComputerEnvironmentType.BROWSER,
            viewport=Viewport(width=1280, height=720),
            screenshot=image_reference(),
            active_url=url,
        )


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {
                "type": ComputerActionType.SCREENSHOT,
                "url": "https://example.com",
            },
            "does not accept a URL",
        ),
        (
            {"type": ComputerActionType.SCREENSHOT, "text": "secret"},
            "does not accept text",
        ),
        (
            {"type": ComputerActionType.SCREENSHOT, "keys": ["ENTER"]},
            "does not accept keys",
        ),
        (
            {"type": ComputerActionType.SCREENSHOT, "delta_y": 0.5},
            "does not accept scroll deltas",
        ),
        (
            {
                "type": ComputerActionType.SCREENSHOT,
                "start": {"x": 0.1, "y": 0.1},
            },
            "does not accept drag points",
        ),
        (
            {"type": ComputerActionType.SCREENSHOT, "duration_ms": 10},
            "does not accept duration_ms",
        ),
    ],
)
def test_action_rejects_fields_for_another_action_type(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ComputerAction.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"type": ComputerActionType.TYPE, "text": ""},
            "requires non-empty text",
        ),
        (
            {"type": ComputerActionType.KEYPRESS, "keys": ["CTRL", "CTRL"]},
            "keys must be unique",
        ),
        (
            {"type": ComputerActionType.KEYPRESS, "keys": ["Ctrl", "ctrl"]},
            "keys must be unique",
        ),
        (
            {
                "type": ComputerActionType.DRAG,
                "start": {"x": 0.25, "y": 0.5},
                "end": {"x": 0.25, "y": 0.5},
            },
            "distinct start and end",
        ),
        (
            {
                "type": ComputerActionType.DRAG,
                "start": {"x": 0.25, "y": 0.5},
                "end": {"x": 0.25 + 1e-12, "y": 0.5},
            },
            "distinct start and end",
        ),
    ],
)
def test_action_rejects_validated_no_ops(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ComputerAction.model_validate(payload)


def test_replace_text_allows_empty_text_to_clear_a_field() -> None:
    action = ComputerAction(
        type=ComputerActionType.REPLACE_TEXT,
        target=ComputerTarget(element_id="input-1"),
        text="",
    )

    assert action.text == ""


def test_drag_accepts_motion_outside_coordinate_epsilon() -> None:
    action = ComputerAction(
        type=ComputerActionType.DRAG,
        start=NormalizedPoint(x=0.25, y=0.5),
        end=NormalizedPoint(x=0.25 + 1e-6, y=0.5),
    )

    assert action.start is not None
    assert action.end is not None
    assert action.end.x > action.start.x


def test_action_batch_uses_normalized_target_and_expected_frame() -> None:
    batch = ComputerActionBatch(
        session_id=" session-1 ",
        expected_frame_id=" frame-1 ",
        actions=[
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(point=NormalizedPoint(x=0.25, y=0.75)),
            )
        ],
    )

    assert batch.session_id == "session-1"
    assert batch.expected_frame_id == "frame-1"
    assert batch.actions[0].target is not None
    assert batch.actions[0].target.point == NormalizedPoint(x=0.25, y=0.75)


def test_action_batch_requires_a_new_frame_after_each_action() -> None:
    with pytest.raises(ValidationError, match="at most 1 item"):
        ComputerActionBatch(
            session_id="session-1",
            expected_frame_id="frame-1",
            actions=[
                ComputerAction(type=ComputerActionType.SCREENSHOT),
                ComputerAction(type=ComputerActionType.SCREENSHOT),
            ],
        )


def test_observation_requires_matching_screenshot_frame() -> None:
    with pytest.raises(ValidationError, match="must match observation frame_id"):
        ComputerObservation(
            session_id="session-1",
            frame_id="frame-2",
            environment=ComputerEnvironmentType.BROWSER,
            viewport=Viewport(width=1280, height=720),
            screenshot=image_reference(frame_id="frame-1"),
        )


def test_observation_requires_matching_screenshot_session() -> None:
    with pytest.raises(ValidationError, match="must match observation session_id"):
        ComputerObservation(
            session_id="session-2",
            frame_id="frame-1",
            environment=ComputerEnvironmentType.BROWSER,
            viewport=Viewport(width=1280, height=720),
            screenshot=image_reference(session_id="session-1"),
        )


def test_observation_rejects_duplicate_element_tokens() -> None:
    with pytest.raises(ValidationError, match="element_id values must be unique"):
        ComputerObservation(
            session_id="session-1",
            frame_id="frame-1",
            environment=ComputerEnvironmentType.BROWSER,
            viewport=Viewport(width=1280, height=720),
            screenshot=image_reference(),
            elements=[observation_element("button-1"), observation_element("button-1")],
        )


def test_observation_bounds_element_count() -> None:
    with pytest.raises(ValidationError, match="at most 100 items"):
        ComputerObservation(
            session_id="session-1",
            frame_id="frame-1",
            environment=ComputerEnvironmentType.BROWSER,
            viewport=Viewport(width=1280, height=720),
            screenshot=image_reference(),
            elements=[
                observation_element(f"element-{index}")
                for index in range(MAX_OBSERVATION_ELEMENTS + 1)
            ],
        )


def test_computer_metadata_must_be_json_safe() -> None:
    with pytest.raises(ValidationError, match="valid JSON"):
        ComputerAction(
            type=ComputerActionType.SCREENSHOT,
            metadata={"not_json": {object()}},
        )


def test_computer_metadata_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValidationError, match="metadata must be JSON serializable"):
        ComputerAction(
            type=ComputerActionType.SCREENSHOT,
            metadata={"not_json": float("nan")},
        )


def test_computer_metadata_has_a_serialized_size_limit() -> None:
    with pytest.raises(ValidationError, match="must be at most"):
        ComputerAction(
            type=ComputerActionType.SCREENSHOT,
            metadata={"payload": "x" * MAX_COMPUTER_METADATA_FIELD_BYTES},
        )
