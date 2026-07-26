from __future__ import annotations

import pytest
from pydantic import ValidationError

from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerEnvironmentType,
    ComputerMediaKind,
    ComputerObservation,
    ComputerTarget,
    NormalizedPoint,
    NormalizedRect,
    Viewport,
)
from xagent.core.context_ref import ContextReference, ContextReferencePurpose


def image_reference(*, frame_id: str = "frame-1") -> ContextReference:
    return ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
            "file_path": "/private/frame.png",
        },
        purpose=ContextReferencePurpose.OBSERVATION,
        frame_id=frame_id,
    )


def test_normalized_rect_must_fit_viewport() -> None:
    with pytest.raises(ValidationError, match="fit within the viewport"):
        NormalizedRect(x=0.8, y=0.1, width=0.3, height=0.2)


@pytest.mark.parametrize(
    "payload",
    [
        {"type": ComputerActionType.CLICK},
        {"type": ComputerActionType.TYPE},
        {"type": ComputerActionType.REPLACE_TEXT, "text": "replacement"},
        {
            "type": ComputerActionType.REPLACE_TEXT,
            "target": {"point": {"x": 0.5, "y": 0.5}},
        },
        {"type": ComputerActionType.KEYPRESS},
        {"type": ComputerActionType.SCROLL},
        {"type": ComputerActionType.NAVIGATE},
        {
            "type": ComputerActionType.DRAG,
            "start": {"x": 0.1, "y": 0.1},
        },
    ],
)
def test_action_requires_type_specific_payload(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ComputerAction.model_validate(payload)


def test_action_batch_uses_normalized_target_and_expected_frame() -> None:
    batch = ComputerActionBatch(
        session_id="session-1",
        expected_frame_id="frame-1",
        actions=[
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(point=NormalizedPoint(x=0.25, y=0.75)),
            )
        ],
    )

    assert batch.actions[0].target is not None
    assert batch.actions[0].target.point == NormalizedPoint(x=0.25, y=0.75)


def test_replace_text_requires_target_and_text() -> None:
    action = ComputerAction(
        type=ComputerActionType.REPLACE_TEXT,
        target=ComputerTarget(point=NormalizedPoint(x=0.25, y=0.75)),
        text="Xagent Introduction",
    )

    assert action.text == "Xagent Introduction"


def test_capture_media_requires_kind_and_bounded_duration() -> None:
    action = ComputerAction(
        type=ComputerActionType.CAPTURE_MEDIA,
        media_kind=ComputerMediaKind.VIDEO,
        duration_ms=30_000,
        output_filename="demo.mp4",
    )

    assert action.media_kind is ComputerMediaKind.VIDEO
    with pytest.raises(ValidationError, match="media_kind"):
        ComputerAction(type=ComputerActionType.CAPTURE_MEDIA, duration_ms=1_000)
    with pytest.raises(ValidationError, match="less than or equal to 30000"):
        ComputerAction(
            type=ComputerActionType.CAPTURE_MEDIA,
            media_kind=ComputerMediaKind.AUDIO,
            duration_ms=30_001,
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


def test_context_reference_strips_local_file_path() -> None:
    reference = image_reference()

    assert reference.file_ref["file_id"] == "image-1"
    assert "file_path" not in reference.file_ref
