from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from xagent.core.agent.context import ExecutionContext, Message
from xagent.core.context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    ImageDetail,
)


def image_reference() -> ContextReference:
    return ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
            "file_path": "/private/frame.png",
        },
        text_fallback="A settings dialog",
    )


def test_legacy_message_payload_is_unchanged_without_refs() -> None:
    message = Message(role="user", content="hello")

    assert message.to_dict() == {"role": "user", "content": "hello"}


def test_message_payload_contains_only_durable_reference() -> None:
    message = Message(
        role="user",
        content="inspect",
        context_refs=(image_reference(),),
    )

    payload = message.to_dict()

    assert payload["content"] == "inspect"
    assert payload[CONTEXT_REFS_KEY][0]["file_ref"]["file_id"] == "image-1"
    assert "file_path" not in json.dumps(payload)
    assert "base64" not in json.dumps(payload)


def test_context_reference_requires_registered_image_file_ref() -> None:
    with pytest.raises(ValidationError, match="registered file_id"):
        ContextReference(
            file_ref={
                "filename": "frame.png",
                "mime_type": "image/png",
            }
        )

    with pytest.raises(ValidationError, match="image MIME type"):
        ContextReference(
            file_ref={
                "file_id": "file-1",
                "filename": "notes.txt",
                "mime_type": "text/plain",
            }
        )


def test_context_reference_rejects_materialized_image_in_metadata() -> None:
    with pytest.raises(ValidationError, match="materialized image data"):
        ContextReference(
            file_ref={
                "file_id": "image-1",
                "filename": "frame.png",
                "mime_type": "image/png",
            },
            metadata={"image": "data:image/png;base64,c2NyZWVuc2hvdA=="},
        )


def test_context_reference_rejects_paths_nested_in_metadata() -> None:
    with pytest.raises(ValidationError, match="must not contain a path"):
        ContextReference(
            file_ref={
                "file_id": "image-1",
                "filename": "frame.png",
                "mime_type": "image/png",
            },
            metadata={"capture": {"file_path": "/private/frame.png"}},
        )

    with pytest.raises(ValidationError, match="absolute filesystem paths"):
        ContextReference(
            file_ref={
                "file_id": "image-1",
                "filename": "frame.png",
                "mime_type": "image/png",
            },
            metadata={"source": "/private/frame.png"},
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"source": r"\\server\share\frame.png"},
        {"source": r"\rooted\frame.png"},
        {"source": "//server/share/frame.png"},
        {"capture": {"source": r"\\server\share\frame.png"}},
        {"values": [r"\rooted\frame.png"]},
    ],
)
def test_context_reference_rejects_cross_platform_absolute_metadata_paths(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="absolute filesystem paths"):
        ContextReference(
            file_ref={
                "file_id": "image-1",
                "filename": "frame.png",
                "mime_type": "image/png",
            },
            metadata=metadata,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"description": "saved at (/private/frame.png)"},
        {"capture": {"description": "file:///private/frame.png"}},
        {"values": [r"source=C:\Users\name\frame.png"]},
        {"description": r"saved at (\\server\share\frame.png)"},
        {"description": r"source=\rooted\frame.png"},
    ],
)
def test_context_reference_rejects_embedded_absolute_metadata_paths(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="absolute filesystem paths"):
        ContextReference(
            file_ref={
                "file_id": "image-1",
                "filename": "frame.png",
                "mime_type": "image/png",
            },
            metadata=metadata,
        )


def test_context_reference_allows_remote_urls_with_path_segments() -> None:
    reference = ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
        },
        metadata={"source": "https://example.com/private/frame.png"},
    )

    assert reference.metadata["source"] == "https://example.com/private/frame.png"


def test_context_reference_validates_json_normalized_tuple_metadata() -> None:
    reference = ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
        },
        metadata={"viewport": (1280, 720)},
    )

    assert reference.metadata == {"viewport": [1280, 720]}
    assert ContextReference.model_validate(reference.durable_dict()) == reference

    with pytest.raises(ValidationError, match="absolute filesystem paths"):
        ContextReference(
            file_ref={
                "file_id": "image-1",
                "filename": "frame.png",
                "mime_type": "image/png",
            },
            metadata={"values": ("/private/frame.png",)},
        )


def test_context_reference_rejects_materialized_image_in_text_fallback() -> None:
    with pytest.raises(ValidationError, match="text_fallback"):
        ContextReference(
            file_ref={
                "file_id": "image-1",
                "filename": "frame.png",
                "mime_type": "image/png",
            },
            text_fallback="data:image/png;base64,c2NyZWVuc2hvdA==",
        )


def test_durable_dict_resanitizes_nested_mutations() -> None:
    reference = image_reference()
    reference.file_ref["file_path"] = "/private/late-mutation.png"
    reference.metadata["image"] = "data:image/png;base64,c2NyZWVuc2hvdA=="

    with pytest.raises(ValueError, match="materialized image data"):
        reference.durable_dict()

    reference.metadata.pop("image")
    assert "file_path" not in reference.durable_dict()["file_ref"]

    reference.metadata["capture"] = {"file_path": "/private/late-mutation.png"}
    with pytest.raises(ValueError, match="must not contain a path"):
        reference.durable_dict()

    reference.metadata["capture"] = {
        "description": "saved at (/private/late-mutation.png)"
    }
    with pytest.raises(ValueError, match="absolute filesystem paths"):
        reference.durable_dict()


@pytest.mark.parametrize(
    ("unsafe_key", "message"),
    [
        ("data:image/png;base64,c2NyZWVuc2hvdA==", "materialized image data"),
        ("/private/frame.png", "absolute filesystem paths"),
        (r"\\server\share\frame.png", "absolute filesystem paths"),
    ],
)
def test_durable_dict_rejects_sensitive_metadata_keys(
    unsafe_key: str,
    message: str,
) -> None:
    reference = image_reference()
    reference.metadata[unsafe_key] = "value"

    with pytest.raises(ValueError, match=message):
        reference.durable_dict()


def test_context_reference_survives_checkpoint_round_trip() -> None:
    context = ExecutionContext(execution_id="execution-1")
    context.add_user_message("inspect", context_refs=[image_reference()])

    serialized = context.to_dict()
    restored = ExecutionContext.from_dict(serialized)

    assert restored.messages[0].context_refs == (image_reference(),)
    assert "file_path" not in json.dumps(serialized)
    assert "base64" not in json.dumps(serialized)


def test_tool_result_reserved_envelope_moves_refs_onto_message() -> None:
    context = ExecutionContext()

    message = context.add_tool_result(
        "inspect_asset",
        {
            "success": True,
            CONTEXT_REFS_KEY: [image_reference().durable_dict()],
        },
        tool_call_id="call-1",
    )

    assert message.context_refs == (image_reference(),)
    assert CONTEXT_REFS_KEY not in message.metadata["raw_result"]
    assert CONTEXT_REFS_KEY not in message.content


def test_tool_result_deduplicates_explicit_and_embedded_refs() -> None:
    context = ExecutionContext()

    message = context.add_tool_result(
        "inspect_asset",
        {
            "success": True,
            CONTEXT_REFS_KEY: [image_reference().durable_dict()],
        },
        context_refs=[image_reference()],
    )

    assert message.context_refs == (image_reference(),)


def test_compaction_transcript_preserves_file_id_without_binary_data() -> None:
    context = ExecutionContext()
    context.add_user_message("inspect", context_refs=[image_reference()])

    transcript = context._compact_transcript(context.messages)

    assert "file_id=image-1" in transcript
    assert "base64" not in transcript
    assert context.estimate_context_tokens() > len("inspect") // 4


def test_llm_compaction_deduplicates_refs_already_on_latest_user() -> None:
    context = ExecutionContext()
    context.add_user_message("inspect", context_refs=[image_reference()])
    context.add_assistant_message("continuing")
    context.add_user_message("use it again", context_refs=[image_reference()])

    context.compact_with_llm_response("Continue inspecting the same image.")

    assert context.messages[0].context_refs == ()
    assert context.messages[1].context_refs == (image_reference(),)


def test_llm_compaction_bounds_structured_refs_and_keeps_recent_handles() -> None:
    context = ExecutionContext()
    for index in range(5):
        context.add_user_message(
            f"inspect {index}",
            context_refs=[
                ContextReference(
                    file_ref={
                        "file_id": f"image-{index}",
                        "filename": f"frame-{index}.png",
                        "mime_type": "image/png",
                    },
                    detail=ImageDetail.HIGH,
                )
            ],
        )
    context.add_user_message("continue")

    result = context.compact_with_llm_response("Continue the image analysis.")

    retained_ids = [reference.file_id for reference in context.messages[0].context_refs]
    assert retained_ids == ["image-3", "image-4"]
    assert "file_id=image-2" in context.messages[0].content
    assert "file_id=image-0" in context.messages[0].content
    assert result.metadata["retained_context_ref_count"] == 2
    assert result.metadata["dropped_context_ref_count"] == 3
