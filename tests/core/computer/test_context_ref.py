from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from xagent.core.agent.context import ExecutionContext, Message
from xagent.core.context_ref import (
    CONTEXT_REFS_KEY,
    ContextReference,
    ContextReferencePurpose,
)


def reference() -> ContextReference:
    return ContextReference(
        file_ref={
            "file_id": "image-1",
            "filename": "frame.png",
            "mime_type": "image/png",
            "file_path": "/private/frame.png",
        },
        purpose=ContextReferencePurpose.OBSERVATION,
        frame_id="frame-1",
    )


def test_legacy_message_model_payload_is_unchanged_without_refs() -> None:
    message = Message(role="user", content="hello")

    assert message.to_dict() == {"role": "user", "content": "hello"}


def test_message_model_payload_contains_only_durable_reference() -> None:
    message = Message(role="user", content="inspect", context_refs=(reference(),))

    payload = message.to_dict()

    assert payload["content"] == "inspect"
    assert payload[CONTEXT_REFS_KEY][0]["file_ref"]["file_id"] == "image-1"
    assert "file_path" not in json.dumps(payload)
    assert "base64" not in json.dumps(payload)


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


def test_durable_dict_resanitizes_nested_mutations() -> None:
    context_ref = reference()
    context_ref.file_ref["file_path"] = "/private/late-mutation.png"
    context_ref.metadata["image"] = "data:image/png;base64,c2NyZWVuc2hvdA=="

    with pytest.raises(ValueError, match="materialized image data"):
        context_ref.durable_dict()

    context_ref.metadata.pop("image")
    assert "file_path" not in context_ref.durable_dict()["file_ref"]


def test_context_reference_survives_checkpoint_round_trip() -> None:
    context = ExecutionContext(execution_id="execution-1")
    context.add_user_message("inspect", context_refs=[reference()])

    serialized = context.to_dict()
    restored = ExecutionContext.from_dict(serialized)

    assert restored.messages[0].context_refs == (reference(),)
    assert "file_path" not in json.dumps(serialized)
    assert "base64" not in json.dumps(serialized)


def test_tool_result_can_carry_durable_observation_reference() -> None:
    context = ExecutionContext()

    message = context.add_tool_result(
        "computer",
        {"success": True, "frame_id": "frame-1"},
        tool_call_id="call-1",
        context_refs=[reference()],
    )

    assert message.role == "tool"
    assert message.context_refs == (reference(),)
    assert message.to_dict()[CONTEXT_REFS_KEY][0]["frame_id"] == "frame-1"


def test_tool_result_reserved_envelope_moves_refs_onto_message() -> None:
    context = ExecutionContext()

    message = context.add_tool_result(
        "computer",
        {
            "success": True,
            "frame_id": "frame-1",
            CONTEXT_REFS_KEY: [reference().durable_dict()],
        },
        tool_call_id="call-1",
    )

    assert message.context_refs == (reference(),)
    assert CONTEXT_REFS_KEY not in message.metadata["raw_result"]
    assert CONTEXT_REFS_KEY not in message.content


def test_tool_result_deduplicates_explicit_and_embedded_refs() -> None:
    context = ExecutionContext()

    message = context.add_tool_result(
        "computer",
        {
            "success": True,
            CONTEXT_REFS_KEY: [reference().durable_dict()],
        },
        context_refs=[reference()],
    )

    assert message.context_refs == (reference(),)


def test_compaction_transcript_preserves_file_id_without_binary_data() -> None:
    context = ExecutionContext()
    context.add_user_message("inspect", context_refs=[reference()])

    transcript = context._compact_transcript(context.messages)

    assert "file_id=image-1" in transcript
    assert "base64" not in transcript
    assert context.estimate_context_tokens() > len("inspect") // 4
