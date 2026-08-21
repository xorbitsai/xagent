from xagent.core.agent.transcript import normalize_transcript_messages
from xagent.core.context_ref import CONTEXT_REFS_KEY, ContextReference


def _image_reference() -> ContextReference:
    return ContextReference(
        file_ref={
            "file_id": "image-id",
            "filename": "diagram.png",
            "mime_type": "image/png",
        },
        metadata={"source": "user_upload"},
    )


def test_normalize_transcript_retains_refs_only_message_and_alias() -> None:
    reference = _image_reference()

    normalized = normalize_transcript_messages(
        [
            {
                "role": "user",
                "content": "",
                "context_refs": [reference.durable_dict()],
            }
        ]
    )

    assert normalized == [
        {
            "role": "user",
            "content": "",
            CONTEXT_REFS_KEY: [reference.durable_dict()],
        }
    ]


def test_normalize_transcript_filters_malformed_refs_without_losing_text() -> None:
    normalized = normalize_transcript_messages(
        [
            {
                "role": "user",
                "content": "Keep this text",
                CONTEXT_REFS_KEY: [{"type": "image"}],
            },
            {
                "role": "user",
                "content": "",
                CONTEXT_REFS_KEY: [{"type": "image"}],
            },
        ]
    )

    assert normalized == [{"role": "user", "content": "Keep this text"}]
