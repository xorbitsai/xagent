from xagent.web.api.websocket import (
    _append_uploaded_files_context_to_message,
    _build_uploaded_files_context,
)


def test_build_uploaded_files_context_includes_agent_builder_kb_instruction():
    context = _build_uploaded_files_context(
        [
            {
                "file_id": "file-123",
                "name": "faq.docx",
                "original_name": "FAQ.docx",
            }
        ],
        is_agent_builder=True,
    )

    assert "FAQ.docx: file_id=file-123" in context
    assert "create_knowledge_base_from_file" in context
    assert 'file_ids = ["file-123"]' in context
    assert "Do NOT ask the user to upload again" in context


def test_append_uploaded_files_context_to_message_is_idempotent():
    context = _build_uploaded_files_context(
        [{"file_id": "file-123", "name": "faq.docx"}],
        is_agent_builder=False,
    )

    message = _append_uploaded_files_context_to_message("Upload File", context)
    assert message.startswith("Upload File\n\n## UPLOADED FILES")
    assert _append_uploaded_files_context_to_message(message, context) == message
