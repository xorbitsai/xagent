from xagent.core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from xagent.web.services.client_error_messages import (
    CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE,
    CLIENT_SAFE_GUIDANCE_IN_PROGRESS,
    CLIENT_SAFE_TASK_FAILURE,
    CLIENT_SAFE_VALIDATION_ERROR,
    ClientErrorCode,
    client_error_message,
    required_mcp_unavailable_client_message,
)


def test_client_error_codes_have_fixed_safe_fallbacks() -> None:
    assert (
        client_error_message(ClientErrorCode.MESSAGE_PROCESSING_FAILED)
        == CLIENT_SAFE_VALIDATION_ERROR
    )
    assert (
        client_error_message(ClientErrorCode.TASK_EXECUTION_FAILED)
        == CLIENT_SAFE_TASK_FAILURE
    )
    assert (
        client_error_message(ClientErrorCode.AUTO_MODEL_UNAVAILABLE)
        == CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE
    )
    assert (
        client_error_message(ClientErrorCode.GUIDANCE_IN_PROGRESS)
        == CLIENT_SAFE_GUIDANCE_IN_PROGRESS
    )
    assert {code.value: client_error_message(code) for code in ClientErrorCode} == {
        "message_processing_failed": "The message could not be processed. Please try again.",
        "task_execution_failed": "Task execution failed.",
        "auto_model_unavailable": (
            "Your Auto model configuration has no usable candidate models. "
            "Review your Auto model settings."
        ),
        "guidance_in_progress": (
            "A previous guidance message is still being applied. "
            "Please wait for it to finish."
        ),
        "message_rate_limited": (
            "You're sending messages too quickly. Please wait a moment and try again."
        ),
        "message_id_conflict": (
            "Message id was already used for different content or files."
        ),
        "message_delivery_failed": (
            "The message could not be delivered. Please retry the draft."
        ),
        "message_continuation_unsupported": (
            "Task does not support message continuation."
        ),
        "task_pause_in_progress": (
            "Task pause is still being applied; please retry shortly."
        ),
        "message_acceptance_pending": (
            "Message acceptance is still being reconciled. Please retry shortly."
        ),
        "task_unavailable": "Task is no longer available.",
        "task_busy": (
            "Task is currently busy; please wait for the previous turn to finish "
            "before sending another message."
        ),
        "workforce_unavailable": (
            "This workforce conversation can no longer accept messages; "
            "please start a new conversation."
        ),
        "workforce_archived": (
            "This workforce has been archived. Unarchive and publish it before "
            "starting a new conversation, or select an active workforce."
        ),
        "message_attachment_corrupt": (
            "A stored file for this message failed its integrity check "
            "and must be re-uploaded."
        ),
        "message_attachment_unavailable": (
            "A stored file for this message could not be read. Please try again."
        ),
        "task_checkpoint_unreadable": "The task's saved progress could not be read.",
        "authentication_required": "Authentication is required to send this message.",
        "task_access_denied": "You do not have access to this task.",
        "invalid_message": "The message format is invalid.",
    }


def test_required_mcp_error_preserves_its_curated_client_message() -> None:
    error = RequiredMCPUnavailableError([])

    assert required_mcp_unavailable_client_message(error) == str(error)


def test_required_mcp_adapter_rejects_incidental_exceptions() -> None:
    error = RuntimeError("provider token=secret")

    assert (
        required_mcp_unavailable_client_message(
            error,
            fallback=CLIENT_SAFE_TASK_FAILURE,
        )
        == CLIENT_SAFE_TASK_FAILURE
    )
