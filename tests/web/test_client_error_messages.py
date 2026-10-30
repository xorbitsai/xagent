import re
from pathlib import Path

import pytest

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
from xagent.web.services.external_task_cancel import (
    EXTERNAL_CANCEL_NOT_APPLIED_MESSAGE,
    EXTERNAL_TURN_INTERRUPTED_MESSAGE,
)
from xagent.web.services.task_execution import (
    _CALLER_ASSERTED_COMMAND_ERROR_CODES,
    _CALLER_ASSERTED_TERMINAL_CODES,
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
    # The cancel core broadcasts and persists its own constant; this table
    # keeps an independent literal, so the two are tied here.
    assert (
        client_error_message(ClientErrorCode.EXTERNAL_TURN_INTERRUPTED)
        == EXTERNAL_TURN_INTERRUPTED_MESSAGE
    )
    assert (
        client_error_message(ClientErrorCode.EXTERNAL_STOP_NOT_APPLIED)
        == EXTERNAL_CANCEL_NOT_APPLIED_MESSAGE
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
        "excel_workbook_access_denied": (
            "The Excel workbook could not be opened or modified. Check that you "
            "have edit permission or reconnect the Excel connector."
        ),
        "excel_site_not_found": (
            "The Excel workbook or its SharePoint location could not be found. "
            "Check the workbook path or reconnect the Excel connector."
        ),
        "excel_workbook_locked": "The Excel workbook is locked. Close it elsewhere and try again.",
        "invalid_message": "The message format is invalid.",
        "message_outcome_unknown": (
            "The message may or may not have been applied. Check the "
            "conversation before sending it again."
        ),
        "external_turn_interrupted": "This response was interrupted.",
        "external_stop_not_applied": (
            "Stopping this response didn't go through — please try again."
        ),
        "model_error": "Model provider call failed.",
        "execution_paused": (
            "New work is briefly paused for a service update. "
            "Nothing was started; please try again in a moment."
        ),
        "auto_resume_not_cancellable": (
            "Automatic resume of a task created through the API cannot be "
            "cancelled here."
        ),
        "auto_resume_cancel_raced": (
            "The task changed while cancelling its automatic resume; please try again."
        ),
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


_FRONTEND_TABLE = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "lib"
    / "client-errors.ts"
)


def test_caller_asserted_code_sets_are_disjoint() -> None:
    """A caller-asserted code names one outcome on one kind of frame.

    The terminal ``task_error`` frame and the external command ``agent_error``
    frame admit codes from separate sets. A code in both would let one frame
    claim the other's outcome: "the turn ended interrupted" and "the stop may
    not have taken effect" cannot share a code.
    """
    assert not (_CALLER_ASSERTED_TERMINAL_CODES & _CALLER_ASSERTED_COMMAND_ERROR_CODES)


@pytest.mark.parametrize(
    "code",
    sorted(_CALLER_ASSERTED_TERMINAL_CODES | _CALLER_ASSERTED_COMMAND_ERROR_CODES),
)
def test_every_caller_asserted_code_is_in_the_frontend_error_table(
    code: ClientErrorCode,
) -> None:
    """A caller-asserted code is only useful if the client table lists it.

    A client that does not render server prose shows a generic fallback for
    any code ``CLIENT_ERROR_CODES`` does not list, so renaming either side
    alone would silently undo the localized wording.
    """
    if not _FRONTEND_TABLE.exists():
        pytest.skip(f"frontend source not present: {_FRONTEND_TABLE}")
    source = _FRONTEND_TABLE.read_text(encoding="utf-8")
    table = source.split("export const CLIENT_ERROR_CODES = [", 1)
    assert len(table) == 2, "could not parse: CLIENT_ERROR_CODES moved"
    listed = set(re.findall(r'^\s*"(\w+)",', table[1].split("] as const", 1)[0], re.M))
    assert code.value in listed
