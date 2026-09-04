"""Client-visible projections of server-side failures.

Holds the fixed fallback strings used when a failure has nothing safe to
say, the per-exception adapters that pass a curated message through, and
the projector that lifts a connector-runtime failure's code onto a
task_error frame.
"""

from enum import StrEnum

from ...core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_NOT_FOUND,
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ERROR_INVALID_RUNTIME_CONTEXT,
    ERROR_MISSING_RUNTIME_CONTEXT,
    ERROR_RUNTIME_CONTEXT_IMMUTABLE,
    ERROR_RUNTIME_SECRET_NOT_ALLOWED,
    ERROR_RUNTIME_SECRET_UNAVAILABLE,
    ERROR_SCHEDULED_SECRET_UNAVAILABLE,
    ConnectorRuntimeError,
)

CLIENT_SAFE_VALIDATION_ERROR = "The message could not be processed. Please try again."

# Task audiences did not necessarily initiate the failing operation, so a
# task-level failure uses neutral wording instead of the validation fallback.
CLIENT_SAFE_TASK_FAILURE = "Task execution failed."
CLIENT_SAFE_GUIDANCE_IN_PROGRESS = (
    "A previous guidance message is still being applied. Please wait for it to finish."
)


class ClientErrorCode(StrEnum):
    """Stable identifiers clients may localize without trusting server prose."""

    MESSAGE_PROCESSING_FAILED = "message_processing_failed"
    TASK_EXECUTION_FAILED = "task_execution_failed"
    GUIDANCE_IN_PROGRESS = "guidance_in_progress"
    MESSAGE_RATE_LIMITED = "message_rate_limited"
    MESSAGE_ID_CONFLICT = "message_id_conflict"
    MESSAGE_DELIVERY_FAILED = "message_delivery_failed"
    MESSAGE_CONTINUATION_UNSUPPORTED = "message_continuation_unsupported"
    TASK_PAUSE_IN_PROGRESS = "task_pause_in_progress"
    MESSAGE_ACCEPTANCE_PENDING = "message_acceptance_pending"
    TASK_UNAVAILABLE = "task_unavailable"
    TASK_BUSY = "task_busy"
    WORKFORCE_UNAVAILABLE = "workforce_unavailable"
    WORKFORCE_ARCHIVED = "workforce_archived"
    MESSAGE_ATTACHMENT_CORRUPT = "message_attachment_corrupt"
    MESSAGE_ATTACHMENT_UNAVAILABLE = "message_attachment_unavailable"
    TASK_CHECKPOINT_UNREADABLE = "task_checkpoint_unreadable"
    AUTHENTICATION_REQUIRED = "authentication_required"
    TASK_ACCESS_DENIED = "task_access_denied"
    INVALID_MESSAGE = "invalid_message"


def client_error_message(code: ClientErrorCode) -> str:
    """Return the fixed safe fallback for a stable client error code."""

    return {
        ClientErrorCode.MESSAGE_PROCESSING_FAILED: CLIENT_SAFE_VALIDATION_ERROR,
        ClientErrorCode.TASK_EXECUTION_FAILED: CLIENT_SAFE_TASK_FAILURE,
        ClientErrorCode.GUIDANCE_IN_PROGRESS: CLIENT_SAFE_GUIDANCE_IN_PROGRESS,
        ClientErrorCode.MESSAGE_RATE_LIMITED: (
            "You're sending messages too quickly. Please wait a moment and try again."
        ),
        ClientErrorCode.MESSAGE_ID_CONFLICT: (
            "Message id was already used for different content or files."
        ),
        ClientErrorCode.MESSAGE_DELIVERY_FAILED: (
            "The message could not be delivered. Please retry the draft."
        ),
        ClientErrorCode.MESSAGE_CONTINUATION_UNSUPPORTED: (
            "Task does not support message continuation."
        ),
        ClientErrorCode.TASK_PAUSE_IN_PROGRESS: (
            "Task pause is still being applied; please retry shortly."
        ),
        ClientErrorCode.MESSAGE_ACCEPTANCE_PENDING: (
            "Message acceptance is still being reconciled. Please retry shortly."
        ),
        ClientErrorCode.TASK_UNAVAILABLE: "Task is no longer available.",
        ClientErrorCode.TASK_BUSY: (
            "Task is currently busy; please wait for the previous turn to finish "
            "before sending another message."
        ),
        ClientErrorCode.WORKFORCE_UNAVAILABLE: (
            "This workforce conversation can no longer accept messages; "
            "please start a new conversation."
        ),
        ClientErrorCode.WORKFORCE_ARCHIVED: (
            "This workforce has been archived. Unarchive and publish it before "
            "starting a new conversation, or select an active workforce."
        ),
        ClientErrorCode.MESSAGE_ATTACHMENT_CORRUPT: (
            "A stored file for this message failed its integrity check "
            "and must be re-uploaded."
        ),
        ClientErrorCode.MESSAGE_ATTACHMENT_UNAVAILABLE: (
            "A stored file for this message could not be read. Please try again."
        ),
        ClientErrorCode.TASK_CHECKPOINT_UNREADABLE: (
            "The task's saved progress could not be read."
        ),
        ClientErrorCode.AUTHENTICATION_REQUIRED: (
            "Authentication is required to send this message."
        ),
        ClientErrorCode.TASK_ACCESS_DENIED: "You do not have access to this task.",
        ClientErrorCode.INVALID_MESSAGE: "The message format is invalid.",
    }[code]


def required_mcp_unavailable_client_message(
    error: BaseException,
    *,
    fallback: str = CLIENT_SAFE_VALIDATION_ERROR,
) -> str:
    """Adapt the curated required-MCP failure without opening a generic escape.

    The runtime check keeps this boundary fail-closed even if a future caller
    passes an incidental exception despite the function's specific name.
    """

    if not isinstance(error, RequiredMCPUnavailableError):
        return fallback
    message = str(error)
    if message.strip():
        return message
    return fallback


def connector_runtime_client_message(error: BaseException) -> str:
    """Adapt the curated connector-runtime failure without a generic escape.

    The runtime check keeps this boundary fail-closed even if a future caller
    passes an incidental exception despite the function's specific name.
    """

    if not isinstance(error, ConnectorRuntimeError):
        return CLIENT_SAFE_TASK_FAILURE
    message = error.safe_message
    if isinstance(message, str) and message.strip():
        return message
    return CLIENT_SAFE_TASK_FAILURE


def connector_runtime_client_code(error: BaseException) -> str | None:
    """Project a connector-runtime failure onto its wire-safe error code.

    Returns ``None`` for anything else, so a caller cannot widen the surface
    by passing an incidental exception. Membership in the client-visible
    closed set is checked by the frame builder, not here: this function
    only decides whether the exception is one we project at all.

    This is not the only client-visible projection of this exception.
    ``_raise_v1_connector_runtime_error`` (``web/api/v1/tasks.py``) projects
    it for the SDK surface and ships ``to_public_error()["details"]``
    whole, ``connector_ref`` included. The two differ because their
    audiences do: that one answers an API key held by a caller already
    authorized for the task, while this one feeds ``broadcast_to_task``,
    which reaches every connection under the task id including anonymous
    widget and share-link visitors. Keep them as two projectors with one
    audience each.
    """

    if not isinstance(error, ConnectorRuntimeError):
        return None
    code = error.code
    return code if isinstance(code, str) else None


# The connector-runtime codes a terminal task_error frame may carry. Every
# member is raised as a ``ConnectorRuntimeError`` somewhere in this
# repository today, and none of them states who owns the task or how an
# authorization check resolved -- the two questions a value has to answer
# "no" to before it may reach anonymous widget and share-link visitors.
# ``mcp_oauth_authorization_failed`` and ``delegated_authorization_failed``
# are deliberately absent: nothing here raises them as this exception, and
# each one is the outcome of an authorization check. Add a code here in the
# same change that adds the raise site, never ahead of it.
CONNECTOR_RUNTIME_CLIENT_ERROR_CODES = frozenset(
    {
        ERROR_CONNECTOR_NOT_FOUND,
        ERROR_INVALID_RUNTIME_CONTEXT,
        ERROR_MISSING_RUNTIME_CONTEXT,
        ERROR_RUNTIME_CONTEXT_IMMUTABLE,
        ERROR_RUNTIME_SECRET_NOT_ALLOWED,
        ERROR_RUNTIME_SECRET_UNAVAILABLE,
        ERROR_SCHEDULED_SECRET_UNAVAILABLE,
        ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    }
)
