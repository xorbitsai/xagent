"""Client-visible projections of server-side failures.

Holds the fixed fallback strings used when a failure has nothing safe to
say, the per-exception adapters that pass a curated message through, the
projector that lifts a connector-runtime failure's code onto a task_error
frame, and the projector that turns a recorded model-provider failure into a
bounded message for task-stream audiences. It also provides the credential
masking used for the owner-facing ``tasks.error_message`` text.
"""

import re
import unicodedata
from enum import StrEnum
from typing import Any

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
from ...core.utils.security import redact_sensitive_text

CLIENT_SAFE_VALIDATION_ERROR = "The message could not be processed. Please try again."

# Task audiences did not necessarily initiate the failing operation, so a
# task-level failure uses neutral wording instead of the validation fallback.
CLIENT_SAFE_TASK_FAILURE = "Task execution failed."


CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE = (
    "Your Auto model configuration has no usable candidate models. "
    "Review your Auto model settings."
)
CLIENT_SAFE_GUIDANCE_IN_PROGRESS = (
    "A previous guidance message is still being applied. Please wait for it to finish."
)

# Head of every model-provider failure sentence shown to task-stream audiences.
MODEL_ERROR_CLIENT_HEAD = "Model provider call failed"
# Fixed fallback for ``ClientErrorCode.MODEL_ERROR``; equals the projection of a
# failure that carries no status, no provider code and no kind phrase.
CLIENT_SAFE_MODEL_ERROR = f"{MODEL_ERROR_CLIENT_HEAD}."
MODEL_ERROR_CLIENT_MESSAGE_MAX_CHARS = 200
MODEL_ERROR_PROVIDER_CODE_MAX_CHARS = 64
# Statuses whose provider message body may reach task-stream audiences. Every
# other status carries no body: 400 keeps the generic task-failure sentence, 401
# bodies echo the submitted credential, and a missing status says nothing about
# who produced the text.
MODEL_ERROR_BODY_STATUSES = frozenset({403, 404, 408, 409, 429}) | frozenset(
    range(500, 600)
)
_PROVIDER_CODE_PATTERN = re.compile(
    rf"^[A-Za-z0-9_.:-]{{1,{MODEL_ERROR_PROVIDER_CODE_MAX_CHARS}}}$"
)
# One short clause per failure kind, used when the body is withheld. ``unknown``
# has no phrase: the tail is omitted.
MODEL_ERROR_KIND_PHRASES: dict[str, str | None] = {
    "timeout": "the request timed out",
    "connection_failed": "the provider could not be reached",
    "bad_request": "the provider rejected the request",
    "authentication_failed": "the credential was rejected",
    "access_denied": "access to the model was denied",
    "not_found": "the model was not found",
    "rate_limited": "the provider rate limit was reached",
    "server_error": "the provider returned a server error",
    "rejected": "the provider rejected the request",
    "unknown": None,
}


class ClientErrorCode(StrEnum):
    """Stable identifiers clients may localize without trusting server prose."""

    MESSAGE_PROCESSING_FAILED = "message_processing_failed"
    TASK_EXECUTION_FAILED = "task_execution_failed"
    AUTO_MODEL_UNAVAILABLE = "auto_model_unavailable"
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
    EXCEL_WORKBOOK_ACCESS_DENIED = "excel_workbook_access_denied"
    EXCEL_SITE_NOT_FOUND = "excel_site_not_found"
    EXCEL_WORKBOOK_LOCKED = "excel_workbook_locked"
    INVALID_MESSAGE = "invalid_message"
    MESSAGE_OUTCOME_UNKNOWN = "message_outcome_unknown"
    # The turn ended interrupted after an external stop was requested (that
    # stop, or a shutdown that settled the turn first). Only the external
    # cancel core puts it on a frame, by passing it to the shared terminal
    # builder as ``asserted_code``; passed as ``code`` it is dropped.
    # TerminalTaskEventMessageCode has a member with the same value and a
    # different meaning: on a cancel command's audit record it means the
    # command ended in failure while its task was already COMPLETED or
    # FAILED. That record never reaches a client.
    EXTERNAL_TURN_INTERRUPTED = "external_turn_interrupted"
    # An external stop request ended without the server confirming it took
    # effect, and the server could not confirm the turn is over: it may still
    # be running, or its row may be unreadable. Only the terminal command
    # broadcast for an external-scope cancel puts it on a frame, as
    # ``error_code`` on an ``agent_error``, by passing it to that frame's
    # builder as ``asserted_code``.
    EXTERNAL_STOP_NOT_APPLIED = "external_stop_not_applied"
    MODEL_ERROR = "model_error"
    # New execution is briefly paused on this server for a planned restart
    # (``release_hold``); nothing was accepted, and the same request may be
    # sent again shortly.
    EXECUTION_PAUSED = "execution_paused"
    # A task whose caller only reads its status (an SDK task): a cancelled
    # automatic resume would leave that caller waiting on a pause forever.
    AUTO_RESUME_NOT_CANCELLABLE = "auto_resume_not_cancellable"
    # Cancelling an automatic resume kept losing a race with another write to
    # the task; nothing was changed, and the cancel may be sent again.
    AUTO_RESUME_CANCEL_RACED = "auto_resume_cancel_raced"


def client_error_message(code: ClientErrorCode) -> str:
    """Return the fixed safe fallback for a stable client error code."""

    return {
        ClientErrorCode.MESSAGE_PROCESSING_FAILED: CLIENT_SAFE_VALIDATION_ERROR,
        ClientErrorCode.TASK_EXECUTION_FAILED: CLIENT_SAFE_TASK_FAILURE,
        ClientErrorCode.AUTO_MODEL_UNAVAILABLE: CLIENT_SAFE_AUTO_MODEL_UNAVAILABLE,
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
        ClientErrorCode.EXCEL_WORKBOOK_ACCESS_DENIED: (
            "The Excel workbook could not be opened or modified. Check that you "
            "have edit permission or reconnect the Excel connector."
        ),
        ClientErrorCode.EXCEL_SITE_NOT_FOUND: (
            "The Excel workbook or its SharePoint location could not be found. "
            "Check the workbook path or reconnect the Excel connector."
        ),
        ClientErrorCode.EXCEL_WORKBOOK_LOCKED: (
            "The Excel workbook is locked. Close it elsewhere and try again."
        ),
        ClientErrorCode.INVALID_MESSAGE: "The message format is invalid.",
        ClientErrorCode.MESSAGE_OUTCOME_UNKNOWN: (
            "The message may or may not have been applied. Check the "
            "conversation before sending it again."
        ),
        ClientErrorCode.EXTERNAL_TURN_INTERRUPTED: "This response was interrupted.",
        ClientErrorCode.EXTERNAL_STOP_NOT_APPLIED: (
            "Stopping this response didn't go through — please try again."
        ),
        ClientErrorCode.MODEL_ERROR: CLIENT_SAFE_MODEL_ERROR,
        ClientErrorCode.EXECUTION_PAUSED: (
            "New work is briefly paused for a service update. "
            "Nothing was started; please try again in a moment."
        ),
        ClientErrorCode.AUTO_RESUME_NOT_CANCELLABLE: (
            "Automatic resume of a task created through the API cannot be "
            "cancelled here."
        ),
        ClientErrorCode.AUTO_RESUME_CANCEL_RACED: (
            "The task changed while cancelling its automatic resume; please try again."
        ),
    }[code]


def with_task_reference(message: str, task_id: int | None) -> str:
    """Append the reportable task id to a client-visible failure line."""
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
        return message
    return f"{message} (Task ID: {task_id})"


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


# Credential shapes masked in provider text. A token counts as starting
# where no ASCII letter or digit precedes it, so ``my_sk-…`` is caught;
# ``key`` and ``token`` are left out as prefixes on purpose: they would also
# mask ordinary words such as ``token_limit_exceeded_for_model``.
_SECRET_PREFIX_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(sk|pk|rk|gsk|hf|xai|nvapi)[-_][A-Za-z0-9_-]{8,}"
)
# Account identifiers, masked only for task-stream audiences: the owner text
# keeps them because a ``proj-`` prefix also starts some model names.
_ACCOUNT_ID_PREFIX_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(org|proj)[-_][A-Za-z0-9_-]{8,}"
)
_JWT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_.-]+)?"
)
_GOOGLE_API_KEY_PATTERN = re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{20,}")
_AWS_ACCESS_KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"
)
# A bare ``Bearer <token>``. The token must hold a digit and be at least 16
# characters long, so ``Bearer authentication required`` stays as it is.
_BARE_BEARER_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])(bearer\s+)(?=[A-Za-z0-9._~+/-]*[0-9])[A-Za-z0-9._~+/-]{16,}=*"
)
# A quoted ``"api_key": "…"`` pair, which ``redact_sensitive_text`` does not
# recognise (it handles ``identifier=value``).
_QUOTED_CREDENTIAL_PATTERN = re.compile(
    r"""(?i)(["'](?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|refresh[_-]?token|secret|secret[_-]?key|client[_-]?secret|password|token)["']\s*:\s*["'])[^"'\s]+(["'])"""
)
_STRIPPED_CHARACTER_CATEGORIES = frozenset({"Cc", "Cf"})


def mask_provider_secrets(text: str) -> str:
    """Mask credential-shaped tokens in provider-authored text.

    ``redact_sensitive_text`` runs first so the header and assignment shapes
    it knows keep their usual ``***1234`` form; the shape patterns above
    then mask what it does not recognise. Account identifiers are not
    touched here.
    """

    masked = redact_sensitive_text(text)
    masked = _SECRET_PREFIX_PATTERN.sub(lambda match: f"{match.group(1)}-***", masked)
    masked = _JWT_PATTERN.sub("***", masked)
    masked = _GOOGLE_API_KEY_PATTERN.sub("***", masked)
    masked = _AWS_ACCESS_KEY_PATTERN.sub("***", masked)
    masked = _BARE_BEARER_PATTERN.sub(lambda match: f"{match.group(1)}***", masked)
    return _QUOTED_CREDENTIAL_PATTERN.sub(
        lambda match: f"{match.group(1)}***{match.group(2)}", masked
    )


def _clean_provider_text(text: str) -> str:
    """Make one provider message safe and short enough for task-stream audiences.

    Whitespace is collapsed before control and format characters are dropped, so
    separate lines stay separate words. Account identifiers are masked next,
    then ``mask_provider_secrets`` masks the credential shapes, and the result
    is capped at ``MODEL_ERROR_CLIENT_MESSAGE_MAX_CHARS`` characters.
    """

    collapsed = " ".join(text.split())
    visible = "".join(
        ch
        for ch in collapsed
        if unicodedata.category(ch) not in _STRIPPED_CHARACTER_CATEGORIES
    )
    masked = _ACCOUNT_ID_PREFIX_PATTERN.sub(
        lambda match: f"{match.group(1)}-***", visible
    )
    return mask_provider_secrets(masked)[:MODEL_ERROR_CLIENT_MESSAGE_MAX_CHARS]


def model_error_client_projection(value: object) -> dict[str, Any] | None:
    """Project a recorded model-provider failure for task-stream audiences.

    ``value`` is the dict a ``ModelProviderError`` publishes (``kind``,
    ``status_code``, ``provider_code``, ``message``). Every field is validated
    by exact type; anything that fails validation is treated as missing.
    Returns ``None`` when ``value`` is not a dict or the failure is an HTTP 400,
    whose body never reaches these audiences: callers then show the generic
    task-failure sentence. Otherwise returns ``error_code``, ``error_message``,
    ``kind``, ``status_code`` and ``provider_code``. The provider ``message``
    appears only for ``MODEL_ERROR_BODY_STATUSES``, cleaned and capped.
    """

    if type(value) is not dict:
        return None
    raw_kind = value.get("kind")
    kind = (
        raw_kind
        if type(raw_kind) is str and raw_kind in MODEL_ERROR_KIND_PHRASES
        else "unknown"
    )
    raw_status = value.get("status_code")
    status_code = (
        raw_status if type(raw_status) is int and 100 <= raw_status <= 599 else None
    )
    if status_code == 400:
        return None
    raw_code = value.get("provider_code")
    provider_code = (
        raw_code
        if type(raw_code) is str and _PROVIDER_CODE_PATTERN.fullmatch(raw_code)
        else None
    )
    message: str | None = None
    raw_message = value.get("message")
    if (
        status_code in MODEL_ERROR_BODY_STATUSES
        and type(raw_message) is str
        and raw_message.strip()
    ):
        try:
            message = _clean_provider_text(raw_message)
        except Exception:
            message = None

    parts = [str(status_code) if status_code is not None else None, provider_code]
    joined = " ".join(part for part in parts if part is not None)
    paren = f" ({joined})" if joined else ""
    phrase = MODEL_ERROR_KIND_PHRASES[kind]
    if message:
        text = f"{MODEL_ERROR_CLIENT_HEAD}{paren}: {message}"
    elif phrase:
        text = f"{MODEL_ERROR_CLIENT_HEAD}{paren}: {phrase}."
    else:
        text = f"{MODEL_ERROR_CLIENT_HEAD}{paren}."
    return {
        "error_code": ClientErrorCode.MODEL_ERROR.value,
        "error_message": text,
        "kind": kind,
        "status_code": status_code,
        "provider_code": provider_code,
    }
