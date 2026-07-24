"""Stable error schema for the ``/v1/*`` SDK surface.

Every error response on the SDK surface is shaped like:

    {
        "error": {
            "code": "<stable enum>",
            "message": "<human-readable, may change>"
        }
    }

The ``code`` is the contract SDK clients pin against; ``message`` is
free text that may be updated for clarity without breaking clients.

Why a separate error envelope from ``/api/*``:
    ``/api/*`` uses FastAPI's default ``{"detail": "..."}`` shape, which
    is fine for an in-house web UI but a poor SDK surface -- clients
    would have to string-match against ``detail`` to distinguish "wrong
    key" from "no such task" from "rate-limited". Coding the cause in a
    stable enum lets each SDK language map server errors to typed
    exceptions deterministically.
"""

from enum import Enum
from typing import Any, NoReturn

from fastapi import Request
from fastapi.responses import JSONResponse


class V1ErrorCode(str, Enum):
    """Stable error codes for ``/v1/*`` responses.

    Values are the on-the-wire strings; SDK clients pin against these.
    Adding new codes is allowed; renaming or removing existing ones is
    a breaking change.
    """

    # Authentication failures (401). One opaque code covers
    # "missing header", "format wrong", "prefix not found", "secret
    # wrong", and "key revoked" so attackers cannot distinguish them
    # by error code (we already keep the timing constant via
    # verify_dummy).
    INVALID_API_KEY = "invalid_api_key"

    # Caller is authenticated but the agent_id they asked for is not
    # bound to their key. Returned as 404 (not 403) so the existence
    # of other tenants' agents is not leaked.
    AGENT_NOT_FOUND = "agent_not_found"

    # Caller is authenticated but the workforce_id they asked for is
    # not bound to their key. Returned as 404 (not 403) for the same
    # leak-prevention reason as AGENT_NOT_FOUND.
    WORKFORCE_NOT_FOUND = "workforce_not_found"

    # The key-bound workforce has been archived; run creation and new
    # turns are permanently rejected (409 -- retrying cannot succeed).
    WORKFORCE_ARCHIVED = "workforce_archived"

    # The workforce exists but is not in the ``active`` status runs
    # require (e.g. still a draft). 409.
    WORKFORCE_NOT_ACTIVE = "workforce_not_active"

    # The workforce configuration changed since this conversation
    # started; the pinned run snapshot no longer matches, so appends
    # are permanently rejected (409). Client should create a new run.
    WORKFORCE_CONFIG_CHANGED = "workforce_config_changed"

    # An idempotency key was already used by a run whose task no longer
    # exists -- the original result can't be replayed and the key can't
    # be reused. 409.
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"

    # A file id passed in ``message.files`` on run creation isn't
    # accessible to the caller. 404 -- distinct from workforce_not_found
    # so SDK clients can tell "bad file" from "bad workforce".
    FILE_NOT_FOUND = "file_not_found"

    # Caller is authenticated and agent_id is right, but the task_id
    # they asked for doesn't exist or doesn't belong to their agent.
    # Also 404 for the same leak-prevention reason.
    TASK_NOT_FOUND = "task_not_found"

    # Template id is unknown or unavailable to the caller.
    TEMPLATE_NOT_FOUND = "template_not_found"

    # The task is currently running and cannot accept new input.
    # 409 Conflict. Client should retry after polling task status.
    TASK_BUSY = "task_busy"

    # Request body failed Pydantic validation (empty content, wrong role,
    # missing required field, etc.). 422. The default ``{"detail": [...]}``
    # FastAPI shape is rewritten to this enum on the /v1/* path so SDK
    # clients always switch on ``body.error.code``.
    INVALID_INPUT = "invalid_input"

    # Reserved for rate limiting. The server does not emit this yet, but
    # it stays in the enum so SDK clients can encode the mapping now.
    RATE_LIMITED = "rate_limited"

    # Monthly execution quota exhausted. 402 Payment Required. Two distinct
    # codes so a client application can tell "my own sub-quota is spent" apart
    # from "the whole team's ceiling is spent" — the team ceiling always wins,
    # so QUOTA_EXCEEDED means there is no headroom at all, while
    # CLIENT_QUOTA_EXCEEDED means the team still has room but this application's
    # slice is used up.
    QUOTA_EXCEEDED = "quota_exceeded"
    CLIENT_QUOTA_EXCEEDED = "client_quota_exceeded"

    # Server-side bug. Detail is sanitized; the raw exception stays in
    # the server log.
    INTERNAL_ERROR = "internal_error"

    CONNECTOR_NOT_FOUND = "connector_not_found"
    INVALID_RUNTIME_CONTEXT = "invalid_runtime_context"
    MISSING_RUNTIME_CONTEXT = "missing_runtime_context"
    RUNTIME_CONTEXT_IMMUTABLE = "runtime_context_immutable"
    RUNTIME_SECRET_NOT_ALLOWED = "runtime_secret_not_allowed"
    RUNTIME_SECRET_UNAVAILABLE = "runtime_secret_unavailable"
    SCHEDULED_SECRET_UNAVAILABLE = "scheduled_secret_unavailable"
    CONNECTOR_RUNTIME_UNAVAILABLE = "connector_runtime_unavailable"
    MCP_OAUTH_AUTHORIZATION_FAILED = "mcp_oauth_authorization_failed"
    DELEGATED_AUTHORIZATION_FAILED = "delegated_authorization_failed"


# Default human-readable text per code. Endpoints may override the
# message when more specific context is available (e.g. surfacing
# ``"task is running"`` vs the default ``"task is busy"``).
_DEFAULT_MESSAGES: dict[V1ErrorCode, str] = {
    V1ErrorCode.INVALID_API_KEY: "Invalid or revoked API key.",
    V1ErrorCode.AGENT_NOT_FOUND: "Agent not found or not accessible with this key.",
    V1ErrorCode.WORKFORCE_NOT_FOUND: (
        "Workforce not found or not accessible with this key."
    ),
    V1ErrorCode.WORKFORCE_ARCHIVED: (
        "This workforce has been archived; it can no longer accept new "
        "runs or messages."
    ),
    V1ErrorCode.WORKFORCE_NOT_ACTIVE: "Workforce must be active to run.",
    V1ErrorCode.WORKFORCE_CONFIG_CHANGED: (
        "The workforce configuration has changed since this conversation "
        "started; please create a new run."
    ),
    V1ErrorCode.IDEMPOTENCY_CONFLICT: (
        "This idempotency key was already used by a run that can no longer be replayed."
    ),
    V1ErrorCode.FILE_NOT_FOUND: "One or more file ids are not accessible.",
    V1ErrorCode.TASK_NOT_FOUND: "Task not found or not accessible with this key.",
    V1ErrorCode.TEMPLATE_NOT_FOUND: "Template not found.",
    V1ErrorCode.TASK_BUSY: "Task is currently running; retry after it completes.",
    V1ErrorCode.INVALID_INPUT: "Request body failed validation.",
    V1ErrorCode.RATE_LIMITED: "Rate limit exceeded; retry later.",
    V1ErrorCode.QUOTA_EXCEEDED: "Monthly execution quota exceeded for this team.",
    V1ErrorCode.CLIENT_QUOTA_EXCEEDED: (
        "Monthly execution quota exceeded for this client application."
    ),
    V1ErrorCode.INTERNAL_ERROR: "Internal server error.",
    V1ErrorCode.CONNECTOR_NOT_FOUND: "Connector not found or not accessible.",
    V1ErrorCode.INVALID_RUNTIME_CONTEXT: "Invalid connector runtime context.",
    V1ErrorCode.MISSING_RUNTIME_CONTEXT: "Required connector runtime context is missing.",
    V1ErrorCode.RUNTIME_CONTEXT_IMMUTABLE: "Connector runtime context cannot change after task creation.",
    V1ErrorCode.RUNTIME_SECRET_NOT_ALLOWED: "Runtime secret is not allowed for this entrypoint.",
    V1ErrorCode.RUNTIME_SECRET_UNAVAILABLE: "Required runtime secret is unavailable.",
    V1ErrorCode.SCHEDULED_SECRET_UNAVAILABLE: "Required scheduled runtime secret is unavailable.",
    V1ErrorCode.CONNECTOR_RUNTIME_UNAVAILABLE: "Connector runtime context is unavailable.",
    V1ErrorCode.MCP_OAUTH_AUTHORIZATION_FAILED: "MCP OAuth authorization is unavailable.",
    V1ErrorCode.DELEGATED_AUTHORIZATION_FAILED: "Delegated authorization failed.",
}


class V1ApiError(Exception):
    """Raise from any ``/v1/*`` handler to return a stable error response.

    Use this instead of ``HTTPException`` so the response envelope is
    consistent across the SDK surface. The exception handler in
    ``web/app.py`` translates this to:

        HTTP <http_status>
        {"error": {"code": <code>, "message": <message>}}

    Args:
        code: One of the :class:`V1ErrorCode` enum values.
        http_status: HTTP status code to send (401 / 404 / 409 / 429 / 500
            per code, but the endpoint chooses, not the code).
        message: Optional override of the default human-readable text.
    """

    def __init__(
        self,
        code: V1ErrorCode,
        http_status: int,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or _DEFAULT_MESSAGES[code])
        self.code = code
        self.http_status = http_status
        self.message = message or _DEFAULT_MESSAGES[code]
        self.details = details


# Turn-rejection reasons (``TaskTurnError.reason``) that are NOT the
# transient "busy" case. The default TASK_BUSY message tells the client
# to retry, which is actively misleading for workforce rejections where
# retrying can never succeed -- mirror of the WebSocket layer's
# ``_TURN_REJECTION_MESSAGES`` map, expressed as stable v1 error codes.
_TURN_REJECTION_CODES: dict[str, tuple[V1ErrorCode, int]] = {
    "workforce_archived": (V1ErrorCode.WORKFORCE_ARCHIVED, 409),
    "workforce_config_changed": (V1ErrorCode.WORKFORCE_CONFIG_CHANGED, 409),
    # The run row backing this conversation is gone; from the SDK's
    # perspective the task is simply no longer available.
    "workforce_run_not_found": (V1ErrorCode.TASK_NOT_FOUND, 404),
}


def raise_for_turn_rejection(reason: str) -> NoReturn:
    """Translate a ``TaskTurnError.reason`` into the v1 error envelope.

    Workforce-specific rejections get their own stable codes; anything
    else (``busy``, ``bg_inflight``, future reasons) stays the generic
    retryable ``task_busy`` 409.
    """
    code_status = _TURN_REJECTION_CODES.get(reason)
    if code_status is not None:
        code, http_status = code_status
        raise V1ApiError(code, http_status)
    raise V1ApiError(V1ErrorCode.TASK_BUSY, 409)


async def v1_api_error_handler(_request: Request, exc: V1ApiError) -> JSONResponse:
    """FastAPI exception handler -- mount via ``app.add_exception_handler``."""
    error: dict[str, Any] = {"code": exc.code.value, "message": exc.message}
    if exc.details is not None:
        error["details"] = exc.details
    return JSONResponse(status_code=exc.http_status, content={"error": error})
