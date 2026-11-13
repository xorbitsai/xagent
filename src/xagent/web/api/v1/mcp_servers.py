"""SDK endpoints for the MCP connectors the key holder can see and create."""

import asyncio
import logging
import math
import re
import weakref
from dataclasses import dataclass
from typing import Any, Callable, NoReturn, Optional, Sequence

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path

from ....config import get_mcp_tool_init_timeout_seconds
from ....core.tools.adapters.vibe.config import (
    CLIENT_REJECTED_REASON,
    is_client_rejected_reason,
)
from ....core.tools.core.mcp.http import _origin as mcp_endpoint_origin
from ....core.tools.core.mcp.model import MASKED_SECRET_VALUE, SENSITIVE_AUTH_FIELDS
from ....core.utils.public_http_transport import UnsafePublicHttpTarget
from ...models.database import get_session_local
from ...models.user import User
from ...schemas.v1 import (
    V1McpServer,
    V1McpServerCreateRequest,
    V1McpServerDeleteResponse,
    V1McpServerEnabledRequest,
    V1McpServerTools,
    V1McpServerUpdateRequest,
    V1McpTool,
    V1McpToolLoadFailure,
)
from ...services import mcp_connector_session
from ...services.connector_name_policy import (
    catalog_app_name_detail,
    folded_name_conflict_detail,
    has_unfoldable_edge_whitespace,
    unfoldable_edge_whitespace_detail,
)
from ...services.mcp_connector_session import (
    ConnectorToolsOutcome,
    ConnectorToolsResult,
)
from ...services.mcp_server_management import (
    _AUTHORIZATION_MESSAGE,
    _CATALOG_AUTH_KEYS,
    _REMOTE_TRANSPORTS,
    McpServerManagementRuntime,
)
from .deps import (
    PersonalApiKeySnapshot,
    UserPrincipalSnapshot,
    get_user_from_personal_key,
)
from .errors import V1ApiError, V1ErrorCode

router = APIRouter(prefix="/mcp-servers")

logger = logging.getLogger(__name__)

_INVALID_CONFIGURATION_PREFIX = "Invalid configuration: "
_MAX_PASSED_THROUGH_CHARS = 1000
_REASON_PATTERN = re.compile(r"[a-z_]{1,64}")


def _summarize_invalid_configuration(detail: str) -> str:
    """The first validation error of a web "Invalid configuration" message.

    The web text carries pydantic's full report, whose reason lines end with
    the rejected input (``input_value=...``), and a model-level error's input
    is the whole request, secrets included. Only the head line, the field path
    and the reason are kept, and only for the one layout this function can
    recognize; anything else yields the head line alone.
    """
    lines = detail.split("\n")
    head = lines[0]
    for i in range(1, len(lines)):
        line = lines[i]
        # pydantic indents each error's reason by exactly two spaces and its
        # "For further information" line by four.
        if line.startswith("  ") and not line.startswith("   "):
            if " [type=" not in line:
                return head
            reason = line[2:].split(" [type=", 1)[0]
            previous = lines[i - 1]
            if i - 1 >= 1 and not previous.startswith(" "):
                return f"{head}: {previous}: {reason}"
            return f"{head}: {reason}"
    return head


def _passed_through_message(detail: Any, secrets: Sequence[str]) -> Optional[str]:
    """The part of a web 400 message that is safe to show, or ``None``."""
    if not isinstance(detail, str):
        return None
    if detail.startswith(_INVALID_CONFIGURATION_PREFIX) and "\n" in detail:
        message = _summarize_invalid_configuration(detail)
    else:
        message = detail.split("\n", 1)[0]
    if (
        not message
        or len(message) > _MAX_PASSED_THROUGH_CHARS
        or any(secret in message for secret in secrets)
    ):
        return None
    return message


def _is_exact_name_conflict(exc: HTTPException, name: str) -> bool:
    return (
        exc.status_code == 400 and exc.detail == f"MCP server '{name}' already exists"
    )


@dataclass(frozen=True)
class _WriteErrorRow:
    """One way a web write can fail, and the SDK error it becomes.

    ``message`` returns the text to show, or ``None`` for the code's default.
    """

    id: str
    matches: Callable[[HTTPException, str, Sequence[str]], bool]
    status: int
    code: V1ErrorCode
    message: Callable[[HTTPException, str, Sequence[str]], Optional[str]]


def _no_message(_exc: HTTPException, _name: str, _secrets: Sequence[str]) -> None:
    return None


def _is_bad_request(exc: HTTPException) -> bool:
    return exc.status_code == 400


def _has_status(
    status_code: int,
) -> Callable[[HTTPException, str, Sequence[str]], bool]:
    return lambda exc, _name, _secrets: exc.status_code == status_code


# The web update answers this when the caller's account disappears while it
# waits for the definition row lock.
_ACCOUNT_GONE_DETAIL = "Requesting user account no longer exists"

_MANAGED_ELSEWHERE_MESSAGE = (
    "This connector has connections managed outside your account; remove it in "
    "the web app."
)


# The first matching row wins; the last row matches every remaining failure.
_MCP_WRITE_ERROR_ROWS: tuple[_WriteErrorRow, ...] = (
    _WriteErrorRow(
        "exact_name_conflict",
        lambda exc, name, _s: _is_exact_name_conflict(exc, name),
        409,
        V1ErrorCode.CONNECTOR_NAME_CONFLICT,
        _no_message,
    ),
    _WriteErrorRow(
        "folded_name_conflict",
        lambda exc, name, _s: (
            _is_bad_request(exc) and exc.detail == folded_name_conflict_detail(name)
        ),
        409,
        V1ErrorCode.CONNECTOR_NAME_CONFLICT,
        _no_message,
    ),
    _WriteErrorRow(
        "catalog_name_conflict",
        lambda exc, name, _s: (
            _is_bad_request(exc) and exc.detail == catalog_app_name_detail(name)
        ),
        409,
        V1ErrorCode.CONNECTOR_NAME_CONFLICT,
        lambda _exc, name, _s: catalog_app_name_detail(name),
    ),
    _WriteErrorRow(
        "passed_through_message",
        lambda exc, _n, secrets: (
            _is_bad_request(exc)
            and _passed_through_message(exc.detail, secrets) is not None
        ),
        400,
        V1ErrorCode.INVALID_INPUT,
        lambda exc, _n, secrets: _passed_through_message(exc.detail, secrets),
    ),
    _WriteErrorRow(
        "unclassified_bad_request",
        lambda exc, _n, _s: _is_bad_request(exc),
        400,
        V1ErrorCode.INVALID_INPUT,
        _no_message,
    ),
    _WriteErrorRow(
        "forbidden",
        _has_status(403),
        403,
        V1ErrorCode.CONNECTOR_FORBIDDEN,
        _no_message,
    ),
    _WriteErrorRow(
        "account_gone",
        lambda exc, _n, _s: (
            exc.status_code == 404 and exc.detail == _ACCOUNT_GONE_DETAIL
        ),
        401,
        V1ErrorCode.INVALID_API_KEY,
        _no_message,
    ),
    _WriteErrorRow(
        "not_found",
        _has_status(404),
        404,
        V1ErrorCode.CONNECTOR_NOT_FOUND,
        _no_message,
    ),
    _WriteErrorRow(
        "managed_elsewhere",
        _has_status(409),
        409,
        V1ErrorCode.CONNECTOR_NOT_EDITABLE,
        _no_message,
    ),
    _WriteErrorRow(
        "internal_error",
        lambda _exc, _n, _s: True,
        500,
        V1ErrorCode.INTERNAL_ERROR,
        _no_message,
    ),
)


def _raise_v1_from_mcp_write_error(
    exc: HTTPException,
    *,
    action: str,
    requested_name: str,
    secrets: Sequence[str],
    owned_connector_id: Optional[int] = None,
) -> NoReturn:
    """Translate an ``HTTPException`` from a web connector write.

    ``secrets`` are the request's header values and string auth values; a web
    message containing one is not shown. ``owned_connector_id`` is the holder's
    own connector of exactly the requested name, so a client that retries a
    create whose answer it lost can find the connector it already made.
    """
    for row in _MCP_WRITE_ERROR_ROWS:
        if not row.matches(exc, requested_name, secrets):
            continue
        if row.code is V1ErrorCode.INTERNAL_ERROR:
            # The web detail can carry exception text, so only its status is logged.
            logger.error(
                "personal key mcp server write failed: action=%s status=%s",
                action,
                exc.status_code,
            )
        message = row.message(exc, requested_name, secrets)
        details: Optional[dict[str, Any]] = None
        if row.id == "exact_name_conflict" and owned_connector_id is not None:
            details = {"connector_id": owned_connector_id}
        elif row.id == "forbidden":
            details = {"reason": "not_permitted"}
        elif row.id == "managed_elsewhere":
            # Of the web writes called here, only the delete answers 409: the
            # connector has OAuth connections another account manages. Any
            # other action has no such 409 today.
            if action == "delete":
                message = _MANAGED_ELSEWHERE_MESSAGE
                details = {"reason": "managed_elsewhere"}
            else:
                details = {"reason": "unknown"}
        raise V1ApiError(row.code, row.status, message, details) from exc
    raise AssertionError("the last row matches every failure")  # pragma: no cover


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [text for item in value for text in _string_values(item)]
    return []


def _request_secrets(
    request: V1McpServerCreateRequest | V1McpServerUpdateRequest,
) -> list[str]:
    return [*_string_values(request.headers), *_string_values(request.auth)]


def _reject_invalid_fields(
    *, name: Optional[str], url: Optional[str], auth: Optional[dict[str, Any]]
) -> None:
    """Refuse a name, URL or auth the web code would store wrongly or not reject.

    ``None`` means the field is not being set.
    """
    if name is not None:
        # The web name rules reject a blank name too; this gives a clearer answer.
        if name.strip() == "":
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT, 400, "Connector name cannot be blank."
            )
        # The web name check lets a trailing line break through and then stores
        # the trimmed name under a row it cannot find again.
        if has_unfoldable_edge_whitespace(name):
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT, 400, unfoldable_edge_whitespace_detail()
            )
    if url is not None:
        try:
            mcp_endpoint_origin(url)
        except (UnsafePublicHttpTarget, ValueError, httpx.InvalidURL) as exc:
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                400,
                "url must be an absolute http(s) URL without user info.",
            ) from exc
    # A catalog marker in ``auth`` would make the connector pass for a
    # platform app.
    if auth and any(key in auth for key in _CATALOG_AUTH_KEYS):
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            "auth cannot set app_id, provider or builtin_provenance; these mark "
            "catalog connectors.",
        )
    # The web code encrypts a non-empty value in these fields and fails unless
    # it is a string; a falsy non-string skips encryption and is stored as sent.
    for key in SENSITIVE_AUTH_FIELDS:
        value = (auth or {}).get(key)
        if value is not None and not isinstance(value, str):
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT, 400, f"auth.{key} must be a string or null."
            )


def _reject_invalid_create(request: V1McpServerCreateRequest) -> None:
    """Refuse a create body the web code would store wrongly or not reject."""
    if request.transport not in _REMOTE_TRANSPORTS:
        raise V1ApiError(V1ErrorCode.TRANSPORT_NOT_SUPPORTED, 400)
    _reject_invalid_fields(name=request.name, url=request.url, auth=request.auth)
    if MASKED_SECRET_VALUE in [
        *request.headers.values(),
        *(request.auth or {}).values(),
    ]:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            f"A new connector has no stored value for '{MASKED_SECRET_VALUE}'; "
            "send the real value.",
        )


# Sent as null these are cleared; every other field must be omitted to keep
# its stored value.
_CLEARABLE_UPDATE_FIELDS = frozenset(
    {"runtime_input_schema", "runtime_bindings", "signed_uploads"}
)


def _reject_invalid_update(request: V1McpServerUpdateRequest) -> None:
    """Refuse an update body the web code would store wrongly or not reject."""
    sent = request.model_fields_set
    for field in V1McpServerUpdateRequest.model_fields:
        if (
            field in sent
            and field not in _CLEARABLE_UPDATE_FIELDS
            and getattr(request, field) is None
        ):
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT,
                400,
                f"{field} cannot be null; omit it to keep the stored value.",
            )
    if "transport" in sent and request.transport not in _REMOTE_TRANSPORTS:
        raise V1ApiError(V1ErrorCode.TRANSPORT_NOT_SUPPORTED, 400)
    _reject_invalid_fields(name=request.name, url=request.url, auth=request.auth)
    # Only the secret fields keep a stored value; the web code would store the
    # mask itself in any other.
    if any(
        value == MASKED_SECRET_VALUE and key not in SENSITIVE_AUTH_FIELDS
        for key, value in (request.auth or {}).items()
    ):
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            f"Only secret auth fields can keep their stored value with "
            f"'{MASKED_SECRET_VALUE}'.",
        )


@router.get("", response_model=list[V1McpServer])
async def list_mcp_servers(
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> list[V1McpServer]:
    """List the connectors the key holder can see.

    The holder's own connectors come first, including ones the holder switched
    off, then connectors visible only because the holder's team shares them.
    Custom APIs are not included.
    """
    user, _key = authed
    return await McpServerManagementRuntime().list_servers(user_id=int(user.id))


@router.get("/{server_id}", response_model=V1McpServer)
async def get_mcp_server(
    server_id: int = Path(gt=0, le=2**31 - 1),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1McpServer:
    """Show one connector the key holder can see.

    A connector the holder cannot see and one that does not exist both answer
    404 ``connector_not_found``.
    """
    user, _key = authed
    server = await McpServerManagementRuntime().get_server(
        user_id=int(user.id), server_id=server_id
    )
    if server is None:
        raise V1ApiError(V1ErrorCode.CONNECTOR_NOT_FOUND, 404)
    return server


@router.post("", response_model=V1McpServer)
async def create_mcp_server(
    request: V1McpServerCreateRequest,
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1McpServer:
    """Create a remote MCP connector owned by the key holder.

    Only the ``streamable_http`` and ``sse`` transports can be created. The
    connector is private to the holder until the holder's team shares it. A
    name that is taken, reserved or equal to a taken name after folding
    answers 409 ``connector_name_conflict``; when the exact name is the
    holder's own connector, ``details.connector_id`` names it.
    """
    user, key = authed
    _reject_invalid_create(request)
    runtime = McpServerManagementRuntime()
    try:
        return await runtime.create_server(
            user_id=int(user.id), key_prefix=key.key_prefix, request=request
        )
    except HTTPException as exc:
        owned = (
            await runtime.owned_connector_id(user_id=int(user.id), name=request.name)
            if _is_exact_name_conflict(exc, request.name)
            else None
        )
        _raise_v1_from_mcp_write_error(
            exc,
            action="create",
            requested_name=request.name,
            secrets=_request_secrets(request),
            owned_connector_id=owned,
        )


@router.patch("/{server_id}", response_model=V1McpServer)
async def update_mcp_server(
    request: V1McpServerUpdateRequest,
    server_id: int = Path(gt=0, le=2**31 - 1),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1McpServer:
    """Change the fields of one of the key holder's remote connectors.

    Only the fields present in the body change; an empty body changes nothing.
    A connector the holder cannot see answers 404; one seen only through the
    team answers 403 ``connector_forbidden``; a catalog app connection answers
    409 ``connector_not_editable``; one with another transport answers 400
    ``transport_not_supported``. A holder who may not edit the shared
    configuration answers 403 ``connector_forbidden`` for any non-empty body.
    A change that moves ``url``, ``auth.issuer`` or ``auth.resource_metadata_url``
    answers 409 ``connector_not_editable`` while other accounts hold
    authorizations it would keep: make that change in the web app. Otherwise
    it removes the authorizations team members made for themselves, and they
    must authorize again.
    """
    user, key = authed
    try:
        return await McpServerManagementRuntime().update_server(
            user_id=int(user.id),
            key_prefix=key.key_prefix,
            server_id=server_id,
            request=request,
            validate=_reject_invalid_update,
        )
    except HTTPException as exc:
        owned = (
            await McpServerManagementRuntime().owned_connector_id(
                user_id=int(user.id), name=request.name
            )
            if request.name is not None and _is_exact_name_conflict(exc, request.name)
            else None
        )
        _raise_v1_from_mcp_write_error(
            exc,
            action="update",
            requested_name=request.name or "",
            secrets=_request_secrets(request),
            owned_connector_id=owned,
        )


@router.delete("/{server_id}", response_model=V1McpServerDeleteResponse)
async def delete_mcp_server(
    server_id: int = Path(gt=0, le=2**31 - 1),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1McpServerDeleteResponse:
    """Delete one of the key holder's connectors.

    This removes the holder's link to the connector; the definition goes with
    the last link, unless the team keeps it. A connector the holder cannot
    see answers 404; one seen only through the team answers 403
    ``connector_forbidden``; a catalog app connection, or a connector with
    OAuth connections that another account (such as an automation account)
    manages through your link, answers 409 ``connector_not_editable``.
    """
    user, key = authed
    try:
        await McpServerManagementRuntime().delete_server(
            user_id=int(user.id), key_prefix=key.key_prefix, server_id=server_id
        )
    except HTTPException as exc:
        _raise_v1_from_mcp_write_error(
            exc, action="delete", requested_name="", secrets=()
        )
    return V1McpServerDeleteResponse(id=server_id)


@router.put("/{server_id}/enabled", response_model=V1McpServer)
async def set_mcp_server_enabled(
    request: V1McpServerEnabledRequest,
    server_id: int = Path(gt=0, le=2**31 - 1),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1McpServer:
    """Switch one of the key holder's connectors on or off.

    Setting the value a connector already has is not an error. Only the
    holder's own switch changes; catalog and non-remote connectors can be
    switched too. A connector seen only through the team answers 403
    ``connector_forbidden``.
    """
    user, key = authed
    try:
        return await McpServerManagementRuntime().set_enabled(
            user_id=int(user.id),
            key_prefix=key.key_prefix,
            server_id=server_id,
            enabled=request.enabled,
        )
    except HTTPException as exc:
        _raise_v1_from_mcp_write_error(
            exc,
            action="enable" if request.enabled else "disable",
            requested_name="",
            secrets=(),
        )


# ===== listing a connector's tools =====

_V1_TOOL_LISTING_MAX_INFLIGHT = 4
_TOOL_LISTING_BUSY_MESSAGE = (
    "Too many connector tool listings are running; retry shortly."
)

# Keyed by loop, weakly, like the web connection-test gate: a semaphore binds
# to the loop it first waits on, and a finished loop must not pin it. This gate
# is independent of the web one, so the two never queue behind each other.
_v1_tool_listing_gates: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()

# Listings that hold a slot. The done-callback gives the slot back; this set
# keeps the task referenced until then.
_v1_tool_listing_loads: "set[asyncio.Task[Any]]" = set()


def _v1_tool_listing_gate() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    gate = _v1_tool_listing_gates.get(loop)
    if gate is None:
        gate = asyncio.Semaphore(_V1_TOOL_LISTING_MAX_INFLIGHT)
        _v1_tool_listing_gates[loop] = gate
    return gate


async def _list_saved_connector_tools(
    server_id: int, user_id: int
) -> ConnectorToolsResult:
    # The session belongs to the listing task, not to the request: a cancelled
    # request must not close a session the listing still uses.
    with get_session_local()() as db:
        user = db.get(User, user_id)
        if user is None:
            raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
        try:
            return await mcp_connector_session.list_connector_tools(
                db,
                user=user,
                server_id=server_id,
                name_prefix=f"server_{server_id}_",
            )
        except Exception as exc:
            # The exception text can carry server URLs or credentials.
            logger.error("personal key mcp tool listing failed: %s", type(exc).__name__)
            raise V1ApiError(V1ErrorCode.INTERNAL_ERROR, 500) from None


def _consume_abandoned_listing(task: "asyncio.Task[Any]") -> None:
    # The request stopped waiting for this listing, so nothing else will
    # retrieve its outcome; do it here so a late failure is not reported as an
    # exception that was never retrieved.
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("abandoned MCP tool listing finished with %s", type(exc).__name__)


async def _run_tool_listing(server_id: int, user_id: int) -> ConnectorToolsResult:
    """List a saved connector's tools under this module's own concurrency cap.

    * At most ``_V1_TOOL_LISTING_MAX_INFLIGHT`` listings run per event loop,
      shared by every key holder in the process.
    * Waiting for a slot is bounded by the MCP initialization timeout; a
      listing that cannot get one answers 429 ``rate_limited`` with a
      ``Retry-After`` of that timeout in whole seconds.
    * With a timeout above 0 a slot is given back when the listing returns,
      never earlier: a cancelled request leaves the listing running. Starting
      the connector is bounded by the initialization timeout; reading its
      settings before that, which can refresh an OAuth token, is not.
    * A timeout of 0 means no deadline: wait for a slot as long as it takes
      and await the listing directly. A client disconnect does not cancel the
      request, so a listing that never returns keeps its slot until the
      process stops.
    """
    gate = _v1_tool_listing_gate()
    timeout = get_mcp_tool_init_timeout_seconds()
    if timeout <= 0:
        async with gate:
            return await _list_saved_connector_tools(server_id, user_id)
    try:
        await asyncio.wait_for(gate.acquire(), timeout)
    except TimeoutError:
        raise V1ApiError(
            V1ErrorCode.RATE_LIMITED,
            429,
            _TOOL_LISTING_BUSY_MESSAGE,
            headers={"Retry-After": str(max(1, math.ceil(timeout)))},
        )
    load = asyncio.ensure_future(_list_saved_connector_tools(server_id, user_id))
    _v1_tool_listing_loads.add(load)

    def _give_back(task: "asyncio.Task[Any]") -> None:
        _v1_tool_listing_loads.discard(task)
        gate.release()

    load.add_done_callback(_give_back)
    try:
        await asyncio.wait({load})
    except asyncio.CancelledError:
        load.add_done_callback(_consume_abandoned_listing)
        raise
    return load.result()


def _reason_of(code: Any) -> str:
    return code if isinstance(code, str) and _REASON_PATTERN.fullmatch(code) else ""


def _tools_response(server_id: int, listing: ConnectorToolsResult) -> V1McpServerTools:
    """Map the connector session's outcome to the /v1 answer.

    Only a short reason and the load phases are passed on; the diagnostic can
    carry server names, issuers and exception text.
    """
    outcome, reason, loaded = listing.outcome, listing.reason, listing.load_result
    if outcome is ConnectorToolsOutcome.NOT_FOUND:
        raise V1ApiError(V1ErrorCode.CONNECTOR_NOT_FOUND, 404)
    if loaded is None:
        if (
            outcome is ConnectorToolsOutcome.NOT_CONNECTED
            or listing.diagnostic is not None
        ):
            # NOT_CONNECTED is the session's own verdict that the credential is
            # missing or unusable, except for a rejected platform client, which
            # is reported as an unavailable connector.
            if outcome is ConnectorToolsOutcome.NOT_CONNECTED and (
                not is_client_rejected_reason(reason)
            ):
                raise V1ApiError(
                    V1ErrorCode.MCP_OAUTH_AUTHORIZATION_FAILED,
                    409,
                    _AUTHORIZATION_MESSAGE,
                    {"reason": _reason_of(reason) or "authorization_required"},
                )
            # A refresh the vendor rejects for the platform's own OAuth client
            # may name the provider after a colon; the provider is not passed
            # on.
            code = (
                CLIENT_REJECTED_REASON
                if is_client_rejected_reason(reason)
                else _reason_of(reason) or "unknown"
            )
            raise V1ApiError(
                V1ErrorCode.CONNECTOR_UNAVAILABLE, 409, details={"reason": code}
            )
        if outcome is ConnectorToolsOutcome.UNSUPPORTED:
            raise V1ApiError(
                V1ErrorCode.CONNECTOR_UNAVAILABLE,
                409,
                details={"reason": "unsupported"},
            )
        raise V1ApiError(
            V1ErrorCode.CONNECTOR_TOOLS_UNAVAILABLE,
            502,
            details={"reason": _reason_of(reason) or "no_tools", "failures": []},
        )
    failures = [
        V1McpToolLoadFailure(phase=failure.phase.value, attempts=failure.attempts)
        for failure in loaded.failures
    ]
    if not loaded.tools:
        raise V1ApiError(
            V1ErrorCode.CONNECTOR_TOOLS_UNAVAILABLE,
            502,
            details={
                "reason": "no_tools",
                "failures": [failure.model_dump() for failure in failures],
            },
        )
    tools: Sequence[Any] = loaded.tools
    return V1McpServerTools(
        id=server_id,
        tools=[
            V1McpTool(
                name=str(getattr(tool, "name", tool)),
                mcp_name=str(tool.mcp_tool.name),
                description=str(
                    getattr(tool, "description", "No description available")
                ),
            )
            for tool in tools
        ],
        failures=failures,
    )


@router.get("/{server_id}/tools", response_model=V1McpServerTools)
async def list_mcp_server_tools(
    server_id: int = Path(gt=0, le=2**31 - 1),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1McpServerTools:
    """Start one of the holder's remote connectors and list its tools.

    The connector is started exactly as a task would start it. The request
    first waits for a free slot (at most one MCP initialization timeout) and
    then lists the tools. Starting the connector takes at most one more
    initialization timeout, plus the time to read its settings. A connector
    the holder has switched off, one shared by the team, or one whose
    transport is not ``streamable_http`` or ``sse`` (catalog connectors
    included) is refused before anything starts. Authorization is done in
    the web app: an unauthorized OAuth connector answers 409
    ``mcp_oauth_authorization_failed``.
    """
    user, _key = authed
    user_id = int(user.id)
    await McpServerManagementRuntime().require_listable(
        user_id=user_id, server_id=server_id
    )
    return _tools_response(server_id, await _run_tool_listing(server_id, user_id))
