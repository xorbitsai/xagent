"""SDK endpoints for the MCP connectors the key holder can see and create."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, NoReturn, Optional, Sequence

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path

from ....core.tools.core.mcp.http import _origin as mcp_endpoint_origin
from ....core.tools.core.mcp.model import MASKED_SECRET_VALUE, SENSITIVE_AUTH_FIELDS
from ....core.utils.public_http_transport import UnsafePublicHttpTarget
from ...schemas.v1 import V1McpServer, V1McpServerCreateRequest
from ...services.connector_name_policy import (
    catalog_app_name_detail,
    folded_name_conflict_detail,
    has_unfoldable_edge_whitespace,
    unfoldable_edge_whitespace_detail,
)
from ...services.mcp_server_management import (
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
        details = (
            {"connector_id": owned_connector_id}
            if row.id == "exact_name_conflict" and owned_connector_id is not None
            else None
        )
        raise V1ApiError(
            row.code, row.status, row.message(exc, requested_name, secrets), details
        ) from exc
    raise AssertionError("the last row matches every failure")  # pragma: no cover


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [text for item in value for text in _string_values(item)]
    return []


def _request_secrets(request: V1McpServerCreateRequest) -> list[str]:
    return [*_string_values(request.headers), *_string_values(request.auth)]


def _reject_invalid_create(request: V1McpServerCreateRequest) -> None:
    """Refuse a create body the web code would store wrongly or not reject."""
    if request.transport not in _REMOTE_TRANSPORTS:
        raise V1ApiError(V1ErrorCode.TRANSPORT_NOT_SUPPORTED, 400)
    # The web name rules reject a blank name too; this gives a clearer answer.
    if request.name.strip() == "":
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT, 400, "Connector name cannot be blank."
        )
    # The web name check lets a trailing line break through and then stores the
    # trimmed name under a row it cannot find again.
    if has_unfoldable_edge_whitespace(request.name):
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT, 400, unfoldable_edge_whitespace_detail()
        )
    try:
        mcp_endpoint_origin(request.url)
    except (UnsafePublicHttpTarget, ValueError, httpx.InvalidURL) as exc:
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            "url must be an absolute http(s) URL without user info.",
        ) from exc
    # A catalog marker in ``auth`` would make the connector pass for a
    # platform app.
    if request.auth and any(key in request.auth for key in _CATALOG_AUTH_KEYS):
        raise V1ApiError(
            V1ErrorCode.INVALID_INPUT,
            400,
            "auth cannot set app_id, provider or builtin_provenance; these mark "
            "catalog connectors.",
        )
    # The web code encrypts a non-empty value in these fields and fails unless
    # it is a string; a falsy non-string skips encryption and is stored as sent.
    for key in SENSITIVE_AUTH_FIELDS:
        value = (request.auth or {}).get(key)
        if value is not None and not isinstance(value, str):
            raise V1ApiError(
                V1ErrorCode.INVALID_INPUT, 400, f"auth.{key} must be a string or null."
            )
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
