"""
MCP Server Management API Endpoints

Provides REST API endpoints for managing MCP server configurations
in the web application.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import shlex
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Callable, Dict, List, Optional, Union, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import get_app_base_url, get_session_secret
from ...core.tools.core.mcp.data_config import MCPServerConfig
from ...core.tools.core.mcp.manager.db import DatabaseMCPServerManager
from ...core.tools.core.mcp.model import MASKED_SECRET_VALUE, SENSITIVE_AUTH_FIELDS
from ...core.utils.encryption import decrypt_value, encrypt_value
from ..auth_dependencies import get_current_user
from ..mcp_apps import get_all_mcp_apps, get_app_by_name
from ..models.database import get_db
from ..models.mcp import MCPServer, UserMCPServer
from ..models.mcp_oauth import (
    MCPOAuthClient,
    MCPOAuthFlowState,
    MCPOAuthGrant,
    mcp_oauth_client_lookup_hash,
    mcp_oauth_grant_lookup_hash,
)
from ..models.user import User
from ..services.mcp_oauth import (
    MCP_OAUTH_HTTP_TIMEOUT_SECONDS,
    MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH,
    MCP_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH,
    MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH,
    MCP_OAUTH_TOKEN_TYPE_MAX_LENGTH,
    MCPOAuthDiscoveryError,
    _same_url,
    create_mcp_oauth_http_client,
    discover_mcp_oauth_metadata,
    normalize_mcp_oauth_scope,
    oauth_error_log_payload,
    oauth_error_message,
    oauth_exception_message,
    oauth_post,
    oauth_token_expires_at,
    select_mcp_oauth_grants,
    validate_mcp_oauth_persisted_value,
)
from ..services.mcp_runtime import HTTP_MCP_TRANSPORTS

logger = logging.getLogger(__name__)

MCP_OAUTH_STATE_COOKIE = "xagent_mcp_oauth_state"
MCP_OAUTH_STATE_TTL = timedelta(minutes=10)
MCP_OAUTH_STATE_COOKIE_MAX_AGE_SECONDS = int(MCP_OAUTH_STATE_TTL.total_seconds())
MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHODS = frozenset(
    {"none", "client_secret_post", "client_secret_basic"}
)


# Pydantic models for API
class MCPServerCreate(BaseModel):
    """Request model for creating MCP server."""

    name: str = Field(..., min_length=1, max_length=100, description="Server name")
    transport: str = Field(
        ..., description="Transport type (stdio, sse, websocket, streamable_http)"
    )
    description: Optional[str] = Field(None, description="Server description")
    config: dict = Field(..., description="Transport-specific configuration")
    is_active: bool = Field(True, description="Whether the server is active")
    user_env: Optional[dict] = Field(
        None, description="Per-user env overrides (merged over global env at runtime)"
    )


class MCPServerUpdate(BaseModel):
    """Request model for updating MCP server."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="Server name"
    )
    transport: Optional[str] = Field(None, description="Transport type")
    description: Optional[str] = Field(None, description="Server description")
    config: Optional[dict] = Field(None, description="Transport-specific configuration")
    is_active: Optional[bool] = Field(None, description="Whether the server is active")
    user_env: Optional[dict] = Field(
        None, description="Per-user env overrides (merged over global env at runtime)"
    )


class MCPServerResponse(BaseModel):
    """Response model for MCP server."""

    id: int
    user_id: int
    name: str
    transport: str
    description: Optional[str]
    config: dict
    is_active: bool
    is_default: bool
    user_env: Optional[dict] = None
    can_edit_global: bool = False
    transport_display: str
    created_at: Optional[str]
    updated_at: Optional[str]
    connected_account: Optional[str] = None
    app_id: Optional[str] = None
    provider: Optional[str] = None

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class MCPConnectionTest(BaseModel):
    """Request model for testing MCP connection."""

    name: str = Field(..., description="Connection name")
    transport: str = Field(..., description="Transport type")
    config: dict[str, Any] = Field(..., description="Connection configuration")


class MCPConnectionTestResponse(BaseModel):
    """Response model for MCP connection test."""

    success: bool
    message: str
    details: Optional[dict] = None


class MCPOAuthDiscoverRequest(BaseModel):
    """Request model for MCP OAuth metadata discovery."""

    model_config = ConfigDict(extra="forbid")


class MCPOAuthDiscoverResponse(BaseModel):
    """Selected MCP OAuth metadata for a configured MCP server."""

    resource: str
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    scopes: list[str]
    authorization_servers: list[str]
    client_id_metadata_document_supported: bool


class MCPOAuthConnectRequest(MCPOAuthDiscoverRequest):
    """Request model for starting MCP OAuth Authorization Code + PKCE."""

    redirect_after: Optional[str] = None


class MCPOAuthGrantResponse(BaseModel):
    """Public-safe MCP OAuth grant status."""

    id: int
    resource_owner_key: str
    issuer: str
    resource: str
    scope: str
    token_type: str
    status: str
    expires_at: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    revoked_at: Optional[str]


class MCPOAuthStatusResponse(BaseModel):
    """MCP OAuth connection status for the current user."""

    server_id: int
    auth_type: Optional[str]
    resource: Optional[str]
    issuer: Optional[str]
    scope: Optional[str]
    grants: list[MCPOAuthGrantResponse]


# Create router
mcp_router = APIRouter(prefix="/api/mcp", tags=["MCP Management"])


class ConfigFieldParser:
    """Modular parser for configuration fields with type-specific parsing strategies."""

    @staticmethod
    def parse_string_list(value: str) -> List[str]:
        """Parse a string into a list of strings."""
        try:
            # Try JSON first
            result = json.loads(value)
            if isinstance(result, list):
                return result
            raise ValueError("Not a list")
        except (json.JSONDecodeError, ValueError):
            try:
                # Try to parse as shell command line
                return shlex.split(value)
            except ValueError:
                # Fall back to splitting by whitespace and newlines
                return [
                    arg.strip()
                    for arg in value.replace("\n", " ").split()
                    if arg.strip()
                ]

    @staticmethod
    def parse_key_value_dict(value: str) -> Dict[str, str]:
        """Parse a string into a dictionary of key-value pairs."""
        try:
            # Try JSON first
            result = json.loads(value)
            if isinstance(result, dict):
                return result
            raise ValueError("Not a dictionary")
        except (json.JSONDecodeError, ValueError):
            # Parse as key=value pairs (one per line or space-separated)
            result = {}
            lines = value.replace("\n", " ").split()
            for line in lines:
                if "=" in line:
                    key, val = line.split("=", 1)
                    result[key.strip()] = val.strip()
            return result

    @staticmethod
    def parse_port_mappings(value: str) -> Dict[str, Union[int, str]]:
        """Parse port mappings as container_port:host_port."""
        try:
            # Try JSON first
            result = json.loads(value)
            if isinstance(result, dict):
                return result
            raise ValueError("Not a dictionary")
        except (json.JSONDecodeError, ValueError):
            # Parse as port:port pairs
            result = {}
            lines = value.replace("\n", " ").split()
            for line in lines:
                if ":" in line:
                    container_port, host_port = line.split(":", 1)
                    result[container_port.strip()] = host_port.strip()
            return result

    @staticmethod
    def parse_boolean(value: str) -> bool:
        """Parse a string into a boolean."""
        return value.lower() in ("true", "1", "yes", "on")

    @staticmethod
    def parse_json_or_fallback(
        value: str, fallback_parser: Callable[[Any], Any] | None = None
    ) -> Any:
        """Try to parse as JSON, fall back to another parser if provided."""
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            if fallback_parser:
                return fallback_parser(value)
            return value


def _format_optional_datetime(value: object) -> Optional[str]:
    """Serialize datetimes while tolerating ORM attributes without DB timestamps."""
    return value.isoformat() if isinstance(value, datetime) else None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _default_mcp_oauth_redirect_uri() -> str:
    base_url = get_app_base_url() or "http://localhost:8000"
    return f"{base_url.rstrip('/')}/api/mcp/oauth/callback"


def _safe_mcp_oauth_redirect_after(value: str | None) -> str:
    if not value:
        return "/tools"
    parsed = urlsplit(value)
    if (
        parsed.scheme
        or parsed.netloc
        or not value.startswith("/")
        or value.startswith("//")
        or value.startswith("/\\")
        or len(value) > MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH
    ):
        return "/tools"
    return value


def _mcp_oauth_cookie_secure() -> bool:
    base_url = get_app_base_url()
    return bool(base_url and base_url.lower().startswith("https://"))


def _mcp_oauth_state_cookie_signature(state_value: str) -> str:
    return hmac.new(
        get_session_secret().encode("utf-8"),
        state_value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _mcp_oauth_state_cookie_value(state_value: str) -> str:
    return f"{state_value}.{_mcp_oauth_state_cookie_signature(state_value)}"


def _set_mcp_oauth_state_cookie(response: Response, state_value: str) -> None:
    response.set_cookie(
        MCP_OAUTH_STATE_COOKIE,
        _mcp_oauth_state_cookie_value(state_value),
        max_age=MCP_OAUTH_STATE_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        secure=_mcp_oauth_cookie_secure(),
        samesite="lax",
        path="/api/mcp",
    )


def _clear_mcp_oauth_state_cookie(response: Response) -> None:
    response.delete_cookie(MCP_OAUTH_STATE_COOKIE, path="/api/mcp")


def _mcp_oauth_callback_error_redirect(
    flow_state: MCPOAuthFlowState,
    *,
    error_code: str,
    message: str,
) -> RedirectResponse:
    raw_redirect_after = (
        str(flow_state.redirect_after) if flow_state.redirect_after else None
    )
    redirect_after = _safe_mcp_oauth_redirect_after(raw_redirect_after)
    parts = urlsplit(redirect_after)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.extend(
        (
            ("mcp_oauth_error", error_code),
            (
                "mcp_oauth_error_message",
                oauth_error_message(message, "MCP OAuth authorization failed"),
            ),
        )
    )
    response = RedirectResponse(
        urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
    )
    _clear_mcp_oauth_state_cookie(response)
    return response


def _validate_mcp_oauth_state_cookie(request: Request, state_value: str) -> None:
    cookie_value = request.cookies.get(MCP_OAUTH_STATE_COOKIE)
    if not cookie_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_state",
                "message": "OAuth callback state was not initiated by this browser session",
            },
        )
    try:
        cookie_state, cookie_signature = cookie_value.rsplit(".", 1)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_state", "message": "Invalid OAuth state cookie"},
        ) from exc
    expected_signature = _mcp_oauth_state_cookie_signature(cookie_state)
    if not (
        hmac.compare_digest(cookie_state, state_value)
        and hmac.compare_digest(cookie_signature, expected_signature)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_state",
                "message": "OAuth callback state did not match this browser session",
            },
        )


def _default_resource_owner_key(user_id: int) -> str:
    return f"xagent:user:{user_id}"


def _oauth_authorization_url(endpoint: str, params: dict[str, str]) -> str:
    parts = urlsplit(endpoint)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(params)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def _scope_string(scopes: list[str] | tuple[str, ...] | str | None) -> str:
    try:
        return normalize_mcp_oauth_scope(scopes)
    except MCPOAuthDiscoveryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


def _bounded_mcp_oauth_value(
    value: str,
    *,
    field_name: str,
    max_length: int = MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH,
) -> str:
    try:
        return validate_mcp_oauth_persisted_value(
            value, field_name=field_name, max_length=max_length
        )
    except MCPOAuthDiscoveryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


def _pkce_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _get_user_mcp_server_or_404(
    db: Session, *, user_id: int, server_id: int, require_active: bool = False
) -> tuple[UserMCPServer, MCPServer]:
    query = (
        db.query(UserMCPServer, MCPServer)
        .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
    )
    if require_active:
        query = query.filter(UserMCPServer.is_active)
    result = query.first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
        )
    return cast(tuple[UserMCPServer, MCPServer], result)


def _get_mcp_oauth_config(server: MCPServer) -> dict[str, Any]:
    config = server.to_config_dict()
    auth_config = config.get("auth")
    if not isinstance(auth_config, dict) or auth_config.get("type") != "mcp_oauth":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MCP server is not configured for MCP OAuth",
        )
    if server.transport not in HTTP_MCP_TRANSPORTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MCP OAuth is only supported for HTTP MCP transports",
        )
    if not server.url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MCP OAuth server requires a URL",
        )
    return auth_config


def _configured_mcp_oauth_value(
    request_value: str | None, auth_config: dict[str, Any], key: str
) -> str | None:
    value = request_value if request_value is not None else auth_config.get(key)
    return str(value).strip() if value else None


async def _discover_mcp_oauth_for_server(
    server: MCPServer,
    auth_config: dict[str, Any],
) -> Any:
    if not server.url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MCP OAuth server requires a URL",
        )
    try:
        return await discover_mcp_oauth_metadata(
            str(server.url),
            headers=None,
            configured_resource_metadata_url=_configured_mcp_oauth_value(
                None,
                auth_config,
                "resource_metadata_url",
            ),
            configured_issuer=_configured_mcp_oauth_value(None, auth_config, "issuer"),
            configured_resource=_configured_mcp_oauth_value(
                None, auth_config, "resource"
            ),
        )
    except MCPOAuthDiscoveryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": exc.code, "message": exc.message},
        ) from exc


def _mcp_oauth_discovery_response(discovery: Any) -> MCPOAuthDiscoverResponse:
    return MCPOAuthDiscoverResponse(
        resource=discovery.resource,
        issuer=discovery.authorization_server.issuer,
        authorization_endpoint=discovery.authorization_server.authorization_endpoint,
        token_endpoint=discovery.authorization_server.token_endpoint,
        scopes=list(discovery.scopes),
        authorization_servers=list(discovery.protected_resource.authorization_servers),
        client_id_metadata_document_supported=(
            discovery.authorization_server.client_id_metadata_document_supported
        ),
    )


def _upsert_mcp_oauth_client(
    db: Session,
    *,
    server_id: int,
    discovery: Any,
    client_id: str,
    client_secret: str | None,
    token_endpoint_auth_method: str,
    redirect_uri: str,
) -> MCPOAuthClient:
    issuer = _bounded_mcp_oauth_value(
        str(discovery.authorization_server.issuer), field_name="issuer"
    )
    authorization_endpoint = _bounded_mcp_oauth_value(
        str(discovery.authorization_server.authorization_endpoint),
        field_name="authorization_endpoint",
    )
    token_endpoint = _bounded_mcp_oauth_value(
        str(discovery.authorization_server.token_endpoint), field_name="token_endpoint"
    )
    client_id = _bounded_mcp_oauth_value(client_id, field_name="client_id")
    token_endpoint_auth_method = _bounded_mcp_oauth_value(
        token_endpoint_auth_method,
        field_name="token_endpoint_auth_method",
        max_length=MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH,
    )
    redirect_uri = _bounded_mcp_oauth_value(redirect_uri, field_name="redirect_uri")
    lookup_hash = mcp_oauth_client_lookup_hash(server_id, issuer, client_id)

    def load_existing_client() -> MCPOAuthClient | None:
        return (
            db.query(MCPOAuthClient)
            .filter(
                MCPOAuthClient.lookup_hash == lookup_hash,
            )
            .first()
        )

    def apply_client_values(existing: MCPOAuthClient | None) -> MCPOAuthClient:
        encrypted_client_secret: str | None
        if client_secret == MASKED_SECRET_VALUE:
            if existing is None or not existing.client_secret:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "invalid_resource",
                        "message": "Masked MCP OAuth client_secret has no stored value",
                    },
                )
            encrypted_client_secret = str(existing.client_secret)
        else:
            encrypted_client_secret = (
                encrypt_value(client_secret) if client_secret else None
            )
        client = existing or MCPOAuthClient(
            mcp_server_id=server_id,
            lookup_hash=lookup_hash,
            issuer=issuer,
            client_id=client_id,
        )
        setattr(client, "authorization_endpoint", authorization_endpoint)
        setattr(client, "token_endpoint", token_endpoint)
        setattr(client, "client_secret", encrypted_client_secret)
        setattr(client, "token_endpoint_auth_method", token_endpoint_auth_method)
        setattr(client, "redirect_uri", redirect_uri)
        setattr(client, "metadata_json", discovery.authorization_server.raw)
        if existing is None:
            db.add(client)
        return client

    try:
        client = apply_client_values(load_existing_client())
        db.flush()
        return client
    except IntegrityError as exc:
        db.rollback()
        existing_after_conflict = load_existing_client()
        if existing_after_conflict is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "oauth_client_conflict",
                    "message": "MCP OAuth client configuration changed concurrently",
                },
            ) from exc
        client = apply_client_values(existing_after_conflict)
        db.flush()
        return client


def _mcp_oauth_grant_response(grant: MCPOAuthGrant) -> MCPOAuthGrantResponse:
    return MCPOAuthGrantResponse(
        id=cast(int, grant.id),
        resource_owner_key=str(grant.resource_owner_key),
        issuer=str(grant.issuer),
        resource=str(grant.resource),
        scope=str(grant.scope),
        token_type=str(grant.token_type),
        status=str(grant.status),
        expires_at=_format_optional_datetime(grant.expires_at),
        created_at=_format_optional_datetime(grant.created_at),
        updated_at=_format_optional_datetime(grant.updated_at),
        revoked_at=_format_optional_datetime(grant.revoked_at),
    )


def _validate_mcp_oauth_callback_issuer(
    *,
    request: Request,
    client: MCPOAuthClient,
    flow_state: MCPOAuthFlowState,
) -> None:
    metadata: dict[str, Any] = (
        client.metadata_json if isinstance(client.metadata_json, dict) else {}
    )
    issuer_required = (
        metadata.get("authorization_response_iss_parameter_supported") is True
    )
    response_issuer = request.query_params.get("iss")
    expected_issuer = str(flow_state.issuer)

    if response_issuer is None:
        if issuer_required:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "issuer_mismatch",
                    "message": "Authorization response issuer is required",
                },
            )
        return

    if not _same_url(response_issuer, expected_issuer):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "issuer_mismatch",
                "message": "Authorization response issuer did not match flow state",
            },
        )


def _mcp_oauth_flow_state_error(
    db: Session, flow_state: MCPOAuthFlowState
) -> tuple[str, str] | None:
    if flow_state.consumed_at is not None:
        return "state_already_consumed", "OAuth state consumed"
    if _as_aware_utc(flow_state.expires_at) <= _utc_now():
        return "expired_state", "OAuth state expired"
    if (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == flow_state.user_id,
            UserMCPServer.mcpserver_id == flow_state.mcp_server_id,
            UserMCPServer.is_active,
        )
        .first()
        is None
    ):
        return (
            "invalid_state",
            "OAuth state is no longer associated with MCP server access",
        )
    return None


def _claim_mcp_oauth_flow_state(
    db: Session, flow_state: MCPOAuthFlowState
) -> tuple[str, str] | None:
    claimed_at = _utc_now()
    updated = (
        db.query(MCPOAuthFlowState)
        .filter(
            MCPOAuthFlowState.id == flow_state.id,
            MCPOAuthFlowState.consumed_at.is_(None),
            MCPOAuthFlowState.expires_at > claimed_at,
        )
        .update(
            {MCPOAuthFlowState.consumed_at: claimed_at},
            synchronize_session=False,
        )
    )
    if updated != 1:
        db.rollback()
        return "state_already_consumed", "OAuth state consumed"
    db.commit()
    db.refresh(flow_state)
    return None


async def _exchange_mcp_oauth_code(
    *,
    client: MCPOAuthClient,
    code: str,
    code_verifier: str,
    resource: str,
) -> dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client.client_id,
        "redirect_uri": client.redirect_uri,
        "code_verifier": code_verifier,
        "resource": resource,
    }
    auth: httpx.Auth | None = None
    client_secret = (
        decrypt_value(str(client.client_secret)) if client.client_secret else ""
    )
    auth_method = str(client.token_endpoint_auth_method or "none")
    if auth_method == "client_secret_post" and client_secret:
        data["client_secret"] = client_secret
    elif auth_method == "client_secret_basic" and client_secret:
        auth = httpx.BasicAuth(str(client.client_id), client_secret)
    elif auth_method not in {"none", "client_secret_post", "client_secret_basic"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "unsupported_auth_server",
                "message": f"Unsupported token endpoint auth method: {auth_method}",
            },
        )

    try:
        post_kwargs: dict[str, Any] = {
            "data": data,
            "headers": {"Content-Type": "application/x-www-form-urlencoded"},
        }
        if auth is not None:
            post_kwargs["auth"] = auth
        async with create_mcp_oauth_http_client(
            timeout=MCP_OAUTH_HTTP_TIMEOUT_SECONDS,
        ) as http_client:
            response = await oauth_post(
                str(client.token_endpoint),
                client=http_client,
                **post_kwargs,
            )
        payload = response.json()
    except (MCPOAuthDiscoveryError, httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "token_exchange_failed",
                "message": oauth_exception_message(
                    exc, "MCP OAuth token exchange failed"
                ),
            },
        ) from exc

    if (
        response.status_code >= 400
        or not isinstance(payload, dict)
        or payload.get("error")
    ):
        logger.warning(
            "MCP OAuth token exchange failed with token endpoint payload: %s",
            oauth_error_log_payload(payload),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "token_exchange_failed",
                "message": oauth_error_message(
                    payload, "MCP OAuth token exchange failed"
                ),
            },
        )
    if not payload.get("access_token"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "token_exchange_failed",
                "message": "Token response did not include access_token",
            },
        )
    return payload


async def _revoke_mcp_oauth_grant_externally(
    *,
    client: MCPOAuthClient,
    grant: MCPOAuthGrant,
) -> None:
    metadata: dict[str, Any] = (
        client.metadata_json if isinstance(client.metadata_json, dict) else {}
    )
    revocation_endpoint = metadata.get("revocation_endpoint")
    if not isinstance(revocation_endpoint, str) or not revocation_endpoint:
        return

    try:
        client_secret = (
            decrypt_value(str(client.client_secret)) if client.client_secret else ""
        )
    except Exception as exc:
        logger.warning(
            "Skipping MCP OAuth token revocation for grant %s because client secret "
            "could not be decrypted: %s",
            grant.id,
            exc,
        )
        return
    auth_method = str(client.token_endpoint_auth_method or "none")
    auth: httpx.Auth | None = None
    base_data: dict[str, str] = {"client_id": str(client.client_id)}
    if auth_method == "client_secret_post" and client_secret:
        base_data["client_secret"] = client_secret
    elif auth_method == "client_secret_basic" and client_secret:
        auth = httpx.BasicAuth(str(client.client_id), client_secret)
    elif auth_method not in {"none", "client_secret_post", "client_secret_basic"}:
        logger.warning(
            "Skipping MCP OAuth token revocation for unsupported auth method %s",
            auth_method,
        )
        return

    encrypted_tokens = (
        (grant.access_token, "access_token"),
        (grant.refresh_token, "refresh_token"),
    )
    async with create_mcp_oauth_http_client(
        timeout=MCP_OAUTH_HTTP_TIMEOUT_SECONDS,
    ) as http_client:
        for encrypted_token, token_type_hint in encrypted_tokens:
            if not encrypted_token:
                continue
            try:
                decrypted_token = decrypt_value(str(encrypted_token))
            except Exception as exc:
                logger.warning(
                    "Skipping MCP OAuth %s revocation for grant %s because token "
                    "could not be decrypted: %s",
                    token_type_hint,
                    grant.id,
                    exc,
                )
                continue
            data = {
                **base_data,
                "token": decrypted_token,
                "token_type_hint": token_type_hint,
            }
            request_kwargs: dict[str, Any] = {
                "data": data,
                "headers": {"Content-Type": "application/x-www-form-urlencoded"},
            }
            if auth is not None:
                request_kwargs["auth"] = auth
            try:
                response = await oauth_post(
                    revocation_endpoint,
                    client=http_client,
                    **request_kwargs,
                )
                if response.status_code >= 400:
                    logger.warning(
                        "MCP OAuth token revocation returned HTTP %s for grant %s",
                        response.status_code,
                        grant.id,
                    )
            except (MCPOAuthDiscoveryError, httpx.HTTPError) as exc:
                logger.warning(
                    "MCP OAuth token revocation failed for grant %s: %s",
                    grant.id,
                    exc,
                )


def _upsert_mcp_oauth_grant(
    db: Session,
    *,
    flow_state: MCPOAuthFlowState,
    token_data: dict[str, Any],
) -> MCPOAuthGrant:
    scope = _scope_string(str(token_data.get("scope") or flow_state.scope))
    lookup_hash = mcp_oauth_grant_lookup_hash(
        flow_state.mcp_server_id,
        flow_state.user_id,
        flow_state.resource_owner_key,
        flow_state.mcp_oauth_client_id,
        flow_state.issuer,
        flow_state.resource,
        scope,
    )
    existing = (
        db.query(MCPOAuthGrant)
        .filter(
            MCPOAuthGrant.lookup_hash == lookup_hash,
        )
        .first()
    )
    grant = existing or MCPOAuthGrant(
        mcp_server_id=flow_state.mcp_server_id,
        user_id=flow_state.user_id,
        mcp_oauth_client_id=flow_state.mcp_oauth_client_id,
        lookup_hash=lookup_hash,
        resource_owner_key=flow_state.resource_owner_key,
        issuer=flow_state.issuer,
        resource=flow_state.resource,
        scope=scope,
    )
    setattr(grant, "access_token", encrypt_value(str(token_data["access_token"])))
    refresh_token = token_data.get("refresh_token")
    if refresh_token:
        setattr(grant, "refresh_token", encrypt_value(str(refresh_token)))
    setattr(
        grant,
        "token_type",
        _bounded_mcp_oauth_value(
            str(token_data.get("token_type") or "Bearer"),
            field_name="token_type",
            max_length=MCP_OAUTH_TOKEN_TYPE_MAX_LENGTH,
        ),
    )
    setattr(grant, "status", "active")
    setattr(grant, "revoked_at", None)
    setattr(
        grant,
        "metadata_json",
        {
            key: value
            for key, value in token_data.items()
            if key not in {"access_token", "refresh_token"}
        },
    )
    setattr(grant, "expires_at", oauth_token_expires_at(token_data))
    if existing is None:
        db.add(grant)
    return grant


class MCPConfigFieldRegistry:
    """Registry of field parsers for different configuration fields."""

    # Field type mappings
    STRING_LIST_FIELDS = {"args", "volumes"}
    KEY_VALUE_DICT_FIELDS = {"env", "headers", "docker_environment"}
    PORT_MAPPING_FIELDS = {"bind_ports"}
    BOOLEAN_FIELDS = {"auto_start", "concurrency_safe"}
    JSON_FIELDS = {"headers"}  # Fields that should prefer JSON parsing

    @classmethod
    def get_parser_for_field(cls, field_name: str) -> Optional[Callable]:
        """Get the appropriate parser function for a field."""
        if field_name in cls.STRING_LIST_FIELDS:
            return ConfigFieldParser.parse_string_list
        elif field_name in cls.KEY_VALUE_DICT_FIELDS:
            return ConfigFieldParser.parse_key_value_dict
        elif field_name in cls.PORT_MAPPING_FIELDS:
            return ConfigFieldParser.parse_port_mappings
        elif field_name in cls.BOOLEAN_FIELDS:
            return ConfigFieldParser.parse_boolean
        return None


class TransportFieldValidator:
    """Validate fields based on transport type."""

    TRANSPORT_REQUIRED_FIELDS = {
        "stdio": {"command"},
        "sse": {"url"},
        "websocket": {"url"},
        "streamable_http": {"url"},
    }

    TRANSPORT_OPTIONAL_FIELDS = {
        "stdio": {"args", "env", "cwd"},
        "sse": {"headers"},
        "websocket": {"headers"},
        "streamable_http": {"headers"},
    }

    @classmethod
    def validate_transport_fields(
        cls, transport: str, config_dict: Dict[str, Any]
    ) -> None:
        """Validate that required fields are present for the transport type."""
        required_fields = cls.TRANSPORT_REQUIRED_FIELDS.get(transport, set())

        for field in required_fields:
            if field not in config_dict or config_dict[field] is None:
                raise ValueError(f"Transport '{transport}' requires field '{field}'")


def _build_server_config(
    server_data: MCPServerCreate, existing_server: Optional[MCPServer] = None
) -> MCPServerConfig:
    """Build MCPServerConfig from API request data using modular parsing."""
    # Start with base config
    config_dict = {
        "name": server_data.name,
        "transport": server_data.transport,
        "description": server_data.description,
        "managed": "external",  # Default for user-created servers
    }

    # Parse and add config fields
    if server_data.config:
        for field_name, value in server_data.config.items():
            if field_name not in [
                "name",
                "transport",
                "description",
            ]:  # Skip already handled fields
                try:
                    parsed_value = _parse_config_field(
                        field_name, value, server_data.transport
                    )

                    if parsed_value is not None:
                        config_dict[field_name] = parsed_value
                except ValueError as e:
                    raise ValueError(
                        f"Configuration error in field '{field_name}': {str(e)}"
                    )

    # For updates, preserve existing values if not provided
    if existing_server:
        existing_config = existing_server.to_config_dict()
        for key, value in existing_config.items():
            if key not in config_dict and value is not None:
                config_dict[key] = value

    TransportFieldValidator.validate_transport_fields(
        server_data.transport, config_dict
    )

    return MCPServerConfig(**config_dict)


def _update_server_from_config(server: MCPServer, config: MCPServerConfig) -> None:
    """Update database server object from MCPServerConfig."""
    # Map config fields to database fields
    field_mapping = {
        "name": "name",
        "description": "description",
        "transport": "transport",
        "managed": "managed",
        "command": "command",
        "args": "args",
        "url": "url",
        "env": "env",
        "cwd": "cwd",
        "headers": "headers",
        "timeout": "timeout",
        "auth": "auth",
        "concurrency_safe": "concurrency_safe",
        "concurrent_tools": "concurrent_tools",
        "docker_url": "docker_url",
        "docker_image": "docker_image",
        "docker_environment": "docker_environment",
        "docker_working_dir": "docker_working_dir",
        "volumes": "volumes",
        "bind_ports": "bind_ports",
        "restart_policy": "restart_policy",
        "auto_start": "auto_start",
    }

    for config_field, db_field in field_mapping.items():
        if hasattr(config, config_field) and hasattr(server, db_field):
            value = getattr(config, config_field)
            if config_field == "env" and value and isinstance(value, dict):
                from xagent.core.utils.encryption import encrypt_env_dict

                # Masked values ("********") mean "keep the stored secret".
                value = encrypt_env_dict(
                    _merge_masked_env(value, getattr(server, "env", None) or {})
                )
            elif config_field == "auth" and value and isinstance(value, dict):
                from xagent.core.utils.encryption import encrypt_value

                encrypted_auth = value.copy()
                for key in SENSITIVE_AUTH_FIELDS:
                    if key in encrypted_auth and encrypted_auth[key]:
                        # If masked, retain the existing encrypted value from the database
                        if encrypted_auth[key] == MASKED_SECRET_VALUE:
                            existing_auth: Any = server.auth or {}
                            encrypted_auth[key] = existing_auth.get(key)
                        else:
                            # encrypt_value is idempotent (skips already-encrypted)
                            encrypted_auth[key] = encrypt_value(encrypted_auth[key])
                value = encrypted_auth
            setattr(server, db_field, value)


def _parse_config_field(
    field_name: str, value: Any, transport: str | None = None
) -> Any:
    """
    Parse configuration field based on its expected type.

    Args:
        field_name: Name of the configuration field
        value: Raw value to parse
        transport: Transport type (for transport-specific parsing if needed)

    Returns:
        Parsed value in the appropriate type
    """
    # Handle None or empty values
    if value is None or value == "":
        return None

    # If not a string, return as-is (already parsed)
    if not isinstance(value, str):
        return value

    # Clean up string value
    value = value.strip()
    if not value:
        return None

    # Get parser for this field
    parser = MCPConfigFieldRegistry.get_parser_for_field(field_name)

    if parser:
        try:
            result = parser(value)
            # Return None for empty results
            if isinstance(result, (dict, list)) and not result:
                return None
            return result
        except Exception as e:
            raise ValueError(f"Failed to parse field '{field_name}': {str(e)}")

    # Default: return string value as-is
    return value


def _mask_env(env: Any) -> dict:
    """Mask env values for API responses (keys stay visible for editing)."""
    return {k: (MASKED_SECRET_VALUE if v else v) for k, v in env.items()}


def _merge_masked_env(new_env: dict, old_env: dict) -> dict:
    """Apply an incoming env dict, restoring the stored value for masked entries.

    A masked entry with no stored value (e.g. a renamed key) is dropped rather
    than persisted as None, which would crash stdio subprocess launch.
    """
    merged = {}
    for k, v in new_env.items():
        if v == MASKED_SECRET_VALUE:
            if k in old_env and old_env[k] is not None:
                merged[k] = old_env[k]
        else:
            merged[k] = v
    return merged


def _check_mcp_permission(
    user_mcp: UserMCPServer, is_admin: bool, require: str = "edit"
) -> bool:
    """Whether the user may mutate shared MCP config.

    ``edit`` gates changes to the shared global config; ``delete`` gates
    removing the shared server. Admins bypass both.
    """
    if is_admin:
        return True
    is_owner = bool(getattr(user_mcp, "is_owner", False))
    if require == "delete":
        # The owner can always delete; can_delete additionally grants it to a
        # non-owner. Checking is_owner too covers rows created before can_delete
        # was set (e.g. OAuth provisioning, migration-skipped is_owner rows).
        return is_owner or bool(getattr(user_mcp, "can_delete", False))
    return is_owner


# Owner-only global fields that are safe to compare (non-secret; secret values
# like env/headers and auth's SENSITIVE_AUTH_FIELDS round-trip as masks and
# can't be diffed reliably, so they keep the silent-preserve behavior).
_GLOBAL_CONFIG_KEYS = ("command", "args", "url")


def _auth_metadata_tampered(incoming_auth: Any, current_auth: Any) -> bool:
    """True if a payload changes non-secret auth metadata (client_id, issuer …)."""
    if not isinstance(incoming_auth, dict):
        return False
    current = current_auth if isinstance(current_auth, dict) else {}
    return any(
        key not in SENSITIVE_AUTH_FIELDS and value != current.get(key)
        for key, value in incoming_auth.items()
    )


def _global_config_tampered(server_data: MCPServerUpdate, server: MCPServer) -> bool:
    """True if a payload changes owner-only global fields (non-secret ones)."""
    if server_data.name is not None and server_data.name != server.name:
        return True
    if server_data.transport is not None and server_data.transport != server.transport:
        return True
    if (
        server_data.description is not None
        and server_data.description != server.description
    ):
        return True
    incoming = server_data.config or {}
    current = server.to_config_dict()
    if any(
        key in incoming and incoming[key] != current.get(key)
        for key in _GLOBAL_CONFIG_KEYS
    ):
        return True
    return _auth_metadata_tampered(incoming.get("auth"), current.get("auth"))


def _db_server_to_response(
    server: MCPServer,
    user_mcp: UserMCPServer,
    manager: DatabaseMCPServerManager,
    connected_account: Optional[str] = None,
    app_id: Optional[str] = None,
    provider: Optional[str] = None,
    is_admin: bool = False,
) -> MCPServerResponse:
    """Convert database MCPServer to response model."""
    # Get status from manager if available
    config = server.to_config_dict()

    # Mask sensitive auth fields for the frontend
    auth_config = config.get("auth")
    if auth_config and isinstance(auth_config, dict):
        masked_auth = auth_config.copy()
        for key in SENSITIVE_AUTH_FIELDS:
            if key in masked_auth and masked_auth[key]:
                masked_auth[key] = MASKED_SECRET_VALUE
        config["auth"] = masked_auth

    # Env values are secrets: mask them (keys stay visible so the UI can edit them).
    if isinstance(config.get("env"), dict):
        config["env"] = _mask_env(config["env"])

    return MCPServerResponse(
        id=server.id,
        user_id=user_mcp.user_id,
        name=server.name,
        transport=server.transport,
        description=server.description,
        config=config,
        is_active=user_mcp.is_active,
        is_default=user_mcp.is_default,
        user_env=_mask_env(getattr(user_mcp, "env", None)) if user_mcp.env else None,
        can_edit_global=_check_mcp_permission(user_mcp, is_admin, require="edit"),
        transport_display=server.transport_display,
        created_at=_format_optional_datetime(server.created_at),
        updated_at=_format_optional_datetime(server.updated_at),
        connected_account=connected_account,
        app_id=app_id,
        provider=provider,
    )


def _enrich_oauth_server_info(
    db: Session, server: MCPServer, oauth_emails: dict
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (app_id, provider, connected_account) for an OAuth-based MCPServer.
    This encapsulates the logic of looking up app information in O(1) time.
    """
    if server.transport != "oauth":
        return None, None, None

    app_info = get_app_by_name(db, str(server.name))
    if not app_info:
        return None, None, None

    provider = app_info.get("provider")
    app_id = app_info.get("id")
    connected_account = oauth_emails.get(app_id) or oauth_emails.get(provider)

    return app_id, provider, connected_account


def _normalize_app_key(value: object) -> Optional[str]:
    if value is None:
        return None
    normalized = "-".join(str(value).strip().lower().split())
    return normalized or None


def _app_lookup_keys(*values: object) -> list[str]:
    keys = []
    for value in values:
        key = _normalize_app_key(value)
        if key and key not in keys:
            keys.append(key)
    return keys


def _oauth_account_can_connect(oauth_account: object) -> bool:
    access_token = getattr(oauth_account, "access_token", None)
    if not access_token:
        return False

    expires_at = getattr(oauth_account, "expires_at", None)
    if not isinstance(expires_at, datetime):
        return True

    if getattr(oauth_account, "refresh_token", None):
        return True

    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    return expires_at > now


def _oauth_keys_for_app(app: dict) -> list[str]:
    return _app_lookup_keys(app.get("id"), app.get("provider"))


def _is_oauth_server_for_app(server: MCPServer, app: dict) -> bool:
    if server.transport != "oauth":
        return False

    app_id = _normalize_app_key(app.get("id"))
    app_name = _normalize_app_key(app.get("name"))
    provider = _normalize_app_key(app.get("provider"))
    server_name = _normalize_app_key(server.name)

    auth = getattr(server, "auth", None)
    if isinstance(auth, dict):
        auth_app_id = _normalize_app_key(auth.get("app_id"))
        auth_provider = _normalize_app_key(auth.get("provider"))

        if auth_app_id and auth_app_id != app_id:
            return False
        if auth_provider and provider and auth_provider != provider:
            return False
        if auth_app_id or auth_provider:
            return True

    # Legacy OAuth server rows created before app metadata was stored in auth.
    return bool(server_name and server_name in {app_id, app_name})


def _connected_oauth_server_for_app(
    app: dict,
    oauth_server_lookup: dict[str, list[MCPServer]],
    oauth_account_lookup: dict[str, object],
) -> tuple[Optional[int], Optional[str]]:
    oauth_account = next(
        (
            oauth_account_lookup[key]
            for key in _oauth_keys_for_app(app)
            if key in oauth_account_lookup
        ),
        None,
    )
    if not oauth_account:
        return None, None

    server = _lookup_oauth_server_for_app(app, oauth_server_lookup)
    if not server:
        return None, None

    email = getattr(oauth_account, "email", None)
    return cast(int, server.id), str(email) if email else None


def _build_oauth_account_lookup(oauth_accounts: list[object]) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for account in oauth_accounts:
        key = _normalize_app_key(getattr(account, "provider", None))
        if key and key not in lookup and _oauth_account_can_connect(account):
            lookup[key] = account
    return lookup


def _oauth_server_lookup_keys(server: MCPServer) -> list[str]:
    auth = getattr(server, "auth", None)
    if isinstance(auth, dict):
        auth_app_id = _normalize_app_key(auth.get("app_id"))
        if auth_app_id:
            return [auth_app_id]

        auth_provider = _normalize_app_key(auth.get("provider"))
        if auth_provider:
            return [auth_provider]

    return _app_lookup_keys(server.name)


def _build_active_oauth_server_lookup(
    user_mcps: list[tuple[MCPServer, UserMCPServer]],
) -> dict[str, list[MCPServer]]:
    lookup: dict[str, list[MCPServer]] = {}
    for server, user_mcp in user_mcps:
        if not user_mcp.is_active or _normalize_app_key(server.transport) != "oauth":
            continue
        for key in _oauth_server_lookup_keys(server):
            lookup.setdefault(key, []).append(server)
    return lookup


def _lookup_oauth_server_for_app(
    app: dict, oauth_server_lookup: dict[str, list[MCPServer]]
) -> Optional[MCPServer]:
    seen_servers: set[int] = set()
    for key in _app_lookup_keys(app.get("id"), app.get("provider"), app.get("name")):
        for server in oauth_server_lookup.get(key, []):
            marker = id(server)
            if marker in seen_servers:
                continue
            seen_servers.add(marker)
            if _is_oauth_server_for_app(server, app):
                return server
    return None


def _build_active_non_oauth_server_lookup(
    user_mcps: list[tuple[MCPServer, UserMCPServer]],
) -> dict[tuple[str, str], MCPServer]:
    lookup: dict[tuple[str, str], MCPServer] = {}
    for server, user_mcp in user_mcps:
        transport = _normalize_app_key(server.transport)
        server_name = _normalize_app_key(server.name)
        if (
            not user_mcp.is_active
            or not transport
            or transport == "oauth"
            or not server_name
        ):
            continue
        lookup.setdefault((transport, server_name), server)
    return lookup


def _connected_non_oauth_server_for_app(
    app: dict, non_oauth_server_lookup: dict[tuple[str, str], MCPServer]
) -> Optional[int]:
    app_transport = _normalize_app_key(app.get("transport"))
    if not app_transport or app_transport == "oauth":
        return None

    server = next(
        (
            non_oauth_server_lookup[(app_transport, app_key)]
            for app_key in _app_lookup_keys(app.get("id"), app.get("name"))
            if (app_transport, app_key) in non_oauth_server_lookup
        ),
        None,
    )
    if not server:
        return None

    return cast(int, server.id)


@mcp_router.get("/apps", response_model=List[dict])
def list_mcp_apps(
    search: Optional[str] = None,
    category: Optional[str] = "All",
    location: Optional[str] = "remote",
    status: Optional[str] = "all",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[dict]:
    """Get the list of available MCP applications in the library."""

    # Query connected servers for the current user
    user_mcps = [
        (server, user_mcp)
        for server, user_mcp in (
            db.query(MCPServer, UserMCPServer)
            .join(UserMCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == current_user.id)
            .all()
        )
    ]

    # Also fetch user oauth accounts to get the connected email
    from ..models.user_oauth import UserOAuth

    oauth_accounts = (
        db.query(UserOAuth).filter(UserOAuth.user_id == current_user.id).all()
    )

    results = []
    library_apps = (
        get_all_mcp_apps(db) if location in ["remote", "local", "all"] else []
    )
    oauth_account_lookup = _build_oauth_account_lookup(list(oauth_accounts))
    oauth_server_lookup = _build_active_oauth_server_lookup(user_mcps)
    non_oauth_server_lookup = _build_active_non_oauth_server_lookup(user_mcps)

    if location in ["remote", "all"]:
        for app in library_apps:
            if app.get("transport") == "oauth":
                server_id, connected_account = _connected_oauth_server_for_app(
                    app, oauth_server_lookup, oauth_account_lookup
                )
            else:
                server_id = _connected_non_oauth_server_for_app(
                    app, non_oauth_server_lookup
                )
                connected_account = None
            is_connected = server_id is not None
            is_visible_in_connector = app.get("is_visible_in_connector", True)

            # Strong hide mode: hidden public apps are removed from the
            # connector catalog for everyone, including already connected users.
            if not is_visible_in_connector:
                continue

            if search:
                search_lower = search.lower()
                if (
                    search_lower not in app["name"].lower()
                    and search_lower not in (app.get("description") or "").lower()
                ):
                    continue

            if category and category != "All":
                if app.get("category") != category:
                    continue

            app_copy = app.copy()
            app_copy["is_connected"] = is_connected

            if is_connected:
                app_copy["server_id"] = server_id

                if connected_account:
                    app_copy["connected_account"] = connected_account

            if status == "verified" and not app_copy["is_connected"]:
                continue

            results.append(app_copy)

    if location in ["local", "all"]:
        library_names = {app["name"].lower() for app in library_apps}
        for server, user_mcp in user_mcps:
            if server.name.lower() in library_names:
                continue

            if search:
                search_lower = search.lower()
                if search_lower not in server.name.lower() and (
                    server.description
                    and search_lower not in server.description.lower()
                ):
                    continue

            if category and category != "All":
                continue

            results.append(
                {
                    "id": server.name,
                    "name": server.name,
                    "description": server.description or "Custom MCP Server",
                    "icon": "",
                    "users": "1",
                    "transport": server.transport,
                    "is_connected": True,
                    "provider": "custom",
                    "category": "Local",
                    "is_local": True,
                    "server_id": server.id,
                }
            )

        # Append Custom APIs
        from ..models.custom_api import CustomApi, UserCustomApi

        user_custom_apis = (
            db.query(UserCustomApi, CustomApi)
            .join(CustomApi, UserCustomApi.custom_api_id == CustomApi.id)
            .filter(UserCustomApi.user_id == current_user.id)
            .all()
        )

        for user_api, api in user_custom_apis:
            if search:
                search_lower = search.lower()
                if search_lower not in api.name.lower() and (
                    api.description and search_lower not in api.description.lower()
                ):
                    continue

            if category and category != "All":
                continue

            results.append(
                {
                    "id": api.name,
                    "name": api.name,
                    "description": api.description or "Custom API",
                    "icon": "",
                    "users": "1",
                    "transport": "custom_api",
                    "is_connected": True,
                    "provider": "custom",
                    "category": "Local",
                    "is_local": True,
                    "server_id": api.id,
                    "is_custom": True,
                }
            )

    return results


@mcp_router.get("/servers", response_model=List[MCPServerResponse])
def get_mcp_servers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[MCPServerResponse]:
    """List MCP servers for the current user."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Get user's MCP servers
        user_mcps = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id)
            .order_by(MCPServer.created_at.desc())
            .all()
        )

        logger.info(f"user_mcps: {user_mcps}")

        # Fetch oauth emails
        from ..models.user_oauth import UserOAuth

        oauth_accounts = db.query(UserOAuth).filter(UserOAuth.user_id == user_id).all()
        oauth_emails = {
            str(oauth.provider): str(oauth.email)
            for oauth in oauth_accounts
            if oauth.email
        }

        is_admin = getattr(current_user, "is_admin", False)
        responses = []
        for user_mcp, server in user_mcps:
            app_id, provider, connected_account = _enrich_oauth_server_info(
                db, server, oauth_emails
            )
            responses.append(
                _db_server_to_response(
                    server,
                    user_mcp,
                    manager,
                    connected_account,
                    app_id,
                    provider,
                    is_admin=is_admin,
                )
            )

        # Append Custom APIs
        from ..models.custom_api import CustomApi, UserCustomApi

        user_custom_apis = (
            db.query(UserCustomApi, CustomApi)
            .join(CustomApi, UserCustomApi.custom_api_id == CustomApi.id)
            .filter(UserCustomApi.user_id == user_id)
            .all()
        )

        for user_api, api in user_custom_apis:
            # Mask env values
            masked_env = {}
            if api.env and isinstance(api.env, dict):
                masked_env = {k: "********" for k in api.env.keys()}

            config_dict = {"env": masked_env}
            if api.url:
                config_dict["url"] = api.url
            if api.method:
                config_dict["method"] = api.method
            if api.headers:
                config_dict["headers"] = api.headers
            if api.body:
                config_dict["body"] = api.body

            responses.append(
                MCPServerResponse(
                    id=api.id,
                    user_id=user_api.user_id,
                    name=api.name,
                    transport="custom_api",
                    description=api.description,
                    config=config_dict,
                    is_active=user_api.is_active,
                    is_default=user_api.is_default,
                    transport_display="Custom API",
                    created_at=_format_optional_datetime(api.created_at),
                    updated_at=_format_optional_datetime(api.updated_at),
                )
            )

        return responses

    except Exception as e:
        logger.error(f"Failed to list MCP servers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list MCP servers",
        )


@mcp_router.get("/servers/{server_id}", response_model=MCPServerResponse)
def get_mcp_server(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPServerResponse:
    """Get a specific MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        user_mcp, server = result

        # Fetch oauth emails for this user to enrich the server info
        from ..models.user_oauth import UserOAuth

        oauth_accounts = db.query(UserOAuth).filter(UserOAuth.user_id == user_id).all()
        oauth_emails = {
            oauth.provider: oauth.email for oauth in oauth_accounts if oauth.email
        }

        app_id, provider, connected_account = _enrich_oauth_server_info(
            db, server, oauth_emails
        )

        return _db_server_to_response(
            server,
            user_mcp,
            manager,
            connected_account,
            app_id,
            provider,
            is_admin=getattr(current_user, "is_admin", False),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP server",
        )


@mcp_router.post(
    "/servers", response_model=MCPServerResponse, status_code=status.HTTP_201_CREATED
)
def create_mcp_server(
    server_data: MCPServerCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPServerResponse:
    """Create a new MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check if server name already exists
        existing = (
            db.query(MCPServer).filter(MCPServer.name == server_data.name).first()
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"MCP server '{server_data.name}' already exists",
            )

        # Build and validate config
        try:
            config = _build_server_config(server_data)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration: {str(e)}",
            )

        # Add server using manager
        manager.add_server(config)

        # Get the created server
        server = db.query(MCPServer).filter(MCPServer.name == server_data.name).first()
        if not server:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create server",
            )

        # Create user-server association. The creator owns the global config.
        from xagent.core.utils.encryption import encrypt_env_dict

        created_user_env = None
        if server_data.user_env:
            # No stored values yet: drop masked entries, then encrypt at rest.
            created_user_env = (
                encrypt_env_dict(_merge_masked_env(server_data.user_env, {})) or None
            )
        user_mcp = UserMCPServer(
            user_id=user_id,
            mcpserver_id=server.id,
            is_active=server_data.is_active,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            env=created_user_env,
        )
        db.add(user_mcp)
        db.commit()
        db.refresh(user_mcp)

        logger.info(f"Created MCP server '{server_data.name}' for user {user_id}")
        return _db_server_to_response(
            server, user_mcp, manager, is_admin=getattr(current_user, "is_admin", False)
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create MCP server: {str(e)}",
        )


@mcp_router.put("/servers/{server_id}", response_model=MCPServerResponse)
def update_mcp_server(
    server_id: int,
    server_data: MCPServerUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPServerResponse:
    """Update an existing MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        user_mcp, server = result
        can_edit_global = _check_mcp_permission(
            user_mcp, getattr(current_user, "is_admin", False), require="edit"
        )

        # Non-owners may not touch the shared global config (env, command, etc.);
        # they only get to set their own per-user env override below. Reject a
        # tampered payload outright (defense-in-depth for direct/stale-UI calls)
        # rather than silently normalizing it back to a 200.
        incoming_config = dict(server_data.config or {})
        if not can_edit_global:
            if _global_config_tampered(server_data, server):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only the server owner can change the shared configuration",
                )
            incoming_config = {}

        # Check for name conflicts if updating name
        if can_edit_global and server_data.name and server_data.name != server.name:
            existing = (
                db.query(MCPServer)
                .filter(MCPServer.name == server_data.name, MCPServer.id != server_id)
                .first()
            )
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"MCP server '{server_data.name}' already exists",
                )

        # Build update config - only include provided fields. Non-owners keep the
        # existing global config untouched.
        update_data = MCPServerCreate(
            name=(server_data.name if can_edit_global else None) or server.name,
            transport=(server_data.transport if can_edit_global else None)
            or server.transport,
            description=server_data.description
            if can_edit_global and server_data.description is not None
            else server.description,
            config=incoming_config,
            is_active=server_data.is_active
            if server_data.is_active is not None
            else user_mcp.is_active,
        )

        # Build and validate config
        try:
            config = _build_server_config(update_data, server)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration: {str(e)}",
            )

        # Update server fields (global config; no-op values for non-owners)
        _update_server_from_config(server, config)

        # Store this user's per-user env override (masked values keep stored secrets)
        if server_data.user_env is not None:
            from xagent.core.utils.encryption import encrypt_env_dict

            merged_user_env = _merge_masked_env(
                server_data.user_env, getattr(user_mcp, "env", None) or {}
            )
            user_mcp.env = encrypt_env_dict(merged_user_env) or None

        # Update user association if needed
        if server_data.is_active is not None:
            user_mcp.is_active = server_data.is_active

        db.commit()
        db.refresh(server)
        db.refresh(user_mcp)

        logger.info(f"Updated MCP server '{server.name}' for user {user_id}")
        return _db_server_to_response(
            server, user_mcp, manager, is_admin=getattr(current_user, "is_admin", False)
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update MCP server: {str(e)}",
        )


@mcp_router.delete("/servers/{server_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_mcp_server(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete an MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        user_mcp, server = result

        # Deleting cascades to the shared config once no associations remain;
        # gate it on ownership, consistent with the update handler.
        if not _check_mcp_permission(
            user_mcp, getattr(current_user, "is_admin", False), require="delete"
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to delete this MCP server",
            )

        server_name = server.name

        # If it's an OAuth server, also delete the corresponding OAuth tokens
        if server.transport == "oauth":
            from ..mcp_apps import get_app_by_name
            from ..models.user_oauth import UserOAuth

            # Find the corresponding app_id and provider
            app_info = get_app_by_name(db, str(server.name))
            if app_info:
                provider = app_info.get("provider")
                app_id = app_info.get("id")

                # Delete tokens for this specific app
                providers_to_delete = [p for p in [provider, app_id] if p is not None]
                if providers_to_delete:
                    db.query(UserOAuth).filter(
                        UserOAuth.user_id == user_id,
                        UserOAuth.provider.in_(providers_to_delete),
                    ).delete(synchronize_session=False)

        # Remove user-server association
        db.delete(user_mcp)
        db.commit()

        # Check if any other users are using this server
        other_users = (
            db.query(UserMCPServer)
            .filter(UserMCPServer.mcpserver_id == server_id)
            .first()
        )

        # Only remove from manager and delete if no other users
        if not other_users:
            manager.remove_server(server_name)
            logger.info(f"Deleted MCP server '{server_name}'")
        else:
            logger.info(f"Removed user {user_id} access to MCP server '{server_name}'")

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete MCP server",
        )


@mcp_router.post("/servers/{server_id}/toggle", response_model=MCPServerResponse)
async def toggle_mcp_server(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPServerResponse:
    """Toggle MCP server active status."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        user_mcp, server = result

        # Toggle active status
        user_mcp.is_active = not user_mcp.is_active
        db.commit()
        db.refresh(user_mcp)

        status_text = "activated" if user_mcp.is_active else "deactivated"
        logger.info(
            f"{status_text.capitalize()} MCP server '{server.name}' for user {user_id}"
        )

        return _db_server_to_response(
            server, user_mcp, manager, is_admin=getattr(current_user, "is_admin", False)
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to toggle MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to toggle MCP server",
        )


@mcp_router.get("/servers/{server_id}/logs")
async def get_mcp_server_logs(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    lines: int = 100,
) -> Dict[str, Any]:
    """Get logs for an internal MCP server."""
    try:
        manager = DatabaseMCPServerManager(db)
        user_id = current_user.id

        if not (1 <= lines <= 1000):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="lines must be between 1 and 1000",
            )

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(UserMCPServer.user_id == user_id, MCPServer.id == server_id)
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="MCP server not found"
            )

        _, server = result

        if server.managed != "internal":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Logs only available for internal servers",
            )

        log_lines = manager.get_logs(server.name, lines)
        return {"server_name": server.name, "logs": log_lines or []}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP server logs",
        )


@mcp_router.post("/test-connection", response_model=MCPConnectionTestResponse)
async def test_mcp_connection(
    test_data: MCPConnectionTest, db: Session = Depends(get_db)
) -> MCPConnectionTestResponse:
    """Test MCP server connection without saving."""
    try:
        from ...core.tools.adapters.vibe.mcp_adapter import (
            load_mcp_tools_as_agent_tools,
        )

        connection: dict[str, Any] = {
            "name": test_data.name,
            "transport": test_data.transport,
        }

        connection.update(**test_data.config)

        try:
            connections_dict: Dict[str, Any] = {"test": connection}
            tools = await load_mcp_tools_as_agent_tools(
                connections_dict, name_prefix="test_"
            )

            if tools:
                return MCPConnectionTestResponse(
                    success=True,
                    message=f"Successfully connected to {test_data.name}. Loaded {len(tools)} tools.",
                    details={"tool_count": len(tools)},
                )
            else:
                return MCPConnectionTestResponse(
                    success=True,
                    message=f"Connected to {test_data.name}, but no tools were loaded.",
                    details={"tool_count": 0},
                )

        except Exception as conn_error:
            return MCPConnectionTestResponse(
                success=False,
                message=f"Failed to connect to {test_data.name}: {str(conn_error)}",
                details={"error": str(conn_error)},
            )

    except Exception as e:
        logger.error(f"Failed to test MCP connection: {e}")
        return MCPConnectionTestResponse(
            success=False,
            message=f"Connection test failed: {str(e)}",
            details={"error": str(e)},
        )


@mcp_router.get("/transports")
def get_supported_transports() -> dict:
    """Get list of supported transport types with descriptions."""
    return {
        "transports": [
            {
                "id": "stdio",
                "name": "STDIO",
                "description": "Standard input/output transport for local processes",
                "config_fields": [
                    {
                        "name": "command",
                        "type": "string",
                        "required": True,
                        "description": "Command to execute",
                    },
                    {
                        "name": "args",
                        "type": "array",
                        "required": False,
                        "description": "Command arguments",
                    },
                    {
                        "name": "env",
                        "type": "object",
                        "required": False,
                        "description": "Environment variables",
                    },
                    {
                        "name": "cwd",
                        "type": "string",
                        "required": False,
                        "description": "Working directory",
                    },
                ],
            },
            {
                "id": "sse",
                "name": "Server-Sent Events",
                "description": "HTTP-based transport using Server-Sent Events",
                "config_fields": [
                    {
                        "name": "url",
                        "type": "string",
                        "required": True,
                        "description": "Server URL",
                    },
                    {
                        "name": "headers",
                        "type": "object",
                        "required": False,
                        "description": "HTTP headers",
                    },
                ],
            },
            {
                "id": "websocket",
                "name": "WebSocket",
                "description": "WebSocket-based transport for real-time communication",
                "config_fields": [
                    {
                        "name": "url",
                        "type": "string",
                        "required": True,
                        "description": "WebSocket URL",
                    },
                    {
                        "name": "headers",
                        "type": "object",
                        "required": False,
                        "description": "WebSocket headers",
                    },
                ],
            },
            {
                "id": "streamable_http",
                "name": "Streamable HTTP",
                "description": "HTTP transport with streaming capabilities",
                "config_fields": [
                    {
                        "name": "url",
                        "type": "string",
                        "required": True,
                        "description": "Server URL",
                    },
                    {
                        "name": "headers",
                        "type": "object",
                        "required": False,
                        "description": "HTTP headers",
                    },
                ],
            },
        ]
    }


@mcp_router.get("/servers/{server_id}/tools")
async def get_mcp_server_tools(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Get tools available from a specific MCP server."""
    try:
        user_id = int(current_user.id)

        # Check user has access to this server
        result = (
            db.query(UserMCPServer, MCPServer)
            .join(MCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
            .filter(
                UserMCPServer.user_id == user_id,
                UserMCPServer.is_active,
                MCPServer.id == server_id,
            )
            .first()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MCP server not found",
            )

        _, server = result

        from ..services.mcp_runtime import build_mcp_runtime_connection

        runtime_build = await build_mcp_runtime_connection(
            db,
            server,
            user_id=user_id,
        )
        if runtime_build.connection is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=runtime_build.diagnostic
                or {
                    "code": "authorization_required",
                    "message": "MCP server authorization is required",
                },
            )
        connection = runtime_build.connection

        # Try to load tools
        from ...core.tools.adapters.vibe.mcp_adapter import (
            load_mcp_tools_as_agent_tools,
        )

        server_name = server.name
        tools: List[Any] = []
        if isinstance(server_name, str):
            connections_dict: Dict[str, Any] = {server_name: connection}
            tools = await load_mcp_tools_as_agent_tools(
                connections_dict, name_prefix=f"server_{server_id}_"
            )

        tools_list: List[Any] = tools if isinstance(tools, list) else []

        return {
            "server_name": server.name,
            "tool_count": len(tools_list),
            "tools": [
                {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(
                        tool, "description", "No description available"
                    ),
                }
                for tool in tools_list
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get MCP server tools",
        )


@mcp_router.get("/oauth/callback")
async def mcp_oauth_callback(
    request: Request,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """Complete MCP OAuth Authorization Code + PKCE and store an encrypted grant."""
    code = request.query_params.get("code")
    state_value = request.query_params.get("state")
    error = request.query_params.get("error")
    if not state_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_state", "message": "Missing OAuth state"},
        )
    flow_state = (
        db.query(MCPOAuthFlowState)
        .filter(MCPOAuthFlowState.state == state_value)
        .first()
    )
    if not flow_state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_state", "message": "Invalid OAuth state"},
        )
    try:
        _validate_mcp_oauth_state_cookie(request, state_value)
    except HTTPException as exc:
        cookie_detail: dict[str, Any] = (
            exc.detail if isinstance(exc.detail, dict) else {}
        )
        return _mcp_oauth_callback_error_redirect(
            flow_state,
            error_code=str(cookie_detail.get("code") or "invalid_state"),
            message=str(
                cookie_detail.get("message")
                or "OAuth callback state did not match this browser session"
            ),
        )
    state_error = _mcp_oauth_flow_state_error(db, flow_state)
    if state_error is not None:
        error_code, message = state_error
        return _mcp_oauth_callback_error_redirect(
            flow_state,
            error_code=error_code,
            message=message,
        )

    client = (
        db.query(MCPOAuthClient)
        .filter(MCPOAuthClient.id == flow_state.mcp_oauth_client_id)
        .first()
    )
    if not client:
        return _mcp_oauth_callback_error_redirect(
            flow_state,
            error_code="token_exchange_failed",
            message="OAuth client metadata not found",
        )

    try:
        _validate_mcp_oauth_callback_issuer(
            request=request,
            client=client,
            flow_state=flow_state,
        )
    except HTTPException as exc:
        issuer_detail: dict[str, Any] = (
            exc.detail if isinstance(exc.detail, dict) else {}
        )
        return _mcp_oauth_callback_error_redirect(
            flow_state,
            error_code=str(issuer_detail.get("code") or "issuer_mismatch"),
            message=str(
                issuer_detail.get("message")
                or "Authorization response issuer did not match flow state"
            ),
        )
    claim_error = _claim_mcp_oauth_flow_state(db, flow_state)
    if claim_error is not None:
        error_code, message = claim_error
        return _mcp_oauth_callback_error_redirect(
            flow_state,
            error_code=error_code,
            message=message,
        )
    if error:
        return _mcp_oauth_callback_error_redirect(
            flow_state,
            error_code="token_exchange_failed",
            message=oauth_error_message(
                {"error": error}, "MCP OAuth authorization failed"
            ),
        )
    if not code:
        return _mcp_oauth_callback_error_redirect(
            flow_state,
            error_code="invalid_state",
            message="Missing authorization code",
        )
    try:
        token_data = await _exchange_mcp_oauth_code(
            client=client,
            code=code,
            code_verifier=decrypt_value(str(flow_state.code_verifier)),
            resource=str(flow_state.resource),
        )
        _upsert_mcp_oauth_grant(db, flow_state=flow_state, token_data=token_data)
        db.commit()
    except HTTPException as exc:
        db.rollback()
        detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
        error_code = str(detail.get("code") or "token_exchange_failed")
        message = str(detail.get("message") or "MCP OAuth authorization failed")
        return _mcp_oauth_callback_error_redirect(
            flow_state,
            error_code=error_code,
            message=message,
        )
    except Exception:
        db.rollback()
        logger.exception("MCP OAuth callback failed after state claim")
        return _mcp_oauth_callback_error_redirect(
            flow_state,
            error_code="token_exchange_failed",
            message="MCP OAuth authorization failed",
        )

    response = RedirectResponse(
        _safe_mcp_oauth_redirect_after(str(flow_state.redirect_after))
    )
    _clear_mcp_oauth_state_cookie(response)
    return response


@mcp_router.post("/{server_id}/oauth/discover", response_model=MCPOAuthDiscoverResponse)
async def discover_mcp_oauth(
    server_id: int,
    request_data: MCPOAuthDiscoverRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPOAuthDiscoverResponse:
    """Discover MCP OAuth protected-resource and authorization-server metadata."""
    _, server = _get_user_mcp_server_or_404(
        db,
        user_id=cast(int, current_user.id),
        server_id=server_id,
        require_active=True,
    )
    auth_config = _get_mcp_oauth_config(server)
    discovery = await _discover_mcp_oauth_for_server(server, auth_config)
    return _mcp_oauth_discovery_response(discovery)


@mcp_router.post("/{server_id}/oauth/connect", response_model=None)
async def connect_mcp_oauth(
    server_id: int,
    request_data: MCPOAuthConnectRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    accept: Annotated[str | None, Header()] = None,
) -> RedirectResponse | JSONResponse:
    """Start MCP OAuth Authorization Code + PKCE for the current user."""
    user_id = cast(int, current_user.id)
    _, server = _get_user_mcp_server_or_404(
        db, user_id=user_id, server_id=server_id, require_active=True
    )
    auth_config = _get_mcp_oauth_config(server)
    discovery = await _discover_mcp_oauth_for_server(server, auth_config)

    client_id = _configured_mcp_oauth_value(None, auth_config, "client_id")
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "unsupported_auth_server",
                "message": "MCP OAuth client_id is required",
            },
        )
    client_id = _bounded_mcp_oauth_value(client_id, field_name="client_id")
    client_secret = _configured_mcp_oauth_value(None, auth_config, "client_secret")
    token_endpoint_auth_method = str(
        auth_config.get("token_endpoint_auth_method")
        or ("client_secret_post" if client_secret else "none")
    )
    token_endpoint_auth_method = _bounded_mcp_oauth_value(
        token_endpoint_auth_method,
        field_name="token_endpoint_auth_method",
        max_length=MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH,
    )
    if token_endpoint_auth_method not in MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHODS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "unsupported_auth_server",
                "message": (
                    "Unsupported token endpoint auth method: "
                    f"{token_endpoint_auth_method}"
                ),
            },
        )
    redirect_uri = (
        _configured_mcp_oauth_value(None, auth_config, "redirect_uri")
        or _default_mcp_oauth_redirect_uri()
    )
    redirect_uri = _bounded_mcp_oauth_value(redirect_uri, field_name="redirect_uri")
    selected_scope = _scope_string(auth_config.get("scope") or discovery.scopes)
    resource_owner_key = _bounded_mcp_oauth_value(
        _default_resource_owner_key(user_id),
        field_name="resource_owner_key",
        max_length=MCP_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH,
    )
    selected_issuer = _bounded_mcp_oauth_value(
        str(discovery.authorization_server.issuer), field_name="issuer"
    )
    selected_resource = _bounded_mcp_oauth_value(
        str(discovery.resource), field_name="resource"
    )

    oauth_client = _upsert_mcp_oauth_client(
        db,
        server_id=server_id,
        discovery=discovery,
        client_id=client_id,
        client_secret=client_secret,
        token_endpoint_auth_method=token_endpoint_auth_method,
        redirect_uri=redirect_uri,
    )

    state_value = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(64)
    flow_state = MCPOAuthFlowState(
        state=state_value,
        mcp_server_id=server_id,
        user_id=user_id,
        mcp_oauth_client_id=oauth_client.id,
        resource_owner_key=resource_owner_key,
        issuer=selected_issuer,
        resource=selected_resource,
        scope=selected_scope,
        code_verifier=encrypt_value(code_verifier),
        redirect_after=_safe_mcp_oauth_redirect_after(request_data.redirect_after),
        expires_at=_utc_now() + MCP_OAUTH_STATE_TTL,
    )
    db.add(flow_state)
    db.commit()

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state_value,
        "code_challenge": _pkce_code_challenge(code_verifier),
        "code_challenge_method": "S256",
        "resource": selected_resource,
    }
    if selected_scope:
        params["scope"] = selected_scope
    authorization_url = _oauth_authorization_url(
        discovery.authorization_server.authorization_endpoint, params
    )
    if accept and "application/json" in accept.lower():
        json_response = JSONResponse({"authorization_url": authorization_url})
        _set_mcp_oauth_state_cookie(json_response, state_value)
        return json_response
    redirect_response = RedirectResponse(
        authorization_url, status_code=status.HTTP_303_SEE_OTHER
    )
    _set_mcp_oauth_state_cookie(redirect_response, state_value)
    return redirect_response


@mcp_router.get("/{server_id}/oauth/status", response_model=MCPOAuthStatusResponse)
async def get_mcp_oauth_status(
    server_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MCPOAuthStatusResponse:
    """Return MCP OAuth grants owned by the current user for one MCP server."""
    user_id = cast(int, current_user.id)
    _, server = _get_user_mcp_server_or_404(
        db, user_id=user_id, server_id=server_id, require_active=True
    )
    config = server.to_config_dict()
    auth_config = config.get("auth") if isinstance(config.get("auth"), dict) else {}
    grants = select_mcp_oauth_grants(
        db,
        server_id=server_id,
        user_id=user_id,
        auth_config=auth_config if isinstance(auth_config, dict) else {},
    )
    return MCPOAuthStatusResponse(
        server_id=server_id,
        auth_type=auth_config.get("type") if isinstance(auth_config, dict) else None,
        resource=auth_config.get("resource") if isinstance(auth_config, dict) else None,
        issuer=auth_config.get("issuer") if isinstance(auth_config, dict) else None,
        scope=auth_config.get("scope") if isinstance(auth_config, dict) else None,
        grants=[_mcp_oauth_grant_response(grant) for grant in grants],
    )


@mcp_router.delete(
    "/{server_id}/oauth/grants/{grant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_mcp_oauth_grant(
    server_id: int,
    grant_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Revoke an MCP OAuth grant owned by the current user."""
    user_id = cast(int, current_user.id)
    _get_user_mcp_server_or_404(
        db, user_id=user_id, server_id=server_id, require_active=True
    )
    grant = (
        db.query(MCPOAuthGrant)
        .filter(
            MCPOAuthGrant.id == grant_id,
            MCPOAuthGrant.mcp_server_id == server_id,
            MCPOAuthGrant.user_id == user_id,
        )
        .first()
    )
    if not grant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="MCP OAuth grant not found"
        )
    if isinstance(grant.oauth_client, MCPOAuthClient):
        await _revoke_mcp_oauth_grant_externally(
            client=grant.oauth_client,
            grant=grant,
        )
    setattr(grant, "status", "revoked")
    setattr(grant, "revoked_at", _utc_now())
    db.commit()
