from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Iterable, Sequence
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit
from urllib.request import parse_http_list, parse_keqv_list

import httpx
from sqlalchemy.orm import sessionmaker

from ...config import get_mcp_oauth_allow_private_hosts, get_mcp_oauth_proxy_url
from ...core.utils.security import (
    PrivateNetworkHostError,
    reject_private_network_host,
)

logger = logging.getLogger(__name__)

MCP_OAUTH_HTTP_TIMEOUT_SECONDS = 10.0
DEFAULT_MCP_OAUTH_DISCOVERY_TIMEOUT = MCP_OAUTH_HTTP_TIMEOUT_SECONDS
MCP_OAUTH_MAX_REDIRECTS = 5
MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH = 1000
MCP_OAUTH_SCOPE_MAX_LENGTH = 1000
MCP_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH = 512
MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH = 100
MCP_OAUTH_TOKEN_TYPE_MAX_LENGTH = 50
OAUTH_ERROR_MESSAGE_MAX_LENGTH = 500
OAUTH_LOG_PAYLOAD_MAX_LENGTH = 2000
OAUTH_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
OAUTH_CROSS_ORIGIN_STRIPPED_HEADERS = frozenset(
    {"authorization", "cookie", "proxy-authorization"}
)
OAUTH_SENSITIVE_PAYLOAD_KEYS = frozenset(
    {"access_token", "refresh_token", "id_token", "client_secret"}
)


class MCPOAuthDiscoveryError(RuntimeError):
    """Raised when MCP OAuth discovery cannot produce usable metadata."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class MCPOAuthRuntimeError(RuntimeError):
    """Raised when runtime cannot prepare an MCP OAuth bearer token."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class _RetryRuntimeGrantSelection(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class MCPAuthorizationChallenge:
    """Bearer challenge advertised by a protected MCP resource."""

    resource_metadata_url: str | None
    scope: str | None
    params: dict[str, str]


@dataclass(frozen=True)
class MCPProtectedResourceMetadata:
    """OAuth Protected Resource Metadata for an MCP endpoint."""

    url: str
    resource: str | None
    authorization_servers: tuple[str, ...]
    scopes_supported: tuple[str, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class OAuthAuthorizationServerMetadata:
    """OAuth/OIDC authorization server metadata needed for MCP OAuth."""

    url: str
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None
    client_id_metadata_document_supported: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class MCPOAuthDiscoveryResult:
    """Complete metadata selected for an MCP OAuth authorization flow."""

    challenge: MCPAuthorizationChallenge | None
    protected_resource: MCPProtectedResourceMetadata
    authorization_server: OAuthAuthorizationServerMetadata
    resource: str
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class MCPOAuthRuntimeAuth:
    """Prepared MCP OAuth bearer authorization for runtime MCP connections."""

    access_token: str
    resource_owner_key: str
    issuer: str
    resource: str
    scope: str
    grant_id: int
    refreshed: bool = False


class SafeOAuthAsyncHTTPTransport(httpx.AsyncBaseTransport):
    """HTTP transport that resolves and pins OAuth hosts before connecting."""

    def __init__(self) -> None:
        proxy_url = get_mcp_oauth_proxy_url()
        self._transport = httpx.AsyncHTTPTransport(
            limits=httpx.Limits(max_keepalive_connections=0),
            trust_env=False,
            http2=False,
            proxy=proxy_url,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_url = request.url
        original_host = request.headers.get("Host")
        resolved_ips = await _resolve_allowed_addresses(str(original_url))
        last_connect_error: httpx.TransportError | None = None
        for index, resolved_ip in enumerate(resolved_ips):
            request.url = original_url.copy_with(host=resolved_ip)
            request.headers["Host"] = _host_header_value(str(original_url))
            if original_url.scheme == "https":
                request.extensions["sni_hostname"] = _hostname_for_url(
                    str(original_url)
                )
            try:
                return await self._transport.handle_async_request(request)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_connect_error = exc
                if index == len(resolved_ips) - 1:
                    raise
            finally:
                request.url = original_url
                if original_host is None:
                    request.headers.pop("Host", None)
                else:
                    request.headers["Host"] = original_host
        if last_connect_error is not None:
            raise last_connect_error
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "Could not resolve OAuth metadata host",
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


def create_mcp_oauth_http_client(
    *, timeout: float, follow_redirects: bool = False
) -> httpx.AsyncClient:
    """Create an OAuth HTTP client that applies the MCP OAuth URL policy."""
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=follow_redirects,
        transport=SafeOAuthAsyncHTTPTransport(),
    )


async def oauth_get(
    url: str,
    *,
    client: httpx.AsyncClient,
    headers: dict[str, str] | None = None,
    allow_redirects: bool = True,
    resolve_dns_for_url_policy: bool = False,
) -> httpx.Response:
    """GET an OAuth URL through the shared URL policy and redirect rules."""
    current_url = url
    current_headers = dict(headers or {})
    for redirect_count in range(MCP_OAUTH_MAX_REDIRECTS + 1):
        await validate_oauth_http_url(
            current_url, resolve_dns=resolve_dns_for_url_policy
        )
        try:
            response = await client.get(
                current_url, headers=current_headers, follow_redirects=False
            )
        except httpx.RemoteProtocolError as exc:
            raise MCPOAuthDiscoveryError(
                "metadata_not_found",
                "OAuth metadata redirect Location was invalid",
            ) from exc
        if not allow_redirects or response.status_code not in OAUTH_REDIRECT_STATUSES:
            return response
        if redirect_count >= MCP_OAUTH_MAX_REDIRECTS:
            raise MCPOAuthDiscoveryError(
                "metadata_not_found",
                "OAuth metadata redirects exceeded the maximum allowed hops",
            )
        location = response.headers.get("location")
        if not location:
            raise MCPOAuthDiscoveryError(
                "metadata_not_found",
                "OAuth metadata redirect did not include a Location header",
            )
        try:
            next_url = urljoin(str(response.url), location)
            await validate_oauth_http_url(
                next_url, resolve_dns=resolve_dns_for_url_policy
            )
        except ValueError as exc:
            raise MCPOAuthDiscoveryError(
                "metadata_not_found",
                "OAuth metadata redirect Location was invalid",
            ) from exc
        if not _same_origin(current_url, next_url):
            current_headers = _headers_without_cross_origin_secrets(current_headers)
        current_url = next_url

    raise MCPOAuthDiscoveryError(
        "metadata_not_found",
        "OAuth metadata redirects exceeded the maximum allowed hops",
    )


async def oauth_post(
    url: str,
    *,
    client: httpx.AsyncClient,
    resolve_dns_for_url_policy: bool = False,
    **request_kwargs: Any,
) -> httpx.Response:
    """POST to an OAuth token-like endpoint without following redirects."""
    await validate_oauth_http_url(url, resolve_dns=resolve_dns_for_url_policy)
    try:
        response = await client.post(url, follow_redirects=False, **request_kwargs)
    except httpx.RemoteProtocolError as exc:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth token endpoint redirects are not supported",
        ) from exc
    if 300 <= response.status_code < 400:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth token endpoint redirects are not supported",
        )
    return response


def parse_www_authenticate_bearer(
    headers: str | Sequence[str] | None,
) -> MCPAuthorizationChallenge | None:
    """Parse the first Bearer challenge from one or more WWW-Authenticate headers."""
    for header_value in _iter_header_values(headers):
        challenge = _parse_bearer_challenge(header_value)
        if challenge is not None:
            return challenge
    return None


def protected_resource_metadata_urls(endpoint_url: str) -> tuple[str, ...]:
    """Return MCP protected-resource metadata candidates in spec priority order."""
    parts = urlsplit(endpoint_url)
    if not parts.scheme or not parts.netloc:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "MCP endpoint URL must be an absolute HTTP(S) URL",
        )

    root_metadata = urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            "/.well-known/oauth-protected-resource",
            "",
            "",
        )
    )
    path = parts.path.rstrip("/")
    if not path:
        return (root_metadata,)

    path_metadata = urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            f"/.well-known/oauth-protected-resource{path}",
            "",
            "",
        )
    )
    return _dedupe_urls((path_metadata, root_metadata))


def authorization_server_metadata_urls(issuer_url: str) -> tuple[str, ...]:
    """Return OAuth/OIDC metadata candidates for an authorization server issuer."""
    parts = urlsplit(issuer_url)
    if not parts.scheme or not parts.netloc:
        raise MCPOAuthDiscoveryError(
            "authorization_server_not_found",
            "Authorization server URL must be absolute",
        )

    base = (parts.scheme.lower(), parts.netloc.lower())
    issuer_path = parts.path.rstrip("/")
    if issuer_path:
        return _dedupe_urls(
            (
                urlunsplit(
                    (
                        *base,
                        f"/.well-known/oauth-authorization-server{issuer_path}",
                        "",
                        "",
                    )
                ),
                urlunsplit(
                    (
                        *base,
                        f"/.well-known/openid-configuration{issuer_path}",
                        "",
                        "",
                    )
                ),
                urlunsplit(
                    (
                        *base,
                        f"{issuer_path}/.well-known/openid-configuration",
                        "",
                        "",
                    )
                ),
            )
        )

    return _dedupe_urls(
        (
            urlunsplit((*base, "/.well-known/oauth-authorization-server", "", "")),
            urlunsplit((*base, "/.well-known/openid-configuration", "", "")),
        )
    )


async def discover_mcp_oauth_metadata(  # noqa: PLR0913
    endpoint_url: str,
    *,
    headers: dict[str, str] | None = None,
    configured_resource_metadata_url: str | None = None,
    configured_issuer: str | None = None,
    configured_resource: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> MCPOAuthDiscoveryResult:
    """Discover MCP protected-resource and OAuth authorization-server metadata."""
    if client is not None:
        return await _discover_with_client(
            endpoint_url,
            headers=headers,
            configured_resource_metadata_url=configured_resource_metadata_url,
            configured_issuer=configured_issuer,
            configured_resource=configured_resource,
            client=client,
            resolve_dns_for_url_policy=False,
        )

    async with create_mcp_oauth_http_client(
        timeout=DEFAULT_MCP_OAUTH_DISCOVERY_TIMEOUT,
    ) as owned_client:
        return await _discover_with_client(
            endpoint_url,
            headers=headers,
            configured_resource_metadata_url=configured_resource_metadata_url,
            configured_issuer=configured_issuer,
            configured_resource=configured_resource,
            client=owned_client,
            resolve_dns_for_url_policy=False,
        )


async def resolve_mcp_oauth_runtime_auth(  # noqa: PLR0913
    db: Any,
    *,
    server_id: int,
    user_id: int,
    auth_config: dict[str, Any],
    resource_owner_key: str,
    resource: str | None = None,
    scope: str | None = None,
    issuer: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> MCPOAuthRuntimeAuth:
    """Resolve and refresh an MCP OAuth grant for one runtime MCP connection."""
    from ...core.utils.encryption import decrypt_value

    normalized_resource = _runtime_config_value(resource, auth_config, "resource")
    if normalized_resource:
        normalized_resource = _canonical_resource(normalized_resource)
    selected_scope = scope if scope is not None else auth_config.get("scope")
    try:
        normalized_scope = (
            normalize_mcp_oauth_scope(selected_scope) if selected_scope else None
        )
    except MCPOAuthDiscoveryError as exc:
        raise MCPOAuthRuntimeError(exc.code, exc.message) from exc
    if not normalized_resource:
        raise MCPOAuthRuntimeError(
            "authorization_required",
            "MCP OAuth runtime requires a configured resource or runtime resource",
        )

    required_scopes = _scope_set(normalized_scope) if normalized_scope else set()

    for attempt in range(2):
        candidate_grants = select_mcp_oauth_grants(
            db,
            server_id=server_id,
            user_id=user_id,
            auth_config=auth_config,
            resource_owner_key=resource_owner_key,
            resource=resource,
            issuer=issuer,
            scope="",
        )
        grant = _select_runtime_grant(candidate_grants, required_scopes)
        if not _grant_needs_refresh(grant.expires_at):
            access_token = decrypt_value(str(grant.access_token))
            return MCPOAuthRuntimeAuth(
                access_token=access_token,
                resource_owner_key=str(grant.resource_owner_key),
                issuer=str(grant.issuer),
                resource=str(grant.resource),
                scope=str(grant.scope),
                grant_id=int(grant.id),
                refreshed=False,
            )

        try:
            runtime_auth = await _refresh_runtime_grant_in_dedicated_session(
                db,
                grant_id=int(grant.id),
                server_id=server_id,
                user_id=user_id,
                auth_config=auth_config,
                resource_owner_key=resource_owner_key,
                resource=resource,
                issuer=issuer,
                required_scopes=required_scopes,
                client=client,
            )
            db.expire(grant)
            return runtime_auth
        except _RetryRuntimeGrantSelection as exc:
            db.expire_all()
            if attempt == 0:
                continue
            raise MCPOAuthRuntimeError(exc.code, exc.message) from exc

    raise MCPOAuthRuntimeError(
        "authorization_required",
        "MCP OAuth grant changed while preparing runtime authorization",
    )


async def _refresh_runtime_grant_in_dedicated_session(  # noqa: PLR0913
    db: Any,
    *,
    grant_id: int,
    server_id: int,
    user_id: int,
    auth_config: dict[str, Any],
    resource_owner_key: str,
    resource: str | None,
    issuer: str | None,
    required_scopes: set[str],
    client: httpx.AsyncClient | None,
) -> MCPOAuthRuntimeAuth:
    """Refresh one runtime grant without committing or rolling back caller state."""
    from ...core.utils.encryption import decrypt_value, encrypt_value
    from ..models.mcp_oauth import MCPOAuthGrant

    SessionLocal = sessionmaker(
        bind=db.get_bind(),
        autoflush=False,
        autocommit=False,
    )
    refresh_db = SessionLocal()
    try:
        grant: Any = (
            refresh_db.query(MCPOAuthGrant).filter(MCPOAuthGrant.id == grant_id).first()
        )
        if grant is None or not _runtime_grant_matches(
            grant,
            server_id=server_id,
            user_id=user_id,
            auth_config=auth_config,
            resource_owner_key=resource_owner_key,
            resource=resource,
            issuer=issuer,
            required_scopes=required_scopes,
        ):
            raise _RetryRuntimeGrantSelection(
                "authorization_required",
                "MCP OAuth grant changed while preparing runtime authorization",
            )

        if not _grant_needs_refresh(grant.expires_at):
            access_token = decrypt_value(str(grant.access_token))
            return MCPOAuthRuntimeAuth(
                access_token=access_token,
                resource_owner_key=str(grant.resource_owner_key),
                issuer=str(grant.issuer),
                resource=str(grant.resource),
                scope=str(grant.scope),
                grant_id=int(grant.id),
                refreshed=False,
            )
        if not grant.refresh_token:
            raise _RetryRuntimeGrantSelection(
                "authorization_required",
                "MCP OAuth grant is expired and requires reauthorization",
            )
        if grant.oauth_client is None:
            raise MCPOAuthRuntimeError(
                "token_refresh_failed",
                "MCP OAuth client metadata not found for grant refresh",
            )

        encrypted_refresh_token = str(grant.refresh_token)
        refresh_token = decrypt_value(encrypted_refresh_token)
        refresh_resource = str(grant.resource)
        oauth_client_snapshot = SimpleNamespace(
            token_endpoint=str(grant.oauth_client.token_endpoint),
            client_id=str(grant.oauth_client.client_id),
            client_secret=grant.oauth_client.client_secret,
            token_endpoint_auth_method=str(
                grant.oauth_client.token_endpoint_auth_method or "none"
            ),
        )
        refresh_db.rollback()

        token_data = await _refresh_mcp_oauth_grant(
            oauth_client_snapshot,
            refresh_token=refresh_token,
            resource=refresh_resource,
            client=client,
        )

        with refresh_db.begin():
            locked_grant: Any = (
                refresh_db.query(MCPOAuthGrant)
                .filter(MCPOAuthGrant.id == grant_id)
                .with_for_update()
                .first()
            )
            if locked_grant is None or not _runtime_grant_matches(
                locked_grant,
                server_id=server_id,
                user_id=user_id,
                auth_config=auth_config,
                resource_owner_key=resource_owner_key,
                resource=resource,
                issuer=issuer,
                required_scopes=required_scopes,
            ):
                raise _RetryRuntimeGrantSelection(
                    "authorization_required",
                    "MCP OAuth grant changed while preparing runtime authorization",
                )

            if not _grant_needs_refresh(locked_grant.expires_at):
                access_token = decrypt_value(str(locked_grant.access_token))
                return MCPOAuthRuntimeAuth(
                    access_token=access_token,
                    resource_owner_key=str(locked_grant.resource_owner_key),
                    issuer=str(locked_grant.issuer),
                    resource=str(locked_grant.resource),
                    scope=str(locked_grant.scope),
                    grant_id=int(locked_grant.id),
                    refreshed=False,
                )
            if (
                not locked_grant.refresh_token
                or str(locked_grant.refresh_token) != encrypted_refresh_token
            ):
                raise _RetryRuntimeGrantSelection(
                    "authorization_required",
                    "MCP OAuth grant changed while preparing runtime authorization",
                )

            try:
                refreshed_scope = (
                    normalize_mcp_oauth_scope(token_data.get("scope"))
                    if token_data.get("scope") is not None
                    else str(locked_grant.scope)
                )
            except MCPOAuthDiscoveryError as exc:
                raise MCPOAuthRuntimeError("token_refresh_failed", exc.message) from exc
            if required_scopes and not required_scopes.issubset(
                _scope_set(refreshed_scope)
            ):
                raise MCPOAuthRuntimeError(
                    "insufficient_scope",
                    "Refreshed MCP OAuth grant does not include the required scope",
                )
            locked_grant.access_token = encrypt_value(str(token_data["access_token"]))
            access_token = str(token_data["access_token"])
            if token_data.get("refresh_token"):
                locked_grant.refresh_token = encrypt_value(
                    str(token_data["refresh_token"])
                )
            try:
                locked_grant.token_type = validate_mcp_oauth_persisted_value(
                    str(token_data.get("token_type") or "Bearer"),
                    field_name="token_type",
                    max_length=MCP_OAUTH_TOKEN_TYPE_MAX_LENGTH,
                )
            except MCPOAuthDiscoveryError as exc:
                raise MCPOAuthRuntimeError("token_refresh_failed", exc.message) from exc
            if token_data.get("scope") is not None:
                locked_grant.scope = refreshed_scope
            locked_grant.metadata_json = {
                key: value
                for key, value in token_data.items()
                if key not in {"access_token", "refresh_token"}
            }
            locked_grant.expires_at = oauth_token_expires_at(token_data)
            runtime_auth = MCPOAuthRuntimeAuth(
                access_token=access_token,
                resource_owner_key=str(locked_grant.resource_owner_key),
                issuer=str(locked_grant.issuer),
                resource=str(locked_grant.resource),
                scope=str(locked_grant.scope),
                grant_id=int(locked_grant.id),
                refreshed=True,
            )

        return runtime_auth
    finally:
        refresh_db.close()


def select_mcp_oauth_grants(  # noqa: PLR0913
    db: Any,
    *,
    server_id: int,
    user_id: int,
    auth_config: dict[str, Any],
    resource_owner_key: str | None = None,
    resource: str | None = None,
    issuer: str | None = None,
    scope: str | None = None,
) -> list[Any]:
    """Return active MCP OAuth grants matching the current server auth config.

    This selector is intentionally side-effect free: it does not decrypt tokens,
    refresh grants, or mutate persistence. Runtime code can build on the selected
    grants when it needs bearer material.
    """
    from ..models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant

    normalized_resource = _runtime_config_value(resource, auth_config, "resource")
    if normalized_resource:
        normalized_resource = _canonical_resource(normalized_resource)
    normalized_issuer = _runtime_config_value(issuer, auth_config, "issuer")
    if normalized_issuer:
        normalized_issuer = _canonical_issuer(normalized_issuer)
    selected_scope = scope if scope is not None else auth_config.get("scope")
    normalized_scope = _normalize_scope(selected_scope) if selected_scope else None
    configured_client_id = _runtime_config_value(None, auth_config, "client_id")
    if not normalized_resource:
        return []

    query = (
        db.query(MCPOAuthGrant)
        .join(MCPOAuthClient, MCPOAuthGrant.mcp_oauth_client_id == MCPOAuthClient.id)
        .filter(
            MCPOAuthGrant.mcp_server_id == server_id,
            MCPOAuthGrant.user_id == user_id,
            MCPOAuthGrant.resource == normalized_resource,
            MCPOAuthGrant.status == "active",
            MCPOAuthClient.mcp_server_id == server_id,
        )
    )
    if resource_owner_key:
        query = query.filter(MCPOAuthGrant.resource_owner_key == resource_owner_key)
    if normalized_issuer:
        query = query.filter(MCPOAuthGrant.issuer == normalized_issuer)
    if configured_client_id:
        query = query.filter(MCPOAuthClient.client_id == configured_client_id)

    grants = query.order_by(MCPOAuthGrant.updated_at.desc()).all()
    required_scopes = _scope_set(normalized_scope) if normalized_scope else set()
    if not required_scopes:
        return list(grants)
    return [
        grant for grant in grants if required_scopes.issubset(_scope_set(grant.scope))
    ]


async def _discover_with_client(  # noqa: PLR0913
    endpoint_url: str,
    *,
    headers: dict[str, str] | None,
    configured_resource_metadata_url: str | None,
    configured_issuer: str | None,
    configured_resource: str | None,
    client: httpx.AsyncClient,
    resolve_dns_for_url_policy: bool,
) -> MCPOAuthDiscoveryResult:
    challenge: MCPAuthorizationChallenge | None = None
    metadata_urls: tuple[str, ...]

    if configured_resource_metadata_url:
        metadata_urls = (configured_resource_metadata_url,)
    else:
        challenge = await _probe_authorization_challenge(
            endpoint_url,
            headers=headers,
            client=client,
            resolve_dns_for_url_policy=resolve_dns_for_url_policy,
        )
        if challenge and challenge.resource_metadata_url:
            metadata_urls = (challenge.resource_metadata_url,)
        else:
            metadata_urls = protected_resource_metadata_urls(endpoint_url)

    protected_resource = await _fetch_first_protected_resource_metadata(
        metadata_urls,
        client=client,
        resolve_dns_for_url_policy=resolve_dns_for_url_policy,
    )
    resource = _canonical_resource(protected_resource.resource or endpoint_url)
    if configured_resource and not _same_url(configured_resource, resource):
        raise MCPOAuthDiscoveryError(
            "resource_mismatch",
            "Configured MCP OAuth resource does not match protected resource metadata",
        )

    authorization_server_url = _select_authorization_server(
        protected_resource.authorization_servers,
        configured_issuer=configured_issuer,
    )
    authorization_server = await _fetch_authorization_server_metadata(
        authorization_server_url,
        configured_issuer=configured_issuer,
        client=client,
        resolve_dns_for_url_policy=resolve_dns_for_url_policy,
    )
    scopes = _select_scopes(challenge, protected_resource)

    return MCPOAuthDiscoveryResult(
        challenge=challenge,
        protected_resource=protected_resource,
        authorization_server=authorization_server,
        resource=resource,
        scopes=scopes,
    )


async def _probe_authorization_challenge(
    endpoint_url: str,
    *,
    headers: dict[str, str] | None,
    client: httpx.AsyncClient,
    resolve_dns_for_url_policy: bool,
) -> MCPAuthorizationChallenge | None:
    try:
        response = await oauth_get(
            endpoint_url,
            headers=headers,
            client=client,
            resolve_dns_for_url_policy=resolve_dns_for_url_policy,
        )
    except httpx.HTTPError as exc:
        raise MCPOAuthDiscoveryError(
            "metadata_not_found",
            f"Failed to probe MCP endpoint for OAuth challenge: {exc}",
        ) from exc

    return parse_www_authenticate_bearer(response.headers.get_list("WWW-Authenticate"))


async def _fetch_first_protected_resource_metadata(
    metadata_urls: Sequence[str],
    *,
    client: httpx.AsyncClient,
    resolve_dns_for_url_policy: bool,
) -> MCPProtectedResourceMetadata:
    last_error: Exception | None = None
    for metadata_url in metadata_urls:
        try:
            response = await oauth_get(
                metadata_url,
                client=client,
                resolve_dns_for_url_policy=resolve_dns_for_url_policy,
            )
            if response.status_code >= 400:
                last_error = MCPOAuthDiscoveryError(
                    "metadata_not_found",
                    f"Protected resource metadata returned HTTP {response.status_code}",
                )
                continue
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("metadata response is not a JSON object")
            return _parse_protected_resource_metadata(metadata_url, payload)
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc

    raise MCPOAuthDiscoveryError(
        "metadata_not_found",
        f"Could not load MCP protected resource metadata: {last_error}",
    )


async def _fetch_authorization_server_metadata(
    authorization_server_url: str,
    *,
    configured_issuer: str | None,
    client: httpx.AsyncClient,
    resolve_dns_for_url_policy: bool,
) -> OAuthAuthorizationServerMetadata:
    last_error: Exception | None = None
    for metadata_url in authorization_server_metadata_urls(authorization_server_url):
        try:
            response = await oauth_get(
                metadata_url,
                client=client,
                resolve_dns_for_url_policy=resolve_dns_for_url_policy,
            )
            if response.status_code >= 400:
                last_error = MCPOAuthDiscoveryError(
                    "authorization_server_not_found",
                    f"Authorization server metadata returned HTTP {response.status_code}",
                )
                continue
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("metadata response is not a JSON object")
            metadata = _parse_authorization_server_metadata(metadata_url, payload)
            await _validate_authorization_server_endpoints(
                metadata,
                resolve_dns_for_url_policy=resolve_dns_for_url_policy,
            )
            _validate_issuer(
                selected_authorization_server=authorization_server_url,
                metadata=metadata,
                configured_issuer=configured_issuer,
            )
            return metadata
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc

    raise MCPOAuthDiscoveryError(
        "authorization_server_not_found",
        f"Could not load OAuth authorization server metadata: {last_error}",
    )


def _parse_protected_resource_metadata(
    metadata_url: str, payload: dict[str, Any]
) -> MCPProtectedResourceMetadata:
    authorization_servers = _string_tuple(payload.get("authorization_servers"))
    if not authorization_servers:
        raise MCPOAuthDiscoveryError(
            "authorization_server_not_found",
            "Protected resource metadata did not include authorization_servers",
        )
    return MCPProtectedResourceMetadata(
        url=metadata_url,
        resource=_optional_string(payload.get("resource"), field_name="resource"),
        authorization_servers=authorization_servers,
        scopes_supported=_string_tuple(payload.get("scopes_supported")),
        raw=payload,
    )


def _parse_authorization_server_metadata(
    metadata_url: str, payload: dict[str, Any]
) -> OAuthAuthorizationServerMetadata:
    issuer = _canonical_issuer(_required_string(payload, "issuer"))
    authorization_endpoint = _required_string(payload, "authorization_endpoint")
    token_endpoint = _required_string(payload, "token_endpoint")
    return OAuthAuthorizationServerMetadata(
        url=metadata_url,
        issuer=issuer,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        registration_endpoint=_optional_string(
            payload.get("registration_endpoint"), field_name="registration_endpoint"
        ),
        client_id_metadata_document_supported=bool(
            payload.get("client_id_metadata_document_supported")
        ),
        raw=payload,
    )


async def _validate_authorization_server_endpoints(
    metadata: OAuthAuthorizationServerMetadata, *, resolve_dns_for_url_policy: bool
) -> None:
    await validate_oauth_http_url(
        metadata.authorization_endpoint, resolve_dns=resolve_dns_for_url_policy
    )
    await validate_oauth_http_url(
        metadata.token_endpoint, resolve_dns=resolve_dns_for_url_policy
    )


def _parse_bearer_challenge(header_value: str) -> MCPAuthorizationChallenge | None:
    match = re.search(r"(?i)(?:^|,\s*)Bearer(?:\s+|$)", header_value)
    if not match:
        return None

    params_text = header_value[match.end() :].strip()
    try:
        params = {
            str(key).lower(): str(value)
            for key, value in parse_keqv_list(parse_http_list(params_text)).items()
            if value is not None
        }
    except Exception:
        return None
    return MCPAuthorizationChallenge(
        resource_metadata_url=params.get("resource_metadata"),
        scope=params.get("scope"),
        params=params,
    )


def _select_authorization_server(
    authorization_servers: Sequence[str], *, configured_issuer: str | None
) -> str:
    if not authorization_servers:
        raise MCPOAuthDiscoveryError(
            "authorization_server_not_found",
            "Protected resource metadata did not include authorization servers",
        )
    if configured_issuer:
        for authorization_server in authorization_servers:
            if _same_url(authorization_server, configured_issuer):
                return authorization_server
        raise MCPOAuthDiscoveryError(
            "issuer_mismatch",
            "Configured issuer is not advertised by protected resource metadata",
        )
    return authorization_servers[0]


def _validate_issuer(
    *,
    selected_authorization_server: str,
    metadata: OAuthAuthorizationServerMetadata,
    configured_issuer: str | None,
) -> None:
    expected = configured_issuer or selected_authorization_server
    if not _same_url(metadata.issuer, expected):
        raise MCPOAuthDiscoveryError(
            "issuer_mismatch",
            "Authorization server metadata issuer did not match the selected issuer",
        )


def _select_scopes(
    challenge: MCPAuthorizationChallenge | None,
    protected_resource: MCPProtectedResourceMetadata,
) -> tuple[str, ...]:
    if challenge and challenge.scope:
        return tuple(scope for scope in challenge.scope.split() if scope)
    return protected_resource.scopes_supported


async def _refresh_mcp_oauth_grant(
    oauth_client: Any,
    *,
    refresh_token: str,
    resource: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": oauth_client.client_id,
        "resource": resource,
    }
    auth: httpx.Auth | None = None
    client_secret = ""
    if oauth_client.client_secret:
        from ...core.utils.encryption import decrypt_value

        client_secret = decrypt_value(str(oauth_client.client_secret))
    auth_method = str(oauth_client.token_endpoint_auth_method or "none")
    if auth_method == "client_secret_post" and client_secret:
        data["client_secret"] = client_secret
    elif auth_method == "client_secret_basic" and client_secret:
        auth = httpx.BasicAuth(str(oauth_client.client_id), client_secret)
    elif auth_method not in {"none", "client_secret_post", "client_secret_basic"}:
        raise MCPOAuthRuntimeError(
            "token_refresh_failed",
            f"Unsupported token endpoint auth method: {auth_method}",
        )

    try:
        request_kwargs: dict[str, Any] = {
            "data": data,
            "headers": {"Content-Type": "application/x-www-form-urlencoded"},
        }
        if auth is not None:
            request_kwargs["auth"] = auth
        if client is not None:
            response = await oauth_post(
                str(oauth_client.token_endpoint),
                client=client,
                resolve_dns_for_url_policy=False,
                **request_kwargs,
            )
        else:
            async with create_mcp_oauth_http_client(
                timeout=MCP_OAUTH_HTTP_TIMEOUT_SECONDS,
            ) as owned_client:
                response = await oauth_post(
                    str(oauth_client.token_endpoint),
                    client=owned_client,
                    resolve_dns_for_url_policy=False,
                    **request_kwargs,
                )
        payload = response.json()
    except (MCPOAuthDiscoveryError, httpx.HTTPError, ValueError) as exc:
        raise MCPOAuthRuntimeError(
            "token_refresh_failed",
            oauth_exception_message(exc, "MCP OAuth refresh failed"),
        ) from exc

    if (
        response.status_code >= 400
        or not isinstance(payload, dict)
        or payload.get("error")
        or not payload.get("access_token")
    ):
        logger.warning(
            "MCP OAuth refresh failed with token endpoint payload: %s",
            oauth_error_log_payload(payload),
        )
        raise MCPOAuthRuntimeError(
            "token_refresh_failed",
            oauth_error_message(payload, "MCP OAuth refresh failed"),
        )
    return payload


def _runtime_config_value(
    request_value: str | None, auth_config: dict[str, Any], key: str
) -> str | None:
    value = request_value if request_value is not None else auth_config.get(key)
    return str(value).strip() if value else None


def _normalize_scope(value: Any) -> str:
    # OAuth scope values are set-like; canonical ordering keeps lookup keys stable
    # and is safe to reuse for authorization requests to compliant servers.
    if isinstance(value, str):
        return " ".join(sorted({item for item in value.split() if item}))
    if isinstance(value, (list, tuple, set)):
        return " ".join(sorted({str(item) for item in value if str(item)}))
    return ""


def normalize_mcp_oauth_scope(value: Any) -> str:
    """Canonicalize a scope string that can be persisted in grant lookup keys."""
    scope = _normalize_scope(value)
    if len(scope) > MCP_OAUTH_SCOPE_MAX_LENGTH:
        raise MCPOAuthDiscoveryError(
            "invalid_scope",
            f"MCP OAuth scope must be at most {MCP_OAUTH_SCOPE_MAX_LENGTH} characters",
        )
    return scope


def validate_mcp_oauth_persisted_value(
    value: str,
    *,
    field_name: str,
    max_length: int = MCP_OAUTH_PERSISTED_VALUE_MAX_LENGTH,
) -> str:
    """Validate a string that will be stored in a bounded OAuth column."""
    if len(value) > max_length:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            f"MCP OAuth {field_name} must be at most {max_length} characters",
        )
    return value


def oauth_token_expires_at(token_data: dict[str, Any]) -> datetime | None:
    """Map OAuth token response expiry into stored grant expiry.

    OAuth access tokens can be opaque/non-expiring from the client's point of
    view when ``expires_in`` is omitted. Store that as ``None`` so refresh
    selection does not keep treating a newly refreshed token as expired.
    """
    expires_in = token_data.get("expires_in")
    if expires_in is None:
        return None
    return _utc_now() + timedelta(seconds=int(expires_in))


def _scope_set(value: Any) -> set[str]:
    return set(_normalize_scope(value).split())


def _select_runtime_grant(grants: Sequence[Any], required_scopes: set[str]) -> Any:
    scope_matches = [
        grant
        for grant in grants
        if not required_scopes or required_scopes.issubset(_scope_set(grant.scope))
    ]
    if not scope_matches:
        if grants and required_scopes:
            raise MCPOAuthRuntimeError(
                "insufficient_scope",
                "No active MCP OAuth grant includes the required scope",
            )
        raise MCPOAuthRuntimeError(
            "authorization_required",
            "No active MCP OAuth grant exists for the selected resource owner",
        )

    for grant in scope_matches:
        if not _grant_needs_refresh(grant.expires_at):
            return grant
    for grant in scope_matches:
        if grant.refresh_token:
            return grant
    raise MCPOAuthRuntimeError(
        "authorization_required",
        "MCP OAuth grant is expired and requires reauthorization",
    )


def _runtime_grant_matches(  # noqa: PLR0913
    grant: Any,
    *,
    server_id: int,
    user_id: int,
    auth_config: dict[str, Any],
    resource_owner_key: str,
    resource: str | None,
    issuer: str | None,
    required_scopes: set[str],
) -> bool:
    normalized_resource = _runtime_config_value(resource, auth_config, "resource")
    if normalized_resource:
        normalized_resource = _canonical_resource(normalized_resource)
    normalized_issuer = _runtime_config_value(issuer, auth_config, "issuer")
    if normalized_issuer:
        normalized_issuer = _canonical_issuer(normalized_issuer)
    configured_client_id = _runtime_config_value(None, auth_config, "client_id")
    oauth_client = getattr(grant, "oauth_client", None)
    return bool(
        grant.mcp_server_id == server_id
        and grant.user_id == user_id
        and grant.resource_owner_key == resource_owner_key
        and grant.resource == normalized_resource
        and grant.status == "active"
        and (not normalized_issuer or grant.issuer == normalized_issuer)
        and (
            not configured_client_id
            or (
                oauth_client is not None
                and oauth_client.mcp_server_id == server_id
                and oauth_client.client_id == configured_client_id
            )
        )
        and (not required_scopes or required_scopes.issubset(_scope_set(grant.scope)))
    )


def _grant_needs_refresh(expires_at: Any) -> bool:
    if not isinstance(expires_at, datetime):
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return bool(expires_at <= _utc_now() + timedelta(minutes=5))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_resource(endpoint_url: str) -> str:
    return _canonical_url_identifier(endpoint_url)


def _canonical_issuer(issuer_url: str) -> str:
    return _canonical_url_identifier(issuer_url)


def _canonical_url_identifier(endpoint_url: str) -> str:
    parts = urlsplit(endpoint_url)
    if not parts.scheme or not parts.netloc:
        return endpoint_url.rstrip("/")
    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    hostname = (parts.hostname or "").rstrip(".").lower()
    netloc = (
        f"[{hostname}]"
        if ":" in hostname and not hostname.startswith("[")
        else hostname
    )
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{netloc}:{port}"
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise MCPOAuthDiscoveryError(
            "unsupported_auth_server",
            f"Authorization server metadata missing required field '{key}'",
        )
    return validate_mcp_oauth_persisted_value(value, field_name=key)


def _optional_string(value: Any, *, field_name: str) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return validate_mcp_oauth_persisted_value(value, field_name=field_name)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _iter_header_values(headers: str | Sequence[str] | None) -> Iterable[str]:
    if headers is None:
        return ()
    if isinstance(headers, str):
        return (headers,)
    return tuple(str(header) for header in headers)


def _dedupe_urls(urls: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return tuple(deduped)


def _same_url(left: str, right: str) -> bool:
    return _url_comparison_key(left) == _url_comparison_key(right)


def _same_origin(left: str, right: str) -> bool:
    left_parts = urlsplit(left)
    right_parts = urlsplit(right)
    return (
        left_parts.scheme.lower(),
        _host_header_value(left),
    ) == (
        right_parts.scheme.lower(),
        _host_header_value(right),
    )


def _url_comparison_key(value: str) -> str:
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value.rstrip("/")

    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    hostname = (parts.hostname or "").lower()
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parts.query, ""))


async def validate_oauth_http_url(value: str, *, resolve_dns: bool) -> None:
    """Reject OAuth metadata/token URLs that can target local infrastructure.

    Production discovery creates its own HTTP client, so it resolves DNS and
    blocks private, loopback, link-local, reserved, multicast, and unspecified
    addresses before connecting. Unit tests often pass a MockTransport-backed
    client for synthetic domains; those still get literal IP / localhost checks
    without turning tests into live DNS lookups.
    """
    await _validate_and_resolve_oauth_http_url(value, resolve_dns=resolve_dns)


async def _resolve_allowed_addresses(value: str) -> list[str]:
    addresses = await _validate_and_resolve_oauth_http_url(value, resolve_dns=True)
    if not addresses:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "Could not resolve OAuth metadata host",
        )
    return addresses


async def _validate_and_resolve_oauth_http_url(
    value: str, *, resolve_dns: bool
) -> list[str]:
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth metadata URL must be an absolute HTTP(S) URL",
        )
    if parts.username or parts.password:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth metadata URL must not include userinfo",
        )

    port = _url_port(parts)
    hostname = parts.hostname.rstrip(".").lower()
    _reject_blocked_host(hostname)
    if not resolve_dns:
        return []

    try:
        loop = asyncio.get_running_loop()
        addresses = await loop.run_in_executor(
            None,
            socket.getaddrinfo,
            hostname,
            port or (443 if parts.scheme.lower() == "https" else 80),
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            f"Could not resolve OAuth metadata host: {hostname}",
        ) from exc

    resolved: list[str] = []
    for address in addresses:
        sockaddr = address[4]
        if sockaddr:
            ip_value = str(sockaddr[0])
            _reject_blocked_host(ip_value)
            if ip_value not in resolved:
                resolved.append(ip_value)
    return resolved


def _hostname_for_url(value: str) -> str:
    hostname = urlsplit(value).hostname
    if not hostname:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth metadata URL must include a hostname",
        )
    return hostname.rstrip(".").lower()


def _host_header_value(value: str) -> str:
    parts = urlsplit(value)
    hostname = _hostname_for_url(value)
    port = _url_port(parts)
    default_port = 443 if parts.scheme.lower() == "https" else 80
    header_host = (
        f"[{hostname}]"
        if ":" in hostname and not hostname.startswith("[")
        else hostname
    )
    if port and port != default_port:
        return f"{header_host}:{port}"
    return header_host


def _url_port(parts: SplitResult) -> int | None:
    try:
        return parts.port
    except ValueError as exc:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth metadata URL has an invalid port",
        ) from exc


def _headers_without_cross_origin_secrets(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in OAUTH_CROSS_ORIGIN_STRIPPED_HEADERS
    }


def oauth_error_message(payload: Any, fallback: str) -> str:
    """Return a bounded user-safe OAuth error message."""
    message: Any = None
    if isinstance(payload, dict):
        message = payload.get("error_description") or payload.get("error")
    elif isinstance(payload, str):
        message = payload
    text = str(message or fallback)
    return _truncate(text, OAUTH_ERROR_MESSAGE_MAX_LENGTH)


def oauth_exception_message(exc: Exception, fallback: str) -> str:
    """Return a bounded user-safe OAuth transport/parsing error message."""
    if isinstance(exc, MCPOAuthDiscoveryError):
        message = exc.message
    elif isinstance(exc, httpx.TimeoutException):
        message = "OAuth request timed out"
    elif isinstance(exc, httpx.HTTPError):
        message = "OAuth request failed"
    elif isinstance(exc, ValueError):
        message = "OAuth response was not valid JSON"
    else:
        message = fallback
    return oauth_error_message(str(message), fallback)


def oauth_error_log_payload(payload: Any) -> str:
    """Return a masked, bounded OAuth payload representation for logs."""
    masked = _mask_oauth_payload(payload)
    try:
        text = json.dumps(masked, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = repr(masked)
    return _truncate(text, OAUTH_LOG_PAYLOAD_MAX_LENGTH)


def _mask_oauth_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "********"
                if str(key).lower() in OAUTH_SENSITIVE_PAYLOAD_KEYS
                else _mask_oauth_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_mask_oauth_payload(item) for item in value]
    return value


def _truncate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."


def _reject_blocked_host(hostname: str) -> None:
    if get_mcp_oauth_allow_private_hosts():
        return
    try:
        reject_private_network_host(hostname)
    except PrivateNetworkHostError as exc:
        raise MCPOAuthDiscoveryError(
            "invalid_resource",
            "OAuth metadata host must not resolve to local addresses",
        ) from exc
