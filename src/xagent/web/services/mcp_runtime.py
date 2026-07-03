from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mcp_oauth import MCPOAuthRuntimeError, resolve_mcp_oauth_runtime_auth

HTTP_MCP_TRANSPORTS = frozenset({"sse", "websocket", "streamable_http"})


@dataclass(frozen=True)
class MCPRuntimeConnectionBuild:
    """Executable MCP connection plus any runtime authorization diagnostic."""

    connection: dict[str, Any] | None
    diagnostic: dict[str, Any] | None = None


def load_user_env_overrides(db: Any, user_id: int | None) -> dict[int, dict]:
    """Batch-load a user's decrypted per-user env overrides, keyed by server id.

    One query for all of a user's active overrides, so callers building many
    connections in a loop avoid an N+1 per-server lookup.
    """
    if not isinstance(user_id, int) or db is None:
        return {}
    from ...core.utils.encryption import decrypt_env_dict
    from ..models.mcp import UserMCPServer

    rows = (
        db.query(UserMCPServer.mcpserver_id, UserMCPServer.env)
        .filter(
            UserMCPServer.user_id == user_id,
            UserMCPServer.is_active,
            UserMCPServer.env.isnot(None),
        )
        .all()
    )
    overrides: dict[int, dict] = {}
    for mcpserver_id, env in rows:
        decrypted = decrypt_env_dict(env)
        if decrypted:
            overrides[mcpserver_id] = decrypted
    return overrides


def merge_stdio_env(
    global_env: dict[str, Any] | None, user_env: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Merge a user's per-user env over the global env (global = fallback, user wins)."""
    if not user_env:
        return global_env
    return {**(global_env or {}), **user_env}


async def build_mcp_runtime_connection(
    db: Any,
    server: Any,
    *,
    user_id: int | None,
    mcp_auth_context: dict[str, Any] | None = None,
    user_env_overrides: dict[int, dict] | None = None,
) -> MCPRuntimeConnectionBuild:
    """Build an executable MCP connection for a specific user runtime.

    Static MCP connection serialization belongs to the MCP server model. Per-user
    MCP OAuth grant selection belongs here, at the web runtime boundary.
    """
    connection = server.to_connection_dict()

    # Merge this user's per-user env override on top of the global env (stdio
    # only; global env acts as the fallback, user values win). Overrides are
    # prefetched by the caller (see load_user_env_overrides) to avoid N+1.
    if connection.get("transport") == "stdio" and user_env_overrides:
        server_id = getattr(server, "id", None)
        if isinstance(server_id, int):
            merged = merge_stdio_env(
                connection.get("env"), user_env_overrides.get(server_id)
            )
            if merged:
                connection["env"] = merged

    auth_config = server._decrypt_auth_config(getattr(server, "auth", None))
    if not _is_mcp_oauth_http_server(server, auth_config):
        return MCPRuntimeConnectionBuild(connection=connection)

    server_id = getattr(server, "id", None)
    if not isinstance(server_id, int) or not isinstance(user_id, int) or db is None:
        return MCPRuntimeConnectionBuild(
            connection=None,
            diagnostic=mcp_oauth_runtime_diagnostic(
                server,
                code="authorization_required",
                message=(
                    "MCP OAuth runtime requires a persisted server, user, and "
                    "database session"
                ),
            ),
        )

    selection = mcp_oauth_selection(mcp_auth_context, server_id)
    resource_owner_key = selection.get("resource_owner_key") or f"xagent:user:{user_id}"
    try:
        runtime_auth = await resolve_mcp_oauth_runtime_auth(
            db,
            server_id=server_id,
            user_id=user_id,
            auth_config=auth_config,
            resource_owner_key=str(resource_owner_key),
            resource=selection.get("resource"),
            scope=selection.get("scope"),
            issuer=selection.get("issuer"),
        )
    except MCPOAuthRuntimeError as exc:
        return MCPRuntimeConnectionBuild(
            connection=None,
            diagnostic=mcp_oauth_runtime_diagnostic(
                server,
                code=exc.code,
                message=exc.message,
                resource_owner_key=str(resource_owner_key),
                resource=selection.get("resource") or auth_config.get("resource"),
                scope=selection.get("scope") or auth_config.get("scope"),
                issuer=selection.get("issuer") or auth_config.get("issuer"),
            ),
        )

    headers = headers_without_authorization(
        connection.get("headers")
        if isinstance(connection.get("headers"), dict)
        else None
    )
    headers["Authorization"] = f"Bearer {runtime_auth.access_token}"
    connection["headers"] = headers
    connection.pop("auth", None)
    return MCPRuntimeConnectionBuild(connection=connection)


def connection_to_transport_config(connection: dict[str, Any]) -> dict[str, Any]:
    """Convert a direct MCP connection dict into WebToolConfig transport config."""
    return {
        key: value
        for key, value in connection.items()
        if key not in {"name", "transport"}
    }


def mcp_oauth_selection(
    mcp_auth_context: dict[str, Any] | None, server_id: int
) -> dict[str, Any]:
    """Return the runtime grant selection for one server."""
    selection = (mcp_auth_context or {}).get(str(server_id))
    return selection if isinstance(selection, dict) else {}


def headers_without_authorization(headers: dict[str, Any] | None) -> dict[str, Any]:
    """Copy headers while removing any static Authorization credential."""
    return {
        str(key): value
        for key, value in dict(headers or {}).items()
        if str(key).lower() != "authorization"
    }


def mcp_oauth_runtime_diagnostic(
    server: Any,
    *,
    code: str,
    message: str,
    resource_owner_key: str | None = None,
    resource: Any | None = None,
    scope: Any | None = None,
    issuer: Any | None = None,
) -> dict[str, Any]:
    """Build the common runtime diagnostic payload for MCP OAuth failures."""
    return {
        "code": code,
        "message": message,
        "server_id": getattr(server, "id", None),
        "server_name": getattr(server, "name", None),
        "resource_owner_key": resource_owner_key,
        "resource": str(resource) if resource else None,
        "scope": str(scope) if scope else "",
        "issuer": str(issuer) if issuer else None,
    }


def _is_mcp_oauth_http_server(server: Any, auth_config: Any) -> bool:
    return (
        getattr(server, "transport", None) in HTTP_MCP_TRANSPORTS
        and isinstance(auth_config, dict)
        and auth_config.get("type") == "mcp_oauth"
    )
