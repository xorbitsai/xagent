"""Authorization, identity, and dedicated-sandbox binding for Chrome MCP."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, cast
from uuid import UUID

from ...core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    CHROME_DEVTOOLS_APP_ID,
    CHROME_SANDBOX_LIFECYCLE_TYPE,
    ChromeExecutionScope,
    ChromeExecutionSessionPool,
    ChromeSandboxHandle,
    ChromeSessionContractError,
)
from ...core.tools.core.mcp.sessions import Connection
from ..builtin_mcp_registry import get_builtin_stdio_session_scope
from ..sandbox_manager import SandboxCapacityError, get_sandbox_manager
from .actor_mcp_runtime import (
    ActorMCPStdioConnectionIdentity,
    ActorMCPStdioSessionIdentity,
)
from .mcp_runtime import CALLER_ID_ENV_VAR, MCPActorExecutionIdentity

_CHROME_SCOPE_HASH_DOMAIN = b"xagent.chrome.execution-session.v1\x00"
_chrome_pool_manager: object | None = None
_chrome_pool: ChromeExecutionSessionPool | None = None


def _hash_identity_key(key: tuple[Any, ...]) -> str:
    digest = hashlib.sha256(_CHROME_SCOPE_HASH_DOMAIN)
    for value in key:
        if isinstance(value, bool):
            raise ChromeSessionContractError("Chrome session identity is invalid")
        if isinstance(value, int):
            encoded = f"i:{value}".encode()
        elif isinstance(value, str):
            encoded = b"s:" + value.encode("utf-8")
        elif isinstance(value, UUID):
            encoded = b"u:" + value.bytes
        else:
            raise ChromeSessionContractError("Chrome session identity is invalid")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def bind_chrome_execution_scope(
    server_name: str,
    connection: Mapping[str, Any],
    session_identity: ActorMCPStdioSessionIdentity,
) -> tuple[ChromeExecutionScope, Mapping[str, Any]]:
    """Bind an exact host-only identity to a secret-free Chrome connection."""

    if (
        server_name != CHROME_DEVTOOLS_APP_ID
        or get_builtin_stdio_session_scope(server_name) != "execution"
    ):
        raise ChromeSessionContractError(
            "Chrome session consumer requires the canonical Chrome app"
        )
    if "actor_stdio_session_identity" in connection:
        raise ChromeSessionContractError(
            "Chrome session identity must use the host-only side channel"
        )
    executable_connection = dict(connection)
    identity = session_identity
    if type(identity) is not ActorMCPStdioSessionIdentity:
        raise ChromeSessionContractError(
            "execution-scoped Chrome requires an exact session identity"
        )
    if (
        type(identity.execution) is not MCPActorExecutionIdentity
        or type(identity.connection) is not ActorMCPStdioConnectionIdentity
        or identity.connection.app_id != server_name
    ):
        raise ChromeSessionContractError("Chrome session identity does not match")
    env = executable_connection.get("env")
    if not isinstance(env, Mapping) or env.get(CALLER_ID_ENV_VAR) != str(
        identity.connection.user_id
    ):
        raise ChromeSessionContractError(
            "Chrome session caller identity does not match"
        )
    child_env = dict(env)
    child_env.pop(CALLER_ID_ENV_VAR, None)
    if child_env:
        raise ChromeSessionContractError(
            "Chrome session connection contains unexpected environment"
        )
    executable_connection["env"] = {}
    key = identity.key
    return (
        ChromeExecutionScope(key=key, digest=_hash_identity_key(key)),
        executable_connection,
    )


async def _create_chrome_sandbox(scope_digest: str) -> ChromeSandboxHandle:
    """Create and actively pin one sandbox whose name contains only a digest."""

    manager = get_sandbox_manager()
    if manager is None:
        raise ChromeSessionContractError("Chrome sandbox is unavailable")
    try:
        provider = await manager.get_or_create_lease_provider(
            CHROME_SANDBOX_LIFECYCLE_TYPE,
            scope_digest,
        )
        attached = await manager.attach_provider(
            CHROME_SANDBOX_LIFECYCLE_TYPE,
            scope_digest,
            provider,
        )
    except SandboxCapacityError as exc:
        raise ChromeSessionContractError("Chrome sandbox is unavailable") from exc
    except Exception as exc:
        raise ChromeSessionContractError("Chrome sandbox creation failed") from exc
    if not attached:
        raise ChromeSessionContractError("Chrome sandbox attachment failed")

    async def delete() -> None:
        await manager.delete_sandbox(CHROME_SANDBOX_LIFECYCLE_TYPE, scope_digest)

    return ChromeSandboxHandle(sandbox=provider.primary_sandbox, delete=delete)


def get_chrome_execution_session_pool() -> ChromeExecutionSessionPool:
    """Return the process-local coordinator for sandbox-owned Chrome sessions."""

    global _chrome_pool, _chrome_pool_manager
    manager = get_sandbox_manager()
    if manager is None:
        raise ChromeSessionContractError("Chrome sandbox is unavailable")
    if _chrome_pool is not None and _chrome_pool_manager is not manager:
        raise ChromeSessionContractError("Chrome sandbox manager changed")
    if _chrome_pool is None:
        _chrome_pool = ChromeExecutionSessionPool(_create_chrome_sandbox)
        _chrome_pool_manager = manager
    return _chrome_pool


async def shutdown_chrome_execution_session_pool() -> None:
    """Drain the process-local pool before the sandbox manager is stopped."""

    global _chrome_pool, _chrome_pool_manager
    pool = _chrome_pool
    try:
        if pool is not None:
            await pool.close_all()
    finally:
        _chrome_pool = None
        _chrome_pool_manager = None


async def consume_chrome_actor_stdio_session(
    *,
    server_name: str,
    connection: Mapping[str, Any],
    session_identity: ActorMCPStdioSessionIdentity,
    sandbox: object | None,
) -> list[Any]:
    """Consume one host-only identity without using the generic MCP loader."""

    del sandbox  # Chrome always owns a dedicated sandbox; there is no fallback.
    scope, executable_connection = bind_chrome_execution_scope(
        server_name, connection, session_identity
    )
    from ...core.tools.adapters.vibe.mcp_adapter import (
        load_execution_scoped_chrome_tools,
    )

    result = await load_execution_scoped_chrome_tools(
        server_name,
        cast(Connection, executable_connection),
        scope=scope,
    )
    if result.adapter_error_types or not result.tools:
        raise ChromeSessionContractError("Chrome session tools are unavailable")
    return list(result.tools)
