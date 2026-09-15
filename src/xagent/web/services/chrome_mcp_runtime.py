"""Authorization, identity, and dedicated-sandbox binding for Chrome MCP."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from typing import Any, cast
from uuid import UUID

from ...core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    CHROME_BACKEND_OPERATION_TIMEOUT_SECONDS,
    CHROME_DEVTOOLS_APP_ID,
    CHROME_SANDBOX_LIFECYCLE_TYPE,
    ChromeExecutionFence,
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
from .chrome_lifecycle import ChromeLifecycleCoordinator
from .db_runtime import await_task_settlement
from .mcp_runtime import CALLER_ID_ENV_VAR, MCPActorExecutionIdentity

_CHROME_SCOPE_HASH_DOMAIN = b"xagent.chrome.execution-session.v1\x00"
_chrome_pool_manager: object | None = None
_chrome_pool: ChromeExecutionSessionPool | None = None
_chrome_lifecycle_coordinator: ChromeLifecycleCoordinator | None = None

logger = logging.getLogger(__name__)


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
        ChromeExecutionScope(
            key=key,
            digest=_hash_identity_key(key),
            execution=ChromeExecutionFence(
                task_id=identity.execution.task_id,
                run_id=identity.execution.run_id,
                lease_attempt_id=identity.execution.lease_attempt_id,
                turn_id=identity.execution.turn_id,
            ),
        ),
        executable_connection,
    )


def get_chrome_lifecycle_coordinator() -> ChromeLifecycleCoordinator:
    """Return the lifecycle coordinator bound to the current sandbox manager."""

    global _chrome_lifecycle_coordinator
    manager = get_sandbox_manager()
    if manager is None:
        raise ChromeSessionContractError("Chrome sandbox is unavailable")
    coordinator = _chrome_lifecycle_coordinator
    if coordinator is not None and coordinator.backend is not manager:
        raise ChromeSessionContractError("Chrome lifecycle coordinator changed")
    if coordinator is None:
        coordinator = ChromeLifecycleCoordinator.production(manager)
        _chrome_lifecycle_coordinator = coordinator
    return coordinator


async def _create_chrome_sandbox(scope: ChromeExecutionScope) -> ChromeSandboxHandle:
    """Create and actively pin one sandbox whose name contains only a digest."""

    manager = get_sandbox_manager()
    if manager is None:
        raise ChromeSessionContractError("Chrome sandbox is unavailable")
    execution = scope.execution
    if execution is None:
        raise ChromeSessionContractError(
            "Chrome durable execution fence is unavailable"
        )
    coordinator = get_chrome_lifecycle_coordinator()
    try:
        lifecycle = await coordinator.register(
            scope_digest=scope.digest,
            task_id=execution.task_id,
            run_id=execution.run_id,
            lease_attempt_id=execution.lease_attempt_id,
            turn_id=execution.turn_id,
        )
    except Exception as exc:
        raise ChromeSessionContractError(
            "Chrome lifecycle registration failed"
        ) from exc

    async def compensate(*, uncertain_create: bool = False) -> None:
        try:
            if uncertain_create:
                await lifecycle.defer_unknown_create()
            else:
                await lifecycle.delete()
        except Exception:
            logger.error(
                "Chrome lifecycle compensation failed for backend %s",
                lifecycle.backend_lifecycle_digest,
                exc_info=True,
            )

    async def drain_compensation(*, uncertain_create: bool = False) -> None:
        cleanup = asyncio.create_task(compensate(uncertain_create=uncertain_create))
        _, cancellation = await await_task_settlement(cleanup)
        if cancellation is not None:
            raise cancellation

    try:
        provider = await asyncio.wait_for(
            manager.get_or_create_lease_provider(
                CHROME_SANDBOX_LIFECYCLE_TYPE,
                lifecycle.backend_lifecycle_digest,
            ),
            timeout=CHROME_BACKEND_OPERATION_TIMEOUT_SECONDS,
        )
    except SandboxCapacityError as exc:
        await drain_compensation()
        raise ChromeSessionContractError("Chrome sandbox is unavailable") from exc
    except BaseException as exc:
        try:
            await drain_compensation(uncertain_create=True)
        except asyncio.CancelledError as cancellation:
            raise cancellation from exc
        if not isinstance(exc, Exception):
            raise
        raise ChromeSessionContractError("Chrome sandbox creation failed") from exc

    try:
        attached = await manager.attach_provider(
            CHROME_SANDBOX_LIFECYCLE_TYPE,
            lifecycle.backend_lifecycle_digest,
            provider,
        )
    except BaseException as exc:
        try:
            await drain_compensation()
        except asyncio.CancelledError as cancellation:
            raise cancellation from exc
        if not isinstance(exc, Exception):
            raise
        raise ChromeSessionContractError("Chrome sandbox attachment failed") from exc
    if not attached:
        await drain_compensation()
        raise ChromeSessionContractError("Chrome sandbox attachment failed")

    try:
        await lifecycle.mark_ready()
    except BaseException as exc:
        try:
            await drain_compensation()
        except asyncio.CancelledError as cancellation:
            raise cancellation from exc
        if not isinstance(exc, Exception):
            raise
        raise ChromeSessionContractError("Chrome lifecycle ready CAS failed") from exc

    async def delete() -> None:
        await lifecycle.delete()

    return ChromeSandboxHandle(
        sandbox=provider.primary_sandbox,
        delete=delete,
        before_backend=lifecycle.renew,
    )


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

    global _chrome_lifecycle_coordinator, _chrome_pool, _chrome_pool_manager
    pool = _chrome_pool
    try:
        if pool is not None:
            await pool.close_all()
    finally:
        _chrome_pool = None
        _chrome_pool_manager = None
        _chrome_lifecycle_coordinator = None


async def start_chrome_lifecycle_recovery(app: Any) -> asyncio.Task[None] | None:
    """Run immediate recovery, then start this worker's periodic sweeper."""

    if get_sandbox_manager() is None:
        app.state.chrome_lifecycle_recovery_task = None
        return None
    existing = cast(
        "asyncio.Task[None] | None",
        getattr(app.state, "chrome_lifecycle_recovery_task", None),
    )
    if existing is not None and not existing.done():
        return existing
    coordinator = get_chrome_lifecycle_coordinator()

    async def recover() -> None:
        try:
            await coordinator.sweep_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Chrome remains hidden/default-off. Recovery is retryable and must
            # not turn an optional connector outage into global startup loss.
            logger.warning("Initial durable Chrome recovery failed", exc_info=True)
        await coordinator.run_sweep_loop()

    # Schedule the first pass immediately without making application startup
    # wait for up to one full backend deadline per stale generation.
    task = asyncio.create_task(recover())
    app.state.chrome_lifecycle_recovery_task = task
    return task


async def stop_chrome_lifecycle_recovery(app: Any) -> None:
    """Stop the periodic sweeper before draining locally owned sessions."""

    task = getattr(app.state, "chrome_lifecycle_recovery_task", None)
    app.state.chrome_lifecycle_recovery_task = None
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.warning(
            "Durable Chrome recovery loop stopped after failure", exc_info=True
        )


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
