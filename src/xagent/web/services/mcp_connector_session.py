"""Start one connected MCP server the way an agent task does, outside a task.

A task builds each connector's connection through ``WebToolConfig``: a builtin
OAuth row becomes a stdio launch with the user's token injected (refreshed when
it is about to expire), a stdio row gets the user/shared/env-source env layers,
and a remote ``mcp_oauth`` row gets the user's bearer grant. This module runs
that same builder for a single server and then lists the server's tools, so a
caller outside a task (the tools route, the connector smoke checks) sees what a
task would see instead of a parallel, partial reconstruction of it. The
GET /api/mcp/servers/{id}/tools route uses it today; the connector smoke
checks (#3246, #3247) are its intended next callers.
"""

from __future__ import annotations

import logging
import shlex
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Mapping

if TYPE_CHECKING:
    from ...core.tools.adapters.vibe.mcp_adapter import MCPLoadResult

logger = logging.getLogger(__name__)


class ConnectorToolsOutcome(str, Enum):
    OK = "ok"
    NOT_FOUND = "not_found"
    NOT_CONNECTED = "not_connected"
    UNSUPPORTED = "unsupported"
    FAILED_TO_START = "failed_to_start"
    NO_TOOLS = "no_tools"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class ConnectorToolsResult:
    """Outcome of starting one connector and listing its tools.

    ``reason`` is a public-safe code: the task builder's unavailable reason, or
    the load failure phase. ``diagnostic`` is the remote MCP OAuth diagnostic
    the builder produced, which carries no credential. ``load_result`` holds
    the loaded tools and is kept out of ``repr``.
    """

    outcome: ConnectorToolsOutcome
    server_id: int
    server_name: str | None = None
    reason: str | None = None
    diagnostic: Mapping[str, Any] | None = None
    load_result: MCPLoadResult | None = field(default=None, repr=False, compare=False)


# Unavailable reasons from the task builder that mean the user's credential is
# missing or unusable, as opposed to a broken server definition.
_CREDENTIAL_REASONS = frozenset(
    {
        "authorization_required",
        "insufficient_scope",
        "oauth_token_refresh_failed",
        "oauth_token_required",
        "oauth_token_resolver_failed",
        "token_refresh_failed",
    }
)

# The same lease lifecycle the tool listing route (api/tools.py) uses.
_SANDBOX_LIFECYCLE = "tools"


class _SandboxUnavailable(Exception):
    pass


async def list_connector_tools(
    db: Any,
    *,
    user: Any,
    server_id: int,
    name_prefix: str,
) -> ConnectorToolsResult:
    """Start ``server_id`` for ``user`` as a task would and list its tools.

    Only a server the user has an active personal link to is started. Tool
    names get ``name_prefix``; the persisted server id travels as the
    connector ref, as at every other MCP materialization seam, so a gated
    call built from these tools can never fail closed for want of an
    identity.
    """
    from ...core.tools.adapters.vibe import mcp_adapter
    from ...core.tools.adapters.vibe.config import MCPConfigLoadError
    from ...core.tools.adapters.vibe.connector_runtime import ConnectorRef
    from ...core.tools.adapters.vibe.mcp_tools import _apply_stdio_output_limit_env
    from ..models.mcp import MCPServer, UserMCPServer
    from ..tools.config import WebToolConfig

    user_id = int(user.id)
    server = (
        db.query(MCPServer)
        .join(UserMCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(
            UserMCPServer.user_id == user_id,
            UserMCPServer.is_active,
            MCPServer.id == server_id,
        )
        .first()
    )
    if server is None:
        return ConnectorToolsResult(ConnectorToolsOutcome.NOT_FOUND, server_id)
    server_name = server.name

    def result(outcome: ConnectorToolsOutcome, **fields: Any) -> ConnectorToolsResult:
        return ConnectorToolsResult(outcome, server_id, server_name, **fields)

    if _is_execution_scoped(db, server):
        return result(ConnectorToolsOutcome.UNSUPPORTED, reason="execution_scoped")

    tool_config = WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=user_id,
        is_admin=bool(getattr(user, "is_admin", False)),
    )
    try:
        server_config = await tool_config.load_mcp_server_config(server_id)
    except MCPConfigLoadError:
        return result(
            ConnectorToolsOutcome.FAILED_TO_START, reason="config_load_failed"
        )
    if server_config is None:
        return result(ConnectorToolsOutcome.NOT_FOUND)

    inner_config = server_config.get("config")
    if not isinstance(inner_config, dict):
        return result(ConnectorToolsOutcome.FAILED_TO_START, reason="invalid_config")
    if inner_config.get("unavailable"):
        reason = inner_config.get("reason")
        return result(
            ConnectorToolsOutcome.NOT_CONNECTED
            if reason in _CREDENTIAL_REASONS
            else ConnectorToolsOutcome.FAILED_TO_START,
            reason=reason,
            diagnostic=inner_config.get("diagnostic"),
        )

    (server_config,) = _apply_stdio_output_limit_env([server_config], tool_config)
    connection = _connection_from_config(server_config)
    # What follows is network and process I/O: do not hold a pooled
    # connection across it, as create_mcp_tools does not (issue #889).
    tool_config.release_db_connection()

    try:
        async with _task_sandbox(connection, user_id) as sandbox:
            load_kwargs: dict[str, Any] = {}
            if sandbox is not None:
                load_kwargs["sandbox"] = sandbox
            # Typed like ToolFactory's own connection map (factory.py).
            connections: dict[str, Any] = {server_name: connection}
            load_result = await mcp_adapter.load_mcp_tools_as_agent_tools(
                connections,
                name_prefix=name_prefix,
                connector_refs={server_name: ConnectorRef("mcp", int(server_id))},
                **load_kwargs,
            )
    except _SandboxUnavailable:
        return result(
            ConnectorToolsOutcome.FAILED_TO_START, reason="sandbox_unavailable"
        )

    outcome, reason = _classify_load(load_result)
    return result(outcome, reason=reason, load_result=load_result)


def _is_execution_scoped(db: Any, server: Any) -> bool:
    """Whether the server's canonical runtime needs a task's execution identity
    (chrome-devtools), which a caller outside a task cannot provide."""
    from ..builtin_mcp_registry import get_builtin_stdio_session_scope
    from ..mcp_apps import get_app_for_mcp_server

    app = get_app_for_mcp_server(db, server)
    app_id = app.get("id") if app else None
    return (
        isinstance(app_id, str)
        and get_builtin_stdio_session_scope(app_id) == "execution"
    )


def _connection_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Turn one task server config into the loader's connection mapping.

    Mirrors the per-config conversion in
    ``ToolFactory._create_mcp_tools_from_configs`` (factory.py); the parity
    tests in tests/web/services/test_mcp_connector_session.py hold the two
    together.
    """
    connection: dict[str, Any] = {
        "transport": config["transport"],
        **config["config"],
    }
    for runtime_key in (
        "runtime_bindings",
        "runtime_input_schema",
        "connector_runtime",
        "allow_delegated_authorization",
    ):
        if runtime_key in config:
            connection[runtime_key] = config[runtime_key]
    args = connection.get("args")
    if isinstance(args, str):
        try:
            connection["args"] = shlex.split(args)
        except ValueError:
            connection["args"] = args.split()
    return connection


@asynccontextmanager
async def _task_sandbox(connection: Any, user_id: int) -> AsyncIterator[Any]:
    """Yield the sandbox a task would run this connection in, or None.

    A task sandboxes only ``npx``/``uvx`` stdio connections, and only when a
    sandbox manager is configured; every other connection runs on the host, as
    it does in a task. When a sandbox is required but cannot be leased, this
    fails closed instead of running the package on the host.
    """
    from ...core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
        should_sandbox_mcp_connection,
    )
    from ..sandbox_manager import get_sandbox_manager

    sandbox_required = should_sandbox_mcp_connection(connection)
    manager = get_sandbox_manager() if sandbox_required else None
    if manager is None:
        if sandbox_required:
            raise _SandboxUnavailable()
        yield None
        return

    lease_user_id = str(user_id)
    try:
        sandbox = await manager.get_or_create_lease_provider(
            _SANDBOX_LIFECYCLE, lease_user_id
        )
        attached = await manager.attach_provider(
            _SANDBOX_LIFECYCLE, lease_user_id, sandbox
        )
    except Exception as error:
        logger.warning(
            "Could not lease a sandbox to start an MCP server (%s)",
            type(error).__name__,
        )
        raise _SandboxUnavailable() from error
    if not attached:
        raise _SandboxUnavailable()
    try:
        yield sandbox
    finally:
        await manager.release(_SANDBOX_LIFECYCLE, lease_user_id)


def _classify_load(
    load_result: MCPLoadResult,
) -> tuple[ConnectorToolsOutcome, str | None]:
    from ...core.tools.adapters.vibe.mcp_adapter import MCPFailurePhase

    if load_result.tools:
        return ConnectorToolsOutcome.OK, None
    failures = tuple(load_result.failures)
    if not failures:
        return ConnectorToolsOutcome.NO_TOOLS, MCPFailurePhase.NO_TOOLS_RETURNED.value
    first = failures[0]
    reason = first.phase.value
    # The loader's wall-clock bound surfaces as TimeoutError
    # (mcp_adapter._load_server_tools_bounded).
    if any(failure.error_type == "TimeoutError" for failure in failures):
        return ConnectorToolsOutcome.TIMED_OUT, reason
    if first.phase is MCPFailurePhase.NO_TOOLS_RETURNED:
        return ConnectorToolsOutcome.NO_TOOLS, reason
    if first.phase is MCPFailurePhase.AUTH_REJECTED:
        return ConnectorToolsOutcome.NOT_CONNECTED, reason
    return ConnectorToolsOutcome.FAILED_TO_START, reason
