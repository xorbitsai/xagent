"""Start one connected MCP server the way an agent task does, outside a task.

A task builds each connector's connection through ``WebToolConfig``: a builtin
OAuth row becomes a stdio launch with the user's token injected (refreshed when
it is about to expire), a stdio row gets the user/shared/env-source env layers,
and a remote ``mcp_oauth`` row gets the user's bearer grant. This module runs
that same builder for a single server and then lists the server's tools, so a
caller outside a task (the tools route, the connector smoke checks) sees what a
task would see instead of a parallel, partial reconstruction of it. The
GET /api/mcp/servers/{id}/tools route and the connector probe use it.

A caller that checks a connector the way an actor-scoped task loads it (the
Toby connector probe, #3288) passes the task's actor policy, actor stdio
adapter, execution identity and governing team through unchanged. An actor
stdio connection has no ``MCPServer`` row, so it is started by catalog
``app_id`` through ``list_actor_stdio_connector_tools`` instead.
"""

from __future__ import annotations

import logging
import shlex
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Mapping, cast

if TYPE_CHECKING:
    from ...core.tools.adapters.vibe.mcp_adapter import MCPLoadResult
    from ...core.tools.core.mcp.sessions import Connection
    from ..tools.config import WebToolConfig
    from .actor_mcp_runtime import ActorMCPStdioConnectionAdapter
    from .mcp_runtime import MCPActorAuthorizationPolicy, MCPActorExecutionIdentity

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
    the loaded tools and is kept out of ``repr``. ``server_id`` is None for
    an actor stdio connection, which has no server row; its ``server_name``
    is the catalog app id.
    """

    outcome: ConnectorToolsOutcome
    server_id: int | None
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
    refuse_sandboxed: bool = False,
    mcp_runtime_authorization_policy: MCPActorAuthorizationPolicy | None = None,
    mcp_actor_stdio_connection_adapter: ActorMCPStdioConnectionAdapter | None = None,
    mcp_actor_execution_identity: MCPActorExecutionIdentity | None = None,
    connector_team_id: int | None = None,
) -> ConnectorToolsResult:
    """Start ``server_id`` for ``user`` as a task would and list its tools.

    Only a server visible to the user is started: one the user has an active
    personal link to or, when ``connector_team_id`` names the governing
    agent's team, one that team owns, the same visibility the task's config
    uses. Tool names get ``name_prefix``; the persisted server id travels as
    the connector ref, as at every other MCP materialization seam, so a gated
    call built from these tools can never fail closed for want of an
    identity.

    The actor keywords are handed to ``WebToolConfig`` unchanged, so a
    builtin OAuth row reads the policy owner's credential, a remote
    ``mcp_oauth`` row prefers the owner's grant, and the policy's catalog
    checks apply. Every default keeps the behaviour of a call without them.
    A failure to resolve the team scope raises the config's typed
    ``ConnectorRuntimeError``, as a task's would.

    ``connector_team_id`` and the actor policy, adapter and identity carry
    authorization meaning and are trusted as given: they must come from an
    agent or turn the caller has already authorized ``user`` to run (the
    governing agent's owning team, never request input), as ``WebToolConfig``
    documents for ``connector_team_id``. The team hook does no membership
    check.

    The sandbox lease is released when this returns, so a caller that goes on
    to call a tool would open a new session on the web host. Such a caller
    passes ``refuse_sandboxed``: an effective connection a task would sandbox
    (``npx``/``uvx``) is refused whether or not a sandbox manager is
    configured, with UNSUPPORTED and reason ``sandboxed``. The refusal comes
    after the task's config build (database lookups, env decryption and, for
    an OAuth row, a possible token refresh) but before any sandbox lease,
    process spawn or loader call.
    """
    from ...core.tools.adapters.vibe.config import MCPConfigLoadError

    user_id = int(user.id)
    server = find_visible_server(
        db,
        user_id=user_id,
        server_id=server_id,
        connector_team_id=connector_team_id,
    )
    if server is None:
        return ConnectorToolsResult(ConnectorToolsOutcome.NOT_FOUND, server_id)
    server_name = server.name

    def result(outcome: ConnectorToolsOutcome, **fields: Any) -> ConnectorToolsResult:
        return ConnectorToolsResult(outcome, server_id, server_name, **fields)

    if _is_execution_scoped(db, server):
        return result(ConnectorToolsOutcome.UNSUPPORTED, reason="execution_scoped")

    tool_config = _task_tool_config(
        db,
        user,
        mcp_runtime_authorization_policy=mcp_runtime_authorization_policy,
        mcp_actor_stdio_connection_adapter=mcp_actor_stdio_connection_adapter,
        mcp_actor_execution_identity=mcp_actor_execution_identity,
        connector_team_id=connector_team_id,
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

    return await _start_and_list(
        tool_config,
        server_config,
        user_id=user_id,
        server_name=server_name,
        name_prefix=name_prefix,
        refuse_sandboxed=refuse_sandboxed,
        result=result,
    )


async def list_actor_stdio_connector_tools(
    db: Any,
    *,
    user: Any,
    app_id: str,
    name_prefix: str,
    mcp_runtime_authorization_policy: MCPActorAuthorizationPolicy,
    mcp_actor_stdio_connection_adapter: ActorMCPStdioConnectionAdapter,
    mcp_actor_execution_identity: MCPActorExecutionIdentity | None = None,
    connector_team_id: int | None = None,
    refuse_sandboxed: bool = False,
) -> ConnectorToolsResult:
    """Start the actor stdio connection for catalog ``app_id`` as a task would.

    An actor stdio connection has no ``MCPServer`` row, so it is addressed by
    its catalog app id. The connection is the one the task's config
    synthesizes for the same policy and adapter
    (``resolve_actor_mcp_stdio_configs``), checked against the user's whole
    visible server set, as a task checks it, so a visible row that shadows
    the app suppresses it here too. It carries no connector ref, as a task
    gives none to a config without a server id.

    An execution-scoped app (chrome-devtools) needs a task's session runtime,
    so it is UNSUPPORTED before anything else is checked, whether or not the
    actor has connected it. Otherwise NOT_FOUND covers every case in which a
    task would silently leave the connection out: the policy does not allow
    builtin stdio, the deployment flag is off, the actor has no connection
    for the app, a visible row shadows it, or the resolver dropped it because
    its storage read failed, its credentials were incomplete, or its
    lifecycle generation was stale. NOT_FOUND therefore does not prove the
    actor never connected the app. A failure to resolve the team scope
    raises the config's typed ``ConnectorRuntimeError``, as in
    ``list_connector_tools``; ``refuse_sandboxed`` behaves as it does there.

    The team id, policy and adapter are trusted as given, under the same
    contract as ``list_connector_tools``: they must come from an agent or
    turn the caller has already authorized ``user`` to run, and the team hook
    does no membership check. ``mcp_actor_execution_identity`` has no effect
    here: execution-scoped apps are UNSUPPORTED before resolution, and the
    resolver reads the identity only for those apps. It is accepted for
    signature parity with ``list_connector_tools``.
    """
    from ..builtin_mcp_registry import get_builtin_stdio_session_scope
    from .actor_mcp_runtime import resolve_actor_mcp_stdio_configs

    user_id = int(user.id)

    def result(outcome: ConnectorToolsOutcome, **fields: Any) -> ConnectorToolsResult:
        return ConnectorToolsResult(outcome, None, app_id, **fields)

    if get_builtin_stdio_session_scope(app_id) == "execution":
        return result(ConnectorToolsOutcome.UNSUPPORTED, reason="execution_scoped")

    # The team scope is resolved outside the guard, and its typed error
    # propagates, as in the task's _load_mcp_server_configs; a scan failure
    # inside it is the task's config_load_failed.
    visible_query = _visible_server_query(
        db, user_id=user_id, connector_team_id=connector_team_id
    )
    try:
        resolution = resolve_actor_mcp_stdio_configs(
            db,
            user_id=user_id,
            policy=mcp_runtime_authorization_policy,
            adapter=mcp_actor_stdio_connection_adapter,
            visible_servers=visible_query.all(),
            execution_identity=mcp_actor_execution_identity,
        )
    except Exception as error:
        logger.warning(
            "Failed to scan actor stdio configs with %s", type(error).__name__
        )
        return result(
            ConnectorToolsOutcome.FAILED_TO_START, reason="config_load_failed"
        )
    server_config = next(
        (config for config in resolution.configs if config.get("name") == app_id),
        None,
    )
    if server_config is None:
        return result(ConnectorToolsOutcome.NOT_FOUND)

    tool_config = _task_tool_config(
        db,
        user,
        mcp_runtime_authorization_policy=mcp_runtime_authorization_policy,
        mcp_actor_stdio_connection_adapter=mcp_actor_stdio_connection_adapter,
        mcp_actor_execution_identity=mcp_actor_execution_identity,
        connector_team_id=connector_team_id,
    )
    return await _start_and_list(
        tool_config,
        server_config,
        user_id=user_id,
        server_name=app_id,
        name_prefix=name_prefix,
        refuse_sandboxed=refuse_sandboxed,
        result=result,
    )


def find_visible_server(
    db: Any, *, user_id: int, server_id: int, connector_team_id: int | None
) -> Any | None:
    """The ``MCPServer`` row ``server_id`` if the task's config would see it.

    Visibility is the config's own: an active personal link, or ownership by
    the governing team ``connector_team_id``. With no team id this is exactly
    the personal link check.
    """
    from ..models.mcp import MCPServer

    return (
        _visible_server_query(db, user_id=user_id, connector_team_id=connector_team_id)
        .filter(MCPServer.id == server_id)
        .first()
    )


def _visible_server_query(
    db: Any, *, user_id: int, connector_team_id: int | None
) -> Any:
    """The config's own ``_visible_mcp_server_query``, run with the team ids
    resolved by the same function ``_load_mcp_server_configs`` uses, so the
    two cannot drift."""
    from ..tools.config import WebToolConfig
    from .connector_team_scope import resolve_team_connector_selection_or_raise

    team_selection = resolve_team_connector_selection_or_raise(
        db, team_id=connector_team_id, log_subject=user_id
    )
    return WebToolConfig(
        db=db, request=None, user_id=user_id
    )._visible_mcp_server_query(team_selection.mcp_ids)


def _task_tool_config(
    db: Any,
    user: Any,
    *,
    mcp_runtime_authorization_policy: MCPActorAuthorizationPolicy | None,
    mcp_actor_stdio_connection_adapter: ActorMCPStdioConnectionAdapter | None,
    mcp_actor_execution_identity: MCPActorExecutionIdentity | None,
    connector_team_id: int | None,
) -> WebToolConfig:
    from ..tools.config import WebToolConfig

    return WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=int(user.id),
        is_admin=bool(getattr(user, "is_admin", False)),
        mcp_runtime_authorization_policy=mcp_runtime_authorization_policy,
        mcp_actor_stdio_connection_adapter=mcp_actor_stdio_connection_adapter,
        mcp_actor_execution_identity=mcp_actor_execution_identity,
        connector_team_id=connector_team_id,
    )


async def _start_and_list(
    tool_config: WebToolConfig,
    server_config: dict[str, Any],
    *,
    user_id: int,
    server_name: str,
    name_prefix: str,
    refuse_sandboxed: bool,
    result: Callable[..., ConnectorToolsResult],
) -> ConnectorToolsResult:
    """Start one built task config and list its tools, as a task would."""
    from ...core.tools.adapters.vibe import mcp_adapter
    from ...core.tools.adapters.vibe.factory import ToolFactory
    from ...core.tools.adapters.vibe.mcp_tools import _apply_stdio_output_limit_env
    from ...core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
        should_sandbox_mcp_connection,
    )

    (server_config,) = _apply_stdio_output_limit_env([server_config], tool_config)
    connection = _connection_from_config(server_config)
    if refuse_sandboxed and should_sandbox_mcp_connection(
        cast("Connection", connection)
    ):
        return result(ConnectorToolsOutcome.UNSUPPORTED, reason="sandboxed")
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
                connector_refs=ToolFactory._mcp_connector_refs(
                    {server_name: server_config}
                ),
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
        "signed_uploads",
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

    A task prefers the sandbox for ``npx``/``uvx`` stdio connections when a
    manager is configured. Custom connections explicitly marked
    ``sandbox_required`` fail closed instead of running on the host.
    """
    from ...core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
        requires_sandbox_mcp_connection,
        should_sandbox_mcp_connection,
    )
    from ..sandbox_manager import get_sandbox_manager

    sandbox_required = should_sandbox_mcp_connection(connection)
    fail_closed = requires_sandbox_mcp_connection(connection)
    manager = get_sandbox_manager() if sandbox_required else None
    if manager is None:
        if fail_closed:
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
