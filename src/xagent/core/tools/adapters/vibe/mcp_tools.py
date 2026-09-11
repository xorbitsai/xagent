"""MCP tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from ....utils.setup_metrics import mcp_setup
from .config import (
    MCPConfigLoadError,
    MCPToolLoadSummary,
    MCPUnavailableSummary,
    enforce_mcp_failure_policy,
)
from .connector_runtime import ConnectorRuntimeError
from .factory import register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


def _stable_server_names(values: Any) -> tuple[str, ...]:
    """Return exact-string server identities once, preserving input order."""
    from .selection_spec import normalize_mcp_server_name

    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = value if type(value) is str else "MCP server"
        key = normalize_mcp_server_name(name)
        if key not in seen:
            names.append(name)
            seen.add(key)
    return tuple(names)


def _select_config_load_failures(
    error: MCPConfigLoadError,
    spec: Any,
) -> tuple[MCPUnavailableSummary, ...]:
    """Apply MCP server selection before enforcing config-load failures."""
    if spec is None:
        return error.summaries

    scoped = spec.scoped_mcp_servers()
    if scoped == frozenset():
        return ()
    if scoped is None:
        return error.summaries

    from .selection_spec import normalize_mcp_server_name

    by_name = {
        normalize_mcp_server_name(summary.server_name): summary
        for summary in error.summaries
    }
    return tuple(
        by_name.get(
            name,
            MCPUnavailableSummary.from_values(name, "config_load_failed"),
        )
        for name in sorted(scoped)
    )


def _build_mcp_load_summary(
    mcp_configs: List[dict[str, Any]],
    tools: List[Any],
    *,
    requested_servers: tuple[str, ...] | None = None,
    forced_failure_reason: str | None = None,
) -> MCPToolLoadSummary:
    """Build the single safe summary owned by the selected MCP boundary."""
    from .mcp_adapter import UnavailableMCPTool
    from .selection_spec import normalize_mcp_server_name

    requested = (
        requested_servers
        if requested_servers is not None
        else _stable_server_names(config.get("name") for config in mcp_configs)
    )
    display_by_normalized = {
        normalize_mcp_server_name(name): name for name in requested
    }

    loaded: list[str] = []
    loaded_keys: set[str] = set()
    failures_by_key: dict[str, MCPUnavailableSummary] = {}
    successful_tool_count = 0

    for tool in tools:
        if isinstance(tool, UnavailableMCPTool):
            failure = MCPUnavailableSummary.from_values(
                tool.server_name,
                tool.unavailability_reason,
            )
            key = normalize_mcp_server_name(failure.server_name)
            failures_by_key.setdefault(key, failure)
            continue

        successful_tool_count += 1
        source_server = getattr(getattr(tool, "metadata", None), "source_server", None)
        if source_server is None:
            source_server = getattr(tool, "source_server", None)
        if type(source_server) is not str:
            continue
        key = normalize_mcp_server_name(source_server)
        name = display_by_normalized.get(key)
        if name is not None and key not in loaded_keys:
            loaded.append(name)
            loaded_keys.add(key)

    if forced_failure_reason is not None:
        for name in requested:
            key = normalize_mcp_server_name(name)
            failures_by_key.setdefault(
                key,
                MCPUnavailableSummary.from_values(name, forced_failure_reason),
            )

    failures: list[MCPUnavailableSummary] = []
    emitted_failure_keys: set[str] = set()
    for name in requested:
        key = normalize_mcp_server_name(name)
        requested_failure = failures_by_key.get(key)
        if requested_failure is None and key not in loaded_keys:
            requested_failure = MCPUnavailableSummary.from_values(
                name, "no_tools_returned"
            )
        if requested_failure is not None and key not in emitted_failure_keys:
            failures.append(requested_failure)
            emitted_failure_keys.add(key)

    for key, failure in failures_by_key.items():
        if key not in emitted_failure_keys:
            failures.append(failure)
            emitted_failure_keys.add(key)

    return MCPToolLoadSummary(
        requested_servers=requested,
        loaded_servers=tuple(loaded),
        failures=tuple(failures),
        successful_tool_count=successful_tool_count,
    )


async def _emit_mcp_load_summary(
    config: "BaseToolConfig", summary: MCPToolLoadSummary
) -> None:
    try:
        emitter = getattr(config, "emit_mcp_load_summary", None)
        if not callable(emitter):
            return
        await emitter(summary)
    except Exception as exc:
        logger.warning(
            "Failed to emit MCP load summary (%s)",
            type(exc).__name__,
        )


async def _finish_mcp_setup(
    config: "BaseToolConfig",
    summary: MCPToolLoadSummary,
    tools: List[Any],
) -> List[Any]:
    """Emit once before enforcing the caller-owned setup policy."""
    await _emit_mcp_load_summary(config, summary)
    enforce_mcp_failure_policy(config.get_mcp_failure_policy(), summary.failures)
    return tools


@register_tool(categories={"mcp"}, selection_gate="mcp")
@mcp_setup.measure()
async def create_mcp_tools(config: "BaseToolConfig") -> List[Any]:
    """Create MCP tools from configuration.

    Registry dispatch goes through ``selection_gate="mcp"`` ->
    ``spec.includes_mcp()`` (see ``ToolRegistry._should_run_creator``),
    not the plain category intersection: a ``mcp:<server>`` scope lands
    in ``mcp_servers`` only and leaves ``categories`` without ``"mcp"``,
    so a category-only gate would skip this creator for server-only specs.

    Internal short-circuit via ``ToolSelectionSpec.includes_mcp()``:
    when the spec excludes MCP this creator returns early WITHOUT calling
    ``config.get_mcp_server_configs()`` — that call goes through the
    MCP server scan / DB lookup / per-server session-initialize path
    which dominates the 25-30s setup window for tasks that don't
    actually want MCP tools (see issue #427). The check is redundant with
    the dispatch gate but kept as defense and to cover the spec=None path.
    """
    spec = (
        config.get_tool_selection_spec()
        if hasattr(config, "get_tool_selection_spec")
        else None
    )
    if spec is not None and not spec.includes_mcp():
        return []
    try:
        mcp_configs = await config.get_mcp_server_configs()
    except MCPConfigLoadError as error:
        failures = _select_config_load_failures(error, spec)
        if not failures:
            return []

        from .factory import ToolFactory

        tools = [
            ToolFactory._create_unavailable_mcp_tool(
                server_name=failure.server_name,
                reason=failure.reason,
                message="MCP server configuration is unavailable.",
            )
            for failure in failures
        ]
        summary = MCPToolLoadSummary(
            requested_servers=tuple(failure.server_name for failure in failures),
            failures=failures,
            successful_tool_count=0,
        )
        return await _finish_mcp_setup(config, summary, tools)
    requested_without_configs: tuple[str, ...] = ()
    if not mcp_configs and spec is not None:
        scoped = spec.scoped_mcp_servers()
        if scoped:
            requested_without_configs = _stable_server_names(sorted(scoped))
    if not mcp_configs:
        summary = _build_mcp_load_summary(
            [], [], requested_servers=requested_without_configs
        )
        return await _finish_mcp_setup(config, summary, [])

    # Pre-build per-server restriction comes from the single policy method
    # ``spec.scoped_mcp_servers()`` so it stays consistent with the parent/
    # child rule ``compute_allowed_names`` applies post-build:
    #   - frozenset(): MCP not selected -> initialize nothing.
    #   - None: no restriction (plain "mcp" parent, or ALL) -> keep all.
    #   - non-empty: keep only those servers.
    # Dropping configs here matters because ``_create_mcp_tools_from_configs``
    # performs the actual session initialization (network I/O). The config
    # ``name`` is folded through the same ``normalize_mcp_server_name`` SSOT
    # as the scoped keys, so case / whitespace / hyphen never drop a server.
    if spec is not None:
        scoped = spec.scoped_mcp_servers()
        if scoped == frozenset():
            return []
        if scoped is not None:
            from .selection_spec import normalize_mcp_server_name

            mcp_configs = [
                cfg
                for cfg in mcp_configs
                if normalize_mcp_server_name(cfg.get("name", "")) in scoped
            ]
            if not mcp_configs:
                requested_servers = _stable_server_names(sorted(scoped))
                summary = _build_mcp_load_summary(
                    [], [], requested_servers=requested_servers
                )
                return await _finish_mcp_setup(config, summary, [])

    # Everything DB-backed is loaded at this point; what follows is pure
    # network I/O against remote MCP servers (initialize + list-tools, with
    # retries). Release the config session's pooled connection first so a
    # slow or hung server doesn't pin a pool slot in ``idle in transaction``
    # for the whole handshake (issue #889).
    release = getattr(config, "release_db_connection", None)
    if callable(release):
        release()

    try:
        from .factory import ToolFactory

        tools = await ToolFactory._create_mcp_tools_from_configs(
            mcp_configs,
            sandbox=config.get_sandbox(),
        )
    except ConnectorRuntimeError:
        summary = _build_mcp_load_summary(
            mcp_configs,
            [],
            forced_failure_reason="runtime_connection_failed",
        )
        await _emit_mcp_load_summary(config, summary)
        raise
    except Exception as e:
        logger.warning("Failed to create MCP tools (%s)", type(e).__name__)
        tools = []
        summary = _build_mcp_load_summary(
            mcp_configs,
            tools,
            forced_failure_reason="loader_failed",
        )
    else:
        summary = _build_mcp_load_summary(mcp_configs, tools)

    return await _finish_mcp_setup(config, summary, tools)
