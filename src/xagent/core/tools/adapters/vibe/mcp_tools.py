"""MCP tools registration using @register_tool decorator."""

import logging
from typing import TYPE_CHECKING, Any, List

from .connector_runtime import ConnectorRuntimeError
from .factory import register_tool

if TYPE_CHECKING:
    from .config import BaseToolConfig

logger = logging.getLogger(__name__)


@register_tool(categories={"mcp"}, selection_gate="mcp")
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
    mcp_configs = await config.get_mcp_server_configs()
    if not mcp_configs:
        return []

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
                return []

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

        return await ToolFactory._create_mcp_tools_from_configs(
            mcp_configs,
            sandbox=config.get_sandbox(),
        )
    except ConnectorRuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Failed to create MCP tools: {e}")
        return []
