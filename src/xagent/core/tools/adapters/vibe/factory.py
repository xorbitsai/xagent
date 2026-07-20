"""
Tool Factory for xagent

Provides a unified interface for creating tools with proper workspace binding
and configuration management.
"""

# mypy: ignore-errors

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Tuple,
)

from sqlalchemy.orm import Session

from .....config import get_uploads_dir
from .....core.workspace import TaskWorkspace
from .base import AbstractBaseTool, Tool
from .config import (
    BaseToolConfig,
    MCPFailurePolicy,
    MCPUnavailableSummary,
    RequiredMCPUnavailableError,
    enforce_mcp_failure_policy,
    normalize_tool_allowlist,
)
from .connector_runtime import ConnectorRuntimeError
from .output_filter_wrapper import OutputFilteredToolWrapper
from .selection_spec import ToolSelectionSpec

if TYPE_CHECKING:
    from .....sandbox.base import Sandbox
    from .mcp_adapter import MCPServerLoadFailure

logger = logging.getLogger(__name__)

__all__ = ["ToolFactory", "ToolRegistry", "register_tool"]


class ToolRegistry:
    """
    Global registry for tool creators using decorator pattern.

    Tools are registered using @register_tool decorator and automatically
    discovered during create_all_tools().

    Each registration may declare the tool ``categories`` it produces so
    that ``create_registered_tools`` can skip the creator entirely when
    a :class:`ToolSelectionSpec` excludes those categories. Creators that
    produce tools across multiple categories or that produce categories
    dynamically (MCP / Custom API) should leave ``categories`` unset and
    short-circuit internally based on the spec. Published-agent
    delegation uses a creator-specific dispatch because workforce worker
    tools can be injected by exact name without enabling the whole
    ``agent`` category.
    """

    # (creator, declared_categories, selection_gate) — declared_categories is
    # None for dynamic creators that filter internally based on the spec.
    _tool_creators: List[Tuple[Callable, Optional[FrozenSet[str]], Optional[str]]] = []
    _modules_imported = False

    @classmethod
    def register(
        cls,
        creator: Optional[Callable] = None,
        *,
        categories: Optional[set] = None,
        selection_gate: Optional[str] = None,
    ) -> Callable:
        """
        Register a tool creator function.

        The creator function will be called during create_all_tools()
        with the current config.

        Usage (bare decorator, no category metadata):
            @register_tool
            def create_my_tools(config: BaseToolConfig) -> List[Tool]:
                return [MyTool(...)]

        Usage (with categories — registry can skip this creator when a
        ToolSelectionSpec excludes all declared categories):
            @register_tool(categories={"basic"})
            def create_basic_tools(config: BaseToolConfig) -> List[Tool]:
                return [BasicTool(...)]

        Usage (with a creator-specific selection gate):
            @register_tool(categories={"agent"}, selection_gate="published_agent")
            def create_agent_tools(config: BaseToolConfig) -> List[Tool]:
                return get_published_agents_tools(...)
        """
        declared = frozenset(categories) if categories else None

        def _do_register(fn: Callable) -> Callable:
            cls._tool_creators.append((fn, declared, selection_gate))
            return fn

        # Bare form: ``@register_tool`` (no parens) — ``creator`` is the
        # decorated callable; apply immediately.
        if creator is not None:
            return _do_register(creator)
        # Parameterized form: ``@register_tool(categories=...)`` —
        # ``creator`` is None; return the actual decorator.
        return _do_register

    @classmethod
    def _import_tool_modules(cls):
        """Import tool modules to trigger @register_tool decorator registration."""
        if cls._modules_imported:
            return

        try:
            # Import tool modules in priority order - these imports trigger @register_tool decorators
            from . import (  # noqa: F401 - imports trigger @register_tool decorators
                a2a_agent_tool,
                agent_tool,
                ask_user_tool,
                audio_tool,
                basic_tools,
                browser_tools,
                custom_api_factory,
                file_ingestion_tool,
                image_tool,
                knowledge_tools,
                mcp_tools,
                music_tool,
                pptx_tool,
                skill_tools,
                sound_effect_tool,
                sql_tool,
                translate_json,
                video_tool,
                vision_tool,
                web_ingestion_tool,
                workspace_file_tool,
            )

            cls._modules_imported = True
            logger.info("Tool modules imported and registered")
        except Exception as e:
            logger.warning(f"Failed to import tool modules: {e}")

    @staticmethod
    def _should_run_creator(
        declared_cats: Optional[FrozenSet[str]],
        spec: Optional[ToolSelectionSpec],
        selection_gate: Optional[str],
    ) -> bool:
        if spec is None or declared_cats is None or spec.categories is None:
            return True

        if selection_gate == "published_agent":
            return spec.includes_published_agent()

        if selection_gate == "mcp":
            # ``mcp:<server>`` scopes land in ``mcp_servers`` only, leaving
            # ``categories`` without ``"mcp"``. Dispatch must read the spec's
            # own MCP predicate (which honors both the plain ``"mcp"`` category
            # and a server scope) rather than the category intersection below,
            # or a server-only spec would skip the MCP creator entirely.
            return spec.includes_mcp()

        if declared_cats & spec.categories:
            return True

        return False

    @classmethod
    async def create_registered_tools(cls, config: BaseToolConfig) -> List[Tool]:
        """Create tools from all registered creators.

        When ``config.get_tool_selection_spec()`` returns a spec,
        creators whose declared categories don't intersect
        ``spec.categories`` are skipped at the registry level (no
        creator call, no I/O). Creators with no declared categories
        (dynamic ones: MCP / Custom API / Image / Audio) are always
        dispatched and are responsible for
        short-circuiting internally based on the spec.
        """
        # Import tool modules on first call to trigger decorator registration
        cls._import_tool_modules()

        spec: Optional[ToolSelectionSpec] = (
            config.get_tool_selection_spec()
            if hasattr(config, "get_tool_selection_spec")
            else None
        )
        tools: List[Tool] = []
        for creator, declared_cats, selection_gate in cls._tool_creators:
            # Registry-level skip: declared categories known and no
            # intersection with the spec's allowed categories. The helper
            # keeps the published-agent workforce exception in one place.
            if not cls._should_run_creator(declared_cats, spec, selection_gate):
                continue
            try:
                created_tools = await creator(config)
                tools.extend(created_tools)
            except ConnectorRuntimeError:
                raise
            except RequiredMCPUnavailableError:
                raise
            except Exception as e:
                logger.warning(f"Tool creator {creator.__name__} failed: {e}")

        # Sort tools by category priority
        tools = cls._sort_tools_by_category(tools)
        return tools

    @classmethod
    def _sort_tools_by_category(cls, tools: List[Tool]) -> List[Tool]:
        """Sort tools by category priority.

        Priority order (most important first):
        1. BASIC - Basic tools (code execution, calculator)
        2. WEB_SEARCH - Web search and webpage fetching
        3. KNOWLEDGE - Knowledge base search
        4. FILE - File operations
        5. VISION - Vision understanding
        6. IMAGE - Image generation
        7. VIDEO - Video generation
        8. AUDIO - Speech and sound effect tools
        9. BROWSER - Browser automation
        10. PPT - PPT tools
        11. DATABASE - Database tools (SQL query)
        12. MCP - MCP tools
        13. SKILL - Skill documentation access tools
        14. AGENT - Agent tools (delegation)
        15. OTHER - Other tools
        """
        from .base import ToolCategory

        # Define category priority order
        category_order = {
            ToolCategory.BASIC: 0,
            ToolCategory.WEB_SEARCH: 1,
            ToolCategory.KNOWLEDGE: 2,
            ToolCategory.FILE: 3,
            ToolCategory.VISION: 4,
            ToolCategory.IMAGE: 5,
            ToolCategory.VIDEO: 6,
            ToolCategory.AUDIO: 7,
            ToolCategory.BROWSER: 8,
            ToolCategory.PPT: 9,
            ToolCategory.DATABASE: 10,
            ToolCategory.MCP: 11,
            ToolCategory.SKILL: 12,
            ToolCategory.AGENT: 13,
            ToolCategory.OTHER: 14,
        }

        def get_tool_priority(tool: Tool) -> int:
            """Get priority for a tool based on its category."""
            tool_category = tool.metadata.category
            return category_order.get(tool_category, 99)

        return sorted(tools, key=get_tool_priority)


# Decorator for easy import
register_tool = ToolRegistry.register


class ToolFactory:
    """
    Unified tool factory that handles tool creation with proper workspace binding.

    Tool categories are self-describing - each tool declares its own category
    via the metadata.category field. No need for manual category mapping.
    """

    @staticmethod
    async def create_all_tools(
        config: BaseToolConfig, apply_user_override_filter: bool = True
    ) -> List[Tool]:
        """
        Create all tools based on configuration.

        This is the unified entry point for tool creation. All tools are discovered
        automatically via @register_tool decorators based on the provided configuration.

        Args:
            config: Tool configuration object
            apply_user_override_filter: If True (default), tools disabled by the
                per-user override hook are filtered out. Set to False for the
                display layer so that disabled tools remain visible with
                ``enabled=False``.

        Returns:
            List of configured tools
        """
        # Auto-discover tools from @register_tool decorators
        tools = await ToolRegistry.create_registered_tools(config)

        # Name-level filter via the spec's ``compute_allowed_names``
        # dispatch. The three return shapes encode the three modes:
        #
        #   None             — ALL mode, keep every tool from the registry
        #   frozenset()      — NONE mode, drop every tool
        #   frozenset({...}) — BY_CATEGORIES mode, keep only matching names
        #                      (plus any workforce ``name_allowlist`` injection)
        #
        # Sealed-type dispatch — the three modes are mutually exclusive
        # and impossible to confuse, unlike the older raw list whose
        # ``None`` vs ``[]`` distinction was a runtime truthiness check.
        # Configs that don't carry a spec default to ALL (full set).
        spec = (
            config.get_tool_selection_spec()
            if hasattr(config, "get_tool_selection_spec")
            else None
        )
        if spec is not None:
            # Prefer the spec. If a legacy concrete ``allowed_tools`` list
            # is ALSO present, warn rather than silently intersecting with
            # a possibly-stale list (issue #539): the spec is the source
            # of truth once supplied.
            legacy_when_spec = (
                config.get_allowed_tools()
                if hasattr(config, "get_allowed_tools")
                else None
            )
            if legacy_when_spec is not None:
                logger.warning(
                    "Both a ToolSelectionSpec and a legacy allowed_tools "
                    "list are set on %s; using the spec and ignoring the "
                    "legacy list (%d name(s)).",
                    type(config).__name__,
                    len(legacy_when_spec),
                )
            allowed_names = spec.compute_allowed_names(tools)
        else:
            # Legacy contract: ``BaseToolConfig.get_allowed_tools()`` is
            # still a public accessor on non-WebToolConfig subclasses
            # (e.g. the standalone ``ToolConfig`` in
            # core/tools/adapters/vibe/config.py:201). A caller that
            # hasn't migrated to ToolSelectionSpec still expresses the
            # name allow-list there; honour it so legacy ``ToolConfig``
            # callers (third-party / standalone embedding) keep working.
            #   None       — no filter (full default set)
            #   []         — explicit zero tools
            #   [...]      — concrete name allow-list
            legacy_list = (
                config.get_allowed_tools()
                if hasattr(config, "get_allowed_tools")
                else None
            )
            allowed_names = None if legacy_list is None else frozenset(legacy_list)

        if allowed_names is not None:
            tools = [tool for tool in tools if tool.name in allowed_names]
            if allowed_names:
                logger.info(
                    f"Filtered tools to {len(tools)} allowed tools: "
                    f"{[t.name for t in tools]}"
                )

        # Filter out tools disabled by per-user hook policy (execution layer)
        if apply_user_override_filter:
            overrides = getattr(config, "get_user_tool_overrides", lambda: {})()
            if overrides:
                disabled_by_hook = {
                    name
                    for name, ov in overrides.items()
                    if ov and ov.get("enabled") is False
                }
                if disabled_by_hook:
                    tools = [
                        tool for tool in tools if tool.name not in disabled_by_hook
                    ]

            # Positive allowlist filter (execution layer). When the hook
            # returns a concrete list, keep only tools whose name is in it.
            # Applied to the already-built list — including dynamically loaded
            # MCP tools — so no tool-universe enumeration is needed. ``None``
            # means "no allowlist configured" and skips filtering; an empty
            # list is an explicit "no tools allowed".
            allowlist = normalize_tool_allowlist(
                getattr(config, "get_user_tool_allowlist", lambda: None)()
            )
            if allowlist is not None:
                allowed_by_hook = set(allowlist)
                tools = [tool for tool in tools if tool.name in allowed_by_hook]

        # Wrap sandbox-enabled tools if sandbox is available
        sandbox = config.get_sandbox()
        if sandbox is not None:
            # The override/allowlist loads above may have re-opened a read
            # transaction on the config session after the MCP creator's
            # release; workspace setup below awaits sandbox exec (external
            # I/O), so release the pooled connection again first
            # (issue #889).
            release = getattr(config, "release_db_connection", None)
            if callable(release):
                release()
            workspace = ToolFactory._create_workspace(config.get_workspace_config())
            if workspace is not None:
                from .sandboxed_tool.sandboxed_tool_wrapper import (
                    create_workspace_in_sandbox,
                )

                setup_sandbox = getattr(sandbox, "primary_sandbox", sandbox)
                await create_workspace_in_sandbox(setup_sandbox, workspace)
            tools = await ToolFactory._wrap_sandbox_tools(tools, sandbox)

        # Apply output filtering to all tools
        tools = ToolFactory._apply_output_filters(tools, config)

        logger.info(f"Created {len(tools)} tools from configuration")
        return tools

    @staticmethod
    def _apply_output_filters(tools: List[Tool], config: BaseToolConfig) -> List[Tool]:
        """Apply output filtering to all tools.

        Args:
            tools: Original tool list
            config: Tool configuration

        Returns:
            Tool list with output filtering applied
        """
        max_chars = config.get_max_output_length()
        max_fields = config.get_max_field_count()
        max_recursion = config.get_max_recursion_depth()

        filtered_tools: List[Tool] = []
        for tool in tools:
            # Only wrap AbstractBaseTool instances
            if isinstance(tool, AbstractBaseTool):
                wrapper = OutputFilteredToolWrapper(
                    target_tool=tool,
                    max_chars=max_chars,
                    max_fields=max_fields,
                    max_recursion=max_recursion,
                )
                filtered_tools.append(wrapper)
            else:
                # For non-AbstractBaseTool tools, keep as is
                filtered_tools.append(tool)

        if filtered_tools:
            logger.debug(
                f"Applied output filtering to {len(filtered_tools)} tools "
                f"(max_chars={max_chars}, max_fields={max_fields}, max_recursion={max_recursion})"
            )

        return filtered_tools

    @staticmethod
    async def _wrap_sandbox_tools(tools: List[Tool], sandbox: Any) -> List[Tool]:
        """Wrap sandbox-enabled tools with SandboxedToolWrapper.

        Args:
            tools: Original tool list
            sandbox: Sandbox instance

        Returns:
            Tool list with sandbox-enabled tools wrapped
        """
        from .sandboxed_tool.sandbox_config import resolve_sandbox_config
        from .sandboxed_tool.sandboxed_tool_wrapper import create_sandboxed_tool

        wrapped_tools: List[Tool] = []
        for tool in tools:
            sb_config = resolve_sandbox_config(tool)
            if sb_config is not None and sb_config.enabled:
                try:
                    wrapped = await create_sandboxed_tool(
                        tool=tool,
                        sandbox=sandbox,
                    )
                    wrapped_tools.append(wrapped)
                    logger.info(f"Wrapped tool '{tool.name}' with sandbox")
                except Exception as e:
                    logger.warning(
                        f"Failed to wrap tool '{tool.name}' with sandbox: {e}, "
                        f"using original tool"
                    )
                    wrapped_tools.append(tool)
            else:
                wrapped_tools.append(tool)
        return wrapped_tools

    # New unified tool creation methods
    @staticmethod
    def _create_workspace(
        workspace_config: Optional[Dict[str, Any]],
    ) -> Optional[TaskWorkspace]:
        """Create workspace from configuration.

        Uses MockWorkspace for tool listing scenarios to avoid creating
        unnecessary directories on disk.
        """
        if not workspace_config:
            return None

        try:
            task_id = workspace_config.get("task_id")

            # Use MockWorkspace for tool listing scenarios
            # This avoids creating unnecessary directories on disk
            if task_id in ("tools_list", "_mock_", None):
                from ....workspace import MockWorkspace

                logger.debug(f"Using MockWorkspace for task_id='{task_id}'")
                return MockWorkspace(
                    id=task_id or "_mock_",
                    base_dir=workspace_config.get("base_dir") or str(get_uploads_dir()),
                )

            # Real task - create actual workspace.
            # IMPORTANT: forward `allowed_external_dirs` so that file tools can
            # access files outside the per-task workspace directory (e.g. the
            # user's upload directory). Otherwise read_file/read_csv_file will
            # reject every uploaded file as "outside the allowed directory".
            from ....workspace import WorkspaceManager

            workspace_manager = WorkspaceManager()
            workspace = workspace_manager.get_or_create_workspace(
                workspace_config.get("base_dir") or str(get_uploads_dir()),
                task_id or "default",
                allowed_external_dirs=workspace_config.get("allowed_external_dirs"),
                db_task_id=workspace_config.get("db_task_id"),
                scope_segments=tuple(workspace_config.get("scope_segments") or ()),
            )
            user_id = workspace_config.get("user_id")
            if isinstance(user_id, int):
                workspace.owner_user_id = user_id
            return workspace
        except Exception as e:
            logger.warning(f"Failed to create workspace: {e}")
            return None

    @staticmethod
    def _create_unavailable_mcp_tool(
        *,
        server_name: object,
        server_id: object = None,
        allow_users: object = None,
        reason: object = None,
        message: object = None,
        failure_code: object = None,
    ) -> Tool:
        """Build the shared server-scoped unavailable MCP tool."""
        from ....agent.result import normalize_tool_failure_code
        from .mcp_adapter import UnavailableMCPTool

        kwargs: Dict[str, Any] = {
            "server_name": server_name if isinstance(server_name, str) else "",
            "server_id": server_id,
            "allow_users": allow_users if isinstance(allow_users, list) else None,
            "failure_code": normalize_tool_failure_code(failure_code),
        }
        if isinstance(reason, str):
            kwargs["reason"] = reason
        if isinstance(message, str):
            kwargs["message"] = message
        return UnavailableMCPTool(**kwargs)

    @classmethod
    def _unavailable_mcp_tools_from_load_failures(
        cls,
        failures: Tuple["MCPServerLoadFailure", ...],
        configs_by_name: Dict[str, Dict[str, Any]],
    ) -> List[Tool]:
        """Convert structured load failures to one unavailable tool per server."""
        from .mcp_adapter import mcp_load_failure_message

        tools: List[Tool] = []
        seen_servers: set[str] = set()
        for failure in failures:
            if failure.server_name in seen_servers:
                continue
            seen_servers.add(failure.server_name)
            config = configs_by_name.get(failure.server_name, {})
            inner_config = config.get("config")
            server_id = config.get("id")
            if server_id is None and isinstance(inner_config, dict):
                server_id = inner_config.get("server_id")
            tools.append(
                cls._create_unavailable_mcp_tool(
                    server_name=failure.server_name,
                    server_id=server_id,
                    allow_users=config.get("allow_users"),
                    reason=failure.phase.value,
                    message=mcp_load_failure_message(failure.phase),
                )
            )
        return tools

    @staticmethod
    def _mcp_unavailable_summaries(
        tools: Iterable[Tool],
    ) -> tuple[MCPUnavailableSummary, ...]:
        """Project unavailable tools into the shared strict-policy contract."""
        from .mcp_adapter import UnavailableMCPTool

        return tuple(
            MCPUnavailableSummary.from_values(
                tool.server_name,
                tool.unavailability_reason,
            )
            for tool in tools
            if isinstance(tool, UnavailableMCPTool)
        )

    @staticmethod
    async def _create_mcp_tools_from_configs(
        mcp_configs: List[Dict[str, Any]],
        sandbox: Optional["Sandbox"] = None,
    ) -> List[Tool]:
        """Create MCP tools from configurations."""
        try:
            from .mcp_adapter import load_mcp_tools_as_agent_tools

            unavailable_tools: List[Tool] = []
            normal_configs: List[Dict[str, Any]] = []

            for config in mcp_configs:
                inner_config = config.get("config")
                if isinstance(inner_config, dict) and inner_config.get("unavailable"):
                    try:
                        server_name = config.get("name")
                        allow_users = config.get("allow_users")
                        unavailable_tools.append(
                            ToolFactory._create_unavailable_mcp_tool(
                                server_name=server_name,
                                server_id=inner_config.get("server_id"),
                                allow_users=allow_users,
                                reason=inner_config.get("reason"),
                                message=inner_config.get("message"),
                                failure_code=inner_config.get("failure_code"),
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to create unavailable MCP tool for server '%s': %s",
                            config.get("name", "<unknown>"),
                            type(e).__name__,
                        )
                    continue
                normal_configs.append(config)

            normal_tools: List[Tool] = []
            if normal_configs:
                connections: Dict[str, Any] = {}
                configs_by_name: Dict[str, Dict[str, Any]] = {}
                try:
                    # Convert configs to connection format
                    for config in normal_configs:
                        inner_config = config.get("config")
                        if not isinstance(inner_config, dict):
                            logger.warning(
                                "MCP server config 'config' field for server "
                                "'%s' must be a dictionary, got %s",
                                config.get("name", "<unknown>"),
                                type(inner_config).__name__,
                            )
                            unavailable_tools.append(
                                ToolFactory._create_unavailable_mcp_tool(
                                    server_name=config.get("name"),
                                    server_id=config.get("id"),
                                    allow_users=config.get("allow_users"),
                                    reason="invalid_config",
                                    message="MCP server configuration is unavailable.",
                                )
                            )
                            continue
                        server_name = config.get("name")
                        if not isinstance(server_name, str):
                            unavailable_tools.append(
                                ToolFactory._create_unavailable_mcp_tool(
                                    server_name=server_name,
                                    server_id=config.get("id"),
                                    allow_users=config.get("allow_users"),
                                    reason="invalid_config",
                                    message="MCP server configuration is unavailable.",
                                )
                            )
                            continue
                        transport = config.get("transport")
                        if not isinstance(transport, str):
                            unavailable_tools.append(
                                ToolFactory._create_unavailable_mcp_tool(
                                    server_name=server_name,
                                    server_id=config.get("id"),
                                    allow_users=config.get("allow_users"),
                                    reason="invalid_config",
                                    message="MCP server configuration is unavailable.",
                                )
                            )
                            continue
                        connection_config = {
                            "transport": transport,
                            **inner_config,
                        }
                        for runtime_key in (
                            "runtime_bindings",
                            "runtime_input_schema",
                            "connector_runtime",
                            "allow_delegated_authorization",
                        ):
                            if runtime_key in config:
                                connection_config[runtime_key] = config[runtime_key]

                        # Fix args field if it's a string instead of list
                        if "args" in connection_config and isinstance(
                            connection_config["args"], str
                        ):
                            # Split args string to list, handling quoted arguments
                            import shlex

                            try:
                                connection_config["args"] = shlex.split(
                                    connection_config["args"]
                                )
                                logger.info(
                                    f"Converted args string to list: {connection_config['args']}"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to parse args string: {e}")
                                # Fallback to simple split
                                connection_config["args"] = connection_config[
                                    "args"
                                ].split()

                        connections[server_name] = connection_config
                        configs_by_name[server_name] = config

                    # Load MCP tools
                    if connections:
                        load_result = await load_mcp_tools_as_agent_tools(
                            connections,
                            sandbox=sandbox,
                        )  # type: ignore[arg-type]
                        normal_tools = list(load_result.tools)
                        unavailable_tools.extend(
                            ToolFactory._unavailable_mcp_tools_from_load_failures(
                                load_result.failures,
                                configs_by_name,
                            )
                        )
                except ConnectorRuntimeError:
                    raise
                except Exception as e:
                    logger.warning("Failed to create MCP tools (%s)", type(e).__name__)
                    for server_name, config in configs_by_name.items():
                        unavailable_tools.append(
                            ToolFactory._create_unavailable_mcp_tool(
                                server_name=server_name,
                                server_id=config.get("id"),
                                allow_users=config.get("allow_users"),
                                reason="loader_failed",
                                message="MCP server tools could not be loaded.",
                            )
                        )

            return unavailable_tools + normal_tools
        except ConnectorRuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Failed to create MCP tools: {e}")
            return []

    @classmethod
    async def create_mcp_tools(
        cls,
        db: Session,
        user_id: int | None = None,
        *,
        mcp_failure_policy: MCPFailurePolicy = MCPFailurePolicy.BEST_EFFORT,
    ):
        """Create MCP tools from database configuration.

        Args:
            db: Database session
            user_id: User ID for filtering MCP servers
            mcp_failure_policy: Caller-owned handling for unavailable servers

        Returns:
            List of MCP tools
        """
        try:
            from .....web.models.mcp import MCPServer, UserMCPServer
            from .....web.services.mcp_runtime import (
                build_mcp_runtime_connection,
                load_shared_env_overrides,
                load_user_env_overrides,
                load_user_env_sources,
            )
            from .mcp_adapter import load_mcp_tools_as_agent_tools

            query = db.query(MCPServer)
            if user_id:
                query = query.join(
                    UserMCPServer, MCPServer.id == UserMCPServer.mcpserver_id
                ).filter(UserMCPServer.user_id == user_id, UserMCPServer.is_active)

            user_env_overrides = load_user_env_overrides(db, user_id)
            shared_env_overrides = load_shared_env_overrides(db, user_id)
            env_source_overrides = load_user_env_sources(db, user_id)

            connections: Dict[str, Any] = {}
            configs_by_name: Dict[str, Dict[str, Any]] = {}
            unavailable_tools: List[Tool] = []
            for server in query.all():
                if isinstance(server, tuple):
                    server = server[0]
                try:
                    build = await build_mcp_runtime_connection(
                        db,
                        server,
                        user_id=user_id,
                        user_env_overrides=user_env_overrides,
                        shared_env_overrides=shared_env_overrides,
                        env_source_overrides=env_source_overrides,
                    )
                except ConnectorRuntimeError:
                    raise
                except Exception as e:
                    logger.warning(
                        "MCP runtime connection build failed for server '%s' (%s)",
                        getattr(server, "name", "<unknown>"),
                        type(e).__name__,
                    )
                    unavailable_tools.append(
                        cls._create_unavailable_mcp_tool(
                            server_name=getattr(server, "name", ""),
                            server_id=getattr(server, "id", None),
                            allow_users=[str(user_id)] if user_id is not None else None,
                            reason="runtime_connection_failed",
                            message="MCP server configuration is unavailable.",
                        )
                    )
                    continue
                if build.connection is not None:
                    server_name = str(server.name)
                    connections[server_name] = build.connection
                    configs_by_name[server_name] = {
                        "id": getattr(server, "id", None),
                        "name": server_name,
                        "allow_users": [str(user_id)] if user_id is not None else None,
                    }
                    continue
                diagnostic = build.diagnostic or {}
                logger.warning(
                    "MCP runtime connection unavailable for server '%s' (%s)",
                    getattr(server, "name", "<unknown>"),
                    diagnostic.get("code", "runtime_connection_unavailable"),
                )
                unavailable_tools.append(
                    cls._create_unavailable_mcp_tool(
                        server_name=getattr(server, "name", ""),
                        server_id=getattr(server, "id", None),
                        allow_users=[str(user_id)] if user_id is not None else None,
                        reason=diagnostic.get("code", "runtime_connection_unavailable"),
                        message="MCP server configuration is unavailable.",
                    )
                )

            if not connections:
                enforce_mcp_failure_policy(
                    mcp_failure_policy,
                    cls._mcp_unavailable_summaries(unavailable_tools),
                )
                return unavailable_tools

            # Load MCP tools
            try:
                load_result = await load_mcp_tools_as_agent_tools(connections)
            except ConnectorRuntimeError:
                raise
            except Exception as e:
                logger.warning(
                    "Failed to create MCP tools from database (%s)", type(e).__name__
                )
                unavailable_tools.extend(
                    cls._create_unavailable_mcp_tool(
                        server_name=server_name,
                        server_id=config.get("id"),
                        allow_users=config.get("allow_users"),
                        reason="loader_failed",
                        message="MCP server tools could not be loaded.",
                    )
                    for server_name, config in configs_by_name.items()
                )
                enforce_mcp_failure_policy(
                    mcp_failure_policy,
                    cls._mcp_unavailable_summaries(unavailable_tools),
                )
                return unavailable_tools

            unavailable_tools.extend(
                cls._unavailable_mcp_tools_from_load_failures(
                    load_result.failures,
                    configs_by_name,
                )
            )
            enforce_mcp_failure_policy(
                mcp_failure_policy,
                cls._mcp_unavailable_summaries(unavailable_tools),
            )
            return unavailable_tools + list(load_result.tools)
        except ConnectorRuntimeError:
            raise
        except RequiredMCPUnavailableError:
            raise
        except Exception as e:
            logger.warning(
                "Failed to create MCP tools from database (%s)", type(e).__name__
            )
            enforce_mcp_failure_policy(
                mcp_failure_policy,
                [MCPUnavailableSummary.from_values(None, "config_load_failed")],
            )
            return []

    @classmethod
    def _create_mcp_tools(
        cls,
        db,
        user_id: int,
        *,
        mcp_failure_policy: MCPFailurePolicy = MCPFailurePolicy.BEST_EFFORT,
    ):
        """Synchronous wrapper for create_mcp_tools.

        Args:
            db: Database session
            user_id: User ID for filtering MCP servers

        Returns:
            List of MCP tools
        """
        import asyncio

        try:
            # Run async method in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, we need to create a new one
                import queue
                import threading

                result_queue = queue.Queue()

                def run_async():
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(
                            cls.create_mcp_tools(
                                db,
                                user_id,
                                mcp_failure_policy=mcp_failure_policy,
                            )
                        )
                        result_queue.put(result)
                    except Exception as e:
                        result_queue.put(e)
                    finally:
                        new_loop.close()

                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()

                result = result_queue.get()
                if isinstance(result, Exception):
                    raise result
                return result
            else:
                # If no event loop is running, use the current one
                return loop.run_until_complete(
                    cls.create_mcp_tools(
                        db,
                        user_id,
                        mcp_failure_policy=mcp_failure_policy,
                    )
                )
        except RequiredMCPUnavailableError:
            raise
        except Exception as e:
            logger.warning(f"Failed to create MCP tools (sync wrapper): {e}")
            return []
