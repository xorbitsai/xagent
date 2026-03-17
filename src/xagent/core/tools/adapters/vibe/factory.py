"""
Tool Factory for xagent

Provides a unified interface for creating tools with proper workspace binding
and configuration management.
"""

# mypy: ignore-errors

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

if TYPE_CHECKING:
    pass

from ....workspace import TaskWorkspace
from .base import Tool
from .config import BaseToolConfig

logger = logging.getLogger(__name__)

__all__ = ["ToolFactory", "ToolRegistry", "register_tool"]


class ToolRegistry:
    """
    Global registry for tool creators using decorator pattern.

    Tools are registered using @register_tool decorator and automatically
    discovered during create_all_tools().
    """

    _tool_creators: List[Callable] = []
    _modules_imported = False

    @classmethod
    def register(cls, creator: Callable) -> Callable:
        """
        Register a tool creator function.

        The creator function will be called during create_all_tools()
        with the current config.

        Usage:
            @register_tool
            def create_my_tools(config: BaseToolConfig) -> List[Tool]:
                return [MyTool(...)]
        """
        cls._tool_creators.append(creator)
        return creator

    @classmethod
    def _import_tool_modules(cls):
        """Import tool modules to trigger @register_tool decorator registration."""
        if cls._modules_imported:
            return

        try:
            # Import tool modules in priority order - these imports trigger @register_tool decorators
            from . import (  # noqa: F401 - imports trigger @register_tool decorators
                agent_tool,
                basic_tools,
                browser_tools,
                image_tool,
                knowledge_tools,
                mcp_tools,
                pptx_tool,
                skill_tools,
                special_image_tools,
                sql_tool,
                vision_tool,
                workspace_file_tool,
            )

            cls._modules_imported = True
            logger.info("Tool modules imported and registered")
        except Exception as e:
            logger.warning(f"Failed to import tool modules: {e}")

    @classmethod
    async def create_registered_tools(cls, config: BaseToolConfig) -> List[Tool]:
        """Create tools from all registered creators."""
        # Import tool modules on first call to trigger decorator registration
        cls._import_tool_modules()

        tools = []
        for creator in cls._tool_creators:
            try:
                created_tools = await creator(config)
                tools.extend(created_tools)
            except Exception as e:
                logger.warning(f"Tool creator {creator.__name__} failed: {e}")

        # Sort tools by category priority
        tools = cls._sort_tools_by_category(tools)
        return tools

    @classmethod
    def _sort_tools_by_category(cls, tools: List[Tool]) -> List[Tool]:
        """Sort tools by category priority.

        Priority order (most important first):
        1. BASIC - Basic tools (search, code execution)
        2. KNOWLEDGE - Knowledge base search
        3. FILE - File operations
        4. VISION - Vision understanding
        5. IMAGE - Image generation
        6. BROWSER - Browser automation
        7. PPT - PPT tools
        8. DATABASE - Database tools (SQL query)
        9. MCP - MCP tools
        10. SKILL - Skill documentation access tools
        11. AGENT - Agent tools (delegation)
        12. OTHER - Other tools
        """
        from .base import ToolCategory

        # Define category priority order
        category_order = {
            ToolCategory.BASIC: 0,
            ToolCategory.KNOWLEDGE: 1,
            ToolCategory.FILE: 2,
            ToolCategory.VISION: 3,
            ToolCategory.IMAGE: 4,
            ToolCategory.BROWSER: 5,
            ToolCategory.PPT: 6,
            ToolCategory.DATABASE: 7,
            ToolCategory.MCP: 8,
            ToolCategory.SKILL: 9,
            ToolCategory.AGENT: 10,
            ToolCategory.OTHER: 11,
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
    async def create_all_tools(config: BaseToolConfig) -> List[Tool]:
        """
        Create all tools based on configuration.

        This is the unified entry point for tool creation. All tools are discovered
        automatically via @register_tool decorators based on the provided configuration.

        Args:
            config: Tool configuration object

        Returns:
            List of configured tools
        """
        # Auto-discover tools from @register_tool decorators
        tools = await ToolRegistry.create_registered_tools(config)

        # Filter tools by allowed_tools if specified
        allowed_tools = config.get_allowed_tools()
        if allowed_tools is not None and len(allowed_tools) > 0:
            tools = [tool for tool in tools if tool.name in allowed_tools]
            logger.info(
                f"Filtered tools to {len(tools)} allowed tools: {[t.name for t in tools]}"
            )
        elif allowed_tools is not None and len(allowed_tools) == 0:
            logger.warning(
                "⚠️ allowed_tools is empty list - this will filter out all tools! If you want to allow all tools, set allowed_tools to None"
            )

        # Wrap sandbox-enabled tools if sandbox is available
        sandbox = config.get_sandbox()
        if sandbox is not None:
            tools = await ToolFactory._wrap_sandbox_tools(tools, sandbox)

        logger.info(f"Created {len(tools)} tools from configuration")
        return tools

    @staticmethod
    async def _wrap_sandbox_tools(tools: List[Tool], sandbox: Any) -> List[Tool]:
        """Wrap sandbox-enabled tools with SandboxedToolWrapper.

        Args:
            tools: Original tool list
            sandbox: Sandbox instance

        Returns:
            Tool list with sandbox-enabled tools wrapped
        """
        from .sandboxed_tool.sandboxed_tool_config import is_sandbox_enabled
        from .sandboxed_tool.sandboxed_tool_wrapper import create_sandboxed_tool

        wrapped_tools: List[Tool] = []
        for tool in tools:
            if is_sandbox_enabled(tool.name):
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
                    base_dir=workspace_config.get("base_dir", "./uploads"),
                )

            # Real task - create actual workspace
            from ....workspace import WorkspaceManager

            workspace_manager = WorkspaceManager()
            workspace = workspace_manager.get_or_create_workspace(
                workspace_config.get("base_dir", "./workspace"),
                task_id or "default",
            )
            return workspace
        except Exception as e:
            logger.warning(f"Failed to create workspace: {e}")
            return None

    @staticmethod
    async def _create_mcp_tools_from_configs(
        mcp_configs: List[Dict[str, Any]],
    ) -> List[Tool]:
        """Create MCP tools from configurations."""
        try:
            from .mcp_adapter import load_mcp_tools_as_agent_tools

            # Convert configs to connection format
            connections = {}
            for config in mcp_configs:
                connection_config = {
                    "transport": config["transport"],
                    **config["config"],
                }

                # Fix args field if it's a string instead of list
                if "args" in connection_config and isinstance(
                    connection_config["args"], str
                ):
                    # Split args string into list, handling quoted arguments
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
                        connection_config["args"] = connection_config["args"].split()

                connections[config["name"]] = connection_config

            # Load MCP tools
            mcp_tools = await load_mcp_tools_as_agent_tools(connections)  # type: ignore[arg-type]
            return mcp_tools if mcp_tools else []  # type: ignore[return-value]
        except Exception as e:
            logger.warning(f"Failed to create MCP tools: {e}")
            return []

    @classmethod
    async def create_mcp_tools(cls, db: Session, user_id: int | None = None):
        """Create MCP tools from database configuration.

        Args:
            db: Database session
            user_id: User ID for filtering MCP servers

        Returns:
            List of MCP tools
        """
        try:
            from .....web.models.mcp import MCPServer, UserMCPServer
            from ...core.mcp.manager.db import DatabaseMCPServerManager
            from .mcp_adapter import load_mcp_tools_as_agent_tools

            # Load MCP server connections for the specific user
            manager = DatabaseMCPServerManager(db)

            if user_id:

                def filter_by_user(query):
                    return query.join(
                        UserMCPServer, MCPServer.id == UserMCPServer.mcpserver_id
                    ).filter(UserMCPServer.user_id == user_id, UserMCPServer.is_active)

                connections = manager.get_connections(filter_by_user)
            else:
                connections = manager.get_connections()

            if not connections:
                return []

            # Load MCP tools
            mcp_tools = await load_mcp_tools_as_agent_tools(connections)
            return mcp_tools if mcp_tools else []
        except Exception as e:
            logger.warning(f"Failed to create MCP tools from database: {e}")
            return []

    @classmethod
    def _create_mcp_tools(cls, db, user_id: int):
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
                            cls.create_mcp_tools(db, user_id)
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
                return loop.run_until_complete(cls.create_mcp_tools(db, user_id))
        except Exception as e:
            logger.warning(f"Failed to create MCP tools (sync wrapper): {e}")
            return []
