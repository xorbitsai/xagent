"""
Tool Factory for xagent

Provides a unified interface for creating tools with proper workspace binding
and configuration management.
"""

# mypy: ignore-errors

import logging
import os
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .....web.models.mcp import MCPServer, UserMCPServer
from ....workspace import TaskWorkspace
from .base import Tool
from .config import ToolConfig
from .image_tool import create_image_tool

# Import MCP function for test compatibility
from .mcp_adapter import load_mcp_tools_as_agent_tools
from .vision_tool import get_vision_tool
from .web_search import WebSearchTool
from .zhipu_web_search import ZhipuWebSearchTool

logger = logging.getLogger(__name__)


class ToolFactory:
    """
    Unified tool factory that handles tool creation with proper workspace binding.
    """

    @staticmethod
    async def create_all_tools(config: ToolConfig) -> List[Tool]:
        """
        Create all tools based on configuration.

        This is the unified entry point for tool creation. All tools are created
        based on the provided configuration, which can come from web context
        or standalone usage.

        Args:
            config: Tool configuration object

        Returns:
            List of configured tools
        """
        tools: List[Tool] = []

        # Create workspace from configuration
        workspace = ToolFactory._create_workspace(config.get_workspace_config())

        # Basic tools (always enabled if config allows)
        if config.get_basic_tools_enabled():
            basic_tools = ToolFactory._create_basic_tools(workspace)
            tools.extend(basic_tools)

        # Knowledge base tools
        embedding_model = config.get_embedding_model()
        allowed_collections = config.get_allowed_collections()
        user_id = config.get_user_id()
        is_admin = config.is_admin()
        knowledge_tools = ToolFactory._create_knowledge_tools(
            embedding_model, allowed_collections, user_id, is_admin
        )
        tools.extend(knowledge_tools)

        # File tools (workspace-bound)
        if config.get_file_tools_enabled() and workspace:
            file_tools = ToolFactory._create_file_tools(workspace)
            tools.extend(file_tools)

        # Vision tools
        vision_model = config.get_vision_model()
        if vision_model:
            vision_tools = get_vision_tool(
                vision_model=vision_model, workspace=workspace
            )
            tools.extend(vision_tools)

        # Image tools
        image_models = config.get_image_models()
        if image_models:
            default_generate_model = config.get_image_generate_model()
            default_edit_model = config.get_image_edit_model()
            image_tools = create_image_tool(
                image_models,
                workspace=workspace,
                default_generate_model=default_generate_model,
                default_edit_model=default_edit_model,
            )
            tools.extend(image_tools)

        # Special image tools (workspace-bound)
        if workspace:
            special_image_tools = ToolFactory._create_special_image_tools(workspace)
            tools.extend(special_image_tools)

        # MCP tools
        mcp_configs = config.get_mcp_server_configs()
        if mcp_configs:
            mcp_tools = await ToolFactory._create_mcp_tools_from_configs(mcp_configs)
            tools.extend(mcp_tools)

        # Browser automation tools
        task_id = config.get_task_id()
        if config.get_browser_tools_enabled():
            browser_tools = ToolFactory._create_browser_tools(task_id, workspace)
            tools.extend(browser_tools)

        # Published agent tools
        if config.get_enable_agent_tools():
            agent_tools = ToolFactory._create_agent_tools(
                db=config.get_db(),
                user_id=config.get_user_id(),
                task_id=task_id,
                config=config,
            )
            tools.extend(agent_tools)

        # PPTX presentation tools (6 tools: read, generate, unpack, pack, add_slide, clean)
        pptx_tools = ToolFactory._create_pptx_tools(
            db=config.get_db(),
            user_id=config.get_user_id(),
            task_id=config.get_task_id(),
            config=config,
        )
        tools.extend(pptx_tools)

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

        logger.info(f"Created {len(tools)} tools from configuration")
        return tools

    # New unified tool creation methods
    @staticmethod
    def _create_workspace(
        workspace_config: Optional[Dict[str, Any]],
    ) -> Optional[TaskWorkspace]:
        """Create workspace from configuration."""
        if not workspace_config:
            return None

        try:
            from ....workspace import WorkspaceManager

            workspace_manager = WorkspaceManager()
            workspace = workspace_manager.get_or_create_workspace(
                workspace_config.get("base_dir", "./workspace"),
                workspace_config.get("task_id", "default"),
            )
            return workspace
        except Exception as e:
            logger.warning(f"Failed to create workspace: {e}")
            return None

    @staticmethod
    def _create_basic_tools(workspace: Optional[TaskWorkspace]) -> List[Tool]:
        """Create basic tools that are always available."""
        tools: List[Tool] = []

        # Web search tool preference: Zhipu -> Google -> none
        zhipu_api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("BIGMODEL_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")

        if zhipu_api_key:
            tools.append(ZhipuWebSearchTool())
        elif google_api_key and google_cse_id:
            tools.append(WebSearchTool())

        # Python executor tool (if workspace available)
        if workspace:
            from .python_executor import get_python_executor_tool

            python_tool = get_python_executor_tool({"workspace": workspace})
            tools.append(python_tool)

        # JavaScript executor tool (if workspace available)
        if workspace:
            from .javascript_executor import get_javascript_executor_tool

            js_tool = get_javascript_executor_tool({"workspace": workspace})
            tools.append(js_tool)

        return tools

    @staticmethod
    def _create_knowledge_tools(
        embedding_model: Optional[str] = None,
        allowed_collections: Optional[List[str]] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Tool]:
        """Create knowledge base search tools.

        Args:
            embedding_model: Optional embedding model ID to use for searches.
            allowed_collections: Optional list of allowed collection names to filter.
            user_id: Optional user ID for multi-tenancy filtering.
            is_admin: Whether current user is admin.
        """
        tools: List[Tool] = []

        try:
            from .document_search import (
                get_knowledge_search_tool,
                get_list_knowledge_bases_tool,
            )

            # Add list knowledge bases tool first
            list_tool = get_list_knowledge_bases_tool(
                allowed_collections=allowed_collections,
                user_id=user_id,
                is_admin=is_admin,
            )
            tools.append(list_tool)
            logger.info("Added list knowledge bases tool")

            # Add search tool with embedding model and allowed collections
            knowledge_tool = get_knowledge_search_tool(
                embedding_model_id=embedding_model,
                allowed_collections=allowed_collections,
                user_id=user_id,
                is_admin=is_admin,
            )
            tools.append(knowledge_tool)
            logger.info("Added knowledge base search tool")
        except Exception as e:
            logger.warning(f"Failed to create knowledge tools: {e}")

        return tools

    @staticmethod
    def _create_file_tools(workspace: TaskWorkspace) -> List[Tool]:
        """Create workspace-bound file tools."""
        try:
            from .workspace_file_tool import create_workspace_file_tools

            return create_workspace_file_tools(workspace)
        except Exception as e:
            logger.warning(f"Failed to create file tools: {e}")
            return []

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
            from ...core.mcp.manager.db import DatabaseMCPServerManager

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

    @staticmethod
    def _create_special_image_tools(workspace: TaskWorkspace) -> List[Tool]:
        """Create special image tools that require workspace binding."""
        tools: List[Tool] = []

        try:
            # Image web search tool
            from .image_web_search import create_image_web_search_tool

            image_search_tool = create_image_web_search_tool(workspace)
            tools.append(image_search_tool)
            logger.info("Added image web search tool")
        except Exception as e:
            logger.warning(f"Failed to create image web search tool: {e}")

        try:
            # Logo overlay tool
            from .logo_overlay import create_logo_overlay_tool

            logo_overlay_tool = create_logo_overlay_tool(workspace)
            tools.append(logo_overlay_tool)
            logger.info("Added logo overlay tool")
        except Exception as e:
            logger.warning(f"Failed to create logo overlay tool: {e}")

        return tools

    @staticmethod
    def _create_browser_tools(
        task_id: Optional[str] = None, workspace: Optional[TaskWorkspace] = None
    ) -> List[Tool]:
        """Create browser automation tools.

        Args:
            task_id: Optional task ID for session tracking
            workspace: Optional workspace for saving screenshots
        """
        try:
            from .browser_use import create_browser_tools

            browser_tools = create_browser_tools(task_id=task_id, workspace=workspace)
            logger.info(f"Added {len(browser_tools)} browser automation tools")
            return browser_tools
        except Exception as e:
            logger.error(f"Failed to create browser tools: {e}", exc_info=True)
            return []

    @staticmethod
    def _create_agent_tools(
        db: Any, user_id: int, task_id: Optional[str] = None, config: Any = None
    ) -> List[Tool]:
        """Create tools from published agents.

        Args:
            db: Database session
            user_id: User ID for model access
            task_id: Optional task ID for workspace isolation
            config: Tool configuration for getting excluded agent ID

        Returns:
            List of AgentTool instances
        """
        try:
            from .agent_tool import get_published_agents_tools

            excluded_agent_id = config.get_excluded_agent_id() if config else None

            agent_tools = get_published_agents_tools(
                db=db,
                user_id=user_id,
                task_id=task_id,
                workspace_base_dir="uploads",
                excluded_agent_id=excluded_agent_id,
            )
            logger.info(f"Added {len(agent_tools)} published agent tools")
            return agent_tools
        except Exception as e:
            logger.error(f"Failed to create agent tools: {e}", exc_info=True)
            return []

    @staticmethod
    def _create_pptx_tools(
        db: Any = None,
        user_id: Optional[int] = None,
        task_id: Optional[str] = None,
        config: Any = None,
    ) -> List[Tool]:
        """Create PPTX presentation tools.

        Args:
            db: Database session (unused, kept for compatibility)
            user_id: User ID (unused, kept for compatibility)
            task_id: Task ID (unused, kept for compatibility)
            config: Tool configuration (unused, kept for compatibility)

        Returns:
            List of PPTX tools (read, generate, unpack, pack, add_slide, clean)
        """
        try:
            from .pptx_tool import create_pptx_tool

            # Get workspace from config
            workspace = (
                ToolFactory._create_workspace(config.get_workspace_config())
                if config
                else None
            )

            pptx_tools = create_pptx_tool(workspace=workspace)
            logger.info(f"Added {len(pptx_tools)} PPTX tools")
            return pptx_tools
        except Exception as e:
            logger.error(f"Failed to create PPTX tools: {e}", exc_info=True)
            return []
