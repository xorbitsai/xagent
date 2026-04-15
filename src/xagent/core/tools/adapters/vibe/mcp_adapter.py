"""MCP Tool Adapter for Agent System.

This module provides adapters to convert MCP tools into Agent system Tool format,
enabling MCP tools to be used in DAG plan-execute patterns and other agent workflows.
"""

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Type

from mcp.types import Tool as MCPTool
from pydantic import BaseModel, Field, create_model

from .....web.user_context import UserContext
from ...core.mcp.sessions import Connection, create_session
from .base import AbstractBaseTool, ToolVisibility


class EmptyArgsModel(BaseModel):
    pass


logger = logging.getLogger(__name__)


class MCPToolAdapter(AbstractBaseTool):
    """
    Adapter that converts an MCP tool into an Agent system Tool.

    This adapter handles:
    - MCP session management
    - Argument schema conversion
    - Async execution with proper session lifecycle
    - User isolation and validation
    - Error handling and logging
    """

    def __init__(
        self,
        mcp_tool: MCPTool,
        connection: Connection,
        *,
        name_prefix: Optional[str] = None,
        visibility: Optional[ToolVisibility] = None,
        allow_users: Optional[List[str]] = None,
    ):
        """Initialize MCP tool adapter.

        Args:
            mcp_tool: The MCP tool to wrap
            connection: MCP server connection configuration
            name_prefix: Optional prefix for tool name (e.g., "mcp_")
            visibility: Tool visibility setting
            allow_users: List of allowed user IDs
        """
        self.mcp_tool = mcp_tool
        self.connection = connection
        self._name_prefix = name_prefix or ""
        self._visibility = visibility or ToolVisibility.PRIVATE
        self._allow_users = allow_users
        from .base import ToolCategory

        self.category = ToolCategory.MCP

        # Build models from MCP tool schema
        self._args_type = self._build_args_model()
        self._return_type = self._build_return_model()

    @property
    def name(self) -> str:
        """Get tool name with optional prefix, formatted for LLM requirements."""
        raw_name = f"{self._name_prefix}{self.mcp_tool.name}"
        # Replace spaces and dashes with underscores to match LLM tool naming constraints
        # This matches the frontend/chat.py filtering logic
        return raw_name.replace(" ", "_").replace("-", "_")

    @property
    def description(self) -> str:
        """Get tool description from MCP tool."""
        return self.mcp_tool.description or f"Execute MCP tool: {self.mcp_tool.name}"

    @property
    def tags(self) -> List[str]:
        """Get tags for this tool."""
        tags = ["mcp"]
        if hasattr(self.mcp_tool, "annotations") and self.mcp_tool.annotations:
            # Add any annotations as tags
            if hasattr(self.mcp_tool.annotations, "audience"):
                tags.extend(self.mcp_tool.annotations.audience or [])
        return tags

    def args_type(self) -> Type[BaseModel]:
        """Get argument model type."""
        return self._args_type

    def return_type(self) -> Type[BaseModel]:
        """Get return model type."""
        return self._return_type

    def state_type(self) -> Optional[Type[BaseModel]]:
        """MCP tools are stateless."""
        return None

    def is_async(self) -> bool:
        """MCP tools are always async."""
        return True

    def _build_args_model(self) -> Type[BaseModel]:
        """Build Pydantic model from MCP tool input schema."""
        try:
            if not self.mcp_tool.inputSchema:
                # No input parameters
                return EmptyArgsModel

            # Convert JSON schema to Pydantic model
            schema = self.mcp_tool.inputSchema

            if not isinstance(schema, dict):
                logger.warning(
                    f"Invalid input schema for MCP tool {self.mcp_tool.name}"
                )

                return EmptyArgsModel

            # Extract properties and required fields
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            if not properties:
                return EmptyArgsModel

            # Build field definitions for create_model
            fields: Dict[str, Any] = {}

            for field_name, field_schema in properties.items():
                field_type = self._json_schema_to_python_type(field_schema)

                # Check if field is required
                if field_name in required:
                    fields[field_name] = (field_type, ...)
                else:
                    # Optional field with default
                    default_value = field_schema.get("default", None)
                    fields[field_name] = (Optional[field_type], default_value)

            # Create the model
            model_name = f"{self.mcp_tool.name.title().replace('_', '')}Args"
            return create_model(model_name, **fields)

        except Exception as e:
            logger.error(
                f"Failed to build args model for MCP tool {self.mcp_tool.name}: {e}"
            )

            return EmptyArgsModel

    def _build_return_model(self) -> Type[BaseModel]:
        """Build return model for MCP tool output."""

        # MCP tools return CallToolResult which contains content
        class MCPToolResult(BaseModel):
            content: List[Dict[str, Any]] = Field(
                default_factory=list, description="Tool execution result content"
            )
            is_error: bool = Field(
                default=False,
                description="Whether the tool execution resulted in an error",
            )

        return MCPToolResult

    def _json_schema_to_python_type(self, schema: Dict[str, Any]) -> Type:
        """Convert JSON schema type to Python type."""
        schema_type = schema.get("type", "string")

        # Handle list types (e.g. ["string", "null"])
        if isinstance(schema_type, list):
            # Pick the first non-null type
            types = [t for t in schema_type if t != "null"]
            schema_type = types[0] if types else "string"

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List[Any],
            "object": Dict[str, Any],
        }

        return type_mapping.get(schema_type, Any)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute MCP tool asynchronously with user validation and context."""
        try:
            # Get current user ID with improved detection
            current_user_id = self._get_current_user_id()

            # Validate user permissions
            if not self._is_user_allowed(current_user_id):
                error_msg = f"User {current_user_id} is not authorized to use tool {self.mcp_tool.name}"
                logger.warning(error_msg)
                return {
                    "content": [{"text": f"Access denied: {error_msg}"}],
                    "is_error": True,
                }

            # Validate arguments
            parsed_args = self._args_type(**args)
            tool_args = parsed_args.model_dump(exclude_none=True)

            logger.debug(
                f"Executing MCP tool {self.mcp_tool.name} with args: {tool_args} for user {current_user_id}"
            )

            # Set user context for execution
            user_context = UserContext(current_user_id)

            with user_context.set_context():
                # Create session and execute tool
                async with create_session(self.connection) as session:
                    await session.initialize()

                    # Call MCP tool
                    result = await session.call_tool(self.mcp_tool.name, tool_args)

                    # Convert result to our format
                    content = []
                    if result.content:
                        for content_item in result.content:
                            if hasattr(content_item, "model_dump"):
                                content.append(content_item.model_dump())
                            else:
                                content.append({"text": str(content_item)})

                    return {
                        "content": content,
                        "is_error": result.isError
                        if hasattr(result, "isError")
                        else False,
                    }

        except Exception as e:
            logger.error(f"MCP tool {self.mcp_tool.name} execution failed: {e}")

            # Extract real errors from ExceptionGroup if present
            error_msg = str(e)
            if isinstance(e, BaseExceptionGroup):
                sub_msgs: list[str] = []
                for sub_e in e.exceptions:
                    if isinstance(sub_e, BaseExceptionGroup):
                        sub_msgs.extend(
                            str(sub_sub_e) for sub_sub_e in sub_e.exceptions
                        )
                    else:
                        sub_msgs.append(str(sub_e))
                if sub_msgs:
                    error_msg = f"{error_msg}: " + ", ".join(sub_msgs)

            return {
                "content": [{"text": f"Error executing MCP tool: {error_msg}"}],
                "is_error": True,
            }

    def _get_current_user_id(self) -> Optional[str]:
        """Get current user ID from environment or context."""
        # Try to get user ID from environment variable (set by web system)
        user_id = os.environ.get("XAGENT_USER_ID")
        if user_id:
            return user_id

        # If no user ID found, this might be a system-level execution
        # In production, this should be replaced with proper context passing
        logger.warning(
            "No user ID found in environment, MCP tool may not be properly isolated"
        )
        return None

    def _is_user_allowed(self, user_id: Optional[str]) -> bool:
        """Check if user is allowed to use this tool."""
        if not user_id:
            # If no user ID, this might be a system execution
            # For security, we should deny access unless explicitly allowed
            return self._allow_users is None or "system" in self._allow_users

        if self._allow_users is None:
            # No specific user restrictions, allow access
            return True

        return user_id in self._allow_users

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """MCP tools are async only."""
        raise RuntimeError(
            f"MCP tool {self.mcp_tool.name} is async only; please use run_json_async()"
        )

    async def save_state_json(self) -> Mapping[str, Any]:
        """MCP tools are stateless."""
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        """MCP tools are stateless."""
        pass

    def return_value_as_string(self, value: Any) -> str:
        """Convert return value to string representation."""
        try:
            if isinstance(value, dict):
                content = value.get("content", [])
                if content:
                    # Extract text from content items
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        else:
                            texts.append(str(item))
                    return "\n".join(texts)
                return "No content returned"
            return str(value)
        except Exception as e:
            logger.warning(f"Failed to convert return value to string: {e}")
            return str(value)


async def load_mcp_tools_as_agent_tools(
    connection_map: Dict[str, Connection],
    *,
    name_prefix: str = "mcp_",
    visibility: Optional[ToolVisibility] = None,
    allow_users: Optional[List[str]] = None,
) -> List[MCPToolAdapter]:
    """Load MCP tools from multiple servers and convert to Agent tools.

    Args:
        connection_map: Map of server names to connection configurations
        name_prefix: Prefix for tool names (default: "mcp_")
        visibility: Tool visibility setting
        allow_users: List of allowed user IDs

    Returns:
        List of MCPToolAdapter instances

    Raises:
        Exception: If any MCP server connection fails
    """
    agent_tools: List[MCPToolAdapter] = []

    for server_name, connection in connection_map.items():
        try:
            logger.info(f"Loading tools from MCP server: {server_name}")

            # Create session and list tools
            async with create_session(connection) as session:
                await session.initialize()

                # List available tools
                tools_result = await session.list_tools()
                mcp_tools = tools_result.tools if tools_result.tools else []

                logger.info(f"Found {len(mcp_tools)} tools from server {server_name}")

                # Convert each MCP tool to Agent tool
                for mcp_tool in mcp_tools:
                    try:
                        # Clean server name for prefix (replace spaces/dashes with underscores)
                        clean_server_name = server_name.replace(" ", "_").replace(
                            "-", "_"
                        )

                        # Create tool name with server prefix
                        tool_prefix = (
                            f"{name_prefix}{clean_server_name}_"
                            if name_prefix
                            else f"{clean_server_name}_"
                        )

                        adapter = MCPToolAdapter(
                            mcp_tool=mcp_tool,
                            connection=connection,
                            name_prefix=tool_prefix,
                            visibility=visibility,
                            allow_users=allow_users,
                        )

                        agent_tools.append(adapter)
                        logger.debug(f"Created adapter for tool: {adapter.name}")

                    except Exception as e:
                        logger.error(
                            f"Failed to create adapter for tool {mcp_tool.name}: {e}"
                        )
                        continue

        except Exception as e:
            logger.error(f"Failed to load tools from MCP server {server_name}: {e}")
            # Continue with other servers rather than failing completely
            continue

    logger.info(f"Successfully loaded {len(agent_tools)} MCP tools as Agent tools")
    return agent_tools


def create_mcp_tool_adapter(
    mcp_tool: MCPTool,
    connection: Connection,
    *,
    name_prefix: Optional[str] = None,
    visibility: Optional[ToolVisibility] = None,
    allow_users: Optional[List[str]] = None,
) -> MCPToolAdapter:
    """Create a single MCP tool adapter.

    Convenience function for creating individual tool adapters.

    Args:
        mcp_tool: The MCP tool to wrap
        connection: MCP server connection configuration
        name_prefix: Optional prefix for tool name
        visibility: Tool visibility setting
        allow_users: List of allowed user IDs

    Returns:
        MCPToolAdapter instance
    """
    return MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection=connection,
        name_prefix=name_prefix,
        visibility=visibility,
        allow_users=allow_users,
    )
