"""MCP Tool Adapter for Agent System.

This module provides adapters to convert MCP tools into Agent system Tool format,
enabling MCP tools to be used in DAG plan-execute patterns and other agent workflows.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

from mcp.types import Tool as MCPTool
from pydantic import BaseModel, Field, create_model

from .....sandbox.base import Sandbox
from ...core.mcp.sessions import Connection, create_session
from ...core.mcp.tools import load_mcp_tools
from .base import AbstractBaseTool, ToolVisibility
from .sandboxed_tool.sandboxed_mcp_tool_helper import (
    load_sandboxed_mcp_tools,
    should_sandbox_mcp_connection,
)


class EmptyArgsModel(BaseModel):
    pass


logger = logging.getLogger(__name__)


def _format_exception_group_messages(exc: BaseExceptionGroup) -> str:
    """Flatten nested exception groups into a readable error string."""
    messages: list[str] = []

    def _collect(group: BaseExceptionGroup) -> None:
        for sub_exc in group.exceptions:
            if isinstance(sub_exc, BaseExceptionGroup):
                _collect(sub_exc)
            else:
                messages.append(str(sub_exc))

    _collect(exc)
    if not messages:
        return str(exc)
    return f"{exc}: " + ", ".join(messages)


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
        """Convert JSON schema type to a Python type for Pydantic model creation."""
        if not isinstance(schema, dict):
            return Any

        for union_key in ("anyOf", "oneOf"):
            options = schema.get(union_key)
            if isinstance(options, list) and options:
                non_null_options = [
                    option for option in options if not self._is_null_schema(option)
                ]
                if len(non_null_options) == 1:
                    return self._json_schema_to_python_type(non_null_options[0])
                resolved_types: list[Type[Any]] = []
                for option in non_null_options:
                    resolved_type = self._json_schema_to_python_type(option)
                    if resolved_type is not Any and resolved_type not in resolved_types:
                        resolved_types.append(resolved_type)
                return self._build_union_type(resolved_types)

        all_of = schema.get("allOf")
        if isinstance(all_of, list) and all_of:
            for option in all_of:
                resolved_type = self._json_schema_to_python_type(option)
                if resolved_type is not Any:
                    return resolved_type

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            concrete_types = [item for item in schema_type if item != "null"]
            concrete_resolved_types: list[Type[Any]] = []
            for concrete_type in concrete_types:
                resolved_type = self._json_schema_to_python_type(
                    {"type": concrete_type}
                )
                if (
                    resolved_type is not Any
                    and resolved_type not in concrete_resolved_types
                ):
                    concrete_resolved_types.append(resolved_type)
            return self._build_union_type(concrete_resolved_types)

        if schema_type == "array":
            return list
        if schema_type == "object":
            return Dict[str, Any]
        if schema_type == "string":
            return str
        if schema_type == "integer":
            return int
        if schema_type == "number":
            return float
        if schema_type == "boolean":
            return bool
        return Any

    def _build_union_type(self, resolved_types: list[Type[Any]]) -> Type[Any]:
        """Build a runtime union for multiple candidate schema types."""
        if not resolved_types:
            return Any
        if len(resolved_types) == 1:
            return resolved_types[0]
        return cast(Type[Any], Union.__getitem__(tuple(resolved_types)))

    def _is_null_schema(self, schema: Any) -> bool:
        """Return True when the schema represents a JSON null type."""
        if not isinstance(schema, dict):
            return False
        schema_type = schema.get("type")
        if schema_type == "null":
            return True
        if isinstance(schema_type, list):
            return all(item == "null" for item in schema_type)
        return False

    def _normalize_args_by_schema(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        """Normalize common LLM argument shape mistakes using the MCP input schema."""
        normalized_args = dict(args)
        schema = self.mcp_tool.inputSchema
        if not isinstance(schema, dict):
            return normalized_args

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return normalized_args

        for field_name, field_schema in properties.items():
            if field_name not in normalized_args:
                continue
            value = normalized_args[field_name]
            if value is None:
                continue
            if self._schema_is_array_only(field_schema) and not isinstance(value, list):
                normalized_args[field_name] = [value]

        return normalized_args

    def _schema_accepts_array(self, schema: Any) -> bool:
        """Return True when a JSON schema allows array input."""
        if not isinstance(schema, dict):
            return False

        schema_type = schema.get("type")
        if schema_type == "array":
            return True
        if isinstance(schema_type, list) and "array" in schema_type:
            return True

        for composite_key in ("anyOf", "oneOf", "allOf"):
            variants = schema.get(composite_key)
            if isinstance(variants, list) and any(
                self._schema_accepts_array(variant) for variant in variants
            ):
                return True

        return False

    def _schema_is_array_only(self, schema: Any) -> bool:
        """Return True when array is the only accepted non-null JSON shape."""
        if not isinstance(schema, dict):
            return False

        schema_type = schema.get("type")
        if schema_type == "array":
            return True
        if isinstance(schema_type, list):
            concrete_types = [item for item in schema_type if item != "null"]
            return bool(concrete_types) and all(
                concrete_type == "array" for concrete_type in concrete_types
            )

        for union_key in ("anyOf", "oneOf"):
            options = schema.get(union_key)
            if isinstance(options, list) and options:
                non_null_options = [
                    option for option in options if not self._is_null_schema(option)
                ]
                return bool(non_null_options) and all(
                    self._schema_is_array_only(option) for option in non_null_options
                )

        return False

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
            normalized_args = self._normalize_args_by_schema(args)
            parsed_args = self._args_type(**normalized_args)
            tool_args = parsed_args.model_dump(exclude_none=True)

            logger.debug(
                f"Executing MCP tool {self.mcp_tool.name} with args: {tool_args} for user {current_user_id}"
            )

            # Set user context for execution
            # Lazy import to avoid core → web layer dependency at module level.
            from .....web.user_context import UserContext

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

        except BaseExceptionGroup as e:
            logger.error(
                f"MCP tool {self.mcp_tool.name} execution failed with exception group: {e}"
            )
            error_msg = _format_exception_group_messages(e)
            return {
                "content": [{"text": f"Error executing MCP tool: {error_msg}"}],
                "is_error": True,
            }

        except Exception as e:
            logger.error(f"MCP tool {self.mcp_tool.name} execution failed: {e}")
            return {
                "content": [{"text": f"Error executing MCP tool: {e}"}],
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


def _build_mcp_tool_adapter(
    server_name: str,
    connection: Connection,
    mcp_tool: MCPTool,
    *,
    name_prefix: str = "mcp_",
    visibility: Optional[ToolVisibility] = None,
    allow_users: Optional[List[str]] = None,
) -> MCPToolAdapter:
    """Create MCP tool adapter."""
    # Create tool name with server prefix
    tool_prefix = f"{name_prefix}{server_name}_" if name_prefix else f"{server_name}_"

    return MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection=connection,
        name_prefix=tool_prefix,
        visibility=visibility,
        allow_users=allow_users,
    )


async def _load_direct_mcp_tools(
    server_name: str,
    connection: Connection,
    *,
    name_prefix: str,
    visibility: Optional[ToolVisibility],
    allow_users: Optional[List[str]],
) -> list[AbstractBaseTool]:
    """Load MCP tools directly on the host."""
    agent_tools: list[AbstractBaseTool] = []
    mcp_tools: list[MCPTool] = []
    last_error: Exception = RuntimeError(f"Failed to load tools from {server_name}")
    transport = connection.get("transport", "")
    non_retryable = {"oauth", "unknown"}
    max_attempts = 1 if transport in non_retryable else 3

    for attempt in range(max_attempts):
        try:
            async with create_session(connection) as session:
                await session.initialize()
                # Use the shared loader to keep pagination behavior consistent.
                mcp_tools = await load_mcp_tools(session)
            break
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Attempt {attempt + 1} failed to load tools from MCP server {server_name}: {e}, retrying..."
                )
                await asyncio.sleep(1)
    else:
        raise last_error

    for mcp_tool in mcp_tools:
        try:
            adapter = _build_mcp_tool_adapter(
                server_name,
                connection,
                mcp_tool,
                name_prefix=name_prefix,
                visibility=visibility,
                allow_users=allow_users,
            )

            agent_tools.append(adapter)
            logger.debug(f"Created adapter for tool: {adapter.name}")

        except Exception as e:
            logger.error(f"Failed to create adapter for tool {mcp_tool.name}: {e}")
            continue

    return agent_tools


async def load_mcp_tools_as_agent_tools(
    connection_map: Dict[str, Connection],
    *,
    name_prefix: str = "mcp_",
    visibility: Optional[ToolVisibility] = None,
    allow_users: Optional[List[str]] = None,
    sandbox: Sandbox | None = None,
) -> List[AbstractBaseTool]:
    """Load MCP tools from multiple servers and convert to Agent tools.

    Args:
        connection_map: Map of server names to connection configurations
        name_prefix: Prefix for tool names (default: "mcp_")
        visibility: Tool visibility setting
        allow_users: List of allowed user IDs
        sandbox: Optional sandbox instance. When provided, stdio connections
            using npx/uvx will be routed through the sandbox for isolation.

    Returns:
        List of MCP-backed agent tools, including sandboxed wrappers when needed

    Notes:
        Failures loading tools from individual MCP servers are logged and skipped.
        The function continues processing remaining servers instead of raising.
    """
    agent_tools: List[AbstractBaseTool] = []

    for server_name, connection in connection_map.items():
        try:
            logger.info(f"Loading tools from MCP server: {server_name}")
            if sandbox is not None and should_sandbox_mcp_connection(connection):

                def tool_builder(
                    mcp_tool: MCPTool,
                    _server_name: str = server_name,
                    _connection: Connection = connection,
                ) -> MCPToolAdapter:
                    return _build_mcp_tool_adapter(
                        _server_name,
                        _connection,
                        mcp_tool,
                        name_prefix=name_prefix,
                        visibility=visibility,
                        allow_users=allow_users,
                    )

                server_tools = await load_sandboxed_mcp_tools(
                    connection,
                    sandbox,
                    tool_builder,
                )
            else:
                server_tools = await _load_direct_mcp_tools(
                    server_name,
                    connection,
                    name_prefix=name_prefix,
                    visibility=visibility,
                    allow_users=allow_users,
                )

            agent_tools.extend(server_tools)
            logger.info(f"Found {len(server_tools)} tools from server {server_name}")

        except Exception as e:
            logger.error(f"Failed to load tools from MCP server {server_name}: {e}")
            # Continue with other servers rather than failing completely
            continue

    logger.info(f"Successfully loaded {len(agent_tools)} MCP tools as Agent tools")
    return agent_tools
