"""MCP Tool Adapter for Agent System.

This module provides adapters to convert MCP tools into Agent system Tool format,
enabling MCP tools to be used in DAG plan-execute patterns and other agent workflows.
"""

import asyncio
import inspect
import logging
import os
import re
import weakref
from collections.abc import Coroutine, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar, Union, cast

import httpx
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, Field, create_model

from ..... import config as _root_config
from .....sandbox.base import Sandbox
from ...core.mcp.sessions import Connection, create_session
from ...core.mcp.tools import load_mcp_tools
from .base import AbstractBaseTool, ToolVisibility
from .connector_runtime import (
    ERROR_DELEGATED_AUTHORIZATION_FAILED,
    MISSING_RUNTIME_VALUE,
    RUNTIME_INPUT_CONTEXT,
    TARGET_MCP_META,
    TARGET_TOOL_ARGUMENTS,
    binding_source_value,
    binding_target,
    connector_runtime_from_config,
    runtime_bindings_from_config,
)
from .sandboxed_tool.sandboxed_mcp_tool_helper import (
    load_sandboxed_mcp_tools,
    should_sandbox_mcp_connection,
)


class MCPFailurePhase(str, Enum):
    """Public-safe phase where an MCP server failed to load."""

    SESSION_START = "session_start"
    INITIALIZE = "initialize"
    LIST_TOOLS = "list_tools"
    ADAPTER_CONSTRUCTION = "adapter_construction"
    SANDBOX_LIST_TOOLS = "sandbox_list_tools"
    SANDBOX_TOOL_WRAP = "sandbox_tool_wrap"
    NO_TOOLS_RETURNED = "no_tools_returned"


_DEFAULT_UNAVAILABLE_MCP_MESSAGE = "MCP server credentials are unavailable."
_MCP_LOAD_FAILURE_MESSAGES: dict[MCPFailurePhase, str] = {
    MCPFailurePhase.SESSION_START: "MCP server could not be started.",
    MCPFailurePhase.INITIALIZE: "MCP server initialization failed.",
    MCPFailurePhase.LIST_TOOLS: "MCP server tools could not be loaded.",
    MCPFailurePhase.ADAPTER_CONSTRUCTION: (
        "Some MCP server tools could not be prepared."
    ),
    MCPFailurePhase.SANDBOX_LIST_TOOLS: "MCP server tools could not be loaded.",
    MCPFailurePhase.SANDBOX_TOOL_WRAP: ("Some MCP server tools could not be prepared."),
    MCPFailurePhase.NO_TOOLS_RETURNED: "MCP server returned no available tools.",
}


def mcp_load_failure_message(phase: MCPFailurePhase) -> str:
    """Return the public-safe message owned by an MCP load failure phase."""
    return _MCP_LOAD_FAILURE_MESSAGES[phase]


@dataclass(frozen=True)
class MCPServerLoadFailure:
    """Safe MCP load failure data that excludes raw exception details."""

    server_name: str
    phase: MCPFailurePhase
    error_type: str | None
    attempts: int = 1


@dataclass(frozen=True)
class MCPLoadResult:
    """Structured outcome for loading one or more MCP servers."""

    tools: tuple[AbstractBaseTool, ...]
    loaded_servers: tuple[str, ...]
    failures: tuple[MCPServerLoadFailure, ...]


class EmptyArgsModel(BaseModel):
    pass


logger = logging.getLogger(__name__)
_RUNTIME_CONNECTION_REFRESH_KEY = "_connector_runtime_refresh"
_OAUTH_TOKEN_RESOLVER_REFRESH_KEY = "_oauth_token_resolver_refresh"
_RESOLVER_HTTP_401_NODE_LIMIT = 64
_HTTP_401_TEXT_RE = re.compile(
    r"\b(?:http(?:\s+status)?|status(?:\s+code)?|response|code)\s*[:=]?\s*401\b|"
    r"\b401\s+unauthorized\b",
    re.IGNORECASE,
)


def _bounded_exception_nodes(
    exc: BaseException, *, excluded_subtree_ids: frozenset[int] = frozenset()
) -> Iterator[BaseException]:
    pending = [(exc, True)]
    visited: set[int] = set()
    visited_count = 0
    while pending and visited_count < _RESOLVER_HTTP_401_NODE_LIMIT:
        current, is_root = pending.pop()
        current_id = id(current)
        if current_id in visited or (
            current_id in excluded_subtree_ids and not is_root
        ):
            continue
        visited.add(current_id)
        visited_count += 1
        yield current

        if current_id in excluded_subtree_ids:
            continue
        linked: list[BaseException] = []
        if isinstance(current, BaseExceptionGroup):
            linked.extend(current.exceptions)
        if isinstance(current.__cause__, BaseException):
            linked.append(current.__cause__)
        if isinstance(current.__context__, BaseException):
            linked.append(current.__context__)
        pending.extend((node, False) for node in reversed(linked))


def _strict_http_401_responses(
    exc: BaseException,
    *,
    excluded_response_ids: frozenset[int] = frozenset(),
    excluded_subtree_ids: frozenset[int] = frozenset(),
) -> Iterator[httpx.Response]:
    for current in _bounded_exception_nodes(
        exc, excluded_subtree_ids=excluded_subtree_ids
    ):
        if not isinstance(current, httpx.HTTPStatusError):
            continue
        response = current.response
        if (
            isinstance(response, httpx.Response)
            and response.status_code == 401
            and id(response) not in excluded_response_ids
        ):
            yield response


def _resolver_401_evidence(exc: BaseException) -> tuple[Any | None, frozenset[int]]:
    # Lazy import keeps the core adapter independent from the web layer at import time.
    from .....web.services.mcp_oauth import parse_www_authenticate_bearer

    challenge = None
    response_ids: set[int] = set()
    for response in _strict_http_401_responses(exc):
        response_ids.add(id(response))
        if challenge is not None:
            continue
        candidate = parse_www_authenticate_bearer(
            response.headers.get_list("WWW-Authenticate")
        )
        if candidate is not None and candidate.params.get("error") == "invalid_token":
            challenge = candidate
    return challenge, frozenset(response_ids)


def _resolver_invalid_token_challenge(exc: BaseException) -> Any | None:
    challenge, _ = _resolver_401_evidence(exc)
    return challenge


def _is_executable_remote_connection(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    transport = value.get("transport")
    if transport not in {"sse", "streamable_http", "websocket"}:
        return False
    url = value.get("url")
    return isinstance(url, str) and bool(url)


def _exception_indicates_http_401(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return any(_exception_indicates_http_401(sub_exc) for sub_exc in exc.exceptions)
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if value == 401 or value == "401":
            return True
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code == 401 or status_code == "401":
            return True
    text = str(exc).lower()
    return bool(_HTTP_401_TEXT_RE.search(text))


def _delegated_authorization_failed_result(
    *, failure_code: object = None
) -> dict[str, Any]:
    from ....agent.result import normalize_tool_failure_code

    result: dict[str, Any] = {
        "content": [
            {
                "text": (
                    "Error executing MCP tool: delegated authorization failed "
                    f"({ERROR_DELEGATED_AUTHORIZATION_FAILED})"
                )
            }
        ],
        "is_error": True,
    }
    normalized_failure_code = normalize_tool_failure_code(failure_code)
    if normalized_failure_code is not None:
        result["failure_code"] = normalized_failure_code
    return result


def _delegated_retry_failed_result() -> dict[str, Any]:
    return {
        "content": [
            {"text": ("Error executing MCP tool after delegated authorization retry.")}
        ],
        "is_error": True,
    }


def _normalize_concurrent_tools(value: Any) -> list[str]:
    """Normalize raw MCP tool-name allowlists from server config."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _connection_concurrency_config(
    connection: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    return (
        bool(connection.get("concurrency_safe", False)),
        _normalize_concurrent_tools(connection.get("concurrent_tools")),
    )


def _mcp_tool_is_concurrency_safe(
    tool_name: str, *, concurrency_safe: bool, concurrent_tools: list[str]
) -> bool:
    if not concurrency_safe:
        return False
    if not concurrent_tools:
        return True
    return tool_name in set(concurrent_tools)


def _get_current_mcp_user_id() -> Optional[str]:
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


def _is_mcp_user_allowed(
    user_id: Optional[str], allow_users: Optional[List[str]]
) -> bool:
    if not user_id:
        # If no user ID, this might be a system execution. For security, deny
        # access unless the tool explicitly allows the system identity.
        return allow_users is None or "system" in allow_users

    if allow_users is None:
        return True

    return user_id in allow_users


def _mcp_access_denied_result(user_id: Optional[str], tool_name: str) -> dict[str, Any]:
    error_msg = f"User {user_id} is not authorized to use tool {tool_name}"
    logger.warning(error_msg)
    return {
        "content": [{"text": f"Access denied: {error_msg}"}],
        "is_error": True,
    }


def _mcp_return_value_as_string(value: Any) -> str:
    try:
        if isinstance(value, dict):
            content = value.get("content", [])
            if isinstance(content, list) and content:
                texts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    else:
                        texts.append(str(item))
                return "\n".join(texts)
            if content:
                return str(content)
            return "No content returned"
        return str(value)
    except Exception as e:
        logger.warning(f"Failed to convert return value to string: {e}")
        return str(value)


def _format_unavailable_mcp_tool_name(server_name: str, server_id: Any | None) -> str:
    from .selection_spec import normalize_mcp_server_name

    normalized_server = re.sub(
        r"[^A-Za-z0-9_]+", "_", normalize_mcp_server_name(server_name)
    ).strip("_")
    parts = ["mcp"]
    if normalized_server:
        parts.append(normalized_server)
    if server_id is not None:
        normalized_id = re.sub(r"[^A-Za-z0-9_]+", "_", str(server_id)).strip("_")
        if normalized_id:
            parts.append(normalized_id)
    parts.append("unavailable")
    name = "_".join(parts)
    return name if name != "mcp_unavailable" else "mcp_server_unavailable"


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
        source_server: Optional[str] = None,
        concurrency_safe: bool = False,
        concurrent_tools: Optional[List[str]] = None,
    ):
        """Initialize MCP tool adapter.

        Args:
            mcp_tool: The MCP tool to wrap
            connection: MCP server connection configuration
            name_prefix: Optional prefix for tool name (e.g., "mcp_")
            visibility: Tool visibility setting
            allow_users: List of allowed user IDs
            source_server: Normalized identity of the originating MCP server
                (``normalize_mcp_server_name``), surfaced on
                ``metadata.source_server`` so server-scoped selection matches
                by structured equality rather than re-parsing the tool name.
            concurrency_safe: Whether the server operator guarantees these MCP
                tools are both concurrency-safe and idempotent when retried
                after interruption.
            concurrent_tools: Optional allowlist of raw MCP tool names. Empty
                means every tool from an opted-in server is safe.
        """
        self.mcp_tool = mcp_tool
        self.connection = connection
        self._name_prefix = name_prefix or ""
        self._visibility = visibility or ToolVisibility.PRIVATE
        self._allow_users = allow_users
        self.source_server = source_server
        self.concurrency_safe = _mcp_tool_is_concurrency_safe(
            self.mcp_tool.name,
            concurrency_safe=concurrency_safe,
            concurrent_tools=_normalize_concurrent_tools(concurrent_tools),
        )
        runtime_config = connection if isinstance(connection, Mapping) else {}
        self._runtime_bindings = runtime_bindings_from_config(runtime_config)
        self._connector_runtime = connector_runtime_from_config(runtime_config)
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
            runtime_bound_args = self._runtime_bound_tool_argument_names(properties)

            for field_name, field_schema in properties.items():
                if field_name in runtime_bound_args:
                    continue
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
                return _mcp_access_denied_result(current_user_id, self.mcp_tool.name)

            # Validate arguments
            normalized_args = self._normalize_args_by_schema(args)
            runtime_bound_args = self._runtime_bound_tool_argument_names(
                self._input_schema_properties()
            )
            for field_name in runtime_bound_args:
                if field_name in normalized_args:
                    logger.warning(
                        "Ignoring LLM-supplied runtime-bound MCP argument "
                        "%s for tool %s",
                        field_name,
                        self.mcp_tool.name,
                    )
                    normalized_args.pop(field_name, None)
            parsed_args = self._args_type(**normalized_args)
            tool_args = parsed_args.model_dump(exclude_none=True)
            tool_args.update(self._runtime_tool_arguments())
            tool_meta = self._runtime_mcp_meta()

            logger.debug(
                "Executing MCP tool %s with args keys: %s for user %s",
                self.mcp_tool.name,
                sorted(tool_args),
                current_user_id,
            )

            # Set user context for execution
            # Lazy import to avoid core → web layer dependency at module level.
            from .....web.user_context import UserContext

            user_context = UserContext(current_user_id)

            with user_context.set_context():
                try:
                    return await self._execute_mcp_call(
                        self.connection, tool_args, tool_meta
                    )
                except (BaseExceptionGroup, Exception) as exc:
                    retry_result = await self._retry_after_authorization_failure(
                        exc, tool_args, tool_meta
                    )
                    if retry_result is not None:
                        return retry_result
                    raise

        except BaseExceptionGroup as e:
            logger.error(
                "MCP tool %s execution failed with exception group %s",
                self.mcp_tool.name,
                type(e).__name__,
            )
            return {
                "content": [{"text": "Error executing MCP tool."}],
                "is_error": True,
            }

        except Exception as e:
            logger.error(
                "MCP tool %s execution failed with %s",
                self.mcp_tool.name,
                type(e).__name__,
            )
            return {
                "content": [{"text": "Error executing MCP tool."}],
                "is_error": True,
            }

    async def _execute_mcp_call(
        self,
        connection: Connection,
        tool_args: Mapping[str, Any],
        tool_meta: Mapping[str, Any],
    ) -> dict[str, Any]:
        async with create_session(connection) as session:
            await session.initialize()
            result = await session.call_tool(
                self.mcp_tool.name,
                dict(tool_args),
                meta=dict(tool_meta) or None,
            )

            content = []
            if result.content:
                for content_item in result.content:
                    if hasattr(content_item, "model_dump"):
                        content.append(content_item.model_dump())
                    else:
                        content.append({"text": str(content_item)})

            return {
                "content": content,
                "is_error": result.isError if hasattr(result, "isError") else False,
            }

    async def _retry_after_authorization_failure(
        self,
        exc: BaseException,
        tool_args: Mapping[str, Any],
        tool_meta: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if isinstance(self.connection, Mapping) and (
            _OAUTH_TOKEN_RESOLVER_REFRESH_KEY in self.connection
        ):
            return await self._retry_resolver_401(exc, tool_args, tool_meta)

        if not _exception_indicates_http_401(exc):
            return None
        if not isinstance(self.connection, Mapping):
            return None
        refresh = self.connection.get(_RUNTIME_CONNECTION_REFRESH_KEY)
        if not callable(refresh):
            return None

        logger.info(
            "Retrying MCP tool %s after delegated authorization failure",
            self.mcp_tool.name,
        )
        refreshed = refresh()
        if inspect.isawaitable(refreshed):
            refreshed = await refreshed
        if not isinstance(refreshed, dict):
            return _delegated_authorization_failed_result()
        try:
            return await self._execute_mcp_call(
                cast(Connection, refreshed), tool_args, tool_meta
            )
        except (BaseExceptionGroup, Exception) as retry_exc:
            if _exception_indicates_http_401(retry_exc):
                return _delegated_authorization_failed_result()
            logger.error(
                "MCP tool %s delegated authorization retry failed with %s",
                self.mcp_tool.name,
                type(retry_exc).__name__,
            )
            return _delegated_retry_failed_result()

    async def _retry_resolver_401(
        self,
        exc: BaseException,
        tool_args: Mapping[str, Any],
        tool_meta: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        challenge, initial_response_ids = _resolver_401_evidence(exc)
        if not initial_response_ids:
            return None

        refresh = self.connection.get(_OAUTH_TOKEN_RESOLVER_REFRESH_KEY)
        if challenge is None or not callable(refresh):
            return _delegated_authorization_failed_result()

        logger.info(
            "Retrying MCP tool %s after resolver authorization failure",
            self.mcp_tool.name,
        )
        try:
            refreshed = refresh(challenge)
            if inspect.isawaitable(refreshed):
                refreshed = await refreshed
        except (BaseExceptionGroup, Exception) as refresh_exc:
            logger.error(
                "MCP tool %s resolver authorization refresh failed with %s",
                self.mcp_tool.name,
                type(refresh_exc).__name__,
            )
            return _delegated_authorization_failed_result()

        from ....agent.result import ClassifiedToolFailure

        if isinstance(refreshed, ClassifiedToolFailure):
            return _delegated_authorization_failed_result(
                failure_code=refreshed.failure_code
            )

        if not _is_executable_remote_connection(refreshed):
            return _delegated_authorization_failed_result()

        try:
            return await self._execute_mcp_call(
                cast(Connection, refreshed), tool_args, tool_meta
            )
        except (BaseExceptionGroup, Exception) as retry_exc:
            excluded_response_ids = (
                frozenset() if retry_exc is exc else initial_response_ids
            )
            if (
                next(
                    _strict_http_401_responses(
                        retry_exc,
                        excluded_response_ids=excluded_response_ids,
                        excluded_subtree_ids=frozenset({id(exc)}),
                    ),
                    None,
                )
                is not None
            ):
                return _delegated_authorization_failed_result()
            logger.error(
                "MCP tool %s resolver authorization retry failed with %s",
                self.mcp_tool.name,
                type(retry_exc).__name__,
            )
            return _delegated_retry_failed_result()

    def _get_current_user_id(self) -> Optional[str]:
        """Get current user ID from environment or context."""
        return _get_current_mcp_user_id()

    def _input_schema_properties(self) -> dict[str, Any]:
        schema = self.mcp_tool.inputSchema
        if not isinstance(schema, dict):
            return {}
        properties = schema.get("properties")
        return properties if isinstance(properties, dict) else {}

    def _runtime_bound_tool_argument_names(
        self, properties: Mapping[str, Any]
    ) -> set[str]:
        bound: set[str] = set()
        for binding in self._runtime_bindings:
            target = binding_target(binding)
            if target.get("target_type") != TARGET_TOOL_ARGUMENTS:
                continue
            target_key = target.get("key")
            if isinstance(target_key, str) and target_key in properties:
                bound.add(target_key)
        return bound

    def _runtime_tool_arguments(self) -> dict[str, Any]:
        properties = self._input_schema_properties()
        runtime_args: dict[str, Any] = {}
        for binding in self._runtime_bindings:
            target = binding_target(binding)
            if target.get("target_type") != TARGET_TOOL_ARGUMENTS:
                continue
            target_key = target.get("key")
            if not isinstance(target_key, str) or target_key not in properties:
                continue
            value = binding_source_value(
                binding,
                self._connector_runtime,
                allowed_input_types={RUNTIME_INPUT_CONTEXT},
            )
            if value is MISSING_RUNTIME_VALUE:
                logger.warning(
                    "Skipping runtime MCP tool argument binding for missing "
                    "context source while setting %s on tool %s",
                    target_key,
                    self.mcp_tool.name,
                )
                continue
            runtime_args[target_key] = value
        return runtime_args

    def _runtime_mcp_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for binding in self._runtime_bindings:
            target = binding_target(binding)
            if target.get("target_type") != TARGET_MCP_META:
                continue
            target_key = target.get("key")
            if not isinstance(target_key, str) or not target_key:
                continue
            value = binding_source_value(
                binding,
                self._connector_runtime,
                allowed_input_types={RUNTIME_INPUT_CONTEXT},
            )
            if value is not MISSING_RUNTIME_VALUE:
                meta[target_key] = value
        return meta

    def _is_user_allowed(self, user_id: Optional[str]) -> bool:
        """Check if user is allowed to use this tool."""
        return _is_mcp_user_allowed(user_id, self._allow_users)

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
        return _mcp_return_value_as_string(value)


class _UnavailableMCPToolResult(BaseModel):
    success: bool = Field(default=False, description="Whether execution succeeded")
    status: str = Field(default="error", description="Tool execution status")
    error: str | None = Field(
        default=None, description="Public-safe tool failure message when available"
    )
    failure_code: str | None = Field(
        default=None, description="Allowlisted public tool failure classification"
    )
    reason: str | None = Field(
        default=None, description="Public-safe MCP unavailability reason"
    )
    content: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tool execution result content"
    )
    is_error: bool = Field(
        default=True,
        description="Whether the tool execution resulted in an error",
    )


class UnavailableMCPTool(AbstractBaseTool):
    """Server-level MCP tool returned when a selected server is unavailable."""

    read_only = True
    concurrency_safe = True

    def __init__(
        self,
        *,
        server_name: str,
        server_id: Any | None,
        allow_users: Optional[List[str]] = None,
        failure_code: str | None = None,
        reason: str | None = None,
        message: str = _DEFAULT_UNAVAILABLE_MCP_MESSAGE,
    ) -> None:
        from ....agent.result import normalize_tool_failure_code
        from .base import ToolCategory
        from .selection_spec import normalize_mcp_server_name

        self._server_name = server_name
        self._server_id = server_id
        self._allow_users = allow_users
        self._failure_code = normalize_tool_failure_code(failure_code)
        self._reason = reason
        self._message = message
        self._name = _format_unavailable_mcp_tool_name(server_name, server_id)
        self.source_server = normalize_mcp_server_name(server_name)
        self.category = ToolCategory.MCP

    @property
    def name(self) -> str:
        return self._name

    @property
    def server_name(self) -> str:
        """Public server identity used by strict setup diagnostics."""
        return self._server_name

    @property
    def unavailability_reason(self) -> str | None:
        """Public-safe reason code used by strict setup diagnostics."""
        return self._reason

    @property
    def description(self) -> str:
        return self._message

    @property
    def tags(self) -> List[str]:
        return ["mcp"]

    def args_type(self) -> Type[BaseModel]:
        return EmptyArgsModel

    def return_type(self) -> Type[BaseModel]:
        return _UnavailableMCPToolResult

    def state_type(self) -> Optional[Type[BaseModel]]:
        return None

    def _run_unavailable(self) -> Dict[str, Any]:
        current_user_id = _get_current_mcp_user_id()
        if not _is_mcp_user_allowed(current_user_id, self._allow_users):
            return _mcp_access_denied_result(current_user_id, self.name)
        content_message = self._message
        if self._message == _DEFAULT_UNAVAILABLE_MCP_MESSAGE:
            content_message = (
                "MCP server credentials are unavailable. Please reconnect "
                "the MCP server credentials and retry."
            )
        result: Dict[str, Any] = {
            "success": False,
            "status": "error",
            "error": self._message,
            "content": [{"text": content_message}],
            "is_error": True,
        }
        if self._reason is not None:
            result["reason"] = self._reason
        if self._failure_code is not None:
            result["failure_code"] = self._failure_code
        return result

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return self._run_unavailable()

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        return self._run_unavailable()

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        pass

    def return_value_as_string(self, value: Any) -> str:
        return _mcp_return_value_as_string(value)


def _build_mcp_tool_adapter(
    server_name: str,
    connection: Connection,
    mcp_tool: MCPTool,
    *,
    name_prefix: str = "mcp_",
    visibility: Optional[ToolVisibility] = None,
    allow_users: Optional[List[str]] = None,
    concurrency_safe: bool = False,
    concurrent_tools: Optional[List[str]] = None,
) -> MCPToolAdapter:
    """Create MCP tool adapter."""
    # Create tool name with server prefix
    tool_prefix = f"{name_prefix}{server_name}_" if name_prefix else f"{server_name}_"

    # Carry the originating server identity as structured metadata, normalized
    # once here through the same SSOT the selector parse / config filter use,
    # so server-scoped selection matches by equality (no tool-name re-parse).
    from .selection_spec import normalize_mcp_server_name

    return MCPToolAdapter(
        mcp_tool=mcp_tool,
        connection=connection,
        name_prefix=tool_prefix,
        visibility=visibility,
        allow_users=allow_users,
        source_server=normalize_mcp_server_name(server_name),
        concurrency_safe=concurrency_safe,
        concurrent_tools=concurrent_tools,
    )


async def _load_direct_mcp_tools(
    server_name: str,
    connection: Connection,
    *,
    name_prefix: str,
    visibility: Optional[ToolVisibility],
    allow_users: Optional[List[str]],
) -> MCPLoadResult:
    """Load MCP tools directly on the host."""
    agent_tools: list[AbstractBaseTool] = []
    mcp_tools: list[MCPTool] = []
    transport = connection.get("transport", "")
    non_retryable = {"oauth", "unknown"}
    max_attempts = 1 if transport in non_retryable else 3
    concurrency_safe, concurrent_tools = _connection_concurrency_config(connection)
    failure_phase = MCPFailurePhase.SESSION_START
    error_type: str | None = None

    for attempt in range(max_attempts):
        current_phase = MCPFailurePhase.SESSION_START
        try:
            async with create_session(connection) as session:
                current_phase = MCPFailurePhase.INITIALIZE
                await session.initialize()
                # Use the shared loader to keep pagination behavior consistent.
                current_phase = MCPFailurePhase.LIST_TOOLS
                mcp_tools = await load_mcp_tools(session)
            break
        except Exception as e:
            failure_phase = current_phase
            error_type = type(e).__name__
            if attempt < max_attempts - 1:
                logger.warning(
                    "Attempt %d failed to load tools from MCP server %s during "
                    "%s (%s); retrying",
                    attempt + 1,
                    server_name,
                    current_phase.value,
                    error_type,
                )
                await asyncio.sleep(1)
    else:
        return MCPLoadResult(
            tools=(),
            loaded_servers=(),
            failures=(
                MCPServerLoadFailure(
                    server_name=server_name,
                    phase=failure_phase,
                    error_type=error_type,
                    attempts=max_attempts,
                ),
            ),
        )

    if not mcp_tools:
        return MCPLoadResult(
            tools=(),
            loaded_servers=(),
            failures=(
                MCPServerLoadFailure(
                    server_name=server_name,
                    phase=MCPFailurePhase.NO_TOOLS_RETURNED,
                    error_type=None,
                ),
            ),
        )

    adapter_error_type: str | None = None
    for mcp_tool in mcp_tools:
        try:
            adapter = _build_mcp_tool_adapter(
                server_name,
                connection,
                mcp_tool,
                name_prefix=name_prefix,
                visibility=visibility,
                allow_users=allow_users,
                concurrency_safe=concurrency_safe,
                concurrent_tools=concurrent_tools,
            )

            agent_tools.append(adapter)
            logger.debug(f"Created adapter for tool: {adapter.name}")

        except Exception as e:
            adapter_error_type = adapter_error_type or type(e).__name__
            logger.error(
                "Failed to create adapter for MCP tool %s from server %s (%s)",
                mcp_tool.name,
                server_name,
                type(e).__name__,
            )
            continue

    failures: tuple[MCPServerLoadFailure, ...] = ()
    if adapter_error_type is not None:
        failures = (
            MCPServerLoadFailure(
                server_name=server_name,
                phase=MCPFailurePhase.ADAPTER_CONSTRUCTION,
                error_type=adapter_error_type,
            ),
        )

    return MCPLoadResult(
        tools=tuple(agent_tools),
        loaded_servers=(server_name,) if agent_tools else (),
        failures=failures,
    )


# Hard cap on concurrent (including abandoned) initializations per server.
# The timeout below bounds the CALLER's wait but not the underlying task:
# a cancellation-resistant cleanup keeps its transport/socket alive after
# the caller has moved on. Without a per-server bound, a burst of tasks
# against one hung server accumulates abandoned loads (and CLOSE-WAIT
# sockets) without limit — the gate slot is only released when the load
# task actually finishes, so abandoned loads keep counting against the cap
# and later callers fail fast instead of opening yet another transport.
_MAX_INFLIGHT_LOADS_PER_SERVER = 4

# Semaphores are bound to an event loop; key by loop (weakly, so a
# discarded loop doesn't pin its gates) then by server name. Web and
# Celery processes each get their own gates.
_server_load_gates: "weakref.WeakKeyDictionary[Any, Dict[str, asyncio.Semaphore]]" = (
    weakref.WeakKeyDictionary()
)


# Strong references to in-flight/abandoned load tasks. The event loop only
# keeps weak references to tasks; if GC collected a still-pending task its
# done-callback — which releases the gate slot — would never fire. Tasks
# remove themselves on completion.
_active_load_tasks: "set[asyncio.Task[Any]]" = set()
_BoundedLoadResult = TypeVar("_BoundedLoadResult")


def _get_server_load_gate(server_name: str) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    gates = _server_load_gates.get(loop)
    if gates is None:
        gates = {}
        _server_load_gates[loop] = gates
    gate = gates.get(server_name)
    if gate is None:
        gate = asyncio.Semaphore(_MAX_INFLIGHT_LOADS_PER_SERVER)
        gates[server_name] = gate
    return gate


async def _load_server_tools_bounded(
    server_name: str,
    load_coro: "Coroutine[Any, Any, _BoundedLoadResult]",
    timeout_seconds: int,
) -> _BoundedLoadResult:
    """Run one server's tool load with a hard wall-clock bound and a
    per-server in-flight cap.

    ``asyncio.wait_for`` alone is not a hard bound: it awaits the cancelled
    task's cleanup, and a hung streamable-HTTP server can stall inside the
    session context manager's ``__aexit__`` just as easily as inside
    ``initialize()`` (issue #889). ``asyncio.wait`` + fire-and-forget cancel
    guarantees the caller resumes at the deadline even if cleanup never
    completes; the abandoned task is logged and its eventual exception is
    consumed by a done-callback so it never surfaces as "exception was never
    retrieved".

    The per-server gate bounds the resource side: at most
    ``_MAX_INFLIGHT_LOADS_PER_SERVER`` load tasks (live or abandoned) exist
    per server per event loop. Callers that cannot get a slot within their
    timeout budget fail fast without creating a transport. The acquire and
    the load share one deadline, so the end-to-end caller bound is
    unchanged.
    """
    gate = _get_server_load_gate(server_name)

    if timeout_seconds <= 0:
        # Timeout disabled: still bound the fan-out, waiting as long as
        # needed for a slot. Cancellation while waiting must close the
        # never-started coroutine or it warns "was never awaited" at GC.
        try:
            await gate.acquire()
        except asyncio.CancelledError:
            load_coro.close()
            raise
        try:
            return await load_coro
        finally:
            gate.release()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    try:
        await asyncio.wait_for(gate.acquire(), timeout_seconds)
    except (asyncio.TimeoutError, TimeoutError):
        # Never started: close the coroutine so it doesn't warn about
        # being un-awaited, and don't touch the gate (nothing acquired).
        load_coro.close()
        raise TimeoutError(
            f"MCP server {server_name}: no initialization slot freed within "
            f"{timeout_seconds}s ({_MAX_INFLIGHT_LOADS_PER_SERVER} loads "
            "already in flight, possibly abandoned by earlier timeouts); "
            "skipping without opening another connection. Slots free when "
            "those loads finish; if the server is permanently hung they "
            "recover only on process restart"
        ) from None
    except asyncio.CancelledError:
        # Caller cancelled while queued at the gate: the load never
        # started, so just close the coroutine and let the cancel out.
        load_coro.close()
        raise

    task = asyncio.ensure_future(load_coro)
    # Keep a strong reference until completion, then release the slot only
    # when the task truly finishes — an abandoned (cancellation-resistant)
    # load keeps counting against the cap.
    _active_load_tasks.add(task)

    def _on_task_done(t: "asyncio.Task[Any]") -> None:
        _active_load_tasks.discard(t)
        gate.release()

    task.add_done_callback(_on_task_done)

    def _consume_result(t: "asyncio.Task[Any]") -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.debug(
                "Abandoned MCP load task for server %s finished with: %s",
                server_name,
                exc,
            )

    remaining = max(0.0, deadline - loop.time())
    try:
        done, _pending = await asyncio.wait({task}, timeout=remaining)
    except asyncio.CancelledError:
        # Caller cancelled (run cancelled, lease lost): ``asyncio.wait``
        # does not cancel its awaited tasks, so propagate the cancel to
        # the load task ourselves or it would run — and hold its gate
        # slot and transport — forever. A well-behaved load unwinds and
        # frees the slot; a cancellation-resistant cleanup keeps its slot
        # until it truly finishes, same as the timeout path.
        task.cancel()
        task.add_done_callback(_consume_result)
        raise

    if task in done:
        return task.result()

    task.cancel()
    task.add_done_callback(_consume_result)
    raise TimeoutError(
        f"MCP server {server_name} initialization timed out after "
        f"{timeout_seconds}s (including cleanup); abandoning it"
    )


async def load_mcp_tools_as_agent_tools(
    connection_map: Dict[str, Connection],
    *,
    name_prefix: str = "mcp_",
    visibility: Optional[ToolVisibility] = None,
    allow_users: Optional[List[str]] = None,
    sandbox: Sandbox | None = None,
) -> MCPLoadResult:
    """Load MCP tools from multiple servers and convert to Agent tools.

    Args:
        connection_map: Map of server names to connection configurations
        name_prefix: Prefix for tool names (default: "mcp_")
        visibility: Tool visibility setting
        allow_users: List of allowed user IDs
        sandbox: Optional sandbox instance. When provided, stdio connections
            using npx/uvx will be routed through the sandbox for isolation.

    Returns:
        Structured MCP tools, loaded server names, and public-safe failures.

    Notes:
        Failures loading tools from individual MCP servers are preserved while
        the function continues processing remaining servers.
    """
    agent_tools: list[AbstractBaseTool] = []
    loaded_servers: list[str] = []
    failures: list[MCPServerLoadFailure] = []
    timeout_seconds = _root_config.get_mcp_tool_init_timeout_seconds()

    for server_name, connection in connection_map.items():
        try:
            logger.info(f"Loading tools from MCP server: {server_name}")
            if sandbox is not None and should_sandbox_mcp_connection(connection):
                concurrency_safe, concurrent_tools = _connection_concurrency_config(
                    connection
                )

                def tool_builder(
                    mcp_tool: MCPTool,
                    _server_name: str = server_name,
                    _connection: Connection = connection,
                    _concurrency_safe: bool = concurrency_safe,
                    _concurrent_tools: list[str] = concurrent_tools,
                ) -> MCPToolAdapter:
                    return _build_mcp_tool_adapter(
                        _server_name,
                        _connection,
                        mcp_tool,
                        name_prefix=name_prefix,
                        visibility=visibility,
                        allow_users=allow_users,
                        concurrency_safe=_concurrency_safe,
                        concurrent_tools=_concurrent_tools,
                    )

                try:
                    sandbox_result = await _load_server_tools_bounded(
                        server_name,
                        load_sandboxed_mcp_tools(
                            connection,
                            sandbox,
                            tool_builder,
                        ),
                        timeout_seconds,
                    )
                except Exception as e:
                    error_type = type(e).__name__
                    logger.error(
                        "Failed to list sandboxed MCP tools from server %s (%s)",
                        server_name,
                        error_type,
                    )
                    failures.append(
                        MCPServerLoadFailure(
                            server_name=server_name,
                            phase=MCPFailurePhase.SANDBOX_LIST_TOOLS,
                            error_type=error_type,
                        )
                    )
                    continue

                server_tools = sandbox_result.tools
                if sandbox_result.adapter_error_types:
                    failures.append(
                        MCPServerLoadFailure(
                            server_name=server_name,
                            phase=MCPFailurePhase.ADAPTER_CONSTRUCTION,
                            error_type=sandbox_result.adapter_error_types[0],
                        )
                    )
                if sandbox_result.wrap_error_types:
                    failures.append(
                        MCPServerLoadFailure(
                            server_name=server_name,
                            phase=MCPFailurePhase.SANDBOX_TOOL_WRAP,
                            error_type=sandbox_result.wrap_error_types[0],
                        )
                    )
                if (
                    not server_tools
                    and not sandbox_result.adapter_error_types
                    and not sandbox_result.wrap_error_types
                ):
                    failures.append(
                        MCPServerLoadFailure(
                            server_name=server_name,
                            phase=MCPFailurePhase.NO_TOOLS_RETURNED,
                            error_type=None,
                        )
                    )
            else:
                direct_result = await _load_server_tools_bounded(
                    server_name,
                    _load_direct_mcp_tools(
                        server_name,
                        connection,
                        name_prefix=name_prefix,
                        visibility=visibility,
                        allow_users=allow_users,
                    ),
                    timeout_seconds,
                )
                server_tools = direct_result.tools
                failures.extend(direct_result.failures)

            agent_tools.extend(server_tools)
            if server_tools:
                loaded_servers.append(server_name)
            logger.info(f"Found {len(server_tools)} tools from server {server_name}")

        except Exception as e:
            error_type = type(e).__name__
            failure_phase = (
                MCPFailurePhase.INITIALIZE
                if isinstance(e, TimeoutError)
                else MCPFailurePhase.SESSION_START
            )
            logger.error(
                "Unexpected failure loading tools from MCP server %s (%s)",
                server_name,
                error_type,
            )
            failures.append(
                MCPServerLoadFailure(
                    server_name=server_name,
                    phase=failure_phase,
                    error_type=error_type,
                )
            )
            continue

    logger.info(
        "Loaded %d MCP tools from %d servers with %d server failures",
        len(agent_tools),
        len(loaded_servers),
        len(failures),
    )
    return MCPLoadResult(
        tools=tuple(agent_tools),
        loaded_servers=tuple(loaded_servers),
        failures=tuple(failures),
    )
