"""Custom API Tool Adapter for Agent System."""

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Type

from pydantic import BaseModel, Field, model_validator

from ....utils.encryption import decrypt_value
from ...core.api_tool import call_api
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .connector_runtime import (
    MISSING_RUNTIME_VALUE,
    RUNTIME_INPUT_CONTEXT,
    RUNTIME_INPUT_SECRETS,
    TARGET_BODY_FIELD,
    TARGET_HEADERS,
    ConnectorRuntimeError,
    binding_source_value,
    binding_target,
    connector_runtime_from_config,
    is_runtime_header_scalar,
    runtime_bindings_from_config,
)
from .tool_naming_limits import MAX_AGENT_TOOL_NAME_LENGTH

logger = logging.getLogger(__name__)


class _CustomApiToolArgsBase(BaseModel):
    """Shared arguments for Custom API tools."""

    url: Optional[str] = Field(
        default=None,
        description="The full URL to call. Omit this when the Custom API has a configured endpoint. You can use variables like $SECRET_KEY in the URL.",
    )
    method: Optional[str] = Field(
        default=None,
        description="HTTP method (GET, POST, PUT, DELETE, etc.). Omit this to use the Custom API configured method.",
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Query parameters."
    )

    @model_validator(mode="before")
    @classmethod
    def parse_string_to_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field in ["headers", "params", "body"]:
                val = data.get(field)
                if isinstance(val, str):
                    try:
                        data[field] = json.loads(val)
                    except json.JSONDecodeError:
                        pass
        return data


class CustomApiToolArgs(_CustomApiToolArgsBase):
    """Arguments for Custom API Tool."""

    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="HTTP headers. You can use variables like $SECRET_KEY in the header values.",
    )
    body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON body for the request. You can use variables like $SECRET_KEY in string values.",
    )


class _CustomApiToolArgsWithoutHeaders(_CustomApiToolArgsBase):
    body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON body for the request. You can use variables like $SECRET_KEY in string values.",
    )


class _CustomApiToolArgsWithoutBody(_CustomApiToolArgsBase):
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="HTTP headers. You can use variables like $SECRET_KEY in the header values.",
    )


class _CustomApiToolArgsWithoutHeadersAndBody(_CustomApiToolArgsBase):
    pass


_CUSTOM_API_ARGS_BY_RUNTIME_VISIBILITY: dict[tuple[bool, bool], Type[BaseModel]] = {
    (False, False): CustomApiToolArgs,
    (True, False): _CustomApiToolArgsWithoutHeaders,
    (False, True): _CustomApiToolArgsWithoutBody,
    (True, True): _CustomApiToolArgsWithoutHeadersAndBody,
}


class CustomApiToolResult(BaseModel):
    """Result of Custom API execution."""

    success: bool = Field(description="Whether the API call was successful")
    status_code: int = Field(description="HTTP status code")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Response headers"
    )
    body: Optional[Any] = Field(
        default=None, description="Response body (JSON or text)"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class CustomApiTool(AbstractBaseTool):
    """
    A generic API tool created from a Custom API configuration.
    It automatically replaces environment variables (secrets) in the request parameters.
    """

    category = ToolCategory.OTHER

    def __init__(
        self,
        name: str,
        description: str,
        env: Dict[str, str],
        url: Optional[str] = None,
        method: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        runtime_bindings: Optional[List[Dict[str, Any]]] = None,
        connector_runtime: Optional[Dict[str, Any]] = None,
        allow_delegated_authorization: bool = False,
        visibility: ToolVisibility = ToolVisibility.PUBLIC,
    ):
        # Format name for LLM (replace spaces/dashes with underscores)
        sanitized_name = name.replace(" ", "_").replace("-", "_")
        # `name` is free-form user input (CustomApiCreate.name only bounds
        # length, not charset), so anything the two replacements above
        # didn't catch -- a `.` in a domain-like name, punctuation, etc. --
        # would otherwise survive into `function.name` and 400 the whole
        # LLM call the same way an unsanitized MCP tool name does.
        sanitized_name = re.sub(r"[^A-Za-z0-9_-]", "_", sanitized_name)
        # Ensure name doesn't start with api_ twice if already prefixed
        prefix = "" if sanitized_name.startswith("api_") else "api_"
        suffix = "_call"
        # Same failure mode as the character check above -- an over-long
        # name is rejected by the same providers, just on length instead of
        # charset. Budget the truncation around the fixed prefix/suffix so
        # `_call` (the part that signals "this is an API call") survives
        # instead of being cut off by a blind truncation of the whole
        # string. `MAX_AGENT_TOOL_NAME_LENGTH` is this repo's own record of
        # that provider limit (tool_naming_limits.py), shared here rather
        # than redeclared so the two adapters can't drift apart on the number.
        budget = MAX_AGENT_TOOL_NAME_LENGTH - len(prefix) - len(suffix)
        if len(sanitized_name) > budget:
            # `name` is a free-form, DB-unique string with no server
            # prefix to squeeze the way the MCP adapter does -- the whole
            # sanitized name is what a truncation would cut. Two long,
            # descriptive Custom API names that share a common prefix and
            # differ only in a trailing qualifier (a normal naming habit,
            # e.g. two services named for the same team) truncate to the
            # same string otherwise, and nothing downstream detects that
            # collision -- mix in a short hash of the *original* name so a
            # truncated name stays distinct instead of silently dispatching
            # to whichever of the two tools registers first.
            digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
            hash_suffix = f"_{digest}"
            sanitized_name = sanitized_name[: budget - len(hash_suffix)] + hash_suffix
        self._name = f"{prefix}{sanitized_name[:budget]}{suffix}"

        # Structured originating-server identity, normalized once through the
        # shared SSOT. A scoped mcp:<server> selector fronts this Custom-API
        # wrapper, so server-scoped selection matches on this field by equality
        # instead of re-parsing ``api_<server>_call``.
        from .selection_spec import normalize_mcp_server_name

        self.source_server = normalize_mcp_server_name(name)

        default_info = ""
        if url:
            default_info += f"\nConfigured endpoint: {url}"
        if method:
            default_info += f"\nConfigured method: {method}"
        if headers:
            default_info += "\nConfigured headers are applied automatically."
        if body:
            default_info += f"\nConfigured body template: {body}"

        # Add env vars info to description so LLM knows how to use them
        env_info = ""
        if env:
            env_info = "\n\nAvailable Secrets (use them as $SECRET_NAME in url, headers, or body):\n"
            for k in env.keys():
                env_info += f"- {k}\n"

        self._description = f"Custom API: {name}\n{description}{default_info}{env_info}"
        self._default_url = url
        self._default_method = method or "GET"
        self._default_headers = headers or {}
        self._default_body = body
        self._runtime_bindings = runtime_bindings_from_config(
            {"runtime_bindings": runtime_bindings}
        )
        self._connector_runtime = connector_runtime_from_config(
            {"connector_runtime": connector_runtime}
        )
        self._allow_delegated_authorization = bool(allow_delegated_authorization)
        self._env = {}
        self._env_patterns = []
        for k, v in (env or {}).items():
            decrypted_v = decrypt_value(v)
            self._env[k] = decrypted_v
            # Pre-compile regex for this key to optimize recursive replacement
            pattern = re.compile(rf"\${{{re.escape(k)}}}|\${re.escape(k)}(?!\w)")
            self._env_patterns.append((pattern, decrypted_v))
        self._visibility = visibility

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tags(self) -> List[str]:
        return ["api", "custom", "http"]

    def args_type(self) -> Type[BaseModel]:
        hidden_headers, hidden_body = self._runtime_hidden_arg_fields()
        return _CUSTOM_API_ARGS_BY_RUNTIME_VISIBILITY[(hidden_headers, hidden_body)]

    def return_type(self) -> Type[BaseModel]:
        return CustomApiToolResult

    def state_type(self) -> Optional[Type[BaseModel]]:
        return None

    def is_async(self) -> bool:
        return True

    def _replace_secrets(self, value: Any) -> Any:
        """Recursively replace $SECRET_NAME in strings."""
        if isinstance(value, str):
            for pattern, v in self._env_patterns:
                value = pattern.sub(v, value)
            return value
        elif isinstance(value, dict):
            return {k: self._replace_secrets(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._replace_secrets(v) for v in value]
        return value

    def _runtime_hidden_arg_fields(self) -> tuple[bool, bool]:
        hidden_headers = False
        hidden_body = False
        for binding in self._runtime_bindings:
            target = binding_target(binding)
            target_type = target.get("target_type")
            if target_type == TARGET_HEADERS:
                hidden_headers = True
            elif target_type == TARGET_BODY_FIELD:
                hidden_body = True
        return hidden_headers, hidden_body

    def sanitize_tool_args_for_trace(self, args: Mapping[str, Any]) -> Dict[str, Any]:
        sanitized = dict(args)
        hidden_headers, hidden_body = self._runtime_hidden_arg_fields()
        if hidden_headers and "headers" in sanitized:
            logger.warning(
                "Runtime Custom API header binding ignores caller-supplied "
                "headers for tool %s",
                self._name,
            )
            sanitized.pop("headers", None)
        if hidden_body and "body" in sanitized:
            logger.warning(
                "Runtime Custom API body binding ignores caller-supplied body "
                "for tool %s",
                self._name,
            )
            sanitized.pop("body", None)
        return sanitized

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        try:
            parsed_args = CustomApiToolArgs(**args)

            # Replace secrets
            url = self._replace_secrets(parsed_args.url or self._default_url)
            if not url:
                return CustomApiToolResult(
                    success=False,
                    status_code=0,
                    headers={},
                    body=None,
                    error="URL is required because this Custom API has no configured endpoint.",
                ).model_dump()

            merged_headers = dict(self._default_headers)
            if parsed_args.headers:
                merged_headers.update(parsed_args.headers)
            runtime_headers = self._runtime_headers()
            for header_name in runtime_headers:
                if header_name in merged_headers:
                    logger.warning(
                        "Runtime Custom API header binding overrides caller/static "
                        "header %s for tool %s",
                        header_name,
                        self._name,
                    )
            merged_headers.update(runtime_headers)
            headers = self._replace_secrets(merged_headers) if merged_headers else {}
            params = (
                self._replace_secrets(parsed_args.params) if parsed_args.params else {}
            )

            body = None
            if parsed_args.body:
                body = self._replace_secrets(parsed_args.body)
            elif self._default_body:
                # Parse default body string if it exists
                try:
                    body = self._replace_secrets(json.loads(self._default_body))
                except json.JSONDecodeError:
                    body = self._replace_secrets(self._default_body)
            body = self._apply_runtime_body_fields(body)

            # Execute API call
            result = await call_api(
                url=url,
                method=parsed_args.method or self._default_method,
                headers=headers,
                params=params,
                body=body,
            )

            if not result.get("success"):
                logger.warning(f"Custom API {self._name} failed: {result.get('error')}")

            return CustomApiToolResult(
                success=result.get("success", False),
                status_code=result.get("status_code", 0),
                headers=result.get("headers", {}),
                body=result.get("body"),
                error=result.get("error"),
            ).model_dump()

        except Exception as e:
            logger.error(
                "Error executing Custom API %s with %s",
                self._name,
                type(e).__name__,
            )
            return CustomApiToolResult(
                success=False,
                status_code=0,
                headers={},
                body=None,
                error="Error executing Custom API.",
            ).model_dump()

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            raise RuntimeError(
                f"Event loop is already running. Use run_json_async instead for tool '{self.name}'."
            )

        return asyncio.run(self.run_json_async(args))

    def _runtime_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for binding in self._runtime_bindings:
            target = binding_target(binding)
            if target.get("target_type") != TARGET_HEADERS:
                continue
            header_name = target.get("key")
            if not isinstance(header_name, str) or not header_name:
                continue
            if (
                header_name.lower() == "authorization"
                and not self._allow_delegated_authorization
            ):
                logger.warning(
                    "Ignoring runtime Authorization header binding for tool %s "
                    "because delegated authorization is disabled",
                    self._name,
                )
                continue
            value = binding_source_value(
                binding,
                self._connector_runtime,
                allowed_input_types={RUNTIME_INPUT_CONTEXT, RUNTIME_INPUT_SECRETS},
            )
            if value is MISSING_RUNTIME_VALUE:
                continue
            if not is_runtime_header_scalar(value):
                logger.warning(
                    "Ignoring non-scalar runtime header binding %s for tool %s",
                    header_name,
                    self._name,
                )
                continue
            headers[header_name] = str(value)
        return headers

    def _apply_runtime_body_fields(self, body: Any) -> Any:
        runtime_fields = self._runtime_body_fields()
        if not runtime_fields:
            return body
        if not isinstance(body, dict):
            if body not in (None, ""):
                logger.warning(
                    "Runtime Custom API body bindings discard non-object body "
                    "for tool %s",
                    self._name,
                )
            body = {}
        merged_body = dict(body)
        for path, value in runtime_fields.items():
            if _dot_path_exists(merged_body, path):
                logger.warning(
                    "Runtime Custom API body binding overrides caller/static "
                    "body field %s for tool %s",
                    path,
                    self._name,
                )
            _set_dot_path(merged_body, path, value, tool_name=self._name)
        return merged_body

    def _runtime_body_fields(self) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        for binding in self._runtime_bindings:
            target = binding_target(binding)
            if target.get("target_type") != TARGET_BODY_FIELD:
                continue
            path = target.get("path")
            if not isinstance(path, str) or not _is_simple_dot_path(path):
                continue
            value = binding_source_value(
                binding,
                self._connector_runtime,
                allowed_input_types={RUNTIME_INPUT_CONTEXT},
            )
            if value is not MISSING_RUNTIME_VALUE:
                fields[path] = value
        return fields

    async def save_state_json(self) -> Mapping[str, Any]:
        return {}

    async def load_state_json(self, state: Mapping[str, Any]) -> None:
        pass

    def return_value_as_string(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)


def create_custom_api_tools(configs: List[Dict[str, Any]]) -> List[CustomApiTool]:
    """Create CustomApiTool instances from configs."""
    tools = []
    for config in configs:
        try:
            name = config.get("name", "custom_api")
            desc = config.get("description", "")
            env = config.get("env", {})
            url = config.get("url")
            method = config.get("method")
            headers = config.get("headers")
            body = config.get("body")

            tool = CustomApiTool(
                name=name,
                description=desc,
                env=env,
                url=url,
                method=method,
                headers=headers,
                body=body,
                runtime_bindings=config.get("runtime_bindings"),
                connector_runtime=config.get("connector_runtime"),
                allow_delegated_authorization=bool(
                    config.get("allow_delegated_authorization", False)
                ),
            )
            tools.append(tool)
        except ConnectorRuntimeError:
            raise
        except Exception as e:
            logger.error(
                "Failed to create Custom API tool %s: %s",
                config.get("name", "custom_api"),
                e,
            )
    return tools


def _is_simple_dot_path(path: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", path))


def _dot_path_exists(value: Mapping[str, Any], path: str) -> bool:
    current: Any = value
    parts = path.split(".")
    for part in parts:
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _set_dot_path(
    value: Dict[str, Any], path: str, field_value: Any, *, tool_name: str
) -> None:
    current = value
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            if part in current:
                logger.warning(
                    "Runtime Custom API body binding overrides non-object "
                    "intermediate field %s while setting %s for tool %s",
                    part,
                    path,
                    tool_name,
                )
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = field_value
