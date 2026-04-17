"""Custom API Tool Adapter for Agent System."""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Type

from pydantic import BaseModel, Field

from ....utils.encryption import decrypt_value
from ...core.api_tool import call_api
from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)


class CustomApiToolArgs(BaseModel):
    """Arguments for Custom API Tool."""

    url: str = Field(
        description="The full URL to call, e.g., 'https://api.example.com/v1/users'. You can use variables like $SECRET_KEY in the URL."
    )
    method: str = Field(
        default="GET", description="HTTP method (GET, POST, PUT, DELETE, etc.)"
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="HTTP headers. You can use variables like $SECRET_KEY in the header values.",
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Query parameters."
    )
    body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON body for the request. You can use variables like $SECRET_KEY in string values.",
    )


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
        visibility: ToolVisibility = ToolVisibility.PUBLIC,
    ):
        # Format name for LLM (replace spaces/dashes with underscores)
        sanitized_name = name.replace(" ", "_").replace("-", "_")
        # Ensure name doesn't start with api_ twice if already prefixed
        if sanitized_name.startswith("api_"):
            self._name = f"{sanitized_name}_call"
        else:
            self._name = f"api_{sanitized_name}_call"

        # Add env vars info to description so LLM knows how to use them
        env_info = ""
        if env:
            env_info = "\n\nAvailable Secrets (use them as $SECRET_NAME in url, headers, or body):\n"
            for k in env.keys():
                env_info += f"- {k}\n"

        self._description = f"Custom API: {name}\n{description}{env_info}"
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
        return CustomApiToolArgs

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

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        try:
            parsed_args = CustomApiToolArgs(**args)

            # Replace secrets
            url = self._replace_secrets(parsed_args.url)
            headers = (
                self._replace_secrets(parsed_args.headers)
                if parsed_args.headers
                else {}
            )
            params = (
                self._replace_secrets(parsed_args.params) if parsed_args.params else {}
            )
            body = self._replace_secrets(parsed_args.body) if parsed_args.body else None

            # Execute API call
            result = await call_api(
                url=url,
                method=parsed_args.method,
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
            logger.error(f"Error executing Custom API {self._name}: {e}")
            return CustomApiToolResult(
                success=False, status_code=0, headers={}, body=None, error=str(e)
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

            tool = CustomApiTool(name=name, description=desc, env=env)
            tools.append(tool)
        except Exception as e:
            logger.error(f"Failed to create Custom API tool for config {config}: {e}")
    return tools
