"""
API Tool for xagent
HTTP client for making arbitrary API calls with support for various auth methods
"""

import json
import logging
from typing import Any, Dict, Mapping, Optional, Type, Union

from pydantic import BaseModel, Field

from ...core.api_tool import APIClientCore
from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)


class APICallArgs(BaseModel):
    url: str = Field(description="Target URL for the API call")
    method: str = Field(
        default="GET",
        description="HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="Request headers as key-value pairs"
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Query parameters as key-value pairs"
    )
    body: Optional[Union[Dict[str, Any], str]] = Field(
        default=None,
        description="Request body (dict for JSON, string for raw content)",
    )
    auth_type: Optional[str] = Field(
        default=None,
        description="Authentication type: 'bearer', 'basic', 'api_key', 'api_key_query'",
    )
    auth_token: Optional[str] = Field(
        default=None,
        description="Authentication token. For basic auth, use username:password format",
    )
    api_key_param: Optional[str] = Field(
        default="api_key",
        description="Parameter name for API key in query string (for api_key_query auth)",
    )
    timeout: Optional[int] = Field(
        default=None, description="Request timeout in seconds (default: 30)"
    )
    retry_count: Optional[int] = Field(
        default=None, description="Number of retries on failure (default: 3)"
    )
    allow_redirects: bool = Field(
        default=True, description="Whether to follow HTTP redirects"
    )


class APICallResult(BaseModel):
    success: bool = Field(description="Whether the API call was successful")
    status_code: int = Field(description="HTTP status code")
    headers: Dict[str, str] = Field(description="Response headers")
    body: Any = Field(description="Response body (parsed JSON or text)")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class APITool(AbstractBaseTool):
    """Framework wrapper for the API client tool"""

    category = ToolCategory.BASIC

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PUBLIC
        self._client = APIClientCore()

    @property
    def name(self) -> str:
        return "api_call"

    @property
    def description(self) -> str:
        return """Make HTTP requests to arbitrary APIs.
        Supports GET, POST, PUT, DELETE, PATCH methods with custom headers, body, and authentication.
        Authentication types: 'bearer' (Bearer token), 'basic' (Basic auth), 'api_key' (X-API-Key header), 'api_key_query' (API key in query params).
        Returns parsed JSON response or text content with status code and headers.
        Useful for integrating with external services, APIs, and webhooks."""

    @property
    def tags(self) -> list[str]:
        return ["api", "http", "rest", "web", "integration"]

    def args_type(self) -> Type[BaseModel]:
        return APICallArgs

    def return_type(self) -> Type[BaseModel]:
        return APICallResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("APITool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        api_args = APICallArgs.model_validate(args)

        # Make API call - api_key_query logic is now handled in core client
        result = await self._client.call_api(
            url=api_args.url,
            method=api_args.method,
            headers=api_args.headers,
            params=api_args.params,
            body=api_args.body,
            auth_type=api_args.auth_type,
            auth_token=api_args.auth_token,
            api_key_param=api_args.api_key_param or "api_key",
            timeout=api_args.timeout,
            retry_count=api_args.retry_count,
            allow_redirects=api_args.allow_redirects,
        )

        return APICallResult.model_validate(result).model_dump()

    def return_value_as_string(self, value: Any) -> str:
        """Format API response as readable string"""
        if isinstance(value, dict):
            if value.get("success"):
                body = value.get("body")
                if isinstance(body, (dict, list)):
                    body_str = json.dumps(body, indent=2, ensure_ascii=False)
                else:
                    body_str = str(body)
                return f"✅ API call successful (HTTP {value.get('status_code')})\n\nResponse:\n{body_str}"
            else:
                error = value.get("error", "Unknown error")
                return f"❌ API call failed: {error}"
        return str(value)
