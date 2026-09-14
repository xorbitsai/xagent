"""
API Tool for xagent
HTTP client for making arbitrary API calls with support for various auth methods
"""

import json
import logging
from typing import Any, Dict, Mapping, Optional, Type, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ...core.api_tool import (
    APIClientCore,
    append_known_connector_domain_hint,
    has_auth_credentials,
    match_known_connector_domain,
)
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
        Useful for integrating with external services, APIs, and webhooks.
        This tool has no stored credential for any external service. A 401/403 from a host with its own dedicated MCP connector (e.g. HubSpot, Slack, GitHub) comes back with a hint to use that connector's tools instead - pass your own credential via auth_type/auth_token, a header, or a query parameter if you have one."""

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

        # Make API call - api_key_query logic is now handled in core client.
        # Always actually attempted, even against a domain a dedicated MCP
        # connector also covers: a domain-based preflight block was tried
        # first here and reverted (see git history) because it couldn't
        # distinguish a request that needs that connector's credential from
        # one that doesn't need any - e.g. GitHub's public, credential-less
        # repo-read endpoints, which must still succeed normally. Only a
        # REAL 401/403 this call actually received gets annotated below.
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

        if result.get("status_code") in (401, 403):
            self._hint_known_connector_if_uncredentialed(result, api_args)

        return APICallResult.model_validate(result).model_dump()

    def _hint_known_connector_if_uncredentialed(
        self, result: Dict[str, Any], api_args: APICallArgs
    ) -> None:
        """Append a hint to `result["error"]` in place when a request that
        already received a real 401/403 targeted a domain covered by a
        dedicated MCP connector and carried no credential this tool
        recognizes - see the HubSpot deal-write incident this exists for
        (reads worked via the real connector throughout; an unauthenticated
        raw fallback call's 401 got diagnosed as "connection broken").
        Purely informational: it never affects whether the call was made or
        what status_code/body it got, so a public, credential-less endpoint
        on the same host that returns 2xx is entirely unaffected.
        """
        # The domain that actually produced this response - not necessarily
        # api_args.url's host, if a redirect crossed hosts along the way
        # (allow_redirects defaults to True). call_api's only path to a real
        # 401/403 always populates final_url from the real response, so this
        # is never missing here in practice; api_args.url is kept only as a
        # defensive fallback, not because it's expected to be used.
        response_url = result.get("final_url") or api_args.url
        connector_label = match_known_connector_domain(
            urlparse(response_url).hostname or ""
        )
        if not connector_label:
            return
        # Whether a credential actually reached the connector host - not
        # necessarily the same as what the caller originally attached.
        # httpx strips the Authorization header (and a Location with no
        # query string drops any api_key_query credential) on a
        # cross-origin redirect, so a request that started out
        # credentialed can still land on the connector host with nothing
        # attached. `final_request_has_credential` reflects the request
        # httpx actually sent, after any redirect; api_args is only a
        # fallback for a result predating that field (e.g. from a mocked
        # or older core client in tests).
        final_request_has_credential = result.get("final_request_has_credential")
        if final_request_has_credential is None:
            final_request_has_credential = has_auth_credentials(
                api_args.url,
                api_args.headers,
                api_args.params,
                api_args.auth_type,
                api_args.auth_token,
            )
        if final_request_has_credential:
            return
        result["error"] = append_known_connector_domain_hint(
            result.get("error"), connector_label
        )
        logger.info(
            f"ℹ️ API Call {api_args.method} {api_args.url} got "
            f"{result['status_code']} with no recognized credential - "
            f"hinting at the {connector_label} connector"
        )

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
