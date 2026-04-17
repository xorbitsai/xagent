"""
API Tool - HTTP Client for making arbitrary API calls

Supports various HTTP methods, authentication, headers, and error handling.
"""

import base64
import json
import logging
import os
from typing import Any, Dict, Optional, Union
from urllib.parse import urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)


class APIClientCore:
    """Core API client for making HTTP requests"""

    def __init__(
        self,
        default_timeout: int = 30,
        max_response_size: int = 10 * 1024 * 1024,  # 10MB
        default_retry_count: int = 3,
    ):
        """
        Initialize API client.

        Args:
            default_timeout: Default request timeout in seconds
            max_response_size: Maximum response size in bytes
            default_retry_count: Default number of retries on failure
        """
        self.default_timeout = default_timeout
        self.max_response_size = max_response_size
        self.default_retry_count = default_retry_count

    async def call_api(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Union[Dict[str, Any], str]] = None,
        auth_type: Optional[str] = None,
        auth_token: Optional[str] = None,
        api_key_param: str = "api_key",
        timeout: Optional[int] = None,
        retry_count: Optional[int] = None,
        allow_redirects: bool = True,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to an API.

        Args:
            url: Target URL
            method: HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
            headers: Request headers
            params: Query parameters
            body: Request body (dict for JSON, str for raw)
            auth_type: Authentication type ('bearer', 'basic', 'api_key', 'api_key_query')
            auth_token: Authentication token/credentials (for 'basic' auth, use "username:password" format)
            api_key_param: Parameter name for API key when using 'api_key_query' auth
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
            allow_redirects: Whether to follow redirects

        Returns:
            Dictionary with success status, status_code, headers, body, and error
        """
        logger.info(
            f"🌐 API Call: {method} {url}"
            + (f" (auth: {auth_type})" if auth_type else "")
        )

        # Validate URL
        if not self._is_valid_url(url):
            return {
                "success": False,
                "status_code": 0,
                "headers": {},
                "body": None,
                "error": f"Invalid URL: {url}",
            }

        # Prepare request
        method = method.upper()
        timeout = timeout or self.default_timeout
        retry_count = (
            retry_count if retry_count is not None else self.default_retry_count
        )
        # Ensure retry_count is non-negative to avoid empty range
        retry_count = max(0, retry_count)

        # Handle API key in query parameters
        request_params: Dict[str, Any] = dict(params) if params else {}
        if auth_type == "api_key_query" and auth_token:
            request_params[api_key_param] = auth_token

        final_request_params: Optional[Dict[str, Any]] = request_params

        # Merge params directly into URL to prevent httpx from stripping existing query strings
        if final_request_params:
            url = str(httpx.URL(url).copy_merge_params(final_request_params))
            final_request_params = None

        # If request_params is empty, set to None
        if not final_request_params:
            final_request_params = None

        # Prepare headers
        request_headers = self._prepare_headers(headers, auth_type, auth_token, body)

        # Prepare body
        request_body = self._prepare_body(body, request_headers)

        # Get proxy configuration
        proxy_url = self._get_proxy_url()

        # Attempt request with retries
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                result = await self._make_request(
                    url=url,
                    method=method,
                    headers=request_headers,
                    params=final_request_params,
                    data=request_body,
                    timeout=timeout,
                    proxy_url=proxy_url,
                    allow_redirects=allow_redirects,
                )
                logger.info(
                    f"✅ API Call successful: {method} {url} -> {result['status_code']}"
                )
                return result

            except Exception as e:
                last_error = e
                if attempt < retry_count:
                    logger.warning(
                        f"⚠️ API Call failed (attempt {attempt + 1}/{retry_count + 1}): {str(e)}"
                    )
                else:
                    logger.error(
                        f"❌ API Call failed after {retry_count + 1} attempts: {str(e)}"
                    )

        # All retries failed
        return {
            "success": False,
            "status_code": 0,
            "headers": {},
            "body": None,
            "error": f"Request failed after {retry_count + 1} attempts: {str(last_error)}",
        }

    async def _make_request(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]],
        data: Optional[Union[str, bytes]],
        timeout: int,
        proxy_url: Optional[str],
        allow_redirects: bool,
    ) -> Dict[str, Any]:
        """Make the actual HTTP request with streaming to limit download size"""
        client_kwargs: Dict[str, Any] = {"timeout": timeout}
        if proxy_url:
            client_kwargs["proxy"] = proxy_url
            logger.debug(f"   Using proxy: {proxy_url}")

        async with httpx.AsyncClient(**client_kwargs) as client:
            # Use streaming to limit download size
            async with client.stream(
                method=method,
                url=url,
                headers=headers,
                params=params,
                content=data,
                follow_redirects=allow_redirects,
            ) as response:
                # Check response size by reading only up to max_response_size
                response_size = 0
                content_chunks = []
                async for chunk in response.aiter_bytes():
                    response_size += len(chunk)
                    if response_size > self.max_response_size:
                        logger.warning(
                            f"⚠️ Response size exceeds limit ({self.max_response_size} bytes), aborting download"
                        )
                        return {
                            "success": False,
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "body": None,
                            "error": f"Response too large (exceeds {self.max_response_size} bytes), download aborted",
                        }
                    content_chunks.append(chunk)

                response_content = b"".join(content_chunks)

                # Parse response body
                body = self._parse_response_body_from_content(
                    response_content, dict(response.headers)
                )

                return {
                    "success": 200 <= response.status_code < 300,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": body,
                    "error": None
                    if 200 <= response.status_code < 300
                    else f"HTTP {response.status_code}",
                }

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and scheme"""
        try:
            result = urlparse(url)
            if result.scheme not in ("http", "https"):
                logger.error(f"❌ Invalid URL scheme: {result.scheme}")
                return False
            if not result.netloc:
                logger.error("❌ Invalid URL: missing network location")
                return False
            return True
        except Exception as e:
            logger.error(f"❌ URL parsing failed: {str(e)}")
            return False

    def _prepare_headers(
        self,
        headers: Optional[Dict[str, str]],
        auth_type: Optional[str],
        auth_token: Optional[str],
        body: Optional[Union[Dict[str, Any], str]],
    ) -> Dict[str, str]:
        """Prepare request headers with authentication"""
        request_headers = {}

        # Default headers
        request_headers["User-Agent"] = "Xagent-API-Tool/1.0"
        request_headers["Accept"] = "application/json"

        # Add custom headers
        if headers:
            request_headers.update(headers)

        # Add authentication
        if auth_type and auth_token:
            auth_type_lower = auth_type.lower()
            if auth_type_lower == "bearer":
                request_headers["Authorization"] = f"Bearer {auth_token}"
            elif auth_type_lower == "basic":
                credentials = base64.b64encode(auth_token.encode()).decode()
                request_headers["Authorization"] = f"Basic {credentials}"
            elif auth_type_lower == "api_key":
                # Default to X-API-Key header
                if "x-api-key" not in [h.lower() for h in request_headers.keys()]:
                    request_headers["X-API-Key"] = auth_token

        # Set Content-Type for body
        if body and "content-type" not in [h.lower() for h in request_headers.keys()]:
            if isinstance(body, dict):
                request_headers["Content-Type"] = "application/json"

        return request_headers

    def _prepare_body(
        self, body: Optional[Union[Dict[str, Any], str]], headers: Dict[str, str]
    ) -> Optional[Union[str, bytes]]:
        """Prepare request body"""
        if body is None:
            return None

        if isinstance(body, dict):
            # Check content type case-insensitively
            content_type = ""
            for key, value in headers.items():
                if key.lower() == "content-type":
                    content_type = value
                    break

            if "application/json" in content_type:
                return json.dumps(body)
            else:
                # For dict with non-JSON content type, convert to form data
                # Use urlencode to properly escape special characters
                return urlencode(body)
        elif isinstance(body, str):
            return body
        else:
            return str(body)

    def _parse_response_body_from_content(
        self, content: bytes, headers: Dict[str, str]
    ) -> Any:
        """Parse response body from raw content based on content type"""
        if not content:
            return None

        content_type = headers.get("content-type", "").lower()

        # Try to determine encoding from content-type header
        encoding = "utf-8"
        if "charset=" in content_type:
            try:
                encoding = content_type.split("charset=")[-1].split(";")[0].strip()
            except Exception:
                pass

        try:
            if "application/json" in content_type:
                return json.loads(content.decode(encoding))
            else:
                # Return text for non-JSON responses
                return content.decode(encoding)
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse response body: {str(e)}")
            # Try to return text as fallback with ignored errors
            try:
                return content.decode(encoding, errors="ignore")
            except Exception:
                return None

    def _get_proxy_url(self) -> Optional[str]:
        """Get proxy URL from environment variables"""
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        return https_proxy or http_proxy


# Convenience function for direct usage
async def call_api(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Union[Dict[str, Any], str]] = None,
    auth_type: Optional[str] = None,
    auth_token: Optional[str] = None,
    api_key_param: str = "api_key",
    timeout: Optional[int] = None,
    retry_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Make an HTTP request to an API.

    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE, PATCH, etc.)
        headers: Request headers
        params: Query parameters
        body: Request body (dict for JSON, str for raw)
        auth_type: Authentication type ('bearer', 'basic', 'api_key', 'api_key_query')
        auth_token: Authentication token/credentials (for 'basic' auth, use "username:password" format)
        api_key_param: Parameter name for API key when using 'api_key_query' auth
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure

    Returns:
        Dictionary with success status, status_code, headers, body, and error

    Example:
        >>> # GET request
        >>> result = await call_api("https://api.example.com/data")

        >>> # POST request with JSON body
        >>> result = await call_api(
        ...     "https://api.example.com/users",
        ...     method="POST",
        ...     body={"name": "John", "email": "john@example.com"},
        ...     auth_type="bearer",
        ...     auth_token="your-token"
        ... )

        >>> # GET with query parameters
        >>> result = await call_api(
        ...     "https://api.example.com/search",
        ...     params={"q": "test", "limit": 10}
        ... )
    """
    client = APIClientCore()
    return await client.call_api(
        url=url,
        method=method,
        headers=headers,
        params=params,
        body=body,
        auth_type=auth_type,
        auth_token=auth_token,
        api_key_param=api_key_param,
        timeout=timeout,
        retry_count=retry_count,
    )
