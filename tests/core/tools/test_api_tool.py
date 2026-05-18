"""
Tests for API Tool
"""

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from xagent.core.tools.adapters.vibe.api_tool import APICallArgs, APITool
from xagent.core.tools.core.api_tool import APIClientCore, call_api


def _single_value_query_args(url: str) -> dict[str, str]:
    return {key: values[-1] for key, values in parse_qs(urlparse(url).query).items()}


@pytest.fixture
def mock_httpbin(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_make_request(
        self: APIClientCore,
        *,
        url: str,
        method: str,
        headers: dict[str, str],
        params: dict[str, Any] | None,
        data: str | bytes | None,
        timeout: int,
        proxy_url: str | None,
        allow_redirects: bool,
    ) -> dict[str, Any]:
        parsed = urlparse(url)
        path = parsed.path
        body: dict[str, Any]
        status_code = 200

        if path == "/get":
            body = {"args": _single_value_query_args(url)}
        elif path == "/post":
            parsed_data = json.loads(
                data.decode() if isinstance(data, bytes) else data or "{}"
            )
            body = {"json": parsed_data}
        elif path == "/bearer":
            token = headers.get("Authorization", "").removeprefix("Bearer ")
            body = {"authenticated": bool(token), "token": token}
        elif path == "/headers":
            body = {"headers": headers}
        elif path == "/status/404":
            status_code = 404
            body = {}
        else:
            status_code = 404
            body = {}

        return {
            "success": 200 <= status_code < 300,
            "status_code": status_code,
            "headers": {"content-type": "application/json"},
            "body": body,
            "error": None if 200 <= status_code < 300 else f"HTTP {status_code}",
        }

    monkeypatch.setattr(APIClientCore, "_make_request", mock_make_request)


class TestAPIClientCore:
    """Test core API client functionality"""

    @pytest.mark.asyncio
    async def test_get_request(self, mock_httpbin: None):
        """Test basic GET request"""
        client = APIClientCore()
        result = await client.call_api(
            url="https://httpbin.org/get",
            method="GET",
            params={"test": "value"},
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert "body" in result
        assert result["body"]["args"]["test"] == "value"

    @pytest.mark.asyncio
    async def test_post_json_request(self, mock_httpbin: None):
        """Test POST request with JSON body"""
        client = APIClientCore()
        result = await client.call_api(
            url="https://httpbin.org/post",
            method="POST",
            body={"name": "test", "value": 123},
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["body"]["json"]["name"] == "test"
        assert result["body"]["json"]["value"] == 123

    @pytest.mark.asyncio
    async def test_bearer_auth(self, mock_httpbin: None):
        """Test Bearer token authentication"""
        client = APIClientCore()
        result = await client.call_api(
            url="https://httpbin.org/bearer",
            method="GET",
            auth_type="bearer",
            auth_token="test-token-123",
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["body"]["authenticated"] is True
        assert result["body"]["token"] == "test-token-123"

    @pytest.mark.asyncio
    async def test_api_key_query_auth(self, mock_httpbin: None):
        """Test API key in query parameters using convenience function"""
        result = await call_api(
            url="https://httpbin.org/get",
            method="GET",
            auth_type="api_key_query",
            auth_token="test-key-123",
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        # Verify the api_key was added to query params
        assert result["body"]["args"]["api_key"] == "test-key-123"

    @pytest.mark.asyncio
    async def test_api_key_query_custom_param(self, mock_httpbin: None):
        """Test API key in custom query parameter"""
        result = await call_api(
            url="https://httpbin.org/get",
            method="GET",
            auth_type="api_key_query",
            auth_token="test-key-123",
            api_key_param="token",
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        # Verify the custom param was added to query params
        assert result["body"]["args"]["token"] == "test-key-123"

    @pytest.mark.asyncio
    async def test_custom_headers(self, mock_httpbin: None):
        """Test custom headers"""
        client = APIClientCore()
        result = await client.call_api(
            url="https://httpbin.org/headers",
            method="GET",
            headers={"X-Custom-Header": "custom-value"},
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert "X-Custom-Header" in result["body"]["headers"]

    @pytest.mark.asyncio
    async def test_invalid_url(self):
        """Test invalid URL handling"""
        client = APIClientCore()
        result = await client.call_api(url="not-a-valid-url")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, mock_httpbin: None):
        """Test retry mechanism on failure"""
        client = APIClientCore(default_retry_count=2)
        # This will fail with 404
        result = await client.call_api(url="https://httpbin.org/status/404")

        assert result["success"] is False
        assert result["status_code"] == 404


class TestAPITool:
    """Test APITool adapter"""

    def test_tool_metadata(self):
        """Test tool metadata"""
        tool = APITool()

        assert tool.name == "api_call"
        assert "HTTP requests to arbitrary APIs" in tool.description
        assert "api" in tool.tags
        assert "http" in tool.tags
        assert tool.category.value == "basic"

    def test_args_schema(self):
        """Test argument schema"""
        tool = APITool()
        args_type = tool.args_type()

        assert args_type == APICallArgs

        # Validate args
        args = APICallArgs.model_validate(
            {
                "url": "https://api.example.com",
                "method": "POST",
                "body": {"test": "value"},
            }
        )
        assert args.url == "https://api.example.com"
        assert args.method == "POST"
        assert args.body == {"test": "value"}

    @pytest.mark.asyncio
    async def test_api_call_execution(self, monkeypatch):
        """Test API tool wrapper delegates to client core"""
        tool = APITool()

        async def mock_call_api(**kwargs):
            assert kwargs["url"] == "https://httpbin.org/get"
            assert kwargs["method"] == "GET"
            assert kwargs["params"] == {"test": "value"}
            return {
                "success": True,
                "status_code": 200,
                "headers": {"content-type": "application/json"},
                "body": {"ok": True},
                "error": None,
            }

        monkeypatch.setattr(tool._client, "call_api", mock_call_api)

        result = await tool.run_json_async(
            {
                "url": "https://httpbin.org/get",
                "method": "GET",
                "params": {"test": "value"},
            }
        )

        assert result["success"] is True
        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_api_call_with_auth(self, monkeypatch):
        """Test API call forwards authentication args"""
        tool = APITool()

        async def mock_call_api(**kwargs):
            assert kwargs["auth_type"] == "bearer"
            assert kwargs["auth_token"] == "test-token"
            return {
                "success": True,
                "status_code": 200,
                "headers": {},
                "body": {"authenticated": True},
                "error": None,
            }

        monkeypatch.setattr(tool._client, "call_api", mock_call_api)

        result = await tool.run_json_async(
            {
                "url": "https://httpbin.org/bearer",
                "method": "GET",
                "auth_type": "bearer",
                "auth_token": "test-token",
            }
        )

        assert result["success"] is True
        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_api_call_with_post_body(self, monkeypatch):
        """Test API call forwards POST body and headers"""
        tool = APITool()

        async def mock_call_api(**kwargs):
            assert kwargs["method"] == "POST"
            assert kwargs["body"] == {"name": "test", "value": 123}
            assert kwargs["headers"] == {"Content-Type": "application/json"}
            return {
                "success": True,
                "status_code": 200,
                "headers": {},
                "body": {"name": "test", "value": 123},
                "error": None,
            }

        monkeypatch.setattr(tool._client, "call_api", mock_call_api)

        result = await tool.run_json_async(
            {
                "url": "https://httpbin.org/post",
                "method": "POST",
                "body": {"name": "test", "value": 123},
                "headers": {"Content-Type": "application/json"},
            }
        )

        assert result["success"] is True
        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_api_call_with_api_key_query(self, monkeypatch):
        """Test API key query options are forwarded via tool"""
        tool = APITool()

        async def mock_call_api(**kwargs):
            assert kwargs["auth_type"] == "api_key_query"
            assert kwargs["auth_token"] == "my-secret-key-123"
            assert kwargs["api_key_param"] == "key"
            return {
                "success": True,
                "status_code": 200,
                "headers": {},
                "body": {"args": {"key": "my-secret-key-123"}},
                "error": None,
            }

        monkeypatch.setattr(tool._client, "call_api", mock_call_api)

        result = await tool.run_json_async(
            {
                "url": "https://httpbin.org/get",
                "method": "GET",
                "auth_type": "api_key_query",
                "auth_token": "my-secret-key-123",
                "api_key_param": "key",
            }
        )

        assert result["success"] is True
        assert result["status_code"] == 200
        assert result["body"]["args"]["key"] == "my-secret-key-123"

    def test_return_value_formatting(self):
        """Test return value formatting"""
        tool = APITool()

        # Success case
        success_result = {
            "success": True,
            "status_code": 200,
            "body": {"result": "success"},
        }
        formatted = tool.return_value_as_string(success_result)
        assert "✅" in formatted
        assert "200" in formatted

        # Error case
        error_result = {
            "success": False,
            "error": "Connection failed",
        }
        formatted = tool.return_value_as_string(error_result)
        assert "❌" in formatted
        assert "Connection failed" in formatted


class TestConvenienceFunctions:
    """Test convenience functions"""

    @pytest.mark.asyncio
    async def test_call_api_function(self, mock_httpbin: None):
        """Test convenience call_api function"""
        result = await call_api(
            url="https://httpbin.org/get",
            method="GET",
            params={"test": "value"},
        )

        assert result["success"] is True
        assert result["status_code"] == 200
