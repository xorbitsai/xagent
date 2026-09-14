"""
Tests for API Tool
"""

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from xagent.core.tools.adapters.vibe.api_tool import APICallArgs, APITool
from xagent.core.tools.core.api_tool import (
    APIClientCore,
    _has_auth_credentials,
    _hostname_matches_connector_domain,
    _match_known_connector_domain,
    call_api,
)


def _single_value_query_args(url: str) -> dict[str, str]:
    return {key: values[-1] for key, values in parse_qs(urlparse(url).query).items()}


class TestMatchKnownConnectorDomain:
    def test_matches_exact_host(self):
        assert _match_known_connector_domain("api.github.com") == "GitHub"

    def test_matches_subdomain_of_a_multi_tenant_service(self):
        assert _match_known_connector_domain("acme.myshopify.com") == "Shopify"
        assert _match_known_connector_domain("acme.zendesk.com") == "Zendesk"

    def test_matches_any_googleapis_subdomain(self):
        assert _match_known_connector_domain("gmail.googleapis.com") == "Google"
        assert _match_known_connector_domain("oauth2.googleapis.com") == "Google"

    def test_is_case_insensitive(self):
        assert _match_known_connector_domain("API.HUBAPI.COM") == "HubSpot"

    def test_does_not_match_an_unrelated_domain(self):
        assert _match_known_connector_domain("httpbin.org") is None

    def test_does_not_match_a_lookalike_suffix(self):
        """ "notzendesk.com" shares a suffix with "zendesk.com" as a raw
        string but is not "*.zendesk.com" - the dot-boundary check in
        _hostname_matches_connector_domain must reject it, not just do a
        substring/endswith check on the bare strings."""
        assert _match_known_connector_domain("notzendesk.com") is None

    def test_does_not_match_domain_as_a_trailing_path_look_alike(self):
        """A hostname that merely contains the target domain as a substring
        elsewhere (not as its own suffix) must not match."""
        assert (
            _hostname_matches_connector_domain("zendesk.com.evil.example", "zendesk.com") is False
        )


class TestHasAuthCredentials:
    def test_true_for_explicit_auth_token(self):
        assert _has_auth_credentials("https://api.hubapi.com/x", None, None, "tok")

    def test_true_for_authorization_header(self):
        assert _has_auth_credentials(
            "https://api.hubapi.com/x", {"Authorization": "Bearer tok"}, None, None
        )

    def test_true_for_service_specific_credential_header(self):
        """Not every connector's direct-call convention uses the literal
        "Authorization" header name - matching must key off any header
        whose name looks credential-shaped, not an exact-name allowlist."""
        assert _has_auth_credentials(
            "https://acme.myshopify.com/x",
            {"X-Shopify-Access-Token": "shpat_abc"},
            None,
            None,
        )

    def test_true_for_auth_query_param_in_params_dict(self):
        assert _has_auth_credentials(
            "https://maps.googleapis.com/x", None, {"key": "AIza123"}, None
        )

    def test_true_for_auth_query_param_in_url_itself(self):
        assert _has_auth_credentials("https://maps.googleapis.com/x?key=AIza123", None, None, None)

    def test_false_when_nothing_looks_like_a_credential(self):
        assert not _has_auth_credentials(
            "https://api.hubapi.com/x?limit=10", {"Content-Type": "application/json"}, None, None
        )

    def test_false_for_no_arguments_at_all(self):
        assert not _has_auth_credentials("https://api.hubapi.com/x", None, None, None)


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
            parsed_data = json.loads(data.decode() if isinstance(data, bytes) else data or "{}")
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
    async def test_blocks_unauthenticated_call_to_known_connector_domain(self):
        """A raw call to a domain covered by a dedicated connector (e.g. the
        agent falling back to api_call because it couldn't find a write tool
        on the real connector) must fail fast with a clear message, not a
        misleading 401/403 that looks like a broken connection."""
        client = APIClientCore()
        result = await client.call_api(
            url="https://api.hubapi.com/crm/v3/objects/deals/1",
            method="PATCH",
        )

        assert result["success"] is False
        assert result["status_code"] == 0
        assert "HubSpot" in result["error"]
        assert "connector tools instead" in result["error"]

    @pytest.mark.asyncio
    async def test_blocks_known_connector_subdomain(self):
        """Multi-tenant connectors (Zendesk, Shopify, etc.) are matched by
        domain suffix, not just the exact host."""
        client = APIClientCore()
        result = await client.call_api(url="https://acme.zendesk.com/api/v2/tickets")

        assert result["success"] is False
        assert "Zendesk" in result["error"]

    @pytest.mark.asyncio
    async def test_allows_known_connector_domain_with_explicit_auth_token(self, mock_httpbin: None):
        """An explicit auth_token means the caller has their own credential
        and genuinely intends a direct call - the guard must not block it."""
        client = APIClientCore()
        result = await client.call_api(
            url="https://api.hubapi.com/some/endpoint",
            auth_type="bearer",
            auth_token="a-private-app-token",
        )

        assert "error" not in result or result.get("status_code") != 0

    @pytest.mark.asyncio
    async def test_allows_known_connector_domain_with_authorization_header(
        self, mock_httpbin: None
    ):
        client = APIClientCore()
        result = await client.call_api(
            url="https://api.hubapi.com/some/endpoint",
            headers={"Authorization": "Bearer a-private-app-token"},
        )

        assert "error" not in result or result.get("status_code") != 0

    @pytest.mark.asyncio
    async def test_allows_known_connector_domain_with_service_specific_header(
        self, mock_httpbin: None
    ):
        """Not every connector uses a plain "Authorization" header - Shopify
        uses X-Shopify-Access-Token, so the guard must recognize that too,
        not just the one header name."""
        client = APIClientCore()
        result = await client.call_api(
            url="https://acme.myshopify.com/admin/api/2024-01/products.json",
            headers={"X-Shopify-Access-Token": "shpat_abc123"},
        )

        assert "error" not in result or result.get("status_code") != 0

    @pytest.mark.asyncio
    async def test_allows_known_connector_domain_with_query_param_in_url(self, mock_httpbin: None):
        """Google APIs commonly carry the API key as a "key" query
        parameter directly in the URL rather than a header."""
        client = APIClientCore()
        result = await client.call_api(
            url="https://maps.googleapis.com/maps/api/geocode/json?key=AIzaSy_123",
        )

        assert "error" not in result or result.get("status_code") != 0

    @pytest.mark.asyncio
    async def test_allows_known_connector_domain_with_query_param_in_params(
        self, mock_httpbin: None
    ):
        client = APIClientCore()
        result = await client.call_api(
            url="https://maps.googleapis.com/maps/api/geocode/json",
            params={"key": "AIzaSy_123"},
        )

        assert "error" not in result or result.get("status_code") != 0

    @pytest.mark.asyncio
    async def test_still_blocks_known_connector_domain_with_unrelated_query_param(
        self, mock_httpbin: None
    ):
        """A query param that isn't one of the recognized auth-carrying
        names must not itself count as evidence of a credential."""
        client = APIClientCore()
        result = await client.call_api(
            url="https://api.hubapi.com/some/endpoint?limit=10",
        )

        assert result["success"] is False
        assert "HubSpot" in result["error"]

    @pytest.mark.asyncio
    async def test_does_not_block_an_unrelated_domain(self, mock_httpbin: None):
        client = APIClientCore()
        result = await client.call_api(url="https://httpbin.org/get")

        assert result["success"] is True

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
