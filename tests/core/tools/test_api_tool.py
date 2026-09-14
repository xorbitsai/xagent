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
    _hostname_matches_connector_domain,
    _sanitize_final_url,
    call_api,
    has_auth_credentials,
    match_known_connector_domain,
)


def _single_value_query_args(url: str) -> dict[str, str]:
    return {key: values[-1] for key, values in parse_qs(urlparse(url).query).items()}


class TestMatchKnownConnectorDomain:
    def test_matches_exact_host(self):
        assert match_known_connector_domain("api.github.com") == "GitHub"

    def test_matches_subdomain_of_a_multi_tenant_service(self):
        assert match_known_connector_domain("acme.myshopify.com") == "Shopify"
        assert match_known_connector_domain("acme.zendesk.com") == "Zendesk"

    def test_matches_salesforce_and_xero(self):
        """Both are live, registered OAuth connectors that were previously
        missing from the domain list entirely, despite Salesforce being
        named explicitly in the code's own comment."""
        assert match_known_connector_domain("acme.my.salesforce.com") == "Salesforce"
        assert match_known_connector_domain("login.salesforce.com") == "Salesforce"
        assert match_known_connector_domain("api.xero.com") == "Xero"

    def test_matches_specific_google_product_hosts(self):
        assert match_known_connector_domain("gmail.googleapis.com") == "Gmail"
        assert match_known_connector_domain("googleads.googleapis.com") == "Google Ads"
        assert (
            match_known_connector_domain("analyticsdata.googleapis.com")
            == "Google Analytics"
        )
        assert (
            match_known_connector_domain("searchconsole.googleapis.com")
            == "Google Search Console"
        )

    def test_does_not_match_an_uncovered_googleapis_subdomain(self):
        """Regression: a blanket "googleapis.com" entry used to match every
        Google Cloud API subdomain, including ones with no xagent connector
        at all (Cloud Storage, BigQuery, Translate, ...) - blocking them
        with a "use the dedicated connector" message that was actively
        wrong since no such connector exists for that API."""
        assert match_known_connector_domain("storage.googleapis.com") is None
        assert match_known_connector_domain("bigquery.googleapis.com") is None

    def test_does_not_match_slack_incoming_webhook_subdomain(self):
        """hooks.slack.com (incoming webhooks) is a different,
        self-authenticating product from the credentialed Slack Web API at
        slack.com/api - the guard must not touch it even though it shares
        the slack.com suffix."""
        assert match_known_connector_domain("slack.com") == "Slack"
        assert match_known_connector_domain("hooks.slack.com") is None

    def test_is_case_insensitive(self):
        assert match_known_connector_domain("API.HUBAPI.COM") == "HubSpot"

    def test_does_not_match_an_unrelated_domain(self):
        assert match_known_connector_domain("httpbin.org") is None

    def test_does_not_match_a_lookalike_suffix(self):
        """ "notzendesk.com" shares a suffix with "zendesk.com" as a raw
        string but is not "*.zendesk.com" - the dot-boundary check in
        _hostname_matches_connector_domain must reject it, not just do a
        substring/endswith check on the bare strings."""
        assert match_known_connector_domain("notzendesk.com") is None

    def test_does_not_match_domain_as_a_trailing_path_look_alike(self):
        """A hostname that merely contains the target domain as a substring
        elsewhere (not as its own suffix) must not match."""
        assert (
            _hostname_matches_connector_domain(
                "zendesk.com.evil.example", "zendesk.com"
            )
            is False
        )


class TestHasAuthCredentials:
    def test_true_for_explicit_auth_token_with_supported_auth_type(self):
        assert has_auth_credentials(
            "https://api.hubapi.com/x", None, None, "bearer", "tok"
        )

    def test_false_for_auth_token_without_auth_type(self):
        """_prepare_headers only attaches Authorization when BOTH auth_type
        and auth_token are set - a bare auth_token with no auth_type never
        reaches the outgoing request, so it must not count as a credential
        here either."""
        assert not has_auth_credentials(
            "https://api.hubapi.com/x", None, None, None, "tok"
        )

    def test_false_for_auth_token_with_unsupported_auth_type(self):
        assert not has_auth_credentials(
            "https://api.hubapi.com/x", None, None, "not-a-real-mode", "tok"
        )

    def test_false_for_auth_type_with_no_auth_token(self):
        """Regression: str(None) is the non-empty text "None", so a naive
        blankness check on auth_token=None (auth_type set, auth_token
        omitted - e.g. the caller forgot it) would wrongly read as "present".
        _prepare_headers requires both fields truthy, so this combination
        sends no real credential and must not count as one here either."""
        assert not has_auth_credentials(
            "https://api.hubapi.com/x", None, None, "bearer", None
        )

    def test_false_for_none_valued_query_param(self):
        """Same root cause as test_false_for_auth_type_with_no_auth_token,
        different call site: params={"api_key": None} is valid input per
        APICallArgs' Optional[Dict[str, Any]] typing (pydantic allows a null
        value), and must not be misread as a present credential."""
        assert not has_auth_credentials(
            "https://api.hubapi.com/x", None, {"api_key": None}, None, None
        )

    def test_true_for_authorization_header(self):
        assert has_auth_credentials(
            "https://api.hubapi.com/x",
            {"Authorization": "Bearer tok"},
            None,
            None,
            None,
        )

    def test_true_for_service_specific_credential_header(self):
        """Not every connector's direct-call convention uses the literal
        "Authorization" header name - matching must key off any header
        whose name looks credential-shaped, not an exact-name allowlist."""
        assert has_auth_credentials(
            "https://acme.myshopify.com/x",
            {"X-Shopify-Access-Token": "shpat_abc"},
            None,
            None,
            None,
        )

    def test_true_for_auth_query_param_in_params_dict(self):
        assert has_auth_credentials(
            "https://maps.googleapis.com/x", None, {"key": "AIza123"}, None, None
        )

    def test_true_for_auth_query_param_in_url_itself(self):
        assert has_auth_credentials(
            "https://maps.googleapis.com/x?key=AIza123", None, None, None, None
        )

    def test_true_for_hyphenated_api_key_query_param(self):
        """The header-marker list already recognized both "api-key" and
        "apikey" spellings; the query-param list was missing the hyphenated
        one, so a real credential passed as ?api-key=... was incorrectly
        treated as no-credential-attached."""
        assert has_auth_credentials(
            "https://api.hubapi.com/x", None, {"api-key": "secret"}, None, None
        )
        assert has_auth_credentials(
            "https://api.hubapi.com/x?api-key=secret", None, None, None, None
        )

    def test_true_for_basic_auth_userinfo_in_url(self):
        """HTTP Basic-Auth credentials embedded directly in the URL
        (user:pass@host) are a legitimate, already-authenticated call shape
        that headers/query-param checks alone can't see."""
        assert has_auth_credentials(
            "https://user:pass@api.github.com/x", None, None, None, None
        )

    def test_false_when_nothing_looks_like_a_credential(self):
        assert not has_auth_credentials(
            "https://api.hubapi.com/x?limit=10",
            {"Content-Type": "application/json"},
            None,
            None,
            None,
        )

    def test_false_for_no_arguments_at_all(self):
        assert not has_auth_credentials(
            "https://api.hubapi.com/x", None, None, None, None
        )

    def test_false_for_blank_header_value(self):
        """A header whose NAME looks credential-shaped but whose VALUE is
        blank carries nothing real - e.g. {"Authorization": ""}, which a
        caller could send by accident (an unresolved template variable,
        say). Treating the name alone as proof of a credential would skip
        the connector hint on a 401 caused by exactly that missing value."""
        assert not has_auth_credentials(
            "https://api.hubapi.com/x", {"Authorization": ""}, None, None, None
        )
        assert not has_auth_credentials(
            "https://api.hubapi.com/x", None, {"api_key": ""}, None, None
        )

    def test_false_for_authority_header(self):
        """ "authority" (the HTTP/2 pseudo-header carrying the target host,
        set automatically by many HTTP/2 clients and proxies - not a
        credential) must not be treated as a credential."""
        assert not has_auth_credentials(
            "https://api.hubapi.com/x",
            {"Content-Type": "application/json", "authority": "api.hubapi.com"},
            None,
            None,
            None,
        )

    def test_false_for_other_headers_merely_containing_auth_as_a_substring(self):
        """Regression: the marker list used to include the bare "auth" and
        "signature" substrings, which matched non-credential headers like
        "X-Author", "X-OAuth-Client-Id" (a public client id, not a secret),
        and GitHub's outbound "X-Hub-Signature-256" webhook-verification
        header - silently treating an actually-unauthenticated call as
        credentialed and defeating the guard entirely."""
        assert not has_auth_credentials(
            "https://api.github.com/x",
            {"X-Author": "jane@example.com"},
            None,
            None,
            None,
        )
        assert not has_auth_credentials(
            "https://api.github.com/x",
            {"X-OAuth-Client-Id": "public-id"},
            None,
            None,
            None,
        )
        assert not has_auth_credentials(
            "https://api.github.com/x",
            {"X-Hub-Signature-256": "sha256=abc"},
            None,
            None,
            None,
        )

    def test_ignores_non_string_header_keys(self):
        assert not has_auth_credentials(
            "https://api.hubapi.com/x", {1: "x"}, None, None, None
        )  # type: ignore[dict-item]

    def test_ignores_non_string_param_keys(self):
        assert not has_auth_credentials(
            "https://api.hubapi.com/x",
            None,
            {1: "x"},
            None,
            None,  # type: ignore[dict-item]
        )


class TestSanitizeFinalUrl:
    """Regression: final_url reports the real post-redirect response URL,
    but the raw URL can itself carry a credential - an api_key_query
    auth_token merged into the query string, or Basic-Auth userinfo a
    caller embedded directly in the URL. Only the host identifies which
    connector answered, so everything else must be stripped before this
    value leaves _make_request."""

    def test_strips_query_string_credential(self):
        assert (
            _sanitize_final_url("https://api.hubapi.com/crm/v3/deals?api_key=secret")
            == "https://api.hubapi.com/crm/v3/deals"
        )

    def test_strips_basic_auth_userinfo(self):
        assert (
            _sanitize_final_url("https://user:pass@api.hubapi.com/crm/v3/deals")
            == "https://api.hubapi.com/crm/v3/deals"
        )

    def test_strips_both_userinfo_and_query_credential(self):
        assert (
            _sanitize_final_url("https://user:pass@example.com/path?api_key=secret")
            == "https://example.com/path"
        )

    def test_preserves_port(self):
        assert (
            _sanitize_final_url("https://example.com:8443/path?key=secret")
            == "https://example.com:8443/path"
        )

    def test_no_op_for_a_url_with_nothing_to_strip(self):
        assert (
            _sanitize_final_url("https://api.hubapi.com/crm/v3/deals")
            == "https://api.hubapi.com/crm/v3/deals"
        )


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
        elif path.startswith("/status/"):
            status_code = int(path.removeprefix("/status/"))
            body = {}
        else:
            status_code = 404
            body = {}

        # A "__redirected_to__" query param lets a test simulate httpx
        # having followed a cross-host redirect before landing on this
        # response - the real _make_request reports response.url (the
        # final, post-redirect URL) here, which can differ from the
        # request's starting url.
        final_url = _single_value_query_args(url).get("__redirected_to__", url)

        # Mirror httpx's own redirect behavior (see
        # httpx.Client._redirect_headers) closely enough for these tests:
        # Authorization is dropped when the redirect crosses hosts, and a
        # redirect target's own query string (not the original request's)
        # is what the final request carries - final_url above already is
        # that target URL, so no extra query handling is needed here.
        final_headers = dict(headers)
        if urlparse(final_url).hostname != parsed.hostname:
            final_headers.pop("Authorization", None)
        final_request_has_credential = has_auth_credentials(
            final_url, final_headers, None, None, None
        )

        return {
            "success": 200 <= status_code < 300,
            "status_code": status_code,
            "headers": {"content-type": "application/json"},
            "body": body,
            "error": None if 200 <= status_code < 300 else f"HTTP {status_code}",
            "final_url": final_url,
            "final_request_has_credential": final_request_has_credential,
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
    async def test_api_key_query_auth_type_is_case_insensitive(
        self, mock_httpbin: None
    ):
        """Regression: this check used to be an exact-case string compare,
        unlike _prepare_headers' auth_type.lower() handling of bearer/
        basic/api_key just below it and has_auth_credentials' own lower()
        call - auth_type="API_KEY_QUERY" (or any non-lowercase spelling)
        would silently attach no credential here while has_auth_credentials
        reported one was present, suppressing the connector hint for
        exactly the resulting unauthenticated 401/403."""
        result = await call_api(
            url="https://httpbin.org/get",
            method="GET",
            auth_type="API_KEY_QUERY",
            auth_token="test-key-123",
        )

        assert result["success"] is True
        assert result["body"]["args"]["api_key"] == "test-key-123"

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
    async def test_url_httpx_rejects_after_merging_params_does_not_raise(self):
        """Regression: _is_valid_url's cheap urlparse-based check (scheme +
        non-empty netloc) lets through a URL httpx.URL() itself later
        rejects (e.g. a non-numeric port) - only reachable once query
        params get merged into the URL, since a request with none never
        reaches that merge step. That merge used to run outside the retry
        loop's try/except, so it raised straight out of call_api instead of
        returning the documented dict shape."""
        client = APIClientCore()
        result = await client.call_api(
            url="http://example.com:abc/path", params={"q": "1"}
        )

        assert result["success"] is False
        assert result["status_code"] == 0
        assert isinstance(result["error"], str)

    @pytest.mark.asyncio
    async def test_does_not_block_known_connector_domains(self, mock_httpbin: None):
        """The connector-domain guard lives in the APITool adapter (the
        agent-facing `api_call` tool), not in APIClientCore/call_api - this
        core client is also used directly by CustomApiTool (the
        user-configured "Custom API" connector-builder feature), which must
        be unaffected by a guard meant for a different problem (an agent
        falling back to a raw call instead of a real MCP connector tool)."""
        client = APIClientCore()
        result = await client.call_api(
            url="https://api.hubapi.com/headers",
            method="GET",
            headers={"X-Client-Secret": "not-a-recognized-marker-name"},
        )

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
        assert "dedicated MCP connector" in tool.description
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

    @pytest.mark.asyncio
    async def test_public_credential_less_call_to_known_connector_domain_succeeds(
        self, mock_httpbin: None
    ):
        """The core regression this whole design is for: a domain covered
        by a dedicated connector (GitHub) can still have genuinely public,
        credential-less endpoints (e.g. GET /repos/{owner}/{repo}). An
        earlier pre-flight version of this guard blocked every
        credential-less call to a known connector domain outright,
        converting a legitimate 200 into a synthetic local failure. The
        request must actually be attempted and its real 2xx returned
        untouched."""
        tool = APITool()
        result = await tool.run_json_async(
            {"url": "https://api.github.com/get", "method": "GET"}
        )

        assert result["success"] is True
        assert result["status_code"] == 200

    @pytest.mark.asyncio
    async def test_annotates_a_real_401_with_connector_hint_when_uncredentialed(
        self, mock_httpbin: None
    ):
        """A request that actually reached a known connector domain and
        actually got a 401/403 back, with no credential this tool
        recognizes, gets that real status/body preserved plus a hint
        pointing at the dedicated connector tools - not a synthetic
        pre-flight refusal."""
        tool = APITool()
        result = await tool.run_json_async(
            {"url": "https://api.hubapi.com/status/401", "method": "PATCH"}
        )

        assert result["success"] is False
        assert result["status_code"] == 401
        assert "HubSpot" in result["error"]
        assert "connector tools" in result["error"]

    @pytest.mark.asyncio
    async def test_annotates_a_real_401_on_a_multi_tenant_subdomain(
        self, mock_httpbin: None
    ):
        """Multi-tenant connectors (Zendesk, Shopify, etc.) are matched by
        domain suffix, not just the exact host."""
        tool = APITool()
        result = await tool.run_json_async(
            {"url": "https://acme.zendesk.com/status/403"}
        )

        assert result["status_code"] == 403
        assert "Zendesk" in result["error"]

    @pytest.mark.asyncio
    async def test_does_not_annotate_a_401_with_credentials_attached(
        self, mock_httpbin: None
    ):
        """A caller that already attached its own credential - even if that
        credential turns out to be invalid and the real call still 401s -
        genuinely intended a direct call, so the hint (which exists to
        explain an UNEXPLAINED 401) must not be added."""
        tool = APITool()
        result = await tool.run_json_async(
            {
                "url": "https://api.hubapi.com/status/401",
                "auth_type": "bearer",
                "auth_token": "a-private-app-token-that-happens-to-be-invalid",
            }
        )

        assert result["status_code"] == 401
        assert "connector tools" not in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_does_not_annotate_a_non_401_403_status(self, mock_httpbin: None):
        """Only 401/403 are ambiguous ("is this a broken connection or a
        missing credential?"); any other status is left exactly as the
        server returned it."""
        tool = APITool()
        result = await tool.run_json_async({"url": "https://api.hubapi.com/status/500"})

        assert result["status_code"] == 500
        assert "connector tools" not in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_does_not_annotate_an_unrelated_domain(self, mock_httpbin: None):
        tool = APITool()
        result = await tool.run_json_async({"url": "https://httpbin.org/status/401"})

        assert result["status_code"] == 401
        assert "connector tools" not in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_annotates_a_trailing_dot_host(self, mock_httpbin: None):
        """A trailing "." is a valid DNS root-label marker for the same
        host (api.hubapi.com. == api.hubapi.com); the hint must still
        apply, not silently skip a spelling a caller could legitimately
        send."""
        tool = APITool()
        result = await tool.run_json_async(
            {"url": "https://api.hubapi.com./status/401"}
        )

        assert "HubSpot" in result["error"]

    @pytest.mark.asyncio
    async def test_annotates_based_on_the_post_redirect_host_not_the_original(
        self, mock_httpbin: None
    ):
        """Regression: allow_redirects defaults to True, so the response
        that actually comes back can be from a different host than the URL
        the caller started with. A request that starts on an unrelated host
        and redirects to a known connector domain, which then 401s, must
        still get the hint - matching against the original URL's host
        (unrelated here) would silently miss it."""
        tool = APITool()
        result = await tool.run_json_async(
            {
                "url": "https://httpbin.org/status/401"
                "?__redirected_to__=https://api.hubapi.com/crm/v3/deals/1"
            }
        )

        assert "HubSpot" in result["error"]

    @pytest.mark.asyncio
    async def test_annotates_when_a_cross_origin_redirect_drops_the_original_credential(
        self, mock_httpbin: None
    ):
        """Regression: httpx strips the Authorization header when a
        redirect crosses hosts (see httpx.Client._redirect_headers). A
        request that started out credentialed can still land on a known
        connector host with nothing attached on the actual final request -
        the hint must reflect that, not the caller's original (now-
        stripped) credential."""
        tool = APITool()
        result = await tool.run_json_async(
            {
                "url": "https://httpbin.org/status/401"
                "?__redirected_to__=https://api.hubapi.com/crm/v3/deals/1",
                "auth_type": "bearer",
                "auth_token": "original-host-token",
            }
        )

        assert "HubSpot" in result["error"]

    @pytest.mark.asyncio
    async def test_does_not_annotate_when_a_same_origin_redirect_keeps_the_credential(
        self, mock_httpbin: None
    ):
        """Control for the cross-origin case above: a same-origin redirect
        does not strip Authorization, so a genuinely still-credentialed
        request must keep suppressing the hint."""
        tool = APITool()
        result = await tool.run_json_async(
            {
                "url": "https://api.hubapi.com/status/401"
                "?__redirected_to__=https://api.hubapi.com/crm/v3/deals/1",
                "auth_type": "bearer",
                "auth_token": "still-valid-token",
            }
        )

        assert "connector tools" not in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_falls_back_to_original_args_when_final_request_has_credential_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A result from a core client that predates final_request_has_
        credential (e.g. an older or differently-mocked call_api) must
        still work - falling back to checking the original api_args, same
        as before that field existed."""
        tool = APITool()

        async def mock_call_api(**kwargs):
            return {
                "success": False,
                "status_code": 401,
                "headers": {},
                "body": None,
                "error": "HTTP 401",
                "final_url": "https://api.hubapi.com/crm/v3/deals/1",
            }

        monkeypatch.setattr(tool._client, "call_api", mock_call_api)

        result = await tool.run_json_async(
            {
                "url": "https://api.hubapi.com/crm/v3/deals/1",
                "auth_type": "bearer",
                "auth_token": "still-valid-token",
            }
        )

        assert "connector tools" not in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_does_not_annotate_when_redirected_off_a_known_connector_host(
        self, mock_httpbin: None
    ):
        """Symmetric regression: a request starting on a known connector
        host that redirects OFF that host to an unrelated 401 must not get
        a hint naming the original (wrong) service."""
        tool = APITool()
        result = await tool.run_json_async(
            {
                "url": "https://api.github.com/status/401"
                "?__redirected_to__=https://httpbin.org/status/401"
            }
        )

        assert "connector tools" not in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_hint_does_not_produce_a_run_on_sentence(self, mock_httpbin: None):
        """The existing error text ("HTTP 401") has no trailing punctuation,
        so appending the hint sentence directly after it would otherwise
        read as one run-on sentence - a period must be inserted between
        them."""
        tool = APITool()
        result = await tool.run_json_async({"url": "https://api.hubapi.com/status/401"})

        assert result["error"].startswith("HTTP 401. This looks like")

    @pytest.mark.asyncio
    async def test_malformed_url_does_not_crash(self):
        """A malformed URL (invalid IPv6 host syntax) must still come back
        as a normal structured result - not an uncaught exception. This URL
        fails core's own validation (_is_valid_url) before a real request is
        ever attempted, so status_code is 0 and the connector-hint check
        never runs at all."""
        tool = APITool()
        result = await tool.run_json_async({"url": "https://[::1"})

        assert result["success"] is False
        assert result["status_code"] == 0
        assert isinstance(result.get("error"), str)

    @pytest.mark.asyncio
    async def test_does_not_block_custom_api_tool_through_shared_client(
        self, mock_httpbin: None
    ):
        """The connector hint is applied in APITool.run_json_async, not
        shared APIClientCore/call_api - confirms a caller going through the
        raw client directly (as CustomApiTool does) is entirely unaffected,
        since that's a different, user-authored integration this feature
        must not touch."""
        client = APIClientCore()
        result = await client.call_api(url="https://api.hubapi.com/headers")

        assert result["success"] is True

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
