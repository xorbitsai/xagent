import json
import socket
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import posthog


def _fake_getaddrinfo(*ips):
    def _impl(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET6 if ":" in ip else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (ip, port),
            )
            for ip in ips
        ]

    return _impl


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        text: str = "",
        url: str = "",
        json_raises: bool = False,
        headers: dict | None = None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self._json_raises = json_raises
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()
        self.url = url
        self.headers = headers or {}

    def json(self):
        if self._json_raises:
            raise json.JSONDecodeError("Expecting value", self.text, 0)
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error for url: {self.url}", response=self
            )


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("POSTHOG_API_KEY", "phx_test_key")
    monkeypatch.setenv("POSTHOG_HOST", "https://us.posthog.com")
    # _base_url() resolves DNS to catch a hostname that rebinds to a private
    # address; tests must not depend on real network/DNS, so every test gets
    # a fake resolver returning an unambiguously public IP by default.
    monkeypatch.setattr(posthog.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1"))


def test_headers_require_api_key(monkeypatch):
    monkeypatch.delenv("POSTHOG_API_KEY")

    with pytest.raises(ValueError, match="POSTHOG_API_KEY"):
        posthog._headers()


def test_headers_include_bearer_token_and_json_content_type():
    assert posthog._headers() == {
        "Authorization": "Bearer phx_test_key",
        "Content-Type": "application/json",
    }


def test_base_url_requires_host(monkeypatch):
    monkeypatch.delenv("POSTHOG_HOST")

    with pytest.raises(ValueError, match="POSTHOG_HOST"):
        posthog._base_url()


@pytest.mark.parametrize(
    "host, expected",
    [
        ("https://eu.posthog.com/", "https://eu.posthog.com"),
        ("us.posthog.com", "https://us.posthog.com"),
        ("  https://eu.posthog.com  ", "https://eu.posthog.com"),
        ("HTTPS://us.posthog.com", "https://us.posthog.com"),
        ("https://eu.posthog.com///", "https://eu.posthog.com"),
        ("https://us.posthog.com.", "https://us.posthog.com"),
        # the default port spelled out explicitly is a no-op, not a real
        # customization -- it's exactly the port every request already
        # goes to, so rejecting it would break a previously-valid,
        # harmless config value for no functional reason.
        ("us.posthog.com:443", "https://us.posthog.com"),
    ],
)
def test_base_url_normalizes_valid_host(monkeypatch, host, expected):
    monkeypatch.setenv("POSTHOG_HOST", host)

    assert posthog._base_url() == expected


@pytest.mark.parametrize(
    "host, match",
    [
        ("   ", "POSTHOG_HOST"),  # whitespace-only
        ("/", "posthog.com"),  # slash-only -> empty hostname -> allowlist
        ("https://", "posthog.com"),  # scheme-only -> empty hostname
        ("http://us.posthog.com", "https"),  # wrong scheme
        ("user:pass@us.posthog.com", "credentials"),  # embedded userinfo
        ("us.posthog.com/api", "path"),  # path component
        ("us.posthog.com?x=1", "path"),  # query component
        ("https://evil.example.com", "posthog.com"),  # unrelated domain
        # a naive host.endswith("posthog.com") (missing the leading dot)
        # would wrongly accept this:
        ("https://evilposthog.com", "posthog.com"),
        # the bare apex domain isn't one of the two allowed hosts either:
        ("https://posthog.com", "posthog.com"),
        ("us.posthog.com:8443", "port"),  # only PostHog Cloud is supported,
        # which never uses a non-default port
        ("us.posthog.com:notaport", "port"),  # unparsable port
        # an empty DNS label that a suffix-based allowlist would have let
        # through as ".posthog.com", crashing socket.getaddrinfo with a
        # UnicodeEncodeError instead of failing cleanly here:
        ("https://.posthog.com", "posthog.com"),
        # literal private/loopback/link-local hosts, plus decimal- and
        # hex-encoded IP obfuscation attempts -- none of these are one of
        # the two allowed hosts, so all are rejected by the allowlist
        # without ever reaching DNS resolution:
        ("127.0.0.1", "posthog.com"),
        ("localhost", "posthog.com"),
        ("169.254.169.254", "posthog.com"),
        ("10.0.0.5", "posthog.com"),
        # a bracket-less IPv6 literal: urlsplit can't tell the trailing
        # ":1" from a port separator, so this is rejected as an invalid
        # port before ever reaching the allowlist check -- still rejected,
        # just via a different diagnostic than the other rows here.
        ("::1", "port"),
        ("2130706433", "posthog.com"),
        ("0x7f000001", "posthog.com"),
    ],
)
def test_base_url_rejects_invalid_host(monkeypatch, host, match):
    monkeypatch.setenv("POSTHOG_HOST", host)

    with pytest.raises(ValueError, match=match):
        posthog._base_url()


def test_base_url_rejects_host_resolving_to_private_ip(monkeypatch):
    # An allowed host (so it clears the domain allowlist) that only
    # *resolves* to a private address -- the DNS-rebinding case a literal
    # host/IP check alone can't catch.
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com")
    monkeypatch.setattr(posthog.socket, "getaddrinfo", _fake_getaddrinfo("10.0.0.5"))

    with pytest.raises(ValueError, match="not allowed"):
        posthog._base_url()


def test_base_url_rejects_when_any_resolved_address_is_private(monkeypatch):
    # A hostname resolving to more than one address (common for a load
    # balanced service) must be rejected if ANY resolved address is
    # private, not just the first one checked.
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com")
    monkeypatch.setattr(
        posthog.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1", "10.0.0.5")
    )

    with pytest.raises(ValueError, match="not allowed"):
        posthog._base_url()


def test_base_url_rejects_ipv6_private_resolved_address(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com")
    monkeypatch.setattr(posthog.socket, "getaddrinfo", _fake_getaddrinfo("fe80::1"))

    with pytest.raises(ValueError, match="not allowed"):
        posthog._base_url()


def test_base_url_raises_when_dns_resolution_fails(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com")

    def _raise(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(posthog.socket, "getaddrinfo", _raise)

    with pytest.raises(ValueError, match="could not be resolved"):
        posthog._base_url()


def test_base_url_revalidates_host_on_every_call(monkeypatch):
    # _base_url() is called fresh on every _request() -- it doesn't cache
    # a validated host across calls -- so a POSTHOG_HOST change between
    # two calls in the same process must be caught on the very next one.
    assert posthog._base_url() == "https://us.posthog.com"

    monkeypatch.setenv("POSTHOG_HOST", "https://evil.example.com")

    with pytest.raises(ValueError, match="posthog.com"):
        posthog._base_url()


@pytest.mark.parametrize(
    "limit, expected",
    [
        (0, 1),  # clamped up to the minimum of 1, not 0 (an empty page)
        (-5, 1),
        (1, 1),
        (posthog.MAX_LIMIT, posthog.MAX_LIMIT),
        (posthog.MAX_LIMIT + 1, posthog.MAX_LIMIT),
        (10_000, posthog.MAX_LIMIT),
    ],
)
def test_clamp_limit_boundaries(limit, expected):
    assert posthog._clamp_limit(limit) == expected


def test_path_segment_encodes_traversal_attempt():
    assert posthog._path_segment("1/../2") == "1%2F..%2F2"


def test_path_segment_leaves_at_current_sentinel_unescaped():
    # "@" is explicitly kept literal (RFC 3986 permits it unescaped in a
    # path segment) so the "@current"/"@me" sentinel every tool here
    # defaults to stays readable, without relying on the server decoding it.
    assert posthog._path_segment("@current") == "@current"


def test_paginated_results_returns_next_offset_when_truncated():
    page, truncated, next_offset = posthog._paginated_results(
        {
            "results": [{"id": 1}, {"id": 2}],
            "next": "https://us.posthog.com/x?offset=2",
        },
        limit=2,
        offset=0,
    )

    assert page == [{"id": 1}, {"id": 2}]
    assert truncated is True
    assert next_offset == 2


def test_paginated_results_no_next_offset_when_not_truncated():
    page, truncated, next_offset = posthog._paginated_results(
        {"results": [{"id": 1}], "next": None}, limit=50, offset=10
    )

    assert truncated is False
    assert next_offset is None


def test_paginated_results_no_next_offset_when_truncated_but_page_empty():
    # A server response that claims more pages exist (`next` is set) but
    # returns zero results this call -- if this still handed back a
    # next_offset, it would equal the offset just requested, and a caller
    # that mechanically retries with it would loop forever on one request.
    page, truncated, next_offset = posthog._paginated_results(
        {"results": [], "next": "https://us.posthog.com/x?offset=999"},
        limit=50,
        offset=10,
    )

    assert page == []
    assert truncated is True
    assert next_offset is None


def test_paginated_results_rejects_non_dict_payload():
    with pytest.raises(ValueError, match="JSON object"):
        posthog._paginated_results(["not", "a", "dict"], limit=50, offset=0)


def test_paginated_results_rejects_non_list_results():
    with pytest.raises(ValueError, match="results"):
        posthog._paginated_results({"results": "not-a-list"}, limit=50, offset=0)


def test_path_segment_rejects_empty_value():
    with pytest.raises(ValueError, match="project_id"):
        posthog._path_segment("", "project_id")


def test_path_segment_rejects_none_value():
    # str(None) == "None", a non-empty string -- a naive `not str(value)`
    # check would miss this and build a literal ".../None/" path.
    with pytest.raises(ValueError, match="person_id"):
        posthog._path_segment(None, "person_id")


def test_get_person_rejects_empty_person_id():
    result = json.loads(posthog.posthog_get_person("", project_id="proj1"))

    assert result["status"] == "error"
    assert "person_id" in result["message"]


def test_create_annotation_rejects_empty_project_id():
    result = json.loads(posthog.posthog_create_annotation("Deployed v2", ""))

    assert result["status"] == "error"
    assert "project_id" in result["message"]


def test_create_annotation_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                status_code=403, json_data={"detail": "Permission denied."}
            )
        ),
    )

    result = json.loads(posthog.posthog_create_annotation("Deployed v2", "proj1"))

    assert result["status"] == "error"
    assert "Permission denied" in result["message"]


def test_request_uses_configured_host_and_headers(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = posthog._request("GET", "/api/users/@me/")

    assert result == {"ok": True}
    assert (
        mock_request.call_args.kwargs["url"] == "https://us.posthog.com/api/users/@me/"
    )
    assert (
        mock_request.call_args.kwargs["headers"]["Authorization"]
        == "Bearer phx_test_key"
    )
    assert mock_request.call_args.kwargs["allow_redirects"] is False


def test_request_passes_empty_proxies_when_none_configured(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    posthog._request("GET", "/api/users/@me/")

    assert mock_request.call_args.kwargs["proxies"] == {}


def test_request_raises_when_proxy_is_configured_but_not_trusted(monkeypatch):
    # An ambient proxy makes the *proxy* resolve DNS for the real
    # connection, silently bypassing _base_url()'s own private-network
    # validation of POSTHOG_HOST -- this must fail loudly instead of
    # quietly connecting through an unvetted proxy.
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
    mock_request = Mock()
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    with pytest.raises(ValueError, match="not marked as trusted"):
        posthog._request("GET", "/api/users/@me/")

    mock_request.assert_not_called()


def test_request_passes_proxy_explicitly_when_trusted(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = posthog._request("GET", "/api/users/@me/")

    assert result == {"ok": True}
    assert mock_request.call_args.kwargs["proxies"] == {
        "http": "http://proxy.internal:8080",
        "https": "http://proxy.internal:8080",
    }


def test_make_request_disables_trust_env(monkeypatch):
    # requests.request() always runs on a fresh trust_env=True Session,
    # which -- regardless of what proxies= is passed -- still falls back to
    # get_environ_proxies() -> urllib.request.getproxies(), which itself
    # falls back *past* HTTP_PROXY/HTTPS_PROXY/ALL_PROXY to the OS's own
    # proxy configuration (getproxies_macosx_sysconf() on macOS,
    # getproxies_registry() on Windows) once none of those env vars are
    # set. trust_env=False skips that entire environment-and-OS-native
    # lookup in one place, so assert _make_request actually sets it before
    # issuing the real request.
    captured = {}

    class FakeSession:
        def __enter__(self):
            self.trust_env = True
            return self

        def __exit__(self, *exc_info):
            return False

        def request(self, **kwargs):
            captured["trust_env"] = self.trust_env
            return MockResponse(json_data={"ok": True})

    monkeypatch.setattr(posthog.requests, "Session", FakeSession)

    response = posthog._make_request(
        method="GET",
        url="https://us.posthog.com/api/users/@me/",
        headers={},
        params=None,
        json=None,
        timeout=1,
        proxies={},
        allow_redirects=False,
    )

    assert captured["trust_env"] is False
    assert response.json() == {"ok": True}


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_request_rejects_redirect_response(monkeypatch, status_code):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                status_code=status_code, url="https://us.posthog.com/api/users/@me/"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="redirect"):
        posthog._request("GET", "/api/users/@me/")


def test_request_passes_configured_timeout(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    posthog._request("GET", "/api/users/@me/")

    assert mock_request.call_args.kwargs["timeout"] == posthog.DEFAULT_TIMEOUT_SECONDS


def test_request_returns_empty_dict_for_204(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(return_value=MockResponse(status_code=204, text="")),
    )

    assert posthog._request("GET", "/api/users/@me/") == {}


def test_request_retries_once_on_429_with_retry_after(monkeypatch):
    responses = [
        MockResponse(status_code=429, url="x", headers={"Retry-After": "1"}),
        MockResponse(json_data={"ok": True}),
    ]
    mock_request = Mock(side_effect=responses)
    monkeypatch.setattr(posthog, "_make_request", mock_request)
    monkeypatch.setattr(posthog.time, "sleep", Mock())

    result = posthog._request("GET", "/api/users/@me/")

    assert result == {"ok": True}
    assert mock_request.call_count == 2
    posthog.time.sleep.assert_called_once_with(1)


def test_request_does_not_retry_a_second_429(monkeypatch):
    response = MockResponse(status_code=429, url="x", headers={"Retry-After": "1"})
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(posthog, "_make_request", mock_request)
    monkeypatch.setattr(posthog.time, "sleep", Mock())

    with pytest.raises(RuntimeError):
        posthog._request("GET", "/api/users/@me/")

    assert mock_request.call_count == 2


def test_request_redacts_connection_error_message(monkeypatch):
    # A ProxyError connecting through a trusted proxy can echo the full
    # proxy URL, credentials included.
    def _raise(*args, **kwargs):
        raise requests.exceptions.ProxyError(
            "Unable to connect to proxy: "
            "https://user:sp-secret-proxy-pass@proxy.internal:8080/"
        )

    monkeypatch.setattr(posthog, "_make_request", _raise)

    with pytest.raises(RuntimeError) as excinfo:
        posthog._request("GET", "/api/users/@me/")

    assert "sp-secret-proxy-pass" not in str(excinfo.value)


def test_request_raises_with_structured_error_detail(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={
                    "type": "authentication_error",
                    "code": "invalid_personal_api_key",
                    "detail": "Invalid Personal API key.",
                },
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Invalid Personal API key"):
        posthog._request("GET", "/api/users/@me/")


def test_request_redacts_bearer_token_in_error_detail(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                status_code=502,
                json_data={
                    "detail": (
                        "Upstream error, request had Authorization: "
                        "Bearer sk-super-secret-token-12345 rejected"
                    )
                },
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        posthog._request("GET", "/api/users/@me/")

    assert "sk-super-secret-token-12345" not in str(excinfo.value)
    assert "Bearer ***" in str(excinfo.value)


def test_request_raises_clear_error_for_non_json_2xx_body(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                status_code=200, text="<html>not json</html>", json_raises=True
            )
        ),
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        posthog._request("GET", "/api/users/@me/")


def test_extract_error_detail_returns_none_for_non_json_body():
    response = MockResponse(status_code=500, text="not json", json_raises=True)

    assert posthog._extract_error_detail(response) is None


def test_request_truncates_unstructured_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        posthog._request("GET", "/api/users/@me/")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_get_current_user_returns_profile(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "uuid": "u1",
                    "email": "ada@example.com",
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                }
            )
        ),
    )

    result = json.loads(posthog.posthog_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["email"] == "ada@example.com"


def test_get_current_user_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={"detail": "Invalid Personal API key."},
            )
        ),
    )

    result = json.loads(posthog.posthog_get_current_user())

    assert result["status"] == "error"
    assert "Invalid Personal API key" in result["message"]


def test_list_organizations_returns_results_and_truncated_flag(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "results": [{"id": "org1", "name": "Acme"}],
                    "next": "https://us.posthog.com/api/organizations/?offset=50",
                }
            )
        ),
    )

    result = json.loads(posthog.posthog_list_organizations())

    assert result["status"] == "success"
    assert result["organizations"] == [{"id": "org1", "name": "Acme"}]
    assert result["truncated"] is True
    assert result["next_offset"] == 1


def test_list_organizations_not_truncated_when_no_next_page(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={"results": [{"id": "org1", "name": "Acme"}], "next": None}
            )
        ),
    )

    result = json.loads(posthog.posthog_list_organizations())

    assert result["status"] == "success"
    assert result["truncated"] is False
    assert result["next_offset"] is None


def test_list_organizations_passes_offset_and_returns_next_offset(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [{"id": "org1"}],
                "next": "https://us.posthog.com/api/organizations/?offset=51",
            }
        )
    )
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = json.loads(posthog.posthog_list_organizations(offset=50))

    assert mock_request.call_args.kwargs["params"]["offset"] == 50
    assert result["next_offset"] == 51


def test_list_organizations_clamps_negative_offset(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    posthog.posthog_list_organizations(offset=-5)

    assert mock_request.call_args.kwargs["params"]["offset"] == 0


def test_list_projects_uses_organization_id_in_path(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": [{"id": 1, "name": "Default Project"}]}
        )
    )
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = json.loads(posthog.posthog_list_projects(organization_id="org1"))

    assert result["status"] == "success"
    assert result["projects"] == [{"id": 1, "name": "Default Project"}]
    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/organizations/org1/projects/"
    )


def test_list_projects_defaults_organization_id_to_current(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    posthog.posthog_list_projects()

    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/organizations/@current/projects/"
    )


def test_list_projects_encodes_hostile_organization_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    posthog.posthog_list_projects(organization_id="1/../2")

    url = mock_request.call_args.kwargs["url"]
    assert "/../" not in url
    assert url.endswith("/api/organizations/1%2F..%2F2/projects/")


def test_query_sends_hogql_query_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "columns": ["event", "count"],
                "results": [["$pageview", 42]],
                "hogql": "SELECT event, count() FROM events",
            }
        )
    )
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = json.loads(
        posthog.posthog_query("select event, count() from events", name="my query")
    )

    assert result["status"] == "success"
    assert result["results"] == [["$pageview", 42]]
    assert mock_request.call_args.kwargs["json"] == {
        "query": {
            "kind": "HogQLQuery",
            "query": "select event, count() from events",
        },
        "name": "my query",
    }
    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/projects/@current/query/"
    )


def test_query_omits_name_when_not_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    posthog.posthog_query("select 1")

    assert "name" not in mock_request.call_args.kwargs["json"]


def test_query_clamps_results_to_limit_and_reports_truncated(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": [[1], [2], [3]], "columns": ["n"]}
        )
    )
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = json.loads(posthog.posthog_query("select n from numbers", limit=2))

    assert result["results"] == [[1], [2]]
    assert result["truncated"] is True


def test_query_not_truncated_when_results_within_limit(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [[1]], "columns": ["n"]})
    )
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = json.loads(posthog.posthog_query("select 1", limit=50))

    assert result["truncated"] is False


def test_query_reports_clear_error_for_malformed_results_shape(monkeypatch):
    # Exercises the same _paginated_results shape guard the list tools get,
    # rather than a silent bad slice (e.g. slicing a dict, or slicing a
    # string into a garbled row) or an opaque AttributeError/TypeError.
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(return_value=MockResponse(json_data={"results": {"not": "a-list"}})),
    )

    result = json.loads(posthog.posthog_query("select 1"))

    assert result["status"] == "error"
    assert "results" in result["message"]


def test_list_persons_includes_search_param(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": [{"id": 1, "distinct_ids": ["abc"]}]}
        )
    )
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = json.loads(posthog.posthog_list_persons(search="ada@example.com"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["search"] == "ada@example.com"


def test_get_person_returns_person(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"id": 1, "distinct_ids": ["abc"]})
    )
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = json.loads(posthog.posthog_get_person("1"))

    assert result["status"] == "success"
    assert result["person"]["id"] == 1
    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/projects/@current/persons/1/"
    )


def test_get_person_encodes_hostile_person_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": 1}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    posthog.posthog_get_person("1/../2", project_id="proj1")

    url = mock_request.call_args.kwargs["url"]
    assert "/../" not in url
    assert url.endswith("/api/projects/proj1/persons/1%2F..%2F2/")


def test_list_insights_requests_basic_shape(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [{"id": 1, "name": "Signups"}]})
    )
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = json.loads(posthog.posthog_list_insights())

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["basic"] == "true"


def test_get_insight_returns_insight(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(return_value=MockResponse(json_data={"id": 1, "name": "Signups"})),
    )

    result = json.loads(posthog.posthog_get_insight("1"))

    assert result["status"] == "success"
    assert result["insight"]["name"] == "Signups"


def test_list_feature_flags_returns_results(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={"results": [{"id": 1, "key": "new-onboarding"}]}
            )
        ),
    )

    result = json.loads(posthog.posthog_list_feature_flags())

    assert result["status"] == "success"
    assert result["feature_flags"] == [{"id": 1, "key": "new-onboarding"}]


def test_list_dashboards_returns_results(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={"results": [{"id": 1, "name": "KPIs"}]}
            )
        ),
    )

    result = json.loads(posthog.posthog_list_dashboards())

    assert result["status"] == "success"
    assert result["dashboards"] == [{"id": 1, "name": "KPIs"}]


def test_list_annotations_returns_results(monkeypatch):
    monkeypatch.setattr(
        posthog,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={"results": [{"id": 1, "content": "Deployed v2"}]}
            )
        ),
    )

    result = json.loads(posthog.posthog_list_annotations())

    assert result["status"] == "success"
    assert result["annotations"] == [{"id": 1, "content": "Deployed v2"}]


def test_create_annotation_sends_content_and_scope(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"id": 1, "content": "Deployed v2"})
    )
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    result = json.loads(posthog.posthog_create_annotation("Deployed v2", "proj1"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"] == {
        "content": "Deployed v2",
        "scope": "project",
    }
    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/projects/proj1/annotations/"
    )


def test_create_annotation_includes_date_marker_when_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": 1}))
    monkeypatch.setattr(posthog, "_make_request", mock_request)

    posthog.posthog_create_annotation(
        "Deployed v2", "proj1", date_marker="2026-08-18T00:00:00Z"
    )

    assert (
        mock_request.call_args.kwargs["json"]["date_marker"] == "2026-08-18T00:00:00Z"
    )


def test_create_annotation_requires_project_id():
    with pytest.raises(TypeError):
        posthog.posthog_create_annotation("Deployed v2")


def test_posthog_app_registry_requires_api_key_and_host():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    posthog_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "posthog"
    )
    assert posthog_app["provider_name"] is None
    assert posthog_app["launch_config"]["required_env"] == [
        "POSTHOG_API_KEY",
        "POSTHOG_HOST",
    ]
