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
    ):
        self._json_data = json_data if json_data is not None else {}
        self._json_raises = json_raises
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()
        self.url = url

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


def test_base_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "https://eu.posthog.com/")

    assert posthog._base_url() == "https://eu.posthog.com"


def test_base_url_rejects_whitespace_only_host(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "   ")

    with pytest.raises(ValueError, match="POSTHOG_HOST"):
        posthog._base_url()


def test_base_url_rejects_slash_only_host(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "/")

    with pytest.raises(ValueError, match="hostname"):
        posthog._base_url()


def test_base_url_prepends_https_when_scheme_missing(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com")

    assert posthog._base_url() == "https://us.posthog.com"


def test_base_url_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "  https://eu.posthog.com  ")

    assert posthog._base_url() == "https://eu.posthog.com"


def test_base_url_accepts_uppercase_scheme(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "HTTPS://us.posthog.com")

    assert posthog._base_url() == "https://us.posthog.com"


def test_base_url_strips_multiple_trailing_slashes(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "https://eu.posthog.com///")

    assert posthog._base_url() == "https://eu.posthog.com"


def test_base_url_rejects_host_resolving_to_private_ip(monkeypatch):
    # A posthog.com host (so it clears the domain allowlist below) that
    # only *resolves* to a private address -- the DNS-rebinding case the
    # literal-string check in test_base_url_rejects_private_network_host
    # can't catch on its own.
    monkeypatch.setenv("POSTHOG_HOST", "fake.posthog.com")
    monkeypatch.setattr(posthog.socket, "getaddrinfo", _fake_getaddrinfo("10.0.0.5"))

    with pytest.raises(ValueError, match="not allowed"):
        posthog._base_url()


def test_base_url_rejects_when_any_resolved_address_is_private(monkeypatch):
    # A hostname resolving to more than one address (common for a load
    # balanced service) must be rejected if ANY resolved address is
    # private, not just the first one checked.
    monkeypatch.setenv("POSTHOG_HOST", "fake.posthog.com")
    monkeypatch.setattr(
        posthog.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1", "10.0.0.5")
    )

    with pytest.raises(ValueError, match="not allowed"):
        posthog._base_url()


def test_base_url_rejects_ipv6_private_resolved_address(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "fake.posthog.com")
    monkeypatch.setattr(posthog.socket, "getaddrinfo", _fake_getaddrinfo("fe80::1"))

    with pytest.raises(ValueError, match="not allowed"):
        posthog._base_url()


def test_base_url_raises_when_dns_resolution_fails(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "fake.posthog.com")

    def _raise(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(posthog.socket, "getaddrinfo", _raise)

    with pytest.raises(ValueError, match="could not be resolved"):
        posthog._base_url()


def test_base_url_preserves_explicit_port(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com:8443")

    assert posthog._base_url() == "https://us.posthog.com:8443"


def test_base_url_rejects_invalid_port(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com:notaport")

    with pytest.raises(ValueError, match="port"):
        posthog._base_url()


def test_base_url_rejects_scheme_only_host(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "https://")

    with pytest.raises(ValueError, match="hostname"):
        posthog._base_url()


def test_base_url_rejects_non_posthog_domain(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "https://evil.example.com")

    with pytest.raises(ValueError, match="posthog.com"):
        posthog._base_url()


def test_base_url_rejects_domain_that_merely_ends_with_posthog_com(monkeypatch):
    # A naive `host.endswith("posthog.com")` (missing the leading dot)
    # would wrongly accept this -- the real check requires a "." boundary.
    monkeypatch.setenv("POSTHOG_HOST", "https://evilposthog.com")

    with pytest.raises(ValueError, match="posthog.com"):
        posthog._base_url()


def test_base_url_accepts_posthog_com_apex_domain(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "https://posthog.com")

    assert posthog._base_url() == "https://posthog.com"


def test_base_url_accepts_trailing_dns_root_dot(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "https://us.posthog.com.")

    assert posthog._base_url() == "https://us.posthog.com"


def test_base_url_rejects_http_scheme(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "http://us.posthog.com")

    with pytest.raises(ValueError, match="https"):
        posthog._base_url()


def test_base_url_rejects_embedded_credentials(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "user:pass@us.posthog.com")

    with pytest.raises(ValueError, match="credentials"):
        posthog._base_url()


def test_base_url_rejects_path_in_host(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com/api")

    with pytest.raises(ValueError, match="path"):
        posthog._base_url()


def test_base_url_rejects_query_in_host(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com?x=1")

    with pytest.raises(ValueError, match="path"):
        posthog._base_url()


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "localhost",
        "169.254.169.254",
        "10.0.0.5",
        "::1",
        "2130706433",  # decimal-encoded 127.0.0.1
        "0x7f000001",  # hex-encoded 127.0.0.1
    ],
)
def test_base_url_rejects_private_network_host(monkeypatch, host):
    # None of these are a posthog.com host, so most are rejected by the
    # domain allowlist without even reaching DNS resolution -- still
    # correct, just via a more fundamental gate than the literal-IP check
    # alone. "::1" is the exception: urlsplit can't parse a bracket-less
    # IPv6 literal, so parsed.hostname comes back empty and it's rejected
    # by the earlier "must include a hostname" check instead.
    monkeypatch.setenv("POSTHOG_HOST", host)

    with pytest.raises(ValueError, match="POSTHOG_HOST"):
        posthog._base_url()


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
        posthog.requests,
        "request",
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
    monkeypatch.setattr(posthog.requests, "request", mock_request)

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


def test_request_rejects_redirect_response(monkeypatch):
    monkeypatch.setattr(
        posthog.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=302, url="https://us.posthog.com/api/users/@me/"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="redirect"):
        posthog._request("GET", "/api/users/@me/")


def test_request_raises_with_structured_error_detail(monkeypatch):
    monkeypatch.setattr(
        posthog.requests,
        "request",
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
        posthog.requests,
        "request",
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
        posthog.requests,
        "request",
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
        posthog.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        posthog._request("GET", "/api/users/@me/")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_get_current_user_returns_profile(monkeypatch):
    monkeypatch.setattr(
        posthog.requests,
        "request",
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
        posthog.requests,
        "request",
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
        posthog.requests,
        "request",
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
        posthog.requests,
        "request",
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
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    result = json.loads(posthog.posthog_list_organizations(offset=50))

    assert mock_request.call_args.kwargs["params"]["offset"] == 50
    assert result["next_offset"] == 51


def test_list_organizations_clamps_negative_offset(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    posthog.posthog_list_organizations(offset=-5)

    assert mock_request.call_args.kwargs["params"]["offset"] == 0


def test_list_projects_uses_organization_id_in_path(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": [{"id": 1, "name": "Default Project"}]}
        )
    )
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    result = json.loads(posthog.posthog_list_projects(organization_id="org1"))

    assert result["status"] == "success"
    assert result["projects"] == [{"id": 1, "name": "Default Project"}]
    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/organizations/org1/projects/"
    )


def test_list_projects_defaults_organization_id_to_current(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    posthog.posthog_list_projects()

    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/organizations/@current/projects/"
    )


def test_list_projects_encodes_hostile_organization_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(posthog.requests, "request", mock_request)

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
    monkeypatch.setattr(posthog.requests, "request", mock_request)

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
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    posthog.posthog_query("select 1")

    assert "name" not in mock_request.call_args.kwargs["json"]


def test_query_clamps_results_to_limit_and_reports_truncated(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": [[1], [2], [3]], "columns": ["n"]}
        )
    )
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    result = json.loads(posthog.posthog_query("select n from numbers", limit=2))

    assert result["results"] == [[1], [2]]
    assert result["truncated"] is True


def test_query_not_truncated_when_results_within_limit(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [[1]], "columns": ["n"]})
    )
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    result = json.loads(posthog.posthog_query("select 1", limit=50))

    assert result["truncated"] is False


def test_query_reports_clear_error_for_malformed_results_shape(monkeypatch):
    # Exercises the same _paginated_results shape guard the list tools get,
    # rather than a silent bad slice (e.g. slicing a dict, or slicing a
    # string into a garbled row) or an opaque AttributeError/TypeError.
    monkeypatch.setattr(
        posthog.requests,
        "request",
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
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    result = json.loads(posthog.posthog_list_persons(search="ada@example.com"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["search"] == "ada@example.com"


def test_get_person_returns_person(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"id": 1, "distinct_ids": ["abc"]})
    )
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    result = json.loads(posthog.posthog_get_person("1"))

    assert result["status"] == "success"
    assert result["person"]["id"] == 1
    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/projects/@current/persons/1/"
    )


def test_get_person_encodes_hostile_person_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": 1}))
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    posthog.posthog_get_person("1/../2", project_id="proj1")

    url = mock_request.call_args.kwargs["url"]
    assert "/../" not in url
    assert url.endswith("/api/projects/proj1/persons/1%2F..%2F2/")


def test_list_insights_requests_basic_shape(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [{"id": 1, "name": "Signups"}]})
    )
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    result = json.loads(posthog.posthog_list_insights())

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["params"]["basic"] == "true"


def test_get_insight_returns_insight(monkeypatch):
    monkeypatch.setattr(
        posthog.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"id": 1, "name": "Signups"})),
    )

    result = json.loads(posthog.posthog_get_insight("1"))

    assert result["status"] == "success"
    assert result["insight"]["name"] == "Signups"


def test_list_feature_flags_returns_results(monkeypatch):
    monkeypatch.setattr(
        posthog.requests,
        "request",
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
        posthog.requests,
        "request",
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
        posthog.requests,
        "request",
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
    monkeypatch.setattr(posthog.requests, "request", mock_request)

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
    monkeypatch.setattr(posthog.requests, "request", mock_request)

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
