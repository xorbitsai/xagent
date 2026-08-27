import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import deputy
from xagent.web.tools.mcp import utils as mcp_utils


class MockResponse:
    def __init__(self, json_data=None, status_code: int = 200, text: str = ""):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("DEPUTY_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("DEPUTY_INSTANCE_URL", "https://acme.au.deputy.com")


# ---------------------------------------------------------------------------
# _headers
# ---------------------------------------------------------------------------


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("DEPUTY_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="DEPUTY_ACCESS_TOKEN"):
        deputy._headers()


def test_headers_include_bearer_token():
    assert deputy._headers() == {
        "Authorization": "Bearer access-token",
        "Content-Type": "application/json",
    }


def test_headers_reject_whitespace_only_token(monkeypatch):
    monkeypatch.setenv("DEPUTY_ACCESS_TOKEN", "   ")

    with pytest.raises(ValueError, match="DEPUTY_ACCESS_TOKEN"):
        deputy._headers()


def test_headers_strip_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("DEPUTY_ACCESS_TOKEN", "  access-token\n")

    assert deputy._headers() == {
        "Authorization": "Bearer access-token",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# _instance_url
# ---------------------------------------------------------------------------


def test_instance_url_requires_env_var(monkeypatch):
    monkeypatch.delenv("DEPUTY_INSTANCE_URL")

    with pytest.raises(ValueError, match="DEPUTY_INSTANCE_URL"):
        deputy._instance_url()


def test_instance_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("DEPUTY_INSTANCE_URL", "https://acme.au.deputy.com/")

    assert deputy._instance_url() == "https://acme.au.deputy.com"


@pytest.mark.parametrize(
    "value",
    [
        "acme.au.deputy.com",  # no scheme at all
        "http://acme.au.deputy.com",  # not https
        "https://attacker.example.com",  # wrong host entirely
        "https://deputy.com.attacker.com",  # suffix-match bypass attempt
    ],
)
def test_instance_url_rejects_invalid_hosts(monkeypatch, value):
    monkeypatch.setenv("DEPUTY_INSTANCE_URL", value)

    with pytest.raises(ValueError, match="not a valid Deputy host"):
        deputy._instance_url()


@pytest.mark.parametrize(
    "value",
    [
        "https://acme.au.deputy.com",
        "https://deputy.com",
    ],
)
def test_instance_url_accepts_real_deputy_hosts(monkeypatch, value):
    monkeypatch.setenv("DEPUTY_INSTANCE_URL", value)

    assert deputy._instance_url() == value


def test_instance_url_preserves_non_default_port(monkeypatch):
    monkeypatch.setenv("DEPUTY_INSTANCE_URL", "https://acme.au.deputy.com:8443")

    assert deputy._instance_url() == "https://acme.au.deputy.com:8443"


def test_instance_url_strips_path_query_and_userinfo(monkeypatch):
    """_instance_url() is used as a raw prefix for every outbound request
    URL, so an extra path/query/userinfo component that passed the
    scheme+host check would otherwise silently ride along into every
    request this connector makes."""
    monkeypatch.setenv(
        "DEPUTY_INSTANCE_URL",
        "https://user:pw@acme.au.deputy.com/evil/path?x=1",
    )

    assert deputy._instance_url() == "https://acme.au.deputy.com"


def test_instance_url_rejects_non_numeric_port_with_clear_message(monkeypatch):
    monkeypatch.setenv("DEPUTY_INSTANCE_URL", "https://acme.au.deputy.com:abc")

    with pytest.raises(ValueError, match="not a valid Deputy host"):
        deputy._instance_url()


def test_instance_url_strips_trailing_dot_from_hostname(monkeypatch):
    monkeypatch.setenv("DEPUTY_INSTANCE_URL", "https://acme.au.deputy.com.")

    assert deputy._instance_url() == "https://acme.au.deputy.com"


def test_instance_url_rejects_ipv6_literal_host_with_clear_message(monkeypatch):
    """urlparse() itself (not just the .port access) raises ValueError on
    an IPv6-literal-like host -- must not escape uncaught."""
    monkeypatch.setenv("DEPUTY_INSTANCE_URL", "https://[::1].deputy.com")

    with pytest.raises(ValueError, match="not a valid Deputy host"):
        deputy._instance_url()


# ---------------------------------------------------------------------------
# _extract_error_detail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["error_description", "error", "Message", "message"])
def test_extract_error_detail_tries_each_fallback_key(key):
    response = MockResponse(json_data={key: "something went wrong"})

    assert deputy._extract_error_detail(response) == "something went wrong"


def test_extract_error_detail_returns_none_for_non_json_body():
    response = MockResponse(text="<html>gateway error</html>")

    def _raise():
        raise ValueError("not json")

    response.json = _raise  # type: ignore[method-assign]

    assert deputy._extract_error_detail(response) is None


def test_extract_error_detail_returns_none_for_non_dict_body():
    response = MockResponse(json_data=["unexpected", "array"])

    assert deputy._extract_error_detail(response) is None


def test_extract_error_detail_prefers_first_matching_key():
    response = MockResponse(
        json_data={"error_description": "primary", "error": "secondary"}
    )

    assert deputy._extract_error_detail(response) == "primary"


# ---------------------------------------------------------------------------
# deputy_get_current_user
# ---------------------------------------------------------------------------


def test_get_current_user_returns_profile(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"Id": 123, "FirstName": "Ada", "Email": "ada@example.com"}
        )
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["Email"] == "ada@example.com"
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.au.deputy.com/api/v1/me"
    )
    assert mock_request.call_args.kwargs["method"] == "GET"
    assert (
        mock_request.call_args.kwargs["headers"]["Authorization"]
        == "Bearer access-token"
    )


def test_get_current_user_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={"error_description": "Session expired"},
            )
        ),
    )

    result = json.loads(deputy.deputy_get_current_user())

    assert result["status"] == "error"
    assert "Session expired" in result["message"]


def test_get_current_user_rejects_non_dict_response(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(json_data=["unexpected"])),
    )

    result = json.loads(deputy.deputy_get_current_user())

    assert result["status"] == "error"


# ---------------------------------------------------------------------------
# deputy_list_resource
# ---------------------------------------------------------------------------


def test_list_resource_returns_records(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data=[{"Id": 1, "Name": "Ada"}, {"Id": 2, "Name": "Bob"}]
        )
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_list_resource("Employee"))

    assert result["status"] == "success"
    assert result["records"] == [{"Id": 1, "Name": "Ada"}, {"Id": 2, "Name": "Bob"}]
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.au.deputy.com/api/v1/resource/Employee"
    )
    assert mock_request.call_args.kwargs["method"] == "GET"


def test_list_resource_returns_error_on_failure(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="gateway error")),
    )

    result = json.loads(deputy.deputy_list_resource("Employee"))

    assert result["status"] == "error"


def test_list_resource_errors_on_non_list_response(monkeypatch):
    """A non-list body is an unexpected Deputy response shape, not
    "genuinely zero records" -- must surface as an error (matching
    deputy_get_resource/deputy_get_current_user's own unexpected-shape
    handling), not silently coerce to an empty, indistinguishable-from-
    real-zero-records list."""
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"unexpected": "shape"})),
    )

    result = json.loads(deputy.deputy_list_resource("Employee"))

    assert result["status"] == "error"


class _NullJsonResponse:
    """A response whose body is the literal JSON `null` -- a common REST
    idiom (e.g. an ASP.NET-style API serializing a null collection
    reference) for "no records", distinct from an unexpected shape like a
    dict or string."""

    status_code = 200
    text = "null"
    content = b"null"

    def json(self):
        return None


def test_list_resource_treats_null_response_as_empty(monkeypatch):
    monkeypatch.setattr(
        deputy.requests, "request", Mock(return_value=_NullJsonResponse())
    )

    result = json.loads(deputy.deputy_list_resource("Employee"))

    assert result["status"] == "success"
    assert result["records"] == []


def test_list_resource_rejects_surrounding_whitespace_resource_without_raising(
    monkeypatch,
):
    # url_path_id (via require_clean_identifier) only rejects *surrounding*
    # whitespace, not internal whitespace (which is a legal, percent-encoded
    # segment character) -- so this must use a leading/trailing space, not
    # an embedded one, to actually exercise the rejection path.
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_list_resource(" Employee"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_list_resource_rejects_empty_resource_without_raising(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_list_resource(""))

    assert result["status"] == "error"
    mock_request.assert_not_called()


# ---------------------------------------------------------------------------
# deputy_get_resource
# ---------------------------------------------------------------------------


def test_get_resource_returns_record(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Id": 123, "Name": "Ada"}))
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_get_resource("Employee", "123"))

    assert result["status"] == "success"
    assert result["record"] == {"Id": 123, "Name": "Ada"}
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.au.deputy.com/api/v1/resource/Employee/123"
    )


def test_get_resource_returns_error_on_failure(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(status_code=404, json_data={"error": "gone"})),
    )

    result = json.loads(deputy.deputy_get_resource("Employee", "999"))

    assert result["status"] == "error"
    assert "gone" in result["message"]


def test_get_resource_rejects_non_dict_response(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(json_data=["unexpected"])),
    )

    result = json.loads(deputy.deputy_get_resource("Employee", "123"))

    assert result["status"] == "error"


@pytest.mark.parametrize(
    "resource,resource_id",
    [
        (" Employee", "123"),
        ("", "123"),
        ("Employee", " 123"),
        ("Employee", ""),
    ],
)
def test_get_resource_rejects_invalid_ids_without_raising(
    monkeypatch, resource, resource_id
):
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_get_resource(resource, resource_id))

    assert result["status"] == "error"
    mock_request.assert_not_called()


# ---------------------------------------------------------------------------
# deputy_query_resource
# ---------------------------------------------------------------------------


def test_query_resource_omits_falsy_optional_params(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=[]))
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    deputy.deputy_query_resource("Roster")

    assert mock_request.call_args.kwargs["json"] == {}
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.au.deputy.com/api/v1/resource/Roster/QUERY"
    )
    assert mock_request.call_args.kwargs["method"] == "POST"


def test_query_resource_includes_only_provided_params(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=[]))
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    search = {"s1": {"field": "Date", "data": "2026-08-01", "type": "gt"}}
    deputy.deputy_query_resource("Roster", search=search)

    assert mock_request.call_args.kwargs["json"] == {"search": search}


def test_query_resource_includes_all_provided_params(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=[]))
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    search = {"s1": {"field": "Date", "data": "2026-08-01", "type": "gt"}}
    sort = {"field": "Date", "order": "asc"}
    join = ["TimesheetObject"]
    deputy.deputy_query_resource("Roster", search=search, sort=sort, join=join)

    assert mock_request.call_args.kwargs["json"] == {
        "search": search,
        "sort": sort,
        "join": join,
    }


def test_query_resource_returns_records(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data=[{"Id": 1}, {"Id": 2}, {"Id": 3}])
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_query_resource("Roster"))

    assert result["status"] == "success"
    assert result["records"] == [{"Id": 1}, {"Id": 2}, {"Id": 3}]


def test_query_resource_returns_error_on_failure(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(status_code=400, json_data={"Message": "bad"})),
    )

    result = json.loads(deputy.deputy_query_resource("Roster"))

    assert result["status"] == "error"
    assert "bad" in result["message"]


def test_query_resource_errors_on_non_list_response(monkeypatch):
    """See test_list_resource_errors_on_non_list_response's docstring --
    same reasoning applies here."""
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"unexpected": "shape"})),
    )

    result = json.loads(deputy.deputy_query_resource("Roster"))

    assert result["status"] == "error"


def test_query_resource_treats_null_response_as_empty(monkeypatch):
    monkeypatch.setattr(
        deputy.requests, "request", Mock(return_value=_NullJsonResponse())
    )

    result = json.loads(deputy.deputy_query_resource("Roster"))

    assert result["status"] == "success"
    assert result["records"] == []


def test_query_resource_rejects_surrounding_whitespace_resource_without_raising(
    monkeypatch,
):
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_query_resource(" Roster"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


# ---------------------------------------------------------------------------
# _success_with_capped_list
# ---------------------------------------------------------------------------


def test_capped_list_halves_and_reports_message_when_output_too_large(monkeypatch):
    monkeypatch.setattr(deputy, "get_tool_max_output_length", lambda: 200)

    raw = deputy._success_with_capped_list(
        "records", [{"Id": i, "Name": "x" * 50} for i in range(50)]
    )
    result = json.loads(raw)

    assert len(raw) <= 200
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["records"]) < 50
    assert "cannot be recovered" in result["message"]


def test_capped_list_drops_message_when_it_alone_exceeds_the_limit(monkeypatch):
    monkeypatch.setattr(deputy, "get_tool_max_output_length", lambda: 40)

    raw = deputy._success_with_capped_list(
        "records", [{"Id": str(i)} for i in range(50)]
    )
    result = json.loads(raw)

    assert result["records"] == []
    assert result["truncated"] is True
    assert "message" not in result


def test_capped_list_does_not_halve_when_output_already_fits(monkeypatch):
    monkeypatch.setattr(deputy, "get_tool_max_output_length", lambda: 100_000)

    raw = deputy._success_with_capped_list("records", [{"Id": 1}, {"Id": 2}])
    result = json.loads(raw)

    assert result["records"] == [{"Id": 1}, {"Id": 2}]
    assert result["truncated"] is False
    assert "message" not in result


# ---------------------------------------------------------------------------
# success_with_capped_dict (via deputy_get_current_user / deputy_get_resource)
# ---------------------------------------------------------------------------


def test_get_current_user_caps_output_size(monkeypatch):
    big_record = {"Id": 123, "Notes": "x" * 5000}
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(json_data=big_record)),
    )
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 200)

    raw = deputy.deputy_get_current_user()
    result = json.loads(raw)

    assert len(raw) <= 200
    assert result["status"] == "success"
    assert result["truncated"] is True
