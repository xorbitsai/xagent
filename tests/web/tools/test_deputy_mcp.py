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


def test_list_resource_treats_204_response_as_empty(monkeypatch):
    """_request() normalizes a 204/empty-content response to `{}` (needed
    so deputy_get_resource/deputy_get_current_user treat it as a valid
    empty dict) -- deputy_list_resource must also recognize that `{}`,
    not just a literal JSON null, as "no records", not an unexpected
    shape."""
    monkeypatch.setattr(
        deputy.requests, "request", Mock(return_value=MockResponse(status_code=204))
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
# deputy_resource_info
# ---------------------------------------------------------------------------


def test_resource_info_returns_field_metadata(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"FirstName": {"type": "String", "required": True}}
        )
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_resource_info("Employee"))

    assert result["status"] == "success"
    assert result["info"] == {"FirstName": {"type": "String", "required": True}}
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.au.deputy.com/api/v1/resource/Employee/INFO"
    )
    assert mock_request.call_args.kwargs["method"] == "GET"


def test_resource_info_returns_error_on_failure(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(status_code=404, json_data={"error": "gone"})),
    )

    result = json.loads(deputy.deputy_resource_info("Employee"))

    assert result["status"] == "error"
    assert "gone" in result["message"]


def test_resource_info_rejects_non_dict_response(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(json_data=["unexpected"])),
    )

    result = json.loads(deputy.deputy_resource_info("Employee"))

    assert result["status"] == "error"


def test_resource_info_rejects_invalid_resource_without_raising(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_resource_info(" Employee"))

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


def test_query_resource_treats_204_response_as_empty(monkeypatch):
    """See test_list_resource_treats_204_response_as_empty's docstring --
    same reasoning applies here."""
    monkeypatch.setattr(
        deputy.requests, "request", Mock(return_value=MockResponse(status_code=204))
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
# deputy_create_resource
# ---------------------------------------------------------------------------


def test_create_resource_sends_data_and_returns_record(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"Id": 123, "FirstName": "Peter"})
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_create_resource("Roster", {"FirstName": "Peter"}))

    assert result["status"] == "success"
    assert result["record"] == {"Id": 123, "FirstName": "Peter"}
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.au.deputy.com/api/v1/resource/Roster"
    )
    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["json"] == {"FirstName": "Peter"}


def test_create_resource_rejects_employee_without_calling_api(monkeypatch):
    """Employee must go through deputy_add_employee instead: a bare
    POST /resource/Employee inserts a row with no location/workplace
    membership, which Deputy's permission model then hides from list/get
    calls (403 "Access to object denied") even though the write itself
    succeeded and returned an id -- see incident from 2026-09-21."""
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(
        deputy.deputy_create_resource("Employee", {"FirstName": "Peter"})
    )

    assert result["status"] == "error"
    assert "deputy_add_employee" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize("resource", ["employee", "EMPLOYEE", " Employee", "Employee "])
def test_create_resource_rejects_employee_regardless_of_case_or_whitespace(
    monkeypatch, resource
):
    """The guard must not be an exact-string match a caller can bypass by
    guessing a different casing -- that would silently fall through to a
    live POST and reproduce the exact incident this rejection exists to
    prevent."""
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_create_resource(resource, {"FirstName": "Peter"}))

    assert result["status"] == "error"
    assert "deputy_add_employee" in result["message"]
    mock_request.assert_not_called()


def test_create_resource_returns_error_on_failure(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400, json_data={"error": "invalid field"}
            )
        ),
    )

    result = json.loads(deputy.deputy_create_resource("Roster", {"Bad": "x"}))

    assert result["status"] == "error"
    assert "invalid field" in result["message"]


def test_create_resource_rejects_non_dict_response(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(json_data=["unexpected"])),
    )

    result = json.loads(deputy.deputy_create_resource("Roster", {"FirstName": "P"}))

    assert result["status"] == "error"


def test_create_resource_rejects_invalid_resource_without_raising(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_create_resource(" Roster", {"FirstName": "P"}))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_resource_rejects_empty_data_without_calling_api(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_create_resource("Roster", {}))

    assert result["status"] == "error"
    assert "No data provided" in result["message"]
    mock_request.assert_not_called()


def test_create_resource_strips_caller_supplied_id(monkeypatch):
    """ "Id" is server-assigned on create; a caller-supplied value must not
    be forwarded, since whether Deputy would honor, ignore, or reject a
    client-chosen id is undocumented."""
    mock_request = Mock(
        return_value=MockResponse(json_data={"Id": 123, "FirstName": "Peter"})
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    deputy.deputy_create_resource("Roster", {"Id": 999, "FirstName": "Peter"})

    assert mock_request.call_args.kwargs["json"] == {"FirstName": "Peter"}


def test_create_resource_rejects_id_only_data_without_calling_api(monkeypatch):
    """An "Id"-only data dict has nothing left to create once "Id" is
    stripped -- must be rejected the same as genuinely empty data, not
    silently posted as an empty body."""
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_create_resource("Roster", {"Id": 999}))

    assert result["status"] == "error"
    assert "No data provided" in result["message"]
    mock_request.assert_not_called()


def test_create_resource_warns_instead_of_confident_success_on_empty_response(
    monkeypatch,
):
    """A 204/empty response (_request normalizes both to {}) after an
    otherwise-successful POST means the record may exist with no id to
    confirm or safely retry against -- must not be reported as a plain,
    confident success with no signal that the id is unknown."""
    monkeypatch.setattr(
        deputy.requests, "request", Mock(return_value=MockResponse(status_code=204))
    )

    result = json.loads(deputy.deputy_create_resource("Roster", {"FirstName": "Peter"}))

    assert result["status"] == "success"
    assert result["record"] == {}
    assert "warning" in result
    assert "deputy_query_resource" in result["warning"]


# ---------------------------------------------------------------------------
# deputy_add_employee
# ---------------------------------------------------------------------------


def test_add_employee_sends_required_fields_and_returns_record(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"Id": 5, "DisplayName": "Peter Parker"})
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_add_employee("Peter", "Parker", 1))

    assert result["status"] == "success"
    assert result["record"] == {"Id": 5, "DisplayName": "Peter Parker"}
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.au.deputy.com/api/v1/supervise/employee"
    )
    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["json"] == {
        "strFirstName": "Peter",
        "strLastName": "Parker",
        "intCompanyId": 1,
        "blnSendInvite": 0,
    }


def test_add_employee_includes_optional_fields_when_given(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Id": 5}))
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    deputy.deputy_add_employee(
        "Peter",
        "Parker",
        1,
        email="peter.parker@example.com",
        mobile_phone="639322112345",
        role_id=50,
        stress_profile_id=2,
        start_date="2026-09-21",
        date_of_birth="1990-01-01",
        gender=1,
        country_code="AU",
        payroll_id="P123",
        weekday_rate=30.5,
        send_invite=True,
    )

    assert mock_request.call_args.kwargs["json"] == {
        "strFirstName": "Peter",
        "strLastName": "Parker",
        "intCompanyId": 1,
        "blnSendInvite": 1,
        "strEmail": "peter.parker@example.com",
        "strMobilePhone": "639322112345",
        "intRoleId": 50,
        "intStressProfile": 2,
        "strStartDate": "2026-09-21",
        "strDob": "1990-01-01",
        "intGender": 1,
        "strCountryCode": "AU",
        "strPayrollId": "P123",
        "fltWeekDayRate": 30.5,
    }


def test_add_employee_omits_unset_optional_fields(monkeypatch):
    """An optional field left as its default None must not appear in the
    request body at all -- Deputy should see the same request whether the
    caller omitted a field or the tool merely defaults it, not a field
    explicitly set to null."""
    mock_request = Mock(return_value=MockResponse(json_data={"Id": 5}))
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    deputy.deputy_add_employee("Peter", "Parker", 1)

    sent = mock_request.call_args.kwargs["json"]
    for optional_key in (
        "strEmail",
        "strMobilePhone",
        "intRoleId",
        "intStressProfile",
        "strStartDate",
        "strDob",
        "intGender",
        "strCountryCode",
        "strPayrollId",
        "fltWeekDayRate",
    ):
        assert optional_key not in sent


def test_add_employee_returns_error_on_failure(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=417, json_data={"error": {"message": "bad payload"}}
            )
        ),
    )

    result = json.loads(deputy.deputy_add_employee("Peter", "Parker", 1))

    assert result["status"] == "error"


def test_add_employee_rejects_non_dict_response(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(json_data=["unexpected"])),
    )

    result = json.loads(deputy.deputy_add_employee("Peter", "Parker", 1))

    assert result["status"] == "error"


def test_add_employee_warns_instead_of_confident_success_on_empty_response(
    monkeypatch,
):
    """Mirrors deputy_create_resource's identical empty-response handling
    -- this create is equally non-idempotent, so a 204/empty body must not
    read as a plain, confident success with no signal the id is unknown."""
    monkeypatch.setattr(
        deputy.requests, "request", Mock(return_value=MockResponse(status_code=204))
    )

    result = json.loads(deputy.deputy_add_employee("Peter", "Parker", 1))

    assert result["status"] == "success"
    assert result["record"] == {}
    assert "warning" in result
    assert "deputy_query_resource" in result["warning"]


def test_add_employee_is_annotated_as_non_idempotent_non_destructive_write():
    """Must declare idempotentHint=False (a retried create on a timeout
    can duplicate the employee, like deputy_create_resource) so the ReAct
    duplicate-write guard enrolls it, and destructiveHint=False to match
    deputy_create_resource -- this is a pure additive create (per
    mcp_adapter.py's own definition, "destructiveHint false = only
    additive updates"), not an overwrite of an existing record like
    deputy_update_resource."""
    tool = deputy.mcp._tool_manager.get_tool("deputy_add_employee")

    assert tool.annotations is not None
    assert tool.annotations.idempotentHint is False
    assert tool.annotations.destructiveHint is False


# ---------------------------------------------------------------------------
# deputy_update_resource
# ---------------------------------------------------------------------------


def test_update_resource_merges_data_into_the_fetched_record(monkeypatch):
    """Deputy's V1 Resource API requires the full object on update, with
    no partial-update support -- so deputy_update_resource must fetch the
    current record first and merge `data` into it, not send `data` alone,
    or every field the caller didn't mention would be dropped/reset."""
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"Id": 123, "FirstName": "Ada", "Active": True}),
            MockResponse(json_data={"Id": 123, "FirstName": "Ada", "Active": False}),
        ]
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(
        deputy.deputy_update_resource("Employee", "123", {"Active": False})
    )

    assert result["status"] == "success"
    assert result["record"] == {"Id": 123, "FirstName": "Ada", "Active": False}
    assert mock_request.call_count == 2

    get_call, post_call = mock_request.call_args_list
    assert get_call.kwargs["method"] == "GET"
    assert get_call.kwargs["url"] == (
        "https://acme.au.deputy.com/api/v1/resource/Employee/123"
    )
    assert post_call.kwargs["method"] == "POST"
    assert post_call.kwargs["url"] == (
        "https://acme.au.deputy.com/api/v1/resource/Employee/123"
    )
    # The fetched record's other fields (FirstName) are preserved in the
    # write, not dropped just because the caller only mentioned Active.
    assert post_call.kwargs["json"] == {
        "Id": 123,
        "FirstName": "Ada",
        "Active": False,
    }


def test_update_resource_ignores_caller_supplied_id(monkeypatch):
    """The URL's resource_id is what actually identifies the record being
    written; a caller-supplied "Id" in data must not override the fetched
    record's own Id and decouple the body from what the URL targets."""
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"Id": 123, "Active": True}),
            MockResponse(json_data={"Id": 123, "Active": False}),
        ]
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    deputy.deputy_update_resource("Employee", "123", {"Id": 999, "Active": False})

    post_call = mock_request.call_args_list[1]
    assert post_call.kwargs["json"] == {"Id": 123, "Active": False}


def test_update_resource_ignores_caller_supplied_id_even_when_current_lacks_one(
    monkeypatch,
):
    """The old conditional re-add (`{"Id": current["Id"]} if "Id" in
    current else {}`) only protected Id when the fetched record itself
    had one -- stripping it from data unconditionally must protect it
    even when current doesn't have an "Id" key at all."""
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"Active": True}),
            MockResponse(json_data={"Active": False}),
        ]
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    deputy.deputy_update_resource("Employee", "123", {"Id": 999, "Active": False})

    post_call = mock_request.call_args_list[1]
    assert post_call.kwargs["json"] == {"Active": False}


def test_update_resource_rejects_id_only_data_without_calling_api(monkeypatch):
    """An "Id"-only data dict has nothing left to change once "Id" is
    stripped -- must be rejected the same as genuinely empty data,
    not turned into a needless no-op GET+POST round trip."""
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_update_resource("Employee", "123", {"Id": 999}))

    assert result["status"] == "error"
    assert "No data provided" in result["message"]
    mock_request.assert_not_called()


def test_update_resource_rejects_empty_fetch_response(monkeypatch):
    """_request normalizes a 204/empty response to {} -- an empty dict
    passes a bare isinstance(current, dict) check, which would silently
    collapse the merge to just `data` and wipe every other field on the
    record. Must be rejected the same as a non-dict response."""
    mock_request = Mock(return_value=MockResponse(status_code=204))
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(
        deputy.deputy_update_resource("Employee", "123", {"Active": False})
    )

    assert result["status"] == "error"
    mock_request.assert_called_once()


def test_update_resource_rejects_empty_data_without_calling_api(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(deputy.deputy_update_resource("Employee", "123", {}))

    assert result["status"] == "error"
    assert "No data provided" in result["message"]
    mock_request.assert_not_called()


def test_update_resource_returns_error_when_fetch_fails(monkeypatch):
    monkeypatch.setattr(
        deputy.requests,
        "request",
        Mock(return_value=MockResponse(status_code=404, json_data={"error": "gone"})),
    )

    result = json.loads(
        deputy.deputy_update_resource("Employee", "999", {"Active": False})
    )

    assert result["status"] == "error"
    assert "gone" in result["message"]


def test_update_resource_returns_error_when_write_fails_after_fetch(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"Id": 123, "Active": True}),
            MockResponse(status_code=400, json_data={"error": "invalid"}),
        ]
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(
        deputy.deputy_update_resource("Employee", "123", {"Active": False})
    )

    assert result["status"] == "error"
    assert "invalid" in result["message"]
    assert mock_request.call_count == 2


def test_update_resource_rejects_non_dict_fetch_response(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=["unexpected"]))
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(
        deputy.deputy_update_resource("Employee", "123", {"Active": False})
    )

    assert result["status"] == "error"
    # The write must never be attempted once the fetch itself is
    # malformed -- only the GET should have happened.
    mock_request.assert_called_once()


def test_update_resource_rejects_non_dict_write_response(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"Id": 123, "Active": True}),
            MockResponse(json_data=["unexpected"]),
        ]
    )
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(
        deputy.deputy_update_resource("Employee", "123", {"Active": False})
    )

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
def test_update_resource_rejects_invalid_ids_without_raising(
    monkeypatch, resource, resource_id
):
    mock_request = Mock()
    monkeypatch.setattr(deputy.requests, "request", mock_request)

    result = json.loads(
        deputy.deputy_update_resource(resource, resource_id, {"Active": False})
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


# ---------------------------------------------------------------------------
# deputy_create_resource / deputy_update_resource tool annotations
# ---------------------------------------------------------------------------


def test_create_resource_is_annotated_as_non_idempotent_write():
    """deputy_create_resource must declare idempotentHint=False so the
    ReAct duplicate-write guard (classify_non_idempotent_write) enrolls it
    -- a retried create on a timeout/connection error is not safe to
    silently repeat. A future accidental edit that swaps this tool's
    annotations with deputy_update_resource's (or drops them) would
    otherwise leave create unprotected without any test catching it."""
    tool = deputy.mcp._tool_manager.get_tool("deputy_create_resource")

    assert tool.annotations is not None
    assert tool.annotations.idempotentHint is False
    assert tool.annotations.destructiveHint is False


def test_update_resource_is_annotated_as_idempotent_destructive_write():
    """deputy_update_resource must declare idempotentHint=True (repeating
    the same update has no additional effect, so it's safe to retry and
    must NOT be enrolled in the duplicate-write guard) and
    destructiveHint=True (it overwrites existing field values -- e.g.
    deactivating an employee or rewriting a timesheet -- so it is not a
    purely additive write like create)."""
    tool = deputy.mcp._tool_manager.get_tool("deputy_update_resource")

    assert tool.annotations is not None
    assert tool.annotations.idempotentHint is True
    assert tool.annotations.destructiveHint is True


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


def test_success_with_capped_dict_last_resort_fallback_keeps_capitalized_id(
    monkeypatch,
):
    """The severe-truncation last resort used to check only lowercase "id",
    so Deputy's capitalized "Id" (and MYOB's "Uid") was silently dropped
    once a record got truncated all the way down -- exercised directly
    against the shared utility (not through a deputy_* tool) because
    reaching this exact branch also requires extra_fields to push the
    unstripped candidate over the limit, which no deputy_* call site uses."""
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 100)

    raw = mcp_utils.success_with_capped_dict(
        "record",
        {"Id": 123, "Notes": "x" * 5000},
        extra_fields={"note": "y" * 60},
    )
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["record"] == {"Id": 123}
