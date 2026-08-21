import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import salesforce


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
    monkeypatch.setenv("SALESFORCE_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", "https://acme.my.salesforce.com")


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("SALESFORCE_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="SALESFORCE_ACCESS_TOKEN"):
        salesforce._headers()


def test_headers_include_bearer_token():
    assert salesforce._headers() == {
        "Authorization": "Bearer access-token",
        "Content-Type": "application/json",
    }


def test_instance_url_requires_env_var(monkeypatch):
    monkeypatch.delenv("SALESFORCE_INSTANCE_URL")

    with pytest.raises(ValueError, match="SALESFORCE_INSTANCE_URL"):
        salesforce._instance_url()


def test_instance_url_strips_trailing_slash(monkeypatch):
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", "https://acme.my.salesforce.com/")

    assert salesforce._instance_url() == "https://acme.my.salesforce.com"


@pytest.mark.parametrize(
    "value",
    [
        "http://acme.my.salesforce.com",  # not https
        "https://attacker.example.com",  # wrong host entirely
        "https://salesforce.com.attacker.com",  # suffix-match bypass attempt
        # force.com hosts Salesforce Sites/Experience Cloud pages, which can
        # serve customer-authored content -- broader than the OAuth token
        # endpoint's actual instance_url pattern, so deliberately not allowed.
        "https://acme.lightning.force.com",
    ],
)
def test_instance_url_rejects_non_salesforce_hosts(monkeypatch, value):
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", value)

    with pytest.raises(ValueError, match="not a valid Salesforce host"):
        salesforce._instance_url()


@pytest.mark.parametrize(
    "value",
    [
        "https://acme.my.salesforce.com",
        "https://cs123.salesforce.com",
    ],
)
def test_instance_url_accepts_real_salesforce_hosts(monkeypatch, value):
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", value)

    assert salesforce._instance_url() == value


def test_instance_url_strips_path_query_and_userinfo(monkeypatch):
    """_instance_url() is used as a raw prefix for every outbound request
    URL, so a scheme+host check alone isn't enough -- an extra path/query/
    userinfo component that passed that check would otherwise silently
    ride along into every request this connector makes."""
    monkeypatch.setenv(
        "SALESFORCE_INSTANCE_URL",
        "https://user:pw@acme.my.salesforce.com/evil/path?x=1",
    )

    assert salesforce._instance_url() == "https://acme.my.salesforce.com"


def test_instance_url_preserves_non_default_port(monkeypatch):
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", "https://acme.my.salesforce.com:8443")

    assert salesforce._instance_url() == "https://acme.my.salesforce.com:8443"


def test_instance_url_rejects_non_numeric_port_with_clear_message(monkeypatch):
    """urlparse().port is a lazy property that raises a bare ValueError
    ("Port could not be cast to integer value...") on access for a
    non-numeric port -- this must still surface as _instance_url's own
    clear message, not that cryptic one escaping uncaught."""
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", "https://acme.my.salesforce.com:abc")

    with pytest.raises(ValueError, match="not a valid Salesforce host"):
        salesforce._instance_url()


def test_instance_url_strips_trailing_dot_from_hostname(monkeypatch):
    """A trailing-dot FQDN is a valid, equivalent hostname that just
    wouldn't satisfy the endswith host-suffix check otherwise -- accepted,
    and the dot is canonicalized away in the returned value along with
    everything else _instance_url() strips."""
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", "https://acme.my.salesforce.com.")

    assert salesforce._instance_url() == "https://acme.my.salesforce.com"


def test_request_uses_instance_url_and_headers(monkeypatch):
    # A generic sobjects path, not /services/oauth2/userinfo: in production
    # that one is only ever fetched via _request_absolute against the fixed
    # USERINFO_URL host, never through _request/_instance_url like every
    # other tool's path.
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = salesforce._request(
        "GET", f"/services/data/{salesforce.API_VERSION}/sobjects"
    )

    assert result == {"ok": True}
    assert mock_request.call_args.kwargs["url"] == (
        f"https://acme.my.salesforce.com/services/data/{salesforce.API_VERSION}/sobjects"
    )
    assert (
        mock_request.call_args.kwargs["headers"]["Authorization"]
        == "Bearer access-token"
    )


# require_clean_identifier/url_path_id themselves are unit-tested in
# test_mcp_utils.py, where they now live (src/xagent/web/tools/mcp/utils.py) --
# this file keeps only the salesforce-specific call-site integration below.


def test_get_record_percent_encodes_ids_in_the_request_url(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Id": "001xx"}))
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    salesforce.salesforce_get_record("Account/001abc", "001x?fields=Id")

    url = mock_request.call_args.kwargs["url"]
    assert url.endswith(
        "/services/data/v59.0/sobjects/Account%2F001abc/001x%3Ffields%3DId"
    )


def test_get_record_rejects_empty_record_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_get_record("Account", ""))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_get_record_rejects_empty_sobject_type(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_get_record("", "001xx"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_get_record_rejects_dot_segment_record_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_get_record("Account", ".."))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_describe_sobject_rejects_empty_sobject_type(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_describe_sobject(""))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_record_rejects_empty_sobject_type(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_create_record("", {"Name": "Acme"}))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_update_record_rejects_empty_sobject_type(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(
        salesforce.salesforce_update_record("", "001xx", {"Industry": "Finance"})
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_update_record_rejects_empty_record_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(
        salesforce.salesforce_update_record("Account", "", {"Industry": "Finance"})
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_request_raises_with_joined_array_error_messages(monkeypatch):
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data=[
                    {
                        "message": "Required fields are missing: [Name]",
                        "errorCode": "REQUIRED_FIELD_MISSING",
                    },
                    {"message": "Session expired", "errorCode": "INVALID_SESSION_ID"},
                ],
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        salesforce._request("GET", "/services/data/v59.0/sobjects/Account/1")

    assert "Required fields are missing" in str(excinfo.value)
    assert "Session expired" in str(excinfo.value)


def test_request_falls_back_to_error_code_for_empty_message(monkeypatch):
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data=[{"message": "", "errorCode": "MALFORMED_ID"}],
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        salesforce._request("GET", "/services/data/v59.0/sobjects/Account/1")

    assert "MALFORMED_ID" in str(excinfo.value)
    assert "{'message'" not in str(excinfo.value)


def test_request_truncates_structured_error_body(monkeypatch):
    long_message = "x" * 5000
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=400,
                json_data=[{"message": long_message, "errorCode": "FIELD_TOO_LONG"}],
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        salesforce._request("GET", "/services/data/v59.0/sobjects/Account/1")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_message)


def test_request_truncates_unstructured_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        salesforce._request("GET", "/services/data/v59.0/sobjects/Account/1")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_get_current_user_returns_profile(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "user_id": "005xx",
                "organization_id": "00Dxx",
                "name": "Ada Lovelace",
                "email": "ada@example.com",
                "preferred_username": "ada@example.com",
            }
        )
    )
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["email"] == "ada@example.com"
    # The OIDC userinfo endpoint is documented as this fixed
    # login.salesforce.com host, not the per-org instance_url every other
    # tool here uses -- Salesforce routes it internally based on the token.
    assert mock_request.call_args.kwargs["url"] == (
        "https://login.salesforce.com/services/oauth2/userinfo"
    )


def test_get_current_user_does_not_require_instance_url(monkeypatch):
    """The userinfo call uses the fixed login host, not instance_url, so it
    must succeed even when SALESFORCE_INSTANCE_URL isn't set."""
    monkeypatch.delenv("SALESFORCE_INSTANCE_URL")
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"user_id": "005xx"})),
    )

    result = json.loads(salesforce.salesforce_get_current_user())

    assert result["status"] == "success"


def test_get_current_user_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data=[
                    {
                        "message": "Session expired or invalid",
                        "errorCode": "INVALID_SESSION_ID",
                    }
                ],
            )
        ),
    )

    result = json.loads(salesforce.salesforce_get_current_user())

    assert result["status"] == "error"
    assert "Session expired" in result["message"]


def test_get_current_user_falls_back_to_raw_text_on_dict_shaped_error(monkeypatch):
    # The OIDC userinfo endpoint (unlike every /services/data/* endpoint
    # this connector otherwise calls) returns dict-shaped errors like
    # {"error": ..., "error_description": ...}, not the top-level array
    # _extract_error_detail otherwise expects -- it should return None for
    # this shape and let the raw response text through instead of crashing.
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={
                    "error": "invalid_grant",
                    "error_description": "expired access/refresh token",
                },
            )
        ),
    )

    result = json.loads(salesforce.salesforce_get_current_user())

    assert result["status"] == "error"
    assert "expired access/refresh token" in result["message"]


def test_query_sends_soql_as_query_param(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "totalSize": 1,
                "done": True,
                "records": [{"Id": "001xx", "Name": "Acme"}],
            }
        )
    )
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_query("SELECT Id, Name FROM Account"))

    assert result["status"] == "success"
    assert result["records"] == [{"Id": "001xx", "Name": "Acme"}]
    assert result["total_size"] == 1
    assert result["truncated"] is False
    assert mock_request.call_args.kwargs["params"] == {
        "q": "SELECT Id, Name FROM Account"
    }
    assert mock_request.call_args.kwargs["url"].endswith(
        f"/services/data/{salesforce.API_VERSION}/query"
    )


def test_query_reports_truncated_when_not_done(monkeypatch):
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"totalSize": 5000, "done": False, "records": []}
            )
        ),
    )

    result = json.loads(salesforce.salesforce_query("SELECT Id FROM Contact"))

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_query_caps_output_size(monkeypatch):
    big_records = [{"Id": str(i), "Name": "x" * 1000} for i in range(50)]
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"totalSize": 50, "done": True, "records": big_records}
            )
        ),
    )
    monkeypatch.setattr(salesforce, "get_tool_max_output_length", lambda: 2000)

    # The raw returned string, not a reparsed-and-reserialized approximation
    # of it (json.loads then json.dumps can differ in length from the
    # original due to formatting/escaping) -- this is the exact string the
    # production output filter would apply its own hard-truncation to.
    raw = salesforce.salesforce_query("SELECT Id, Name FROM Account")
    result = json.loads(raw)

    assert len(raw) <= 2000
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["records"]) < len(big_records)
    assert "cannot be recovered" in result["message"]


def test_capped_list_drops_message_when_it_alone_exceeds_the_limit(monkeypatch):
    monkeypatch.setattr(salesforce, "get_tool_max_output_length", lambda: 40)

    raw = salesforce._success_with_capped_list(
        "records", [{"Id": str(i)} for i in range(50)]
    )
    result = json.loads(raw)

    assert result["records"] == []
    assert result["truncated"] is True
    assert "message" not in result


def test_search_sends_sosl_as_query_param(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"searchRecords": [{"Id": "001xx", "Name": "Acme"}]}
        )
    )
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(
        salesforce.salesforce_search("FIND {Acme} IN ALL FIELDS RETURNING Account(Id)")
    )

    assert result["status"] == "success"
    assert result["results"] == [{"Id": "001xx", "Name": "Acme"}]
    assert mock_request.call_args.kwargs["params"] == {
        "q": "FIND {Acme} IN ALL FIELDS RETURNING Account(Id)"
    }


def test_search_caps_output_size(monkeypatch):
    big_results = [{"Id": str(i), "Name": "x" * 1000} for i in range(50)]
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"searchRecords": big_results})),
    )
    monkeypatch.setattr(salesforce, "get_tool_max_output_length", lambda: 2000)

    raw = salesforce.salesforce_search(
        "FIND {Acme} IN ALL FIELDS RETURNING Account(Id)"
    )
    result = json.loads(raw)

    assert len(raw) <= 2000
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["results"]) < len(big_results)
    assert "cannot be recovered" in result["message"]


def test_list_sobjects_returns_summaries(monkeypatch):
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "sobjects": [
                        {
                            "name": "Account",
                            "label": "Account",
                            "queryable": True,
                            "createable": True,
                            "updateable": True,
                            "deletable": True,
                            "custom": False,
                        }
                    ]
                }
            )
        ),
    )

    result = json.loads(salesforce.salesforce_list_sobjects())

    assert result["status"] == "success"
    assert result["sobjects"][0]["name"] == "Account"


def test_list_sobjects_caps_output_size(monkeypatch):
    big_sobjects = [
        {"name": f"Object{i}__c", "label": "x" * 1000, "queryable": True}
        for i in range(50)
    ]
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"sobjects": big_sobjects})),
    )
    monkeypatch.setattr(salesforce, "get_tool_max_output_length", lambda: 2000)

    raw = salesforce.salesforce_list_sobjects()
    result = json.loads(raw)

    assert len(raw) <= 2000
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["sobjects"]) < len(big_sobjects)
    assert "cannot be recovered" in result["message"]


def test_describe_sobject_extracts_picklist_values(monkeypatch):
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "name": "Account",
                    "label": "Account",
                    "fields": [
                        {
                            "name": "Industry",
                            "label": "Industry",
                            "type": "picklist",
                            "nillable": True,
                            "createable": True,
                            "updateable": True,
                            "picklistValues": [
                                {"value": "Technology", "active": True},
                                {"value": "Retired", "active": False},
                            ],
                        },
                        {
                            "name": "Name",
                            "label": "Account Name",
                            "type": "string",
                            "nillable": False,
                            "createable": True,
                            "updateable": True,
                        },
                    ],
                }
            )
        ),
    )

    result = json.loads(salesforce.salesforce_describe_sobject("Account"))

    assert result["status"] == "success"
    industry_field = next(f for f in result["fields"] if f["name"] == "Industry")
    assert industry_field["picklist_values"] == ["Technology"]
    name_field = next(f for f in result["fields"] if f["name"] == "Name")
    assert name_field["picklist_values"] is None


def test_describe_sobject_caps_output_size(monkeypatch):
    big_fields = [
        {"name": f"Field{i}__c", "label": "x" * 1000, "type": "string"}
        for i in range(50)
    ]
    monkeypatch.setattr(
        salesforce.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"name": "Account", "label": "Account", "fields": big_fields}
            )
        ),
    )
    monkeypatch.setattr(salesforce, "get_tool_max_output_length", lambda: 2000)

    raw = salesforce.salesforce_describe_sobject("Account")
    result = json.loads(raw)

    assert len(raw) <= 2000
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["fields"]) < len(big_fields)
    assert "cannot be recovered" in result["message"]


def test_get_record_sends_fields_param_when_provided(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"Id": "001xx", "Name": "Acme", "Industry": "Technology"}
        )
    )
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(
        salesforce.salesforce_get_record("Account", "001xx", fields="Name,Industry")
    )

    assert result["status"] == "success"
    assert result["record"]["Name"] == "Acme"
    assert mock_request.call_args.kwargs["params"] == {"fields": "Name,Industry"}
    assert mock_request.call_args.kwargs["url"].endswith(
        f"/services/data/{salesforce.API_VERSION}/sobjects/Account/001xx"
    )


def test_get_record_omits_fields_param_when_not_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"Id": "001xx"}))
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    salesforce.salesforce_get_record("Account", "001xx")

    assert mock_request.call_args.kwargs["params"] == {}


def test_create_record_sends_fields_as_json_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"id": "001xx", "success": True})
    )
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(
        salesforce.salesforce_create_record(
            "Account", {"Name": "Acme Corp", "Industry": "Technology"}
        )
    )

    assert result["status"] == "success"
    assert result["id"] == "001xx"
    assert mock_request.call_args.kwargs["json"] == {
        "Name": "Acme Corp",
        "Industry": "Technology",
    }
    assert mock_request.call_args.kwargs["url"].endswith(
        f"/services/data/{salesforce.API_VERSION}/sobjects/Account"
    )
    assert mock_request.call_args.kwargs["method"] == "POST"


def test_create_record_omits_errors_key_on_plain_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"id": "001xx", "success": True})
    )
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(
        salesforce.salesforce_create_record("Account", {"Name": "Acme Corp"})
    )

    assert "errors" not in result


def test_create_record_propagates_errors_on_failure(monkeypatch):
    # Real Salesforce create failures surface via a non-2xx status (already
    # exercised by test_request_raises_with_joined_array_error_messages, via
    # the exception path in _request_absolute), not a 200 carrying
    # success=false. This defends the {success:false, errors:[...]} shape
    # anyway since Salesforce's own docs describe it as a possible response
    # for this endpoint -- if that shape is ever hit, errors must not be
    # silently dropped.
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "id": None,
                "success": False,
                "errors": [{"statusCode": "REQUIRED_FIELD_MISSING"}],
            }
        )
    )
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(
        salesforce.salesforce_create_record("Account", {"Name": "Acme Corp"})
    )

    assert result["errors"] == [{"statusCode": "REQUIRED_FIELD_MISSING"}]


def test_create_record_reports_error_on_realistic_non_2xx_failure(monkeypatch):
    """The documented, actually-reachable create failure contract: Salesforce
    returns a non-2xx status with a top-level JSON array body, not a 200
    carrying success=false (that shape is defended separately above, but
    isn't what production traffic hits)."""
    mock_request = Mock(
        return_value=MockResponse(
            status_code=400,
            json_data=[
                {
                    "message": "Required fields are missing: [Name]",
                    "errorCode": "REQUIRED_FIELD_MISSING",
                }
            ],
        )
    )
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(
        salesforce.salesforce_create_record("Account", {"Industry": "Technology"})
    )

    assert result["status"] == "error"
    assert "Required fields are missing" in result["message"]


def test_update_record_sends_fields_as_json_body(monkeypatch):
    mock_request = Mock(return_value=MockResponse(status_code=204))
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(
        salesforce.salesforce_update_record("Account", "001xx", {"Industry": "Finance"})
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"] == {"Industry": "Finance"}
    assert mock_request.call_args.kwargs["method"] == "PATCH"
    assert mock_request.call_args.kwargs["url"].endswith(
        f"/services/data/{salesforce.API_VERSION}/sobjects/Account/001xx"
    )


def test_update_record_requires_at_least_one_field(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_update_record("Account", "001xx", {}))

    assert result["status"] == "error"
    assert "No fields" in result["message"]
    mock_request.assert_not_called()


def test_salesforce_app_registry_requires_refresh_token_and_openid_scopes():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    salesforce_app = next(
        row
        for row in get_builtin_public_mcp_app_rows()
        if row["app_id"] == "salesforce"
    )
    assert "refresh_token" in salesforce_app["oauth_scopes"]
    assert "openid" in salesforce_app["oauth_scopes"]
    assert salesforce_app["launch_config"]["env_mapping"] == {
        "SALESFORCE_ACCESS_TOKEN": "access_token",
        "SALESFORCE_INSTANCE_URL": "instance_url",
    }
