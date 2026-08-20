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


def test_request_uses_instance_url_and_headers(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = salesforce._request("GET", "/services/oauth2/userinfo")

    assert result == {"ok": True}
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.my.salesforce.com/services/oauth2/userinfo"
    )
    assert (
        mock_request.call_args.kwargs["headers"]["Authorization"]
        == "Bearer access-token"
    )


def test_require_clean_identifier_rejects_empty_and_whitespace():
    with pytest.raises(ValueError, match="record_id"):
        salesforce._require_clean_identifier("", "record_id")
    with pytest.raises(ValueError, match="record_id"):
        salesforce._require_clean_identifier(" 001xx ", "record_id")
    assert salesforce._require_clean_identifier("001xx", "record_id") == "001xx"


def test_url_path_id_percent_encodes_reserved_characters():
    # A literal ".." blocklist misses "/" and "?", which redirect the
    # request to a different endpoint or inject query params without ever
    # containing "..". Percent-encoding closes off all of them at once.
    assert salesforce._url_path_id("Account/001abc", "sobject_type") == (
        "Account%2F001abc"
    )
    assert salesforce._url_path_id("001x?fields=Id", "record_id") == (
        "001x%3Ffields%3DId"
    )
    with pytest.raises(ValueError):
        salesforce._url_path_id("", "record_id")


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


def test_delete_record_rejects_empty_record_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_delete_record("Account", ""))

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


def test_delete_record_sends_delete_method(monkeypatch):
    mock_request = Mock(return_value=MockResponse(status_code=204))
    monkeypatch.setattr(salesforce.requests, "request", mock_request)

    result = json.loads(salesforce.salesforce_delete_record("Account", "001xx"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["method"] == "DELETE"
    assert mock_request.call_args.kwargs["url"].endswith(
        f"/services/data/{salesforce.API_VERSION}/sobjects/Account/001xx"
    )


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
