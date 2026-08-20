import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import posthog


class MockResponse:
    def __init__(
        self, json_data=None, status_code: int = 200, text: str = "", url: str = ""
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = self.text.encode()
        self.url = url

    def json(self):
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

    with pytest.raises(ValueError, match="POSTHOG_HOST"):
        posthog._base_url()


def test_base_url_prepends_https_when_scheme_missing(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "us.posthog.com")

    assert posthog._base_url() == "https://us.posthog.com"


def test_base_url_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("POSTHOG_HOST", "  https://eu.posthog.com  ")

    assert posthog._base_url() == "https://eu.posthog.com"


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

    result = json.loads(posthog.posthog_create_annotation("Deployed v2"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"] == {
        "content": "Deployed v2",
        "scope": "project",
    }


def test_create_annotation_includes_date_marker_when_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": 1}))
    monkeypatch.setattr(posthog.requests, "request", mock_request)

    posthog.posthog_create_annotation("Deployed v2", date_marker="2026-08-18T00:00:00Z")

    assert (
        mock_request.call_args.kwargs["json"]["date_marker"] == "2026-08-18T00:00:00Z"
    )


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
