import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import mixpanel


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        text: str = "",
        url: str = "",
        json_raises: bool = False,
        headers: dict | None = None,
        content: bytes | None = None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self._json_raises = json_raises
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = content if content is not None else self.text.encode()
        self.url = url
        self.headers = headers or {}
        self.closed = False

    def json(self):
        if self._json_raises:
            raise json.JSONDecodeError("Expecting value", self.text, 0)
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error for url: {self.url}", response=self
            )

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def iter_lines(self):
        for line in self.text.splitlines():
            yield line.encode("utf-8")


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv(
        "MIXPANEL_SERVICE_ACCOUNT_USERNAME", "svc-account.mp-service-account"
    )
    monkeypatch.setenv("MIXPANEL_SERVICE_ACCOUNT_SECRET", "test-secret")
    monkeypatch.setenv("MIXPANEL_PROJECT_ID", "12345")
    monkeypatch.setenv("MIXPANEL_REGION", "us")


def test_auth_requires_username(monkeypatch):
    monkeypatch.delenv("MIXPANEL_SERVICE_ACCOUNT_USERNAME")

    with pytest.raises(ValueError, match="MIXPANEL_SERVICE_ACCOUNT_USERNAME"):
        mixpanel._auth()


def test_auth_requires_secret(monkeypatch):
    monkeypatch.delenv("MIXPANEL_SERVICE_ACCOUNT_SECRET")

    with pytest.raises(ValueError, match="MIXPANEL_SERVICE_ACCOUNT_SECRET"):
        mixpanel._auth()


def test_auth_returns_username_secret_tuple():
    assert mixpanel._auth() == ("svc-account.mp-service-account", "test-secret")


def test_auth_strips_whitespace(monkeypatch):
    monkeypatch.setenv(
        "MIXPANEL_SERVICE_ACCOUNT_USERNAME", "  svc-account.mp-service-account\n"
    )
    monkeypatch.setenv("MIXPANEL_SERVICE_ACCOUNT_SECRET", " test-secret ")

    assert mixpanel._auth() == ("svc-account.mp-service-account", "test-secret")


def test_auth_treats_whitespace_only_values_as_missing(monkeypatch):
    monkeypatch.setenv("MIXPANEL_SERVICE_ACCOUNT_USERNAME", "   ")

    with pytest.raises(ValueError, match="MIXPANEL_SERVICE_ACCOUNT_USERNAME"):
        mixpanel._auth()


def test_project_id_requires_env(monkeypatch):
    monkeypatch.delenv("MIXPANEL_PROJECT_ID")

    with pytest.raises(ValueError, match="MIXPANEL_PROJECT_ID"):
        mixpanel._project_id()


def test_project_id_strips_whitespace(monkeypatch):
    monkeypatch.setenv("MIXPANEL_PROJECT_ID", " 12345\n")

    assert mixpanel._project_id() == "12345"


def test_project_id_treats_whitespace_only_value_as_missing(monkeypatch):
    monkeypatch.setenv("MIXPANEL_PROJECT_ID", "   ")

    with pytest.raises(ValueError, match="MIXPANEL_PROJECT_ID"):
        mixpanel._project_id()


@pytest.mark.parametrize(
    "region, query_host, export_host",
    [
        ("us", "mixpanel.com", "data.mixpanel.com"),
        ("eu", "eu.mixpanel.com", "data-eu.mixpanel.com"),
        ("in", "in.mixpanel.com", "data-in.mixpanel.com"),
        ("EU", "eu.mixpanel.com", "data-eu.mixpanel.com"),
    ],
)
def test_region_hosts_resolves_known_regions(
    monkeypatch, region, query_host, export_host
):
    monkeypatch.setenv("MIXPANEL_REGION", region)

    hosts = mixpanel._region_hosts()

    assert hosts == {"query": query_host, "export": export_host}


def test_region_hosts_defaults_to_us_when_unset(monkeypatch):
    monkeypatch.delenv("MIXPANEL_REGION")

    assert mixpanel._region_hosts()["query"] == "mixpanel.com"


def test_region_hosts_rejects_unknown_region(monkeypatch):
    monkeypatch.setenv("MIXPANEL_REGION", "apac")

    with pytest.raises(ValueError, match="MIXPANEL_REGION"):
        mixpanel._region_hosts()


@pytest.mark.parametrize(
    "limit, expected",
    [
        (0, 1),
        (-5, 1),
        (1, 1),
        (mixpanel.MAX_LIMIT, mixpanel.MAX_LIMIT),
        (mixpanel.MAX_LIMIT + 1, mixpanel.MAX_LIMIT),
        (10_000, mixpanel.MAX_LIMIT),
    ],
)
def test_clamp_limit_boundaries(limit, expected):
    assert mixpanel._clamp_limit(limit) == expected


def test_validate_date_accepts_iso_date():
    assert mixpanel._validate_date("2026-01-15", "from_date") == "2026-01-15"


@pytest.mark.parametrize(
    "value",
    [
        "2026/01/15",
        "01-15-2026",
        "2026-1-15",
        "not-a-date",
        "",
        None,
        # `$` (unlike `\Z`) matches just before a trailing newline as well
        # as at the true end of the string -- this must still be rejected.
        "2026-01-15\n",
    ],
)
def test_validate_date_rejects_malformed_value(value):
    with pytest.raises(ValueError, match="from_date"):
        mixpanel._validate_date(value, "from_date")


@pytest.mark.parametrize("value", ["2026-02-30", "2026-13-01", "2026-00-10"])
def test_validate_date_rejects_calendar_invalid_date(value):
    # Shape-correct (matches the regex) but not a real calendar date --
    # date.fromisoformat() is what actually catches this.
    with pytest.raises(ValueError, match="valid calendar date"):
        mixpanel._validate_date(value, "from_date")


def test_extract_error_detail_returns_error_string():
    response = MockResponse(json_data={"error": "invalid credentials"})

    assert mixpanel._extract_error_detail(response) == "invalid credentials"


def test_extract_error_detail_returns_none_for_non_json_body():
    response = MockResponse(status_code=500, text="not json", json_raises=True)

    assert mixpanel._extract_error_detail(response) is None


def test_extract_error_detail_returns_none_when_error_field_missing():
    response = MockResponse(json_data={"detail": "no error key here"})

    assert mixpanel._extract_error_detail(response) is None


def test_extract_error_detail_returns_none_for_non_dict_json_body():
    response = MockResponse(json_data=["not", "a", "dict"])

    assert mixpanel._extract_error_detail(response) is None


def test_extract_error_detail_returns_none_for_empty_string_error():
    response = MockResponse(json_data={"error": ""})

    assert mixpanel._extract_error_detail(response) is None


def test_extract_error_detail_returns_none_for_non_string_error():
    response = MockResponse(json_data={"error": {"nested": "object"}})

    assert mixpanel._extract_error_detail(response) is None


def test_request_uses_configured_host_project_id_and_auth(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert result == {"ok": True}
    assert (
        mock_request.call_args.kwargs["url"]
        == "https://mixpanel.com/api/query/events/names"
    )
    assert mock_request.call_args.kwargs["auth"] == (
        "svc-account.mp-service-account",
        "test-secret",
    )
    assert mock_request.call_args.kwargs["params"]["project_id"] == "12345"
    assert mock_request.call_args.kwargs["allow_redirects"] is False


def test_request_omits_project_id_when_include_project_id_false(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel._request(
        "GET",
        "mixpanel.com",
        "/api/app/projects/12345/annotations",
        include_project_id=False,
    )

    assert "project_id" not in mock_request.call_args.kwargs["params"]


def test_request_sends_json_body(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel._request(
        "POST",
        "mixpanel.com",
        "/api/app/projects/12345/annotations",
        json_data={"date": "2026-01-15 00:00:00", "description": "x"},
        include_project_id=False,
    )

    assert mock_request.call_args.kwargs["json"] == {
        "date": "2026-01-15 00:00:00",
        "description": "x",
    }


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_request_rejects_redirect_response(monkeypatch, status_code):
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=status_code, url="https://mixpanel.com/x"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="redirect"):
        mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")


def test_request_closes_streamed_response_on_redirect(monkeypatch):
    # A stream=True response left open on this path would leak its
    # connection, since it's discarded here instead of ever being iterated.
    response = MockResponse(status_code=302, url="https://data.mixpanel.com/x")
    monkeypatch.setattr(mixpanel.requests, "request", Mock(return_value=response))

    with pytest.raises(RuntimeError, match="redirect"):
        mixpanel._request("GET", "data.mixpanel.com", "/api/2.0/export", stream=True)

    assert response.closed is True


def test_request_passes_configured_timeout(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert mock_request.call_args.kwargs["timeout"] == mixpanel.DEFAULT_TIMEOUT_SECONDS


def test_request_retries_once_on_429_with_retry_after(monkeypatch):
    responses = [
        MockResponse(status_code=429, url="x", headers={"Retry-After": "1"}),
        MockResponse(json_data={"ok": True}),
    ]
    mock_request = Mock(side_effect=responses)
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)
    monkeypatch.setattr(mixpanel.time, "sleep", Mock())

    result = mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert result == {"ok": True}
    assert mock_request.call_count == 2
    mixpanel.time.sleep.assert_called_once_with(1)


def test_request_closes_discarded_streamed_response_before_429_retry(monkeypatch):
    # The first attempt's connection is never read when stream=True -- it
    # must be closed explicitly before being discarded for the retry, or
    # it leaks for the duration of the sleep.
    first = MockResponse(status_code=429, url="x", headers={"Retry-After": "1"})
    second = MockResponse(status_code=200, text="")
    mock_request = Mock(side_effect=[first, second])
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)
    monkeypatch.setattr(mixpanel.time, "sleep", Mock())

    result = mixpanel._request(
        "GET", "data.mixpanel.com", "/api/2.0/export", stream=True
    )

    assert result is second
    assert first.closed is True
    assert second.closed is False


def test_request_does_not_retry_a_second_429(monkeypatch):
    response = MockResponse(status_code=429, url="x", headers={"Retry-After": "1"})
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)
    monkeypatch.setattr(mixpanel.time, "sleep", Mock())

    with pytest.raises(RuntimeError):
        mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert mock_request.call_count == 2


def test_request_does_not_retry_429_without_retry_after_header(monkeypatch):
    mock_request = Mock(return_value=MockResponse(status_code=429, url="x"))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)
    monkeypatch.setattr(mixpanel.time, "sleep", Mock())

    with pytest.raises(RuntimeError):
        mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert mock_request.call_count == 1
    mixpanel.time.sleep.assert_not_called()


def test_request_does_not_retry_429_with_non_integer_retry_after(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            status_code=429, url="x", headers={"Retry-After": "not-a-number"}
        )
    )
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)
    monkeypatch.setattr(mixpanel.time, "sleep", Mock())

    with pytest.raises(RuntimeError):
        mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert mock_request.call_count == 1
    mixpanel.time.sleep.assert_not_called()


def test_request_does_not_retry_429_with_retry_after_exceeding_max(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            status_code=429,
            url="x",
            headers={"Retry-After": str(mixpanel.MAX_RETRY_AFTER_SECONDS + 1)},
        )
    )
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)
    monkeypatch.setattr(mixpanel.time, "sleep", Mock())

    with pytest.raises(RuntimeError):
        mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert mock_request.call_count == 1
    mixpanel.time.sleep.assert_not_called()


def test_request_returns_empty_dict_for_non_streamed_empty_body(monkeypatch):
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(return_value=MockResponse(status_code=200, text="")),
    )

    result = mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert result == {}


def test_request_redacts_connection_error_message(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.exceptions.ProxyError(
            "Unable to connect to proxy: "
            "https://user:sp-secret-proxy-pass@proxy.internal:8080/"
        )

    monkeypatch.setattr(mixpanel.requests, "request", _raise)

    with pytest.raises(RuntimeError) as excinfo:
        mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert "sp-secret-proxy-pass" not in str(excinfo.value)


def test_request_raises_with_structured_error_detail(monkeypatch):
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=401, json_data={"error": "invalid API key"}
            )
        ),
    )

    with pytest.raises(RuntimeError, match="invalid API key"):
        mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")


def test_request_raises_clear_error_for_non_json_2xx_body(monkeypatch):
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=200, text="<html>not json</html>", json_raises=True
            )
        ),
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")


def test_request_truncates_unstructured_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_request_with_stream_true_returns_raw_response(monkeypatch):
    mock_response = MockResponse(status_code=200, text='{"a": 1}\n{"b": 2}\n')
    monkeypatch.setattr(mixpanel.requests, "request", Mock(return_value=mock_response))

    result = mixpanel._request(
        "GET", "data.mixpanel.com", "/api/2.0/export", stream=True
    )

    assert result is mock_response


def test_request_passes_stream_flag_to_requests_request(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel._request("GET", "mixpanel.com", "/api/query/events/names", stream=True)

    assert mock_request.call_args.kwargs["stream"] is True


def test_request_defaults_stream_to_false(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel._request("GET", "mixpanel.com", "/api/query/events/names")

    assert mock_request.call_args.kwargs["stream"] is False


def test_list_event_names_returns_names(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=["Signup", "Purchase"]))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_list_event_names())

    assert result["status"] == "success"
    assert result["event_names"]["event_names"] == ["Signup", "Purchase"]
    assert mock_request.call_args.kwargs["url"].endswith("/api/query/events/names")
    assert mock_request.call_args.kwargs["params"]["type"] == "general"


def test_list_event_names_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(
            return_value=MockResponse(status_code=401, json_data={"error": "bad auth"})
        ),
    )

    result = json.loads(mixpanel.mixpanel_list_event_names())

    assert result["status"] == "error"
    assert "bad auth" in result["message"]


def test_get_top_events_returns_capped_dict(monkeypatch):
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"Signup": {"amount": 42}})),
    )

    result = json.loads(mixpanel.mixpanel_get_top_events())

    assert result["status"] == "success"
    assert result["top_events"] == {"Signup": {"amount": 42}}


def test_get_event_properties_hits_top_endpoint_without_type_param(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"$browser": {"count": 10}})
    )
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_get_event_properties("Signup"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/query/events/properties/top"
    )
    assert mock_request.call_args.kwargs["params"]["event"] == "Signup"
    assert "type" not in mock_request.call_args.kwargs["params"]


def test_query_segmentation_requires_valid_dates():
    result = json.loads(
        mixpanel.mixpanel_query_segmentation("Signup", "not-a-date", "2026-01-31")
    )

    assert result["status"] == "error"
    assert "from_date" in result["message"]


def test_query_segmentation_includes_optional_params(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {}}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(
        mixpanel.mixpanel_query_segmentation(
            "Signup",
            "2026-01-01",
            "2026-01-31",
            on='properties["$browser"]',
            where='properties["plan"] == "pro"',
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/api/query/segmentation")
    params = mock_request.call_args.kwargs["params"]
    assert params["on"] == 'properties["$browser"]'
    assert params["where"] == 'properties["plan"] == "pro"'


def test_query_segmentation_omits_optional_params_when_not_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {}}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel.mixpanel_query_segmentation("Signup", "2026-01-01", "2026-01-31")

    params = mock_request.call_args.kwargs["params"]
    assert "on" not in params
    assert "where" not in params


def test_query_retention_defaults_and_optional_params(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {}}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(
        mixpanel.mixpanel_query_retention(
            "2026-01-01", "2026-01-31", born_event="Signup"
        )
    )

    assert result["status"] == "success"
    params = mock_request.call_args.kwargs["params"]
    assert params["retention_type"] == "birth"
    assert params["born_event"] == "Signup"
    assert "event" not in params


def test_query_retention_requires_born_event_for_birth_type(monkeypatch):
    # Mixpanel has no "any event" fallback for retention_type="birth"
    # (unlike "compounded") -- born_event is required in that case.
    mock_request = Mock()
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_query_retention("2026-01-01", "2026-01-31"))

    assert result["status"] == "error"
    assert "born_event" in result["message"]
    mock_request.assert_not_called()


def test_query_retention_allows_missing_born_event_for_compounded_type(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {}}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(
        mixpanel.mixpanel_query_retention(
            "2026-01-01", "2026-01-31", retention_type="compounded"
        )
    )

    assert result["status"] == "success"
    assert "born_event" not in mock_request.call_args.kwargs["params"]


def test_list_funnels_returns_list(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data=[{"funnel_id": 1, "name": "Onboarding"}])
    )
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_list_funnels())

    assert result["status"] == "success"
    assert result["funnels"]["funnels"] == [{"funnel_id": 1, "name": "Onboarding"}]
    assert mock_request.call_args.kwargs["url"].endswith("/api/query/funnels/list")


def test_query_funnel_sends_funnel_id_and_dates(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"data": {}}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_query_funnel(1, "2026-01-01", "2026-01-31"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/api/query/funnels")
    params = mock_request.call_args.kwargs["params"]
    assert params["funnel_id"] == 1
    assert params["from_date"] == "2026-01-01"


def test_query_engage_posts_form_body_without_limit(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"results": [], "session_id": "s1", "page": 0}
        )
    )
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel.mixpanel_query_engage()

    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["url"].endswith("/api/query/engage")
    assert mock_request.call_args.kwargs["data"] == {}
    assert "limit" not in (mock_request.call_args.kwargs["params"] or {})


def test_query_engage_sends_where_and_output_properties_in_form_body(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel.mixpanel_query_engage(
        where='properties["$email"] is set', output_properties="$email, $last_name"
    )

    form = mock_request.call_args.kwargs["data"]
    assert form["where"] == 'properties["$email"] is set'
    assert form["output_properties"] == json.dumps(["$email", "$last_name"])


def test_query_engage_includes_pagination_params_when_continuing(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel.mixpanel_query_engage(session_id="s1", page=2)

    form = mock_request.call_args.kwargs["data"]
    assert form["session_id"] == "s1"
    assert form["page"] == 2


def test_query_engage_rejects_page_without_session_id(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_query_engage(page=2))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_list_annotations_requires_valid_dates(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_list_annotations("bad", "2026-01-31"))

    assert result["status"] == "error"
    assert "from_date" in result["message"]
    mock_request.assert_not_called()


def test_list_annotations_returns_empty_list_when_results_key_missing(monkeypatch):
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(return_value=MockResponse(json_data={"status": "ok"})),
    )

    result = json.loads(mixpanel.mixpanel_list_annotations("2026-01-01", "2026-01-31"))

    assert result["status"] == "success"
    assert result["annotations"]["annotations"] == []


def test_list_annotations_uses_app_api_path_and_camelcase_dates(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "status": "ok",
                "results": [{"id": 1, "description": "Deployed"}],
            }
        )
    )
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_list_annotations("2026-01-01", "2026-01-31"))

    assert result["status"] == "success"
    assert result["annotations"]["annotations"] == [
        {"id": 1, "description": "Deployed"}
    ]
    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/app/projects/12345/annotations"
    )
    params = mock_request.call_args.kwargs["params"]
    assert params["fromDate"] == "2026-01-01"
    assert params["toDate"] == "2026-01-31"
    assert "project_id" not in params


def test_create_annotation_sends_json_body_to_app_api(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"id": 1, "description": "Deployed v2"})
    )
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(
        mixpanel.mixpanel_create_annotation("2026-01-15 00:00:00", "Deployed v2")
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"] == {
        "date": "2026-01-15 00:00:00",
        "description": "Deployed v2",
    }
    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/api/app/projects/12345/annotations"
    )
    assert "project_id" not in mock_request.call_args.kwargs["params"]


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-15",
        "2026-01-15T00:00:00",
        "01-15-2026 00:00:00",
        "2026-02-30 00:00:00",
        "not-a-datetime",
        "",
    ],
)
def test_create_annotation_rejects_malformed_date(value, monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_create_annotation(value, "Deployed v2"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_export_events_parses_ndjson_and_caps_result(monkeypatch):
    lines = "\n".join(json.dumps({"event": "Signup", "n": i}) for i in range(3))
    mock_request = Mock(return_value=MockResponse(status_code=200, text=lines))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_export_events("2026-01-01", "2026-01-31"))

    assert result["status"] == "success"
    assert len(result["events"]["events"]) == 3
    assert result["events"]["row_limit_reached"] is False
    assert result["events"]["stream_error"] is False
    assert result["events"]["events"][0] == {"event": "Signup", "n": 0}
    # No "count" field: it would go stale if success_with_capped_dict's own
    # size-based capping halves the list further after this point.
    assert "count" not in result["events"]
    # Streamed, not buffered whole -- a wide date range must not pull the
    # entire NDJSON body into memory before the row cap applies.
    assert mock_request.call_args.kwargs["stream"] is True


def test_export_events_truncates_at_max_events(monkeypatch):
    lines = "\n".join(
        json.dumps({"event": "Signup", "n": i})
        for i in range(mixpanel.MAX_EXPORT_EVENTS + 5)
    )
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(return_value=MockResponse(status_code=200, text=lines)),
    )

    result = json.loads(mixpanel.mixpanel_export_events("2026-01-01", "2026-01-31"))

    assert len(result["events"]["events"]) == mixpanel.MAX_EXPORT_EVENTS
    assert result["events"]["row_limit_reached"] is True
    assert result["events"]["stream_error"] is False


def test_export_events_treats_malformed_trailing_line_as_end_of_stream(monkeypatch):
    # A truncated final line (a realistic mid-stream connection reset) must
    # not discard the events already parsed before it.
    lines = (
        "\n".join(json.dumps({"event": "Signup", "n": i}) for i in range(3))
        + '\n{"event": "Signup", "n": 3, truncated'
    )
    monkeypatch.setattr(
        mixpanel.requests,
        "request",
        Mock(return_value=MockResponse(status_code=200, text=lines)),
    )

    result = json.loads(mixpanel.mixpanel_export_events("2026-01-01", "2026-01-31"))

    assert result["status"] == "success"
    assert len(result["events"]["events"]) == 3
    # Distinct from row_limit_reached: this stopped because the stream
    # broke, not because MAX_EXPORT_EVENTS was hit -- a caller must not
    # read this partial result as "the complete answer for the range."
    assert result["events"]["row_limit_reached"] is False
    assert result["events"]["stream_error"] is True


def test_export_events_uses_export_host_and_encodes_event_filter(monkeypatch):
    mock_request = Mock(return_value=MockResponse(status_code=200, text=""))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel.mixpanel_export_events("2026-01-01", "2026-01-31", event="Signup")

    assert (
        mock_request.call_args.kwargs["url"]
        == "https://data.mixpanel.com/api/2.0/export"
    )
    assert mock_request.call_args.kwargs["params"]["event"] == json.dumps(["Signup"])
    assert (
        mock_request.call_args.kwargs["params"]["limit"] == mixpanel.MAX_EXPORT_EVENTS
    )


def test_export_events_requires_valid_dates(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    result = json.loads(mixpanel.mixpanel_export_events("2026/01/01", "2026-01-31"))

    assert result["status"] == "error"
    assert "from_date" in result["message"]
    mock_request.assert_not_called()


def test_export_events_uses_eu_export_host_for_eu_region(monkeypatch):
    monkeypatch.setenv("MIXPANEL_REGION", "eu")
    mock_request = Mock(return_value=MockResponse(status_code=200, text=""))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel.mixpanel_export_events("2026-01-01", "2026-01-31")

    assert mock_request.call_args.kwargs["url"] == (
        "https://data-eu.mixpanel.com/api/2.0/export"
    )


def test_list_annotations_uses_eu_query_host_for_eu_region(monkeypatch):
    monkeypatch.setenv("MIXPANEL_REGION", "eu")
    mock_request = Mock(return_value=MockResponse(json_data={"results": []}))
    monkeypatch.setattr(mixpanel.requests, "request", mock_request)

    mixpanel.mixpanel_list_annotations("2026-01-01", "2026-01-31")

    assert mock_request.call_args.kwargs["url"] == (
        "https://eu.mixpanel.com/api/app/projects/12345/annotations"
    )


@pytest.mark.parametrize("value", [None, 123])
def test_validate_annotation_datetime_rejects_non_string(value):
    with pytest.raises(ValueError, match="date"):
        mixpanel._validate_annotation_datetime(value, "date")


def test_mixpanel_app_registry_requires_service_account_and_project():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    mixpanel_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "mixpanel"
    )
    assert mixpanel_app["provider_name"] is None
    assert mixpanel_app["category"] == "Analytics"
    assert mixpanel_app["transport"] == "stdio"
    assert mixpanel_app["launch_config"]["required_env"] == [
        "MIXPANEL_SERVICE_ACCOUNT_USERNAME",
        "MIXPANEL_SERVICE_ACCOUNT_SECRET",
        "MIXPANEL_PROJECT_ID",
        "MIXPANEL_REGION",
    ]
