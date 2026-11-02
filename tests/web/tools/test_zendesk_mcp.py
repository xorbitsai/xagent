import json
import re
import socket
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import zendesk


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
        content: bytes | None = None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self._json_raises = json_raises
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.content = content if content is not None else self.text.encode()
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
    monkeypatch.setenv("ZENDESK_SUBDOMAIN", "acme")
    monkeypatch.setenv("ZENDESK_ACCESS_TOKEN", "test-token")
    # Unconfigured by default, so the exact-header assertions below also
    # cover a deployment without the Marketplace settings.
    for name in (
        "ZENDESK_MARKETPLACE_NAME",
        "ZENDESK_MARKETPLACE_ORGANIZATION_ID",
        "ZENDESK_MARKETPLACE_APP_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    # _base_url() resolves DNS to catch a hostname that rebinds to a private
    # address; tests must not depend on real network/DNS, so every test gets
    # a fake resolver returning an unambiguously public IP by default.
    monkeypatch.setattr(zendesk.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1"))


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("ZENDESK_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="ZENDESK_ACCESS_TOKEN"):
        zendesk._headers()


def test_headers_treat_a_whitespace_only_access_token_as_missing(monkeypatch):
    monkeypatch.setenv("ZENDESK_ACCESS_TOKEN", " \n")

    with pytest.raises(ValueError, match="ZENDESK_ACCESS_TOKEN"):
        zendesk._headers()


def test_headers_send_the_access_token_as_a_bearer_credential():
    assert zendesk._headers() == {"Authorization": "Bearer test-token"}


def test_headers_carry_the_configured_marketplace_headers(monkeypatch):
    monkeypatch.setenv("ZENDESK_MARKETPLACE_NAME", "Xagent")
    monkeypatch.setenv("ZENDESK_MARKETPLACE_ORGANIZATION_ID", "12345")
    monkeypatch.setenv("ZENDESK_MARKETPLACE_APP_ID", "67890")

    assert zendesk._headers() == {
        "Authorization": "Bearer test-token",
        "X-Zendesk-Marketplace-Name": "Xagent",
        "X-Zendesk-Marketplace-Organization-Id": "12345",
        "X-Zendesk-Marketplace-App-Id": "67890",
    }


def test_headers_drop_an_invalid_marketplace_value(monkeypatch):
    monkeypatch.setenv("ZENDESK_MARKETPLACE_NAME", "Xagent\r\nX-Injected: 1")
    monkeypatch.setenv("ZENDESK_MARKETPLACE_ORGANIZATION_ID", "12345")

    assert zendesk._headers() == {
        "Authorization": "Bearer test-token",
        "X-Zendesk-Marketplace-Organization-Id": "12345",
    }


def test_legacy_api_token_credentials_are_ignored(monkeypatch):
    monkeypatch.delenv("ZENDESK_ACCESS_TOKEN")
    monkeypatch.setenv("ZENDESK_EMAIL", "agent@acme.com")
    monkeypatch.setenv("ZENDESK_API_TOKEN", "legacy-token")

    with pytest.raises(ValueError, match="ZENDESK_ACCESS_TOKEN"):
        zendesk._headers()


def test_base_url_requires_subdomain(monkeypatch):
    monkeypatch.delenv("ZENDESK_SUBDOMAIN")

    with pytest.raises(ValueError, match="ZENDESK_SUBDOMAIN"):
        zendesk._base_url()


def test_base_url_builds_from_subdomain():
    assert zendesk._base_url() == "https://acme.zendesk.com/api/v2"


def test_base_url_lowercases_subdomain(monkeypatch):
    monkeypatch.setenv("ZENDESK_SUBDOMAIN", "ACME")

    assert zendesk._base_url() == "https://acme.zendesk.com/api/v2"


@pytest.mark.parametrize(
    "subdomain",
    [
        "",
        "   ",
        "acme.evil.com",
        "acme/../evil",
        "https://acme",
        "acme:8080",
        "-acme",
        "acme-",
        "acme evil",
    ],
)
def test_base_url_rejects_invalid_subdomain(monkeypatch, subdomain):
    monkeypatch.setenv("ZENDESK_SUBDOMAIN", subdomain)

    with pytest.raises(ValueError, match="ZENDESK_SUBDOMAIN"):
        zendesk._base_url()


def test_base_url_rejects_subdomain_resolving_to_private_ip(monkeypatch):
    # A syntactically valid subdomain that only *resolves* to a private
    # address -- the DNS-rebinding case a literal-string check alone can't
    # catch.
    monkeypatch.setattr(zendesk.socket, "getaddrinfo", _fake_getaddrinfo("10.0.0.5"))

    with pytest.raises(ValueError, match="not allowed"):
        zendesk._base_url()


def test_base_url_rejects_when_any_resolved_address_is_private(monkeypatch):
    monkeypatch.setattr(
        zendesk.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1", "10.0.0.5")
    )

    with pytest.raises(ValueError, match="not allowed"):
        zendesk._base_url()


def test_base_url_raises_when_dns_resolution_fails(monkeypatch):
    def _raise(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(zendesk.socket, "getaddrinfo", _raise)

    with pytest.raises(ValueError, match="could not be resolved"):
        zendesk._base_url()


@pytest.mark.parametrize(
    "limit, expected",
    [
        (0, 1),
        (-5, 1),
        (1, 1),
        (zendesk.MAX_LIMIT, zendesk.MAX_LIMIT),
        (zendesk.MAX_LIMIT + 1, zendesk.MAX_LIMIT),
        (10_000, zendesk.MAX_LIMIT),
    ],
)
def test_clamp_limit_boundaries(limit, expected):
    assert zendesk._clamp_limit(limit) == expected


@pytest.mark.parametrize("value", ["", "   ", None])
def test_require_non_blank_rejects_empty_values(value):
    with pytest.raises(ValueError, match="field"):
        zendesk._require_non_blank(value, "field")


def test_require_non_blank_accepts_non_empty_value():
    assert zendesk._require_non_blank("hello", "field") == "hello"


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", None), ("   ", None), ("  hi  ", "hi"), ("hi", "hi")],
)
def test_blank_to_none(value, expected):
    assert zendesk._blank_to_none(value) == expected


@pytest.mark.parametrize(
    "tags, expected",
    [
        (["vip", "urgent"], ["vip", "urgent"]),
        (["vip ", "  urgent"], ["vip", "urgent"]),
        (["vip ", "", "  "], ["vip"]),
        ([], []),
    ],
)
def test_clean_tags(tags, expected):
    assert zendesk._clean_tags(tags) == expected


def test_resolve_tags_none_means_not_provided():
    assert zendesk._resolve_tags(None) is None


def test_resolve_tags_empty_list_means_explicit_clear():
    assert zendesk._resolve_tags([]) == []


def test_resolve_tags_cleans_a_non_empty_list():
    assert zendesk._resolve_tags(["vip ", "", "urgent"]) == ["vip", "urgent"]


def test_resolve_tags_rejects_a_non_empty_list_that_cleans_to_nothing():
    with pytest.raises(ValueError, match="no usable values"):
        zendesk._resolve_tags(["  ", ""])


def test_require_one_of_normalizes_case():
    allowed = frozenset({"solved", "open"})
    assert zendesk._require_one_of("Solved", allowed, "status") == "solved"


def test_require_one_of_rejects_unknown_value():
    with pytest.raises(ValueError, match="must be one of"):
        zendesk._require_one_of("archived", frozenset({"solved", "open"}), "status")


def test_search_sends_stripped_query(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [], "count": 0})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_search("  type:ticket  ")

    assert mock_request.call_args.kwargs["params"]["query"] == "type:ticket"


def test_create_ticket_sends_stripped_subject_and_comment(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "subject": "Help"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_create_ticket("  Help  ", "  Something's broken  ")

    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"]["subject"] == "Help"
    assert body["ticket"]["comment"] == {"body": "Something's broken"}


def test_reply_to_ticket_sends_stripped_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "status": "open"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_reply_to_ticket(1, "  Thanks for reaching out  ")

    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"]["comment"]["body"] == "Thanks for reaching out"


def test_unwrap_extracts_key_from_envelope():
    assert zendesk._unwrap({"ticket": {"id": 1}}, "ticket") == {"id": 1}


def test_unwrap_falls_back_to_raw_value_when_not_a_dict():
    assert zendesk._unwrap([1, 2], "ticket") == [1, 2]


def test_unwrap_falls_back_to_payload_when_key_missing():
    assert zendesk._unwrap({"other": 1}, "ticket") == {"other": 1}


def test_extract_error_detail_prefers_description():
    response = MockResponse(
        json_data={"error": "RecordNotFound", "description": "Not found"}
    )

    assert zendesk._extract_error_detail(response.text) == "Not found"


def test_extract_error_detail_falls_back_to_string_error():
    response = MockResponse(json_data={"error": "Couldn't authenticate you"})

    assert zendesk._extract_error_detail(response.text) == "Couldn't authenticate you"


def test_extract_error_detail_handles_nested_error_object():
    response = MockResponse(
        json_data={"error": {"title": "Unauthorized", "message": "Bad credentials"}}
    )

    assert zendesk._extract_error_detail(response.text) == "Bad credentials"


def test_extract_error_detail_falls_back_to_title_when_message_absent():
    response = MockResponse(json_data={"error": {"title": "Unauthorized"}})

    assert zendesk._extract_error_detail(response.text) == "Unauthorized"


def test_extract_error_detail_returns_none_for_non_json_body():
    response = MockResponse(status_code=500, text="not json", json_raises=True)

    assert zendesk._extract_error_detail(response.text) is None


def test_extract_error_detail_appends_record_invalid_field_reasons():
    # RecordInvalid's "description" is a generic "Record validation
    # errors" -- the actual reason lives in "details", keyed by field.
    response = MockResponse(
        json_data={
            "error": "RecordInvalid",
            "description": "Record validation errors",
            "details": {"base": [{"description": "Status: closed is not valid"}]},
        }
    )

    assert (
        zendesk._extract_error_detail(response.text)
        == "Record validation errors: Status: closed is not valid"
    )


def test_extract_error_detail_falls_back_to_field_reasons_without_description():
    response = MockResponse(
        json_data={
            "details": {
                "email": [
                    {
                        "description": "Email: a@example.com already in use",
                        "error": "DuplicateValue",
                    }
                ]
            }
        }
    )

    assert (
        zendesk._extract_error_detail(response.text)
        == "Email: a@example.com already in use"
    )


def test_finalize_capped_returns_response_unchanged_when_it_fits():
    assert zendesk._finalize_capped('{"status": "success"}', 2000) == (
        '{"status": "success"}'
    )


def test_finalize_capped_shrinks_its_own_fallback_to_fit_a_small_cap():
    # A reviewer-reported example: XAGENT_TOOL_MAX_OUTPUT_LENGTH=80 is well
    # under the ~155-char detailed fallback message this function used to
    # return unconditionally, which itself exceeded the cap it was meant to
    # respect. The shorter "output cap too small" tier must be used instead.
    raw = zendesk._finalize_capped("x" * 5000, 80)

    assert len(raw) <= 80
    assert json.loads(raw)["status"] == "error"
    assert json.loads(raw)["message"] == "output cap too small"


def test_finalize_capped_falls_back_to_minimal_envelope_below_any_message():
    # Below the size of even the shortest fixed error message this function
    # can construct, it must still return valid, in-cap JSON -- the
    # unsatisfiable case is a misconfigured cap, not a bug in the fallback.
    raw = zendesk._finalize_capped("x" * 5000, 20)

    assert len(raw) <= 20
    assert json.loads(raw) == {"status": "error"}


def test_scrub_query_strings_redacts_a_real_url_query():
    text = "Max retries exceeded with url: /search.json?query=email:jane@example.com"

    assert "jane@example.com" not in zendesk._scrub_query_strings(text)
    assert "/search.json?<query redacted>" in zendesk._scrub_query_strings(text)


def test_scrub_query_strings_leaves_a_plain_question_mark_alone():
    # A "?" alone doesn't mean a URL query string -- only one immediately
    # preceded by something path-shaped (containing a "/") is a real query
    # string; ordinary prose with a "?" must not get mangled.
    text = "Is this a duplicate ticket? Contact support."

    assert zendesk._scrub_query_strings(text) == text


def test_cursor_page_returns_next_cursor_when_has_more():
    page, has_more, after_cursor = zendesk._cursor_page(
        {
            "tickets": [{"id": 1}, {"id": 2}],
            "meta": {"has_more": True, "after_cursor": "abc"},
        },
        "tickets",
        limit=2,
    )

    assert page == [{"id": 1}, {"id": 2}]
    assert has_more is True
    assert after_cursor == "abc"


def test_cursor_page_no_next_cursor_when_not_truncated():
    page, has_more, after_cursor = zendesk._cursor_page(
        {"tickets": [{"id": 1}], "meta": {"has_more": False}}, "tickets", limit=50
    )

    assert has_more is False
    assert after_cursor is None


def test_cursor_page_rejects_non_dict_payload():
    with pytest.raises(ValueError, match="JSON object"):
        zendesk._cursor_page([], "tickets", limit=10)


def test_cursor_page_rejects_non_list_field():
    with pytest.raises(ValueError, match="tickets"):
        zendesk._cursor_page({"tickets": "not-a-list"}, "tickets", limit=10)


def test_cursor_page_raises_when_more_results_but_no_cursor():
    # Zendesk's cursor-pagination contract guarantees after_cursor whenever
    # has_more is true (or the page overflowed limit) -- a response that
    # violates that must fail loudly rather than hand back a has_more=true
    # page the caller can never resume past.
    with pytest.raises(RuntimeError, match="resume cursor"):
        zendesk._cursor_page(
            {"tickets": [{"id": 1}, {"id": 2}], "meta": {"has_more": True}},
            "tickets",
            limit=1,
        )


def test_cursor_page_raises_when_more_results_but_empty_page_and_no_cursor():
    # An empty page with has_more=true (e.g. a concurrent deletion emptied
    # out the window Zendesk reported as non-empty) is exactly the case
    # with the least to fall back on -- it must raise too, not slip through
    # as a silent has_more=true/after_cursor=null dead end just because the
    # page itself happens to be empty.
    with pytest.raises(RuntimeError, match="resume cursor"):
        zendesk._cursor_page(
            {"tickets": [], "meta": {"has_more": True}}, "tickets", limit=10
        )


def test_offset_page_has_more_when_next_page_present():
    page, has_more = zendesk._offset_page(
        {"results": [{"id": 1}], "next_page": "https://acme.zendesk.com/x"},
        "results",
        limit=1,
    )

    assert page == [{"id": 1}]
    assert has_more is True


def test_offset_page_not_more_when_no_next_page():
    _page, has_more = zendesk._offset_page(
        {"results": [{"id": 1}], "next_page": None}, "results", limit=50
    )

    assert has_more is False


def test_request_uses_configured_host_and_auth(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = zendesk._request("GET", "/tickets.json")

    assert result == {"ok": True}
    assert (
        mock_request.call_args.kwargs["url"]
        == "https://acme.zendesk.com/api/v2/tickets.json"
    )
    assert mock_request.call_args.kwargs["headers"] == {
        "Authorization": "Bearer test-token"
    }
    assert "auth" not in mock_request.call_args.kwargs
    assert mock_request.call_args.kwargs["allow_redirects"] is False


def test_request_and_its_429_retry_carry_the_marketplace_headers(monkeypatch):
    monkeypatch.setenv("ZENDESK_MARKETPLACE_NAME", "Xagent")
    monkeypatch.setenv("ZENDESK_MARKETPLACE_ORGANIZATION_ID", "12345")
    monkeypatch.setattr(zendesk.time, "sleep", lambda _seconds: None)
    mock_request = Mock(
        side_effect=[
            MockResponse(status_code=429, headers={"Retry-After": "1"}),
            MockResponse(json_data={"ok": True}),
        ]
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    assert zendesk._request("GET", "/tickets.json") == {"ok": True}

    assert mock_request.call_count == 2
    for call in mock_request.call_args_list:
        assert call.kwargs["headers"] == {
            "Authorization": "Bearer test-token",
            "X-Zendesk-Marketplace-Name": "Xagent",
            "X-Zendesk-Marketplace-Organization-Id": "12345",
        }


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_request_rejects_redirect_response(monkeypatch, status_code):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=status_code, url="https://acme.zendesk.com/x"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="redirect"):
        zendesk._request("GET", "/tickets.json")


def test_request_passes_configured_timeout(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk._request("GET", "/tickets.json")

    assert mock_request.call_args.kwargs["timeout"] == zendesk.DEFAULT_TIMEOUT_SECONDS


def test_request_returns_empty_dict_for_204(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(return_value=MockResponse(status_code=204, text="")),
    )

    assert zendesk._request("DELETE", "/tickets/1.json") == {}


def test_request_retries_once_on_429_with_retry_after(monkeypatch):
    responses = [
        MockResponse(status_code=429, url="x", headers={"Retry-After": "1"}),
        MockResponse(json_data={"ok": True}),
    ]
    mock_request = Mock(side_effect=responses)
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    monkeypatch.setattr(zendesk.time, "sleep", Mock())

    result = zendesk._request("GET", "/tickets.json")

    assert result == {"ok": True}
    assert mock_request.call_count == 2
    zendesk.time.sleep.assert_called_once_with(1)


def test_request_does_not_retry_a_second_429(monkeypatch):
    response = MockResponse(status_code=429, url="x", headers={"Retry-After": "1"})
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    monkeypatch.setattr(zendesk.time, "sleep", Mock())

    with pytest.raises(RuntimeError):
        zendesk._request("GET", "/tickets.json")

    assert mock_request.call_count == 2


def test_request_does_not_retry_on_non_numeric_retry_after(monkeypatch):
    # A non-numeric Retry-After (or, per test below, a missing one) must
    # fall back to "no usable wait signal" -- not raise out of the retry
    # logic itself.
    response = MockResponse(
        status_code=429, url="x", headers={"Retry-After": "not-a-number"}
    )
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    monkeypatch.setattr(zendesk.time, "sleep", Mock())

    with pytest.raises(RuntimeError):
        zendesk._request("GET", "/tickets.json")

    assert mock_request.call_count == 1
    zendesk.time.sleep.assert_not_called()


def test_request_does_not_retry_on_missing_retry_after(monkeypatch):
    response = MockResponse(status_code=429, url="x", headers={})
    mock_request = Mock(return_value=response)
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    monkeypatch.setattr(zendesk.time, "sleep", Mock())

    with pytest.raises(RuntimeError):
        zendesk._request("GET", "/tickets.json")

    assert mock_request.call_count == 1
    zendesk.time.sleep.assert_not_called()


def test_request_redacts_connection_error_message(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.exceptions.ProxyError(
            "Unable to connect to proxy: "
            "https://user:sp-secret-proxy-pass@proxy.internal:8080/"
        )

    monkeypatch.setattr(zendesk._session, "request", _raise)

    with pytest.raises(RuntimeError) as excinfo:
        zendesk._request("GET", "/tickets.json")

    assert "sp-secret-proxy-pass" not in str(excinfo.value)


def test_request_http_error_message_omits_query_bearing_url(monkeypatch):
    # requests.HTTPError's default str() is "<status> ... for url: <full
    # url>", and the full url includes the query string -- for the search
    # tools, the caller's own query, documented with an email-address
    # example. The raised message is built from the status code and
    # Zendesk's own error detail only, never str(exc), so the query never
    # reaches it in the first place.
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=422,
                url="https://acme.zendesk.com/api/v2/search.json"
                "?query=email:jane@example.com",
                text="not json",
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        zendesk._request(
            "GET", "/search.json", params={"query": "email:jane@example.com"}
        )

    assert "jane@example.com" not in str(excinfo.value)
    assert "email:jane@example.com" not in str(excinfo.value)


def test_request_scrubs_query_string_from_connection_error_message(monkeypatch):
    # Unlike the HTTPError path above, a connection-level failure's own
    # message (urllib3's "Max retries exceeded with url: ...") is the only
    # thing available and does get surfaced -- so it goes through
    # _sanitize_exception_text's query-string scrub rather than str(exc)
    # verbatim.
    def _raise(*args, **kwargs):
        raise requests.exceptions.ConnectionError(
            "Max retries exceeded with url: "
            "/api/v2/search.json?query=email:jane@example.com "
            "(Caused by NewConnectionError(...))"
        )

    monkeypatch.setattr(zendesk._session, "request", _raise)

    with pytest.raises(RuntimeError) as excinfo:
        zendesk._request("GET", "/search.json")

    assert "jane@example.com" not in str(excinfo.value)
    assert "<query redacted>" in str(excinfo.value)


def test_request_raises_with_structured_error_detail(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=404,
                json_data={"error": "RecordNotFound", "description": "Not found"},
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Not found"):
        zendesk._request("GET", "/tickets/999.json")


@pytest.mark.parametrize("status_code", [401, 403])
def test_request_auth_failure_tells_the_user_to_reconnect(monkeypatch, status_code):
    # An expired/revoked OAuth token (401) or a grant lacking the needed
    # scope (403) can only be fixed by reconnecting Zendesk, so the error
    # says so rather than leaving the agent to retry.
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=status_code,
                json_data={"error": "Couldn't authenticate you"},
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        zendesk._request("GET", "/tickets/999.json")

    message = str(excinfo.value)
    assert f"HTTP {status_code}" in message
    assert "Couldn't authenticate you" in message
    assert "reconnect" in message.lower()


def test_request_error_never_echoes_the_access_token(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={"error": "invalid_token", "description": "test-token"},
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        zendesk._request("GET", "/tickets/999.json")

    assert "test-token" not in str(excinfo.value)


def test_request_error_redacts_a_token_echoed_past_the_truncation_point(monkeypatch):
    token = "t" * 40
    monkeypatch.setenv("ZENDESK_ACCESS_TOKEN", token)
    body = "x" * 10_000 + token
    monkeypatch.setattr(zendesk, "truncate_error_text", lambda text: text[:10_020])
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(return_value=MockResponse(status_code=401, text=body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        zendesk._request("GET", "/tickets/999.json")

    assert "t" * 20 not in str(excinfo.value)


def test_request_error_redacts_a_token_cut_by_the_details_truncation(monkeypatch):
    # The field-level reasons are truncated while being extracted, so a token
    # echoed across that cut must be removed from the raw body beforehand.
    token = "k" * 40
    monkeypatch.setenv("ZENDESK_ACCESS_TOKEN", token)
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=422,
                json_data={
                    "error": "RecordInvalid",
                    "description": "Record validation errors",
                    "details": {"base": [{"description": "y" * 980 + token}]},
                },
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        zendesk._request("POST", "/tickets.json")

    message = str(excinfo.value)
    assert "Record validation errors" in message
    assert "k" * 20 not in message


def test_request_error_is_not_a_reconnect_hint_for_other_statuses(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(status_code=404, json_data={"error": "Not found"})
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        zendesk._request("GET", "/tickets/999.json")

    assert "reconnect" not in str(excinfo.value).lower()


def test_request_raises_clear_error_for_non_json_2xx_body(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=200, text="<html>not json</html>", json_raises=True
            )
        ),
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        zendesk._request("GET", "/tickets.json")


def test_request_truncates_unstructured_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        zendesk._request("GET", "/tickets.json")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_search_requires_non_blank_query():
    result = json.loads(zendesk.zendesk_search("   "))

    assert result["status"] == "error"


def test_search_handles_none_query_without_crashing_the_error_handler():
    # _require_non_blank raises for a None query, but the except block's
    # own len(query) used to run on the original (still-None) parameter --
    # turning the error handler itself into an uncaught TypeError instead
    # of a clean _error() response.
    result = json.loads(zendesk.zendesk_search(None))

    assert result["status"] == "error"


def test_search_sends_per_page_and_page_params(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [
                    {"result_type": "ticket", "id": 1, "subject": "Help"},
                    {"result_type": "user", "id": 2, "name": "Jane"},
                ],
                "count": 2,
                "next_page": None,
            }
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_search("type:ticket status:open", limit=10, page=2)
    )

    assert result["status"] == "success"
    results = result["results"]
    assert results[0]["result_type"] == "ticket"
    assert results[0]["subject"] == "Help"
    assert results[1]["result_type"] == "user"
    assert mock_request.call_args.kwargs["params"] == {
        "query": "type:ticket status:open",
        "per_page": 10,
        "page": 2,
    }


def test_search_forces_has_more_when_output_truncated(monkeypatch):
    # Zendesk's own next_page says this is the last page (has_more would
    # normally be False), but the results are large enough to get locally
    # truncated for output size -- has_more must reflect that truncation
    # regardless of what Zendesk said, or a caller trusting has_more alone
    # would stop paging and silently lose the trimmed results.
    big_results = [
        {"result_type": "ticket", "id": i, "subject": "x" * 1000} for i in range(50)
    ]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"results": big_results, "count": 50, "next_page": None}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 2000)

    raw = zendesk.zendesk_search("type:ticket", limit=50)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert 0 < len(result["results"]) < len(big_results)
    assert result["has_more"] is True
    assert len(raw) <= 2000  # _finalize_capped guarantees this now


def test_search_explains_a_single_oversized_item(monkeypatch):
    one_huge_result = [{"result_type": "ticket", "id": 1, "subject": "x" * 5000}]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"results": one_huge_result, "count": 1, "next_page": None}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 2000)

    raw = zendesk.zendesk_search("type:ticket", limit=50)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["results"] == []
    assert result["truncated"] is True
    assert result["has_more"] is True
    assert "individually too large" in result["message"]
    assert len(raw) <= 2000


def test_search_uses_short_message_when_full_explanation_does_not_fit(monkeypatch):
    # Same gap as the cursor paginator's equivalent test: a cap too small
    # for the long explanation but large enough for a shorter one must
    # still explain the oversized-item case, not silently revert to a bare
    # has_more=true/empty-results envelope with no message.
    one_huge_result = [{"result_type": "ticket", "id": 1, "subject": "x" * 5000}]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"results": one_huge_result, "count": 1, "next_page": None}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 200)

    raw = zendesk.zendesk_search("type:ticket", limit=50)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["results"] == []
    assert result["has_more"] is True
    assert "too large to fit" in result["message"]
    assert len(raw) <= 200


def test_search_reports_error_when_even_short_message_does_not_fit(monkeypatch):
    one_huge_result = [{"result_type": "ticket", "id": 1, "subject": "x" * 5000}]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"results": one_huge_result, "count": 1, "next_page": None}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 150)

    raw = zendesk.zendesk_search("type:ticket", limit=50)
    result = json.loads(raw)

    assert result["status"] == "error"
    assert len(raw) <= 150


def test_search_clamps_page_to_at_least_one(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [], "count": 0})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_search("type:ticket", page=0)

    assert mock_request.call_args.kwargs["params"]["page"] == 1


def test_past_search_window_checks_the_windows_start_not_its_end():
    # limit=30, page=34: the page starts at (page-1)*limit=990, still under
    # the 1000 ceiling, so it can contain real results (991-1000) even
    # though it also straddles past the ceiling. Rejecting it locally
    # (checking the end) would silently discard those results with no way
    # to recover them through this tool -- forwarding it to Zendesk instead
    # either serves the valid tail or gets a visible 422, never silent loss.
    # page=34 itself (start=1020) is genuinely past the window and must be
    # rejected.
    ceiling = zendesk._MAX_SEARCH_RESULT_WINDOW
    assert (
        zendesk._past_search_window(page=33, max_results=30, ceiling=ceiling) is False
    )
    assert (
        zendesk._past_search_window(page=34, max_results=30, ceiling=ceiling) is False
    )
    assert zendesk._past_search_window(page=35, max_results=30, ceiling=ceiling) is True


def test_past_search_window_uses_the_ceiling_it_is_given():
    # zendesk_search (1,000) and zendesk_search_users (10,000) have
    # different documented result-window ceilings -- the same page/limit
    # combination must be judged against whichever ceiling the caller
    # passes in, not a single shared constant.
    assert zendesk._past_search_window(page=11, max_results=100, ceiling=1000) is True
    assert zendesk._past_search_window(page=11, max_results=100, ceiling=10000) is False


def test_search_returns_empty_past_result_window_without_calling_zendesk(monkeypatch):
    # Past Zendesk's own documented result-window ceiling, Zendesk itself
    # would answer with an opaque HTTP error -- a caller mechanically
    # incrementing `page` must get a clean "no more results" instead.
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    page = zendesk._MAX_SEARCH_RESULT_WINDOW // 100 + 1

    result = json.loads(zendesk.zendesk_search("type:ticket", limit=100, page=page))

    # Same flat shape as the normal success path -- this tool must return
    # one consistent shape regardless of how far it paged.
    assert result == {
        "status": "success",
        "results": [],
        "count": None,
        "has_more": False,
        "truncated": False,
    }
    mock_request.assert_not_called()


def test_search_past_result_window_respects_the_output_cap(monkeypatch):
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 60)
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    page = zendesk._MAX_SEARCH_RESULT_WINDOW // 100 + 1

    raw = zendesk.zendesk_search("type:ticket", limit=100, page=page)

    assert len(raw) <= 60
    assert json.loads(raw)["status"] == "error"
    mock_request.assert_not_called()


def test_search_past_result_window_still_surfaces_missing_credentials(monkeypatch):
    # The past-window guard returns before ever calling _request(), which
    # is normally what validates ZENDESK_SUBDOMAIN/ACCESS_TOKEN -- a
    # misconfigured environment must not be masked as a clean "no results"
    # just because the caller happened to page past the window.
    monkeypatch.delenv("ZENDESK_ACCESS_TOKEN")
    page = zendesk._MAX_SEARCH_RESULT_WINDOW // 100 + 1

    result = json.loads(zendesk.zendesk_search("type:ticket", limit=100, page=page))

    assert result["status"] == "error"
    assert "ZENDESK_ACCESS_TOKEN" in result["message"]


def test_list_tickets_sends_page_size_and_cursor(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"tickets": [{"id": 1}], "meta": {"has_more": False}}
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_list_tickets(limit=10, after_cursor="cur1")

    params = mock_request.call_args.kwargs["params"]
    assert params["page[size]"] == 10
    assert params["page[after]"] == "cur1"


def test_list_tickets_caps_output_size(monkeypatch):
    big_tickets = [{"id": i, "subject": "x" * 1000} for i in range(50)]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"tickets": big_tickets, "meta": {"has_more": False}}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 2000)

    raw = zendesk.zendesk_list_tickets(limit=50)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert 0 < len(result["tickets"]) < len(big_tickets)
    # Zendesk's own meta said this was the last page (has_more: False), but
    # truncation dropped tickets this call never returned -- has_more must
    # reflect that regardless of what Zendesk's meta said, or a caller that
    # trusts has_more alone will stop paging and silently lose them.
    assert result["has_more"] is True
    assert len(raw) <= 2000  # _finalize_capped guarantees this now


def test_list_tickets_truncation_retries_same_page_not_zendesks_next_page(monkeypatch):
    # A truncated page must report ITS OWN input cursor so the caller
    # retries the same starting point with a smaller limit -- returning
    # Zendesk's real after_cursor instead would skip every item this call
    # couldn't fit, silently dropping them for good.
    big_tickets = [{"id": i, "subject": "x" * 1000} for i in range(50)]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "tickets": big_tickets,
                    "meta": {
                        "has_more": True,
                        "after_cursor": "zendesks_real_next_cursor",
                    },
                }
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 2000)

    result = json.loads(zendesk.zendesk_list_tickets(limit=50, after_cursor="cur0"))

    assert result["truncated"] is True
    assert result["has_more"] is True
    assert result["after_cursor"] == "cur0"


def test_list_tickets_explains_a_single_oversized_item(monkeypatch):
    # If even the single largest remaining ticket doesn't fit alone, the
    # halving loop collapses to an empty list while has_more/truncated stay
    # true -- "retry with a smaller limit" (the tool's own general advice)
    # cannot help here, since limit only controls item *count*, not the
    # size of one item. Say so explicitly instead of implying a fix that
    # doesn't exist.
    one_huge_ticket = [{"id": 1, "subject": "x" * 5000}]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"tickets": one_huge_ticket, "meta": {"has_more": False}}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 2000)

    raw = zendesk.zendesk_list_tickets(limit=50)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["tickets"] == []
    assert result["truncated"] is True
    assert result["has_more"] is True
    assert "individually too large" in result["message"]
    assert len(raw) <= 2000


def test_list_tickets_falls_back_to_error_when_cap_is_impossibly_small(monkeypatch):
    # An operator-configured XAGENT_TOOL_MAX_OUTPUT_LENGTH small enough that
    # even the empty, message-less envelope exceeds it must still produce
    # valid JSON -- not an oversized string for the platform's own output
    # filter to hard-truncate into garbage. The fallback itself must also
    # fit under the cap, not just be "a short string" in the abstract.
    one_huge_ticket = [{"id": 1, "subject": "x" * 5000}]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"tickets": one_huge_ticket, "meta": {"has_more": False}}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 20)

    raw = zendesk.zendesk_list_tickets(limit=50)
    result = json.loads(raw)  # must not raise -- valid JSON either way

    assert result["status"] == "error"
    assert len(raw) <= 20


def test_list_tickets_uses_short_message_when_full_explanation_does_not_fit(
    monkeypatch,
):
    # A cap too small for _OVERSIZED_ITEMS_MESSAGE (342 chars in this
    # scenario) but large enough for a shorter one must still explain the
    # oversized-item case -- never silently fall back to a bare
    # has_more=true/empty-list envelope just because the long explanation
    # didn't fit, which would be indistinguishable from ordinary truncation
    # and would send the caller into a retry loop that can never progress.
    one_huge_ticket = [{"id": 1, "subject": "x" * 5000}]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"tickets": one_huge_ticket, "meta": {"has_more": False}}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 200)

    raw = zendesk.zendesk_list_tickets(limit=50)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["tickets"] == []
    assert result["has_more"] is True
    assert "too large to fit" in result["message"]
    assert len(raw) <= 200


def test_list_tickets_reports_error_when_even_short_message_does_not_fit(
    monkeypatch,
):
    # Below the short message's own size, neither explanation fits -- this
    # must fall to _finalize_capped's own error tier rather than ever
    # returning the misleading bare has_more=true envelope, which fits at
    # this cap but has no way to tell the caller why the page is empty.
    one_huge_ticket = [{"id": 1, "subject": "x" * 5000}]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"tickets": one_huge_ticket, "meta": {"has_more": False}}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 150)

    raw = zendesk.zendesk_list_tickets(limit=50)
    result = json.loads(raw)

    assert result["status"] == "error"
    assert len(raw) <= 150


def test_ticket_summary_includes_group_id(monkeypatch):
    # F14: zendesk_update_ticket accepts and sends group_id, but the shared
    # projection used by every ticket-returning tool omitted it -- callers
    # couldn't observe the group assignment they just requested.
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(json_data={"ticket": {"id": 1, "group_id": 42}})
        ),
    )

    result = json.loads(zendesk.zendesk_get_ticket(1))

    assert result["ticket"]["group_id"] == 42


def test_get_ticket_returns_summary(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"ticket": {"id": 1, "subject": "Help", "status": "open"}}
            )
        ),
    )

    result = json.loads(zendesk.zendesk_get_ticket(1))

    assert result["status"] == "success"
    assert result["ticket"]["subject"] == "Help"


def test_get_ticket_includes_description(monkeypatch):
    # A single ticket fetch is exactly the case where the caller wants the
    # actual text, not just metadata -- without this, the only way to read
    # a ticket's body was a second call to list comments and guess the
    # first one is the description. Mirrors linear.py's detail-vs-summary
    # split.
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "ticket": {
                        "id": 1,
                        "subject": "Help",
                        "description": "The full body text of the ticket.",
                        "organization_id": 42,
                        "type": "incident",
                        "via": {"channel": "web"},
                    }
                }
            )
        ),
    )

    result = json.loads(zendesk.zendesk_get_ticket(1))

    assert result["ticket"]["description"] == "The full body text of the ticket."
    assert result["ticket"]["organization_id"] == 42
    assert result["ticket"]["type"] == "incident"
    assert result["ticket"]["via"] == {"channel": "web"}


def test_get_ticket_shrinks_an_oversized_description_before_erroring(monkeypatch):
    # description can be up to 64KiB, well past a typical output cap --
    # this must degrade the same way an oversized comment body already
    # does (shrink with a truncated flag), not fail the whole call via
    # _finalize_capped's blunter last-resort fallback.
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "ticket": {
                        "id": 1,
                        "subject": "Help",
                        "description": "x" * 5000,
                    }
                }
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 2000)

    raw = zendesk.zendesk_get_ticket(1)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["ticket"]["description_truncated"] is True
    assert len(result["ticket"]["description"]) < 5000
    assert len(raw) <= 2000


def test_get_ticket_falls_back_to_error_when_response_is_too_large(monkeypatch):
    # zendesk_get_ticket builds a single fixed-shape object with no list to
    # shrink -- unlike the list/comment tools, it had no size guarantee at
    # all before _finalize_capped was added here. A ticket with a large
    # tag set (caller/account-controlled content) must still produce valid
    # JSON, not a response the platform's own output filter then
    # hard-truncates into garbage.
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "ticket": {
                        "id": 1,
                        "subject": "x" * 5000,
                        "tags": [f"tag{i}" for i in range(100)],
                    }
                }
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 200)

    raw = zendesk.zendesk_get_ticket(1)
    result = json.loads(raw)  # must not raise -- valid JSON either way

    assert result["status"] == "error"


def test_get_ticket_accepts_agent_ui_url(monkeypatch):
    # A caller copying a link out of the Zendesk agent UI (rather than a
    # bare id) must resolve to the same ticket, not error out.
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 123, "subject": "Help"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_get_ticket("https://acme.zendesk.com/agent/tickets/123")
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/tickets/123.json")


def test_create_ticket_requires_subject_and_comment():
    result = json.loads(zendesk.zendesk_create_ticket("", "body"))
    assert result["status"] == "error"

    result = json.loads(zendesk.zendesk_create_ticket("subject", ""))
    assert result["status"] == "error"


def test_create_ticket_sends_expected_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "subject": "Help"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_create_ticket(
            "Help",
            "Something's broken",
            requester_email="jane@example.com",
            priority="high",
            tags=["bug", "urgent"],
        )
    )

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"]["subject"] == "Help"
    assert body["ticket"]["comment"] == {"body": "Something's broken"}
    assert body["ticket"]["requester"] == {"email": "jane@example.com"}
    assert body["ticket"]["priority"] == "high"
    assert body["ticket"]["tags"] == ["bug", "urgent"]
    assert mock_request.call_args.kwargs["method"] == "POST"


def test_create_ticket_treats_whitespace_only_priority_as_not_provided(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "subject": "Help"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_create_ticket("Help", "Something's broken", priority="   ")

    body = mock_request.call_args.kwargs["json"]
    assert "priority" not in body["ticket"]


def test_create_ticket_sends_requester_name_with_email(monkeypatch):
    # F11: Zendesk requires a name for a requester_email that doesn't
    # already match an existing user; name-only requesters aren't
    # supported by this connector, but name+email must reach Zendesk.
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "subject": "Help"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_create_ticket(
        "Help",
        "Something's broken",
        requester_email="new.user@example.com",
        requester_name="New User",
    )

    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"]["requester"] == {
        "email": "new.user@example.com",
        "name": "New User",
    }


def test_create_ticket_rejects_requester_name_without_email():
    result = json.loads(
        zendesk.zendesk_create_ticket(
            "Help", "Something's broken", requester_name="New User"
        )
    )

    assert result["status"] == "error"


def test_create_ticket_treats_whitespace_only_requester_name_as_not_provided(
    monkeypatch,
):
    # A blank-looking name must not silently satisfy "name is required for
    # a new requester" -- it's treated the same as name not being provided
    # at all (matching status/priority's blank-vs-unset handling), not
    # sent to Zendesk verbatim.
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "subject": "Help"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_create_ticket(
        "Help",
        "Something's broken",
        requester_email="new@example.com",
        requester_name="   ",
    )

    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"]["requester"] == {"email": "new@example.com"}


def test_create_ticket_treats_whitespace_only_requester_email_as_not_provided():
    result = json.loads(
        zendesk.zendesk_create_ticket(
            "Help",
            "Something's broken",
            requester_email="   ",
            requester_name="New User",
        )
    )

    # requester_email is blank, so requester_name has nothing to attach
    # to -- same error as omitting requester_email entirely.
    assert result["status"] == "error"


def test_create_ticket_cleans_tags(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "subject": "Help"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_create_ticket(
        "Help", "Something's broken", tags=["vip ", "", "  priority"]
    )

    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"]["tags"] == ["vip", "priority"]


def test_update_ticket_requires_at_least_one_field():
    result = json.loads(zendesk.zendesk_update_ticket(1))

    assert result["status"] == "error"


def test_update_ticket_sends_only_provided_fields(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "status": "solved"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_update_ticket(1, status="solved")

    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"] == {"status": "solved"}
    assert mock_request.call_args.kwargs["url"].endswith("/tickets/1.json")
    assert mock_request.call_args.kwargs["method"] == "PUT"


def test_update_ticket_rejects_invalid_status():
    result = json.loads(zendesk.zendesk_update_ticket(1, status="archived"))

    assert result["status"] == "error"


def test_update_ticket_rejects_closed_status(monkeypatch):
    # "closed" is a valid value the status field can *read* as, but
    # reaching it is automation-only (Zendesk rejects a direct attempt to
    # set it remotely) -- it must not be in the locally-accepted set
    # either, or local validation gives false confidence it's settable.
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_update_ticket(1, status="closed"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_update_ticket_rejects_invalid_priority():
    result = json.loads(zendesk.zendesk_update_ticket(1, priority="critical"))

    assert result["status"] == "error"


def test_update_ticket_normalizes_status_and_priority_case(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ticket": {"id": 1}}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_update_ticket(1, status="Solved", priority="URGENT")

    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"] == {"status": "solved", "priority": "urgent"}


def test_create_ticket_rejects_invalid_priority():
    result = json.loads(
        zendesk.zendesk_create_ticket("Help", "Something's broken", priority="asap")
    )

    assert result["status"] == "error"


def test_update_ticket_treats_empty_status_and_priority_as_not_provided(monkeypatch):
    # There is no valid "clear the status/priority" value in Zendesk (unlike
    # tags, which support an explicit empty list) -- an LLM caller filling
    # in "" for a field it doesn't want to change must not turn into a
    # request that fails the whole update.
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "priority": "urgent"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_update_ticket(1, status="", priority="urgent"))

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"] == {"priority": "urgent"}


def test_update_ticket_treats_whitespace_only_status_as_not_provided(monkeypatch):
    # Same as the blank-string case above, but whitespace-only -- truthy in
    # Python, so a bare `if status:` check alone would forward it.
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "priority": "urgent"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_update_ticket(1, status="   ", priority="urgent")
    )

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"] == {"priority": "urgent"}


def test_update_ticket_cleans_tags(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ticket": {"id": 1}}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_update_ticket(1, tags=["vip ", "", "  priority"])

    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"]["tags"] == ["vip", "priority"]


def test_update_ticket_sends_assignee_and_group(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ticket": {"id": 1}}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_update_ticket(1, assignee_id=42, group_id=7))

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"] == {"assignee_id": 42, "group_id": 7}


def test_delete_ticket_sends_delete_request(monkeypatch):
    mock_request = Mock(return_value=MockResponse(status_code=204, text=""))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_delete_ticket(1))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["method"] == "DELETE"
    assert mock_request.call_args.kwargs["url"].endswith("/tickets/1.json")


def test_delete_ticket_accepts_agent_ui_url(monkeypatch):
    mock_request = Mock(return_value=MockResponse(status_code=204, text=""))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_delete_ticket("https://acme.zendesk.com/agent/tickets/123")
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/tickets/123.json")
    # The response must echo the resolved numeric id as a real int, not the
    # raw URL the caller passed in and not a string -- so it's directly
    # comparable with the id every other tool's response returns.
    assert result["ticket_id"] == 123


def test_update_ticket_clears_tags_with_explicit_empty_list(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ticket": {"id": 1}}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_update_ticket(1, tags=[]))

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"] == {"tags": []}


def test_update_ticket_rejects_all_blank_tags_instead_of_clearing(monkeypatch):
    # tags=["  ", ""] is a non-empty list that cleans down to nothing --
    # unlike an explicit tags=[], this must not be silently treated as
    # "clear all tags" (Zendesk's tag update is a full replace, so that
    # would destructively wipe an existing tag set the caller never asked
    # to clear, with no undo available in this connector).
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_update_ticket(1, tags=["  ", ""]))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_ticket_rejects_all_blank_tags_instead_of_clearing(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_create_ticket("Help", "Something's broken", tags=["  ", ""])
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_list_ticket_comments_returns_summaries(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "comments": [
                        {"id": 1, "plain_body": "Hi", "public": True, "author_id": 5}
                    ],
                    "meta": {"has_more": False},
                }
            )
        ),
    )

    result = json.loads(zendesk.zendesk_list_ticket_comments(1))

    assert result["status"] == "success"
    assert result["comments"][0]["body"] == "Hi"


def test_reply_to_ticket_sends_public_true(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "status": "open"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_reply_to_ticket(1, "Thanks for reaching out"))

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"]["comment"] == {
        "body": "Thanks for reaching out",
        "public": True,
    }


def test_add_internal_note_sends_public_false(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"ticket": {"id": 1, "status": "open"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_add_internal_note(1, "Escalating to tier 2")

    body = mock_request.call_args.kwargs["json"]
    assert body["ticket"]["comment"] == {
        "body": "Escalating to tier 2",
        "public": False,
    }


def test_add_comment_rejects_blank_body():
    result = json.loads(zendesk.zendesk_reply_to_ticket(1, "   "))

    assert result["status"] == "error"


def test_add_comment_rejects_body_over_zendesk_limit(monkeypatch):
    # Zendesk silently truncates a comment body past 64KiB instead of
    # rejecting it -- reject locally first so the caller gets a clear
    # error instead of a write that "succeeds" with less content than sent.
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    oversized_body = "x" * (zendesk._MAX_COMMENT_BODY_BYTES + 1)

    result = json.loads(zendesk.zendesk_reply_to_ticket(1, oversized_body))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_add_comment_accepts_body_at_exactly_the_zendesk_limit(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(return_value=MockResponse(json_data={"ticket": {"id": 1}})),
    )
    exact_limit_body = "x" * zendesk._MAX_COMMENT_BODY_BYTES

    result = json.loads(zendesk.zendesk_reply_to_ticket(1, exact_limit_body))

    assert result["status"] == "success"


def test_add_comment_rejects_multibyte_body_over_the_byte_limit(monkeypatch):
    # The limit is a *byte* limit (Zendesk's own wording is "64kB"), and
    # _require_comment_body measures value.encode("utf-8"), not len(value)
    # -- confirm the byte-vs-character distinction actually holds: a CJK
    # character is 3 bytes in UTF-8, so 25,000 of them is well under
    # 65,536 *characters* but well over 65,536 *bytes* (75,000). A
    # character-count check would wrongly accept this.
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    multibyte_body = "你" * 25000
    assert len(multibyte_body) < zendesk._MAX_COMMENT_BODY_BYTES
    assert len(multibyte_body.encode("utf-8")) > zendesk._MAX_COMMENT_BODY_BYTES

    result = json.loads(zendesk.zendesk_reply_to_ticket(1, multibyte_body))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_ticket_rejects_comment_over_zendesk_limit(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    oversized_comment = "x" * (zendesk._MAX_COMMENT_BODY_BYTES + 1)

    result = json.loads(zendesk.zendesk_create_ticket("Help", oversized_comment))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_add_comment_truncates_echoed_body_to_fit_output_cap(monkeypatch):
    # F9: the reply/note tools are the only mutation responses in this file
    # that echo caller-supplied free text back verbatim -- a large-but-
    # under-Zendesk's-64KiB-limit body must still be bounded before being
    # baked into the JSON response, or the platform's own output filter
    # (which truncates raw strings, not JSON-aware) can corrupt it.
    long_body = "x" * 60000
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "ticket": {"id": 1, "status": "open"},
                    "audit": {
                        "events": [
                            {
                                "type": "Comment",
                                "id": 999,
                                "body": long_body,
                                "public": True,
                            }
                        ]
                    },
                }
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 2000)

    raw = zendesk.zendesk_reply_to_ticket(1, long_body)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert len(result["comment"]["body"]) < len(long_body)
    assert result["comment"]["body_truncated"] is True
    # id/public still confirm what was posted and that it went out
    # publicly, even with the body preview cut down.
    assert result["comment"]["id"] == 999
    assert result["comment"]["public"] is True
    assert len(raw) <= 2000  # _finalize_capped guarantees this now


def test_reply_to_ticket_returns_created_comment_from_audit_events(monkeypatch):
    # PUT /tickets/{id}.json returns {"ticket": ..., "audit": {"events":
    # [...]}} -- the just-created comment's own id/body/public only appear
    # in the audit trail, not in the "ticket" object itself.
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "ticket": {"id": 1, "status": "open"},
                "audit": {
                    "events": [
                        {"type": "Create"},
                        {
                            "type": "Comment",
                            "id": 999,
                            "body": "Thanks for reaching out",
                            "public": True,
                            "author_id": 5,
                        },
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_reply_to_ticket(1, "Thanks for reaching out"))

    assert result["status"] == "success"
    assert result["ticket"]["id"] == 1
    assert result["comment"] == {
        "id": 999,
        "author_id": 5,
        "body": "Thanks for reaching out",
        "public": True,
        "created_at": None,
    }


def test_add_internal_note_returns_none_comment_when_audit_missing(monkeypatch):
    # Not every Zendesk response necessarily carries an audit trail (e.g.
    # a stubbed/older API version) -- the comment field must degrade to
    # None rather than raise.
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(return_value=MockResponse(json_data={"ticket": {"id": 1}})),
    )

    result = json.loads(zendesk.zendesk_add_internal_note(1, "Escalating to tier 2"))

    assert result["status"] == "success"
    assert result["comment"] is None


def test_add_internal_note_does_not_echo_a_mismatched_public_comment(monkeypatch):
    # A residual case where an account-configured integration adds its own
    # comment into the same audit trail: the first Comment-type event isn't
    # necessarily the one this call created. Matching on `public` is cheap
    # insurance -- a candidate whose visibility differs from what this call
    # actually sent can never be it, so it must not be echoed back
    # mislabeled (an internal note reported as having gone out publicly
    # would be a real, visible-to-nobody-but-us safety issue).
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "ticket": {"id": 1},
                    "audit": {
                        "events": [
                            {
                                "type": "Comment",
                                "id": 111,
                                "body": "A public reply from another integration",
                                "public": True,
                            },
                            {
                                "type": "Comment",
                                "id": 222,
                                "body": "Escalating to tier 2",
                                "public": False,
                            },
                        ]
                    },
                }
            )
        ),
    )

    result = json.loads(zendesk.zendesk_add_internal_note(1, "Escalating to tier 2"))

    assert result["status"] == "success"
    assert result["comment"]["id"] == 222
    assert result["comment"]["public"] is False


def test_add_comment_reports_ticket_and_comment_truncation_independently(monkeypatch):
    # A long subject plus many tags can make the ticket summary alone
    # exceed the cap even after the comment body is fully shrunk -- the
    # halving loop must fall through to trimming the ticket summary too
    # (down to its id, the correlation key) rather than returning an
    # over-cap response once comment_body has nothing left to give.
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "ticket": {
                        "id": 1,
                        "subject": "x" * 5000,
                        "tags": [f"tag{i}" for i in range(100)],
                    },
                    "audit": {
                        "events": [
                            {
                                "type": "Comment",
                                "id": 1,
                                "body": "Thanks!",
                                "public": True,
                            }
                        ]
                    },
                }
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 2000)

    raw = zendesk.zendesk_reply_to_ticket(1, "Thanks!")
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["ticket"] == {"id": 1}
    assert len(raw) <= 2000


def test_list_cursor_paginated_requests_boundary_indicators(monkeypatch):
    # F7: users/organizations omit meta.has_more entirely unless
    # include_boundary_indicators=true is passed, which _cursor_page then
    # reads as "no more pages" -- silently dropping every item past the
    # first page. Every cursor-paginated call must request it.
    mock_request = Mock(
        return_value=MockResponse(json_data={"users": [], "meta": {"has_more": False}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_list_users()

    assert (
        mock_request.call_args.kwargs["params"]["include_boundary_indicators"] == "true"
    )


def test_list_users_returns_summaries(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "users": [{"id": 1, "name": "Jane", "email": "jane@example.com"}],
                    "meta": {"has_more": False},
                }
            )
        ),
    )

    result = json.loads(zendesk.zendesk_list_users())

    assert result["status"] == "success"
    assert result["users"][0]["email"] == "jane@example.com"


def test_get_user_returns_summary(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(return_value=MockResponse(json_data={"user": {"id": 1, "name": "Jane"}})),
    )

    result = json.loads(zendesk.zendesk_get_user(1))

    assert result["status"] == "success"
    assert result["user"]["name"] == "Jane"


def test_get_user_falls_back_to_error_when_response_is_too_large(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(json_data={"user": {"id": 1, "name": "x" * 5000}})
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 200)

    raw = zendesk.zendesk_get_user(1)
    result = json.loads(raw)  # must not raise -- valid JSON either way

    assert result["status"] == "error"


def test_get_user_accepts_agent_ui_url(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"user": {"id": 42, "name": "Jane"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_get_user("https://acme.zendesk.com/agent/users/42")
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/users/42.json")


def test_search_users_requires_non_blank_query():
    result = json.loads(zendesk.zendesk_search_users(""))

    assert result["status"] == "error"


def test_search_users_handles_none_query_without_crashing_the_error_handler():
    result = json.loads(zendesk.zendesk_search_users(None))

    assert result["status"] == "error"


def test_search_users_returns_empty_past_result_window_without_calling_zendesk(
    monkeypatch,
):
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    # /users/search.json's own ceiling is 10,000 (not the unified
    # /search.json's 1,000) -- use the user-search-specific constant so
    # this test doesn't silently start testing the wrong endpoint's window.
    page = zendesk._MAX_USER_SEARCH_RESULT_WINDOW // 100 + 1

    result = json.loads(
        zendesk.zendesk_search_users("jane@example.com", limit=100, page=page)
    )

    assert result == {
        "status": "success",
        "users": [],
        "has_more": False,
        "truncated": False,
    }
    mock_request.assert_not_called()


def test_search_users_past_result_window_respects_the_output_cap(monkeypatch):
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 60)
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    page = zendesk._MAX_USER_SEARCH_RESULT_WINDOW // 100 + 1

    raw = zendesk.zendesk_search_users("jane@example.com", limit=100, page=page)

    assert len(raw) <= 60
    assert json.loads(raw)["status"] == "error"
    mock_request.assert_not_called()


def test_search_users_does_not_trip_the_unified_search_window(monkeypatch):
    # Regression for F6: reusing /search.json's 1,000-result ceiling for
    # /users/search.json (which supports 10,000) would silently return an
    # empty page for pages 11-100 even though Zendesk has real results
    # there.
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"users": [{"id": 1, "name": "Jane"}], "next_page": None}
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_search_users("jane@example.com", limit=100, page=50)
    )

    assert result["status"] == "success"
    assert len(result["users"]) == 1
    mock_request.assert_called_once()


def test_search_users_past_result_window_still_surfaces_missing_credentials(
    monkeypatch,
):
    monkeypatch.delenv("ZENDESK_ACCESS_TOKEN")
    page = zendesk._MAX_USER_SEARCH_RESULT_WINDOW // 100 + 1

    result = json.loads(
        zendesk.zendesk_search_users("jane@example.com", limit=100, page=page)
    )

    assert result["status"] == "error"
    assert "ZENDESK_ACCESS_TOKEN" in result["message"]


def test_search_users_sends_query_and_page_params(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"users": [{"id": 1, "name": "Jane"}], "next_page": None}
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_search_users("jane@example.com", limit=5))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/users/search.json")
    assert mock_request.call_args.kwargs["params"] == {
        "query": "jane@example.com",
        "per_page": 5,
        "page": 1,
    }


def test_search_users_forces_has_more_when_output_truncated(monkeypatch):
    big_users = [
        {"id": i, "name": "x" * 1000, "email": "jane@example.com"} for i in range(50)
    ]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(json_data={"users": big_users, "next_page": None})
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 2000)

    raw = zendesk.zendesk_search_users("jane@example.com", limit=50)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert 0 < len(result["users"]) < len(big_users)
    assert result["has_more"] is True
    assert len(raw) <= 2000  # _finalize_capped guarantees this now


def test_list_organizations_returns_summaries(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "organizations": [{"id": 1, "name": "Acme"}],
                    "meta": {"has_more": False},
                }
            )
        ),
    )

    result = json.loads(zendesk.zendesk_list_organizations())

    assert result["status"] == "success"
    assert result["organizations"][0]["name"] == "Acme"


def test_get_organization_returns_summary(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"organization": {"id": 1, "name": "Acme"}}
            )
        ),
    )

    result = json.loads(zendesk.zendesk_get_organization(1))

    assert result["status"] == "success"
    assert result["organization"]["name"] == "Acme"


def test_get_organization_accepts_agent_ui_url(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"organization": {"id": 7, "name": "Acme"}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_get_organization(
            "https://acme.zendesk.com/agent/organizations/7"
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/organizations/7.json")


def test_session_is_a_requests_session():
    # Each MCP tool call runs in its own fresh subprocess (see
    # mcp_adapter._execute_mcp_call), so this session is never actually
    # reused *across* calls in production -- it only saves a connection
    # within the retry-on-429 path of a single call. This just confirms
    # `_request()` has something to call `.request()` on, not cross-call
    # pooling (which doesn't happen here).
    assert isinstance(zendesk._session, requests.Session)


def test_session_disables_trust_env():
    # requests.Session()'s default trust_env=True falls back to the OS's
    # own proxy configuration once no HTTP(S)_PROXY env var is set, which
    # would let an ambient/OS-native proxy silently perform DNS resolution
    # for the real connection -- bypassing _base_url()'s private-network
    # check of the addresses this process itself resolved. This is the
    # actual behavioral assertion; isinstance(_session, requests.Session)
    # alone (the previous version of this test) doesn't verify it.
    assert zendesk._session.trust_env is False


def test_request_passes_empty_proxies_when_none_trusted(monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk._request("GET", "/tickets.json")

    assert mock_request.call_args.kwargs["proxies"] == {}


def test_request_raises_when_ambient_proxy_is_not_trusted(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    with pytest.raises(ValueError, match="not marked as trusted"):
        zendesk._request("GET", "/tickets.json")
    mock_request.assert_not_called()


def test_request_forwards_explicitly_trusted_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk._request("GET", "/tickets.json")

    assert mock_request.call_args.kwargs["proxies"] == {
        "http": "http://proxy.internal:8080",
        "https": "http://proxy.internal:8080",
    }


@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        (
            "zendesk_create_ticket",
            {
                "subject": "Help",
                "comment": "Something broke",
                "requester_email": None,
                "priority": None,
                "tags": None,
            },
        ),
        (
            "zendesk_update_ticket",
            # status/priority both None exercises their null-acceptance;
            # tags=[] (an explicit, meaningful empty list) is the one field
            # that satisfies "at least one of status/priority/tags must be
            # provided" without reintroducing a truthy status/priority.
            {
                "ticket_id": "1",
                "status": None,
                "priority": None,
                "tags": [],
                "assignee_id": None,
                "group_id": None,
            },
        ),
        ("zendesk_list_tickets", {"limit": 25, "after_cursor": None}),
        ("zendesk_list_ticket_comments", {"ticket_id": "1", "after_cursor": None}),
        ("zendesk_list_users", {"after_cursor": None}),
        ("zendesk_list_organizations", {"after_cursor": None}),
        ("zendesk_list_macros", {"after_cursor": None}),
    ],
)
async def test_optional_params_accept_explicit_none_through_tool_schema(
    monkeypatch, tool_name, arguments
):
    """Regression test for a real MCP-specific failure mode: a `str = ""`
    (non-Optional) tool parameter's generated JSON schema has no "null"
    variant, so an MCP client explicitly passing null for an unused
    optional argument (common LLM behavior) gets a hard Pydantic
    ValidationError before this module's own code ever runs -- confirmed
    by reproducing it directly against FastMCP's schema validation. Every
    optional parameter listed here must accept an explicit None through
    zendesk.mcp.call_tool (not just a plain Python call, which bypasses
    Pydantic's schema validation entirely and would not have caught the
    original bug)."""
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "ticket": {"id": 1},
                    "tickets": [],
                    "comments": [],
                    "users": [],
                    "organizations": [],
                    "macros": [],
                    "meta": {"has_more": False},
                }
            )
        ),
    )

    content, structured = await zendesk.mcp.call_tool(tool_name, arguments)

    result = json.loads(structured["result"])
    assert result["status"] == "success"


@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("zendesk_get_ticket", {"ticket_id": 1}),
        ("zendesk_delete_ticket", {"ticket_id": 1}),
        ("zendesk_list_ticket_comments", {"ticket_id": 1, "after_cursor": None}),
        ("zendesk_reply_to_ticket", {"ticket_id": 1, "body": "hi"}),
        ("zendesk_add_internal_note", {"ticket_id": 1, "body": "hi"}),
        ("zendesk_get_user", {"user_id": 1}),
        ("zendesk_get_organization", {"organization_id": 1}),
        ("zendesk_get_article", {"article_id": 1}),
        ("zendesk_get_macro", {"macro_id": 1}),
        ("zendesk_list_side_conversations", {"ticket_id": 1}),
        (
            "zendesk_get_side_conversation",
            {"ticket_id": 1, "side_conversation_id": "5c62"},
        ),
        ("zendesk_apply_macro", {"ticket_id": 1, "macro_id": 1}),
        (
            "zendesk_update_ticket",
            {
                "ticket_id": 1,
                "status": None,
                "priority": None,
                "tags": [],
                "assignee_id": None,
                "group_id": None,
            },
        ),
    ],
)
async def test_id_params_accept_a_bare_int_through_tool_schema(
    monkeypatch, tool_name, arguments
):
    """Every id-based tool's own responses echo Zendesk's ids as JSON
    integers (e.g. zendesk_list_tickets' `tickets[].id`), so an LLM that
    just listed tickets and now wants to act on one is the most natural
    caller to pass that same int straight back in -- but ticket_id/user_id/
    organization_id were typed plain `str`, whose generated schema has no
    "integer" variant, so FastMCP's Pydantic validation rejected an int
    before this module's own code (which already coerces via
    _resolve_path_id's str(value)) ever ran. Confirmed by reproducing the
    rejection directly against real int args through zendesk.mcp.call_tool
    (not a plain Python call, which bypasses schema validation entirely)."""
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "ticket": {"id": 1},
                    "user": {"id": 1},
                    "organization": {"id": 1},
                    "article": {"id": 1},
                    "macro_attachments": [],
                    "macro": {
                        "id": 1,
                        "actions": [{"field": "status", "value": "open"}],
                    },
                    "result": {
                        "ticket": {"id": 1, "status": "open", "updated_at": "t"}
                    },
                    "side_conversations": [],
                    "side_conversation": {"id": "5c62"},
                    "events": [],
                    "comments": [],
                    "audit": {"events": []},
                    "meta": {"has_more": False},
                }
            )
        ),
    )

    content, structured = await zendesk.mcp.call_tool(tool_name, arguments)

    result = json.loads(structured["result"])
    assert result["status"] == "success"


_ALL_TOOL_CALLS = [
    ("zendesk_search", {"query": "status:open"}),
    ("zendesk_list_tickets", {}),
    ("zendesk_get_ticket", {"ticket_id": 1}),
    ("zendesk_create_ticket", {"subject": "s", "comment": "c"}),
    ("zendesk_update_ticket", {"ticket_id": 1, "status": "open"}),
    ("zendesk_delete_ticket", {"ticket_id": 1}),
    ("zendesk_list_ticket_comments", {"ticket_id": 1}),
    ("zendesk_reply_to_ticket", {"ticket_id": 1, "body": "hi"}),
    ("zendesk_add_internal_note", {"ticket_id": 1, "body": "hi"}),
    ("zendesk_list_users", {}),
    ("zendesk_get_user", {"user_id": 1}),
    ("zendesk_search_users", {"query": "alice"}),
    ("zendesk_list_organizations", {}),
    ("zendesk_get_organization", {"organization_id": 1}),
    ("zendesk_search_articles", {"query": "password"}),
    ("zendesk_get_article", {"article_id": 1}),
    ("zendesk_list_macros", {}),
    ("zendesk_get_macro", {"macro_id": 1}),
    ("zendesk_list_side_conversations", {"ticket_id": 1}),
    (
        "zendesk_get_side_conversation",
        {"ticket_id": 1, "side_conversation_id": "5c62"},
    ),
    ("zendesk_apply_macro", {"ticket_id": 1, "macro_id": 1}),
]


async def test_tool_call_table_covers_every_registered_tool():
    registered = {tool.name for tool in await zendesk.mcp.list_tools()}

    assert registered == {name for name, _arguments in _ALL_TOOL_CALLS}


@pytest.mark.parametrize("tool_name, arguments", _ALL_TOOL_CALLS)
async def test_every_tool_sends_the_access_token_as_bearer(
    monkeypatch, tool_name, arguments
):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "ticket": {"id": 1},
                "tickets": [],
                "results": [],
                "comments": [],
                "users": [],
                "user": {"id": 1},
                "organizations": [],
                "organization": {"id": 1},
                "article": {"id": 1},
                "macros": [],
                "macro_attachments": [],
                "macro": {
                    "id": 1,
                    "actions": [{"field": "status", "value": "open"}],
                },
                "result": {"ticket": {"id": 1, "status": "open", "updated_at": "t"}},
                "side_conversations": [],
                "side_conversation": {"id": "5c62"},
                "events": [],
                "audit": {"events": []},
                "meta": {"has_more": False},
            }
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    _content, structured = await zendesk.mcp.call_tool(tool_name, arguments)

    assert json.loads(structured["result"])["status"] == "success"
    assert mock_request.call_count >= 1
    for call in mock_request.call_args_list:
        assert call.kwargs["headers"] == {"Authorization": "Bearer test-token"}
        assert "auth" not in call.kwargs


def test_zendesk_app_registry_uses_account_oauth():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    zendesk_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "zendesk"
    )
    assert zendesk_app["provider_name"] == "zendesk"
    assert zendesk_app["category"] == "Support"
    assert zendesk_app["transport"] == "oauth"
    assert zendesk_app["launch_config"]["env_mapping"] == {
        "ZENDESK_ACCESS_TOKEN": "access_token",
        "ZENDESK_SUBDOMAIN": "instance_url",
    }
    assert "required_env" not in zendesk_app["launch_config"]
    assert zendesk_app["launch_config"]["builtin_provenance"] == {
        "registry": "xagent",
        "app_id": "zendesk",
        "version": 1,
    }
    # The release gate this connector's unverified write tools ship behind.
    assert zendesk_app["is_visible_in_connector"] is False


def test_search_articles_sends_query_and_returns_article_summaries(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "results": [
                    {
                        "id": 360001,
                        "title": "Reset your password",
                        "html_url": "https://acme.zendesk.com/hc/en-us/articles/360001",
                        "locale": "en-us",
                        "section_id": 77,
                        "snippet": "To <em>reset</em> your password...",
                        "body": "<p>" + "x" * 500 + "</p>",
                        "updated_at": "2026-10-01T00:00:00Z",
                    }
                ],
                "count": 1,
                "next_page": None,
            }
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_search_articles(" reset ", limit=10, page=2))

    assert result["status"] == "success"
    assert result["count"] == 1
    assert result["has_more"] is False
    assert result["results"] == [
        {
            "id": 360001,
            "title": "Reset your password",
            "html_url": "https://acme.zendesk.com/hc/en-us/articles/360001",
            "locale": "en-us",
            "section_id": 77,
            "snippet": "To <em>reset</em> your password...",
            "updated_at": "2026-10-01T00:00:00Z",
        }
    ]
    assert mock_request.call_args.kwargs["method"] == "GET"
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.zendesk.com/api/v2/help_center/articles/search.json"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "query": "reset",
        "per_page": 10,
        "page": 2,
    }


def test_search_articles_reports_more_pages_from_next_page(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "results": [{"id": 1, "title": "A"}],
                    "count": 40,
                    "next_page": "https://acme.zendesk.com/api/v2/help_center/articles/search.json?page=2",
                }
            )
        ),
    )

    result = json.loads(zendesk.zendesk_search_articles("a", limit=1))

    assert result["has_more"] is True


@pytest.mark.parametrize("query", ["   ", None])
def test_search_articles_requires_non_blank_query(monkeypatch, query):
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_search_articles(query))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_search_articles_returns_empty_past_result_window_without_calling_zendesk(
    monkeypatch,
):
    # Help Center search serves at most 1,000 results: page 11 at 100 per
    # page starts at result 1,001.
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_search_articles("a", limit=100, page=11))

    assert result == {
        "status": "success",
        "results": [],
        "count": None,
        "has_more": False,
        "truncated": False,
    }
    mock_request.assert_not_called()


def test_search_articles_past_result_window_respects_the_output_cap(monkeypatch):
    # The past-window early return must still go through the output cap: a
    # configured cap smaller than the fixed success envelope may not be
    # bypassed.
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 60)
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    raw = zendesk.zendesk_search_articles("a", limit=100, page=11)

    assert len(raw) <= 60
    assert json.loads(raw)["status"] == "error"
    mock_request.assert_not_called()


def test_search_articles_surfaces_a_help_center_error(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"error": "RecordNotFound", "description": "Not found"},
                status_code=404,
            )
        ),
    )

    result = json.loads(zendesk.zendesk_search_articles("a"))

    assert result["status"] == "error"
    assert "HTTP 404" in result["message"]


def test_get_article_returns_detail_with_body(monkeypatch):
    article = {
        "id": 360001,
        "title": "Reset your password",
        "html_url": "https://acme.zendesk.com/hc/en-us/articles/360001",
        "locale": "en-us",
        "section_id": 77,
        "author_id": 9,
        "draft": False,
        "label_names": ["account"],
        "body": "<p>Open <b>Settings</b>.</p>",
        "created_at": "2026-09-01T00:00:00Z",
        "updated_at": "2026-10-01T00:00:00Z",
        "vote_sum": 3,
    }
    mock_request = Mock(return_value=MockResponse(json_data={"article": article}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_get_article(360001))

    assert result == {
        "status": "success",
        "article": {
            "id": 360001,
            "title": "Reset your password",
            "html_url": "https://acme.zendesk.com/hc/en-us/articles/360001",
            "locale": "en-us",
            "section_id": 77,
            "author_id": 9,
            "draft": False,
            "label_names": ["account"],
            "body": "<p>Open <b>Settings</b>.</p>",
            "created_at": "2026-09-01T00:00:00Z",
            "updated_at": "2026-10-01T00:00:00Z",
        },
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.zendesk.com/api/v2/help_center/articles/360001.json"
    )


@pytest.mark.parametrize(
    "article_url",
    [
        "https://acme.zendesk.com/hc/en-us/articles/360001-Reset-your-password",
        "https://acme.zendesk.com/knowledge/articles/360001/en-us?brand_id=1",
    ],
)
def test_get_article_accepts_a_help_center_or_knowledge_url(monkeypatch, article_url):
    mock_request = Mock(
        return_value=MockResponse(json_data={"article": {"id": 360001}})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_get_article(article_url))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/help_center/articles/360001.json"
    )


def test_get_article_shrinks_an_oversized_body_before_erroring(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"article": {"id": 1, "title": "T", "body": "x" * 5000}}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 1000)

    raw = zendesk.zendesk_get_article(1)
    result = json.loads(raw)

    assert len(raw) <= 1000
    assert result["status"] == "success"
    assert result["article"]["title"] == "T"
    assert result["article"]["body_truncated"] is True
    assert 0 < len(result["article"]["body"]) < 5000


def test_get_article_surfaces_a_not_found_error(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"error": "RecordNotFound", "description": "Not found"},
                status_code=404,
            )
        ),
    )

    result = json.loads(zendesk.zendesk_get_article(1))

    assert result["status"] == "error"
    assert "HTTP 404" in result["message"]


def test_list_macros_pages_active_macros_by_cursor(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "macros": [
                    {
                        "id": 5,
                        "title": "Close and redirect",
                        "description": "Redirect to the Help Center",
                        "active": True,
                        "actions": [{"field": "status", "value": "solved"}],
                        "updated_at": "2026-10-01T00:00:00Z",
                    }
                ],
                "meta": {"has_more": True, "after_cursor": "cur2"},
            }
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_list_macros(limit=1, after_cursor="cur1"))

    assert result == {
        "status": "success",
        "macros": [
            {
                "id": 5,
                "title": "Close and redirect",
                "description": "Redirect to the Help Center",
                "updated_at": "2026-10-01T00:00:00Z",
            }
        ],
        "has_more": True,
        "after_cursor": "cur2",
        "truncated": False,
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.zendesk.com/api/v2/macros/active.json"
    )
    assert mock_request.call_args.kwargs["params"] == {
        "page[size]": 1,
        "page[after]": "cur1",
        "include_boundary_indicators": "true",
    }


def test_list_macros_surfaces_a_permission_error(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"error": "Forbidden", "description": "Forbidden"},
                status_code=403,
            )
        ),
    )

    result = json.loads(zendesk.zendesk_list_macros())

    assert result["status"] == "error"
    assert "HTTP 403" in result["message"]


def test_get_macro_returns_its_actions_and_restriction(monkeypatch):
    macro = {
        "id": 5,
        "title": "Close and redirect",
        "description": "Redirect to the Help Center",
        "active": True,
        "actions": [
            {"field": "status", "value": "solved"},
            {"field": "comment_value_html", "value": "<p>See our guide.</p>"},
        ],
        "restriction": {"type": "Group", "id": 7, "ids": [7]},
        "created_at": "2026-09-01T00:00:00Z",
        "updated_at": "2026-10-01T00:00:00Z",
        "position": 3,
    }
    mock_request = Mock(return_value=MockResponse(json_data={"macro": macro}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_get_macro(5))

    assert result == {
        "status": "success",
        "macro": {
            "id": 5,
            "title": "Close and redirect",
            "description": "Redirect to the Help Center",
            "updated_at": "2026-10-01T00:00:00Z",
            "active": True,
            "actions": [
                {"field": "status", "value": "solved"},
                {"field": "comment_value_html", "value": "<p>See our guide.</p>"},
            ],
            "restriction": {"type": "Group", "id": 7, "ids": [7]},
            "created_at": "2026-09-01T00:00:00Z",
        },
    }
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.zendesk.com/api/v2/macros/5.json"
    )


def test_get_macro_accepts_an_admin_center_url(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"macro": {"id": 5}}))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_get_macro(
            "https://acme.zendesk.com/admin/workspaces/agent-workspace/macros/5"
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("/macros/5.json")


def test_get_macro_errors_rather_than_dropping_actions_when_too_large(monkeypatch):
    # A macro's actions are what it does; returning only some of them
    # would misdescribe it, so an oversized macro is an error, not a cut.
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "macro": {
                        "id": 5,
                        "actions": [{"field": "comment_value", "value": "x" * 5000}],
                    }
                }
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 1000)

    raw = zendesk.zendesk_get_macro(5)

    assert len(raw) <= 1000
    assert json.loads(raw)["status"] == "error"


def test_search_articles_forces_has_more_when_output_truncated(monkeypatch):
    big_results = [{"id": i, "title": "x" * 1000} for i in range(20)]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"results": big_results, "count": 20, "next_page": None}
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 3000)

    raw = zendesk.zendesk_search_articles("x", limit=20)
    result = json.loads(raw)

    assert len(raw) <= 3000
    assert result["truncated"] is True
    assert result["has_more"] is True
    assert 0 < len(result["results"]) < 20


def test_get_macro_surfaces_a_not_found_error(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"error": "RecordNotFound", "description": "Not found"},
                status_code=404,
            )
        ),
    )

    result = json.loads(zendesk.zendesk_get_macro(5))

    assert result["status"] == "error"
    assert "HTTP 404" in result["message"]


_SIDE_CONVERSATION = {
    "id": "5c623337-d419-11f1-8809-7698c7f03fad",
    "ticket_id": 1,
    "subject": "Shipping delay",
    "state": "open",
    "preview_text": "Can you confirm the new ship date?",
    "participants": [
        {"user_id": 7, "name": "Agent", "email": "agent@acme.com"},
        {"user_id": 8, "name": "Vendor", "email": "vendor@example.com"},
    ],
    "created_at": "2026-10-30T04:21:21.440Z",
    "updated_at": "2026-10-30T04:21:22.661Z",
    "message_added_at": "2026-10-30T04:21:21.440Z",
    "state_updated_at": "2026-10-30T04:21:21.440Z",
    "external_ids": {},
    "url": "https://acme.zendesk.com/api/v2/tickets/1/side_conversations/5c62",
}

_SIDE_CONVERSATION_SUMMARY = {
    "id": "5c623337-d419-11f1-8809-7698c7f03fad",
    "subject": "Shipping delay",
    "state": "open",
    "preview_text": "Can you confirm the new ship date?",
    "participants": [
        {"user_id": 7, "name": "Agent", "email": "agent@acme.com"},
        {"user_id": 8, "name": "Vendor", "email": "vendor@example.com"},
    ],
    "created_at": "2026-10-30T04:21:21.440Z",
    "updated_at": "2026-10-30T04:21:22.661Z",
    "message_added_at": "2026-10-30T04:21:21.440Z",
}


def test_list_side_conversations_pages_by_page_number(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "side_conversations": [_SIDE_CONVERSATION],
                "count": 3,
                "next_page": "https://acme.zendesk.com/api/v2/tickets/1/side_conversations.json?page=3",
                "previous_page": None,
            }
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_list_side_conversations(1, limit=1, page=2))

    assert result == {
        "status": "success",
        "side_conversations": [_SIDE_CONVERSATION_SUMMARY],
        "count": 3,
        "has_more": True,
        "truncated": False,
    }
    assert mock_request.call_args.kwargs["method"] == "GET"
    assert mock_request.call_args.kwargs["url"] == (
        "https://acme.zendesk.com/api/v2/tickets/1/side_conversations.json"
    )
    assert mock_request.call_args.kwargs["params"] == {"per_page": 1, "page": 2}


def test_list_side_conversations_accepts_a_ticket_url_and_clamps_page(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"side_conversations": [], "count": 0, "next_page": None}
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_list_side_conversations(
            "https://acme.zendesk.com/agent/tickets/42", page=0
        )
    )

    assert result["status"] == "success"
    assert result["has_more"] is False
    assert mock_request.call_args.kwargs["url"].endswith(
        "/tickets/42/side_conversations.json"
    )
    assert mock_request.call_args.kwargs["params"]["page"] == 1


def test_list_side_conversations_surfaces_a_permission_error(monkeypatch):
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"error": "Forbidden", "description": "Forbidden"},
                status_code=403,
            )
        ),
    )

    result = json.loads(zendesk.zendesk_list_side_conversations(999999))

    assert result["status"] == "error"
    assert "HTTP 403" in result["message"]


def _side_conversation_event(event_id, body):
    return {
        "id": event_id,
        "side_conversation_id": "5c623337-d419-11f1-8809-7698c7f03fad",
        "ticket_id": 1,
        "type": "create",
        "via": "support",
        "created_at": "2026-10-30T04:21:21.440Z",
        "actor": {"user_id": 7, "name": "Agent", "email": "agent@acme.com"},
        "updates": {},
        "message": {
            "subject": "Shipping delay",
            "preview_text": body[:20],
            "from": {"user_id": 7, "name": "Agent", "email": "agent@acme.com"},
            "to": [{"user_id": 8, "name": "Vendor", "email": "vendor@example.com"}],
            "body": body,
            "html_body": f"<div>{body}</div>",
            "attachments": [],
            "external_ids": {},
        },
    }


def _side_conversation_responses(events, next_page=None):
    """Zendesk's two responses, in request order: the side conversation,
    then one page of its events."""
    return [
        MockResponse(json_data={"side_conversation": _SIDE_CONVERSATION}),
        MockResponse(
            json_data={
                "events": events,
                "count": len(events),
                "next_page": next_page,
                "previous_page": None,
            }
        ),
    ]


def test_get_side_conversation_returns_it_with_a_page_of_messages(monkeypatch):
    mock_request = Mock(
        side_effect=_side_conversation_responses(
            [_side_conversation_event("e1", "Can you confirm the new ship date?")],
            next_page="https://acme.zendesk.com/api/v2/next",
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_get_side_conversation(
            1, "5c623337-d419-11f1-8809-7698c7f03fad", limit=1, page=2
        )
    )

    assert result == {
        "status": "success",
        "events": [
            {
                "id": "e1",
                "type": "create",
                "created_at": "2026-10-30T04:21:21.440Z",
                "actor": {"user_id": 7, "name": "Agent", "email": "agent@acme.com"},
                "updates": {},
                "message": {
                    "subject": "Shipping delay",
                    "from": {"user_id": 7, "name": "Agent", "email": "agent@acme.com"},
                    "to": [
                        {"user_id": 8, "name": "Vendor", "email": "vendor@example.com"}
                    ],
                    "body": "Can you confirm the new ship date?",
                    "body_is_html": False,
                    "attachments": [],
                },
            }
        ],
        "side_conversation": {
            **_SIDE_CONVERSATION_SUMMARY,
            "state_updated_at": "2026-10-30T04:21:21.440Z",
        },
        "count": 1,
        "has_more": True,
        "truncated": False,
    }
    show_call, events_call = mock_request.call_args_list
    base = "https://acme.zendesk.com/api/v2/tickets/1/side_conversations"
    assert show_call.kwargs["url"] == (
        f"{base}/5c623337-d419-11f1-8809-7698c7f03fad.json"
    )
    assert events_call.kwargs["url"] == (
        f"{base}/5c623337-d419-11f1-8809-7698c7f03fad/events.json"
    )
    assert events_call.kwargs["params"] == {"per_page": 1, "page": 2}


def test_get_side_conversation_reports_an_event_without_a_message(monkeypatch):
    state_change = {
        "id": "e2",
        "type": "update",
        "created_at": "2026-10-30T05:00:00Z",
        "actor": {"user_id": 7, "name": "Agent", "email": "agent@acme.com"},
        "updates": {"state": "closed"},
    }
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(side_effect=_side_conversation_responses([state_change])),
    )

    result = json.loads(zendesk.zendesk_get_side_conversation(1, "5c62"))

    assert result["events"] == [
        {
            "id": "e2",
            "type": "update",
            "created_at": "2026-10-30T05:00:00Z",
            "actor": {"user_id": 7, "name": "Agent", "email": "agent@acme.com"},
            "updates": {"state": "closed"},
            "message": None,
        }
    ]


@pytest.mark.parametrize("side_conversation_id", ["", "   ", None, "..", " 5c62 "])
def test_get_side_conversation_rejects_an_unusable_id_without_calling_zendesk(
    monkeypatch, side_conversation_id
):
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_get_side_conversation(1, side_conversation_id))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_get_side_conversation_encodes_the_id_into_one_path_segment(monkeypatch):
    mock_request = Mock(side_effect=_side_conversation_responses([]))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_get_side_conversation(1, "abc/../../users")

    assert mock_request.call_args_list[0].kwargs["url"] == (
        "https://acme.zendesk.com/api/v2/tickets/1/side_conversations/"
        "abc%2F..%2F..%2Fusers.json"
    )


def test_get_side_conversation_trims_events_but_keeps_the_conversation(
    monkeypatch,
):
    events = [_side_conversation_event(f"e{i}", "x" * 1000) for i in range(10)]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(side_effect=_side_conversation_responses(events)),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 4000)

    raw = zendesk.zendesk_get_side_conversation(1, "5c62", limit=10)
    result = json.loads(raw)

    assert len(raw) <= 4000
    assert result["status"] == "success"
    assert result["side_conversation"]["subject"] == "Shipping delay"
    assert result["truncated"] is True
    assert result["has_more"] is True
    assert 0 < len(result["events"]) < 10
    kept = [e["id"] for e in result["events"]]
    assert kept == [f"e{i}" for i in range(len(kept))]
    assert result["count"] == 10
    assert result["side_conversation"]["id"] == "5c623337-d419-11f1-8809-7698c7f03fad"


def test_get_side_conversation_clamps_limit_to_100(monkeypatch):
    mock_request = Mock(side_effect=_side_conversation_responses([]))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_get_side_conversation(1, "5c62", limit=500)

    _, events_call = mock_request.call_args_list
    assert events_call.kwargs["params"] == {"per_page": 100, "page": 1}


def test_list_side_conversations_trims_to_fit_and_keeps_count(monkeypatch):
    side_conversations = [
        {**_SIDE_CONVERSATION, "id": f"sc{i}", "preview_text": "x" * 1000}
        for i in range(10)
    ]
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "side_conversations": side_conversations,
                    "count": 10,
                    "next_page": None,
                }
            )
        ),
    )
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 4000)

    raw = zendesk.zendesk_list_side_conversations(1, limit=10)
    result = json.loads(raw)

    assert len(raw) <= 4000
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["has_more"] is True
    assert result["count"] == 10
    kept = [sc["id"] for sc in result["side_conversations"]]
    assert kept == [f"sc{i}" for i in range(len(kept))]
    assert 0 < len(kept) < 10


def test_get_side_conversation_surfaces_a_not_found_error(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"error": "RecordNotFound", "description": "Not found"},
            status_code=404,
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_get_side_conversation(1, "missing"))

    assert result["status"] == "error"
    assert "HTTP 404" in result["message"]
    assert mock_request.call_count == 1


def test_get_side_conversation_falls_back_to_the_html_body(monkeypatch):
    event = _side_conversation_event("e1", "")
    event["message"]["html_body"] = "<div>Only HTML</div>"
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(side_effect=_side_conversation_responses([event])),
    )

    result = json.loads(zendesk.zendesk_get_side_conversation(1, "5c62"))

    assert result["events"][0]["message"]["body"] == "<div>Only HTML</div>"
    assert result["events"][0]["message"]["body_is_html"] is True


def test_get_side_conversation_accepts_a_ticket_url(monkeypatch):
    mock_request = Mock(side_effect=_side_conversation_responses([]))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_get_side_conversation(
            "https://acme.zendesk.com/agent/tickets/42", "5c62"
        )
    )

    assert result["status"] == "success"
    assert (
        mock_request.call_args_list[0]
        .kwargs["url"]
        .endswith("/tickets/42/side_conversations/5c62.json")
    )


def test_get_side_conversation_surfaces_an_events_error(monkeypatch):
    mock_request = Mock(
        side_effect=[
            MockResponse(json_data={"side_conversation": _SIDE_CONVERSATION}),
            MockResponse(
                json_data={"error": "Forbidden", "description": "Forbidden"},
                status_code=403,
            ),
        ]
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_get_side_conversation(1, "5c62"))

    assert result["status"] == "error"
    assert "HTTP 403" in result["message"]
    assert mock_request.call_count == 2


def _macro(
    actions, macro_id=5, title="Close and redirect", updated_at="2026-10-29T00:00:00Z"
):
    return MockResponse(
        json_data={
            "macro": {
                "id": macro_id,
                "title": title,
                "actions": actions,
                "updated_at": updated_at,
            }
        }
    )


def _macro_preview(_drop=(), **ticket):
    """GET /tickets/{id}/macros/{id}/apply: the whole ticket as it would be
    after the macro, placeholders already rendered."""
    base = {
        "id": 1,
        "subject": "Help",
        "status": "pending",
        "priority": "normal",
        "tags": ["vip", "probe_macro_tag"],
        "updated_at": "2026-10-30T04:21:22Z",
        "collaborator_ids": [],
        "follower_ids": [],
        "comment": {"body": "Hi Jane", "html_body": "<p>Hi Jane</p>"},
    }
    merged = {**base, **ticket}
    for key in _drop:
        del merged[key]
    return MockResponse(json_data={"result": {"ticket": merged}})


_STAMP = "2026-10-30T04:21:22Z"
# A save that is refused before the token is compared never reads it.
_UNUSED_TOKEN = "unused"


_PROBE_MACRO_ACTIONS = [
    {
        "field": "comment_value_html",
        "value": "<p>Hi {{ticket.requester.first_name}}</p>",
    },
    {"field": "current_tags", "value": "probe_macro_tag"},
    {"field": "status", "value": "pending"},
]


def _no_attachments():
    """GET /macros/{id}/attachments: the macro has no attached files."""
    return MockResponse(json_data={"macro_attachments": []})


def _macro_reads(actions=_PROBE_MACRO_ACTIONS, preview=None, **macro_kwargs):
    """Every GET a preview makes, in order: the macro, its attachments, the
    ticket-scoped render, and the macro again."""
    return [
        _macro(actions, **macro_kwargs),
        _no_attachments(),
        _macro_preview() if preview is None else preview,
        _macro(actions, **macro_kwargs),
    ]


def _preview_token(monkeypatch, actions=_PROBE_MACRO_ACTIONS, **preview_ticket):
    """The preview_token a dry run returns for macro 5 (updated 2026-10-29)
    on the ticket, so a save test approves exactly what it then saves."""
    monkeypatch.setattr(
        zendesk._session,
        "request",
        Mock(
            side_effect=_macro_reads(actions, preview=_macro_preview(**preview_ticket))
        ),
    )
    result = json.loads(zendesk.zendesk_apply_macro(preview_ticket.get("id", 1), 5))
    assert result["status"] == "success"
    return result["preview_token"]


def test_apply_macro_refuses_an_action_outside_the_allowlist_before_saving(
    monkeypatch,
):
    mock_request = Mock(
        side_effect=[
            _macro(
                [
                    {"field": "status", "value": "pending"},
                    {"field": "side_conversation", "value": ["Hi", "x@y.com"]},
                    {"field": "cc", "value": "42"},
                ]
            )
        ]
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=_UNUSED_TOKEN)
    )

    assert result["status"] == "error"
    assert "side_conversation" in result["message"]
    assert "cc" in result["message"]
    assert mock_request.call_count == 1
    assert mock_request.call_args.kwargs["url"].endswith("/macros/5.json")


def test_apply_macro_previews_without_saving_by_default(monkeypatch):
    mock_request = Mock(side_effect=_macro_reads())
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 5))

    token = result.pop("preview_token")
    assert token.startswith("1|2026-10-30T04:21:22Z|5|2026-10-29T00:00:00Z|")
    assert re.fullmatch(r"[0-9a-f]{32}", token.rsplit("|", 1)[1])
    assert result == {
        "status": "success",
        "dry_run": True,
        "ticket_id": 1,
        "macro": {"id": 5, "title": "Close and redirect"},
        "changes": {
            "status": "pending",
            "tags": ["vip", "probe_macro_tag"],
            "comment": {
                "body": "<p>Hi Jane</p>",
                "body_is_html": True,
                "public": True,
            },
        },
    }
    assert [call.kwargs["method"] for call in mock_request.call_args_list] == [
        "GET",
        "GET",
        "GET",
        "GET",
    ]
    assert (
        mock_request.call_args_list[2]
        .kwargs["url"]
        .endswith("/tickets/1/macros/5/apply.json")
    )


def test_apply_macro_saves_the_rendered_changes_with_a_safe_update(monkeypatch):
    saved = MockResponse(
        json_data={
            "ticket": {
                "id": 1,
                "status": "pending",
                "tags": ["vip", "probe_macro_tag"],
            },
            "audit": {
                "events": [
                    {
                        "type": "Comment",
                        "id": 900,
                        "author_id": 7,
                        "body": "Hi Jane",
                        "public": True,
                    }
                ]
            },
        }
    )
    token = _preview_token(monkeypatch)
    mock_request = Mock(side_effect=[*_macro_reads(), saved])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )

    put = mock_request.call_args_list[4]
    assert put.kwargs["method"] == "PUT"
    assert put.kwargs["url"].endswith("/tickets/1.json")
    # Only the fields the macro's actions touch: the preview's empty
    # collaborator_ids/follower_ids must not be forwarded, or they would
    # clear CCs and followers. Tags are the preview's final set, because a
    # single-ticket update ignores additional_tags/remove_tags.
    assert put.kwargs["json"] == {
        "ticket": {
            "status": "pending",
            "tags": ["vip", "probe_macro_tag"],
            "comment": {"html_body": "<p>Hi Jane</p>", "public": True},
            "safe_update": True,
            "updated_stamp": "2026-10-30T04:21:22Z",
        }
    }
    assert result["status"] == "success"
    assert result["dry_run"] is False
    assert result["ticket"]["status"] == "pending"
    assert result["comment"]["id"] == 900
    assert result["comment"]["public"] is True
    assert "preview_token" not in result


@pytest.mark.parametrize("mode", [False, "false", " FALSE "])
def test_apply_macro_keeps_an_internal_comment_internal(monkeypatch, mode):
    actions = [
        *_PROBE_MACRO_ACTIONS,
        {"field": "comment_mode_is_public", "value": mode},
    ]
    token = _preview_token(monkeypatch, actions)
    saved = MockResponse(json_data={"ticket": {"id": 1}, "audit": {"events": []}})
    mock_request = Mock(side_effect=[*_macro_reads(actions), saved])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"]["ticket"]["comment"]["public"] is False
    assert result["changes"]["comment"]["public"] is False


@pytest.mark.parametrize("mode", [True, "true", " TRUE "])
def test_apply_macro_posts_a_public_comment_when_the_macro_says_so(monkeypatch, mode):
    actions = [
        *_PROBE_MACRO_ACTIONS,
        {"field": "comment_mode_is_public", "value": mode},
    ]
    token = _preview_token(monkeypatch, actions)
    saved = MockResponse(json_data={"ticket": {"id": 1}, "audit": {"events": []}})
    mock_request = Mock(side_effect=[*_macro_reads(actions), saved])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"]["ticket"]["comment"]["public"] is True
    assert result["changes"]["comment"]["public"] is True


@pytest.mark.parametrize("modes", [[True, False], ["true", "maybe"]])
def test_apply_macro_refuses_a_comment_mode_set_more_than_once(monkeypatch, modes):
    actions = [
        *_PROBE_MACRO_ACTIONS,
        *({"field": "comment_mode_is_public", "value": mode} for mode in modes),
    ]
    mock_request = Mock(side_effect=[_macro(actions)])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=_UNUSED_TOKEN)
    )

    assert result["status"] == "error"
    assert "more than once" in result["message"]
    assert mock_request.call_count == 1


def test_apply_macro_refuses_an_unrecognized_comment_mode(monkeypatch):
    actions = [
        *_PROBE_MACRO_ACTIONS,
        {"field": "comment_mode_is_public", "value": "maybe"},
    ]
    mock_request = Mock(side_effect=[_macro(actions)])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=_UNUSED_TOKEN)
    )

    assert result["status"] == "error"
    assert "comment_mode_is_public" in result["message"]
    assert mock_request.call_count == 1


def test_apply_macro_sends_the_plain_body_when_there_is_no_html(monkeypatch):
    actions = [{"field": "comment_value", "value": "Hello {{ticket.requester.name}}"}]
    plain_comment = {"body": "Hello Jane", "html_body": ""}
    token = _preview_token(monkeypatch, actions, comment=plain_comment)
    saved = MockResponse(json_data={"ticket": {"id": 1}, "audit": {"events": []}})
    mock_request = Mock(
        side_effect=[
            *_macro_reads(actions, preview=_macro_preview(comment=plain_comment)),
            saved,
        ]
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)

    assert mock_request.call_args.kwargs["json"]["ticket"] == {
        "comment": {"body": "Hello Jane", "public": True},
        "safe_update": True,
        "updated_stamp": "2026-10-30T04:21:22Z",
    }


def test_apply_macro_takes_resolved_field_values_from_the_preview(monkeypatch):
    # A macro can assign to "current_user"; the preview resolves it to an id.
    mock_request = Mock(
        side_effect=_macro_reads(
            [{"field": "assignee_id", "value": "current_user"}],
            preview=_macro_preview(assignee_id=77),
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 5))

    assert result["changes"] == {"assignee_id": 77}


def test_apply_macro_reports_a_safe_update_conflict(monkeypatch):
    conflict = MockResponse(
        json_data={
            "error": "UpdateConflict",
            "description": "Safe update timestamp prevented the update due to outdated ticket data.",
        },
        status_code=409,
    )
    token = _preview_token(monkeypatch)
    mock_request = Mock(side_effect=[*_macro_reads(), conflict])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )

    assert result["status"] == "error"
    assert "HTTP 409" in result["message"]
    assert "Safe update timestamp" in result["message"]


@pytest.mark.parametrize(
    "actions", [[], [{"field": "comment_mode_is_public", "value": "true"}]]
)
def test_apply_macro_refuses_a_macro_with_nothing_to_apply(monkeypatch, actions):
    mock_request = Mock(side_effect=[_macro(actions)])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=_UNUSED_TOKEN)
    )

    assert result["status"] == "error"
    assert "no actions to apply" in result["message"]
    assert mock_request.call_count == 1


def test_apply_macro_refuses_an_empty_rendered_comment(monkeypatch):
    mock_request = Mock(
        side_effect=_macro_reads(
            preview=_macro_preview(comment={"body": "", "html_body": ""})
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=_UNUSED_TOKEN)
    )

    assert result["status"] == "error"
    assert "empty comment" in result["message"]
    assert mock_request.call_count == 4


def test_apply_macro_accepts_ticket_and_macro_urls(monkeypatch):
    mock_request = Mock(side_effect=_macro_reads())
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_apply_macro(
        "https://acme.zendesk.com/agent/tickets/42",
        "https://acme.zendesk.com/admin/workspaces/agent-workspace/macros/5",
    )

    macro_call, attachments_call, preview_call, reread_call = (
        mock_request.call_args_list
    )
    assert macro_call.kwargs["url"].endswith("/macros/5.json")
    assert attachments_call.kwargs["url"].endswith("/macros/5/attachments.json")
    assert preview_call.kwargs["url"].endswith("/tickets/42/macros/5/apply.json")
    assert reread_call.kwargs["url"].endswith("/macros/5.json")


def test_apply_macro_gives_no_token_when_the_comment_is_too_long_to_show(
    monkeypatch,
):
    long_body = "<p>" + "x" * 5000 + "</p>"
    mock_request = Mock(
        side_effect=_macro_reads(
            preview=_macro_preview(comment={"body": long_body, "html_body": long_body})
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 1000)

    raw = zendesk.zendesk_apply_macro(1, 5)
    result = json.loads(raw)

    assert result["status"] == "error"
    assert "too large to show" in result["message"]
    assert "preview_token" not in raw


@pytest.mark.parametrize("token", [None, "", "   "])
def test_apply_macro_refuses_to_save_without_a_preview_token(monkeypatch, token):
    mock_request = Mock()
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )

    assert result["status"] == "error"
    assert "preview_token" in result["message"]
    mock_request.assert_not_called()


def test_apply_macro_refuses_to_save_when_the_ticket_changed_since_the_dry_run(
    monkeypatch,
):
    token = _preview_token(monkeypatch)
    mock_request = Mock(
        side_effect=_macro_reads(
            preview=_macro_preview(updated_at="2026-10-30T09:00:00Z")
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )

    assert result["status"] == "error"
    assert "changed since" in result["message"]
    assert [call.kwargs["method"] for call in mock_request.call_args_list] == [
        "GET",
        "GET",
        "GET",
        "GET",
    ]


def test_apply_macro_refuses_an_inactive_macro(monkeypatch):
    inactive = MockResponse(
        json_data={
            "macro": {
                "id": 5,
                "title": "Old",
                "active": False,
                "actions": _PROBE_MACRO_ACTIONS,
            }
        }
    )
    mock_request = Mock(side_effect=[inactive])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 5))

    assert result["status"] == "error"
    assert "inactive" in result["message"]
    assert mock_request.call_count == 1


@pytest.mark.parametrize("field", ["type", "custom_status_id"])
def test_apply_macro_refuses_fields_outside_the_acceptance_list(monkeypatch, field):
    mock_request = Mock(side_effect=[_macro([{"field": field, "value": "x"}])])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 5))

    assert result["status"] == "error"
    assert field in result["message"]


@pytest.mark.parametrize(
    "action, expected",
    [
        ({"field": "priority", "value": "high"}, {"priority": "high"}),
        ({"field": "group_id", "value": "7"}, {"group_id": 7}),
        ({"field": "set_tags", "value": "a b"}, {"tags": ["a", "b"]}),
        ({"field": "remove_tags", "value": "vip"}, {"tags": ["probe_macro_tag"]}),
    ],
)
def test_apply_macro_sends_each_allowlisted_field_from_the_preview(
    monkeypatch, action, expected
):
    preview_fields = {"priority": "high", "group_id": 7}
    if action["field"] == "set_tags":
        preview_fields["tags"] = ["a", "b"]
    if action["field"] == "remove_tags":
        preview_fields["tags"] = ["probe_macro_tag"]
    token = _preview_token(monkeypatch, [action], **preview_fields)
    saved = MockResponse(json_data={"ticket": {"id": 1}, "audit": {"events": []}})
    mock_request = Mock(
        side_effect=[
            *_macro_reads([action], preview=_macro_preview(**preview_fields)),
            saved,
        ]
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)

    assert mock_request.call_args.kwargs["json"]["ticket"] == {
        **expected,
        "safe_update": True,
        "updated_stamp": _STAMP,
    }


def test_apply_macro_refuses_a_tag_action_when_the_preview_has_no_tags(monkeypatch):
    mock_request = Mock(
        side_effect=_macro_reads(
            [{"field": "current_tags", "value": "x"}],
            preview=_macro_preview(tags=None),
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=_UNUSED_TOKEN)
    )

    assert result["status"] == "error"
    assert "no tag list" in result["message"]
    assert mock_request.call_count == 4


def test_apply_macro_surfaces_a_missing_macro(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={"error": "RecordNotFound", "description": "Not found"},
            status_code=404,
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 999))

    assert result["status"] == "error"
    assert "HTTP 404" in result["message"]
    assert mock_request.call_count == 1


def test_apply_macro_never_reports_a_saved_change_as_an_error(monkeypatch):
    # A saved macro must come back as success however small the output cap:
    # an error would invite a retry that posts the comment twice.
    many_tags = [f"tag_{i:03d}" for i in range(300)]
    saved = MockResponse(
        json_data={
            "ticket": {"id": 1, "subject": "s" * 200, "tags": many_tags},
            "audit": {"events": [{"type": "Comment", "id": 900, "public": True}]},
        }
    )
    token = _preview_token(monkeypatch, tags=many_tags)
    mock_request = Mock(
        side_effect=[*_macro_reads(preview=_macro_preview(tags=many_tags)), saved]
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 300)

    raw = zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    result = json.loads(raw)

    assert len(raw) <= 300
    assert result["status"] == "success"
    assert result["dry_run"] is False
    assert result["ticket_id"] == 1


def test_apply_macro_refuses_when_the_macro_renders_differently_at_save(
    monkeypatch,
):
    # The macro definition and the ticket stamp are unchanged, but the
    # placeholder renders differently (for example the requester's details
    # changed), so the rendered comment is not what the user approved.
    token = _preview_token(monkeypatch)
    mock_request = Mock(
        side_effect=_macro_reads(
            preview=_macro_preview(
                comment={"body": "Internal only", "html_body": "<p>Internal only</p>"}
            )
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )

    assert result["status"] == "error"
    assert "renders differently" in result["message"]
    assert [call.kwargs["method"] for call in mock_request.call_args_list] == [
        "GET",
        "GET",
        "GET",
        "GET",
    ]


@pytest.mark.parametrize(
    "tamper",
    [
        # No stamps at all.
        lambda token: _UNUSED_TOKEN,
        # Another ticket's token.
        lambda token: "9|" + token.split("|", 1)[1],
        # Another macro version.
        lambda token: token.replace("|2026-10-29T", "|2026-10-28T", 1),
    ],
)
def test_apply_macro_names_a_changed_ticket_or_macro_when_the_token_is_stale(
    monkeypatch, tamper
):
    token = tamper(_preview_token(monkeypatch))
    mock_request = Mock(side_effect=_macro_reads())
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )

    assert result["status"] == "error"
    assert "changed since the preview" in result["message"]
    assert "renders differently" not in result["message"]
    assert "PUT" not in [call.kwargs["method"] for call in mock_request.call_args_list]


@pytest.mark.parametrize("cap", [60, 40, 22])
def test_apply_macro_reports_a_committed_save_as_success_under_a_tiny_cap(
    monkeypatch, cap
):
    token = _preview_token(monkeypatch)
    saved = MockResponse(json_data={"ticket": {"id": 1}, "audit": {"events": []}})
    mock_request = Mock(side_effect=[*_macro_reads(), saved])
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: cap)

    raw = zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)

    assert len(raw) <= cap
    assert json.loads(raw)["status"] == "success"
    assert mock_request.call_count == 5
    assert mock_request.call_args.kwargs["method"] == "PUT"


@pytest.mark.parametrize(
    "macro_kwargs",
    [
        # Another macro than the one previewed.
        {"macro_id": 6},
        # The previewed macro, edited after the preview.
        {"updated_at": "2026-10-30T09:00:00Z"},
    ],
)
def test_apply_macro_refuses_a_token_from_another_macro_or_version(
    monkeypatch, macro_kwargs
):
    token = _preview_token(monkeypatch)
    mock_request = Mock(side_effect=_macro_reads(**macro_kwargs))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )

    assert result["status"] == "error"
    assert "changed since" in result["message"]
    assert [call.kwargs["method"] for call in mock_request.call_args_list] == [
        "GET",
        "GET",
        "GET",
        "GET",
    ]


def test_apply_macro_refuses_a_token_from_another_ticket(monkeypatch):
    token = _preview_token(monkeypatch)
    mock_request = Mock(side_effect=_macro_reads(preview=_macro_preview(id=2)))
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(2, 5, dry_run=False, preview_token=token)
    )

    assert result["status"] == "error"
    assert "changed since" in result["message"]
    assert mock_request.call_count == 4


def test_apply_macro_gives_no_token_for_a_preview_too_large_to_show(monkeypatch):
    # A token for changes the caller never saw would let it save them.
    many_tags = [f"tag_{i:03d}" for i in range(300)]
    mock_request = Mock(
        side_effect=_macro_reads(preview=_macro_preview(tags=many_tags))
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    monkeypatch.setattr(zendesk, "get_tool_max_output_length", lambda: 1500)

    raw = zendesk.zendesk_apply_macro(1, 5)
    result = json.loads(raw)

    assert result["status"] == "error"
    assert "preview_token" not in raw


def test_apply_macro_marks_a_plain_text_comment(monkeypatch):
    mock_request = Mock(
        side_effect=_macro_reads(
            [{"field": "comment_value", "value": "Hello"}],
            preview=_macro_preview(comment={"body": "Hello Jane", "html_body": ""}),
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 5))

    assert result["changes"]["comment"] == {
        "body": "Hello Jane",
        "body_is_html": False,
        "public": True,
    }


def _save_after_preview(monkeypatch, third):
    token = _preview_token(monkeypatch)
    mock_request = Mock(side_effect=[*_macro_reads(), third])
    monkeypatch.setattr(zendesk._session, "request", mock_request)
    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=token)
    )
    return result, mock_request


@pytest.mark.parametrize(
    "outcome",
    [
        requests.exceptions.ReadTimeout("read timed out"),
        MockResponse(json_data={"error": "BadGateway"}, status_code=502),
    ],
    ids=["transport-error", "5xx"],
)
def test_apply_macro_warns_a_save_may_have_applied_when_the_outcome_is_unknown(
    monkeypatch, outcome
):
    result, mock_request = _save_after_preview(monkeypatch, outcome)

    assert result["status"] == "error"
    assert "may have been applied" in result["message"]
    assert "zendesk_get_ticket" in result["message"]
    assert mock_request.call_count == 5


def test_apply_macro_does_not_warn_for_a_5xx_before_saving(monkeypatch):
    mock_request = Mock(
        side_effect=[
            _macro(_PROBE_MACRO_ACTIONS),
            _no_attachments(),
            MockResponse(json_data={"error": "Oops"}, status_code=503),
        ]
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 5))

    assert result["status"] == "error"
    assert "HTTP 503" in result["message"]
    assert "may have been applied" not in result["message"]


def test_apply_macro_refuses_a_preview_without_updated_at(monkeypatch):
    mock_request = Mock(
        side_effect=_macro_reads(preview=_macro_preview(_drop=("updated_at",)))
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 5))

    assert result["status"] == "error"
    assert "no updated_at to pin a save to" in result["message"]


def test_apply_macro_refuses_a_preview_without_a_ticket(monkeypatch):
    mock_request = Mock(
        side_effect=_macro_reads(preview=MockResponse(json_data={"result": {}}))
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 5))

    assert result["status"] == "error"
    assert "no preview of this macro's result" in result["message"]


def test_apply_macro_refuses_a_preview_missing_a_field_it_sets(monkeypatch):
    mock_request = Mock(
        side_effect=_macro_reads(
            [{"field": "status", "value": "pending"}],
            preview=_macro_preview(_drop=("status",)),
        )
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(zendesk.zendesk_apply_macro(1, 5))

    assert result["status"] == "error"
    assert "has no status" in result["message"]


def test_apply_macro_accepts_a_token_with_surrounding_whitespace(monkeypatch):
    saved = MockResponse(
        json_data={"ticket": {"id": 1, "status": "pending"}, "audit": {"events": []}}
    )
    token = _preview_token(monkeypatch)
    mock_request = Mock(side_effect=[*_macro_reads(), saved])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    result = json.loads(
        zendesk.zendesk_apply_macro(1, 5, dry_run=False, preview_token=f"  {token}\n")
    )

    assert result["status"] == "success"
    assert mock_request.call_args_list[4].kwargs["method"] == "PUT"
    assert mock_request.call_count == 5


@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "save"])
def test_apply_macro_refuses_a_macro_with_attached_files(monkeypatch, dry_run):
    # Attached files are a separate resource, not in `actions`: saving would
    # post the comment without them.
    token = _preview_token(monkeypatch) if not dry_run else None
    attached = MockResponse(
        json_data={"macro_attachments": [{"id": 1, "filename": "guide.pdf"}]}
    )
    mock_request = Mock(side_effect=[_macro(_PROBE_MACRO_ACTIONS), attached])
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    raw = zendesk.zendesk_apply_macro(
        1, 5, dry_run=dry_run, preview_token=token if not dry_run else None
    )
    result = json.loads(raw)

    assert result["status"] == "error"
    assert "attached files" in result["message"]
    assert "preview_token" not in raw
    assert mock_request.call_count == 2
    assert mock_request.call_args.kwargs["url"].endswith("/macros/5/attachments.json")
    assert all(call.kwargs["method"] == "GET" for call in mock_request.call_args_list)


def test_apply_macro_refuses_when_zendesk_does_not_say_about_attachments(
    monkeypatch,
):
    mock_request = Mock(
        side_effect=[_macro(_PROBE_MACRO_ACTIONS), MockResponse(json_data={})]
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    raw = zendesk.zendesk_apply_macro(1, 5)
    result = json.loads(raw)

    assert result["status"] == "error"
    assert "didn't say whether this macro has attached files" in result["message"]
    assert "preview_token" not in raw
    assert mock_request.call_count == 2


@pytest.mark.parametrize("dry_run", [True, False], ids=["preview", "save"])
def test_apply_macro_refuses_when_the_macro_gains_an_action_while_it_is_applied(
    monkeypatch, dry_run
):
    # The allowlist ran on the first macro read. An admin adding an
    # unsupported action before the render leaves the render, and so the
    # write digest, unchanged: only the second macro read can tell.
    token = _preview_token(monkeypatch) if not dry_run else None
    mock_request = Mock(
        side_effect=[
            _macro(_PROBE_MACRO_ACTIONS),
            _no_attachments(),
            _macro_preview(),
            _macro(
                [*_PROBE_MACRO_ACTIONS, {"field": "cc", "value": "42"}],
                updated_at="2026-10-29T00:00:05Z",
            ),
        ]
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    raw = zendesk.zendesk_apply_macro(
        1, 5, dry_run=dry_run, preview_token=token if not dry_run else None
    )
    result = json.loads(raw)

    assert result["status"] == "error"
    assert "changed while it was being applied" in result["message"]
    assert "preview_token" not in raw
    assert mock_request.call_count == 4
    assert all(call.kwargs["method"] == "GET" for call in mock_request.call_args_list)
