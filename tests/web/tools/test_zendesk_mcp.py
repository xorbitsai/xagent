import json
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
    monkeypatch.setenv("ZENDESK_EMAIL", "agent@acme.com")
    monkeypatch.setenv("ZENDESK_API_TOKEN", "test-token")
    # _base_url() resolves DNS to catch a hostname that rebinds to a private
    # address; tests must not depend on real network/DNS, so every test gets
    # a fake resolver returning an unambiguously public IP by default.
    monkeypatch.setattr(zendesk.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1"))


def test_auth_requires_email(monkeypatch):
    monkeypatch.delenv("ZENDESK_EMAIL")

    with pytest.raises(ValueError, match="ZENDESK_EMAIL"):
        zendesk._auth()


def test_auth_requires_api_token(monkeypatch):
    monkeypatch.delenv("ZENDESK_API_TOKEN")

    with pytest.raises(ValueError, match="ZENDESK_API_TOKEN"):
        zendesk._auth()


def test_auth_returns_email_token_username_and_token_password():
    assert zendesk._auth() == ("agent@acme.com/token", "test-token")


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

    assert zendesk._extract_error_detail(response) == "Not found"


def test_extract_error_detail_falls_back_to_string_error():
    response = MockResponse(json_data={"error": "Couldn't authenticate you"})

    assert zendesk._extract_error_detail(response) == "Couldn't authenticate you"


def test_extract_error_detail_handles_nested_error_object():
    response = MockResponse(
        json_data={"error": {"title": "Unauthorized", "message": "Bad credentials"}}
    )

    assert zendesk._extract_error_detail(response) == "Bad credentials"


def test_extract_error_detail_returns_none_for_non_json_body():
    response = MockResponse(status_code=500, text="not json", json_raises=True)

    assert zendesk._extract_error_detail(response) is None


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
    assert mock_request.call_args.kwargs["auth"] == (
        "agent@acme.com/token",
        "test-token",
    )
    assert mock_request.call_args.kwargs["allow_redirects"] is False


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
    assert len(raw) <= 2000 + 200  # last halving step can overshoot


def test_search_clamps_page_to_at_least_one(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"results": [], "count": 0})
    )
    monkeypatch.setattr(zendesk._session, "request", mock_request)

    zendesk.zendesk_search("type:ticket", page=0)

    assert mock_request.call_args.kwargs["params"]["page"] == 1


def test_past_search_window_checks_the_windows_end_not_its_start():
    # limit=30, page=34: (page-1)*limit=990 is under the 1000 ceiling, but
    # the requested range is results 991-1020, which straddles past it --
    # the check must catch the window's *end*, not just where it starts,
    # or a non-divisor limit slips a straddling page through to an opaque
    # Zendesk HTTP error instead of this connector's own clean handling.
    ceiling = zendesk._MAX_SEARCH_RESULT_WINDOW
    assert zendesk._past_search_window(page=34, max_results=30, ceiling=ceiling) is True
    assert (
        zendesk._past_search_window(page=33, max_results=30, ceiling=ceiling) is False
    )


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


def test_search_past_result_window_still_surfaces_missing_credentials(monkeypatch):
    # The past-window guard returns before ever calling _request(), which
    # is normally what validates ZENDESK_SUBDOMAIN/EMAIL/API_TOKEN -- a
    # misconfigured environment must not be masked as a clean "no results"
    # just because the caller happened to page past the window.
    monkeypatch.delenv("ZENDESK_API_TOKEN")
    page = zendesk._MAX_SEARCH_RESULT_WINDOW // 100 + 1

    result = json.loads(zendesk.zendesk_search("type:ticket", limit=100, page=page))

    assert result["status"] == "error"
    assert "ZENDESK_API_TOKEN" in result["message"]


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
    assert len(raw) <= 2000 + 200  # last halving step can overshoot


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
    # The response must echo the resolved numeric id, not the raw URL
    # the caller passed in -- so it's directly comparable with the id
    # every other tool's response returns.
    assert result["ticket_id"] == "123"


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
    assert len(raw) <= 2000 + 200  # last halving step can overshoot


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
    monkeypatch.delenv("ZENDESK_API_TOKEN")
    page = zendesk._MAX_USER_SEARCH_RESULT_WINDOW // 100 + 1

    result = json.loads(
        zendesk.zendesk_search_users("jane@example.com", limit=100, page=page)
    )

    assert result["status"] == "error"
    assert "ZENDESK_API_TOKEN" in result["message"]


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
    assert len(raw) <= 2000 + 200  # last halving step can overshoot


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
                    "meta": {"has_more": False},
                }
            )
        ),
    )

    content, structured = await zendesk.mcp.call_tool(tool_name, arguments)

    result = json.loads(structured["result"])
    assert result["status"] == "success"


def test_zendesk_app_registry_requires_subdomain_email_and_token():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    zendesk_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "zendesk"
    )
    assert zendesk_app["provider_name"] is None
    assert zendesk_app["category"] == "Support"
    assert zendesk_app["transport"] == "stdio"
    assert zendesk_app["launch_config"]["required_env"] == [
        "ZENDESK_SUBDOMAIN",
        "ZENDESK_EMAIL",
        "ZENDESK_API_TOKEN",
    ]
