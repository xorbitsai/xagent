"""Tests for the Freshdesk MCP connector.

Scope note: these are mocked against the REST contract published at
developers.freshdesk.com/api. They pin the request this module builds and how
it translates a response; they cannot show that the documented shape matches a
live tenant, which is still outstanding on xorbitsai/xagent-saas#1409.
"""

import json
import logging
from typing import Any

import pytest
import requests

from xagent.web.tools.mcp import freshdesk


class _FakeResponse:
    """Minimal stand-in for requests.Response.

    Only the attributes _request()/_extract_error_detail() actually read are
    implemented, so a drift in what they read shows up as an AttributeError
    here rather than as a test passing against a mock that quietly answers
    everything.
    """

    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = content if content is not None else text.encode()
        self.headers = headers or {}

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


# ---------------------------------------------------------------------------
# Subdomain validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "acme",
        "acme-support",
        "a",
        "a1",
        "1acme",
        "a" * 63,
    ],
)
def test_subdomain_accepts_valid_labels(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", value)
    assert freshdesk._subdomain() == value


def test_subdomain_lowercases_and_strips(monkeypatch: pytest.MonkeyPatch):
    """A copy-pasted value routinely carries case and surrounding whitespace.

    Both are safe to normalize (DNS labels are case-insensitive), and doing so
    here keeps them from reaching the composed URL, where a stray space would
    become %20 in the Host rather than an actionable error.
    """
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", "  ACME-Support \n")
    assert freshdesk._subdomain() == "acme-support"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        # A full host, or anything carrying the domain, is the most likely
        # user error: accepting it would compose acme.freshdesk.com.freshdesk.com.
        "acme.freshdesk.com",
        "https://acme.freshdesk.com",
        "acme.freshdesk.com/api/v2",
        # Separators that would escape the label position in the composed URL.
        "acme/../evil",
        "acme/x",
        "acme:8443",
        "acme@evil.com",
        "acme?x=1",
        "acme#frag",
        "acme evil",
        # Label-shape rules: no leading/trailing hyphen, no underscore, <= 63.
        "-acme",
        "acme-",
        "acme_support",
        "a" * 64,
        # Non-ASCII: freshdesk.com subdomains are ASCII labels, and accepting
        # Unicode here would leave the IDNA question open at the URL layer.
        "acmé",
    ],
)
def test_subdomain_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", value)
    with pytest.raises(ValueError, match="FRESHDESK_SUBDOMAIN"):
        freshdesk._subdomain()


def test_subdomain_missing_env_is_actionable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FRESHDESK_SUBDOMAIN", raising=False)
    with pytest.raises(ValueError, match="FRESHDESK_SUBDOMAIN"):
        freshdesk._subdomain()


def test_base_url_is_composed_not_user_supplied(monkeypatch: pytest.MonkeyPatch):
    """The whole SSRF story for this connector is that the host is composed
    from a validated label, never taken from user input. Pin it.
    """
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", "acme")
    assert freshdesk._base_url() == "https://acme.freshdesk.com/api/v2"


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def test_api_key_strips_whitespace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FRESHDESK_API_KEY", "  abc123\n")
    assert freshdesk._api_key() == "abc123"


@pytest.mark.parametrize("value", ["", "   "])
def test_api_key_missing_or_blank_is_actionable(
    monkeypatch: pytest.MonkeyPatch, value: str
):
    monkeypatch.setenv("FRESHDESK_API_KEY", value)
    with pytest.raises(ValueError, match="FRESHDESK_API_KEY"):
        freshdesk._api_key()


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def test_extract_error_detail_prefers_description():
    response = _FakeResponse(payload={"description": "Validation failed"})
    assert freshdesk._extract_error_detail(response) == "Validation failed"


def test_extract_error_detail_includes_per_field_errors():
    """Freshdesk's 400 body carries the actionable part in `errors`, not in
    `description` (which is the generic "Validation failed"). Dropping the
    field list would turn a fixable mistake into an opaque failure.
    """
    response = _FakeResponse(
        payload={
            "description": "Validation failed",
            "errors": [
                {
                    "field": "priority",
                    "message": "is not a valid value",
                    "code": "invalid_value",
                }
            ],
        }
    )
    detail = freshdesk._extract_error_detail(response)
    assert detail is not None
    assert "Validation failed" in detail
    assert "priority" in detail
    assert "is not a valid value" in detail


def test_extract_error_detail_handles_message_only_body():
    response = _FakeResponse(payload={"message": "Access denied"})
    assert freshdesk._extract_error_detail(response) == "Access denied"


def test_extract_error_detail_returns_none_for_non_json():
    response = _FakeResponse(text="<html>502 Bad Gateway</html>")
    assert freshdesk._extract_error_detail(response) is None


def test_extract_error_detail_returns_none_for_non_dict_json():
    response = _FakeResponse(payload=["unexpected"])
    assert freshdesk._extract_error_detail(response) is None


# ---------------------------------------------------------------------------
# _request
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast if a test reaches the real session instead of a recorder.

    Without this, a test that forgets to patch _session.request silently opens
    a socket to acme.freshdesk.com and shows up as a multi-second hang rather
    than a readable failure.
    """

    def _unpatched(**kwargs: Any) -> None:
        raise AssertionError(
            "this test issued a real HTTP request; install a recorder with "
            f"_install(monkeypatch, ...) first (url={kwargs.get('url')!r})"
        )

    monkeypatch.setattr(freshdesk._session, "request", _unpatched)


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """_base_url() resolves the tenant host and rejects private addresses.

    Autouse so no test in this module can reach real DNS -- one that did would
    make the suite depend on the network and on whatever *.freshdesk.com
    happens to resolve to. The two tests that exercise resolution override this
    with their own addresses.
    """
    monkeypatch.setattr(
        freshdesk.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", "acme")
    monkeypatch.setenv("FRESHDESK_API_KEY", "secret-key")


def test_request_uses_basic_auth_with_key_as_username(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Freshdesk's REST API takes the API key as the Basic Auth *username*
    with an ignored password, not as a bearer token. Getting this wrong
    authenticates as nobody and returns 401 on every call.
    """
    captured: dict[str, Any] = {}

    def fake_request(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(payload={"ok": True}, content=b"{}")

    monkeypatch.setattr(freshdesk._session, "request", fake_request)
    freshdesk._request("GET", "/tickets")

    assert captured["auth"] == ("secret-key", "X")
    assert captured["url"] == "https://acme.freshdesk.com/api/v2/tickets"
    assert captured["timeout"] == freshdesk.DEFAULT_TIMEOUT_SECONDS


def test_request_drops_none_and_empty_params(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """An LLM passing status="" to mean "no filter" must not become ?status=."""
    captured: dict[str, Any] = {}

    def fake_request(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(payload={}, content=b"{}")

    monkeypatch.setattr(freshdesk._session, "request", fake_request)
    freshdesk._request(
        "GET", "/tickets", params={"status": "", "priority": None, "page": 2}
    )

    assert captured["params"] == {"page": 2}


def test_request_raises_with_detail_on_error_status(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    monkeypatch.setattr(
        freshdesk._session,
        "request",
        lambda **_: _FakeResponse(
            status_code=404,
            payload={"description": "Resource not found"},
            content=b"{}",
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets/999999999")

    message = str(excinfo.value)
    assert "404" in message
    assert "Resource not found" in message


def test_request_surfaces_rate_limit_retry_after(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A 429 is the one error a caller can actually act on, and Freshdesk puts
    the wait in Retry-After. Without it the LLM only learns "rate limited" and
    has no basis for when to try again -- issue #1409 lists quota exhaustion as
    an error that must be actionable.
    """
    monkeypatch.setattr(
        freshdesk._session,
        "request",
        lambda **_: _FakeResponse(
            status_code=429,
            payload={"description": "You have exceeded the limit"},
            content=b"{}",
            headers={"Retry-After": "42"},
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    message = str(excinfo.value)
    assert "429" in message
    assert "42" in message


def test_request_redacts_credentials_from_transport_errors(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """setup_proxy_env() exports whatever proxy the OS has configured, and a
    ProxyError echoes that URL -- which can embed user:pass@ credentials.
    """

    def raise_proxy_error(**_: Any) -> None:
        raise requests.RequestException(
            "ProxyError: https://bob:hunter2@proxy.internal:3128"
        )

    monkeypatch.setattr(freshdesk._session, "request", raise_proxy_error)
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    assert "hunter2" not in str(excinfo.value)


def test_request_returns_empty_dict_for_204(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A 204 or an empty body must decode to {} rather than raising.

    No tool here issues a DELETE today, but _request is shared by all of them
    and Freshdesk answers several writes with an empty body.
    """
    monkeypatch.setattr(
        freshdesk._session,
        "request",
        lambda **_: _FakeResponse(status_code=204, content=b""),
    )
    assert freshdesk._request("DELETE", "/tickets/1") == {}


def test_request_rejects_non_json_success_body(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A 200 carrying an HTML body means a gateway answered, not Freshdesk."""
    monkeypatch.setattr(
        freshdesk._session,
        "request",
        lambda **_: _FakeResponse(status_code=200, text="<html>hi</html>"),
    )
    with pytest.raises(RuntimeError, match="non-JSON"):
        freshdesk._request("GET", "/tickets")


# ---------------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------------


def test_success_and_error_envelopes_are_json():
    assert json.loads(freshdesk._success(tickets=[])) == {
        "status": "success",
        "tickets": [],
    }
    assert json.loads(freshdesk._error("nope")) == {
        "status": "error",
        "message": "nope",
    }


def test_clamp_per_page_bounds_to_freshdesk_maximum():
    assert freshdesk._clamp_per_page(0) == 1
    assert freshdesk._clamp_per_page(-5) == 1
    assert freshdesk._clamp_per_page(10) == 10
    assert freshdesk._clamp_per_page(5000) == freshdesk.MAX_PER_PAGE


def test_validated_list_rejects_non_list_payload():
    """Freshdesk list endpoints return a bare JSON array, not an envelope."""
    assert freshdesk._validated_list([], "/tickets") == []
    assert freshdesk._validated_list([{"id": 1}], "/tickets") == [{"id": 1}]
    with pytest.raises(RuntimeError, match="/tickets"):
        freshdesk._validated_list({"description": "nope"}, "/tickets")


def test_validated_dict_rejects_non_dict_payload():
    assert freshdesk._validated_dict({"id": 1}, "/tickets/1") == {"id": 1}
    with pytest.raises(RuntimeError, match="/tickets/1"):
        freshdesk._validated_dict([1], "/tickets/1")


# ---------------------------------------------------------------------------
# Catalog wiring
# ---------------------------------------------------------------------------


def test_registry_entry_declares_the_env_vars_this_module_reads():
    """The catalog row is what the connect dialog renders fields for, so a
    rename on either side silently produces a connector that collects the
    wrong values and then fails at tool-call time with "missing env".
    """
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "freshdesk"
    )
    assert row["transport"] == "stdio"
    assert row["provider_name"] is None
    assert row["category"] == "Support"
    assert row["launch_config"]["args"] == ["-m", "xagent.web.tools.mcp.freshdesk"]
    assert row["launch_config"]["required_env"] == [
        "FRESHDESK_SUBDOMAIN",
        "FRESHDESK_API_KEY",
    ]


def test_registry_entry_classifies_as_api_key():
    """classify_app_auth is the single source the backend connect gate and both
    frontend dialogs read; "unconnectable" here would mean no Connect button.
    """
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
    from xagent.web.mcp_apps import classify_app_auth

    row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "freshdesk"
    )
    assert classify_app_auth(row["transport"], row["launch_config"]) == "api_key"


def test_request_hints_at_wrong_subdomain_on_bodyless_404(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """freshdesk.com is wildcard-resolved, so a mistyped subdomain answers
    every path with a body-less 404 from the edge rather than anything naming
    the real problem. Without the hint that is indistinguishable from a
    deleted ticket.
    """
    monkeypatch.setattr(
        freshdesk._session,
        "request",
        lambda **_: _FakeResponse(status_code=404, text="", content=b""),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    message = str(excinfo.value)
    assert "FRESHDESK_SUBDOMAIN" in message
    assert "acme" in message


def test_request_keeps_a_real_404_description_over_the_subdomain_hint(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A genuine Freshdesk 404 names the missing resource; burying that under
    a subdomain hint would send the caller chasing the wrong problem.
    """
    monkeypatch.setattr(
        freshdesk._session,
        "request",
        lambda **_: _FakeResponse(
            status_code=404,
            payload={"description": "Resource not found"},
            content=b"{}",
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets/999999999")

    message = str(excinfo.value)
    assert "Resource not found" in message
    assert "FRESHDESK_SUBDOMAIN" not in message


def test_request_hints_at_credentials_on_bodyless_401(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    monkeypatch.setattr(
        freshdesk._session,
        "request",
        lambda **_: _FakeResponse(status_code=401, text="", content=b""),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    assert "FRESHDESK_API_KEY" in str(excinfo.value)


def test_subdomain_hint_echoes_the_subdomain_and_nothing_from_its_neighbour(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The hint exists to name the wrong subdomain, so pin that it does.

    An earlier version of this test only asserted the API key was absent,
    which the hint never interpolates -- it could not fail and proved nothing.
    """
    monkeypatch.setattr(
        freshdesk._session,
        "request",
        lambda **_: _FakeResponse(status_code=404, text="", content=b""),
    )
    with pytest.raises(RuntimeError) as excinfo:
        freshdesk._request("GET", "/tickets")

    message = str(excinfo.value)
    assert "'acme'" in message, "the hint must name the subdomain it used"
    assert "FRESHDESK_SUBDOMAIN" in message
    assert "secret-key" not in message


# ---------------------------------------------------------------------------
# Tools
#
# Every tool is exercised against a recorded stand-in for the REST contract
# documented at developers.freshdesk.com/api. These prove the request this
# module builds and how it translates a response -- NOT that the documented
# shape matches a live tenant, which is still outstanding.
# ---------------------------------------------------------------------------


class _Recorder:
    """Capture each outbound request and answer from a scripted queue."""

    def __init__(self, *responses: _FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return (
            self.responses.pop(0)
            if self.responses
            else _FakeResponse(payload={}, content=b"{}")
        )

    @property
    def call(self) -> dict[str, Any]:
        assert len(self.calls) == 1, f"expected one request, made {len(self.calls)}"
        return self.calls[0]


def _install(monkeypatch: pytest.MonkeyPatch, *responses: _FakeResponse) -> _Recorder:
    recorder = _Recorder(*responses)
    monkeypatch.setattr(freshdesk._session, "request", recorder)
    return recorder


def _payload(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


def _json_response(
    body: Any, *, headers: dict[str, str] | None = None, status_code: int = 200
) -> _FakeResponse:
    return _FakeResponse(
        status_code=status_code,
        payload=body,
        content=json.dumps(body).encode(),
        headers=headers,
    )


# --- list_tickets ----------------------------------------------------------


def test_list_tickets_sends_filters_and_clamped_page_size(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response([{"id": 1}]))

    result = _payload(
        freshdesk.freshdesk_list_tickets(
            filter_name="new_and_my_open",
            updated_since="2026-09-01T00:00:00Z",
            order_by="updated_at",
            page=0,
            per_page=5000,
        )
    )

    assert result["status"] == "success"
    assert result["tickets"] == [{"id": 1}]
    params = recorder.call["params"]
    # `filter` is the wire name; `filter_name` is the tool parameter, because
    # `filter` shadows a builtin in the signature.
    assert params["filter"] == "new_and_my_open"
    assert params["updated_since"] == "2026-09-01T00:00:00Z"
    assert params["order_by"] == "updated_at"
    assert params["page"] == 1, "page must be clamped to >= 1"
    assert params["per_page"] == freshdesk.MAX_PER_PAGE
    assert "include" not in params, "unset optional filters must not be sent"


def test_list_tickets_reads_has_more_from_the_link_header(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Freshdesk returns no total on list endpoints, so has_more comes from
    its Link header. Inferring it from a full page would report a phantom
    next page every time the last page is exactly full.
    """
    full_page = [{"id": n} for n in range(freshdesk.DEFAULT_PER_PAGE)]
    _install(
        monkeypatch,
        _json_response(
            full_page,
            headers={
                "Link": '<https://acme.freshdesk.com/api/v2/tickets?page=2>; rel="next"'
            },
        ),
    )
    assert _payload(freshdesk.freshdesk_list_tickets())["has_more"] is True

    _install(monkeypatch, _json_response(full_page))
    assert _payload(freshdesk.freshdesk_list_tickets())["has_more"] is False


def test_list_tickets_rejects_an_envelope_where_an_array_is_documented(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The list endpoints are documented to return a bare array. If a tenant
    ever answers with an object instead, failing loudly beats reporting zero
    tickets as though the helpdesk were empty.
    """
    _install(monkeypatch, _json_response({"tickets": [{"id": 1}]}))
    result = _payload(freshdesk.freshdesk_list_tickets())
    assert result["status"] == "error"
    assert "/tickets" in result["message"]


def test_list_tickets_truncates_an_oversized_page(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A ticket carries its full description, so a page of 100 can exceed the
    platform's output cap. The dropped ones are not on the next page, so the
    response must say so rather than silently shrinking.
    """
    monkeypatch.setattr(freshdesk, "get_tool_max_output_length", lambda: 2000)
    fat = [{"id": n, "description": "x" * 500} for n in range(40)]
    _install(monkeypatch, _json_response(fat))

    result = _payload(freshdesk.freshdesk_list_tickets(per_page=40))

    assert result["truncated"] is True
    assert len(result["tickets"]) < 40
    assert "smaller per_page" in result["message"]


# --- get_ticket ------------------------------------------------------------


def test_get_ticket_requests_the_id_path(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response({"id": 42, "subject": "hi"}))

    result = _payload(freshdesk.freshdesk_get_ticket(42, include="requester"))

    assert result["ticket"]["id"] == 42
    assert recorder.call["url"].endswith("/api/v2/tickets/42")
    assert recorder.call["params"]["include"] == "requester"


@pytest.mark.parametrize("bad", ["12 OR 1=1", "../agents", "abc", 0, -3, None])
def test_ticket_id_is_validated_before_it_reaches_the_path(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, bad: Any
):
    """The id is interpolated into the request path, so a non-numeric value
    must fail locally rather than being pasted into the URL.
    """
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_get_ticket(bad))
    assert result["status"] == "error"
    assert recorder.calls == [], "no request may be issued for an invalid id"


# --- search_tickets --------------------------------------------------------


def test_search_tickets_wraps_the_expression_in_double_quotes(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Freshdesk requires the filter expression to arrive double-quoted. The
    tool adds them so a caller passing a bare expression works, and so a
    pre-quoted one cannot become a doubly-quoted always-empty search.
    """
    recorder = _install(monkeypatch, _json_response({"results": [], "total": 0}))

    freshdesk.freshdesk_search_tickets("status:2 AND priority:4")

    assert recorder.call["params"]["query"] == '"status:2 AND priority:4"'
    assert recorder.call["url"].endswith("/api/v2/search/tickets")


def test_search_tickets_returns_total_and_results(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    _install(monkeypatch, _json_response({"results": [{"id": 7}], "total": 1}))
    result = _payload(freshdesk.freshdesk_search_tickets("status:2"))
    assert result["results"] == [{"id": 7}]
    assert result["total"] == 1
    assert result["has_more"] is False


def test_search_tickets_reports_more_only_while_pages_remain(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Page 10 is Freshdesk's hard ceiling, whatever `total` says.

    The signal itself comes from `total` where available -- see
    test_search_has_more_comes_from_total_not_page_fullness; this pins the
    ceiling that caps it either way.
    """
    full = [{"id": n} for n in range(freshdesk.SEARCH_PAGE_SIZE)]
    _install(monkeypatch, _json_response({"results": full, "total": 900}))
    assert _payload(freshdesk.freshdesk_search_tickets("status:2"))["has_more"] is True

    _install(monkeypatch, _json_response({"results": full, "total": 900}))
    last = _payload(
        freshdesk.freshdesk_search_tickets("status:2", page=freshdesk.MAX_SEARCH_PAGE)
    )
    assert last["has_more"] is False, "page 10 is the last page Freshdesk serves"


def test_search_tickets_refuses_a_page_past_the_vendor_ceiling(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_search_tickets("status:2", page=11))
    assert result["status"] == "error"
    assert "narrow the query" in result["message"]
    assert recorder.calls == []


def test_search_tickets_rejects_a_blank_query(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch)
    assert _payload(freshdesk.freshdesk_search_tickets("   "))["status"] == "error"
    assert recorder.calls == []


# --- create_ticket ---------------------------------------------------------


def test_create_ticket_posts_defaults_and_requester(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response({"id": 9}))

    result = _payload(
        freshdesk.freshdesk_create_ticket(
            subject="Printer down",
            description="It is on fire",
            email="user@example.com",
        )
    )

    assert result["ticket"]["id"] == 9
    body = recorder.call["json"]
    assert body["subject"] == "Printer down"
    assert body["email"] == "user@example.com"
    assert body["status"] == freshdesk.STATUS_OPEN
    assert body["priority"] == freshdesk.PRIORITY_LOW
    assert "responder_id" not in body, "unset optionals must not be sent"


def test_create_ticket_requires_a_requester_identifier(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Freshdesk rejects a ticket with no requester; catching it here names
    the three fields that would satisfy it instead of relaying a 400.
    """
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_create_ticket(subject="s", description="d"))
    assert result["status"] == "error"
    assert "requester" in result["message"]
    assert recorder.calls == []


@pytest.mark.parametrize("field,value", [("status", 1), ("priority", 9)])
def test_create_ticket_rejects_values_below_or_outside_the_vendor_range(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, field: str, value: int
):
    """status=1 is the trap: it is a legal *priority* and not a legal status,
    so an LLM reaches for it. Naming the legal values beats "Validation failed".
    """
    recorder = _install(monkeypatch)
    result = _payload(
        freshdesk.freshdesk_create_ticket(
            subject="s", description="d", email="a@b.c", **{field: value}
        )
    )
    assert result["status"] == "error"
    assert field in result["message"]
    assert recorder.calls == []


@pytest.mark.parametrize("custom_status", [6, 7, 42])
def test_custom_ticket_statuses_are_forwarded_not_rejected(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, custom_status: int
):
    """A helpdesk can define custom statuses with instance-specific numeric
    values above the four built-ins. This connector cannot know a tenant's
    legal set, so closing the enum at 5 would make it narrower than the API it
    fronts and break every helpdesk that defines one -- Freshdesk decides.
    """
    recorder = _install(monkeypatch, _json_response({"id": 3}))

    result = _payload(freshdesk.freshdesk_update_ticket(3, status=custom_status))

    assert result["status"] == "success"
    assert recorder.call["json"] == {"status": custom_status}


def test_priority_stays_a_closed_set(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Priority, unlike status, is not customizable, so an out-of-range value
    is a caller mistake worth catching locally.
    """
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_update_ticket(3, priority=7))
    assert result["status"] == "error"
    assert "priority" in result["message"]
    assert recorder.calls == []


@pytest.mark.parametrize("blank", ["", "   "])
def test_create_ticket_rejects_blank_subject_or_description(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, blank: str
):
    recorder = _install(monkeypatch)
    assert (
        _payload(
            freshdesk.freshdesk_create_ticket(
                subject=blank, description="d", email="a@b.c"
            )
        )["status"]
        == "error"
    )
    assert (
        _payload(
            freshdesk.freshdesk_create_ticket(
                subject="s", description=blank, email="a@b.c"
            )
        )["status"]
        == "error"
    )
    assert recorder.calls == []


# --- update_ticket ---------------------------------------------------------


def test_update_ticket_sends_only_the_fields_given(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response({"id": 3}))

    freshdesk.freshdesk_update_ticket(3, status=4)

    assert recorder.call["method"] == "PUT"
    assert recorder.call["json"] == {"status": 4}


def test_update_ticket_distinguishes_clearing_tags_from_leaving_them(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """tags=[] means "remove every tag" and must reach Freshdesk; tags=None
    means "leave them alone" and must not. Collapsing the two would either
    silently wipe a ticket's tags or make clearing impossible.
    """
    recorder = _install(monkeypatch, _json_response({"id": 3}))
    freshdesk.freshdesk_update_ticket(3, tags=[])
    assert recorder.call["json"] == {"tags": []}

    recorder = _install(monkeypatch, _json_response({"id": 3}))
    freshdesk.freshdesk_update_ticket(3, status=2, tags=None)
    assert "tags" not in recorder.call["json"]


def test_update_ticket_requires_at_least_one_field(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """An empty PUT would report success while changing nothing."""
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_update_ticket(3))
    assert result["status"] == "error"
    assert recorder.calls == []


# --- conversations ---------------------------------------------------------


def test_list_ticket_conversations_pages_the_ticket_subpath(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(
        monkeypatch,
        _json_response(
            [{"id": 1, "private": True}],
            headers={"Link": '<...?page=2>; rel="next"'},
        ),
    )

    result = _payload(freshdesk.freshdesk_list_ticket_conversations(8, per_page=10))

    assert result["conversations"] == [{"id": 1, "private": True}]
    assert result["has_more"] is True
    assert recorder.call["url"].endswith("/api/v2/tickets/8/conversations")
    assert recorder.call["params"]["per_page"] == 10


def test_reply_posts_to_the_reply_subpath(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response({"id": 11}))

    freshdesk.freshdesk_reply_to_ticket(8, "on our way", cc_emails=["b@c.d"])

    assert recorder.call["method"] == "POST"
    assert recorder.call["url"].endswith("/api/v2/tickets/8/reply")
    assert recorder.call["json"] == {"body": "on our way", "cc_emails": ["b@c.d"]}


def test_note_defaults_to_private(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A note that defaults to public would disclose internal discussion to
    the requester, and nothing in the call site would show it.
    """
    recorder = _install(monkeypatch, _json_response({"id": 12}))

    freshdesk.freshdesk_add_note_to_ticket(8, "refund approved by finance")

    assert recorder.call["url"].endswith("/api/v2/tickets/8/notes")
    assert recorder.call["json"]["private"] is True


def test_note_can_be_made_public_explicitly(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response({"id": 12}))
    freshdesk.freshdesk_add_note_to_ticket(8, "visible", private=False)
    assert recorder.call["json"]["private"] is False


@pytest.mark.parametrize("blank", ["", "   "])
def test_reply_and_note_reject_an_empty_body(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, blank: str
):
    recorder = _install(monkeypatch)
    assert _payload(freshdesk.freshdesk_reply_to_ticket(8, blank))["status"] == "error"
    assert (
        _payload(freshdesk.freshdesk_add_note_to_ticket(8, blank))["status"] == "error"
    )
    assert recorder.calls == []


# --- contacts and agents ---------------------------------------------------


def test_bad_contact_id_names_the_contact_field_not_the_ticket_field(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The id validator is shared with the ticket tools; reporting
    "ticket_id must be an integer" for a contact sends the caller to the wrong
    argument.
    """
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_get_contact("abc"))
    assert result["status"] == "error"
    assert "contact_id" in result["message"]
    assert "ticket_id" not in result["message"]
    assert recorder.calls == []


def test_get_contact_requests_the_id_path(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response({"id": 5, "name": "Jo"}))
    result = _payload(freshdesk.freshdesk_get_contact(5))
    assert result["contact"]["name"] == "Jo"
    assert recorder.call["url"].endswith("/api/v2/contacts/5")


def test_search_contacts_uses_autocomplete_for_a_term(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response([{"id": 5}]))

    result = _payload(freshdesk.freshdesk_search_contacts(term="jo"))

    assert result["contacts"] == [{"id": 5}]
    assert recorder.call["url"].endswith("/api/v2/contacts/autocomplete")
    assert recorder.call["params"]["term"] == "jo"
    assert result["has_more"] is False


def test_search_contacts_uses_exact_filters_without_a_term(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response([{"id": 5}]))

    freshdesk.freshdesk_search_contacts(email="jo@example.com")

    assert recorder.call["url"].endswith("/api/v2/contacts")
    assert recorder.call["params"]["email"] == "jo@example.com"


def test_search_contacts_refuses_term_mixed_with_exact_filters(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The two go to different endpoints, so honouring one and dropping the
    other would silently answer a narrower question than was asked.
    """
    recorder = _install(monkeypatch)
    result = _payload(
        freshdesk.freshdesk_search_contacts(term="jo", email="jo@example.com")
    )
    assert result["status"] == "error"
    assert "email" in result["message"]
    assert recorder.calls == []


def test_list_agents_filters_and_pages(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response([{"id": 2}]))

    result = _payload(freshdesk.freshdesk_list_agents(state="fulltime", per_page=101))

    assert result["agents"] == [{"id": 2}]
    assert recorder.call["params"]["state"] == "fulltime"
    assert recorder.call["params"]["per_page"] == freshdesk.MAX_PER_PAGE


# --- error propagation through the tool boundary ---------------------------


def test_a_vendor_error_becomes_an_error_envelope_not_an_exception(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Every tool returns a JSON string; an exception escaping one would
    surface to the agent as a tool crash instead of a readable message.
    """
    _install(
        monkeypatch,
        _json_response(
            {"description": "Validation failed", "errors": [{"field": "status"}]},
            status_code=400,
        ),
    )
    result = _payload(freshdesk.freshdesk_update_ticket(3, status=4))
    assert result["status"] == "error"
    assert "Validation failed" in result["message"]
    assert "status" in result["message"]


def test_a_missing_credential_surfaces_through_the_tool_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRESHDESK_SUBDOMAIN", "acme")
    monkeypatch.delenv("FRESHDESK_API_KEY", raising=False)
    recorder = _install(monkeypatch)

    result = _payload(freshdesk.freshdesk_list_tickets())

    assert result["status"] == "error"
    assert "FRESHDESK_API_KEY" in result["message"]
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# Preflight regressions
#
# Each of these pins a defect the preflight fan-out found in the code written
# to answer the previous review round.
# ---------------------------------------------------------------------------


def test_log_output_does_not_leak_a_proxy_credential(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    caplog: pytest.LogCaptureFixture,
):
    """The error envelope was redacted but the LOG was not.

    _send raises RuntimeError(redacted) `from exc`, so anything that formats
    the __cause__ chain -- an exc_info=True this module no longer passes --
    reaches the original ProxyError, which echoes HTTPS_PROXY with its
    embedded user:pass@. The redaction in _send exists precisely for that
    string. Reproduced against the leaking version before it was fixed; this
    keeps the log line from regaining the chain.
    """

    def raise_proxy_error(**_: Any) -> None:
        raise requests.exceptions.ProxyError(
            "HTTPSConnectionPool: Max retries exceeded "
            "(Caused by ProxyError('https://bob:s3cr3tpass@proxy.internal:8080'))"
        )

    monkeypatch.setattr(freshdesk._session, "request", raise_proxy_error)

    with caplog.at_level(logging.ERROR, logger="freshdesk-mcp"):
        result = freshdesk.freshdesk_list_tickets()

    assert _payload(result)["status"] == "error"
    assert "s3cr3tpass" not in result
    assert "s3cr3tpass" not in caplog.text
    assert caplog.records, "the failure must still be logged for operators"


def test_search_terms_are_scrubbed_from_error_text(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    caplog: pytest.LogCaptureFixture,
):
    """A urllib3 exception embeds the full URL, and a Freshdesk URL carries the
    caller's search terms -- end-user PII -- in its query string.
    redact_sensitive_text only knows credential-shaped keys, not `query=`.
    """

    def raise_with_url(**_: Any) -> None:
        raise requests.exceptions.ConnectionError(
            "HTTPSConnectionPool(host='acme.freshdesk.com', port=443): "
            "Max retries exceeded with url: "
            "/api/v2/contacts?email=patient%40clinic.example "
            "(Caused by NewConnectionError())"
        )

    monkeypatch.setattr(freshdesk._session, "request", raise_with_url)

    with caplog.at_level(logging.ERROR, logger="freshdesk-mcp"):
        result = freshdesk.freshdesk_search_contacts(email="patient@clinic.example")

    assert "patient%40clinic.example" not in result
    assert "patient%40clinic.example" not in caplog.text
    assert "<query redacted>" in result


def test_a_redirect_is_refused_rather_than_followed(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Following a 3xx would attach the Basic credential to whatever Location
    names, and a 307/308 would replay the request body there too.
    """
    recorder = _install(monkeypatch, _json_response({}, status_code=302))
    monkeypatch.setattr(freshdesk._session, "request", recorder)

    result = _payload(freshdesk.freshdesk_list_tickets())

    assert result["status"] == "error"
    assert "redirect" in result["message"].lower()
    assert recorder.call["allow_redirects"] is False


def test_the_session_does_not_trust_ambient_proxy_env():
    """An ambient OS proxy does its own DNS resolution, bypassing the
    private-network check _base_url() performs on the addresses this process
    resolved -- the hole that check exists to close.
    """
    assert freshdesk._session.trust_env is False


def test_base_url_refuses_a_host_that_resolves_privately(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A legitimate *.freshdesk.com label can still be rebound by DNS to an
    internal address at request time; the label check is orthogonal to that.
    """
    monkeypatch.setattr(
        freshdesk.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="not allowed"):
        freshdesk._base_url()


def test_base_url_accepts_a_public_address(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    monkeypatch.setattr(
        freshdesk.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    assert freshdesk._base_url() == "https://acme.freshdesk.com/api/v2"


@pytest.mark.parametrize("bad", [9.7, 2.5, True, False])
def test_non_integral_values_are_refused_rather_than_truncated(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, bad: Any
):
    """int() truncates: status=9.7 used to be rejected by the closed enum, but
    once the enum opened for custom statuses it would have shipped as 9.
    bool is an int subclass, so True would pass as priority 1.
    """
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_update_ticket(3, status=bad))
    assert result["status"] == "error"
    assert recorder.calls == []


def test_an_overlong_search_query_is_refused_locally(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The tool adds the two enclosing quotes, and they count toward
    Freshdesk's 512-character limit -- so a 511-character expression would
    have been sent as 513 and rejected remotely.
    """
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_search_tickets("a" * 511))
    assert result["status"] == "error"
    assert "512" in result["message"]
    assert recorder.calls == []

    # 510 + 2 quotes == exactly the limit, and must still be sent.
    recorder = _install(monkeypatch, _json_response({"results": [], "total": 0}))
    assert (
        _payload(freshdesk.freshdesk_search_tickets("a" * 510))["status"] == "success"
    )


def test_phone_only_create_requires_a_name(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """ "If the phone number is set and the email address is not, then the name
    attribute is mandatory." Without the local guard this always 400s.
    """
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_create_ticket("s", "d", phone="+15551234567"))
    assert result["status"] == "error"
    assert "name" in result["message"]
    assert recorder.calls == []

    recorder = _install(monkeypatch, _json_response({"id": 1}))
    ok = _payload(
        freshdesk.freshdesk_create_ticket("s", "d", phone="+15551234567", name="Jo")
    )
    assert ok["status"] == "success"
    assert recorder.call["json"]["name"] == "Jo"


def test_email_create_does_not_require_a_name(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The name requirement is phone-only; demanding it for an email create
    would be narrower than the API.
    """
    _install(monkeypatch, _json_response({"id": 1}))
    result = _payload(freshdesk.freshdesk_create_ticket("s", "d", email="a@b.c"))
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Review round 1 (xorbitsai/xagent#2601)
# ---------------------------------------------------------------------------


def test_an_untrusted_ambient_proxy_is_refused(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """setup_proxy_env() promotes whatever proxy the OS has into the
    environment. A proxy resolves the target host itself, bypassing the
    private-network check _base_url() performs, so an untrusted one must stop
    the request rather than silently carry the Basic credential.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
    recorder = _install(monkeypatch)

    result = _payload(freshdesk.freshdesk_list_tickets())

    assert result["status"] == "error"
    assert recorder.calls == [], "no request may go through an untrusted proxy"


def test_a_trusted_proxy_is_forwarded_explicitly(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Opting in routes through the proxy via an explicit proxies= argument
    rather than trust_env, which stays off.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    recorder = _install(monkeypatch, _json_response([]))

    result = _payload(freshdesk.freshdesk_list_tickets())

    assert result["status"] == "success"
    assert recorder.call["proxies"]["https"] == "http://proxy.internal:8080"
    assert freshdesk._session.trust_env is False


def test_an_unresolvable_host_is_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The OSError branch of _base_url()'s lookup was never exercised."""

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("Name or service not known")

    monkeypatch.setattr(freshdesk.socket, "getaddrinfo", boom)
    recorder = _install(monkeypatch)

    result = _payload(freshdesk.freshdesk_list_tickets())

    assert result["status"] == "error"
    assert "could not be resolved" in result["message"]
    assert recorder.calls == []


def test_a_429_without_retry_after_still_reports_the_rate_limit(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Retry-After is documented but not guaranteed; the status must still be
    legible without it rather than producing a bare "(retry after s)".
    """
    _install(
        monkeypatch,
        _json_response({"description": "You have exceeded the limit"}, status_code=429),
    )
    result = _payload(freshdesk.freshdesk_list_tickets())

    assert result["status"] == "error"
    assert "429" in result["message"]
    assert "retry after" not in result["message"], (
        "no Retry-After header means no retry hint, not an empty one"
    )


@pytest.mark.parametrize("wrapped", ['"status:2"', '  "status:2"  '])
def test_a_pre_quoted_expression_is_unwrapped_not_double_quoted(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, wrapped: str
):
    """A caller that already wrapped the expression is the common mistake, and
    it used to produce '""status:2""' -- syntactically fine, always empty.
    """
    recorder = _install(monkeypatch, _json_response({"results": [], "total": 0}))
    freshdesk.freshdesk_search_tickets(wrapped)
    assert recorder.call["params"]["query"] == '"status:2"'


def test_an_interior_double_quote_is_refused(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Freshdesk quotes string values with single quotes, so an interior
    double quote cannot be legitimate -- and guessing its intent would
    silently change the query.
    """
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_search_tickets('tag:"urgent"'))
    assert result["status"] == "error"
    assert "single quotes" in result["message"]
    assert recorder.calls == []


def test_search_has_more_comes_from_total_not_page_fullness(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """total is exact and arrives in the same response; the full-page
    heuristic reported a phantom page whenever the last page was exactly full.
    """
    full = [{"id": n} for n in range(freshdesk.SEARCH_PAGE_SIZE)]

    # Exactly one full page and total says that is all there is.
    _install(monkeypatch, _json_response({"results": full, "total": 30}))
    assert _payload(freshdesk.freshdesk_search_tickets("status:2"))["has_more"] is False

    # A full page with more behind it.
    _install(monkeypatch, _json_response({"results": full, "total": 90}))
    assert _payload(freshdesk.freshdesk_search_tickets("status:2"))["has_more"] is True

    # No usable total -- fall back to the heuristic.
    _install(monkeypatch, _json_response({"results": full}))
    assert _payload(freshdesk.freshdesk_search_tickets("status:2"))["has_more"] is True


@pytest.mark.parametrize("field", ["requester_id", "responder_id", "group_id"])
def test_body_ids_are_validated_like_path_ids(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, field: str
):
    """These reach Freshdesk in the request body, so they escaped the
    validation every path-embedded id gets: requester_id=0 was forwarded.
    """
    recorder = _install(monkeypatch)
    result = _payload(
        freshdesk.freshdesk_create_ticket("s", "d", email="a@b.c", **{field: 0})
    )
    assert result["status"] == "error"
    assert field in result["message"]
    assert recorder.calls == []


def test_update_ticket_validates_its_body_ids_too(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_update_ticket(3, responder_id=-1))
    assert result["status"] == "error"
    assert "responder_id" in result["message"]
    assert recorder.calls == []


def test_the_catalog_row_ships_hidden_until_verified():
    """Nothing here has been run against a live tenant and the connector can
    email a requester, so it ships hidden like zendesk and intercom.
    """
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    rows = {r["app_id"]: r for r in get_builtin_public_mcp_app_rows()}
    assert rows["freshdesk"]["is_visible_in_connector"] is False
    assert rows["zendesk"]["is_visible_in_connector"] is False, (
        "the precedent this follows"
    )


def test_create_ticket_sends_normalized_identifiers(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The guards read stripped values while the payload sent the raw ones, so
    a padded address went to Freshdesk with its whitespace intact.
    """
    recorder = _install(monkeypatch, _json_response({"id": 1}))

    freshdesk.freshdesk_create_ticket(
        "  s  ", "  d  ", email="  a@b.c  ", name="  Jo  "
    )

    body = recorder.call["json"]
    assert body["email"] == "a@b.c"
    assert body["name"] == "Jo"
    assert body["subject"] == "s"
    assert body["description"] == "d"


def test_a_whitespace_only_identifier_is_absent_not_sent_blank(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """`email="   "` was judged absent by the guard and still sent, because
    the payload filter drops "" but not "   ".
    """
    recorder = _install(monkeypatch, _json_response({"id": 1}))

    freshdesk.freshdesk_create_ticket(
        "s", "d", phone="+15551234567", email="   ", name="Jo"
    )

    assert "email" not in recorder.call["json"]
    assert recorder.call["json"]["phone"] == "+15551234567"


def test_a_401_with_a_body_still_gets_the_credential_hint(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A real Freshdesk 401 carries a message, so gating the hint on an empty
    detail meant it never fired in the only case it was written for.
    """
    _install(
        monkeypatch,
        _json_response({"message": "Access denied"}, status_code=401),
    )
    result = _payload(freshdesk.freshdesk_list_tickets())

    assert "Access denied" in result["message"], "the vendor's own message survives"
    assert "FRESHDESK_API_KEY" in result["message"], "and the hint is appended"


def test_a_403_gets_the_permission_hint(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    _install(
        monkeypatch,
        _json_response({"description": "Forbidden"}, status_code=403),
    )
    result = _payload(freshdesk.freshdesk_list_tickets())
    assert "Forbidden" in result["message"]
    assert "permission" in result["message"]


def test_tags_are_stripped_and_blanks_dropped(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """ "vip " silently fails to match the canonical "vip" already on the
    account, and a blank entry is a wasted tag.
    """
    recorder = _install(monkeypatch, _json_response({"id": 3}))
    freshdesk.freshdesk_update_ticket(3, tags=["vip ", "", "  urgent"])
    assert recorder.call["json"]["tags"] == ["vip", "urgent"]


def test_an_all_blank_tag_list_is_refused_not_treated_as_a_clear(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """Freshdesk replaces the whole tag list, so collapsing ["  "] into the
    same request as an explicit clear would wipe the tags off a ticket nobody
    asked to untag.
    """
    recorder = _install(monkeypatch)
    result = _payload(freshdesk.freshdesk_update_ticket(3, tags=["  ", ""]))

    assert result["status"] == "error"
    assert "explicitly clear" in result["message"]
    assert recorder.calls == []


def test_an_explicit_empty_tag_list_still_clears(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _json_response({"id": 3}))
    freshdesk.freshdesk_update_ticket(3, tags=[])
    assert recorder.call["json"]["tags"] == []


def test_create_ticket_cleans_tags_too(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """The same field, the same replace semantics -- fixing only update_ticket
    would leave the sibling to be re-raised.
    """
    recorder = _install(monkeypatch, _json_response({"id": 1}))
    freshdesk.freshdesk_create_ticket("s", "d", email="a@b.c", tags=["vip ", ""])
    assert recorder.call["json"]["tags"] == ["vip"]
