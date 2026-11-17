"""Tests for the Chargebee MCP connector.

Scope note: these are mocked against the REST API v2 contract published at
apidocs.chargebee.com. They pin the request this module builds and how it
translates a response; they cannot show that the documented shape matches a
live site, which is still outstanding (xorbitsai/xagent-saas#2466).
"""

import asyncio
import json
from typing import Any

import pytest
import requests

from xagent.web.tools.mcp import chargebee


class _FakeResponse:
    """Only the attributes the connector reads, so a drift shows up as an
    AttributeError instead of a mock that quietly answers everything."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: Any = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text if payload is None else json.dumps(payload)
        self.headers = headers or {}

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class _Recorder:
    def __init__(self, *responses: _FakeResponse) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return self._responses.pop(0)

    @property
    def call(self) -> dict[str, Any]:
        assert len(self.calls) == 1
        return self.calls[0]


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unpatched(**kwargs: Any) -> None:
        raise AssertionError(f"real HTTP request to {kwargs.get('url')!r}")

    monkeypatch.setattr(chargebee._session, "request", _unpatched)


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        chargebee.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(chargebee.time, "sleep", slept.append)
    return slept


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHARGEBEE_SITE", "acme-test")
    monkeypatch.setenv("CHARGEBEE_API_KEY", "test_cbkey123secret")


def _install(monkeypatch: pytest.MonkeyPatch, *responses: _FakeResponse) -> _Recorder:
    recorder = _Recorder(*responses)
    monkeypatch.setattr(chargebee._session, "request", recorder)
    return recorder


def _ok(body: Any, **kwargs: Any) -> _FakeResponse:
    return _FakeResponse(payload=body, **kwargs)


def _payload(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


# ---------------------------------------------------------------------------
# Site name and host pinning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "acme-test",
        "  ACME-Test \n",
        "acme-test.chargebee.com",
        "https://acme-test.chargebee.com",
        "https://acme-test.chargebee.com/",
    ],
)
def test_site_accepts_the_label_or_its_hostname(
    monkeypatch: pytest.MonkeyPatch, value: str
):
    """Users copy the address bar, so the full hostname or URL is accepted,
    but only for the chargebee.com label itself."""
    monkeypatch.setenv("CHARGEBEE_SITE", value)
    assert chargebee._site() == "acme-test"


@pytest.mark.parametrize(
    "value",
    [
        "acme.evil.com",
        "acme.chargebee.com.evil.com",
        "evil.com/acme",
        "acme:8443",
        "user@acme",
        "http://acme.chargebee.com",
        "https://acme.chargebee.com/api/v2",
        "acme.eu.chargebee.com",
        "-acme",
        "acme_test",
        "a" * 64,
        "acme\n.evil",
    ],
)
def test_site_rejects_anything_that_changes_the_host(
    monkeypatch: pytest.MonkeyPatch, value: str
):
    monkeypatch.setenv("CHARGEBEE_SITE", value)
    with pytest.raises(ValueError, match="CHARGEBEE_SITE"):
        chargebee._site()


def test_missing_settings_are_actionable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CHARGEBEE_SITE", raising=False)
    monkeypatch.setenv("CHARGEBEE_API_KEY", "   ")
    with pytest.raises(ValueError, match="CHARGEBEE_SITE"):
        chargebee._site()
    with pytest.raises(ValueError, match="CHARGEBEE_API_KEY"):
        chargebee._api_key()


def test_base_url_is_pinned_to_the_site(configured_env: None):
    assert chargebee._base_url() == "https://acme-test.chargebee.com/api/v2"


def test_base_url_refuses_a_site_that_resolves_privately(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    monkeypatch.setattr(
        chargebee.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("10.0.0.5", 443))],
    )
    with pytest.raises(ValueError, match="not allowed"):
        chargebee._base_url()


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def test_get_uses_basic_auth_with_the_key_as_username_and_no_redirects(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _ok({"customer": {"id": "c1"}}))
    chargebee._get("/customers/c1")
    call = recorder.call
    assert call["method"] == "GET"
    assert call["url"] == "https://acme-test.chargebee.com/api/v2/customers/c1"
    assert call["auth"] == ("test_cbkey123secret", "")
    assert call["allow_redirects"] is False
    assert call["timeout"] == chargebee.DEFAULT_TIMEOUT_SECONDS


def test_get_drops_unset_params(monkeypatch: pytest.MonkeyPatch, configured_env: None):
    recorder = _install(monkeypatch, _ok({"list": []}))
    chargebee._get("/customers", {"email[is]": None, "company[is]": "", "limit": 5})
    assert recorder.call["params"] == {"limit": 5}


def test_redirect_is_refused(monkeypatch: pytest.MonkeyPatch, configured_env: None):
    _install(
        monkeypatch,
        _FakeResponse(status_code=302, headers={"Location": "https://evil.example"}),
    )
    with pytest.raises(RuntimeError, match="redirect"):
        chargebee._get("/customers")


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (
            401,
            {"api_error_code": "api_authentication_failed", "message": "bad key"},
            "CHARGEBEE_API_KEY",
        ),
        (
            403,
            {"api_error_code": "api_authorization_failed", "message": "no access"},
            "permission",
        ),
        (
            404,
            {"api_error_code": "site_not_found", "message": "Site not found"},
            "CHARGEBEE_SITE",
        ),
        (
            404,
            {"api_error_code": "resource_not_found", "message": "Customer x not found"},
            "Customer x not found",
        ),
        (
            503,
            {"api_error_code": "site_read_only_mode", "message": "read only"},
            "read-only",
        ),
        (
            429,
            {"api_error_code": "lock_timeout", "message": "locked"},
            "another request",
        ),
        (
            400,
            {
                "api_error_code": "param_wrong_value",
                "param": "status[is]",
                "message": "status[is] : invalid value",
            },
            "status[is]",
        ),
    ],
)
def test_error_responses_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    status: int,
    body: dict[str, Any],
    expected: str,
):
    _install(monkeypatch, _FakeResponse(status_code=status, payload=body))
    with pytest.raises(RuntimeError) as excinfo:
        chargebee._get("/customers")
    assert str(status) in str(excinfo.value)
    assert expected in str(excinfo.value)


def test_error_text_never_carries_the_key(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """A proxy or WAF error page can echo the request back."""
    _install(
        monkeypatch,
        _FakeResponse(
            status_code=500, text="upstream echo: user test_cbkey123secret rejected"
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        chargebee._get("/customers")
    assert "test_cbkey123secret" not in str(excinfo.value)


@pytest.mark.parametrize(
    "echo",
    [
        "api_key=test_cbkey123secret",
        "Authorization: Basic test_cbkey123secret",
        "token: test_cbkey123secret",
    ],
)
def test_a_credential_shaped_echo_leaks_no_part_of_the_key(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, echo: str
):
    """Generic redaction keeps the last four characters of a credential-shaped
    value; the key has to be removed before it runs."""
    _install(monkeypatch, _FakeResponse(status_code=500, text=f"upstream: {echo}"))
    with pytest.raises(RuntimeError) as excinfo:
        chargebee._get("/customers")
    assert "cret" not in str(excinfo.value)


def test_transport_error_text_is_redacted(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    def _raise(**_: Any) -> None:
        raise requests.ConnectionError(
            "ProxyError https://bob:hunter2@proxy.internal:3128 key test_cbkey123secret"
        )

    monkeypatch.setattr(chargebee._session, "request", _raise)
    with pytest.raises(RuntimeError) as excinfo:
        chargebee._get("/customers")
    assert "hunter2" not in str(excinfo.value)
    assert "test_cbkey123secret" not in str(excinfo.value)


def test_transport_error_text_drops_the_filters_in_the_url(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """List filters carry customer data; a transport error embeds the URL."""

    def _raise(**_: Any) -> None:
        raise requests.ConnectionError(
            "HTTPSConnectionPool(host='acme-test.chargebee.com', port=443): Max "
            "retries exceeded with url: /api/v2/customers?email%5Bis%5D=ada%40"
            "example.com&company%5Bis%5D=Acme (Caused by NewConnectionError)"
        )

    monkeypatch.setattr(chargebee._session, "request", _raise)
    raw = chargebee.chargebee_list_customers(email="ada@example.com")
    assert "ada%40example.com" not in raw
    assert "Acme" not in raw
    assert "/api/v2/customers?<query redacted>" in raw


def test_rate_limit_with_retry_after_waits_and_retries_once(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, _no_real_sleep: list[float]
):
    recorder = _install(
        monkeypatch,
        _FakeResponse(
            status_code=429,
            payload={"api_error_code": "api_request_limit_exceeded", "message": "slow"},
            headers={"Retry-After": "3"},
        ),
        _ok({"customer": {"id": "c1"}}),
    )
    assert chargebee._get("/customers/c1") == {"customer": {"id": "c1"}}
    assert len(recorder.calls) == 2
    assert _no_real_sleep == [3]


def test_rate_limit_is_reported_after_the_one_retry(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, _no_real_sleep: list[float]
):
    limited = _FakeResponse(
        status_code=429,
        payload={"api_error_code": "lock_timeout", "message": "busy"},
        headers={"Retry-After": "2"},
    )
    recorder = _install(monkeypatch, limited, limited)
    with pytest.raises(RuntimeError, match="429"):
        chargebee._get("/customers")
    assert len(recorder.calls) == 2
    assert _no_real_sleep == [2]


@pytest.mark.parametrize("retry_after", [None, "0", "120", "soon", "²"])
def test_rate_limit_without_a_usable_wait_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    _no_real_sleep: list[float],
    retry_after: str | None,
):
    recorder = _install(
        monkeypatch,
        _FakeResponse(
            status_code=429,
            payload={"api_error_code": "api_request_limit_exceeded", "message": "x"},
            headers={"Retry-After": retry_after} if retry_after else {},
        ),
    )
    with pytest.raises(RuntimeError, match="429"):
        chargebee._get("/customers")
    assert len(recorder.calls) == 1
    assert _no_real_sleep == []


def test_non_json_success_is_an_error(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    _install(monkeypatch, _FakeResponse(status_code=200, text="<html>"))
    with pytest.raises(RuntimeError, match="non-JSON"):
        chargebee._get("/customers")


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


_NOV_1 = 1793491200  # 2026-11-01T00:00:00Z
_DAY = 86400


@pytest.mark.parametrize(
    "value, bound, expected",
    [
        # Chargebee's bounds are exclusive; a bare date includes its whole day.
        ("2026-11-01", "after", _NOV_1 - 1),
        ("2026-11-01", "before", _NOV_1 + _DAY),
        # A datetime is sent as given.
        ("2026-11-01T08:00:00+08:00", "after", _NOV_1),
        ("2026-11-01T00:00:00Z", "before", _NOV_1),
    ],
)
def test_timestamp_counts_a_bare_date_as_its_whole_day(
    value: str, bound: str, expected: int
):
    assert chargebee._timestamp(value, "date_after", bound) == expected


@pytest.mark.parametrize("value", ["2026-11-01T08:00:00", "yesterday", "1793491200"])
def test_timestamp_rejects_ambiguous_values(value: str):
    with pytest.raises(ValueError, match="date_after"):
        chargebee._timestamp(value, "date_after", "after")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_LIST_TOOLS = [
    (chargebee.chargebee_list_customers, "/customers", "customer", "customers"),
    (
        chargebee.chargebee_list_subscriptions,
        "/subscriptions",
        "subscription",
        "subscriptions",
    ),
    (chargebee.chargebee_list_invoices, "/invoices", "invoice", "invoices"),
    (
        chargebee.chargebee_list_transactions,
        "/transactions",
        "transaction",
        "transactions",
    ),
]


@pytest.mark.parametrize("tool, path, resource, field", _LIST_TOOLS)
def test_list_tools_unwrap_entries_and_pass_the_next_offset(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    tool: Any,
    path: str,
    resource: str,
    field: str,
):
    """Chargebee wraps each list entry by resource name and pages by an
    opaque next_offset, which the caller passes back unchanged."""
    recorder = _install(
        monkeypatch,
        _ok(
            {
                "list": [{resource: {"id": "a"}, "card": {"id": "side"}}],
                "next_offset": '["1700000000000","42"]',
            }
        ),
    )
    result = _payload(tool(limit=1000, offset='["1690000000000","7"]'))
    assert result["status"] == "success"
    assert result[field] == [{"id": "a"}]
    assert result["next_offset"] == '["1700000000000","42"]'
    assert result["has_more"] is True
    call = recorder.call
    assert call["url"].endswith(f"/api/v2{path}")
    assert call["params"]["limit"] == chargebee.MAX_LIMIT
    assert call["params"]["offset"] == '["1690000000000","7"]'


@pytest.mark.parametrize("tool, path, resource, field", _LIST_TOOLS)
def test_list_tools_report_the_last_page(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    tool: Any,
    path: str,
    resource: str,
    field: str,
):
    recorder = _install(monkeypatch, _ok({"list": []}))
    result = _payload(tool(limit=0))
    assert result[field] == []
    assert result["next_offset"] is None
    assert result["has_more"] is False
    assert recorder.call["params"] == {"limit": 1}


@pytest.mark.parametrize("tool, path, resource, field", _LIST_TOOLS)
def test_include_deleted_is_sent_as_a_lowercase_boolean(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    tool: Any,
    path: str,
    resource: str,
    field: str,
):
    """Python's str(True) is "True", which form-encoded APIs reject or misread
    (the Stripe connector shipped that bug once, #1462)."""
    recorder = _install(monkeypatch, _ok({"list": []}))
    tool(include_deleted=True)
    assert recorder.call["params"]["include_deleted"] == "true"


def test_list_customers_sends_filters_in_chargebee_operator_form(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _ok({"list": []}))
    chargebee.chargebee_list_customers(
        email="a@example.com",
        company="Acme",
        created_after="2026-11-01",
        created_before="2026-12-01",
    )
    params = recorder.call["params"]
    assert params["email[is]"] == "a@example.com"
    assert params["company[is]"] == "Acme"
    assert params["created_at[after]"] == _NOV_1 - 1
    # 2026-12-01 counts as a whole day: up to the start of 12-02.
    assert params["created_at[before]"] == 1796083200 + _DAY


def test_list_subscriptions_filters_by_customer_and_status(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch, _ok({"list": []}))
    chargebee.chargebee_list_subscriptions(customer_id="cust_1", status="active")
    params = recorder.call["params"]
    assert params["customer_id[is]"] == "cust_1"
    assert params["status[is]"] == "active"


@pytest.mark.parametrize(
    "tool",
    [chargebee.chargebee_list_invoices, chargebee.chargebee_list_transactions],
)
def test_invoice_and_transaction_lists_filter_by_date(
    monkeypatch: pytest.MonkeyPatch, configured_env: None, tool: Any
):
    recorder = _install(monkeypatch, _ok({"list": []}))
    tool(
        customer_id="cust_1",
        subscription_id="sub_1",
        status="paid",
        date_after="2026-11-01",
        date_before="2026-12-01",
    )
    params = recorder.call["params"]
    assert params["customer_id[is]"] == "cust_1"
    assert params["subscription_id[is]"] == "sub_1"
    assert params["status[is]"] == "paid"
    assert params["date[after]"] == _NOV_1 - 1
    assert params["date[before]"] == 1796083200 + _DAY


def test_list_tool_reports_a_bad_date_without_sending(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    recorder = _install(monkeypatch)
    result = _payload(chargebee.chargebee_list_invoices(date_after="last week"))
    assert result["status"] == "error"
    assert "date_after" in result["message"]
    assert recorder.calls == []


def test_list_rejects_an_unexpected_envelope(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    _install(monkeypatch, _ok({"customers": []}))
    result = _payload(chargebee.chargebee_list_customers())
    assert result["status"] == "error"


def test_list_halves_an_oversized_page(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    monkeypatch.setattr(chargebee, "get_tool_max_output_length", lambda: 2000)
    entries = [{"customer": {"id": f"c{n}", "notes": "x" * 200}} for n in range(20)]
    _install(monkeypatch, _ok({"list": entries, "next_offset": "next"}))
    raw = chargebee.chargebee_list_customers(limit=20)
    result = _payload(raw)
    assert len(raw) <= 2000
    assert result["truncated"] is True
    kept = len(result["customers"])
    assert 0 < kept < 20
    # The dropped entries are not on the next page; their ids are returned.
    assert result["omitted_ids"] == [f"c{n}" for n in range(kept, 20)]
    assert "smaller limit" in result["message"]


def test_list_stays_valid_json_when_even_the_ids_do_not_fit(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    monkeypatch.setattr(chargebee, "get_tool_max_output_length", lambda: 200)
    entries = [{"customer": {"id": f"cust_{n:040d}"}} for n in range(20)]
    _install(monkeypatch, _ok({"list": entries}))
    raw = chargebee.chargebee_list_customers(limit=20)
    result = _payload(raw)
    assert len(raw) <= 200
    assert result["customers"] == []
    assert result["truncated"] is True
    assert result["omitted_count"] == 20


def test_list_names_a_record_too_large_to_return(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    """No smaller page shows a record larger than the limit; its id lets the
    caller fetch it with the get tool, which trims a single record."""
    monkeypatch.setattr(chargebee, "get_tool_max_output_length", lambda: 1000)
    _install(monkeypatch, _ok({"list": [{"customer": {"id": "big", "n": "x" * 5000}}]}))
    result = _payload(chargebee.chargebee_list_customers(limit=1))
    assert result["customers"] == []
    assert result["truncated"] is True
    assert result["omitted_ids"] == ["big"]


_GET_TOOLS = [
    (chargebee.chargebee_get_customer, "customers", "customer"),
    (chargebee.chargebee_get_subscription, "subscriptions", "subscription"),
    (chargebee.chargebee_get_invoice, "invoices", "invoice"),
    (chargebee.chargebee_get_transaction, "transactions", "transaction"),
]


@pytest.mark.parametrize("tool, path, resource", _GET_TOOLS)
def test_get_tools_fetch_one_resource(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    tool: Any,
    path: str,
    resource: str,
):
    recorder = _install(monkeypatch, _ok({resource: {"id": "abc"}, "card": {}}))
    result = _payload(tool("abc"))
    assert result == {"status": "success", resource: {"id": "abc"}, "truncated": False}
    assert recorder.call["url"].endswith(f"/api/v2/{path}/abc")


@pytest.mark.parametrize("tool, path, resource", _GET_TOOLS)
def test_get_tools_encode_the_id_into_one_path_segment(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    tool: Any,
    path: str,
    resource: str,
):
    recorder = _install(monkeypatch, _ok({resource: {"id": "x"}}))
    tool("a/../b?limit=1")
    assert recorder.call["url"].endswith(f"/{path}/a%2F..%2Fb%3Flimit%3D1")


@pytest.mark.parametrize("tool, path, resource", _GET_TOOLS)
@pytest.mark.parametrize("bad_id", ["", " abc", ".."])
def test_get_tools_reject_a_bad_id_without_sending(
    monkeypatch: pytest.MonkeyPatch,
    configured_env: None,
    tool: Any,
    path: str,
    resource: str,
    bad_id: str,
):
    recorder = _install(monkeypatch)
    result = _payload(tool(bad_id))
    assert result["status"] == "error"
    assert recorder.calls == []


def test_get_rejects_a_body_without_the_resource(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    _install(monkeypatch, _ok({"invoice": {"id": "x"}}))
    result = _payload(chargebee.chargebee_get_customer("x"))
    assert result["status"] == "error"


def test_tool_errors_never_carry_the_key(
    monkeypatch: pytest.MonkeyPatch, configured_env: None
):
    _install(
        monkeypatch,
        _FakeResponse(
            status_code=401,
            payload={
                "api_error_code": "api_authentication_failed",
                "message": "key test_cbkey123secret is invalid",
            },
        ),
    )
    raw = chargebee.chargebee_list_customers()
    assert "test_cbkey123secret" not in raw
    assert _payload(raw)["status"] == "error"


def test_every_tool_is_registered_read_only():
    tools = asyncio.run(chargebee.mcp.list_tools())
    assert {tool.name for tool in tools} == {
        "chargebee_list_customers",
        "chargebee_get_customer",
        "chargebee_list_subscriptions",
        "chargebee_get_subscription",
        "chargebee_list_invoices",
        "chargebee_get_invoice",
        "chargebee_list_transactions",
        "chargebee_get_transaction",
    }
    for tool in tools:
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True, tool.name
