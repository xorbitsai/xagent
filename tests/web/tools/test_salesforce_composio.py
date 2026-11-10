"""Salesforce tools on a Composio-backed grant (#3646).

Such a grant holds no Salesforce token and no instance URL: the subprocess is
launched with the Composio ids only (``tools/config._composio_grant_env``),
and every call goes through Composio's Proxy Execute as an absolute URL on
the org's host. That host is read once from the connected account's
``base_url``, which Composio keeps as the org's host plus
``/services/data/vNN.0`` -- a relative path would be appended to it.
Composio is faked here; no request leaves the process.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.web.tools import config as tools_config
from xagent.web.tools.mcp import composio_proxy, salesforce

API_KEY = "test-composio-api-key-value"
ACCOUNT_ID = "ca_salesforce_account_1"
COMPOSIO_USER = "xagent-0123456789abcdef"
ORG = "https://acme.my.salesforce.com"
# What Composio stores for a live Salesforce connection (#3646 live test).
BASE_URL = f"{ORG}/services/data/v67.0"
API = f"/services/data/{salesforce.API_VERSION}"
ACCOUNT_URL = f"https://backend.composio.dev/api/v3.1/connected_accounts/{ACCOUNT_ID}"


class _Response:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeProxy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.answers: list[_Response] = []
        self.lookups: list[dict[str, Any]] = []
        self.account = _Response(
            200, {"id": ACCOUNT_ID, "data": {"base_url": BASE_URL}}
        )

    def answer(self, upstream_status: int, data: Any = None) -> None:
        self.answers.append(_Response(200, {"data": data, "status": upstream_status}))

    def request(self, **kwargs: Any) -> _Response:
        # salesforce and composio_proxy share one requests module, so this
        # one fake sees every call either makes.
        if kwargs["url"] == ACCOUNT_URL:
            self.lookups.append(kwargs)
            return self.account
        if kwargs["url"] != composio_proxy.COMPOSIO_PROXY_URL:
            raise AssertionError("a Composio grant must never call Salesforce directly")
        self.calls.append(kwargs["json"])
        return self.answers.pop(0)


@pytest.fixture
def proxy(monkeypatch) -> FakeProxy:
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    # A Composio grant launches without these.
    monkeypatch.delenv("SALESFORCE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("SALESFORCE_INSTANCE_URL", raising=False)
    fake = FakeProxy()
    monkeypatch.setattr(composio_proxy.requests, "request", fake.request)
    monkeypatch.setattr(salesforce, "_COMPOSIO_ORG_ORIGINS", {})
    return fake


def _secrets_in(text: str) -> list[str]:
    return [value for value in (API_KEY, ACCOUNT_ID, COMPOSIO_USER) if value in text]


def test_a_query_goes_through_the_proxy_as_a_url_on_the_orgs_host(proxy):
    proxy.answer(
        200,
        {"totalSize": 1, "done": True, "records": [{"Id": "001xx", "Name": "Acme"}]},
    )

    result = json.loads(salesforce.salesforce_query("SELECT Id, Name FROM Account"))

    assert result["status"] == "success"
    assert result["records"] == [{"Id": "001xx", "Name": "Acme"}]
    (call,) = proxy.calls
    assert call["endpoint"] == f"{ORG}{API}/query"
    assert call["method"] == "GET"
    assert call["connected_account_id"] == ACCOUNT_ID
    assert call["parameters"] == [
        {"name": "q", "value": "SELECT Id, Name FROM Account", "type": "query"}
    ]
    # Composio adds the token; we never send one.
    assert all(p["type"] != "header" for p in call["parameters"])


def test_the_current_user_is_read_from_the_orgs_own_userinfo(proxy):
    """login.salesforce.com does not serve sandbox orgs; the org's own host
    does, and Composio knows it."""
    proxy.answer(
        200,
        {
            "user_id": "005xx0000012345AAA",
            "organization_id": "00Dxx0000001234EAA",
            "name": "Ada Admin",
            "email": "ada@example.com",
            "preferred_username": "ada@acme.example",
        },
    )

    result = json.loads(salesforce.salesforce_get_current_user())

    assert result["status"] == "success"
    assert result["user"]["user_id"] == "005xx0000012345AAA"
    (call,) = proxy.calls
    assert call["endpoint"] == f"{ORG}/services/oauth2/userinfo"


def test_a_record_is_created_with_its_fields_as_the_body(proxy):
    proxy.answer(201, {"id": "001xx0000099999AAA", "success": True, "errors": []})

    result = json.loads(
        salesforce.salesforce_create_record("Account", {"Name": "Acme Corp"})
    )

    assert result == {"status": "success", "id": "001xx0000099999AAA", "success": True}
    (call,) = proxy.calls
    assert (call["method"], call["endpoint"]) == (
        "POST",
        f"{ORG}{API}/sobjects/Account",
    )
    assert call["body"] == {"Name": "Acme Corp"}


@pytest.mark.parametrize("no_content", [None, ""])
def test_an_update_answered_with_no_content_succeeds(proxy, no_content):
    """Composio relays a 204's empty body as ""."""
    proxy.answer(204, no_content)

    result = json.loads(
        salesforce.salesforce_update_record(
            "Account", "001xx0000099999AAA", {"Industry": "Finance"}
        )
    )

    assert result == {"status": "success", "id": "001xx0000099999AAA"}
    (call,) = proxy.calls
    assert call["method"] == "PATCH"
    assert call["endpoint"] == f"{ORG}{API}/sobjects/Account/001xx0000099999AAA"


def test_salesforces_error_array_reaches_the_model(proxy):
    proxy.answer(
        400,
        [
            {"errorCode": "INVALID_FIELD", "message": "No such column 'Foo'"},
            {"errorCode": "MALFORMED_QUERY", "message": ""},
        ],
    )

    result = json.loads(salesforce.salesforce_query("SELECT Foo FROM Account"))

    assert result["status"] == "error"
    assert result["message"] == (
        "Salesforce API error (status 400): No such column 'Foo'; MALFORMED_QUERY"
    )


def test_an_error_that_is_not_an_array_is_shown_bounded(proxy):
    proxy.answer(502, {"detail": "x" * (salesforce.MAX_ERROR_RESPONSE_TEXT_CHARS * 2)})

    result = json.loads(salesforce.salesforce_query("SELECT Id FROM Account"))

    assert result["status"] == "error"
    assert result["message"].startswith("Salesforce API error (status 502): ")
    assert result["message"].endswith("... [truncated]")
    assert len(result["message"]) == len(
        "Salesforce API error (status 502): "
    ) + salesforce.MAX_ERROR_RESPONSE_TEXT_CHARS + len("... [truncated]")


def test_a_composio_refusal_is_an_error_without_secrets(proxy):
    echoed = f"account {ACCOUNT_ID} of {COMPOSIO_USER} with key {API_KEY}"
    proxy.answers.append(
        _Response(
            404, {"error": {"slug": "ConnectedAccount_NotFound", "message": echoed}}
        )
    )

    result = json.loads(salesforce.salesforce_query("SELECT Id FROM Account"))

    assert result["status"] == "error"
    assert "ConnectedAccount_NotFound" in result["message"]
    assert _secrets_in(result["message"]) == []


def test_a_native_grant_still_calls_its_org_directly(monkeypatch):
    for name in (
        composio_proxy.COMPOSIO_API_KEY_ENV,
        composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
        composio_proxy.COMPOSIO_USER_ID_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SALESFORCE_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", "https://acme.my.salesforce.com")
    sent: list[dict[str, Any]] = []

    class _Direct:
        status_code = 200
        content = b"{}"

        def json(self) -> Any:
            return {"user_id": "005xx0000012345AAA"}

    def direct(**kwargs: Any) -> _Direct:
        if kwargs["url"] == composio_proxy.COMPOSIO_PROXY_URL:
            raise AssertionError("a native grant must never go through Composio")
        sent.append(kwargs)
        return _Direct()

    monkeypatch.setattr(salesforce.requests, "request", direct)

    json.loads(salesforce.salesforce_get_current_user())

    (request,) = sent
    assert request["url"] == salesforce.USERINFO_URL
    assert request["headers"]["Authorization"] == "Bearer access-token"


def test_a_composio_salesforce_grant_launches_with_the_composio_ids(monkeypatch):
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)

    env = tools_config._composio_grant_env(
        app_id="salesforce",
        connected_account_id=ACCOUNT_ID,
        composio_user_id=COMPOSIO_USER,
    )

    assert env == {
        composio_proxy.COMPOSIO_API_KEY_ENV: API_KEY,
        composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV: ACCOUNT_ID,
        composio_proxy.COMPOSIO_USER_ID_ENV: COMPOSIO_USER,
    }


def test_a_composio_id_echoed_in_an_error_is_redacted(proxy):
    proxy.answer(
        403,
        [{"errorCode": "INSUFFICIENT_ACCESS", "message": f"denied for {ACCOUNT_ID}"}],
    )

    result = json.loads(salesforce.salesforce_query("SELECT Id FROM Account"))

    assert result["status"] == "error"
    assert "INSUFFICIENT_ACCESS" not in result["message"]  # the message wins
    assert "denied for" in result["message"]
    assert _secrets_in(result["message"]) == []


def test_multiple_choices_is_returned_as_data(proxy):
    """requests does not follow 300, so the direct path returns its body as
    data; the proxy path does the same."""
    proxy.answer(300, [{"id": "001xx", "url": "/services/data/v59.0/x"}])

    result = salesforce._request("GET", f"{API}/sobjects/Account/ExtId__c/x")

    assert result == [{"id": "001xx", "url": "/services/data/v59.0/x"}]


@pytest.mark.parametrize("empty", [None, ""])
def test_an_empty_answer_reads_as_an_empty_object(proxy, empty):
    """As on the direct path, which returns {} for an empty body."""
    proxy.answer(204, empty)

    assert salesforce._request("DELETE", f"{API}/sobjects/Account/001xx") == {}


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_composio_did_not_follow_is_an_error(proxy, status):
    """The direct path follows redirects; an unfollowed one through Composio
    must not read as an empty, successful result."""
    proxy.answer(status, None)

    result = json.loads(salesforce.salesforce_query("SELECT Id FROM Account"))

    assert result["message"] == (
        f"Salesforce API error (status {status}): "
        "the request was redirected and the redirect not followed"
    )


def test_an_echoed_id_cut_by_the_bound_is_redacted_whole(proxy):
    """The id straddles the cut: cutting before redacting would leave a prefix
    of it in the message."""
    kept = 8
    padding = "x" * (salesforce.MAX_ERROR_RESPONSE_TEXT_CHARS - kept)
    proxy.answer(403, [{"errorCode": "DENIED", "message": padding + ACCOUNT_ID}])

    result = json.loads(salesforce.salesforce_query("SELECT Id FROM Account"))

    assert result["status"] == "error"
    assert result["message"].endswith("... [truncated]")
    assert ACCOUNT_ID[:kept] not in result["message"]


def test_a_3xx_requests_would_not_follow_is_not_turned_into_a_redirect_error(proxy):
    """Only the redirects requests follows are errors; 304 is not one."""
    proxy.answer(304, None)

    assert salesforce._request("GET", f"{API}/sobjects") == {}


def test_the_orgs_host_is_read_from_composio_once(proxy):
    proxy.answer(200, {"totalSize": 0, "done": True, "records": []})
    proxy.answer(200, {"totalSize": 0, "done": True, "records": []})

    salesforce.salesforce_query("SELECT Id FROM Account")
    salesforce.salesforce_query("SELECT Id FROM Contact")

    (lookup,) = proxy.lookups
    assert lookup["method"] == "GET"
    assert lookup["headers"]["x-api-key"] == API_KEY
    assert [call["endpoint"] for call in proxy.calls] == [f"{ORG}{API}/query"] * 2


@pytest.mark.parametrize(
    "base_url",
    [
        "https://evil.example.com/services/data/v67.0",
        "http://acme.my.salesforce.com/services/data/v67.0",
        "https://acme.my.salesforce.com.evil.example/services/data/v67.0",
        "",
        None,
    ],
)
def test_a_base_url_that_is_not_a_salesforce_host_sends_nothing(proxy, base_url):
    proxy.account = _Response(200, {"id": ACCOUNT_ID, "data": {"base_url": base_url}})

    result = json.loads(salesforce.salesforce_query("SELECT Id FROM Account"))

    assert result["status"] == "error"
    assert "not a valid Salesforce host" in result["message"]
    assert proxy.calls == []


def test_a_failed_lookup_is_an_error_without_secrets(proxy):
    echoed = f"account {ACCOUNT_ID} with key {API_KEY}"
    proxy.account = _Response(
        403, {"error": {"slug": "APIKey_InsufficientPermissions", "message": echoed}}
    )

    result = json.loads(salesforce.salesforce_query("SELECT Id FROM Account"))

    assert result["status"] == "error"
    assert "APIKey_InsufficientPermissions" in result["message"]
    assert _secrets_in(result["message"]) == []
    assert proxy.calls == []


def test_a_failed_lookup_is_retried_on_the_next_call(proxy):
    proxy.account = _Response(503, None)
    salesforce.salesforce_query("SELECT Id FROM Account")
    proxy.account = _Response(200, {"id": ACCOUNT_ID, "data": {"base_url": BASE_URL}})
    proxy.answer(200, {"totalSize": 0, "done": True, "records": []})

    result = json.loads(salesforce.salesforce_query("SELECT Id FROM Account"))

    assert result["status"] == "success"
    assert len(proxy.lookups) == 2
