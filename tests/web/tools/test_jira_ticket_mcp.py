"""Tests for the create-only Jira ticket connector.

Mocked against the Jira Cloud REST v2 contract (create issue).
They pin the requests this module builds and how it translates responses; they
cannot show the documented shapes match a live site.
"""

import asyncio
import json
from typing import Any

import pytest
import requests

from xagent.web.tools.mcp import jira_ticket

SITE = "https://acme.atlassian.net"
TOKEN = "tok-SECRET-123"
EMAIL = "intake@example.com"


class _FakeResponse:
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
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.content = self.text.encode()
        self.headers = headers or {}

    def json(self) -> Any:
        if self._payload is None:
            raise requests.exceptions.JSONDecodeError("no JSON body", "", 0)
        return self._payload


class _FakeSession:
    """Records every request and answers from a queue (exceptions are raised)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.queue: list[Any] = []

    def request(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    fake = _FakeSession()
    monkeypatch.setattr(jira_ticket, "_session", fake)
    return fake


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_TICKET_SITE", "acme")
    monkeypatch.setenv("JIRA_TICKET_EMAIL", EMAIL)
    monkeypatch.setenv("JIRA_TICKET_API_TOKEN", TOKEN)
    monkeypatch.setenv("JIRA_TICKET_PROJECT_KEY", "SUP")
    # Resolve every host to a public address by default.
    monkeypatch.setattr(
        jira_ticket.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("104.192.136.1", 443))],
    )


def _created(key: str = "SUP-7") -> _FakeResponse:
    return _FakeResponse(
        status_code=201, payload={"id": "10010", "key": key, "self": "x"}
    )


def _call(**kwargs: Any) -> dict[str, Any]:
    args = {"summary": "Login broken", "description": "Cannot sign in."}
    args.update(kwargs)
    return json.loads(jira_ticket.jira_ticket_create(**args))


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_exposes_only_the_create_tool() -> None:
    tools = asyncio.run(jira_ticket.mcp.list_tools())
    assert [tool.name for tool in tools] == ["jira_ticket_create"]


def test_tool_has_no_argument_that_selects_project_site_or_url() -> None:
    tools = asyncio.run(jira_ticket.mcp.list_tools())
    assert set(tools[0].inputSchema["properties"]) == {
        "summary",
        "description",
        "issue_type",
        "labels",
    }


# ---------------------------------------------------------------------------
# Site validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("acme", SITE),
        ("ACME", SITE),
        (" acme \n", SITE),
        ("acme.atlassian.net", SITE),
        ("https://acme.atlassian.net", SITE),
        ("https://acme.atlassian.net/", SITE),
        ("HTTPS://Acme.Atlassian.Net", SITE),
        ("my-team-1", "https://my-team-1.atlassian.net"),
    ],
)
def test_site_is_normalized(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: str
) -> None:
    monkeypatch.setenv("JIRA_TICKET_SITE", value)
    assert jira_ticket._site() == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "evil.com",
        "acme.example.com",
        "acme.atlassian.net.evil.com",
        "evil.com/acme.atlassian.net",
        "http://acme.atlassian.net",
        "https://acme.atlassian.net/path",
        "https://acme.atlassian.net/?a=1",
        "https://acme.atlassian.net#frag",
        "acme.atlassian.net:8443",
        "user@acme.atlassian.net",
        "https://user:pw@acme.atlassian.net",
        "a.b.atlassian.net",
        "atlassian.net",
        ".atlassian.net",
        "-acme",
        "acme-",
        "ac me",
        "acme_x",
        "a" * 64,
        "127.0.0.1",
    ],
)
def test_site_rejects_anything_but_one_label_under_atlassian_net(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("JIRA_TICKET_SITE", value)
    with pytest.raises(ValueError, match="JIRA_TICKET_SITE"):
        jira_ticket._site()


def test_site_rejects_private_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        jira_ticket.socket,
        "getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("10.0.0.5", 443))],
    )
    with pytest.raises(ValueError, match="JIRA_TICKET_SITE"):
        jira_ticket._base_url()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["SUP", "AB", "A1_B", "ABCDEFGHIJ"])
def test_project_key_accepts_jira_keys(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("JIRA_TICKET_PROJECT_KEY", f" {value} ")
    assert jira_ticket._project_key() == value


@pytest.mark.parametrize(
    "value", ["", "  ", "sup", "S", "1SUP", "SUP-1", "ABCDEFGHIJK", "SU P", "SUP\nX"]
)
def test_project_key_rejects_non_keys(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("JIRA_TICKET_PROJECT_KEY", value)
    with pytest.raises(ValueError, match="JIRA_TICKET_PROJECT_KEY"):
        jira_ticket._project_key()


@pytest.mark.parametrize("name", ["JIRA_TICKET_EMAIL", "JIRA_TICKET_API_TOKEN"])
def test_missing_credentials_name_the_setting(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "  ")
    with pytest.raises(ValueError, match=name):
        jira_ticket._credentials()


def test_email_must_look_like_an_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JIRA_TICKET_EMAIL", "not-an-email")
    with pytest.raises(ValueError, match="JIRA_TICKET_EMAIL"):
        jira_ticket._credentials()


def test_bad_config_is_a_tool_error_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, session: _FakeSession
) -> None:
    monkeypatch.setenv("JIRA_TICKET_PROJECT_KEY", "nope")
    result = _call()
    assert result["status"] == "error"
    assert "JIRA_TICKET_PROJECT_KEY" in result["message"]
    assert session.calls == []


# ---------------------------------------------------------------------------
# Create request shape
# ---------------------------------------------------------------------------


def test_create_request_shape_and_result(session: _FakeSession) -> None:
    session.queue = [_created("SUP-7")]
    result = _call(summary="  Login broken  ", description="Cannot sign in.")

    assert result == {
        "status": "success",
        "key": "SUP-7",
        "url": f"{SITE}/browse/SUP-7",
    }
    (call,) = session.calls
    assert call["method"] == "POST"
    assert call["url"] == f"{SITE}/rest/api/2/issue"
    assert call["auth"] == (EMAIL, TOKEN)
    assert call["allow_redirects"] is False
    assert call["timeout"] == jira_ticket.CREATE_TIMEOUT_SECONDS
    assert call["json"] == {
        "fields": {
            "project": {"key": "SUP"},
            "summary": "Login broken",
            "description": "Cannot sign in.",
            "issuetype": {"name": "Task"},
        }
    }


def test_issue_type_and_labels_are_forwarded(session: _FakeSession) -> None:
    session.queue = [_created()]
    _call(issue_type=" Bug ", labels=["web", "guest-intake"])
    assert session.calls[0]["json"]["fields"]["issuetype"] == {"name": "Bug"}
    assert session.calls[0]["json"]["fields"]["labels"] == ["web", "guest-intake"]


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"summary": "   "}, "summary"),
        ({"summary": "x" * 256}, "255"),
        ({"description": "  "}, "description"),
        ({"description": "x" * 32001}, "32000"),
        ({"labels": ["has space"]}, "label"),
        ({"labels": [""]}, "label"),
        ({"labels": [5]}, "label"),
        ({"labels": ["a"] * 11}, "10"),
    ],
)
def test_invalid_arguments_are_rejected_before_any_request(
    session: _FakeSession, kwargs: dict[str, Any], fragment: str
) -> None:
    result = _call(**kwargs)
    assert result["status"] == "error"
    assert fragment in result["message"]
    assert session.calls == []


def test_summary_at_the_limit_is_accepted(session: _FakeSession) -> None:
    session.queue = [_created()]
    assert _call(summary="x" * 255)["status"] == "success"


def test_create_response_without_a_usable_key_is_an_error(
    session: _FakeSession,
) -> None:
    session.queue = [_FakeResponse(status_code=201, payload={"id": "1"})]
    result = _call()
    assert result["status"] == "error"
    assert "key" in result["message"]


def test_create_response_with_a_path_like_key_is_refused(
    session: _FakeSession,
) -> None:
    session.queue = [_created("../../SUP-1")]
    result = _call()
    assert result["status"] == "error"
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, fragment",
    [
        (401, "JIRA_TICKET_EMAIL"),
        (403, "Create Issues"),
        (404, "project or site"),
        (429, "rate limit"),
        (500, "500"),
        (503, "503"),
    ],
)
def test_http_errors_map_to_short_messages(
    session: _FakeSession, status: int, fragment: str
) -> None:
    body = f"<html>raw body sentinel {TOKEN}</html>"
    session.queue = [_FakeResponse(status_code=status, text=body)]
    result = _call()
    assert result["status"] == "error"
    assert fragment in result["message"]
    assert "sentinel" not in result["message"]
    assert TOKEN not in result["message"]


def test_rate_limit_surfaces_retry_after(session: _FakeSession) -> None:
    session.queue = [
        _FakeResponse(status_code=429, text="slow down", headers={"Retry-After": "12"})
    ]
    assert "12" in _call()["message"]


@pytest.mark.parametrize("status", [500, 501, 502, 503, 504, 599])
def test_gateway_error_on_create_says_the_ticket_may_exist(
    session: _FakeSession, status: int
) -> None:
    session.queue = [_FakeResponse(status_code=status, text="gw")]
    message = _call()["message"]
    assert "may have been created" in message
    assert "check the project" in message
    assert "try again" not in message.lower()


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.ReadTimeout("slow"),
        requests.exceptions.ChunkedEncodingError("cut"),
        requests.exceptions.ConnectionError("reset"),
        requests.exceptions.ContentDecodingError("bad gzip"),
    ],
)
def test_post_send_failure_on_create_says_the_ticket_may_exist(
    session: _FakeSession, exc: Exception
) -> None:
    session.queue = [exc]
    message = _call()["message"]
    assert "may have been created" in message
    assert "try again" not in message.lower()


def test_connect_timeout_on_create_is_safe_to_retry(session: _FakeSession) -> None:
    session.queue = [requests.exceptions.ConnectTimeout("no route")]
    message = _call()["message"]
    assert "may have been created" not in message


def test_create_advice_survives_a_very_long_underlying_message(
    session: _FakeSession,
) -> None:
    session.queue = [requests.exceptions.ReadTimeout("x" * 5000)]
    message = _call()["message"]
    assert message.startswith("The ticket may have been created")
    assert "check the project" in message
    assert len(message) < 1500


def test_rate_limit_on_create_is_not_outcome_unknown(session: _FakeSession) -> None:
    session.queue = [_FakeResponse(status_code=429, text="slow")]
    assert "may have been created" not in _call()["message"]


def test_one_dns_check_per_call(
    session: _FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookups: list[Any] = []

    def _resolve(*a: Any, **k: Any) -> list[Any]:
        lookups.append(a)
        return [(2, 1, 6, "", ("104.192.136.1", 443))]

    monkeypatch.setattr(jira_ticket.socket, "getaddrinfo", _resolve)
    session.queue = [_created()]
    assert _call()["status"] == "success"
    assert len(lookups) == 1


def test_400_names_the_offending_fields(session: _FakeSession) -> None:
    session.queue = [
        _FakeResponse(
            status_code=400,
            payload={
                "errorMessages": ["Project is required"],
                "errors": {"issuetype": "Specify a valid issue type"},
            },
        )
    ]
    message = _call(issue_type="Nope")["message"]
    assert "issuetype" in message
    assert "Specify a valid issue type" in message
    assert "Project is required" in message


def test_400_does_not_dump_unstructured_bodies(session: _FakeSession) -> None:
    session.queue = [_FakeResponse(status_code=400, text="<html>" + "x" * 5000)]
    message = _call()["message"]
    assert "<html>" not in message
    assert len(message) < 400


def test_400_message_is_capped(session: _FakeSession) -> None:
    session.queue = [
        _FakeResponse(
            status_code=400,
            payload={"errors": {f"f{i}": "y" * 1000 for i in range(50)}},
        )
    ]
    assert len(_call()["message"]) < 1500


def test_redirect_is_refused(session: _FakeSession) -> None:
    session.queue = [
        _FakeResponse(status_code=302, headers={"Location": "https://evil.example/"})
    ]
    message = _call()["message"]
    assert "redirect" in message
    assert "evil.example" not in message


def test_transport_error_never_echoes_the_token(session: _FakeSession) -> None:
    session.queue = [
        requests.ConnectionError(f"proxy http://u:{TOKEN}@proxy.local:8080 refused")
    ]
    result = _call()
    assert result["status"] == "error"
    assert TOKEN not in result["message"]


def test_non_json_success_body_is_an_error(session: _FakeSession) -> None:
    session.queue = [_FakeResponse(status_code=201, text="<html>ok</html>")]
    result = _call()
    assert result["status"] == "error"
    assert "<html>" not in result["message"]
