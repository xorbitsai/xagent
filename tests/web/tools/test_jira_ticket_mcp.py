"""Tests for the create-only Jira ticket connector.

Mocked against the Jira Cloud REST v3 contract (create issue).
They pin the requests this module builds and how it translates responses; they
cannot show the documented shapes match a live site.
"""

import asyncio
import json
from typing import Any

import pytest
import requests
from urllib3.exceptions import (
    ConnectTimeoutError,
    MaxRetryError,
    NameResolutionError,
    NewConnectionError,
    ProtocolError,
    ReadTimeoutError,
)

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


def test_module_uses_the_public_requests_facade() -> None:
    from xagent.core.utils import public_requests

    assert jira_ticket.requests is public_requests
    assert isinstance(jira_ticket._session, public_requests.Session)


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
    assert call["url"] == f"{SITE}/rest/api/3/issue"
    assert call["headers"] == {"Accept": "application/json"}
    assert call["auth"] == (EMAIL, TOKEN)
    assert call["allow_redirects"] is False
    assert call["timeout"] == jira_ticket.CREATE_TIMEOUT_SECONDS
    assert call["json"] == {
        "fields": {
            "project": {"key": "SUP"},
            "summary": "Login broken",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Cannot sign in."}],
                    }
                ],
            },
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
        (401, "classic (unscoped)"),
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
    assert "may already have been created" in message
    assert "do not call jira_ticket_create again" in message
    assert "try again" not in message.lower()


_URL = f"{SITE}/rest/api/3/issue"


def _maxretry(reason: Exception) -> requests.ConnectionError:
    return requests.ConnectionError(MaxRetryError(None, _URL, reason=reason))  # type: ignore[arg-type]


def _facade_dns_error() -> requests.ConnectionError:
    try:
        raise OSError("Name or service not known")
    except OSError as cause:
        try:
            raise requests.ConnectionError(
                "Public API host could not be resolved"
            ) from cause
        except requests.ConnectionError as exc:
            return exc


_NOTHING_SENT = [
    requests.exceptions.ConnectTimeout("no route"),
    requests.exceptions.InvalidURL("private address"),
    requests.exceptions.ProxyError("proxy refused"),
    _maxretry(NewConnectionError(None, "refused")),  # type: ignore[arg-type]
    _maxretry(ConnectTimeoutError(None, "connect timed out")),  # type: ignore[arg-type]
    _maxretry(NameResolutionError("acme.atlassian.net", None, OSError("dns"))),  # type: ignore[arg-type]
    _facade_dns_error(),
]

_OUTCOME_UNKNOWN = [
    requests.exceptions.ReadTimeout("slow"),
    requests.exceptions.ChunkedEncodingError("cut"),
    requests.exceptions.ConnectionError("reset"),
    requests.exceptions.ContentDecodingError("bad gzip"),
    requests.exceptions.SSLError("handshake"),
    _maxretry(ProtocolError("connection aborted")),
    _maxretry(ReadTimeoutError(None, _URL, "read timed out")),  # type: ignore[arg-type]
    _maxretry(None),  # type: ignore[arg-type]
]


@pytest.mark.parametrize("exc", _OUTCOME_UNKNOWN)
def test_post_send_failure_on_create_says_the_ticket_may_exist(
    session: _FakeSession, exc: Exception
) -> None:
    session.queue = [exc]
    message = _call()["message"]
    assert "may already have been created" in message
    assert "try again" not in message.lower()


@pytest.mark.parametrize("exc", _NOTHING_SENT)
def test_failure_before_anything_was_sent_is_not_outcome_unknown(
    session: _FakeSession, exc: Exception
) -> None:
    session.queue = [exc]
    result = _call()
    assert result["status"] == "error"
    assert result["message"].startswith("Jira request failed")
    assert "may already have been created" not in result["message"]
    assert TOKEN not in result["message"]


@pytest.mark.parametrize(
    "failure",
    [OSError("Name or service not known"), ValueError("private address")],
    ids=["unresolvable-host", "private-address"],
)
def test_real_facade_refusals_before_sending_are_not_outcome_unknown(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """Through the real public_requests session: its adapter refuses before
    connecting, and the connector must not call that a possibly-created
    ticket."""
    from xagent.core.utils import public_requests

    def refuse(url: str) -> list[str]:
        raise failure

    monkeypatch.setattr(public_requests, "validate_public_http_url_sync", refuse)
    result = _call()
    assert result["status"] == "error"
    assert result["message"].startswith("Jira request failed")
    assert "may already have been created" not in result["message"]
    assert TOKEN not in result["message"]


@pytest.mark.parametrize("exc", _NOTHING_SENT)
def test_nothing_sent_failures_raise_the_dedicated_error(
    session: _FakeSession, exc: Exception
) -> None:
    session.queue = [exc]
    with pytest.raises(jira_ticket._NothingSentError):
        jira_ticket._send(SITE, "POST", "/rest/api/3/issue", json_data={})
    assert jira_ticket._outcome_unknown(jira_ticket._NothingSentError("x")) is False


@pytest.mark.parametrize("exc", _OUTCOME_UNKNOWN)
def test_other_transport_failures_stay_request_exceptions(
    session: _FakeSession, exc: Exception
) -> None:
    session.queue = [exc]
    with pytest.raises(requests.RequestException) as caught:
        jira_ticket._send(SITE, "POST", "/rest/api/3/issue", json_data={})
    assert not isinstance(caught.value, jira_ticket._NothingSentError)
    assert jira_ticket._outcome_unknown(caught.value) is True


def test_summary_newlines_and_whitespace_runs_are_collapsed(
    session: _FakeSession,
) -> None:
    session.queue = [_created()]
    _call(summary="  Login\r\nbroken\n\n on \t mobile \r")
    assert session.calls[0]["json"]["fields"]["summary"] == "Login broken on mobile"


def test_description_keeps_its_newlines_in_the_adf_body(
    session: _FakeSession,
) -> None:
    session.queue = [_created()]
    _call(description="one\ntwo")
    content = session.calls[0]["json"]["fields"]["description"]["content"]
    assert content == [
        {
            "type": "paragraph",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "hardBreak"},
                {"type": "text", "text": "two"},
            ],
        }
    ]


@pytest.mark.parametrize(
    "text",
    [
        "[x|https://e.test]",
        "!https://e.test/p.png!",
        "{noformat}",
        "*bold* _it_ {code}x{code} [~user]",
    ],
)
def test_markup_looking_description_arrives_verbatim(text: str) -> None:
    assert jira_ticket._adf_document(text) == {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}],
    }


def test_adf_paragraphs_and_hard_breaks() -> None:
    doc = jira_ticket._adf_document("a\r\nb\r\r\nc\n \n\n\nd\ne")
    paragraphs = doc["content"]
    assert [p["type"] for p in paragraphs] == ["paragraph"] * 3
    assert paragraphs[0]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "hardBreak"},
        {"type": "text", "text": "b"},
    ]
    assert paragraphs[1]["content"] == [{"type": "text", "text": "c"}]
    assert paragraphs[2]["content"] == [
        {"type": "text", "text": "d"},
        {"type": "hardBreak"},
        {"type": "text", "text": "e"},
    ]


def test_adf_never_emits_an_empty_text_node() -> None:
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "text":
                assert node["text"] != ""
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for text in ["x\n\n\n\ny", "\nx\n", "a\n   \nb", "  \n x"]:
        walk(jira_ticket._adf_document(text))


def test_create_advice_survives_a_very_long_underlying_message(
    session: _FakeSession,
) -> None:
    session.queue = [requests.exceptions.ReadTimeout("x" * 5000)]
    message = _call()["message"]
    assert message.startswith("The ticket may already have been created")
    assert "do not call jira_ticket_create again" in message
    assert len(message) < 1500


def test_rate_limit_on_create_is_not_outcome_unknown(session: _FakeSession) -> None:
    session.queue = [_FakeResponse(status_code=429, text="slow")]
    assert "may already have been created" not in _call()["message"]


def test_base_url_resolves_the_site_once_per_call(
    session: _FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Counts only _base_url's pre-check: the session fixture replaces the
    # facade, whose own resolve on connect is outside this test.
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


def test_400_field_error_never_echoes_the_token(session: _FakeSession) -> None:
    """The bare token (no URL userinfo for redact_sensitive_text to catch) in
    a field error Jira echoes back is removed by _safe_text's own replace."""
    session.queue = [
        _FakeResponse(
            status_code=400,
            payload={"errorMessages": [], "errors": {"summary": f"bad {TOKEN}"}},
        )
    ]
    result = _call()
    assert result["status"] == "error"
    assert "summary" in result["message"]
    assert TOKEN not in result["message"]


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
