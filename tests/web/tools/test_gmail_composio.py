"""Gmail tools on a Composio-backed grant.

Such a grant holds no Google token: the tool process is launched with the
Composio ids instead (``tools/config._composio_grant_env``), the service is
built on the proxy transport, and every call goes through Composio's Proxy
Execute endpoint. Composio is faked here; no request leaves the process.
"""

from __future__ import annotations

import base64
import json
from email import policy
from email.parser import BytesParser
from typing import Any
from unittest.mock import Mock

import httplib2
import pytest
from googleapiclient.discovery import build

from xagent.web.tools import config as tools_config
from xagent.web.tools.mcp import composio_proxy, gmail, google_api_proxy

API_KEY = "test-composio-api-key-value"
ACCOUNT_ID = "ca_gmail_account_1"
COMPOSIO_USER = "xagent-0123456789abcdef"

_MESSAGE = {
    "id": "m1",
    "threadId": "t1",
    "snippet": "hello there",
    "payload": {
        "headers": [
            {"name": "Subject", "value": "Hello"},
            {"name": "From", "value": "sender@example.com"},
        ]
    },
}
_NOT_FOUND = {
    "error": {
        "code": 404,
        "message": "Requested entity was not found.",
        "errors": [{"message": "Not Found", "domain": "global"}],
    }
}
_BATCH_BOUNDARY = "gmail-tools-batch"


class _Response:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeProxy:
    """Answers by endpoint substring, checked in insertion order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.by_endpoint: list[tuple[str, _Response]] = []
        self.default = _Response(200, {"status": 200, "data": {}})

    def answer(self, needle: str, status: int, data: Any = None) -> None:
        self.by_endpoint.append(
            (needle, _Response(200, {"status": status, "data": data}))
        )

    def request(self, **kwargs: Any) -> _Response:
        assert kwargs["url"] == composio_proxy.COMPOSIO_PROXY_URL
        payload = kwargs["json"]
        self.calls.append(payload)
        for needle, response in self.by_endpoint:
            if needle in payload["endpoint"]:
                return response
        return self.default


@pytest.fixture
def composio_env(monkeypatch) -> None:
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    # A Composio grant launches without these.
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)


@pytest.fixture
def proxy(monkeypatch, composio_env) -> FakeProxy:
    fake = FakeProxy()
    monkeypatch.setattr(composio_proxy.requests, "request", fake.request)
    return fake


def _proxy_service() -> Any:
    return build(
        "gmail",
        "v1",
        http=google_api_proxy.GoogleApiProxyHttp(
            allowed_hosts=("gmail.googleapis.com",)
        ),
        static_discovery=True,
    )


def _secrets_in(text: str) -> list[str]:
    return [value for value in (API_KEY, ACCOUNT_ID, COMPOSIO_USER) if value in text]


def _batch_part(request_id: str, status: int, body: str) -> str:
    return (
        f"--{_BATCH_BOUNDARY}\r\n"
        "Content-Type: application/http\r\n"
        "Content-Transfer-Encoding: binary\r\n"
        f"Content-ID: <base + {request_id}>\r\n"
        "\r\n"
        f"HTTP/1.1 {status} OK\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "\r\n"
        f"{body}\r\n"
    )


class FakeNativeHttp:
    """A native grant's transport: one JSON answer, or a batch answer."""

    def __init__(
        self, list_body: dict[str, Any], parts: list[tuple[str, int, str]]
    ) -> None:
        self._list_body = json.dumps(list_body)
        self._batch_body = (
            "".join(_batch_part(*part) for part in parts) + f"--{_BATCH_BOUNDARY}--\r\n"
        )

    def request(self, uri, method=None, body=None, headers=None):
        if "batch" in uri:
            response = httplib2.Response(
                {"content-type": f'multipart/mixed; boundary="{_BATCH_BOUNDARY}"'}
            )
            return response, self._batch_body.encode("utf-8")
        response = httplib2.Response({"content-type": "application/json"})
        return response, self._list_body.encode("utf-8")


def test_a_composio_grant_builds_the_service_on_the_proxy_transport(
    monkeypatch, composio_env
):
    built: dict[str, Any] = {}

    def fake_build(*args: Any, **kwargs: Any) -> Any:
        built["args"] = args
        built["kwargs"] = kwargs
        return Mock()

    monkeypatch.setattr(gmail, "build", fake_build)

    gmail.get_gmail_service()

    assert built["args"] == ("gmail", "v1")
    kwargs = built["kwargs"]
    assert "credentials" not in kwargs
    assert kwargs["static_discovery"] is True
    transport = kwargs["http"]
    assert isinstance(transport, google_api_proxy.GoogleApiProxyHttp)
    # Gmail's own host only: Composio's injected token cannot be sent to
    # another Google API.
    with pytest.raises(ValueError):
        transport.request("https://www.googleapis.com/drive/v3/files")


def test_a_native_grant_still_builds_credentials(monkeypatch):
    for name in (
        composio_proxy.COMPOSIO_API_KEY_ENV,
        composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
        composio_proxy.COMPOSIO_USER_ID_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    built: dict[str, Any] = {}

    def fake_build(*args: Any, **kwargs: Any) -> Any:
        built["kwargs"] = kwargs
        return Mock()

    monkeypatch.setattr(gmail, "build", fake_build)

    gmail.get_gmail_service()

    kwargs = built["kwargs"]
    assert kwargs["credentials"].token == "access-token"
    assert "http" not in kwargs
    assert "static_discovery" not in kwargs


def test_a_native_batch_is_still_one_batch(monkeypatch):
    monkeypatch.delenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, raising=False)
    batch = Mock()

    gmail._execute_batch(batch)

    batch.execute.assert_called_once_with()


def test_a_composio_batch_is_run_by_the_proxy_fallback(monkeypatch, composio_env):
    batch = Mock()
    fallback = Mock()
    monkeypatch.setattr(google_api_proxy, "execute_batch", fallback)

    gmail._execute_batch(batch)

    fallback.assert_called_once_with(batch)
    batch.execute.assert_not_called()


def test_search_reads_each_message_through_the_proxy(proxy, monkeypatch):
    proxy.answer("/messages/m1", 200, _MESSAGE)
    proxy.answer("/messages", 200, {"messages": [{"id": "m1"}]})
    service = _proxy_service()
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: service)

    result = json.loads(gmail.gmail_search_messages("from:someone", max_results=1))

    assert result == {
        "status": "success",
        "messages": [
            {
                "id": "m1",
                "threadId": "t1",
                "snippet": "hello there",
                "subject": "Hello",
                "from": "sender@example.com",
            }
        ],
        "errors": None,
    }
    list_call, message_call = proxy.calls
    assert list_call["method"] == "GET"
    assert list_call["endpoint"] == (
        "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    )
    assert {
        "name": "q",
        "value": "from:someone",
        "type": "query",
    } in list_call["parameters"]
    assert message_call["endpoint"] == (
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1"
    )


def test_search_with_no_messages_sends_only_the_list_request(proxy, monkeypatch):
    proxy.answer("/messages", 200, {"messages": []})
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(gmail.gmail_search_messages("nothing"))

    assert result == {"status": "success", "messages": []}
    assert len(proxy.calls) == 1


def test_read_threads_sends_one_request_per_thread(proxy, monkeypatch):
    proxy.answer("/threads/t1", 200, {"id": "t1", "messages": [{"id": "m1"}]})
    proxy.answer("/threads/t2", 200, {"id": "t2", "messages": []})
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(gmail.gmail_read_threads(["t1", "t2"]))

    assert result == {
        "status": "success",
        "threads": [
            {"id": "t1", "messages": [{"id": "m1"}]},
            {"id": "t2", "messages": []},
        ],
        "errors": None,
    }
    assert [call["endpoint"] for call in proxy.calls] == [
        "https://gmail.googleapis.com/gmail/v1/users/me/threads/t1",
        "https://gmail.googleapis.com/gmail/v1/users/me/threads/t2",
    ]


def test_creating_a_label_sends_its_body_through_the_proxy(proxy, monkeypatch):
    proxy.answer("/labels", 200, {"id": "Label_1", "name": "Receipts"})
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(gmail.gmail_manage_labels("create", name="Receipts"))

    assert result["label"] == {"id": "Label_1", "name": "Receipts"}
    (call,) = proxy.calls
    assert call["method"] == "POST"
    assert call["endpoint"] == "https://gmail.googleapis.com/gmail/v1/users/me/labels"
    assert call["body"] == {
        "name": "Receipts",
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show",
    }
    assert {"name": "content-type", "value": "application/json", "type": "header"} in (
        call["parameters"]
    )


def test_sending_with_an_attachment_sends_the_encoded_message(
    proxy, monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    attachment = tmp_path / "notes.txt"
    attachment.write_text("attached notes", encoding="utf-8")
    proxy.answer("/messages/send", 200, {"id": "sent-1"})
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "a@example.com",
                    "subject": "Notes",
                    "body": "See attached.",
                    "attachments": [str(attachment)],
                }
            ],
            action="send",
        )
    )

    assert result["results"] == [
        {
            "status": "sent",
            "id": "sent-1",
            "attachments": [{"filename": "notes.txt", "size": 14}],
        }
    ]
    (call,) = proxy.calls
    assert call["endpoint"] == (
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    )
    encoded = call["body"]["raw"]
    message = BytesParser(policy=policy.default).parsebytes(
        base64.urlsafe_b64decode(encoded)
    )
    assert [part.get_filename() for part in message.iter_attachments()] == ["notes.txt"]
    assert message["Subject"] == "Notes"


def test_a_message_too_large_for_the_proxy_is_refused_clearly(
    proxy, monkeypatch, tmp_path
):
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    attachment = tmp_path / "big.bin"
    attachment.write_bytes(b"\x00" * 4096)
    monkeypatch.setattr(composio_proxy, "MAX_JSON_BODY_BYTES", 2000)
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "a@example.com",
                    "subject": "Big",
                    "body": "B",
                    "attachments": [str(attachment)],
                }
            ],
            action="send",
        )
    )

    assert result["status"] == "error"
    assert "too large to send through Composio" in result["message"]
    assert "2000 byte limit" in result["message"]
    assert "smaller attachment" in result["message"]
    assert proxy.calls == []


def test_an_upstream_error_reaches_the_tool_as_googles_message(proxy, monkeypatch):
    proxy.answer("/labels/missing", 404, _NOT_FOUND)
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(gmail.gmail_manage_labels("get", label_id="missing"))

    assert result["status"] == "error"
    assert "HttpError 404" in result["message"]
    assert "Requested entity was not found." in result["message"]


def test_a_composio_refusal_is_reported_without_secrets(proxy, monkeypatch):
    proxy.default = _Response(
        404,
        {
            "error": {
                "slug": "ConnectedAccount_NotFound",
                "message": f"account {ACCOUNT_ID} of {COMPOSIO_USER} key {API_KEY}",
            }
        },
    )
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(gmail.gmail_manage_labels("list"))

    assert result["status"] == "error"
    assert "ConnectedAccount_NotFound" in result["message"]
    assert _secrets_in(result["message"]) == []


def test_the_batch_fallback_output_matches_a_native_batch(monkeypatch, proxy):
    """The same messages, one tool run through a real native batch and one
    through the proxy, come back as the same JSON."""
    list_body = {"messages": [{"id": "m1"}, {"id": "m2"}]}
    native_service = build(
        "gmail",
        "v1",
        http=FakeNativeHttp(
            list_body,
            [
                ("1", 200, json.dumps(_MESSAGE)),
                ("2", 404, json.dumps(_NOT_FOUND)),
            ],
        ),
        static_discovery=True,
    )

    for name in (
        composio_proxy.COMPOSIO_API_KEY_ENV,
        composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
        composio_proxy.COMPOSIO_USER_ID_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: native_service)
    native_output = json.loads(gmail.gmail_search_messages("q", max_results=2))

    proxy.answer("/messages/m1", 200, _MESSAGE)
    proxy.answer("/messages/m2", 404, _NOT_FOUND)
    proxy.answer("/messages", 200, list_body)
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())
    proxied_output = json.loads(gmail.gmail_search_messages("q", max_results=2))

    assert proxied_output == native_output
    assert native_output["status"] == "success"
    assert native_output["messages"][0]["subject"] == "Hello"
    assert "HttpError 404" in native_output["errors"][0]


def test_a_composio_gmail_grant_gets_the_composio_launch_variables(monkeypatch):
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)

    env = tools_config._composio_grant_env(
        app_id="gmail",
        connected_account_id=ACCOUNT_ID,
        composio_user_id=COMPOSIO_USER,
    )

    assert env == {
        composio_proxy.COMPOSIO_API_KEY_ENV: API_KEY,
        composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV: ACCOUNT_ID,
        composio_proxy.COMPOSIO_USER_ID_ENV: COMPOSIO_USER,
    }
