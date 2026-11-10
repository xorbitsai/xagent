"""Gmail tools on a Composio-backed grant.

Such a grant holds no Google token: the tool process is launched with the
Composio ids instead (``tools/config._composio_grant_env``), the service is
built on the proxy transport, and every call goes through Composio's Proxy
Execute endpoint. Composio is faked here; no request leaves the process.
"""

from __future__ import annotations

import base64
import json
import logging
from email import policy
from email.parser import BytesParser
from typing import Any
from unittest.mock import Mock

import httplib2
import pytest
from googleapiclient.discovery import build

from tests.web.tools.conftest import (
    ACCOUNT_ID,
    API_KEY,
    COMPOSIO_USER,
    _Response,
)
from xagent.web.tools import config as tools_config
from xagent.web.tools.mcp import composio_proxy, gmail, google_api_proxy

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


def test_building_the_service_fetches_no_discovery_document(proxy):
    """The discovery document is the library's bundled copy; fetching it
    through the proxy would spend a call and could be refused cross-domain."""
    service = gmail.get_gmail_service()

    assert service is not None
    assert proxy.calls == []


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


def test_the_real_body_limit_binds_at_about_2_5_mb_of_attachments(
    proxy, monkeypatch, tmp_path
):
    """An attachment is base64 inside the MIME part and the whole message is
    base64 again as ``raw``, so the JSON body is about 1.8 times the file, not
    4/3. Against the real 4.5 MB limit a 2.4 MB attachment passes and a 2.6 MB
    one is refused before sending; the boundary is about 2.5 MB (a 3.3 MB
    attachment is 5.9 MB of JSON)."""
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())
    proxy.answer("/messages/send", 200, {"id": "sent-1"})
    limit = composio_proxy.MAX_JSON_BODY_BYTES

    accepted = tmp_path / "under.bin"
    accepted.write_bytes(b"\x00" * 2_400_000)

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "a@example.com",
                    "subject": "Under",
                    "body": "B",
                    "attachments": [str(accepted)],
                }
            ],
            action="send",
        )
    )

    assert result["status"] == "success"
    (call,) = proxy.calls
    body_bytes = len(json.dumps(call))
    assert body_bytes <= limit
    assert 1.79 < body_bytes / 2_400_000 < 1.81

    refused = tmp_path / "over.bin"
    refused.write_bytes(b"\x00" * 2_600_000)

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {
                    "to": "a@example.com",
                    "subject": "Over",
                    "body": "B",
                    "attachments": [str(refused)],
                }
            ],
            action="send",
        )
    )

    assert result["status"] == "error"
    assert "too large to send through Composio" in result["message"]
    assert f"{limit} byte limit" in result["message"]
    assert len(proxy.calls) == 1, "the refused message must not be sent"


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


def test_an_oversized_second_message_refuses_the_whole_batch_before_sending(
    proxy, monkeypatch, tmp_path
):
    """#3647 regression guard: the proxy refuses an oversized JSON body
    locally, so a batch whose second message is too large must fail entirely
    up front instead of sending the first one and then erroring."""
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    attachment = tmp_path / "big.bin"
    attachment.write_bytes(b"\x00" * 4096)
    monkeypatch.setattr(composio_proxy, "MAX_JSON_BODY_BYTES", 3000)
    proxy.answer("/messages/send", 200, {"id": "sent-1"})
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {"to": "a@example.com", "subject": "Small", "body": "hi"},
                {
                    "to": "b@example.com",
                    "subject": "Big",
                    "body": "B",
                    "attachments": [str(attachment)],
                },
            ],
            action="send",
        )
    )

    assert result["status"] == "error"
    assert "too large to send through Composio" in result["message"]
    assert "3000 byte limit" in result["message"]
    assert proxy.calls == [], "no message may be sent before the whole batch is checked"


def test_sending_more_messages_than_the_cap_is_refused_before_sending(
    proxy, monkeypatch
):
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())
    messages = [
        {"to": "a@example.com", "subject": str(number), "body": "B"}
        for number in range(google_api_proxy.MAX_BATCH_REQUESTS + 1)
    ]

    result = json.loads(gmail.gmail_send_messages(messages, action="send"))

    assert result["status"] == "error"
    assert "request limit for one tool call" in result["message"]
    assert proxy.calls == []


def test_a_send_that_fails_midway_returns_the_messages_already_sent(proxy, monkeypatch):
    """Messages already sent cannot be taken back; the error must report
    them instead of dropping the partial result."""
    calls: list[dict[str, Any]] = []

    def request(**kwargs: Any) -> _Response:
        calls.append(kwargs["json"])
        if len(calls) == 1:
            return _Response(200, {"status": 200, "data": {"id": "sent-1"}})
        return _Response(500, {"error": {"message": "upstream is down"}})

    monkeypatch.setattr(composio_proxy.requests, "request", request)
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(
        gmail.gmail_send_messages(
            [
                {"to": "a@example.com", "subject": "One", "body": "1"},
                {"to": "b@example.com", "subject": "Two", "body": "2"},
            ],
            action="send",
        )
    )

    assert result["status"] == "error"
    assert result["results"] == [{"status": "sent", "id": "sent-1", "attachments": []}]
    assert "1 of 2 messages were sent" in result["message"]
    assert "upstream is down" in result["message"]


def test_a_search_above_the_request_cap_is_refused_before_the_list_call(
    proxy, monkeypatch
):
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(
        gmail.gmail_search_messages(
            "q", max_results=google_api_proxy.MAX_BATCH_REQUESTS + 1
        )
    )

    assert result["status"] == "error"
    assert "request limit for one tool call" in result["message"]
    # The list request is one more proxy call than the message cap counts.
    assert "101 message requests (plus the list request)" in result["message"]
    assert proxy.calls == []


def test_reading_more_threads_than_the_request_cap_is_refused_before_sending(
    proxy, monkeypatch
):
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())
    thread_ids = [
        f"t{number}" for number in range(google_api_proxy.MAX_BATCH_REQUESTS + 1)
    ]

    result = json.loads(gmail.gmail_read_threads(thread_ids))

    assert result["status"] == "error"
    assert "request limit for one tool call" in result["message"]
    assert proxy.calls == []


def test_deleting_a_label_sends_a_delete_through_the_proxy(proxy, monkeypatch):
    proxy.default = _Response(200, {"status": 204, "data": None})
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(gmail.gmail_manage_labels("delete", label_id="L1"))

    assert result == {"status": "success", "message": "Label L1 deleted"}
    (call,) = proxy.calls
    assert call["method"] == "DELETE"
    assert (
        call["endpoint"] == "https://gmail.googleapis.com/gmail/v1/users/me/labels/L1"
    )


def test_updating_a_label_patches_through_the_proxy(proxy, monkeypatch):
    proxy.answer("/labels/L1", 200, {"id": "L1", "name": "New"})
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(gmail.gmail_manage_labels("update", label_id="L1", name="New"))

    assert result["label"] == {"id": "L1", "name": "New"}
    (call,) = proxy.calls
    assert call["method"] == "PATCH"
    assert (
        call["endpoint"] == "https://gmail.googleapis.com/gmail/v1/users/me/labels/L1"
    )
    assert call["body"] == {"name": "New"}


def test_modifying_a_message_posts_its_label_changes(proxy, monkeypatch):
    proxy.answer("/messages/m1/modify", 200, {"id": "m1"})
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(
        gmail.gmail_manage_labels(
            "modify_message",
            message_id="m1",
            add_label_ids=["STARRED"],
            remove_label_ids=["UNREAD"],
        )
    )

    assert result["message"] == {"id": "m1"}
    (call,) = proxy.calls
    assert call["method"] == "POST"
    assert call["endpoint"] == (
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1/modify"
    )
    assert call["body"] == {"addLabelIds": ["STARRED"], "removeLabelIds": ["UNREAD"]}


def test_drafting_a_message_through_the_proxy(proxy, monkeypatch):
    proxy.answer("/drafts", 200, {"id": "draft-1"})
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    result = json.loads(
        gmail.gmail_send_messages(
            [{"to": "a@example.com", "subject": "Draft", "body": "later"}],
            action="draft",
        )
    )

    assert result["results"] == [
        {"status": "drafted", "id": "draft-1", "attachments": []}
    ]
    (call,) = proxy.calls
    assert call["endpoint"] == "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
    assert "raw" in call["body"]["message"]


def test_an_error_that_echoes_the_message_hides_its_body_and_attachment(
    proxy, monkeypatch, tmp_path, caplog
):
    """The vendor can echo the refused request body; the base64 ``raw`` in it
    carries the mail body and attachment, and must not reach the tool result
    or the server log."""
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    attachment = tmp_path / "secret.txt"
    attachment.write_text("ATTACHMENT-SECRET", encoding="utf-8")
    leaked: dict[str, str] = {}

    def request(**kwargs: Any) -> _Response:
        raw = kwargs["json"]["body"]["raw"]
        leaked["raw"] = raw
        return _Response(
            400, {"error": {"message": f"invalid body {json.dumps({'raw': raw})}"}}
        )

    monkeypatch.setattr(composio_proxy.requests, "request", request)
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    with caplog.at_level(logging.ERROR, logger="gmail-mcp"):
        result = json.loads(
            gmail.gmail_send_messages(
                [
                    {
                        "to": "a@example.com",
                        "subject": "Secret",
                        "body": "BODY-SECRET",
                        "attachments": [str(attachment)],
                    }
                ],
                action="send",
            )
        )

    assert result["status"] == "error"
    assert leaked["raw"] not in result["message"]
    assert leaked["raw"] not in "\n".join(
        record.getMessage() for record in caplog.records
    )
    decoded = base64.urlsafe_b64decode(leaked["raw"])
    assert b"BODY-SECRET" in decoded
    # The attachment body is base64 inside the MIME message.
    assert base64.b64encode(b"ATTACHMENT-SECRET") in decoded


def test_an_error_that_echoes_the_decoded_message_hides_a_long_recipient_list(
    proxy, monkeypatch, tmp_path, caplog
):
    """A group send keeps the recipients on one long ``To:`` line. When the
    vendor echoes the decoded message instead of the request body, redacting
    that line used to shorten the text in front of the mail body and the
    attached file, which then reached the tool result and the server log."""
    monkeypatch.setenv("XAGENT_GMAIL_FILE_ALLOWED_DIRS", str(tmp_path))
    attachment_bytes = b"ATTACHMENT-SECRET-" + bytes(range(256)) * 20
    attachment = tmp_path / "secret.bin"
    attachment.write_bytes(attachment_bytes)
    recipients = ",".join(f"user{index:03d}@example.com" for index in range(70))
    body = "TOOL-BODY-MARKER this is the mail body"
    encoded_attachment = base64.b64encode(attachment_bytes).decode()
    attachment_lines = [
        encoded_attachment[index : index + 76]
        for index in range(0, len(encoded_attachment), 76)
    ]

    def request(**kwargs: Any) -> _Response:
        raw = kwargs["json"]["body"]["raw"]
        decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
        return _Response(400, {"error": {"message": f"invalid body {decoded}"}})

    monkeypatch.setattr(composio_proxy.requests, "request", request)
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _proxy_service())

    with caplog.at_level(logging.ERROR, logger="gmail-mcp"):
        result = json.loads(
            gmail.gmail_send_messages(
                [
                    {
                        "to": recipients,
                        "subject": "Group send",
                        "body": body,
                        "attachments": [str(attachment)],
                    }
                ],
                action="send",
            )
        )

    assert result["status"] == "error"
    seen = (
        result["message"]
        + "\n"
        + "\n".join(record.getMessage() for record in caplog.records)
    )
    assert body not in seen
    assert "user000@example.com" not in seen
    assert not [line for line in attachment_lines if line[:24] in seen]
