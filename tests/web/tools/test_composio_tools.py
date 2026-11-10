"""Composio's Files API and tool execution, faked at the HTTP boundary.

No request leaves the process: ``requests.request`` is replaced, so these
tests pin the request shapes Composio's documented API takes and the
messages a caller gets back, including that no credential or presigned URL
reaches them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xagent.web.tools.mcp import composio_proxy, composio_tools
from xagent.web.tools.mcp.google_drive_composio import (
    GOOGLEDRIVE_CREATE_FILE,
    GOOGLEDRIVE_TOOLKIT_NAME,
    GOOGLEDRIVE_TOOLKIT_SLUG,
    GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
)

API_KEY = "test-composio-api-key-value"
ACCOUNT_ID = "ca_drive_account_1"
COMPOSIO_USER = "xagent-0123456789abcdef"
UPLOAD_URL = (
    "https://storage.composio.dev/projects/pr_1/files/report.pdf?X-Amz-Signature=secret"
)


class _Response:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeHttp:
    """Answers by URL and method, recording every call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.by_url: list[tuple[str, str, _Response]] = []
        self.default = _Response(200, {})

    def answer(self, url: str, status: int, payload: Any, method: str = "POST") -> None:
        self.by_url.append((method, url, _Response(status, payload)))

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        call = {"method": method, "url": url, **kwargs}
        data = call.get("data")
        if data is not None and not isinstance(data, (bytes, bytearray)):
            # What requests does with a stream body: read it once.
            call["data"] = data.read()
        self.calls.append(call)
        for expected_method, needle, response in self.by_url:
            if method == expected_method and needle in url:
                return response
        return self.default

    def calls_to(self, needle: str, method: str | None = None) -> list[dict[str, Any]]:
        return [
            call
            for call in self.calls
            if needle in call["url"] and (method is None or call["method"] == method)
        ]


@pytest.fixture
def http(monkeypatch) -> FakeHttp:
    fake = FakeHttp()
    monkeypatch.setattr(composio_proxy.requests, "request", fake.request)
    # composio_tools and composio_proxy share the same public_requests module,
    # so one patch covers the upload, the tool call and a download.
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)
    return fake


def _upload_answer(**overrides: Any) -> dict[str, Any]:
    answer = {
        "id": "req_file_1",
        "key": "projects/pr_1/requests/googledrive/report.pdf",
        "new_presigned_url": UPLOAD_URL,
        "metadata": {"storage_backend": "s3"},
    }
    answer.update(overrides)
    return answer


_BYTES = b"%PDF-1.4 bytes"


def _execute(slug: str, arguments: Any) -> composio_tools.ToolResult:
    """Run a Drive tool the way google_drive.py does."""
    return composio_tools.execute_tool(
        slug, arguments, toolkit=GOOGLEDRIVE_TOOLKIT_NAME
    )


def _stage(
    http: FakeHttp,
    *,
    answer: dict[str, Any] | None = None,
    put_status: int = 200,
    on_dedup: Any = None,
) -> dict[str, str]:
    http.answer("/files/upload/request", 200, answer or _upload_answer())
    http.answer(UPLOAD_URL, put_status, None, method="PUT")
    return composio_tools.stage_file(
        toolkit_slug=GOOGLEDRIVE_TOOLKIT_SLUG,
        tool_slug=GOOGLEDRIVE_CREATE_FILE,
        filename="report.pdf",
        mimetype="application/pdf",
        md5="0" * 32,
        data=_BYTES,
        length=len(_BYTES),
        on_dedup=on_dedup,
    )


def test_stage_file_uploads_the_bytes_and_returns_the_tool_argument(http):
    staged = _stage(http)

    upload_call = http.calls_to("/files/upload/request")[0]
    assert upload_call["json"] == {
        "toolkit_slug": "googledrive",
        "tool_slug": "GOOGLEDRIVE_CREATE_FILE",
        "filename": "report.pdf",
        "mimetype": "application/pdf",
        "md5": "0" * 32,
    }
    assert upload_call["headers"]["x-api-key"] == API_KEY
    put_call = http.calls_to(UPLOAD_URL, method="PUT")[0]
    assert put_call["data"] == _BYTES
    assert put_call["headers"]["Content-Type"] == "application/pdf"
    assert put_call["headers"]["Content-Length"] == str(len(_BYTES))
    assert "x-ms-blob-type" not in put_call["headers"]
    assert put_call["allow_redirects"] is False
    assert staged == {
        "name": "report.pdf",
        "mimetype": "application/pdf",
        "s3key": "projects/pr_1/requests/googledrive/report.pdf",
    }


def test_stage_file_sets_the_azure_blob_header_for_the_azure_backend(http):
    _stage(
        http, answer=_upload_answer(metadata={"storage_backend": "azure_blob_storage"})
    )

    put_headers = http.calls_to(UPLOAD_URL, method="PUT")[0]["headers"]
    assert put_headers["x-ms-blob-type"] == "BlockBlob"


def test_stage_file_skips_the_upload_when_composio_already_has_the_bytes(http):
    staged = _stage(http, answer=_upload_answer(new_presigned_url=None))

    assert staged["s3key"] == "projects/pr_1/requests/googledrive/report.pdf"
    assert http.calls_to(UPLOAD_URL) == []


def test_stage_file_calls_the_dedup_hook_only_when_no_upload_happened(http):
    deduped: list[bool] = []
    staged = _stage(
        http,
        answer=_upload_answer(new_presigned_url=None),
        on_dedup=lambda: deduped.append(True),
    )

    assert staged["s3key"]
    assert deduped == [True]

    # A fresh answer with a URL must upload and must not call the hook.
    http.by_url.clear()
    deduped.clear()
    _stage(http, on_dedup=lambda: deduped.append(True))
    assert deduped == []


def test_stage_file_refuses_a_non_string_presigned_url(http):
    """An answer field with the wrong type is not a deduplication: the bytes
    would otherwise never be uploaded and the tool would name a stale key."""
    with pytest.raises(composio_tools.ComposioToolError, match="unusable upload URL"):
        _stage(http, answer=_upload_answer(new_presigned_url={"url": UPLOAD_URL}))

    assert http.calls_to(UPLOAD_URL) == []


def test_stage_file_refuses_a_bool_as_a_length(http):
    """``bool`` is an ``int`` in Python, but ``True`` is not a byte count."""
    with pytest.raises(ValueError, match="byte count"):
        composio_tools.stage_file(
            toolkit_slug=GOOGLEDRIVE_TOOLKIT_SLUG,
            tool_slug=GOOGLEDRIVE_CREATE_FILE,
            filename="report.pdf",
            mimetype="application/pdf",
            md5="0" * 32,
            data=b"x",
            length=True,
        )


def test_a_private_upload_destination_keeps_the_composio_error_contract():
    """The real public-only client refuses a private host while the request is
    prepared, before any socket is opened, so this is the shape that refusal
    arrives in. The answer keeps the ComposioToolError contract and does not
    name the host."""
    with pytest.raises(composio_tools.ComposioToolError, match="unusable upload URL"):
        composio_tools._put_file(
            "https://127.0.0.1/report.pdf",
            data=b"x",
            length=1,
            mimetype="text/plain",
            backend=None,
        )


def test_a_destination_refused_at_resolution_keeps_the_same_contract(monkeypatch):
    """A host that resolves to a private address is refused later, in the
    adapter's send, and the session reports that as an InvalidURL
    (public_requests.py). Same contract, and still no host or URL named."""

    def refuse(method: str, url: str, **kwargs: Any) -> Any:
        raise composio_proxy.requests.exceptions.InvalidURL(
            "Host must not resolve to a private network."
        )

    monkeypatch.setattr(composio_proxy.requests, "request", refuse)

    with pytest.raises(
        composio_tools.ComposioToolError, match="unusable upload URL"
    ) as raised:
        composio_tools._put_file(
            UPLOAD_URL,
            data=b"x",
            length=1,
            mimetype="text/plain",
            backend=None,
        )

    assert UPLOAD_URL not in str(raised.value)


def test_a_programming_error_in_the_http_client_is_not_hidden(http, monkeypatch):
    """Only transport failures become ``ComposioToolError``; a bug in the
    caller's payload must keep its own type instead of looking like an
    unreachable server."""
    http.answer("/tools/execute/GOOGLEDRIVE_CREATE_FILE", 200, {})

    def boom(**kwargs: Any) -> Any:
        raise ValueError("payload is not serializable")

    monkeypatch.setattr(composio_proxy.requests, "request", boom)

    with pytest.raises(ValueError, match="not serializable"):
        _execute(GOOGLEDRIVE_CREATE_FILE, {})


def test_stage_file_refuses_an_answer_without_a_key(http):
    with pytest.raises(composio_tools.ComposioToolError, match="no key"):
        _stage(http, answer={"new_presigned_url": UPLOAD_URL})

    assert http.calls_to(UPLOAD_URL) == []


def test_stage_file_refuses_an_upload_url_that_is_not_https(http):
    http.answer(
        "/files/upload/request",
        200,
        _upload_answer(new_presigned_url="http://storage.example.com/report.pdf"),
    )

    with pytest.raises(composio_tools.ComposioToolError, match="unusable upload URL"):
        composio_tools.stage_file(
            toolkit_slug=GOOGLEDRIVE_TOOLKIT_SLUG,
            tool_slug=GOOGLEDRIVE_CREATE_FILE,
            filename="report.pdf",
            mimetype="application/pdf",
            md5="0" * 32,
            data=b"bytes",
            length=5,
        )

    assert [call for call in http.calls if call["method"] == "PUT"] == []


def test_a_refused_storage_upload_does_not_name_the_presigned_url(http):
    with pytest.raises(composio_tools.ComposioToolError) as raised:
        _stage(http, put_status=403)

    assert "HTTP 403" in str(raised.value)
    assert UPLOAD_URL not in str(raised.value)
    assert "X-Amz-Signature" not in str(raised.value)


def test_an_unreachable_upload_names_only_its_type(http, monkeypatch):
    def boom(**kwargs: Any) -> Any:
        raise ConnectionError(f"cannot reach {kwargs['url']}")

    monkeypatch.setattr(composio_proxy.requests, "request", boom)

    with pytest.raises(composio_tools.ComposioToolError) as raised:
        composio_tools.stage_file(
            toolkit_slug=GOOGLEDRIVE_TOOLKIT_SLUG,
            tool_slug=GOOGLEDRIVE_CREATE_FILE,
            filename="report.pdf",
            mimetype="application/pdf",
            md5="0" * 32,
            data=b"bytes",
            length=5,
        )

    assert str(raised.value) == "Composio could not be reached"
    assert API_KEY not in str(raised.value)


def test_stage_file_sends_the_toolkit_it_is_given(http):
    """The module is not Drive's own: the toolkit comes from the caller."""
    http.answer("/files/upload/request", 200, _upload_answer())
    http.answer(UPLOAD_URL, 200, None, method="PUT")

    composio_tools.stage_file(
        toolkit_slug="gmail",
        tool_slug="GMAIL_SEND_EMAIL",
        filename="report.pdf",
        mimetype="application/pdf",
        md5="0" * 32,
        data=_BYTES,
        length=len(_BYTES),
    )

    assert http.calls_to("/files/upload/request")[0]["json"]["toolkit_slug"] == "gmail"


def test_execute_tool_names_the_toolkit_it_is_given(http):
    http.answer(
        "/tools/execute/GMAIL_SEND_EMAIL",
        422,
        {"error": {"message": "refused", "slug": "BadRequest"}},
    )

    with pytest.raises(composio_tools.ComposioToolError) as raised:
        composio_tools.execute_tool("GMAIL_SEND_EMAIL", {}, toolkit="Gmail")

    assert "could not run the Gmail tool" in str(raised.value)


def test_execute_tool_sends_the_account_the_user_and_the_arguments(http):
    http.answer(
        "/tools/execute/GOOGLEDRIVE_CREATE_FILE",
        200,
        {"data": {"id": "drive1"}, "error": None, "successful": True},
    )

    result = _execute(
        GOOGLEDRIVE_CREATE_FILE,
        {"name": "notes.txt", "file_to_upload": {"s3key": "k"}},
    )

    assert result.successful is True
    assert result.data == {"id": "drive1"}
    assert result.error is None
    call = http.calls_to("/tools/execute/GOOGLEDRIVE_CREATE_FILE")[0]
    assert call["json"] == {
        "connected_account_id": ACCOUNT_ID,
        "user_id": COMPOSIO_USER,
        "arguments": {"name": "notes.txt", "file_to_upload": {"s3key": "k"}},
    }
    assert call["headers"]["x-api-key"] == API_KEY
    # A redirect is not followed: replaying the JSON body to another origin
    # fails after the POST was sent (see the redirect test below).
    assert call["allow_redirects"] is False


def test_a_redirect_on_the_tool_post_is_reported_as_an_unknown_result(http):
    """A cross-origin 307/308 cannot be replayed, and the client's own
    refusal arrives after the POST was sent. The 3xx status is reported
    instead, so a caller never reads it as "nothing was sent"."""
    http.answer(
        "/tools/execute/GOOGLEDRIVE_CREATE_FILE",
        307,
        {"error": {"message": "moved"}},
    )

    with pytest.raises(composio_tools.ComposioToolError) as raised:
        _execute(GOOGLEDRIVE_CREATE_FILE, {})

    assert raised.value.status == 307
    assert "redirect (HTTP 307)" in str(raised.value)
    assert "the request was sent" in str(raised.value)
    assert "nothing was sent" not in str(raised.value)


def test_execute_tool_returns_a_tool_failure_without_raising(http):
    http.answer(
        "/tools/execute/GOOGLEDRIVE_UPLOAD_UPDATE_FILE",
        200,
        {
            "data": {},
            "error": "Google Drive refused the upload",
            "successful": False,
        },
    )

    result = _execute(GOOGLEDRIVE_UPLOAD_UPDATE_FILE, {"fileId": "f1"})

    assert result.successful is False
    assert result.error == "Google Drive refused the upload"


def test_execute_tool_redacts_a_secret_composio_echoes_back(http):
    http.answer(
        "/tools/execute/GOOGLEDRIVE_CREATE_FILE",
        400,
        {
            "error": {
                "message": f"bad key {API_KEY} for {ACCOUNT_ID}",
                "slug": "InvalidApiKey",
                "status": 400,
            }
        },
    )

    with pytest.raises(composio_tools.ComposioToolError) as raised:
        _execute(GOOGLEDRIVE_CREATE_FILE, {})

    message = str(raised.value)
    assert "InvalidApiKey" in message
    assert API_KEY not in message
    assert ACCOUNT_ID not in message
    assert raised.value.status == 400


def test_an_error_text_redacts_a_token_assignment_and_a_signed_url():
    """A vendor can echo a credential by member name or a presigned URL; the
    error must not carry either into the model's transcript or the log. The
    shapes are the ones ``composio_proxy`` redacts, so the same answer is
    cleaned the same way whichever Composio client relays it."""
    error = composio_tools.ComposioToolError(
        'bad access_token=ACCESS-SECRET and "refresh_token": "REFRESH-SECRET" '
        "at https://storage.example.com/a?X-Amz-Signature=SIGNED-SECRET&x=1"
    )

    text = str(error)
    assert "ACCESS-SECRET" not in text
    assert "REFRESH-SECRET" not in text
    assert "SIGNED-SECRET" not in text
    assert "storage.example.com" not in text
    assert "access_token=[redacted]" in text
    assert '"refresh_token": "[redacted]"' in text
    assert "[redacted-url]" in text
    # The name survives so the message still says what leaked.
    assert "bad " in text


def test_execute_tool_refuses_an_answer_without_a_result(http):
    http.answer(
        "/tools/execute/GOOGLEDRIVE_CREATE_FILE",
        200,
        {"data": {"id": "drive1"}},
    )

    with pytest.raises(composio_tools.ComposioToolError, match="without a result"):
        _execute(GOOGLEDRIVE_CREATE_FILE, {})


def test_execute_tool_refuses_an_answer_that_is_not_an_object(http):
    http.answer("/tools/execute/GOOGLEDRIVE_CREATE_FILE", 200, ["not", "an", "object"])

    with pytest.raises(composio_tools.ComposioToolError, match="unusable tool answer"):
        _execute(GOOGLEDRIVE_CREATE_FILE, {})


def test_execute_tool_names_a_missing_launch_variable(http, monkeypatch):
    monkeypatch.delenv(composio_proxy.COMPOSIO_API_KEY_ENV)

    with pytest.raises(ValueError, match="COMPOSIO_API_KEY"):
        _execute(GOOGLEDRIVE_CREATE_FILE, {})


def test_a_staged_body_is_read_as_a_stream(http):
    class Body:
        def __init__(self) -> None:
            self._data = [b"first ", b"second"]
            self.reads = 0

        def __len__(self) -> int:
            return 11

        def read(self, amount: int = -1) -> bytes:
            self.reads += 1
            return b"".join(self._data)

    body = Body()
    http.answer("/files/upload/request", 200, _upload_answer())
    http.answer(UPLOAD_URL, 200, None, method="PUT")

    composio_tools.stage_file(
        toolkit_slug=GOOGLEDRIVE_TOOLKIT_SLUG,
        tool_slug=GOOGLEDRIVE_CREATE_FILE,
        filename="report.pdf",
        mimetype="application/pdf",
        md5="0" * 32,
        data=body,
        length=len(body),
    )

    assert body.reads == 1
    assert http.calls_to(UPLOAD_URL, method="PUT")[0]["data"] == b"first second"


def test_the_tool_failure_error_is_kept_to_its_text(http):
    http.answer(
        "/tools/execute/GOOGLEDRIVE_CREATE_FILE",
        200,
        {"data": None, "error": "x" * 5000, "successful": False},
    )

    result = _execute(GOOGLEDRIVE_CREATE_FILE, {})

    assert result.error is not None
    assert result.error.endswith("...[truncated]")
    assert len(result.error) <= composio_proxy.MAX_ERROR_TEXT_CHARS + len(
        "...[truncated]"
    )


def test_the_upload_request_is_not_sent_without_the_api_key(monkeypatch):
    monkeypatch.delenv(composio_proxy.COMPOSIO_API_KEY_ENV)

    with pytest.raises(ValueError, match="COMPOSIO_API_KEY"):
        composio_tools.stage_file(
            toolkit_slug=GOOGLEDRIVE_TOOLKIT_SLUG,
            tool_slug=GOOGLEDRIVE_CREATE_FILE,
            filename="report.pdf",
            mimetype="application/pdf",
            md5="0" * 32,
            data=b"bytes",
            length=5,
        )


def test_the_staged_file_argument_is_json_serializable(http):
    staged = _stage(http)

    assert json.loads(json.dumps(staged)) == staged
