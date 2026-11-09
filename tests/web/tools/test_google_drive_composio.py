"""Google Drive tools on a Composio-backed grant.

Such a grant holds no Google token: the tool process is launched with the
Composio ids instead (``tools/config._composio_grant_env``), JSON and media
reads go through the proxy transport, and the three byte-writing tools stage
the file through Composio's Files API and run Composio's Drive tool.
Composio is faked here; no request leaves the process.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from xagent.web.tools import config as tools_config
from xagent.web.tools.mcp import (
    composio_proxy,
    composio_tools,
    google_api_proxy,
    google_drive,
)

API_KEY = "test-composio-api-key-value"
ACCOUNT_ID = "ca_drive_account_1"
COMPOSIO_USER = "xagent-0123456789abcdef"
BINARY_URL = (
    "https://storage.composio.dev/projects/pr_1/staged/report.pdf"
    "?X-Amz-Signature=binary-secret"
)
UPLOAD_URL = (
    "https://storage.composio.dev/projects/pr_1/requests/googledrive/report.pdf"
    "?X-Amz-Signature=upload-secret"
)
DECK_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
OLD_CONTENT = b"old deck bytes"
NEW_CONTENT = b"new deck bytes, edited"
_NEW_ID = "new-file-1"


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


class _Response:
    def __init__(
        self, status_code: int, payload: Any = None, content: bytes = b""
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeHttp:
    """The single ``requests.request`` the Drive tool path calls, routed by URL.

    ``public_requests`` is one module shared by the proxy client, the tool
    client and the transport, so one patch covers every call a Drive tool
    makes: Proxy Execute, Composio's Files API, Composio's tool execution,
    the presigned PUT and a ``binary_data`` download.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.proxy_calls: list[dict[str, Any]] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.puts: list[dict[str, Any]] = []
        self.downloads: dict[str, bytes] = {}
        self._proxy_answers: dict[str, deque[dict[str, Any]]] = {}
        self._tool_answers: dict[str, dict[str, Any]] = {}
        self.upload_answer: dict[str, Any] = {
            "id": "req_file_1",
            "key": "projects/pr_1/requests/googledrive/report.pdf",
            "new_presigned_url": UPLOAD_URL,
            "metadata": {"storage_backend": "s3"},
        }
        self.put_status = 200
        # Called just before the Files API answers, to change a file mid-stage.
        self.before_upload: Any = None

    def proxy(
        self,
        needle: str,
        status: int = 200,
        data: Any = None,
        *,
        binary_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        answer: dict[str, Any] = {"status": status, "data": data}
        if binary_data is not None:
            answer["binary_data"] = binary_data
        if headers is not None:
            answer["headers"] = headers
        self._proxy_answers.setdefault(needle, deque()).append(answer)

    def tool(
        self,
        slug: str,
        *,
        successful: bool = True,
        data: Any = None,
        error: str | None = None,
    ) -> None:
        self._tool_answers[slug] = {
            "successful": successful,
            "data": data if data is not None else {},
            "error": error,
        }

    def download(self, url: str, content: bytes) -> None:
        self.downloads[url] = content

    def calls_to(self, needle: str, method: str | None = None) -> list[dict[str, Any]]:
        return [
            call
            for call in self.calls
            if needle in call["url"] and (method is None or call["method"] == method)
        ]

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        call = {"method": method, "url": url, **kwargs}
        data = call.get("data")
        if data is not None and not isinstance(data, (bytes, bytearray)):
            # What requests does with a stream body: read it once.
            call["data"] = data.read()
        self.calls.append(call)
        if url == composio_proxy.COMPOSIO_PROXY_URL:
            return self._proxy_answer(kwargs["json"])
        if url == f"{composio_tools.COMPOSIO_API_BASE_URL}/files/upload/request":
            if self.before_upload is not None:
                hook, self.before_upload = self.before_upload, None
                hook()
            return _Response(200, self.upload_answer)
        if "/tools/execute/" in url:
            slug = url.rsplit("/", 1)[-1]
            self.tool_calls.append((slug, kwargs["json"]["arguments"]))
            return _Response(200, self._tool_answers[slug])
        if method == "PUT":
            self.puts.append(call)
            return _Response(self.put_status)
        if method == "GET" and url in self.downloads:
            return _Response(200, content=self.downloads[url])
        raise AssertionError(f"unexpected request: {method} {url}")

    def _proxy_answer(self, payload: dict[str, Any]) -> _Response:
        self.proxy_calls.append(payload)
        signature = (
            payload["endpoint"]
            + " "
            + " ".join(
                f"{item['type']}:{item['name']}={item['value']}"
                for item in payload.get("parameters", [])
            )
        )
        for needle, answers in self._proxy_answers.items():
            if needle in signature:
                answer = answers.popleft() if len(answers) > 1 else answers[0]
                return _Response(200, answer)
        return _Response(200, {"status": 200, "data": {}})


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch, tmp_path):
    """No test inherits a Composio grant or the host's Drive credentials."""
    for name in (
        composio_proxy.COMPOSIO_API_KEY_ENV,
        composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
        composio_proxy.COMPOSIO_USER_ID_ENV,
        "GOOGLE_ACCESS_TOKEN",
        "GOOGLE_REFRESH_TOKEN",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", str(tmp_path))
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS", str(allowed))
    return allowed


@pytest.fixture
def composio_env(_isolate_environment, monkeypatch):
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, ACCOUNT_ID)
    monkeypatch.setenv(composio_proxy.COMPOSIO_USER_ID_ENV, COMPOSIO_USER)


@pytest.fixture
def http(monkeypatch, composio_env) -> FakeHttp:
    fake = FakeHttp()
    monkeypatch.setattr(composio_proxy.requests, "request", fake.request)
    return fake


def _query(payload: dict[str, Any]) -> dict[str, str]:
    return {
        item["name"]: item["value"]
        for item in payload.get("parameters", [])
        if item["type"] == "query"
    }


def _headers(payload: dict[str, Any]) -> dict[str, str]:
    return {
        item["name"]: item["value"]
        for item in payload.get("parameters", [])
        if item["type"] == "header"
    }


def _metadata(**overrides: Any) -> dict[str, Any]:
    metadata = {
        "id": "deck1",
        "name": "Quarterly Deck.pptx",
        "mimeType": DECK_MIME,
        "modifiedTime": "2026-05-12T08:00:00.000Z",
        "version": "7",
        "headRevisionId": "rev-1",
        "shared": True,
        "webViewLink": "https://drive.google.com/file/d/deck1/view",
    }
    metadata.update(overrides)
    return metadata


def _current(**overrides: Any) -> dict[str, Any]:
    current = _metadata(
        trashed=False,
        size=str(len(OLD_CONTENT)),
        md5Checksum=_md5(OLD_CONTENT),
        capabilities={"canModifyContent": True},
    )
    current.update(overrides)
    return current


def _updated(data: bytes = NEW_CONTENT, **overrides: Any) -> dict[str, Any]:
    updated = {
        "id": "deck1",
        "name": "Quarterly Deck.pptx",
        "mimeType": DECK_MIME,
        "size": str(len(data)),
        "md5Checksum": _md5(data),
        "headRevisionId": "rev-2",
        "version": "8",
        "modifiedTime": "2026-09-01T08:00:00.000Z",
        "webViewLink": "https://drive.google.com/file/d/deck1/view",
        "shared": True,
    }
    updated.update(overrides)
    return updated


def _replacement(allowed: Path, data: bytes = NEW_CONTENT) -> Path:
    path = allowed / "Quarterly Deck.pptx"
    path.write_bytes(data)
    return path


def _update(http: FakeHttp, current: dict[str, Any], now: dict[str, Any]) -> None:
    """Answer the pre-update read, then the post-update read."""
    http.proxy("capabilities", 200, current)
    http.proxy("md5Checksum", 200, now)


def _call_update(path: Path, **kwargs: Any) -> dict[str, Any]:
    return json.loads(
        google_drive.google_drive_update_file_content("deck1", str(path), **kwargs)
    )


def _staged(http: FakeHttp) -> dict[str, Any]:
    return http.calls_to("/files/upload/request")[0]["json"]


# --- service construction and launch wiring --------------------------------


def test_a_composio_grant_builds_the_drive_service_on_the_proxy_transport(
    http, monkeypatch
):
    built: dict[str, Any] = {}

    def fake_build(*args: Any, **kwargs: Any) -> Any:
        built["args"] = args
        built["kwargs"] = kwargs
        return Mock()

    monkeypatch.setattr(google_drive, "build", fake_build)

    google_drive.get_drive_service()

    assert built["args"] == ("drive", "v3")
    kwargs = built["kwargs"]
    assert "credentials" not in kwargs
    assert kwargs["static_discovery"] is True
    transport = kwargs["http"]
    assert isinstance(transport, google_api_proxy.GoogleApiProxyHttp)
    # Composio's injected token cannot be sent to another Google API.
    with pytest.raises(ValueError):
        transport.request("https://gmail.googleapis.com/gmail/v1/users/me")


def test_building_the_drive_service_fetches_no_discovery_document(http):
    assert google_drive.get_drive_service() is not None
    assert http.calls == []


def test_a_native_grant_still_builds_credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    built: dict[str, Any] = {}

    def fake_build(*args: Any, **kwargs: Any) -> Any:
        built["kwargs"] = kwargs
        return Mock()

    monkeypatch.setattr(google_drive, "build", fake_build)

    google_drive.get_drive_service()

    kwargs = built["kwargs"]
    assert kwargs["credentials"].token == "access-token"
    assert "http" not in kwargs
    assert "static_discovery" not in kwargs


def test_a_native_grant_still_requires_an_access_token(monkeypatch):
    with pytest.raises(ValueError, match="GOOGLE_ACCESS_TOKEN"):
        google_drive.get_drive_service()


def test_a_composio_drive_grant_gets_the_composio_launch_variables(monkeypatch):
    monkeypatch.setenv(composio_proxy.COMPOSIO_API_KEY_ENV, API_KEY)

    env = tools_config._composio_grant_env(
        app_id="google-drive",
        connected_account_id=ACCOUNT_ID,
        composio_user_id=COMPOSIO_USER,
    )

    assert env == {
        composio_proxy.COMPOSIO_API_KEY_ENV: API_KEY,
        composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV: ACCOUNT_ID,
        composio_proxy.COMPOSIO_USER_ID_ENV: COMPOSIO_USER,
    }


# --- JSON tools through the transport --------------------------------------


def test_search_sends_the_query_and_fields_through_the_proxy(http):
    http.proxy(
        "/drive/v3/files",
        200,
        {"files": [{"id": "f1", "name": "notes.txt", "mimeType": "text/plain"}]},
    )

    result = json.loads(google_drive.google_drive_search("name contains 'notes'", 5))

    assert result["status"] == "success"
    assert result["files"] == [
        {"id": "f1", "name": "notes.txt", "mimeType": "text/plain"}
    ]
    call = http.proxy_calls[0]
    assert call["method"] == "GET"
    assert call["endpoint"] == "https://www.googleapis.com/drive/v3/files"
    query = _query(call)
    assert query["q"] == "name contains 'notes'"
    assert query["pageSize"] == "5"
    assert query["supportsAllDrives"] == "true"
    assert query["includeItemsFromAllDrives"] == "true"
    assert query["fields"] == "nextPageToken, files(id, name, mimeType, modifiedTime)"


def test_a_create_folder_sends_its_body_and_parameters_through_the_proxy(http):
    http.proxy(
        "/drive/v3/files",
        200,
        {
            "id": "folder1",
            "name": "Reports",
            "mimeType": "application/vnd.google-apps.folder",
        },
    )

    result = json.loads(google_drive.google_drive_create_folder("Reports", "parent1"))

    assert result["status"] == "success"
    assert result["folder"]["id"] == "folder1"
    call = http.proxy_calls[0]
    assert call["method"] == "POST"
    assert call["endpoint"] == "https://www.googleapis.com/drive/v3/files"
    query = _query(call)
    assert query["supportsAllDrives"] == "true"
    assert query["fields"] == "id, name, webViewLink, mimeType"
    assert call["body"] == {
        "name": "Reports",
        "mimeType": "application/vnd.google-apps.folder",
        "parents": ["parent1"],
    }


def test_an_upstream_404_still_becomes_the_same_tool_error(http):
    http.proxy(
        "/drive/v3/files",
        404,
        {"error": {"code": 404, "message": "File not found: deck1."}},
    )

    result = json.loads(google_drive.google_drive_search("name = 'x'"))

    assert result["status"] == "error"
    assert "404" in result["message"]


def test_a_file_read_sends_the_resource_key_header(http):
    url = "https://drive.google.com/file/d/deck1/view?resourcekey=0-Rkey123"
    http.proxy(
        "mimeType",
        200,
        {"id": "deck1", "name": "notes.txt", "mimeType": "text/plain"},
    )
    http.proxy(
        "alt=media",
        200,
        binary_data={"url": BINARY_URL, "content_type": "text/plain", "size": 3},
    )
    http.download(BINARY_URL, b"txt")

    google_drive.google_drive_get_file_content(url)

    media_call = [
        call for call in http.proxy_calls if _query(call).get("alt") == "media"
    ][0]
    assert _headers(media_call)["X-Goog-Drive-Resource-Keys"] == "deck1/0-Rkey123"


# --- downloads and exports stay on the proxy -------------------------------


def test_download_file_writes_the_bytes_behind_binary_data(http, tmp_path):
    http.proxy(
        "version, headRevisionId",
        200,
        _metadata(mimeType="application/pdf", name="Report.pdf"),
    )
    http.proxy(
        "alt=media",
        200,
        binary_data={
            "url": BINARY_URL,
            "content_type": "application/pdf",
            "size": len(NEW_CONTENT),
        },
    )
    http.download(BINARY_URL, NEW_CONTENT)

    result = json.loads(google_drive.google_drive_download_file("deck1"))

    assert result["status"] == "success"
    assert result["size"] == len(NEW_CONTENT)
    written = Path(result["path"])
    assert written.read_bytes() == NEW_CONTENT
    media_call = [
        call for call in http.proxy_calls if _query(call).get("alt") == "media"
    ][0]
    assert media_call["endpoint"] == "https://www.googleapis.com/drive/v3/files/deck1"
    # The downloader's Range header is not forwarded: the transport always
    # carries the whole file.
    assert "Range" not in _headers(media_call)
    assert http.calls_to(BINARY_URL, method="GET")[0]["timeout"] > 0


def test_get_file_content_exports_a_google_doc(http):
    http.proxy(
        "fields",
        200,
        {"id": "doc1", "name": "Notes", "mimeType": GOOGLE_DOC_MIME},
    )
    http.proxy(
        "/export",
        200,
        binary_data={
            "url": BINARY_URL,
            "content_type": "text/plain",
            "size": len(b"hello from docs"),
        },
    )
    http.download(BINARY_URL, b"hello from docs")

    result = json.loads(google_drive.google_drive_get_file_content("doc1"))

    assert result["status"] == "success"
    assert result["content"] == "hello from docs"
    assert result["encoding"] == "utf-8"
    export_call = [call for call in http.proxy_calls if "/export" in call["endpoint"]][
        0
    ]
    assert _query(export_call)["mimeType"] == "text/plain"


def test_get_file_content_accepts_text_relayed_in_data(http):
    """Whether Composio puts a text body in ``binary_data`` or ``data`` (a
    stage unknown), both arrive as the same UTF-8 text."""
    http.proxy(
        "mimeType",
        200,
        {"id": "f1", "name": "notes.txt", "mimeType": "text/plain"},
    )
    http.proxy("alt=media", 200, data="plain text")

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "success"
    assert result["content"] == "plain text"


def test_a_zero_byte_download_finishes(http):
    http.proxy(
        "mimeType",
        200,
        {"id": "f1", "name": "empty.txt", "mimeType": "text/plain"},
    )
    http.proxy(
        "alt=media",
        200,
        binary_data={"url": BINARY_URL, "content_type": "text/plain", "size": 0},
    )
    http.download(BINARY_URL, b"")

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "success"
    assert result["content"] == ""
    assert result["truncated"] is False


def test_a_binary_answer_without_an_https_url_is_refused(http):
    http.proxy(
        "mimeType",
        200,
        {"id": "f1", "name": "notes.txt", "mimeType": "text/plain"},
    )
    http.proxy(
        "alt=media",
        200,
        binary_data={
            "url": "http://storage.example.com/file",
            "content_type": "text/plain",
            "size": 3,
        },
    )

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "error"
    assert "download URL" in result["message"]
    assert "storage.example.com" not in result["message"]


# --- upload_file on a Composio grant ---------------------------------------


def _created_metadata(**overrides: Any) -> dict[str, Any]:
    metadata = {
        "id": _NEW_ID,
        "name": "Report.pdf",
        "webViewLink": f"https://drive.google.com/file/d/{_NEW_ID}/view",
        "mimeType": "application/pdf",
    }
    metadata.update(overrides)
    return metadata


def test_upload_file_stages_the_bytes_and_normalizes_the_tool_answer(
    http, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(composio_tools.GOOGLEDRIVE_CREATE_FILE, data={"id": _NEW_ID})
    http.proxy("webViewLink", 200, _created_metadata())

    result = json.loads(
        google_drive.google_drive_upload_file(
            str(local), mime_type="application/pdf", parent_id="folder1"
        )
    )

    assert result == {"status": "success", "file": _created_metadata()}
    staged_request = _staged(http)
    assert staged_request["filename"] == "Report.pdf"
    assert staged_request["mimetype"] == "application/pdf"
    assert staged_request["md5"] == _md5(NEW_CONTENT)
    assert http.puts[0]["data"] == NEW_CONTENT
    slug, arguments = http.tool_calls[0]
    assert slug == composio_tools.GOOGLEDRIVE_CREATE_FILE
    assert arguments == {
        "name": "Report.pdf",
        "mimeType": "application/pdf",
        "file_to_upload": {
            "name": "Report.pdf",
            "mimetype": "application/pdf",
            "s3key": "projects/pr_1/requests/googledrive/report.pdf",
        },
        "parents": ["folder1"],
    }
    assert _query(http.proxy_calls[-1])["fields"] == "id, name, webViewLink, mimeType"


def test_upload_file_streams_a_file_above_the_in_memory_threshold(
    http, monkeypatch, _isolate_environment
):
    monkeypatch.setattr(google_drive, "_STAGED_UPLOAD_IN_MEMORY_BYTES", 4)
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(composio_tools.GOOGLEDRIVE_CREATE_FILE, data={"id": _NEW_ID})
    http.proxy("webViewLink", 200, _created_metadata())

    result = json.loads(
        google_drive.google_drive_upload_file(str(local), mime_type="application/pdf")
    )

    assert result["status"] == "success"
    put_call = http.puts[0]
    assert put_call["data"] == NEW_CONTENT
    assert put_call["headers"]["Content-Length"] == str(len(NEW_CONTENT))


def test_upload_file_rejects_an_empty_file_before_any_composio_call(
    http, _isolate_environment
):
    local = _isolate_environment / "empty.pdf"
    local.write_bytes(b"")

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "empty" in result["message"].lower()
    assert http.calls == []


def test_upload_file_does_not_claim_success_when_the_tool_fails(
    http, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(
        composio_tools.GOOGLEDRIVE_CREATE_FILE,
        successful=False,
        error="insufficient permissions",
    )

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "insufficient permissions" in result["message"]
    assert "success" not in result


def test_upload_file_does_not_claim_success_without_the_new_file_id(
    http, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(composio_tools.GOOGLEDRIVE_CREATE_FILE, data={})

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "did not return the new file's id" in result["message"]


def test_upload_file_uses_the_tool_answer_when_the_read_back_fails(
    http, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(composio_tools.GOOGLEDRIVE_CREATE_FILE, data=_created_metadata())
    http.proxy("webViewLink", 404, {"error": {"code": 404, "message": "not found"}})

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    # The file exists; reporting a failure would invite a duplicate upload.
    assert result["status"] == "success"
    assert result["file"]["id"] == _NEW_ID
    assert result["file"]["name"] == "Report.pdf"


def test_upload_file_never_puts_the_api_key_or_upload_url_in_an_error(
    http, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(
        composio_tools.GOOGLEDRIVE_CREATE_FILE,
        successful=False,
        error="the drive account is not connected",
    )
    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    message = result["message"]
    assert API_KEY not in message
    assert UPLOAD_URL not in message
    assert "X-Amz-Signature" not in message


def test_upload_file_on_a_native_grant_still_uses_the_media_upload(
    monkeypatch, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    files = Mock()
    files.create.return_value.execute.return_value = _created_metadata()
    service = Mock()
    service.files.return_value = files
    monkeypatch.setattr(google_drive, "get_drive_service", lambda: service)

    result = json.loads(
        google_drive.google_drive_upload_file(str(local), mime_type="application/pdf")
    )

    assert result["status"] == "success"
    kwargs = files.create.call_args.kwargs
    assert kwargs["body"] == {"name": "Report.pdf", "mimeType": "application/pdf"}
    media = kwargs["media_body"]
    assert media.mimetype() == "application/pdf"
    assert media.size() == len(NEW_CONTENT)


# --- create_file on a Composio grant ---------------------------------------


def test_create_file_converts_text_to_a_google_doc(http):
    name = "Meeting notes"
    content = "hello from the meeting"
    http.tool(composio_tools.GOOGLEDRIVE_CREATE_FILE, data={"id": "doc1"})
    http.proxy(
        "webViewLink",
        200,
        {
            "id": "doc1",
            "name": name,
            "webViewLink": "https://docs.google.com/document/d/doc1/edit",
            "mimeType": GOOGLE_DOC_MIME,
        },
    )

    result = json.loads(
        google_drive.google_drive_create_file(
            name, content, mime_type=GOOGLE_DOC_MIME, parent_id="folder1"
        )
    )

    assert result["status"] == "success"
    assert result["file"]["mimeType"] == GOOGLE_DOC_MIME
    staged = _staged(http)
    assert staged["mimetype"] == "text/plain"
    assert staged["md5"] == _md5(content.encode("utf-8"))
    assert http.puts[0]["data"] == content.encode("utf-8")
    slug, arguments = http.tool_calls[0]
    assert slug == composio_tools.GOOGLEDRIVE_CREATE_FILE
    assert arguments["mimeType"] == GOOGLE_DOC_MIME
    assert arguments["parents"] == ["folder1"]
    assert (
        arguments["file_to_upload"]["s3key"]
        == "projects/pr_1/requests/googledrive/report.pdf"
    )


def test_create_file_keeps_a_plain_text_type(http):
    http.tool(composio_tools.GOOGLEDRIVE_CREATE_FILE, data={"id": "file1"})
    http.proxy(
        "webViewLink",
        200,
        {
            "id": "file1",
            "name": "notes.txt",
            "webViewLink": "https://drive.google.com/file/d/file1/view",
            "mimeType": "text/plain",
        },
    )

    google_drive.google_drive_create_file("notes.txt", "hello")

    assert _staged(http)["mimetype"] == "text/plain"
    assert http.tool_calls[0][1]["mimeType"] == "text/plain"


def test_create_file_stages_empty_content(http):
    http.tool(composio_tools.GOOGLEDRIVE_CREATE_FILE, data={"id": "file1"})
    http.proxy(
        "webViewLink",
        200,
        {
            "id": "file1",
            "name": "empty.txt",
            "webViewLink": "https://drive.google.com/file/d/file1/view",
            "mimeType": "text/plain",
        },
    )

    result = json.loads(google_drive.google_drive_create_file("empty.txt", ""))

    assert result["status"] == "success"
    assert _staged(http)["md5"] == _md5(b"")
    assert http.puts[0]["data"] == b""
    assert http.puts[0]["headers"]["Content-Length"] == "0"


def test_create_file_keeps_the_binary_name_guard_before_any_composio_call(http):
    result = json.loads(google_drive.google_drive_create_file("report.pdf", "text"))

    assert result["status"] == "error"
    assert "looks like a binary file" in result["message"]
    assert http.calls == []


def test_create_file_does_not_claim_success_without_the_new_file_id(http):
    http.tool(composio_tools.GOOGLEDRIVE_CREATE_FILE, data={})

    result = json.loads(google_drive.google_drive_create_file("notes.txt", "hello"))

    assert result["status"] == "error"
    assert "did not return" in result["message"]
    assert result["status"] != "success"


# --- update_file_content on a Composio grant -------------------------------


def test_update_content_stages_and_runs_a_media_update(http, _isolate_environment):
    local = _replacement(_isolate_environment)
    _update(http, _current(), _updated())
    http.tool(composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE, data={})

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "success"
    assert result["changed"] is True
    assert result["file"]["md5Checksum"] == _md5(NEW_CONTENT)
    assert result["previous"]["headRevisionId"] == "rev-1"
    assert "Replaced the content of 'Quarterly Deck.pptx'" in result["message"]
    staged = _staged(http)
    assert staged["md5"] == _md5(NEW_CONTENT)
    assert staged["mimetype"] == DECK_MIME
    assert http.puts[0]["data"] == NEW_CONTENT
    slug, arguments = http.tool_calls[0]
    assert slug == composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE
    assert arguments == {
        "fileId": "deck1",
        "file_to_upload": {
            "name": "Quarterly Deck.pptx",
            "mimetype": DECK_MIME,
            "s3key": "projects/pr_1/requests/googledrive/report.pdf",
        },
        "uploadType": "media",
        "supportsAllDrives": True,
    }


def test_update_content_still_refuses_a_stale_expected_head(http, _isolate_environment):
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, _current(headRevisionId="rev-9"))

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "does not match expected_head_revision_id" in result["message"]
    assert http.calls_to("/files/upload/request") == []
    assert http.tool_calls == []


def test_update_content_still_short_circuits_identical_content(
    http, _isolate_environment
):
    local = _replacement(_isolate_environment, OLD_CONTENT)
    http.proxy("capabilities", 200, _current())

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "unchanged"
    assert result["changed"] is False
    assert http.calls_to("/files/upload/request") == []
    assert http.tool_calls == []


@pytest.mark.parametrize(
    "current",
    [
        _current(
            mimeType="application/vnd.google-apps.document",
            md5Checksum=None,
        ),
        _current(trashed=True),
        _current(capabilities={"canModifyContent": False}),
    ],
)
def test_update_content_still_refuses_a_native_or_unwritable_file(
    http, _isolate_environment, current
):
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, current)

    result = _call_update(local)

    assert result["status"] == "error"
    assert "nothing was changed" in result["message"].lower()
    assert http.calls_to("/files/upload/request") == []
    assert http.tool_calls == []


def test_update_content_still_refuses_a_replacement_of_another_type(
    http, _isolate_environment
):
    local = _isolate_environment / "Quarterly Deck.pdf"
    local.write_bytes(NEW_CONTENT)
    http.proxy("capabilities", 200, _current())

    result = _call_update(local)

    assert result["status"] == "error"
    assert "replacement file is a .pdf file" in result["message"]
    assert http.calls_to("/files/upload/request") == []
    assert http.tool_calls == []


def test_update_content_is_unknown_when_another_writer_changed_the_file(
    http, _isolate_environment
):
    local = _replacement(_isolate_environment)
    _update(http, _current(), _updated(b"another writer's bytes"))
    http.tool(
        composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
        successful=False,
        error="the upload failed",
    )

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "the file has changed since it was read" in result["message"]
    assert "not known whether this call changed it" in result["message"]


def test_update_content_reports_a_definite_failure_when_the_file_is_unchanged(
    http, _isolate_environment
):
    local = _replacement(_isolate_environment)
    _update(http, _current(), _current())
    http.tool(
        composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
        successful=False,
        error="Google Drive refused the upload",
    )

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "did not accept the new content" in result["message"]
    assert "Nothing was changed in Google Drive." in result["message"]
    assert "Google Drive refused the upload" in result["message"]


def test_update_content_is_verified_by_drive_when_the_tool_reports_failure(
    http, _isolate_environment
):
    """A timeout or a 5xx can arrive after Drive stored the content. The
    checksum read after it decides the result, not the tool's answer."""
    local = _replacement(_isolate_environment)
    _update(http, _current(), _updated())
    http.tool(
        composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
        successful=False,
        error="the upload timed out",
    )

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "success"
    assert result["file"]["md5Checksum"] == _md5(NEW_CONTENT)
    assert "Composio reported an error" in result["message"]


def test_update_content_is_not_reported_as_done_when_the_checksum_disagrees(
    http, _isolate_environment
):
    local = _replacement(_isolate_environment)
    _update(http, _current(), _updated(b"someone else's bytes"))
    http.tool(composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE, data={})

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "does not confirm the new content" in result["message"]
    assert "checksum does not match" in result["message"]


def test_update_content_is_not_reported_as_done_when_the_read_back_fails(
    http, _isolate_environment
):
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, _current())
    http.proxy("md5Checksum", 500, {"error": {"code": 500, "message": "backend error"}})
    http.tool(composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE, data={})

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "accepted the upload" in result["message"]
    assert "not known whether" in result["message"]
    assert "updated or as unchanged" in result["message"]


def test_update_content_says_the_outcome_is_unknown_when_the_re_read_is_a_404(
    http, _isolate_environment
):
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, _current())
    http.proxy(
        "md5Checksum", 404, {"error": {"code": 404, "message": "File not found."}}
    )
    http.tool(
        composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
        successful=False,
        error="the upload failed",
    )

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "not known whether" in result["message"]
    assert "no longer exists" in result["message"]
    # A Composio grant has full Drive access, so the per-file caveat is gone.
    assert "per-file Drive access" not in result["message"]


def test_update_content_stops_before_the_tool_when_the_local_file_changes(
    http, _isolate_environment, monkeypatch
):
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 4)
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, _current())
    original = local.read_bytes()

    def change_the_file() -> None:
        local.write_bytes(b"z" * len(original))

    http.before_upload = change_the_file

    result = _call_update(local)

    assert result["status"] == "error"
    assert "local file changed" in result["message"].lower()
    assert "Google Drive kept the file as it was" in result["message"]
    assert http.tool_calls == []


def test_a_large_replacement_is_staged_as_a_verified_stream(
    http, _isolate_environment, monkeypatch
):
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 4)
    local = _replacement(_isolate_environment)
    _update(http, _current(), _updated())
    http.tool(composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE, data={})

    result = _call_update(local)

    assert result["status"] == "success"
    put_body = http.puts[0]["data"]
    assert put_body == NEW_CONTENT
    # The stream reports its length, so the upload is not chunked.
    assert http.puts[0]["headers"]["Content-Length"] == str(len(NEW_CONTENT))


def test_update_content_does_not_leak_the_api_key_or_the_upload_url(
    http, _isolate_environment
):
    local = _replacement(_isolate_environment)
    _update(http, _current(), _current())
    http.tool(
        composio_tools.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
        successful=False,
        error=f"bad key {API_KEY}",
    )

    result = _call_update(local)

    message = result["message"]
    assert API_KEY not in message
    assert UPLOAD_URL not in message
    assert "X-Amz-Signature" not in message


# --- docstrings and per-file wording ---------------------------------------


def test_tool_docstrings_do_not_describe_the_per_file_limit():
    assert "drive.file" not in (google_drive.google_drive_search.__doc__ or "")
    assert "drive.file" not in (google_drive.google_drive_move_file.__doc__ or "")


def test_a_composio_grant_leaves_the_per_file_caveat_out(http):
    assert google_drive._per_file_access_note() == ""
    assert "per-file Drive access" not in google_drive._empty_search_note()
    assert "per-file Drive access" not in google_drive._unavailable_file_message(
        RuntimeError("x")
    )
    assert (
        "per-file Drive access"
        not in google_drive._unavailable_content_update_message(RuntimeError("x"))
    )
    assert "per-file Drive access" not in google_drive._no_visible_parent_message()


def test_a_native_grant_keeps_the_per_file_caveat():
    assert "per-file Drive access" in google_drive._per_file_access_note()
    assert "per-file Drive access" in google_drive._empty_search_note()
