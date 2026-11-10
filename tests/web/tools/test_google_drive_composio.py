"""Google Drive tools on a Composio-backed grant.

Such a grant holds no Google token: the tool process is launched with the
Composio ids instead (``tools/config._composio_grant_env``), JSON and media
reads go through the proxy transport, and the three byte-writing tools stage
the file through Composio's Files API and run Composio's Drive tool.
Composio is faked here; no request leaves the process.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import deque
from datetime import datetime, timedelta, timezone
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
    google_drive_composio,
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
        self,
        status_code: int,
        payload: Any = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = content
        # The transport reads these off a signed-URL download.
        self.headers = headers or {}

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
        self._tool_status: dict[str, int] = {}
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
        status: int = 200,
        successful: bool = True,
        data: Any = None,
        error: str | None = None,
    ) -> None:
        self._tool_answers[slug] = {
            "successful": successful,
            "data": data if data is not None else {},
            "error": error,
        }
        self._tool_status[slug] = status

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
            return _Response(self._tool_status.get(slug, 200), self._tool_answers[slug])
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
        {
            "id": "doc1",
            "name": "Notes",
            "mimeType": GOOGLE_DOC_MIME,
            # Drive reports a size for a Workspace file too (the document's
            # own blob), and it never equals the exported bytes.
            "size": "4096",
        },
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


def test_download_file_exports_a_document_despite_its_reported_size(
    http, _isolate_environment
):
    """The #3648 acceptance case: export a Google Doc to PDF. Drive's own
    size for the document is not the export's size, so it must not be
    compared."""
    http.proxy(
        "version, headRevisionId",
        200,
        _metadata(
            id="doc1",
            name="Notes",
            mimeType=GOOGLE_DOC_MIME,
            size="4096",
            md5Checksum=None,
        ),
    )
    http.proxy(
        "/export",
        200,
        binary_data={
            "url": BINARY_URL,
            "content_type": "application/pdf",
            "size": len(NEW_CONTENT),
        },
    )
    http.download(BINARY_URL, NEW_CONTENT)

    result = json.loads(
        google_drive.google_drive_download_file("doc1", mime_type="application/pdf")
    )

    assert result["status"] == "success"
    assert result["size"] == len(NEW_CONTENT)
    assert Path(result["path"]).read_bytes() == NEW_CONTENT
    assert Path(result["path"]).name.endswith(".pdf")
    # The check-only fields are not part of the tool's output shape.
    assert "size" not in result["file"]


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


def test_get_file_content_rejects_a_short_binary_answer(http):
    http.proxy(
        "mimeType",
        200,
        {"id": "f1", "name": "notes.txt", "mimeType": "text/plain"},
    )
    http.proxy(
        "alt=media",
        200,
        binary_data={
            "url": BINARY_URL,
            "content_type": "text/plain",
            "size": 100,
        },
    )
    http.download(BINARY_URL, b"short")

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "error"
    assert "truncated" in result["message"]


def test_download_file_writes_nothing_when_the_body_is_short(
    http, _isolate_environment, tmp_path
):
    http.proxy(
        "version, headRevisionId",
        200,
        _metadata(
            mimeType="application/pdf",
            name="Report.pdf",
            size="100",
            md5Checksum="0" * 32,
        ),
    )
    http.proxy(
        "alt=media",
        200,
        binary_data={
            "url": BINARY_URL,
            "content_type": "application/pdf",
            "size": 100,
        },
    )
    http.download(BINARY_URL, b"short")

    result = json.loads(google_drive.google_drive_download_file("deck1"))

    assert result["status"] == "error"
    assert "truncated" in result["message"]
    assert not (tmp_path / "output").exists()


def test_download_file_writes_nothing_when_drive_checksum_disagrees(
    http, _isolate_environment, tmp_path
):
    other = b"x" * len(NEW_CONTENT)
    http.proxy(
        "version, headRevisionId",
        200,
        _metadata(
            mimeType="application/pdf",
            name="Report.pdf",
            size=str(len(NEW_CONTENT)),
            md5Checksum=_md5(NEW_CONTENT),
        ),
    )
    http.proxy(
        "alt=media",
        200,
        binary_data={
            "url": BINARY_URL,
            "content_type": "application/pdf",
            "size": len(other),
        },
    )
    http.download(BINARY_URL, other)

    result = json.loads(google_drive.google_drive_download_file("deck1"))

    assert result["status"] == "error"
    assert "could not be verified" in result["message"]
    assert not (tmp_path / "output").exists()


def test_get_file_content_rejects_a_json_body_composio_parsed(http):
    """A JSON file relayed in ``data`` is re-serialized, not the original
    bytes. Drive's own checksum catches that instead of returning text that
    looks successful."""
    original = b'{"a": "\xc3\xa9", "b": 2}'
    http.proxy(
        "mimeType",
        200,
        {
            "id": "f1",
            "name": "data.json",
            "mimeType": "application/json",
            "size": str(len(original)),
            "md5Checksum": _md5(original),
        },
    )
    http.proxy("alt=media", 200, data={"a": "é", "b": 2})

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "error"
    assert "parsed object" in result["message"]
    assert "exact bytes" in result["message"]


def test_get_file_content_accepts_a_text_body_that_matches_the_file(http):
    text = '{"a": "é", "b": 2}'
    data = text.encode("utf-8")
    http.proxy(
        "mimeType",
        200,
        {
            "id": "f1",
            "name": "data.json",
            "mimeType": "application/json",
            "size": str(len(data)),
            "md5Checksum": _md5(data),
        },
    )
    http.proxy("alt=media", 200, data=text)

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "success"
    assert result["content"] == text
    # The check-only fields are not added to the native output shape.
    assert "md5Checksum" not in result["file"]
    assert "size" not in result["file"]


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
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": _NEW_ID})
    # Drive's read-back reports the uploaded checksum and size, so the
    # integrity check compares them rather than having nothing to check.
    http.proxy(
        "webViewLink",
        200,
        _created_metadata(md5Checksum=_md5(NEW_CONTENT), size=str(len(NEW_CONTENT))),
    )

    result = json.loads(
        google_drive.google_drive_upload_file(
            str(local), mime_type="application/pdf", parent_id="folder1"
        )
    )

    assert result == {"status": "success", "file": _created_metadata()}
    # The two check-only fields settle the upload and stay out of the result.
    assert "md5Checksum" not in result["file"]
    assert "size" not in result["file"]
    staged_request = _staged(http)
    assert staged_request["filename"] == "Report.pdf"
    assert staged_request["mimetype"] == "application/pdf"
    assert staged_request["md5"] == _md5(NEW_CONTENT)
    assert http.puts[0]["data"] == NEW_CONTENT
    slug, arguments = http.tool_calls[0]
    assert slug == google_drive_composio.GOOGLEDRIVE_CREATE_FILE
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
    assert _query(http.proxy_calls[-1])["fields"] == (
        "id, name, webViewLink, mimeType, md5Checksum, size"
    )


def test_upload_file_streams_a_file_above_the_in_memory_threshold(
    http, monkeypatch, _isolate_environment
):
    monkeypatch.setattr(google_drive_composio, "_STAGED_UPLOAD_IN_MEMORY_BYTES", 4)
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": _NEW_ID})
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


def test_upload_file_keeps_a_failed_tool_answer_an_unknown_outcome(
    http, _isolate_environment
):
    """Composio can fail the tool after Drive stored the file, so a failed
    answer is no proof it was not uploaded: the lookup decides."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(
        google_drive_composio.GOOGLEDRIVE_CREATE_FILE,
        successful=False,
        error="insufficient permissions",
    )
    http.proxy("name = ", 200, {"files": []})

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "insufficient permissions" in result["message"]
    assert "found no file named 'Report.pdf'" in result["message"]
    assert "not known whether it was uploaded" in result["message"]
    assert "was not uploaded" not in result["message"]
    lookup = [call for call in http.proxy_calls if "name = " in str(call)]
    assert len(lookup) == 1


def test_upload_file_does_not_claim_success_without_the_new_file_id(
    http, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={})
    http.proxy("name = ", 200, {"files": []})

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "did not confirm" in result["message"]
    assert "found no file named 'Report.pdf'" in result["message"]
    assert "not known whether it was uploaded" in result["message"]
    # The lookup was made once; the upload itself is not retried.
    assert len(http.tool_calls) == 1
    assert [call for call in http.proxy_calls if "name = " in str(call)] != []


def _same_name_file(**overrides: Any) -> dict[str, Any]:
    """A file the same-name lookup can find.

    It carries the checksum, size and type the old matching rules looked at as
    well as the id, name and creation time the lookup now asks for, so a test
    that feeds one of these to a create would fail if a match on any of them
    were ever reported as this call's result again.
    """
    entry = {
        "id": _NEW_ID,
        "name": "Report.pdf",
        "mimeType": "application/pdf",
        "size": str(len(NEW_CONTENT)),
        "md5Checksum": _md5(NEW_CONTENT),
        "webViewLink": f"https://drive.google.com/file/d/{_NEW_ID}/view",
        "createdTime": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    entry.update(overrides)
    return entry


def _same_name_folder(**overrides: Any) -> dict[str, Any]:
    """A same-name folder in the folder being written to."""
    entry = {
        "id": "folder1",
        "name": "Report.pdf",
        "mimeType": "application/vnd.google-apps.folder",
        "webViewLink": "https://drive.google.com/drive/folders/folder1",
        "createdTime": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    entry.update(overrides)
    return entry


def _candidate(file: dict[str, Any]) -> dict[str, str]:
    """The diagnostics an unknown create's answer reports for ``file``."""
    return {key: file[key] for key in ("id", "name", "createdTime") if key in file}


def _seconds_ago(seconds: int) -> str:
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    return moment.isoformat().replace("+00:00", "Z")


def _timeout_on_execute(monkeypatch: Any) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    def timeout(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        raise composio_tools.ComposioToolError("Composio could not be reached")

    monkeypatch.setattr(composio_tools, "execute_tool", timeout)
    return calls


def test_upload_file_says_nothing_was_uploaded_when_staging_fails(
    http, monkeypatch, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)

    def refuse(**kwargs: Any) -> Any:
        raise composio_tools.ComposioToolError(
            "Composio refused the file upload request (HTTP 403)"
        )

    monkeypatch.setattr(composio_tools, "stage_file", refuse)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "was not uploaded to Google Drive" in result["message"]
    assert "could not be staged" in result["message"]
    # The tool never ran, so Drive was not touched.
    assert http.tool_calls == []


def test_upload_file_does_not_take_a_same_bytes_file_for_the_upload(
    http, monkeypatch, _isolate_environment
):
    """A same-name file holding the uploaded bytes, created 20 seconds before
    this call, is exactly what the reviewer's counter-example describes: its
    name, its folder, its checksum and its creation time all match, and none
    of them is an operation identity. It is reported as a candidate to check,
    and the upload's outcome stays unknown instead of becoming a success."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    candidate = _same_name_file(createdTime=_seconds_ago(20))
    http.proxy("name = ", 200, {"files": [candidate]})
    calls = _timeout_on_execute(monkeypatch)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "file" not in result
    assert f"id '{_NEW_ID}'" in result["message"]
    assert "may be the file this call was making" in result["message"]
    assert "It is not known whether this call uploaded a file" in result["message"]
    assert result["candidates"] == [_candidate(candidate)]
    # One tool call, no retry: a retry could create a duplicate.
    assert len(calls) == 1
    lookup = [call for call in http.proxy_calls if "name = " in str(call)]
    assert len(lookup) == 1
    query = _query(lookup[0])
    assert query["q"].startswith("name = 'Report.pdf'")
    assert "'root' in parents" in query["q"]
    # The lookup asks only for what a person needs to check the candidate.
    assert query["fields"] == "files(id, name, createdTime)"


def test_upload_file_looks_in_the_parent_a_lost_answer_may_have_used(
    http, monkeypatch, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.proxy("name = ", 200, {"files": [_same_name_file()]})
    _timeout_on_execute(monkeypatch)

    json.loads(google_drive.google_drive_upload_file(str(local), parent_id="folder1"))

    lookup = [call for call in http.proxy_calls if "name = " in str(call)][0]
    assert "'folder1' in parents" in _query(lookup)["q"]


def test_upload_file_says_the_outcome_is_unknown_when_the_lookup_finds_nothing(
    http, monkeypatch, _isolate_environment
):
    """A timed-out tool may still be running in Composio, so a lookup that
    finds nothing is "not yet", not "not created"."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.proxy("name = ", 200, {"files": []})
    _timeout_on_execute(monkeypatch)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "found no file named 'Report.pdf'" in result["message"]
    assert "the upload may still be finishing in Composio" in result["message"]
    assert "not known whether it was uploaded" in result["message"]
    assert "was not uploaded" not in result["message"]
    assert "not created" not in result["message"]


def test_upload_file_does_not_take_a_same_name_folder_for_the_upload(
    http, monkeypatch, _isolate_environment
):
    """A same-name folder is only another candidate: it is never the upload's
    result, and it is listed for the user to check like any other file."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    folder = _same_name_folder()
    http.proxy("name = ", 200, {"files": [folder]})
    _timeout_on_execute(monkeypatch)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "file" not in result
    assert f"id '{folder['id']}'" in result["message"]
    assert "It is not known whether this call uploaded a file" in result["message"]
    assert result["candidates"] == [_candidate(folder)]


def test_upload_file_treats_a_408_as_unknown_and_looks_the_file_up(
    http, monkeypatch, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.proxy("name = ", 200, {"files": []})

    def timed_out(*args: Any, **kwargs: Any) -> Any:
        raise composio_tools.ComposioToolError(
            "Composio timed out waiting for the tool", status=408
        )

    monkeypatch.setattr(composio_tools, "execute_tool", timed_out)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    # A 408 can arrive after the tool ran, so it is not a definite rejection.
    assert result["status"] == "error"
    assert "rejected the request" not in result["message"]
    assert "not known whether it was uploaded" in result["message"]
    lookup = [call for call in http.proxy_calls if "name = " in str(call)]
    assert len(lookup) == 1


@pytest.mark.parametrize(
    ("status", "definite"),
    [
        (400, True),
        (404, True),
        (422, True),
        (429, True),
        (408, False),
        (500, False),
        (503, False),
        (None, False),
    ],
)
def test_a_composio_rejection_is_definite_only_for_a_4xx_other_than_408(
    status: int | None, definite: bool
):
    error = composio_tools.ComposioToolError("Composio refused the call", status=status)

    assert google_drive_composio._composio_rejection_is_definite(error) is definite


def test_upload_file_says_the_outcome_is_unknown_when_drive_cannot_be_checked(
    http, monkeypatch, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.proxy("name = ", 500, {"error": {"code": 500, "message": "backend error"}})
    _timeout_on_execute(monkeypatch)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "could not be checked" in result["message"]
    assert "not known whether it exists" in result["message"]
    assert "was not uploaded" not in result["message"]


def test_upload_file_says_nothing_was_uploaded_when_composio_rejects_the_call(
    http, monkeypatch, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)

    def reject(*args: Any, **kwargs: Any) -> Any:
        raise composio_tools.ComposioToolError(
            "Composio could not run the Google Drive tool (HTTP 422)",
            status=422,
        )

    monkeypatch.setattr(composio_tools, "execute_tool", reject)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "was not uploaded to Google Drive" in result["message"]
    assert "nothing was sent to Google Drive" in result["message"]
    assert "HTTP 422" in result["message"]
    # A definite rejection needs no lookup.
    assert http.proxy_calls == []


def test_upload_file_treats_a_5xx_as_unknown_and_looks_the_file_up(
    http, monkeypatch, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.proxy("name = ", 200, {"files": [_same_name_file()]})

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise composio_tools.ComposioToolError(
            "Composio could not run the Google Drive tool (HTTP 503)",
            status=503,
        )

    monkeypatch.setattr(composio_tools, "execute_tool", fail)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    # A 5xx can arrive after the tool created the file, so the answer is not a
    # definite failure: the lookup is made, and its finding is a candidate.
    assert result["status"] == "error"
    assert "It is not known whether this call uploaded a file" in result["message"]
    assert "was not uploaded" not in result["message"]
    assert f"id '{_NEW_ID}'" in result["message"]
    assert [call for call in http.proxy_calls if "name = " in str(call)] != []


def test_upload_file_refuses_a_resourcekey_parent_before_any_call(
    http, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    parent = "https://drive.google.com/drive/folders/folder1?resourcekey=0-Rkey"

    result = json.loads(
        google_drive.google_drive_upload_file(str(local), parent_id=parent)
    )

    assert result["status"] == "error"
    assert "resource key" in result["message"]
    assert "Nothing was uploaded" in result["message"]
    assert http.calls == []


def test_upload_file_does_not_report_success_when_drive_stored_other_bytes(
    http, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": _NEW_ID})
    http.proxy(
        "webViewLink",
        200,
        _created_metadata(md5Checksum="0" * 32, size=str(len(NEW_CONTENT))),
    )

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "checksum does not match" in result["message"]
    assert "would create a duplicate" in result["message"]


def test_upload_file_looks_the_file_up_when_the_read_back_is_a_404(
    http, _isolate_environment
):
    """The id the tool answered with does not resolve, so it proves nothing:
    the same-name files are reported as candidates instead of one of them
    being reported as a success."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    candidate = _same_name_file()
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data=_created_metadata())
    # The read-back is a files.get of the id the tool answered with; the
    # lookup that follows is a files.list, so the id names only the read-back.
    http.proxy(_NEW_ID, 404, {"error": {"code": 404, "message": "not found"}})
    http.proxy("name = ", 200, {"files": [candidate]})

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "file" not in result
    assert f"id '{candidate['id']}'" in result["message"]
    assert result["candidates"] == [_candidate(candidate)]
    lookup = [call for call in http.proxy_calls if "name = " in str(call)]
    assert len(lookup) == 1


def test_upload_file_keeps_the_tool_answer_when_the_read_back_fails(
    http, _isolate_environment
):
    """A read-back that fails for any other reason still leaves the tool's own
    answer: the file exists, and reporting a failure would invite a duplicate
    upload."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data=_created_metadata())
    http.proxy(_NEW_ID, 500, {"error": {"code": 500, "message": "backend error"}})

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "success"
    assert result["file"]["id"] == _NEW_ID
    assert result["file"]["name"] == "Report.pdf"


def test_upload_file_sends_a_type_outside_the_tool_enum_as_octet_stream(
    http, _isolate_environment
):
    """The create tool's mimeType is a closed enum. A legacy .doc type is not
    in it, so the tool is asked for application/octet-stream while the staged
    object keeps the file's real type."""
    local = _isolate_environment / "Notes.doc"
    local.write_bytes(NEW_CONTENT)
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": _NEW_ID})
    http.proxy(
        "webViewLink",
        200,
        _created_metadata(name="Notes.doc", mimeType="application/octet-stream"),
    )

    result = json.loads(
        google_drive.google_drive_upload_file(
            str(local), mime_type="application/msword"
        )
    )

    assert result["status"] == "success"
    assert _staged(http)["mimetype"] == "application/msword"
    assert http.tool_calls[0][1]["mimeType"] == "application/octet-stream"


def test_upload_file_does_not_take_an_older_same_name_file_for_the_upload(
    http, monkeypatch, _isolate_environment
):
    """An older file with the same name is a candidate like any other: it is
    listed for the user to check, and the upload stays unknown."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    older = _same_name_file(createdTime="2026-01-02T03:04:05.000Z")
    http.proxy("name = ", 200, {"files": [older]})
    _timeout_on_execute(monkeypatch)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "file" not in result
    assert f"id '{_NEW_ID}'" in result["message"]
    assert "It is not known whether this call uploaded a file" in result["message"]
    assert "the upload may still be finishing in Composio" not in result["message"]
    assert result["candidates"] == [_candidate(older)]


def test_the_create_lookup_searches_every_drive(
    http, monkeypatch, _isolate_environment
):
    """Without corpora=allDrives a file created in a shared drive is never
    found by the lookup."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.proxy("name = ", 200, {"files": []})
    _timeout_on_execute(monkeypatch)

    json.loads(google_drive.google_drive_upload_file(str(local)))

    lookup = [call for call in http.proxy_calls if "name = " in str(call)][0]
    query = _query(lookup)
    assert query["corpora"] == "allDrives"
    assert query["includeItemsFromAllDrives"] == "true"
    assert query["supportsAllDrives"] == "true"


def test_upload_file_says_nothing_was_sent_when_the_call_never_started(
    http, monkeypatch, _isolate_environment
):
    """A missing launch variable raises before any request goes out, so
    nothing reached Drive and no lookup is needed."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("COMPOSIO_USER_ID environment variable is missing")

    monkeypatch.setattr(composio_tools, "execute_tool", missing)

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "was not uploaded to Google Drive" in result["message"]
    assert "nothing was sent to Google Drive" in result["message"]
    assert "COMPOSIO_USER_ID" in result["message"]
    assert http.proxy_calls == []


def test_upload_file_does_not_say_nothing_was_sent_after_a_redirect(
    http, _isolate_environment
):
    """An answer of 307/308 means the POST went out and its result is not
    known: the redirect is not followed, so this is not a pre-send failure
    and must not be reported as one."""
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, status=307, data={})
    http.proxy("name = ", 200, {"files": []})

    result = json.loads(google_drive.google_drive_upload_file(str(local)))

    assert result["status"] == "error"
    assert "nothing was sent to Google Drive" not in result["message"]
    assert "not known whether it was uploaded" in result["message"]
    # The tool's own words say the request was sent.
    assert "redirect (HTTP 307)" in result["message"]
    assert "the request was sent" in result["message"]
    tool_call = [call for call in http.calls if "/tools/execute/" in call["url"]][0]
    assert tool_call["allow_redirects"] is False
    # One tool call, one lookup: a retry could create a duplicate.
    assert len(http.tool_calls) == 1
    assert len([call for call in http.proxy_calls if "name = " in str(call)]) == 1


def test_upload_file_never_puts_the_api_key_or_upload_url_in_an_error(
    http, _isolate_environment
):
    local = _isolate_environment / "Report.pdf"
    local.write_bytes(NEW_CONTENT)
    http.tool(
        google_drive_composio.GOOGLEDRIVE_CREATE_FILE,
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
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": "doc1"})
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
    assert slug == google_drive_composio.GOOGLEDRIVE_CREATE_FILE
    assert arguments["mimeType"] == GOOGLE_DOC_MIME
    assert arguments["parents"] == ["folder1"]
    assert (
        arguments["file_to_upload"]["s3key"]
        == "projects/pr_1/requests/googledrive/report.pdf"
    )


def test_create_file_keeps_a_plain_text_type(http):
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": "file1"})
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
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": "file1"})
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
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={})
    http.proxy("name = ", 200, {"files": []})

    result = json.loads(google_drive.google_drive_create_file("notes.txt", "hello"))

    assert result["status"] == "error"
    assert "did not confirm" in result["message"]
    assert "found no file named 'notes.txt'" in result["message"]
    assert "not known whether it was created" in result["message"]


def test_create_file_keeps_a_recent_same_name_doc_an_unknown_outcome(http, monkeypatch):
    """A same-name, same-type Google Doc created 20 seconds before this call,
    with the create's own answer lost, is the case this rule exists for: the
    lookup cannot show it is this call's Doc, so it is reported as a
    candidate and the create stays unknown rather than becoming a success."""
    earlier_doc = {
        "id": "doc1",
        "name": "Meeting notes",
        # The type the lookup used to compare. A Workspace file has no
        # checksum, which is exactly why the old no-checksum branch took it.
        "mimeType": GOOGLE_DOC_MIME,
        "createdTime": _seconds_ago(20),
    }
    http.proxy("name = ", 200, {"files": [earlier_doc]})
    calls = _timeout_on_execute(monkeypatch)

    result = json.loads(
        google_drive.google_drive_create_file(
            "Meeting notes", "hi", mime_type=GOOGLE_DOC_MIME
        )
    )

    assert result["status"] == "error"
    assert "file" not in result
    assert "'doc1'" in result["message"]
    assert earlier_doc["createdTime"] in result["message"]
    assert "may be the file this call was making" in result["message"]
    assert "It is not known whether this call created a file" in result["message"]
    assert result["candidates"] == [_candidate(earlier_doc)]
    assert len(calls) == 1


def test_create_file_keeps_a_verified_read_back_a_success(http):
    """The negative control for the rule above: when the tool's own answer is
    confirmed by Drive's read-back, the create is still a success."""
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": "doc1"})
    http.proxy(
        "webViewLink",
        200,
        {
            "id": "doc1",
            "name": "Meeting notes",
            "webViewLink": "https://docs.google.com/document/d/doc1/edit",
            "mimeType": GOOGLE_DOC_MIME,
        },
    )
    # Nothing is looked up: the read-back settled the result.
    http.proxy("name = ", 200, {"files": [_same_name_file(id="doc1")]})

    result = json.loads(
        google_drive.google_drive_create_file(
            "Meeting notes", "hi", mime_type=GOOGLE_DOC_MIME
        )
    )

    assert result["status"] == "success"
    assert result["file"]["id"] == "doc1"
    assert [call for call in http.proxy_calls if "name = " in str(call)] == []


def test_create_file_does_not_take_a_same_name_folder_for_the_created_doc(
    http, monkeypatch
):
    """A same-name folder is only another candidate, never the created Doc's
    result: the answer stays unknown and lists it to check."""
    folder = _same_name_folder(name="Meeting notes")
    http.proxy("name = ", 200, {"files": [folder]})
    _timeout_on_execute(monkeypatch)

    result = json.loads(
        google_drive.google_drive_create_file(
            "Meeting notes", "hi", mime_type=GOOGLE_DOC_MIME
        )
    )

    assert result["status"] == "error"
    assert "file" not in result
    assert f"id '{folder['id']}'" in result["message"]
    assert "It is not known whether this call created a file" in result["message"]
    assert result["candidates"] == [_candidate(folder)]


def test_create_file_says_the_outcome_is_unknown_when_the_lookup_finds_nothing(
    http, monkeypatch
):
    """A timed-out create may still be running in Composio, so a lookup that
    finds nothing is "not yet", not "not created"."""
    http.proxy("name = ", 200, {"files": []})
    _timeout_on_execute(monkeypatch)

    result = json.loads(google_drive.google_drive_create_file("notes.txt", "hello"))

    assert result["status"] == "error"
    assert "found no file named 'notes.txt'" in result["message"]
    assert "the create may still be finishing in Composio" in result["message"]
    assert "not known whether it was created" in result["message"]
    assert "was not created in Google Drive" not in result["message"]
    assert "not created under" not in result["message"]


def test_create_file_does_not_report_success_when_drive_stored_other_bytes(http):
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": "file1"})
    http.proxy(
        "webViewLink",
        200,
        {
            "id": "file1",
            "name": "notes.txt",
            "webViewLink": "https://drive.google.com/file/d/file1/view",
            "mimeType": "text/plain",
            "md5Checksum": "0" * 32,
            "size": "5",
        },
    )

    result = json.loads(google_drive.google_drive_create_file("notes.txt", "hello"))

    assert result["status"] == "error"
    assert "checksum does not match" in result["message"]
    assert "would create a duplicate" in result["message"]


def test_create_file_keeps_a_failed_tool_answer_an_unknown_outcome(http):
    """Composio can fail the create tool after Drive created the file, so a
    failed answer is no proof it does not exist."""
    http.tool(
        google_drive_composio.GOOGLEDRIVE_CREATE_FILE,
        successful=False,
        error="the drive account is not connected",
    )
    http.proxy("name = ", 200, {"files": []})

    result = json.loads(google_drive.google_drive_create_file("notes.txt", "hello"))

    assert result["status"] == "error"
    assert "the drive account is not connected" in result["message"]
    assert "not known whether it was created" in result["message"]
    assert "was not created in Google Drive" not in result["message"]
    lookup = [call for call in http.proxy_calls if "name = " in str(call)]
    assert len(lookup) == 1


def test_create_file_sends_a_type_outside_the_tool_enum_as_octet_stream(http):
    """text/x-python is a valid text type natively but not one of the create
    tool's mimeType values, so the tool is asked for octet-stream."""
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": "file1"})
    http.proxy(
        "webViewLink",
        200,
        {
            "id": "file1",
            "name": "gen.py",
            "webViewLink": "https://drive.google.com/file/d/file1/view",
            "mimeType": "application/octet-stream",
        },
    )

    result = json.loads(
        google_drive.google_drive_create_file(
            "gen.py", "print(1)", mime_type="text/x-python"
        )
    )

    assert result["status"] == "success"
    assert _staged(http)["mimetype"] == "text/x-python"
    assert http.tool_calls[0][1]["mimeType"] == "application/octet-stream"


def test_create_file_accepts_a_non_conversion_type_drive_stored_as_sent(http):
    """Outside the tool's enum, the tool is asked for octet-stream while the
    staged object keeps the real type. Drive may store either label, and a
    create that is not a conversion must not be reported as a failed one."""
    content = "print(1)"
    data = content.encode("utf-8")
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": "file1"})
    http.proxy(
        "webViewLink",
        200,
        {
            "id": "file1",
            "name": "gen.py",
            "webViewLink": "https://drive.google.com/file/d/file1/view",
            "mimeType": "text/x-python",
            "md5Checksum": _md5(data),
            "size": str(len(data)),
        },
    )

    result = json.loads(
        google_drive.google_drive_create_file(
            "gen.py", content, mime_type="text/x-python"
        )
    )

    assert result["status"] == "success"
    assert result["file"]["mimeType"] == "text/x-python"
    assert http.tool_calls[0][1]["mimeType"] == "application/octet-stream"


def test_create_file_reports_a_conversion_that_did_not_happen(http):
    """The bytes match, but the read-back type is not the requested one: the
    Google Docs conversion did not happen, so this is not a success."""
    content = "hello from the meeting"
    data = content.encode("utf-8")
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": "file1"})
    http.proxy(
        "webViewLink",
        200,
        {
            "id": "file1",
            "name": "Meeting notes",
            "webViewLink": "https://drive.google.com/file/d/file1/view",
            "mimeType": "text/plain",
            "md5Checksum": _md5(data),
            "size": str(len(data)),
        },
    )

    result = json.loads(
        google_drive.google_drive_create_file(
            "Meeting notes", content, mime_type=GOOGLE_DOC_MIME
        )
    )

    assert result["status"] == "error"
    assert "did not happen" in result["message"]
    assert "'text/plain'" in result["message"]
    assert f"'{GOOGLE_DOC_MIME}'" in result["message"]
    assert "would create a duplicate" in result["message"]


def test_create_file_does_not_take_an_unconverted_same_bytes_file_for_the_doc(
    http, monkeypatch
):
    """A lost answer plus a same-name file holding the uploaded bytes is only
    a candidate: nothing the lookup can see shows whether this call made it,
    or whether the requested conversion happened there."""
    content = NEW_CONTENT.decode("utf-8")
    file = _same_name_file(name="Meeting notes")
    http.proxy("name = ", 200, {"files": [file]})
    _timeout_on_execute(monkeypatch)

    result = json.loads(
        google_drive.google_drive_create_file(
            "Meeting notes", content, mime_type=GOOGLE_DOC_MIME
        )
    )

    assert result["status"] == "error"
    assert "file" not in result
    assert f"id '{_NEW_ID}'" in result["message"]
    assert "It is not known whether this call created a file" in result["message"]
    assert result["candidates"] == [_candidate(file)]


@pytest.mark.parametrize(
    "tool_data",
    [
        # The usual shape: only the new id.
        {"id": "doc1"},
        # A tool answer that claims the requested type is still its own claim,
        # not what Drive stored.
        {"id": "doc1", "name": "Meeting notes", "mimeType": GOOGLE_DOC_MIME},
    ],
)
def test_create_file_reports_an_unverified_conversion_when_the_type_is_unreadable(
    http, tool_data
):
    """The tool answered with an id, but the read-back failed, so the stored
    type is unknown: an id proves the file exists, not that Drive converted
    it, so this is an unverified outcome rather than a success."""
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data=tool_data)
    http.proxy("webViewLink", 500, {"error": {"code": 500, "message": "backend"}})

    result = json.loads(
        google_drive.google_drive_create_file(
            "Meeting notes", "hi", mime_type=GOOGLE_DOC_MIME
        )
    )

    assert result["status"] == "error"
    assert "file" not in result
    assert "'doc1'" in result["message"]
    assert f"'{GOOGLE_DOC_MIME}'" in result["message"]
    assert "not known whether the conversion" in result["message"]
    assert "unverified" in result["message"]
    assert "would create a duplicate" in result["message"]
    # The read-back failed, so there is no lookup either: the id is known.
    assert [call for call in http.proxy_calls if "name = " in str(call)] == []


def test_create_file_keeps_a_non_conversion_success_when_the_type_is_unreadable(
    http,
):
    """The negative control for the rule above: a create that asked for no
    conversion is not held to a type the read-back could not establish."""
    http.tool(google_drive_composio.GOOGLEDRIVE_CREATE_FILE, data={"id": "file1"})
    http.proxy("webViewLink", 500, {"error": {"code": 500, "message": "backend"}})

    result = json.loads(google_drive.google_drive_create_file("notes.txt", "hello"))

    assert result["status"] == "success"
    assert result["file"]["id"] == "file1"


# --- update_file_content on a Composio grant -------------------------------


def test_update_content_stages_and_runs_a_media_update(http, _isolate_environment):
    local = _replacement(_isolate_environment)
    _update(http, _current(), _updated())
    http.tool(google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE, data={})

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
    assert slug == google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE
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
        google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
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
        google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
        successful=False,
        error="Google Drive refused the upload",
    )

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "did not accept the new content" in result["message"]
    assert "Nothing was changed in Google Drive." in result["message"]
    assert "Google Drive refused the upload" in result["message"]


def test_update_content_says_the_outcome_is_unknown_after_a_timed_out_upload(
    http, monkeypatch, _isolate_environment
):
    """Our client-side timeout can arrive while Composio's tool is still
    running, so an unchanged read-back is "not yet", not "nothing changed"."""
    local = _replacement(_isolate_environment)
    _update(http, _current(), _current())
    calls = _timeout_on_execute(monkeypatch)

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "did not finish normally" in result["message"]
    assert "not known whether 'Quarterly Deck.pptx' changed" in result["message"]
    assert "Do not report it as updated or as unchanged" in result["message"]
    assert "version history" in result["message"]
    assert "Nothing was changed" not in result["message"]
    assert result["previous"]["headRevisionId"] == "rev-1"
    # One read-back settles the answer; the tool itself is not retried.
    assert len(calls) == 1


def test_update_content_keeps_a_composio_rejection_as_a_definite_failure(
    http, monkeypatch, _isolate_environment
):
    """A 4xx other than 408 is Composio refusing the call before the tool
    ran, so nothing changed."""
    local = _replacement(_isolate_environment)
    _update(http, _current(), _current())

    def reject(*args: Any, **kwargs: Any) -> Any:
        raise composio_tools.ComposioToolError(
            "Composio could not run the Google Drive tool (HTTP 422)",
            status=422,
        )

    monkeypatch.setattr(composio_tools, "execute_tool", reject)

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "did not accept the new content" in result["message"]
    assert "Nothing was changed in Google Drive." in result["message"]
    assert "not known" not in result["message"]
    assert result["file"]["headRevisionId"] == "rev-1"


def test_update_content_says_nothing_changed_when_composio_refuses_and_the_re_read_fails(
    http, monkeypatch, _isolate_environment
):
    """A 4xx other than 408 is a definite refusal: nothing was applied, so
    that answer does not depend on the read-back that failed."""
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, _current())
    http.proxy("md5Checksum", 500, {"error": {"code": 500, "message": "backend error"}})

    def reject(*args: Any, **kwargs: Any) -> Any:
        raise composio_tools.ComposioToolError(
            "Composio could not run the Google Drive tool (HTTP 422)",
            status=422,
        )

    monkeypatch.setattr(composio_tools, "execute_tool", reject)

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "did not accept the new content" in result["message"]
    assert "Nothing was changed in Google Drive." in result["message"]
    assert "HTTP 422" in result["message"]
    assert "not known" not in result["message"]
    # The file was not read again, so the pre-read state is what is reported.
    assert "file" not in result
    assert result["previous"]["headRevisionId"] == "rev-1"


def test_update_content_keeps_an_unclear_refusal_unknown_when_the_re_read_fails(
    http, monkeypatch, _isolate_environment
):
    """The negative control for the rule above: a 5xx may arrive after the
    write landed, so a read-back that fails leaves that one unknown."""
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, _current())
    http.proxy("md5Checksum", 500, {"error": {"code": 500, "message": "backend error"}})

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise composio_tools.ComposioToolError(
            "Composio could not run the Google Drive tool (HTTP 503)",
            status=503,
        )

    monkeypatch.setattr(composio_tools, "execute_tool", fail)

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "not known whether 'Quarterly Deck.pptx' changed" in result["message"]
    assert "Nothing was changed" not in result["message"]


def test_update_content_is_verified_by_drive_when_the_tool_reports_failure(
    http, _isolate_environment
):
    """A timeout or a 5xx can arrive after Drive stored the content. The
    checksum read after it decides the result, not the tool's answer."""
    local = _replacement(_isolate_environment)
    _update(http, _current(), _updated())
    http.tool(
        google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
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
    http.tool(google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE, data={})

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
    http.tool(google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE, data={})

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
        google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
        successful=False,
        error="the upload failed",
    )

    result = _call_update(local, expected_head_revision_id="rev-1")

    assert result["status"] == "error"
    assert "not known whether" in result["message"]
    assert "no longer exists" in result["message"]
    # A Composio grant has full Drive access, so the per-file caveat is gone.
    assert "per-file Drive access" not in result["message"]


def test_update_content_says_the_outcome_is_unknown_when_the_re_read_is_a_403(
    http, _isolate_environment
):
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, _current())
    http.proxy(
        "md5Checksum",
        403,
        {"error": {"code": 403, "message": "The user has not granted access."}},
    )
    http.tool(
        google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
        successful=False,
        error="the upload failed",
    )

    result = _call_update(local)

    assert result["status"] == "error"
    assert "not known whether" in result["message"]
    assert "refused permission" in result["message"]
    assert "updated or as unchanged" in result["message"]


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
    http.tool(google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE, data={})

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
        google_drive_composio.GOOGLEDRIVE_UPLOAD_UPDATE_FILE,
        successful=False,
        error=f"bad key {API_KEY}",
    )

    result = _call_update(local)

    message = result["message"]
    assert API_KEY not in message
    assert UPLOAD_URL not in message
    assert "X-Amz-Signature" not in message


def test_a_deduplicated_update_detects_a_local_change(
    http, _isolate_environment, monkeypatch
):
    """Composio deduplication skips the PUT, so the reader is never asked to
    verify the bytes; the file's size and mtime still catch a change that
    happened after the hash pass."""
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 4)
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, _current())
    original = local.read_bytes()

    def change_the_file() -> None:
        local.write_bytes(b"z" * len(original))

    http.before_upload = change_the_file
    http.upload_answer = {
        key: value
        for key, value in http.upload_answer.items()
        if key != "new_presigned_url"
    }

    result = _call_update(local)

    assert result["status"] == "error"
    assert "local file changed" in result["message"].lower()
    assert http.puts == []
    assert http.tool_calls == []


def test_update_content_refuses_a_resourcekey_file_before_staging(
    http, _isolate_environment
):
    """The update tool takes a bare fileId and cannot carry a resource key, so
    a link-shared file is refused before its bytes are staged."""
    local = _replacement(_isolate_environment)
    http.proxy("capabilities", 200, _current())
    file_id = "https://drive.google.com/file/d/deck1/view?resourcekey=0-Rkey"

    result = json.loads(
        google_drive.google_drive_update_file_content(file_id, str(local))
    )

    assert result["status"] == "error"
    assert "resource key" in result["message"]
    assert "Nothing was changed" in result["message"]
    assert http.calls_to("/files/upload/request") == []
    assert http.tool_calls == []


def test_the_staged_reader_reads_each_block_once():
    """The reader must serve a 16 KiB read from a block it already checked,
    not re-read and re-hash the whole 1 MiB block each time."""
    payload = bytes(range(256)) * (2 * 1024 * 1024 // 256) + b"tail"
    size = len(payload)
    digests = [
        google_drive._block_digest(
            payload[offset : offset + google_drive._HASH_BLOCK_BYTES]
        )
        for offset in range(0, size, google_drive._HASH_BLOCK_BYTES)
    ]

    class CountingFile:
        def __init__(self, data: bytes) -> None:
            self._io = io.BytesIO(data)
            self.bytes_read = 0

        def seek(self, offset: int, whence: int = 0) -> int:
            return self._io.seek(offset, whence)

        def read(self, amount: int = -1) -> bytes:
            chunk = self._io.read(amount)
            self.bytes_read += len(chunk)
            return chunk

    counted = CountingFile(payload)
    reader = google_drive_composio._VerifiedStagedReader(
        counted, size=size, block_digests=digests
    )
    chunks: list[bytes] = []
    while chunk := reader.read(16 * 1024):
        chunks.append(chunk)

    assert b"".join(chunks) == payload
    # Every byte was read once; the old implementation read each 1 MiB block
    # 64 times.
    assert counted.bytes_read == size


def test_the_staged_reader_still_refuses_a_block_changed_after_the_first_read():
    payload = (
        b"a" * google_drive._HASH_BLOCK_BYTES + b"b" * google_drive._HASH_BLOCK_BYTES
    )
    size = len(payload)
    digests = [
        google_drive._block_digest(
            payload[offset : offset + google_drive._HASH_BLOCK_BYTES]
        )
        for offset in range(0, size, google_drive._HASH_BLOCK_BYTES)
    ]
    fh = io.BytesIO(payload)
    reader = google_drive_composio._VerifiedStagedReader(
        fh, size=size, block_digests=digests
    )

    assert reader.read(16 * 1024) == b"a" * (16 * 1024)
    # Change the second block after the first one has been cached.
    fh.seek(google_drive._HASH_BLOCK_BYTES)
    fh.write(b"z" * google_drive._HASH_BLOCK_BYTES)

    with pytest.raises(google_drive._ReplacementChangedError):
        while reader.read(16 * 1024):
            pass


def test_a_tool_answer_prefers_a_nested_file_and_requires_a_drive_id():
    assert (
        google_drive_composio._tool_file_object(
            {"id": "execution-1", "file": {"id": "file-1"}}
        )["id"]
        == "file-1"
    )
    assert (
        google_drive_composio._tool_file_object(
            {"id": "execution-1", "data": {"id": "file-1"}}
        )["id"]
        == "file-1"
    )
    # A blank or execution-shaped id is not a Drive file id.
    assert google_drive_composio._tool_file_object({"id": " "}) == {}
    assert google_drive_composio._tool_file_object({"id": "not a drive id"}) == {}


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
