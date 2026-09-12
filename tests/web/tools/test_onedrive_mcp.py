import base64
import io
import json
import os
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import onedrive


class MockResponse:
    def __init__(
        self, json_data=None, status_code=200, content=b"{}", url=None, headers=None
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        # Real requests.HTTPError messages embed the request URL (e.g.
        # "500 Server Error: ... for url: https://...") -- defaulting this
        # to a URL-shaped string rather than leaving it out entirely means
        # every test that triggers raise_for_status() exercises the real
        # message shape the production code must keep out of caller-visible
        # errors, not a synthetic one with no URL to protect at all.
        self.url = url or "https://upload.example/session-default"
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Error for url: {self.url}",
                response=self,
            )


class _FakeSession:
    """Stand-in for requests.Session() used by _upload_large_file_content --
    a plain Mock doesn't support the `with ... as` context-manager protocol
    on its own, and mocking Session.put/.delete at the class level would
    leak between tests since the module only ever creates one Session."""

    def __init__(self, put=None, get=None, delete=None):
        self.put = put if put is not None else Mock(return_value=MockResponse({}))
        self.get = (
            get
            if get is not None
            else Mock(return_value=MockResponse({"nextExpectedRanges": ["0-"]}))
        )
        self.delete = (
            delete if delete is not None else Mock(return_value=MockResponse({}))
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _patch_session(monkeypatch, fake_session):
    monkeypatch.setattr(onedrive.requests, "Session", Mock(return_value=fake_session))


def _quickxor_hash(data: bytes, *split_points: int) -> str:
    digest = onedrive._QuickXorHash()
    start = 0
    for end in (*split_points, len(data)):
        digest.update(data[start:end])
        start = end
    return digest.base64_digest()


def _microsoft_quickxor_reference(data: bytes) -> str:
    """Direct oracle from Microsoft's published rotate/XOR pseudocode.

    This intentionally uses one 160-bit rotation per input byte instead of
    production's optimized 160-byte period folding:
    https://learn.microsoft.com/onedrive/developer/code-snippets/quickxorhash
    """
    width = 160
    mask = (1 << width) - 1
    value = 0
    for index, byte in enumerate(data):
        shift = (index * 11) % width
        rotated = byte if shift == 0 else ((byte << shift) | (byte >> (width - shift)))
        value ^= rotated & mask
    digest = bytearray(value.to_bytes(20, "little"))
    for index, byte in enumerate(len(data).to_bytes(8, "little")):
        digest[12 + index] ^= byte
    return base64.b64encode(digest).decode("ascii")


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")
    monkeypatch.setattr(onedrive.time, "sleep", Mock())


@pytest.fixture(autouse=True)
def _upload_allowed_dirs_env(tmp_path, monkeypatch):
    """Scope onedrive_upload_file's read allowlist to an isolated per-test
    directory so tests aren't order-dependent on whatever the real working
    directory holds."""
    allowed_dir = tmp_path / "workspace"
    allowed_dir.mkdir()
    monkeypatch.setenv("XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS", str(allowed_dir))
    return allowed_dir


# ---------------------------------------------------------------------------
# onedrive_upload_file
# ---------------------------------------------------------------------------


def test_quickxor_hash_matches_microsoft_reference_vector_across_updates():
    # Values generated from Microsoft's published per-byte 160-bit rotate/XOR
    # reference algorithm, rather than from the implementation under test.
    assert _quickxor_hash(b"hello world", 1, 5, 9) == ("aCgDG9jwBhDc4Q1yawMZAAAAAAA=")
    assert _quickxor_hash(bytes(range(21)), 7, 19, 20) == (
        "4AmQiIZEpkIZ4AAIXYACFsCABjg="
    )
    periodic_data = bytes(range(256)) * 2
    assert _quickxor_hash(periodic_data, 7, 159, 160, 161, 333) == (
        "edJlP68QDhntUYpkxf/vpP5uDuY="
    )


@pytest.mark.parametrize("size", [21, 159, 160, 161, 512, 4097])
def test_quickxor_hash_matches_independent_microsoft_algorithm(size):
    data = bytes((index * 37 + 11) % 256 for index in range(size))
    split_points = tuple(point for point in (13, 157, 401) if point < size)
    assert _quickxor_hash(data, *split_points) == _microsoft_quickxor_reference(data)


@pytest.mark.asyncio
async def test_upload_file_is_registered_with_expected_schema():
    tools = {tool.name: tool for tool in await onedrive.mcp.list_tools()}

    assert "onedrive_upload_file" in tools
    schema = tools["onedrive_upload_file"].inputSchema
    assert schema["required"] == ["local_file_path"]
    assert set(schema["properties"]) == {
        "local_file_path",
        "remote_path",
        "mime_type",
    }


def test_upload_file_sends_real_binary_content(monkeypatch, _upload_allowed_dirs_env):
    """Regression guard for the actual production bug: uploading an
    already-generated spreadsheet must send its real bytes with a real
    mimeType — not a text/plain placeholder string, which is all
    onedrive_upload_text_file's str content parameter can carry."""
    local_file = _upload_allowed_dirs_env / "Regional_Performance_Data-v3.xlsx"
    local_file.write_bytes(b"PK\x03\x04 fake xlsx bytes")

    mock_request = Mock(
        return_value=MockResponse(
            {
                "id": "item-1",
                "name": "Regional_Performance_Data-v3.xlsx",
                "file": {
                    "mimeType": (
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    )
                },
            }
        )
    )
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert result["item"]["name"] == "Regional_Performance_Data-v3.xlsx"

    kwargs = mock_request.call_args.kwargs
    assert kwargs["method"] == "PUT"
    assert kwargs["url"].endswith(
        "/me/drive/root:/Regional_Performance_Data-v3.xlsx:/content"
    )
    assert kwargs["data"] == b"PK\x03\x04 fake xlsx bytes"
    assert kwargs["headers"]["Content-Type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert kwargs["timeout"] == onedrive._BINARY_UPLOAD_TIMEOUT_SECONDS


def test_upload_file_accepts_explicit_remote_path_and_mime_type(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "data.bin"
    local_file.write_bytes(b"\x00\x01\x02")

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(local_file),
            remote_path="Sandbox/custom.dat",
            mime_type="application/octet-stream",
        )
    )

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["url"].endswith("/me/drive/root:/Sandbox/custom.dat:/content")
    assert kwargs["headers"]["Content-Type"] == "application/octet-stream"


def test_upload_file_defaults_mime_type_when_unguessable(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "mystery_file_no_extension"
    local_file.write_bytes(b"some bytes")

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["headers"]["Content-Type"] == "application/octet-stream"


def test_upload_file_guesses_mime_type_from_remote_name(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "generated-artifact"
    local_file.write_bytes(b"%PDF-1.7")
    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(local_file), remote_path="Documents/report.pdf"
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["headers"]["Content-Type"] == (
        "application/pdf"
    )


def test_upload_file_falls_back_to_local_name_for_mime_type(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"%PDF-1.7")
    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(str(local_file), remote_path="report")
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["headers"]["Content-Type"] == (
        "application/pdf"
    )


@pytest.mark.parametrize(
    "file_name,inner_type",
    [("report.pdf.gz", "application/pdf"), ("data.csv.gz", "text/csv")],
)
def test_upload_file_resolves_gzip_content_type_not_the_inner_type(
    monkeypatch, _upload_allowed_dirs_env, file_name, inner_type
):
    """Regression guard: mimetypes.guess_type("report.pdf.gz") returns
    ("application/pdf", "gzip") -- the type element describes the
    *decompressed* content, not what's actually going out over the wire.
    Blindly using that type element alone as Content-Type (discarding the
    encoding element) would tell any client trusting that header to parse
    a raw gzip stream as an uncompressed PDF (or CSV)."""
    local_file = _upload_allowed_dirs_env / file_name
    local_file.write_bytes(b"\x1f\x8b\x08\x00fake gzip bytes")

    # Confirm the premise directly against the real stdlib, independent of
    # whatever this host's mime.types happens to add on top.
    assert onedrive.mimetypes.guess_type(file_name) == (inner_type, "gzip")

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["headers"]["Content-Type"] == (
        "application/gzip"
    )


@pytest.mark.parametrize(
    "extension,expected_mime_type",
    [
        (
            ".docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        (
            ".pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        (".odt", "application/vnd.oasis.opendocument.text"),
    ],
)
def test_upload_file_resolves_ooxml_mime_type_without_relying_on_host_mime_db(
    monkeypatch, _upload_allowed_dirs_env, extension, expected_mime_type
):
    """Regression guard: stdlib mimetypes.guess_type() only recognizes OOXML/
    ODF extensions when a system mime.types file happens to be installed —
    verified directly (mimetypes.MimeTypes(filenames=()) returns (None, None)
    for all of these). A minimal/slim container image has no such file, so
    without _MIME_TYPE_OVERRIDES these would silently fall back to
    "application/octet-stream" instead of the correct, real mime type."""
    local_file = _upload_allowed_dirs_env / f"report{extension}"
    local_file.write_bytes(b"binary content")

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert (
        mock_request.call_args.kwargs["headers"]["Content-Type"] == expected_mime_type
    )


def test_upload_file_rejects_empty_or_root_remote_path(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: remote_path="/" previously reached _content_path
    (simple-PUT path) as an effectively empty target, raising a confusing
    "file_path is required" that names the wrong parameter, or reached
    _item_path (large-file path) silently building a request against the
    drive root itself instead of a named file."""
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file), remote_path="/"))

    assert result["status"] == "error"
    assert "remote_path" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize("remote_path", ["Documents/", "Documents/Reports/"])
def test_upload_file_rejects_trailing_slash_remote_path(
    monkeypatch, _upload_allowed_dirs_env, remote_path
):
    """Regression guard: a trailing "/" reads as "put it in this folder" --
    the obvious way an LLM caller would express that intent -- but
    _normalize_path strips it right off, so without this check the file
    would silently be uploaded as an item literally *named* "Documents"
    (or "Reports") at the parent location instead of placed inside that
    folder, with no error and nothing to signal the mistake."""
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(str(local_file), remote_path=remote_path)
    )

    assert result["status"] == "error"
    assert "filename" in result["message"]
    mock_request.assert_not_called()


def test_upload_file_rejects_trailing_period_remote_path(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(local_file), remote_path="Documents/report.pdf."
        )
    )

    assert result["status"] == "error"
    assert "period" in result["message"]
    mock_request.assert_not_called()


def test_upload_file_accepts_remote_path_with_folder_and_filename(
    monkeypatch, _upload_allowed_dirs_env
):
    """Complementary case: a remote_path that already includes a filename
    after the folder must still work normally."""
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    mock_request = Mock(return_value=MockResponse({"id": "f1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(local_file), remote_path="Documents/report.pdf"
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/me/drive/root:/Documents/report.pdf:/content"
    )


@pytest.mark.parametrize(
    "traversal_remote_path",
    [
        "../../etc/passwd",
        "../../../../me/messages",
        "foo/../../bar",
        "./secret",
        r"..\..\me\messages",
    ],
)
def test_upload_file_rejects_dot_segments_in_remote_path(
    monkeypatch, _upload_allowed_dirs_env, traversal_remote_path
):
    """Regression guard for a confirmed request-forgery bug: requests'
    own URL preparation collapses ".." segments the same way a browser
    does (verified directly: 'root:/../../etc/x:/content' becomes
    '/me/etc/x:/content'), so an unvalidated remote_path could walk the
    actual HTTP request Graph receives entirely out of
    '/me/drive/root:/' and onto a different, unrelated Graph API endpoint
    under the same OAuth token -- not just the wrong file within Drive."""
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(local_file), remote_path=traversal_remote_path
        )
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_normalize_path_rejects_dot_segments_directly():
    """Direct regression guard on the shared choke point every path-
    building helper (_item_path/_children_path/_content_path) goes
    through, independent of which public tool calls it."""
    with pytest.raises(ValueError, match=r"\.\.|\bpath must not contain"):
        onedrive._normalize_path("../../etc/passwd")
    with pytest.raises(ValueError):
        onedrive._normalize_path("a/../b")
    with pytest.raises(ValueError):
        onedrive._normalize_path("./a")
    # A path with no dot-segments at all must be unaffected.
    assert onedrive._normalize_path("Documents/report.pdf") == "Documents/report.pdf"


@pytest.mark.parametrize(
    "call",
    [
        lambda: onedrive.onedrive_list_items(folder_path="../../etc"),
        lambda: onedrive.onedrive_get_item(path="../../etc/passwd"),
        lambda: onedrive.onedrive_get_file_content("../../etc/passwd"),
        lambda: onedrive.onedrive_create_folder("new-folder", parent_path="../../etc"),
        lambda: onedrive.onedrive_upload_text_file("../../etc/passwd", "x"),
    ],
    ids=[
        "onedrive_list_items",
        "onedrive_get_item",
        "onedrive_get_file_content",
        "onedrive_create_folder",
        "onedrive_upload_text_file",
    ],
)
def test_path_based_tools_reject_dot_segments_end_to_end(monkeypatch, call):
    """Regression guard: _normalize_path's dot-segment rejection is unit-
    tested directly above, and onedrive_upload_file's own remote_path is
    covered separately -- but that leaves the five *pre-existing* tools
    that also route a caller-supplied path through _item_path/
    _children_path/_content_path (and therefore _normalize_path)
    unexercised end-to-end. A future refactor that accidentally bypassed
    _normalize_path for one of these specific call sites (while the
    direct unit test above kept passing) would ship undetected without
    this."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(call())

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_upload_file_uses_upload_session_for_large_files(
    monkeypatch, _upload_allowed_dirs_env
):
    """Files over Graph's ~4MB simple-PUT cap must go through
    createUploadSession + chunked PUTs instead of a single content PUT."""
    local_file = _upload_allowed_dirs_env / "big.bin"
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    # Distinguishable per-region content (not a single repeated byte) so the
    # assertions below can catch a chunk-boundary regression in the read-
    # from-disk path (e.g. an off-by-one that shifts bytes between chunks)
    # that a uniform b"\x01" * total_size body would silently pass.
    first_chunk_bytes = bytes([1]) * chunk_size
    second_chunk_bytes = bytes([2]) * (total_size - chunk_size)
    local_file.write_bytes(first_chunk_bytes + second_chunk_bytes)

    mock_request = Mock(
        return_value=MockResponse({"uploadUrl": "https://upload.example/session-1"})
    )
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    mock_put = Mock(
        side_effect=[
            MockResponse({}, content=b""),
            MockResponse({"id": "item-1", "name": "big.bin"}),
        ]
    )
    _patch_session(monkeypatch, _FakeSession(put=mock_put))

    result = json.loads(
        onedrive.onedrive_upload_file(str(local_file), mime_type="application/x-custom")
    )

    assert result["status"] == "success"
    assert result["item"]["id"] == "item-1"

    session_call = mock_request.call_args
    assert session_call.kwargs["method"] == "POST"
    assert session_call.kwargs["url"].endswith(
        "/me/drive/root:/big.bin:/createUploadSession"
    )
    assert session_call.kwargs["json"] == {
        "item": {"@microsoft.graph.conflictBehavior": "replace"}
    }

    assert mock_put.call_count == 2
    first_call, second_call = mock_put.call_args_list
    assert first_call.args[0] == "https://upload.example/session-1"
    assert first_call.kwargs["data"] == first_chunk_bytes
    assert second_call.kwargs["data"] == second_chunk_bytes
    assert first_call.kwargs["headers"]["Content-Range"] == (
        f"bytes 0-{chunk_size - 1}/{total_size}"
    )
    assert second_call.kwargs["headers"]["Content-Range"] == (
        f"bytes {chunk_size}-{total_size - 1}/{total_size}"
    )
    # The explicit mime_type must reach every chunk, not just the simple-PUT
    # branch — Graph doesn't document Content-Type as authoritative for the
    # resumable path, but there's no other client-controllable lever, and
    # sending it costs nothing.
    assert first_call.kwargs["headers"]["Content-Type"] == "application/x-custom"
    assert second_call.kwargs["headers"]["Content-Type"] == "application/x-custom"
    # Chunk uploads use a longer timeout than the small-JSON-call default —
    # a multi-megabyte PUT over a slow link can legitimately take longer
    # than DEFAULT_TIMEOUT_SECONDS.
    assert first_call.kwargs["timeout"] == onedrive._BINARY_UPLOAD_TIMEOUT_SECONDS
    # The pre-authenticated upload session URL must never carry our own
    # Authorization header alongside its own query-string token.
    assert "Authorization" not in first_call.kwargs["headers"]
    assert "Authorization" not in second_call.kwargs["headers"]


def test_upload_large_file_content_reads_bounded_chunks_from_disk(monkeypatch):
    """Regression guard for the actual production bug's efficiency half:
    _upload_large_file_content must pull each chunk straight off the file
    handle rather than the caller loading the whole file into memory first
    — verified directly against the function (not through the mimetypes/
    allowlist plumbing of onedrive_upload_file) by tracking the largest
    single read() request it issues against a fake file object."""
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = 2 * chunk_size + 10
    content = bytes(range(256)) * (total_size // 256 + 1)
    content = content[:total_size]

    class _TrackingBuffer(io.BytesIO):
        max_read_size = 0

        def read(self, size=-1, *a, **kw):
            if isinstance(size, int) and size > 0:
                type(self).max_read_size = max(type(self).max_read_size, size)
            return super().read(size, *a, **kw)

    fh = _TrackingBuffer(content)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    put_calls = []

    def _fake_put(url, data, headers, timeout):
        put_calls.append(data)
        is_last = sum(len(d) for d in put_calls) >= total_size
        return MockResponse(
            {"id": "item-1"} if is_last else {}, content=b"{}" if is_last else b""
        )

    _patch_session(monkeypatch, _FakeSession(put=Mock(side_effect=_fake_put)))

    result = onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    assert result == {"id": "item-1"}
    assert b"".join(put_calls) == content
    assert _TrackingBuffer.max_read_size <= chunk_size


def test_upload_large_file_content_handles_exact_chunk_multiple(monkeypatch):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = 2 * chunk_size
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(
        side_effect=[
            MockResponse({}, status_code=202),
            MockResponse({"id": "item-1"}, status_code=201),
        ]
    )
    _patch_session(monkeypatch, _FakeSession(put=mock_put))

    result = onedrive._upload_large_file_content(
        "big.bin",
        io.BytesIO(b"\x00" * total_size),
        total_size,
        "application/octet-stream",
    )

    assert result == {"id": "item-1"}
    assert mock_put.call_count == 2
    assert mock_put.call_args_list[-1].kwargs["headers"]["Content-Range"] == (
        f"bytes {chunk_size}-{total_size - 1}/{total_size}"
    )


def test_upload_large_file_content_does_not_forward_chunk_error_body(
    monkeypatch,
):
    """Upload responses can echo a preauthenticated URL in arbitrary
    encodings, so their body must never be exposed to the caller."""
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    error_response = MockResponse(
        {"error": {"message": "Invalid upload session"}},
        status_code=400,
        content=b'{"error": {"message": "Invalid upload session"}}',
    )
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(
        monkeypatch,
        _FakeSession(put=Mock(return_value=error_response), delete=mock_delete),
    )

    with pytest.raises(RuntimeError, match="HTTP 400") as exc_info:
        onedrive._upload_large_file_content(
            "big.bin", fh, total_size, "application/octet-stream"
        )

    assert "Invalid upload session" not in str(exc_info.value)

    # A failed upload must cancel its now-abandoned session immediately
    # rather than leaving it for Graph's own ~15-minute expiry. The
    # cancellation itself carries no body, so it uses the short default
    # timeout rather than the long one sized for a multi-megabyte PUT --
    # otherwise a stalled cleanup call would needlessly delay surfacing the
    # real failure by up to another _BINARY_UPLOAD_TIMEOUT_SECONDS.
    mock_delete.assert_called_once_with(
        "https://upload.example/s", timeout=onedrive.DEFAULT_TIMEOUT_SECONDS
    )


def test_upload_large_file_content_never_leaks_upload_url_on_chunk_failure(
    monkeypatch, caplog
):
    """Regression guard: Graph's preauthenticated upload-session URL is
    itself usable for PUT/GET/DELETE without the OAuth bearer token, so a
    rejected chunk must never format requests' own HTTPError (whose default
    message embeds the full request URL) into the error the caller/LLM
    sees or into a log line."""
    sentinel_url = "https://upload.example/session-with-a-secret-token-abc123"
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": sentinel_url})),
    )
    error_response = MockResponse(
        {"error": {"message": "range conflict"}},
        status_code=416,
        content=b'{"error": {"message": "range conflict"}}',
    )
    _patch_session(monkeypatch, _FakeSession(put=Mock(return_value=error_response)))

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError) as exc_info:
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    assert sentinel_url not in str(exc_info.value)
    assert "made no forward progress" in str(exc_info.value)
    for record in caplog.records:
        assert sentinel_url not in record.getMessage()


@pytest.mark.parametrize(
    "echoed_url",
    [
        "https://upload.example/session?tempauth=SECRETVALUE&foo=bar",
        "https://upload.example/session?tempauth=SECRETVALUE&amp;foo=bar",
        "/session?tempauth%3DSECRETVALUE%26foo%3Dbar",
    ],
)
def test_upload_large_file_content_never_forwards_encoded_url_from_response_body(
    monkeypatch, caplog, echoed_url
):
    """A proxy/WAF can echo the upload URL using encodings that defeat
    string-replacement redaction, so the response body is discarded."""
    sentinel_url = "https://upload.example/session?tempauth=SECRETVALUE&foo=bar"
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": sentinel_url})),
    )
    waf_body = f"Blocked: request to {echoed_url} was denied".encode()
    error_response = MockResponse({}, status_code=403, content=waf_body)
    _patch_session(monkeypatch, _FakeSession(put=Mock(return_value=error_response)))

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError) as exc_info:
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    assert "SECRETVALUE" not in str(exc_info.value)
    assert "Blocked" not in str(exc_info.value)
    assert "HTTP 403" in str(exc_info.value)
    assert "SECRETVALUE" not in caplog.text


def test_upload_large_file_content_never_leaks_upload_url_on_transport_failure(
    monkeypatch, caplog
):
    """Regression guard: a chunk PUT that fails at the transport layer
    (connection error, timeout, TLS failure) before any HTTP response
    exists at all raises a requests exception whose own default message
    embeds the full request URL (verified directly against a real failed
    request) -- a sanitize step that only covers the "got a non-2xx
    response" path would miss this entirely. Every exception the chunk
    loop can raise must have the URL scrubbed, not just HTTPError."""
    sentinel_url = "https://upload.example/session-with-a-secret-token-abc123"
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": sentinel_url})),
    )
    transport_error = requests.ConnectionError(
        f"HTTPSConnectionPool(...): Max retries exceeded with url: "
        f"{sentinel_url} (Caused by ...)"
    )
    _patch_session(monkeypatch, _FakeSession(put=Mock(side_effect=transport_error)))

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError) as exc_info:
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    assert sentinel_url not in str(exc_info.value)
    for record in caplog.records:
        assert sentinel_url not in record.getMessage()


def test_upload_large_file_content_never_leaks_token_when_host_and_path_are_reported_separately(
    monkeypatch, caplog
):
    """Regression guard for the real urllib3 message shape, not just a
    synthetic one: verified directly against an actual failed request that
    a genuine ConnectionError/SSLError never embeds "scheme://host/path?
    query" as one contiguous string the way the test above's synthetic
    message does -- it reports the host separately (inside
    "HTTPSConnectionPool(host=..., port=...)") from the path+query (inside
    "Max retries exceeded with url: /path?query"). A sanitize step built
    around a single `message.replace(upload_url, ...)` passes the
    synthetic-message test above while still leaking the token-bearing
    query string here."""
    sentinel_host = "upload.example.invalid"
    sentinel_token = "SECRETTOKEN123"
    sentinel_url = f"https://{sentinel_host}/session-1?token={sentinel_token}"
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": sentinel_url})),
    )
    transport_error = requests.ConnectionError(
        f"HTTPSConnectionPool(host='{sentinel_host}', port=443): Max "
        f"retries exceeded with url: /session-1?token={sentinel_token} "
        "(Caused by SSLError(...))"
    )
    _patch_session(monkeypatch, _FakeSession(put=Mock(side_effect=transport_error)))

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError) as exc_info:
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    assert sentinel_token not in str(exc_info.value)
    assert sentinel_host not in str(exc_info.value)
    for record in caplog.records:
        assert sentinel_token not in record.getMessage()
        assert sentinel_host not in record.getMessage()


def test_upload_large_file_content_treats_cleanup_404_as_fine(monkeypatch, caplog):
    """Regression guard: a 404 on the cancellation DELETE means the session
    is already gone (expired, or already completed/cancelled) -- exactly
    the outcome cleanup wants, not a failure of it, so it must not warn."""
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    error_response = MockResponse(
        {"error": {"message": "boom"}}, status_code=400, content=b'{"error": "boom"}'
    )
    delete_404_response = MockResponse({}, status_code=404)
    mock_delete = Mock(return_value=delete_404_response)
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(return_value=error_response),
            delete=mock_delete,
        ),
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="HTTP 400"):
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    mock_delete.assert_called_once()
    assert not any(
        "cancellation returned" in record.getMessage() for record in caplog.records
    )


def test_upload_large_file_content_rejects_non_positive_total(monkeypatch):
    """Regression guard: total=0 would make `range(0, 0, chunk_size)` skip
    the loop entirely, silently returning the empty `result` this function
    initializes before ever running the "did OneDrive confirm this"
    check -- not reachable through onedrive_upload_file today, but this
    function should refuse to silently no-op if called directly."""
    monkeypatch.setattr(onedrive.requests, "request", Mock())

    with pytest.raises(ValueError, match="must be positive"):
        onedrive._upload_large_file_content(
            "big.bin", io.BytesIO(b""), 0, "application/octet-stream"
        )


def test_upload_large_file_content_warns_without_leaking_url_when_cleanup_fails(
    monkeypatch, caplog
):
    """Regression guard: if the best-effort cancellation DELETE itself
    raises (e.g. a network-level requests exception, whose own message
    commonly embeds the request URL), the warning log must have that URL
    scrubbed out of the logged message rather than leaking it verbatim."""
    sentinel_url = "https://upload.example/session-with-a-secret-token-abc123"
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": sentinel_url})),
    )
    error_response = MockResponse({}, status_code=400, content=b"{}")
    cleanup_error = requests.ConnectionError(f"Failed to reach {sentinel_url}")
    mock_delete = Mock(side_effect=cleanup_error)
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(return_value=error_response),
            delete=mock_delete,
        ),
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError):
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    for record in caplog.records:
        assert sentinel_url not in record.getMessage()
        assert sentinel_url not in caplog.text
    mock_delete.assert_called_once()
    assert "Failed to cancel abandoned" in caplog.text


def test_upload_large_file_content_warns_on_failed_cleanup_status(monkeypatch, caplog):
    """Regression guard: requests doesn't raise on its own for an HTTP
    error status -- a 429/5xx response to the cancellation DELETE must not
    be silently treated as a successful cancellation.

    Uses a 400 chunk failure so the test reaches cleanup immediately."""
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    error_response = MockResponse(
        {"error": {"message": "boom"}}, status_code=400, content=b'{"error": "boom"}'
    )
    delete_failure_response = MockResponse({}, status_code=429)
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(return_value=error_response),
            delete=Mock(return_value=delete_failure_response),
        ),
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="HTTP 400"):
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    assert any("429" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_upload_large_file_content_retries_then_cancels_transient_http_failure(
    monkeypatch, caplog, status_code
):
    """Transient failures retry on the same session; an exhausted session
    is cancelled because no cross-call resume state is persisted."""
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    error_response = MockResponse({}, status_code=status_code)
    mock_put = Mock(return_value=error_response)
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(
        monkeypatch,
        _FakeSession(put=mock_put, delete=mock_delete),
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError):
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    assert mock_put.call_count == onedrive._UPLOAD_CHUNK_MAX_ATTEMPTS
    assert onedrive.time.sleep.call_count == onedrive._UPLOAD_CHUNK_MAX_ATTEMPTS - 1
    mock_delete.assert_called_once()
    assert "Retrying OneDrive upload fragment" in caplog.text


def test_upload_large_file_content_retries_then_cancels_network_failure(
    monkeypatch, caplog
):
    """Transport failures use the same bounded retry and cleanup path."""
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(side_effect=requests.ConnectionError("boom"))
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(
        monkeypatch,
        _FakeSession(put=mock_put, delete=mock_delete),
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError):
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    assert mock_put.call_count == onedrive._UPLOAD_CHUNK_MAX_ATTEMPTS
    mock_delete.assert_called_once()


def test_upload_large_file_content_recovers_from_transient_failure(monkeypatch):
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    transient = MockResponse({}, status_code=503)
    intermediate = MockResponse({}, status_code=202, content=b"")
    completed = MockResponse({"id": "item-1"})
    mock_put = Mock(side_effect=[transient, intermediate, completed])
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(monkeypatch, _FakeSession(put=mock_put, delete=mock_delete))

    result = onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    assert result == {"id": "item-1"}
    assert mock_put.call_count == 3
    onedrive.time.sleep.assert_called_once()
    mock_delete.assert_not_called()


@pytest.mark.parametrize(
    "ambiguous_result",
    [
        requests.ConnectionError("response lost"),
        MockResponse({}, status_code=416),
    ],
)
def test_upload_large_file_content_advances_when_server_has_fragment(
    monkeypatch, ambiguous_result
):
    """A lost response or 416 is reconciled before resending the range."""
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    fh = io.BytesIO(b"\x00" * total_size)
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(
        side_effect=[
            ambiguous_result,
            MockResponse({"id": "item-1", "size": total_size}),
        ]
    )
    mock_get = Mock(
        return_value=MockResponse({"nextExpectedRanges": [f"{chunk_size}-"]})
    )
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(
        monkeypatch,
        _FakeSession(put=mock_put, get=mock_get, delete=mock_delete),
    )

    result = onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    assert result["id"] == "item-1"
    assert mock_put.call_count == 2
    assert (
        mock_put.call_args_list[0]
        .kwargs["headers"]["Content-Range"]
        .startswith("bytes 0-")
    )
    assert (
        mock_put.call_args_list[1]
        .kwargs["headers"]["Content-Range"]
        .startswith(f"bytes {chunk_size}-")
    )
    mock_get.assert_called_once_with(
        "https://upload.example/s", timeout=onedrive.DEFAULT_TIMEOUT_SECONDS
    )
    mock_delete.assert_not_called()


def test_upload_large_file_content_resumes_inside_ambiguous_fragment(monkeypatch):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    # Graph explicitly says nextExpectedRanges need not use the client's
    # fragment boundaries. Choose an unaligned offset so this test catches a
    # retry that sends only the old suffix (an invalid non-final fragment).
    partial_offset = 12_345
    total_size = 2 * chunk_size + 10
    content = b"a" * partial_offset + b"b" * (total_size - partial_offset)
    fh = io.BytesIO(content)
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(
        side_effect=[
            requests.ConnectionError("response lost"),
            MockResponse({}, status_code=202),
            MockResponse({"id": "item-1"}, status_code=201),
        ]
    )
    mock_get = Mock(
        return_value=MockResponse({"nextExpectedRanges": [f"{partial_offset}-"]})
    )
    _patch_session(monkeypatch, _FakeSession(put=mock_put, get=mock_get))

    result = onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    assert result == {"id": "item-1"}
    resumed = mock_put.call_args_list[1].kwargs
    assert resumed["headers"]["Content-Range"] == (
        f"bytes {partial_offset}-{partial_offset + chunk_size - 1}/{total_size}"
    )
    assert resumed["headers"]["Content-Length"] == str(chunk_size)
    assert (
        bytes(resumed["data"]) == content[partial_offset : partial_offset + chunk_size]
    )
    final = mock_put.call_args_list[2].kwargs
    assert final["headers"]["Content-Range"] == (
        f"bytes {partial_offset + chunk_size}-{total_size - 1}/{total_size}"
    )


def test_upload_large_file_content_retries_transient_reconciliation_get(monkeypatch):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(
        side_effect=[
            requests.ConnectionError("response lost"),
            MockResponse({}, status_code=202),
            MockResponse({"id": "item-1"}, status_code=201),
        ]
    )
    mock_get = Mock(
        side_effect=[
            MockResponse({}, status_code=503),
            MockResponse({"nextExpectedRanges": ["0-"]}),
        ]
    )
    _patch_session(monkeypatch, _FakeSession(put=mock_put, get=mock_get))

    result = onedrive._upload_large_file_content(
        "big.bin",
        io.BytesIO(b"\x00" * total_size),
        total_size,
        "application/octet-stream",
    )

    assert result == {"id": "item-1"}
    assert mock_get.call_count == 2
    assert mock_put.call_count == 3


def test_upload_large_file_content_retries_put_after_reconcile_exhaustion(monkeypatch):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(
        side_effect=[
            requests.ConnectionError("response lost"),
            MockResponse({}, status_code=202),
            MockResponse({"id": "item-1"}, status_code=201),
        ]
    )
    mock_get = Mock(
        side_effect=[
            MockResponse({}, status_code=503)
            for _ in range(onedrive._UPLOAD_CHUNK_MAX_ATTEMPTS)
        ]
    )
    _patch_session(monkeypatch, _FakeSession(put=mock_put, get=mock_get))

    result = onedrive._upload_large_file_content(
        "big.bin",
        io.BytesIO(b"\x00" * total_size),
        total_size,
        "application/octet-stream",
    )

    assert result == {"id": "item-1"}
    assert mock_put.call_count == 3
    assert mock_get.call_count == onedrive._UPLOAD_CHUNK_MAX_ATTEMPTS


def test_upload_large_file_content_resets_failures_after_forward_progress(monkeypatch):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = 3 * chunk_size
    partial_offsets = [1 * 1024 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024]
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(
        side_effect=[requests.ConnectionError("response lost") for _ in partial_offsets]
        + [
            MockResponse({}, status_code=202),
            MockResponse({}, status_code=202),
            MockResponse({"id": "item-1"}, status_code=201),
        ]
    )
    mock_get = Mock(
        side_effect=[
            MockResponse({"nextExpectedRanges": [f"{offset}-"]})
            for offset in partial_offsets
        ]
    )
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(
        monkeypatch,
        _FakeSession(put=mock_put, get=mock_get, delete=mock_delete),
    )

    result = onedrive._upload_large_file_content(
        "big.bin",
        io.BytesIO(b"\x00" * total_size),
        total_size,
        "application/octet-stream",
    )

    assert result == {"id": "item-1"}
    assert mock_get.call_count == len(partial_offsets)
    assert mock_put.call_count == 6
    mock_delete.assert_not_called()


def test_upload_large_file_content_confirms_lost_final_response(monkeypatch):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    fh = io.BytesIO(b"\x00" * total_size)
    expected_hash = _quickxor_hash(b"\x00" * total_size)
    graph_request = Mock(
        side_effect=[
            MockResponse({"uploadUrl": "https://upload.example/s"}),
            MockResponse(
                {
                    "id": "item-1",
                    "name": "big.bin",
                    "size": total_size,
                    "eTag": "new-version",
                    "file": {"hashes": {"quickXorHash": expected_hash}},
                }
            ),
        ]
    )
    monkeypatch.setattr(onedrive.requests, "request", graph_request)
    mock_put = Mock(
        side_effect=[
            MockResponse({}, status_code=202, content=b""),
            requests.ConnectionError("final response lost"),
        ]
    )
    mock_get = Mock(return_value=MockResponse({}, status_code=404))
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(
        monkeypatch,
        _FakeSession(put=mock_put, get=mock_get, delete=mock_delete),
    )

    result = onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    assert result == {
        "id": "item-1",
        "name": "big.bin",
        "size": total_size,
        "eTag": "new-version",
        "file": {"hashes": {"quickXorHash": expected_hash}},
    }
    assert graph_request.call_count == 2
    mock_delete.assert_not_called()


def test_upload_large_file_content_treats_empty_ranges_as_final_completion(
    monkeypatch,
):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    content = b"\x00" * total_size
    expected_hash = _quickxor_hash(content)
    graph_request = Mock(
        side_effect=[
            MockResponse({"uploadUrl": "https://upload.example/s"}),
            MockResponse(
                {
                    "id": "item-1",
                    "size": total_size,
                    "file": {"hashes": {"quickXorHash": expected_hash}},
                }
            ),
        ]
    )
    monkeypatch.setattr(onedrive.requests, "request", graph_request)
    mock_get = Mock(return_value=MockResponse({"nextExpectedRanges": []}))
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(
                side_effect=[
                    MockResponse({}, status_code=202),
                    requests.ConnectionError("final response lost"),
                ]
            ),
            get=mock_get,
        ),
    )

    result = onedrive._upload_large_file_content(
        "big.bin", io.BytesIO(content), total_size, "application/octet-stream"
    )

    assert result["id"] == "item-1"
    mock_get.assert_called_once()


def test_upload_large_file_content_accepts_completed_item_from_status_get(monkeypatch):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    content = b"\x00" * total_size
    expected_hash = _quickxor_hash(content)
    completed_item = {
        "id": "item-1",
        "size": total_size,
        "file": {"hashes": {"quickXorHash": expected_hash}},
    }
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse({"uploadUrl": "https://upload.example/s"}),
                MockResponse(completed_item),
            ]
        ),
    )
    mock_get = Mock(return_value=MockResponse(completed_item, status_code=200))
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(
                side_effect=[
                    MockResponse({}, status_code=202),
                    requests.ConnectionError("final response lost"),
                ]
            ),
            get=mock_get,
        ),
    )

    result = onedrive._upload_large_file_content(
        "big.bin", io.BytesIO(content), total_size, "application/octet-stream"
    )

    assert result == completed_item
    mock_get.assert_called_once()


@pytest.mark.parametrize("fragment_status", [404, 409])
def test_upload_large_file_content_reconciles_final_fragment_status(
    monkeypatch, fragment_status
):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    content = b"\x00" * total_size
    expected_hash = _quickxor_hash(content)
    completed_item = {
        "id": "item-1",
        "size": total_size,
        "file": {"hashes": {"quickXorHash": expected_hash}},
    }
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse({"uploadUrl": "https://upload.example/s"}),
                MockResponse(completed_item),
            ]
        ),
    )
    mock_get = Mock(return_value=MockResponse({}, status_code=404))
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(
                side_effect=[
                    MockResponse({}, status_code=202),
                    MockResponse({}, status_code=fragment_status),
                ]
            ),
            get=mock_get,
        ),
    )

    result = onedrive._upload_large_file_content(
        "big.bin", io.BytesIO(content), total_size, "application/octet-stream"
    )

    assert result == completed_item
    mock_get.assert_called_once()


def test_upload_large_file_content_rejects_same_size_concurrent_item(
    monkeypatch,
):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    fh = io.BytesIO(b"\x00" * total_size)
    concurrent_item = {
        "id": "concurrent-item",
        "name": "big.bin",
        "size": total_size,
        "file": {"hashes": {"quickXorHash": _quickxor_hash(b"x" * total_size)}},
    }
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse({"uploadUrl": "https://upload.example/s"}),
                *[
                    MockResponse(concurrent_item)
                    for _ in range(onedrive._UPLOAD_CHUNK_MAX_ATTEMPTS)
                ],
            ]
        ),
    )
    mock_delete = Mock(return_value=MockResponse({}, status_code=404))
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(
                side_effect=[
                    MockResponse({}, status_code=202, content=b""),
                    requests.ConnectionError("final response lost"),
                ]
            ),
            get=Mock(return_value=MockResponse({}, status_code=404)),
            delete=mock_delete,
        ),
    )

    with pytest.raises(RuntimeError, match="could not be confirmed"):
        onedrive._upload_large_file_content(
            "big.bin", fh, total_size, "application/octet-stream"
        )

    mock_delete.assert_called_once()


def test_completed_upload_item_handles_null_file_facet(monkeypatch):
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"id": "item-1", "size": 123, "file": None})),
    )

    with pytest.raises(onedrive._UploadError, match="could not be confirmed"):
        onedrive._completed_upload_item("big.bin", 123, "expected-hash")


def test_upload_large_file_content_honors_bounded_retry_after(monkeypatch):
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    throttled = MockResponse({}, status_code=429, headers={"Retry-After": "999"})
    intermediate = MockResponse({}, status_code=202, content=b"")
    completed = MockResponse({"id": "item-1"})
    _patch_session(
        monkeypatch,
        _FakeSession(put=Mock(side_effect=[throttled, intermediate, completed])),
    )

    onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    onedrive.time.sleep.assert_called_once_with(onedrive._UPLOAD_RETRY_MAX_SECONDS)


def test_upload_retry_delay_accepts_http_date(monkeypatch):
    monkeypatch.setattr(onedrive.time, "time", Mock(return_value=0.0))

    assert onedrive._retry_delay(
        {"Retry-After": "Thu, 01 Jan 1970 00:00:07 GMT"}, 1
    ) == pytest.approx(7.0)


@pytest.mark.parametrize("invalid_delay", ["nan", "inf", "-inf"])
def test_upload_retry_delay_rejects_non_finite_seconds(invalid_delay):
    assert onedrive._retry_delay({"Retry-After": invalid_delay}, 2) == pytest.approx(
        2.0
    )


def test_upload_retry_delay_treats_naive_http_date_as_utc(monkeypatch):
    monkeypatch.setattr(onedrive.time, "time", Mock(return_value=0.0))

    assert onedrive._retry_delay(
        {"Retry-After": "Thu, 01 Jan 1970 00:00:07"}, 1
    ) == pytest.approx(7.0)


def test_upload_large_file_content_still_cancels_on_a_non_retriable_failure(
    monkeypatch,
):
    """Complementary case: a genuine client-side rejection (not a 429/5xx,
    not a network failure) still gets the session cancelled immediately,
    exactly like before -- nothing about a retry against the same session
    would fix a 400, so there's no reason to withhold the cleanup."""
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    error_response = MockResponse({}, status_code=400)
    mock_delete = Mock(return_value=MockResponse({}, status_code=204))
    _patch_session(
        monkeypatch,
        _FakeSession(put=Mock(return_value=error_response), delete=mock_delete),
    )

    with pytest.raises(RuntimeError):
        onedrive._upload_large_file_content(
            "big.bin", fh, total_size, "application/octet-stream"
        )

    mock_delete.assert_called_once()


def test_upload_large_file_content_fails_on_a_non_first_chunk(monkeypatch):
    """Regression guard: earlier test coverage only ever failed the first
    chunk of a session — verifying a mid-sequence failure also propagates
    (and still triggers cleanup) catches a bug that only manifests once the
    loop has state to lose (e.g. an exception handler scoped to the first
    iteration only)."""
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = 3 * chunk_size
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    error_response = MockResponse(
        {"error": {"message": "range conflict"}},
        status_code=416,
        content=b'{"error": {"message": "range conflict"}}',
    )
    mock_put = Mock(side_effect=[MockResponse({}, content=b"")] + [error_response] * 3)
    mock_get = Mock(
        return_value=MockResponse({"nextExpectedRanges": [f"{chunk_size}-"]})
    )
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(
        monkeypatch,
        _FakeSession(put=mock_put, get=mock_get, delete=mock_delete),
    )

    with pytest.raises(RuntimeError, match="made no forward progress"):
        onedrive._upload_large_file_content(
            "big.bin", fh, total_size, "application/octet-stream"
        )

    assert mock_put.call_count == 4
    mock_delete.assert_called_once()


def test_upload_large_file_content_recovers_when_final_response_is_202(
    monkeypatch,
):
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)
    expected_hash = _quickxor_hash(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse({"uploadUrl": "https://upload.example/s"}),
                MockResponse(
                    {
                        "id": "item-1",
                        "size": total_size,
                        "file": {"hashes": {"quickXorHash": expected_hash}},
                    }
                ),
            ]
        ),
    )
    mock_put = Mock(
        side_effect=[
            MockResponse({}, content=b""),
            MockResponse(
                {"expirationDateTime": "2099-01-01T00:00:00Z"}, status_code=202
            ),
        ]
    )
    mock_get = Mock(return_value=MockResponse({}, status_code=404))
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(
        monkeypatch,
        _FakeSession(put=mock_put, get=mock_get, delete=mock_delete),
    )

    result = onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    assert result["id"] == "item-1"
    mock_get.assert_called_once()
    mock_delete.assert_not_called()


@pytest.mark.parametrize("final_content", [b"", b"not valid json"])
def test_upload_large_file_content_recovers_from_missing_or_unparsable_final_response(
    monkeypatch, final_content
):
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)
    expected_hash = _quickxor_hash(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(
            side_effect=[
                MockResponse({"uploadUrl": "https://upload.example/s"}),
                MockResponse(
                    {
                        "id": "item-1",
                        "size": total_size,
                        "file": {"hashes": {"quickXorHash": expected_hash}},
                    }
                ),
            ]
        ),
    )
    final_response = MockResponse({}, status_code=201, content=final_content)
    if final_content:
        final_response.json = Mock(side_effect=ValueError("Expecting value"))
    mock_put = Mock(side_effect=[MockResponse({}, content=b""), final_response])
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(monkeypatch, _FakeSession(put=mock_put, delete=mock_delete))

    result = onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    assert result["id"] == "item-1"
    mock_delete.assert_not_called()


def test_upload_large_file_content_retries_nested_metadata_failure(monkeypatch):
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    content = b"\x00" * total_size
    fh = io.BytesIO(content)
    expected_hash = _quickxor_hash(content)
    transient = MockResponse({}, status_code=503)
    graph_request = Mock(
        side_effect=[
            MockResponse({"uploadUrl": "https://upload.example/s"}),
            transient,
            MockResponse(
                {
                    "id": "item-1",
                    "size": total_size,
                    "file": {"hashes": {"quickXorHash": expected_hash}},
                }
            ),
        ]
    )
    monkeypatch.setattr(onedrive.requests, "request", graph_request)
    malformed_final = MockResponse({}, status_code=201, content=b"not valid json")
    malformed_final.json = Mock(side_effect=ValueError("Expecting value"))
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(side_effect=[MockResponse({}, status_code=202), malformed_final])
        ),
    )

    result = onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    assert result["id"] == "item-1"
    assert graph_request.call_count == 3
    onedrive.time.sleep.assert_called_once()


def test_reconcile_exhaustion_preserves_retriable_cause(monkeypatch):
    response = MockResponse({}, status_code=503)
    http = _FakeSession(get=Mock(return_value=response))

    with pytest.raises(onedrive._UploadError) as exc_info:
        onedrive._reconcile_upload_progress(
            http,
            "https://upload.example/s",
            "big.bin",
            0,
            onedrive._UPLOAD_SESSION_CHUNK_SIZE,
            onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10,
            "unused-hash",
        )

    assert onedrive._is_retriable_upload_error(exc_info.value)


def test_reconcile_rejects_progress_beyond_submitted_fragment():
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    http = _FakeSession(
        get=Mock(
            return_value=MockResponse({"nextExpectedRanges": [f"{chunk_size + 1}-"]})
        )
    )

    with pytest.raises(
        onedrive._UploadError, match="progress beyond the submitted fragment"
    ):
        onedrive._reconcile_upload_progress(
            http,
            "https://upload.example/s",
            "big.bin",
            0,
            chunk_size,
            2 * chunk_size,
            "unused-hash",
        )


def test_upload_large_file_content_retries_unaccepted_final_202(monkeypatch):
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 10
    fh = io.BytesIO(b"\x00" * total_size)
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(
        side_effect=[
            MockResponse({}, status_code=202),
            MockResponse({}, status_code=202, headers={"Retry-After": "999"}),
            MockResponse({"id": "item-1"}, status_code=201),
        ]
    )
    mock_get = Mock(
        return_value=MockResponse({"nextExpectedRanges": [f"{chunk_size}-"]})
    )
    _patch_session(monkeypatch, _FakeSession(put=mock_put, get=mock_get))

    result = onedrive._upload_large_file_content(
        "big.bin", fh, total_size, "application/octet-stream"
    )

    assert result == {"id": "item-1"}
    assert mock_put.call_count == 3
    mock_get.assert_called_once()
    onedrive.time.sleep.assert_called_once_with(onedrive._UPLOAD_RETRY_MAX_SECONDS)


def test_upload_large_file_content_rejects_short_chunk_read(monkeypatch):
    """Regression guard: if the local file shrinks mid-upload (a concurrent
    rewrite/truncation), a short fh.read() must fail loudly instead of
    silently sending a Content-Length/Content-Range that doesn't match the
    actual bytes transmitted -- which would desynchronize every later
    chunk's byte-range accounting."""
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    total_size = chunk_size + 100

    class _ShortReadBuffer:
        def __init__(self, data):
            self._data = data
            self._pos = 0

        def read(self, size=-1):
            # Always return at most half of what was asked for (but never
            # zero, so this isn't just simulating ordinary EOF).
            actual = max(1, size // 2) if size and size > 1 else size
            chunk = self._data[self._pos : self._pos + actual]
            self._pos += len(chunk)
            return chunk

    fh = _ShortReadBuffer(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(return_value=MockResponse({}, content=b""))
    _patch_session(monkeypatch, _FakeSession(put=mock_put))

    with pytest.raises(RuntimeError, match="changed size during upload"):
        onedrive._upload_large_file_content(
            "big.bin", fh, total_size, "application/octet-stream"
        )

    # Must fail before ever sending the short chunk to Graph.
    mock_put.assert_not_called()


def test_upload_large_file_content_rejects_growth_before_final_chunk(monkeypatch):
    """The final over-read catches bytes appended after the size snapshot."""
    chunk_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE
    snapshotted_size = chunk_size + 100
    fh = io.BytesIO(b"\x00" * (snapshotted_size + 1))
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(return_value=MockResponse({}, status_code=202, content=b""))
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(monkeypatch, _FakeSession(put=mock_put, delete=mock_delete))

    with pytest.raises(RuntimeError, match="changed size during upload"):
        onedrive._upload_large_file_content(
            "big.bin", fh, snapshotted_size, "application/octet-stream"
        )

    assert mock_put.call_count == 1
    mock_delete.assert_called_once()


def test_simple_upload_max_bytes_is_at_or_below_graphs_4mb_limit():
    """Regression guard for the actual production boundary bug: Microsoft's
    own docs disagree on the simple content PUT's real limit (the OneDrive
    API concepts page says "4 MB", the Graph v1.0 API reference for the
    same endpoint says "250 MB" -- see _SIMPLE_UPLOAD_MAX_BYTES's own
    comment), and some deployments enforce the smaller figure as the
    decimal 4,000,000 bytes rather than the binary 4 MiB (4,194,304 bytes).
    The cutoff must stay at or below the smaller, decimal figure so a file
    anywhere in the ambiguous gap always takes the resumable upload-session
    path instead of ever risking rejection at the simple-PUT boundary."""
    assert onedrive._SIMPLE_UPLOAD_MAX_BYTES <= 4_000_000


def test_upload_file_at_exact_boundary_uses_simple_put(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: a purely static assertion on the constant (see
    above) can't catch a routing-condition regression like `<=` silently
    flipped to `<` -- this exercises onedrive_upload_file itself with a
    file sized exactly at the boundary and confirms it takes the simple-PUT
    branch, not the upload-session one."""
    local_file = _upload_allowed_dirs_env / "at_boundary.bin"
    local_file.write_bytes(b"\x00" * onedrive._SIMPLE_UPLOAD_MAX_BYTES)

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)
    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["method"] == "PUT"


def test_upload_file_bounds_the_actual_read_before_upload(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "growing.bin"
    local_file.write_bytes(b"initially small")
    real_file = local_file.open("rb")

    class GrowingFile:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            real_file.close()
            return False

        def fileno(self):
            return real_file.fileno()

        def read(self, size):
            with open(local_file, "ab") as append_fh:
                append_fh.write(b"x" * (onedrive._SIMPLE_UPLOAD_MAX_BYTES + 1))
            return real_file.read(size)

    monkeypatch.setattr(onedrive.Path, "open", lambda self, mode: GrowingFile())
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "error"
    assert "grew during upload" in result["message"]
    assert local_file.stat().st_size > onedrive._SIMPLE_UPLOAD_MAX_BYTES
    mock_request.assert_not_called()


def test_upload_file_one_byte_over_boundary_uses_upload_session(
    monkeypatch, _upload_allowed_dirs_env
):
    """The complementary case to the exact-boundary test above: one byte
    over the cutoff must take the resumable upload-session path."""
    local_file = _upload_allowed_dirs_env / "over_boundary.bin"
    local_file.write_bytes(b"\x00" * (onedrive._SIMPLE_UPLOAD_MAX_BYTES + 1))

    mock_request = Mock(
        return_value=MockResponse({"uploadUrl": "https://upload.example/s"})
    )
    monkeypatch.setattr(onedrive.requests, "request", mock_request)
    mock_put = Mock(return_value=MockResponse({"id": "item-1"}))
    _patch_session(monkeypatch, _FakeSession(put=mock_put))

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("createUploadSession")
    mock_put.assert_called_once()


@pytest.mark.parametrize(
    "file_path",
    [
        # Pre-existing binary-format coverage.
        "font.woff2",
        "font.woff",
        "font.ttf",
        "data.parquet",
        "cache.sqlite",
        "cache.sqlite3",
        "budget.numbers",
        "notes.pages",
        "vault.key",
        "extension.crx",
        "app.pub",
        "installer.dmg",
        "image.iso",
        "app.apk",
        "book.mobi",
        "app.jar",
        "Main.class",
        "installer.cab",
        "package.deb",
        "file.torrent",
        "keystore.p12",
        "cert.pfx",
        "movie.swf",
        # ML/data-science/VM-image/keystore formats a positive
        # known-binary-formats list kept missing across review rounds --
        # the specific gap that motivated switching to a default-deny,
        # known-*text*-formats allowlist instead (see
        # _KNOWN_TEXT_EXTENSIONS's own docstring). None of these need to be
        # individually enumerated anywhere for this test to pass -- that's
        # the point of the redesign: anything not affirmatively listed as
        # text is rejected, with no separate blocklist to keep patching.
        "model.safetensors",
        "weights.pkl",
        "array.npy",
        "array.npz",
        "model.onnx",
        "graph.pb",
        "module.pyc",
        "module.pyd",
        "pkg.whl",
        "pkg.egg",
        "blob.dat",
        "archive.xz",
        "archive.zst",
        "archive.lz4",
        "data.avro",
        "db.accdb",
        "df.feather",
        "table.arrow",
        "logs.orc",
        "disk.img",
        "disk.vmdk",
        "disk.qcow2",
        "store.jks",
        "model.sav",  # codespell:ignore sav
        "mesh.stl",
        "weights.h5",
        "weights.hdf5",
        "settings.plist",
        # Deliberately excluded from _KNOWN_TEXT_EXTENSIONS despite often
        # being PEM/text in practice -- see that set's own docstring on
        # why ".crt" specifically isn't given the same "text" judgment
        # call as ".bat"/".ts"/".scm"/".sc".
        "host.crt",
    ],
)
def test_upload_text_file_rejects_binary_extensions(monkeypatch, file_path):
    """Regression guard for the default-deny classifier: any extension not
    affirmatively listed in _KNOWN_TEXT_EXTENSIONS is rejected, regardless
    of whether it's a format anyone has thought to test mimetypes against.
    """
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "error"
    assert "onedrive_upload_file" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize(
    "file_path",
    [
        "chart.svg",
        "app.ts",
        "deploy.bat",
        "session.scm",
        "worksheet.sc",
        "deploy.ps1",
        "schema.sql",
        "index.php",
        "main.dart",
        "paper.tex",
        "build.csh",
        "deploy.sh",
        "subs.srt",
        "schema.dtd",
        "layout.tpl",
    ],
)
def test_upload_text_file_allows_known_text_extensions_regardless_of_host_mimetypes(
    monkeypatch, file_path
):
    """Regression guard for the default-deny classifier's whole point:
    unlike the mimetype-driven designs this module cycled through in
    earlier rounds, _name_looks_binary never consults mimetypes.guess_type
    at all, so it can't be affected by whatever a given host's mime.types
    database happens to say. Proven directly here by stubbing
    mimetypes.guess_type to a value that would misclassify every one of
    these names if it were still consulted (a real binary-looking type
    with no text-safe signal at all) -- if the guard incorrectly fell back
    to mimetypes for any of these, this would catch it.

    ".ts"/".bat"/".scm"/".sc" are judgment calls (see
    _KNOWN_TEXT_EXTENSIONS's own docstring: each also names a real,
    unrelated binary format, but an agent-generated file with one of these
    names is overwhelmingly more likely to be genuine source text)."""
    monkeypatch.setattr(
        onedrive.mimetypes,
        "guess_type",
        lambda name: ("application/octet-stream", None),
    )
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "success"


def test_upload_text_file_allows_extensionless_names(monkeypatch):
    """Regression guard: a name with no extension at all (e.g. "Dockerfile",
    "README") isn't itself evidence of binary intent the way an
    unrecognized extension is -- _name_looks_binary's default-deny design
    only rejects a *recognized-as-suspicious* extension, not the absence of
    one."""
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    for name in ["Dockerfile", "README", "LICENSE"]:
        result = json.loads(onedrive.onedrive_upload_text_file(name, "some text"))
        assert result["status"] == "success", name


def test_upload_text_file_rejects_dotfile_shaped_binary_extension(monkeypatch):
    """Regression guard: a name that's *entirely* a leading dot plus
    extension (e.g. ".pdf") has an empty Path(...).suffix per pathlib's
    Unix-dotfile convention -- _split_stem_suffix exists specifically so
    this doesn't fall through _name_looks_binary as "extensionless" (and
    therefore accepted) the same way "Dockerfile" correctly does above."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(".pdf", "some text"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_upload_text_file_rejects_nested_dotfile_shaped_binary_extension(
    monkeypatch,
):
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_text_file("Documents/.pdf", "some text")
    )

    assert result["status"] == "error"
    assert "onedrive_upload_file" in result["message"]
    mock_request.assert_not_called()


def test_upload_file_resolves_real_mime_type_for_ambiguous_extensions(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: _AMBIGUOUS_TEXT_EXTENSIONS forces ".ts"/".bat"/
    ".scm"/".sc"/".ps1" to be treated as non-binary by the
    onedrive_upload_text_file guard, but onedrive_upload_file uploads a
    real local file's real bytes -- a genuine ".ts" file is very commonly
    an actual MPEG transport-stream video chunk, not TypeScript source.
    An earlier version of this fix applied the override inside
    _guess_mime_type itself, which also corrupted this tool's real
    Content-Type resolution (sending "text/plain" for what Graph is told
    is a ".ts" file) whenever no explicit mime_type is passed.

    What mimetypes.guess_type() itself resolves ".ts" to is host-dependent
    -- confirmed directly: this file's own dev host resolves it to
    "video/mp2t", while a CI run on a different OS/Python resolved it to
    "text/vnd.trolltech.linguist" (Qt Linguist translation source, yet
    another real format that happens to share this extension) instead.
    Monkeypatching mimetypes.guess_type to a fixed value makes this test
    deterministic instead of depending on whichever mime.types database
    happens to be installed on whatever machine runs it -- what's actually
    under test is that _guess_mime_type's real answer reaches Content-Type
    unmodified, not what that real answer happens to be for ".ts"
    specifically."""
    monkeypatch.setattr(
        onedrive.mimetypes, "guess_type", lambda name: ("video/mp2t", None)
    )
    local_file = _upload_allowed_dirs_env / "segment001.ts"
    local_file.write_bytes(b"\x47" * 100)  # MPEG-TS sync byte, not text

    mock_request = Mock(return_value=MockResponse({"id": "f1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    sent_headers = mock_request.call_args.kwargs["headers"]
    assert sent_headers["Content-Type"] == "video/mp2t"


def test_upload_file_raises_when_upload_session_has_no_url(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "big.bin"
    local_file.write_bytes(b"\x01" * (onedrive._UPLOAD_SESSION_CHUNK_SIZE + 1))

    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({}))
    )

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "error"
    assert "upload session" in result["message"]


def test_upload_file_sanitizes_upload_session_creation_failure(
    monkeypatch, _upload_allowed_dirs_env, caplog
):
    local_file = _upload_allowed_dirs_env / "big.bin"
    local_file.write_bytes(b"\x01" * (onedrive._UPLOAD_SESSION_CHUNK_SIZE + 1))
    secret = "provider-internal-secret"
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {"error": {"message": secret}},
                status_code=500,
                content=f'{{"error":{{"message":"{secret}"}}}}'.encode(),
            )
        ),
    )

    with caplog.at_level("ERROR"):
        result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result == {
        "status": "error",
        "message": "Could not create OneDrive upload session",
    }
    assert secret not in caplog.text


def test_upload_file_rejects_path_outside_allowed_directories(
    monkeypatch, tmp_path, _upload_allowed_dirs_env
):
    outside_file = tmp_path / "outside.pdf"
    outside_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(outside_file)))

    assert result["status"] == "error"
    assert "allowed upload directories" in result["message"]
    # The absolute host path must never leak into the message the LLM sees.
    assert str(outside_file) not in result["message"]
    mock_request.assert_not_called()


def test_upload_file_rejects_sibling_directory_name_collision(
    monkeypatch, tmp_path, _upload_allowed_dirs_env
):
    """Regression guard: an allowed dir like "/workspace" must not
    accidentally admit a sibling "/workspace-other" just because it starts
    with the same string -- containment must be a real path-relative check
    (is_relative_to), not a naive string prefix comparison. Passing today,
    but pinned down directly so a future refactor to str.startswith()
    can't silently regress it while every other test stays green."""
    sibling_dir = tmp_path / (_upload_allowed_dirs_env.name + "-other")
    sibling_dir.mkdir()
    outside_file = sibling_dir / "secret.pdf"
    outside_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(outside_file)))

    assert result["status"] == "error"
    assert "allowed upload directories" in result["message"]
    mock_request.assert_not_called()


def test_upload_file_rejects_symlink_escaping_allowed_dir(
    monkeypatch, tmp_path, _upload_allowed_dirs_env
):
    """A symlink physically inside the allowed directory but pointing
    outside it must not grant access to its target -- resolve() follows
    the symlink to its real location before the containment check runs."""
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("secret")
    link = _upload_allowed_dirs_env / "escape_link.txt"
    link.symlink_to(secret_file)

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(link)))

    assert result["status"] == "error"
    assert "allowed upload directories" in result["message"]
    mock_request.assert_not_called()


def test_upload_file_allows_symlink_whose_target_is_inside_allowed_dir(
    monkeypatch, _upload_allowed_dirs_env
):
    """Complementary case to the escaping-symlink test above: a symlink
    that resolves to a target still *inside* the allowed directory must be
    accepted, not rejected outright -- guards against a future
    over-tightening (e.g. "reject any symlink at all") that would pass
    every existing test here while breaking a legitimate same-workspace
    symlink an agent's own tooling might create."""
    real_file = _upload_allowed_dirs_env / "real.pdf"
    real_file.write_bytes(b"content")
    link = _upload_allowed_dirs_env / "alias.pdf"
    link.symlink_to(real_file)

    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_file(str(link)))

    assert result["status"] == "success"


def test_upload_file_rejects_relative_traversal_outside_allowed_dir(
    monkeypatch, tmp_path, _upload_allowed_dirs_env
):
    (tmp_path / "secret.txt").write_text("secret")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(_upload_allowed_dirs_env / ".." / "secret.txt")
        )
    )

    assert result["status"] == "error"
    assert "allowed upload directories" in result["message"]
    mock_request.assert_not_called()


def test_allowed_upload_dirs_falls_back_to_cwd_when_unset(monkeypatch):
    """onedrive.py now delegates to the shared allowed_dirs_from_env helper
    (mcp/utils.py) rather than keeping its own copy of this parsing
    logic -- see that helper's own docstring for why the copies drifted
    and produced a real bug before consolidation."""
    monkeypatch.delenv("XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS", raising=False)

    assert onedrive.allowed_dirs_from_env(onedrive._UPLOAD_ALLOWED_DIRS_ENV_VAR) == [
        onedrive.Path.cwd().resolve()
    ]


def test_allowed_upload_dirs_falls_back_to_cwd_when_malformed(monkeypatch):
    """Regression guard: a malformed-but-nonblank env value (e.g. a lone
    "," or " , ") must fall back to CWD the same way a fully-unset/blank
    value does, not silently resolve to an empty allowlist that rejects
    every upload with no diagnostic pointing at the env var as the cause."""
    monkeypatch.setenv("XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS", " , ")

    assert onedrive.allowed_dirs_from_env(onedrive._UPLOAD_ALLOWED_DIRS_ENV_VAR) == [
        onedrive.Path.cwd().resolve()
    ]


def test_upload_file_honors_explicit_empty_allowed_dirs(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")
    monkeypatch.setenv("XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS", "[]")
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "error"
    assert "allowed upload directories" in result["message"]
    mock_request.assert_not_called()


def test_upload_file_scrubs_malformed_allowed_dirs_configuration(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")
    monkeypatch.setenv("XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS", "[")
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result == {
        "status": "error",
        "message": "Upload directory configuration is invalid",
    }
    assert "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS" not in result["message"]
    mock_request.assert_not_called()


def test_upload_file_rejects_missing_file(monkeypatch, _upload_allowed_dirs_env):
    missing_path = _upload_allowed_dirs_env / "does_not_exist.pdf"

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(missing_path)))

    assert result["status"] == "error"
    assert "not found" in result["message"].lower()
    mock_request.assert_not_called()


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges")
def test_upload_file_does_not_leak_path_from_symlink_loop(
    monkeypatch, _upload_allowed_dirs_env
):
    loop = _upload_allowed_dirs_env / "loop.pdf"
    loop.symlink_to(loop)
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(loop)))

    assert result["status"] == "error"
    assert result["message"] == "Could not resolve local_file_path"
    assert str(loop) not in result["message"]
    mock_request.assert_not_called()


def test_upload_file_rejects_directory_with_a_distinct_message(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: a directory (or other non-regular file) used to
    raise the same "File not found" as a genuinely missing path, which
    reads as "retry, it'll show up" even though a directory never will."""
    a_directory = _upload_allowed_dirs_env / "not_a_file"
    a_directory.mkdir()

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(a_directory)))

    assert result["status"] == "error"
    assert "not a regular file" in result["message"].lower()
    mock_request.assert_not_called()


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="permission bits aren't meaningful on Windows or when running as root",
)
def test_upload_file_does_not_leak_absolute_path_on_permission_error(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: local_path.open("rb") previously had no OSError
    guard, so a permission-denied file (or a TOCTOU race between the
    allowlist check and the open) fell through to the generic
    `except Exception` and returned str(OSError) -- which embeds the
    absolute resolved host path (e.g. "[Errno 13] Permission denied:
    '/full/host/path'") -- straight to the caller/LLM, the same detail
    _resolve_upload_file_path's own error deliberately scrubs."""
    unreadable_file = _upload_allowed_dirs_env / "secret.pdf"
    unreadable_file.write_bytes(b"content")
    unreadable_file.chmod(0o000)
    try:
        mock_request = Mock()
        monkeypatch.setattr(onedrive.requests, "request", mock_request)

        result = json.loads(onedrive.onedrive_upload_file(str(unreadable_file)))

        assert result["status"] == "error"
        assert str(unreadable_file) not in result["message"]
        assert "secret.pdf" not in result["message"]
        mock_request.assert_not_called()
    finally:
        unreadable_file.chmod(0o644)


def test_upload_file_rejects_empty_file(monkeypatch, _upload_allowed_dirs_env):
    empty_file = _upload_allowed_dirs_env / "empty.pdf"
    empty_file.write_bytes(b"")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(empty_file)))

    assert result["status"] == "error"
    assert "empty" in result["message"].lower()
    mock_request.assert_not_called()


def test_upload_file_rejects_file_over_max_upload_bytes(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: onedrive_upload_file previously had no upper size
    bound at all (only rejected 0 bytes), so a mistargeted large file (an
    unrelated log directory, the wrong generated artifact) would trigger
    an unbounded chunked upload with no early feedback."""
    local_file = _upload_allowed_dirs_env / "huge.bin"
    local_file.write_bytes(b"x")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)
    real_fstat = onedrive.os.fstat
    fake_size = onedrive._MAX_UPLOAD_BYTES + 1

    def _fake_fstat(fd):
        real = real_fstat(fd)
        return type(real)(
            (
                real.st_mode,
                real.st_ino,
                real.st_dev,
                real.st_nlink,
                real.st_uid,
                real.st_gid,
                fake_size,
                real.st_atime,
                real.st_mtime,
                real.st_ctime,
            )
        )

    monkeypatch.setattr(onedrive.os, "fstat", _fake_fstat)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "error"
    assert "limit" in result["message"].lower()
    mock_request.assert_not_called()


def test_upload_file_returns_error_payload_on_api_failure(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {"error": "boom"}, status_code=500, content=b'{"error": "boom"}'
            )
        ),
    )

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_upload_file_requires_completed_item_confirmation(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({}, status_code=204, content=b"")),
    )

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "error"
    assert "did not confirm" in result["message"]


# ---------------------------------------------------------------------------
# onedrive_upload_text_file — binary-content guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "file_path", ["report.pdf", "Documents/photo.PNG", "deck.pptx", "archive.zip"]
)
def test_upload_text_file_rejects_binary_looking_names(monkeypatch, file_path):
    """Regression guard for the actual production bug: onedrive_upload_text_file
    can only write text (content is utf-8 encoded), so a target path that
    looks like a binary format must be steered to onedrive_upload_file
    instead of silently getting a text/plain file with a misleading name."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "error"
    assert "onedrive_upload_file" in result["message"]
    mock_request.assert_not_called()


def test_upload_text_file_rejects_folder_shaped_path(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_text_file("Documents/Reports/", "text")
    )

    assert result["status"] == "error"
    assert "filename" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize(
    "file_path", ["report.pdf.", "Documents/archive.zip.", "model.pkl. "]
)
def test_upload_text_file_rejects_trailing_period_names(monkeypatch, file_path):
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "error"
    assert "period" in result["message"]
    mock_request.assert_not_called()


def test_upload_text_file_requires_completed_item_confirmation(monkeypatch):
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({}, status_code=204, content=b"")),
    )

    result = json.loads(onedrive.onedrive_upload_text_file("notes.txt", "text"))

    assert result["status"] == "error"
    assert "did not confirm" in result["message"]


@pytest.mark.parametrize(
    "file_path",
    [
        "notes.txt",
        "README.md",
        "config.json",
        "config.yaml",
        "config.yml",
        "data.csv",
        "script.py",
        "page.html",
        "server.log",
        "styles.css",
        "main.js",
        "data.xml",
        "notes",
    ],
)
def test_upload_text_file_allows_plain_text_names(monkeypatch, file_path):
    """Regression guard: the only broad "should be accepted" case used to
    be ".txt" alone, with every other positive case a narrow one-off added
    reactively after a specific extension was found broken in a prior
    round -- which is exactly how ".sh" shipped genuinely broken for a
    whole round before anyone tested it. This covers a broader set of
    everyday text formats an agent is likely to actually generate."""
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "hello world"))

    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# item_id-based tools (onedrive_get_item, onedrive_rename_item,
# onedrive_delete_item) -- dot-segment rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda item_id: onedrive.onedrive_get_item(item_id=item_id),
        lambda item_id: onedrive.onedrive_rename_item(item_id, "new-name.txt"),
        lambda item_id: onedrive.onedrive_delete_item(item_id),
    ],
    ids=["onedrive_get_item", "onedrive_rename_item", "onedrive_delete_item"],
)
@pytest.mark.parametrize("bad_id", [".", ".."])
def test_item_id_tools_reject_dot_segments(monkeypatch, call, bad_id):
    """Regression guard: each of the three item_id-based tools must refuse
    a "." or ".." item_id before ever making a request, not just when a
    Drive-relative path is used instead."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(call(bad_id))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_get_item_by_item_id_still_works_for_a_normal_id(monkeypatch):
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"id": "abc123", "name": "report.pdf"})),
    )

    result = json.loads(onedrive.onedrive_get_item(item_id="abc123"))

    assert result["status"] == "success"
    assert result["item"]["id"] == "abc123"
