import io
import json
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import onedrive


class MockResponse:
    def __init__(self, json_data=None, status_code=200, content=b"{}"):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="replace")

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


class _FakeSession:
    """Stand-in for requests.Session() used by _upload_large_file_content --
    a plain Mock doesn't support the `with ... as` context-manager protocol
    on its own, and mocking Session.put/.delete at the class level would
    leak between tests since the module only ever creates one Session."""

    def __init__(self, put=None, delete=None):
        self.put = put if put is not None else Mock(return_value=MockResponse({}))
        self.delete = (
            delete if delete is not None else Mock(return_value=MockResponse({}))
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _patch_session(monkeypatch, fake_session):
    monkeypatch.setattr(onedrive.requests, "Session", Mock(return_value=fake_session))


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")


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


def test_upload_large_file_content_enriches_chunk_error_with_response_body(
    monkeypatch,
):
    """Regression guard: a rejected chunk must surface Graph's actual error
    body, not a bare HTTPError with no detail — every other error path in
    this module (via _graph_request) already does this."""
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

    with pytest.raises(RuntimeError, match="Invalid upload session"):
        onedrive._upload_large_file_content(
            "big.bin", fh, total_size, "application/octet-stream"
        )

    # A failed upload must cancel its now-abandoned session immediately
    # rather than leaving it for Graph's own ~15-minute expiry. The
    # cancellation itself carries no body, so it uses the short default
    # timeout rather than the long one sized for a multi-megabyte PUT --
    # otherwise a stalled cleanup call would needlessly delay surfacing the
    # real failure by up to another _BINARY_UPLOAD_TIMEOUT_SECONDS.
    mock_delete.assert_called_once_with(
        "https://upload.example/s", timeout=onedrive.DEFAULT_TIMEOUT_SECONDS
    )


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
    mock_put = Mock(
        side_effect=[
            MockResponse({}, content=b""),  # chunk 1 succeeds
            error_response,  # chunk 2 fails
        ]
    )
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(monkeypatch, _FakeSession(put=mock_put, delete=mock_delete))

    with pytest.raises(RuntimeError, match="range conflict"):
        onedrive._upload_large_file_content(
            "big.bin", fh, total_size, "application/octet-stream"
        )

    assert mock_put.call_count == 2
    mock_delete.assert_called_once()


def test_upload_large_file_content_raises_when_final_response_has_no_item(
    monkeypatch,
):
    """Regression guard: the final chunk's response must actually carry a
    completed driveItem before this reports success -- an unexpected 202 (or
    any body without an "id") on what this loop computed as the last range
    must surface as an error, not a hollow {"status": "success", "item": {}}."""
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    mock_put = Mock(
        side_effect=[
            MockResponse({}, content=b""),
            MockResponse({"expirationDateTime": "2099-01-01T00:00:00Z"}),
        ]
    )
    mock_delete = Mock(return_value=MockResponse({}))
    _patch_session(monkeypatch, _FakeSession(put=mock_put, delete=mock_delete))

    with pytest.raises(RuntimeError, match="did not confirm"):
        onedrive._upload_large_file_content(
            "big.bin", fh, total_size, "application/octet-stream"
        )

    # Regression guard: this validation error must go through the same
    # cleanup path as a genuine transfer failure, not just the try/except
    # around the HTTP call itself -- an earlier version of this check ran
    # after the `with requests.Session()` block had already exited
    # normally, so it never attempted to cancel the (still-incomplete, by
    # this check's own logic) upload session.
    mock_delete.assert_called_once()


def test_upload_large_file_content_reports_unparsable_final_response_distinctly(
    monkeypatch,
):
    """Regression guard: when every chunk PUT succeeds (no HTTPError) but the
    final response body can't be parsed as JSON, the error must say the
    upload itself was accepted -- distinguishing "transfer succeeded, can't
    confirm the result" from a genuine transfer failure, since OneDrive has
    already committed the file by this point."""
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    malformed_final = MockResponse({}, content=b"not valid json")
    malformed_final.json = Mock(side_effect=ValueError("Expecting value"))
    mock_put = Mock(side_effect=[MockResponse({}, content=b""), malformed_final])
    _patch_session(monkeypatch, _FakeSession(put=mock_put))

    with pytest.raises(RuntimeError, match="accepted the final chunk"):
        onedrive._upload_large_file_content(
            "big.bin", fh, total_size, "application/octet-stream"
        )


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


def test_simple_upload_max_bytes_is_at_or_below_graphs_4mb_limit():
    """Regression guard for the actual production boundary bug: Graph's
    simple content PUT is documented as accepting files up to "4 MB", which
    some deployments enforce as the decimal 4,000,000 bytes rather than the
    binary 4 MiB (4,194,304 bytes). The cutoff must stay at or below the
    smaller, decimal figure so a file in that ambiguous gap always takes the
    resumable upload-session path instead of risking rejection right at the
    simple-PUT boundary."""
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
    session_factory = Mock()
    monkeypatch.setattr(onedrive.requests, "Session", session_factory)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["method"] == "PUT"
    session_factory.assert_not_called()


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
    ],
)
def test_upload_text_file_rejects_additional_binary_extensions(monkeypatch, file_path):
    """Regression guard: reviewer-flagged gap in the original binary-
    extension set — fonts, WASM, columnar/DB, and several common formats
    stdlib mimetypes also has no opinion on were missing from it, so with
    no other signal they used to sail straight through the guard and get
    silently created as mislabeled text files."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "error"
    assert "onedrive_upload_file" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize("file_path", ["app.ts", "deploy.bat", "session.scm"])
def test_upload_text_file_allows_ambiguous_extensions_that_collide_with_binary_mimetypes(
    monkeypatch, file_path
):
    """Regression guard for a bug introduced by switching from a fixed
    extension allowlist to mimetype-driven detection: mimetypes.guess_type
    resolves ".ts" to "video/mp2t", ".bat" to "application/x-msdownload",
    and ".scm" to "application/vnd.lotus-screencam" -- none of which are
    text-safe -- even though all three extensions are overwhelmingly used
    for genuine text/source content (TypeScript, Windows batch scripts,
    Scheme source) in practice. Without an explicit carve-out, onedrive_
    upload_text_file would reject these with no working alternative (
    onedrive_upload_file needs an existing local file, not raw text)."""
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "success"


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
    monkeypatch.delenv("XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS", raising=False)

    assert onedrive._allowed_upload_dirs() == [onedrive.Path.cwd().resolve()]


def test_upload_file_rejects_missing_file(monkeypatch, _upload_allowed_dirs_env):
    missing_path = _upload_allowed_dirs_env / "does_not_exist.pdf"

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(missing_path)))

    assert result["status"] == "error"
    assert "not found" in result["message"].lower()
    mock_request.assert_not_called()


def test_upload_file_rejects_empty_file(monkeypatch, _upload_allowed_dirs_env):
    empty_file = _upload_allowed_dirs_env / "empty.pdf"
    empty_file.write_bytes(b"")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(empty_file)))

    assert result["status"] == "error"
    assert "empty" in result["message"].lower()
    mock_request.assert_not_called()


def test_upload_file_returns_error_payload_on_api_failure(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"error": "boom"}, status_code=500)),
    )

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "error"


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


def test_upload_text_file_allows_plain_text_names(monkeypatch):
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file("notes.txt", "hello world"))

    assert result["status"] == "success"
