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


@pytest.mark.parametrize(
    "traversal_remote_path",
    ["../../etc/passwd", "../../../../me/messages", "foo/../../bar", "./secret"],
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
    assert "range conflict" in str(exc_info.value)
    for record in caplog.records:
        assert sentinel_url not in record.getMessage()


def test_upload_large_file_content_redacts_url_even_if_echoed_in_response_body(
    monkeypatch, caplog
):
    """Regression guard: the redaction must not rely on the URL only ever
    appearing in requests' own HTTPError message -- an intervening proxy
    or WAF error page that happens to echo the request URL into the
    response *body* must be scrubbed too, since the body is appended
    verbatim to the raised error."""
    sentinel_url = "https://upload.example/session-with-a-secret-token-abc123"
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": sentinel_url})),
    )
    waf_body = f"Blocked: request to {sentinel_url} was denied".encode()
    error_response = MockResponse({}, status_code=403, content=waf_body)
    _patch_session(monkeypatch, _FakeSession(put=Mock(return_value=error_response)))

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError) as exc_info:
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    assert sentinel_url not in str(exc_info.value)
    assert "Blocked" in str(exc_info.value)


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
        {"error": {"message": "boom"}}, status_code=500, content=b'{"error": "boom"}'
    )
    delete_404_response = MockResponse({}, status_code=404)
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(return_value=error_response),
            delete=Mock(return_value=delete_404_response),
        ),
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="boom"):
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

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
    error_response = MockResponse({}, status_code=500, content=b"{}")
    cleanup_error = requests.ConnectionError(f"Failed to reach {sentinel_url}")
    _patch_session(
        monkeypatch,
        _FakeSession(
            put=Mock(return_value=error_response),
            delete=Mock(side_effect=cleanup_error),
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


def test_upload_large_file_content_warns_on_failed_cleanup_status(monkeypatch, caplog):
    """Regression guard: requests doesn't raise on its own for an HTTP
    error status -- a 429/5xx response to the cancellation DELETE must not
    be silently treated as a successful cancellation."""
    total_size = onedrive._UPLOAD_SESSION_CHUNK_SIZE + 10
    fh = io.BytesIO(b"\x00" * total_size)

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"uploadUrl": "https://upload.example/s"})),
    )
    error_response = MockResponse(
        {"error": {"message": "boom"}}, status_code=500, content=b'{"error": "boom"}'
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
        with pytest.raises(RuntimeError, match="boom"):
            onedrive._upload_large_file_content(
                "big.bin", fh, total_size, "application/octet-stream"
            )

    assert any("429" in record.getMessage() for record in caplog.records)


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


@pytest.mark.parametrize(
    "mime_type",
    [
        "application/x-sql",
        "application/sql",
        "application/x-httpd-php",
        "application/vnd.dart",
        "application/x-tex",
        "application/x-csh",
        "application/vnd.groove-tool-template",
    ],
)
def test_is_binary_mime_type_excludes_known_text_formats_directly(mime_type):
    """Regression guard independent of the host's own mimetypes database:
    whether mimetypes.guess_type actually resolves a given extension to one
    of these types varies by host (e.g. ".php" only resolves to
    "application/x-httpd-php" with a fuller system mime.types installed,
    not reproduced on every machine/CI image). Under the positive
    known-binary-formats design, these types are "not binary" simply by
    not appearing in _BINARY_MIME_TYPES/_BINARY_MIME_PREFIXES -- this pins
    that down directly so a future addition to that set can't silently
    re-capture one of these genuinely-text formats."""
    assert not onedrive._is_binary_mime_type(mime_type)


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
        "app.pub",
        "installer.dmg",
        "image.iso",
        "app.apk",
        "book.mobi",
    ],
)
def test_upload_text_file_rejects_additional_binary_extensions(monkeypatch, file_path):
    """Regression guard: reviewer-flagged gap in the original binary-
    extension set — fonts, WASM, columnar/DB, and several common formats
    stdlib mimetypes also has no opinion on were missing from it, so with
    no other signal they used to sail straight through the guard and get
    silently created as mislabeled text files. (.pub/.dmg/.iso/.apk/.mobi
    were re-verified after switching to a positive known-binary-formats
    list, since a host WITH a system mime.types installed resolves them to
    a specific mimetype that must itself be in _BINARY_MIME_TYPES -- the
    "no mimetype at all" fallback alone only covers a bare-stdlib host.)"""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "error"
    assert "onedrive_upload_file" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize(
    "file_path",
    [
        "app.jar",
        "Main.class",
        "installer.cab",
        "package.deb",
        "file.torrent",
        "keystore.p12",
        "cert.pfx",
        "movie.swf",
    ],
)
def test_upload_text_file_rejects_further_binary_extensions(monkeypatch, file_path):
    """Regression guard: a follow-up self-review sweep of the positive
    _BINARY_MIME_TYPES/fallback-set design found these archive/executable/
    key-bundle formats were still missing (verified directly against
    mimetypes.guess_type on both a bare-stdlib and a full-mime.types host),
    so they used to sail through the guard as "text" the same way the
    extensions above once did."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "error"
    assert "onedrive_upload_file" in result["message"]
    mock_request.assert_not_called()


def test_upload_text_file_allows_svg(monkeypatch):
    """Regression guard: "image/svg+xml" starts with the "image/" prefix
    in _BINARY_MIME_PREFIXES, but SVG is a plain-text XML format an agent
    may legitimately generate and upload as text -- without the
    _TEXT_SAFE_MIME_SUFFIXES carve-out for "+xml"/"+json"/"+yaml", the
    positive-list redesign would misclassify it as binary, reproducing the
    exact false-positive bug class this whole redesign exists to fix."""
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file("chart.svg", "<svg></svg>"))

    assert result["status"] == "success"


@pytest.mark.parametrize(
    "file_path", ["app.ts", "deploy.bat", "session.scm", "worksheet.sc"]
)
def test_upload_text_file_allows_ambiguous_extensions_that_collide_with_binary_mimetypes(
    monkeypatch, file_path
):
    """Regression guard for a bug introduced by switching from a fixed
    extension allowlist to mimetype-driven detection: mimetypes.guess_type
    resolves ".ts" to "video/mp2t", ".bat" to "application/x-msdownload",
    ".scm" to "application/vnd.lotus-screencam", and ".sc" to
    "application/vnd.ibm.secure-container" -- none of which are
    text-safe -- even though all four extensions are overwhelmingly used
    for genuine text/source content (TypeScript, Windows batch scripts,
    Scheme source, Scala worksheets) in practice. Without an explicit
    carve-out, onedrive_upload_text_file would reject these with no
    working alternative (onedrive_upload_file needs an existing local
    file, not raw text)."""
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "success"


@pytest.mark.parametrize(
    "file_path,simulated_mime_type",
    [
        ("schema.sql", "application/x-sql"),
        ("schema.sql", "application/sql"),
        ("index.php", "application/x-httpd-php"),
        ("main.dart", "application/vnd.dart"),
        ("paper.tex", "application/x-tex"),
        ("build.csh", "application/x-csh"),
        ("layout.tpl", "application/vnd.groove-tool-template"),
        ("deploy.sh", "application/x-sh"),
        ("subs.srt", "application/x-subrip"),
        ("schema.dtd", "application/xml-dtd"),
        ("host.crt", "application/x-x509-ca-cert"),
    ],
)
def test_upload_text_file_allows_source_extensions_with_non_text_mime_guess(
    monkeypatch, file_path, simulated_mime_type
):
    """Regression guard: a reviewer-verified false-positive class --
    ".sql"/".php"/".dart"/".tex"/".csh"/".tpl"/".sh"/".srt"/".dtd"/".crt"
    resolve via mimetypes on at least some hosts to non-text application/*
    types. None of these are in _BINARY_MIME_TYPES (nor collide with a
    real binary format the way .ts/.bat/.scm/.sc do), so _is_binary_mime_type
    correctly treats them as text by default rather than needing an
    explicit text-safe entry -- ".sh" in particular is a common,
    frequently-generated format that shipped genuinely broken (rejected
    with no working alternative) before this fix.

    The mimetype each extension actually resolves to is host- and Python-
    version-dependent (confirmed directly: this file's own dev host and a
    CI run on a different OS/Python disagreed on ".sql" -- "application/
    x-sql" locally, "application/sql" on CI) -- monkeypatching
    mimetypes.guess_type to a fixed value per case makes this test
    deterministic instead of silently depending on whichever mime.types
    database happens to be installed on whatever machine runs it.
    """
    monkeypatch.setattr(
        onedrive.mimetypes, "guess_type", lambda name: (simulated_mime_type, None)
    )
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "success"


def test_upload_text_file_allows_ps1_regardless_of_host_mimetypes(monkeypatch):
    """Regression guard: ".ps1" was mistakenly placed in the hand-
    maintained binary-extension fallback set in an earlier commit
    (PowerShell scripts are genuine text) and is now in
    _AMBIGUOUS_TEXT_EXTENSIONS instead -- unlike the mime-type-allowlist
    cases above, this must hold regardless of what mimetypes.guess_type
    returns for it on any given host, which is exactly what the ambiguous-
    extension short-circuit guarantees."""
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file("deploy.ps1", "some text"))

    assert result["status"] == "success"


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
    is a ".ts" file) whenever no explicit mime_type is passed."""
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


def test_normalize_item_id_rejects_dot_segments_directly():
    """Regression guard: item_id-based endpoints are built as
    f"/me/drive/items/{quote(item_id, safe='')}" without ever going through
    _normalize_path/_item_path -- but quote() never percent-encodes a bare
    "." or ".." (they're in urllib's own "always safe" unreserved set), so
    without this explicit check an item_id of "." or ".." still reaches
    requests as a literal dot-segment and collapses the request path onto
    a different Graph endpoint under the same OAuth token, the same
    traversal class _normalize_path exists to stop for path-based tools."""
    for bad_id in [".", ".."]:
        with pytest.raises(ValueError, match="must not be"):
            onedrive._normalize_item_id(bad_id)


def test_normalize_item_id_rejects_empty_value():
    with pytest.raises(ValueError, match="required"):
        onedrive._normalize_item_id("   ")


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
