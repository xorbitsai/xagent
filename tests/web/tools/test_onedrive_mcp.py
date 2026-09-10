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


def test_upload_file_uses_upload_session_for_large_files(
    monkeypatch, _upload_allowed_dirs_env
):
    """Files over Graph's 4 MiB simple-PUT cap must go through
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
    monkeypatch.setattr(onedrive.requests, "put", mock_put)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

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
    # Chunk uploads use a longer timeout than the small-JSON-call default —
    # a multi-megabyte PUT over a slow link can legitimately take longer
    # than DEFAULT_TIMEOUT_SECONDS.
    assert first_call.kwargs["timeout"] == onedrive._CHUNK_UPLOAD_TIMEOUT_SECONDS
    # The pre-authenticated upload session URL must never carry our own
    # Authorization header alongside its own query-string token.
    assert "Authorization" not in first_call.kwargs["headers"]
    assert "Authorization" not in second_call.kwargs["headers"]


def test_upload_large_file_content_reads_bounded_chunks_from_disk(
    monkeypatch, _upload_allowed_dirs_env
):
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

    monkeypatch.setattr(onedrive.requests, "put", Mock(side_effect=_fake_put))

    result = onedrive._upload_large_file_content("big.bin", fh, total_size)

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
    monkeypatch.setattr(onedrive.requests, "put", Mock(return_value=error_response))

    with pytest.raises(RuntimeError, match="Invalid upload session"):
        onedrive._upload_large_file_content("big.bin", fh, total_size)


def test_simple_upload_max_bytes_is_at_or_below_graphs_4mb_limit():
    """Regression guard for the actual production boundary bug: Graph's
    simple content PUT is documented as accepting files up to "4 MB", which
    some deployments enforce as the decimal 4,000,000 bytes rather than the
    binary 4 MiB (4,194,304 bytes). The cutoff must stay at or below the
    smaller, decimal figure so a file in that ambiguous gap always takes the
    resumable upload-session path instead of risking rejection right at the
    simple-PUT boundary."""
    assert onedrive._SIMPLE_UPLOAD_MAX_BYTES <= 4_000_000


@pytest.mark.parametrize(
    "file_path", ["font.woff2", "font.woff", "font.ttf", "data.parquet", "cache.sqlite"]
)
def test_upload_text_file_rejects_additional_binary_extensions(monkeypatch, file_path):
    """Regression guard: reviewer-flagged gap in the original binary-
    extension set — fonts, WASM, and columnar/DB formats were missing from
    it, so with no other signal they used to sail straight through the
    guard and get silently created as mislabeled text files."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "error"
    assert "onedrive_upload_file" in result["message"]
    mock_request.assert_not_called()


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
