import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import google_drive


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)


@pytest.fixture(autouse=True)
def _output_dir_env(tmp_path, monkeypatch):
    """Every test gets its own isolated output root so nothing here ever
    writes into the real working directory."""
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", str(tmp_path))
    return tmp_path


def _mock_drive_service(monkeypatch, files_mock):
    service = Mock()
    service.files.return_value = files_mock
    monkeypatch.setattr(google_drive, "get_drive_service", lambda: service)
    return service


class _FakeDownloader:
    """Mirrors googleapiclient.http.MediaIoBaseDownload's interface closely
    enough for this file's download loop: write `content` into the target
    buffer across one call to next_chunk()."""

    def __init__(self, fh, request, content: bytes = b"") -> None:
        self._fh = fh
        self._content = content

    def next_chunk(self):
        self._fh.write(self._content)
        return None, True


def _patch_downloader(monkeypatch, content: bytes):
    monkeypatch.setattr(
        google_drive,
        "MediaIoBaseDownload",
        lambda fh, request: _FakeDownloader(fh, request, content),
    )


# ---------------------------------------------------------------------------
# google_drive_get_file_content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mime_type", ["text/plain", "text/csv", "text/markdown", "application/json"]
)
def test_get_file_content_accepts_text_mime_types(monkeypatch, mime_type):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "notes.txt",
        "mimeType": "text/plain",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"hello world")

    result = json.loads(google_drive.google_drive_get_file_content("f1", mime_type))

    assert result["status"] == "success"
    assert result["content"] == "hello world"


@pytest.mark.parametrize(
    "mime_type",
    [
        "application/pdf",
        "image/png",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ],
)
def test_get_file_content_rejects_binary_mime_types(monkeypatch, mime_type):
    """Regression guard: decoding binary content as UTF-8 with
    errors="replace" silently corrupts it — reject before even calling the
    API rather than return garbage that looks superficially like success."""
    files = Mock()
    _mock_drive_service(monkeypatch, files)

    result = json.loads(google_drive.google_drive_get_file_content("f1", mime_type))

    assert result["status"] == "error"
    assert "google_drive_download_file" in result["message"]
    files.get.assert_not_called()


def test_get_file_content_rejects_regular_file_whose_real_mime_type_is_binary(
    monkeypatch,
):
    """Regression guard for the actual bug this tool exists to fix: for a
    regular (non-Workspace) file, get_media() ignores the mime_type
    parameter entirely and returns the file's own real bytes — so it's
    file_mime_type (from Drive's metadata), not the request's mime_type
    default of "text/plain", that must gate whether decoding as UTF-8 is
    safe. Calling with the default text/plain param on an actual PDF must
    still be rejected, not silently corrupted."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "report.pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"%PDF-1.4 real pdf bytes")

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "error"
    assert "google_drive_download_file" in result["message"]


def test_get_file_content_accepts_regular_text_file(monkeypatch):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "notes.txt",
        "mimeType": "text/plain",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"hello world")

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "success"
    assert result["content"] == "hello world"


def test_get_file_content_accepts_rtf(monkeypatch):
    """Regression guard: RTF is 7-bit-ASCII-clean per spec (non-ASCII
    content is escaped, not raw bytes), so it round-trips through UTF-8
    decoding safely — the binary-content guard must not reject it."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "notes.rtf",
        "mimeType": "application/rtf",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, rb"{\rtf1 hello}")

    result = json.loads(
        google_drive.google_drive_get_file_content("f1", "application/rtf")
    )

    assert result["status"] == "success"


def test_get_file_content_exports_workspace_doc(monkeypatch):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Notes",
        "mimeType": "application/vnd.google-apps.document",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"exported text")

    result = json.loads(google_drive.google_drive_get_file_content("f1", "text/plain"))

    assert result["status"] == "success"
    assert result["content"] == "exported text"
    files.export_media.assert_called_once_with(fileId="f1", mimeType="text/plain")
    files.get_media.assert_not_called()


def test_get_file_content_returns_error_payload_on_api_failure(monkeypatch):
    files = Mock()
    files.get.return_value.execute.side_effect = RuntimeError("boom")
    _mock_drive_service(monkeypatch, files)

    result = json.loads(google_drive.google_drive_get_file_content("f1", "text/plain"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# google_drive_download_file
# ---------------------------------------------------------------------------


def test_download_file_writes_regular_binary_file(monkeypatch, tmp_path):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "photo.png",
        "mimeType": "image/png",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"\x89PNG-fake-bytes")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    assert result["size"] == len(b"\x89PNG-fake-bytes")
    output_path = tmp_path / "output" / "photo.png"
    assert result["path"] == str(output_path)
    assert output_path.read_bytes() == b"\x89PNG-fake-bytes"
    files.get_media.assert_called_once_with(fileId="f1")
    files.export_media.assert_not_called()


def test_download_file_exports_workspace_doc_with_extension_appended(
    monkeypatch, tmp_path
):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Onboarding Deck",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"%PDF-1.4 fake pdf")

    result = json.loads(
        google_drive.google_drive_download_file("f1", mime_type="application/pdf")
    )

    assert result["status"] == "success"
    output_path = tmp_path / "output" / "Onboarding Deck.pdf"
    assert result["path"] == str(output_path)
    assert output_path.read_bytes() == b"%PDF-1.4 fake pdf"
    files.export_media.assert_called_once_with(fileId="f1", mimeType="application/pdf")
    files.get_media.assert_not_called()


def test_download_file_requires_mime_type_for_workspace_doc(monkeypatch):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Onboarding Deck",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service(monkeypatch, files)

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "error"
    assert "mime_type" in result["message"]
    files.export_media.assert_not_called()


def test_download_file_uses_explicit_filename(monkeypatch, tmp_path):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Onboarding Deck",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file(
            "f1", mime_type="application/pdf", filename="custom.pdf"
        )
    )

    assert result["status"] == "success"
    assert result["path"] == str(tmp_path / "output" / "custom.pdf")


def test_download_file_appends_extension_to_explicit_filename_missing_one(
    monkeypatch, tmp_path
):
    """Regression guard: an explicit filename with no extension must still
    get the export mime_type's extension appended, exactly like the
    default (Drive-name-derived) filename already does — the caller
    shouldn't lose the .pdf just because they named the file themselves."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Onboarding Deck",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file(
            "f1", mime_type="application/pdf", filename="report"
        )
    )

    assert result["status"] == "success"
    assert result["path"] == str(tmp_path / "output" / "report.pdf")


def test_download_file_does_not_double_extension_on_case_mismatch(
    monkeypatch, tmp_path
):
    """Regression guard: matching the target extension must be
    case-insensitive — a Drive name already ending in ".PDF" (any case)
    exported to "application/pdf" must not become "....PDF.pdf"."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Report.PDF",
        "mimeType": "application/vnd.google-apps.document",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file("f1", mime_type="application/pdf")
    )

    assert result["status"] == "success"
    assert result["path"] == str(tmp_path / "output" / "Report.PDF")


def test_download_file_sanitizes_path_traversal_in_filename(monkeypatch, tmp_path):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "../../etc/passwd",
        "mimeType": "text/plain",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.parent == tmp_path / "output"
    assert output_path.is_relative_to(tmp_path / "output")


def test_download_file_sanitizes_unsafe_characters_in_filename(monkeypatch, tmp_path):
    """Regression guard: a name with characters Path.name alone would leave
    untouched (no "/" to strip) must still be neutralized by the character
    allowlist — otherwise this test would pass even with the sanitizer
    reduced to a no-op Path(name).name call."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "weird:name?.txt",
        "mimeType": "text/plain",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.parent == tmp_path / "output"
    assert ":" not in output_path.name
    assert "?" not in output_path.name
    assert output_path.name.endswith(".txt")


def test_download_file_preserves_extension_for_degenerate_drive_name(
    monkeypatch, tmp_path
):
    """Regression guard: sanitizing a degenerate name (all characters the
    allowlist/strip would remove) must happen *before* the export extension
    is appended — otherwise the trailing ".strip('._')" eats into the
    extension's own leading dot and produces a bare "pdf" instead of a
    usable "file.pdf"."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "...",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file("f1", mime_type="application/pdf")
    )

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.name == "file.pdf"


def test_download_file_preserves_extension_for_non_ascii_drive_name(
    monkeypatch, tmp_path
):
    """Regression guard: a name whose entire stem is non-ASCII (e.g. CJK)
    sanitizes down to nothing on its own, but the extension must survive —
    this is the regular-file (get_media) branch, which has no separate
    "extension" variable to fall back on the way the Workspace-export
    branch does, so the fix has to live in _safe_output_filename itself."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "季度报告.pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.name == "file.pdf"


def test_download_file_preserves_extension_only_drive_name(monkeypatch, tmp_path):
    """Regression guard: Path(".pdf").suffix is '' per pathlib's dotfile
    convention (a leading dot with nothing before it never counts as an
    extension separator) — so a Drive name that's *exactly* an extension
    must still be recognized as one, not silently reduced to "pdf" with
    the leading dot dropped."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": ".pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.name == "file.pdf"


def test_download_file_sanitizes_path_traversal_in_explicit_filename(
    monkeypatch, tmp_path
):
    """Regression guard: only the Drive-reported name was tested for
    traversal/unsafe-character sanitization elsewhere — an explicit
    `filename` argument goes through the exact same _safe_output_filename
    call and must be sanitized identically."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "report.txt",
        "mimeType": "text/plain",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file("f1", filename="../../etc/passwd")
    )

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.parent == tmp_path / "output"
    assert output_path.is_relative_to(tmp_path / "output")


def test_download_file_truncates_overlong_filename(monkeypatch, tmp_path):
    """Regression guard: an unbounded sanitized filename can exceed the
    ~255-byte NAME_MAX most filesystems enforce, making the final
    write_bytes() raise OSError/ENAMETOOLONG and fail the whole call
    instead of just using a shorter name."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": ("a" * 500) + ".pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert len(output_path.name) <= 210
    assert output_path.name.endswith(".pdf")


def test_download_file_truncates_overlong_suffix_too(monkeypatch, tmp_path):
    """Regression guard: the length cap must also bound the suffix, not
    just the stem — a name like "a." + 300 chars has almost its entire
    length in what _split_stem_suffix treats as the "extension" (everything
    from the last dot onward), so truncating only the stem would still
    leave a 300+ char filename."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "a." + ("x" * 300),
        "mimeType": "text/plain",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert len(output_path.name) <= 30


def test_download_file_errors_when_no_task_workspace_is_configured(
    monkeypatch, tmp_path
):
    """Regression guard: an unset output-dir env var must fail loudly
    rather than silently writing into whatever directory the MCP
    subprocess happens to have as its cwd."""
    monkeypatch.delenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", raising=False)
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "report.pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "error"
    assert "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR" in result["message"]
    assert not (tmp_path / "output").exists()


def test_download_file_dedupes_existing_filename(monkeypatch, tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "deck.pdf").write_bytes(b"already here")

    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "deck.pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service(monkeypatch, files)
    _patch_downloader(monkeypatch, b"new content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    assert result["path"] == str(tmp_path / "output" / "deck (1).pdf")
    # The original file must survive untouched.
    assert (tmp_path / "output" / "deck.pdf").read_bytes() == b"already here"


def test_download_file_returns_error_payload_on_api_failure(monkeypatch):
    files = Mock()
    files.get.return_value.execute.side_effect = RuntimeError("boom")
    _mock_drive_service(monkeypatch, files)

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "error"
    assert "boom" in result["message"]
