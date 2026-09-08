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
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS", str(tmp_path))
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
