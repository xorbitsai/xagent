"""Tests for Google Drive long-running Workspace downloads."""

from __future__ import annotations

import errno
import importlib
import logging
from pathlib import Path
from typing import Any

import pytest
from google.auth.credentials import AnonymousCredentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class _Request:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.headers: dict[str, str] = {}
        self.execute_calls: list[int] = []

    def execute(self, *, num_retries: int = 0) -> dict[str, Any]:
        self.execute_calls.append(num_retries)
        return self.response


class _FilesResource:
    def __init__(self, operation: dict[str, Any]) -> None:
        self.request = _Request(operation)
        self.download_calls: list[dict[str, str]] = []

    def download(self, **kwargs: str) -> _Request:
        self.download_calls.append(kwargs)
        return self.request


class _OperationsResource:
    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = list(responses or [])
        self.get_calls: list[str] = []
        self.requests: list[_Request] = []

    def get(self, *, name: str) -> _Request:
        self.get_calls.append(name)
        if not self.responses:
            raise AssertionError(f"Unexpected poll for operation {name}")
        request = _Request(self.responses.pop(0))
        self.requests.append(request)
        return request


class _DriveService:
    def __init__(
        self,
        operation: dict[str, Any],
        *,
        poll_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self.files_resource = _FilesResource(operation)
        self.operations_resource = _OperationsResource(poll_responses)

    def files(self) -> _FilesResource:
        return self.files_resource

    def operations(self) -> _OperationsResource:
        return self.operations_resource


class _Response:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        error: Exception | None = None,
        status_code: int = 200,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.status_code = status_code
        self.raise_for_status_called = False

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True
        if self.error is not None:
            raise self.error

    def iter_content(self, *, chunk_size: int) -> list[bytes]:
        assert chunk_size == 1024 * 1024
        return self.chunks


class _AuthorizedSession:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.get_calls: list[dict[str, Any]] = []

    def __enter__(self) -> _AuthorizedSession:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.get_calls.append({"url": url, **kwargs})
        return self.response


def _module() -> Any:
    return importlib.import_module("xagent.web.services.google_drive_download")


def test_download_google_workspace_file_writes_completed_operation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService(
        {
            "name": "operations/download-1",
            "done": True,
            "response": {"downloadUri": "https://drive.example/download/1"},
        }
    )
    response = _Response([b"first", b"", b"second"])
    session = _AuthorizedSession(response)
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)
    destination = tmp_path / "slides.pptx"

    module.download_google_workspace_file(
        service=service,
        credentials=object(),
        file_id="slides-1",
        mime_type="application/vnd.test.presentation",
        destination=destination,
        timeout_seconds=600,
    )

    assert destination.read_bytes() == b"firstsecond"
    assert service.files_resource.download_calls == [
        {
            "fileId": "slides-1",
            "mimeType": "application/vnd.test.presentation",
        }
    ]
    assert response.raise_for_status_called is True
    assert session.get_calls == [
        {
            "url": "https://drive.example/download/1",
            "stream": True,
            "timeout": 60,
            "headers": {},
        }
    ]


def test_download_google_workspace_file_polls_pending_operation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService(
        {"name": "operations/download-2"},
        poll_responses=[
            {
                "name": "operations/download-2",
                "done": True,
                "response": {"downloadUri": "https://drive.example/download/2"},
            }
        ],
    )
    session = _AuthorizedSession(_Response([b"complete"]))
    sleep_calls: list[float] = []
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)
    monkeypatch.setattr(module, "sleep", sleep_calls.append)
    destination = tmp_path / "slides.pptx"

    module.download_google_workspace_file(
        service=service,
        credentials=object(),
        file_id="slides-2",
        mime_type="application/vnd.test.presentation",
        destination=destination,
        timeout_seconds=600,
    )

    assert destination.read_bytes() == b"complete"
    assert service.operations_resource.get_calls == ["download-2"]
    assert service.files_resource.request.execute_calls == [3]
    assert service.operations_resource.requests[0].execute_calls == [3]
    assert sleep_calls == [2]


def test_download_google_workspace_file_wraps_poll_http_error(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService({"name": "operations/download-poll-error"})

    class _HttpResponse(dict):
        status = 503
        reason = "Service Unavailable"

    poll_error = HttpError(_HttpResponse(), b"Drive unavailable")

    class _FailingPollRequest:
        headers: dict[str, str] = {}

        def execute(self, *, num_retries: int = 0) -> dict[str, Any]:
            assert num_retries == 3
            raise poll_error

    monkeypatch.setattr(module, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        service.operations_resource,
        "get",
        lambda *, name: _FailingPollRequest(),
    )

    with pytest.raises(
        module.GoogleDriveDownloadError,
        match="Drive operation polling failed",
    ) as exc_info:
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-poll-error",
            mime_type="application/vnd.test.presentation",
            destination=tmp_path / "slides.pptx",
            timeout_seconds=600,
        )

    assert exc_info.value.__cause__ is poll_error


def test_download_google_workspace_file_raises_completed_operation_error(
    tmp_path: Path,
) -> None:
    module = _module()
    service = _DriveService(
        {
            "name": "operations/download-3",
            "done": True,
            "error": {"code": 13, "message": "Drive could not create the export"},
        }
    )

    with pytest.raises(
        module.GoogleDriveDownloadError,
        match=r"Drive operation failed \(13\): Drive could not create the export",
    ):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-3",
            mime_type="application/vnd.test.presentation",
            destination=tmp_path / "slides.pptx",
            timeout_seconds=600,
        )

    assert not (tmp_path / "slides.pptx").exists()


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (None, "Unknown Drive operation error"),
        ("malformed error", "malformed error"),
    ],
)
def test_download_google_workspace_file_wraps_malformed_operation_error(
    error: object,
    expected_message: str,
    tmp_path: Path,
) -> None:
    module = _module()
    service = _DriveService({"done": True, "error": error})

    with pytest.raises(
        module.GoogleDriveDownloadError,
        match=f"Drive operation failed \\(unknown\\): {expected_message}",
    ):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-malformed-error",
            mime_type="application/vnd.test.presentation",
            destination=tmp_path / "slides.pptx",
            timeout_seconds=600,
        )


def test_download_google_workspace_file_clamps_sleep_to_remaining_deadline(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The poll sleep must not overshoot a shorter remaining deadline."""
    module = _module()
    service = _DriveService(
        {"name": "operations/download-clamped"},
        poll_responses=[
            {
                "name": "operations/download-clamped",
                "done": True,
                "response": {"downloadUri": "https://drive.example/download/clamped"},
            }
        ],
    )
    session = _AuthorizedSession(_Response([b"complete"]))
    times = iter([0.0, 4.0, 5.0])
    sleep_calls: list[float] = []
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)
    monkeypatch.setattr(module, "monotonic", lambda: next(times))
    monkeypatch.setattr(module, "sleep", sleep_calls.append)

    module.download_google_workspace_file(
        service=service,
        credentials=object(),
        file_id="slides-clamped",
        mime_type="application/vnd.test.presentation",
        destination=tmp_path / "slides.pptx",
        timeout_seconds=5,
    )

    assert sleep_calls == [1]


def test_download_google_workspace_file_stops_at_operation_timeout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService(
        {"name": "operations/download-4"},
        poll_responses=[{"name": "operations/download-4"}],
    )
    times = iter([0.0, 0.0, 10.0, 10.0])
    sleep_calls: list[float] = []
    monkeypatch.setattr(module, "monotonic", lambda: next(times))
    monkeypatch.setattr(module, "sleep", sleep_calls.append)

    with pytest.raises(
        module.GoogleDriveDownloadTimeout,
        match="Drive download for slides-4 did not finish within 10 seconds",
    ):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-4",
            mime_type="application/vnd.test.presentation",
            destination=tmp_path / "slides.pptx",
            timeout_seconds=10,
        )

    assert service.operations_resource.get_calls == ["download-4"]
    assert sleep_calls == [2]
    assert not (tmp_path / "slides.pptx").exists()


@pytest.mark.parametrize(
    ("operation", "expected_message"),
    [
        ({}, "Drive returned a pending operation without a name"),
        ({"done": True}, "Drive completed the operation without a response"),
        (
            {"done": True, "response": {}},
            "Drive completed the operation without a download URI",
        ),
    ],
)
def test_download_google_workspace_file_rejects_malformed_operation(
    operation: dict[str, Any],
    expected_message: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService(operation)
    monkeypatch.setattr(module, "sleep", lambda _seconds: None)

    with pytest.raises(module.GoogleDriveDownloadError, match=expected_message):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-invalid",
            mime_type="application/vnd.test.presentation",
            destination=tmp_path / "slides.pptx",
            timeout_seconds=600,
        )


def test_download_google_workspace_file_sends_supplied_resource_key_on_start(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService(
        {
            "done": True,
            "response": {"downloadUri": "https://drive.example/download/linked"},
        }
    )
    session = _AuthorizedSession(_Response([b"linked"]))
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)

    module.download_google_workspace_file(
        service=service,
        credentials=object(),
        file_id="slides-linked",
        mime_type="application/vnd.test.presentation",
        destination=tmp_path / "slides.pptx",
        timeout_seconds=600,
        resource_key="link-resource-key",
    )

    expected_headers = {"X-Goog-Drive-Resource-Keys": "slides-linked/link-resource-key"}
    assert service.files_resource.request.headers == expected_headers
    assert session.get_calls[0]["headers"] == expected_headers


def test_download_google_workspace_file_sends_resource_key_without_logging_secrets(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: Any,
) -> None:
    module = _module()
    service = _DriveService(
        {"name": "operations/download-shared"},
        poll_responses=[
            {
                "name": "operations/download-shared",
                "done": True,
                "response": {"downloadUri": "https://drive.example/download/shared"},
            }
        ],
    )
    session = _AuthorizedSession(_Response([b"shared"]))
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)
    monkeypatch.setattr(module, "sleep", lambda _seconds: None)
    caplog.set_level(logging.INFO, logger=module.__name__)

    module.download_google_workspace_file(
        service=service,
        credentials=object(),
        file_id="slides-shared",
        mime_type="application/vnd.test.presentation",
        destination=tmp_path / "slides.pptx",
        timeout_seconds=600,
        resource_key="resource-secret",
    )

    expected_headers = {"X-Goog-Drive-Resource-Keys": "slides-shared/resource-secret"}
    assert service.files_resource.request.headers == expected_headers
    assert service.operations_resource.requests[0].headers == expected_headers
    assert session.get_calls[0]["headers"] == expected_headers
    assert "Drive download operation started file_id=slides-shared" in caplog.text
    assert "Drive download operation completed file_id=slides-shared" in caplog.text
    assert "resource-secret" not in caplog.text
    assert "drive.example" not in caplog.text


def test_drive_discovery_exposes_long_running_download_contract() -> None:
    """The minimum client contract includes Drive download and polling methods."""
    service = build(
        "drive",
        "v3",
        credentials=AnonymousCredentials(),
        cache_discovery=False,
    )

    download_request = service.files().download(
        fileId="slides-contract",
        mimeType="application/vnd.test.presentation",
    )
    poll_request = service.operations().get(name="download-contract")

    assert isinstance(download_request.headers, dict)
    assert isinstance(poll_request.headers, dict)


def test_download_google_workspace_file_caps_poll_backoff(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService(
        {"name": "operations/download-backoff"},
        poll_responses=[
            {"name": "operations/download-backoff"},
            {"name": "operations/download-backoff", "done": False},
            {"name": "operations/download-backoff"},
            {"name": "operations/download-backoff", "done": False},
            {"name": "operations/download-backoff"},
            {"name": "operations/download-backoff", "done": False},
            {
                "name": "operations/download-backoff",
                "done": True,
                "response": {"downloadUri": "https://drive.example/download/backoff"},
            },
        ],
    )
    session = _AuthorizedSession(_Response([b"complete"]))
    sleep_calls: list[float] = []
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)
    monkeypatch.setattr(module, "sleep", sleep_calls.append)

    module.download_google_workspace_file(
        service=service,
        credentials=object(),
        file_id="slides-backoff",
        mime_type="application/vnd.test.presentation",
        destination=tmp_path / "slides.pptx",
        timeout_seconds=600,
    )

    assert sleep_calls == [2, 4, 8, 16, 32, 60, 60]


def test_download_google_workspace_file_rejects_export_over_max_bytes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The exported artifact must not exceed the caller's byte limit."""
    module = _module()
    service = _DriveService(
        {
            "done": True,
            "response": {"downloadUri": "https://drive.example/download/large"},
        }
    )
    session = _AuthorizedSession(_Response([b"123", b"45"]))
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)
    destination = tmp_path / "slides.pptx"

    with pytest.raises(
        module.GoogleDriveDownloadError,
        match="Exported file exceeds the maximum allowed size",
    ):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-large-export",
            mime_type="application/vnd.test.presentation",
            destination=destination,
            timeout_seconds=600,
            max_bytes=4,
        )

    assert destination.read_bytes() == b"123"


def test_download_google_workspace_file_rejects_zero_byte_export(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """An empty Drive export should fail before the parser sees it."""
    module = _module()
    service = _DriveService(
        {
            "done": True,
            "response": {"downloadUri": "https://drive.example/download/empty"},
        }
    )
    session = _AuthorizedSession(_Response([b""]))
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)
    destination = tmp_path / "slides.pptx"

    with pytest.raises(
        module.GoogleDriveDownloadError,
        match="Drive export produced no content",
    ):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-empty-export",
            mime_type="application/vnd.test.presentation",
            destination=destination,
            timeout_seconds=600,
        )

    assert destination.read_bytes() == b""


def test_download_google_workspace_file_preserves_partial_file_on_stream_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService(
        {
            "done": True,
            "response": {"downloadUri": "https://drive.example/download/partial"},
        }
    )

    class _MidstreamFailureResponse(_Response):
        def iter_content(self, *, chunk_size: int):
            assert chunk_size == 1024 * 1024
            yield b"partial"
            raise RuntimeError("connection reset")

    session = _AuthorizedSession(_MidstreamFailureResponse([]))
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)
    destination = tmp_path / "slides.pptx"

    with pytest.raises(
        module.GoogleDriveDownloadError,
        match="Final Drive download request failed",
    ):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-partial",
            mime_type="application/vnd.test.presentation",
            destination=destination,
            timeout_seconds=600,
        )

    assert destination.read_bytes() == b"partial"


def test_download_google_workspace_file_reports_local_write_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService(
        {
            "done": True,
            "response": {"downloadUri": "https://drive.example/download/write"},
        }
    )
    session = _AuthorizedSession(_Response([b"content"]))
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)

    with pytest.raises(
        module.GoogleDriveDownloadError,
        match="Failed to write Drive download: No such file or directory",
    ):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-write-failure",
            mime_type="application/vnd.test.presentation",
            destination=tmp_path / "missing" / "slides.pptx",
            timeout_seconds=600,
        )


def test_download_google_workspace_file_reports_midstream_write_failure(
    monkeypatch: Any,
) -> None:
    """A disk failure after a partial write must use the service error contract."""
    module = _module()
    service = _DriveService(
        {
            "done": True,
            "response": {"downloadUri": "https://drive.example/download/disk-full"},
        }
    )
    session = _AuthorizedSession(_Response([b"first", b"second"]))
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)

    class _FailingOutput:
        def __init__(self) -> None:
            self.writes: list[bytes] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def write(self, chunk: bytes) -> None:
            if self.writes:
                raise OSError(errno.ENOSPC, "No space left on device")
            self.writes.append(chunk)

    output = _FailingOutput()

    class _Destination:
        def open(self, mode: str) -> _FailingOutput:
            assert mode == "wb"
            return output

    destination: Any = _Destination()

    with pytest.raises(
        module.GoogleDriveDownloadError,
        match="Failed to write Drive download: No space left on device",
    ):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-disk-full",
            mime_type="application/vnd.test.presentation",
            destination=destination,
            timeout_seconds=600,
        )

    assert output.writes == [b"first"]


def test_download_google_workspace_file_wraps_final_http_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = _module()
    service = _DriveService(
        {
            "done": True,
            "response": {"downloadUri": "https://drive.example/download/failure"},
        }
    )
    response = _Response([], error=RuntimeError("signed URI"), status_code=503)
    session = _AuthorizedSession(response)
    monkeypatch.setattr(module, "AuthorizedSession", lambda _credentials: session)
    destination = tmp_path / "slides.pptx"

    with pytest.raises(
        module.GoogleDriveDownloadError,
        match="Final Drive download failed with HTTP status 503",
    ):
        module.download_google_workspace_file(
            service=service,
            credentials=object(),
            file_id="slides-failure",
            mime_type="application/vnd.test.presentation",
            destination=destination,
            timeout_seconds=600,
        )

    assert not destination.exists()
