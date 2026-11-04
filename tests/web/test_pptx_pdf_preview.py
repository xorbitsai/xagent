"""Regression tests for the .pptx -> PDF preview cache and endpoint.

The preview endpoint must keep its existing contract while reusing the shared
slide-conversion helper: managed cache key/location, cache hits without
LibreOffice, a same-filesystem temporary output with an atomic publish, the
None fallback used for the client-side canvas renderer, and the inline
application/pdf response. Database release and publication-contention
behavior is covered by the existing
tests/web/services/test_complete_upload_cleanup.py and
tests/web/services/test_cleanup_publication_contracts.py suites.
"""

import asyncio
import os
import pathlib

import pytest

from tests.web.test_file_upload import auth_headers as _file_upload_auth_headers
from tests.web.test_file_upload import client as _file_upload_client
from tests.web.test_file_upload import temp_uploads_dir as _file_upload_temp_uploads_dir
from tests.web.test_file_upload import test_db as _file_upload_test_db
from xagent.web.api import files
from xagent.web.services import uploaded_file_cleanup_publication


# Reuse the upload-endpoint fixtures from the files suite instead of copying
# its database/storage bootstrap. Wrapping keeps this module's fixture names
# distinct from the imported helpers that ruff sees at module scope.
@pytest.fixture
def test_db(monkeypatch):
    yield from _file_upload_test_db.__wrapped__(monkeypatch)


@pytest.fixture
def auth_headers(test_db):
    return _file_upload_auth_headers.__wrapped__(test_db)


@pytest.fixture
def client(test_db):
    return _file_upload_client.__wrapped__(test_db)


@pytest.fixture
def temp_uploads_dir(monkeypatch):
    yield from _file_upload_temp_uploads_dir.__wrapped__(monkeypatch)


PPTX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)


def _require_shared_conversion():
    module = getattr(files, "pptx_conversion", None)
    assert module is not None, (
        "the .pptx preview must reuse the shared pptx conversion helper"
    )
    return module


def _write_pptx(directory: pathlib.Path, name: str = "deck.pptx") -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"PK\x03\x04 synthetic preview deck")
    return path


def _outdir_from_arguments(arguments) -> pathlib.Path:
    return pathlib.Path(arguments[arguments.index("--outdir") + 1])


_simulated_processes = {}


@pytest.fixture(autouse=True)
def simulated_process_groups(monkeypatch):
    kill_group = os.killpg

    def kill(group, signal):
        if group in _simulated_processes:
            _simulated_processes[group].kill()
        else:
            kill_group(group, signal)

    monkeypatch.setattr(os, "killpg", kill)
    yield
    _simulated_processes.clear()


class _FakeProcess:
    def __init__(self, *, returncode: int = 0, stderr: bytes = b"", write_pdf=None):
        self.returncode = returncode
        self._stderr = stderr
        self._write_pdf = write_pdf
        self.pid = id(self)
        _simulated_processes[self.pid] = self
        self.killed = 0
        self.waits = 0

    def kill(self) -> None:
        self.killed += 1

    async def wait(self):
        self.waits += 1
        if self.killed:
            self.returncode = -9
        elif self._write_pdf is not None:
            self._write_pdf()
        return self.returncode


def _install_fake_process(monkeypatch, process_factory):
    calls: list[dict] = []

    async def create_subprocess_exec(*arguments, **kwargs):
        calls.append({"arguments": list(arguments), "kwargs": dict(kwargs)})
        process = process_factory(calls)
        kwargs["stderr"].write(process._stderr)
        kwargs["stderr"].flush()
        return process

    monkeypatch.setattr(files.asyncio, "create_subprocess_exec", create_subprocess_exec)
    return calls


def test_preview_cache_path_is_keyed_by_file_id_then_path_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)

    by_id = files._pptx_pdf_cache_path(pptx, "52a86d2f-1d4a-4b6f-8a71-2f2c4e3b1a90")
    legacy = files._pptx_pdf_cache_path(pptx)

    assert by_id.parent == tmp_path / "storage" / "pptx_pdf_cache"
    assert by_id.name == "52a86d2f-1d4a-4b6f-8a71-2f2c4e3b1a90.preview.pdf"
    assert legacy.parent == by_id.parent
    assert legacy.name != by_id.name
    assert legacy.name.endswith(".preview.pdf")
    assert files._pptx_pdf_cache_path(pptx) == legacy


@pytest.mark.asyncio
async def test_preview_cache_hit_never_starts_libreoffice(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)
    cache_path = files._pptx_pdf_cache_path(pptx)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"%PDF-1.4 cached")

    async def unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("a fresh cache entry must not start a conversion")

    monkeypatch.setattr(files.asyncio, "create_subprocess_exec", unexpected_subprocess)

    assert await files._convert_pptx_to_pdf(pptx) == cache_path
    assert cache_path.read_bytes() == b"%PDF-1.4 cached"


@pytest.mark.asyncio
async def test_preview_publishes_the_pdf_from_a_same_filesystem_temp_dir(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)
    cache_path = files._pptx_pdf_cache_path(pptx)

    def write_pdf():
        outdir = _outdir_from_arguments(calls[0]["arguments"])
        (outdir / "deck.pdf").write_bytes(b"%PDF-1.4 atomic")

    calls = _install_fake_process(
        monkeypatch, lambda _calls: _FakeProcess(write_pdf=write_pdf)
    )

    published = await files._convert_pptx_to_pdf(pptx)

    assert published == cache_path
    assert published.read_bytes() == b"%PDF-1.4 atomic"
    temp_dir = _outdir_from_arguments(calls[0]["arguments"])
    assert not temp_dir.exists()
    assert temp_dir.parent == cache_path.parent
    assert not list(cache_path.parent.glob("*.tmp"))


@pytest.mark.asyncio
async def test_preview_does_not_accept_a_stray_pdf_name(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)

    def write_stray_pdf():
        outdir = _outdir_from_arguments(calls[0]["arguments"])
        (outdir / "some-other-name.pdf").write_bytes(b"%PDF-1.4 stray")

    calls = _install_fake_process(
        monkeypatch, lambda _calls: _FakeProcess(write_pdf=write_stray_pdf)
    )

    assert await files._convert_pptx_to_pdf(pptx) is None
    assert not files._pptx_pdf_cache_path(pptx).exists()


@pytest.mark.asyncio
async def test_preview_passes_a_per_call_libreoffice_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)

    def write_pdf():
        outdir = _outdir_from_arguments(calls[0]["arguments"])
        (outdir / "deck.pdf").write_bytes(b"%PDF-1.4 profiled")

    calls = _install_fake_process(
        monkeypatch, lambda _calls: _FakeProcess(write_pdf=write_pdf)
    )

    assert await files._convert_pptx_to_pdf(pptx) is not None

    arguments = calls[0]["arguments"]
    profile_arguments = [
        argument
        for argument in arguments
        if argument.startswith("-env:UserInstallation=")
    ]
    assert len(profile_arguments) == 1
    profile_uri = profile_arguments[0].split("=", 1)[1]
    assert profile_uri.startswith("file://")
    profile_dir = pathlib.Path(profile_uri[len("file://") :])
    assert profile_dir != _outdir_from_arguments(arguments)
    assert not profile_dir.exists()


@pytest.mark.asyncio
async def test_preview_returns_none_when_libreoffice_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)

    async def missing_binary(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'soffice'")

    monkeypatch.setattr(files.asyncio, "create_subprocess_exec", missing_binary)

    assert await files._convert_pptx_to_pdf(pptx) is None
    assert not files._pptx_pdf_cache_path(pptx).exists()


@pytest.mark.asyncio
async def test_preview_returns_none_when_conversion_times_out(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)
    process = _FakeProcess(returncode=None)

    async def create_subprocess_exec(*_args, **_kwargs):
        return process

    waits = 0

    async def fake_wait_for(awaitable, timeout=None):
        nonlocal waits
        waits += 1
        if waits == 1:
            awaitable.close()
            raise asyncio.TimeoutError
        return await awaitable

    monkeypatch.setattr(files.asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr(files.asyncio, "wait_for", fake_wait_for)

    assert await files._convert_pptx_to_pdf(pptx) is None
    assert process.killed == 1
    assert process.waits == 1


@pytest.mark.asyncio
async def test_preview_returns_none_when_libreoffice_exits_non_zero(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)
    _install_fake_process(
        monkeypatch,
        lambda _calls: _FakeProcess(returncode=1, stderr=b"conversion failed"),
    )

    assert await files._convert_pptx_to_pdf(pptx) is None


@pytest.mark.asyncio
async def test_preview_guard_wraps_the_uncached_conversion(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)

    assert hasattr(files._convert_pptx_to_pdf_uncached, "__wrapped__")

    def busy_lock(_file_id):
        from filelock import Timeout

        raise Timeout("publication lock is held")

    async def unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("conversion must not start while publication is busy")

    monkeypatch.setattr(
        uploaded_file_cleanup_publication, "file_cleanup_lock", busy_lock
    )
    monkeypatch.setattr(files.asyncio, "create_subprocess_exec", unexpected_subprocess)

    file_id = "52a86d2f-1d4a-4b6f-8a71-2f2c4e3b1a90"
    assert await files._convert_pptx_to_pdf(pptx, file_id) is None


@pytest.mark.asyncio
async def test_preview_uses_the_shared_conversion_helper(tmp_path, monkeypatch):
    conversion = _require_shared_conversion()
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(tmp_path / "storage"))
    pptx = _write_pptx(tmp_path)
    records: dict[str, object] = {}

    async def fake_convert(pptx_path, output_dir, *, timeout_seconds=60.0, **kwargs):
        records["source"] = pathlib.Path(pptx_path)
        records["output_dir"] = pathlib.Path(output_dir)
        records["timeout_seconds"] = timeout_seconds
        assert not kwargs.get("include_hidden_slides", False)
        pdf_path = pathlib.Path(output_dir) / (pathlib.Path(pptx_path).stem + ".pdf")
        pdf_path.write_bytes(b"%PDF-1.4 shared helper")
        return pdf_path

    monkeypatch.setattr(conversion, "convert_pptx_to_pdf", fake_convert)
    if hasattr(files, "convert_pptx_to_pdf"):
        monkeypatch.setattr(files, "convert_pptx_to_pdf", fake_convert)

    cache_path = await files._convert_pptx_to_pdf(pptx)

    assert cache_path is not None
    assert cache_path.read_bytes() == b"%PDF-1.4 shared helper"
    assert records["source"] == pptx
    assert records["timeout_seconds"] == 60
    assert records["timeout_seconds"] == (
        conversion.DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS
    )
    assert pathlib.Path(records["output_dir"]).parent == cache_path.parent


def _upload_pptx(client, auth_headers) -> str:
    upload = client.post(
        "/api/files/upload",
        files={"file": ("deck.pptx", b"PK\x03\x04 synthetic upload", PPTX_MEDIA_TYPE)},
        data={"task_type": "general"},
        headers=auth_headers,
    )
    assert upload.status_code == 200, upload.text
    return upload.json()["file_id"]


def _install_endpoint_fake_process(monkeypatch, behaviour) -> list[dict]:
    """Patch the conversion subprocess the preview endpoint reaches."""

    def factory(calls):
        def write_pdf():
            outdir = _outdir_from_arguments(calls[0]["arguments"])
            behaviour(outdir)

        return _FakeProcess(write_pdf=write_pdf)

    return _install_fake_process(monkeypatch, factory)


def test_preview_pdf_endpoint_serves_inline_application_pdf(
    client, auth_headers, temp_uploads_dir, monkeypatch
):
    file_id = _upload_pptx(client, auth_headers)

    def write_pdf(outdir):
        (outdir / "deck.pdf").write_bytes(b"%PDF-1.4 endpoint preview")

    _install_endpoint_fake_process(monkeypatch, write_pdf)

    response = client.get(f"/api/files/preview-pdf/{file_id}", headers=auth_headers)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.headers["content-disposition"] == "inline"
    assert response.content == b"%PDF-1.4 endpoint preview"


def test_preview_pdf_endpoint_rejects_non_pptx(client, auth_headers, temp_uploads_dir):
    upload = client.post(
        "/api/files/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
        data={"task_type": "general"},
        headers=auth_headers,
    )
    assert upload.status_code == 200, upload.text
    file_id = upload.json()["file_id"]

    response = client.get(f"/api/files/preview-pdf/{file_id}", headers=auth_headers)

    assert response.status_code == 400
    assert "pptx" in response.json()["detail"]


def test_preview_pdf_endpoint_returns_404_for_an_unknown_file(
    client, auth_headers, temp_uploads_dir
):
    response = client.get(
        "/api/files/preview-pdf/00000000-0000-4000-8000-000000000000",
        headers=auth_headers,
    )

    assert response.status_code == 404


def test_preview_pdf_endpoint_falls_back_with_503_when_libreoffice_is_missing(
    client, auth_headers, temp_uploads_dir, monkeypatch
):
    file_id = _upload_pptx(client, auth_headers)

    async def missing_binary(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'soffice'")

    monkeypatch.setattr(files.asyncio, "create_subprocess_exec", missing_binary)

    response = client.get(f"/api/files/preview-pdf/{file_id}", headers=auth_headers)

    assert response.status_code == 503
    assert "LibreOffice" in response.json()["detail"]
