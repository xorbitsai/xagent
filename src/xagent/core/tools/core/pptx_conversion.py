"""Shared LibreOffice conversion from ``.pptx`` to a one-off PDF.

Both the ``.pptx`` preview cache and the slide render tool need the same
LibreOffice invocation contract, so the process handling lives here:

* a parameter list (never a shell command) with ``--headless`` and
  ``--convert-to pdf``;
* one LibreOffice user profile per call, so two concurrent conversions cannot
  fight over the same configuration lock;
* a bounded wait whose timeout terminates the isolated process group;
* at most :data:`MAX_CONCURRENT_CONVERSIONS` LibreOffice processes at once in this
  backend process, shared by previews and slide rendering; later calls wait
  for a slot before their timeout starts;
* cleanup of the profile on every exit, and of the process group on failure
  or cancellation.

Callers publish complete PDFs into the shared preview cache namespace.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Final

from ....config import get_storage_root

DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS: Final[float] = 60.0
DEFAULT_SOFFICE_BINARY: Final[str] = "soffice"
MAX_CONCURRENT_CONVERSIONS: Final[int] = 2
_conversion_slots = asyncio.Semaphore(MAX_CONCURRENT_CONVERSIONS)


class PptxConversionError(RuntimeError):
    """Base class for a ``.pptx`` -> PDF conversion failure."""


class PptxConverterUnavailableError(PptxConversionError):
    """LibreOffice is not installed or not reachable through PATH."""


class PptxConversionTimeoutError(PptxConversionError):
    """LibreOffice did not finish inside the configured timeout."""


class PptxConversionFailedError(PptxConversionError):
    """LibreOffice ran but did not produce a usable PDF."""


def build_soffice_arguments(
    pptx_path: Path,
    output_dir: Path,
    profile_dir: Path,
    *,
    include_hidden_slides: bool = False,
) -> list[str]:
    """Return the argv for one headless ``.pptx`` -> PDF conversion.

    ``profile_dir`` becomes ``-env:UserInstallation=file://...``: a writable
    LibreOffice configuration directory that is separate from the PDF output
    directory. It is what keeps concurrent conversions from serialising on (or
    corrupting) a shared profile lock.
    """
    pdf_filter = (
        'pdf:impress_pdf_Export:{"ExportHiddenSlides":{"type":"boolean","value":"true"}}'
        if include_hidden_slides
        else "pdf"
    )
    return [
        DEFAULT_SOFFICE_BINARY,
        "--headless",
        f"-env:UserInstallation={profile_dir.as_uri()}",
        "--convert-to",
        pdf_filter,
        "--outdir",
        str(output_dir),
        str(pptx_path),
    ]


def expected_pdf_path(pptx_path: Path, output_dir: Path) -> Path:
    """Return the PDF path LibreOffice is expected to create for a source file."""
    return Path(output_dir) / f"{Path(pptx_path).stem}.pdf"


PROCESS_REAP_TIMEOUT_SECONDS: Final[float] = 1.0


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Terminate the isolated group and bound reaping, even with inherited pipes."""
    _kill_process_group(process)
    try:
        await asyncio.wait_for(process.wait(), PROCESS_REAP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        pass


async def convert_pptx_to_pdf(
    pptx_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
    include_hidden_slides: bool = False,
) -> Path:
    """Convert ``pptx_path`` into ``output_dir`` and return the produced PDF.

    The caller supplies the output directory and decides whether to keep or
    publish the PDF. The call first waits for one of the process-wide
    conversion slots. A unique LibreOffice profile directory is created for this
    call and removed again on every exit path. Hidden slides are included only
    when explicitly requested; previews retain the default export behavior.

    Raises:
        PptxConverterUnavailableError: LibreOffice cannot be started.
        PptxConversionTimeoutError: the conversion exceeded ``timeout_seconds``;
            process-group termination was attempted and reaping was bounded.
        PptxConversionFailedError: a non-zero exit or no expected PDF.
    """
    async with _conversion_slots:
        pptx_path = Path(pptx_path)
        output_dir = Path(output_dir)
        profile_dir = Path(tempfile.mkdtemp(prefix="xagent-pptx-profile-"))
        process: asyncio.subprocess.Process | None = None
        try:
            arguments = build_soffice_arguments(
                pptx_path,
                output_dir,
                profile_dir,
                include_hidden_slides=include_hidden_slides,
            )
            stderr_path = profile_dir / "stderr.log"
            with stderr_path.open("wb") as stderr_file:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *arguments,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=stderr_file,
                        start_new_session=True,
                    )
                except FileNotFoundError as exc:
                    raise PptxConverterUnavailableError(
                        f"LibreOffice is not installed or not on PATH ({DEFAULT_SOFFICE_BINARY!r}); "
                        f"cannot convert {pptx_path.name} to PDF."
                    ) from exc
                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
                except asyncio.TimeoutError as exc:
                    raise PptxConversionTimeoutError(
                        f"LibreOffice conversion of {pptx_path.name} timed out after "
                        f"{timeout_seconds:g} seconds."
                    ) from exc
            if process.returncode != 0:
                detail = await asyncio.to_thread(_stderr_summary, stderr_path)
                raise PptxConversionFailedError(
                    f"LibreOffice exited {process.returncode} while converting "
                    f"{pptx_path.name}: {detail or 'no stderr output'}"
                )
            pdf_path = expected_pdf_path(pptx_path, output_dir)
            if not await asyncio.to_thread(pdf_path.is_file):
                raise PptxConversionFailedError(
                    f"LibreOffice finished but did not produce the expected PDF "
                    f"{pdf_path.name} for {pptx_path.name}."
                )
            return pdf_path
        except (Exception, asyncio.CancelledError):
            if process is not None:
                await _terminate_process(process)
            raise
        finally:
            await asyncio.to_thread(shutil.rmtree, profile_dir, ignore_errors=True)


def _stderr_summary(path: Path) -> str:
    with path.open("rb") as stream:
        return stream.read(800).decode("utf-8", "replace").strip()[:200]


def pptx_pdf_cache_path(
    source: Path, file_id: str | None = None, *, content_hash: str | None = None
) -> Path:
    """Use the preview cache namespace for default and hidden-slide exports."""
    directory = get_storage_root() / "pptx_pdf_cache"
    directory.mkdir(parents=True, exist_ok=True)
    owner = file_id or hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:24]
    variant = f".{content_hash}.hidden" if content_hash is not None else ""
    return directory / f"{owner}{variant}.preview.pdf"
