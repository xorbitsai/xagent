"""Render the slides of a workspace ``.pptx`` into one PNG per slide.

The module shares the managed PDF cache with the web preview path and converts
cache misses through :mod:`xagent.core.tools.core.pptx_conversion`, rasterises only
the selected PDF pages with PyMuPDF, and registers each produced PNG as a
durable workspace file so a later turn can show it or pass its ``file_id`` to a
vision model.

Bounds are part of the tool contract, not caller advice:

* at most :data:`MAX_RENDERED_PAGES` PNG pages per call;
* at most :data:`MAX_PAGE_PIXELS` pixels per PNG, decided from the PDF page
  rectangle before a pixmap is allocated;
* one LibreOffice conversion with the shared
  :data:`~xagent.core.tools.core.pptx_conversion.DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS`
  timeout.

LibreOffice converts and caches the whole deck, including hidden slides, so PDF page
numbers match source slide numbers; a PDF whose page count differs from the deck's
slide count is refused instead of returning shifted pages. Only the PNG output is
page-bounded.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from defusedxml.ElementTree import fromstring

from ....config import get_max_upload_size_bytes
from ...file_ref import build_workspace_file_ref
from ...workspace import ExternalUploadSizeLimitError, TaskWorkspace
from ..artifacts import build_inline_artifact
from .pptx_conversion import (
    DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
    PptxConversionFailedError,
    PptxConversionTimeoutError,
    PptxConverterUnavailableError,
    convert_pptx_to_pdf,
    pptx_pdf_cache_path,
)

MAX_RENDERED_PAGES = 20
MAX_PAGE_PIXELS = 4_000_000
DEFAULT_RENDER_SCALE = 2.0
PNG_MIME_TYPE = "image/png"
RENDER_OUTPUT_DIR_PREFIX = "pptx_render_"
_PRESENTATION_NAMESPACES = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main"
}
# A larger presentation part fails to parse instead of being expanded in memory.
_MAX_PRESENTATION_PART_BYTES = 8 * 1024 * 1024


class PptxRenderError(ValueError):
    """Base class for render failures that map to a tool error result."""

    error_code = "render_error"


class PptxRenderArgumentError(PptxRenderError):
    error_code = "invalid_argument"


class PptxRenderInputError(PptxRenderError):
    error_code = "input_not_found"


class PptxRenderNotAFileError(PptxRenderError):
    error_code = "input_not_a_file"


class PptxRenderUnsupportedInputError(PptxRenderError):
    error_code = "unsupported_format"


class PptxRenderOutsideWorkspaceError(PptxRenderError):
    error_code = "input_outside_workspace"


class PptxRenderEmptyPresentationError(PptxRenderError):
    error_code = "empty_presentation"


class PptxRenderPageRangeError(PptxRenderError):
    error_code = "page_range_invalid"


class PptxRenderPdfError(PptxRenderError):
    error_code = "pdf_unreadable"


class PptxRenderRasterError(PptxRenderError):
    error_code = "raster_failed"


@dataclass(frozen=True)
class PageSelection:
    """Resolved 1-based inclusive page range for one render call."""

    page_numbers: tuple[int, ...]
    truncated: bool
    total_pages: int


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    path: Path
    width: int
    height: int


def page_file_name(page_number: int) -> str:
    """Return the deterministic output filename for a 1-based page number."""
    return f"slide_{page_number:03d}.png"


def resolve_page_selection(
    total_pages: int,
    start_page: int | None = None,
    end_page: int | None = None,
) -> PageSelection:
    """Resolve the page range against the deck size.

    With no end bound, at most ``MAX_RENDERED_PAGES`` pages from the start
    are selected and ``truncated`` says the page bound was applied. An explicit
    range is never silently rewritten: a range over the bound or outside the
    deck raises instead of selecting different pages.
    """
    if total_pages < 1:
        raise PptxRenderEmptyPresentationError(
            "The presentation contains no slides to render."
        )
    start, end = start_page, end_page
    if (start is not None and start < 1) or (end is not None and end < 1):
        raise PptxRenderArgumentError("Page numbers must be at least 1.")
    if start is not None and end is not None and start > end:
        raise PptxRenderArgumentError("start_page must not be greater than end_page.")
    resolved_start = 1 if start is None else start
    resolved_end = (
        min(total_pages, resolved_start + MAX_RENDERED_PAGES - 1)
        if end is None
        else end
    )
    if resolved_start > total_pages:
        raise PptxRenderPageRangeError(
            f"start_page ({resolved_start}) is beyond the last page "
            f"({total_pages}) of this presentation."
        )
    if resolved_end > total_pages:
        raise PptxRenderPageRangeError(
            f"end_page ({resolved_end}) is beyond the last page "
            f"({total_pages}) of this presentation."
        )
    requested = resolved_end - resolved_start + 1
    if requested > MAX_RENDERED_PAGES:
        raise PptxRenderArgumentError(
            f"The requested range covers {requested} slides, but at most "
            f"{MAX_RENDERED_PAGES} pages can be rendered per call. "
            "Narrow start_page/end_page."
        )
    return PageSelection(
        page_numbers=tuple(range(resolved_start, resolved_end + 1)),
        truncated=end is None and resolved_end < total_pages,
        total_pages=total_pages,
    )


def _page_scale(page: Any) -> float:
    import fitz

    width, height = float(page.rect.width), float(page.rect.height)
    area = width * height
    if width <= 0 or height <= 0 or not math.isfinite(area) or area <= 0:
        raise PptxRenderRasterError("The PDF page has invalid or oversized dimensions.")
    scale = min(DEFAULT_RENDER_SCALE, math.sqrt(MAX_PAGE_PIXELS / area))
    while scale > 0:
        extent = (page.rect * fitz.Matrix(scale, scale)).irect
        if extent.width <= 0 or extent.height <= 0:
            break
        if extent.width * extent.height <= MAX_PAGE_PIXELS:
            return scale
        scale *= 0.99
    raise PptxRenderRasterError(
        "The PDF page is too large to render within the pixel limit."
    )


def _open_pdf(pdf_path: Path) -> Any:
    import fitz

    try:
        return fitz.open(str(pdf_path))
    except Exception as exc:
        raise PptxRenderPdfError(
            f"The converted PDF {pdf_path.name} could not be opened: {exc}"
        ) from exc


def render_pdf_pages_to_pngs(
    document: Any,
    page_numbers: list[int] | tuple[int, ...],
    output_dir: Path,
    *,
    output_fd: int,
    cancelled: threading.Event,
) -> list[RenderedPage]:
    """Rasterize selected pages through a held directory handle."""
    import fitz

    try:
        rendered = []
        for number in page_numbers:
            if cancelled.is_set():
                raise asyncio.CancelledError
            page = document[number - 1]
            effective_scale = _page_scale(page)
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(effective_scale, effective_scale), alpha=False
            )
            target = output_dir / page_file_name(number)
            descriptor = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=output_fd,
            )
            with os.fdopen(descriptor, "wb") as image_file:
                pixmap.pil_save(image_file, format="PNG")
            rendered.append(RenderedPage(number, target, pixmap.width, pixmap.height))
        return rendered
    except PptxRenderError:
        raise
    except Exception as exc:
        raise PptxRenderRasterError(
            f"Could not render the slide images: {exc}"
        ) from exc


def _resolve_pptx_input(workspace: TaskWorkspace, raw_selector: str) -> Path:
    """Resolve an authorized workspace input that must be an existing .pptx."""
    selector = str(raw_selector).strip()
    try:
        path = Path(
            workspace.resolve_path_with_search(selector, detached_file_ids=True)
        )
    except FileNotFoundError as exc:
        raise PptxRenderInputError(
            f"The presentation {selector!r} was not found in this task workspace."
        ) from exc
    except ValueError as exc:
        raise PptxRenderOutsideWorkspaceError(
            f"The path {selector!r} is outside the task workspace or its "
            "authorized upload directories."
        ) from exc
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PptxRenderInputError(
            f"The presentation {selector!r} could not be resolved to a file."
        ) from exc
    if not path.is_file():
        raise PptxRenderNotAFileError(
            f"The presentation {selector!r} is not a regular file."
        )
    if path.suffix.lower() != ".pptx":
        raise PptxRenderUnsupportedInputError(
            f"The presentation {selector!r} must be a .pptx file."
        )
    return path


def _count_slides(pptx_path: Path) -> int:
    """Count source slides, hidden ones included, from ``ppt/presentation.xml``."""
    try:
        with zipfile.ZipFile(pptx_path) as package:
            with package.open("ppt/presentation.xml") as part:
                root = fromstring(part.read(_MAX_PRESENTATION_PART_BYTES))
    except Exception as exc:
        raise PptxRenderUnsupportedInputError(
            "The presentation is not a readable .pptx file."
        ) from exc
    return len(root.findall("p:sldIdLst/p:sldId", _PRESENTATION_NAMESPACES))


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "error_code": code,
        "error": message,
    }
    result.update(extra)
    return result


def _selection_payload(selection: PageSelection) -> dict[str, Any]:
    first, last = selection.page_numbers[0], selection.page_numbers[-1]
    total = selection.total_pages
    return {
        "total_pages": total,
        "rendered_pages": len(selection.page_numbers),
        "page_numbers": list(selection.page_numbers),
        "truncated": selection.truncated,
        "unrendered_pages": total - len(selection.page_numbers),
        "unrendered_page_ranges": [
            [start, end]
            for start, end in ((1, first - 1), (last + 1, total))
            if start <= end
        ],
    }


def _build_pages_payload(
    workspace: TaskWorkspace,
    rendered: list[RenderedPage],
    persisted: dict[int, str],
) -> list[dict[str, Any]]:
    """Build FileRefs and inline artifacts; runs in a worker thread."""
    pages: list[dict[str, Any]] = []
    for page in rendered:
        file_id = persisted[page.page_number]
        file_ref = build_workspace_file_ref(
            workspace=workspace,
            file_path=page.path,
            file_id=file_id,
            mime_type=PNG_MIME_TYPE,
        )
        pages.append(
            {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "file_id": file_id,
                "file_ref": file_ref,
            }
        )
    return pages


def _render_and_register(
    pdf_path: Path,
    slide_count: int,
    workspace: TaskWorkspace,
    start_page: int | None,
    end_page: int | None,
    cancelled: threading.Event,
) -> dict[str, Any]:
    document = _open_pdf(pdf_path)
    try:
        selection = resolve_page_selection(slide_count, start_page, end_page)
        if document.page_count != slide_count:
            raise PptxConversionFailedError(
                f"LibreOffice exported {document.page_count} PDF pages for "
                f"{slide_count} slides, so pages would not match source slides."
            )
        directory_name = RENDER_OUTPUT_DIR_PREFIX + uuid.uuid4().hex
        output_dir = workspace.output_dir / directory_name
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent_fd = os.open(workspace.output_dir, flags)
        try:
            os.mkdir(directory_name, mode=0o700, dir_fd=parent_fd)
            directory_fd = None
            registration_started = False
            try:
                directory_fd = os.open(directory_name, flags, dir_fd=parent_fd)
                rendered = render_pdf_pages_to_pngs(
                    document,
                    selection.page_numbers,
                    output_dir,
                    output_fd=directory_fd,
                    cancelled=cancelled,
                )
                if cancelled.is_set():
                    raise asyncio.CancelledError
                registration_started = True
                persisted: dict[int, str] = {}
                try:
                    for page in rendered:
                        persisted[page.page_number] = workspace.register_delivery_file(
                            str(page.path)
                        )
                    pages = _build_pages_payload(workspace, rendered, persisted)
                except Exception as exc:
                    return _error(
                        "registration_failed",
                        f"Could not register slide images: {exc}",
                        registered_pages=persisted,
                        output_dir=str(output_dir),
                    )
                return {
                    "success": True,
                    "output_dir": str(output_dir),
                    **_selection_payload(selection),
                    "pages": pages,
                    "artifacts": [
                        build_inline_artifact(page["file_ref"]) for page in pages
                    ],
                }
            finally:
                try:
                    if not registration_started:
                        if directory_fd is not None:
                            for number in selection.page_numbers:
                                try:
                                    os.unlink(
                                        page_file_name(number), dir_fd=directory_fd
                                    )
                                except FileNotFoundError:
                                    pass
                        # rmdir never follows a replacement symlink and only removes an empty directory.
                        try:
                            os.rmdir(directory_name, dir_fd=parent_fd)
                        except (FileNotFoundError, NotADirectoryError):
                            pass
                finally:
                    if directory_fd is not None:
                        os.close(directory_fd)
        finally:
            os.close(parent_fd)

    finally:
        document.close()


def _content_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish_hidden_pdf(converted: Path, cache_path: Path) -> Path:
    """Publish an export and remove its owner's exports of older content."""
    converted.replace(cache_path)
    owner = cache_path.name.split(".", 1)[0]
    for stale in cache_path.parent.glob(f"{owner}.*.hidden.preview.pdf"):
        if stale != cache_path:
            stale.unlink(missing_ok=True)
    return cache_path


async def _cached_render_pdf(
    staged_input: Path, source: Path, workspace: TaskWorkspace
) -> Path | None:
    """Return the hidden-slide PDF, or ``None`` while its publication is busy."""
    from ....web.services.uploaded_file_cleanup_publication import (
        guard_preview_publication,
        preview_cache_hit,
    )

    file_id = await asyncio.to_thread(workspace.get_file_id_from_path, str(source))
    digest = await asyncio.to_thread(_content_hash, staged_input)
    cache_path = await asyncio.to_thread(
        pptx_pdf_cache_path, source, file_id, content_hash=digest
    )
    cached = await asyncio.to_thread(preview_cache_hit, source, cache_path, file_id)
    if cached is not None:
        return cached

    # As in the preview path, the guard spans the temporary directory,
    # conversion and publication, and reports a busy cleanup lock as None.
    @guard_preview_publication
    async def convert_and_publish(
        path: Path, file_id: str | None = None
    ) -> Path | None:
        with tempfile.TemporaryDirectory(
            dir=cache_path.parent, prefix=f".{file_id or cache_path.stem}.preview-"
        ) as directory:
            converted = await convert_pptx_to_pdf(
                path,
                Path(directory),
                timeout_seconds=DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
                include_hidden_slides=True,
            )
            return await asyncio.to_thread(_publish_hidden_pdf, converted, cache_path)

    return await convert_and_publish(staged_input, file_id)


async def render_pptx_to_images(
    pptx_path: str,
    workspace: TaskWorkspace | None,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
) -> dict[str, Any]:
    """Render selected source slides, including hidden ones, into registered PNGs.

    Registration failures preserve the images. Failures before registration
    remove this call's output. Cancellation during a conversion waits for that
    conversion under the preview publication guard, as previews do. Later
    cancellation propagates without waiting for workers; the worker checks it
    between pages and before registration. Registration already in progress may
    finish and its images are preserved.
    """
    try:
        import fitz  # noqa: F401 - optional dependency checked at invocation
    except ImportError:
        return _error(
            "dependency_unavailable", "PyMuPDF is required to render slide images."
        )
    if workspace is None:
        return _error(
            "workspace_unavailable", "Rendering slides requires a task workspace."
        )
    if workspace.current_task_id is None:
        return _error(
            "durable_registration_unavailable",
            "Rendering slides requires a persistent task owner.",
        )
    staged_input = None
    cancelled = threading.Event()
    try:
        source = await asyncio.to_thread(_resolve_pptx_input, workspace, pptx_path)
        staged_input = await asyncio.to_thread(
            workspace.stage_resolved_path_for_external_upload,
            source,
            pptx_path,
            max_bytes=get_max_upload_size_bytes(),
            host_private=True,
        )
        slide_count = await asyncio.to_thread(_count_slides, staged_input)
        pdf_path = await _cached_render_pdf(staged_input, source, workspace)
        if pdf_path is None:
            return _error(
                "pdf_conversion_unavailable",
                "The presentation is busy with file cleanup or another conversion; "
                "try again.",
            )
        return await asyncio.to_thread(
            _render_and_register,
            pdf_path,
            slide_count,
            workspace,
            start_page,
            end_page,
            cancelled,
        )
    except asyncio.CancelledError:
        cancelled.set()
        raise
    except ExternalUploadSizeLimitError as exc:
        return _error("input_too_large", str(exc))
    except PptxRenderError as exc:
        return _error(exc.error_code, str(exc))
    except PptxConverterUnavailableError as exc:
        return _error("pdf_conversion_unavailable", str(exc))
    except PptxConversionTimeoutError as exc:
        return _error("pdf_conversion_timeout", str(exc))
    except PptxConversionFailedError as exc:
        return _error("pdf_conversion_failed", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return _error("input_not_found", str(exc))
    except OSError as exc:
        return _error("output_unavailable", str(exc))
    finally:
        if staged_input is not None:
            await asyncio.to_thread(
                workspace.discard_staged_external_upload, staged_input
            )
