import io
import json
import logging
import os
from copy import deepcopy
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.presentation import Presentation as PresentationType

from ....core.tools.core.file_analysis import iter_pptx_shapes
from .utils import setup_proxy_env, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("powerpoint-mcp")

setup_proxy_env()

mcp = FastMCP("powerpoint-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30
_BINARY_TIMEOUT_SECONDS = 60

_POWERPOINT_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)

# Microsoft Graph exposes a PowerPoint (.pptx) file only as a driveItem
# content blob -- there is no structured "PowerPoint API" resource (no
# per-slide/per-shape Graph endpoints). Every tool here downloads the whole
# file, edits it in memory with python-pptx, and re-uploads the whole file.
# Kept entirely in memory (io.BytesIO, never written to local disk).
#
# Above this, the simple content PUT gives way to the resumable
# upload-session API instead of failing outright -- matching onedrive.py's
# own use of the same threshold to pick between the two, since a deck with
# embedded media routinely exceeds this.
_SIMPLE_UPLOAD_MAX_BYTES = 4_000_000

# A deliberately arbitrary product ceiling (independent of any Graph-side
# limit) on the *edited* presentation's serialized size -- unlike
# onedrive.py's file upload, which streams a chunk at a time straight from
# disk, this module's whole download-edit-reupload cycle already holds the
# serialized bytes in memory at least twice over (the downloaded bytes, the
# python-pptx object graph, and the re-saved buffer), so an unbounded size
# here is an unbounded memory commitment, not just a slow upload.
_MAX_PRESENTATION_BYTES = 200_000_000

# Graph requires every upload-session fragment but the last to be a
# multiple of 320 KiB, and documents a 60 MiB hard maximum per PUT; 5 MiB
# is also in its recommended 5-10 MiB "best practice" range, and matches
# onedrive.py's own chunk size for the same API.
_UPLOAD_SESSION_CHUNK_BYTES = 5 * 1024 * 1024

# python-pptx has no public API for adding a slide at other than the layout
# picked, or for removing one at all -- see _delete_slide's own docstring
# for the latter. Layout 1 ("Title and Content") is the conventional
# default for "just add a slide" the same way Word's own UI defaults a new
# document to the Normal style.
_DEFAULT_SLIDE_LAYOUT_INDEX = 1


class _GraphRequestError(RuntimeError):
    """Graph HTTP failure that retains its status without response parsing."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _graph_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    token = os.environ.get("AUTH_TOKEN")
    if not token:
        raise ValueError("AUTH_TOKEN environment variable is missing")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _graph_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    data: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
    raw: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    # Graph's /content GET 302s to a preauthenticated download URL (a
    # bearer secret in its own query string), which requests follows
    # automatically -- so *any* failure once that redirect has happened,
    # not just an HTTP error status, can carry that signed URL: a
    # connection-layer failure (timeout, reset, TLS error) raises a
    # RequestException whose own str() embeds it exactly like HTTPError's
    # does. Both branches below build their own message from method/path
    # (this function's own input, never the possibly-redirected
    # response/request URL) instead of str(exc), and raise "from None"
    # rather than chaining the original exception as __cause__ -- chaining
    # would still let a future traceback/log/APM capture surface that URL
    # even with a clean top-level message, matching _create_only_upload's
    # identical guard on its own upload-session URL.
    try:
        response = requests.request(
            method=method,
            url=f"{GRAPH_BASE_URL}{path}",
            headers=_graph_headers(extra_headers),
            params=params,
            json=body,
            data=data,
            timeout=timeout,
        )
    except requests.RequestException:
        raise RuntimeError(f"Graph request failed: {method} {path}") from None

    try:
        response.raise_for_status()
    except requests.HTTPError:
        response_text = response.text.strip()
        message = f"Graph {method} {path} failed with HTTP {response.status_code}"
        if response_text:
            message = f"{message} - {response_text}"
        raise _GraphRequestError(message, status_code=response.status_code) from None

    if raw:
        return response.content
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _site_segment(site_id: str) -> str:
    """Percent-encode a caller-supplied Graph site identifier for
    interpolation into a URL path segment.

    A Graph site id is one of: the literal "root", a composite id
    ("hostname,spSiteId,spWebId"), or a "hostname:/server-relative-path"
    form. ':' and '/' stay unescaped because they're structural to the
    third shape, while a '.'/'..' segment is rejected outright -- standard
    HTTP client URL normalization could otherwise walk the request off
    "/sites/{id}/..." and onto a different Graph endpoint under the same
    OAuth token.
    """
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError("site_id is required")
    value = site_id.strip()
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"site_id must not contain '.' or '..' segments: {site_id!r}")
    return quote(value, safe=":/,")


def _normalize_relative_path(path: str) -> str:
    """Normalize a drive-relative file path for a root:/{path}: request URL,
    rejecting '.'/'..' segments, a trailing folder separator, and a filename
    ending in a period (Graph/SharePoint's backing storage can silently
    normalize a trailing dot away, so "Deck.pptx." could silently resolve
    to a real, different "Deck.pptx")."""
    value = path.strip().strip("/")
    if not value:
        raise ValueError("file_path is required")
    if path.strip().endswith("/"):
        raise ValueError(
            "file_path must include a filename, not end with a folder separator"
        )
    if "\\" in value:
        raise ValueError("file_path must use '/' separators and must not contain '\\'")
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"file_path must not contain '.' or '..' segments: {path!r}")
    if value.rsplit("/", 1)[-1].endswith("."):
        raise ValueError(f"file_path filename must not end with a period: {path!r}")
    return value


def _item_path(file_path: str, site_id: str | None, drive_id: str | None) -> str:
    normalized = _normalize_relative_path(file_path)
    if site_id:
        site_segment = _site_segment(site_id)
        # A "hostname:/server-relative-path" site_id needs a second,
        # closing colon before appending another resource segment, to
        # transition Graph's parser back from path-based site addressing to
        # resource-based addressing (confirmed against Graph's own
        # documented example: ".../sites/contoso.sharepoint.com:/teams/hr:
        # /drive") -- the other two site_id shapes ("root" and the
        # composite "hostname,spSiteId,spWebId" form) never contain ':' and
        # are unaffected.
        site_suffix = ":" if ":" in site_segment else ""
        drive_base = (
            f"/sites/{site_segment}{site_suffix}/drives/{url_path_id(drive_id, 'drive_id')}"
            if drive_id
            else f"/sites/{site_segment}{site_suffix}/drive"
        )
    elif drive_id:
        drive_base = f"/drives/{url_path_id(drive_id, 'drive_id')}"
    else:
        drive_base = "/me/drive"
    return f"{drive_base}/root:/{quote(normalized, safe='/')}:"


def _content_path(file_path: str, site_id: str | None, drive_id: str | None) -> str:
    return f"{_item_path(file_path, site_id, drive_id)}/content"


def _download_presentation(
    file_path: str, site_id: str | None, drive_id: str | None
) -> PresentationType:
    content = _graph_request(
        "GET", _content_path(file_path, site_id, drive_id), raw=True
    )
    try:
        return Presentation(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(
            f"Could not open {file_path!r} as a PowerPoint presentation -- it may "
            "not be a valid .pptx file"
        ) from exc


def _upload_presentation(
    presentation: PresentationType,
    file_path: str,
    site_id: str | None,
    drive_id: str | None,
) -> dict[str, Any]:
    buffer = io.BytesIO()
    presentation.save(buffer)
    content = buffer.getvalue()
    if len(content) > _MAX_PRESENTATION_BYTES:
        raise ValueError(
            f"The updated presentation is {len(content)} bytes, over the "
            f"{_MAX_PRESENTATION_BYTES // 1_000_000} MB limit this tool currently "
            "supports"
        )
    if len(content) > _SIMPLE_UPLOAD_MAX_BYTES:
        return _upload_presentation_session(content, file_path, site_id, drive_id)
    result = _graph_request(
        "PUT",
        _content_path(file_path, site_id, drive_id),
        extra_headers={"Content-Type": _POWERPOINT_MIME_TYPE},
        data=content,
        timeout=_BINARY_TIMEOUT_SECONDS,
    )
    if not isinstance(result, dict) or not result.get("id"):
        raise RuntimeError("Graph did not confirm the presentation upload completed")
    safe_item = dict(result)
    safe_item.pop("@microsoft.graph.downloadUrl", None)
    return safe_item


def _upload_presentation_session(
    content: bytes, file_path: str, site_id: str | None, drive_id: str | None
) -> dict[str, Any]:
    """Upload content over _SIMPLE_UPLOAD_MAX_BYTES via Graph's
    upload-session API instead of the simple content PUT, in
    _UPLOAD_SESSION_CHUNK_BYTES-aligned fragments.

    Unlike onedrive.py's own use of this API, content here is always
    already fully resident in memory (python-pptx has no streaming save),
    so this has no need for that module's disk-streaming/resume-on-
    reconnect machinery -- a fragment failure simply fails the whole
    upload, matching this module's existing _create_only_upload, which
    takes the same no-retry stance on the same API for its own (always
    single-fragment) use of it.
    """
    session = _graph_request(
        "POST",
        f"{_item_path(file_path, site_id, drive_id)}/createUploadSession",
        body={"item": {"@microsoft.graph.conflictBehavior": "replace"}},
    )
    upload_url = session.get("uploadUrl") if isinstance(session, dict) else None
    if not isinstance(upload_url, str) or not upload_url:
        raise RuntimeError("Graph did not return an upload session URL")

    total = len(content)
    result: Any = None
    # The upload-session URL is itself pre-authenticated (a token in its
    # own query string) -- as with _create_only_upload's identical use of
    # this API, this goes through a plain requests.Session() rather than
    # _graph_request (which always attaches an Authorization header, which
    # Graph's docs warn can itself cause a 401 here), and every exception
    # is re-raised "from None" rather than "from exc" so a future
    # traceback/log/APM capture can never surface this URL as __cause__.
    with requests.Session() as http:
        for start in range(0, total, _UPLOAD_SESSION_CHUNK_BYTES):
            end = min(start + _UPLOAD_SESSION_CHUNK_BYTES, total)
            try:
                response = http.put(
                    upload_url,
                    data=content[start:end],
                    headers={
                        "Content-Range": f"bytes {start}-{end - 1}/{total}",
                        "Content-Type": _POWERPOINT_MIME_TYPE,
                    },
                    timeout=_BINARY_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
            except requests.HTTPError:
                raise _GraphRequestError(
                    "PowerPoint presentation upload failed with HTTP "
                    f"{response.status_code}",
                    status_code=response.status_code,
                ) from None
            except requests.RequestException:
                raise RuntimeError("PowerPoint presentation upload failed") from None
            if end == total:
                try:
                    result = response.json()
                except ValueError:
                    result = None

    if not isinstance(result, dict) or not result.get("id"):
        raise RuntimeError("Graph did not confirm the presentation upload completed")
    safe_item = dict(result)
    safe_item.pop("@microsoft.graph.downloadUrl", None)
    return safe_item


def _create_only_upload(
    content: bytes, file_path: str, site_id: str | None, drive_id: str | None
) -> dict[str, Any]:
    """Upload content, failing instead of silently overwriting if a file
    already exists at file_path.

    A separate exists-check GET followed by a plain _upload_presentation PUT
    would still race: Graph's own "Upload small files" reference documents
    no If-None-Match or other conditional-write option on the simple
    content PUT, so two concurrent callers can both pass the check and the
    second would silently clobber the first. The upload-session API is
    documented to accept a conflictBehavior of "fail" on session creation,
    which Graph enforces atomically server-side (a 409 nameAlreadyExists if
    the target already exists) -- used here even though the content (a
    blank new presentation) is always small enough for a single-PUT
    session, specifically for that atomicity guarantee.
    """
    item_path = _item_path(file_path, site_id, drive_id)
    try:
        session = _graph_request(
            "POST",
            f"{item_path}/createUploadSession",
            body={"item": {"@microsoft.graph.conflictBehavior": "fail"}},
        )
    except _GraphRequestError as exc:
        if exc.status_code == 409:
            raise ValueError(
                f"{file_path!r} already exists; use the other powerpoint_* "
                "tools to edit it instead of recreating it"
            ) from exc
        raise
    upload_url = session.get("uploadUrl") if isinstance(session, dict) else None
    if not isinstance(upload_url, str) or not upload_url:
        raise RuntimeError("Graph did not return an upload session URL")

    # The upload-session URL is itself pre-authenticated (a token in its own
    # query string) -- Graph's createUploadSession docs warn that including
    # an Authorization header on this PUT can cause a 401 -- so this goes
    # through a plain requests.put, not _graph_request (which always
    # attaches one). Both requests.put's own exception (a connection failure
    # or timeout) and the HTTPError from raise_for_status() embed the full
    # request URL in their default str(), so neither is stringified into a
    # message below, and each is re-raised with "from None" rather than
    # "from exc" -- chaining the original would still attach it as
    # __cause__, which a future traceback/log/APM capture could surface --
    # matching onedrive.py's identical guard on the same hazard.
    try:
        response = requests.put(
            upload_url,
            data=content,
            headers={
                "Content-Range": f"bytes 0-{len(content) - 1}/{len(content)}",
                "Content-Type": _POWERPOINT_MIME_TYPE,
            },
            timeout=_BINARY_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.HTTPError:
        status_code = response.status_code
        if status_code == 409:
            raise ValueError(
                f"{file_path!r} already exists; use the other powerpoint_* "
                "tools to edit it instead of recreating it"
            ) from None
        raise _GraphRequestError(
            f"PowerPoint presentation upload failed with HTTP {status_code}",
            status_code=status_code,
        ) from None
    except requests.RequestException:
        raise RuntimeError("PowerPoint presentation upload failed") from None
    try:
        result = response.json()
    except ValueError:
        result = None
    if not isinstance(result, dict) or not result.get("id"):
        raise RuntimeError("Graph did not confirm the presentation upload completed")
    safe_item = dict(result)
    safe_item.pop("@microsoft.graph.downloadUrl", None)
    return safe_item


def _shape_text(shape: Any) -> str | None:
    return shape.text_frame.text if shape.has_text_frame else None


def _slide_texts(slide: Any) -> list[str]:
    """All shape text on slide, recursing into any group and pulling each
    table cell's text.

    slide.shapes alone only yields top-level shapes, and neither a group
    (it has no text frame of its own -- the text lives on the shapes
    nested inside it) nor a table (a GraphicFrame, whose has_text_frame is
    also always False -- the text lives on its individual cells) would
    otherwise contribute anything, so a plain has_text_frame filter over
    slide.shapes silently omits both.
    """
    texts: list[str] = []
    for shape in iter_pptx_shapes(slide.shapes):
        text = _shape_text(shape)
        if text:
            texts.append(text)
        elif getattr(shape, "has_table", False):
            for row in shape.table.rows:
                for cell in row.cells:
                    text = cell.text_frame.text
                    if text:
                        texts.append(text)
    return texts


def _slide_notes(slide: Any) -> str | None:
    """slide's speaker notes text, or None if it has no notes slide or no
    notes placeholder on it.

    Checked via has_notes_slide rather than a bare getattr/try -- python-
    pptx's own notes_slide property *creates* a notes slide on first access
    if one doesn't already exist, which would silently mutate a
    presentation this module never intends to write back for a read-only
    tool. notes_text_frame can still be None even when has_notes_slide is
    True -- per its own docstring, that happens if the notes placeholder
    was deleted from the notes slide (the notes-slide XML part survives,
    just without a body placeholder).
    """
    if not slide.has_notes_slide:
        return None
    notes_text_frame = slide.notes_slide.notes_text_frame
    if notes_text_frame is None:
        return None
    return notes_text_frame.text or None


def _select_body_placeholder(slide: Any) -> Any | None:
    """Find the first non-title placeholder on slide that can hold text.

    A placeholder's idx alone doesn't guarantee it has a text frame -- a
    picture, chart, or other content placeholder can sit at a lower idx
    than a real text placeholder (verified: python-pptx's own
    SlidePlaceholders iterates in idx order, with no guarantee idx order
    matches "text placeholders first"), so this skips non-text
    placeholders entirely rather than stopping at the first non-title one
    regardless of whether it can actually hold text.
    """
    return next(
        (
            ph
            for ph in slide.placeholders
            if ph.placeholder_format.idx != 0 and ph.has_text_frame
        ),
        None,
    )


def _require_int(value: Any, field_name: str) -> int:
    """Validate an integer-typed tool argument.

    FastMCP/Pydantic validates and coerces arguments against a tool's type
    hints for a real call over the MCP protocol, but that validation is
    bypassed by a caller that invokes this Python function directly (e.g. a
    test, or any other in-process caller) -- without this, a non-int index
    reaches python-pptx's own indexing/comparison and fails with a raw,
    less clear TypeError instead. bool is excluded even though it's a
    subclass of int in Python, since True/False are never valid indices.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _text_frame_has_dynamic_content(text_frame: Any) -> bool:
    """Whether text_frame holds a hyperlink or a dynamic field (e.g. an
    auto-updating slide number or date placeholder) that a full-frame text
    replacement would silently discard.

    python-pptx's TextFrame.text setter clears every existing paragraph
    and rebuilds the frame from a single new run per line (verified
    directly: assigning .text removes all <a:p> children) -- a run's
    <a:hlinkClick> and a paragraph-level <a:fld> element are both lost
    with no error or warning.
    """
    tx_body = text_frame._txBody
    return (
        tx_body.find(".//" + qn("a:hlinkClick")) is not None
        or tx_body.find(".//" + qn("a:fld")) is not None
    )


def _replace_text_frame_text(text_frame: Any, text: str) -> None:
    """Replace text_frame's text, preserving each existing paragraph's
    paragraph-level formatting (alignment, bullet/numbering, indent level)
    and its first run's character formatting (bold, italic, font, size,
    color) by position.

    TextFrame.text's own setter (python-pptx) clears every <a:p> and
    rebuilds one new paragraph per "\n"-separated line, each holding a
    single run with no <a:pPr>/<a:rPr> at all -- verified directly against
    its source (CT_TextBody.clear_content / CT_TextParagraph.append_text)
    -- so a plain assignment silently discards all of that. A paragraph
    added or removed by this call (the new text has more/fewer lines than
    the frame had paragraphs) has nothing to carry formatting over
    to/from, so it gets/loses it; every other paragraph keeps its look.
    """
    saved: list[tuple[Any, Any]] = []
    for paragraph in text_frame.paragraphs:
        p_pr = paragraph._p.find(qn("a:pPr"))
        first_run = paragraph.runs[0] if paragraph.runs else None
        r_pr = first_run._r.find(qn("a:rPr")) if first_run is not None else None
        saved.append(
            (
                deepcopy(p_pr) if p_pr is not None else None,
                deepcopy(r_pr) if r_pr is not None else None,
            )
        )

    text_frame.text = text

    for (p_pr, r_pr), paragraph in zip(saved, text_frame.paragraphs):
        if p_pr is not None:
            existing_p_pr = paragraph._p.find(qn("a:pPr"))
            if existing_p_pr is not None:
                paragraph._p.remove(existing_p_pr)
            paragraph._p.insert(0, p_pr)
        if r_pr is not None and paragraph.runs:
            run = paragraph.runs[0]
            existing_r_pr = run._r.find(qn("a:rPr"))
            if existing_r_pr is not None:
                run._r.remove(existing_r_pr)
            run._r.insert(0, r_pr)


def _require_slide(presentation: PresentationType, slide_index: int) -> Any:
    slides = presentation.slides
    if not 0 <= slide_index < len(slides):
        raise ValueError(
            f"slide_index {slide_index} is out of range for a presentation with "
            f"{len(slides)} slides"
        )
    return slides[slide_index]


def _delete_slide(presentation: PresentationType, slide_index: int) -> None:
    """Remove a slide by index.

    python-pptx has no public API for removing a slide (confirmed: no
    Slides.remove()/delete() method exists in the library). The documented
    community workaround -- dropping the slide's <p:sldId> entry from the
    presentation's slide-id list, which is what actually determines slide
    order/membership in the underlying XML -- is used here instead.

    Removing only the <p:sldId> entry leaves the slide's own XML part and
    the presentation-to-slide relationship in the saved package -- verified
    directly: python-pptx's serializer does no orphan-pruning on save, so a
    caller reading the package's parts/relationships directly (rather than
    walking presentation.slides) would still see the "deleted" slide's
    content, and repeated add/delete cycles would accumulate dead parts.
    part.drop_rel() below removes that relationship (and, since it's the
    part's only remaining reference, the part itself) so the deleted
    slide's content doesn't survive in the saved file at all.
    """
    slide_id_list = presentation.slides._sldIdLst
    slide_ids = list(slide_id_list)
    if not 0 <= slide_index < len(slide_ids):
        raise ValueError(
            f"slide_index {slide_index} is out of range for a presentation with "
            f"{len(slide_ids)} slides"
        )
    target = slide_ids[slide_index]
    relationship_id = target.get(qn("r:id"))
    slide_id_list.remove(target)
    if relationship_id:
        presentation.part.drop_rel(relationship_id)


@mcp.tool()
def powerpoint_create_presentation(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """Create a new, blank PowerPoint presentation at file_path. Fails if a
    file already exists there -- edit it with the other powerpoint_* tools
    instead of recreating it."""
    try:
        buffer = io.BytesIO()
        Presentation().save(buffer)
        item = _create_only_upload(buffer.getvalue(), file_path, site_id, drive_id)
        return _success(item=item)
    except Exception as e:
        logger.error("Error creating PowerPoint presentation %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def powerpoint_get_presentation_text(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """Get all text in a PowerPoint presentation, as a list of slides each
    with the text of every text-bearing shape on it (including a shape
    nested inside a group, and each cell of a table) plus that slide's
    speaker notes, if any."""
    try:
        presentation = _download_presentation(file_path, site_id, drive_id)
        slides = [
            {
                "slide_index": slide_index,
                "shapes": _slide_texts(slide),
                "notes": _slide_notes(slide),
            }
            for slide_index, slide in enumerate(presentation.slides)
        ]
        return _success(slides=slides)
    except Exception as e:
        logger.error(
            "Error getting text for PowerPoint presentation %s: %s", file_path, e
        )
        return _error(str(e))


@mcp.tool()
def powerpoint_list_slides(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """List a PowerPoint presentation's slides, with each slide's index,
    layout name, and shape count."""
    try:
        presentation = _download_presentation(file_path, site_id, drive_id)
        slides = [
            {
                "slide_index": slide_index,
                "layout_name": slide.slide_layout.name,
                "shape_count": len(slide.shapes),
            }
            for slide_index, slide in enumerate(presentation.slides)
        ]
        return _success(slides=slides)
    except Exception as e:
        logger.error(
            "Error listing slides for PowerPoint presentation %s: %s", file_path, e
        )
        return _error(str(e))


@mcp.tool()
def powerpoint_get_slide_text(
    file_path: str,
    slide_index: int,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Get one slide's shapes with their index, type, and text (needed by
    powerpoint_set_shape_text)."""
    try:
        slide_index = _require_int(slide_index, "slide_index")
        presentation = _download_presentation(file_path, site_id, drive_id)
        slide = _require_slide(presentation, slide_index)
        shapes = [
            {
                "shape_index": shape_index,
                "shape_type": str(shape.shape_type),
                "is_placeholder": shape.is_placeholder,
                "text": _shape_text(shape),
            }
            for shape_index, shape in enumerate(slide.shapes)
        ]
        return _success(shapes=shapes)
    except Exception as e:
        logger.error(
            "Error getting slide %s text for PowerPoint presentation %s: %s",
            slide_index,
            file_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def powerpoint_set_shape_text(
    file_path: str,
    slide_index: int,
    shape_index: int,
    text: str,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Replace a shape's text on a slide, by slide_index and shape_index
    (from powerpoint_get_slide_text). Only works on a shape that already
    has a text frame (a title, body, or plain text box); errors otherwise.
    Preserves each existing paragraph's alignment/bullet/indent formatting
    and its first run's character formatting (bold, italic, font, size,
    color) by position; a paragraph this call adds or removes (a "\\n" in
    text starts a new paragraph) has no corresponding old/new paragraph to
    carry that formatting from/to. Also errors (rather than silently
    discarding it) if the shape's text contains a hyperlink or a dynamic
    field such as an auto-updating slide number or date, since replacing
    the whole frame's text has no way to carry those over -- edit that
    shape directly in PowerPoint instead."""
    try:
        slide_index = _require_int(slide_index, "slide_index")
        shape_index = _require_int(shape_index, "shape_index")
        presentation = _download_presentation(file_path, site_id, drive_id)
        slide = _require_slide(presentation, slide_index)
        shapes = list(slide.shapes)
        if not 0 <= shape_index < len(shapes):
            raise ValueError(
                f"shape_index {shape_index} is out of range for slide {slide_index} "
                f"with {len(shapes)} shapes"
            )
        shape = shapes[shape_index]
        if not shape.has_text_frame:
            raise ValueError(
                f"shape {shape_index} on slide {slide_index} has no text frame "
                "(e.g. it's an image or a plain line/connector) and cannot hold text"
            )
        if _text_frame_has_dynamic_content(shape.text_frame):
            raise ValueError(
                f"shape {shape_index} on slide {slide_index} contains a hyperlink "
                "or a dynamic field (e.g. slide number or date) that replacing its "
                "text would silently delete -- edit this shape directly in "
                "PowerPoint instead"
            )
        _replace_text_frame_text(shape.text_frame, text)
        item = _upload_presentation(presentation, file_path, site_id, drive_id)
        return _success(item=item)
    except Exception as e:
        logger.error(
            "Error setting shape %s text on slide %s in PowerPoint presentation %s: %s",
            shape_index,
            slide_index,
            file_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def powerpoint_add_slide(
    file_path: str,
    title: str | None = None,
    body_text: str | None = None,
    layout_index: int = _DEFAULT_SLIDE_LAYOUT_INDEX,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Add a new slide at the end of a PowerPoint presentation.

    layout_index selects a slide layout from the presentation's slide
    master (0 is usually "Title Slide", 1 "Title and Content", 6 "Blank" --
    use powerpoint_list_slide_layouts to see what this presentation
    actually has, since layouts vary by template). title, if given, is set
    on the layout's title placeholder when present. body_text, if given, is
    set on the first non-title placeholder (by idx order) that has a text
    frame -- a non-text placeholder earlier in idx order (e.g. a picture or
    chart placeholder) is skipped rather than causing body_text to be left
    unset. This is still a heuristic, not a guaranteed "body" role: on a
    "Title Slide" layout, for example, that placeholder is actually the
    subtitle. Use powerpoint_get_slide_text after adding the slide to
    confirm where each piece of text actually landed."""
    try:
        layout_index = _require_int(layout_index, "layout_index")
        presentation = _download_presentation(file_path, site_id, drive_id)
        layouts = presentation.slide_layouts
        if not 0 <= layout_index < len(layouts):
            raise ValueError(
                f"layout_index {layout_index} is out of range for this "
                f"presentation's {len(layouts)} slide layouts"
            )
        slide = presentation.slides.add_slide(layouts[layout_index])
        if title is not None and slide.shapes.title is not None:
            slide.shapes.title.text = title
        if body_text is not None:
            body_placeholder = _select_body_placeholder(slide)
            if body_placeholder is not None:
                body_placeholder.text_frame.text = body_text
        item = _upload_presentation(presentation, file_path, site_id, drive_id)
        return _success(item=item, slide_index=len(presentation.slides) - 1)
    except Exception as e:
        logger.error(
            "Error adding slide to PowerPoint presentation %s: %s", file_path, e
        )
        return _error(str(e))


@mcp.tool()
def powerpoint_list_slide_layouts(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """List the slide layouts available in a presentation's template, with
    the index powerpoint_add_slide's layout_index expects."""
    try:
        presentation = _download_presentation(file_path, site_id, drive_id)
        layouts = [
            {"layout_index": index, "name": layout.name}
            for index, layout in enumerate(presentation.slide_layouts)
        ]
        return _success(layouts=layouts)
    except Exception as e:
        logger.error(
            "Error listing slide layouts for PowerPoint presentation %s: %s",
            file_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def powerpoint_delete_slide(
    file_path: str,
    slide_index: int,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Delete a slide from a PowerPoint presentation by index."""
    try:
        slide_index = _require_int(slide_index, "slide_index")
        presentation = _download_presentation(file_path, site_id, drive_id)
        _delete_slide(presentation, slide_index)
        item = _upload_presentation(presentation, file_path, site_id, drive_id)
        return _success(item=item)
    except Exception as e:
        logger.error(
            "Error deleting slide %s from PowerPoint presentation %s: %s",
            slide_index,
            file_path,
            e,
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
