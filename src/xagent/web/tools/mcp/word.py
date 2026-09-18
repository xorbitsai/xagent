import io
import json
import logging
import os
from typing import Any
from urllib.parse import quote

import requests
from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.text.run import Run
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("word-mcp")

setup_proxy_env()

mcp = FastMCP("word-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30
_BINARY_TIMEOUT_SECONDS = 60

_WORD_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

# Microsoft Graph exposes a Word (.docx) file only as a driveItem content
# blob -- there is no structured "Word API" resource the way there is a
# Workbook API for Excel (verified: no paragraph/section-level Graph
# endpoints exist for Word documents). Every tool here therefore downloads
# the whole file, edits it in memory with python-docx, and re-uploads the
# whole file. Kept entirely in memory (io.BytesIO, never written to local
# disk) since there is no local file for a caller to name or an allowlist
# to enforce -- unlike onedrive_upload_file/sharepoint_upload_file, which
# upload a real local file a caller points at.
#
# Graph's simple content PUT is documented inconsistently -- see
# sharepoint.py's _SIMPLE_UPLOAD_MAX_BYTES for the same ambiguity between
# the OneDrive concepts page (< 4 MB) and the v1.0 API reference (< 250 MB)
# -- so this module keeps the same conservative 4 MB bound rather than
# implementing a resumable upload session.
_SIMPLE_UPLOAD_MAX_BYTES = 4_000_000


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
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=_graph_headers(extra_headers),
        params=params,
        json=body,
        data=data,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text.strip()
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
        raise _GraphRequestError(message, status_code=response.status_code) from exc

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
    normalize a trailing dot away, so "Report.docx." could silently resolve
    to a real, different "Report.docx")."""
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
        drive_base = (
            f"/sites/{site_segment}/drives/{url_path_id(drive_id, 'drive_id')}"
            if drive_id
            else f"/sites/{site_segment}/drive"
        )
    elif drive_id:
        drive_base = f"/drives/{url_path_id(drive_id, 'drive_id')}"
    else:
        drive_base = "/me/drive"
    return f"{drive_base}/root:/{quote(normalized, safe='/')}:"


def _content_path(file_path: str, site_id: str | None, drive_id: str | None) -> str:
    return f"{_item_path(file_path, site_id, drive_id)}/content"


def _download_document(
    file_path: str, site_id: str | None, drive_id: str | None
) -> DocumentType:
    content = _graph_request(
        "GET", _content_path(file_path, site_id, drive_id), raw=True
    )
    try:
        return Document(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(
            f"Could not open {file_path!r} as a Word document -- it may not be a "
            "valid .docx file"
        ) from exc


def _upload_document(
    document: DocumentType, file_path: str, site_id: str | None, drive_id: str | None
) -> dict[str, Any]:
    buffer = io.BytesIO()
    document.save(buffer)
    content = buffer.getvalue()
    if len(content) > _SIMPLE_UPLOAD_MAX_BYTES:
        raise ValueError(
            f"The updated document is {len(content)} bytes, over the "
            f"{_SIMPLE_UPLOAD_MAX_BYTES // 1_000_000} MB limit this tool currently "
            "supports"
        )
    result = _graph_request(
        "PUT",
        _content_path(file_path, site_id, drive_id),
        extra_headers={"Content-Type": _WORD_MIME_TYPE},
        data=content,
        timeout=_BINARY_TIMEOUT_SECONDS,
    )
    if not isinstance(result, dict) or not result.get("id"):
        raise RuntimeError("Graph did not confirm the document upload completed")
    safe_item = dict(result)
    safe_item.pop("@microsoft.graph.downloadUrl", None)
    return safe_item


def _create_only_upload(
    content: bytes, file_path: str, site_id: str | None, drive_id: str | None
) -> dict[str, Any]:
    """Upload content, failing instead of silently overwriting if a file
    already exists at file_path.

    A separate exists-check GET followed by a plain _upload_document PUT
    would still race: Graph's own "Upload small files" reference documents
    no If-None-Match or other conditional-write option on the simple
    content PUT, so two concurrent callers can both pass the check and the
    second would silently clobber the first. The upload-session API is
    documented to accept a conflictBehavior of "fail" on session creation,
    which Graph enforces atomically server-side (a 409 nameAlreadyExists
    if the target already exists) -- used here even though the content
    (a blank new document) is always small enough for a single-PUT
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
                f"{file_path!r} already exists; use the other word_* tools to "
                "edit it instead of recreating it"
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
                "Content-Type": _WORD_MIME_TYPE,
            },
            timeout=_BINARY_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.HTTPError:
        status_code = response.status_code
        if status_code == 409:
            raise ValueError(
                f"{file_path!r} already exists; use the other word_* tools to "
                "edit it instead of recreating it"
            ) from None
        raise _GraphRequestError(
            f"Word document upload failed with HTTP {status_code}",
            status_code=status_code,
        ) from None
    except requests.RequestException:
        raise RuntimeError("Word document upload failed") from None
    result = response.json()
    if not isinstance(result, dict) or not result.get("id"):
        raise RuntimeError("Graph did not confirm the document upload completed")
    safe_item = dict(result)
    safe_item.pop("@microsoft.graph.downloadUrl", None)
    return safe_item


_SAFE_RUN_CHILD_TAGS = (qn("w:rPr"), qn("w:t"))


# Assigning Run.text clears every child of <w:r> except <w:rPr> (formatting)
# and re-adds only <w:t>/<w:tab>/<w:br>/<w:cr> elements built from the new
# text -- verified directly against python-docx's own CT_R.text getter and
# setter. So rather than enumerate the specific tags known to be at risk (an
# inline image/drawing, a field character, a symbol, a footnote reference,
# and more -- an open-ended and version-fragile list), a run is treated as
# unsafe to rewrite unless every one of its children is <w:rPr> or <w:t>.
# This is deliberately conservative: some elements that do round-trip
# correctly today (a <w:tab>, for instance) are refused too, and a <w:cr>
# is silently downgraded to a <w:br> even before this guard runs -- both
# acceptable trade-offs against a helper that has to be re-verified by hand
# against python-docx's undocumented internals every time it's relied on.
def _run_has_non_text_content(run: Any) -> bool:
    return any(child.tag not in _SAFE_RUN_CHILD_TAGS for child in run._r)


# Runs and hyperlinks reached through one of these ancestors are not part of
# a paragraph's own edit history: w:ins/w:del/w:moveFrom/w:moveTo mark
# tracked-change insertions, deletions, and moves attributed to someone
# else's not-yet-accepted edit, and w:txbxContent is a text box's own
# separate content stream, merely anchored to (not part of) the run that
# holds its <w:drawing> -- none of this is visible through
# word_get_document_text/word_list_paragraphs either. A plain recursive
# ".//w:r" or ".//w:hyperlink" search would reach into all of these and
# silently rewrite or misdetect content the caller never saw (verified: a
# tracked insertion's run text is readable and matchable even though
# paragraph.runs excludes it, and a text box's nested <w:p> is invisible to
# document.paragraphs but still reachable via ".//" from its anchor run).
_PARAGRAPH_FLOW_PREDICATE = (
    "not(ancestor::w:ins) and not(ancestor::w:del) "
    "and not(ancestor::w:moveFrom) and not(ancestor::w:moveTo) "
    "and not(ancestor::w:txbxContent)"
)


def _paragraph_flow_elements(paragraph: Any, local_tag: str) -> Any:
    """Descendants of paragraph matching local_tag (e.g. "w:r", "w:hyperlink"),
    excluding ones nested in tracked-change markup or a text box -- see
    _PARAGRAPH_FLOW_PREDICATE."""
    return paragraph._p.xpath(f".//{local_tag}[{_PARAGRAPH_FLOW_PREDICATE}]")


# Elements that hold their own runs outside paragraph.runs' direct-children
# view, the same way a hyperlink does: w:ins/w:del/w:moveFrom/w:moveTo (a
# tracked-change insertion, deletion, or move) and w:sdt (a content
# control). _set_paragraph_text only rewrites paragraph.runs[0] and empties
# the rest, so any of these left untouched keeps its old text -- verified
# directly: the old text is still physically present in the saved XML
# (readable via ".//w:t", just not reflected in paragraph.text), and Word
# would render it alongside the new text. Excludes ones inside a text box,
# which is a separate content stream this rewrite never touches anyway.
_UNSUPPORTED_NESTED_ELEMENT_TAGS = (
    "w:hyperlink",
    "w:ins",
    "w:del",
    "w:moveFrom",
    "w:moveTo",
    "w:sdt",
)


def _paragraph_has_unsupported_structure(paragraph: Any) -> bool:
    predicate = " or ".join(f"self::{tag}" for tag in _UNSUPPORTED_NESTED_ELEMENT_TAGS)
    return bool(paragraph._p.xpath(f".//*[{predicate}][not(ancestor::w:txbxContent)]"))


def _set_paragraph_text(paragraph: Any, text: str) -> None:
    """Replace a paragraph's visible text, keeping its first run's
    character formatting (font, bold, etc.) and its own paragraph style.
    Any additional runs are emptied rather than removed, since python-docx
    doesn't expose a supported run-removal API; an empty run has no visible
    effect but a paragraph with zero runs to begin with needs one added.

    Refuses (rather than silently corrupting the document) a paragraph
    containing a hyperlink, a tracked-change insertion/deletion/move, or a
    content control -- each holds its own runs outside paragraph.runs, so
    the rewrite below would leave their old text stale rather than
    replacing it -- or a paragraph containing a run with an image, break,
    or field character, which the plain-text runs[0].text assignment below
    would silently delete.
    """
    if _paragraph_has_unsupported_structure(paragraph):
        raise ValueError(
            "paragraph contains a hyperlink, tracked change, or content "
            "control; word_set_paragraph_text does not support editing it "
            "here (its own run text is outside python-docx's paragraph.runs, "
            "so it would be left stale rather than replaced) -- edit this "
            "paragraph directly in Word instead"
        )
    for run in paragraph.runs:
        if _run_has_non_text_content(run):
            raise ValueError(
                "paragraph contains a run with non-text content (an image, "
                "line/page break, or field such as a table of contents entry) "
                "that word_set_paragraph_text would silently delete -- edit "
                "this paragraph directly in Word instead"
            )
    if not paragraph.runs:
        paragraph.add_run(text)
        return
    paragraph.runs[0].text = text
    for run in paragraph.runs[1:]:
        run.text = ""


@mcp.tool()
def word_create_document(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """Create a new, blank Word document at file_path. Fails if a file
    already exists there -- edit it with the other word_* tools instead of
    recreating it."""
    try:
        buffer = io.BytesIO()
        Document().save(buffer)
        item = _create_only_upload(buffer.getvalue(), file_path, site_id, drive_id)
        return _success(item=item)
    except Exception as e:
        logger.error("Error creating Word document %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def word_get_document_text(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """Get a Word document's full text, as its paragraphs joined by
    newlines. Table contents are not included -- see word_list_paragraphs
    for per-paragraph detail."""
    try:
        document = _download_document(file_path, site_id, drive_id)
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return _success(text=text, table_count=len(document.tables))
    except Exception as e:
        logger.error("Error getting text for Word document %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def word_list_paragraphs(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """List a Word document's paragraphs with their index, text, and
    style name. The index is needed by word_set_paragraph_text."""
    try:
        document = _download_document(file_path, site_id, drive_id)
        paragraphs = [
            {
                "index": index,
                "text": paragraph.text,
                "style": paragraph.style.name if paragraph.style else None,
            }
            for index, paragraph in enumerate(document.paragraphs)
        ]
        return _success(paragraphs=paragraphs)
    except Exception as e:
        logger.error("Error listing paragraphs for Word document %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def word_set_paragraph_text(
    file_path: str,
    paragraph_index: int,
    text: str,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Replace a Word document paragraph's text by index (from
    word_list_paragraphs). Keeps the paragraph's style and its first run's
    character formatting; does not preserve formatting that varied across
    multiple runs within the paragraph."""
    try:
        document = _download_document(file_path, site_id, drive_id)
        paragraphs = document.paragraphs
        if not 0 <= paragraph_index < len(paragraphs):
            raise ValueError(
                f"paragraph_index {paragraph_index} is out of range for a document "
                f"with {len(paragraphs)} paragraphs"
            )
        _set_paragraph_text(paragraphs[paragraph_index], text)
        item = _upload_document(document, file_path, site_id, drive_id)
        return _success(item=item)
    except Exception as e:
        logger.error(
            "Error setting paragraph %s text in Word document %s: %s",
            paragraph_index,
            file_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def word_append_paragraph(
    file_path: str,
    text: str,
    style: str | None = None,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Append a new paragraph to the end of a Word document. style, if
    given, is a built-in Word style name (e.g. "List Bullet", "Quote");
    invalid names raise an error rather than silently falling back to
    Normal."""
    try:
        document = _download_document(file_path, site_id, drive_id)
        document.add_paragraph(text, style=style)
        item = _upload_document(document, file_path, site_id, drive_id)
        return _success(item=item)
    except Exception as e:
        logger.error("Error appending paragraph to Word document %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def word_add_heading(
    file_path: str,
    text: str,
    level: int = 1,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Append a heading to the end of a Word document. level is 0 (the
    document title style) through 9."""
    try:
        if not 0 <= level <= 9:
            raise ValueError("level must be between 0 and 9")
        document = _download_document(file_path, site_id, drive_id)
        document.add_heading(text, level=level)
        item = _upload_document(document, file_path, site_id, drive_id)
        return _success(item=item)
    except Exception as e:
        logger.error("Error adding heading to Word document %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def word_replace_text(
    file_path: str,
    find: str,
    replace: str,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Find and replace text across a Word document's paragraphs.

    Best-effort: a match is only found when it falls entirely within one
    run (python-docx exposes text at the run level, and Word frequently
    splits one visible word across multiple runs for formatting reasons).
    Matches spanning a run boundary are not found. Check the returned
    replacement count against expectations; use word_list_paragraphs plus
    word_set_paragraph_text for a guaranteed replacement in a specific
    paragraph.

    Searches every run in a paragraph, including ones nested inside a
    hyperlink or other wrapper (a content control, a smart tag) --
    paragraph.runs itself only sees direct children, which would otherwise
    make hyperlink text invisible to this tool even though it's included
    when reading the document back with word_get_document_text. Does not
    search inside a text box or a tracked-change insertion/deletion/move,
    since neither is visible through this module's read tools either.

    Refuses (rather than silently corrupting the document) a match found in
    a run that also holds non-text content -- an image, break, or field --
    since assigning that run's text would delete it the same way
    word_set_paragraph_text's identical guard prevents."""
    try:
        if not find:
            raise ValueError("find must not be empty")
        document = _download_document(file_path, site_id, drive_id)
        replacements = 0
        for paragraph in document.paragraphs:
            for r in _paragraph_flow_elements(paragraph, "w:r"):
                run = Run(r, paragraph)
                if find in run.text:
                    if _run_has_non_text_content(run):
                        raise ValueError(
                            "found a match in a run that also contains non-text "
                            "content (an image, break, or field) that replacing "
                            "its text would silently delete -- edit this "
                            "paragraph directly in Word instead"
                        )
                    replacements += run.text.count(find)
                    run.text = run.text.replace(find, replace)
        item = _upload_document(document, file_path, site_id, drive_id)
        return _success(item=item, replacements=replacements)
    except Exception as e:
        logger.error("Error replacing text in Word document %s: %s", file_path, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
