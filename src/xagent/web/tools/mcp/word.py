import io
import json
import logging
import os
import zipfile
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests
from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.text.run import Run
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import clamp_limit, clamp_offset, setup_proxy_env, url_path_id

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
# Bounds the compressed bytes read off the wire (see _read_capped_content).
# On its own this does not bound what python-docx's ZIP/XML parser expands
# those bytes into -- a small, highly compressible .docx can still pass
# this cap and exhaust memory once unpacked, which is what
# _validate_docx_archive checks separately, before Document() unpacks
# anything. Matches sharepoint.py's identical _MAX_DOWNLOAD_BYTES/
# _read_capped_content guard on the compressed-size half of this hazard.
_MAX_DOWNLOAD_BYTES = 10_000_000

# The decompressed-size half of the same hazard: matches powerpoint.py's
# identical _MAX_PRESENTATION_UNCOMPRESSED_BYTES/_MAX_PRESENTATION_PARTS
# guard (via _validate_presentation_archive) for the same OOXML
# ZIP-archive structure a .docx shares with a .pptx.
_MAX_DECOMPRESSED_BYTES = 500_000_000
_MAX_ARCHIVE_PARTS = 10_000

# _upload_document sends the whole edited document in a single PUT to an
# upload-session URL (not Graph's separate small-file content PUT), for
# which Graph's own upload-session guidance recommends staying under 60
# MiB per request. Set equal to _MAX_DOWNLOAD_BYTES rather than some
# smaller, independently-chosen figure: any document this module can
# successfully read back must also be re-uploadable after an edit, or
# every document between the two limits would download and parse
# successfully only to fail on every single write -- verified this was
# exactly the prior behavior when this constant was a leftover 4 MB from
# before this function used an upload session.
_MAX_UPLOAD_BYTES = _MAX_DOWNLOAD_BYTES
_DEFAULT_TEXT_PAGE_CHARS = 20_000
_MAX_TEXT_PAGE_CHARS = 40_000
_DEFAULT_PARAGRAPH_PAGE_SIZE = 100
_MAX_PARAGRAPH_PAGE_SIZE = 500


class _GraphRequestError(RuntimeError):
    """Graph HTTP failure that retains its status without response parsing."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class _EditSnapshot:
    """Stable Graph identity and version observed for an edit download."""

    item_path: str
    etag: str


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


def _read_capped_content(response: requests.Response, *, max_bytes: int) -> bytes:
    """Read a streamed response body, rejecting it once it exceeds ``max_bytes``.

    The caller sends Accept-Encoding: identity so Content-Length reflects the
    actual byte count (requests would otherwise transparently decompress a
    gzip/deflate body, making that header describe the wire size rather than
    the size actually buffered here) -- the Content-Encoding check below is
    a fallback for a server that ignores the request header anyway.

    ``with response`` releases the connection on every exit path, not just
    a fully-consumed one -- matching sharepoint.py's identical
    _read_capped_content for the same reasoning.
    """
    with response:
        encoding = response.headers.get("Content-Encoding", "identity")
        if encoding.lower() != "identity":
            raise ValueError(f"unsupported compressed content encoding: {encoding!r}")

        content_length = response.headers.get("Content-Length")
        if content_length is not None and content_length.isdigit():
            if int(content_length) > max_bytes:
                raise ValueError(
                    f"file is too large to download ({content_length} bytes, "
                    f"limit is {max_bytes} bytes)"
                )

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(
                    f"file is too large to download (limit is {max_bytes} bytes)"
                )
            chunks.append(chunk)
        return b"".join(chunks)


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
    request_headers = _graph_headers(extra_headers)
    if raw:
        request_headers["Accept-Encoding"] = "identity"

    if raw:
        # Graph's /content redirects to a short-lived preauthenticated
        # download URL -- itself a bearer credential, since whoever has it
        # can download the file with no further auth until it expires.
        # requests follows that redirect transparently, so both a
        # RequestException raised while connecting to (or reading from) the
        # final host (its message embeds the URL it was trying to reach)
        # and response.url/str(HTTPError) after a failed response (same
        # reason) would leak that credential into the returned/logged
        # error -- matching sharepoint.py's identical guard on the same
        # hazard for its own document downloads. response.text is not the
        # same risk -- it's Graph's own JSON error body -- so it's kept for
        # diagnostic detail (matching the non-raw branch below) while the
        # URL-bearing exception/response.url are not.
        try:
            response = requests.request(
                method=method,
                url=f"{GRAPH_BASE_URL}{path}",
                headers=request_headers,
                params=params,
                json=body,
                data=data,
                timeout=timeout,
                stream=True,
            )
        except requests.RequestException:
            # "from None": chaining the original exception would still
            # attach it as __cause__, which embeds the same download URL
            # this branch exists to keep out of the raised message -- a
            # future traceback/log/APM capture could surface it from there
            # instead, matching the same guard already applied to the
            # upload-session path in _create_only_upload/_upload_document.
            raise _GraphRequestError(
                "network error downloading file content", status_code=0
            ) from None
        try:
            response.raise_for_status()
        except requests.HTTPError:
            response_text = response.text.strip()
            message = f"{response.status_code} error downloading file content"
            if response_text:
                message = f"{message} - {response_text}"
            raise _GraphRequestError(
                message, status_code=response.status_code
            ) from None
        try:
            return _read_capped_content(response, max_bytes=_MAX_DOWNLOAD_BYTES)
        except requests.RequestException:
            # A connection drop mid-body (after the 200 OK, while streaming
            # via response.iter_content() inside _read_capped_content) isn't
            # covered by the try/except above it -- redact it the same way
            # rather than letting it propagate with the download URL
            # embedded in its own message.
            raise _GraphRequestError(
                "network error downloading file content", status_code=0
            ) from None

    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=request_headers,
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


def _site_subresource_base(site_id: str) -> str:
    """Site path prefix for appending a further path segment (e.g. "/drive").
    A path-addressed site id ("hostname:/path") must be closed with a
    second colon before appending more segments -- confirmed against
    Graph's own sharepoint-addressing documentation, which shows
    "/sites/{hostname}:/{path}:/drive" as the one-call compound form -- or
    Graph parses the appended segment as part of the site's own
    server-relative path instead of as a sub-resource name (matching
    sharepoint.py's identical _site_subresource_base). The "root" and
    composite-id ("hostname,siteId,webId") forms contain no colon and are
    returned unchanged."""
    value = site_id.strip()
    segment = _site_segment(site_id)
    if ":" in value and not value.endswith(":"):
        return f"/sites/{segment}:"
    return f"/sites/{segment}"


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
        site_base = _site_subresource_base(site_id)
        drive_base = (
            f"{site_base}/drives/{url_path_id(drive_id, 'drive_id')}"
            if drive_id
            else f"{site_base}/drive"
        )
    elif drive_id:
        drive_base = f"/drives/{url_path_id(drive_id, 'drive_id')}"
    else:
        drive_base = "/me/drive"
    return f"{drive_base}/root:/{quote(normalized, safe='/')}:"


def _content_path(file_path: str, site_id: str | None, drive_id: str | None) -> str:
    return f"{_item_path(file_path, site_id, drive_id)}/content"


def _validate_docx_archive(content: bytes) -> None:
    """Reject a .docx whose ZIP entries would expand far beyond what a
    legitimate Word document needs, before Document() unpacks any of them.
    _read_capped_content only bounds the compressed bytes read off the
    wire; a small, highly compressible archive can still pass that check
    and exhaust memory once decompressed. Matches powerpoint.py's identical
    _validate_presentation_archive guard on the same OOXML ZIP-archive
    structure."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            if len(infos) > _MAX_ARCHIVE_PARTS:
                raise ValueError(
                    "The document contains too many OOXML parts to process safely"
                )
            expanded_size = sum(info.file_size for info in infos)
            if expanded_size > _MAX_DECOMPRESSED_BYTES:
                raise ValueError(
                    "The document expands beyond the safe OOXML processing limit"
                )
    except zipfile.BadZipFile as exc:
        raise ValueError("The file is not a valid Word document OOXML archive") from exc


def _download_document(
    file_path: str, site_id: str | None, drive_id: str | None, *, need_etag: bool = True
) -> tuple[DocumentType, _EditSnapshot | None]:
    """Download file_path and parse it, also returning the driveItem's
    stable driveItem identity and current eTag for edit callers. An edit
    fails closed if Graph omits any required identity/version field: silently
    falling back to a path-addressed unconditional upload would reintroduce
    the stale-write race this snapshot exists to prevent. need_etag=False
    skips the metadata GET entirely for read-only tools."""
    item_path = _item_path(file_path, site_id, drive_id)
    snapshot = None
    download_path = _content_path(file_path, site_id, drive_id)
    if need_etag:
        metadata = _graph_request(
            "GET", item_path, params={"$select": "id,eTag,parentReference"}
        )
        item_id = metadata.get("id") if isinstance(metadata, dict) else None
        etag = metadata.get("eTag") if isinstance(metadata, dict) else None
        parent_reference = (
            metadata.get("parentReference") if isinstance(metadata, dict) else None
        )
        metadata_drive_id = (
            parent_reference.get("driveId")
            if isinstance(parent_reference, dict)
            else None
        )
        if (
            not isinstance(item_id, str)
            or not item_id.strip()
            or not isinstance(etag, str)
            or not etag.strip()
            or not isinstance(metadata_drive_id, str)
            or not metadata_drive_id.strip()
        ):
            raise RuntimeError(
                "Graph did not return the driveItem id, drive id, and eTag "
                "required for a concurrency-safe Word edit"
            )
        stable_item_path = (
            f"/drives/{url_path_id(metadata_drive_id, 'drive_id')}/items/"
            f"{url_path_id(item_id, 'item_id')}"
        )
        snapshot = _EditSnapshot(item_path=stable_item_path, etag=etag)
        download_path = f"{stable_item_path}/content"
    content = _graph_request("GET", download_path, raw=True)
    _validate_docx_archive(content)
    try:
        return Document(io.BytesIO(content)), snapshot
    except Exception as exc:
        raise ValueError(
            f"Could not open {file_path!r} as a Word document -- it may not be a "
            "valid .docx file"
        ) from exc


def _upload_document(
    document: DocumentType,
    file_path: str,
    snapshot: _EditSnapshot,
) -> dict[str, Any]:
    """Save document and upload it in place of file_path's current content.

    Goes through an upload session (like _create_only_upload) instead of
    the simpler content PUT this used before, specifically to attach
    If-Match: etag on session creation -- confirmed in Graph's
    createUploadSession reference, unlike the plain content-PUT endpoint's
    own reference, which documents no conditional-write header at all.
    Without this, two callers racing a download-mutate-upload cycle on the
    same file could have the later one silently overwrite the earlier
    one's change; a stale etag now fails the write with 412 instead.
    Both fields in snapshot are mandatory. The upload session is addressed
    by the immutable driveItem id rather than by the caller's path, so a
    concurrent rename or delete-and-recreate cannot redirect the write to a
    different item.
    """
    buffer = io.BytesIO()
    document.save(buffer)
    content = buffer.getvalue()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise ValueError(
            f"The updated document is {len(content)} bytes, over the "
            f"{_MAX_UPLOAD_BYTES // 1_000_000} MB limit this tool currently "
            "supports"
        )
    conflict_message = (
        f"{file_path!r} was changed by someone else since this edit "
        "started; re-read it and retry"
    )
    try:
        session = _graph_request(
            "POST",
            f"{snapshot.item_path}/createUploadSession",
            # "replace" (rather than the "fail" default) is required here:
            # this path always resolves to the file already being edited,
            # so the default's job -- refusing to silently clobber an
            # unrelated file that happens to share this name -- is instead
            # done by the If-Match check below.
            body={"item": {"@microsoft.graph.conflictBehavior": "replace"}},
            extra_headers={"If-Match": snapshot.etag},
        )
    except _GraphRequestError as exc:
        if exc.status_code == 412:
            raise ValueError(conflict_message) from exc
        raise
    upload_url = session.get("uploadUrl") if isinstance(session, dict) else None
    if not isinstance(upload_url, str) or not upload_url:
        raise RuntimeError("Graph did not return an upload session URL")

    # See _create_only_upload's identical comment: the upload-session URL
    # is itself a bearer secret, so neither requests.put's own exception
    # nor the HTTPError from raise_for_status() is stringified into a
    # message here, and each is re-raised "from None".
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
        if status_code == 412:
            raise ValueError(conflict_message) from None
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


# Runs protected from word_replace_text's in-place rewrite the same way
# _UNSUPPORTED_NESTED_ELEMENT_TAGS protects word_set_paragraph_text's whole
# paragraph: a content control, a legacy smart tag, a custom XML region, or
# a simple field's cached result -- rewriting any of these in place would
# desync it from a data binding or a field instruction this tool has no way
# to update. w:hyperlink is deliberately absent here: unlike
# word_set_paragraph_text (which must replace a paragraph's entire text at
# once and so cannot handle a hyperlink's separately-addressed run at all),
# word_replace_text only rewrites the one matched run in place, and a
# hyperlink's run is an ordinary <w:r> once reached -- safe to edit.
_PROTECTED_RUN_ANCESTOR_TAGS = ("w:sdt", "w:fldSimple", "w:smartTag", "w:customXml")
_PROTECTED_RUN_ANCESTOR_XPATH = (
    "boolean("
    + " or ".join(f"ancestor::{tag}" for tag in _PROTECTED_RUN_ANCESTOR_TAGS)
    + ")"
)


def _iter_replaceable_runs(paragraph: Any, field_depth_stack: list[bool]) -> Any:
    """Yield (run_element, is_field_protected) for each <w:r> in
    paragraph's own flow (excluding tracked-change/text-box content, per
    _PARAGRAPH_FLOW_PREDICATE) -- a single pass over one xpath() call, so
    every run's identity stays valid for the whole loop, unlike comparing
    id()s collected from two separate xpath() calls: lxml only guarantees a
    node's proxy object identity for as long as something still holds a
    reference to it, and a set of bare id() ints doesn't -- verified
    directly, this silently mismatched two different runs after the
    resulting document was saved and reloaded.

    Tracks complex-field state (a <w:fldChar w:fldCharType="begin"/>, the
    field's instruction text, a "separate" marker, its cached display-text
    runs, then an "end" marker) via field_depth_stack, a list the *caller*
    creates once and passes to every paragraph in the document. A field
    commonly opens in one paragraph and closes many paragraphs later (a
    multi-entry table of contents has its own begin/separate in the first
    entry's paragraph, one entry per following paragraph, and end in the
    last) -- verified directly: tracking this state fresh per paragraph
    call, as an earlier version of this function did, loses protection for
    every entry after the first. The stack (not a single flag) also matters
    for nesting: a field's own "end" must only close its own frame, not any
    enclosing field's still-open result region -- verified directly: a
    plain boolean lets a nested field's "end" prematurely un-protect the
    outer field's remaining result text.

    The exclusion check runs before the field-marker check, not after, so
    a field marker hidden inside tracked-change/text-box markup can never
    touch field_depth_stack -- verified directly: checking exclusion second
    let a field marker inside an unrelated, unaccepted tracked-change
    insertion still toggle (and leave toggled) the shared field state for
    runs later in the same paragraph.
    """
    for r in paragraph._p.xpath(".//w:r"):
        if not r.xpath(_PARAGRAPH_FLOW_PREDICATE):
            continue
        fld_char = r.find(qn("w:fldChar"))
        if fld_char is not None:
            fld_type = fld_char.get(qn("w:fldCharType"))
            if fld_type == "begin":
                field_depth_stack.append(False)
            elif fld_type == "separate":
                if field_depth_stack:
                    field_depth_stack[-1] = True
            elif fld_type == "end":
                if field_depth_stack:
                    field_depth_stack.pop()
            continue
        field_protected = (field_depth_stack and field_depth_stack[-1]) or bool(
            r.xpath(_PROTECTED_RUN_ANCESTOR_XPATH)
        )
        yield r, field_protected


# Elements that hold their own runs outside paragraph.runs' direct-children
# view, the same way a hyperlink does: w:ins/w:del/w:moveFrom/w:moveTo (a
# tracked-change insertion, deletion, or move), w:sdt (a content control),
# w:smartTag (a legacy Office smart tag), w:customXml (a custom XML markup
# region), and w:fldSimple (a simple field, e.g. PAGE/DATE/a TOC entry --
# its cached display text lives in a run here). _set_paragraph_text only
# rewrites paragraph.runs[0] and empties the rest, so any of these left
# untouched keeps its old text -- verified directly for each: the old text
# is still physically present in the saved XML (readable via ".//w:t", just
# not reflected in paragraph.text), and Word would render it alongside the
# new text. Excludes ones inside a text box, which is a separate content
# stream this rewrite never touches anyway.
_UNSUPPORTED_NESTED_ELEMENT_TAGS = (
    "w:hyperlink",
    "w:ins",
    "w:del",
    "w:moveFrom",
    "w:moveTo",
    "w:sdt",
    "w:smartTag",
    "w:customXml",
    "w:fldSimple",
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
    containing a hyperlink, a tracked-change insertion/deletion/move, a
    content control, a smart tag, custom XML markup, or a simple field --
    each holds its own runs outside paragraph.runs, so the rewrite below
    would leave their old text stale rather than replacing it -- or a
    paragraph containing a run with an image, break, or field character,
    which the plain-text runs[0].text assignment below would silently
    delete.
    """
    if _paragraph_has_unsupported_structure(paragraph):
        raise ValueError(
            "paragraph contains a hyperlink, tracked change, content "
            "control, smart tag, custom XML region, or field; "
            "word_set_paragraph_text does not support editing it here "
            "(its own run text is outside python-docx's paragraph.runs, "
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


def _text_page_response(text: str, *, offset: int, limit: int, table_count: int) -> str:
    """Return a recoverable character page that fits the tool output cap."""
    start = min(clamp_offset(offset), len(text))
    requested = clamp_limit(limit, max_limit=_MAX_TEXT_PAGE_CHARS)
    end = min(start + requested, len(text))
    max_output_length = get_tool_max_output_length()

    def _build(page_end: int) -> str:
        has_more = page_end < len(text)
        return _success(
            text=text[start:page_end],
            offset=start,
            total_characters=len(text),
            has_more=has_more,
            next_offset=page_end if has_more else None,
            table_count=table_count,
        )

    response = _build(end)
    while len(response) > max_output_length and end > start:
        end = start + (end - start) // 2
        response = _build(end)
    if len(response) > max_output_length or (end == start and start < len(text)):
        raise ValueError(
            "The configured tool output limit is too small to return a "
            "recoverable Word text page"
        )
    return response


def _paragraph_page_response(
    paragraphs: list[Any], *, offset: int, limit: int, text_offset: int
) -> str:
    """Return paragraph records without ever emitting unrecoverable JSON.

    Pages normally advance by paragraph index. If one paragraph alone is too
    large, its text is split and next_offset/next_text_offset resume within
    that same paragraph, so no content is silently discarded.
    """
    total = len(paragraphs)
    page_start = min(clamp_offset(offset), total)
    index = page_start
    first_text_offset = clamp_offset(text_offset)
    page_limit = clamp_limit(limit, max_limit=_MAX_PARAGRAPH_PAGE_SIZE)
    max_output_length = get_tool_max_output_length()
    items: list[dict[str, Any]] = []
    next_offset: int | None = None
    next_text_offset: int | None = None

    def _build(
        page_items: list[dict[str, Any]],
        resume_index: int | None,
        resume_text_offset: int | None,
    ) -> str:
        has_more = resume_index is not None
        return _success(
            paragraphs=page_items,
            offset=page_start,
            total_paragraphs=total,
            has_more=has_more,
            next_offset=resume_index,
            next_text_offset=resume_text_offset if has_more else None,
        )

    while index < total and len(items) < page_limit:
        paragraph = paragraphs[index]
        full_text = paragraph.text
        start = min(first_text_offset if index == page_start else 0, len(full_text))
        item = {
            "index": index,
            "text": full_text[start:],
            "style": paragraph.style.name if paragraph.style else None,
            "text_offset": start,
            "total_text_characters": len(full_text),
            "text_truncated": False,
        }
        following_index = index + 1
        trial_resume = following_index if following_index < total else None
        trial = _build(items + [item], trial_resume, 0 if trial_resume else None)
        if len(trial) <= max_output_length:
            items.append(item)
            index = following_index
            first_text_offset = 0
            continue

        if items:
            next_offset = index
            next_text_offset = start
            break

        # A single paragraph is larger than the output cap. Find the largest
        # prefix that fits and make continuation resume inside that paragraph.
        low = 0
        high = len(full_text) - start
        best_item: dict[str, Any] | None = None
        while low <= high:
            length = (low + high) // 2
            candidate = {**item, "text": full_text[start : start + length]}
            candidate["text_truncated"] = start + length < len(full_text)
            resume_index: int | None = (
                index if candidate["text_truncated"] else following_index
            )
            if resume_index is not None and resume_index >= total:
                resume_index = None
            resume_text = start + length if candidate["text_truncated"] else 0
            encoded = _build(
                [candidate],
                resume_index,
                resume_text if resume_index is not None else None,
            )
            if len(encoded) <= max_output_length:
                best_item = candidate
                low = length + 1
            else:
                high = length - 1
        if best_item is None or not best_item["text"] and start < len(full_text):
            raise ValueError(
                "The configured tool output limit is too small to return a "
                "recoverable Word paragraph page"
            )
        items.append(best_item)
        if best_item["text_truncated"]:
            next_offset = index
            next_text_offset = start + len(best_item["text"])
        elif following_index < total:
            next_offset = following_index
            next_text_offset = 0
        break

    if next_offset is None and index < total:
        next_offset = index
        next_text_offset = 0
    response = _build(items, next_offset, next_text_offset)
    if len(response) > max_output_length:
        raise ValueError(
            "The configured tool output limit is too small to return Word "
            "paragraph pagination metadata"
        )
    return response


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
    file_path: str,
    site_id: str | None = None,
    drive_id: str | None = None,
    offset: int = 0,
    limit: int = _DEFAULT_TEXT_PAGE_CHARS,
) -> str:
    """Get a page of a Word document's paragraph text joined by newlines.

    Table contents are not included. Pass next_offset from a response with
    has_more=true back as offset to continue without losing content. limit
    is a character count and is capped to a safe maximum."""
    try:
        document, _snapshot = _download_document(
            file_path, site_id, drive_id, need_etag=False
        )
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return _text_page_response(
            text, offset=offset, limit=limit, table_count=len(document.tables)
        )
    except Exception as e:
        logger.error("Error getting text for Word document %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def word_list_paragraphs(
    file_path: str,
    site_id: str | None = None,
    drive_id: str | None = None,
    offset: int = 0,
    limit: int = _DEFAULT_PARAGRAPH_PAGE_SIZE,
    text_offset: int = 0,
) -> str:
    """List a recoverable page of paragraphs with index, text, and style.

    Pass next_offset and next_text_offset from a response with has_more=true
    back as offset/text_offset. next_text_offset is non-zero only when one
    unusually large paragraph had to be split to keep the JSON valid. The
    paragraph index is used by word_set_paragraph_text."""
    try:
        document, _snapshot = _download_document(
            file_path, site_id, drive_id, need_etag=False
        )
        return _paragraph_page_response(
            document.paragraphs,
            offset=offset,
            limit=limit,
            text_offset=text_offset,
        )
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
        document, snapshot = _download_document(file_path, site_id, drive_id)
        assert snapshot is not None
        paragraphs = document.paragraphs
        if not 0 <= paragraph_index < len(paragraphs):
            raise ValueError(
                f"paragraph_index {paragraph_index} is out of range for a document "
                f"with {len(paragraphs)} paragraphs"
            )
        _set_paragraph_text(paragraphs[paragraph_index], text)
        item = _upload_document(document, file_path, snapshot)
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
        document, snapshot = _download_document(file_path, site_id, drive_id)
        assert snapshot is not None
        document.add_paragraph(text, style=style)
        item = _upload_document(document, file_path, snapshot)
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
        document, snapshot = _download_document(file_path, site_id, drive_id)
        assert snapshot is not None
        document.add_heading(text, level=level)
        item = _upload_document(document, file_path, snapshot)
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
    hyperlink -- paragraph.runs itself only sees direct children, which
    would otherwise make hyperlink text invisible to this tool even though
    it's included when reading the document back with
    word_get_document_text. Does not search inside a text box or a
    tracked-change insertion/deletion/move, since neither is visible
    through this module's read tools either.

    Refuses (rather than silently corrupting the document or leaving it
    inconsistent) a match found in: a run that also holds non-text content
    -- an image, break, or field character -- since assigning that run's
    text would delete it the same way word_set_paragraph_text's identical
    guard prevents; or a content control or field's cached result, since
    rewriting that text in place would desync it from the control's data
    binding or the field's own instruction, with no way for this tool to
    update either to match."""
    try:
        if not find:
            raise ValueError("find must not be empty")
        document, snapshot = _download_document(file_path, site_id, drive_id)
        assert snapshot is not None
        replacements = 0
        field_depth_stack: list[bool] = []
        for paragraph in document.paragraphs:
            for r, field_protected in _iter_replaceable_runs(
                paragraph, field_depth_stack
            ):
                run = Run(r, paragraph)
                if find in run.text:
                    if _run_has_non_text_content(run):
                        raise ValueError(
                            "found a match in a run that also contains non-text "
                            "content (an image, break, or field) that replacing "
                            "its text would silently delete -- edit this "
                            "paragraph directly in Word instead"
                        )
                    if field_protected:
                        raise ValueError(
                            "found a match inside a content control or a "
                            "field's cached result; replacing it here would "
                            "leave the control's data binding or the field's "
                            "instruction out of sync -- edit this paragraph "
                            "directly in Word instead"
                        )
                    replacements += run.text.count(find)
                    run.text = run.text.replace(find, replace)
        if replacements == 0:
            # Nothing changed -- uploading the unmodified document would
            # still create a new version/last-modified entry for no reason.
            return _success(replacements=0)
        item = _upload_document(document, file_path, snapshot)
        return _success(item=item, replacements=replacements)
    except Exception as e:
        logger.error("Error replacing text in Word document %s: %s", file_path, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
