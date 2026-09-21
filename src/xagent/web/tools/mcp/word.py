import hashlib
import io
import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from xml.etree.ElementTree import ParseError

import requests
from defusedxml import ElementTree as DefusedElementTree
from defusedxml.common import DefusedXmlException
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

# Layered expansion, XML-part, element, and paragraph ceilings keep both
# archive inflation and python-docx's later object graph bounded.
_MAX_DECOMPRESSED_BYTES = 25_000_000
_MAX_ARCHIVE_PARTS = 2_000
_MAX_XML_PART_BYTES = 8_000_000
_MAX_XML_ELEMENTS = 50_000
_MAX_DOCUMENT_PARAGRAPHS = 2_000
_MAX_DOCUMENT_TEXT_CHARS = 400_000

# Bound work added by one mutation before python-docx materializes and
# serializes it. The archive validator below remains the final source of truth;
# these limits keep a highly repetitive replacement/newline payload from first
# building a much larger in-memory XML object graph only to be rejected later.
_MAX_MUTATION_TEXT_CHARS = _MAX_DOCUMENT_TEXT_CHARS
_MAX_MUTATION_CONTROL_ELEMENTS = 10_000

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
_UPLOAD_RETRY_ATTEMPTS = 3
_UPLOAD_RETRY_BASE_SECONDS = 1.0
_UPLOAD_RETRY_MAX_SECONDS = 4.0
_DEFAULT_TEXT_PAGE_CHARS = 20_000
_MIN_TEXT_PAGE_CHARS = 10_000
_MAX_TEXT_PAGE_CHARS = 40_000
_DEFAULT_PARAGRAPH_PAGE_SIZE = 500
_MIN_PARAGRAPH_PAGE_SIZE = 100
_MAX_PARAGRAPH_PAGE_SIZE = 500
_MIN_PAGINATION_OUTPUT_LENGTH = 16_384
_MIN_TOOL_OUTPUT_LENGTH = 2_048


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
    generation: str = ""


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    encoded = json.dumps(payload, ensure_ascii=False)
    max_length = get_tool_max_output_length()
    if len(encoded) <= max_length:
        return encoded
    for fallback in (
        {"status": "error", "message": "Word tool error exceeds output limit"},
        {"status": "error"},
        {},
    ):
        encoded = json.dumps(fallback, ensure_ascii=False)
        if len(encoded) <= max_length:
            return encoded
    return "0"


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
    requiring a .docx filename and rejecting '.'/'..' segments, a trailing
    folder separator, and a filename ending in a period (Graph/SharePoint's
    backing storage can silently normalize a trailing dot away, so
    "Report.docx." could silently resolve to a different "Report.docx")."""
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
    if not value.casefold().endswith(".docx"):
        raise ValueError("file_path must name a .docx Word document")
    return value


def _item_path(file_path: str, site_id: str | None, drive_id: str | None) -> str:
    normalized = _normalize_relative_path(file_path)
    if site_id is not None:
        site_base = _site_subresource_base(site_id)
        drive_base = (
            f"{site_base}/drives/{url_path_id(drive_id, 'drive_id')}"
            if drive_id is not None
            else f"{site_base}/drive"
        )
    elif drive_id is not None:
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
            xml_infos = [
                info
                for info in infos
                if info.filename.lower().endswith((".xml", ".rels"))
            ]
            if any(info.file_size > _MAX_XML_PART_BYTES for info in xml_infos):
                raise ValueError(
                    "The document contains an XML part beyond the safe processing limit"
                )

            element_count = 0
            paragraph_count = 0
            document_text_characters = 0
            paragraph_tag = (
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
            )
            text_tags = {
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t",
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}delText",
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}instrText",
            }
            for info in xml_infos:
                with archive.open(info) as part:
                    for _event, element in DefusedElementTree.iterparse(
                        part,
                        events=("end",),
                        forbid_dtd=True,
                        forbid_entities=True,
                        forbid_external=True,
                    ):
                        element_count += 1
                        if element_count > _MAX_XML_ELEMENTS:
                            raise ValueError(
                                "The document contains too many XML elements to process safely"
                            )
                        if (
                            info.filename == "word/document.xml"
                            and element.tag == paragraph_tag
                        ):
                            paragraph_count += 1
                            if paragraph_count > _MAX_DOCUMENT_PARAGRAPHS:
                                raise ValueError(
                                    "The document contains too many paragraphs to process safely"
                                )
                        if (
                            info.filename == "word/document.xml"
                            and element.tag in text_tags
                            and element.text
                        ):
                            document_text_characters += len(element.text)
                            if document_text_characters > _MAX_DOCUMENT_TEXT_CHARS:
                                raise ValueError(
                                    "The document contains too much main-body text "
                                    "to process safely"
                                )
                        element.clear()
    except (zipfile.BadZipFile, ParseError, DefusedXmlException) as exc:
        raise ValueError("The file is not a valid Word document OOXML archive") from exc


def _download_document(
    file_path: str,
    site_id: str | None,
    drive_id: str | None,
    *,
    need_etag: bool = True,
    expected_generation: str | None = None,
) -> tuple[DocumentType, _EditSnapshot | None, str]:
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
            "GET",
            item_path,
            params={"$select": "id,eTag,parentReference,publication"},
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
        publication = (
            metadata.get("publication") if isinstance(metadata, dict) else None
        )
        if isinstance(publication, dict) and publication.get("level") == "checkout":
            raise ValueError(
                f"{file_path!r} is already checked out; check it in or discard "
                "that checkout before editing it with this connector"
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
    generation = hashlib.sha256(content).hexdigest()
    if expected_generation is not None and generation != expected_generation:
        raise ValueError(
            f"{file_path!r} changed since the referenced paragraph page was read; "
            "restart pagination and retry with its new generation"
        )
    _validate_docx_archive(content)
    if snapshot is not None:
        snapshot = _EditSnapshot(
            item_path=snapshot.item_path,
            etag=snapshot.etag,
            generation=generation,
        )
    try:
        return Document(io.BytesIO(content)), snapshot, generation
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

    Graph documents If-Match only for upload-session creation, not for the
    later preauthenticated final PUT. Acquire Graph's file checkout lock and
    revalidate the exact downloaded bytes after the lock is held, so no edit
    can slip between the optimistic precondition and final commit. The upload
    session remains item-id addressed and If-Match protected as a second fence.
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
    _validate_docx_archive(content)
    conflict_message = (
        f"{file_path!r} was changed by someone else since this edit "
        "started; re-read it and retry"
    )
    locked_etag = _checkout_edit_snapshot(snapshot, conflict_message)
    locked_snapshot = _EditSnapshot(
        item_path=snapshot.item_path,
        etag=locked_etag,
        generation=snapshot.generation,
    )
    try:
        result = _upload_checked_out_document(
            content, locked_snapshot, conflict_message=conflict_message
        )
        _check_in_edit_snapshot(snapshot, content)
        return result
    except Exception:
        try:
            _discard_edit_checkout(snapshot)
        except Exception:
            raise RuntimeError(
                "Word edit failed and its checkout could not be released; "
                "verify the document in Word before retrying"
            ) from None
        raise


def _check_in_edit_snapshot(snapshot: _EditSnapshot, content: bytes) -> None:
    try:
        _graph_request(
            "POST",
            f"{snapshot.item_path}/checkin",  # codespell:ignore checkin
            body={"comment": "Updated by Xagent Word connector"},
        )
        return
    except (requests.RequestException, _GraphRequestError):
        pass

    # The check-in can commit and then lose its response. Treat it as success
    # only when both the exact intended bytes and an explicit published state
    # can be observed; otherwise the caller's cleanup path fails closed.
    try:
        remote = _graph_request("GET", f"{snapshot.item_path}/content", raw=True)
        metadata = _graph_request(
            "GET", snapshot.item_path, params={"$select": "publication"}
        )
    except (ValueError, requests.RequestException, _GraphRequestError):
        raise RuntimeError(
            "Graph did not confirm the Word check-in completed"
        ) from None
    publication = metadata.get("publication") if isinstance(metadata, dict) else None
    if (
        remote == content
        and isinstance(publication, dict)
        and publication.get("level") == "published"
    ):
        return
    raise RuntimeError("Graph did not confirm the Word check-in completed")


def _discard_edit_checkout(snapshot: _EditSnapshot) -> None:
    _graph_request("POST", f"{snapshot.item_path}/discardCheckout")


def _checkout_edit_snapshot(snapshot: _EditSnapshot, conflict_message: str) -> str:
    """Lock the item, then prove it still contains the downloaded snapshot."""
    if not snapshot.generation:
        raise RuntimeError("Word edit snapshot is missing its content generation")
    try:
        _graph_request("POST", f"{snapshot.item_path}/checkout")
    except requests.RequestException:
        # The lock may have been acquired before the transport failed. A
        # delegated discard can release only this user's checkout; HTTP 400
        # means no checkout existed, while any other cleanup failure remains
        # an explicit unknown state requiring manual verification.
        try:
            _discard_edit_checkout(snapshot)
        except _GraphRequestError as cleanup_error:
            if cleanup_error.status_code == 400:
                raise RuntimeError(
                    "Graph could not acquire the checkout required for a "
                    "concurrency-safe Word edit"
                ) from None
            raise RuntimeError(
                "Word checkout outcome is unknown; verify the document in "
                "Word before retrying"
            ) from None
        except requests.RequestException:
            raise RuntimeError(
                "Word checkout outcome is unknown; verify the document in "
                "Word before retrying"
            ) from None
        raise RuntimeError(
            "Graph did not confirm the Word checkout; the possible checkout "
            "was discarded, so the edit was not applied"
        ) from None
    except _GraphRequestError as exc:
        if exc.status_code in (409, 412, 423):
            raise ValueError(conflict_message) from exc
        raise RuntimeError(
            "Graph could not acquire the checkout required for a "
            "concurrency-safe Word edit"
        ) from exc
    try:
        metadata = _graph_request("GET", snapshot.item_path, params={"$select": "eTag"})
        locked_content = _graph_request(
            "GET", f"{snapshot.item_path}/content", raw=True
        )
        locked_etag = metadata.get("eTag") if isinstance(metadata, dict) else None
        if not isinstance(locked_etag, str) or not locked_etag:
            raise RuntimeError("Graph did not return an eTag for the checked-out file")
        if hashlib.sha256(locked_content).hexdigest() != snapshot.generation:
            raise ValueError(conflict_message)
        return locked_etag
    except Exception:
        try:
            _discard_edit_checkout(snapshot)
        except Exception:
            raise RuntimeError(
                "Word document changed while acquiring its checkout and the "
                "checkout could not be released; verify it in Word"
            ) from None
        raise


def _upload_checked_out_document(
    content: bytes,
    snapshot: _EditSnapshot,
    *,
    conflict_message: str,
) -> dict[str, Any]:
    try:
        session = _graph_request(
            "POST",
            f"{snapshot.item_path}/createUploadSession",
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

    offset = 0
    for attempt in range(_UPLOAD_RETRY_ATTEMPTS):
        response: requests.Response | None = None
        try:
            response = requests.put(
                upload_url,
                data=content[offset:],
                headers={
                    "Content-Range": (
                        f"bytes {offset}-{len(content) - 1}/{len(content)}"
                    ),
                    "Content-Type": _WORD_MIME_TYPE,
                },
                timeout=_BINARY_TIMEOUT_SECONDS,
            )
            if response.status_code == 412:
                raise ValueError(conflict_message)
            if response.status_code in (200, 201):
                try:
                    result = response.json()
                except ValueError:
                    result = None
                if isinstance(result, dict) and result.get("id"):
                    safe_item = dict(result)
                    safe_item.pop("@microsoft.graph.downloadUrl", None)
                    return safe_item
            elif response.status_code == 202:
                try:
                    progress = response.json()
                except ValueError:
                    progress = None
                next_offset = _next_upload_offset(progress, len(content))
                if next_offset is not None:
                    offset = next_offset
                    continue
            elif response.status_code < 500 and response.status_code != 416:
                raise _GraphRequestError(
                    f"Word document upload failed with HTTP {response.status_code}",
                    status_code=response.status_code,
                )
        except requests.RequestException:
            pass

        reconciled = _reconcile_uploaded_document(snapshot, content)
        if reconciled is not None:
            return reconciled
        try:
            status = requests.get(upload_url, timeout=_BINARY_TIMEOUT_SECONDS)
            if status.status_code == 200:
                try:
                    progress = status.json()
                except ValueError:
                    progress = None
                next_offset = _next_upload_offset(progress, len(content))
                if next_offset is not None:
                    offset = next_offset
                    _sleep_before_upload_retry(attempt)
                    continue
        except requests.RequestException:
            pass

        _sleep_before_upload_retry(attempt)

    raise RuntimeError(
        "Word document upload outcome is unknown; its checkout must be "
        "discarded before retrying"
    )


def _sleep_before_upload_retry(attempt: int) -> None:
    """Apply bounded exponential backoff before another upload attempt."""
    if attempt >= _UPLOAD_RETRY_ATTEMPTS - 1:
        return
    delay = min(
        _UPLOAD_RETRY_BASE_SECONDS * (2.0**attempt),
        _UPLOAD_RETRY_MAX_SECONDS,
    )
    time.sleep(delay)


def _next_upload_offset(payload: Any, total_bytes: int) -> int | None:
    if not isinstance(payload, dict):
        return None
    ranges = payload.get("nextExpectedRanges")
    if not isinstance(ranges, list) or not ranges or not isinstance(ranges[0], str):
        return None
    start = ranges[0].split("-", 1)[0]
    if not start.isdigit():
        return None
    offset = int(start)
    return offset if 0 <= offset < total_bytes else None


def _reconcile_uploaded_document(
    snapshot: _EditSnapshot, content: bytes
) -> dict[str, Any] | None:
    """Prove an ambiguous upload committed by comparing its exact bytes."""
    try:
        remote = _graph_request("GET", f"{snapshot.item_path}/content", raw=True)
        if remote != content:
            return None
        item = _graph_request("GET", snapshot.item_path, params={"$select": "id,eTag"})
    except (ValueError, requests.RequestException, _GraphRequestError):
        return None
    if not isinstance(item, dict) or not item.get("id"):
        return None
    safe_item = dict(item)
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
    _validate_docx_archive(content)
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
    # query string) -- never include it in an exception or chained cause.
    # A final PUT can commit server-side and then lose its response, so creation
    # uses the same bounded session-status/resume loop as edits instead of
    # reporting a definite failure for an ambiguous transport outcome.
    offset = 0
    outcome_ambiguous = False
    for attempt in range(_UPLOAD_RETRY_ATTEMPTS):
        try:
            response = requests.put(
                upload_url,
                data=content[offset:],
                headers={
                    "Content-Range": (
                        f"bytes {offset}-{len(content) - 1}/{len(content)}"
                    ),
                    "Content-Type": _WORD_MIME_TYPE,
                },
                timeout=_BINARY_TIMEOUT_SECONDS,
            )
            if response.status_code == 409:
                raise ValueError(
                    f"{file_path!r} already exists; use the other word_* tools "
                    "to edit it instead of recreating it"
                )
            if response.status_code in (200, 201):
                try:
                    result = response.json()
                except ValueError:
                    result = None
                if isinstance(result, dict) and result.get("id"):
                    safe_item = dict(result)
                    safe_item.pop("@microsoft.graph.downloadUrl", None)
                    return safe_item
                outcome_ambiguous = True
            elif response.status_code == 202:
                try:
                    progress = response.json()
                except ValueError:
                    progress = None
                next_offset = _next_upload_offset(progress, len(content))
                if next_offset is not None:
                    offset = next_offset
                    continue
            elif response.status_code < 500 and response.status_code != 416:
                if outcome_ambiguous:
                    break
                raise _GraphRequestError(
                    f"Word document upload failed with HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            else:
                outcome_ambiguous = True
        except requests.RequestException:
            outcome_ambiguous = True

        # Only this upload URL can describe this caller's session. Looking up
        # the destination and comparing bytes is not attribution: two racing
        # blank creates have identical DOCX bytes, so that could incorrectly
        # claim the other caller's item after this session lost a 409 response.
        try:
            status = requests.get(upload_url, timeout=_BINARY_TIMEOUT_SECONDS)
            if status.status_code == 200:
                try:
                    progress = status.json()
                except ValueError:
                    progress = None
                next_offset = _next_upload_offset(progress, len(content))
                if next_offset is not None:
                    offset = next_offset
                    _sleep_before_upload_retry(attempt)
                    continue
            if outcome_ambiguous and status.status_code in (404, 410):
                break
        except requests.RequestException:
            pass
        _sleep_before_upload_retry(attempt)

    raise RuntimeError(
        "Word document creation outcome is unknown; verify whether the file "
        "exists before retrying"
    )


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


def _paragraph_has_unsupported_structure(paragraph: Any) -> bool:
    """Return whether visible flow exists outside ``paragraph.runs``.

    python-docx exposes only direct ``w:r`` children through paragraph.runs.
    Checking that invariant, rather than maintaining a denylist of wrapper
    names, also covers extension elements and future WordprocessingML wrappers.
    Office Math uses its own run vocabulary, so it is rejected explicitly.
    Text boxes are a separate content stream and are checked by the containing
    drawing run's non-text-content guard.
    """
    for run in paragraph._p.xpath(".//w:r[not(ancestor::w:txbxContent)]"):
        if run.getparent() is not paragraph._p:
            return True
    return bool(
        paragraph._p.xpath(
            ".//m:oMath[not(ancestor::w:txbxContent)] | "
            ".//m:oMathPara[not(ancestor::w:txbxContent)]"
        )
    )


def _set_paragraph_text(paragraph: Any, text: str) -> None:
    """Replace a paragraph's visible text, keeping its first run's
    character formatting (font, bold, etc.) and its own paragraph style.
    Any additional runs are emptied rather than removed, since python-docx
    doesn't expose a supported run-removal API; an empty run has no visible
    effect but a paragraph with zero runs to begin with needs one added.

    Refuses (rather than silently corrupting the document) a paragraph with
    visible content outside paragraph.runs (including wrappers, tracked
    changes, hyperlinks, content controls, extension markup, or Office Math),
    which the rewrite below would leave stale rather than replacing, or a
    paragraph containing a run with an image, break, or field character,
    which the plain-text runs[0].text assignment below would silently
    delete.
    """
    _validate_mutation_text(text, field_name="text")
    if _paragraph_has_unsupported_structure(paragraph):
        raise ValueError(
            "paragraph contains wrapped or mathematical content outside "
            "python-docx's paragraph.runs (for example a hyperlink, tracked "
            "change, content control, extension wrapper, or Office Math); "
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


def _mutation_control_element_count(text: str) -> int:
    return text.count("\n") + text.count("\r") + text.count("\t")


def _validate_mutation_text(text: str, *, field_name: str) -> None:
    if not isinstance(text, str):
        raise ValueError(f"{field_name} must be a string")
    if len(text) > _MAX_MUTATION_TEXT_CHARS:
        raise ValueError(
            f"{field_name} exceeds the {_MAX_MUTATION_TEXT_CHARS:,}-character "
            "Word mutation limit"
        )
    if _mutation_control_element_count(text) > _MAX_MUTATION_CONTROL_ELEMENTS:
        raise ValueError(
            f"{field_name} contains too many tabs or line breaks for one Word mutation"
        )


def _ensure_can_add_body_paragraph(document: DocumentType) -> None:
    paragraph_count = len(document.element.body.xpath(".//w:p"))
    if paragraph_count >= _MAX_DOCUMENT_PARAGRAPHS:
        raise ValueError(
            "The document is already at the safe paragraph limit; adding "
            "another paragraph would make it unreadable by this connector"
        )


def _document_main_body_text_characters(document: DocumentType) -> int:
    return sum(
        len(element.text or "")
        for element in document.element.body.xpath(
            ".//w:t | .//w:delText | .//w:instrText"
        )
    )


def _ensure_projected_text_budget(document: DocumentType, character_delta: int) -> None:
    if (
        _document_main_body_text_characters(document) + character_delta
        > _MAX_DOCUMENT_TEXT_CHARS
    ):
        raise ValueError(
            "The requested mutation would exceed the safe main-body text limit"
        )


def _text_page_response(
    text: str, *, offset: int, limit: int, table_count: int, generation: str
) -> str:
    """Return a recoverable character page that fits the tool output cap."""
    start = min(clamp_offset(offset), len(text))
    requested = max(
        _MIN_TEXT_PAGE_CHARS,
        clamp_limit(limit, max_limit=_MAX_TEXT_PAGE_CHARS),
    )
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
            generation=generation,
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
    paragraphs: list[Any],
    *,
    offset: int,
    limit: int,
    text_offset: int,
    generation: str,
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
    page_limit = max(
        _MIN_PARAGRAPH_PAGE_SIZE,
        clamp_limit(limit, max_limit=_MAX_PARAGRAPH_PAGE_SIZE),
    )
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
            generation=generation,
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


def _require_pagination_output_budget() -> None:
    """Reject caps that would turn one bounded document into tiny pages.

    Each MCP invocation starts a fresh process and reparses the document, so
    allowing arbitrarily small output caps would defeat the minimum page sizes
    above. The check runs before download/parse work in both read handlers.
    """
    if get_tool_max_output_length() < _MIN_PAGINATION_OUTPUT_LENGTH:
        raise ValueError(
            "Word pagination requires XAGENT_TOOL_MAX_OUTPUT_LENGTH to be at "
            f"least {_MIN_PAGINATION_OUTPUT_LENGTH} characters"
        )


def _require_tool_output_budget() -> None:
    """Reject caps too small for _success() to guarantee a valid envelope.

    Unlike _error(), _success() always serializes its full payload -- an
    accepted cap below this floor lets a real success response (even a
    no-op) exceed it, so the outer raw-truncating output filter mangles the
    result into invalid JSON instead of the filter ever seeing a complete
    envelope. The check runs before any Graph work in every non-paginated
    Word tool.
    """
    if get_tool_max_output_length() < _MIN_TOOL_OUTPUT_LENGTH:
        raise ValueError(
            "Word tools require XAGENT_TOOL_MAX_OUTPUT_LENGTH to be at least "
            f"{_MIN_TOOL_OUTPUT_LENGTH} characters"
        )


@mcp.tool()
def word_create_document(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """Create a new, blank Word document at file_path. Fails if a file
    already exists there -- edit it with the other word_* tools instead of
    recreating it."""
    try:
        _require_tool_output_budget()
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
    expected_generation: str | None = None,
) -> str:
    """Get a page of top-level paragraph text from the main document body.

    Tables, headers, footers, text boxes, notes, and tracked-change content
    are not included. Pass next_offset and generation from a response with
    has_more=true back as offset/expected_generation. A changed document is
    rejected instead of mixing pages from different versions. limit is a
    character count and is normalized to a safe range of 10,000-40,000.
    Pagination requires an output cap of at least 16,384 characters."""
    try:
        _require_pagination_output_budget()
        if clamp_offset(offset) and not expected_generation:
            raise ValueError("expected_generation is required to continue pagination")
        document, _snapshot, generation = _download_document(
            file_path,
            site_id,
            drive_id,
            need_etag=False,
            expected_generation=expected_generation,
        )
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return _text_page_response(
            text,
            offset=offset,
            limit=limit,
            table_count=len(document.tables),
            generation=generation,
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
    expected_generation: str | None = None,
) -> str:
    """List top-level main-body paragraphs with index, text, and style.

    Tables, headers, footers, text boxes, notes, and tracked-change content
    are excluded. Pass next_offset, next_text_offset, and generation from a
    response with has_more=true back as offset/text_offset/expected_generation.
    next_text_offset is non-zero only when one unusually large paragraph had
    to be split. limit is normalized to 100-500 paragraphs. Pagination
    requires an output cap of at least 16,384 characters. Pass generation to
    word_set_paragraph_text with the index."""
    try:
        _require_pagination_output_budget()
        if (
            clamp_offset(offset) or clamp_offset(text_offset)
        ) and not expected_generation:
            raise ValueError("expected_generation is required to continue pagination")
        document, _snapshot, generation = _download_document(
            file_path,
            site_id,
            drive_id,
            need_etag=False,
            expected_generation=expected_generation,
        )
        return _paragraph_page_response(
            document.paragraphs,
            offset=offset,
            limit=limit,
            text_offset=text_offset,
            generation=generation,
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
    expected_generation: str | None = None,
) -> str:
    """Replace a Word document paragraph's text by index (from
    word_list_paragraphs). Keeps the paragraph's style and its first run's
    character formatting; does not preserve formatting that varied across
    multiple runs within the paragraph. expected_generation is required from
    the word_list_paragraphs response that supplied paragraph_index, preventing
    an intervening edit from redirecting the index to different content."""
    try:
        _require_tool_output_budget()
        if not expected_generation:
            raise ValueError(
                "expected_generation from word_list_paragraphs is required"
            )
        document, snapshot, _generation = _download_document(
            file_path,
            site_id,
            drive_id,
            expected_generation=expected_generation,
        )
        assert snapshot is not None
        paragraphs = document.paragraphs
        if not 0 <= paragraph_index < len(paragraphs):
            raise ValueError(
                f"paragraph_index {paragraph_index} is out of range for a document "
                f"with {len(paragraphs)} paragraphs"
            )
        _ensure_projected_text_budget(
            document, len(text) - len(paragraphs[paragraph_index].text)
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
        _require_tool_output_budget()
        document, snapshot, _generation = _download_document(
            file_path, site_id, drive_id
        )
        assert snapshot is not None
        _validate_mutation_text(text, field_name="text")
        _ensure_can_add_body_paragraph(document)
        _ensure_projected_text_budget(document, len(text))
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
        _require_tool_output_budget()
        if not 0 <= level <= 9:
            raise ValueError("level must be between 0 and 9")
        document, snapshot, _generation = _download_document(
            file_path, site_id, drive_id
        )
        assert snapshot is not None
        _validate_mutation_text(text, field_name="text")
        _ensure_can_add_body_paragraph(document)
        _ensure_projected_text_budget(document, len(text))
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
    """Find and replace text across top-level main-body paragraphs only.

    Headers, footers, tables, text boxes, notes, and tracked-change content
    are outside this tool's scope.

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
        _require_tool_output_budget()
        if not find:
            raise ValueError("find must not be empty")
        _validate_mutation_text(find, field_name="find")
        _validate_mutation_text(replace, field_name="replace")
        document, snapshot, _generation = _download_document(
            file_path, site_id, drive_id
        )
        assert snapshot is not None
        replacements = 0
        added_characters = 0
        character_delta = 0
        added_control_elements = 0
        planned_replacements: list[tuple[Run, str]] = []
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
                    count = run.text.count(find)
                    updated_text = run.text.replace(find, replace)
                    replacements += count
                    character_delta += len(updated_text) - len(run.text)
                    added_characters += max(0, len(updated_text) - len(run.text))
                    added_control_elements += max(
                        0,
                        _mutation_control_element_count(updated_text)
                        - _mutation_control_element_count(run.text),
                    )
                    if added_characters > _MAX_MUTATION_TEXT_CHARS:
                        raise ValueError(
                            "replacement would add too much text in one Word mutation"
                        )
                    if added_control_elements > _MAX_MUTATION_CONTROL_ELEMENTS:
                        raise ValueError(
                            "replacement would add too many tabs or line breaks "
                            "in one Word mutation"
                        )
                    planned_replacements.append((run, updated_text))
        if replacements == 0:
            # Nothing changed -- uploading the unmodified document would
            # still create a new version/last-modified entry for no reason.
            return _success(replacements=0)
        _ensure_projected_text_budget(document, character_delta)
        for run, updated_text in planned_replacements:
            run.text = updated_text
        item = _upload_document(document, file_path, snapshot)
        return _success(item=item, replacements=replacements)
    except Exception as e:
        logger.error("Error replacing text in Word document %s: %s", file_path, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
