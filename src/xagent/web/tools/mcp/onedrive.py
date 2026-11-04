import base64
import errno
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ....core.utils import public_requests as requests
from .utils import allowed_dirs_from_env, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onedrive-mcp")


mcp = FastMCP("onedrive-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30
# Sized for a single binary PUT of up to a few megabytes over a slow link,
# not just a small JSON Graph call -- DEFAULT_TIMEOUT_SECONDS' 30s is
# realistic for the API calls elsewhere in this module but can legitimately
# be too short for a multi-megabyte binary transfer. Used for both the
# simple-PUT branch (up to _SIMPLE_UPLOAD_MAX_BYTES) and each chunk of the
# resumable upload-session path.
_BINARY_UPLOAD_TIMEOUT_SECONDS = 120
# Microsoft's own docs disagree on the simple content PUT's real limit:
# the OneDrive API concepts page ("Uploading files") says simple upload is
# "available for items with less than 4 MB of content", while the Graph
# v1.0 API reference for the same endpoint (driveitem-put-content) says it
# "supports files up to 250 MB in size". Rather than trying to resolve
# which page is authoritative, this deliberately keeps the smaller, more
# conservative bound -- some Graph deployments enforce it as the decimal
# 4,000,000 bytes rather than 4 MiB (4,194,304 bytes), so the decimal
# figure is used here too. This means a file anywhere in the 4MB-250MB gap
# always takes the resumable upload-session path instead of ever risking a
# rejection at the simple-PUT boundary; the only cost of guessing wrong
# this way is an unnecessary chunked upload, never a failed one.
_SIMPLE_UPLOAD_MAX_BYTES = 4_000_000
# Chunk size for the upload-session path. Must be a multiple of 320 KiB
# (327,680 bytes) per Graph's requirement for every non-final chunk; 5 MiB
# is exactly 16 * 320 KiB.
_UPLOAD_SESSION_CHUNK_SIZE = 5 * 1024 * 1024
_UPLOAD_FRAGMENT_MAX_ATTEMPTS = 3
_UPLOAD_RECONCILE_MAX_ATTEMPTS = 3
# Destination metadata can lag behind a committed final fragment. Keep this
# confirmation budget separate from fragment/status retries so an ambiguous
# completion can wait up to 45 seconds (1+2+4+8+10+10+10) for QuickXorHash
# without weakening the exact-content check or making ordinary chunks slower.
_UPLOAD_COMPLETION_MAX_ATTEMPTS = 8
# Forward progress resets the consecutive-failure counter, so keep a separate
# cap on recovery cycles within one buffered fragment to prevent a server that
# advances by tiny ranges from extending a synchronous call without bound.
_UPLOAD_CHUNK_MAX_RECOVERY_CYCLES = 12
_UPLOAD_RETRY_BASE_SECONDS = 1.0
_UPLOAD_RETRY_MAX_SECONDS = 10.0
# Not a Graph API limit (OneDrive itself supports files far larger than
# this via the resumable upload session) -- a guard-rail against a
# mistargeted file (a generated artifact pointed at the wrong path, an
# entire log directory, etc.) tying up a chunked upload for a long time
# with no feedback until it eventually finishes or fails. Unlike gmail.py's
# _MAX_ATTACHMENT_BYTES (a real derivation of Gmail's documented 25MB
# message-size limit), this is a deliberately arbitrary product choice, not
# a limit either this module or Graph actually enforces elsewhere.
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# Appended to a 404 from a path-addressed lookup. In #2875 the agent guessed a
# "Documents/" folder and then passed its own workspace path to this
# connector, after onedrive_search_files missed a file at the drive root.
_PATH_NOT_FOUND_HINT = (
    "No OneDrive item exists at this path. Paths are relative to the drive "
    "root (e.g. 'Reports/Q3.pptx'), not a local task-workspace path. "
    "onedrive_search_files can miss existing items, so list the parent folder "
    "with onedrive_list_items (omit folder_path for the root) to get the exact "
    "path."
)
# Returned with an empty search result. Search missed an existing root-level
# file in #2875, and it cannot be relied on to locate an item.
_SEARCH_MISS_HINT = (
    "No items matched. onedrive_search_files can miss existing items, even "
    "by their exact name. Before guessing a path, list the parent folder "
    "with onedrive_list_items (omit folder_path for the root) to get the "
    "exact path."
)
_DriveFilePath = Annotated[
    str,
    Field(
        description=(
            "File path relative to the drive root, with '/' between folders; "
            "not a local task-workspace path."
        )
    ),
]

_UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS"
_OUTPUT_DIR_ENV_VAR = "XAGENT_ONEDRIVE_OUTPUT_DIR"


class _UploadError(RuntimeError):
    """Upload failure whose message is safe to expose to the caller."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class _GraphRequestError(RuntimeError):
    """Graph HTTP failure that keeps its status and Graph error code."""

    def __init__(self, message: str, status_code: int, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class _QuickXorHash:
    """Streaming implementation of OneDrive's 160-bit QuickXorHash."""

    _WIDTH_BITS = 160
    _DIGEST_BYTES = _WIDTH_BITS // 8
    # rotate(byte, index * 11) repeats after 160 input bytes because 11
    # and 160 are coprime. This is distinct from the 20-byte digest width.
    _INPUT_PERIOD_BYTES = _WIDTH_BITS
    _MASK = (1 << _WIDTH_BITS) - 1

    def __init__(self) -> None:
        self._column_xor = 0
        self._tail = b""
        self._length = 0

    def update(self, data: bytes | bytearray) -> None:
        """Add bytes while reducing full 160-byte periods in C-sized blocks."""
        self._length += len(data)
        if self._tail:
            needed = self._INPUT_PERIOD_BYTES - len(self._tail)
            if len(data) < needed:
                self._tail += bytes(data)
                return
            self._column_xor ^= int.from_bytes(
                self._tail + bytes(data[:needed]), "little"
            )
            data = data[needed:]
            self._tail = b""

        full_length = len(data) - (len(data) % self._INPUT_PERIOD_BYTES)
        view = memoryview(data)
        for offset in range(0, full_length, self._INPUT_PERIOD_BYTES):
            self._column_xor ^= int.from_bytes(
                view[offset : offset + self._INPUT_PERIOD_BYTES], "little"
            )
        self._tail = bytes(view[full_length:])

    def base64_digest(self) -> str:
        """Return the Base64 value exposed as ``file.hashes.quickXorHash``."""
        columns = self._column_xor
        if self._tail:
            columns ^= int.from_bytes(self._tail, "little")

        value = 0
        for index, byte in enumerate(
            columns.to_bytes(self._INPUT_PERIOD_BYTES, "little")
        ):
            shift = (index * 11) % self._WIDTH_BITS
            rotated = (
                byte
                if shift == 0
                else ((byte << shift) | (byte >> (self._WIDTH_BITS - shift)))
            )
            value ^= rotated & self._MASK

        digest = bytearray(value.to_bytes(self._DIGEST_BYTES, "little"))
        for index, byte in enumerate(self._length.to_bytes(8, "little")):
            digest[self._DIGEST_BYTES - 8 + index] ^= byte
        return base64.b64encode(digest).decode("ascii")


# stdlib mimetypes.guess_type() only recognizes these extensions when a
# system mime.types file happens to be installed (e.g. Apache's, common on
# a dev laptop) -- on a minimal/slim host with no such file (a stripped-down
# container image, verified directly: MimeTypes(filenames=()) returns
# (None, None) for every one of these), it silently can't identify them at
# all. Checked before falling back to mimetypes.guess_type() wherever this
# module resolves a *real* mime type (i.e. for onedrive_upload_file's
# Content-Type header, never for the binary/text guard below -- see
# _name_looks_binary's own docstring for why that guard doesn't use
# mimetypes at all), so a real .xlsx/.docx/etc. doesn't get mislabeled
# "application/octet-stream" just because the host is missing a file this
# module has no control over.
_MIME_TYPE_OVERRIDES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".epub": "application/epub+zip",
}


def _guess_mime_type(name: str) -> str | None:
    """Guess ``name``'s real mime type for actual Content-Type resolution:
    _MIME_TYPE_OVERRIDES first, for the formats stdlib mimetypes can't
    reliably identify on its own, then stdlib itself.

    Only the type element of mimetypes.guess_type()'s result is normally
    usable directly -- but when it also reports a non-None *encoding*
    (e.g. "gzip" for ".gz", "compress" for ".Z"), the type element
    describes the *decompressed* content, not what's actually going out
    over the wire: mimetypes.guess_type("report.pdf.gz") returns
    ("application/pdf", "gzip"), and blindly using "application/pdf" as
    Content-Type would tell any client trusting that header to parse a raw
    gzip stream as an uncompressed PDF. Falls back to the standard
    encoding-to-mimetype mapping in that case instead.
    """
    suffix = Path(name).suffix.lower()
    override = _MIME_TYPE_OVERRIDES.get(suffix)
    if override is not None:
        return override
    guessed_type, guessed_encoding = mimetypes.guess_type(name)
    if guessed_encoding is not None:
        return {
            "gzip": "application/gzip",
            "bzip2": "application/x-bzip2",
            "xz": "application/x-xz",
            "compress": "application/x-compress",
        }.get(guessed_encoding, "application/octet-stream")
    return guessed_type


def _split_stem_suffix(base: str) -> tuple[str, str]:
    """Like Path(base).stem/.suffix, except a name that's *entirely* a
    leading dot plus extension (e.g. ".pdf") is treated as having that
    extension. pathlib's own split refuses to do this -- it follows the
    Unix dotfile convention where a single leading dot with nothing before
    it never counts as an extension separator, leaving Path(".pdf").suffix
    empty -- but silently losing the extension here would let a name like
    that slip past _name_looks_binary's guard as "extensionless" (accepted)
    when it's actually naming a real binary format. Only that narrow shape
    is special-cased; e.g. "..pdf" or "..." already split the way we want
    via plain pathlib and are left alone.
    """
    suffix = Path(base).suffix
    if not suffix and base.startswith(".") and base.count(".") == 1 and len(base) > 1:
        return "", base
    return Path(base).stem, suffix


# Stable text-extension allowlist shared in shape with the Google Drive
# guard. Host MIME databases vary, so they cannot safely decide whether the
# text-only tool may use a filename. Unknown extensions are rejected; names
# without an extension remain valid for files such as README and Dockerfile.
# Ambiguous but commonly textual source extensions such as .ts and .bat are
# allowed, while formats with common binary variants such as .crt and .plist
# are not.
_KNOWN_TEXT_EXTENSIONS = {
    # plain text / docs / dotfiles
    ".txt", ".md", ".markdown", ".mdx", ".rst", ".adoc", ".rtf", ".log",
    ".lock", ".gitignore", ".gitattributes", ".editorconfig",
    ".dockerignore", ".env", ".ini", ".cfg", ".conf", ".properties",
    ".toml",
    # structured/data formats
    ".json", ".json5", ".xml", ".yaml", ".yml", ".csv", ".tsv", ".dtd",
    ".xsd", ".xsl", ".xslt", ".proto", ".graphql", ".gql", ".thrift",
    ".avsc", ".ipynb", ".jsonl", ".ndjson", ".geojson",
    # web
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".svg",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".vue", ".svelte", ".astro",
    # source code
    ".py", ".rb", ".php", ".java", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".cxx", ".cs", ".m", ".mm", ".go", ".rs", ".swift", ".kt", ".kts",
    ".scala", ".groovy", ".lua", ".r", ".jl", ".pl", ".pm", ".hs", ".fs",
    ".fsx", ".ml", ".mli", ".clj", ".cljs", ".erl", ".ex", ".exs", ".nim",
    ".zig", ".v", ".d", ".dart", ".elm", ".cr", ".tcl", ".scm", ".sc",
    ".rkt", ".lisp", ".el", ".asm", ".s", ".pas", ".f90", ".for", ".vb",
    ".vbs", ".cabal", ".nix",
    # shell / scripting / templates
    ".sh", ".bash", ".zsh", ".csh", ".ksh", ".fish",
    ".ps1", ".bat", ".cmd", ".awk", ".sed", ".sql", ".j2", ".tpl", ".srt",
    # build / infra
    ".tex", ".latex", ".bib", ".cls", ".sty", ".diff", ".patch",
    ".hcl", ".tf", ".tfvars", ".gradle", ".dockerfile", ".cmake",
    # misc
    ".pem", ".po",
}  # fmt: skip


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _caller_safe_drive_item(item: Any) -> Any:
    """Copy a driveItem (or a list of them) without any short-lived
    preauthenticated download URL.

    The signed link's field name differs per endpoint -- Graph's
    "@microsoft.graph.downloadUrl" on item and children responses, OneDrive
    Personal's upload-session "@content.downloadUrl" and
    "@content.downloadUrlNoAuth" -- and a shared item nests another driveItem
    under "remoteItem", so every key naming a download URL is dropped at any
    depth rather than one known name at the top level. This removes a name
    pattern instead of projecting an allowlist like powerpoint.py does,
    because list and search callers rely on the rest of the driveItem
    (file/folder facets, parentReference). sharepoint.py keeps an identical
    copy; change both together.
    """
    if isinstance(item, dict):
        return {
            key: _caller_safe_drive_item(value)
            for key, value in item.items()
            if "downloadurl" not in key.lower()
        }
    if isinstance(item, list):
        return [_caller_safe_drive_item(value) for value in item]
    return item


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _resource_locked_error(operation: str) -> str:
    """Return a stable, caller-actionable contract for a Graph 423."""
    return json.dumps(
        {
            "status": "error",
            "message": (
                f"The OneDrive item is locked, so it could not be {operation}. "
                "Close the document in Word or Teams and retry later, or use a "
                "different destination path to save a copy."
            ),
            "code": "resource_locked",
            "retryable": True,
            # Graph commonly omits the owner for driveItem 423 responses. Keep
            # a stable field without exposing or guessing an identity.
            "lock_owner": None,
            "suggested_actions": ["close_document_and_retry", "save_as_copy"],
        },
        ensure_ascii=False,
    )


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


def _raise_upload_status(response: Any) -> None:
    """Raise a credential-safe error for a rejected upload fragment.

    Upload-session URLs are preauthenticated secrets. An intermediary can
    echo them into an error body with arbitrary escaping, so no part of the
    response body is forwarded to the caller or logs.
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status_code = getattr(response, "status_code", "unknown")
        raise _UploadError(
            f"OneDrive upload fragment failed with HTTP {status_code}"
        ) from exc


def _safe_upload_error_message(exc: BaseException) -> str:
    """Describe an upload failure without copying a request URL from it."""
    if isinstance(exc, _UploadError):
        status_code = _upload_status_code(exc)
        message = str(exc)
        if status_code is not None and "HTTP" not in message:
            return f"{message} (last HTTP status {status_code})"
        return message
    cause = _request_error_cause(exc)
    if isinstance(cause, requests.Timeout):
        return "OneDrive upload fragment timed out"
    if isinstance(cause, requests.ConnectionError):
        return "OneDrive upload fragment connection failed"
    if isinstance(cause, requests.RequestException):
        return "OneDrive upload fragment request failed"
    return f"OneDrive upload failed ({type(exc).__name__})"


def _request_error_cause(exc: BaseException) -> BaseException:
    """Unwrap nested safe errors to their underlying requests exception."""
    cause = exc
    seen: set[int] = set()
    while cause.__cause__ is not None and id(cause) not in seen:
        seen.add(id(cause))
        cause = cause.__cause__
    return cause


def _is_retriable_upload_error(exc: BaseException) -> bool:
    """Whether ``exc`` might succeed on another bounded fragment attempt.

    A network-level failure (no response was ever received) and Graph's
    documented retriable statuses (429 and any 5xx) are retriable. The
    HTTPError behind an _UploadError is recovered through ``__cause__``.
    """
    status_code = _upload_status_code(exc)
    if status_code is not None:
        return status_code == 429 or 500 <= status_code < 600
    cause = _request_error_cause(exc)
    return isinstance(cause, (requests.ConnectionError, requests.Timeout))


def _retry_delay(headers: Any, retry_number: int) -> float:
    """Return a bounded seconds/HTTP-date Retry-After or exponential delay."""
    headers = headers or {}
    retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            retry_seconds = float(retry_after)
            if math.isfinite(retry_seconds):
                return min(max(retry_seconds, 0.0), _UPLOAD_RETRY_MAX_SECONDS)
        except (TypeError, ValueError):
            pass
        try:
            retry_at = parsedate_to_datetime(str(retry_after))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            retry_seconds = retry_at.timestamp() - time.time()
            if math.isfinite(retry_seconds):
                return min(max(retry_seconds, 0.0), _UPLOAD_RETRY_MAX_SECONDS)
        except (TypeError, ValueError, OverflowError):
            pass
    exponential_delay = _UPLOAD_RETRY_BASE_SECONDS * (2.0 ** (retry_number - 1))
    return min(exponential_delay, _UPLOAD_RETRY_MAX_SECONDS)


def _upload_retry_delay(exc: BaseException, retry_number: int) -> float:
    """Return a retry delay from an upload exception."""
    cause = _request_error_cause(exc)
    response = getattr(cause, "response", None)
    return _retry_delay(getattr(response, "headers", None), retry_number)


def _upload_status_code(exc: BaseException) -> int | None:
    """Return an HTTP status from a credential-safe upload error."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        current = current.__cause__
    cause = _request_error_cause(exc)
    status_code = getattr(getattr(cause, "response", None), "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _next_expected_upload_offset(payload: Any, total: int) -> int:
    """Parse the first missing offset from an upload-session status payload."""
    try:
        ranges = payload["nextExpectedRanges"]
        if not isinstance(ranges, list):
            raise TypeError("nextExpectedRanges must be a list")
        offsets = [int(value.split("-", 1)[0]) for value in ranges]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise _UploadError("OneDrive returned invalid upload-session progress") from exc
    if not offsets:
        return total
    if any(offset < 0 or offset > total for offset in offsets):
        raise _UploadError("OneDrive returned invalid upload-session progress")
    return min(offsets)


def _current_upload_item(remote_path: str) -> dict[str, Any] | None:
    """Read destination metadata, treating an absent item as no baseline."""
    try:
        item = _graph_request(
            "GET",
            _item_path(remote_path),
            params={"$select": "id,name,size,file"},
        )
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        if _upload_status_code(exc) == 404:
            return None
        raise _UploadError("Could not inspect the OneDrive upload target") from exc
    return item if isinstance(item, dict) and item.get("id") else None


def _completed_upload_item(
    remote_path: str,
    total: int,
    expected_final_quickxor_hash: str,
) -> dict[str, Any]:
    """Bind ambiguous completion to the exact uploaded bytes via QuickXorHash."""
    last_error: BaseException | None = None
    for attempt in range(1, _UPLOAD_COMPLETION_MAX_ATTEMPTS + 1):
        try:
            item = _current_upload_item(remote_path)
            file_facet = item.get("file") if isinstance(item, dict) else None
            hashes = file_facet.get("hashes") if isinstance(file_facet, dict) else None
            remote_hash = (
                hashes.get("quickXorHash") if isinstance(hashes, dict) else None
            )
            if (
                isinstance(item, dict)
                and item.get("id")
                and item.get("size") == total
                and remote_hash == expected_final_quickxor_hash
            ):
                return item
            last_error = _UploadError(
                "OneDrive upload completed ambiguously and could not be confirmed"
            )
        except Exception as exc:
            last_error = exc
            if not _is_retriable_upload_error(exc):
                break
        if attempt < _UPLOAD_COMPLETION_MAX_ATTEMPTS:
            time.sleep(_upload_retry_delay(last_error, attempt))
    raise _UploadError(
        "OneDrive upload completed ambiguously and could not be confirmed"
    ) from last_error


def _reconcile_upload_progress(
    http: requests.Session,
    upload_url: str,
    remote_path: str,
    start: int,
    end: int,
    total: int,
    expected_final_quickxor_hash: str,
) -> tuple[int, dict[str, Any] | None]:
    """Return the first missing byte offset, plus a confirmed final item."""
    last_error: BaseException | None = None
    completed = False
    for attempt in range(1, _UPLOAD_RECONCILE_MAX_ATTEMPTS + 1):
        try:
            response = http.get(upload_url, timeout=DEFAULT_TIMEOUT_SECONDS)
            if response.status_code == 404:
                if end == total:
                    completed = True
                    break
                raise _UploadError(
                    "OneDrive upload session disappeared before completion"
                )
            _raise_upload_status(response)
            try:
                payload = response.json()
            except ValueError as exc:
                raise _UploadError(
                    "OneDrive returned invalid upload-session progress"
                ) from exc
            if (
                isinstance(payload, dict)
                and payload.get("id")
                and "nextExpectedRanges" not in payload
            ):
                if end < total:
                    raise _UploadError(
                        "OneDrive reported completion before the local final fragment"
                    )
                completed = True
                break
            next_offset = _next_expected_upload_offset(payload, total)
            if next_offset < start:
                raise _UploadError(
                    "OneDrive returned inconsistent upload-session progress"
                )
            if next_offset > end:
                raise _UploadError(
                    "OneDrive returned progress beyond the submitted fragment"
                )
            if next_offset == total and end < total:
                raise _UploadError(
                    "OneDrive reported completion before the local final fragment"
                )
            if next_offset < end:
                return next_offset, None
            if end == total:
                completed = True
                break
            return next_offset, None
        except Exception as exc:
            last_error = exc
            if not _is_retriable_upload_error(exc):
                raise
            if attempt == _UPLOAD_RECONCILE_MAX_ATTEMPTS:
                break
            time.sleep(_upload_retry_delay(exc, attempt))
    if completed:
        return total, _completed_upload_item(
            remote_path, total, expected_final_quickxor_hash
        )
    raise _UploadError(
        "Could not determine OneDrive upload-session progress"
    ) from last_error


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
    path_lookup: bool = False,
) -> Any:
    """Send one Graph request.

    path_lookup marks a request whose drive path the agent supplied and which
    must already exist (a lookup, or the parent of a new folder); only then
    does a missing item get _PATH_NOT_FOUND_HINT.
    """
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
        # Never forward the response body or requests' own message: Graph and
        # its backing storage may echo a preauthenticated download URL into
        # either. Callers branch on the parsed error code instead; the message
        # format matches powerpoint.py's.
        code = _graph_error_code(response)
        message = f"Graph {method} {path} failed with HTTP {response.status_code}"
        if code:
            message = f"{message} ({code})"
        if path_lookup and _is_path_miss(response.status_code, code):
            message = f"{message}. {_PATH_NOT_FOUND_HINT}"
        raise _GraphRequestError(message, response.status_code, code) from exc

    if raw:
        return response.content
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _is_path_miss(status_code: int, code: str | None) -> bool:
    """Whether a 404 says the item is missing, rather than e.g. the drive.

    A body without a parsable Graph code is still treated as a miss.
    """
    return status_code == 404 and code in (None, "itemNotFound")


def _graph_error_code(response: Any) -> str | None:
    try:
        payload = response.json()
    except (ValueError, TypeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and code else None


def _normalize_path(path: str | None) -> str | None:
    """Normalize a Drive-relative path for use in a root:/{path}: request
    URL, rejecting any "." or ".." segment outright rather than trying to
    resolve it.

    This isn't a filesystem path -- it's spliced directly into the request
    URL below -- but standard HTTP client URL normalization still applies
    to it: verified directly that requests' own PreparedRequest collapses
    ".." segments the same way a browser would (e.g.
    "/me/drive/root:/../../etc/x:/content" becomes "/me/etc/x:/content"),
    which can walk the request entirely out of "/me/drive/root:/" and onto
    a different, unrelated Graph API endpoint under the same OAuth token --
    not merely "the wrong file within Drive." A caller-supplied path with a
    dot-segment is refused rather than silently normalized away.
    """
    if path is None:
        return None
    value = path.strip().strip("/")
    if not value:
        return None
    if "\\" in value:
        raise ValueError("path must use '/' separators and must not contain '\\'")
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"path must not contain '.' or '..' segments: {path!r}")
    return value


def _item_path(base_path: str | None) -> str:
    normalized = _normalize_path(base_path)
    if not normalized:
        return "/me/drive/root"
    return f"/me/drive/root:/{quote(normalized, safe='/')}:"


def _children_path(folder_path: str | None) -> str:
    normalized = _normalize_path(folder_path)
    if not normalized:
        return "/me/drive/root/children"
    return f"/me/drive/root:/{quote(normalized, safe='/')}:/children"


def _content_path(file_path: str, *, field_name: str = "file_path") -> str:
    stripped_path = file_path.strip()
    if stripped_path.endswith("/"):
        raise ValueError(
            f"{field_name} must include a filename, not end with a folder separator"
        )
    if Path(stripped_path).name.endswith("."):
        raise ValueError(f"{field_name} filename must not end with a period")
    normalized = _normalize_path(file_path)
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return f"/me/drive/root:/{quote(normalized, safe='/')}:/content"


def _item_content_path(item_id: str) -> str:
    """Build a stable content URL for a previously resolved drive item."""
    normalized_id = str(item_id).strip()
    if not normalized_id:
        raise ValueError("OneDrive item id is required")
    return f"/me/drive/items/{quote(normalized_id, safe='')}/content"


def _download_output_dir() -> Path:
    """Return the current task's output directory for binary downloads."""
    base = os.environ.get(_OUTPUT_DIR_ENV_VAR, "").strip()
    if not base:
        raise RuntimeError(
            "No task workspace configured for this connector "
            f"({_OUTPUT_DIR_ENV_VAR} is unset) — onedrive_download_file needs "
            "a task workspace to write into."
        )
    resolved_base = Path(base).expanduser()
    try:
        resolved_base = resolved_base.resolve()
    except RuntimeError:
        # Python 3.11/3.12 raise RuntimeError for symlink loops. Leave the
        # unresolved path for mkdir(), which reports the underlying OSError.
        pass
    output_dir = resolved_base / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


_UNSAFE_DOWNLOAD_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.() -]")
_MAX_DOWNLOAD_FILENAME_LENGTH = 200
_MAX_DOWNLOAD_SUFFIX_LENGTH = 20


def _safe_download_filename(name: str) -> str:
    """Keep a remote OneDrive name as one safe local path segment."""
    base = Path(str(name).strip()).name
    stem, suffix = _split_stem_suffix(base)
    stem = _UNSAFE_DOWNLOAD_FILENAME_CHARS.sub("_", stem).strip(" ._") or "file"
    suffix = _UNSAFE_DOWNLOAD_FILENAME_CHARS.sub("_", suffix)[
        :_MAX_DOWNLOAD_SUFFIX_LENGTH
    ]
    max_stem_length = max(1, _MAX_DOWNLOAD_FILENAME_LENGTH - len(suffix))
    return stem[:max_stem_length] + suffix


def _publish_download_file(
    temporary_path: Path, output_dir: Path, filename: str
) -> Path:
    """Publish a completed download without clobbering a concurrent result."""
    stem, suffix = _split_stem_suffix(filename)
    counter = 0
    while True:
        candidate_name = filename if counter == 0 else f"{stem} ({counter}){suffix}"
        candidate = output_dir / candidate_name
        try:
            os.link(temporary_path, candidate)
        except FileExistsError:
            counter += 1
            continue
        except OSError as exc:
            if exc.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.ENOTSUP,
                errno.EOPNOTSUPP,
            }:
                raise
            try:
                fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                counter += 1
                continue
            try:
                with os.fdopen(fd, "wb") as output, temporary_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
            except BaseException:
                candidate.unlink(missing_ok=True)
                raise
        try:
            temporary_path.unlink()
        except OSError:
            logger.warning("Failed to remove OneDrive download staging file")
        return candidate


def _stream_download_to_path(
    url: str,
    output_path: Path,
    expected_size: int,
    *,
    authenticated: bool,
    expected_quickxor_hash: str | None = None,
) -> tuple[int, str]:
    """Stream content, enforcing a safe bound and verifying available hashes."""
    headers = _graph_headers({"Accept": "*/*"}) if authenticated else {"Accept": "*/*"}
    try:
        response = requests.request(
            method="GET",
            url=url,
            headers=headers,
            timeout=_BINARY_UPLOAD_TIMEOUT_SECONDS,
            stream=True,
        )
    except requests.RequestException:
        raise RuntimeError("OneDrive file download failed") from None

    digest = hashlib.sha256()
    quickxor = _QuickXorHash()
    total = 0
    try:
        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(
                f"OneDrive file download failed with HTTP {response.status_code}"
            ) from None
        with output_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                total += len(chunk)
                if total > _MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        "The OneDrive download exceeded the "
                        f"{_MAX_DOWNLOAD_BYTES // (1024 * 1024 * 1024)} GiB limit"
                    )
                if total > expected_size:
                    raise RuntimeError(
                        "OneDrive file size changed while it was being downloaded"
                    )
                output.write(chunk)
                digest.update(chunk)
                quickxor.update(chunk)
    except requests.RequestException:
        raise RuntimeError("OneDrive file download failed") from None
    finally:
        response.close()

    if expected_quickxor_hash is not None:
        if quickxor.base64_digest() != expected_quickxor_hash:
            raise RuntimeError("OneDrive file content hash did not match metadata")
    elif total != expected_size:
        raise RuntimeError("OneDrive file size changed while it was being downloaded")
    return total, digest.hexdigest()


def _decode_bytes(content: bytes) -> tuple[str | None, str | None]:
    try:
        return content.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, base64.b64encode(content).decode("ascii")


def _name_looks_binary(name: str) -> bool:
    """Whether ``name``'s extension is NOT a recognized text format.

    Default-deny: an extension is only accepted if it's in
    _KNOWN_TEXT_EXTENSIONS. A name with no extension at all (e.g.
    "Dockerfile", "README") is accepted too -- the absence of an extension
    isn't evidence of binary intent the way an unrecognized one is.

    Uses _split_stem_suffix rather than plain ``Path.suffix`` for the same
    reason google_drive.py's equivalent does: a name that's *entirely* a
    leading dot plus extension (e.g. ".pdf") has an empty ``Path(...).suffix``
    per pathlib's dotfile convention, which would make this function treat
    it as extensionless (accepted) -- exactly the kind of mislabeling this
    guard exists to catch, just via a name pathlib refuses to split.
    """
    suffix = _split_stem_suffix(Path(name.strip()).name)[1].lower()
    return bool(suffix) and suffix not in _KNOWN_TEXT_EXTENSIONS


def _resolve_upload_file_path(local_file_path: str) -> Path:
    """Restrict onedrive_upload_file to files under an allowlisted
    directory, mirroring the equivalent defenses used by other local-file
    upload tools.

    Containment is checked before existence, so a path that is both
    outside the allowlist and nonexistent reports the allowlist message,
    not "not found" -- the latter would leak whether that host path
    exists at all to a caller who has no business finding out.
    """
    try:
        candidate_path = Path(local_file_path).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = Path.cwd() / candidate_path
        local_path = candidate_path.resolve()
        if candidate_path.is_symlink():
            local_path = candidate_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "Could not resolve OneDrive upload path %r: %s", local_file_path, exc
        )
        raise ValueError("Could not resolve local_file_path") from exc

    try:
        allowed_dirs = allowed_dirs_from_env(_UPLOAD_ALLOWED_DIRS_ENV_VAR)
    except ValueError as exc:
        logger.warning("Invalid OneDrive upload directory configuration: %s", exc)
        raise ValueError("Upload directory configuration is invalid") from None
    if not any(local_path.is_relative_to(d) for d in allowed_dirs):
        logger.warning(
            "Rejected onedrive_upload_file path %s outside allowed directories: %s",
            local_path,
            ", ".join(str(path) for path in allowed_dirs),
        )
        raise PermissionError(
            "local_file_path is outside the allowed upload directories; pass "
            "an absolute path inside the task workspace, or, if this tool's "
            "description says a registered file_id may be supplied, pass the "
            "xagent file_id as file:<file_id> (from list_all_user_files or a "
            "file reference; not a OneDrive item id)"
        )

    if local_path.exists() and not local_path.is_file():
        # Distinguish "not a regular file" (a directory, a device node,
        # etc.) from "does not exist" -- both used to raise the same
        # FileNotFoundError, which reads as "retry, it'll show up" to a
        # caller/agent even though a directory will never become a file.
        raise ValueError("The given path is not a regular file")
    if not local_path.is_file():
        raise FileNotFoundError("File not found at the given path")
    return local_path


def _upload_large_file_content(
    remote_path: str, fh: Any, total: int, mime_type: str
) -> dict[str, Any]:
    """Upload the next ``total`` bytes readable from ``fh`` (too large for
    the simple content PUT) via Graph's upload-session API, in 320 KiB-
    aligned chunks read straight off disk -- never materializing more than
    one chunk of the file in memory at a time, unlike holding the whole
    file as a single ``bytes`` object would.

    Transient fragment failures are retried with bounded backoff on the same
    session. Cross-call resume state is not persisted, so an exhausted or
    non-retriable failure cancels the session before returning an error.
    """
    if total <= 0:
        # Not reachable through onedrive_upload_file today (it rejects an
        # empty file before choosing a path, and anything <= 0 bytes would
        # take the simple-PUT branch regardless) -- guarded directly here
        # too since `for start in range(0, 0, chunk_size)` would otherwise
        # silently skip the whole loop and return the empty `result` this
        # function initializes below, bypassing the "did OneDrive actually
        # confirm this" check that only runs inside the loop.
        raise ValueError(f"total must be positive, got {total}")
    try:
        session = _graph_request(
            "POST",
            f"{_item_path(remote_path)}/createUploadSession",
            body={"item": {"@microsoft.graph.conflictBehavior": "replace"}},
        )
    except (RuntimeError, requests.RequestException) as exc:
        raise _UploadError("Could not create OneDrive upload session") from exc
    upload_url = session.get("uploadUrl") if isinstance(session, dict) else None
    if not isinstance(upload_url, str) or not upload_url:
        raise RuntimeError("OneDrive did not return an upload session URL")

    # One Session for every chunk of this upload -- a fresh top-level
    # requests.put() per chunk would pay a new TCP+TLS handshake each time,
    # which adds up over the ~100 requests a 500MB upload needs.
    with requests.Session() as http:
        result: dict[str, Any] = {}
        quickxor_hash = _QuickXorHash()

        def refill_fragment(
            chunk: bytearray, fragment_start: int, next_offset: int
        ) -> int:
            """Drop an accepted prefix and refill a non-final fragment.

            Graph can report an arbitrary missing-byte offset. Sending only
            the old fragment's suffix could make a non-final PUT shorter than
            the required 320 KiB multiple, so extend it back to the configured
            chunk size while keeping only one fragment buffer resident.
            """
            accepted_size = next_offset - fragment_start
            del chunk[:accepted_size]
            new_end = min(next_offset + _UPLOAD_SESSION_CHUNK_SIZE, total)
            extra_size = new_end - next_offset - len(chunk)
            if extra_size:
                extra = fh.read(extra_size)
                if len(extra) != extra_size:
                    raise _UploadError(
                        f"expected to read {extra_size} bytes at offset "
                        f"{next_offset + len(chunk)} but got {len(extra)} -- the "
                        "local file may have changed size during upload"
                    )
                quickxor_hash.update(extra)
                chunk.extend(extra)
                if new_end == total and fh.read(1):
                    raise _UploadError(
                        "the local file may have changed size during upload"
                    )
            return new_end

        try:
            start = 0
            while start < total:
                end = min(start + _UPLOAD_SESSION_CHUNK_SIZE, total)
                expected_size = end - start
                chunk = bytearray(fh.read(expected_size))
                if len(chunk) != expected_size:
                    # A short read here means the local file shrank out from
                    # under this upload (a concurrent rewrite/truncation, or
                    # an unusual filesystem) -- sending Content-Length/
                    # Content-Range computed from the size we *expected*
                    # rather than what we actually read would either hang
                    # waiting for bytes that never arrive or desynchronize
                    # every later chunk's byte-range accounting. Fail loudly
                    # instead.
                    raise _UploadError(
                        f"expected to read {expected_size} bytes at offset "
                        f"{start} but got {len(chunk)} -- the local file "
                        "may have changed size during upload"
                    )
                # Hash each byte once when it is first read. Fragment retries
                # reuse ``chunk`` and therefore cannot double-count it.
                quickxor_hash.update(chunk)
                # Detect growth after the caller's fstat snapshot before a
                # truncated final item is committed. Keeping this as a
                # separate read preserves the one-chunk memory bound.
                if end == total and fh.read(1):
                    raise _UploadError(
                        "the local file may have changed size during upload"
                    )
                fragment_start = start
                # The upload session URL is itself pre-authenticated (a
                # token in its query string) -- Microsoft's own docs for
                # createUploadSession confirm this explicitly: "If you
                # include the Authorization header when issuing the PUT
                # call, it might result in an HTTP 401 Unauthorized
                # response. ... Don't include it when issuing the PUT
                # call." So this goes straight through the plain Session
                # rather than _graph_request (which always attaches one).
                # Graph's docs don't confirm Content-Type is honored on a
                # chunk PUT the way it is on the simple-PUT endpoint (only
                # Content-Length/Content-Range are documented there), but
                # sending it costs nothing and is the closest available
                # lever to the simple path's behavior.
                consecutive_failures = 0
                recovery_cycles = 0
                response: Any | None = None
                while True:
                    upload_error: BaseException | None = None
                    try:
                        response = http.put(
                            upload_url,
                            data=chunk,
                            headers={
                                "Content-Range": (
                                    f"bytes {fragment_start}-{end - 1}/{total}"
                                ),
                                "Content-Type": mime_type,
                            },
                            timeout=_BINARY_UPLOAD_TIMEOUT_SECONDS,
                        )
                        _raise_upload_status(response)
                        if response.status_code in (200, 201):
                            if end < total:
                                raise _UploadError(
                                    "OneDrive reported completion before the local "
                                    "final fragment"
                                )
                            break
                        if end < total and response.status_code == 202:
                            break
                    except Exception as exc:
                        status_code = _upload_status_code(exc)
                        if not (
                            _is_retriable_upload_error(exc)
                            or status_code in (404, 409, 416)
                        ):
                            raise
                        upload_error = exc

                    recovery_cycles += 1
                    try:
                        next_offset, completed_item = _reconcile_upload_progress(
                            http,
                            upload_url,
                            remote_path,
                            fragment_start,
                            end,
                            total,
                            quickxor_hash.base64_digest(),
                        )
                    except Exception as reconcile_error:
                        if not _is_retriable_upload_error(reconcile_error):
                            raise
                        consecutive_failures += 1
                        if consecutive_failures >= _UPLOAD_FRAGMENT_MAX_ATTEMPTS:
                            raise _UploadError(
                                "Could not determine OneDrive upload-session progress"
                            ) from reconcile_error
                        delay = _upload_retry_delay(
                            reconcile_error, consecutive_failures
                        )
                    else:
                        if completed_item is not None:
                            result = completed_item
                            break
                        if next_offset == end:
                            break
                        if next_offset > fragment_start:
                            end = refill_fragment(chunk, fragment_start, next_offset)
                            fragment_start = next_offset
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                            if consecutive_failures >= _UPLOAD_FRAGMENT_MAX_ATTEMPTS:
                                raise _UploadError(
                                    "OneDrive upload fragment made no forward progress"
                                ) from upload_error
                        retry_number = max(consecutive_failures, 1)
                        delay = (
                            _upload_retry_delay(upload_error, retry_number)
                            if upload_error is not None
                            else _retry_delay(
                                getattr(response, "headers", None), retry_number
                            )
                        )

                    if recovery_cycles >= _UPLOAD_CHUNK_MAX_RECOVERY_CYCLES:
                        raise _UploadError(
                            "OneDrive upload fragment exceeded its recovery budget"
                        ) from upload_error
                    logger.warning(
                        "Retrying OneDrive upload fragment after reconciliation "
                        "(recovery cycle %s/%s, delay %.1fs)",
                        recovery_cycles,
                        _UPLOAD_CHUNK_MAX_RECOVERY_CYCLES,
                        delay,
                    )
                    time.sleep(delay)
                # Only the final chunk's response carries the completed
                # item; Graph's intermediate (202) responses return upload-
                # progress info, not the item -- checked explicitly
                # (end == total) rather than "the last response that
                # happened to have a body", which would silently pick up an
                # unrelated intermediate body if Graph's progress responses
                # ever started including one.
                if end == total:
                    if result.get("id"):
                        break
                    if response is None:
                        raise _UploadError(
                            "OneDrive did not return a final upload response"
                        )
                    try:
                        final_item = response.json() if response.content else {}
                    except ValueError:
                        final_item = {}
                    if isinstance(final_item, dict) and final_item.get("id"):
                        result = final_item
                        break

                    # A 200/201 final PUT can still lose or omit its JSON
                    # driveItem response. Bind destination metadata to the
                    # exact local bytes before reporting success.
                    result = _completed_upload_item(
                        remote_path,
                        total,
                        quickxor_hash.base64_digest(),
                    )
                    break
                start = end
        except Exception as exc:
            cause = _request_error_cause(exc)
            if isinstance(exc, _UploadError) or isinstance(
                cause, requests.RequestException
            ):
                logger.error(
                    "OneDrive large-file upload failed: %s",
                    _safe_upload_error_message(exc),
                )
            else:
                logger.exception(
                    "Unexpected OneDrive large-file upload failure (%s)",
                    type(exc).__name__,
                )
            # No cross-call resume state is persisted. Once bounded retries
            # are exhausted, cancel the unusable session instead of leaving
            # partial data until its provider-defined expiration time.
            try:
                cancel_response = http.delete(
                    upload_url, timeout=DEFAULT_TIMEOUT_SECONDS
                )
            except Exception as cleanup_exc:
                # Never log cleanup exception text: requests exceptions can
                # contain the preauthenticated upload URL.
                logger.warning(
                    "Failed to cancel abandoned OneDrive upload session (%s)",
                    type(cleanup_exc).__name__,
                )
            else:
                if (
                    not (200 <= cancel_response.status_code < 300)
                    and cancel_response.status_code != 404
                ):
                    logger.warning(
                        "OneDrive upload session cancellation returned HTTP %s "
                        "instead of success",
                        cancel_response.status_code,
                    )
            # Keep only the numeric status across the upload-URL redaction
            # boundary. The original exception chain may contain the
            # preauthenticated session URL, so it must not escape, but callers
            # still need to distinguish a 423 lock from an ordinary failure.
            raise _UploadError(
                _safe_upload_error_message(exc),
                status_code=_upload_status_code(exc),
            ) from None

    return result


@mcp.tool()
def onedrive_get_profile() -> str:
    """Get the current Microsoft 365 user profile for OneDrive operations."""
    try:
        me = _graph_request(
            "GET",
            "/me",
            params={"$select": "id,displayName,userPrincipalName,mail"},
        )
        return _success(user=me)
    except Exception as e:
        logger.error("Error getting OneDrive profile: %s", e)
        return _error(str(e))


@mcp.tool()
def onedrive_list_items(
    folder_path: Annotated[
        str | None,
        Field(
            description=(
                "Folder path relative to the drive root, e.g. 'Reports'; omit "
                "for the root; not a local task-workspace path."
            )
        ),
    ] = None,
    top: int = 50,
) -> str:
    """List files and folders in OneDrive, optionally under a folder path."""
    try:
        result = _graph_request(
            "GET",
            _children_path(folder_path),
            params={"$top": max(1, min(top, 200))},
            path_lookup=True,
        )
        return _success(items=_caller_safe_drive_item(result.get("value", [])))
    except Exception as e:
        logger.error("Error listing OneDrive items under %s: %s", folder_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_search_files(query: str, top: int = 25) -> str:
    """Search files and folders in OneDrive by keyword.

    Search can miss existing items. To find a file's exact path, list its
    folder with onedrive_list_items instead of guessing one.
    """
    try:
        if not query.strip():
            raise ValueError("query is required")
        escaped_query = query.replace("'", "''")
        result = _graph_request(
            "GET",
            f"/me/drive/root/search(q='{quote(escaped_query, safe='')}')",
            params={"$top": max(1, min(top, 100))},
        )
        items = _caller_safe_drive_item(result.get("value", []))
        if not items:
            return _success(items=items, hint=_SEARCH_MISS_HINT)
        return _success(items=items)
    except Exception as e:
        logger.error("Error searching OneDrive files: %s", e)
        return _error(str(e))


@mcp.tool()
def onedrive_get_item(
    path: Annotated[
        str | None,
        Field(
            description=(
                "Item path relative to the drive root, e.g. 'Reports/Q3.pptx'; "
                "not a local task-workspace path."
            )
        ),
    ] = None,
    item_id: str | None = None,
) -> str:
    """Get OneDrive metadata by path or item_id."""
    try:
        if item_id:
            result = _graph_request(
                "GET",
                f"/me/drive/items/{url_path_id(item_id, 'item_id')}",
            )
        elif path:
            result = _graph_request("GET", _item_path(path), path_lookup=True)
        else:
            raise ValueError("either path or item_id is required")
        return _success(item=_caller_safe_drive_item(result))
    except Exception as e:
        logger.error("Error getting OneDrive item: %s", e)
        return _error(str(e))


@mcp.tool()
def onedrive_get_file_content(file_path: _DriveFilePath) -> str:
    """Read text content by path; use onedrive_download_file for binaries."""
    try:
        content = _graph_request(
            "GET", _content_path(file_path), raw=True, path_lookup=True
        )
        text_content, base64_content = _decode_bytes(content)
        return _success(
            file_path=file_path,
            text_content=text_content,
            base64_content=base64_content,
            encoding="utf-8" if text_content is not None else "base64",
        )
    except Exception as e:
        logger.error("Error downloading OneDrive file %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_download_file(file_path: _DriveFilePath, filename: str = "") -> str:
    """Download a OneDrive binary file into the current task workspace.

    This tool is intended for binary content that must be passed to another
    connector or processed locally. To change a .pptx, .docx or .xlsx that
    stays in OneDrive, edit it in place with the powerpoint_*, word_* or
    excel_* tools when that connector is enabled; they take the same
    drive-relative file_path. Download it for editing only when no such
    connector can make the change.

    It writes a real local file under the task's ``output/`` directory,
    enforces the download limit, verifies available Graph metadata hashes, and
    computes its SHA-256 plus a workspace path. The MCP host also registers
    that path as a durable FileRef before exposing the result to the agent.
    """
    temporary_path: Path | None = None
    try:
        output_dir = _download_output_dir()
        metadata = _graph_request(
            "GET",
            _item_path(file_path),
            params={"$select": "id,name,size,file,@microsoft.graph.downloadUrl"},
            path_lookup=True,
        )
        if (
            not isinstance(metadata, dict)
            or not metadata.get("id")
            or not isinstance(metadata.get("file"), dict)
        ):
            raise RuntimeError("OneDrive item is not a file")
        size = metadata.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError("OneDrive returned an invalid file size")
        if size > _MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"The OneDrive file is {size} bytes, over the "
                f"{_MAX_DOWNLOAD_BYTES // (1024 * 1024 * 1024)} GiB limit"
            )

        file_metadata = metadata["file"]
        hashes = file_metadata.get("hashes")
        expected_quickxor_hash = (
            hashes.get("quickXorHash") if isinstance(hashes, dict) else None
        )
        if not isinstance(expected_quickxor_hash, str) or not expected_quickxor_hash:
            expected_quickxor_hash = None

        remote_name = str(metadata.get("name") or Path(file_path).name)
        output_name = _safe_download_filename(filename or remote_name)
        staging_dir = output_dir.parent / ".xagent-onedrive-downloads"
        staging_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = staging_dir / f"{uuid4().hex}.part"
        download_url = metadata.get("@microsoft.graph.downloadUrl")
        if isinstance(download_url, str) and download_url:
            authenticated = False
            content_url = download_url
        else:
            authenticated = True
            content_url = f"{GRAPH_BASE_URL}{_item_content_path(metadata['id'])}"
        total, sha256 = _stream_download_to_path(
            content_url,
            temporary_path,
            size,
            authenticated=authenticated,
            expected_quickxor_hash=expected_quickxor_hash,
        )
        output_path = _publish_download_file(temporary_path, output_dir, output_name)
        temporary_path = None
        mime_type = (
            file_metadata.get("mimeType")
            or _guess_mime_type(output_path.name)
            or "application/octet-stream"
        )
        return _success(
            file_path=str(output_path),
            filename=output_path.name,
            size=total,
            sha256=sha256,
            mime_type=mime_type,
        )
    except Exception as e:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to clean OneDrive download temp file")
        raw_message = str(e)
        public_message = "OneDrive file download failed"
        if isinstance(e, _GraphRequestError):
            # Only the metadata lookup by path goes through _graph_request
            # here; a failed content stream raises a plain RuntimeError.
            public_message = f"OneDrive file download failed with HTTP {e.status_code}"
            if _is_path_miss(e.status_code, e.code):
                public_message = f"{public_message}. {_PATH_NOT_FOUND_HINT}"
        elif raw_message.startswith(
            (
                "No task workspace configured for this connector",
                "path must use '/' separators",
                "path must not contain",
                "The OneDrive file is ",
                "OneDrive item is not a file",
                "OneDrive returned ",
                "OneDrive file size changed",
                "OneDrive file content hash did not match metadata",
                "The OneDrive download exceeded ",
                "OneDrive file download failed with HTTP ",
                "AUTH_TOKEN environment variable is missing",
            )
        ):
            public_message = raw_message
        logger.error(
            "Error downloading OneDrive binary file %s (%s): %s",
            file_path,
            type(e).__name__,
            public_message,
        )
        return _error(public_message)


@mcp.tool()
def onedrive_upload_text_file(
    file_path: _DriveFilePath,
    content: str,
) -> str:
    """
    Upload or overwrite a UTF-8 text file in OneDrive by path.

    content is always treated as text (it is UTF-8 encoded before upload)
    -- this tool cannot create a real PDF, image, or Office-format binary;
    naming the target "report.pdf" would produce a file with that name but
    plain-text content, not an actual PDF. To upload an already-generated
    local file's real bytes (a PDF, image, .docx, etc.), use
    onedrive_upload_file with that file's path instead.
    """
    try:
        if _name_looks_binary(file_path):
            return _error(
                f"'{file_path}' looks like a binary file, but "
                "onedrive_upload_text_file only writes text content -- "
                "uploading it here would produce a mislabeled text file, "
                "not real binary data. If you already have this file on "
                "disk (e.g. as a task output), use onedrive_upload_file "
                "with its local path to upload the real binary content "
                "instead."
            )
        result = _graph_request(
            "PUT",
            _content_path(file_path),
            extra_headers={"Content-Type": "text/plain; charset=utf-8"},
            data=content.encode("utf-8"),
        )
        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("OneDrive did not confirm the upload completed")
        return _success(item=_caller_safe_drive_item(result))
    except Exception as e:
        logger.error("Error uploading OneDrive text file %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_upload_file(
    local_file_path: str,
    remote_path: Annotated[
        str,
        Field(
            description=(
                "Destination path relative to the drive root, e.g. "
                "'Reports/report.pdf'; not a local task-workspace path. Empty "
                "uploads to the root under the local file name."
            )
        ),
    ] = "",
    mime_type: str = "",
) -> str:
    """
    Upload a local file's real bytes to OneDrive -- use this (not
    onedrive_upload_text_file) for a PDF, image, Office document, or any
    other binary content, including a file this agent already generated
    into the task workspace (e.g. an exported PDF report).

    local_file_path: path to a file already on disk, e.g. something written to
    the task workspace. Must be inside an allowed directory (automatically
    scoped to the current task workspace). When no allowlist is configured,
    the process working directory is used as a fallback, so this is not an
    absolute guarantee against access to host files. Pass an absolute path;
    a relative path resolves against this process's working directory.
    remote_path: the OneDrive path to upload to (e.g. "Reports/report.pdf"),
    overwriting any existing file there; defaults to the local file's own
    name at the OneDrive root. Prefer the format-specific connector when one
    supports editing the existing Office document. If an existing item is
    locked, this tool returns code="resource_locked"; do not retry the
    same destination immediately. Close the document and retry later, or ask
    the user before choosing a different remote_path to save a copy.
    mime_type: defaults to a guess from the remote filename, then the local
    filename, falling back to "application/octet-stream". It is sent on both
    simple uploads and upload-session chunks.

    Empty files are rejected. Files over 2 GiB are also rejected as a
    guard-rail against a mistargeted upload, not as a OneDrive/Graph limit.
    """
    try:
        local_path = _resolve_upload_file_path(local_file_path)
        resolved_remote_path = remote_path.strip() or local_path.name
        content_path = _content_path(resolved_remote_path, field_name="remote_path")
        resolved_mime_type = mime_type.strip() or _guess_mime_type(resolved_remote_path)
        if resolved_mime_type is None:
            resolved_mime_type = (
                _guess_mime_type(local_path.name) or "application/octet-stream"
            )

        # Read from one open handle throughout -- the size check, the small-
        # file read, and the large-file chunk reads all use this same fh
        # rather than each re-opening/re-stat'ing the file. For a file over
        # _SIMPLE_UPLOAD_MAX_BYTES, _upload_large_file_content reads it
        # chunk-by-chunk straight off this handle rather than this function
        # first loading the whole file into memory -- the resumable upload-
        # session path exists specifically so a large file's memory
        # footprint stays bounded to one chunk at a time.
        # A fresh by-path open, not the same descriptor
        # _resolve_upload_file_path used for its allowlist check -- a
        # symlink swapped in during that window wouldn't be caught.
        # Accepted risk: holding a file descriptor open across the whole
        # allowlist resolution isn't worth it for a local, non-shared task
        # workspace (same tradeoff google_drive_upload_file makes).
        try:
            fh_ctx = local_path.open("rb")
        except OSError as e:
            # str(OSError) embeds the absolute path (e.g. "[Errno 13]
            # Permission denied: '/full/host/path'") -- the same detail
            # _resolve_upload_file_path's own error deliberately scrubs. A
            # permission error or a TOCTOU race (the allowlist-checked path
            # got swapped/removed between the check and this open) would
            # otherwise leak it straight into the caller/LLM-facing message
            # through the generic `except Exception` below.
            logger.warning("Failed to open upload file %s: %s", local_path, e)
            raise ValueError("Could not read the file at the given path") from e
        with fh_ctx as fh:
            file_size = os.fstat(fh.fileno()).st_size
            if file_size == 0:
                # A deliberate product choice, not an API constraint: Graph
                # itself accepts a 0-byte file. An agent uploading an empty
                # file is almost always a symptom of an upstream mistake
                # (e.g. a generation step that silently produced nothing),
                # so this is rejected here rather than silently creating a
                # placeholder-empty item on OneDrive.
                raise ValueError(f"File is empty: {local_file_path}")
            if file_size > _MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"File is {file_size} bytes, over the "
                    f"{_MAX_UPLOAD_BYTES // (1024 * 1024 * 1024)} GiB limit for "
                    "onedrive_upload_file"
                )

            if file_size <= _SIMPLE_UPLOAD_MAX_BYTES:
                # Bound the read even if another process appends after fstat.
                content = fh.read(_SIMPLE_UPLOAD_MAX_BYTES + 1)
                if not content:
                    raise ValueError(f"File is empty: {local_file_path}")
                if len(content) > _SIMPLE_UPLOAD_MAX_BYTES:
                    raise RuntimeError("the local file grew during upload")
                result = _graph_request(
                    "PUT",
                    content_path,
                    extra_headers={"Content-Type": resolved_mime_type},
                    data=content,
                    timeout=_BINARY_UPLOAD_TIMEOUT_SECONDS,
                )
            else:
                result = _upload_large_file_content(
                    resolved_remote_path, fh, file_size, resolved_mime_type
                )

        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("OneDrive did not confirm the upload completed")

        return _success(item=_caller_safe_drive_item(result))
    except Exception as e:
        if isinstance(e, (OSError, ValueError, RuntimeError)):
            logger.error("Error uploading OneDrive file %s: %s", local_file_path, e)
        else:
            logger.exception(
                "Unexpected error uploading OneDrive file %s", local_file_path
            )
        if _upload_status_code(e) in (409, 423):
            return _resource_locked_error("overwritten")
        return _error(str(e))


@mcp.tool()
def onedrive_create_folder(
    folder_name: str,
    parent_path: Annotated[
        str | None,
        Field(
            description=(
                "Existing parent folder path relative to the drive root, e.g. "
                "'Reports'; omit for the root; not a local task-workspace path."
            )
        ),
    ] = None,
    conflict_behavior: str = "rename",
) -> str:
    """Create a OneDrive folder under the specified parent path."""
    try:
        if not folder_name.strip():
            raise ValueError("folder_name is required")
        normalized_behavior = conflict_behavior.strip().lower()
        if normalized_behavior not in {"rename", "fail", "replace"}:
            raise ValueError("conflict_behavior must be one of: rename, fail, replace")
        result = _graph_request(
            "POST",
            _children_path(parent_path),
            body={
                "name": folder_name,
                "folder": {},
                "@microsoft.graph.conflictBehavior": normalized_behavior,
            },
            path_lookup=True,
        )
        return _success(folder=_caller_safe_drive_item(result))
    except Exception as e:
        logger.error("Error creating OneDrive folder %s: %s", folder_name, e)
        return _error(str(e))


@mcp.tool()
def onedrive_rename_item(item_id: str, new_name: str) -> str:
    """Rename a OneDrive file or folder by item_id."""
    try:
        if not new_name.strip():
            raise ValueError("new_name is required")
        result = _graph_request(
            "PATCH",
            f"/me/drive/items/{url_path_id(item_id, 'item_id')}",
            body={"name": new_name},
        )
        return _success(item=_caller_safe_drive_item(result))
    except Exception as e:
        logger.error("Error renaming OneDrive item %s: %s", item_id, e)
        return _error(str(e))


_CROSS_DRIVE_MOVE_MESSAGE = (
    "destination_folder_id is in another drive (for example a folder shared "
    "with you); OneDrive cannot move items between drives"
)


def _read_move_item(item_id: str, field_name: str) -> dict[str, Any]:
    """Read the source or destination of a move, turning 403/404 into an
    error that names which of the two ids failed."""
    try:
        item = _graph_request(
            "GET", f"/me/drive/items/{url_path_id(item_id, field_name)}"
        )
    except _GraphRequestError as exc:
        if exc.status_code == 404:
            raise ValueError(
                f"{field_name} was not found in this OneDrive: {item_id}"
            ) from None
        if exc.status_code == 403:
            raise ValueError(
                f"The connected account has no permission to access {field_name}: "
                f"{item_id}"
            ) from None
        raise
    if not isinstance(item, dict) or not isinstance(item.get("id"), str):
        raise RuntimeError(f"OneDrive returned an invalid item for {field_name}")
    return item


def _parent_reference(item: dict[str, Any], key: str) -> str | None:
    parent = item.get("parentReference")
    value = parent.get(key) if isinstance(parent, dict) else None
    return value if isinstance(value, str) and value else None


def _same_graph_id(left: str | None, right: str | None) -> bool:
    """Compare drive or item ids ignoring letter case: OneDrive Personal has
    been seen to report the same driveId in different case across
    responses, which must not turn a same-drive move into a refusal."""
    return (
        left is not None and right is not None and left.casefold() == right.casefold()
    )


@mcp.tool()
def onedrive_move_item(
    item_id: str, destination_folder_id: str, new_name: str = ""
) -> str:
    """Move a OneDrive file or folder into another folder of the same drive.

    ``item_id`` and ``destination_folder_id`` are OneDrive item ids (from
    onedrive_list_items, onedrive_search_files or onedrive_get_item); use
    "root" as ``destination_folder_id`` for the top level of the drive. The
    item keeps its id, sharing links and version history. ``new_name``
    optionally renames it in the same request; leave it empty to keep the
    current name.

    Both ids are read before anything changes, so a missing item, a
    destination that is not a folder, or a folder in another drive (such as
    one shared with the user) fails without moving anything. If OneDrive
    reports that the destination already holds an item with that name, the
    error says so; retry with a different ``new_name``. If the item is
    locked, typically because it is open in an Office app or checked out,
    the error says so; do not retry until the user confirms it is closed.

    On success the response holds ``item`` (the item after the call),
    ``destination_folder_id``, ``already_in_destination`` (true when the item
    was already in that folder, so it was not moved) and ``renamed``.

    Moving can change who can access the item, because an item can pick up
    the sharing of the folder it is moved into. Confirm the destination and
    this possible access change with the user before calling this tool.
    """
    try:
        if not isinstance(new_name, str):
            raise ValueError("new_name must be a string")
        requested_name = new_name.strip()
        if new_name and not requested_name:
            raise ValueError(
                "new_name must not be blank; leave it empty to keep the current name"
            )
        url_path_id(item_id, "item_id")
        url_path_id(destination_folder_id, "destination_folder_id")
        if item_id == destination_folder_id:
            raise ValueError("An item cannot be moved into itself.")

        source = _read_move_item(item_id, "item_id")
        if "root" in source:
            raise ValueError("The drive root folder cannot be moved")
        destination = _read_move_item(destination_folder_id, "destination_folder_id")
        if "remoteItem" in destination:
            raise ValueError(_CROSS_DRIVE_MOVE_MESSAGE)
        if "folder" not in destination:
            raise ValueError("destination_folder_id must refer to a folder")
        source_drive_id = _parent_reference(source, "driveId")
        destination_drive_id = _parent_reference(destination, "driveId")
        # Graph refuses a cross-drive move on its own; this check only turns
        # that refusal into a clear message before anything is sent, so it
        # is skipped when either side leaves driveId out.
        if (
            source_drive_id
            and destination_drive_id
            and not _same_graph_id(source_drive_id, destination_drive_id)
        ):
            raise ValueError(_CROSS_DRIVE_MOVE_MESSAGE)

        # Graph rejects the "root" alias in parentReference.id, so the move
        # always sends the id Graph returned for the destination.
        resolved_destination_id = destination["id"]
        if _same_graph_id(source["id"], resolved_destination_id):
            raise ValueError("An item cannot be moved into itself.")

        already_in_destination = _same_graph_id(
            _parent_reference(source, "id"), resolved_destination_id
        )
        body: dict[str, Any] = {}
        if not already_in_destination:
            body["parentReference"] = {"id": resolved_destination_id}
        if requested_name and requested_name != source.get("name"):
            body["name"] = requested_name
        if not body:
            return _success(
                item=_caller_safe_drive_item(source),
                destination_folder_id=resolved_destination_id,
                already_in_destination=True,
                renamed=False,
            )

        try:
            result = _graph_request(
                "PATCH",
                f"/me/drive/items/{url_path_id(source['id'], 'item_id')}",
                body=body,
            )
        except _GraphRequestError as exc:
            # Graph also answers 409 for other conflicts (its error docs give
            # a missing parent folder as the example), so only its
            # nameAlreadyExists code is reported as a name clash; any other
            # 409 keeps Graph's own status and error code.
            if exc.status_code == 409 and exc.code == "nameAlreadyExists":
                target_name = body.get("name", source.get("name"))
                raise ValueError(
                    f"The destination folder already contains an item named "
                    f"{target_name!r}; retry with a different new_name"
                ) from None
            if exc.status_code == 403:
                raise ValueError(
                    "The connected account has no permission to move this item "
                    "or to write to the destination folder"
                ) from None
            if exc.status_code == 404:
                raise ValueError(
                    "The item or the destination folder no longer exists"
                ) from None
            # Graph answers 423 whatever its error code (production sent
            # notAllowed) while the file is open or checked out elsewhere;
            # repeating the call at once only meets the same lock. The code
            # stays in the message because the tool result is what reaches
            # the trace.
            if exc.status_code == 423:
                raise ValueError(
                    "OneDrive refused the move because the item, or a file in "
                    "it, is locked, typically because it is open in Word, "
                    "Excel, PowerPoint or Teams (possibly by someone else) or "
                    "checked out. Ask the user to close it or check it in; a "
                    "lock can take a few minutes to clear after the file is "
                    "closed. Do not retry until they confirm"
                    + (f" (Graph error code: {exc.code})" if exc.code else "")
                ) from None
            raise
        if not isinstance(result, dict):
            raise RuntimeError("OneDrive move returned an invalid item")
        # The PATCH has already been applied here, so only a response that
        # names a different parent is reported, and the message says the
        # item may have moved rather than claiming the move failed.
        result_parent_id = _parent_reference(result, "id")
        if result_parent_id is not None and not _same_graph_id(
            result_parent_id, resolved_destination_id
        ):
            raise RuntimeError(
                "OneDrive accepted the move but reported a different parent "
                "folder; the item may have moved. Check its location with "
                "onedrive_get_item before retrying"
            )
        return _success(
            item=_caller_safe_drive_item(result),
            destination_folder_id=resolved_destination_id,
            already_in_destination=already_in_destination,
            renamed="name" in body,
        )
    except Exception as e:
        logger.error("Error moving OneDrive item %s: %s", item_id, e)
        return _error(str(e))


@mcp.tool()
def onedrive_delete_item(item_id: str) -> str:
    """Delete a OneDrive file or folder by item_id."""
    try:
        _graph_request(
            "DELETE",
            f"/me/drive/items/{url_path_id(item_id, 'item_id')}",
        )
        return _success(message="Item deleted successfully")
    except Exception as e:
        logger.error("Error deleting OneDrive item %s: %s", item_id, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
