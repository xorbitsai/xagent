import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onedrive-mcp")

setup_proxy_env()

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
# Microsoft's docs describe the simple content PUT's limit as "4 MB"; some
# Graph deployments enforce that as the decimal 4,000,000 bytes rather than
# 4 MiB (4,194,304 bytes). Using the smaller, decimal figure here means a
# file in that ambiguous ~194KB gap always takes the resumable upload-session
# path instead of risking a rejection right at the simple-PUT boundary.
_SIMPLE_UPLOAD_MAX_BYTES = 4_000_000
# Chunk size for the upload-session path. Must be a multiple of 320 KiB
# (327,680 bytes) per Graph's requirement for every non-final chunk; 5 MiB
# is exactly 16 * 320 KiB.
_UPLOAD_SESSION_CHUNK_SIZE = 5 * 1024 * 1024
# Not a Graph API limit (OneDrive itself supports files far larger than
# this via the resumable upload session) -- a guard-rail against a
# mistargeted file (a generated artifact pointed at the wrong path, an
# entire log directory, etc.) tying up a chunked upload for a long time
# with no feedback until it eventually finishes or fails, mirroring
# gmail.py's own _MAX_ATTACHMENT_BYTES guard for the same kind of mistake.
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

_UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS"

# stdlib mimetypes.guess_type() only recognizes these extensions when a
# system mime.types file happens to be installed (e.g. Apache's, common on
# a dev laptop) -- on a minimal/slim host with no such file (a stripped-down
# container image, verified directly: MimeTypes(filenames=()) returns
# (None, None) for every one of these), it silently can't identify them at
# all. Checked before falling back to mimetypes.guess_type() everywhere
# this module resolves a mime type, so a real .xlsx/.docx/etc. doesn't get
# mislabeled "application/octet-stream" just because the host is missing a
# file this module has no control over.
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


# Extensions whose real mime type collides with a totally unrelated binary
# format's registered mimetype, verified directly: ".ts" -> "video/mp2t"
# (MPEG transport stream, not TypeScript), ".bat" -> "application/x-msdownload",
# ".scm" -> "application/vnd.lotus-screencam" (not Scheme source), ".sc" ->
# "application/vnd.ibm.secure-container" (an obscure, effectively dead IBM
# format, not Scala/SuperCollider source).
#
# Consulted only by _name_looks_binary's text/binary guard, NOT by
# _guess_mime_type -- a real ".ts" file uploaded through onedrive_upload_file
# (a genuine MPEG transport stream, not TypeScript source) still needs its
# actual "video/mp2t" Content-Type, not "text/plain". Forcing the override
# into _guess_mime_type itself (an earlier version of this fix did that)
# would fix the text-upload guard's false positive at the cost of mislabeling
# the exact binary format the collision names.
_AMBIGUOUS_TEXT_EXTENSIONS = {
    ".ts", ".bat", ".scm", ".sc",
    # Not a verified collision like the four above -- mimetypes.guess_type
    # returns None for ".ps1" on every host tested. Included here
    # defensively anyway: no real binary format uses ".ps1", so there's no
    # downside, and it removes any dependency on some as-yet-unseen host's
    # mime database staying silent on it.
    ".ps1",
}  # fmt: skip


def _guess_mime_type(name: str) -> str | None:
    """Guess ``name``'s real mime type for actual Content-Type resolution:
    _MIME_TYPE_OVERRIDES first, for the formats stdlib mimetypes can't
    reliably identify on its own, then stdlib itself. Deliberately does not
    consult _AMBIGUOUS_TEXT_EXTENSIONS -- see that set's own docstring."""
    suffix = Path(name).suffix.lower()
    override = _MIME_TYPE_OVERRIDES.get(suffix)
    if override is not None:
        return override
    return mimetypes.guess_type(name)[0]


# Mime types ending in one of these suffixes are XML/JSON/YAML-based text
# formats even when their registered top-level type collides with a
# _BINARY_MIME_PREFIXES prefix -- verified directly: "image/svg+xml" (a
# plain-text vector format an agent may legitimately generate and upload via
# onedrive_upload_text_file) starts with "image/" and would otherwise be
# misclassified as binary by that prefix check alone. Checked before the
# prefix/positive-list check in _is_binary_mime_type so a real image/audio/
# video/font format that happens to also be XML/JSON/YAML-encoded text is
# never caught by the broader prefix.
_TEXT_SAFE_MIME_SUFFIXES = ("+xml", "+json", "+yaml")

# Mime-type prefixes that are unambiguously binary media containers.
_BINARY_MIME_PREFIXES = ("image/", "audio/", "video/", "font/")

# Specific mime types (documents, archives, executables) that are
# unambiguously binary but don't fall under one of the prefixes above.
# This is deliberately a *positive* list of known binary formats rather
# than the inverse (a "known text" allowlist grown to cover every possible
# script/config/source-code mimetype) -- the false-positive class that
# approach produces (.ts/.bat/.scm/.sc, then .ps1/.sql/.php/.dart, then
# .tex/.csh/.tpl, then .sh/.srt/.dtd, discovered across four separate
# review rounds) keeps recurring because programming/script/config-file
# extensions are effectively unbounded, while real binary container/
# document formats are a small, enumerable set. Anything not matched here
# (or by a prefix above) is treated as text by default -- including a
# mimetype nobody has thought to test yet, which is exactly the case this
# design change exists to stop mishandling.
_BINARY_MIME_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel.sheet.macroEnabled.12",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.presentation",
    "application/epub+zip",
    "application/zip",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
    "application/x-bzip2",
    "application/x-7z-compressed",
    "application/vnd.rar",
    "application/x-rar-compressed",
    "application/x-msdownload",
    "application/x-executable",
    "application/x-mach-binary",
    "application/x-sharedlib",
    "application/octet-stream",
    "application/wasm",
    "application/vnd.sqlite3",
    "application/x-sqlite3",
    "application/x-mspublisher",  # .pub -- a real binary format, verified
    "application/postscript",  # .ps -- pre-existing behavior, unchanged
    # Found while re-auditing _KNOWN_BINARY_EXTENSIONS_WITHOUT_MIME_GUESS's
    # own extensions against a host WITH a system mime.types installed
    # (unlike the bare-stdlib probe that motivated that fallback set in the
    # first place) -- these four resolve to a real, specific mimetype
    # there instead of None, so the "no mimetype at all" fallback alone
    # would miss them on such a host.
    "application/x-mobipocket-ebook",  # .mobi
    "application/x-apple-diskimage",  # .dmg
    "application/x-iso9660-image",  # .iso
    "application/vnd.android.package-archive",  # .apk
    # Found via a follow-up sweep of common archive/executable/key-bundle
    # formats not covered by the categories above -- each verified directly
    # against mimetypes.guess_type.
    "application/java-archive",  # .jar
    "application/java-vm",  # .class
    "application/x-shockwave-flash",  # .swf
    "application/vnd.ms-cab-compressed",  # .cab
    "application/x-debian-package",  # .deb
    "application/x-bittorrent",  # .torrent
    "application/x-pkcs12",  # .p12, .pfx -- private key/certificate bundles
}


def _is_binary_mime_type(mime_type: str) -> bool:
    if mime_type.endswith(_TEXT_SAFE_MIME_SUFFIXES):
        return False
    return (
        mime_type.startswith(_BINARY_MIME_PREFIXES) or mime_type in _BINARY_MIME_TYPES
    )


# Extensions of unambiguously binary formats that resolve to no mime type
# at all (neither _MIME_TYPE_OVERRIDES nor a bare stdlib mimetypes install
# recognizes them), so _name_looks_binary's mime-type check alone would
# silently miss them. onedrive_upload_text_file's content is always UTF-8
# text (see below), so a binary-looking target name is always a caller
# mistake, not a valid use.
_KNOWN_BINARY_EXTENSIONS_WITHOUT_MIME_GUESS = {
    ".mobi", ".psd",
    ".ogg", ".flac", ".m4a",
    ".mkv", ".wmv",
    ".7z", ".rar", ".gz", ".bz2",
    ".dylib", ".dmg", ".iso", ".apk", ".rpm", ".crx",
    ".woff", ".woff2", ".ttf", ".otf",
    ".parquet", ".sqlite", ".sqlite3", ".db", ".db3",
    ".numbers", ".pages", ".key", ".blend", ".indd", ".pub",
    # Resolve to a real, specific mimetype on a host with a full system
    # mime.types (so they're already caught by _BINARY_MIME_TYPES/prefixes
    # there) but to None on a bare stdlib install -- verified directly via
    # MimeTypes(filenames=()) -- so they're listed here too for host
    # independence, same rationale as the rest of this set.
    ".jar", ".class", ".cab", ".deb", ".torrent", ".m4v",
    # Resolve to no mimetype at all on any host tested (bare stdlib or
    # system mime.types) -- camera raw photo and generic macOS/executable
    # bundle formats mimetypes has no opinion on whatsoever.
    ".raw", ".cr2", ".app",
}  # fmt: skip


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


def _raise_for_status_with_body(response: Any) -> None:
    """Like response.raise_for_status(), but a non-2xx status raises a
    RuntimeError with Graph's own error body appended -- a bare HTTPError's
    default message carries only the status line, not the JSON error detail
    Graph actually returns. Shared by every call site in this module that
    talks to Graph directly (both _graph_request and the per-chunk PUTs in
    _upload_large_file_content, which can't go through _graph_request itself
    -- see its own Authorization-header note) so a future change to the
    message format only needs to happen once.
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text.strip()
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
        raise RuntimeError(message) from exc


def _redact_upload_url(message: str, upload_url: str) -> str:
    """Strip ``upload_url`` (a preauthenticated, bearer-token-equivalent
    Graph upload-session URL) out of ``message`` as thoroughly as possible.

    A plain ``message.replace(upload_url, ...)`` only catches the case
    where the exception's own message happens to contain the exact,
    complete "scheme://host/path?query" string. Verified directly against
    a real failed request: urllib3's own ConnectionError/SSLError messages
    never do that -- they report the host separately (inside
    "HTTPSConnectionPool(host=..., port=...)") from the path+query (inside
    "Max retries exceeded with url: /path?query"), so the whole-string
    match silently misses them and the query string -- which carries the
    session's actual bearer-equivalent token -- passes through unredacted.
    Redacting the host and the path+query independently (in addition to
    the full URL, for whichever call site does happen to embed it whole)
    closes that gap without depending on any particular exception's message
    shape.
    """
    redacted = message.replace(upload_url, "<redacted-upload-session-url>")
    parsed = urlsplit(upload_url)
    path_and_query = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    if path_and_query:
        redacted = redacted.replace(path_and_query, "<redacted-upload-session-url>")
    if parsed.query:
        redacted = redacted.replace(parsed.query, "<redacted-upload-session-token>")
    if parsed.netloc:
        redacted = redacted.replace(parsed.netloc, "<redacted-upload-session-host>")
    return redacted


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
    _raise_for_status_with_body(response)

    if raw:
        return response.content
    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


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
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"path must not contain '.' or '..' segments: {path!r}")
    return value


def _normalize_item_id(item_id: str) -> str:
    """Validate an item_id used directly in a /me/drive/items/{id} request
    URL (onedrive_get_item, onedrive_rename_item, onedrive_delete_item).

    These never go through _normalize_path/_item_path -- they're opaque
    Graph item identifiers, not Drive-relative paths -- but they're still
    quoted with safe='' and spliced into the request URL the same way, so
    the same collapsing behavior applies: quote() never percent-encodes a
    bare "." or ".." (they're in urllib's own "always safe" unreserved
    set), so an item_id of exactly "." or ".." reaches requests as a
    literal dot-segment and collapses "/me/drive/items/.." down to
    "/me/drive/" -- a different Graph endpoint under the same OAuth token.
    quote(..., safe='') does escape "/", so a multi-segment "../.." can't
    reach this path, but the single-segment case still could without this
    check.
    """
    value = item_id.strip()
    if not value:
        raise ValueError("item_id is required")
    if value in (".", ".."):
        raise ValueError(f"item_id must not be '.' or '..': {item_id!r}")
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


def _content_path(file_path: str) -> str:
    normalized = _normalize_path(file_path)
    if not normalized:
        raise ValueError("file_path is required")
    return f"/me/drive/root:/{quote(normalized, safe='/')}:/content"


def _decode_bytes(content: bytes) -> tuple[str | None, str | None]:
    try:
        return content.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, base64.b64encode(content).decode("ascii")


def _name_looks_binary(name: str) -> bool:
    """Whether ``name``'s extension names an unambiguously binary format.

    _AMBIGUOUS_TEXT_EXTENSIONS is checked first and short-circuits straight
    to "not binary" -- these extensions' real mimetypes collide with an
    unrelated binary format (see that set's own docstring), which only
    matters for this text/binary guard, not for _guess_mime_type's real
    Content-Type resolution. Otherwise, a resolvable mime type (override or
    stdlib) is binary only if _is_binary_mime_type says so -- anything
    else, including a mime type nobody anticipated, defaults to "not
    binary" (see that function's own docstring for why a positive binary
    list is the safer default here). An unresolvable mime type falls back
    to the small hand-maintained set of formats mimetypes has no opinion
    on at all (_KNOWN_BINARY_EXTENSIONS_WITHOUT_MIME_GUESS above).
    """
    suffix = Path(name).suffix.lower()
    if suffix in _AMBIGUOUS_TEXT_EXTENSIONS:
        return False
    guessed = _guess_mime_type(name)
    if guessed is not None:
        return _is_binary_mime_type(guessed)
    return suffix in _KNOWN_BINARY_EXTENSIONS_WITHOUT_MIME_GUESS


def _allowed_upload_dirs() -> list[Path]:
    raw_dirs = os.environ.get(_UPLOAD_ALLOWED_DIRS_ENV_VAR, "")
    if not raw_dirs.strip():
        return [Path.cwd().resolve()]
    return [
        Path(stripped).expanduser().resolve()
        for raw_dir in raw_dirs.split(",")
        if (stripped := raw_dir.strip())
    ]


def _resolve_upload_file_path(file_path: str) -> Path:
    """Restrict onedrive_upload_file to files under an allowlisted
    directory, mirroring slack_upload_file's already-merged defense (and
    the equivalent google_drive_upload_file guard proposed in the still-
    unmerged sibling PR #2275) -- without it an agent could be tricked into
    exfiltrating arbitrary host files through this tool.

    Containment is checked before existence, so a path that is both
    outside the allowlist and nonexistent reports the allowlist message,
    not "not found" -- the latter would leak whether that host path
    exists at all to a caller who has no business finding out.
    """
    local_path = Path(file_path).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    local_path = local_path.resolve()

    allowed_dirs = _allowed_upload_dirs()
    if not any(local_path.is_relative_to(d) for d in allowed_dirs):
        logger.warning(
            "Rejected onedrive_upload_file path %s outside allowed directories: %s",
            local_path,
            ", ".join(str(path) for path in allowed_dirs),
        )
        raise PermissionError(
            "file_path is outside the allowed upload directories; ask the "
            "user for a file inside the task workspace or another allowed "
            "location"
        )

    if local_path.exists() and not local_path.is_file():
        # Distinguish "not a regular file" (a directory, a device node,
        # etc.) from "does not exist" -- both used to raise the same
        # FileNotFoundError, which reads as "retry, it'll show up" to a
        # caller/agent even though a directory will never become a file.
        raise ValueError(f"Not a regular file: {file_path}")
    if not local_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return local_path


def _upload_large_file_content(
    remote_path: str, fh: Any, total: int, mime_type: str
) -> dict[str, Any]:
    """Upload the next ``total`` bytes readable from ``fh`` (too large for
    the simple content PUT) via Graph's upload-session API, in 320 KiB-
    aligned chunks read straight off disk -- never materializing more than
    one chunk of the file in memory at a time, unlike holding the whole
    file as a single ``bytes`` object would.

    This uses the session only for chunking, not for actual resumability:
    on any failure the session is cancelled outright (see below) rather
    than retried from Graph's ``nextExpectedRanges``, so a transient
    mid-upload failure means the whole file is re-sent from byte 0 on the
    next call, not resumed from where it left off.
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
    session = _graph_request(
        "POST",
        f"{_item_path(remote_path)}/createUploadSession",
        body={"item": {"@microsoft.graph.conflictBehavior": "replace"}},
    )
    upload_url = session.get("uploadUrl")
    if not upload_url:
        raise RuntimeError("OneDrive did not return an upload session URL")

    # One Session for every chunk of this upload -- a fresh top-level
    # requests.put() per chunk would pay a new TCP+TLS handshake each time,
    # which adds up over the ~100 requests a 500MB upload needs.
    with requests.Session() as http:
        result: dict[str, Any] = {}
        try:
            for start in range(0, total, _UPLOAD_SESSION_CHUNK_SIZE):
                end = min(start + _UPLOAD_SESSION_CHUNK_SIZE, total)
                chunk = fh.read(end - start)
                if len(chunk) != end - start:
                    # A short read here means the local file shrank out from
                    # under this upload (a concurrent rewrite/truncation, or
                    # an unusual filesystem) -- sending Content-Length/
                    # Content-Range computed from the size we *expected*
                    # rather than what we actually read would either hang
                    # waiting for bytes that never arrive or desynchronize
                    # every later chunk's byte-range accounting. Fail loudly
                    # instead.
                    raise RuntimeError(
                        f"expected to read {end - start} bytes at offset "
                        f"{start} but got {len(chunk)} -- the local file "
                        "may have changed size during upload"
                    )
                # The upload session URL is itself pre-authenticated (a
                # token in its query string) -- Graph 401s a chunk request
                # that also carries our own Authorization header, so this
                # goes straight through the plain Session rather than
                # _graph_request (which always attaches one). Graph's docs
                # don't confirm Content-Type is honored on a chunk PUT the
                # way it is on the simple-PUT endpoint (only Content-Length/
                # Content-Range are documented there), but sending it costs
                # nothing and is the closest available lever to the simple
                # path's behavior.
                response = http.put(
                    upload_url,
                    data=chunk,
                    headers={
                        "Content-Length": str(end - start),
                        "Content-Range": f"bytes {start}-{end - 1}/{total}",
                        "Content-Type": mime_type,
                    },
                    timeout=_BINARY_UPLOAD_TIMEOUT_SECONDS,
                )
                _raise_for_status_with_body(response)
                # Only the final chunk's response carries the completed
                # item; Graph's intermediate (202) responses return upload-
                # progress info, not the item -- checked explicitly
                # (end == total) rather than "the last response that
                # happened to have a body", which would silently pick up an
                # unrelated intermediate body if Graph's progress responses
                # ever started including one.
                if end == total:
                    try:
                        result = response.json() if response.content else {}
                    except ValueError as exc:
                        # The PUT itself succeeded (no HTTPError above) --
                        # OneDrive has already committed the file at this
                        # point, so this is "the transfer worked but the
                        # confirmation can't be read", not a transfer
                        # failure, even though it's still reported as an
                        # error here (the caller has no completed item to
                        # act on either way).
                        raise RuntimeError(
                            "OneDrive accepted the final chunk but "
                            f"returned an unparsable response: {exc}"
                        ) from exc
                    if "id" not in result:
                        # The final chunk's response didn't actually carry
                        # a completed driveItem (e.g. an unexpected 202 on
                        # what this loop computed as the last range).
                        raise RuntimeError(
                            "OneDrive did not confirm the upload completed "
                            f"(final response: {result!r})"
                        )
        except Exception as exc:
            # Redact upload_url out of the failure's own message before it
            # goes anywhere else -- this catches every source uniformly
            # (an HTTP error response whose body happens to echo the
            # request URL via an intervening proxy/WAF, and -- the gap a
            # status-code-keyed sanitize flag on the PUT call alone would
            # miss -- a transport-level failure before any response
            # existed at all: requests/urllib3's own ConnectionError/
            # Timeout/SSLError messages embed the full request URL,
            # verified directly against a real failed request). A plain
            # `except requests.HTTPError` (or a flag passed only to the
            # PUT's own status check) would leave those other paths
            # unsanitized, so this wraps every exception the chunk loop
            # can raise instead of trusting each one individually to avoid
            # the URL. See _redact_upload_url's own docstring for why a
            # bare whole-string replace isn't enough on its own.
            redacted_message = _redact_upload_url(str(exc), upload_url)
            # Best-effort: free the abandoned session immediately instead
            # of leaving it for Graph's own ~15-minute expiry. A failure
            # here must never mask the real error above, and uses the
            # short, small-request timeout -- this DELETE carries no body,
            # so it shouldn't compound a slow failure with another wait as
            # long as a multi-megabyte upload's own timeout.
            try:
                cancel_response = http.delete(
                    upload_url, timeout=DEFAULT_TIMEOUT_SECONDS
                )
            except Exception as cleanup_exc:
                cleanup_message = _redact_upload_url(str(cleanup_exc), upload_url)
                logger.warning(
                    "Failed to cancel abandoned OneDrive upload session: %s",
                    cleanup_message,
                )
            else:
                # requests does not raise for an HTTP error status on its
                # own -- a 429/5xx here would otherwise look like a
                # successful cancellation while the session (and its
                # partial upload) actually lingers until Graph's own
                # expiry. A 404 is also treated as fine (not warned on):
                # it means the session is already gone -- expired, or
                # completed/cancelled by a previous attempt -- which is
                # the outcome this cleanup wants, not a failure of it.
                # Status only, no body: nothing about the response is
                # expected to carry the upload_url, but there's no reason
                # to risk it for a log line that only needs the status.
                if (
                    not (200 <= cancel_response.status_code < 300)
                    and cancel_response.status_code != 404
                ):
                    logger.warning(
                        "OneDrive upload session cancellation returned "
                        "HTTP %s instead of success",
                        cancel_response.status_code,
                    )
            raise RuntimeError(redacted_message) from exc

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
def onedrive_list_items(folder_path: str | None = None, top: int = 50) -> str:
    """List files and folders in OneDrive, optionally under a folder path."""
    try:
        result = _graph_request(
            "GET",
            _children_path(folder_path),
            params={"$top": max(1, min(top, 200))},
        )
        return _success(items=result.get("value", []))
    except Exception as e:
        logger.error("Error listing OneDrive items under %s: %s", folder_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_search_files(query: str, top: int = 25) -> str:
    """Search files and folders in OneDrive by keyword."""
    try:
        if not query.strip():
            raise ValueError("query is required")
        escaped_query = query.replace("'", "''")
        result = _graph_request(
            "GET",
            f"/me/drive/root/search(q='{quote(escaped_query, safe='')}')",
            params={"$top": max(1, min(top, 100))},
        )
        return _success(items=result.get("value", []))
    except Exception as e:
        logger.error("Error searching OneDrive files: %s", e)
        return _error(str(e))


@mcp.tool()
def onedrive_get_item(path: str | None = None, item_id: str | None = None) -> str:
    """Get OneDrive metadata by path or item_id."""
    try:
        if item_id:
            result = _graph_request(
                "GET",
                f"/me/drive/items/{quote(_normalize_item_id(item_id), safe='')}",
            )
        elif path:
            result = _graph_request("GET", _item_path(path))
        else:
            raise ValueError("either path or item_id is required")
        return _success(item=result)
    except Exception as e:
        logger.error("Error getting OneDrive item: %s", e)
        return _error(str(e))


@mcp.tool()
def onedrive_get_file_content(file_path: str) -> str:
    """Download file content from OneDrive by path. Returns text when possible, otherwise base64."""
    try:
        content = _graph_request("GET", _content_path(file_path), raw=True)
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
def onedrive_upload_text_file(
    file_path: str,
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
        return _success(item=result)
    except Exception as e:
        logger.error("Error uploading OneDrive text file %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_upload_file(
    file_path: str, remote_path: str = "", mime_type: str = ""
) -> str:
    """
    Upload a local file's real bytes to OneDrive -- use this (not
    onedrive_upload_text_file) for a PDF, image, Office document, or any
    other binary content, including a file this agent already generated
    into the task workspace (e.g. an exported PDF report).

    file_path: path to a file already on disk, e.g. something written to
    the task workspace. Must be inside an allowed directory (automatically
    scoped to the current task workspace) so this tool cannot be used to
    exfiltrate arbitrary files from the host. Pass an absolute path — a
    relative path resolves against this process's own working directory,
    not the allowed directory, and will not find a file written to the
    task workspace.
    remote_path: the OneDrive path to upload to (e.g. "Documents/report.pdf"),
    overwriting any existing file there; defaults to the local file's own
    name at the OneDrive root.
    mime_type: defaults to a guess from the file's extension, falling back
    to "application/octet-stream" when it can't be guessed. Reliably honored
    for files small enough to upload in one request (up to a few MB); for
    larger files uploaded via a resumable session, OneDrive doesn't document
    this as something the caller can override, so an explicit value here may
    not take effect and the file's own name/extension is what actually
    determines its type in that case.

    Rejects a file over _MAX_UPLOAD_BYTES (2 GiB) -- a guard-rail against a
    mistargeted upload, not a OneDrive/Graph limit.
    """
    try:
        local_path = _resolve_upload_file_path(file_path)
        resolved_remote_path = remote_path.strip() or local_path.name
        # _normalize_path strips leading/trailing "/" -- a caller passing
        # "/" or "" ends up with nothing to name the uploaded item, which
        # would otherwise reach _content_path/_item_path as an empty target
        # (a confusing "file_path is required" from the simple-PUT path, or
        # a malformed request against the drive root itself from the
        # upload-session path). Reject explicitly, before either path.
        if not _normalize_path(resolved_remote_path):
            raise ValueError("remote_path must not be empty or just '/'")
        resolved_mime_type = (
            mime_type.strip()
            or _guess_mime_type(local_path.name)
            or "application/octet-stream"
        )

        # Read from one open handle throughout -- the size check, the small-
        # file read, and the large-file chunk reads all use this same fh
        # rather than each re-opening/re-stat'ing the file. For a file over
        # _SIMPLE_UPLOAD_MAX_BYTES, _upload_large_file_content reads it
        # chunk-by-chunk straight off this handle rather than this function
        # first loading the whole file into memory -- the resumable upload-
        # session path exists specifically so a large file's memory
        # footprint stays bounded to one chunk at a time.
        with local_path.open("rb") as fh:
            file_size = os.fstat(fh.fileno()).st_size
            if file_size == 0:
                # A deliberate product choice, not an API constraint: Graph
                # itself accepts a 0-byte file. An agent uploading an empty
                # file is almost always a symptom of an upstream mistake
                # (e.g. a generation step that silently produced nothing),
                # so this is rejected here rather than silently creating a
                # placeholder-empty item on OneDrive.
                raise ValueError(f"File is empty: {file_path}")
            if file_size > _MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"File is {file_size} bytes, over the "
                    f"{_MAX_UPLOAD_BYTES // (1024 * 1024 * 1024)}GB limit for "
                    "onedrive_upload_file"
                )

            if file_size <= _SIMPLE_UPLOAD_MAX_BYTES:
                result = _graph_request(
                    "PUT",
                    _content_path(resolved_remote_path),
                    extra_headers={"Content-Type": resolved_mime_type},
                    data=fh.read(),
                    timeout=_BINARY_UPLOAD_TIMEOUT_SECONDS,
                )
            else:
                result = _upload_large_file_content(
                    resolved_remote_path, fh, file_size, resolved_mime_type
                )

        return _success(item=result)
    except Exception as e:
        logger.error("Error uploading OneDrive file %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_create_folder(
    folder_name: str,
    parent_path: str | None = None,
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
        )
        return _success(folder=result)
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
            f"/me/drive/items/{quote(_normalize_item_id(item_id), safe='')}",
            body={"name": new_name},
        )
        return _success(item=result)
    except Exception as e:
        logger.error("Error renaming OneDrive item %s: %s", item_id, e)
        return _error(str(e))


@mcp.tool()
def onedrive_delete_item(item_id: str) -> str:
    """Delete a OneDrive file or folder by item_id."""
    try:
        _graph_request(
            "DELETE",
            f"/me/drive/items/{quote(_normalize_item_id(item_id), safe='')}",
        )
        return _success(message="Item deleted successfully")
    except Exception as e:
        logger.error("Error deleting OneDrive item %s: %s", item_id, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
