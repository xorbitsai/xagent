import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import requests
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ....config import get_tool_max_output_length
from .utils import allowed_dirs_from_env, clamp_limit, setup_proxy_env, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sharepoint-mcp")

setup_proxy_env()

mcp = FastMCP("sharepoint-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30
_BINARY_UPLOAD_TIMEOUT_SECONDS = 120
# Graph's simple content PUT is documented inconsistently -- the OneDrive API
# concepts page says simple upload is "available for items with less than
# 4 MB of content", while the Graph v1.0 API reference for the same endpoint
# says it "supports files up to 250 MB". onedrive.py resolves this by taking
# the smaller, more conservative bound and falling back to a resumable
# upload session above it; that session-based path is not implemented here
# yet, so a file over this bound is rejected outright with a clear error
# rather than silently attempted and possibly failed by Graph.
_SIMPLE_UPLOAD_MAX_BYTES = 4_000_000

# sharepoint_get_file_content buffers the whole download into memory (and then
# again as a base64 string in the tool result), so an unbounded download of a
# large document-library file risks exhausting worker memory and blowing up
# the tool output. Capped at the same order of magnitude as the simple-upload
# limit above.
_MAX_DOWNLOAD_BYTES = 10_000_000

_UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_SHAREPOINT_FILE_ALLOWED_DIRS"


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
    via plain pathlib and are left alone. Matches onedrive.py's identical
    helper.
    """
    suffix = Path(base).suffix
    if not suffix and base.startswith(".") and base.count(".") == 1 and len(base) > 1:
        return "", base
    return Path(base).stem, suffix


# Stable text-extension allowlist, matching onedrive.py's/google_drive.py's
# identical set. Host MIME databases vary, so they cannot safely decide
# whether the text-only tool may use a filename. Unknown extensions are
# rejected; names without an extension remain valid for files such as
# README and Dockerfile. Ambiguous but commonly textual source extensions
# such as .ts and .bat are allowed, while formats with common binary
# variants such as .crt and .plist are not.
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


def _name_looks_binary(name: str) -> bool:
    """Whether ``name``'s extension is NOT a recognized text format.

    Default-deny: an extension is only accepted if it's in
    _KNOWN_TEXT_EXTENSIONS. A name with no extension at all (e.g.
    "Dockerfile", "README") is accepted too -- the absence of an extension
    isn't evidence of binary intent the way an unrecognized one is. Matches
    onedrive.py's identical guard; see _split_stem_suffix above for why it,
    not plain ``Path.suffix``, decides the extension.
    """
    suffix = _split_stem_suffix(Path(name.strip()).name)[1].lower()
    return bool(suffix) and suffix not in _KNOWN_TEXT_EXTENSIONS


# Matches onedrive.py's own _MIME_TYPE_OVERRIDES: stdlib mimetypes.guess_type()
# only recognizes these extensions when a system mime.types file happens to be
# installed, which a minimal/slim host (a stripped-down container image) may
# not have -- verified directly there via MimeTypes(filenames=()) returning
# (None, None) for every one of these.
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


def _success_with_capped_list(
    list_field: str, items: list[Any], *, truncated: bool = False, **extra: Any
) -> str:
    """Build a success payload, halving ``items`` until the response fits
    the platform's output limit.

    _graph_paginate's own ``limit`` cap keeps a normal-sized page well under
    that limit, but a page of items with unusually large per-item content
    (e.g. sharepoint_list_list_items' $expand=fields pulling in long Rich
    Text column values) can still serialize past the runtime's generic
    output filter's fixed character threshold and get hard-truncated into
    broken JSON -- the same corruption sharepoint_get_file_content's own
    output-length guard exists to avoid. Halving (rather than a fixed
    slice) adapts to whatever size a given site's items happen to be, and
    continues down to zero items: a single oversized item must still be
    capped, not returned whole because there's nothing left to halve away
    from it. Matches deputy.py's/salesforce.py's identical
    _success_with_capped_list.
    """

    def _build(items: list[Any], truncated: bool, halved: bool) -> str:
        payload = {list_field: items, "truncated": truncated, **extra}
        if halved:
            payload["message"] = (
                f"Returned {len(items)} {list_field} out of the full result; "
                "the rest did not fit the output size limit and cannot be "
                "recovered via this tool call."
            )
        return _success(**payload)

    max_output_length = get_tool_max_output_length()
    halved = False
    response = _build(items, truncated, halved)
    while len(response) > max_output_length and items:
        items = items[: len(items) // 2]
        truncated = True
        halved = True
        response = _build(items, truncated, halved)
    if halved and len(response) > max_output_length:
        response = _build(items, truncated, False)
    return response


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
    a fully-consumed one: a stream=True response left un-closed on a
    rejection (compressed encoding, an oversized Content-Length, or a body
    that exceeds max_bytes mid-read) would otherwise hold its connection
    until garbage collection gets to it, rather than returning it to the
    pool immediately. Matches mixpanel.py's identical response.close()
    discipline for its own stream=True requests.
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
        # error. response.text is not the same risk -- it's Graph's own
        # JSON error body, which describes the failure but does not itself
        # echo the request URL -- so it's kept for diagnostic detail
        # (matching the non-raw branch below) while the URL-bearing
        # exception/response.url are not.
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
        except requests.RequestException as exc:
            raise _GraphRequestError(
                "network error downloading file content", status_code=0
            ) from exc
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            response_text = response.text.strip()
            message = f"{response.status_code} error downloading file content"
            if response_text:
                message = f"{message} - {response_text}"
            raise _GraphRequestError(message, status_code=response.status_code) from exc
        try:
            return _read_capped_content(response, max_bytes=_MAX_DOWNLOAD_BYTES)
        except requests.RequestException as exc:
            # A connection drop mid-body (after the 200 OK, while streaming
            # via response.iter_content() inside _read_capped_content) is
            # not covered by the try/except above it -- redact it the same
            # way rather than letting it propagate with the download URL
            # embedded in its own message.
            raise _GraphRequestError(
                "network error downloading file content", status_code=0
            ) from exc

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


def _graph_get_absolute(
    url: str, *, timeout: int = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """GET an already-absolute Graph URL (an @odata.nextLink), which carries
    its own host/path/query and must not be re-prefixed with GRAPH_BASE_URL
    the way _graph_request's ``path`` argument is.

    Requires the link to still start with GRAPH_BASE_URL before sending
    this connector's bearer token to it -- matches outlook.py's identical
    _next_link_path guard against a nextLink that's ever off-host (this
    connector's own token must never be handed to an arbitrary URL just
    because Graph's response body named it).
    """
    if not isinstance(url, str) or not url.startswith(f"{GRAPH_BASE_URL}/"):
        raise ValueError("SharePoint returned an invalid pagination next link.")
    # Decode before splitting so an encoded slash cannot hide a dot segment
    # inside one raw segment (matches outlook.py's identical rationale).
    if any(
        segment in {".", ".."} for segment in unquote(urlsplit(url).path).split("/")
    ):
        raise ValueError("SharePoint returned an invalid pagination next link.")
    response = requests.request(
        method="GET", url=url, headers=_graph_headers(), timeout=timeout
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
    payload: dict[str, Any] = response.json()
    return payload


# Defensive stop against a pathological @odata.nextLink chain, well beyond
# any legitimate page count this connector's own $top caps (<= 200 per
# request) would ever need to reach a caller-requested limit.
_MAX_PAGINATION_PAGES = 50


def _graph_paginate(
    path: str, params: dict[str, Any], *, limit: int
) -> tuple[list[dict[str, Any]], bool]:
    """GET a Graph collection, following @odata.nextLink until either
    ``limit`` items are collected or the collection is exhausted.

    Graph pages supported collections independently of the caller's $top --
    a server-side page size smaller than $top is common, and returning only
    the first page would make later records silently inaccessible rather
    than reporting that more exist. Returns (items, truncated): items is
    capped at ``limit``, and truncated is True when the collection has (or
    may have, if _MAX_PAGINATION_PAGES was hit first) more items beyond it.
    """
    items: list[dict[str, Any]] = []
    result = _graph_request("GET", path, params=params)
    items.extend(result.get("value", []))
    next_link = result.get("@odata.nextLink")
    pages = 1
    while next_link and len(items) < limit and pages < _MAX_PAGINATION_PAGES:
        result = _graph_get_absolute(next_link)
        items.extend(result.get("value", []))
        next_link = result.get("@odata.nextLink")
        pages += 1
    truncated = bool(next_link) or len(items) > limit
    return items[:limit], truncated


def _site_segment(site_id: str) -> str:
    """Percent-encode a caller-supplied Graph site identifier for
    interpolation into a URL path segment.

    A Graph site id is one of: the literal "root", a composite id
    ("hostname,spSiteId,spWebId"), or a "hostname:/server-relative-path"
    form -- all three are returned verbatim by sharepoint_search_sites /
    sharepoint_get_root_site / sharepoint_get_site, so callers are expected
    to pass one of those straight back rather than constructing one by
    hand. ':' and '/' are kept unescaped because they are structural to the
    third form; every other character is percent-encoded. A '.'/'..'
    segment is rejected outright (matching onedrive.py's _normalize_path)
    since requests/urllib3 normalize dot-segments out of the final URL,
    which could otherwise walk the request off "/sites/{id}/..." and onto a
    different Graph endpoint under the same token.
    """
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError("site_id is required")
    value = site_id.strip()
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"site_id must not contain '.' or '..' segments: {site_id!r}")
    return quote(value, safe=":/,")


def _site_subresource_base(site_id: str) -> str:
    """Site path prefix for appending a further path segment (e.g.
    "/drives", "/lists"). A path-addressed site id ("hostname:/path") must
    be closed with a second colon before appending more segments -- the
    same convention Graph requires for drive-item path addressing (see
    _drive_children_path) -- otherwise Graph parses the appended segment as
    part of the site's own server-relative path instead of as a
    sub-resource name. The "root" and composite-id ("hostname,siteId,webId")
    forms contain no colon and are returned unchanged."""
    value = site_id.strip()
    segment = _site_segment(site_id)
    if ":" in value and not value.endswith(":"):
        return f"/sites/{segment}:"
    return f"/sites/{segment}"


def _drive_base(site_id: str, drive_id: str | None) -> str:
    site_base = _site_subresource_base(site_id)
    if drive_id:
        return f"{site_base}/drives/{url_path_id(drive_id, 'drive_id')}"
    return f"{site_base}/drive"


def _normalize_relative_path(
    path: str | None, *, field_name: str = "path"
) -> str | None:
    """Normalize a drive-relative path for a root:/{path}: request URL,
    rejecting '.'/'..' segments -- same rationale as _site_segment above.

    field_name lets a caller whose own parameter isn't literally named
    "path" (e.g. "folder_path", or _drive_content_path's own field_name)
    get an error that names its actual argument."""
    if path is None:
        return None
    value = path.strip().strip("/")
    if not value:
        return None
    if "\\" in value:
        raise ValueError(
            f"{field_name} must use '/' separators and must not contain '\\'"
        )
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(
            f"{field_name} must not contain '.' or '..' segments: {path!r}"
        )
    return value


def _drive_children_path(
    site_id: str, folder_path: str | None, drive_id: str | None
) -> str:
    base = _drive_base(site_id, drive_id)
    normalized = _normalize_relative_path(folder_path, field_name="folder_path")
    if not normalized:
        return f"{base}/root/children"
    return f"{base}/root:/{quote(normalized, safe='/')}:/children"


def _drive_content_path(
    site_id: str,
    file_path: str,
    drive_id: str | None,
    *,
    field_name: str = "file_path",
) -> str:
    # field_name lets a caller whose own parameter isn't literally named
    # "file_path" (sharepoint_upload_file's is "remote_path") get an error
    # that names its actual argument instead of a parameter it doesn't
    # have -- matches onedrive.py's identically-parametrized _content_path.
    stripped_path = file_path.strip()
    if stripped_path.endswith("/"):
        raise ValueError(
            f"{field_name} must include a filename, not end with a folder separator"
        )
    if Path(stripped_path).name.endswith("."):
        # Matches onedrive.py's _content_path guard: a trailing-dot filename
        # can be silently normalized by the backing storage (Windows/SharePoint
        # semantics), so e.g. "report.docx." would both bypass
        # sharepoint_upload_text_file's extension denylist (its suffix parses
        # as "", not ".docx") AND land as "report.docx" on the server side --
        # silently overwriting an unrelated real document with mislabeled
        # text content.
        raise ValueError(f"{field_name} {file_path!r} must not end with a period")
    normalized = _normalize_relative_path(file_path, field_name=field_name)
    if not normalized:
        raise ValueError(f"{field_name} is required")
    base = _drive_base(site_id, drive_id)
    return f"{base}/root:/{quote(normalized, safe='/')}:/content"


def _caller_safe_drive_item(item: dict[str, Any]) -> dict[str, Any]:
    """Copy a driveItem without Graph's short-lived preauthenticated URL."""
    safe_item = dict(item)
    safe_item.pop("@microsoft.graph.downloadUrl", None)
    return safe_item


def _decode_bytes(content: bytes) -> tuple[str | None, str | None]:
    try:
        return content.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, base64.b64encode(content).decode("ascii")


def _guess_mime_type(name: str) -> str | None:
    """Guess ``name``'s real mime type: _MIME_TYPE_OVERRIDES first, for the
    formats stdlib mimetypes can't reliably identify on its own, then stdlib
    itself.

    Only the type element of mimetypes.guess_type()'s result is normally
    usable directly -- but when it also reports a non-None *encoding* (e.g.
    "gzip" for ".gz"), the type element describes the *decompressed*
    content, not what's actually going out over the wire:
    mimetypes.guess_type("report.pdf.gz") returns ("application/pdf",
    "gzip"), and blindly using "application/pdf" as Content-Type would tell
    any client trusting that header to parse a raw gzip stream as an
    uncompressed PDF. Falls back to the standard encoding-to-mimetype
    mapping in that case instead, matching onedrive.py's identical guard.
    """
    suffix = _split_stem_suffix(Path(name).name)[1].lower()
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


def _resolve_upload_file_path(local_file_path: str) -> Path:
    """Restrict sharepoint_upload_file to files under an allowlisted
    directory, mirroring onedrive_upload_file's local-file guard.

    Containment is checked before existence, so a path that is both outside
    the allowlist and nonexistent reports the allowlist message, not "not
    found" -- the latter would leak whether that host path exists at all.
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
            "Could not resolve SharePoint upload path %r: %s", local_file_path, exc
        )
        raise ValueError("Could not resolve local_file_path") from exc

    try:
        allowed_dirs = allowed_dirs_from_env(_UPLOAD_ALLOWED_DIRS_ENV_VAR)
    except ValueError as exc:
        logger.warning("Invalid SharePoint upload directory configuration: %s", exc)
        raise ValueError("Upload directory configuration is invalid") from None
    if not any(local_path.is_relative_to(d) for d in allowed_dirs):
        logger.warning(
            "Rejected sharepoint_upload_file path %s outside allowed directories: %s",
            local_path,
            ", ".join(str(path) for path in allowed_dirs),
        )
        raise PermissionError(
            "local_file_path is outside the allowed upload directories; ask the "
            "user for a file inside the task workspace or another allowed "
            "location"
        )

    if local_path.exists() and not local_path.is_file():
        raise ValueError("The given path is not a regular file")
    if not local_path.is_file():
        raise FileNotFoundError("File not found at the given path")
    return local_path


def _parse_fields_json(fields_json: str) -> dict[str, Any]:
    """Parse a SharePoint list item's column values from a JSON object
    string. Taking a JSON string (rather than a raw dict parameter) matches
    google_slides_batch_update's precedent for an open-ended, caller-defined
    payload -- a SharePoint list's columns are defined per-list by whoever
    created it, so there is no fixed schema this tool could validate against.
    """
    try:
        parsed = json.loads(fields_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"fields_json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("fields_json must decode to a JSON object")
    return parsed


@mcp.tool()
def sharepoint_search_sites(query: str, top: int = 25) -> str:
    """Search SharePoint sites by keyword. Returns each matching site's id
    (needed by every other sharepoint_* tool) alongside its name and URL."""
    try:
        if not query.strip():
            raise ValueError("query is required")
        capped_top = clamp_limit(top, max_limit=100)
        sites, truncated = _graph_paginate(
            "/sites", {"search": query, "$top": capped_top}, limit=capped_top
        )
        return _success_with_capped_list("sites", sites, truncated=truncated)
    except Exception as e:
        logger.error("Error searching SharePoint sites: %s", e)
        return _error(str(e))


@mcp.tool()
def sharepoint_get_root_site() -> str:
    """Get the tenant's root SharePoint site, including its id."""
    try:
        result = _graph_request("GET", "/sites/root")
        return _success(site=result)
    except Exception as e:
        logger.error("Error getting SharePoint root site: %s", e)
        return _error(str(e))


@mcp.tool()
def sharepoint_get_site(site_id: str) -> str:
    """Get a SharePoint site's metadata by its Graph site id (from
    sharepoint_search_sites or sharepoint_get_root_site)."""
    try:
        result = _graph_request("GET", f"/sites/{_site_segment(site_id)}")
        return _success(site=result)
    except Exception as e:
        logger.error("Error getting SharePoint site %s: %s", site_id, e)
        return _error(str(e))


@mcp.tool()
def sharepoint_list_drives(site_id: str) -> str:
    """List the document libraries (drives) in a SharePoint site."""
    try:
        drives, truncated = _graph_paginate(
            f"{_site_subresource_base(site_id)}/drives", {"$top": 200}, limit=200
        )
        return _success_with_capped_list("drives", drives, truncated=truncated)
    except Exception as e:
        logger.error("Error listing SharePoint drives for site %s: %s", site_id, e)
        return _error(str(e))


@mcp.tool()
def sharepoint_list_items(
    site_id: str,
    folder_path: str | None = None,
    drive_id: str | None = None,
    top: int = 50,
) -> str:
    """List files and folders in a SharePoint document library, optionally
    under a folder path. Uses the site's default document library unless
    drive_id (from sharepoint_list_drives) is given."""
    try:
        capped_top = clamp_limit(top, max_limit=200)
        items, truncated = _graph_paginate(
            _drive_children_path(site_id, folder_path, drive_id),
            {"$top": capped_top},
            limit=capped_top,
        )
        return _success_with_capped_list(
            "items",
            [_caller_safe_drive_item(item) for item in items],
            truncated=truncated,
        )
    except Exception as e:
        logger.error(
            "Error listing SharePoint items for site %s under %s: %s",
            site_id,
            folder_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def sharepoint_search_files(
    site_id: str, query: str, drive_id: str | None = None, top: int = 25
) -> str:
    """Search files and folders in a SharePoint document library by keyword."""
    try:
        if not query.strip():
            raise ValueError("query is required")
        escaped_query = query.replace("'", "''")
        base = _drive_base(site_id, drive_id)
        capped_top = clamp_limit(top, max_limit=100)
        items, truncated = _graph_paginate(
            f"{base}/root/search(q='{quote(escaped_query, safe='')}')",
            {"$top": capped_top},
            limit=capped_top,
        )
        return _success_with_capped_list(
            "items",
            [_caller_safe_drive_item(item) for item in items],
            truncated=truncated,
        )
    except Exception as e:
        logger.error("Error searching SharePoint files for site %s: %s", site_id, e)
        return _error(str(e))


@mcp.tool()
def sharepoint_get_file_content(
    site_id: str, file_path: str, drive_id: str | None = None
) -> str:
    """Download a file's content from a SharePoint document library by path.
    Returns text when possible, otherwise base64. Rejected outright, rather
    than returned truncated, when the encoded result would be too large for
    a single tool response."""
    try:
        content = _graph_request(
            "GET", _drive_content_path(site_id, file_path, drive_id), raw=True
        )
        # The runtime's generic output filter truncates any string value --
        # the JSON envelope _success returns below is one -- at
        # get_tool_max_output_length() characters without parsing it first,
        # so an encoded payload that slips past that limit would come back
        # as invalid JSON with a '"status": "success"' prefix still intact:
        # silent corruption rather than a clear failure. Rejecting it here,
        # before that filter ever sees it, trades that for an honest,
        # actionable error.
        max_output_length = get_tool_max_output_length()
        text_content, base64_content = _decode_bytes(content)
        if base64_content is not None:
            # Base64 inflates the byte count by exactly 4 * ceil(n/3) --
            # deterministically larger than len(content), so this is a
            # safe, exact lower bound on the eventual JSON string's length
            # without needing to build (and immediately discard) the full
            # envelope to find out. This does NOT hold for the text_content
            # branch below: decoding multi-byte UTF-8 (CJK, emoji, ...)
            # produces FEWER characters than input bytes, so a byte-count
            # check there would reject files that would have fit -- that
            # branch instead falls through to the exact len(result) check.
            if len(base64_content) > max_output_length:
                return _error(
                    f"{file_path!r} is too large to return inline "
                    f"({len(content)} bytes downloaded, limit is "
                    f"{max_output_length} characters)"
                )
        result = _success(
            file_path=file_path,
            text_content=text_content,
            base64_content=base64_content,
            encoding="utf-8" if text_content is not None else "base64",
        )
        if len(result) > max_output_length:
            return _error(
                f"{file_path!r} is too large to return inline "
                f"({len(content)} bytes downloaded, {len(result)} characters "
                f"encoded, limit is {max_output_length} characters)"
            )
        return result
    except Exception as e:
        logger.error(
            "Error downloading SharePoint file %s for site %s: %s",
            file_path,
            site_id,
            e,
        )
        return _error(str(e))


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True))
def sharepoint_upload_text_file(
    site_id: str,
    file_path: str,
    content: str,
    drive_id: str | None = None,
) -> str:
    """Upload or overwrite a UTF-8 text file in a SharePoint document
    library by path.

    content is always treated as text (UTF-8 encoded before upload) -- this
    tool cannot create a real .docx/.xlsx/.pptx/.pdf or other binary
    document. To upload an already-generated local file's real bytes, use
    sharepoint_upload_file with that file's path instead.
    """
    try:
        if _name_looks_binary(file_path):
            return _error(
                f"'{file_path}' looks like a binary file, but "
                "sharepoint_upload_text_file only writes text content -- "
                "uploading it here would produce a mislabeled, broken file. "
                "If you already have this file on disk (e.g. as a task "
                "output), use sharepoint_upload_file with its local path to "
                "upload the real binary content instead."
            )
        result = _graph_request(
            "PUT",
            _drive_content_path(site_id, file_path, drive_id),
            extra_headers={"Content-Type": "text/plain; charset=utf-8"},
            data=content.encode("utf-8"),
        )
        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("SharePoint did not confirm the upload completed")
        return _success(item=_caller_safe_drive_item(result))
    except Exception as e:
        logger.error(
            "Error uploading SharePoint text file %s for site %s: %s",
            file_path,
            site_id,
            e,
        )
        return _error(str(e))


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True))
def sharepoint_upload_file(
    site_id: str,
    local_file_path: str,
    remote_path: str = "",
    drive_id: str | None = None,
    mime_type: str = "",
) -> str:
    """Upload a local file's real bytes to a SharePoint document library --
    use this (not sharepoint_upload_text_file) for a PDF, image, Office
    document, or any other binary content.

    local_file_path: path to a file already on disk (e.g. in the task
    workspace). Must be inside an allowed directory.
    remote_path: the document library path to upload to, overwriting any
    existing file there; defaults to the local file's own name at the
    library root.

    Limited to files up to 4 MB (a resumable upload session for larger
    files, like onedrive_upload_file supports, is not yet implemented here).
    """
    try:
        local_path = _resolve_upload_file_path(local_file_path)
        resolved_remote_path = remote_path.strip() or local_path.name
        content_path = _drive_content_path(
            site_id, resolved_remote_path, drive_id, field_name="remote_path"
        )
        resolved_mime_type = mime_type.strip() or _guess_mime_type(resolved_remote_path)
        if resolved_mime_type is None:
            resolved_mime_type = (
                _guess_mime_type(local_path.name) or "application/octet-stream"
            )

        # Open once and read the size from that same descriptor (rather than
        # a separate Path.stat() before opening) so the size check and the
        # bytes actually uploaded are bound to one file description -- a
        # symlink swapped in between a separate stat() and open() could
        # otherwise upload different content than what was size-checked.
        try:
            fh_ctx = local_path.open("rb")
        except OSError as e:
            # str(OSError) embeds the absolute host path (e.g. "[Errno 13]
            # Permission denied: '/full/host/path'"), which the generic
            # `except Exception` below would otherwise leak straight into
            # the caller/LLM-facing error message.
            logger.warning("Failed to open upload file %s: %s", local_path, e)
            raise ValueError("Could not read the file at the given path") from e
        with fh_ctx as fh:
            file_size = os.fstat(fh.fileno()).st_size
            if file_size == 0:
                raise ValueError(f"File is empty: {local_file_path}")
            if file_size > _SIMPLE_UPLOAD_MAX_BYTES:
                raise ValueError(
                    f"File is {file_size} bytes, over the "
                    f"{_SIMPLE_UPLOAD_MAX_BYTES // 1_000_000} MB limit "
                    "sharepoint_upload_file currently supports"
                )
            content = fh.read(_SIMPLE_UPLOAD_MAX_BYTES + 1)
        if not content:
            # A concurrent truncation of the file between the fstat() size
            # check above and this read() would otherwise fall through
            # (content == b"" is not > _SIMPLE_UPLOAD_MAX_BYTES) and PUT an
            # empty body to SharePoint instead of raising -- matches
            # onedrive.py's identical re-check for the same race.
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
        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("SharePoint did not confirm the upload completed")
        return _success(item=_caller_safe_drive_item(result))
    except Exception as e:
        if isinstance(e, (OSError, ValueError, RuntimeError, PermissionError)):
            logger.error(
                "Error uploading SharePoint file %s for site %s: %s",
                local_file_path,
                site_id,
                e,
            )
        else:
            logger.exception(
                "Unexpected error uploading SharePoint file %s for site %s",
                local_file_path,
                site_id,
            )
        return _error(str(e))


@mcp.tool()
def sharepoint_list_lists(site_id: str, top: int = 50) -> str:
    """List the SharePoint lists (e.g. custom lists, document metadata
    lists) in a site."""
    try:
        capped_top = clamp_limit(top, max_limit=200)
        lists, truncated = _graph_paginate(
            f"{_site_subresource_base(site_id)}/lists",
            {"$top": capped_top},
            limit=capped_top,
        )
        return _success_with_capped_list("lists", lists, truncated=truncated)
    except Exception as e:
        logger.error("Error listing SharePoint lists for site %s: %s", site_id, e)
        return _error(str(e))


@mcp.tool()
def sharepoint_list_list_items(site_id: str, list_id: str, top: int = 50) -> str:
    """List items in a SharePoint list, including each item's column values.

    list_id accepts either the list's Graph id or its display name."""
    try:
        capped_top = clamp_limit(top, max_limit=200)
        items, truncated = _graph_paginate(
            f"{_site_subresource_base(site_id)}/lists/{url_path_id(list_id, 'list_id')}/items",
            {"$top": capped_top, "$expand": "fields"},
            limit=capped_top,
        )
        return _success_with_capped_list("items", items, truncated=truncated)
    except Exception as e:
        logger.error(
            "Error listing SharePoint list items for site %s list %s: %s",
            site_id,
            list_id,
            e,
        )
        return _error(str(e))


@mcp.tool(annotations=ToolAnnotations(destructiveHint=False, idempotentHint=False))
def sharepoint_create_list_item(site_id: str, list_id: str, fields_json: str) -> str:
    """Create an item in a SharePoint list.

    fields_json is a JSON object string of column-name to value pairs, e.g.
    '{"Title": "New task", "Status": "Not Started"}'. Column names and
    valid values are list-specific; use sharepoint_list_list_items or the
    list's own settings to discover them.

    This is not idempotent: retrying after a timeout or connection error
    can create a duplicate item. Use sharepoint_list_list_items to check
    whether the item already exists before retrying a failed call."""
    try:
        fields = _parse_fields_json(fields_json)
        result = _graph_request(
            "POST",
            f"{_site_subresource_base(site_id)}/lists/{url_path_id(list_id, 'list_id')}/items",
            body={"fields": fields},
        )
        return _success(item=result)
    except Exception as e:
        logger.error(
            "Error creating SharePoint list item for site %s list %s: %s",
            site_id,
            list_id,
            e,
        )
        return _error(str(e))


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True))
def sharepoint_update_list_item(
    site_id: str, list_id: str, item_id: str, fields_json: str
) -> str:
    """Update an existing SharePoint list item's column values.

    fields_json is a JSON object string of the column-name to value pairs
    to change; unset columns are left untouched."""
    try:
        fields = _parse_fields_json(fields_json)
        if not fields:
            raise ValueError("fields_json must contain at least one field to update")
        result = _graph_request(
            "PATCH",
            f"{_site_subresource_base(site_id)}/lists/{url_path_id(list_id, 'list_id')}"
            f"/items/{url_path_id(item_id, 'item_id')}/fields",
            body=fields,
        )
        return _success(item=result)
    except Exception as e:
        logger.error(
            "Error updating SharePoint list item %s for site %s list %s: %s",
            item_id,
            site_id,
            list_id,
            e,
        )
        return _error(str(e))


@mcp.tool(annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True))
def sharepoint_delete_list_item(site_id: str, list_id: str, item_id: str) -> str:
    """Delete an item from a SharePoint list."""
    try:
        _graph_request(
            "DELETE",
            f"{_site_subresource_base(site_id)}/lists/{url_path_id(list_id, 'list_id')}"
            f"/items/{url_path_id(item_id, 'item_id')}",
        )
        return _success(message="List item deleted successfully")
    except Exception as e:
        logger.error(
            "Error deleting SharePoint list item %s for site %s list %s: %s",
            item_id,
            site_id,
            list_id,
            e,
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
