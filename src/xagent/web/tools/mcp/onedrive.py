import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from .utils import allowed_dirs_from_env, setup_proxy_env, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onedrive-mcp")

setup_proxy_env()

mcp = FastMCP("onedrive-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30
# Allow more time for binary content than for small Graph JSON requests.
_BINARY_UPLOAD_TIMEOUT_SECONDS = 120
# Deliberate limit for this tool's single-request implementation.
# Larger files require upload-session support, delivered separately.
_SIMPLE_UPLOAD_MAX_BYTES = 4_000_000


_UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS"

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
        raise RuntimeError(message) from exc

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
            "local_file_path is outside the allowed upload directories; ask the "
            "user for a file inside the task workspace or another allowed "
            "location"
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
                f"/me/drive/items/{url_path_id(item_id, 'item_id')}",
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
        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("OneDrive did not confirm the upload completed")
        return _success(item=result)
    except Exception as e:
        logger.error("Error uploading OneDrive text file %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def onedrive_upload_file(
    local_file_path: str, remote_path: str = "", mime_type: str = ""
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
    remote_path: the OneDrive path to upload to (e.g. "Documents/report.pdf"),
    overwriting any existing file there; defaults to the local file's own
    name at the OneDrive root.
    mime_type: defaults to a guess from the remote filename, then the local
    filename, falling back to "application/octet-stream".

    Supports non-empty files up to 4,000,000 bytes (4 MB). Larger files
    are rejected before any upload request is sent.
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

        # Check size and read content through the same open handle.
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
            # A stat-then-unbounded-read sequence can exceed the limit if
            # another process appends to the file between those operations.
            # Reading one byte past the cap bounds memory and makes the bytes
            # actually sent authoritative for both the empty and size checks.
            content = fh.read(_SIMPLE_UPLOAD_MAX_BYTES + 1)
            if not content:
                # A deliberate product choice, not an API constraint: Graph
                # itself accepts a 0-byte file. An agent uploading an empty
                # file is almost always a symptom of an upstream mistake
                # (e.g. a generation step that silently produced nothing),
                # so this is rejected here rather than silently creating a
                # placeholder-empty item on OneDrive.
                raise ValueError(f"File is empty: {local_file_path}")
            if len(content) > _SIMPLE_UPLOAD_MAX_BYTES:
                raise ValueError(
                    f"File is over the {_SIMPLE_UPLOAD_MAX_BYTES:,}-byte "
                    "(4 MB) limit for "
                    "onedrive_upload_file. Large-file uploads are not supported yet."
                )

        result = _graph_request(
            "PUT",
            content_path,
            extra_headers={"Content-Type": resolved_mime_type},
            data=content,
            timeout=_BINARY_UPLOAD_TIMEOUT_SECONDS,
        )
        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("OneDrive did not confirm the upload completed")

        return _success(item=result)
    except Exception as e:
        logger.error("Error uploading OneDrive file %s: %s", local_file_path, e)
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
            f"/me/drive/items/{url_path_id(item_id, 'item_id')}",
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
            f"/me/drive/items/{url_path_id(item_id, 'item_id')}",
        )
        return _success(message="Item deleted successfully")
    except Exception as e:
        logger.error("Error deleting OneDrive item %s: %s", item_id, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
