import base64
import io
import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from googleapiclient.http import (  # type: ignore[import-not-found]
    MediaIoBaseDownload,
    MediaIoBaseUpload,
)
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import clamp_limit, require_clean_identifier, setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-drive-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-drive-mcp")

# A Drive/Docs/Sheets/Slides/Forms/Drawings id is always [a-zA-Z0-9_-]+.
_DRIVE_ID_CHARS = re.compile(r"[a-zA-Z0-9_-]+")

# The share-link path prefixes _resolve_file_id recognizes, as one source
# of truth for both patterns below -- so adding a new Drive-family surface
# (this PR already added forms/drawings once) means editing this tuple
# once, not two hand-copied regex literals that could silently drift apart.
# The optional "u/<n>/" branch tolerates the account-index segment Google
# inserts (e.g. ".../file/u/1/d/<id>/view") whenever more than one Google
# account is signed into the browser -- see google_sheets.py's identical
# (?:u/\d+/)? tolerance for the spreadsheets URL shape.
_DRIVE_PATH_PREFIXES = (
    r"file/(?:u/\d+/)?d",
    r"folders",
    r"document/(?:u/\d+/)?d",
    r"spreadsheets/(?:u/\d+/)?d",
    r"presentation/(?:u/\d+/)?d",
    r"forms/(?:u/\d+/)?d",
    r"drawings/(?:u/\d+/)?d",
)
_DRIVE_PATH_ALTERNATION = "|".join(_DRIVE_PATH_PREFIXES)

# Matches the id segment out of any of the common share-link shapes, since a
# user (and therefore a model relaying the user's words) is far more likely
# to hand over the URL they see in the browser than the bare id. "(?!e/)"
# after each "d/" excludes the "published to web" link shape
# (".../d/e/<publish-id>/pubhtml"): that "e" is a literal path segment, not
# part of the id, and the real Drive file id isn't present in that URL at
# all -- the published-to-web token isn't a valid fileId, so this must not
# match rather than confidently return the wrong string ("e"). Applied only
# to a URL's parsed .path (see _resolve_file_id) rather than searched over
# the whole raw string, so a match can't come from an unrelated substring
# elsewhere in the URL (e.g. a query value).
_DRIVE_PATH_ID_PATTERN = re.compile(
    rf"/(?:{_DRIVE_PATH_ALTERNATION})/(?!e/)({_DRIVE_ID_CHARS.pattern})"
)

# Same alternation as _DRIVE_PATH_ID_PATTERN, built from the same
# _DRIVE_PATH_ALTERNATION so the two can't drift apart, but without the
# id-capturing group or the "(?!e/)" exclusion -- used only to recognize
# that a path *is* one of these share-link shapes at all, including the
# shapes deliberately excluded above (e.g. the published-to-web "/d/e/"
# form). See _resolve_file_id: a path matching this but not
# _DRIVE_PATH_ID_PATTERN was recognized-and-rejected, not merely
# unrecognized, and must not fall through to the "?id=" query fallback
# below -- otherwise the fallback would re-derive an id from the very link
# shape the exclusion above refused.
_DRIVE_PATH_SHAPE_PATTERN = re.compile(rf"/(?:{_DRIVE_PATH_ALTERNATION})/")

# Hosts _resolve_file_id treats as an authoritative Drive/Docs/Sheets/Slides
# share link. Without this check, an untrusted URL (e.g. quoted inside a
# document an agent reads) that merely happens to parse with a matching
# path/query shape would have its id extracted and passed to a share/
# delete/permission call as if it were this connector's own link.
# drive.usercontent.google.com is Google's own host for direct-download
# links (e.g. from a "confirm download" interstitial for large files).
_DRIVE_URL_HOSTS = frozenset(
    {"drive.google.com", "docs.google.com", "drive.usercontent.google.com"}
)

# "owner" is deliberately excluded: ownership transfer needs
# transferOwnership=True, is not reversible the way a role change is, and is
# a distinct product decision from "share this with someone" -- one this
# connector doesn't make on the model's behalf. The Shared-Drive-only
# "organizer"/"fileOrganizer" roles are excluded for the same reason: they
# grant administrative control over the drive/folder (managing membership,
# restructuring), which is a materially different, more privileged action
# than the plain collaboration roles below -- not an oversight of
# supportsAllDrives support, which applies independently of which roles
# this connector chooses to expose.
_SHARE_ROLES = ("reader", "commenter", "writer")

# Same shape as api/auth.py's EMAIL_PATTERN, kept as its own local copy
# rather than imported: this module runs as its own subprocess per MCP tool
# call (see get_drive_service's callers), and auth.py is a ~3800-line
# FastAPI router with heavy top-level imports (fastapi, jose, and an
# import-time os.environ mutation) -- pulling that whole module in for one
# regex would add real per-call import cost and an unwarranted dependency
# from a connector tool onto the web API layer. "a@b" (no dot in the
# domain) and "a@"/"@b" (an empty local or domain part) are not valid
# email addresses; the previous "@" not in email check accepted all three.
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Loop-safety bound for google_drive_list_permissions' pagination, not a
# real expected page count (that's governed by output-size capping instead)
# -- just enough to guarantee termination if a misbehaving API response
# kept returning a nextPageToken forever.
_MAX_PERMISSION_LIST_PAGES = 1000

# Google Workspace types with no files.export support at all, for any
# mimeType (confirmed against Google's own export-formats reference) --
# google_drive_get_file_content rejects these outright with an actionable
# message instead of letting them fall into the generic export branch and
# 400 opaquely. Folder/shortcut aren't documents at all, but are included
# here since they share the same "no content to export" outcome and the
# same "application/vnd.google-apps" prefix that would otherwise route
# them into that branch too.
_GOOGLE_APPS_TYPES_WITH_NO_EXPORT = frozenset(
    {
        "application/vnd.google-apps.folder",
        "application/vnd.google-apps.shortcut",
        "application/vnd.google-apps.form",
        "application/vnd.google-apps.site",
    }
)

# Maps a Google Workspace mimeType to what google_drive_get_file_content
# should export it as when the caller left mime_type at its "text/plain"
# default -- either because text/plain isn't a supported export format for
# that type at all (Sheets, Drawings, Apps Script all 400 on it), or
# because a different format is what a caller who didn't specify one
# almost certainly wants. Each value here is genuine text (CSV/SVG-as-XML/
# JSON), so it decodes sensibly through this function's existing
# decode("utf-8", errors="replace") path, unlike Drawings' other export
# options (pdf/png/jpeg), which would come out as garbled replacement
# characters.
_TEXT_PLAIN_EXPORT_FALLBACK = {
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.drawing": "image/svg+xml",
    "application/vnd.google-apps.script": "application/vnd.google-apps.script+json",
}


def _resolve_file_id(file_id: str, field_name: str = "file_id") -> str:
    """Return the id captured out of a Drive/Docs/Sheets/Slides share link,
    or ``file_id`` itself (stripped) when it's already a bare id -- or, for
    anything URL-shaped that isn't a trusted link, unresolved verbatim so an
    obviously-invalid fileId reaches the API instead of a silently wrong one.

    A real Drive id is always ``[a-zA-Z0-9_-]+`` (``_DRIVE_ID_CHARS``) and
    never contains "/" or "?", so only a value containing one of those
    characters is even considered URL-shaped; anything else is the bare-id
    case and is validated against that same character class before being
    accepted (not returned unvalidated) -- otherwise a literal ``".."``
    (no "/" or "?", so it never reaches the URL-handling code below at all)
    would reach the Drive API as a request whose fileId is a dot-segment,
    which normalization can collapse into an unrelated endpoint; this is
    the same risk the "?id=" query fallback further down is validated
    against, just reachable directly instead of via a URL. Empty/
    whitespace-only input is rejected outright (matching
    ``require_clean_identifier``'s treatment of ``email``/``permission_id``
    elsewhere in this file) rather than silently resolving to ``""``, which
    would otherwise reach the Drive API as a request against the bare
    ``.../files/`` collection endpoint instead of a specific file.

    urlparse only populates ``.scheme``/``.netloc`` for a string with an
    explicit "scheme://" prefix (or at least a leading "//"): a scheme-less
    "evil.com/file/d/<id>/view" (at least as natural a shape for an
    attacker to plant in text an agent reads as a full https:// URL) would
    otherwise parse with no netloc and slip past a naive urlparse(value)
    check. Retrying with a "//" prefix only when the *first* parse found no
    netloc (rather than keying off whether "//" occurs anywhere in the
    string) correctly handles a scheme-less link that itself contains a
    nested "https://" further in, e.g. in a "continue=" query value.

    Once the host is confirmed trusted, the id is extracted from the
    parsed URL's own ``.path``/``.query`` -- not searched over the whole
    raw string -- so a URL whose overall host is allowed but whose query
    *value* merely contains a matching shape (".../search?q=/file/d/<id>")
    can't have that unrelated substring extracted as if it were the URL's
    own resource id. A path recognized as a share-link shape that
    _DRIVE_PATH_ID_PATTERN deliberately excludes (the published-to-web
    "/d/e/" form) is rejected outright rather than falling through to the
    "?id=" query check -- otherwise appending "?id=<anything>" to an
    excluded link would trivially re-derive an id the path exclusion just
    refused to give up. A query "id=" value is validated against the same
    id character class as the path form: unlike a path segment, Drive's own
    client library does not percent-encode "." before sending it, so an
    unvalidated "id=.." would reach the API as a literal ".." path segment,
    which dot-segment normalization can collapse into an unrelated
    endpoint.
    """
    if not isinstance(file_id, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = file_id.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be a non-empty id")
    if not re.search(r"[/?]", stripped):
        if not _DRIVE_ID_CHARS.fullmatch(stripped):
            raise ValueError(f"{field_name} must look like a Drive id")
        return stripped
    try:
        parsed = urlparse(stripped)
        if not parsed.netloc:
            parsed = urlparse(f"//{stripped}")
    except ValueError:
        return stripped
    if parsed.hostname not in _DRIVE_URL_HOSTS:
        return stripped
    match = _DRIVE_PATH_ID_PATTERN.search(parsed.path)
    if match:
        return match.group(1)
    if _DRIVE_PATH_SHAPE_PATTERN.search(parsed.path):
        return stripped
    query_id = parse_qs(parsed.query).get("id")
    if query_id and _DRIVE_ID_CHARS.fullmatch(query_id[0]):
        return query_id[0]
    return stripped


def _require_share_role(role: str) -> str:
    if role not in _SHARE_ROLES:
        allowed = ", ".join(_SHARE_ROLES)
        raise ValueError(f"role must be one of {allowed}")
    return role


def _default_halve(value: Any) -> Any:
    return value[: len(value) // 2]


def _capped_response(
    render: Callable[[Any, bool], str],
    value: Any,
    truncated: bool = False,
    halve: Callable[[Any], Any] = _default_halve,
) -> str:
    """Halve ``value`` (a list or a string) until ``render(value, truncated)``
    fits the platform's output limit -- mirrors deputy.py's/salesforce.py's
    _success_with_capped_list. There is no cursor to resume from here, so
    entries/content dropped this way are gone for this call, not just this
    page. Shared by _capped_list_response and _capped_content_response so a
    future change to the halving strategy only has one loop to update.

    ``truncated`` seeds the initial state rather than always starting
    False, so a caller that already knows the value is incomplete for a
    reason this function can't see itself (e.g. a paginated fetch that gave
    up before exhausting every page) can report that honestly even if the
    partial ``value`` it did collect happens to still fit under the limit.

    ``halve`` defaults to a plain half-length slice, but a caller whose
    ``value`` has structure a byte-offset cut could break (e.g. base64,
    where an arbitrary cut produces a string that isn't just incomplete
    but genuinely invalid) can supply an alignment-preserving version.

    Named ``render``, not ``build``, to avoid shadowing the module-level
    ``build`` imported from googleapiclient.discovery.
    """
    max_output_length = get_tool_max_output_length()
    response = render(value, truncated)
    while len(response) > max_output_length and value:
        value = halve(value)
        truncated = True
        response = render(value, truncated)
    return response


def _capped_list_response(
    field_name: str, items: list[Any], *, truncated: bool = False
) -> str:
    def _build(items: list[Any], truncated: bool) -> str:
        return json.dumps(
            {"status": "success", field_name: items, "truncated": truncated},
            ensure_ascii=False,
        )

    return _capped_response(_build, items, truncated)


def _halve_base64(value: str) -> str:
    return value[: (len(value) // 2) // 4 * 4]


def _capped_content_response(
    file_metadata: dict[str, Any], content: str, encoding: str = "utf-8"
) -> str:
    """Same halving idea as _capped_list_response, applied to a single
    string instead of a list since downloaded/exported file content has no
    natural list of records to shrink.

    ``encoding`` tells the caller how ``content`` is represented (see
    google_drive_get_file_content, which falls back to base64 for content
    that isn't valid UTF-8) and picks the matching halving strategy.
    """

    def _build(content: str, truncated: bool) -> str:
        return json.dumps(
            {
                "status": "success",
                "file": file_metadata,
                "content": content,
                "encoding": encoding,
                "truncated": truncated,
            },
            ensure_ascii=False,
        )

    halve = _halve_base64 if encoding == "base64" else _default_halve
    return _capped_response(_build, content, halve=halve)


def get_drive_service() -> Any:
    token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if not token:
        raise ValueError("GOOGLE_ACCESS_TOKEN environment variable is missing")

    creds_kwargs = {"token": token}
    if refresh_token and client_id and client_secret:
        creds_kwargs.update(
            {
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )

    credentials = Credentials(**creds_kwargs)
    return build("drive", "v3", credentials=credentials)


@mcp.tool()
def google_drive_search(query: str = "", max_results: int = 10) -> str:
    """
    Search for files in Google Drive.
    Use query parameter for Google Drive search syntax (e.g. "name contains 'meeting'").
    """
    try:
        page_size = clamp_limit(max_results, max_limit=1000)
        service = get_drive_service()
        results = (
            service.files()
            .list(
                q=query if query else None,
                pageSize=page_size,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            )
            .execute()
        )
        items = results.get("files", [])

        return _capped_list_response("files", items)
    except Exception as e:
        logger.error(f"Error searching drive: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_get_file_content(file_id: str, mime_type: str = "text/plain") -> str:
    """
    Download or export file content from Google Drive by file_id.
    If it's a Google Workspace document (Docs, Sheets), it will be exported to the requested mime_type.
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        service = get_drive_service()
        file_metadata = (
            service.files()
            .get(
                fileId=resolved_file_id,
                supportsAllDrives=True,
                fields="id, name, mimeType",
            )
            .execute()
        )
        file_mime_type = file_metadata.get("mimeType", "")

        if file_mime_type in _GOOGLE_APPS_TYPES_WITH_NO_EXPORT:
            # These all start with "application/vnd.google-apps" and would
            # otherwise fall into the export_media branch below, which
            # makes no sense for any of them (a folder/shortcut has no
            # document content; Forms and Sites have no files.export
            # support at all, for any mimeType) and would fail with an
            # opaque API error instead of an actionable one. Folder/form
            # share links are directly reachable via _resolve_file_id
            # (forms/d/ was added alongside folders/documents/etc.), not a
            # hypothetical.
            raise ValueError(
                f"file_id resolves to a {file_mime_type.rsplit('.', 1)[-1]}, "
                "which has no content to read."
            )

        if "application/vnd.google-apps" in file_mime_type:
            # Export Google Workspace document. The tool's "text/plain"
            # default is reasonable for the common Docs/Slides case, but
            # several other Workspace types either have no text/plain
            # export at all (400s outright) or export something other
            # than what a caller who left mime_type untouched would want.
            # _TEXT_PLAIN_EXPORT_FALLBACK maps each such type to the
            # format that (a) Drive actually supports exporting it as and
            # (b) is genuinely text (CSV/SVG-XML/JSON), so it decodes
            # sensibly below instead of raw binary bytes coming out as
            # garbled replacement characters. Only consulted when the
            # caller left mime_type at its default -- an explicit
            # mime_type is always honored as-is.
            export_mime_type = mime_type
            if mime_type == "text/plain":
                export_mime_type = _TEXT_PLAIN_EXPORT_FALLBACK.get(
                    file_mime_type, mime_type
                )
            request = service.files().export_media(
                fileId=resolved_file_id, mimeType=export_mime_type
            )
        else:
            # Download regular file
            request = service.files().get_media(
                fileId=resolved_file_id, supportsAllDrives=True
            )

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            _, done = downloader.next_chunk()

        raw_bytes = fh.getvalue()
        try:
            # A strict decode, not errors="replace": the export branch
            # above only ever requests genuinely text formats (plain
            # text/CSV/SVG-as-XML/JSON), so a UTF-8 failure there would be
            # a real bug worth surfacing, not something to paper over. The
            # get_media (regular download) branch can return any binary
            # file -- a PDF, image, zip, or non-UTF-8-encoded text file --
            # which replace-decoding would silently corrupt into
            # replacement characters with no signal anything was lost;
            # base64 preserves it exactly, same pattern as onedrive.py's
            # onedrive_get_file_content/_decode_bytes.
            content = raw_bytes.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(raw_bytes).decode("ascii")
            encoding = "base64"

        return _capped_content_response(file_metadata, content, encoding)
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_create_file(
    name: str, content: str, mime_type: str = "text/plain", parent_id: str | None = None
) -> str:
    """
    Create a new file in Google Drive.
    If you want to create a Google Doc, use mime_type="application/vnd.google-apps.document"
    and pass plain text or HTML in the content. For normal text files, use "text/plain".
    """
    try:
        file_metadata: dict[str, Any] = {"name": name, "mimeType": mime_type}
        if parent_id:
            file_metadata["parents"] = [_resolve_file_id(parent_id, "parent_id")]

        service = get_drive_service()
        fh = io.BytesIO(content.encode("utf-8"))

        # When creating a Google Doc, the upload mime type needs to be the original content's mime type (like text/plain)
        upload_mime_type = "text/plain" if "google-apps" in mime_type else mime_type
        media = MediaIoBaseUpload(fh, mimetype=upload_mime_type, resumable=True)

        file = (
            service.files()
            .create(
                body=file_metadata,
                media_body=media,
                supportsAllDrives=True,
                fields="id, name, webViewLink, mimeType",
            )
            .execute()
        )

        return json.dumps({"status": "success", "file": file}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error creating file: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_create_folder(name: str, parent_id: str | None = None) -> str:
    """
    Create a new folder in Google Drive.
    """
    try:
        file_metadata: dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            file_metadata["parents"] = [_resolve_file_id(parent_id, "parent_id")]

        service = get_drive_service()
        folder = (
            service.files()
            .create(
                body=file_metadata,
                supportsAllDrives=True,
                fields="id, name, webViewLink, mimeType",
            )
            .execute()
        )

        return json.dumps({"status": "success", "folder": folder}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_rename_file(file_id: str, new_name: str) -> str:
    """
    Rename an existing file or folder in Google Drive.
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        service = get_drive_service()
        file_metadata = {"name": new_name}

        updated_file = (
            service.files()
            .update(
                fileId=resolved_file_id,
                body=file_metadata,
                supportsAllDrives=True,
                fields="id, name, webViewLink, mimeType",
            )
            .execute()
        )

        return json.dumps(
            {"status": "success", "file": updated_file}, ensure_ascii=False
        )
    except Exception as e:
        logger.error(f"Error renaming file: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


def _is_confirmed_gone(verify_err: Exception) -> bool:
    """Whether ``verify_err`` (raised by re-fetching the object) means it is
    actually gone, i.e. a 404.

    An ``HttpError``'s own ``resp.status`` is the real, unambiguous HTTP
    status code and is checked first when available. Falling back to
    substring-matching "404"/"not found" in ``str(verify_err)`` is fragile
    on its own: ``HttpError.__str__`` embeds the request URI, which
    contains the resolved file/permission id, so a real 403 (object still
    exists; access was merely denied) on an id that happens to contain the
    digits "404" would otherwise be misread as "confirmed gone" --
    reporting a delete/permission-removal that never happened as a success.
    The string fallback stays for non-HttpError exceptions (e.g. a plain
    error from a different failure path, or in tests).
    """
    status = getattr(getattr(verify_err, "resp", None), "status", None)
    if status is not None:
        # httplib2.Response coerces .status to int, so this is always an
        # int in practice via the real client -- try/except rather than an
        # isinstance/str check so a status representation this doesn't
        # anticipate (a different transport, a mock, a future library
        # version) still compares correctly instead of silently skipping
        # the real-status check it exists to provide.
        try:
            return int(status) == 404
        except (TypeError, ValueError):
            pass
    return "404" in str(verify_err) or "not found" in str(verify_err).lower()


def _execute_ignoring_204_ssl_eof(
    execute: Callable[[], Any], verify_done: Callable[[], None]
) -> None:
    """Run a Drive delete-style call that returns 204 No Content, tolerating
    the SSL EOF error a proxy can raise on that empty response.

    httplib2 (via a proxy) can turn a 204 response into
    ``UNEXPECTED_EOF_WHILE_READING`` even though the delete already
    succeeded server-side. When that happens, confirm the object is
    actually gone (``verify_done`` should raise a 404-shaped error once it
    is) before treating the call as successful; any other error propagates
    unchanged.
    """
    try:
        execute()
    except Exception as e:
        if "UNEXPECTED_EOF_WHILE_READING" not in str(e):
            raise
        logger.warning(
            f"Ignored SSL EOF error (often caused by proxy on 204 response): {e}"
        )
        try:
            verify_done()
        except Exception as verify_err:
            if _is_confirmed_gone(verify_err):
                return  # Successfully completed
            raise e from verify_err

        # verify_done() didn't raise, so the object is still there -- the
        # delete/permission removal genuinely did not happen.
        raise Exception(f"Operation did not complete, SSL error occurred: {e}") from e


@mcp.tool()
def google_drive_delete_file(file_id: str) -> str:
    """
    Delete a file or folder in Google Drive.
    Note: this skips the trash and permanently deletes the file -- there is
    no separate "move to trash" tool on this connector, so confirm with the
    user that permanent deletion (not just removing it from view) is what
    they want before calling this.
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        service = get_drive_service()
        _execute_ignoring_204_ssl_eof(
            lambda: (
                service.files()
                .delete(fileId=resolved_file_id, supportsAllDrives=True)
                .execute()
            ),
            lambda: (
                service.files()
                .get(fileId=resolved_file_id, supportsAllDrives=True)
                .execute()
            ),
        )

        return json.dumps(
            {
                "status": "success",
                "message": f"File/Folder {resolved_file_id} successfully deleted.",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_list_permissions(file_id: str) -> str:
    """
    List who currently has access to a Drive file or folder (owner,
    editors, commenters, viewers) and their permission ids. Use the
    returned permission ids with google_drive_update_permission or
    google_drive_remove_permission.
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        service = get_drive_service()
        max_output_length = get_tool_max_output_length()
        permissions: list[Any] = []
        approx_length = 0
        page_token: str | None = None
        # For an item in a Shared Drive, Drive returns at most 100
        # permissions per page when pageSize isn't set (a non-Shared-Drive
        # item always returns everything in one page); follow
        # nextPageToken so a heavily-shared Shared Drive folder isn't
        # silently under-reported. Bounded so a misbehaving API response
        # can't turn this into an infinite loop.
        for _ in range(_MAX_PERMISSION_LIST_PAGES):
            results = (
                service.permissions()
                .list(
                    fileId=resolved_file_id,
                    supportsAllDrives=True,
                    pageToken=page_token,
                    fields=(
                        "nextPageToken, "
                        "permissions(id, type, role, emailAddress, displayName)"
                    ),
                )
                .execute()
            )
            new_permissions = results.get("permissions", [])
            permissions.extend(new_permissions)
            # _capped_list_response below will halve this back down to fit
            # the output limit regardless, so once there's already enough
            # data to exceed it, fetching further pages is pure waste: more
            # blocking network round-trips for permissions that get thrown
            # away immediately after. Tracked incrementally (each new page's
            # items serialized once, not the whole accumulated list every
            # iteration) to stay O(n) rather than O(n^2) for a long
            # permission list; ensure_ascii=False so this length estimate
            # isn't inflated relative to what _capped_list_response will
            # actually measure -- with the default ensure_ascii=True, a
            # non-ASCII displayName/emailAddress (CJK, accented, emoji names
            # are common on a Shared Drive) would each be escaped to a
            # 6+-byte \uXXXX sequence here even though the real response
            # keeps them as 1-2 bytes, overestimating this list as needing
            # to stop paginating even though the true payload would still
            # fit -- and _capped_list_response would then measure the true,
            # smaller size and leave truncated=False, silently under-
            # reporting collaborators while claiming a complete result.
            approx_length += sum(
                len(json.dumps(p, ensure_ascii=False)) for p in new_permissions
            )
            page_token = results.get("nextPageToken")
            if not page_token:
                break
            if approx_length > max_output_length:
                break
        else:
            # The loop ran out of iterations without either a natural
            # "no more pages" or "already too big" stop -- a nextPageToken
            # may still exist that was never followed, so this can't
            # honestly report a complete list even if the permissions
            # collected so far happen to still fit under the output cap.
            return _capped_list_response("permissions", permissions, truncated=True)

        return _capped_list_response("permissions", permissions)
    except Exception as e:
        logger.error(f"Error listing permissions: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_share_file(
    file_id: str,
    email: str,
    role: str = "reader",
    send_notification: bool = True,
    message: str | None = None,
) -> str:
    """
    Grant a user (by email) access to a Drive file or folder, making it
    visible to someone outside this conversation. This is an external
    action -- confirm the target file, the email, and the role with the
    user before calling it.
    role: "reader" (can view), "commenter" (can view and comment), or
    "writer" (can edit). Sharing a folder gives that access to everything
    inside it. When send_notification is True, Google emails the person
    being added; message is included in that email if given. If
    send_notification is False, message is silently discarded (Google's
    API rejects a notification message when no notification is sent) --
    the response won't flag this, so don't rely on the message being
    delivered without also checking send_notification.
    This grants access to a specific email only; it can't create "anyone
    with the link" link-sharing. google_drive_remove_permission can still
    revoke an existing link-shared permission if one is already present.
    """
    try:
        _require_share_role(role)
        require_clean_identifier(email, "email")
        if not _EMAIL_PATTERN.match(email):
            raise ValueError("email must be a valid email address")
        resolved_file_id = _resolve_file_id(file_id)

        service = get_drive_service()
        # The API rejects emailMessage outright when sendNotificationEmail is
        # false, so it's only included when a notification is actually going out.
        create_kwargs: dict[str, Any] = {
            "fileId": resolved_file_id,
            "body": {"type": "user", "role": role, "emailAddress": email},
            "sendNotificationEmail": send_notification,
            "supportsAllDrives": True,
            "fields": "id, type, role, emailAddress, displayName",
        }
        if send_notification and message is not None:
            create_kwargs["emailMessage"] = message

        permission = service.permissions().create(**create_kwargs).execute()

        return json.dumps(
            {"status": "success", "permission": permission}, ensure_ascii=False
        )
    except Exception as e:
        logger.error(f"Error sharing file: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_update_permission(file_id: str, permission_id: str, role: str) -> str:
    """
    Change an existing collaborator's role on a Drive file or folder (e.g.
    upgrade a viewer to an editor). Get permission_id from
    google_drive_list_permissions. This is an external action -- confirm
    the change with the user before calling it.
    role: "reader", "commenter", or "writer".
    """
    try:
        _require_share_role(role)
        resolved_file_id = _resolve_file_id(file_id)
        resolved_permission_id = require_clean_identifier(
            permission_id, "permission_id"
        )

        service = get_drive_service()
        permission = (
            service.permissions()
            .update(
                fileId=resolved_file_id,
                permissionId=resolved_permission_id,
                body={"role": role},
                supportsAllDrives=True,
                fields="id, type, role, emailAddress, displayName",
            )
            .execute()
        )

        return json.dumps(
            {"status": "success", "permission": permission}, ensure_ascii=False
        )
    except Exception as e:
        logger.error(f"Error updating permission: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_remove_permission(file_id: str, permission_id: str) -> str:
    """
    Revoke a collaborator's access to a Drive file or folder. Get
    permission_id from google_drive_list_permissions. This is an external
    action -- confirm who is losing access with the user before calling it.
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        resolved_permission_id = require_clean_identifier(
            permission_id, "permission_id"
        )
        service = get_drive_service()
        _execute_ignoring_204_ssl_eof(
            lambda: (
                service.permissions()
                .delete(
                    fileId=resolved_file_id,
                    permissionId=resolved_permission_id,
                    supportsAllDrives=True,
                )
                .execute()
            ),
            lambda: (
                service.permissions()
                .get(
                    fileId=resolved_file_id,
                    permissionId=resolved_permission_id,
                    supportsAllDrives=True,
                )
                .execute()
            ),
        )

        return json.dumps(
            {
                "status": "success",
                "message": f"Permission {resolved_permission_id} successfully removed.",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error removing permission: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
