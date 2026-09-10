import base64
import io
import json
import logging
import mimetypes
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, parse_qs, urlparse

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from googleapiclient.http import (  # type: ignore[import-not-found]
    MediaIoBaseDownload,
    MediaIoBaseUpload,
)
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import (
    clamp_limit,
    require_clean_identifier,
    setup_proxy_env,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-drive-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-drive-mcp")

# A Drive/Docs/Sheets/Slides/Forms/Drawings id is always [a-zA-Z0-9_-]+.
_DRIVE_ID_CHARS = re.compile(r"[a-zA-Z0-9_-]+")

# A real Drive id never contains "/" or "?" -- used by both
# _parse_trusted_drive_url and _resolve_file_id to decide "is this even
# URL-shaped" before doing any URL parsing. Precompiled to match every
# other regex in this module rather than left as a literal pattern string
# re-evaluated at each of its two call sites.
_DRIVE_URL_SHAPE_HINT = re.compile(r"[/?]")

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
# elsewhere in the URL (e.g. a query value). The trailing "(?=/|$)" requires
# the captured id to run all the way to the next path separator (or the end
# of the path) -- without it, a malformed segment like ".../file/d/ABC.DEF/
# view" would silently capture just "ABC" (the id charset stops at the
# "."), acting on a truncated, wrong id instead of falling through to the
# "not a recognized shape" branch and returning the whole URL unresolved.
_DRIVE_PATH_ID_PATTERN = re.compile(
    rf"/(?:{_DRIVE_PATH_ALTERNATION})/(?!e/)({_DRIVE_ID_CHARS.pattern})(?=/|$)"
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

# "domain" and "anyone" are deliberately excluded, matching
# google_drive_share_file's own docstring ("can't share with an entire
# domain" / no "anyone with the link" support): both grant access to a much
# broader audience than "one specific user or group", which is a distinct
# product decision this connector doesn't make on the model's behalf, the
# same reasoning _SHARE_ROLES above already applies to "owner"/"organizer".
_SHARE_ENTITY_TYPES = ("user", "group")

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


def _parse_trusted_drive_url(value: str) -> ParseResult | None:
    """Return the parsed URL if ``value`` is URL-shaped, parses cleanly,
    carries no userinfo ambiguity, and its host is in _DRIVE_URL_HOSTS --
    otherwise ``None``.

    This is the one place that decides "is this a URL this connector
    trusts as its own Drive share link", shared by _resolve_file_id and
    _extract_resource_key so that decision can't drift between them.
    Before this was factored out, the two functions each carried their
    own copy of this logic, and they *did* silently disagree: a
    published-to-web "/d/e/" link that _resolve_file_id correctly refuses
    to resolve an id for was nonetheless treated as trusted enough by
    _extract_resource_key to pull a resourcekey out of -- the id-shape
    exclusion lived only in _resolve_file_id's copy. Sharing this parse
    step means a future fix to the trust decision (a new excluded shape,
    another parser-differential case, etc.) applies to both call sites at
    once instead of needing to be found and re-applied twice.

    urlparse only populates ``.scheme``/``.netloc`` for a string with an
    explicit "scheme://" prefix (or at least a leading "//"): a scheme-less
    "evil.com/file/d/<id>/view" (at least as natural a shape for an
    attacker to plant in text an agent reads as a full https:// URL) would
    otherwise parse with no netloc and slip past a naive urlparse(value)
    check. Retrying with a "//" prefix only when the *first* parse found no
    netloc (rather than keying off whether "//" occurs anywhere in the
    string) correctly handles a scheme-less link that itself contains a
    nested "https://" further in, e.g. in a "continue=" query value.

    A userinfo-bearing authority ("evil.com@drive.google.com", or the same
    thing with a literal backslash before the "@") is inherently
    ambiguous: Python's parser (correctly, per RFC 3986) treats everything
    before the last "@" as userinfo and resolves .hostname to the part
    after it, so this would pass the trusted-host check below -- but a
    WHATWG-based parser (e.g. a browser rendering this same string to a
    human, or a different URL consumer downstream) can disagree about
    which side is the real destination. Reject outright rather than trust
    a parse that other consumers of the same string might not agree with.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not _DRIVE_URL_SHAPE_HINT.search(stripped):
        return None
    try:
        parsed = urlparse(stripped)
        if not parsed.netloc:
            parsed = urlparse(f"//{stripped}")
    except ValueError:
        return None
    if parsed.username is not None:
        return None
    if parsed.hostname not in _DRIVE_URL_HOSTS:
        return None
    return parsed


def _resolve_id_from_parsed(
    parsed: ParseResult, query_params: dict[str, list[str]] | None = None
) -> str | None:
    """Given an already-trusted, parsed Drive URL (see
    _parse_trusted_drive_url), return the real fileId it resolves to, or
    ``None`` if it doesn't resolve to one at all -- a recognized-but-
    excluded path shape (e.g. the published-to-web "/d/e/" form), or a
    path/query that doesn't match any known share-link shape.

    The single source of truth for "does this URL resolve to a real
    fileId", shared by _resolve_file_id and _extract_resource_key so their
    answers to that question can't independently drift -- they did,
    twice: once for the "/d/e/" exclusion specifically (fixed by factoring
    out _parse_trusted_drive_url for the host/userinfo trust decision),
    and once for any other unrecognized path shape (fixed by factoring out
    this function for the id-resolution decision itself). A caller that
    already has ``parsed.query`` parsed for its own purposes (e.g.
    _extract_resource_key, which needs it again for "resourcekey") can
    pass ``query_params`` to avoid parsing it a second time here.
    """
    match = _DRIVE_PATH_ID_PATTERN.search(parsed.path)
    if match:
        return match.group(1)
    if _DRIVE_PATH_SHAPE_PATTERN.search(parsed.path):
        return None
    if query_params is None:
        query_params = parse_qs(parsed.query)
    query_id = query_params.get("id")
    if query_id and _DRIVE_ID_CHARS.fullmatch(query_id[0]):
        return query_id[0]
    return None


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

    Once the host is confirmed trusted (see _parse_trusted_drive_url), the
    id is extracted via _resolve_id_from_parsed -- not searched over the
    whole raw string -- so a URL whose overall host is allowed but whose
    query *value* merely contains a matching shape
    (".../search?q=/file/d/<id>") can't have that unrelated substring
    extracted as if it were the URL's own resource id. A path recognized
    as a share-link shape that _DRIVE_PATH_ID_PATTERN deliberately
    excludes (the published-to-web "/d/e/" form) is rejected outright
    rather than falling through to the "?id=" query check -- otherwise
    appending "?id=<anything>" to an excluded link would trivially
    re-derive an id the path exclusion just refused to give up. A query
    "id=" value is validated against the same id character class as the
    path form: unlike a path segment, Drive's own client library does not
    percent-encode "." before sending it, so an unvalidated "id=.." would
    reach the API as a literal ".." path segment, which dot-segment
    normalization can collapse into an unrelated endpoint.
    """
    if not isinstance(file_id, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = file_id.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be a non-empty id")
    if not _DRIVE_URL_SHAPE_HINT.search(stripped):
        # Shares _require_drive_id's character-class check rather than
        # repeating it inline, so file_id's bare-id validation and
        # permission_id's validation can't silently drift apart the way
        # _resolve_file_id/_extract_resource_key's trust logic once did
        # (see _parse_trusted_drive_url). require_clean_identifier's own
        # empty/whitespace check is a no-op here -- stripped is already
        # non-empty and already stripped -- so the only error this can
        # raise is the character-class one.
        return _require_drive_id(stripped, field_name)
    parsed = _parse_trusted_drive_url(stripped)
    if parsed is None:
        return stripped
    resolved = _resolve_id_from_parsed(parsed)
    return resolved if resolved is not None else stripped


def _extract_resource_key(file_id: str) -> str | None:
    """Return the ``resourcekey`` query parameter from a Drive share link,
    or ``None`` if ``file_id`` isn't URL-shaped, isn't a trusted Drive host,
    or carries no resourcekey.

    Google added resourcekeys in 2021 for items shared via link before that
    date: such an item's plain fileId alone now 404s on files.get/
    permissions.list/files.export/etc. unless the request also carries an
    "X-Goog-Drive-Resource-Keys: <fileId>/<resourceKey>" header (see
    google_drive_download.py/kb.py for the same header elsewhere in this
    codebase) -- there is no separate lookup API for a resourcekey; the only
    place it's ever exposed is the "?resourcekey=" query parameter Drive
    puts on the share link itself.

    Shares _parse_trusted_drive_url and _resolve_id_from_parsed with
    _resolve_file_id rather than keeping its own copy of either the host/
    userinfo trust logic or the id-resolution decision -- covering not
    just the published-to-web "/d/e/" exclusion but any other trusted-host
    URL whose path isn't a recognized share-link shape at all. Sharing
    _resolve_id_from_parsed (rather than each function keeping its own
    copy of that three-branch decision) guarantees that whenever this
    function returns a resourcekey, _resolve_file_id on the same input
    made the identical decision and returned a real, _DRIVE_ID_CHARS-
    validated fileId -- never the raw unresolved URL string -- which is
    also what keeps _attach_resource_key's header value free of characters
    (including a literal CR/LF) the fileId half was never validated
    against. Kept as its own function rather than folding into
    _resolve_file_id and returning a tuple: _resolve_file_id's plain-string
    return is relied on by every call site and by the full existing test
    suite, and widening its contract for a value that's usually None (only
    pre-2021 link-shared items carry a resourcekey at all) isn't worth that
    blast radius.
    """
    parsed = _parse_trusted_drive_url(file_id)
    if parsed is None:
        return None
    query_params = parse_qs(parsed.query)
    if _resolve_id_from_parsed(parsed, query_params) is None:
        return None
    resource_key = query_params.get("resourcekey")
    if not resource_key:
        return None
    candidate = resource_key[0]
    # A real resourcekey isn't restricted to _DRIVE_ID_CHARS's fileId
    # charset -- kb.py's own CloudFile.resourceKey field (api/kb.py:3319)
    # deliberately validates only `^[^\r\n]*$`, rejecting nothing but
    # CR/LF (header-injection safety) rather than inventing a narrower
    # grammar; tests/web/api/test_kb_dir.py's
    # test_kb_ingest_cloud_accepts_punctuation_in_drive_identifiers asserts
    # exactly this for a resourceKey containing a colon. Matching that
    # precedent here avoids silently discarding a legitimate resourcekey
    # (and getting a confusing 404) just because it contains a character
    # this connector didn't anticipate.
    if not candidate or "\r" in candidate or "\n" in candidate:
        return None
    return candidate


def _attach_resource_key(request: Any, file_id: str, resource_key: str | None) -> Any:
    """Attach the X-Goog-Drive-Resource-Keys header to an unexecuted
    googleapiclient request when ``resource_key`` is present, mirroring
    google_drive_download.py's ``request.headers.update(...)`` pattern.
    Returns ``request`` so the call site can build/attach/execute in one
    expression.

    ``file_id`` is expected to already be _DRIVE_ID_CHARS-clean by this
    point -- every call site passes the result of _resolve_file_id, and
    _extract_resource_key only ever returns a truthy ``resource_key`` when
    _resolve_id_from_parsed made the identical decision on the same input
    and _resolve_file_id would therefore have taken the same validated-id
    branch (see _extract_resource_key's docstring). Note this is narrower
    than "file_id is always safe everywhere": an untrusted/unresolved
    file_id can still reach the Drive API as the request's own ``fileId=``
    path parameter with no header involved at all -- that path is
    protected separately, by googleapiclient's own URI-template expansion
    percent-encoding reserved characters (including a literal CR/LF)
    before it ever reaches the wire, not by anything in this function.

    The CR/LF check below only guards the one thing THIS function builds:
    the resourcekey header value. It's belt-and-suspenders defense against
    a future call site breaking the invariant above -- raising, rather
    than silently skipping the header, since a silent skip would be a
    *worse* failure mode than surfacing the violation: a pre-2021 link-
    shared item that genuinely needs this header would otherwise get a
    confusing 404 from Drive with no indication a resourcekey was computed
    but silently dropped. Raising here signals a bug in this file's own
    invariant, not a user-input problem -- logged before raising so it's
    distinguishable in logs from an ordinary validation error, even though
    the outer per-tool try/except still turns it into the same JSON error
    envelope as any other exception.
    """
    if not resource_key:
        return request
    if "\r" in file_id or "\n" in file_id:
        logger.error(
            "_attach_resource_key: file_id contains CR/LF, violating the "
            "caller invariant this function relies on -- this indicates a "
            "bug in google_drive.py, not bad user input: %r",
            file_id,
        )
        raise ValueError(
            "internal error (bug in google_drive.py): file_id must not "
            f"contain CR/LF, got {file_id!r}"
        )
    request.headers["X-Goog-Drive-Resource-Keys"] = f"{file_id}/{resource_key}"
    return request


def _apply_parent_id(
    file_metadata: dict[str, Any], parent_id: str | None
) -> tuple[str | None, str | None]:
    """Resolve an optional ``parent_id`` (a bare id or a folder URL/link,
    possibly carrying a pre-2021 resourcekey) and, if given, add it to
    ``file_metadata["parents"]`` in place.

    Returns ``(resolved_parent_id, parent_resource_key)`` — both ``None``
    when ``parent_id`` is falsy — for the caller to pass to
    ``_attach_resource_key`` once its own request object exists (that
    happens after ``file_metadata`` is built, so it can't be folded into
    this function). Shared by google_drive_upload_file, ..._create_file,
    and ..._create_folder, which otherwise each duplicated this exact
    four-line resolve-then-mutate sequence.
    """
    if not parent_id:
        return None, None
    resolved_parent_id = _resolve_file_id(parent_id, "parent_id")
    parent_resource_key = _extract_resource_key(parent_id)
    file_metadata["parents"] = [resolved_parent_id]
    return resolved_parent_id, parent_resource_key


def _require_drive_id(value: str, field_name: str) -> str:
    """Validate a raw (non-URL) Drive id such as a permission_id: non-empty,
    no surrounding whitespace (require_clean_identifier), and restricted to
    the same character class real Drive ids use (_DRIVE_ID_CHARS).

    Closes the same dot-segment risk _resolve_file_id's bare-id path
    guards against, for an id that (unlike file_id) never goes through URL
    resolution at all: require_clean_identifier alone accepts ".."
    (non-empty, no surrounding whitespace), and googleapiclient does not
    percent-encode "." before interpolating an id into a URL path segment
    -- an unvalidated permission_id="..\" reaches the API as a literal
    dot-segment, which normalization can collapse a permission-removal
    request into the file's own delete/get endpoint instead.
    """
    require_clean_identifier(value, field_name)
    if not _DRIVE_ID_CHARS.fullmatch(value):
        raise ValueError(f"{field_name} must look like a Drive id")
    return value


def _require_share_role(role: str) -> str:
    if role not in _SHARE_ROLES:
        allowed = ", ".join(_SHARE_ROLES)
        raise ValueError(f"role must be one of {allowed}")
    return role


def _require_entity_type(entity_type: str) -> str:
    if entity_type not in _SHARE_ENTITY_TYPES:
        allowed = ", ".join(_SHARE_ENTITY_TYPES)
        raise ValueError(f"entity_type must be one of {allowed}")
    return entity_type


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
    field_name: str,
    items: list[Any],
    *,
    truncated: bool = False,
    next_page_token: str | None = None,
) -> str:
    def _build(items: list[Any], truncated: bool) -> str:
        response: dict[str, Any] = {
            "status": "success",
            field_name: items,
            "truncated": truncated,
        }
        if next_page_token:
            # Independent of the halving loop's own `truncated`/`items`
            # cutting -- a caller-resumable page cursor from the upstream
            # API, not something local truncation ever produces or
            # invalidates, so it's attached once here rather than threaded
            # through render/halve.
            response["next_page_token"] = next_page_token
        return json.dumps(response, ensure_ascii=False)

    return _capped_response(_build, items, truncated)


def _capped_permissions_response(
    pages: list[tuple[str | None, list[Any]]],
    next_page_token: str | None,
    max_output_length: int | None = None,
) -> str:
    """Build google_drive_list_permissions' response, shrinking (if
    needed to fit the output cap) by dropping whole pages from the tail
    before ever cutting an individual page's items.

    A flat item-level cut (_capped_list_response's default halving)
    can't be paired safely with a resumable page token: Drive's
    ``pageToken`` only resumes at a page boundary, so if size-capping
    dropped items from *inside* an already-fetched page, a
    next_page_token that only ever points past whole pages would skip
    those dropped items forever, even though they were already fetched
    from the API this call. Dropping only whole pages avoids that: the
    first dropped page's own start token is always available as the
    resume point, so a caller who follows next_page_token is guaranteed
    to see every permission again (at worst once more, never zero
    times).

    If even the single earliest remaining page doesn't fit under the cap
    on its own, this falls back to _capped_list_response's plain
    item-level cut of just that page's contents -- and deliberately never
    passes a next_page_token through to it in that case. This isn't an
    oversight: at that point every candidate token is unsafe to offer --
    ``pages[1][0]`` (the next already-fetched page's start) would skip
    whatever this fallback had to cut from the *current* page, and this
    page's own start token would just refetch the same too-big page
    again. A cut inside one page has no page-granular resume point that
    doesn't silently skip something, so omitting the token entirely is
    the honest answer, not a simplification worth "fixing" by wiring one
    of those back in.

    ``pages`` is the ordered list of ``(start_token, items)`` fetched
    this call; ``next_page_token`` is the token for whatever page comes
    after the LAST page in ``pages`` (``None`` if that was Drive's last
    page). ``max_output_length`` lets a caller that already read it (e.g.
    to drive its own fetch-loop early-exit) pass the same value through
    instead of this function reading it again independently.
    """
    if max_output_length is None:
        max_output_length = get_tool_max_output_length()

    def _token_after(kept: int) -> str | None:
        return next_page_token if kept == len(pages) else pages[kept][0]

    def _render(items: list[Any], truncated: bool, token: str | None) -> str:
        response: dict[str, Any] = {
            "status": "success",
            "permissions": items,
            "truncated": truncated,
        }
        if token:
            response["next_page_token"] = token
        return json.dumps(response, ensure_ascii=False)

    truncated = next_page_token is not None
    kept = len(pages)
    permissions = [item for _, items in pages for item in items]
    response = _render(permissions, truncated, _token_after(kept))

    while len(response) > max_output_length and kept > 1:
        kept -= 1
        truncated = True
        permissions = [item for _, items in pages[:kept] for item in items]
        response = _render(permissions, truncated, _token_after(kept))

    if len(response) > max_output_length:
        # Only the earliest remaining page is left and it alone doesn't
        # fit -- fall through to the same plain item-level halving every
        # other capped list response uses (no page-granular token to
        # offer here regardless, see the docstring above).
        response = _capped_list_response("permissions", permissions, truncated=True)

    return response


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


# mime types safe to decode as UTF-8 text and return inline, or to accept as
# the declared type of a google_drive_create_file() call (whose content is
# always literal UTF-8 text — see that function). Anything else (PDFs,
# Office/OOXML formats, images, etc.) is binary and must go through
# google_drive_download_file / google_drive_upload_file instead —
# decoding/encoding it as UTF-8 would silently corrupt it into unusable
# garbage. RTF is included because it's specified as 7-bit-ASCII-clean
# (escape sequences carry any non-ASCII content), so it round-trips through
# UTF-8 safely, unlike the binary formats this allowlist exists to keep out.
# YAML/JS and the "+json"/"+xml" structured-syntax suffixes (RFC 6839, e.g.
# "application/ld+json", "application/atom+xml") are included for the same
# reason — genuinely text under the hood, just not "text/"-prefixed by
# convention. The script/source-format entries (x-sh, x-sql, x-tex, ...,
# graphql) are each an unambiguous registered mime type for a genuinely-text
# format — no unrelated binary format is ever declared with them — so
# accepting them here is always safe, however the caller arrived at
# declaring one. Thrift/Avro/protobuf are deliberately NOT included here
# even though their .thrift/.avsc/.proto *extensions* are in
# _KNOWN_TEXT_EXTENSIONS below (an IDL/schema file is always text) —
# unlike graphql, a generic Thrift/Avro/protobuf mime type (e.g.
# "application/x-protobuf") can legitimately describe an actual
# binary-encoded message payload in real-world use (Prometheus
# remote-write, gRPC-Web, ...), the same ambiguity ".bat"/".ts"/".scm" have
# at the extension level (see _KNOWN_TEXT_EXTENSIONS's own note on those).
_TEXT_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/rtf",
    "application/javascript",
    "application/ecmascript",
    "application/x-yaml",
    "application/yaml",
    "application/csv",
    "application/x-sh",
    "application/x-csh",
    "application/x-tcl",
    "application/x-sql",
    "application/x-tex",
    "application/x-latex",
    "application/xml-dtd",
    "application/vnd.dart",
    "application/graphql",
}
_TEXT_MIME_TYPE_SUFFIXES = ("+xml", "+json", "+yaml")


def _normalize_mime_type(mime_type: str) -> str:
    """Lowercase and strip any ``;charset=...``-style parameter — callers
    (especially an LLM) commonly include one (e.g. "application/json;
    charset=utf-8", "TEXT/PLAIN") without meaning anything other than the
    plain type. Shared by every mime_type comparison in this module so
    they all treat the same input the same way.

    Also strips any embedded CR/LF. Not currently exploitable — the
    normalized value only ever reaches Drive as a JSON body field or a
    caller-facing error message, never a raw HTTP header the way
    _attach_resource_key's resourcekey value does — but scrubbing it here
    symmetrically is cheap insurance against a future call site adding
    one.
    """
    normalized = mime_type.strip().split(";", 1)[0].strip().lower()
    return normalized.replace("\r", "").replace("\n", "")


def _is_text_mime_type(mime_type: str) -> bool:
    """Whether ``mime_type`` denotes content safe to treat as UTF-8 text."""
    normalized = _normalize_mime_type(mime_type)
    return (
        normalized.startswith("text/")
        or normalized in _TEXT_MIME_TYPES
        or normalized.endswith(_TEXT_MIME_TYPE_SUFFIXES)
    )


def _is_google_workspace_mime_type(mime_type: str) -> bool:
    """Whether ``mime_type`` denotes a native Google Workspace document
    (Docs/Sheets/Slides/...), whose mimeType is always exactly
    "application/vnd.google-apps.<kind>".

    A prefix check on that fixed namespace, not a bare ``"google-apps" in
    mime_type`` substring test — the latter would also match an unrelated
    value that merely contains that text somewhere (e.g. a crafted
    "application/pdf; x=google-apps").
    """
    return _normalize_mime_type(mime_type).startswith("application/vnd.google-apps.")


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.() -]")

# Comfortably under the ~255-byte NAME_MAX most filesystems enforce, leaving
# room for the " (N)" suffix _unique_output_path may append on top.
_MAX_FILENAME_LENGTH = 200
# Generous for any real extension, including compound ones like ".tar.gz" —
# a "suffix" longer than this isn't behaving like an extension anymore (see
# _safe_output_filename), so it gets truncated too rather than left to blow
# the overall length cap on its own.
_MAX_SUFFIX_LENGTH = 20


def _output_dir() -> Path:
    """Root directory google_drive_download_file writes into: the current
    task's workspace output/ subdirectory, mirroring TaskWorkspace.output_dir
    so downloaded files show up alongside other generated deliverables."""
    base = os.environ.get("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", "").strip()
    if not base:
        raise RuntimeError(
            "No task workspace configured for this connector "
            "(XAGENT_GOOGLE_DRIVE_OUTPUT_DIR is unset) — "
            "google_drive_download_file needs a task workspace to write "
            "into."
        )
    output_dir = Path(base).expanduser().resolve() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _split_stem_suffix(base: str) -> tuple[str, str]:
    """Like Path(base).stem/.suffix, except a name that's *entirely* a
    leading dot plus extension (e.g. ".pdf") is treated as having that
    extension. pathlib's own split refuses to do this — it follows the
    Unix dotfile convention where a single leading dot with nothing before
    it never counts as an extension separator, leaving Path(".pdf").suffix
    empty — but Drive occasionally hands back names in exactly this shape,
    and silently losing the extension breaks the downloaded file's type.
    Only that narrow shape is special-cased; e.g. "..pdf" or "..." already
    split the way we want via plain pathlib and are left alone.
    """
    suffix = Path(base).suffix
    if not suffix and base.startswith(".") and base.count(".") == 1 and len(base) > 1:
        return "", base
    return Path(base).stem, suffix


def _safe_output_filename(name: str) -> str:
    """Collapse a Drive file name into a single safe path segment so it
    can't escape the output directory (e.g. via ".." or embedded "/") — the
    name comes from user/Drive data, not a trusted constant.

    Stem and suffix are sanitized separately: sanitizing the whole string
    in one pass would let the trailing ".strip('._')" eat into the
    extension's own leading dot whenever the stem sanitizes down to nothing
    (e.g. an all-non-ASCII or all-punctuation name) — "季度报告.pdf" must
    still come out as "....pdf" (something ending in .pdf), not "pdf".
    """
    base = Path(name).name
    stem, suffix = _split_stem_suffix(base)
    stem = _UNSAFE_FILENAME_CHARS.sub("_", stem).strip("._") or "file"
    suffix = _UNSAFE_FILENAME_CHARS.sub("_", suffix)[:_MAX_SUFFIX_LENGTH]
    max_stem_length = max(1, _MAX_FILENAME_LENGTH - len(suffix))
    return stem[:max_stem_length] + suffix


def _download_media(request: Any) -> bytes:
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    return fh.getvalue()


def _unique_output_path(output_dir: Path, filename: str) -> Path:
    """Avoid silently overwriting a same-named file already in the output
    dir (e.g. downloading the same deck twice) by appending " (1)", " (2)",
    etc. — mirrors common download-manager behavior rather than either
    clobbering data or forcing the caller to pick a unique name upfront."""
    candidate = output_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    counter = 1
    while (candidate := output_dir / f"{stem} ({counter}){suffix}").exists():
        counter += 1
    return candidate


_UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS"


def _upload_allowed_dirs() -> list[Path]:
    """Parse XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS's comma-separated
    directory allowlist, falling back to the current working directory
    when it's unset.

    That CWD fallback is a fail-open default: with no task workspace and
    no configured external dirs, an MCP subprocess's own working
    directory becomes the allowlist. Not reachable from the app's own
    call sites today — _build_oauth_mcp_stdio_transport_config in
    config.py always sets this env var whenever a task_id is present —
    but a standalone or misconfigured launch of this server would fall
    back to it silently.

    Previously lived in utils.py as a name shared with gmail.py/slack.py's
    own equivalents in spirit, but google_drive.py's upload tool is its
    only real caller — gmail.py, slack.py, and linkedin.py each still
    carry their own separate, near-identical copy rather than having been
    migrated to a shared helper (a fast-follow, not part of this fix).
    """
    raw_dirs = os.environ.get(_UPLOAD_ALLOWED_DIRS_ENV_VAR, "")
    parsed_dirs = [
        Path(stripped).expanduser().resolve()
        for raw_dir in raw_dirs.split(",")
        if (stripped := raw_dir.strip())
    ]
    # Falls back to CWD not just when the env var is entirely unset/blank,
    # but also when it normalizes to no real entries (e.g. "," or " , ") --
    # otherwise this would return [] and _resolve_upload_file_path's `any(
    # local_path.is_relative_to(d) for d in allowed_dirs)` is vacuously
    # False for every path, turning the documented fail-open CWD default
    # into an unintended fail-closed "everything rejected" for a malformed
    # value, rather than the same fallback a fully-empty var gets.
    return parsed_dirs or [Path.cwd().resolve()]


def _resolve_upload_file_path(file_path: str) -> Path:
    """Resolve ``file_path`` and ensure it falls under one of the upload
    allowlist's directories — without this, google_drive_upload_file
    could be tricked into reading arbitrary host files.

    Pass an absolute path — a relative path resolves against this
    process's own working directory, not the allowed directory, and will
    not find a file written to the task workspace.

    Containment is checked *before* existence, so a path that is both
    outside the allowlist and nonexistent reports the allowlist message,
    not "not found" — the latter would leak whether that host path exists
    at all to a caller who has no business finding out.
    """
    local_path = Path(file_path).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    local_path = local_path.resolve()

    allowed_dirs = _upload_allowed_dirs()
    if not any(local_path.is_relative_to(d) for d in allowed_dirs):
        # The absolute host path is deliberately kept out of the raised
        # message: it reaches the caller/LLM unfiltered via the error
        # payload otherwise, and host filesystem layout has no business in
        # a model transcript. Full detail (including the allowed
        # directories) is logged server-side.
        logger.warning(
            "Rejected file path %s outside allowed directories: %s",
            local_path,
            ", ".join(str(path) for path in allowed_dirs),
        )
        raise PermissionError(
            "file path is outside the allowed directories; ask the user "
            "for a file inside the task workspace or another allowed "
            "location"
        )

    if not local_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return local_path


# Extensions of formats confidently known to be plain UTF-8 text. Default
# a NAME to "looks binary" unless its extension is in this allowlist (or
# it has no extension at all — an extensionless name like "Dockerfile" or
# "README" isn't itself evidence of binary intent) — the opposite of a
# binary-extension blocklist, which can never enumerate every binary
# format that exists (three separate review rounds each found more gaps
# in this module's blocklist attempts: the original hand list, then a
# mimetypes.guess_type()-based one, then a wider hand list again). A
# false positive here (a legitimate but unlisted text extension) is a
# clear, actionable rejection pointing at google_drive_upload_file; a
# false negative in a blocklist is a silently mislabeled file, a strictly
# worse outcome, so this list defaults toward rejecting the unfamiliar
# rather than accepting it.
#
# Also NOT delegated to mimetypes.guess_type(): that function's result
# for anything beyond a small hardcoded core depends on whatever
# mime.types database happens to be installed on the *host* — verified
# directly, several of this list's own entries (.sql, .tex, .dart, .tcl,
# .dtd) came back non-text from an incidental /etc/apache2/mime.types on
# one development machine, and were silently ABSENT (making them "not
# text" by the old design's logic) in a minimal CI/production container.
# A fixed, in-source list is the only way to get identical behavior
# everywhere this code runs.
#
# ".bat", ".ts", and ".scm" are included as text (batch script,
# TypeScript, Scheme source) even though each also names a real,
# unrelated binary format elsewhere (a Windows .exe-like executable, an
# MPEG-2 transport stream, a Lotus ScreenCam recording) -- a judgment
# call in favor of what an agent generating files is overwhelmingly more
# likely to mean, resolved per-extension since no mime-type rule can
# distinguish the two (see _TEXT_MIME_TYPES's matching note on why
# Thrift/Avro's mime types, unlike protobuf's, are excluded there for
# the same reason). ".plist" gets the same treatment: real Apple property
# lists can be either XML text or a binary-encoded format, but an
# agent-authored one is overwhelmingly more likely to be the XML form.
_KNOWN_TEXT_EXTENSIONS = {
    # plain text / docs / dotfiles
    ".txt", ".md", ".markdown", ".mdx", ".rst", ".adoc", ".rtf", ".log",
    ".lock", ".gitignore", ".gitattributes", ".editorconfig",
    ".dockerignore", ".env", ".ini", ".cfg", ".conf", ".properties",
    ".toml",
    # structured/data formats
    ".json", ".json5", ".xml", ".yaml", ".yml", ".csv", ".tsv", ".dtd",
    ".xsd", ".xsl", ".xslt", ".proto", ".graphql", ".gql", ".thrift",
    ".avsc", ".ipynb", ".jsonl", ".ndjson", ".geojson", ".plist",
    # web
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".svg",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".vue", ".svelte", ".astro",
    # source code
    ".py", ".rb", ".php", ".java", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".cxx", ".cs", ".m", ".mm", ".go", ".rs", ".swift", ".kt", ".kts",
    ".scala", ".groovy", ".lua", ".r", ".jl", ".pl", ".pm", ".hs", ".fs",
    ".fsx", ".ml", ".mli", ".clj", ".cljs", ".erl", ".ex", ".exs", ".nim",
    ".zig", ".v", ".d", ".dart", ".elm", ".cr", ".tcl", ".scm", ".rkt",
    ".lisp", ".el", ".asm", ".s", ".pas", ".f90", ".for", ".vb", ".vbs",
    ".cabal", ".nix",
    # shell / scripting / templates
    ".sh", ".bash", ".zsh", ".csh", ".ksh", ".fish",
    ".ps1", ".bat", ".cmd", ".awk", ".sed", ".sql", ".j2",
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
    "Dockerfile", "README") is accepted too — the absence of an
    extension isn't evidence of binary intent the way an unrecognized
    one is.

    Uses _split_stem_suffix rather than plain ``Path.suffix`` for the
    same reason _safe_output_filename does: a name that's *entirely* a
    leading dot plus extension (e.g. ".pdf") has an empty ``Path(...).suffix``
    per pathlib's dotfile convention, which would make this function treat
    it as extensionless (accepted) — exactly the kind of mislabeling this
    guard exists to catch, just via a name pathlib refuses to split.

    google_drive_create_file's mime_type-based check is the complementary
    signal that catches an explicitly-declared binary type regardless of
    what the name looks like.
    """
    suffix = _split_stem_suffix(name.strip())[1].lower()
    return bool(suffix) and suffix not in _KNOWN_TEXT_EXTENSIONS


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
    The result's "truncated" field is true if the output was too large and
    some matching files were cut to fit -- treat the "files" list as
    possibly incomplete in that case rather than the full result set.
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
    Download or export a text file's content from Google Drive by file_id,
    returned inline as a string. If it's a Google Workspace document (Docs,
    Sheets), it will be exported to the requested mime_type.

    mime_type must be a text format (e.g. "text/plain", "text/csv",
    "application/json") — this tool decodes the result as UTF-8 text, which
    would corrupt a binary format like a PDF or image into garbage. For a
    PDF, an Office format, an image, or any other binary content, use
    google_drive_download_file instead, which writes the real bytes to a
    file instead of decoding them as text.
    The result's "encoding" field is "utf-8" for ordinary text, or "base64"
    for the rare case where content typed as text still wasn't valid UTF-8
    (e.g. a legacy-encoded file) -- check this before treating "content" as
    literal text, since a base64 string decoded and displayed as-is is
    unreadable. "truncated" is true if the output was too large and
    "content" was cut to fit -- treat it as possibly incomplete in that
    case rather than the full file.
    """
    try:
        # Normalized once here so every downstream use (the text-safety
        # check, the "== text/plain" default-detection below, and the
        # export_media mimeType itself) agrees on the same canonical
        # value — an un-normalized "TEXT/PLAIN" or "text/plain; charset=..."
        # would otherwise pass the check above but then fail the exact
        # string comparison used to detect the default, silently skipping
        # the Workspace-export mimeType fallback and sending Drive's API
        # a non-canonical mimeType instead.
        mime_type = _normalize_mime_type(mime_type)
        if not _is_text_mime_type(mime_type):
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"mime_type '{mime_type}' is not a text format -- this "
                        "tool would corrupt binary content by decoding it as "
                        "UTF-8. Use google_drive_download_file instead to "
                        "get the real bytes as a file."
                    ),
                },
                ensure_ascii=False,
            )

        resolved_file_id = _resolve_file_id(file_id)
        resource_key = _extract_resource_key(file_id)
        service = get_drive_service()
        get_request = service.files().get(
            fileId=resolved_file_id,
            supportsAllDrives=True,
            fields="id, name, mimeType",
        )
        _attach_resource_key(get_request, resolved_file_id, resource_key)
        file_metadata = get_request.execute()
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

        if _is_google_workspace_mime_type(file_mime_type):
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
            # mime_type is always honored as-is (and is already validated
            # as a text format by the _is_text_mime_type check above).
            export_mime_type = mime_type
            if mime_type == "text/plain":
                export_mime_type = _TEXT_PLAIN_EXPORT_FALLBACK.get(
                    file_mime_type, mime_type
                )
            request = service.files().export_media(
                fileId=resolved_file_id, mimeType=export_mime_type
            )
        else:
            # Regular file: get_media ignores mime_type entirely and
            # returns the file's own real bytes, so it's file_mime_type --
            # not the caller-supplied mime_type (already validated above)
            # -- that determines whether decoding as UTF-8 is safe here.
            # file_mime_type itself comes straight from Drive's own stored
            # metadata, which can itself be a host-dependent guess made at
            # upload time (e.g. mimetypes.guess_type() in
            # google_drive_upload_file's own mime_type default, or another
            # client's own equivalent) rather than something this tool
            # controls -- a ".docx" uploaded without an explicit mime_type
            # on a host that fails to guess it lands in Drive as
            # "application/octet-stream" and would be rejected here as
            # "not a text file" even though the real content might have
            # been text. Not something this function can correct after the
            # fact; Drive's stored mimeType is the only source of truth it
            # has for a regular (non-Workspace) file.
            if not _is_text_mime_type(file_mime_type):
                return json.dumps(
                    {
                        "status": "error",
                        "message": (
                            f"'{file_metadata.get('name', resolved_file_id)}' is "
                            f"not a text file (mimeType '{file_mime_type}') -- "
                            "this tool would corrupt it by decoding as UTF-8. "
                            "Use google_drive_download_file instead."
                        ),
                    },
                    ensure_ascii=False,
                )
            request = service.files().get_media(
                fileId=resolved_file_id, supportsAllDrives=True
            )
        _attach_resource_key(request, resolved_file_id, resource_key)

        raw_bytes = _download_media(request)
        try:
            # A strict decode, not errors="replace": both branches above
            # now only ever reach here for content already validated as a
            # text format (export_mime_type is always text; the get_media
            # branch just rejected anything file_mime_type doesn't mark as
            # text), so a UTF-8 failure would mean that text-typed content
            # isn't actually valid UTF-8 (e.g. a legacy non-UTF-8 encoded
            # text file) -- a real, if rare, case worth preserving exactly
            # rather than corrupting via errors="replace". base64
            # preserves it exactly, same pattern as onedrive.py's
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
def google_drive_download_file(
    file_id: str, mime_type: str = "", filename: str = ""
) -> str:
    """
    Download or export a Google Drive file to a real file in the task
    workspace, returning its path — use this for any binary content (PDF,
    image, Office format, etc.) that google_drive_get_file_content would
    otherwise corrupt by decoding as text. The returned path can be passed
    directly to another tool that reads local files, e.g. gmail_send_messages's
    'attachments'.

    mime_type: required when file_id is a Google Workspace document (Docs,
    Sheets, Slides) — the format to export to (e.g. "application/pdf").
    Ignored for a file that already has real binary content of its own
    (mime_type is not required and has no effect there).
    filename: optional name for the written file; defaults to the Drive
    file's own name. Either way, if exporting a Workspace document the
    mime_type's extension is appended when not already present (so a bare
    filename="report" for an "application/pdf" export still ends up
    "report.pdf").
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        resource_key = _extract_resource_key(file_id)
        service = get_drive_service()
        get_request = service.files().get(
            fileId=resolved_file_id,
            supportsAllDrives=True,
            fields="id, name, mimeType",
        )
        _attach_resource_key(get_request, resolved_file_id, resource_key)
        file_metadata = get_request.execute()
        file_mime_type = file_metadata.get("mimeType", "")
        drive_name = file_metadata.get("name") or resolved_file_id

        if _is_google_workspace_mime_type(file_mime_type):
            if not mime_type:
                return json.dumps(
                    {
                        "status": "error",
                        "message": (
                            f"'{drive_name}' is a Google Workspace document "
                            "(mimeType "
                            f"'{file_mime_type}'); specify mime_type to "
                            'export it to (e.g. "application/pdf").'
                        ),
                    },
                    ensure_ascii=False,
                )
            request = service.files().export_media(
                fileId=resolved_file_id, mimeType=mime_type
            )
            extension = mimetypes.guess_extension(mime_type) or ""
        else:
            request = service.files().get_media(
                fileId=resolved_file_id, supportsAllDrives=True
            )
            extension = ""
        _attach_resource_key(request, resolved_file_id, resource_key)

        data = _download_media(request)

        # Ensure the export's extension is present regardless of whether
        # the name came from Drive's own file name or an explicit
        # `filename` argument — the caller passing a bare "report" for a
        # PDF export shouldn't lose the ".pdf" any more than the default
        # name would. Case-insensitive so "Report.PDF" doesn't become
        # "Report.PDF.pdf".
        chosen_name = _safe_output_filename(filename or drive_name)
        if extension and not chosen_name.lower().endswith(extension.lower()):
            chosen_name += extension

        output_path = _unique_output_path(_output_dir(), chosen_name)
        output_path.write_bytes(data)

        return json.dumps(
            {
                "status": "success",
                "file": file_metadata,
                "path": str(output_path),
                "size": len(data),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_upload_file(
    file_path: str, name: str = "", mime_type: str = "", parent_id: str | None = None
) -> str:
    """
    Upload a local file's real bytes to Google Drive — use this (not
    google_drive_create_file) for a PDF, image, Office document, or any
    other binary content, including a file this agent already generated
    into the task workspace (e.g. an exported PDF report).

    file_path: path to a file already on disk, e.g. something written to
    the task workspace. Must be inside an allowed directory (automatically
    scoped to the current task workspace), which is meant to keep this
    tool from reading arbitrary host files — not an absolute guarantee
    (see this function's own symlink-timing comment on the `with
    local_path.open("rb")` line below). Pass an absolute path — a relative
    path resolves against this process's own working directory, not the
    allowed directory, and will not find a file written to the task
    workspace.
    name: the file name to give it in Drive; defaults to file_path's own
    basename.
    mime_type: defaults to a guess from the file's extension via the
    standard mimetypes module, falling back to "application/octet-stream"
    when it can't be guessed. This guess is informational only — Drive's
    displayed file type may be wrong on a host whose mimetypes database
    doesn't recognize the extension (see the _KNOWN_TEXT_EXTENSIONS
    comment above for why that's host-dependent), but the uploaded bytes
    are always exactly file_path's real content either way. Pass mime_type
    explicitly for a reliable label, especially for an extension outside
    the common formats mimetypes ships with by default.
    """
    try:
        local_path = _resolve_upload_file_path(file_path)

        resolved_name = name.strip() or local_path.name
        # Normalized the same way google_drive_create_file's mime_type is,
        # so an explicit "Application/PDF" or "application/pdf; foo=bar"
        # doesn't land on Drive uncanonicalized just because this tool
        # (unlike create_file) has no text-safety gate to normalize for.
        resolved_mime_type = (
            _normalize_mime_type(mime_type)
            or mimetypes.guess_type(local_path.name)[0]
            or "application/octet-stream"
        )

        file_metadata: dict[str, Any] = {
            "name": resolved_name,
            "mimeType": resolved_mime_type,
        }
        resolved_parent_id, parent_resource_key = _apply_parent_id(
            file_metadata, parent_id
        )

        service = get_drive_service()

        # local_path.open() here is a fresh by-path open, not the same
        # descriptor _resolve_upload_file_path used for its allowlist
        # check — a symlink swapped in during that window wouldn't be
        # caught. Accepted risk: closing it would mean holding a file
        # descriptor open across the whole allowlist resolution, which
        # isn't worth it for a local, non-shared task workspace. Size is
        # checked from this same handle the upload reads from (rather
        # than a separate stat() call before opening) so at least those
        # two agree with each other. No maximum size cap: MediaIoBaseUpload
        # streams the file in chunks rather than buffering it whole, and
        # Drive's own per-account storage quota is the real limit.
        try:
            fh_ctx = local_path.open("rb")
        except OSError as e:
            # str(OSError) embeds the absolute path (e.g. "[Errno 13]
            # Permission denied: '/full/host/path'") -- the same detail
            # _resolve_upload_file_path's own error deliberately scrubs.
            # A permission error or a TOCTOU race (the allowlist-checked
            # path got swapped/removed between the check and this open)
            # would otherwise leak it straight into the caller/LLM-facing
            # message through the generic `except Exception` below.
            logger.warning("Failed to open upload file %s: %s", local_path, e)
            raise ValueError("Could not read the file at the given path") from e
        with fh_ctx as fh:
            file_size = os.fstat(fh.fileno()).st_size
            if file_size == 0:
                # Drive's API itself accepts 0-byte files; rejecting one
                # here is a deliberate product choice (an agent uploading
                # an empty file is almost always a mistake upstream, e.g.
                # a generation step that silently produced nothing) rather
                # than an API constraint, unlike Gmail/Slack's matching
                # rejects.
                raise ValueError(f"File is empty: {file_path}")

            media = MediaIoBaseUpload(fh, mimetype=resolved_mime_type, resumable=True)

            create_request = service.files().create(
                body=file_metadata,
                media_body=media,
                supportsAllDrives=True,
                fields="id, name, webViewLink, mimeType",
            )
            if resolved_parent_id is not None:
                _attach_resource_key(
                    create_request, resolved_parent_id, parent_resource_key
                )
            file = create_request.execute()

        return json.dumps({"status": "success", "file": file}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_drive_create_file(
    name: str, content: str, mime_type: str | None = None, parent_id: str | None = None
) -> str:
    """
    Create a new Drive file from plain text or HTML content.
    If you want to create a Google Doc, use mime_type="application/vnd.google-apps.document"
    and pass plain text or HTML in the content. For normal text files, leave
    mime_type unset (it defaults to "text/plain").

    content is always treated as text (it is UTF-8 encoded before upload) —
    this tool cannot create a real PDF, image, or Office-format binary; a
    name like "report.pdf" would get a file with that name but text/plain
    content, not an actual PDF. To upload an already-generated file's real
    bytes (a PDF exported earlier, an image, a .docx, etc.), use
    google_drive_upload_file with the file's path instead.

    mime_type left unset also gates `name` against a fixed guess-the-format
    check: a name whose extension isn't recognized as a text format (this
    can be a false positive — extensions this tool doesn't know about, or
    a name like "www.example.com"/"report-v1.2" where the text after the
    last "." isn't really an extension at all) gets rejected. If that
    happens for a name you know is fine, pass mime_type explicitly (even
    "text/plain") — an explicit mime_type is trusted over the name-based
    guess, and is only rejected if it itself isn't a text-safe type.
    """
    try:
        # Stripped once here so every downstream use agrees: the
        # binary-name guard below and the actual Drive file_metadata["name"]
        # sent. _name_looks_binary already stripped internally for its own
        # suffix check, but without also reassigning `name` itself, a
        # trailing-space name like "notes.txt " would pass the guard (the
        # suffix check sees ".txt") yet still land in Drive with the
        # literal trailing space intact — inconsistent with
        # google_drive_upload_file's matching `name.strip()`.
        name = name.strip()
        # None/omitted vs. an explicit value are distinguished on purpose
        # (rather than defaulting the parameter itself to "text/plain"):
        # an explicit mime_type — even one that happens to also be
        # "text/plain" — is the caller making a deliberate assertion about
        # the content's type, which is trusted over the name-based guess
        # below; an omitted one leaves the name as the only signal, same
        # as before. Empty/whitespace-only is treated as omitted too, so a
        # caller that explicitly passes "" gets the same default rather
        # than a confusing rejection.
        # Normalized (case/whitespace/;params stripped) once here so every
        # downstream use agrees: the guard checks below, the Drive
        # file_metadata["mimeType"] actually sent, and the upload's own
        # media mimetype. Without this, a caller passing e.g. "TEXT/PLAIN"
        # or "application/json; charset=utf-8" would pass the guard (which
        # already normalized internally) but then have that same
        # non-canonical string land in Drive's metadata instead of the
        # canonical form the guard just validated.
        #
        # The `mime_type is not None` check is spelled out again here
        # (rather than reusing a separately-assigned bool) so mypy can
        # narrow `mime_type` to `str` for the _normalize_mime_type call --
        # a bool computed from that same condition doesn't carry the
        # narrowing through to this branch.
        if mime_type is not None and mime_type.strip() != "":
            explicit_mime_type = True
            mime_type = _normalize_mime_type(mime_type)
        else:
            explicit_mime_type = False
            mime_type = "text/plain"
        is_google_doc_conversion = _is_google_workspace_mime_type(mime_type)
        # A Google Workspace conversion is exempt from both checks below
        # (Docs/Sheets/Slides always take text/HTML source regardless of
        # the target doc's name), so short-circuit here rather than
        # computing _name_looks_binary and the mime-type check on every
        # such call for nothing.
        if not is_google_doc_conversion:
            mime_type_is_binary = not _is_text_mime_type(mime_type)
            # Two independent signals catch two different mistakes — but
            # only the name-based one is a heuristic guess, and only when
            # the caller left mime_type unset. _KNOWN_TEXT_EXTENSIONS is a
            # default-deny allowlist, so a name that merely CONTAINS a dot
            # not meant as an extension ("www.example.com",
            # "report-v1.2") or uses a real text extension this list
            # doesn't happen to know about would otherwise be rejected
            # with no way to correct it. An explicit mime_type — even
            # "text/plain" — lets the caller override that false positive
            # directly; a caller that left mime_type unset, named the file
            # like a binary format, is still caught (the original
            # production bug this guard exists for). An explicit *binary*
            # mime_type is caught either way, regardless of the name,
            # since content can only ever be UTF-8 text (see the encode()
            # below).
            name_looks_binary = not explicit_mime_type and _name_looks_binary(name)
        else:
            name_looks_binary = mime_type_is_binary = False
        if name_looks_binary or mime_type_is_binary:
            if name_looks_binary:
                message = (
                    f"'{name}' looks like a binary file, but "
                    "google_drive_create_file only writes text content — "
                    "uploading it here would produce a mislabeled text "
                    "file, not real binary data. If this is actually text "
                    "and the name just isn't recognized (e.g. the text "
                    'after the last "." isn\'t really an extension, like '
                    "in a version number or domain name), pass an "
                    'explicit mime_type (e.g. "text/plain") to confirm '
                    "that. If you already generated a real binary file "
                    "(e.g. as a task output), use google_drive_upload_file "
                    "with its file path to upload the real binary content "
                    "instead."
                )
            else:
                message = (
                    f"mime_type '{mime_type}' is not a text format, but "
                    "google_drive_create_file's content is always UTF-8 "
                    "text — either use a text-safe mime_type (e.g. "
                    '"text/plain"), or if this is genuinely binary '
                    "content, write it to a file first and use "
                    "google_drive_upload_file with that file's path "
                    "instead."
                )
            return json.dumps(
                {"status": "error", "message": message}, ensure_ascii=False
            )

        file_metadata: dict[str, Any] = {"name": name, "mimeType": mime_type}
        resolved_parent_id, parent_resource_key = _apply_parent_id(
            file_metadata, parent_id
        )

        service = get_drive_service()
        fh = io.BytesIO(content.encode("utf-8"))

        # When creating a Google Doc, the upload mime type needs to be the original content's mime type (like text/plain)
        upload_mime_type = "text/plain" if is_google_doc_conversion else mime_type
        media = MediaIoBaseUpload(fh, mimetype=upload_mime_type, resumable=True)

        create_request = service.files().create(
            body=file_metadata,
            media_body=media,
            supportsAllDrives=True,
            fields="id, name, webViewLink, mimeType",
        )
        if resolved_parent_id is not None:
            _attach_resource_key(
                create_request, resolved_parent_id, parent_resource_key
            )
        file = create_request.execute()

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
        resolved_parent_id, parent_resource_key = _apply_parent_id(
            file_metadata, parent_id
        )

        service = get_drive_service()
        create_request = service.files().create(
            body=file_metadata,
            supportsAllDrives=True,
            fields="id, name, webViewLink, mimeType",
        )
        if resolved_parent_id is not None:
            _attach_resource_key(
                create_request, resolved_parent_id, parent_resource_key
            )
        folder = create_request.execute()

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
        resource_key = _extract_resource_key(file_id)
        service = get_drive_service()
        file_metadata = {"name": new_name}

        update_request = service.files().update(
            fileId=resolved_file_id,
            body=file_metadata,
            supportsAllDrives=True,
            fields="id, name, webViewLink, mimeType",
        )
        _attach_resource_key(update_request, resolved_file_id, resource_key)
        updated_file = update_request.execute()

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
    Delete a file or folder in Google Drive. This is an external action --
    confirm with the user before calling it.
    Note: this skips the trash and permanently deletes the file -- there is
    no separate "move to trash" tool on this connector, so make sure
    permanent deletion (not just removing it from view) is what they want.
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        resource_key = _extract_resource_key(file_id)
        service = get_drive_service()

        def _delete() -> Any:
            request = service.files().delete(
                fileId=resolved_file_id, supportsAllDrives=True
            )
            return _attach_resource_key(
                request, resolved_file_id, resource_key
            ).execute()

        def _verify_deleted() -> Any:
            request = service.files().get(
                fileId=resolved_file_id, supportsAllDrives=True
            )
            return _attach_resource_key(
                request, resolved_file_id, resource_key
            ).execute()

        _execute_ignoring_204_ssl_eof(_delete, _verify_deleted)

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
def google_drive_list_permissions(file_id: str, page_token: str | None = None) -> str:
    """
    List who currently has access to a Drive file or folder (owner,
    editors, commenters, viewers) and their permission ids. Use the
    returned permission ids with google_drive_update_permission or
    google_drive_remove_permission.
    For an item in a Shared Drive with more collaborators than fit in one
    response, "truncated" is true and the result usually includes a
    "next_page_token" -- pass it back as this call's page_token to
    continue listing from where it left off instead of silently missing
    the rest. In the rare case where even the earliest unreturned page
    alone is too large to fit, "truncated" is true but "next_page_token"
    is absent -- there's no safe way to resume from partway through a
    single page, so the remaining collaborators on it can't be listed
    through this tool.
    A permission whose "permissionDetails" marks it "inherited": true comes
    from a parent folder, not this item directly -- it can't be changed or
    removed here; do that on the folder it's inherited from instead.
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        resource_key = _extract_resource_key(file_id)
        service = get_drive_service()
        max_output_length = get_tool_max_output_length()
        # Each fetched page is kept alongside the token that fetched it
        # (see _capped_permissions_response): if the response must shrink
        # to fit the output cap, only whole pages are dropped from the
        # tail, so the surfaced next_page_token always resumes at a page
        # boundary Drive will refetch from scratch -- never at a point
        # *inside* a page whose later items a flat item-level cut
        # (_capped_list_response's default halving) would have silently
        # discarded after page_token had already advanced past them.
        pages: list[tuple[str | None, list[Any]]] = []
        approx_length = 0
        current_page_token = page_token
        # For an item in a Shared Drive, Drive returns at most 100
        # permissions per page when pageSize isn't set (a non-Shared-Drive
        # item always returns everything in one page); follow
        # nextPageToken so a heavily-shared Shared Drive folder isn't
        # silently under-reported. Bounded so a misbehaving API response
        # can't turn this into an infinite loop.
        for _ in range(_MAX_PERMISSION_LIST_PAGES):
            list_request = service.permissions().list(
                fileId=resolved_file_id,
                supportsAllDrives=True,
                pageToken=current_page_token,
                fields=(
                    "nextPageToken, "
                    "permissions(id, type, role, emailAddress, displayName, "
                    "permissionDetails(permissionType, role, inherited, "
                    "inheritedFrom))"
                ),
            )
            _attach_resource_key(list_request, resolved_file_id, resource_key)
            results = list_request.execute()
            new_permissions = results.get("permissions", [])
            pages.append((current_page_token, new_permissions))
            # _capped_permissions_response below will drop whole pages
            # back off to fit the output limit regardless, so once
            # there's already enough data to exceed it, fetching further
            # pages is pure waste: more blocking network round-trips for
            # permissions that get thrown away immediately after. Tracked
            # incrementally (each new page's items serialized once, not
            # the whole accumulated list every iteration) to stay O(n)
            # rather than O(n^2) for a long permission list;
            # ensure_ascii=False so this length estimate isn't inflated
            # relative to what _capped_permissions_response will actually
            # measure -- with the default ensure_ascii=True, a non-ASCII
            # displayName/emailAddress (CJK, accented, emoji names are
            # common on a Shared Drive) would each be escaped to a
            # 6+-byte \uXXXX sequence here even though the real response
            # keeps them as 1-2 bytes, overestimating this list as needing
            # to stop paginating even though the true payload would still
            # fit -- and _capped_permissions_response would then measure
            # the true, smaller size and leave truncated=False, silently
            # under-reporting collaborators while claiming a complete
            # result.
            approx_length += sum(
                len(json.dumps(p, ensure_ascii=False)) for p in new_permissions
            )
            next_token = results.get("nextPageToken")
            current_page_token = next_token
            if not next_token:
                break
            if approx_length > max_output_length:
                break
        # current_page_token is None only when the loop broke because
        # Drive reported no further pages. If the loop instead exhausted
        # _MAX_PERMISSION_LIST_PAGES without ever seeing a falsy
        # nextPageToken, current_page_token still holds that last (truthy)
        # token -- so, same as the early size-cap break, it correctly
        # signals "more data exists beyond what was fetched" without
        # needing a separate branch for the loop-safety-bound case.

        return _capped_permissions_response(
            pages, current_page_token, max_output_length
        )
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
    entity_type: str = "user",
) -> str:
    """
    Grant a user or Google Group (by email) access to a Drive file or
    folder, making it visible to someone outside this conversation. This
    is an external action -- confirm the target file, the email, the
    role, and whether it's a user or group with the user before calling
    it.
    role: "reader" (can view), "commenter" (can view and comment), or
    "writer" (can edit). Sharing a folder gives that access to everything
    inside it. entity_type: "user" (default) for an individual's email
    address, or "group" for a Google Group's email address -- a group's
    role semantics are otherwise identical to a user's. When
    send_notification is True, Google emails the person or group being
    added; message is included in that email if given. If
    send_notification is False, message is silently discarded (Google's
    API rejects a notification message when no notification is sent) --
    the response won't flag this, so don't rely on the message being
    delivered without also checking send_notification.
    This grants access to a specific user or group only; it can't create
    "anyone with the link" link-sharing or share with an entire domain.
    google_drive_remove_permission can still revoke an existing
    link-shared permission if one is already present.
    """
    try:
        _require_share_role(role)
        _require_entity_type(entity_type)
        require_clean_identifier(email, "email")
        if not _EMAIL_PATTERN.match(email):
            raise ValueError("email must be a valid email address")
        resolved_file_id = _resolve_file_id(file_id)
        resource_key = _extract_resource_key(file_id)

        service = get_drive_service()
        # The API rejects emailMessage outright when sendNotificationEmail is
        # false, so it's only included when a notification is actually going out.
        create_kwargs: dict[str, Any] = {
            "fileId": resolved_file_id,
            "body": {"type": entity_type, "role": role, "emailAddress": email},
            "sendNotificationEmail": send_notification,
            "supportsAllDrives": True,
            "fields": "id, type, role, emailAddress, displayName",
        }
        if send_notification and message is not None:
            create_kwargs["emailMessage"] = message

        create_request = service.permissions().create(**create_kwargs)
        _attach_resource_key(create_request, resolved_file_id, resource_key)
        permission = create_request.execute()

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
        resource_key = _extract_resource_key(file_id)
        resolved_permission_id = _require_drive_id(permission_id, "permission_id")

        service = get_drive_service()
        update_request = service.permissions().update(
            fileId=resolved_file_id,
            permissionId=resolved_permission_id,
            body={"role": role},
            supportsAllDrives=True,
            fields="id, type, role, emailAddress, displayName",
        )
        _attach_resource_key(update_request, resolved_file_id, resource_key)
        permission = update_request.execute()

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
        resource_key = _extract_resource_key(file_id)
        resolved_permission_id = _require_drive_id(permission_id, "permission_id")
        service = get_drive_service()

        def _delete() -> Any:
            request = service.permissions().delete(
                fileId=resolved_file_id,
                permissionId=resolved_permission_id,
                supportsAllDrives=True,
            )
            return _attach_resource_key(
                request, resolved_file_id, resource_key
            ).execute()

        def _verify_deleted() -> Any:
            request = service.permissions().get(
                fileId=resolved_file_id,
                permissionId=resolved_permission_id,
                supportsAllDrives=True,
            )
            return _attach_resource_key(
                request, resolved_file_id, resource_key
            ).execute()

        _execute_ignoring_204_ssl_eof(_delete, _verify_deleted)

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
