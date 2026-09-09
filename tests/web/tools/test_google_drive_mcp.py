import base64
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import google_drive


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    # tests/conftest.py force-loads the project .env into the test process, so
    # any refresh credentials configured there would leak into the "absent"
    # assertions below.
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)


def _mock_drive_service(monkeypatch):
    service = Mock()
    monkeypatch.setattr(google_drive, "get_drive_service", lambda: service)
    return service


@pytest.fixture(autouse=True)
def _output_dir_env(tmp_path, monkeypatch):
    """Every test gets its own isolated output root so nothing here ever
    writes into the real working directory."""
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _upload_allowed_dirs_env(tmp_path, monkeypatch):
    """Scope google_drive_upload_file's read allowlist to an isolated
    per-test directory, mirroring _output_dir_env above for the write
    side — otherwise the default (cwd) allowlist would make tests
    order-dependent on whatever the real working directory holds."""
    allowed_dir = tmp_path / "workspace"
    allowed_dir.mkdir()
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS", str(allowed_dir))
    return allowed_dir


def _mock_drive_service_with_files(monkeypatch, files_mock):
    service = Mock()
    service.files.return_value = files_mock
    monkeypatch.setattr(google_drive, "get_drive_service", lambda: service)
    return service


class _FakeDownloader:
    """Mirrors googleapiclient.http.MediaIoBaseDownload's interface closely
    enough for this file's download loop: write `content` into the target
    buffer across one call to next_chunk()."""

    def __init__(self, fh, request, content: bytes = b"") -> None:
        self._fh = fh
        self._content = content

    def next_chunk(self):
        self._fh.write(self._content)
        return None, True


def _patch_downloader(monkeypatch, content: bytes):
    monkeypatch.setattr(
        google_drive,
        "MediaIoBaseDownload",
        lambda fh, request: _FakeDownloader(fh, request, content),
    )


def test_get_drive_service_requires_access_token(monkeypatch):
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="GOOGLE_ACCESS_TOKEN"):
        google_drive.get_drive_service()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("abc123", "abc123"),
        ("https://drive.google.com/file/d/abc123/view?usp=sharing", "abc123"),
        ("https://drive.google.com/drive/folders/abc123", "abc123"),
        ("https://docs.google.com/document/d/abc123/edit", "abc123"),
        ("https://docs.google.com/spreadsheets/d/abc123/edit#gid=0", "abc123"),
        ("https://docs.google.com/presentation/d/abc123/edit", "abc123"),
        # Google inserts this account-index segment into a copied share link
        # whenever more than one Google account is signed into the browser.
        ("https://drive.google.com/file/u/1/d/abc123/view", "abc123"),
        ("https://drive.google.com/drive/u/1/folders/abc123", "abc123"),
        ("https://docs.google.com/document/u/2/d/abc123/edit", "abc123"),
        ("https://docs.google.com/spreadsheets/u/0/d/abc123/edit#gid=0", "abc123"),
        ("https://docs.google.com/presentation/u/1/d/abc123/edit", "abc123"),
        # Legacy share-link format with no "/d/<id>" path segment at all,
        # still emitted by DriveApp.getUrl() in Apps Script.
        ("https://drive.google.com/open?id=abc123", "abc123"),
        ("https://docs.google.com/open?id=abc123&authuser=0", "abc123"),
    ],
)
def test_resolve_file_id_accepts_bare_id_and_url_forms(value, expected):
    assert google_drive._resolve_file_id(value) == expected


@pytest.mark.parametrize(
    "url",
    [
        # "id" is an extremely common, generic query-param name -- without
        # host-gating, any unrelated site's link would get its id extracted
        # and passed to a share/delete/permission call as if it were this
        # connector's own share link.
        "https://example.com/article?id=12345",
        "https://evil.com/x?id=ATTACKER_CHOSEN_ID",
        # Same risk for the path-shaped patterns: a deliberately-crafted,
        # non-Google URL can be made to match the same path shape.
        "https://evil.com/file/d/ATTACKER_ID/view",
        "https://evil.com/drive/folders/ATTACKER_ID",
        # Scheme-less and protocol-relative variants of the same attack --
        # urlparse only populates .scheme/.hostname for an explicit
        # "scheme://" or leading "//" prefix, so these are at least as
        # natural a shape for an attacker to plant in text an agent reads
        # as the full https:// form, and must be rejected identically.
        "evil.com/file/d/ATTACKER_ID/view",
        "//evil.com/file/d/ATTACKER_ID/view",
        "evil.com/x?id=ATTACKER_CHOSEN_ID",
        "evil.com/drive/folders/ATTACKER_ID",
    ],
)
def test_resolve_file_id_does_not_extract_id_from_untrusted_hosts(url):
    """A prompt-injected agent could be fed a URL like these (e.g. quoted
    inside a document's content) as `file_id`; the resolver must not
    silently redirect a share/delete/permission call onto whatever id that
    URL happens to name -- it should fail as an obviously-invalid fileId
    instead, by returning the URL unresolved."""
    assert google_drive._resolve_file_id(url) == url


@pytest.mark.parametrize(
    "url",
    [
        # A userinfo-bearing authority is inherently ambiguous across URL
        # parsers: Python's urlparse (RFC 3986) resolves .hostname to
        # whatever follows the *last* "@", so each of these has .hostname
        # == "drive.google.com" and would pass a naive host check -- but a
        # WHATWG-based parser (a browser rendering the same string to a
        # human, or a different URL consumer downstream) can disagree about
        # which side is the real destination. The literal backslash variant
        # is the same ambiguity with an extra separator character thrown
        # in; both must be rejected identically, not just the plain form.
        r"https://evil.com@drive.google.com/x?id=ATTACKER_ID",
        r"https://evil.com\@drive.google.com/x?id=ATTACKER_ID",
    ],
)
def test_resolve_file_id_rejects_userinfo_bearing_authority(url):
    assert google_drive._resolve_file_id(url) == url


@pytest.mark.parametrize(
    "url",
    [
        # Untrusted/ambiguous URLs _resolve_file_id refuses to resolve an
        # id for. _resolve_file_id and _extract_resource_key share
        # _parse_trusted_drive_url specifically so their trust decisions
        # for the same input can't drift apart the way they once did (see
        # _parse_trusted_drive_url's docstring) -- this asserts that
        # agreement directly, from one shared list, rather than only
        # exercising each function's own hand-picked cases.
        "https://evil.com/x?id=ATTACKER_CHOSEN_ID",
        "https://evil.com@drive.google.com/x?id=ATTACKER_ID",
        r"https://evil.com\@drive.google.com/x?id=ATTACKER_ID",
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTabc123xyz/pubhtml",
        # A trusted-host URL whose path isn't a recognized share-link
        # shape at all (not even an excluded one) -- both functions must
        # agree this doesn't resolve to a real id.
        "https://drive.google.com/drive/my-drive",
        # A malformed path segment the end-anchor now rejects instead of
        # truncating -- both functions must agree on that too.
        "https://drive.google.com/file/d/ABC.DEF/view",
    ],
)
def test_resolve_file_id_and_extract_resource_key_agree_on_untrusted_input(url):
    separator = "&" if "?" in url else "?"
    resource_key_url = f"{url}{separator}resourcekey=0-Rkey123"
    assert google_drive._resolve_file_id(resource_key_url) == resource_key_url
    assert google_drive._extract_resource_key(resource_key_url) is None


@pytest.mark.parametrize(
    "url",
    [
        "drive.google.com/file/d/abc123/view",
        "drive.google.com/drive/folders/abc123",
        "docs.google.com/open?id=abc123",
    ],
)
def test_resolve_file_id_resolves_scheme_less_trusted_host_urls(url):
    """The scheme-less-URL check above must reject untrusted hosts without
    also rejecting a legitimate Drive/Docs URL a user pasted without its
    "https://" prefix."""
    assert google_drive._resolve_file_id(url) == "abc123"


@pytest.mark.parametrize(
    "url",
    [
        # "Publish to web" links have no real Drive file id in them at all
        # -- "e" is a literal path segment, not part of an id, and the
        # actual publish token isn't usable as a fileId.
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vTabc123xyz/pubhtml",
        "https://docs.google.com/document/d/e/2PACX-1vTabc123xyz/pub",
        "https://docs.google.com/presentation/d/e/2PACX-1vTabc123xyz/pub",
    ],
)
def test_resolve_file_id_does_not_mistake_published_link_e_segment_for_id(url):
    assert google_drive._resolve_file_id(url) == url


def test_resolve_file_id_still_matches_a_real_id_starting_with_e():
    """The "/d/e/" exclusion for published links must be specific to that
    literal segment, not reject any id that happens to start with "e"."""
    assert (
        google_drive._resolve_file_id("https://drive.google.com/file/d/e5f6g7/view")
        == "e5f6g7"
    )


@pytest.mark.parametrize(
    "url",
    [
        # A malformed path segment must be rejected outright, not silently
        # truncated to the leading run of valid characters -- without an
        # end-anchor, ".../file/d/ABC.DEF/view" would capture just "ABC"
        # (the id charset stops at the "."), acting on a wrong, truncated
        # id instead of returning the whole URL unresolved.
        "https://drive.google.com/file/d/ABC.DEF/view",
        "https://drive.google.com/drive/folders/ABC.DEF",
    ],
)
def test_resolve_file_id_rejects_malformed_path_segment_instead_of_truncating(url):
    assert google_drive._resolve_file_id(url) == url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # A scheme-less trusted-host link whose query value itself embeds
        # a full URL. An earlier "does '//' occur anywhere in the string"
        # heuristic failed to prepend "//" here (since "//" already
        # appears, just not at the front), so urlparse saw no netloc and
        # the whole string was rejected as untrusted -- even though this
        # is a perfectly legitimate Drive link.
        (
            "docs.google.com/document/d/ABCID123/edit"
            "?usp=sharing&continue=https://other.com/x",
            "ABCID123",
        ),
        # A harmless double-slash typo in the path must not defeat
        # resolution either, for the same reason.
        ("drive.google.com/file/d/ABCID123//view", "ABCID123"),
    ],
)
def test_resolve_file_id_handles_scheme_less_urls_containing_another_scheme(
    url, expected
):
    assert google_drive._resolve_file_id(url) == expected


def test_resolve_file_id_does_not_extract_id_from_unrelated_query_value():
    """The host check validates the URL's overall authority, but the id
    must come from the URL's own path/query structure -- not a free-text
    search of the whole string. Otherwise a URL whose host is a real,
    trusted drive.google.com but whose query VALUE merely contains a
    matching shape (here, a hypothetical search results page) would have
    that unrelated substring extracted as if it were the URL's own
    resource id."""
    url = "https://drive.google.com/search?q=/file/d/OTHERID/view"
    assert google_drive._resolve_file_id(url) == url


def test_resolve_file_id_rejects_non_string_input():
    with pytest.raises(ValueError, match="file_id must be a string"):
        google_drive._resolve_file_id(12345)


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_resolve_file_id_rejects_empty_or_whitespace_input(value):
    """An unrejected empty file_id would reach the Drive API as a request
    against the bare .../files/ collection endpoint instead of a specific
    file -- reject it the same way email/permission_id already are via
    require_clean_identifier."""
    with pytest.raises(ValueError, match="file_id must be a non-empty id"):
        google_drive._resolve_file_id(value)


def test_resolve_file_id_uses_the_given_field_name_in_errors():
    with pytest.raises(ValueError, match="parent_id must be a non-empty id"):
        google_drive._resolve_file_id("", "parent_id")
    with pytest.raises(ValueError, match="parent_id must be a string"):
        google_drive._resolve_file_id(123, "parent_id")


@pytest.mark.parametrize(
    "url",
    [
        # googleapiclient does not percent-encode "." before sending it, so
        # an unvalidated id reaches the API as a literal path segment;
        # dot-segment normalization could then collapse "files/.." into an
        # unrelated endpoint. Character-class-validating the query "id="
        # value (matching what the path branch already implicitly requires)
        # closes this off.
        "https://drive.google.com/x?id=..",
        "https://drive.google.com/x?id=.",
        "https://drive.google.com/x?id=../../etc",
    ],
)
def test_resolve_file_id_rejects_dot_segments_from_query_fallback(url):
    assert google_drive._resolve_file_id(url) == url


@pytest.mark.parametrize("value", ["..", ".", "...", "~", "%2e%2e"])
def test_resolve_file_id_rejects_dot_segments_from_bare_input(value):
    """The bare-id fast path (no "/" or "?" at all, so it never reaches the
    URL-handling code, or the query-fallback validation above) is the most
    common call shape -- every caller normally passes a plain id, not a
    URL -- and was left completely unvalidated. A literal file_id=".."
    reaches google_drive_delete_file (which "skips the trash and
    permanently deletes") as fileId=".." verbatim, which dot-segment
    normalization can collapse into an unrelated endpoint -- the exact risk
    the query-fallback case above is already guarded against, just
    reachable more directly here."""
    with pytest.raises(ValueError, match="file_id must look like a Drive id"):
        google_drive._resolve_file_id(value)


def test_resolve_file_id_query_fallback_respects_published_link_exclusion():
    """The path pattern's "(?!e/)" exclusion for published-to-web links
    must not be bypassable by simply appending "?id=<anything>": the query
    fallback used to run unconditionally whenever the path match failed,
    without knowing *why* it failed."""
    url = "https://docs.google.com/document/d/e/2PACX-1abcXYZ/pub?id=SNEAKY"
    assert google_drive._resolve_file_id(url) == url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://drive.google.com/forms/d/abc123/edit", "abc123"),
        ("https://drive.google.com/forms/u/1/d/abc123/edit", "abc123"),
        ("https://drive.google.com/drawings/d/abc123/edit", "abc123"),
        ("https://drive.google.com/drawings/u/2/d/abc123/edit", "abc123"),
    ],
)
def test_resolve_file_id_supports_forms_and_drawings(url, expected):
    assert google_drive._resolve_file_id(url) == expected


def test_resolve_file_id_supports_usercontent_host():
    url = "https://drive.usercontent.google.com/download?id=abc123&export=download"
    assert google_drive._resolve_file_id(url) == "abc123"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("abc123", None),  # bare id: never URL-shaped, nothing to extract
        ("https://drive.google.com/file/d/abc123/view", None),  # no resourcekey
        (
            "https://drive.google.com/file/d/abc123/view?resourcekey=0-Rkey123",
            "0-Rkey123",
        ),
        (
            "https://docs.google.com/document/d/abc123/edit?resourcekey=0-Rkey123",
            "0-Rkey123",
        ),
        # Untrusted host: must not leak a resourcekey-shaped query value
        # from a URL this connector doesn't otherwise trust at all.
        ("https://evil.com/x?id=abc123&resourcekey=0-Rkey123", None),
        # Userinfo-bearing authority: same rejection as _resolve_file_id.
        (
            "https://evil.com@drive.google.com/file/d/abc123/view?resourcekey=0-Rkey123",
            None,
        ),
        # A resourcekey isn't restricted to the fileId character class --
        # kb.py's own CloudFile.resourceKey field validates only
        # "^[^\r\n]*$" (see test_kb_ingest_cloud_accepts_punctuation_in_
        # drive_identifiers in test_kb_dir.py), so punctuation like a
        # colon must be preserved rather than silently discarded.
        (
            "https://drive.google.com/file/d/abc123/view?resourcekey=resource:key_1-2",
            "resource:key_1-2",
        ),
        # Published-to-web links ("/d/e/") have no real Drive fileId in
        # them at all -- _resolve_file_id refuses to resolve an id for
        # this exact shape, so a resourcekey must not be extracted for it
        # either; doing so would attach a resourcekey header to a request
        # whose fileId is the unresolved raw URL.
        (
            "https://docs.google.com/spreadsheets/d/e/2PACX-1vTabc123xyz/pubhtml"
            "?resourcekey=0-Rkey123",
            None,
        ),
        # A trusted-host URL whose path isn't a recognized share-link
        # shape at all (not even the excluded "/d/e/" one) must not have a
        # resourcekey extracted either -- _resolve_file_id can't derive a
        # real fileId from it, so a resourcekey extracted here would only
        # ever be attached to a request whose fileId is the unresolved raw
        # URL. Broader than the "/d/e/" case above: this is any
        # unrecognized path, not just a deliberately-excluded known one.
        ("https://drive.google.com/drive/my-drive?resourcekey=0-Rkey123", None),
        # The legacy "?id=" query-fallback share-link shape must still
        # extract a resourcekey normally -- the widened exclusion guard
        # must not accidentally reject this legitimate, already-supported
        # shape along with the unrecognized-path case above.
        (
            "https://drive.google.com/open?id=abc123&resourcekey=0-Rkey123",
            "0-Rkey123",
        ),
    ],
)
def test_extract_resource_key(value, expected):
    assert google_drive._extract_resource_key(value) == expected


@pytest.mark.parametrize(
    "malicious_file_id",
    [
        "evil\r\nX-Injected-Header: 1",
        "evil\rX-Injected-Header: 1",
        "evil\nX-Injected-Header: 1",
    ],
)
def test_attach_resource_key_refuses_a_file_id_containing_crlf(malicious_file_id):
    """Belt-and-suspenders defense: _extract_resource_key's guard should
    make this unreachable in practice (a resourcekey is only ever
    extracted alongside a _DRIVE_ID_CHARS-validated file_id), but if a
    future call site ever breaks that invariant, this must raise rather
    than silently skip the header -- a silent skip would send a request
    that legitimately needed the header without it, surfacing later as a
    confusing 404 instead of an immediate, actionable local error."""
    request = Mock()
    request.headers = {}

    with pytest.raises(ValueError, match="CR/LF"):
        google_drive._attach_resource_key(request, malicious_file_id, "0-Rkey123")

    assert request.headers == {}


def test_attach_resource_key_attaches_normally_for_a_clean_file_id():
    request = Mock()
    request.headers = {}

    result = google_drive._attach_resource_key(request, "abc123", "0-Rkey123")

    assert result.headers == {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}


@pytest.mark.parametrize("bad_role", ["owner", "organizer", ""])
def test_require_share_role_rejects_roles_outside_the_allowed_set(bad_role):
    with pytest.raises(ValueError, match="role"):
        google_drive._require_share_role(bad_role)


def test_require_share_role_error_message_is_not_a_tuple_repr():
    with pytest.raises(ValueError) as exc_info:
        google_drive._require_share_role("owner")
    message = str(exc_info.value)
    assert "(" not in message and "'" not in message
    assert "reader, commenter, writer" in message


@pytest.mark.parametrize("good_role", ["reader", "commenter", "writer"])
def test_require_share_role_accepts_allowed_roles(good_role):
    assert google_drive._require_share_role(good_role) == good_role


@pytest.mark.parametrize("bad_entity_type", ["domain", "anyone", ""])
def test_require_entity_type_rejects_types_outside_the_allowed_set(bad_entity_type):
    with pytest.raises(ValueError, match="entity_type"):
        google_drive._require_entity_type(bad_entity_type)


@pytest.mark.parametrize("good_entity_type", ["user", "group"])
def test_require_entity_type_accepts_allowed_types(good_entity_type):
    assert google_drive._require_entity_type(good_entity_type) == good_entity_type


_FOLDER_URL_WITH_ID = "https://drive.google.com/drive/folders/abc123"


def _stub_happy_path(service):
    """Wire every mocked call this module can make to a benign response, so
    a test driving one specific tool doesn't also need to know how the
    others behave."""
    service.files.return_value.delete.return_value.execute.return_value = {}
    service.files.return_value.get.return_value.execute.return_value = {}
    service.permissions.return_value.list.return_value.execute.return_value = {
        "permissions": []
    }
    service.permissions.return_value.create.return_value.execute.return_value = {
        "id": "perm1"
    }
    service.permissions.return_value.update.return_value.execute.return_value = {
        "id": "perm1"
    }
    service.permissions.return_value.delete.return_value.execute.return_value = {}
    service.permissions.return_value.get.return_value.execute.return_value = {}


@pytest.mark.parametrize(
    ("mock_target", "invoke"),
    [
        (
            lambda s: s.files.return_value.delete,
            lambda fid: google_drive.google_drive_delete_file(fid),
        ),
        (
            lambda s: s.permissions.return_value.list,
            lambda fid: google_drive.google_drive_list_permissions(fid),
        ),
        (
            lambda s: s.permissions.return_value.create,
            lambda fid: google_drive.google_drive_share_file(fid, "a@example.com"),
        ),
        (
            lambda s: s.permissions.return_value.update,
            lambda fid: google_drive.google_drive_update_permission(
                fid, "perm1", "reader"
            ),
        ),
        (
            lambda s: s.permissions.return_value.delete,
            lambda fid: google_drive.google_drive_remove_permission(fid, "perm1"),
        ),
    ],
    ids=[
        "delete_file",
        "list_permissions",
        "share_file",
        "update_permission",
        "remove_permission",
    ],
)
def test_id_taking_tools_resolve_full_drive_urls(monkeypatch, mock_target, invoke):
    """Every id-taking tool touched by this change calls _resolve_file_id
    before hitting the API, but a test that only ever passes an already-bare
    id would not notice if that call were missing from a given tool. Drive
    this through the real call sites with a full folder URL and assert the
    *resolved* id is what reached the mocked API call."""
    service = _mock_drive_service(monkeypatch)
    _stub_happy_path(service)

    invoke(_FOLDER_URL_WITH_ID)

    kwargs = mock_target(service).call_args.kwargs
    assert kwargs["fileId"] == "abc123"
    assert kwargs["supportsAllDrives"] is True


def test_search_passes_shared_drive_support_and_clamps_max_results(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "f1", "name": "notes.txt"}]
    }

    result = json.loads(google_drive.google_drive_search("notes", max_results=999999))

    assert result["status"] == "success"
    assert result["files"][0]["id"] == "f1"
    assert result["truncated"] is False
    kwargs = service.files.return_value.list.call_args.kwargs
    assert kwargs["supportsAllDrives"] is True
    assert kwargs["includeItemsFromAllDrives"] is True
    assert kwargs["pageSize"] == 1000  # clamped, not the raw 999999


def test_search_caps_oversized_output(monkeypatch):
    # Pinned rather than relying on the ~50KB default: XAGENT_TOOL_MAX_OUTPUT_LENGTH
    # is honored and tests/conftest.py force-loads .env, so an environment
    # override could otherwise make this flaky in either direction.
    monkeypatch.setattr(google_drive, "get_tool_max_output_length", lambda: 3000)
    service = _mock_drive_service(monkeypatch)
    huge_files = [
        {"id": f"f{i}", "name": "x" * 200, "mimeType": "text/plain"}
        for i in range(2000)
    ]
    service.files.return_value.list.return_value.execute.return_value = {
        "files": huge_files
    }

    result = json.loads(google_drive.google_drive_search("x"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert 0 < len(result["files"]) < len(huge_files)


def test_search_returns_error_payload_on_failure(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.list.return_value.execute.side_effect = RuntimeError(
        "quota exceeded"
    )

    result = json.loads(google_drive.google_drive_search("x"))

    assert result["status"] == "error"
    assert "quota exceeded" in result["message"]


def test_search_validates_max_results_before_building_service(monkeypatch):
    """Every other tool in this file validates/resolves its input before
    calling get_drive_service() (Credentials construction + a discovery-
    doc build); google_drive_search was the one outlier that built the
    service first and only clamped max_results afterward, paying that
    cost on every malformed call before ever rejecting it."""
    get_service = Mock()
    monkeypatch.setattr(google_drive, "get_drive_service", get_service)

    result = json.loads(
        google_drive.google_drive_search("x", max_results="not-a-number")
    )

    assert result["status"] == "error"
    get_service.assert_not_called()


def test_create_file_resolves_parent_id_url_and_supports_shared_drives(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.create.return_value.execute.return_value = {
        "id": "new1",
        "name": "notes.txt",
    }

    result = json.loads(
        google_drive.google_drive_create_file(
            "notes.txt", "hello", parent_id=_FOLDER_URL_WITH_ID
        )
    )

    assert result["status"] == "success"
    kwargs = service.files.return_value.create.call_args.kwargs
    assert kwargs["body"]["parents"] == ["abc123"]
    assert kwargs["supportsAllDrives"] is True


def test_create_folder_resolves_parent_id_url_and_supports_shared_drives(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.create.return_value.execute.return_value = {
        "id": "new1",
        "name": "Subfolder",
    }

    result = json.loads(
        google_drive.google_drive_create_folder(
            "Subfolder", parent_id=_FOLDER_URL_WITH_ID
        )
    )

    assert result["status"] == "success"
    kwargs = service.files.return_value.create.call_args.kwargs
    assert kwargs["body"]["parents"] == ["abc123"]
    assert kwargs["supportsAllDrives"] is True
    assert kwargs["fields"] == "id, name, webViewLink, mimeType"


def test_create_file_attaches_resource_key_header_for_parent(monkeypatch):
    """A pre-2021 link-shared parent folder needs its own resourcekey
    attached the same way every other file_id-taking tool does, or
    files.create 404s resolving a parent that genuinely exists."""
    service = _mock_drive_service(monkeypatch)
    create_request = service.files.return_value.create.return_value
    create_request.headers = {}
    create_request.execute.return_value = {"id": "new1", "name": "notes.txt"}

    url = "https://drive.google.com/drive/folders/abc123?resourcekey=0-Rkey123"
    google_drive.google_drive_create_file("notes.txt", "hello", parent_id=url)

    assert create_request.headers == {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}


def test_create_folder_attaches_resource_key_header_for_parent(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    create_request = service.files.return_value.create.return_value
    create_request.headers = {}
    create_request.execute.return_value = {"id": "new1", "name": "Subfolder"}

    url = "https://drive.google.com/drive/folders/abc123?resourcekey=0-Rkey123"
    google_drive.google_drive_create_folder("Subfolder", parent_id=url)

    assert create_request.headers == {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}


def test_create_file_omits_resource_key_header_without_parent_id(monkeypatch):
    """No parent_id at all must not call _attach_resource_key with a None
    resolved_parent_id (which would otherwise build a nonsensical
    "None/None" header value)."""
    service = _mock_drive_service(monkeypatch)
    create_request = service.files.return_value.create.return_value
    create_request.headers = {}
    create_request.execute.return_value = {"id": "new1", "name": "notes.txt"}

    google_drive.google_drive_create_file("notes.txt", "hello")

    assert create_request.headers == {}


def test_create_file_validates_parent_id_before_building_service(monkeypatch):
    get_service = Mock()
    monkeypatch.setattr(google_drive, "get_drive_service", get_service)

    result = json.loads(
        google_drive.google_drive_create_file("notes.txt", "hello", parent_id=123)
    )

    assert result["status"] == "error"
    assert "parent_id must be a string" in result["message"]
    get_service.assert_not_called()


def test_create_folder_validates_parent_id_before_building_service(monkeypatch):
    get_service = Mock()
    monkeypatch.setattr(google_drive, "get_drive_service", get_service)

    result = json.loads(
        google_drive.google_drive_create_folder("Subfolder", parent_id=123)
    )

    assert result["status"] == "error"
    assert "parent_id must be a string" in result["message"]
    get_service.assert_not_called()


def test_get_file_content_resolves_full_drive_url(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "abc123",
        "name": "notes.txt",
        "mimeType": "text/plain",
    }

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"hello")
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(google_drive.google_drive_get_file_content(_FOLDER_URL_WITH_ID))

    assert result["status"] == "success"
    get_kwargs = service.files.return_value.get.call_args.kwargs
    assert get_kwargs["fileId"] == "abc123"
    assert get_kwargs["supportsAllDrives"] is True
    media_kwargs = service.files.return_value.get_media.call_args.kwargs
    assert media_kwargs["fileId"] == "abc123"
    assert media_kwargs["supportsAllDrives"] is True
    assert result["content"] == "hello"
    assert result["encoding"] == "utf-8"


def test_get_file_content_attaches_resource_key_header_from_share_link(monkeypatch):
    """A pre-2021 link-shared item's plain fileId 404s on files.get/
    files.get_media/files.export_media without the resourcekey from its
    share link attached as an X-Goog-Drive-Resource-Keys header (see
    google_drive_download.py for the same header used elsewhere in this
    codebase). The header must be attached to both the metadata fetch and
    the content download/export request."""
    service = _mock_drive_service(monkeypatch)
    get_request = service.files.return_value.get.return_value
    get_request.headers = {}
    get_request.execute.return_value = {
        "id": "abc123",
        "name": "notes.txt",
        "mimeType": "text/plain",
    }
    media_request = service.files.return_value.get_media.return_value
    media_request.headers = {}

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"hello")
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    url = "https://drive.google.com/file/d/abc123/view?resourcekey=0-Rkey123"
    result = json.loads(google_drive.google_drive_get_file_content(url))

    assert result["status"] == "success"
    expected_header = {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}
    assert get_request.headers == expected_header
    assert media_request.headers == expected_header


def test_get_file_content_omits_resource_key_header_for_bare_id(monkeypatch):
    """A bare id (or a share link with no resourcekey) never carries a
    resourcekey -- the header must not be attached (or attached as
    "id/None") in that ordinary case."""
    service = _mock_drive_service(monkeypatch)
    get_request = service.files.return_value.get.return_value
    get_request.headers = {}
    get_request.execute.return_value = {
        "id": "abc123",
        "name": "notes.txt",
        "mimeType": "text/plain",
    }
    media_request = service.files.return_value.get_media.return_value
    media_request.headers = {}

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"hello")
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(google_drive.google_drive_get_file_content("abc123"))

    assert result["status"] == "success"
    assert get_request.headers == {}
    assert media_request.headers == {}


def test_get_file_content_falls_back_to_base64_for_non_utf8_text_file(monkeypatch):
    """A regular (non-Workspace) file whose real mimeType marks it as text
    can still fail a strict UTF-8 decode (e.g. a legacy non-UTF-8-encoded
    text file) even though genuine binary content (PDF, image, zip) is now
    rejected outright by the file_mime_type gate before ever reaching this
    decode step. Forcing errors="replace" here would silently corrupt it
    into replacement characters with status: success and no signal
    anything was lost; base64 preserves it exactly, matching onedrive.py's
    onedrive_get_file_content/_decode_bytes pattern for the same problem."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "text1",
        "name": "legacy.txt",
        "mimeType": "text/plain",
    }
    non_utf8_content = "café".encode("latin-1")  # not valid UTF-8

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(non_utf8_content)
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(google_drive.google_drive_get_file_content("text1"))

    assert result["status"] == "success"
    assert result["encoding"] == "base64"
    assert base64.b64decode(result["content"]) == non_utf8_content


def test_get_file_content_rejects_regular_binary_file(monkeypatch):
    """A regular (non-Workspace) file whose real mimeType is genuinely
    binary (PDF, image, zip, etc.) must be rejected before download rather
    than silently corrupted or (post-merge with the disk-write feature)
    routed around that gate -- confirms this connector's own resourcekey/
    supportsAllDrives hardening on the metadata GET didn't accidentally
    reopen the binary-content gap that feature exists to close."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "img1",
        "name": "photo.png",
        "mimeType": "image/png",
    }

    result = json.loads(google_drive.google_drive_get_file_content("img1"))

    assert result["status"] == "error"
    assert "google_drive_download_file" in result["message"]
    service.files.return_value.get_media.assert_not_called()


def test_get_file_content_defaults_spreadsheet_export_to_csv(monkeypatch):
    """Google Sheets has no text/plain export format and 400s on it, so the
    tool's own "text/plain" default -- fine for the far more common Docs
    case -- must not be sent as-is when the target turns out to be a Sheet."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "sheet1",
        "name": "Budget",
        "mimeType": "application/vnd.google-apps.spreadsheet",
    }

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"a,b\n1,2")
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(google_drive.google_drive_get_file_content("sheet1"))

    assert result["status"] == "success"
    assert (
        service.files.return_value.export_media.call_args.kwargs["mimeType"]
        == "text/csv"
    )


def test_get_file_content_respects_explicit_mime_type_for_spreadsheet(monkeypatch):
    """An explicit (non-default) mime_type must be honored as-is rather
    than the text/plain->text/csv smart-fallback mapping overriding it.
    Uses a text mime_type (not "text/plain" itself, which is what the
    fallback mapping keys off of) since an explicit binary mime_type is
    now rejected outright before this ever runs."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "sheet1",
        "name": "Budget",
        "mimeType": "application/vnd.google-apps.spreadsheet",
    }

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b'{"a": 1}')
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(
        google_drive.google_drive_get_file_content(
            "sheet1", mime_type="application/json"
        )
    )

    assert result["status"] == "success"
    assert (
        service.files.return_value.export_media.call_args.kwargs["mimeType"]
        == "application/json"
    )


def test_get_file_content_defaults_drawing_export_to_svg(monkeypatch):
    """Drawings has no text/plain export format either (only
    pdf/png/jpeg/svg), and unlike Sheets there's no other text-shaped
    format to fall back to -- svg is the one of those that's actually text
    (XML) and will decode sensibly, rather than raw image/PDF bytes coming
    out as garbled replacement characters."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "drawing1",
        "name": "Diagram",
        "mimeType": "application/vnd.google-apps.drawing",
    }

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"<svg></svg>")
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(google_drive.google_drive_get_file_content("drawing1"))

    assert result["status"] == "success"
    assert (
        service.files.return_value.export_media.call_args.kwargs["mimeType"]
        == "image/svg+xml"
    )


def test_get_file_content_respects_explicit_mime_type_for_drawing(monkeypatch):
    """Same as the spreadsheet case above: an explicit mime_type overrides
    the text/plain->image/svg+xml smart default for drawings. Uses a text
    mime_type since an explicit binary one (e.g. "image/png", the format
    this test originally used) is now rejected outright before this runs
    -- that binary-content path belongs to google_drive_download_file."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "drawing1",
        "name": "Diagram",
        "mimeType": "application/vnd.google-apps.drawing",
    }

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b'{"shape": "circle"}')
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(
        google_drive.google_drive_get_file_content(
            "drawing1", mime_type="application/json"
        )
    )

    assert result["status"] == "success"
    assert (
        service.files.return_value.export_media.call_args.kwargs["mimeType"]
        == "application/json"
    )


def test_get_file_content_defaults_apps_script_export_to_json(monkeypatch):
    """Apps Script projects have exactly one supported export format
    (application/vnd.google-apps.script+json) -- text/plain 400s on it the
    same way it does for Sheets/Drawings, but unlike those two there's no
    other plausible default a caller might have meant, so this is the only
    fallback that makes sense."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "script1",
        "name": "My Script",
        "mimeType": "application/vnd.google-apps.script",
    }

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b'{"files": []}')
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(google_drive.google_drive_get_file_content("script1"))

    assert result["status"] == "success"
    assert (
        service.files.return_value.export_media.call_args.kwargs["mimeType"]
        == "application/vnd.google-apps.script+json"
    )


def test_get_file_content_keeps_text_plain_default_for_docs(monkeypatch):
    """The text/csv fallback is specific to spreadsheets -- Docs supports
    text/plain export natively and must not be redirected to csv too."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "doc1",
        "name": "Notes",
        "mimeType": "application/vnd.google-apps.document",
    }

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(b"hello")
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(google_drive.google_drive_get_file_content("doc1"))

    assert result["status"] == "success"
    assert (
        service.files.return_value.export_media.call_args.kwargs["mimeType"]
        == "text/plain"
    )


@pytest.mark.parametrize(
    "folder_mime_type",
    [
        "application/vnd.google-apps.folder",
        "application/vnd.google-apps.shortcut",
        "application/vnd.google-apps.form",
        "application/vnd.google-apps.site",
    ],
)
def test_get_file_content_rejects_folder_and_shortcut_with_a_clear_message(
    monkeypatch, folder_mime_type
):
    """All of these mimeTypes start with "application/vnd.google-apps" and
    would otherwise fall into the export_media branch, which makes no
    sense for any of them (a folder has no content; a shortcut is a
    pointer; Forms and Sites have no files.export support at all) -- and
    folder/form share links are now resolvable via _resolve_file_id, so
    this is directly reachable, not hypothetical."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "abc123",
        "name": "Team Folder",
        "mimeType": folder_mime_type,
    }

    result = json.loads(google_drive.google_drive_get_file_content("abc123"))

    assert result["status"] == "error"
    assert "no content to read" in result["message"]
    service.files.return_value.export_media.assert_not_called()
    service.files.return_value.get_media.assert_not_called()


def test_get_file_content_caps_oversized_output(monkeypatch):
    # Pinned rather than relying on the ~50KB default -- see
    # test_search_caps_oversized_output's comment for why.
    monkeypatch.setattr(google_drive, "get_tool_max_output_length", lambda: 3000)
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "abc123",
        "name": "big.txt",
        "mimeType": "text/plain",
    }
    huge_content = ("x" * 200_000).encode("utf-8")

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(huge_content)
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(google_drive.google_drive_get_file_content("abc123"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert 0 < len(result["content"]) < len(huge_content)


def test_get_file_content_caps_oversized_non_utf8_content_as_valid_base64(monkeypatch):
    """Halving a base64 string at an arbitrary character offset (as the
    plain-text halving does) can produce a result that isn't just
    incomplete but genuinely invalid base64 -- unlike truncated text,
    which merely cuts off mid-character/word. The halving used for
    encoding="base64" must stay 4-character aligned so the truncated
    result still decodes. Uses a text-typed file whose actual bytes
    aren't valid UTF-8 (a legacy-encoded text file), since a genuinely
    binary mimeType is now rejected before this ever runs."""
    monkeypatch.setattr(google_drive, "get_tool_max_output_length", lambda: 3000)
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "text1",
        "name": "big.txt",
        "mimeType": "text/plain",
    }
    huge_binary = bytes(range(256)) * 100  # not valid UTF-8

    class _FakeDownloader:
        def __init__(self, fh, _request):
            self._fh = fh

        def next_chunk(self):
            self._fh.write(huge_binary)
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(google_drive.google_drive_get_file_content("img1"))

    assert result["status"] == "success"
    assert result["encoding"] == "base64"
    assert result["truncated"] is True
    # Must not raise -- confirms the truncated string is still valid,
    # correctly-padded base64, not an arbitrary character-offset cut.
    decoded = base64.b64decode(result["content"], validate=True)
    assert 0 < len(decoded) < len(huge_binary)


def test_rename_file_resolves_full_drive_url(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.update.return_value.execute.return_value = {
        "id": "abc123",
        "name": "renamed",
    }

    result = json.loads(
        google_drive.google_drive_rename_file(_FOLDER_URL_WITH_ID, "renamed")
    )

    assert result["status"] == "success"
    kwargs = service.files.return_value.update.call_args.kwargs
    assert kwargs["fileId"] == "abc123"
    assert kwargs["supportsAllDrives"] is True


def test_rename_file_attaches_resource_key_header(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    update_request = service.files.return_value.update.return_value
    update_request.headers = {}
    update_request.execute.return_value = {"id": "abc123", "name": "renamed"}

    url = "https://drive.google.com/file/d/abc123/view?resourcekey=0-Rkey123"
    google_drive.google_drive_rename_file(url, "renamed")

    assert update_request.headers == {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}


def test_list_permissions_returns_permissions(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.list.return_value.execute.return_value = {
        "permissions": [
            {"id": "perm1", "type": "user", "role": "reader", "emailAddress": "a@x.com"}
        ]
    }

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["status"] == "success"
    assert result["permissions"][0]["id"] == "perm1"
    assert result["truncated"] is False
    assert (
        service.permissions.return_value.list.call_args.kwargs["supportsAllDrives"]
        is True
    )


def test_list_permissions_requests_permission_details_field(monkeypatch):
    """Without permissionDetails, a caller can't tell an inherited
    permission (comes from a parent folder, can't be changed/removed on
    this item directly) from a direct one -- only observable for a Shared
    Drive item, but the field must always be requested to expose it when
    it's there."""
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.list.return_value.execute.return_value = {
        "permissions": []
    }

    google_drive.google_drive_list_permissions("fid")

    fields = service.permissions.return_value.list.call_args.kwargs["fields"]
    assert "permissionDetails" in fields
    assert "inherited" in fields


def test_list_permissions_accepts_a_caller_supplied_page_token(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.list.return_value.execute.return_value = {
        "permissions": [{"id": "perm1"}]
    }

    google_drive.google_drive_list_permissions("fid", page_token="resume-here")

    assert (
        service.permissions.return_value.list.call_args.kwargs["pageToken"]
        == "resume-here"
    )


def test_list_permissions_returns_next_page_token_when_stopped_early(monkeypatch):
    """When pagination stops early because accumulated size already
    exceeds the output cap, a nextPageToken may still exist -- it must be
    surfaced so a caller can resume the listing instead of the rest of
    the collaborators being silently lost.

    Uses two pages (a small one that fits, then a huge one that doesn't)
    rather than one huge page: the response must resume from the *first
    dropped whole page*'s own start token, not from whatever nextPageToken
    the last-fetched page happened to report -- otherwise a caller who
    followed that resumed token would skip the second page's contents
    entirely, since size-capping already dropped every one of its items
    from this response.
    """
    monkeypatch.setattr(google_drive, "get_tool_max_output_length", lambda: 3000)
    service = _mock_drive_service(monkeypatch)
    small_page = [{"id": "perm0", "type": "user", "role": "reader"}]
    huge_page = [
        {
            "id": f"perm{i}",
            "type": "user",
            "role": "reader",
            "emailAddress": f"user{i}@example.com",
            "displayName": "x" * 200,
        }
        for i in range(2000)
    ]
    service.permissions.return_value.list.return_value.execute.side_effect = [
        {"permissions": small_page, "nextPageToken": "page2"},
        {"permissions": huge_page, "nextPageToken": "page3"},
    ]

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["truncated"] is True
    # Resumes from page2 (the huge page's own start token) -- not page3
    # (which would skip everything on page2, since none of it survived
    # size-capping in this response).
    assert result["next_page_token"] == "page2"
    assert [p["id"] for p in result["permissions"]] == ["perm0"]


def test_list_permissions_omits_resume_token_when_a_single_page_overflows_alone(
    monkeypatch,
):
    """If even the single earliest page doesn't fit under the cap on its
    own, there is no page-granular resume point to offer honestly (Drive's
    pageToken can't resume from the middle of a page) -- next_page_token
    must be omitted rather than pointing at the next page and silently
    skipping whatever this response had to cut from the current one."""
    monkeypatch.setattr(google_drive, "get_tool_max_output_length", lambda: 3000)
    service = _mock_drive_service(monkeypatch)
    huge_page = [
        {
            "id": f"perm{i}",
            "type": "user",
            "role": "reader",
            "emailAddress": f"user{i}@example.com",
            "displayName": "x" * 200,
        }
        for i in range(2000)
    ]
    service.permissions.return_value.list.return_value.execute.return_value = {
        "permissions": huge_page,
        "nextPageToken": "page2",
    }

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["truncated"] is True
    assert "next_page_token" not in result
    assert 0 < len(result["permissions"]) < len(huge_page)


def test_list_permissions_omits_next_page_token_when_list_is_complete(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.list.return_value.execute.return_value = {
        "permissions": [{"id": "perm1"}]
    }

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["truncated"] is False
    assert "next_page_token" not in result


def test_list_permissions_attaches_resource_key_header(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    list_request = service.permissions.return_value.list.return_value
    list_request.headers = {}
    list_request.execute.return_value = {"permissions": []}

    url = "https://drive.google.com/file/d/abc123/view?resourcekey=0-Rkey123"
    google_drive.google_drive_list_permissions(url)

    assert list_request.headers == {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}


def test_list_permissions_caps_oversized_output(monkeypatch):
    # Pinned rather than relying on the ~50KB default -- see
    # test_search_caps_oversized_output's comment for why.
    monkeypatch.setattr(google_drive, "get_tool_max_output_length", lambda: 3000)
    service = _mock_drive_service(monkeypatch)
    huge_permissions = [
        {
            "id": f"perm{i}",
            "type": "user",
            "role": "reader",
            "emailAddress": f"user{i}@example.com",
            "displayName": "x" * 200,
        }
        for i in range(2000)
    ]
    service.permissions.return_value.list.return_value.execute.return_value = {
        "permissions": huge_permissions
    }

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert 0 < len(result["permissions"]) < len(huge_permissions)


def test_list_permissions_follows_next_page_token(monkeypatch):
    """A Shared Drive item returns at most 100 permissions per page when
    pageSize isn't set; without following nextPageToken, a heavily-shared
    Shared Drive folder would silently under-report collaborators."""
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.list.return_value.execute.side_effect = [
        {
            "permissions": [{"id": "perm1"}],
            "nextPageToken": "page2",
        },
        {
            "permissions": [{"id": "perm2"}],
        },
    ]

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["status"] == "success"
    assert [p["id"] for p in result["permissions"]] == ["perm1", "perm2"]
    calls = service.permissions.return_value.list.call_args_list
    assert calls[0].kwargs["pageToken"] is None
    assert calls[1].kwargs["pageToken"] == "page2"


def test_list_permissions_reports_truncated_when_page_bound_is_exhausted(
    monkeypatch,
):
    """If the loop-safety bound is hit while a nextPageToken still exists
    and the accumulated size never crossed the output cap, the result must
    still say truncated=True -- more permissions may exist on pages that
    were never fetched, so reporting a complete list would be dishonest
    even though what little was collected happens to fit comfortably."""
    monkeypatch.setattr(google_drive, "_MAX_PERMISSION_LIST_PAGES", 3)
    monkeypatch.setattr(google_drive, "get_tool_max_output_length", lambda: 3000)
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.list.return_value.execute.return_value = {
        "permissions": [{"id": "perm"}],
        "nextPageToken": "more",  # always another page available
    }

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert service.permissions.return_value.list.call_count == 3


def test_list_permissions_stops_paginating_once_over_the_output_limit(monkeypatch):
    """_capped_list_response will halve an oversized permissions list back
    down anyway, so once accumulated permissions already exceed the output
    limit, fetching further pages is pure waste -- more blocking API calls
    for data that gets thrown away immediately after."""
    service = _mock_drive_service(monkeypatch)
    monkeypatch.setattr(google_drive, "get_tool_max_output_length", lambda: 3000)
    huge_page = [
        {
            "id": f"perm{i}",
            "type": "user",
            "role": "reader",
            "emailAddress": f"user{i}@example.com",
            "displayName": "x" * 200,
        }
        for i in range(2000)
    ]
    service.permissions.return_value.list.return_value.execute.return_value = {
        "permissions": huge_page,
        "nextPageToken": "page2",  # would keep going forever if not stopped
    }

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    # Stopped after the first oversized page rather than following
    # nextPageToken indefinitely.
    assert service.permissions.return_value.list.call_count == 1


def test_list_permissions_pagination_check_matches_the_real_encoding(monkeypatch):
    """The early-stop length check must use ensure_ascii=False, matching
    _capped_list_response's actual encoding -- otherwise a page of
    non-ASCII displayName/emailAddress values (CJK/accented/emoji names are
    common on a Shared Drive) inflates the check's estimate (each such
    character escapes to a 6+-char \\uXXXX sequence under the ensure_ascii
    default) well past what the real, smaller ensure_ascii=False payload
    needs, causing pagination to stop before a real nextPageToken is
    exhausted -- while the final response, measuring the true smaller size,
    reports truncated=False and silently omits the remaining page(s)."""
    service = _mock_drive_service(monkeypatch)
    monkeypatch.setattr(google_drive, "get_tool_max_output_length", lambda: 3000)

    page1 = [
        {
            "id": f"perm{i}",
            "type": "user",
            "role": "reader",
            "emailAddress": f"user{i}@example.com",
            "displayName": "中文姓名" * 5,
        }
        for i in range(15)
    ]
    page2 = [
        {"id": "perm15", "type": "user", "role": "reader", "emailAddress": "a@x.com"},
        {"id": "perm16", "type": "user", "role": "reader", "emailAddress": "b@x.com"},
    ]
    service.permissions.return_value.list.return_value.execute.side_effect = [
        {"permissions": page1, "nextPageToken": "page2"},
        {"permissions": page2},
    ]

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert service.permissions.return_value.list.call_count == 2
    assert result["status"] == "success"
    assert result["truncated"] is False
    assert {p["id"] for p in result["permissions"]} == {f"perm{i}" for i in range(17)}


def test_list_permissions_defaults_missing_key(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.list.return_value.execute.return_value = {}

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["status"] == "success"
    assert result["permissions"] == []


def test_list_permissions_returns_error_payload_on_failure(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.list.return_value.execute.side_effect = (
        RuntimeError("not found")
    )

    result = json.loads(google_drive.google_drive_list_permissions("fid"))

    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_share_file_grants_role_to_email(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.create.return_value.execute.return_value = {
        "id": "perm1",
        "type": "user",
        "role": "writer",
        "emailAddress": "a@x.com",
    }

    result = json.loads(
        google_drive.google_drive_share_file(
            "fid", "a@x.com", role="writer", send_notification=True, message="hi"
        )
    )

    assert result["status"] == "success"
    assert result["permission"]["id"] == "perm1"
    kwargs = service.permissions.return_value.create.call_args.kwargs
    assert kwargs["body"] == {
        "type": "user",
        "role": "writer",
        "emailAddress": "a@x.com",
    }
    assert kwargs["sendNotificationEmail"] is True
    assert kwargs["emailMessage"] == "hi"
    assert kwargs["supportsAllDrives"] is True


def test_share_file_grants_role_to_group(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.create.return_value.execute.return_value = {
        "id": "perm1",
        "type": "group",
        "role": "reader",
        "emailAddress": "team@x.com",
    }

    result = json.loads(
        google_drive.google_drive_share_file(
            "fid", "team@x.com", role="reader", entity_type="group"
        )
    )

    assert result["status"] == "success"
    kwargs = service.permissions.return_value.create.call_args.kwargs
    assert kwargs["body"] == {
        "type": "group",
        "role": "reader",
        "emailAddress": "team@x.com",
    }


@pytest.mark.parametrize("bad_entity_type", ["domain", "anyone", ""])
def test_share_file_rejects_entity_type_outside_user_or_group(
    monkeypatch, bad_entity_type
):
    service = _mock_drive_service(monkeypatch)

    result = json.loads(
        google_drive.google_drive_share_file(
            "fid", "a@x.com", entity_type=bad_entity_type
        )
    )

    assert result["status"] == "error"
    assert "entity_type" in result["message"]
    service.permissions.return_value.create.assert_not_called()


def test_share_file_attaches_resource_key_header(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    create_request = service.permissions.return_value.create.return_value
    create_request.headers = {}
    create_request.execute.return_value = {
        "id": "perm1",
        "type": "user",
        "role": "reader",
        "emailAddress": "a@x.com",
    }

    url = "https://drive.google.com/file/d/abc123/view?resourcekey=0-Rkey123"
    google_drive.google_drive_share_file(url, "a@x.com")

    assert create_request.headers == {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}


def test_share_file_defaults_to_reader_with_notification(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.create.return_value.execute.return_value = {}

    result = json.loads(google_drive.google_drive_share_file("fid", "a@x.com"))

    assert result["status"] == "success"
    kwargs = service.permissions.return_value.create.call_args.kwargs
    assert kwargs["body"]["role"] == "reader"
    assert kwargs["sendNotificationEmail"] is True


def test_share_file_omits_email_message_when_notification_disabled(monkeypatch):
    """The Drive API rejects emailMessage outright when sendNotificationEmail
    is false, so a message passed alongside send_notification=False must not
    reach the request -- regardless of whether the caller also wanted a
    message, suppressing the notification wins."""
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.create.return_value.execute.return_value = {}

    result = json.loads(
        google_drive.google_drive_share_file(
            "fid", "a@x.com", send_notification=False, message="hi"
        )
    )

    assert result["status"] == "success"
    kwargs = service.permissions.return_value.create.call_args.kwargs
    assert kwargs["sendNotificationEmail"] is False
    assert "emailMessage" not in kwargs


@pytest.mark.parametrize("bad_role", ["owner", "organizer", ""])
def test_share_file_rejects_role_outside_the_share_roles(monkeypatch, bad_role):
    """ "owner" is deliberately not accepted: ownership transfer is a
    distinct, irreversible action this connector does not perform on the
    model's behalf."""
    service = _mock_drive_service(monkeypatch)

    result = json.loads(
        google_drive.google_drive_share_file("fid", "a@x.com", role=bad_role)
    )

    assert result["status"] == "error"
    assert "role" in result["message"]
    service.permissions.return_value.create.assert_not_called()


@pytest.mark.parametrize(
    "bad_email",
    ["", "not-an-email", " a@x.com", "a@", "@b", "a@b"],
)
def test_share_file_rejects_invalid_email(monkeypatch, bad_email):
    """ "a@", "@b", and "a@b" (no dot in the domain) look email-shaped enough
    to pass a bare "@" in email check, but are not valid addresses."""
    service = _mock_drive_service(monkeypatch)

    result = json.loads(google_drive.google_drive_share_file("fid", bad_email))

    assert result["status"] == "error"
    assert "email" in result["message"]
    service.permissions.return_value.create.assert_not_called()


def test_share_file_returns_error_payload_on_failure(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.create.return_value.execute.side_effect = (
        RuntimeError("insufficientFilePermissions")
    )

    result = json.loads(google_drive.google_drive_share_file("fid", "a@x.com"))

    assert result["status"] == "error"
    assert "insufficientFilePermissions" in result["message"]


def test_update_permission_changes_role(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.update.return_value.execute.return_value = {
        "id": "perm1",
        "role": "writer",
    }

    result = json.loads(
        google_drive.google_drive_update_permission("fid", "perm1", "writer")
    )

    assert result["status"] == "success"
    kwargs = service.permissions.return_value.update.call_args.kwargs
    assert kwargs["permissionId"] == "perm1"
    assert kwargs["body"] == {"role": "writer"}
    assert kwargs["fields"] == "id, type, role, emailAddress, displayName"


def test_update_permission_attaches_resource_key_header(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    update_request = service.permissions.return_value.update.return_value
    update_request.headers = {}
    update_request.execute.return_value = {"id": "perm1", "role": "writer"}

    url = "https://drive.google.com/file/d/abc123/view?resourcekey=0-Rkey123"
    google_drive.google_drive_update_permission(url, "perm1", "writer")

    assert update_request.headers == {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}


@pytest.mark.parametrize("bad_permission_id", ["..", "../other", "perm/../file"])
def test_update_permission_rejects_dot_segment_permission_id(
    monkeypatch, bad_permission_id
):
    """permission_id never goes through URL resolution, but googleapiclient
    does not percent-encode "." before interpolating it into the request
    path -- an unvalidated dot-segment id can be collapsed by normalization
    into an unrelated endpoint. require_clean_identifier alone accepts
    ".." (non-empty, no surrounding whitespace); _require_drive_id must
    additionally reject it via the same character class real ids use."""
    service = _mock_drive_service(monkeypatch)

    result = json.loads(
        google_drive.google_drive_update_permission("fid", bad_permission_id, "reader")
    )

    assert result["status"] == "error"
    assert "permission_id" in result["message"]
    service.permissions.return_value.update.assert_not_called()


@pytest.mark.parametrize("bad_permission_id", ["..", "../other", "perm/../file"])
def test_remove_permission_rejects_dot_segment_permission_id(
    monkeypatch, bad_permission_id
):
    service = _mock_drive_service(monkeypatch)

    result = json.loads(
        google_drive.google_drive_remove_permission("fid", bad_permission_id)
    )

    assert result["status"] == "error"
    assert "permission_id" in result["message"]
    service.permissions.return_value.delete.assert_not_called()


@pytest.mark.parametrize("bad_role", ["owner", "organizer", ""])
def test_update_permission_rejects_role_outside_the_share_roles(monkeypatch, bad_role):
    service = _mock_drive_service(monkeypatch)

    result = json.loads(
        google_drive.google_drive_update_permission("fid", "perm1", bad_role)
    )

    assert result["status"] == "error"
    assert "role" in result["message"]
    service.permissions.return_value.update.assert_not_called()


def test_update_permission_returns_error_payload_on_failure(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.update.return_value.execute.side_effect = (
        RuntimeError("permission not found")
    )

    result = json.loads(
        google_drive.google_drive_update_permission("fid", "perm1", "reader")
    )

    assert result["status"] == "error"
    assert "permission not found" in result["message"]


def test_remove_permission_success(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.delete.return_value.execute.return_value = {}

    result = json.loads(google_drive.google_drive_remove_permission("fid", "perm1"))

    assert result["status"] == "success"
    assert "perm1" in result["message"]
    service.permissions.return_value.get.assert_not_called()


def test_remove_permission_attaches_resource_key_header(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    delete_request = service.permissions.return_value.delete.return_value
    delete_request.headers = {}
    delete_request.execute.return_value = {}

    url = "https://drive.google.com/file/d/abc123/view?resourcekey=0-Rkey123"
    google_drive.google_drive_remove_permission(url, "perm1")

    assert delete_request.headers == {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}


def test_remove_permission_returns_error_payload_on_failure(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.delete.return_value.execute.side_effect = (
        RuntimeError("permission not found")
    )

    result = json.loads(google_drive.google_drive_remove_permission("fid", "perm1"))

    assert result["status"] == "error"
    assert "permission not found" in result["message"]


class TestExecuteIgnoring204SslEof:
    """Unit coverage for the helper shared by google_drive_delete_file and
    google_drive_remove_permission, which both call Drive endpoints that
    return 204 No Content -- a response shape a proxy can turn into an SSL
    EOF error even though the operation already succeeded server-side."""

    def test_returns_normally_when_execute_succeeds(self):
        execute = Mock()
        verify_done = Mock()

        google_drive._execute_ignoring_204_ssl_eof(execute, verify_done)

        execute.assert_called_once()
        verify_done.assert_not_called()

    def test_propagates_unrelated_errors_without_verifying(self):
        execute = Mock(side_effect=RuntimeError("quota exceeded"))
        verify_done = Mock()

        with pytest.raises(RuntimeError, match="quota exceeded"):
            google_drive._execute_ignoring_204_ssl_eof(execute, verify_done)

        verify_done.assert_not_called()

    def test_swallows_ssl_eof_when_verify_confirms_it_completed(self):
        execute = Mock(side_effect=Exception("UNEXPECTED_EOF_WHILE_READING"))
        verify_done = Mock(side_effect=Exception("404 not found"))

        google_drive._execute_ignoring_204_ssl_eof(execute, verify_done)

        verify_done.assert_called_once()

    def test_raises_ssl_eof_when_verify_shows_it_did_not_complete(self):
        execute = Mock(side_effect=Exception("UNEXPECTED_EOF_WHILE_READING"))
        verify_done = Mock(return_value=None)  # object is still there

        with pytest.raises(Exception, match="UNEXPECTED_EOF_WHILE_READING"):
            google_drive._execute_ignoring_204_ssl_eof(execute, verify_done)

    def test_trusts_http_status_over_uri_text_on_a_real_http_error(self):
        """A real HttpError's __str__ embeds the request URI (which
        contains the resolved file/permission id) alongside the actual
        status code. If an id happens to contain the digits "404", a naive
        substring match on str(verify_err) would misread a genuine 403
        (object still exists; access merely denied) as "confirmed gone".
        Checking verify_err.resp.status first must not fall for that."""
        from googleapiclient.errors import HttpError

        resp = type("Resp", (), {"status": 403, "reason": "Forbidden"})()
        forbidden = HttpError(
            resp,
            b'{"error": {"message": "Permission denied"}}',
            uri="https://www.googleapis.com/drive/v3/files/1AbC404xyz?fields=id",
        )
        assert "404" in str(forbidden)  # confirms the trap is real

        execute = Mock(side_effect=Exception("UNEXPECTED_EOF_WHILE_READING"))
        verify_done = Mock(side_effect=forbidden)

        with pytest.raises(Exception, match="UNEXPECTED_EOF_WHILE_READING"):
            google_drive._execute_ignoring_204_ssl_eof(execute, verify_done)

    def test_treats_a_real_http_404_as_confirmed_gone(self):
        from googleapiclient.errors import HttpError

        resp = type("Resp", (), {"status": 404, "reason": "Not Found"})()
        not_found = HttpError(
            resp, b'{"error": {"message": "File not found"}}', uri="https://x/1abc"
        )
        execute = Mock(side_effect=Exception("UNEXPECTED_EOF_WHILE_READING"))
        verify_done = Mock(side_effect=not_found)

        google_drive._execute_ignoring_204_ssl_eof(execute, verify_done)  # no raise

    def test_is_confirmed_gone_handles_a_stringified_status(self):
        """httplib2.Response always coerces .status to int in practice, but
        the check shouldn't silently stop working (skip straight to
        returning False, never reaching the string fallback) if some other
        transport, mock, or future library version ever represents it as
        e.g. "404" instead of 404."""
        err = type("Err", (), {"resp": type("Resp", (), {"status": "404"})()})()
        assert google_drive._is_confirmed_gone(err) is True

    def test_raises_even_when_original_error_text_contains_not_found(self):
        """The "not complete" exception raised when verify_done() shows the
        object is still there must never be caught by the except clause that
        is meant to interpret *verify_done's own* exception -- if it were,
        an original SSL error whose text happens to mention "not found" or
        "404" (plausible in a proxy's wrapped error message) would make a
        genuine failure-to-delete look like a success."""
        execute = Mock(
            side_effect=Exception(
                "UNEXPECTED_EOF_WHILE_READING: upstream returned 404 not found"
            )
        )
        verify_done = Mock(return_value=None)  # object is still there

        with pytest.raises(Exception, match="did not complete"):
            google_drive._execute_ignoring_204_ssl_eof(execute, verify_done)


def test_delete_file_tolerates_ssl_eof_on_204_response(monkeypatch):
    """End-to-end regression for the pre-existing SSL EOF tolerance that
    used to live inline in google_drive_delete_file, now routed through the
    shared helper."""
    service = _mock_drive_service(monkeypatch)
    service.files.return_value.delete.return_value.execute.side_effect = Exception(
        "UNEXPECTED_EOF_WHILE_READING"
    )
    service.files.return_value.get.return_value.execute.side_effect = Exception(
        "404: not found"
    )

    result = json.loads(google_drive.google_drive_delete_file("fid"))

    assert result["status"] == "success"


def test_delete_file_rejects_dot_segment_file_id_before_calling_the_api(monkeypatch):
    """google_drive_delete_file "skips the trash and permanently deletes" --
    a literal file_id=".." reaching the API as a dot-segment fileId is the
    highest-stakes reachable case of the bare-input validation gap."""
    get_service = Mock()
    monkeypatch.setattr(google_drive, "get_drive_service", get_service)

    result = json.loads(google_drive.google_drive_delete_file(".."))

    assert result["status"] == "error"
    assert "file_id must look like a Drive id" in result["message"]
    get_service.assert_not_called()


def test_delete_file_attaches_resource_key_header(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    delete_request = service.files.return_value.delete.return_value
    delete_request.headers = {}
    delete_request.execute.return_value = {}

    url = "https://drive.google.com/file/d/abc123/view?resourcekey=0-Rkey123"
    google_drive.google_drive_delete_file(url)

    assert delete_request.headers == {"X-Goog-Drive-Resource-Keys": "abc123/0-Rkey123"}


def test_remove_permission_tolerates_ssl_eof_on_204_response(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.delete.return_value.execute.side_effect = (
        Exception("UNEXPECTED_EOF_WHILE_READING")
    )
    service.permissions.return_value.get.return_value.execute.side_effect = Exception(
        "404: not found"
    )

    result = json.loads(google_drive.google_drive_remove_permission("fid", "perm1"))

    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# google_drive_get_file_content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mime_type", ["text/plain", "text/csv", "text/markdown", "application/json"]
)
def test_get_file_content_accepts_text_mime_types(monkeypatch, mime_type):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "notes.txt",
        "mimeType": "text/plain",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"hello world")

    result = json.loads(google_drive.google_drive_get_file_content("f1", mime_type))

    assert result["status"] == "success"
    assert result["content"] == "hello world"


@pytest.mark.parametrize(
    "mime_type",
    [
        "application/pdf",
        "image/png",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ],
)
def test_get_file_content_rejects_binary_mime_types(monkeypatch, mime_type):
    """Regression guard: decoding binary content as UTF-8 with
    errors="replace" silently corrupts it — reject before even calling the
    API rather than return garbage that looks superficially like success."""
    files = Mock()
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_get_file_content("f1", mime_type))

    assert result["status"] == "error"
    assert "google_drive_download_file" in result["message"]
    files.get.assert_not_called()


def test_get_file_content_rejects_regular_file_whose_real_mime_type_is_binary(
    monkeypatch,
):
    """Regression guard for the actual bug this tool exists to fix: for a
    regular (non-Workspace) file, get_media() ignores the mime_type
    parameter entirely and returns the file's own real bytes — so it's
    file_mime_type (from Drive's metadata), not the request's mime_type
    default of "text/plain", that must gate whether decoding as UTF-8 is
    safe. Calling with the default text/plain param on an actual PDF must
    still be rejected, not silently corrupted."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "report.pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"%PDF-1.4 real pdf bytes")

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "error"
    assert "google_drive_download_file" in result["message"]


def test_get_file_content_accepts_regular_text_file(monkeypatch):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "notes.txt",
        "mimeType": "text/plain",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"hello world")

    result = json.loads(google_drive.google_drive_get_file_content("f1"))

    assert result["status"] == "success"
    assert result["content"] == "hello world"


def test_get_file_content_accepts_rtf(monkeypatch):
    """Regression guard: RTF is 7-bit-ASCII-clean per spec (non-ASCII
    content is escaped, not raw bytes), so it round-trips through UTF-8
    decoding safely — the binary-content guard must not reject it."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "notes.rtf",
        "mimeType": "application/rtf",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, rb"{\rtf1 hello}")

    result = json.loads(
        google_drive.google_drive_get_file_content("f1", "application/rtf")
    )

    assert result["status"] == "success"


def test_get_file_content_exports_workspace_doc(monkeypatch):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Notes",
        "mimeType": "application/vnd.google-apps.document",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"exported text")

    result = json.loads(google_drive.google_drive_get_file_content("f1", "text/plain"))

    assert result["status"] == "success"
    assert result["content"] == "exported text"
    files.export_media.assert_called_once_with(fileId="f1", mimeType="text/plain")
    files.get_media.assert_not_called()


def test_get_file_content_returns_error_payload_on_api_failure(monkeypatch):
    files = Mock()
    files.get.return_value.execute.side_effect = RuntimeError("boom")
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_get_file_content("f1", "text/plain"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# google_drive_download_file
# ---------------------------------------------------------------------------


def test_download_file_writes_regular_binary_file(monkeypatch, tmp_path):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "photo.png",
        "mimeType": "image/png",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"\x89PNG-fake-bytes")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    assert result["size"] == len(b"\x89PNG-fake-bytes")
    output_path = tmp_path / "output" / "photo.png"
    assert result["path"] == str(output_path)
    assert output_path.read_bytes() == b"\x89PNG-fake-bytes"
    files.get_media.assert_called_once_with(fileId="f1", supportsAllDrives=True)
    files.export_media.assert_not_called()


def test_download_file_exports_workspace_doc_with_extension_appended(
    monkeypatch, tmp_path
):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Onboarding Deck",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"%PDF-1.4 fake pdf")

    result = json.loads(
        google_drive.google_drive_download_file("f1", mime_type="application/pdf")
    )

    assert result["status"] == "success"
    output_path = tmp_path / "output" / "Onboarding Deck.pdf"
    assert result["path"] == str(output_path)
    assert output_path.read_bytes() == b"%PDF-1.4 fake pdf"
    files.export_media.assert_called_once_with(fileId="f1", mimeType="application/pdf")
    files.get_media.assert_not_called()


def test_download_file_requires_mime_type_for_workspace_doc(monkeypatch):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Onboarding Deck",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "error"
    assert "mime_type" in result["message"]
    files.export_media.assert_not_called()


def test_download_file_uses_explicit_filename(monkeypatch, tmp_path):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Onboarding Deck",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file(
            "f1", mime_type="application/pdf", filename="custom.pdf"
        )
    )

    assert result["status"] == "success"
    assert result["path"] == str(tmp_path / "output" / "custom.pdf")


def test_download_file_appends_extension_to_explicit_filename_missing_one(
    monkeypatch, tmp_path
):
    """Regression guard: an explicit filename with no extension must still
    get the export mime_type's extension appended, exactly like the
    default (Drive-name-derived) filename already does — the caller
    shouldn't lose the .pdf just because they named the file themselves."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Onboarding Deck",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file(
            "f1", mime_type="application/pdf", filename="report"
        )
    )

    assert result["status"] == "success"
    assert result["path"] == str(tmp_path / "output" / "report.pdf")


def test_download_file_does_not_double_extension_on_case_mismatch(
    monkeypatch, tmp_path
):
    """Regression guard: matching the target extension must be
    case-insensitive — a Drive name already ending in ".PDF" (any case)
    exported to "application/pdf" must not become "....PDF.pdf"."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "Report.PDF",
        "mimeType": "application/vnd.google-apps.document",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file("f1", mime_type="application/pdf")
    )

    assert result["status"] == "success"
    assert result["path"] == str(tmp_path / "output" / "Report.PDF")


def test_download_file_sanitizes_path_traversal_in_filename(monkeypatch, tmp_path):
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "../../etc/passwd",
        "mimeType": "text/plain",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.parent == tmp_path / "output"
    assert output_path.is_relative_to(tmp_path / "output")


def test_download_file_sanitizes_unsafe_characters_in_filename(monkeypatch, tmp_path):
    """Regression guard: a name with characters Path.name alone would leave
    untouched (no "/" to strip) must still be neutralized by the character
    allowlist — otherwise this test would pass even with the sanitizer
    reduced to a no-op Path(name).name call."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "weird:name?.txt",
        "mimeType": "text/plain",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.parent == tmp_path / "output"
    assert ":" not in output_path.name
    assert "?" not in output_path.name
    assert output_path.name.endswith(".txt")


def test_download_file_preserves_extension_for_degenerate_drive_name(
    monkeypatch, tmp_path
):
    """Regression guard: sanitizing a degenerate name (all characters the
    allowlist/strip would remove) must happen *before* the export extension
    is appended — otherwise the trailing ".strip('._')" eats into the
    extension's own leading dot and produces a bare "pdf" instead of a
    usable "file.pdf"."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "...",
        "mimeType": "application/vnd.google-apps.presentation",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file("f1", mime_type="application/pdf")
    )

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.name == "file.pdf"


def test_download_file_preserves_extension_for_non_ascii_drive_name(
    monkeypatch, tmp_path
):
    """Regression guard: a name whose entire stem is non-ASCII (e.g. CJK)
    sanitizes down to nothing on its own, but the extension must survive —
    this is the regular-file (get_media) branch, which has no separate
    "extension" variable to fall back on the way the Workspace-export
    branch does, so the fix has to live in _safe_output_filename itself."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "季度报告.pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.name == "file.pdf"


def test_download_file_preserves_extension_only_drive_name(monkeypatch, tmp_path):
    """Regression guard: Path(".pdf").suffix is '' per pathlib's dotfile
    convention (a leading dot with nothing before it never counts as an
    extension separator) — so a Drive name that's *exactly* an extension
    must still be recognized as one, not silently reduced to "pdf" with
    the leading dot dropped."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": ".pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.name == "file.pdf"


def test_download_file_sanitizes_path_traversal_in_explicit_filename(
    monkeypatch, tmp_path
):
    """Regression guard: only the Drive-reported name was tested for
    traversal/unsafe-character sanitization elsewhere — an explicit
    `filename` argument goes through the exact same _safe_output_filename
    call and must be sanitized identically."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "report.txt",
        "mimeType": "text/plain",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(
        google_drive.google_drive_download_file("f1", filename="../../etc/passwd")
    )

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert output_path.parent == tmp_path / "output"
    assert output_path.is_relative_to(tmp_path / "output")


def test_download_file_truncates_overlong_filename(monkeypatch, tmp_path):
    """Regression guard: an unbounded sanitized filename can exceed the
    ~255-byte NAME_MAX most filesystems enforce, making the final
    write_bytes() raise OSError/ENAMETOOLONG and fail the whole call
    instead of just using a shorter name."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": ("a" * 500) + ".pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert len(output_path.name) <= 210
    assert output_path.name.endswith(".pdf")


def test_download_file_truncates_overlong_suffix_too(monkeypatch, tmp_path):
    """Regression guard: the length cap must also bound the suffix, not
    just the stem — a name like "a." + 300 chars has almost its entire
    length in what _split_stem_suffix treats as the "extension" (everything
    from the last dot onward), so truncating only the stem would still
    leave a 300+ char filename."""
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "a." + ("x" * 300),
        "mimeType": "text/plain",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    output_path = Path(result["path"])
    assert len(output_path.name) <= 30


def test_download_file_errors_when_no_task_workspace_is_configured(
    monkeypatch, tmp_path
):
    """Regression guard: an unset output-dir env var must fail loudly
    rather than silently writing into whatever directory the MCP
    subprocess happens to have as its cwd."""
    monkeypatch.delenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", raising=False)
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "report.pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "error"
    assert "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR" in result["message"]
    assert not (tmp_path / "output").exists()


def test_download_file_dedupes_existing_filename(monkeypatch, tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "deck.pdf").write_bytes(b"already here")

    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "f1",
        "name": "deck.pdf",
        "mimeType": "application/pdf",
    }
    _mock_drive_service_with_files(monkeypatch, files)
    _patch_downloader(monkeypatch, b"new content")

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "success"
    assert result["path"] == str(tmp_path / "output" / "deck (1).pdf")
    # The original file must survive untouched.
    assert (tmp_path / "output" / "deck.pdf").read_bytes() == b"already here"


def test_download_file_returns_error_payload_on_api_failure(monkeypatch):
    files = Mock()
    files.get.return_value.execute.side_effect = RuntimeError("boom")
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_download_file("f1"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# google_drive_upload_file
# ---------------------------------------------------------------------------


def test_upload_file_sends_real_binary_content(monkeypatch, tmp_path):
    """Regression guard for the actual production bug: uploading an
    already-generated PDF must send its real bytes with a real
    application/pdf mimeType — not a text/plain placeholder string, which
    is all google_drive_create_file's str content parameter can carry."""
    allowed_dir = tmp_path / "workspace"
    local_pdf = allowed_dir / "Regional_Performance_Summary.pdf"
    local_pdf.write_bytes(b"%PDF-1.4 fake pdf bytes")

    files = Mock()
    files.create.return_value.execute.return_value = {
        "id": "new-file-id",
        "name": "Regional_Performance_Summary.pdf",
        "mimeType": "application/pdf",
        "webViewLink": "https://drive.google.com/file/d/new-file-id/view",
    }
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_upload_file(str(local_pdf)))

    assert result["status"] == "success"
    assert result["file"]["mimeType"] == "application/pdf"

    _, kwargs = files.create.call_args
    assert kwargs["body"] == {
        "name": "Regional_Performance_Summary.pdf",
        "mimeType": "application/pdf",
    }
    media = kwargs["media_body"]
    assert media.mimetype() == "application/pdf"
    assert media.size() == len(b"%PDF-1.4 fake pdf bytes")


def test_upload_file_accepts_explicit_name_and_mime_type(monkeypatch, tmp_path):
    allowed_dir = tmp_path / "workspace"
    local_file = allowed_dir / "data.bin"
    local_file.write_bytes(b"\x00\x01\x02")

    files = Mock()
    files.create.return_value.execute.return_value = {"id": "f1"}
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(
        google_drive.google_drive_upload_file(
            str(local_file), name="custom.dat", mime_type="application/octet-stream"
        )
    )

    assert result["status"] == "success"
    _, kwargs = files.create.call_args
    assert kwargs["body"]["name"] == "custom.dat"
    assert kwargs["body"]["mimeType"] == "application/octet-stream"


def test_upload_file_defaults_mime_type_when_unguessable(monkeypatch, tmp_path):
    allowed_dir = tmp_path / "workspace"
    local_file = allowed_dir / "mystery_file_no_extension"
    local_file.write_bytes(b"some bytes")

    files = Mock()
    files.create.return_value.execute.return_value = {"id": "f1"}
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_upload_file(str(local_file)))

    assert result["status"] == "success"
    _, kwargs = files.create.call_args
    assert kwargs["body"]["mimeType"] == "application/octet-stream"


def test_upload_file_includes_parent_id_when_given(monkeypatch, tmp_path):
    allowed_dir = tmp_path / "workspace"
    local_file = allowed_dir / "report.pdf"
    local_file.write_bytes(b"content")

    files = Mock()
    files.create.return_value.execute.return_value = {"id": "f1"}
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(
        google_drive.google_drive_upload_file(str(local_file), parent_id="folder-1")
    )

    assert result["status"] == "success"
    _, kwargs = files.create.call_args
    assert kwargs["body"]["parents"] == ["folder-1"]


def test_upload_file_rejects_path_outside_allowed_directories(monkeypatch, tmp_path):
    outside_file = tmp_path / "outside.pdf"
    outside_file.write_bytes(b"content")

    files = Mock()
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_upload_file(str(outside_file)))

    assert result["status"] == "error"
    assert "allowed directories" in result["message"]
    files.create.assert_not_called()


def test_upload_file_rejects_missing_file(monkeypatch, tmp_path):
    allowed_dir = tmp_path / "workspace"
    missing_path = allowed_dir / "does_not_exist.pdf"

    files = Mock()
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_upload_file(str(missing_path)))

    assert result["status"] == "error"
    files.create.assert_not_called()


def test_upload_file_rejects_empty_file(monkeypatch, tmp_path):
    allowed_dir = tmp_path / "workspace"
    empty_file = allowed_dir / "empty.pdf"
    empty_file.write_bytes(b"")

    files = Mock()
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_upload_file(str(empty_file)))

    assert result["status"] == "error"
    assert "empty" in result["message"].lower()
    files.create.assert_not_called()


def test_upload_file_returns_error_payload_on_api_failure(monkeypatch, tmp_path):
    allowed_dir = tmp_path / "workspace"
    local_file = allowed_dir / "report.pdf"
    local_file.write_bytes(b"content")

    files = Mock()
    files.create.return_value.execute.side_effect = RuntimeError("boom")
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_upload_file(str(local_file)))

    assert result["status"] == "error"
    assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# google_drive_create_file — binary-content guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["report.pdf", "photo.PNG", "deck.pptx", "archive.zip"]
)
def test_create_file_rejects_binary_looking_names(monkeypatch, name):
    """Regression guard for the actual production bug: google_drive_create_file
    can only write text (content is utf-8 encoded), so a caller naming the
    file like a binary format must be steered to google_drive_upload_file
    instead of silently getting a text/plain file with a misleading name."""
    files = Mock()
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(google_drive.google_drive_create_file(name, "some text"))

    assert result["status"] == "error"
    assert "google_drive_upload_file" in result["message"]
    files.create.assert_not_called()


def test_create_file_still_allows_binary_looking_name_for_google_doc_export(
    monkeypatch,
):
    """A Google Workspace document conversion legitimately takes plain
    text/HTML content regardless of the target doc's display name, so the
    binary-extension guard must not block it."""
    files = Mock()
    files.create.return_value.execute.return_value = {"id": "f1"}
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(
        google_drive.google_drive_create_file(
            "backup.pdf",
            "<p>hello</p>",
            mime_type="application/vnd.google-apps.document",
        )
    )

    assert result["status"] == "success"
    files.create.assert_called_once()


def test_create_file_allows_plain_text_names(monkeypatch):
    files = Mock()
    files.create.return_value.execute.return_value = {"id": "f1"}
    _mock_drive_service_with_files(monkeypatch, files)

    result = json.loads(
        google_drive.google_drive_create_file("notes.txt", "hello world")
    )

    assert result["status"] == "success"
    files.create.assert_called_once()


@pytest.mark.parametrize(
    "name,mime_type",
    [
        ("report", "application/pdf"),
        ("report.txt", "application/pdf"),
        ("photo", "image/png"),
        ("data.bin", "application/octet-stream"),
    ],
)
def test_create_file_rejects_binary_mime_type_even_with_a_non_flagged_name(
    monkeypatch, name, mime_type
):
    """Regression guard: the name-extension check alone isn't enough — a
    caller that declares an explicit binary mime_type (rather than naming
    the file like a binary format) must be rejected too, since content is
    always UTF-8-encoded text regardless of what mime_type claims. Without
    this, a caller could bypass the binary-looking-name guard entirely by
    using a name with no extension, a .txt extension, or an extension
    outside the hardcoded binary set, while still declaring a binary
    mime_type — producing the exact declared-type-vs-content mismatch this
    guard exists to prevent."""
    files = Mock()
    _mock_drive_service(monkeypatch, files)

    result = json.loads(
        google_drive.google_drive_create_file(name, "some text", mime_type=mime_type)
    )

    assert result["status"] == "error"
    assert "google_drive_upload_file" in result["message"]
    files.create.assert_not_called()


@pytest.mark.parametrize(
    "mime_type", ["text/plain", "text/csv", "application/json", "application/rtf"]
)
def test_create_file_allows_text_safe_mime_types(monkeypatch, mime_type):
    """The new mime_type-based check must not reject legitimate text-safe
    formats that don't start with "text/" (application/json,
    application/rtf) — it should reuse the same text-safety notion as
    google_drive_get_file_content's _is_text_mime_type, not a narrower
    one."""
    files = Mock()
    files.create.return_value.execute.return_value = {"id": "f1"}
    _mock_drive_service(monkeypatch, files)

    result = json.loads(
        google_drive.google_drive_create_file("data.json", "{}", mime_type=mime_type)
    )

    assert result["status"] == "success"
    files.create.assert_called_once()
