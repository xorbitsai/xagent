import json
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


@pytest.mark.parametrize("bad_role", ["owner", "organizer", ""])
def test_require_share_role_rejects_roles_outside_the_allowed_set(bad_role):
    with pytest.raises(ValueError, match="role"):
        google_drive._require_share_role(bad_role)


@pytest.mark.parametrize("good_role", ["reader", "commenter", "writer"])
def test_require_share_role_accepts_allowed_roles(good_role):
    assert google_drive._require_share_role(good_role) == good_role


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


def test_create_file_validates_parent_id_before_building_service(monkeypatch):
    get_service = Mock()
    monkeypatch.setattr(google_drive, "get_drive_service", get_service)

    result = json.loads(
        google_drive.google_drive_create_file("notes.txt", "hello", parent_id=123)
    )

    assert result["status"] == "error"
    assert "file_id must be a string" in result["message"]
    get_service.assert_not_called()


def test_create_folder_validates_parent_id_before_building_service(monkeypatch):
    get_service = Mock()
    monkeypatch.setattr(google_drive, "get_drive_service", get_service)

    result = json.loads(
        google_drive.google_drive_create_folder("Subfolder", parent_id=123)
    )

    assert result["status"] == "error"
    assert "file_id must be a string" in result["message"]
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
            self._fh.write(b"<html></html>")
            return None, True

    monkeypatch.setattr(google_drive, "MediaIoBaseDownload", _FakeDownloader)

    result = json.loads(
        google_drive.google_drive_get_file_content(
            "sheet1", mime_type="application/pdf"
        )
    )

    assert result["status"] == "success"
    assert (
        service.files.return_value.export_media.call_args.kwargs["mimeType"]
        == "application/pdf"
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


def test_get_file_content_caps_oversized_output(monkeypatch):
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


def test_list_permissions_caps_oversized_output(monkeypatch):
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


def test_list_permissions_stops_paginating_once_over_the_output_limit(monkeypatch):
    """_capped_list_response will halve an oversized permissions list back
    down anyway, so once accumulated permissions already exceed the output
    limit, fetching further pages is pure waste -- more blocking API calls
    for data that gets thrown away immediately after."""
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


def test_share_file_defaults_to_reader_with_notification(monkeypatch):
    service = _mock_drive_service(monkeypatch)
    service.permissions.return_value.create.return_value.execute.return_value = {}

    json.loads(google_drive.google_drive_share_file("fid", "a@x.com"))

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

    json.loads(
        google_drive.google_drive_share_file(
            "fid", "a@x.com", send_notification=False, message="hi"
        )
    )

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
