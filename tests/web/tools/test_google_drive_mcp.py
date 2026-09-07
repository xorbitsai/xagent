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

    assert mock_target(service).call_args.kwargs["fileId"] == "abc123"


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
    assert service.files.return_value.get.call_args.kwargs["fileId"] == "abc123"
    assert service.files.return_value.get_media.call_args.kwargs["fileId"] == "abc123"


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
    assert service.files.return_value.update.call_args.kwargs["fileId"] == "abc123"


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


@pytest.mark.parametrize("bad_email", ["", "not-an-email", " a@x.com"])
def test_share_file_rejects_invalid_email(monkeypatch, bad_email):
    service = _mock_drive_service(monkeypatch)

    result = json.loads(google_drive.google_drive_share_file("fid", bad_email))

    assert result["status"] == "error"
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
