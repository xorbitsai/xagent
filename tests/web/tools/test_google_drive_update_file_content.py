"""Tests for google_drive_update_file_content, which replaces the content of
an existing Drive file in place, against a mocked Drive API."""

import hashlib
import io
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import googleapiclient.http
import pytest
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import HttpMockSequence
from mcp.types import CallToolResult, TextContent

from xagent.core.tools.adapters.vibe import mcp_adapter as mcp_adapter_module
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    _build_mcp_tool_adapter,
    classify_non_idempotent_write,
)
from xagent.web.tools.mcp import google_drive

DECK_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
OLD_CONTENT = b"original deck bytes"
NEW_CONTENT = b"edited deck bytes, a little longer"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)


@pytest.fixture(autouse=True)
def allowed_dir(tmp_path, monkeypatch):
    directory = tmp_path / "workspace"
    directory.mkdir()
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS", str(directory))
    return directory


@pytest.fixture
def mock_workspace_db():
    # The cross-run test below registers a real durable file, so the global
    # autouse fixture that stubs out file records must not apply here.
    yield


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _current(**overrides):
    file = {
        "id": "deck1",
        "name": "Quarterly Deck.pptx",
        "mimeType": DECK_MIME,
        "trashed": False,
        "size": str(len(OLD_CONTENT)),
        "md5Checksum": _md5(OLD_CONTENT),
        "headRevisionId": "rev-1",
        "version": "7",
        "modifiedTime": "2026-05-12T08:00:00.000Z",
        "webViewLink": "https://drive.google.com/file/d/deck1/view",
        "shared": False,
        "capabilities": {"canModifyContent": True},
    }
    file.update(overrides)
    # Drive leaves out a field it has no value for.
    return {key: value for key, value in file.items() if value is not None}


def _updated(data: bytes = NEW_CONTENT, **overrides):
    file = {
        "id": "deck1",
        "name": "Quarterly Deck.pptx",
        "mimeType": DECK_MIME,
        "size": str(len(data)),
        "md5Checksum": _md5(data),
        "headRevisionId": "rev-2",
        "version": "8",
        "modifiedTime": "2026-10-09T08:00:00.000Z",
        "webViewLink": "https://drive.google.com/file/d/deck1/view",
        "shared": False,
    }
    file.update(overrides)
    return {key: value for key, value in file.items() if value is not None}


def _only_requested_fields(resource, fields):
    """Like Drive, keep only the fields a `fields` string names, including
    one level of nested selection such as "capabilities(canModifyContent)".
    A field the tool forgets to request is then missing, as it would be."""
    if not isinstance(resource, dict):
        return resource
    selected = {
        name: {part.strip() for part in nested.split(",")} if nested else None
        for name, nested in re.findall(r"(\w+)(?:\(([^)]*)\))?", fields)
    }
    kept = {}
    for key, value in resource.items():
        if key not in selected:
            continue
        nested = selected[key]
        if nested is not None and isinstance(value, dict):
            value = {name: item for name, item in value.items() if name in nested}
        kept[key] = value
    return kept


class _Drive:
    """A mocked Drive service that records the media sent to files.update.

    ``current`` and ``updated`` are the full file before and after the
    update; each request returns only the fields it asked for."""

    def __init__(self, monkeypatch, current=None, updated=None):
        self.current = _current() if current is None else current
        self.updated = _updated() if updated is None else updated
        self.files = Mock()
        self.get_request = Mock(headers={})
        self.update_request = Mock(headers={})
        self.uploaded: list[bytes] = []

        def get(**kwargs):
            self.get_request.execute.return_value = _only_requested_fields(
                self.current, kwargs["fields"]
            )
            return self.get_request

        def update(**kwargs):
            media = kwargs["media_body"]
            self.uploaded.append(media.getbytes(0, media.size()))
            self.update_request.execute.return_value = _only_requested_fields(
                self.updated, kwargs["fields"]
            )
            return self.update_request

        self.files.get.side_effect = get
        self.files.update.side_effect = update
        service = Mock()
        service.files.return_value = self.files
        self.get_service = Mock(return_value=service)
        monkeypatch.setattr(google_drive, "get_drive_service", self.get_service)

    def assert_nothing_written(self):
        self.files.update.assert_not_called()
        self.files.create.assert_not_called()


def _no_service(monkeypatch):
    get_service = Mock()
    monkeypatch.setattr(google_drive, "get_drive_service", get_service)
    return get_service


def _replacement(directory: Path, name: str = "Quarterly Deck.pptx", data=NEW_CONTENT):
    path = directory / name
    path.write_bytes(data)
    return path


def _call(file_id="deck1", file_path="", **kwargs):
    return json.loads(
        google_drive.google_drive_update_file_content(file_id, str(file_path), **kwargs)
    )


def _drive_http_error(status: int, message: str, reason: str | None):
    errors = [] if reason is None else [{"reason": reason, "message": message}]
    body = {"error": {"code": status, "message": message, "errors": errors}}
    return HttpError(
        SimpleNamespace(status=status, reason="error"),
        json.dumps(body).encode("utf-8"),
        uri="https://www.googleapis.com/upload/drive/v3/files/deck1?alt=json",
    )


# --- success paths ---------------------------------------------------------


def test_simple_update_replaces_content_in_place(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch)
    local = _replacement(allowed_dir)

    result = _call("https://drive.google.com/file/d/deck1/view", local)

    assert result["status"] == "success"
    assert result["changed"] is True
    assert "only for a limited time (about 30 days" in result["message"]
    assert result["file"]["id"] == "deck1"
    assert result["file"]["md5Checksum"] == _md5(NEW_CONTENT)
    # The model names and links the updated file from these.
    link = "https://drive.google.com/file/d/deck1/view"
    assert result["file"]["name"] == "Quarterly Deck.pptx"
    assert result["file"]["webViewLink"] == link
    assert result["file"]["modifiedTime"] == "2026-10-09T08:00:00.000Z"
    assert result["previous"]["webViewLink"] == link
    assert result["previous"]["modifiedTime"] == "2026-05-12T08:00:00.000Z"
    assert result["previous"]["headRevisionId"] == "rev-1"
    assert result["previous"]["version"] == "7"
    get_kwargs = drive.files.get.call_args.kwargs
    assert get_kwargs["fileId"] == "deck1"
    assert get_kwargs["supportsAllDrives"] is True
    for field in ("canModifyContent", "headRevisionId", "md5Checksum", "shared"):
        assert field in get_kwargs["fields"]
    assert "canEdit" not in get_kwargs["fields"]
    update_kwargs = drive.files.update.call_args.kwargs
    assert update_kwargs["fileId"] == "deck1"
    assert "body" not in update_kwargs
    assert update_kwargs["supportsAllDrives"] is True
    for param in ("addParents", "removeParents", "keepRevisionForever"):
        assert param not in update_kwargs
    media = update_kwargs["media_body"]
    assert media.resumable() is False
    assert media.mimetype() == DECK_MIME
    assert drive.uploaded == [NEW_CONTENT]
    drive.update_request.execute.assert_called_once()
    execute_kwargs = drive.update_request.execute.call_args.kwargs
    assert execute_kwargs["num_retries"] == 2
    # Requests go through the update's own http object, wrapped to note an
    # attempt that ended without a clear answer.
    assert execute_kwargs["http"]._http is drive.update_request.http
    drive.files.create.assert_not_called()


def test_update_uses_the_id_drive_returns(monkeypatch, allowed_dir):
    drive = _Drive(
        monkeypatch,
        current=_current(id="deck1canonical"),
        updated=_updated(id="deck1canonical"),
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"
    assert drive.files.update.call_args.kwargs["fileId"] == "deck1canonical"


def test_resumable_upload_above_the_simple_limit(monkeypatch, allowed_dir):
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 1024)
    data = bytes(range(256)) * 8
    drive = _Drive(monkeypatch, updated=_updated(data))

    result = _call("deck1", _replacement(allowed_dir, data=data))

    assert result["status"] == "success"
    media = drive.files.update.call_args.kwargs["media_body"]
    assert media.resumable() is True
    assert media.chunksize() == google_drive._RESUMABLE_CHUNK_BYTES
    assert google_drive._RESUMABLE_CHUNK_BYTES % (256 * 1024) == 0
    assert drive.uploaded == [data]
    assert "body" not in drive.files.update.call_args.kwargs


def test_upload_limits_keep_their_documented_values():
    # Google documents a simple upload as "5 MB or less"; 5 MiB would send up
    # to 242,880 bytes too many in one request.
    assert google_drive._SIMPLE_UPLOAD_MAX_BYTES == 5_000_000
    assert google_drive._MAX_CONTENT_UPDATE_BYTES == 2 * 1024 * 1024 * 1024
    # Each resumable chunk is read into memory, so this also bounds memory.
    assert google_drive._RESUMABLE_CHUNK_BYTES == 16 * 1024 * 1024


@pytest.mark.parametrize(("size", "resumable"), [(5_000_000, False), (5_000_001, True)])
def test_simple_upload_boundary(monkeypatch, allowed_dir, size, resumable):
    local = allowed_dir / "Quarterly Deck.pptx"
    with local.open("wb") as fh:
        fh.truncate(size)
    data = bytes(size)
    drive = _Drive(monkeypatch, updated=_updated(data))

    result = _call("deck1", local)

    assert result["status"] == "success"
    assert drive.files.update.call_args.kwargs["media_body"].resumable() is resumable
    assert drive.uploaded == [data]


def test_simple_upload_sends_the_bytes_that_were_hashed(monkeypatch, allowed_dir):
    """The checksum checked after the upload is the one computed before it,
    so a simple upload must send those same bytes, not re-read the file."""
    drive = _Drive(monkeypatch)
    local = _replacement(allowed_dir)
    service = drive.get_service.return_value

    def rewrite_then_connect():
        with local.open("r+b") as fh:
            fh.write(b"x" * len(NEW_CONTENT))
        return service

    monkeypatch.setattr(google_drive, "get_drive_service", rewrite_then_connect)

    result = _call("deck1", local)

    assert local.read_bytes() != NEW_CONTENT
    assert result["status"] == "success"
    assert drive.uploaded == [NEW_CONTENT]


def test_shared_file_success_says_everyone_sees_new_content(monkeypatch, allowed_dir):
    _Drive(
        monkeypatch,
        current=_current(shared=True),
        updated=_updated(shared=True),
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"
    assert result["previous"]["shared"] is True
    assert result["file"]["shared"] is True
    assert "everyone with access now sees the new content" in result["message"]


def test_shared_drive_file_success_says_its_members_see_the_new_content(
    monkeypatch, allowed_dir
):
    # Drive does not set "shared" for a file in a shared drive.
    _Drive(
        monkeypatch,
        current=_current(shared=None, driveId="team1"),
        updated=_updated(shared=None, driveId="team1"),
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"
    assert result["previous"]["driveId"] == "team1"
    assert result["file"]["driveId"] == "team1"
    message = result["message"]
    assert "in a shared drive" in message
    assert "including the drive's members, now sees the new content" in message


def test_unshared_file_success_does_not_mention_sharing(monkeypatch, allowed_dir):
    _Drive(monkeypatch)

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"
    assert "now sees the new content" not in result["message"]


def test_success_reports_a_changed_type(monkeypatch, allowed_dir):
    drive = _Drive(
        monkeypatch,
        current=_current(mimeType="application/octet-stream"),
        updated=_updated(mimeType=DECK_MIME),
    )

    result = _call("deck1", _replacement(allowed_dir), mime_type=DECK_MIME)

    assert result["status"] == "success"
    assert drive.files.update.call_args.kwargs["media_body"].mimetype() == DECK_MIME
    assert "type changed from 'application/octet-stream'" in result["message"]


def test_stored_octet_stream_is_sent_unchanged_without_mime_type(
    monkeypatch, allowed_dir
):
    drive = _Drive(
        monkeypatch,
        current=_current(mimeType="application/octet-stream"),
        updated=_updated(mimeType="application/octet-stream"),
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"
    media = drive.files.update.call_args.kwargs["media_body"]
    assert media.mimetype() == "application/octet-stream"


def test_share_link_resource_key_is_sent_on_read_and_update(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch)

    result = _call(
        "https://drive.google.com/file/d/deck1/view?resourcekey=0-key",
        _replacement(allowed_dir),
    )

    assert result["status"] == "success"
    assert drive.get_request.headers["X-Goog-Drive-Resource-Keys"] == "deck1/0-key"
    assert drive.update_request.headers["X-Goog-Drive-Resource-Keys"] == "deck1/0-key"


# --- unchanged and conflicts -----------------------------------------------


def test_identical_content_is_unchanged_not_success(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch)

    result = _call("deck1", _replacement(allowed_dir, data=OLD_CONTENT))

    assert result["status"] == "unchanged"
    assert result["changed"] is False
    assert result["message"].startswith("Google Drive was NOT modified by this call")
    # Without expected_head_revision_id the tool cannot tell a repeat of an
    # earlier successful call from a replacement that was never edited.
    assert "that save stands" in result["message"]
    assert "original download" in result["message"]
    assert result["file"]["headRevisionId"] == "rev-1"
    drive.assert_nothing_written()


def test_identical_to_the_downloaded_version_is_unchanged(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch)

    result = _call(
        "deck1",
        _replacement(allowed_dir, data=OLD_CONTENT),
        expected_head_revision_id="rev-1",
    )

    message = result["message"]
    assert result["status"] == "unchanged"
    assert message.startswith("Google Drive was NOT modified")
    # The expected version may be the download or this task's last save.
    assert (
        "identical to the version of 'Quarterly Deck.pptx' named by "
        "expected_head_revision_id (the version downloaded or last saved in "
        "this task)"
    ) in message
    assert "Do not report it as updated" in message
    assert "original download or a copy that was already saved" in message
    assert "that save stands" not in message
    drive.assert_nothing_written()


def test_repeat_of_a_completed_update_is_unchanged_not_a_conflict(
    monkeypatch, allowed_dir
):
    """A retry after an unclear failure, or a replay of a call that already
    succeeded, finds Drive's head revision moved by that call and exactly the
    replacement's bytes in place: nothing would be overwritten, so it must
    not be reported as someone else's change."""
    drive = _Drive(monkeypatch)
    local = _replacement(allowed_dir)

    first = _call("deck1", local, expected_head_revision_id="rev-1")
    drive.current = _current(
        size=str(len(NEW_CONTENT)),
        md5Checksum=_md5(NEW_CONTENT),
        headRevisionId="rev-2",
        version="8",
        modifiedTime="2026-10-09T08:00:00.000Z",
    )
    repeat = _call("deck1", local, expected_head_revision_id="rev-1")

    assert first["status"] == "success"
    assert repeat["status"] == "unchanged"
    assert repeat["changed"] is False
    message = repeat["message"]
    assert message.startswith("Google Drive was NOT modified by this call")
    assert "'Quarterly Deck.pptx' already has exactly this content" in message
    assert "Its head revision is now rev-2, not rev-1" in message
    assert (
        "If an earlier call in this task uploaded this replacement, that save "
        "stands: say that the file already has this content, without saying "
        "that this call saved it"
    ) in message
    assert "changed in Google Drive" not in message
    assert "upload the edited file as a new file" not in message
    assert repeat["file"]["headRevisionId"] == "rev-2"
    drive.files.update.assert_called_once()
    drive.files.create.assert_not_called()


@pytest.mark.parametrize(
    ("current_head", "expected"),
    [
        # A wrong value, here the file's version instead of its head revision.
        ("rev-1", "7"),
        # The head revision moved without a change to the content.
        ("rev-3", "rev-1"),
    ],
)
def test_identical_content_under_another_head_is_not_reported_as_saved(
    monkeypatch, allowed_dir, current_head, expected
):
    """The original download passed as file_path, with a head revision other
    than the expected one, looks the same as a repeat of an earlier save. No
    call saved anything here, so the result must not tell the model that the
    edit is already in Google Drive."""
    drive = _Drive(monkeypatch, current=_current(headRevisionId=current_head))

    result = _call(
        "deck1",
        _replacement(allowed_dir, data=OLD_CONTENT),
        expected_head_revision_id=expected,
    )

    message = result["message"]
    assert result["status"] == "unchanged"
    assert result["changed"] is False
    assert message.startswith("Google Drive was NOT modified by this call")
    assert f"Its head revision is now {current_head}, not {expected}" in message
    assert "If an earlier call in this task uploaded this replacement" in message
    assert "Otherwise do not report it as updated" in message
    assert "not the original download" in message
    assert "expected_head_revision_id is the latest headRevisionId" in message
    assert "usually means" not in message
    assert "Tell the user that the file in Google Drive already has" not in message
    drive.assert_nothing_written()


def test_identical_content_without_a_current_head_is_unchanged(
    monkeypatch, allowed_dir
):
    drive = _Drive(monkeypatch, current=_current(headRevisionId=None))

    result = _call(
        "deck1",
        _replacement(allowed_dir, data=OLD_CONTENT),
        expected_head_revision_id="rev-1",
    )

    assert result["status"] == "unchanged"
    drive.assert_nothing_written()


@pytest.mark.parametrize("expected", ["rev-0", "7"])
def test_conflicting_head_revision_is_refused(monkeypatch, allowed_dir, expected):
    drive = _Drive(monkeypatch, current=_current(headRevisionId="rev-9"))

    result = _call(
        "deck1", _replacement(allowed_dir), expected_head_revision_id=expected
    )

    message = result["message"]
    assert result["status"] == "error"
    assert message.startswith("Google Drive was not modified")
    # All the tool knows is that the values differ: the file may have changed,
    # or the value passed (here possibly a version number) is not this file's
    # headRevisionId.
    assert f"does not match expected_head_revision_id {expected}" in message
    assert "rev-9" in message
    assert "2026-05-12T08:00:00.000Z" in message
    assert "changed in Google Drive after it was downloaded" not in message
    assert (
        "may have changed in Google Drive since this task last downloaded or saved it"
    ) in message
    assert "headRevisionId that google_drive_download_file returned" in message
    assert "file.headRevisionId in that call's result" in message
    # The current head shown here may include someone else's change, so a
    # retry that copies it would overwrite that change unchecked.
    assert "Do not use the head revision shown in this message instead" in message
    assert "download the current version again" in message
    assert result["file"]["headRevisionId"] == "rev-9"
    drive.assert_nothing_written()


def test_second_save_passes_the_head_revision_the_first_save_returned(
    monkeypatch, allowed_dir
):
    """Each save moves Drive's head revision, so after this tool has saved the
    file once, a further edit is saved against the head revision in that
    result. Passing the download's value again is refused, and the refusal
    points at the earlier save's result rather than only at a change made
    in Google Drive."""
    drive = _Drive(monkeypatch)
    local = _replacement(allowed_dir)

    first = _call("deck1", local, expected_head_revision_id="rev-1")

    assert first["status"] == "success"
    saved_head = first["file"]["headRevisionId"]
    assert saved_head == "rev-2"
    drive.current = _current(
        size=str(len(NEW_CONTENT)),
        md5Checksum=_md5(NEW_CONTENT),
        headRevisionId="rev-2",
        version="8",
        modifiedTime="2026-10-09T08:00:00.000Z",
    )
    further = NEW_CONTENT + b", and one more change"
    local.write_bytes(further)
    drive.updated = _updated(
        further,
        headRevisionId="rev-3",
        version="9",
        modifiedTime="2026-10-09T09:00:00.000Z",
    )

    stale = _call("deck1", local, expected_head_revision_id="rev-1")
    second = _call("deck1", local, expected_head_revision_id=saved_head)

    stale_message = stale["message"]
    assert stale["status"] == "error"
    assert "unless a successful call of this tool in this task" in stale_message
    assert second["status"] == "success"
    assert second["previous"]["headRevisionId"] == "rev-2"
    assert second["file"]["headRevisionId"] == "rev-3"
    assert drive.uploaded == [NEW_CONTENT, further]
    drive.files.create.assert_not_called()


def test_matching_head_revision_proceeds(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch)

    result = _call(
        "deck1", _replacement(allowed_dir), expected_head_revision_id=" rev-1 "
    )

    assert result["status"] == "success"
    drive.files.update.assert_called_once()


def test_expected_head_revision_without_a_current_head_is_refused(
    monkeypatch, allowed_dir
):
    drive = _Drive(monkeypatch, current=_current(headRevisionId=None))

    result = _call(
        "deck1", _replacement(allowed_dir), expected_head_revision_id="rev-1"
    )

    assert result["status"] == "error"
    assert "could not be checked" in result["message"]
    drive.assert_nothing_written()


# --- refusals after the pre-read -------------------------------------------


@pytest.mark.parametrize(
    ("mime_type", "expected"),
    [
        ("application/vnd.google-apps.document", "Google Docs tools"),
        ("application/vnd.google-apps.spreadsheet", "Google Sheets tools"),
        ("application/vnd.google-apps.presentation", "google_slides_import_pptx"),
        ("application/vnd.google-apps.drawing", "native Google file (drawing)"),
        ("application/vnd.google-apps.form", "native Google file (form)"),
        ("application/vnd.google-apps.folder", "is a folder"),
        ("application/vnd.google-apps.shortcut", "target id target1"),
    ],
)
def test_native_google_files_are_refused(monkeypatch, allowed_dir, mime_type, expected):
    drive = _Drive(
        monkeypatch,
        current=_current(
            mimeType=mime_type,
            name="Quarterly Deck",
            shortcutDetails={"targetId": "target1", "targetMimeType": DECK_MIME},
        ),
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "error"
    assert expected in result["message"]
    assert "Nothing was changed in Google Drive" in result["message"]
    drive.assert_nothing_written()


def test_shortcut_refusal_asks_to_confirm_the_target_first(monkeypatch, allowed_dir):
    _Drive(
        monkeypatch,
        current=_current(
            mimeType="application/vnd.google-apps.shortcut",
            shortcutDetails={"targetId": "target1", "targetMimeType": DECK_MIME},
        ),
    )

    result = _call("deck1", _replacement(allowed_dir))

    message = result["message"]
    assert "target id target1" in message
    assert "confirm with the user first, naming the target file" in message
    assert "whether it is shared" in message


def test_presentation_refusal_names_the_slides_tools(monkeypatch, allowed_dir):
    _Drive(
        monkeypatch,
        current=_current(mimeType="application/vnd.google-apps.presentation"),
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert "Google Slides tools" in result["message"]


def test_trashed_file_is_refused(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch, current=_current(trashed=True))

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "error"
    assert "trash" in result["message"]
    # A refusal carries the file, so the model can name and link it.
    assert result["file"]["id"] == "deck1"
    assert result["file"]["name"] == "Quarterly Deck.pptx"
    assert result["file"]["webViewLink"] == "https://drive.google.com/file/d/deck1/view"
    drive.assert_nothing_written()


def test_file_without_edit_access_is_refused(monkeypatch, allowed_dir):
    drive = _Drive(
        monkeypatch,
        current=_current(capabilities={"canModifyContent": False}),
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "error"
    assert "cannot change the content" in result["message"]
    assert "nothing was changed" in result["message"]
    drive.assert_nothing_written()


def test_missing_capability_is_left_to_the_api(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch, current=_current(capabilities=None))

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"
    drive.files.update.assert_called_once()


def test_different_extension_is_refused(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch)

    result = _call("deck1", _replacement(allowed_dir, name="Quarterly Deck.xlsx"))

    assert result["status"] == "error"
    assert "a .xlsx file" in result["message"]
    assert "a .pptx file" in result["message"]
    drive.assert_nothing_written()


def test_extension_check_ignores_case(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch, current=_current(name="Quarterly Deck.PPTX"))

    refused = _call("deck1", _replacement(allowed_dir, name="Quarterly Deck.xlsx"))

    assert refused["status"] == "error"
    assert "a .xlsx file" in refused["message"]
    assert "a .pptx file" in refused["message"]
    drive.assert_nothing_written()

    accepted = _call("deck1", _replacement(allowed_dir, name="quarterly deck.pptx"))

    assert accepted["status"] == "success"
    drive.files.update.assert_called_once()


def test_different_mime_type_is_refused(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch)

    result = _call("deck1", _replacement(allowed_dir), mime_type="application/pdf")

    assert result["status"] == "error"
    assert "does not match the type" in result["message"]
    drive.assert_nothing_written()


@pytest.mark.parametrize("drive_name", ["Quarterly Deck", "Plan v1.2"])
def test_drive_name_without_an_extension_skips_the_extension_check(
    monkeypatch, allowed_dir, drive_name
):
    _Drive(monkeypatch, current=_current(name=drive_name))

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"


def _typed_drive(monkeypatch, name, mime_type):
    return _Drive(
        monkeypatch,
        current=_current(name=name, mimeType=mime_type),
        updated=_updated(name=name, mimeType=mime_type),
    )


@pytest.mark.parametrize(
    ("drive_name", "mime_type", "local_name"),
    [
        ("Team Photo.jpeg", "image/jpeg", "Team Photo.jpg"),
        ("Team Photo.JPG", "image/jpeg", "team photo.jpeg"),
        ("Scan.tif", "image/tiff", "Scan.tiff"),
        ("Index.htm", "text/html", "Index.html"),
        ("config.yaml", "application/x-yaml", "config.yml"),
        ("config.yml", "application/octet-stream", "config.yaml"),
    ],
)
def test_two_spellings_of_one_extension_are_the_same_type(
    monkeypatch, allowed_dir, drive_name, mime_type, local_name
):
    drive = _typed_drive(monkeypatch, drive_name, mime_type)

    result = _call("deck1", _replacement(allowed_dir, name=local_name))

    assert result["status"] == "success", result
    assert drive.uploaded == [NEW_CONTENT]


@pytest.mark.parametrize(
    ("drive_name", "mime_type", "local_name"),
    [
        # The replacement's extension names the type stored in Drive.
        ("John.Smith", "application/pdf", "John.Smith.pdf"),
        ("report.final", DECK_MIME, "report.final.pptx"),
        # Neither the dotted part nor the stored type says what the file is.
        ("report.final", "application/octet-stream", "report.final.pptx"),
    ],
)
def test_dotted_drive_name_is_not_taken_as_a_different_type(
    monkeypatch, allowed_dir, drive_name, mime_type, local_name
):
    drive = _typed_drive(monkeypatch, drive_name, mime_type)

    result = _call("deck1", _replacement(allowed_dir, name=local_name))

    assert result["status"] == "success", result
    assert drive.uploaded == [NEW_CONTENT]


@pytest.mark.parametrize(
    ("drive_name", "mime_type", "local_name", "expected"),
    [
        (
            "John.Smith",
            "application/pdf",
            "John.Smith.docx",
            "'John.Smith' in Google Drive is stored as application/pdf",
        ),
        (
            "Quarterly Deck.pptx",
            "application/octet-stream",
            "Quarterly Deck.xlsx",
            "'Quarterly Deck.pptx' in Google Drive is a .pptx file",
        ),
        (
            "Team Photo.jpg",
            "image/jpeg",
            "Team Photo.png",
            "'Team Photo.jpg' in Google Drive is a .jpg file",
        ),
    ],
)
def test_replacement_of_another_type_is_still_refused(
    monkeypatch, allowed_dir, drive_name, mime_type, local_name, expected
):
    drive = _typed_drive(monkeypatch, drive_name, mime_type)

    result = _call("deck1", _replacement(allowed_dir, name=local_name))

    assert result["status"] == "error"
    assert expected in result["message"]
    assert "Nothing was changed in Google Drive" in result["message"]
    drive.assert_nothing_written()


@pytest.mark.parametrize(
    ("drive_name", "mime_type", "local_name", "drive_kind"),
    [
        # google_drive_upload_file stores "report.csv.gz" as text/csv.
        ("report.csv.gz", "text/csv", "report.csv", "a .gz file"),
        ("backup.tgz", "application/x-gzip", "backup.tar", "a .tgz file"),
        ("logo.svgz", "image/svg+xml", "logo.svg", "a .svgz file"),
        ("logs.gz", "application/octet-stream", "logs.txt", "a .gz file"),
        ("dump.xz", "application/octet-stream", "dump.sql", "a .xz file"),
        ("backup.tar", "application/x-tar", "backup.tgz", "a .tar file"),
        ("report.csv", "text/csv", "report.csv.gz", "a .csv file"),
        ("data.tar.gz", "application/x-tar", "data.tar", "a .gz file"),
        ("logo.svg.gz", "image/svg+xml", "logo.svg", "a .gz file"),
        ("data.tar.gz", "application/x-tar", "data.svgz", "a .gz file"),
    ],
)
def test_compressed_and_plain_files_are_different_types(
    monkeypatch, allowed_dir, drive_name, mime_type, local_name, drive_kind
):
    drive = _typed_drive(monkeypatch, drive_name, mime_type)

    result = _call("deck1", _replacement(allowed_dir, name=local_name))

    assert result["status"] == "error"
    assert f"'{drive_name}' in Google Drive is {drive_kind}" in result["message"]
    assert "Nothing was changed in Google Drive" in result["message"]
    drive.assert_nothing_written()


@pytest.mark.parametrize(
    ("drive_name", "mime_type", "local_name"),
    [
        ("data.tar.gz", "application/x-tar", "data.tgz"),
        ("data.tgz", "application/x-tar", "data.tar.gz"),
        ("data.tar.bz2", "application/x-tar", "data.tbz2"),
        ("data.tbz2", "application/x-bzip2", "data.tar.bz2"),
        ("data.tar.xz", "application/x-xz", "data.txz"),
        ("logo.svg.gz", "image/svg+xml", "logo.svgz"),
        ("logo.svgz", "image/svg+xml", "logo.svg.gz"),
        ("Data.TAR.GZ", "application/gzip", "data.tgz"),
    ],
)
def test_two_spellings_of_one_compressed_type_are_the_same_type(
    monkeypatch, allowed_dir, drive_name, mime_type, local_name
):
    drive = _typed_drive(monkeypatch, drive_name, mime_type)

    result = _call("deck1", _replacement(allowed_dir, name=local_name))

    assert result["status"] == "success", result
    assert drive.uploaded == [NEW_CONTENT]


@pytest.mark.parametrize(
    ("drive_name", "mime_type", "local_name"),
    [
        # Extensions missing from Python's table, stored under their own type.
        ("Keynote.key", "application/x-iwork-keynote-sffkey", "Keynote.key"),
        ("Site Plan.dwg", "image/vnd.dwg", "site plan v2.DWG"),
        ("logs.gz", "application/gzip", "logs.gz"),
    ],
)
def test_same_extension_is_accepted_whatever_the_stored_type(
    monkeypatch, allowed_dir, drive_name, mime_type, local_name
):
    drive = _typed_drive(monkeypatch, drive_name, mime_type)

    result = _call("deck1", _replacement(allowed_dir, name=local_name))

    assert result["status"] == "success", result
    assert drive.uploaded == [NEW_CONTENT]


def test_extension_types_come_from_python_not_the_host(monkeypatch):
    # mimetypes.guess_type also reads the host's mime.types files, which
    # differ between machines; the check must not depend on them.
    monkeypatch.setattr(
        google_drive.mimetypes,
        "guess_type",
        lambda *args, **kwargs: ("application/x-host-type", None),
    )

    assert google_drive._extension_type(".smith") is None
    assert google_drive._extension_type(".jpeg") == ("image/jpeg", None)
    assert google_drive._extension_type(".jpg") == ("image/jpeg", None)
    assert google_drive._extension_type(".pptx") == (DECK_MIME, None)
    assert google_drive._extension_type(".svgz") == ("image/svg+xml", "gzip")
    assert google_drive._extension_type(".gz") == (None, "gzip")
    tar_gz = ("application/x-tar", "gzip")
    assert google_drive._name_type("data.tar.gz", ".gz") == tar_gz
    assert google_drive._name_type("Data.TAR.GZ", ".gz") == tar_gz
    assert google_drive._name_type("data.tgz", ".tgz") == tar_gz
    assert google_drive._name_type("logs.gz", ".gz") == (None, "gzip")
    assert google_drive._name_type("data.tar", ".tar") == ("application/x-tar", None)


@pytest.mark.parametrize(
    ("status", "api_message", "reason"),
    [
        (404, "File not found: deck1.", "notFound"),
        (
            403,
            "The user has not granted the app 123456 read access to the file deck1.",
            "appNotAuthorizedToFile",
        ),
    ],
)
def test_file_this_connection_cannot_see_is_explained(
    monkeypatch, allowed_dir, status, api_message, reason
):
    drive = _Drive(monkeypatch)
    drive.get_request.execute.side_effect = _drive_http_error(
        status, api_message, reason
    )

    result = _call("deck1", _replacement(allowed_dir))

    message = result["message"]
    assert result["status"] == "error"
    assert "nothing was changed" in message
    assert "per-file Drive access" in message
    assert "file picker" in message
    assert "Upload new version" in message
    assert "google_drive_upload_file" in message
    # The open-by-link hint for Docs/Sheets/Slides does not help with a stored file.
    assert "open it with the Google Docs" not in message
    assert f"HTTP {status}" in message
    drive.assert_nothing_written()


# --- failures of the update request ----------------------------------------


@pytest.mark.parametrize(
    ("status", "reason", "expected"),
    [
        (403, "insufficientFilePermissions", "cannot change the content"),
        (403, None, "cannot change the content"),
        (404, "notFound", "per-file Drive access"),
        (403, "storageQuotaExceeded", "(reason: storageQuotaExceeded)"),
        (403, "userRateLimitExceeded", "(reason: userRateLimitExceeded)"),
    ],
)
def test_update_errors_are_mapped(monkeypatch, allowed_dir, status, reason, expected):
    drive = _Drive(monkeypatch)
    drive.update_request.execute.side_effect = _drive_http_error(
        status, "Drive refused the update", reason
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "error"
    assert expected in result["message"]
    assert "www.googleapis.com" not in result["message"]
    # A 4xx reply is a definite rejection once the file, read again, is
    # still as it was read before the update.
    assert "not known whether" not in result["message"]
    assert result["file"]["headRevisionId"] == "rev-1"
    assert drive.files.get.call_count == 2
    drive.files.create.assert_not_called()


def test_rejected_update_says_drive_did_not_accept_it(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch)
    drive.update_request.execute.side_effect = _drive_http_error(
        400, "Bad Request", "badRequest"
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "error"
    assert "did not accept the new content" in result["message"]
    assert "Do not report the file as updated" in result["message"]
    assert "(reason: badRequest)" in result["message"]
    assert result["file"]["headRevisionId"] == "rev-1"


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("reset"),
        TimeoutError("timed out"),
        TimeoutError(),
        _drive_http_error(503, "Backend Error", "backendError"),
        _drive_http_error(408, "Request Timeout", None),
    ],
)
def test_update_with_an_unknown_outcome_does_not_claim_either_way(
    monkeypatch, allowed_dir, error
):
    """A timeout, a dropped connection or a 5xx left after retries may come
    after Drive stored the upload, so the result must neither say that Drive
    refused the content nor present the pre-read state as the file's state."""
    drive = _Drive(monkeypatch)
    drive.update_request.execute.side_effect = error

    result = _call("deck1", _replacement(allowed_dir))

    message = result["message"]
    assert result["status"] == "error"
    assert "not known whether 'Quarterly Deck.pptx' changed" in message
    assert "Do not report it as updated or as unchanged" in message
    assert "version history" in message
    assert "did not accept" not in message
    assert "nothing was changed" not in message.lower()
    assert "www.googleapis.com" not in message
    assert "file" not in result
    assert result["previous"]["headRevisionId"] == "rev-1"
    drive.files.update.assert_called_once()
    drive.files.create.assert_not_called()


def test_failure_after_the_update_returned_does_not_claim_nothing_changed(
    monkeypatch, allowed_dir
):
    drive = _Drive(monkeypatch)

    def broken_check(*args, **kwargs):
        raise TypeError("can't compare offset-naive and offset-aware datetimes")

    monkeypatch.setattr(google_drive, "_content_update_mismatch", broken_check)

    result = _call("deck1", _replacement(allowed_dir))

    message = result["message"]
    assert result["status"] == "error"
    assert "nothing was changed" not in message.lower()
    assert "not known whether the file changed" in message
    assert "Do not report it as updated" in message
    assert "version history" in message
    drive.files.update.assert_called_once()
    drive.files.create.assert_not_called()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            _drive_http_error(500, "Backend Error", "backendError"),
            "HTTP 500 Backend Error (reason: backendError)",
        ),
        (
            _drive_http_error(403, "Insufficient scopes", "insufficientPermissions"),
            "HTTP 403 Insufficient scopes (reason: insufficientPermissions)",
        ),
        (
            _drive_http_error(403, "Rate limit", "rateLimitExceeded"),
            "HTTP 403 Rate limit (reason: rateLimitExceeded)",
        ),
        (TimeoutError("timed out"), "Error: timed out"),
    ],
)
def test_failed_pre_read_changes_nothing_and_keeps_the_uri_out(
    monkeypatch, allowed_dir, error, expected
):
    drive = _Drive(monkeypatch)
    drive.get_request.execute.side_effect = error

    result = _call("deck1", _replacement(allowed_dir))

    message = result["message"]
    assert result["status"] == "error"
    assert "Could not read the Google Drive file" in message
    assert "Nothing was changed in Google Drive" in message
    assert expected in message
    assert "www.googleapis.com" not in message
    drive.assert_nothing_written()


# --- post-upload verification ----------------------------------------------


@pytest.mark.parametrize(
    ("updated", "expected"),
    [
        (_updated(md5Checksum=_md5(b"something else")), "checksum does not match"),
        (_updated(headRevisionId="rev-1"), "head revision did not change"),
        (_updated(version="7"), "version did not advance"),
        (_updated(id="other"), "different file id"),
        # The checksum is the evidence; size and version do not replace it.
        (_updated(md5Checksum=None), "returned no checksum to check"),
    ],
)
def test_unverified_update_is_not_reported_as_success(
    monkeypatch, allowed_dir, updated, expected
):
    drive = _Drive(monkeypatch, updated=updated)

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "error"
    assert expected in result["message"]
    assert "Do not report this as done" in result["message"]
    assert "version history" in result["message"]
    drive.files.update.assert_called_once()
    drive.files.create.assert_not_called()


def test_checksum_confirms_the_update_when_modified_time_moved_back(
    monkeypatch, allowed_dir
):
    """A client can store a modifiedTime ahead of Google's clock; a content
    update then sets it to the current time, which is earlier."""
    _Drive(monkeypatch, current=_current(modifiedTime="2027-01-01T00:00:00.000Z"))

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"


def test_missing_head_revision_falls_back_to_version(monkeypatch, allowed_dir):
    _Drive(monkeypatch, updated=_updated(headRevisionId=None))

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"


def test_head_revision_missing_on_both_sides_falls_back_to_version(
    monkeypatch, allowed_dir
):
    _Drive(
        monkeypatch,
        current=_current(headRevisionId=None),
        updated=_updated(headRevisionId=None),
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"


def test_update_without_any_sign_of_change_is_not_success(monkeypatch, allowed_dir):
    # The checksum matches the upload, but with no earlier checksum, head
    # revision or version to compare, nothing shows that this call changed it.
    drive = _Drive(
        monkeypatch,
        current=_current(md5Checksum=None),
        updated=_updated(headRevisionId=None, version=None),
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "error"
    assert "shows no sign that the content changed" in result["message"]
    assert "Do not report this as done" in result["message"]
    drive.files.update.assert_called_once()


def test_checksum_change_alone_confirms_the_update(monkeypatch, allowed_dir):
    _Drive(monkeypatch, updated=_updated(headRevisionId=None, version=None))

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"


# --- local checks before any request ---------------------------------------


def _assert_local_refusal(result, get_service, *host_paths):
    message = result["message"]
    assert result["status"] == "error"
    assert "Nothing was changed in Google Drive" in message
    # The reason ends its own sentence, even when the text it comes from
    # (such as the shared file id check) has no full stop.
    assert re.search(r"[^.!?] Nothing was changed in Google Drive", message) is None
    assert ".. Nothing was changed" not in message
    for path in host_paths:
        assert str(path) not in message
    get_service.assert_not_called()


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("file_id must look like a Drive id", "file_id must look like a Drive id."),
        ("Finish writing it first.", "Finish writing it first."),
        ("Is it there?", "Is it there?"),
    ],
)
def test_reason_and_nothing_changed_are_separate_sentences(reason, expected):
    message = google_drive._then_nothing_changed(ValueError(reason))

    assert message == f"{expected} Nothing was changed in Google Drive."


def test_empty_reason_leaves_only_nothing_changed():
    assert google_drive._then_nothing_changed(RuntimeError()) == (
        "Nothing was changed in Google Drive."
    )


def test_invalid_pre_read_reply_changes_nothing(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch, current=["not", "a", "file"])

    result = _call("deck1", _replacement(allowed_dir))

    assert result == {
        "status": "error",
        "message": "Drive returned an invalid file response. Nothing was "
        "changed in Google Drive.",
    }
    drive.assert_nothing_written()


def test_path_outside_allowed_directories_fails_before_any_request(
    monkeypatch, tmp_path, allowed_dir
):
    get_service = _no_service(monkeypatch)
    outside = _replacement(tmp_path)

    result = _call("deck1", outside)

    _assert_local_refusal(result, get_service, allowed_dir)
    assert "outside the allowed directories" in result["message"]
    assert 'file_path="file:<id>"' in result["message"]
    assert "do not rebuild it" in result["message"]


def test_missing_file_fails_before_any_request(monkeypatch, allowed_dir):
    get_service = _no_service(monkeypatch)

    result = _call("deck1", allowed_dir / "missing.pptx")

    _assert_local_refusal(result, get_service)
    assert "File not found" in result["message"]
    assert 'file_path="file:<id>"' in result["message"]


def test_symlink_escaping_the_workspace_fails_before_any_request(
    monkeypatch, tmp_path, allowed_dir
):
    get_service = _no_service(monkeypatch)
    outside = _replacement(tmp_path, name="secret.pptx")
    link = allowed_dir / "Quarterly Deck.pptx"
    link.symlink_to(outside)

    result = _call("deck1", link)

    _assert_local_refusal(result, get_service, outside, allowed_dir)
    assert "outside the allowed directories" in result["message"]


def test_relative_path_fails_before_any_request(monkeypatch, allowed_dir):
    get_service = _no_service(monkeypatch)
    _replacement(allowed_dir)

    result = _call("deck1", "Quarterly Deck.pptx")

    _assert_local_refusal(result, get_service, allowed_dir)


def test_empty_file_fails_before_any_request(monkeypatch, allowed_dir):
    get_service = _no_service(monkeypatch)

    result = _call("deck1", _replacement(allowed_dir, data=b""))

    _assert_local_refusal(result, get_service)
    assert "File is empty" in result["message"]


def test_oversize_file_fails_before_any_request(monkeypatch, allowed_dir):
    monkeypatch.setattr(google_drive, "_MAX_CONTENT_UPDATE_BYTES", 4)
    get_service = _no_service(monkeypatch)

    result = _call("deck1", _replacement(allowed_dir, data=b"12345"))

    _assert_local_refusal(result, get_service)
    assert "not a Google Drive limit" in result["message"]


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores file mode bits, so chmod 000 does not block the read",
)
def test_open_failure_does_not_leak_the_host_path(monkeypatch, allowed_dir):
    get_service = _no_service(monkeypatch)
    local = _replacement(allowed_dir)
    local.chmod(0o000)
    try:
        result = _call("deck1", local)
    finally:
        local.chmod(0o644)

    _assert_local_refusal(result, get_service, local, allowed_dir)
    assert "Could not read the file" in result["message"]


@pytest.mark.parametrize("size_delta", [5, -5])
def test_file_changing_while_read_fails_before_any_request(
    monkeypatch, allowed_dir, size_delta
):
    get_service = _no_service(monkeypatch)
    local = _replacement(allowed_dir)
    real_fstat = google_drive.os.fstat

    with monkeypatch.context() as patch:
        patch.setattr(
            google_drive.os,
            "fstat",
            lambda fd: SimpleNamespace(st_size=real_fstat(fd).st_size + size_delta),
        )
        result = _call("deck1", local)

    _assert_local_refusal(result, get_service)
    assert "changed while it was being read" in result["message"]


def test_hash_reads_exactly_the_stated_size(monkeypatch):
    monkeypatch.setattr(google_drive, "_HASH_BLOCK_BYTES", 4)
    data = b"0123456789"

    digest, kept, blocks = google_drive._hash_replacement_file(
        io.BytesIO(data), len(data), keep_bytes=True
    )

    assert digest == _md5(data)
    assert kept == data
    assert blocks == []
    digest, kept, blocks = google_drive._hash_replacement_file(
        io.BytesIO(data), len(data), keep_bytes=False
    )
    assert digest == _md5(data)
    assert kept is None
    assert blocks == [
        hashlib.sha256(part).digest() for part in (b"0123", b"4567", b"89")
    ]
    for size in (len(data) + 1, len(data) - 1):
        with pytest.raises(ValueError, match="changed while it was being read"):
            google_drive._hash_replacement_file(
                io.BytesIO(data), size, keep_bytes=False
            )


@pytest.mark.parametrize(
    "file_id",
    [
        "3f2b1c9e-4d5a-4b6c-8d7e-9f0a1b2c3d4e",
        "file:3f2b1c9e-4d5a-4b6c-8d7e-9f0a1b2c3d4e",
        "FILE:abc",
    ],
)
def test_workspace_file_id_in_file_id_fails_before_any_request(
    monkeypatch, allowed_dir, file_id
):
    get_service = _no_service(monkeypatch)

    result = _call(file_id, _replacement(allowed_dir))

    _assert_local_refusal(result, get_service)
    assert "looks like a workspace file id" in result["message"]
    assert 'file_path="file:<id>"' in result["message"]


@pytest.mark.parametrize("file_id", ["..", "", "   ", 123])
def test_invalid_file_id_fails_before_any_request(monkeypatch, allowed_dir, file_id):
    get_service = _no_service(monkeypatch)

    result = json.loads(
        google_drive.google_drive_update_file_content(
            file_id, str(_replacement(allowed_dir))
        )
    )

    _assert_local_refusal(result, get_service)


def test_untrusted_link_reaches_the_api_verbatim(monkeypatch, allowed_dir):
    drive = _Drive(monkeypatch)
    drive.get_request.execute.side_effect = _drive_http_error(
        404, "File not found.", "notFound"
    )
    link = "https://evil.example/file/d/deck1/view?resourcekey=0-key"

    result = _call(link, _replacement(allowed_dir))

    assert result["status"] == "error"
    assert drive.files.get.call_args.kwargs["fileId"] == link
    assert "X-Goog-Drive-Resource-Keys" not in drive.get_request.headers
    drive.assert_nothing_written()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"mime_type": "application/vnd.google-apps.document"}, "native Google type"),
        ({"mime_type": 3}, "mime_type must be a string"),
        ({"expected_head_revision_id": "rev\r\nX: y"}, "line breaks"),
    ],
)
def test_invalid_options_fail_before_any_request(
    monkeypatch, allowed_dir, kwargs, expected
):
    get_service = _no_service(monkeypatch)

    result = _call("deck1", _replacement(allowed_dir), **kwargs)

    _assert_local_refusal(result, get_service)
    assert expected in result["message"]


# --- request shapes on the wire --------------------------------------------


class _RecordingHttp(HttpMockSequence):
    """HttpMockSequence that also keeps each request body's bytes, reading a
    streamed chunk at the moment it is sent, and that, like Drive, answers
    a request with a `fields` parameter with only those fields."""

    def request(self, uri, method="GET", body=None, headers=None, **kwargs):
        if hasattr(body, "read"):
            body = body.read()
        if isinstance(self._iterable[0], BaseException):
            # A transport error, such as a timeout, instead of a reply.
            self.request_sequence.append((uri, method, body, headers))
            raise self._iterable.pop(0)
        response, content = super().request(
            uri, method=method, body=body, headers=headers, **kwargs
        )
        fields = parse_qs(urlparse(uri).query).get("fields")
        if fields and content and response.status == 200:
            payload = json.loads(content)
            content = json.dumps(_only_requested_fields(payload, fields[0])).encode()
        return response, content

    @property
    def requests(self):
        return [
            SimpleNamespace(
                uri=uri,
                method=method,
                body=body,
                headers={key.lower(): value for key, value in (headers or {}).items()},
            )
            for uri, method, body, headers in self.request_sequence
        ]


def _wire_drive(monkeypatch, responses):
    http = _RecordingHttp(responses)
    service = build("drive", "v3", http=http, static_discovery=True)
    monkeypatch.setattr(google_drive, "get_drive_service", lambda: service)
    return http


def _ok(payload):
    return ({"status": "200"}, json.dumps(payload))


def test_wire_simple_update_sends_only_the_media(monkeypatch, allowed_dir):
    http = _wire_drive(monkeypatch, [_ok(_current()), _ok(_updated())])

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"
    get, update = http.requests
    assert get.method == "GET"
    assert get.uri.startswith("https://www.googleapis.com/drive/v3/files/deck1?")
    assert update.method == "PATCH"
    assert update.uri.startswith(
        "https://www.googleapis.com/upload/drive/v3/files/deck1?"
    )
    assert "uploadType=media" in update.uri
    assert "supportsAllDrives=true" in update.uri
    assert "multipart" not in update.uri
    assert update.body == NEW_CONTENT
    assert update.headers["content-type"] == DECK_MIME


def test_wire_resumable_update_with_resource_key(monkeypatch, allowed_dir):
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 1000)
    monkeypatch.setattr(google_drive, "_RESUMABLE_CHUNK_BYTES", 256 * 1024)
    data = (b"slide-bytes-" * 30000)[: 300 * 1024]
    session = "https://www.googleapis.com/upload/drive/v3/files/deck1?upload_id=s1"
    http = _wire_drive(
        monkeypatch,
        [
            _ok(_current()),
            ({"status": "200", "location": session}, ""),
            ({"status": "308", "range": "bytes=0-262143"}, ""),
            _ok(_updated(data)),
        ],
    )

    result = _call(
        "https://drive.google.com/file/d/deck1/view?resourcekey=0-key",
        _replacement(allowed_dir, data=data),
    )

    assert result["status"] == "success"
    get, start, first, second = http.requests
    assert get.headers["x-goog-drive-resource-keys"] == "deck1/0-key"
    assert start.method == "PATCH"
    assert "uploadType=resumable" in start.uri
    assert "supportsAllDrives=true" in start.uri
    assert start.headers["x-goog-drive-resource-keys"] == "deck1/0-key"
    assert start.headers["x-upload-content-type"] == DECK_MIME
    assert start.headers["x-upload-content-length"] == str(len(data))
    assert not start.body
    assert [first.method, second.method] == ["PUT", "PUT"]
    assert first.uri == second.uri == session
    assert first.headers["content-range"] == f"bytes 0-262143/{len(data)}"
    assert (
        second.headers["content-range"] == f"bytes 262144-{len(data) - 1}/{len(data)}"
    )
    assert first.body + second.body == data
    assert all("multipart" not in request.uri for request in http.requests)
    assert all(request.method != "POST" for request in http.requests)


def test_wire_update_retries_a_server_error(monkeypatch, allowed_dir):
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    http = _wire_drive(
        monkeypatch,
        [
            _ok(_current()),
            ({"status": "503"}, json.dumps({"error": {"code": 503}})),
            _ok(_updated()),
        ],
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "success"
    _, failed, retried = http.requests
    assert failed.method == retried.method == "PATCH"
    assert failed.body == retried.body == NEW_CONTENT


@pytest.mark.parametrize(
    "failure",
    [
        ({"status": "503"}, json.dumps({"error": {"code": 503}})),
        (
            {"status": "403"},
            json.dumps(
                {
                    "error": {
                        "code": 403,
                        "errors": [{"reason": "rateLimitExceeded"}],
                    }
                }
            ),
        ),
    ],
)
def test_wire_resumable_update_resends_a_retried_chunk_intact(
    monkeypatch, allowed_dir, failure
):
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 1000)
    monkeypatch.setattr(google_drive, "_RESUMABLE_CHUNK_BYTES", 256 * 1024)
    data = (b"slide-bytes-" * 30000)[: 300 * 1024]
    session = "https://www.googleapis.com/upload/drive/v3/files/deck1?upload_id=s1"
    http = _wire_drive(
        monkeypatch,
        [
            _ok(_current()),
            ({"status": "200", "location": session}, ""),
            failure,
            ({"status": "308", "range": "bytes=0-262143"}, ""),
            _ok(_updated(data)),
        ],
    )

    result = _call("deck1", _replacement(allowed_dir, data=data))

    assert result["status"] == "success"
    _, _, failed, retried, last = http.requests
    first_range = f"bytes 0-262143/{len(data)}"
    assert failed.headers["content-range"] == first_range
    assert retried.headers["content-range"] == first_range
    # A retried chunk carries the same bytes, not an already-read stream.
    assert failed.body == retried.body == data[: 256 * 1024]
    assert int(retried.headers["content-length"]) == len(retried.body)
    assert last.body == data[256 * 1024 :]


def test_wire_update_does_not_retry_a_client_error(monkeypatch, allowed_dir):
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    bad_request = {
        "error": {
            "code": 400,
            "message": "Bad Request",
            "errors": [{"reason": "badRequest", "message": "Bad Request"}],
        }
    }
    http = _wire_drive(
        monkeypatch,
        [
            _ok(_current()),
            ({"status": "400"}, json.dumps(bad_request)),
            _ok(_current()),
        ],
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "error"
    assert "did not accept the new content" in result["message"]
    assert "(reason: badRequest)" in result["message"]
    # The 400 is not retried; the file is read again to check it is as it was.
    assert [request.method for request in http.requests] == ["GET", "PATCH", "GET"]


def _error_reply(status, reason):
    body = {
        "error": {"code": status, "message": reason, "errors": [{"reason": reason}]}
    }
    return ({"status": str(status)}, json.dumps(body))


_RATE_LIMITED = _error_reply(429, "rateLimitExceeded")
_RATE_LIMITED_403 = _error_reply(403, "userRateLimitExceeded")


@pytest.mark.parametrize(
    "replies",
    [
        # Drive may have stored the content on the first attempt; the
        # error that is finally raised is about a retry.
        [({"status": "503"}, ""), _RATE_LIMITED, _RATE_LIMITED],
        [({"status": "500"}, ""), _RATE_LIMITED_403, _RATE_LIMITED_403],
        [({"status": "502"}, ""), _error_reply(400, "badRequest")],
        [TimeoutError("timed out"), _RATE_LIMITED, _RATE_LIMITED],
        [
            ConnectionResetError("reset"),
            _error_reply(403, "insufficientFilePermissions"),
        ],
        [({"status": "503"}, ""), _error_reply(404, "notFound")],
    ],
    ids=[
        "503-then-429",
        "500-then-rate-limit-403",
        "502-then-400",
        "timeout-then-429",
        "reset-then-permission-403",
        "503-then-not-found",
    ],
)
def test_wire_rejected_retry_after_an_unclear_attempt_is_an_unknown_outcome(
    monkeypatch, allowed_dir, replies
):
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    http = _wire_drive(monkeypatch, [_ok(_current()), *replies])

    result = _call("deck1", _replacement(allowed_dir))

    message = result["message"]
    assert result["status"] == "error"
    assert "refused a retry of the upload" in message
    assert "not known whether 'Quarterly Deck.pptx' changed" in message
    assert "Do not report it as updated or as unchanged" in message
    assert "did not accept" not in message
    assert "nothing was changed" not in message.lower()
    # The pre-read state is not presented as the file's current state.
    assert "file" not in result
    assert result["previous"]["headRevisionId"] == "rev-1"
    assert [request.method for request in http.requests] == ["GET"] + ["PATCH"] * len(
        replies
    )


@pytest.mark.parametrize("rate_limited", [_RATE_LIMITED, _RATE_LIMITED_403])
def test_wire_rate_limit_on_every_attempt_is_a_rejection(
    monkeypatch, allowed_dir, rate_limited
):
    """Every attempt was refused, and the file read again still has the
    content read before the update, so that is the file's state."""
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    reread = _current(modifiedTime="2026-05-12T09:00:00.000Z")
    http = _wire_drive(
        monkeypatch, [_ok(_current())] + [rate_limited] * 3 + [_ok(reread)]
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert result["status"] == "error"
    assert "did not accept the new content" in result["message"]
    assert "not known whether" not in result["message"]
    assert result["file"]["headRevisionId"] == "rev-1"
    assert result["file"]["modifiedTime"] == reread["modifiedTime"]
    methods = [request.method for request in http.requests]
    assert methods == ["GET"] + ["PATCH"] * 3 + ["GET"]
    reread_request = http.requests[-1]
    assert "md5Checksum" in parse_qs(urlparse(reread_request.uri).query)["fields"][0]


def test_wire_rejected_update_reads_the_file_again_with_its_resource_key(
    monkeypatch, allowed_dir
):
    http = _wire_drive(
        monkeypatch,
        [_ok(_current()), _error_reply(400, "badRequest"), _ok(_current())],
    )

    result = _call(
        "https://drive.google.com/file/d/deck1/view?resourcekey=0-key",
        _replacement(allowed_dir),
    )

    assert "did not accept the new content" in result["message"]
    pre_read, update, reread = http.requests
    assert reread.method == "GET"
    assert reread.uri.startswith("https://www.googleapis.com/drive/v3/files/deck1?")
    assert "supportsAllDrives=true" in reread.uri
    for request in (pre_read, update, reread):
        assert request.headers["x-goog-drive-resource-keys"] == "deck1/0-key"


_NO_HEAD = {"headRevisionId": None}


@pytest.mark.parametrize(
    ("update_replies", "reread_replies", "without_head"),
    [
        # httplib2 sent the request again on its own after the connection
        # closed with no reply, so neither googleapiclient nor the tool saw
        # the attempt that stored the content; every attempt they saw got 429.
        ([_RATE_LIMITED] * 3, [_ok(_updated())], False),
        ([_error_reply(400, "badRequest")], [_ok(_updated())], False),
        # The same, for a file Drive returns no head revision for.
        ([_RATE_LIMITED] * 3, [_ok(_updated(**_NO_HEAD))], True),
        # Someone else changed the file meanwhile.
        (
            [_error_reply(403, "insufficientFilePermissions")],
            [_ok(_current(md5Checksum=_md5(b"other"), headRevisionId="rev-9"))],
            False,
        ),
        # A new revision with the old content, which may sit on top of one
        # with the new content.
        ([_RATE_LIMITED] * 3, [_ok(_current(headRevisionId="rev-3"))], False),
        # The file cannot be read again. A file that is gone or out of
        # reach on the re-read is covered by
        # test_wire_rejection_with_the_file_out_of_reach_on_the_re_read.
        ([_RATE_LIMITED] * 3, [_RATE_LIMITED] * 3, False),
    ],
    ids=[
        "429-after-a-hidden-resend-stored-it",
        "400-after-a-hidden-resend-stored-it",
        "stored-without-head-revisions",
        "changed-by-someone-else",
        "new-revision-with-the-old-content",
        "re-read-rate-limited",
    ],
)
def test_wire_rejection_the_file_read_again_does_not_settle_is_unknown(
    monkeypatch, allowed_dir, update_replies, reread_replies, without_head
):
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    current = _current(**_NO_HEAD) if without_head else _current()
    http = _wire_drive(monkeypatch, [_ok(current), *update_replies, *reread_replies])

    result = _call("deck1", _replacement(allowed_dir))

    message = result["message"]
    assert result["status"] == "error"
    assert "reading the file again did not confirm that it is unchanged" in message
    assert "not known whether 'Quarterly Deck.pptx' changed" in message
    assert "Do not report it as updated or as unchanged" in message
    assert "did not accept" not in message
    assert "nothing was changed" not in message.lower()
    assert "file" not in result
    assert result["previous"]["md5Checksum"] == _md5(OLD_CONTENT)
    assert result["previous"].get("headRevisionId") == current.get("headRevisionId")
    assert [request.method for request in http.requests] == (
        ["GET"] + ["PATCH"] * len(update_replies) + ["GET"] * len(reread_replies)
    )


_PER_FILE_ACCESS = "per-file Drive access"
_NO_LONGER_READABLE = "refused to let the connected Google account read"


@pytest.mark.parametrize(
    ("update_replies", "reread_reply", "expected"),
    [
        (
            [_error_reply(404, "notFound")],
            _error_reply(404, "notFound"),
            _PER_FILE_ACCESS,
        ),
        (
            [_error_reply(400, "badRequest")],
            _error_reply(404, "notFound"),
            _PER_FILE_ACCESS,
        ),
        (
            [_error_reply(403, "appNotAuthorizedToFile")],
            _error_reply(403, "appNotAuthorizedToFile"),
            _PER_FILE_ACCESS,
        ),
        (
            [_RATE_LIMITED] * 3,
            _error_reply(403, "appNotAuthorizedToFile"),
            _PER_FILE_ACCESS,
        ),
        (
            [_error_reply(403, "insufficientFilePermissions")],
            _error_reply(403, "insufficientFilePermissions"),
            _NO_LONGER_READABLE,
        ),
    ],
    ids=[
        "404-then-not-found",
        "400-then-not-found",
        "app-not-authorized-then-app-not-authorized",
        "429-then-app-not-authorized",
        "permission-403-then-permission-403",
    ],
)
def test_wire_rejection_with_the_file_out_of_reach_on_the_re_read(
    monkeypatch, allowed_dir, update_replies, reread_reply, expected
):
    """The file was deleted, or this connection lost access to it, after
    the pre-read, so reading it again cannot show whether it changed. The
    result gives the guidance for a file out of reach, without claiming
    that nothing changed or sending the user to a version history they
    cannot open."""
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    http = _wire_drive(monkeypatch, [_ok(_current()), *update_replies, reread_reply])

    result = _call("deck1", _replacement(allowed_dir))

    message = result["message"]
    assert result["status"] == "error"
    assert expected in message
    assert "not known whether 'Quarterly Deck.pptx' changed" in message
    assert "Do not report it as updated or as unchanged" in message
    assert "google_drive_upload_file" in message
    reread_status = reread_reply[0]["status"]
    update_status = update_replies[-1][0]["status"]
    assert f"HTTP {reread_status} " in message
    to_update = "Google API response to the update"
    assert f"(when reading it again). {to_update}: HTTP {update_status} " in message
    assert "nothing was changed" not in message.lower()
    assert "check the file's version history" not in message
    assert "did not accept" not in message
    assert "www.googleapis.com" not in message
    assert "file" not in result
    assert result["previous"]["headRevisionId"] == "rev-1"
    assert [request.method for request in http.requests] == (
        ["GET"] + ["PATCH"] * len(update_replies) + ["GET"]
    )


def test_wire_rejection_without_head_revisions_is_checked_by_checksum(
    monkeypatch, allowed_dir
):
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    current = _current(**_NO_HEAD)
    _wire_drive(
        monkeypatch, [_ok(current), _error_reply(400, "badRequest"), _ok(current)]
    )

    result = _call("deck1", _replacement(allowed_dir))

    assert "did not accept the new content" in result["message"]
    assert result["file"]["md5Checksum"] == _md5(OLD_CONTENT)


_SESSION = (
    "https://www.googleapis.com/upload/drive/v3/files/deck1"
    "?uploadType=resumable&upload_id=s1"
)
_SESSION_STARTED = ({"status": "200", "location": _SESSION}, "")


_NOT_ACCEPTED = "did not accept the new content"
_NOT_OPENED = "could not open file_id"


@pytest.mark.parametrize(
    ("replies", "methods", "expected"),
    [
        (
            [
                ({"status": "503"}, ""),
                _SESSION_STARTED,
                _error_reply(400, "badRequest"),
            ],
            ["PATCH", "PATCH", "PUT"],
            _NOT_ACCEPTED,
        ),
        (
            [
                _SESSION_STARTED,
                ({"status": "503"}, ""),
                _error_reply(400, "badRequest"),
            ],
            ["PATCH", "PUT", "PUT"],
            _NOT_ACCEPTED,
        ),
        (
            [_SESSION_STARTED, ({"status": "500"}, ""), _error_reply(404, "notFound")],
            ["PATCH", "PUT", "PUT"],
            _NOT_OPENED,
        ),
    ],
    ids=["session-start-503", "first-chunk-503", "first-chunk-500-then-404"],
)
def test_wire_resumable_server_error_before_the_last_chunk_stores_nothing(
    monkeypatch, allowed_dir, replies, methods, expected
):
    """Drive stores a resumable upload only when its last chunk arrives, so
    a server error on the request that starts the session or on an earlier
    chunk leaves a later rejection a definite one."""
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 1000)
    monkeypatch.setattr(google_drive, "_RESUMABLE_CHUNK_BYTES", 256 * 1024)
    data = (b"slide-bytes-" * 30000)[: 300 * 1024]
    http = _wire_drive(monkeypatch, [_ok(_current()), *replies, _ok(_current())])

    result = _call("deck1", _replacement(allowed_dir, data=data))

    assert result["status"] == "error"
    assert expected in result["message"]
    assert "not known whether" not in result["message"]
    assert result["file"]["headRevisionId"] == "rev-1"
    assert [request.method for request in http.requests] == ["GET", *methods, "GET"]
    # The last chunk was never sent.
    assert not any(
        request.headers.get("content-range", "").startswith("bytes 262144-")
        for request in http.requests
    )


def test_wire_resumable_last_chunk_rejected_after_a_server_error_is_unknown(
    monkeypatch, allowed_dir
):
    monkeypatch.setattr(googleapiclient.http.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 1000)
    monkeypatch.setattr(google_drive, "_RESUMABLE_CHUNK_BYTES", 256 * 1024)
    data = (b"slide-bytes-" * 30000)[: 300 * 1024]
    http = _wire_drive(
        monkeypatch,
        [
            _ok(_current()),
            _SESSION_STARTED,
            ({"status": "308", "range": "bytes=0-262143"}, ""),
            # The last chunk may have completed the upload before the 503.
            ({"status": "503"}, ""),
            _error_reply(400, "badRequest"),
        ],
    )

    result = _call("deck1", _replacement(allowed_dir, data=data))

    message = result["message"]
    assert result["status"] == "error"
    assert "refused a retry of the upload" in message
    assert "not known whether 'Quarterly Deck.pptx' changed" in message
    assert "file" not in result
    assert result["previous"]["headRevisionId"] == "rev-1"
    last_range = f"bytes 262144-{len(data) - 1}/{len(data)}"
    assert [request.headers.get("content-range") for request in http.requests] == [
        None,
        None,
        f"bytes 0-262143/{len(data)}",
        last_range,
        last_range,
    ]


def _change_after_hashing(monkeypatch, local, change):
    """Run change(fh) on the replacement once it was hashed, when the tool
    asks for the Drive service."""
    service = google_drive.get_drive_service()

    def change_then_connect():
        with local.open("r+b") as fh:
            change(fh)
        return service

    monkeypatch.setattr(google_drive, "get_drive_service", change_then_connect)


def _overwrite_at(offset):
    def change(fh):
        fh.seek(offset)
        fh.write(b"X")

    return change


def _truncate_to(size):
    return lambda fh: fh.truncate(size)


@pytest.mark.parametrize(
    ("change", "sent_chunks"),
    [
        (_overwrite_at(0), 0),
        (_overwrite_at(256 * 1024 + 5), 1),
        (_truncate_to(300 * 1024 - 10), 1),
        # Cut on a 64 KiB block boundary, every block still read is intact,
        # so only the short read shows the change.
        (_truncate_to(0), 0),
        (_truncate_to(3 * 64 * 1024), 0),
        (_truncate_to(256 * 1024), 1),
    ],
    ids=[
        "first-chunk-changed",
        "last-chunk-changed",
        "truncated",
        "emptied",
        "truncated-on-a-block-boundary",
        "truncated-at-the-chunk-boundary",
    ],
)
def test_wire_resumable_upload_stops_when_the_file_changes_after_hashing(
    monkeypatch, allowed_dir, change, sent_chunks
):
    """A file above the simple-upload limit is read again for the upload.
    Bytes other than the hashed ones are never sent, and the upload stops
    before its last chunk, so Drive never completes it."""
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 1000)
    monkeypatch.setattr(google_drive, "_RESUMABLE_CHUNK_BYTES", 256 * 1024)
    monkeypatch.setattr(google_drive, "_HASH_BLOCK_BYTES", 64 * 1024)
    data = (b"slide-bytes-" * 30000)[: 300 * 1024]
    local = _replacement(allowed_dir, data=data)
    session = "https://www.googleapis.com/upload/drive/v3/files/deck1?upload_id=s1"
    http = _wire_drive(
        monkeypatch,
        [
            _ok(_current()),
            ({"status": "200", "location": session}, ""),
            ({"status": "308", "range": "bytes=0-262143"}, ""),
            _ok(_updated(data)),
        ],
    )
    _change_after_hashing(monkeypatch, local, change)

    result = _call("deck1", local)

    message = result["message"]
    assert result["status"] == "error"
    assert "changed while it was being uploaded" in message
    assert "Nothing was changed in Google Drive" in message
    assert result["file"]["headRevisionId"] == "rev-1"
    methods = [request.method for request in http.requests]
    assert methods == ["GET", "PATCH"] + ["PUT"] * sent_chunks
    for request in http.requests[2:]:
        assert request.body == data[: len(request.body)]
        # No chunk ends the upload at another size than the hashed one.
        assert request.headers["content-range"] == (
            f"bytes 0-{len(request.body) - 1}/{len(data)}"
        )


def test_wire_resumable_upload_sends_only_the_hashed_bytes_of_a_grown_file(
    monkeypatch, allowed_dir
):
    monkeypatch.setattr(google_drive, "_SIMPLE_UPLOAD_MAX_BYTES", 1000)
    monkeypatch.setattr(google_drive, "_RESUMABLE_CHUNK_BYTES", 256 * 1024)
    data = (b"slide-bytes-" * 30000)[: 300 * 1024]
    local = _replacement(allowed_dir, data=data)
    session = "https://www.googleapis.com/upload/drive/v3/files/deck1?upload_id=s1"
    http = _wire_drive(
        monkeypatch,
        [
            _ok(_current()),
            ({"status": "200", "location": session}, ""),
            ({"status": "308", "range": "bytes=0-262143"}, ""),
            _ok(_updated(data)),
        ],
    )

    def append(fh):
        fh.seek(0, os.SEEK_END)
        fh.write(b"appended later")

    _change_after_hashing(monkeypatch, local, append)

    result = _call("deck1", local)

    assert result["status"] == "success", result
    _, start, first, second = http.requests
    assert start.headers["x-upload-content-length"] == str(len(data))
    assert second.headers["content-range"].endswith(f"/{len(data)}")
    assert first.body + second.body == data


def test_verified_upload_checks_the_whole_blocks_around_any_range(monkeypatch):
    monkeypatch.setattr(google_drive, "_HASH_BLOCK_BYTES", 4)
    data = bytes(range(26))
    fh = io.BytesIO(data)
    _, _, blocks = google_drive._hash_replacement_file(fh, len(data), keep_bytes=False)
    media = google_drive._VerifiedChunkUpload(
        fh,
        size=len(data),
        block_digests=blocks,
        mimetype=DECK_MIME,
        chunksize=256 * 1024,
        resumable=True,
    )

    assert media.size() == len(data)
    # Drive may report progress that is not on a block boundary.
    assert media.getbytes(5, 6) == data[5:11]
    assert media.getbytes(24, 10) == data[24:]
    assert media.getbytes(0, 100) == data

    fh.seek(0, os.SEEK_END)
    fh.write(b"appended")
    assert media.size() == len(data)
    assert media.getbytes(20, 100) == data[20:]

    fh.seek(9)
    fh.write(b"\xff")
    with pytest.raises(ValueError, match="changed while it was being read"):
        media.getbytes(5, 6)
    assert media.getbytes(0, 8) == data[:8]


# --- registration and download metadata ------------------------------------


async def test_tool_schema_and_annotations():
    tools = {tool.name: tool for tool in await google_drive.mcp.list_tools()}

    tool = tools["google_drive_update_file_content"]
    schema = tool.inputSchema
    assert set(schema["properties"]) == {
        "file_id",
        "file_path",
        "mime_type",
        "expected_head_revision_id",
    }
    assert sorted(schema["required"]) == ["file_id", "file_path"]
    assert tool.annotations is not None
    assert tool.annotations.destructiveHint is True
    assert tool.annotations.idempotentHint is True
    assert tool.annotations.readOnlyHint is not True
    # An identical repeat is a no-op ("unchanged"), so the tool stays out of
    # the same-turn duplicate-write guard, which would otherwise suppress a
    # retry after a failed attempt as if it had succeeded.
    wire = tool.annotations.model_dump(by_alias=True, exclude_none=True)
    assert classify_non_idempotent_write(wire) is False
    description = " ".join(tool.description.split())
    for phrase in (
        "Confirm with the user before calling",
        "say that its content will be replaced for everyone with access to it",
        "attach the replacement",
        # An approval in a later turn needs these to make the approved call.
        "end the confirmation with the Drive file id, the replacement's "
        "file:<id> and the headRevisionId",
        "Do not promise that the old version can always be restored",
        "Pass it whenever the file was downloaded and edited in this task",
        # After a save in this task, the download's head revision is stale.
        "once this tool saved it, from that save's result",
        "(not a Drive id)",
        "do not rebuild the file without asking the user",
        "google_drive_upload_file",
        '"unchanged"',
    ):
        assert phrase in description
    # The description is sent with every tool listing; details belong in the
    # result messages.
    assert len(tool.description) < 1800


def test_download_returns_revision_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", str(tmp_path))
    files = Mock()
    files.get.return_value.execute.return_value = {
        "id": "deck1",
        "name": "Quarterly Deck.pptx",
        "mimeType": DECK_MIME,
        "headRevisionId": "rev-1",
        "modifiedTime": "2026-05-12T08:00:00.000Z",
        "version": "7",
        "shared": True,
        "driveId": "team1",
    }
    service = Mock()
    service.files.return_value = files
    monkeypatch.setattr(google_drive, "get_drive_service", lambda: service)
    monkeypatch.setattr(google_drive, "_download_media", lambda request: OLD_CONTENT)

    result = json.loads(google_drive.google_drive_download_file("deck1"))

    assert result["status"] == "success"
    fields = files.get.call_args.kwargs["fields"]
    for field in ("headRevisionId", "modifiedTime", "version", "shared", "driveId"):
        assert field in fields
        assert field in result["file"]


# --- across runs, through the adapter ---------------------------------------


async def test_edited_file_from_an_earlier_run_is_staged_for_the_update(
    monkeypatch, tmp_path
):
    """An agent edits a file in one run and saves it to Drive after the user
    approves in a later run. The first run's workspace is gone by then, so
    the durable file:<id> must be staged into the new workspace, pass the
    connector's path guard, and upload byte for byte."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from xagent.core.file_storage.factory import get_unscoped_file_storage
    from xagent.core.workspace import TaskWorkspace
    from xagent.web.models import Base
    from xagent.web.models.task import Task
    from xagent.web.models.uploaded_file import UploadedFile
    from xagent.web.models.user import User

    engine = create_engine(
        f"sqlite:///{tmp_path / 'xagent.db'}",
        connect_args={"check_same_thread": False},
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", (tmp_path / "objects").as_uri())
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    get_unscoped_file_storage.cache_clear()
    monkeypatch.setattr("xagent.core.storage.manager.create_db_session", SessionLocal)
    monkeypatch.setattr("xagent.web.models.database._SessionLocal", SessionLocal)
    edited = bytes(range(256)) * 64
    db = SessionLocal()
    try:
        user = User(username="deck-owner", password_hash="hash")
        db.add(user)
        db.flush()
        db.add(Task(id=4242, user_id=user.id, title="Edit a deck"))
        db.commit()

        base_dir = str(tmp_path / "workspaces")
        first_run = TaskWorkspace(id="web_task_4242", base_dir=base_dir)
        first_copy = first_run.output_dir / "Quarterly Deck.pptx"
        first_copy.write_bytes(edited)
        file_id = first_run.register_file(str(first_copy), db_session=db)
        db.commit()
        record = db.query(UploadedFile).filter(UploadedFile.file_id == file_id).one()
        assert record.storage_status == "available"
    finally:
        db.close()
    first_run.cleanup()
    assert not first_copy.exists()

    second_run = TaskWorkspace(id="web_task_4242", base_dir=base_dir)
    task_dir = second_run.workspace_dir.resolve()
    monkeypatch.setenv(
        "XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS", json.dumps([str(task_dir)])
    )
    drive = _Drive(monkeypatch, updated=_updated(edited))
    tools = {tool.name: tool for tool in await google_drive.mcp.list_tools()}
    adapter = _build_mcp_tool_adapter(
        "Google Drive",
        {"transport": "stdio", "command": "python", "args": []},
        tools["google_drive_update_file_content"],
        workspace=second_run,
    )
    seen = {}

    class InProcessSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            staged = Path(arguments["file_path"])
            seen["file_id"] = arguments["file_id"]
            seen["path"] = staged
            seen["bytes"] = staged.read_bytes()
            text = getattr(google_drive, name)(**arguments)
            return CallToolResult(
                content=[TextContent(type="text", text=text)], isError=False
            )

    @asynccontextmanager
    async def fake_create_session(_connection):
        yield InProcessSession()

    monkeypatch.setattr(mcp_adapter_module, "create_session", fake_create_session)
    try:
        result = await adapter.run_json_async(
            {"file_id": "deck1", "file_path": f"file:{file_id}"}
        )
    finally:
        get_unscoped_file_storage.cache_clear()
        engine.dispose()

    payload = json.loads(result["content"][0]["text"])
    assert payload["status"] == "success", payload
    assert seen["file_id"] == "deck1"
    assert seen["path"].resolve().is_relative_to(task_dir)
    assert seen["path"].suffix == ".pptx"
    assert seen["bytes"] == edited
    assert drive.uploaded == [edited]
    assert _md5(drive.uploaded[0]) == _md5(edited) == payload["file"]["md5Checksum"]
    assert drive.files.update.call_args.kwargs["fileId"] == "deck1"
    assert not seen["path"].exists()
    assert "supplied for file_path" in adapter.description
