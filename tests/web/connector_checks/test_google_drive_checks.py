"""Google Drive's write check (#3567), run end to end.

The declaration in ``registry.py`` runs through the real engine and sweeper,
and every step calls the real tool function in
``web/tools/mcp/google_drive.py``. Only Drive's API is replaced, by an
in-memory drive laid out like the test tenant's seed: a closed seed folder
holding the seed document, and the write folder the write check creates in.
The Composio tests replace Composio's two calls as well, so the create runs
through ``google_drive_composio.create_file`` as it does on a production
grant: stage the content, run ``GOOGLEDRIVE_CREATE_FILE``, read the new file
back through the proxy.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from xagent.web.connector_checks.engine import run_checks
from xagent.web.connector_checks.results import RunResult
from xagent.web.connector_checks.sweeper import run_sweep
from xagent.web.tools.mcp import (
    composio_proxy,
    composio_tools,
    google_drive,
    google_drive_composio,
)

from .fakes import FakePolicyAdapter, FakeTool, listed, rows, targets

SEED_FOLDER = "folder-seed"
SEED_DOC = "doc-seed"
WRITE_FOLDER = "folder-write"
FOLDER_MIME = "application/vnd.google-apps.folder"
DOC_MIME = "application/vnd.google-apps.document"
SEED_DOC_TEXT = "xagent smoke seed document\nSecond line: seeded content\n"
FIXTURES = {
    "google-drive.seed_folder_id": SEED_FOLDER,
    "google-drive.write_folder_id": WRITE_FOLDER,
    "google-drive.write_sweep_query": (
        f"'{WRITE_FOLDER}' in parents and trashed = false"
    ),
    "google-docs.document_id": SEED_DOC,
}
# Toby classifies only Jira's writes, so it withholds neither Drive tool from a
# member who did not install the connector.
VERDICTS = {"google_drive_create_file": "allow", "google_drive_delete_file": "allow"}
NAMED = re.compile(r"xagent-smoke-[0-9]{12}-[0-9a-f]{6}-drive-file")
# The tenant pin's search: the write folder, by the seed's own name rule.
PIN_QUERY = (
    "name = 'xagent-smoke-write' and "
    f"mimeType = '{FOLDER_MIME}' and "
    "'root' in parents and "
    "trashed = false"
)
_NAME_QUERY = re.compile(
    r"name = '(?P<name>[^']*)' and mimeType = '(?P<mime_type>[^']*)'"
    r"(?: and '(?P<root>root)' in parents)? and trashed = false"
)
_PARENT_QUERY = re.compile(r"'(?P<parent>[^']*)' in parents and trashed = false")
# The lookup a create whose answer was lost makes: by name and parent.
_SAME_NAME_QUERY = re.compile(
    r"name = '(?P<name>[^']*)' and trashed = false and '(?P<parent>[^']*)' in parents"
)


def _in_parent(item: dict[str, Any], parent: str | None) -> bool:
    """Whether a name query's parent clause holds: anywhere when it has none,
    the root for the fake's top-level items, or one of the item's parents."""
    if parent is None:
        return True
    if parent == "root":
        return not item["parents"]
    return parent in item["parents"]


class Drive:
    """An in-memory Drive answering the requests google_drive.py sends."""

    def __init__(self) -> None:
        self.revoked = False
        self.items: dict[str, dict[str, Any]] = {}
        self.contents: dict[str, bytes] = {}
        self.requests: list[tuple[str, str]] = []
        self.created: list[dict[str, Any]] = []
        self.add(SEED_FOLDER, "xagent-smoke-seed", FOLDER_MIME)
        self.add(
            SEED_DOC,
            "xagent-smoke-seed-doc",
            DOC_MIME,
            parent=SEED_FOLDER,
            content=SEED_DOC_TEXT.encode("utf-8"),
        )
        self.add(WRITE_FOLDER, "xagent-smoke-write", FOLDER_MIME)

    def add(
        self,
        file_id: str,
        name: str,
        mime_type: str,
        *,
        parent: str | None = None,
        content: bytes = b"",
    ) -> str:
        self.items[file_id] = {
            "id": file_id,
            "name": name,
            "mimeType": mime_type,
            "modifiedTime": "2026-11-01T00:00:00.000Z",
            "parents": [parent] if parent else [],
        }
        self.contents[file_id] = content
        return file_id

    def files(self) -> _Files:
        return _Files(self)

    def refused(self) -> None:
        if self.revoked:
            raise RuntimeError("401 Client Error: Unauthorized")

    def search(self, query: str, page_size: int) -> dict[str, Any]:
        self.refused()
        self.requests.append(("list", query))
        by_name = _NAME_QUERY.fullmatch(query)
        if by_name:
            found = [
                item
                for item in self.items.values()
                if item["name"] == by_name["name"]
                and item["mimeType"] == by_name["mime_type"]
                and _in_parent(item, by_name["root"])
            ]
        else:
            same_name = _SAME_NAME_QUERY.fullmatch(query)
            if same_name is not None:
                found = [
                    {
                        "id": item["id"],
                        "name": item["name"],
                        "createdTime": item["modifiedTime"],
                    }
                    for item in self.items.values()
                    if item["name"] == same_name["name"]
                    and same_name["parent"] in item["parents"]
                ]
            else:
                by_parent = _PARENT_QUERY.fullmatch(query)
                assert by_parent, query
                found = [
                    item
                    for item in self.items.values()
                    if by_parent["parent"] in item["parents"]
                ]
        return {"files": found[:page_size]}

    def create(self, body: dict[str, Any], media: Any) -> dict[str, Any]:
        self.refused()
        (parent,) = body.get("parents") or [None]
        content = media.getbytes(0, 1 << 20)
        file_id = self.add(
            f"file-{len(self.items) + 1}",
            body["name"],
            body["mimeType"],
            parent=parent,
            content=content,
        )
        self.created.append(
            {
                "id": file_id,
                "name": body["name"],
                "parent": parent,
                "mime_type": body["mimeType"],
                "content": content,
            }
        )
        self.requests.append(("create", file_id))
        return {
            "id": file_id,
            "name": body["name"],
            "mimeType": body["mimeType"],
            "webViewLink": f"https://drive.example/{file_id}",
        }

    def get(self, file_id: str) -> dict[str, Any]:
        self.refused()
        self.requests.append(("get", file_id))
        if file_id not in self.items:
            raise RuntimeError("404 Client Error: Not Found")
        item = self.items[file_id]
        if item["mimeType"] == FOLDER_MIME:
            return item
        # What Drive reports for a stored file, and what a Composio create's
        # read-back and a proxied download are checked against.
        data = self.contents[file_id]
        return {
            **item,
            "size": str(len(data)),
            "md5Checksum": hashlib.md5(data, usedforsecurity=False).hexdigest(),
        }

    def media(self, file_id: str) -> _Media:
        self.refused()
        self.requests.append(("media", file_id))
        if file_id not in self.items:
            raise RuntimeError("404 Client Error: Not Found")
        return _Media(file_id)

    def delete(self, file_id: str) -> None:
        self.refused()
        self.requests.append(("delete", file_id))
        if file_id not in self.items:
            raise RuntimeError("404 Client Error: Not Found")
        del self.items[file_id]


class _Request:
    """An unexecuted Drive request: the call runs on ``execute``."""

    def __init__(self, run: Callable[[], Any]) -> None:
        self._run = run

    def execute(self) -> Any:
        return self._run()


class _Media:
    """A media request, whose bytes the test's downloader reads."""

    def __init__(self, file_id: str) -> None:
        self.file_id = file_id


class _Files:
    def __init__(self, drive: Drive) -> None:
        self._drive = drive

    def list(self, **kwargs: Any) -> _Request:
        return _Request(lambda: self._drive.search(kwargs["q"], kwargs["pageSize"]))

    def create(self, **kwargs: Any) -> _Request:
        return _Request(
            lambda: self._drive.create(kwargs["body"], kwargs["media_body"])
        )

    def get(self, **kwargs: Any) -> _Request:
        return _Request(lambda: self._drive.get(kwargs["fileId"]))

    def get_media(self, **kwargs: Any) -> _Media:
        return self._drive.media(kwargs["fileId"])

    def export_media(self, **kwargs: Any) -> _Media:
        return self._drive.media(kwargs["fileId"])

    def delete(self, **kwargs: Any) -> _Request:
        return _Request(lambda: self._drive.delete(kwargs["fileId"]))


def _tool(name: str) -> FakeTool:
    function: Callable[..., str] = getattr(google_drive, name)

    def respond(args: dict[str, Any]) -> Any:
        return {
            "content": [{"type": "text", "text": function(**args)}],
            "structured_content": None,
            "is_error": False,
        }

    return FakeTool(name, respond=respond)


# Every tool the server registers, so the session lists what production does.
_TOOLS = (
    "google_drive_search",
    "google_drive_get_file_content",
    "google_drive_download_file",
    "google_drive_upload_file",
    "google_drive_update_file_content",
    "google_drive_create_file",
    "google_drive_create_folder",
    "google_drive_rename_file",
    "google_drive_move_file",
    "google_drive_delete_file",
    "google_drive_list_permissions",
    "google_drive_share_file",
    "google_drive_update_permission",
    "google_drive_remove_permission",
)


@pytest.fixture()
def drive(monkeypatch: pytest.MonkeyPatch) -> Drive:
    drive = Drive()
    monkeypatch.setattr(google_drive, "get_drive_service", lambda: drive)
    monkeypatch.setattr(
        google_drive, "_download_media", lambda request: drive.contents[request.file_id]
    )
    return drive


class Composio:
    """Composio's two calls a Drive create makes, on the fake drive.

    ``stage_file`` keeps the bytes under a made-up key, and ``execute_tool``
    has ``GOOGLEDRIVE_CREATE_FILE`` create the file in Drive the way the
    tool does: named as the arguments say, in the folder its ``parents``
    names. ``fail_status`` loses the answer after the tool ran (a 5xx), the
    case where the file exists but the call reports it as unconfirmed.
    """

    def __init__(self, drive: Drive, *, fail_status: int | None = None) -> None:
        self.drive = drive
        self.fail_status = fail_status
        self.staged: dict[str, tuple[bytes, str]] = {}
        self.executed: list[tuple[str, dict[str, Any]]] = []

    def stage_file(
        self,
        *,
        toolkit_slug: str,
        tool_slug: str,
        filename: str,
        mimetype: str,
        md5: str,
        data: Any,
        length: int,
        on_dedup: Callable[[], None] | None = None,
    ) -> dict[str, str]:
        key = f"s3key-{len(self.staged) + 1}"
        body = bytes(data)
        assert hashlib.md5(body, usedforsecurity=False).hexdigest() == md5
        assert len(body) == length
        self.staged[key] = (body, mimetype)
        return {"name": filename, "mimetype": mimetype, "s3key": key}

    def execute_tool(
        self, tool_slug: str, arguments: dict[str, Any], *, toolkit: str
    ) -> composio_tools.ToolResult:
        self.executed.append((tool_slug, dict(arguments)))
        body, _ = self.staged[arguments["file_to_upload"]["s3key"]]
        (parent,) = arguments.get("parents") or [None]
        file_id = self.drive.add(
            f"file-{len(self.drive.items) + 1}",
            arguments["name"],
            arguments["mimeType"],
            parent=parent,
            content=body,
        )
        self.drive.created.append(
            {"id": file_id, "name": arguments["name"], "parent": parent}
        )
        if self.fail_status is not None:
            raise composio_tools.ComposioToolError(
                "bad gateway", status=self.fail_status
            )
        return composio_tools.ToolResult(
            successful=True, data={"id": file_id}, error=None
        )


@pytest.fixture()
def composio(drive: Drive, monkeypatch: pytest.MonkeyPatch) -> Composio:
    """A Composio-backed grant: Drive's calls stay on the fake drive, and the
    create stages and runs Composio's tool instead of Drive's files.create."""
    monkeypatch.setenv(composio_proxy.COMPOSIO_CONNECTED_ACCOUNT_ID_ENV, "ca_test")
    composio = Composio(drive)
    monkeypatch.setattr(composio_tools, "stage_file", composio.stage_file)
    monkeypatch.setattr(composio_tools, "execute_tool", composio.execute_tool)
    return composio


@pytest.fixture()
def writes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAGENT_CONNECTOR_SMOKE_WRITE_APPS", "google-drive")


def _tools() -> dict[str, FakeTool]:
    return {name: _tool(name) for name in _TOOLS}


def _adapter(
    fixtures: dict[str, Any] = FIXTURES, tools: dict[str, FakeTool] | None = None
) -> FakePolicyAdapter:
    tools = tools if tools is not None else _tools()
    return FakePolicyAdapter(
        targets("google-drive"),
        {"google-drive": listed(*tools.values())},
        verdicts=VERDICTS,
        fixtures=fixtures,
    )


async def _run(adapter: FakePolicyAdapter) -> RunResult:
    return await run_checks(adapter, check_timeout=5.0, total_timeout=60.0)


def _check_rows(result: RunResult) -> dict[str, tuple[Any, ...]]:
    # The session row carries no check id.
    return {
        row[1] or "session": row[2:] for row in rows(result) if row[0] == "google-drive"
    }


_CHECKS = (
    "google-drive.search_seed_folder",
    "google-drive.read_seed_file",
    "google-drive.file_round_trip",
)


# A run against the seeded drive


@pytest.mark.usefixtures("writes_on")
async def test_every_check_passes_on_the_seeded_drive_and_leaves_nothing(
    drive: Drive,
) -> None:
    seeded = dict(drive.items)
    tools = _tools()

    result = await _run(_adapter(tools=tools))

    assert _check_rows(result) == {
        "session": ("pass", None, None),
        **{check: ("pass", None, None) for check in _CHECKS},
    }
    assert result.complete
    # The write check created one run-scoped file in the write folder, read it
    # back, and deleted it again.
    assert len(drive.created) == 1
    assert len([kind for kind, _ in drive.requests if kind == "delete"]) == 1
    assert drive.items == seeded


@pytest.mark.usefixtures("writes_on")
async def test_the_write_check_names_its_file_for_its_run(drive: Drive) -> None:
    await _run(_adapter())

    (created,) = drive.created
    assert created["parent"] == WRITE_FOLDER
    assert created["mime_type"] == "text/plain"
    assert created["content"] == b"xagent smoke write probe"
    assert NAMED.fullmatch(created["name"])


async def test_with_writes_off_the_reads_still_run(drive: Drive) -> None:
    result = await _run(_adapter())

    checks = _check_rows(result)
    assert checks.pop("google-drive.file_round_trip") == (
        "skip",
        "writes_disabled",
        None,
    )
    assert set(checks.values()) == {("pass", None, None)}
    assert not [kind for kind, _ in drive.requests if kind in ("create", "delete")]


@pytest.mark.usefixtures("writes_on")
async def test_a_revoked_grant_fails_every_check_and_writes_nothing(
    drive: Drive,
) -> None:
    seeded = dict(drive.items)
    drive.revoked = True

    result = await _run(_adapter())

    checks = _check_rows(result)
    assert checks.pop("session") == ("pass", None, None)
    assert set(checks) == set(_CHECKS)
    assert {outcome for outcome, _, _ in checks.values()} == {"fail"}
    assert drive.items == seeded


@pytest.mark.usefixtures("writes_on")
async def test_a_pin_that_reads_another_write_folder_refuses_the_write(
    drive: Drive,
) -> None:
    seeded = dict(drive.items)

    result = await _run(
        _adapter({**FIXTURES, "google-drive.write_folder_id": "folder-2"})
    )

    checks = _check_rows(result)
    # The read checks keep to the seed, so they still pass; the pin reads the
    # write folder and requires the registry's id, so it refuses the write.
    assert checks["google-drive.search_seed_folder"] == ("pass", None, None)
    assert checks["google-drive.file_round_trip"] == (
        "fail",
        "tenant_mismatch",
        "mismatch",
    )
    assert not drive.created
    assert drive.items == seeded


@pytest.mark.usefixtures("writes_on")
async def test_a_same_named_folder_outside_the_root_does_not_fail_the_pin(
    drive: Drive,
) -> None:
    # The seed's own uniqueness rule counts folders in My Drive's root only,
    # so the pin's search does too: a folder with the write folder's name
    # deeper in the tree is not a second write folder.
    drive.add("folder-nested", "xagent-smoke-write", FOLDER_MIME, parent=SEED_FOLDER)

    result = await run_checks(
        _adapter(),
        checks=["google-drive.file_round_trip"],
        check_timeout=5.0,
        total_timeout=60.0,
    )

    assert _check_rows(result)["google-drive.file_round_trip"] == (
        "pass",
        None,
        None,
    )
    assert len(drive.created) == 1


@pytest.mark.usefixtures("writes_on")
async def test_a_write_folder_the_connection_cannot_read_refuses_the_write(
    drive: Drive,
) -> None:
    # A native `drive.file` grant, or a connection to another account, cannot
    # see the write folder the seed made: the pin's search finds no such
    # folder and nothing is written.
    seeded = dict(drive.items)
    del drive.items[WRITE_FOLDER]
    del seeded[WRITE_FOLDER]

    result = await run_checks(
        _adapter(),
        checks=["google-drive.file_round_trip"],
        check_timeout=5.0,
        total_timeout=60.0,
    )

    assert _check_rows(result)["google-drive.file_round_trip"] == (
        "fail",
        "tenant_mismatch",
        "pin_failed",
    )
    assert not drive.created
    assert drive.items == seeded


# A Composio-backed grant, the connection production actually serves


@pytest.mark.usefixtures("writes_on", "composio")
async def test_a_composio_grant_creates_reads_and_deletes_the_file(
    drive: Drive, composio: Composio
) -> None:
    seeded = dict(drive.items)

    result = await _run(_adapter())

    assert _check_rows(result) == {
        "session": ("pass", None, None),
        **{check: ("pass", None, None) for check in _CHECKS},
    }
    # The pin read the write folder before the create, and the create ran
    # Composio's tool with the file staged in that folder.
    assert ("list", PIN_QUERY) in drive.requests
    ((slug, arguments),) = composio.executed
    assert slug == google_drive_composio.GOOGLEDRIVE_CREATE_FILE
    assert arguments["parents"] == [WRITE_FOLDER]
    assert arguments["mimeType"] == "text/plain"
    assert arguments["file_to_upload"]["mimetype"] == "text/plain"
    assert NAMED.fullmatch(arguments["name"])
    # One file was created and deleted, through the same Drive API calls.
    assert len(drive.created) == 1
    assert len([kind for kind, _ in drive.requests if kind == "delete"]) == 1
    assert drive.items == seeded


@pytest.mark.usefixtures("writes_on", "composio")
async def test_a_composio_create_whose_answer_is_lost_leaves_the_file_to_the_sweep(
    drive: Drive, composio: Composio
) -> None:
    seeded = dict(drive.items)
    composio.fail_status = 502

    result = await _run(_adapter())

    # The tool ran, so the file exists; with no id in the answer the create
    # fails its status assertion and nothing is deleted.
    assert _check_rows(result)["google-drive.file_round_trip"] == (
        "fail",
        "assertion_failed",
        "create:status_success",
    )
    (created,) = set(drive.items) - set(seeded)
    assert drive.items[created]["parents"] == [WRITE_FOLDER]
    assert not [kind for kind, _ in drive.requests if kind == "delete"]

    # The next sweep finds it in the write folder and deletes it by id.
    later = datetime.now(timezone.utc) + timedelta(hours=3)
    swept = await run_sweep(
        _adapter(), call_timeout=5.0, total_timeout=60.0, now=lambda: later
    )
    (row,) = swept.to_document()["rows"]
    assert (row["outcome"], row["found"], row["deleted"]) == ("pass", 1, 1)
    assert drive.items == seeded


@pytest.mark.usefixtures("writes_on", "composio")
async def test_a_composio_grant_that_cannot_read_the_write_folder_writes_nothing(
    drive: Drive, composio: Composio
) -> None:
    # The grant reaches Drive (the reads answer) but not the seed's write
    # folder: a search for it finds nothing, so the pin fails and no create
    # runs.
    del drive.items[WRITE_FOLDER]
    seeded = dict(drive.items)

    result = await run_checks(
        _adapter(),
        checks=["google-drive.file_round_trip"],
        check_timeout=5.0,
        total_timeout=60.0,
    )

    assert _check_rows(result)["google-drive.file_round_trip"] == (
        "fail",
        "tenant_mismatch",
        "pin_failed",
    )
    assert not composio.executed
    assert not drive.created
    assert drive.items == seeded


@pytest.mark.usefixtures("writes_on")
async def test_a_file_that_lost_its_run_scoped_name_is_not_deleted(
    drive: Drive,
) -> None:
    seeded = dict(drive.items)
    original = drive.create

    def rename(body: dict[str, Any], media: Any) -> dict[str, Any]:
        file = original(body, media)
        drive.items[file["id"]]["name"] = "another name"
        return file

    drive.create = rename  # type: ignore[method-assign]

    result = await _run(_adapter())

    # The read-back no longer finds the run-scoped name, so the guard stops
    # the cleanup and the file stays for the sweeper.
    assert _check_rows(result)["google-drive.file_round_trip"] == (
        "fail",
        "cleanup_failed",
        "ownership",
    )
    assert not [kind for kind, _ in drive.requests if kind == "delete"]
    assert set(drive.items) == set(seeded) | {drive.created[0]["id"]}


@pytest.mark.usefixtures("writes_on")
async def test_a_create_that_returns_an_error_is_not_cleaned_up(
    drive: Drive,
) -> None:
    seeded = dict(drive.items)

    def refuse(body: dict[str, Any], media: Any) -> dict[str, Any]:
        raise RuntimeError("403 Client Error: Forbidden")

    drive.create = refuse  # type: ignore[method-assign]

    result = await _run(_adapter())

    # The tool turns the Drive error into its own error envelope, so the create
    # fails its status assertion; with no created id, nothing is deleted.
    assert _check_rows(result)["google-drive.file_round_trip"] == (
        "fail",
        "assertion_failed",
        "create:status_success",
    )
    assert not [kind for kind, _ in drive.requests if kind == "delete"]
    assert drive.items == seeded


# Drifted seed objects


async def test_a_seed_document_whose_text_is_gone_fails_its_read(
    drive: Drive,
) -> None:
    drive.contents[SEED_DOC] = b"another document"

    result = await _run(_adapter())

    outcome, failure_class, reason = _check_rows(result)["google-drive.read_seed_file"]
    assert (outcome, failure_class) == ("fail", "assertion_failed")
    assert reason == "contains"


async def test_a_seed_document_that_drifted_fails_its_read(drive: Drive) -> None:
    drive.items[SEED_DOC]["name"] = "another-doc"

    result = await _run(_adapter())

    outcome, failure_class, reason = _check_rows(result)["google-drive.read_seed_file"]
    assert (outcome, failure_class) == ("fail", "assertion_failed")
    assert reason == "equals"


async def test_a_seed_folder_that_is_gone_fails_its_read(drive: Drive) -> None:
    del drive.items[SEED_FOLDER]

    result = await _run(_adapter())

    outcome, failure_class, reason = _check_rows(result)[
        "google-drive.search_seed_folder"
    ]
    assert (outcome, failure_class) == ("fail", "assertion_failed")
    assert reason == "length"


# The sweep


@pytest.mark.usefixtures("writes_on")
async def test_the_sweep_deletes_only_old_leftovers_in_the_write_folder(
    drive: Drive,
) -> None:
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)

    def run_name(age: timedelta) -> str:
        return f"xagent-smoke-{(now - age).strftime('%Y%m%d%H%M')}-a1b2c3-drive-file"

    seeded = dict(drive.items)
    stale = run_name(timedelta(hours=3))
    old = drive.add("file-old", stale, "text/plain", parent=WRITE_FOLDER)
    young = drive.add(
        "file-young", run_name(timedelta(minutes=30)), "text/plain", parent=WRITE_FOLDER
    )
    # The same old run-scoped name outside the write folder: the sweep lists
    # the write folder alone, so it stays.
    elsewhere = drive.add("file-elsewhere", stale, "text/plain", parent=SEED_FOLDER)
    tools = _tools()

    result = await run_sweep(
        _adapter(tools=tools), call_timeout=5.0, total_timeout=60.0, now=lambda: now
    )

    assert tools["google_drive_delete_file"].calls == [{"file_id": old}]
    (row,) = result.to_document()["rows"]
    assert (row["connector"], row["outcome"]) == ("google-drive", "pass")
    assert old not in drive.items
    assert young in drive.items
    assert elsewhere in drive.items
    assert set(drive.items) == set(seeded) | {young, elsewhere}
    assert ("list", FIXTURES["google-drive.write_sweep_query"]) in drive.requests
