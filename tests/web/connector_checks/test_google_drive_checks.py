"""Google Drive's write check (#3567), run end to end.

The declaration in ``registry.py`` runs through the real engine and sweeper,
and every step calls the real tool function in
``web/tools/mcp/google_drive.py``. Only Drive's API is replaced, by an
in-memory drive laid out like the test tenant's seed: a closed seed folder
holding the seed document, and the write folder the write check creates in.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from xagent.web.connector_checks.engine import run_checks
from xagent.web.connector_checks.results import RunResult
from xagent.web.connector_checks.sweeper import run_sweep
from xagent.web.tools.mcp import google_drive

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
_NAME_QUERY = re.compile(
    r"name = '(?P<name>[^']*)' and mimeType = '(?P<mime_type>[^']*)' "
    "and trashed = false"
)
_PARENT_QUERY = re.compile(r"'(?P<parent>[^']*)' in parents and trashed = false")


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
        return self.items[file_id]

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
async def test_a_pin_that_reads_another_seed_folder_refuses_the_write(
    drive: Drive,
) -> None:
    seeded = dict(drive.items)

    result = await _run(
        _adapter({**FIXTURES, "google-drive.seed_folder_id": "folder-2"})
    )

    checks = _check_rows(result)
    # The same search backs the read check, which compares the id it found,
    # and the pin, which compares the id it read.
    assert checks["google-drive.search_seed_folder"] == (
        "fail",
        "assertion_failed",
        "equals",
    )
    assert checks["google-drive.file_round_trip"] == (
        "fail",
        "tenant_mismatch",
        "mismatch",
    )
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
    old = drive.add(
        "file-old", run_name(timedelta(hours=3)), "text/plain", parent=WRITE_FOLDER
    )
    young = drive.add(
        "file-young", run_name(timedelta(minutes=30)), "text/plain", parent=WRITE_FOLDER
    )
    tools = _tools()

    result = await run_sweep(
        _adapter(tools=tools), call_timeout=5.0, total_timeout=60.0, now=lambda: now
    )

    assert tools["google_drive_delete_file"].calls == [{"file_id": old}]
    (row,) = result.to_document()["rows"]
    assert (row["connector"], row["outcome"]) == ("google-drive", "pass")
    assert old not in drive.items
    assert young in drive.items
    assert set(drive.items) == set(seeded) | {young}
    assert ("list", FIXTURES["google-drive.write_sweep_query"]) in drive.requests
