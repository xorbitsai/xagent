"""The Microsoft 365 family's smoke V2 Tier 1 checks (#3569), run end to end.

The declarations in ``registry.py`` run through the real engine, and every
check calls the real tool function of ``web/tools/mcp/{onedrive,teams,word,
excel,powerpoint}.py``. Only Microsoft Graph is replaced, by an in-memory
tenant laid out like the test account's seed: a closed OneDrive folder
holding a text note and the Word, Excel and PowerPoint seed files, and a team
whose one channel lists the seeded message first. The family runs read
checks only, so no test here switches writes on.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any
from urllib.parse import unquote

import pytest
import requests
from docx import Document
from pptx import Presentation
from pptx.util import Inches
from requests.structures import CaseInsensitiveDict

from xagent.core.utils import public_requests
from xagent.web.connector_checks.engine import run_checks
from xagent.web.connector_checks.registry import CONNECTOR_CHECKS
from xagent.web.connector_checks.results import RunResult
from xagent.web.tools.mcp import excel, onedrive, powerpoint, teams, word

from .fakes import FakeAdapter, FakeTool, listed, rows, targets, tool_for

_MODULES: dict[str, ModuleType] = {
    "onedrive": onedrive,
    "teams": teams,
    "word": word,
    "excel": excel,
    "powerpoint": powerpoint,
}

# The seed, as the cross-repo contract of #3569 lays it out.
SEED_FOLDER = "xagent-smoke-seed"
# q7k2 appears in no other file's name or content, so a search finds the note
# alone (contract v2).
NOTE = "xagent-smoke-seed-note-q7k2.txt"
NOTE_TEXT = "seeded by the xagent connector smoke check\n"
DOCX = "xagent-smoke-seed.docx"
XLSX = "xagent-smoke-seed.xlsx"
PPTX = "xagent-smoke-seed.pptx"
PARAGRAPHS = ("xagent smoke seed document", "second seeded paragraph")
SLIDE_TEXT = "xagent smoke seed presentation"
SHEET = "seed"
CELLS: list[list[Any]] = [["item", "quantity"], ["alpha", 1], ["beta", 2]]
MESSAGE_TEXT = "xagent smoke seed message"

DRIVE_ID = "drive-1"
NOTE_ID = "item-note"
TEAM_ID = "team-read"
CHANNEL_ID = "19:general@thread.tacv2"
MESSAGE_ID = "1731400000000"

FIXTURES: dict[str, Any] = {
    "onedrive.seed_folder_path": SEED_FOLDER,
    "onedrive.seed_file_path": f"{SEED_FOLDER}/{NOTE}",
    "onedrive.seed_file_id": NOTE_ID,
    "word.file_path": f"{SEED_FOLDER}/{DOCX}",
    "excel.file_path": f"{SEED_FOLDER}/{XLSX}",
    "powerpoint.file_path": f"{SEED_FOLDER}/{PPTX}",
    "teams.seed_team_id": TEAM_ID,
    "teams.seed_channel_id": CHANNEL_ID,
    "teams.seed_message_id": MESSAGE_ID,
}

# Each connector's checks, in declaration order: the result diff keys on them.
CHECKS: dict[str, tuple[str, ...]] = {
    "onedrive": (
        "onedrive.list_seed_folder",
        "onedrive.search_seed_file",
        "onedrive.current_user",
        "onedrive.get_seed_file",
        "onedrive.read_seed_file",
    ),
    "teams": (
        "teams.auth_status",
        "teams.list_teams",
        "teams.list_chats",
        "teams.list_seed_channels",
        "teams.read_seed_message",
    ),
    "word": ("word.read_fixture", "word.list_paragraphs"),
    "excel": ("excel.read_fixture", "excel.read_seed_range", "excel.read_used_range"),
    "powerpoint": (
        "powerpoint.read_fixture",
        "powerpoint.read_slide_text",
        "powerpoint.read_presentation_text",
    ),
}
ALL_CHECKS = tuple(check for checks in CHECKS.values() for check in checks)


def _docx(paragraphs: tuple[str, ...] = PARAGRAPHS) -> bytes:
    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _pptx(
    slides: tuple[tuple[str, ...], ...] = ((SLIDE_TEXT,),), notes: str | None = None
) -> bytes:
    """A deck with one blank-layout slide per item, each holding one text box
    per text."""
    presentation = Presentation()
    for texts in slides:
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        for index, text in enumerate(texts):
            box = slide.shapes.add_textbox(
                Inches(1), Inches(1 + index), Inches(6), Inches(1)
            )
            box.text_frame.text = text
        if notes is not None:
            slide.notes_slide.notes_text_frame.text = notes
    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


_CELL = re.compile(r"(?P<column>[A-Z])(?P<row>[0-9]+)")


def _response(
    status: int, *, payload: Any = None, content: bytes | None = None, url: str
) -> requests.Response:
    """A real ``requests.Response``, so each tool's own status and streaming
    handling runs unchanged."""
    response = requests.Response()
    response.status_code = status
    response.reason = {200: "OK", 401: "Unauthorized", 404: "Not Found"}.get(
        status, "Error"
    )
    response.url = url
    response.encoding = "utf-8"
    if content is None:
        content = json.dumps(payload).encode("utf-8")
        content_type = "application/json"
    else:
        content_type = "application/octet-stream"
    response._content = content
    response._content_consumed = True  # type: ignore[attr-defined]
    response.headers = CaseInsensitiveDict(
        {"Content-Type": content_type, "Content-Length": str(len(content))}
    )
    return response


class Tenant:
    """An in-memory Microsoft Graph answering the reads the five servers send.

    Anything but a GET fails the test: the family declares no write check.
    """

    def __init__(self) -> None:
        self.revoked = False
        self.me: dict[str, Any] = {
            "id": "user-1",
            "displayName": "Smoke QA",
            "userPrincipalName": "qa@example.invalid",
            "mail": "qa@example.invalid",
        }
        # The seed folder's files, in the order Graph lists them.
        self.files: dict[str, dict[str, Any]] = {}
        self.add_file(NOTE, NOTE_TEXT.encode("utf-8"), item_id=NOTE_ID)
        self.add_file(DOCX, _docx())
        self.add_file(XLSX, b"PK")
        self.add_file(PPTX, _pptx())
        # The workbook as Graph's workbook API reads it, by worksheet.
        self.worksheets: dict[str, list[list[Any]]] = {
            SHEET: [list(row) for row in CELLS]
        }
        # Cells that carry formatting but no value, by worksheet.
        self.formatted: dict[str, list[tuple[int, int]]] = {}
        self.teams: list[dict[str, Any]] = [
            {"id": "team-other", "displayName": "Another team", "description": None},
            {"id": TEAM_ID, "displayName": "xagent-smoke-read", "description": None},
        ]
        self.channels: dict[str, list[dict[str, Any]]] = {
            TEAM_ID: [
                {
                    "id": CHANNEL_ID,
                    "displayName": "General",
                    "membershipType": "standard",
                }
            ]
        }
        # In Graph's order: by the last activity of each reply chain, most
        # recently active first, system messages included.
        self.messages: dict[str, list[dict[str, Any]]] = {
            CHANNEL_ID: [self.message(MESSAGE_ID, MESSAGE_TEXT)]
        }
        # The seed cannot hold a chat, so the account's chat list is empty.
        self.chats: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str]] = []

    @staticmethod
    def message(message_id: str, text: str) -> dict[str, Any]:
        return {
            "id": message_id,
            "messageType": "message",
            "createdDateTime": "2026-11-12T00:00:00Z",
            "from": {"user": {"id": "user-1", "displayName": "Smoke QA"}},
            "body": {"contentType": "text", "content": text},
        }

    def add_file(
        self, name: str, content: bytes, *, item_id: str | None = None
    ) -> None:
        self.files[name] = {
            "id": item_id or f"item-{len(self.files) + 1}",
            "name": name,
            "size": len(content),
            "eTag": f'"{{{name}}},1"',
            "file": {"mimeType": "application/octet-stream"},
            "parentReference": {
                "driveId": DRIVE_ID,
                "path": f"/drive/root:/{SEED_FOLDER}",
            },
            "@microsoft.graph.downloadUrl": f"https://download.invalid/{name}",
            "content": content,
        }

    def set_content(self, name: str, content: bytes) -> None:
        self.files[name]["content"] = content
        self.files[name]["size"] = len(content)

    # Graph

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        path = unquote(url.removeprefix(onedrive.GRAPH_BASE_URL))
        # Recorded before anything else: a tool swallows an exception raised
        # here, so the tests assert on this log instead.
        self.requests.append((method, path))
        if method != "GET" or not url.startswith(onedrive.GRAPH_BASE_URL):
            return _response(405, payload={}, url=url)
        if self.revoked:
            return _response(
                401,
                payload={
                    "error": {
                        "code": "InvalidAuthenticationToken",
                        "message": "Lifetime validation failed, the token is expired.",
                    }
                },
                url=url,
            )
        if path == "/me/chats":
            # Graph takes $top from 1 to 50 here and answers 400 otherwise.
            top = (kwargs.get("params") or {}).get("$top")
            if not isinstance(top, int) or not 1 <= top <= 50:
                return _response(
                    400,
                    payload={"error": {"code": "BadRequest", "message": "$top"}},
                    url=url,
                )
            return _response(200, payload={"value": self.chats}, url=url)
        try:
            answer = self.answer(path)
        except LookupError:
            return _response(
                404,
                payload={"error": {"code": "itemNotFound", "message": "not found"}},
                url=url,
            )
        if isinstance(answer, bytes):
            return _response(200, content=answer, url=url)
        return _response(200, payload=answer, url=url)

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """A preauthenticated download URL, which carries no bearer token."""
        self.requests.append(("GET", url))
        for item in self.files.values():
            if item["@microsoft.graph.downloadUrl"] == url:
                return _response(200, content=item["content"], url=url)
        return _response(404, payload={}, url=url)

    def refuse(self, *args: Any, **kwargs: Any) -> Any:
        self.requests.append(("WRITE", repr(args)))
        raise RuntimeError("the family's checks only read")

    def answer(self, path: str) -> Any:
        if path == "/me":
            return self.me
        if path == "/me/joinedTeams":
            return {"value": self.teams}
        if match := re.fullmatch(r"/teams/([^/]+)/channels", path):
            return {"value": self.channels[match[1]]}
        if match := re.fullmatch(r"/teams/([^/]+)/channels/([^/]+)/messages", path):
            if match[1] not in self.channels:
                raise LookupError(path)
            return {"value": self.messages[match[2]]}
        if match := re.fullmatch(r"/me/drive/root/search\(q='(.*)'\)", path):
            # Graph search matches a file's name and its content.
            query = match[1].lower()
            return {
                "value": [
                    self.listed(item)
                    for item in self.files.values()
                    if query in item["name"].lower()
                    or query.encode("utf-8") in item["content"].lower()
                ]
            }
        if match := re.fullmatch(r"/me/drive/items/([^/]+)", path):
            return self.listed(self.by_id(match[1]))
        if match := re.fullmatch(rf"/drives/{DRIVE_ID}/items/([^/]+)/content", path):
            return self.by_id(match[1])["content"]
        match = re.fullmatch(r"/me/drive/root:/(?P<path>[^:]+):(?P<rest>.*)", path)
        if match is None:
            raise LookupError(path)
        if match["path"] == SEED_FOLDER and match["rest"] == "/children":
            return {"value": [self.listed(item) for item in self.files.values()]}
        folder, _, name = match["path"].rpartition("/")
        if folder != SEED_FOLDER or name not in self.files:
            raise LookupError(path)
        item = self.files[name]
        if match["rest"] == "":
            return self.listed(item)
        if match["rest"] == "/content":
            return item["content"]
        if name == XLSX and match["rest"].startswith("/workbook/"):
            return self.workbook(match["rest"][len("/workbook") :])
        raise LookupError(path)

    def listed(self, item: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in item.items() if key != "content"}

    def by_id(self, item_id: str) -> dict[str, Any]:
        for item in self.files.values():
            if item["id"] == item_id:
                return item
        raise LookupError(item_id)

    def workbook(self, path: str) -> Any:
        if path == "/worksheets":
            return {
                "value": [
                    {"id": f"{{{index}}}", "name": name, "position": index}
                    for index, name in enumerate(self.worksheets)
                ]
            }
        match = re.fullmatch(
            r"/worksheets\('(?P<sheet>[^']+)'\)/"
            r"(?:range\(address='(?P<first>[A-Z][0-9]+):(?P<last>[A-Z][0-9]+)'\)"
            r"|(?P<used>usedRange)(?P<values_only>\(valuesOnly=true\))?)",
            path,
        )
        if match is None or match["sheet"] not in self.worksheets:
            raise LookupError(path)
        grid = self.worksheets[match["sheet"]]
        if match["used"]:
            # Graph's used range also spans formatted empty cells, unless
            # it is asked for valuesOnly.
            cells = [
                (row, column)
                for row, cells_ in enumerate(grid)
                for column, value in enumerate(cells_)
                if value != ""
            ]
            if not match["values_only"]:
                cells += self.formatted.get(match["sheet"], [])
            first = (0, 0)
            last = (max(r for r, _ in cells), max(c for _, c in cells))
        else:
            start, end = _CELL.fullmatch(match["first"]), _CELL.fullmatch(match["last"])
            assert start and end
            first = (int(start["row"]) - 1, ord(start["column"]) - ord("A"))
            last = (int(end["row"]) - 1, ord(end["column"]) - ord("A"))
        values = [
            [
                grid[row][column] if row < len(grid) and column < len(grid[row]) else ""
                for column in range(first[1], last[1] + 1)
            ]
            for row in range(first[0], last[0] + 1)
        ]
        return {
            "address": (
                f"{match['sheet']}!{chr(ord('A') + first[1])}{first[0] + 1}:"
                f"{chr(ord('A') + last[1])}{last[0] + 1}"
            ),
            "rowCount": len(values),
            "columnCount": len(values[0]),
            "values": values,
            "text": [[str(value) for value in row] for row in values],
        }


@pytest.fixture()
def tenant(monkeypatch: pytest.MonkeyPatch) -> Tenant:
    tenant = Tenant()
    monkeypatch.setenv("AUTH_TOKEN", "graph-token")
    # The five servers share this client. Their writes also go through
    # post/put and a Session, which record a refused call here.
    monkeypatch.setattr(public_requests, "request", tenant.request)
    monkeypatch.setattr(public_requests, "get", tenant.get)
    for name in ("post", "put", "patch", "delete", "Session"):
        monkeypatch.setattr(public_requests, name, tenant.refuse)
    return tenant


def _registered(module: ModuleType) -> tuple[str, ...]:
    return tuple(tool.name for tool in asyncio.run(module.mcp.list_tools()))


# Every tool each server registers, so a session lists what production does.
_REGISTERED = {app_id: _registered(module) for app_id, module in _MODULES.items()}


def _tools() -> dict[str, dict[str, FakeTool]]:
    return {
        app_id: {name: tool_for(_MODULES[app_id], name) for name in names}
        for app_id, names in _REGISTERED.items()
    }


async def _run(
    tools: dict[str, dict[str, FakeTool]] | None = None,
    fixtures: dict[str, Any] = FIXTURES,
) -> RunResult:
    tools = tools if tools is not None else _tools()
    adapter = FakeAdapter(
        targets(*_MODULES),
        {app_id: listed(*by_name.values()) for app_id, by_name in tools.items()},
        fixtures=fixtures,
    )
    return await run_checks(adapter, check_timeout=10.0, total_timeout=120.0)


def _outcomes(result: RunResult) -> dict[str, tuple[Any, ...]]:
    """Each row's (outcome, class, reason), by check id or ``<app>.session``."""
    return {
        check or f"{connector}.session": (outcome, failure, reason)
        for connector, check, outcome, failure, reason in rows(result)
    }


PASS = ("pass", None, None)


def _called(tools: dict[str, dict[str, FakeTool]]) -> set[str]:
    return {
        name
        for by_name in tools.values()
        for name, tool in by_name.items()
        if tool.calls
    }


# The declarations


def test_the_family_declares_its_reads_and_no_write_check() -> None:
    """Five reads at most per connector, and no write check or sweep: the
    OneDrive, Excel and PowerPoint writes wait on a tenant pin none of their
    tools can read, and Teams has no tool that deletes a message."""
    for app_id, check_ids in CHECKS.items():
        checks = CONNECTOR_CHECKS[app_id]
        assert tuple(check.id for check in checks.reads) == check_ids
        assert len(checks.reads) <= 5
        assert not checks.writes
        assert checks.sweep is None


# A run against the seeded tenant


async def test_every_check_passes_on_the_seeded_tenant_and_only_reads(
    tenant: Tenant,
) -> None:
    tools = _tools()

    result = await _run(tools)

    assert _outcomes(result) == {
        **{f"{app_id}.session": PASS for app_id in _MODULES},
        **{check: PASS for check in ALL_CHECKS},
    }
    assert result.complete
    assert _called(tools) == {
        tool for app_id in CHECKS for tool in CONNECTOR_CHECKS[app_id].checked_tools()
    }
    assert {method for method, _ in tenant.requests} == {"GET"}
    # auth_status already reads /me; the second /me read is exempt (#3569).
    assert "get_current_user" not in _called(tools)


@pytest.mark.parametrize("chats", [0, 2])
async def test_the_chat_list_is_checked_for_its_shape_alone(
    tenant: Tenant, chats: int
) -> None:
    """The seed cannot hold a chat, so an empty list passes, and so does an
    account that has chats."""
    tenant.chats = [
        {"id": f"19:chat-{index}@thread.v2", "chatType": "oneOnOne", "topic": None}
        for index in range(chats)
    ]

    outcomes = _outcomes(await _run())

    assert outcomes["teams.list_chats"] == PASS
    assert ("GET", "/me/chats") in tenant.requests


async def test_a_revoked_token_fails_every_check(tenant: Tenant) -> None:
    """Acceptance 2 of #3569: a token Graph refuses turns every check red,
    while the session, which only lists tools, still passes."""
    tenant.revoked = True
    tools = _tools()

    result = await _run(tools)

    outcomes = _outcomes(result)
    for app_id in _MODULES:
        assert outcomes.pop(f"{app_id}.session") == PASS
    assert set(outcomes) == set(ALL_CHECKS)
    assert {(outcome, failure) for outcome, failure, _ in outcomes.values()} == {
        ("fail", "assertion_failed")
    }
    assert _called(tools) <= {
        tool for app_id in CHECKS for tool in CONNECTOR_CHECKS[app_id].checked_tools()
    }
    assert {method for method, _ in tenant.requests} == {"GET"}


# Which checks read which fixture key: the tenant registry must record each.
# A key a declaration reads beyond these would skip its check in the passing
# run above.
_KEY_READERS: dict[str, set[str]] = {
    "onedrive.seed_folder_path": {"onedrive.list_seed_folder"},
    "onedrive.seed_file_path": {"onedrive.read_seed_file"},
    "onedrive.seed_file_id": {"onedrive.search_seed_file", "onedrive.get_seed_file"},
    "word.file_path": set(CHECKS["word"]),
    "excel.file_path": set(CHECKS["excel"]),
    "powerpoint.file_path": set(CHECKS["powerpoint"]),
    "teams.seed_team_id": {"teams.list_seed_channels", "teams.read_seed_message"},
    "teams.seed_channel_id": {"teams.list_seed_channels", "teams.read_seed_message"},
    "teams.seed_message_id": {"teams.read_seed_message"},
}


@pytest.mark.parametrize("key", sorted(FIXTURES))
async def test_a_missing_fixture_skips_only_the_checks_that_read_it(
    tenant: Tenant, key: str
) -> None:
    fixtures = {k: v for k, v in FIXTURES.items() if k != key}

    outcomes = _outcomes(await _run(fixtures=fixtures))

    skipped = {check for check, row in outcomes.items() if row != PASS}
    assert skipped == _KEY_READERS[key]
    assert {outcomes[check] for check in skipped} == {("skip", "fixture_missing", key)}


# A seed that drifted


def _edit_note(tenant: Tenant) -> None:
    tenant.set_content(NOTE, b"seeded by the xagent connector smoke check")


def _extra_file(tenant: Tenant) -> None:
    tenant.add_file("stray.txt", b"stray\n")


def _old_note_name_first(tenant: Tenant) -> None:
    # A file named as the note was before the marker, listed first: the
    # search for the marker no longer sees it.
    files = tenant.files
    tenant.files = {}
    tenant.add_file("xagent-smoke-seed-note.txt", NOTE_TEXT.encode("utf-8"))
    tenant.files.update(files)


def _marker_in_another_file(tenant: Tenant) -> None:
    tenant.add_file("stray.txt", b"q7k2\n")


def _renamed_note(tenant: Tenant) -> None:
    # Still found by the marker, but no longer the seeded name.
    tenant.files[NOTE]["name"] = "xagent-smoke-seed-note-q7k2-old.txt"


def _profile_without_id(tenant: Tenant) -> None:
    tenant.me = {**tenant.me, "id": None}


def _profile_without_principal_name(tenant: Tenant) -> None:
    # auth_status reports the profile's fields by name, so only OneDrive's
    # raw profile read can miss one.
    del tenant.me["userPrincipalName"]


def _team_without_name(tenant: Tenant) -> None:
    del tenant.teams[0]["displayName"]


def _chat_without_id(tenant: Tenant) -> None:
    tenant.chats = [{"chatType": "group", "topic": "no id"}]


def _chats_not_a_list(tenant: Tenant) -> None:
    tenant.chats = {"id": "19:chat@thread.v2"}  # type: ignore[assignment]


def _no_teams(tenant: Tenant) -> None:
    # The account has left every team, the read team included.
    tenant.teams = []


def _other_channel(tenant: Tenant) -> None:
    tenant.channels[TEAM_ID][0]["id"] = "19:other@thread.tacv2"


def _extra_channel(tenant: Tenant) -> None:
    tenant.channels[TEAM_ID].append(
        {"id": "19:extra@thread.tacv2", "displayName": "Extra"}
    )


def _html_message(tenant: Tenant) -> None:
    # Teams may store a message sent as text with an HTML body.
    tenant.messages[CHANNEL_ID][0]["body"] = {
        "contentType": "html",
        "content": f"<p>{MESSAGE_TEXT}</p>",
    }


def _newer_message(tenant: Tenant) -> None:
    tenant.messages[CHANNEL_ID].insert(0, tenant.message("1731500000000", "hello"))


def _edited_message(tenant: Tenant) -> None:
    # Still starts like the seed's text, so only the whole text tells them apart.
    tenant.messages[CHANNEL_ID][0]["body"]["content"] = "xagent smoke test message"


def _extra_paragraph(tenant: Tenant) -> None:
    tenant.set_content(DOCX, _docx((*PARAGRAPHS, "third")))


def _other_first_paragraph(tenant: Tenant) -> None:
    tenant.set_content(DOCX, _docx(("another paragraph", PARAGRAPHS[1])))


def _other_paragraph_text(tenant: Tenant) -> None:
    tenant.set_content(DOCX, _docx((PARAGRAPHS[0], "another paragraph")))


def _extra_worksheet(tenant: Tenant) -> None:
    tenant.worksheets["Sheet2"] = []


def _renamed_worksheet(tenant: Tenant) -> None:
    tenant.worksheets = {"Sheet1": tenant.worksheets[SHEET]}


def _quantity_as_text(tenant: Tenant) -> None:
    # A number typed as text reads back as the string "1".
    tenant.worksheets[SHEET][1][1] = "1"


def _other_item(tenant: Tenant) -> None:
    tenant.worksheets[SHEET][2][0] = "gamma"


def _formatted_cell_outside(tenant: Tenant) -> None:
    # A filled or bordered empty cell widens Graph's plain used range.
    tenant.formatted[SHEET] = [(9, 3)]


def _extra_row(tenant: Tenant) -> None:
    tenant.worksheets[SHEET].append(["gamma", 3])


def _extra_slide(tenant: Tenant) -> None:
    tenant.set_content(PPTX, _pptx(((SLIDE_TEXT,), ("second slide",))))


def _other_slide_text(tenant: Tenant) -> None:
    tenant.set_content(PPTX, _pptx((("another presentation",),)))


def _extra_text_box(tenant: Tenant) -> None:
    tenant.set_content(PPTX, _pptx(((SLIDE_TEXT, "second box"),)))


def _speaker_notes(tenant: Tenant) -> None:
    tenant.set_content(PPTX, _pptx(notes="speaker notes"))


@pytest.mark.parametrize(
    ("drift", "failing"),
    [
        (_edit_note, {"onedrive.read_seed_file"}),
        (_extra_file, {"onedrive.list_seed_folder"}),
        (_old_note_name_first, {"onedrive.list_seed_folder"}),
        (
            _marker_in_another_file,
            {"onedrive.list_seed_folder", "onedrive.search_seed_file"},
        ),
        (_renamed_note, {"onedrive.get_seed_file"}),
        (
            _profile_without_id,
            {"onedrive.current_user", "teams.auth_status"},
        ),
        (
            _profile_without_principal_name,
            {"onedrive.current_user"},
        ),
        (_team_without_name, {"teams.list_teams"}),
        (_chat_without_id, {"teams.list_chats"}),
        (_chats_not_a_list, {"teams.list_chats"}),
        (_no_teams, {"teams.list_teams"}),
        (_other_channel, {"teams.list_seed_channels"}),
        (_extra_channel, {"teams.list_seed_channels"}),
        (_newer_message, {"teams.read_seed_message"}),
        (_edited_message, {"teams.read_seed_message"}),
        (_html_message, set()),
        (_extra_paragraph, {"word.list_paragraphs"}),
        (_other_first_paragraph, {"word.read_fixture", "word.list_paragraphs"}),
        (_other_paragraph_text, {"word.read_fixture", "word.list_paragraphs"}),
        (_extra_worksheet, {"excel.read_fixture"}),
        (_renamed_worksheet, set(CHECKS["excel"])),
        (_quantity_as_text, {"excel.read_seed_range"}),
        (_other_item, {"excel.read_seed_range"}),
        (_extra_row, {"excel.read_used_range"}),
        (_formatted_cell_outside, set()),
        (
            _extra_slide,
            {"powerpoint.read_fixture", "powerpoint.read_presentation_text"},
        ),
        (
            _other_slide_text,
            {"powerpoint.read_slide_text", "powerpoint.read_presentation_text"},
        ),
        (
            _extra_text_box,
            {"powerpoint.read_slide_text", "powerpoint.read_presentation_text"},
        ),
        (_speaker_notes, {"powerpoint.read_presentation_text"}),
    ],
)
async def test_a_seed_that_drifted_fails_exactly_the_checks_that_read_it(
    tenant: Tenant, drift: Callable[[Tenant], None], failing: set[str]
) -> None:
    drift(tenant)

    outcomes = _outcomes(await _run())

    failed = {check for check, row in outcomes.items() if row != PASS}
    assert failed == failing
    assert all(outcomes[check][:2] == ("fail", "assertion_failed") for check in failed)


@pytest.mark.parametrize(
    ("row", "column"), [(row, column) for row in range(3) for column in range(2)]
)
async def test_each_seeded_cell_is_read_back(
    tenant: Tenant, row: int, column: int
) -> None:
    tenant.worksheets[SHEET][row][column] = "changed"

    outcomes = _outcomes(await _run())

    assert {check for check, row_ in outcomes.items() if row_ != PASS} == {
        "excel.read_seed_range"
    }


async def test_a_message_id_fixture_must_be_a_string(tenant: Tenant) -> None:
    """Graph returns a message id as a string of digits, and Equals compares
    as JSON does, so a registry that records the id as a number fails the
    check: ``teams.seed_message_id`` is a string."""
    fixtures = {**FIXTURES, "teams.seed_message_id": int(MESSAGE_ID)}

    outcomes = _outcomes(await _run(fixtures=fixtures))

    assert outcomes["teams.read_seed_message"] == (
        "fail",
        "assertion_failed",
        "equals",
    )
