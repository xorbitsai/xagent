"""Every connector's V2 declaration: declared, or derived from its L3 probe."""

from __future__ import annotations

import inspect
import io
import json
from collections.abc import Callable, Iterator, Mapping
from typing import Any
from unittest.mock import Mock

import pytest
from docx import Document
from pptx import Presentation

from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
from xagent.web.connector_checks.assertions import judge_result, resolve_expected
from xagent.web.connector_checks.declarations import (
    Assertion,
    ConnectorChecks,
    Contains,
    DeclarationError,
    EachHasKeys,
    Equals,
    FixtureRef,
    HasKeys,
    IsType,
    L3Envelope,
    Length,
    ReadCheck,
    StatusSuccess,
    UnprobeableCheck,
    validate_connector_checks,
)
from xagent.web.connector_checks.registry import (
    CONNECTOR_CHECKS,
    WRITE_WITHOUT_CLEANUP,
    WRITE_WITHOUT_REVERT,
    _declared_l3,
    connector_checks_for,
    l3_read_check,
)
from xagent.web.connector_checks.results import FailureClass
from xagent.web.tools.mcp import calendar as google_calendar
from xagent.web.tools.mcp import (
    excel,
    gmail,
    google_analytics,
    google_docs,
    google_drive,
    google_sheets,
    google_slides,
    powerpoint,
    word,
    zendesk,
    zoom,
)

# The three Microsoft document connectors, and the read each one declares: the
# tool, the one required argument its fixture binds, the fixture key, and the
# keys only a successful read of the object returns.
_DOCUMENT_READS: Mapping[str, tuple[str, str, str, tuple[str, ...]]] = {
    "excel": (
        "excel_list_worksheets",
        "file_path",
        "excel.file_path",
        ("worksheets", "truncated"),
    ),
    "powerpoint": (
        "powerpoint_list_slides",
        "file_path",
        "powerpoint.file_path",
        ("slides", "etag"),
    ),
    "word": (
        "word_get_document_text",
        "file_path",
        "word.file_path",
        ("text", "generation"),
    ),
}

_APP_MODULES = {
    "excel": excel,
    "powerpoint": powerpoint,
    "word": word,
}


def _expected_checks(app_id: str) -> ConnectorChecks:
    tool, argument, fixture_key, assertion_keys = _DOCUMENT_READS[app_id]
    return ConnectorChecks(
        app_id,
        reads=(
            ReadCheck(
                f"{app_id}.read_fixture",
                tool,
                {argument: FixtureRef(fixture_key)},
                assertions=(StatusSuccess(), HasKeys((), assertion_keys)),
            ),
        ),
    )


def _fixture_refs(value: Any) -> Iterator[FixtureRef]:
    if isinstance(value, FixtureRef):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _fixture_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _fixture_refs(item)


def _assert_declared_output(app_id: str, result: str) -> None:
    """The tool's JSON output satisfies its declared assertions.

    The engine that evaluates declarations is a follow-up, so the two
    assertion kinds the document connectors declare are checked here directly.
    """
    checks = CONNECTOR_CHECKS[app_id]
    payload = json.loads(result)
    assert len(checks.reads) == 1
    for assertion in checks.reads[0].assertions:
        if isinstance(assertion, StatusSuccess):
            assert payload.get("status") == "success"
        elif isinstance(assertion, HasKeys):
            target = payload
            for segment in assertion.path:
                target = target[segment]
            assert isinstance(target, dict)
            assert set(assertion.keys) <= set(target)
        else:
            raise AssertionError(f"unhandled assertion {assertion!r} for {app_id}")


class _GraphResponse:
    """The subset of ``requests.Response`` the Graph tools read."""

    def __init__(
        self,
        *,
        json_data: Any = None,
        content: bytes = b"",
        status_code: int = 200,
    ) -> None:
        self._json_data = json_data
        self.content = (
            json.dumps(json_data).encode("utf-8") if json_data is not None else content
        )
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return self._json_data

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> _GraphResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


def _docx_bytes() -> bytes:
    document = Document()
    document.add_paragraph("Fixture document")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _pptx_bytes() -> bytes:
    presentation = Presentation()
    presentation.slides.add_slide(presentation.slide_layouts[0])
    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def test_a_probed_connector_gets_its_l3_probe_as_one_read_check() -> None:
    assert connector_checks_for("slack") == ConnectorChecks(
        "slack",
        reads=(
            ReadCheck(
                "slack.l3_probe",
                "slack_list_channels",
                {"limit": 1},
                assertions=(L3Envelope(),),
            ),
        ),
    )


def test_the_l3_expect_keys_carry_over() -> None:
    checks = connector_checks_for("linkedin")

    assert checks is not None
    assert checks.reads == (
        ReadCheck(
            "linkedin.l3_probe", "get_profile", assertions=(L3Envelope(("sub",)),)
        ),
    )


@pytest.mark.parametrize(
    ("app_id", "reason"),
    [
        ("hubspot", "covered_by_l2"),
        ("google-maps", "sandboxed"),
        ("chrome-devtools", "execution_scoped"),
    ],
)
def test_an_unprobeable_connector_gets_an_unprobeable_slot(
    app_id: str, reason: str
) -> None:
    assert connector_checks_for(app_id) == ConnectorChecks(
        app_id,
        reads=(UnprobeableCheck(f"{app_id}.l3_probe", reason),),  # type: ignore[arg-type]
    )


def test_an_unknown_app_has_no_declaration() -> None:
    assert connector_checks_for("not-a-catalog-app") is None


@pytest.mark.parametrize(
    ("app_id", "check"),
    [
        (
            "notion",
            ReadCheck(
                "notion.current_user",
                "notion-get-self",
                payload="json_text",
                assertions=(
                    HasKeys((), ("workspace", "user")),
                    HasKeys(("user",), ("id",)),
                ),
                require_read_only_hint=True,
            ),
        ),
        (
            "miro",
            ReadCheck(
                "miro.current_user",
                "user_who_am_i",
                payload="structured_content",
                assertions=(HasKeys((), ("user_id",)),),
                require_read_only_hint=True,
            ),
        ),
        (
            "atlassian",
            ReadCheck(
                "atlassian.current_user",
                "atlassianUserInfo",
                payload="json_text",
                assertions=(HasKeys(("data",), ("accountId",)),),
                require_read_only_hint=True,
            ),
        ),
    ],
)
def test_a_vendor_hosted_read_check_is_declared_exactly(
    app_id: str, check: ReadCheck
) -> None:
    assert connector_checks_for(app_id) == ConnectorChecks(app_id, reads=(check,))


def test_every_remote_mcp_read_check_requires_the_read_only_hint() -> None:
    remote_app_ids = {
        row["app_id"]
        for row in get_builtin_public_mcp_app_rows()
        if "url" in row["launch_config"]
    }
    assert {"notion", "miro", "atlassian"} <= remote_app_ids & set(CONNECTOR_CHECKS)
    for app_id in remote_app_ids & set(CONNECTOR_CHECKS):
        for check in CONNECTOR_CHECKS[app_id].reads:
            if isinstance(check, ReadCheck):
                assert check.require_read_only_hint, (app_id, check.id)


def test_every_builtin_catalog_app_has_a_valid_declaration() -> None:
    app_ids = sorted(row["app_id"] for row in get_builtin_public_mcp_app_rows())

    assert app_ids
    for app_id in app_ids:
        checks = connector_checks_for(app_id)
        assert checks is not None, app_id
        assert checks.app_id == app_id
        validate_connector_checks(checks)


def test_declared_v2_checks_are_valid_and_keyed_by_their_app() -> None:
    app_ids = {row["app_id"] for row in get_builtin_public_mcp_app_rows()}

    assert set(CONNECTOR_CHECKS) <= app_ids
    for app_id, checks in CONNECTOR_CHECKS.items():
        assert checks.app_id == app_id
        validate_connector_checks(checks)


@pytest.mark.parametrize("app_id", sorted(_DOCUMENT_READS))
def test_a_document_connector_declares_its_fixture_read(app_id: str) -> None:
    # Its other tools are exempted; test_tool_coverage.py covers those.
    assert connector_checks_for(app_id).reads == _expected_checks(app_id).reads


def test_every_document_fixture_key_is_namespaced_by_its_app() -> None:
    refs = [
        (app_id, ref)
        for app_id in _DOCUMENT_READS
        for check in CONNECTOR_CHECKS[app_id].reads
        if isinstance(check, ReadCheck)
        for ref in _fixture_refs(check.arguments)
    ]

    # Every document connector contributes at least one fixture.
    assert set(_DOCUMENT_READS) <= {app_id for app_id, _ in refs}
    for app_id, ref in refs:
        assert ref.key.startswith(f"{app_id}."), (app_id, ref.key)


@pytest.mark.parametrize("app_id", sorted(_DOCUMENT_READS))
def test_the_probe_route_still_needs_a_fixture(app_id: str) -> None:
    assert l3_read_check(app_id) == UnprobeableCheck(
        f"{app_id}.l3_probe", "needs_fixture"
    )


@pytest.mark.parametrize("app_id", sorted(_DOCUMENT_READS))
def test_the_fixture_is_bound_to_a_required_parameter(app_id: str) -> None:
    """The tool is the module attribute of its own name: the in-repo servers
    register each ``@mcp.tool`` function under that name, and the decorator
    returns the function itself, so its signature is the tool's schema."""
    module = _APP_MODULES[app_id]
    for check in CONNECTOR_CHECKS[app_id].reads:
        assert isinstance(check, ReadCheck)
        tool = getattr(module, check.tool)
        parameters = inspect.signature(tool).parameters
        bound = [
            name
            for name, value in check.arguments.items()
            if isinstance(value, FixtureRef)
        ]
        assert bound
        for name in bound:
            assert name in parameters
            assert parameters[name].default is inspect.Parameter.empty
        required = {
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        }
        assert required <= set(check.arguments), (check.tool, required)


def test_word_read_fixture_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "51200")
    response = _GraphResponse(content=_docx_bytes())
    monkeypatch.setattr(word.requests, "request", Mock(return_value=response))

    _assert_declared_output("word", word.word_get_document_text("Report.docx"))


def test_excel_read_fixture_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")
    response = _GraphResponse(json_data={"value": [{"id": "ws-1", "name": "Sheet1"}]})
    monkeypatch.setattr(excel.requests, "request", Mock(return_value=response))

    _assert_declared_output("excel", excel.excel_list_worksheets("Book.xlsx"))


def test_powerpoint_read_fixture_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")
    content = _pptx_bytes()
    responses = iter(
        [
            _GraphResponse(
                json_data={
                    "id": "item-1",
                    "size": len(content),
                    "eTag": '"abc"',
                    "parentReference": {"driveId": "drive-1"},
                }
            ),
            _GraphResponse(content=content),
        ]
    )
    monkeypatch.setattr(
        powerpoint.requests, "request", Mock(side_effect=lambda **_: next(responses))
    )

    _assert_declared_output(
        "powerpoint", powerpoint.powerpoint_list_slides("Deck.pptx")
    )


# The Tier 2 read checks (issue #3573): the tool, its arguments, and every
# assertion, so dropping an item key or a whole assertion fails here.
_TIER_2_READS: Mapping[str, tuple[str, dict[str, Any], tuple[Assertion, ...]]] = {
    "zendesk.list_tickets": (
        "zendesk_list_tickets",
        {"limit": 1},
        (
            StatusSuccess(),
            HasKeys((), ("tickets", "has_more", "after_cursor", "truncated")),
            EachHasKeys(
                ("tickets",),
                (
                    "id",
                    "subject",
                    "status",
                    "priority",
                    "requester_id",
                    "assignee_id",
                    "group_id",
                    "tags",
                    "created_at",
                    "updated_at",
                ),
            ),
            Length(("tickets",), max=1),
        ),
    ),
    "zendesk.list_organizations": (
        "zendesk_list_organizations",
        {"limit": 1},
        (
            StatusSuccess(),
            HasKeys((), ("organizations", "has_more", "after_cursor", "truncated")),
            EachHasKeys(
                ("organizations",), ("id", "name", "domain_names", "created_at")
            ),
            Length(("organizations",), max=1),
        ),
    ),
    "zendesk.list_macros": (
        "zendesk_list_macros",
        {"limit": 1},
        (
            StatusSuccess(),
            HasKeys((), ("macros", "has_more", "after_cursor", "truncated")),
            EachHasKeys(("macros",), ("id", "title", "description", "updated_at")),
            Length(("macros",), max=1),
        ),
    ),
    "zoom.list_meetings": (
        "zoom_list_meetings",
        {},
        (
            StatusSuccess(),
            HasKeys((), ("meetings", "next_page_token")),
            IsType(("meetings",), "array"),
        ),
    ),
}


def _read_check(check_id: str) -> ReadCheck:
    app_id = check_id.split(".", 1)[0]
    check = next(c for c in CONNECTOR_CHECKS[app_id].reads if c.id == check_id)
    assert isinstance(check, ReadCheck)
    return check


def _assert_check_output(check_id: str, result: str) -> None:
    check = _read_check(check_id)
    raw = {
        "content": [{"type": "text", "text": result}],
        "structured_content": None,
        "is_error": False,
    }
    assertions = resolve_expected(check.assertions, _FIXTURES)
    assert judge_result(raw, check.payload, assertions) == (None, None)


def _assert_check_fails(check_id: str, result: str, reason: str) -> None:
    check = _read_check(check_id)
    raw = {
        "content": [{"type": "text", "text": result}],
        "structured_content": None,
        "is_error": False,
    }
    assertions = resolve_expected(check.assertions, _FIXTURES)
    assert judge_result(raw, check.payload, assertions) == (
        FailureClass.ASSERTION_FAILED,
        reason,
    )


@pytest.mark.parametrize(("check_id", "expected"), sorted(_TIER_2_READS.items()))
def test_a_tier_2_read_check_is_declared_exactly(
    check_id: str, expected: tuple[str, dict[str, Any], tuple[Assertion, ...]]
) -> None:
    tool, arguments, assertions = expected
    check = _read_check(check_id)
    assert (check.tool, dict(check.arguments)) == (tool, arguments)
    assert check.assertions == assertions


@pytest.mark.parametrize("app_id", ["zendesk", "zoom"])
def test_a_tier_2_l3_check_is_the_probe_derived_one(app_id: str) -> None:
    expected = l3_read_check(app_id)
    assert expected is not None

    l3_checks = [
        check
        for check in CONNECTOR_CHECKS[app_id].reads
        if check.id == f"{app_id}.l3_probe"
    ]
    assert l3_checks == [expected]


@pytest.mark.parametrize("app_id", ["acme", "hubspot"])
def test_the_declared_l3_helper_rejects_anything_but_a_read_check(app_id: str) -> None:
    with pytest.raises(DeclarationError, match=app_id):
        _declared_l3(app_id)


def test_zoom_create_meeting_has_no_cleanup_tool() -> None:
    assert [
        exemption.tool
        for exemption in CONNECTOR_CHECKS["zoom"].exemptions
        if exemption.reason == WRITE_WITHOUT_CLEANUP
    ] == ["zoom_create_meeting"]


def _list_payload(list_key: str, item: dict[str, Any], count: int) -> dict[str, Any]:
    return {list_key: [item] * count, "meta": {"has_more": False}}


@pytest.mark.parametrize(
    ("check_id", "list_key", "item", "tool"),
    [
        (
            "zendesk.list_tickets",
            "tickets",
            {
                "id": 7,
                "subject": "Printer down",
                "status": "open",
                "priority": "normal",
                "requester_id": 3,
                "tags": ["hardware"],
            },
            zendesk.zendesk_list_tickets,
        ),
        (
            "zendesk.list_organizations",
            "organizations",
            {"id": 4, "name": "Acme", "domain_names": ["acme.example"]},
            zendesk.zendesk_list_organizations,
        ),
        (
            "zendesk.list_macros",
            "macros",
            {"id": 5, "title": "Close ticket", "description": "Closes it"},
            zendesk.zendesk_list_macros,
        ),
    ],
)
@pytest.mark.parametrize("count", [0, 1])
def test_a_zendesk_list_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
    check_id: str,
    list_key: str,
    item: dict[str, Any],
    tool: Callable[..., str],
    count: int,
) -> None:
    monkeypatch.setattr(
        zendesk,
        "_request",
        lambda *args, **kwargs: _list_payload(list_key, item, count),
    )

    _assert_check_output(check_id, tool(limit=1))


@pytest.mark.parametrize("count", [0, 1])
def test_zoom_list_meetings_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    monkeypatch.setattr(
        zoom,
        "_request",
        lambda *args, **kwargs: {
            "meetings": [{"id": 111, "topic": "Sync"}] * count,
            "next_page_token": "",
        },
    )

    _assert_check_output("zoom.list_meetings", zoom.zoom_list_meetings())


# The Google family (issue #3567): the read checks of the tenant the seed
# script (xorbitsai/xagent-saas#2533) fills, with the tool, its arguments and
# every assertion, so dropping an argument, an assertion or an expected seed
# value fails here. ``google-calendar`` and ``google-analytics`` keep their
# probe-derived L3 call instead (``_declared_l3``).
_GOOGLE_READS: Mapping[str, tuple[str, dict[str, Any], tuple[Assertion, ...]]] = {
    "gmail.search_seed_message": (
        "gmail_search_messages",
        {"label_ids": [FixtureRef("gmail.seed_label_id")], "max_results": 10},
        (
            StatusSuccess(),
            HasKeys((), ("messages",)),
            Length(("messages",), 1, 1),
            EachHasKeys(
                ("messages",), ("id", "threadId", "snippet", "subject", "from")
            ),
            Equals(("messages", 0, "subject"), "xagent smoke seed message"),
        ),
    ),
    "google-analytics.run_report": (
        "google_analytics_run_report",
        {
            "property_id": FixtureRef("google-analytics.property_id"),
            "metrics": ["sessions"],
            "date_ranges": [{"start_date": "7daysAgo", "end_date": "yesterday"}],
        },
        (
            StatusSuccess(),
            HasKeys(
                (),
                (
                    "query",
                    "metric_headers",
                    "totals",
                    "rows",
                    "row_count",
                    "returned_count",
                    "has_more",
                    "truncated",
                ),
            ),
            HasKeys(
                ("query",),
                ("property_id", "metrics", "date_ranges", "limit", "offset"),
            ),
            Equals(
                ("query", "property_id"), FixtureRef("google-analytics.property_id")
            ),
            IsType(("rows",), "array"),
        ),
    ),
    "google-docs.read_fixture": (
        "google_docs_get_document",
        {"document_id": FixtureRef("google-docs.document_id")},
        (
            StatusSuccess(),
            HasKeys((), ("document_id", "title", "content")),
            Equals(("document_id",), FixtureRef("google-docs.document_id")),
            Equals(("title",), "xagent-smoke-seed-doc"),
            Contains(("content",), "xagent smoke seed document"),
        ),
    ),
    "google-drive.read_seed_file": (
        "google_drive_get_file_content",
        {"file_id": FixtureRef("google-docs.document_id")},
        (
            StatusSuccess(),
            HasKeys((), ("file", "content", "encoding", "truncated")),
            HasKeys(("file",), ("id", "name", "mimeType")),
            Equals(("file", "id"), FixtureRef("google-docs.document_id")),
            Equals(("file", "name"), "xagent-smoke-seed-doc"),
            Equals(("encoding",), "utf-8"),
            Contains(("content",), "xagent smoke seed document"),
        ),
    ),
    "google-drive.search_seed_folder": (
        "google_drive_search",
        {
            "query": (
                "name = 'xagent-smoke-seed' and "
                "mimeType = 'application/vnd.google-apps.folder' and "
                "trashed = false"
            ),
            "max_results": 10,
        },
        (
            StatusSuccess(),
            HasKeys((), ("files", "truncated")),
            Length(("files",), 1, 1),
            EachHasKeys(("files",), ("id", "name", "mimeType", "modifiedTime")),
            Equals(("files", 0, "id"), FixtureRef("google-drive.seed_folder_id")),
        ),
    ),
    "google-sheets.read_fixture": (
        "google_sheets_get_spreadsheet",
        {"spreadsheet_id": FixtureRef("google-sheets.spreadsheet_id")},
        (
            StatusSuccess(),
            HasKeys((), ("spreadsheet_id", "title", "url", "sheets")),
            Equals(("spreadsheet_id",), FixtureRef("google-sheets.spreadsheet_id")),
            Equals(("title",), "xagent-smoke-seed-sheet"),
            Length(("sheets",), min=1),
            EachHasKeys(
                ("sheets",), ("sheet_id", "title", "row_count", "column_count")
            ),
        ),
    ),
    "google-sheets.read_seed_range": (
        "google_sheets_read_range",
        {
            "spreadsheet_id": FixtureRef("google-sheets.spreadsheet_id"),
            "range_name": "A1:B3",
        },
        (
            StatusSuccess(),
            HasKeys((), ("range", "values", "truncated")),
            Length(("values",), 3, 3),
            Equals(("values", 0, 0), "item"),
            Equals(("values", 0, 1), "quantity"),
            Equals(("values", 1, 0), "alpha"),
            Equals(("values", 1, 1), "1"),
            Equals(("values", 2, 0), "beta"),
            Equals(("values", 2, 1), "2"),
        ),
    ),
    "google-slides.read_fixture": (
        "google_slides_get_presentation",
        {"presentation_id": FixtureRef("google-slides.presentation_id")},
        (
            StatusSuccess(),
            HasKeys((), ("presentation_id", "title", "slide_count", "slides")),
            Equals(("presentation_id",), FixtureRef("google-slides.presentation_id")),
            Equals(("title",), "xagent-smoke-seed-slides"),
            IsType(("slide_count",), "number"),
        ),
    ),
}

_GOOGLE_APP_MODULES = {
    "gmail": gmail,
    "google-analytics": google_analytics,
    "google-calendar": google_calendar,
    "google-docs": google_docs,
    "google-drive": google_drive,
    "google-sheets": google_sheets,
    "google-slides": google_slides,
}

# The fixture keys the tenant registry (xorbitsai/xagent-saas#2627) records for
# the Google seed layout; every one a Google read check uses is here.
_GOOGLE_FIXTURE_KEYS = frozenset(
    {
        "gmail.seed_label_id",
        "google-analytics.property_id",
        "google-docs.document_id",
        "google-drive.seed_folder_id",
        "google-sheets.spreadsheet_id",
        "google-slides.presentation_id",
    }
)


class _Fixtures:
    """The Google seed's values, as the identity adapter would resolve them."""

    _VALUES = {
        "gmail.seed_label_id": "Label_1",
        "google-analytics.property_id": "563334905",
        "google-docs.document_id": "doc-1",
        "google-drive.seed_folder_id": "folder-1",
        "google-sheets.spreadsheet_id": "sheet-1",
        "google-slides.presentation_id": "deck-1",
    }

    def resolve_fixture(self, key: str) -> str | None:
        return self._VALUES.get(key)


_FIXTURES = _Fixtures()


def _google_fixture_keys(check: ReadCheck) -> Iterator[FixtureRef]:
    yield from _fixture_refs(check.arguments)
    for assertion in check.assertions:
        if isinstance(assertion, Equals):
            yield from _fixture_refs(assertion.value)


@pytest.mark.parametrize(("check_id", "expected"), sorted(_GOOGLE_READS.items()))
def test_a_google_read_check_is_declared_exactly(
    check_id: str, expected: tuple[str, dict[str, Any], tuple[Assertion, ...]]
) -> None:
    tool, arguments, assertions = expected
    check = _read_check(check_id)
    assert (check.tool, dict(check.arguments), check.assertions) == (
        tool,
        arguments,
        assertions,
    )


@pytest.mark.parametrize(
    ("app_id", "check_ids"),
    [
        ("gmail", ["gmail.search_seed_message"]),
        (
            "google-analytics",
            ["google-analytics.l3_probe", "google-analytics.run_report"],
        ),
        ("google-calendar", ["google-calendar.l3_probe"]),
        ("google-docs", ["google-docs.read_fixture"]),
        (
            "google-drive",
            ["google-drive.search_seed_folder", "google-drive.read_seed_file"],
        ),
        (
            "google-sheets",
            ["google-sheets.read_fixture", "google-sheets.read_seed_range"],
        ),
        ("google-slides", ["google-slides.read_fixture"]),
    ],
)
def test_a_google_connector_declares_exactly_its_reads(
    app_id: str, check_ids: list[str]
) -> None:
    assert [check.id for check in CONNECTOR_CHECKS[app_id].reads] == check_ids


@pytest.mark.parametrize("app_id", ["google-analytics", "google-calendar"])
def test_a_google_l3_check_is_the_probe_derived_one(app_id: str) -> None:
    expected = l3_read_check(app_id)
    assert expected is not None

    l3_checks = [
        check
        for check in CONNECTOR_CHECKS[app_id].reads
        if check.id == f"{app_id}.l3_probe"
    ]
    assert l3_checks == [expected]


@pytest.mark.parametrize("app_id", ["google-docs", "google-sheets", "google-slides"])
def test_a_google_document_probe_still_needs_a_fixture(app_id: str) -> None:
    assert l3_read_check(app_id) == UnprobeableCheck(
        f"{app_id}.l3_probe", "needs_fixture"
    )


def test_every_google_fixture_key_is_one_the_seed_records() -> None:
    keys = {
        ref.key
        for app_id in _GOOGLE_APP_MODULES
        for check in CONNECTOR_CHECKS[app_id].reads
        if isinstance(check, ReadCheck)
        for ref in _google_fixture_keys(check)
    }

    assert keys == _GOOGLE_FIXTURE_KEYS


@pytest.mark.parametrize("app_id", sorted(_GOOGLE_APP_MODULES))
def test_a_google_fixture_is_bound_to_a_parameter_of_its_tool(app_id: str) -> None:
    """A fixture in the arguments names a parameter of the tool, at any depth
    (Gmail's is the one item of ``label_ids``), and every required argument is
    pinned: an omitted one would make the tool prompt for it."""
    module = _GOOGLE_APP_MODULES[app_id]
    for check in CONNECTOR_CHECKS[app_id].reads:
        if not isinstance(check, ReadCheck):
            continue
        tool = getattr(module, check.tool)
        parameters = inspect.signature(tool).parameters
        bound = [
            name for name, value in check.arguments.items() if any(_fixture_refs(value))
        ]
        for name in bound:
            assert name in parameters, (check.tool, name)
        required = {
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        }
        assert required <= set(check.arguments), (check.tool, required)


def test_a_cleanable_write_keeps_a_pending_exemption() -> None:
    """The writes whose own cleanup tool can delete what they create wait for
    round two (#3567); none carries the permanent no-cleanup reason."""
    cleanable = (
        "google_calendar_create_events",
        "google_calendar_delete_events",
        "google_drive_create_file",
        "google_drive_create_folder",
        "google_drive_delete_file",
        "google_sheets_add_sheet",
        "google_sheets_delete_sheet",
        "google_slides_add_slide",
        "google_slides_delete_slide",
    )
    exemptions = {
        exemption.tool: exemption.reason
        for app_id in (
            "google-calendar",
            "google-drive",
            "google-sheets",
            "google-slides",
        )
        for exemption in CONNECTOR_CHECKS[app_id].exemptions
    }

    for tool in cleanable:
        assert exemptions[tool] != WRITE_WITHOUT_CLEANUP, tool
        assert exemptions[tool].endswith("decided in #3567"), tool


# Each Google app's write exemptions that carry a permanent reason instead of a
# pending one. A connector whose writes can all be cleaned up has none.
_GOOGLE_PERMANENT_WRITE_REASONS: Mapping[str, Mapping[str, str]] = {
    "gmail": {"gmail_send_messages": WRITE_WITHOUT_CLEANUP},
    "google-analytics": {},
    "google-calendar": {},
    "google-docs": {
        "google_docs_append_text": WRITE_WITHOUT_REVERT,
        "google_docs_batch_update": WRITE_WITHOUT_REVERT,
        "google_docs_create_document": WRITE_WITHOUT_CLEANUP,
        "google_docs_replace_text": WRITE_WITHOUT_REVERT,
    },
    "google-drive": {},
    "google-sheets": {"google_sheets_create_spreadsheet": WRITE_WITHOUT_CLEANUP},
    "google-slides": {
        "google_slides_create_presentation": WRITE_WITHOUT_CLEANUP,
        "google_slides_import_pptx": WRITE_WITHOUT_CLEANUP,
    },
}


@pytest.mark.parametrize("app_id", sorted(_GOOGLE_APP_MODULES))
def test_each_permanent_write_reason_names_its_own_tools(app_id: str) -> None:
    actual = {
        exemption.tool: exemption.reason
        for exemption in CONNECTOR_CHECKS[app_id].exemptions
        if exemption.reason in (WRITE_WITHOUT_CLEANUP, WRITE_WITHOUT_REVERT)
    }

    assert actual == _GOOGLE_PERMANENT_WRITE_REASONS[app_id], app_id


def test_gmail_manage_labels_is_exempt_for_the_read_step_rule() -> None:
    exemption = next(
        exemption
        for exemption in CONNECTOR_CHECKS["gmail"].exemptions
        if exemption.tool == "gmail_manage_labels"
    )

    assert exemption.reason == (
        "read and write in one tool: it is the only Gmail tool that reads a "
        "label and the only one that creates or deletes one, so a label write "
        "check's verify or cleanup guard would call its own create and cleanup "
        "tool, which #3653 refuses; no label-reading tool is added, which "
        "#3567 allows"
    )


# Output samples for the Google read checks


class _GmailBatch:
    """The batch object ``gmail_search_messages`` drives: it calls back once
    per request with the response its test gave it."""

    def __init__(
        self, callback: Callable[..., None], responses: list[dict[str, Any]]
    ) -> None:
        self._callback = callback
        self._responses = responses

    def add(self, request: Any) -> None:
        return None

    def execute(self) -> None:
        for index, response in enumerate(self._responses):
            self._callback(str(index), response, None)


def _gmail_service(responses: list[dict[str, Any]]) -> Mock:
    service = Mock()
    service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": f"m{index}"} for index in range(len(responses))]
    }
    service.new_batch_http_request.side_effect = lambda callback: _GmailBatch(
        callback, responses
    )
    return service


def _gmail_message(subject: str = "xagent smoke seed message") -> dict[str, Any]:
    return {
        "id": "msg-1",
        "threadId": "thread-1",
        "snippet": "Seeded message for the Gmail read checks.",
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": "seed@xagent-smoke.invalid"},
            ]
        },
    }


def test_gmail_search_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gmail, "get_gmail_service", lambda: _gmail_service([_gmail_message()])
    )

    _assert_check_output(
        "gmail.search_seed_message",
        gmail.gmail_search_messages(label_ids=["Label_1"], max_results=10),
    )


@pytest.mark.parametrize(
    ("messages", "reason"),
    [
        ([_gmail_message(), _gmail_message()], "length"),
        ([_gmail_message("another subject")], "equals"),
    ],
)
def test_gmail_search_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, Any]], reason: str
) -> None:
    monkeypatch.setattr(gmail, "get_gmail_service", lambda: _gmail_service(messages))

    _assert_check_fails(
        "gmail.search_seed_message",
        gmail.gmail_search_messages(label_ids=["Label_1"], max_results=10),
        reason,
    )


_DRIVE_FOLDER = {
    "id": "folder-1",
    "name": "xagent-smoke-seed",
    "mimeType": "application/vnd.google-apps.folder",
    "modifiedTime": "2026-11-01T00:00:00.000Z",
}


def _drive_list_service(files: list[dict[str, Any]]) -> Mock:
    service = Mock()
    service.files.return_value.list.return_value.execute.return_value = {"files": files}
    return service


def _drive_search() -> str:
    return google_drive.google_drive_search(
        query=(
            "name = 'xagent-smoke-seed' and "
            "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        ),
        max_results=10,
    )


def test_google_drive_search_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_drive, "get_drive_service", lambda: _drive_list_service([_DRIVE_FOLDER])
    )

    _assert_check_output("google-drive.search_seed_folder", _drive_search())


@pytest.mark.parametrize(
    ("files", "reason"),
    [
        ([], "length"),
        ([{**_DRIVE_FOLDER, "id": "other"}], "equals"),
        (
            [
                {
                    key: value
                    for key, value in _DRIVE_FOLDER.items()
                    if key != "modifiedTime"
                }
            ],
            "each_has_keys",
        ),
    ],
)
def test_google_drive_search_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch, files: list[dict[str, Any]], reason: str
) -> None:
    monkeypatch.setattr(
        google_drive, "get_drive_service", lambda: _drive_list_service(files)
    )

    _assert_check_fails("google-drive.search_seed_folder", _drive_search(), reason)


def _drive_file_service() -> Mock:
    service = Mock()
    service.files.return_value.get.return_value.execute.return_value = {
        "id": "doc-1",
        "name": "xagent-smoke-seed-doc",
        "mimeType": "application/vnd.google-apps.document",
    }
    return service


def test_google_drive_file_content_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_drive,
        "get_drive_service",
        lambda: _drive_file_service(),
    )
    monkeypatch.setattr(
        google_drive,
        "_download_media",
        lambda request: b"xagent smoke seed document\nSecond line: seeded content\n",
    )

    _assert_check_output(
        "google-drive.read_seed_file",
        google_drive.google_drive_get_file_content("doc-1"),
    )


def test_google_drive_file_content_output_fails_without_the_seed_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_drive,
        "get_drive_service",
        lambda: _drive_file_service(),
    )
    monkeypatch.setattr(
        google_drive, "_download_media", lambda request: b"another document"
    )

    _assert_check_fails(
        "google-drive.read_seed_file",
        google_drive.google_drive_get_file_content("doc-1"),
        "contains",
    )


_DOCUMENT_TEXT = "xagent smoke seed document"


def _document_body(text: str = _DOCUMENT_TEXT) -> dict[str, Any]:
    return {
        "content": [
            {"paragraph": {"elements": [{"textRun": {"content": text + "\n"}}]}},
            {
                "paragraph": {
                    "elements": [
                        {"textRun": {"content": "Second line: seeded content\n"}}
                    ]
                }
            },
        ]
    }


def _docs_service(title: str, text: str = _DOCUMENT_TEXT) -> Mock:
    service = Mock()
    service.documents.return_value.get.return_value.execute.return_value = {
        "documentId": "doc-1",
        "title": title,
        "body": _document_body(text),
    }
    return service


def test_google_docs_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_docs, "get_docs_service", lambda: _docs_service("xagent-smoke-seed-doc")
    )

    _assert_check_output(
        "google-docs.read_fixture", google_docs.google_docs_get_document("doc-1")
    )


@pytest.mark.parametrize(
    ("title", "text", "reason"),
    [
        ("another doc", _DOCUMENT_TEXT, "equals"),
        ("xagent-smoke-seed-doc", "another document", "contains"),
    ],
)
def test_google_docs_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch, title: str, text: str, reason: str
) -> None:
    monkeypatch.setattr(
        google_docs, "get_docs_service", lambda: _docs_service(title, text)
    )

    _assert_check_fails(
        "google-docs.read_fixture",
        google_docs.google_docs_get_document("doc-1"),
        reason,
    )


def _seed_sheet() -> dict[str, Any]:
    return {
        "spreadsheetId": "sheet-1",
        "properties": {"title": "xagent-smoke-seed-sheet"},
        "spreadsheetUrl": "https://docs.google.com/spreadsheets/d/sheet-1",
        "sheets": [
            {
                "properties": {
                    "sheetId": 0,
                    "title": "Sheet1",
                    "gridProperties": {"rowCount": 1000, "columnCount": 26},
                }
            }
        ],
    }


def _spreadsheet_service(title: str) -> Mock:
    service = Mock()
    payload = {**_seed_sheet(), "properties": {"title": title}}
    service.spreadsheets.return_value.get.return_value.execute.return_value = payload
    return service


def test_google_sheets_spreadsheet_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_sheets,
        "get_sheets_service",
        lambda: _spreadsheet_service("xagent-smoke-seed-sheet"),
    )

    _assert_check_output(
        "google-sheets.read_fixture",
        google_sheets.google_sheets_get_spreadsheet("sheet-1"),
    )


@pytest.mark.parametrize(
    ("title", "reason"),
    [
        ("another book", "equals"),
        ("xagent-smoke-seed-sheet", "length"),
    ],
)
def test_google_sheets_spreadsheet_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch, title: str, reason: str
) -> None:
    service = _spreadsheet_service(title)
    if reason == "length":
        payload = (
            service.spreadsheets.return_value.get.return_value.execute.return_value
        )
        payload["sheets"] = []
    monkeypatch.setattr(google_sheets, "get_sheets_service", lambda: service)

    _assert_check_fails(
        "google-sheets.read_fixture",
        google_sheets.google_sheets_get_spreadsheet("sheet-1"),
        reason,
    )


def _values_service(values: list[list[str]]) -> Mock:
    service = Mock()
    service.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = {
        "range": "Sheet1!A1:B3",
        "values": values,
    }
    return service


def _read_seed_range() -> str:
    return google_sheets.google_sheets_read_range(
        spreadsheet_id="sheet-1", range_name="A1:B3"
    )


def test_google_sheets_range_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = [["item", "quantity"], ["alpha", "1"], ["beta", "2"]]
    monkeypatch.setattr(
        google_sheets, "get_sheets_service", lambda: _values_service(values)
    )

    _assert_check_output("google-sheets.read_seed_range", _read_seed_range())


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ([["item", "quantity"], ["alpha", "1"]], "length"),
        ([["item", "quantity"], ["alpha", "2"], ["beta", "2"]], "equals"),
    ],
)
def test_google_sheets_range_output_fails_on_a_bad_result(
    monkeypatch: pytest.MonkeyPatch, values: list[list[str]], reason: str
) -> None:
    monkeypatch.setattr(
        google_sheets, "get_sheets_service", lambda: _values_service(values)
    )

    _assert_check_fails("google-sheets.read_seed_range", _read_seed_range(), reason)


def _presentation_service(title: str) -> Mock:
    service = Mock()
    service.presentations.return_value.get.return_value.execute.return_value = {
        "presentationId": "deck-1",
        "title": title,
        "slides": [{"objectId": "slide-1", "pageElements": []}],
    }
    return service


def test_google_slides_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_slides,
        "get_slides_service",
        lambda: _presentation_service("xagent-smoke-seed-slides"),
    )

    _assert_check_output(
        "google-slides.read_fixture",
        google_slides.google_slides_get_presentation("deck-1"),
    )


def test_google_slides_output_fails_on_another_deck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_slides,
        "get_slides_service",
        lambda: _presentation_service("another deck"),
    )

    _assert_check_fails(
        "google-slides.read_fixture",
        google_slides.google_slides_get_presentation("deck-1"),
        "equals",
    )


def _analytics_report() -> dict[str, Any]:
    return {
        "dimensionHeaders": [],
        "metricHeaders": [{"name": "sessions", "type": "TYPE_INTEGER"}],
        "rows": [],
        "totals": [{"metricValues": [{"value": "0"}]}],
        "rowCount": 0,
    }


def _run_report(property_id: str) -> str:
    return google_analytics.google_analytics_run_report(
        property_id=property_id,
        metrics=["sessions"],
        date_ranges=[{"start_date": "7daysAgo", "end_date": "yesterday"}],
    )


def test_google_analytics_report_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_analytics, "_request", lambda *args, **kwargs: _analytics_report()
    )

    _assert_check_output("google-analytics.run_report", _run_report("563334905"))


def test_google_analytics_report_output_fails_on_another_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_analytics, "_request", lambda *args, **kwargs: _analytics_report()
    )

    _assert_check_fails(
        "google-analytics.run_report", _run_report("999999999"), "equals"
    )


def test_google_analytics_list_properties_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_analytics,
        "_request",
        lambda *args, **kwargs: {
            "accountSummaries": [
                {
                    "displayName": "Seed account",
                    "propertySummaries": [
                        {
                            "property": "properties/563334905",
                            "displayName": "xagent-smoke-seed",
                        }
                    ],
                }
            ]
        },
    )

    _assert_check_output(
        "google-analytics.l3_probe", google_analytics.google_analytics_list_properties()
    )


def test_google_analytics_list_properties_output_fails_on_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("unauthorized")

    monkeypatch.setattr(google_analytics, "_request", fail)

    _assert_check_fails(
        "google-analytics.l3_probe",
        google_analytics.google_analytics_list_properties(),
        "status_not_success",
    )


def _calendar_service(items: list[dict[str, Any]]) -> Mock:
    service = Mock()
    service.events.return_value.list.return_value.execute.return_value = {
        "timeZone": "UTC",
        "items": items,
    }
    return service


def test_google_calendar_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        google_calendar, "get_calendar_service", lambda: _calendar_service([])
    )

    _assert_check_output(
        "google-calendar.l3_probe",
        google_calendar.google_calendar_search_events(max_results=1),
    )


def test_google_calendar_output_fails_on_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    service.events.return_value.list.return_value.execute.side_effect = RuntimeError(
        "unauthorized"
    )
    monkeypatch.setattr(google_calendar, "get_calendar_service", lambda: service)

    _assert_check_fails(
        "google-calendar.l3_probe",
        google_calendar.google_calendar_search_events(max_results=1),
        "status_not_success",
    )
