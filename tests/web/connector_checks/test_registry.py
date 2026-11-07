"""Every connector's V2 declaration: declared, or derived from its L3 probe."""

from __future__ import annotations

import inspect
import io
import json
from collections.abc import Iterator, Mapping
from typing import Any
from unittest.mock import Mock

import pytest
from docx import Document
from pptx import Presentation

from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
from xagent.web.connector_checks.declarations import (
    ConnectorChecks,
    FixtureRef,
    HasKeys,
    L3Envelope,
    ReadCheck,
    StatusSuccess,
    UnprobeableCheck,
    validate_connector_checks,
)
from xagent.web.connector_checks.registry import (
    CONNECTOR_CHECKS,
    connector_checks_for,
    l3_read_check,
)
from xagent.web.tools.mcp import (
    excel,
    google_docs,
    google_sheets,
    google_slides,
    powerpoint,
    word,
)

# The six document connectors, and the read each one declares: the tool, the
# one required argument its fixture binds, the fixture key, and the keys only
# a successful read of the object returns.
_DOCUMENT_READS: Mapping[str, tuple[str, str, str, tuple[str, ...]]] = {
    "excel": (
        "excel_list_worksheets",
        "file_path",
        "excel.file_path",
        ("worksheets", "truncated"),
    ),
    "google-docs": (
        "google_docs_get_document",
        "document_id",
        "google-docs.document_id",
        ("document_id", "title"),
    ),
    "google-sheets": (
        "google_sheets_get_spreadsheet",
        "spreadsheet_id",
        "google-sheets.spreadsheet_id",
        ("spreadsheet_id", "title", "sheets"),
    ),
    "google-slides": (
        "google_slides_get_presentation",
        "presentation_id",
        "google-slides.presentation_id",
        ("presentation_id", "title", "slides"),
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
    "google-docs": google_docs,
    "google-sheets": google_sheets,
    "google-slides": google_slides,
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
    assert connector_checks_for("gmail") == ConnectorChecks(
        "gmail",
        reads=(
            ReadCheck(
                "gmail.l3_probe",
                "gmail_search_messages",
                {"max_results": 1},
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


def test_every_builtin_catalog_app_has_a_valid_declaration() -> None:
    app_ids = sorted(row["app_id"] for row in get_builtin_public_mcp_app_rows())

    assert app_ids
    for app_id in app_ids:
        checks = connector_checks_for(app_id)
        assert checks is not None, app_id
        assert checks.app_id == app_id
        validate_connector_checks(checks)


def test_declared_v2_checks_are_valid_and_keyed_by_their_app() -> None:
    for app_id, checks in CONNECTOR_CHECKS.items():
        assert checks.app_id == app_id
        validate_connector_checks(checks)


@pytest.mark.parametrize("app_id", sorted(_DOCUMENT_READS))
def test_a_document_connector_declares_its_fixture_read(app_id: str) -> None:
    assert connector_checks_for(app_id) == _expected_checks(app_id)


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


def test_google_docs_read_fixture_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    service.documents.return_value.get.return_value.execute.return_value = {
        "documentId": "doc-1",
        "title": "Fixture doc",
        "body": {"content": []},
    }
    monkeypatch.setattr(google_docs, "get_docs_service", lambda: service)

    _assert_declared_output(
        "google-docs", google_docs.google_docs_get_document("doc-1")
    )


def test_google_sheets_read_fixture_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    service.spreadsheets.return_value.get.return_value.execute.return_value = {
        "spreadsheetId": "sheet-1",
        "properties": {"title": "Fixture workbook"},
        "sheets": [{"properties": {"sheetId": 0, "title": "Sheet1"}}],
    }
    monkeypatch.setattr(google_sheets, "get_sheets_service", lambda: service)

    _assert_declared_output(
        "google-sheets",
        google_sheets.google_sheets_get_spreadsheet("sheet-1"),
    )


def test_google_slides_read_fixture_output_satisfies_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = Mock()
    service.presentations.return_value.get.return_value.execute.return_value = {
        "presentationId": "deck-1",
        "title": "Fixture deck",
        "slides": [{"objectId": "slide-1", "pageElements": []}],
    }
    monkeypatch.setattr(google_slides, "get_slides_service", lambda: service)

    _assert_declared_output(
        "google-slides",
        google_slides.google_slides_get_presentation("deck-1"),
    )


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
