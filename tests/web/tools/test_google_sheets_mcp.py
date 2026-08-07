import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import google_sheets


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    # tests/conftest.py force-loads the project .env into the test process, so
    # any refresh credentials configured there would leak into the "absent"
    # assertions below.
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)


def _mock_sheets_service(monkeypatch, spreadsheets_mock):
    service = Mock()
    service.spreadsheets.return_value = spreadsheets_mock
    monkeypatch.setattr(google_sheets, "get_sheets_service", lambda: service)
    return service


def test_get_credentials_requires_access_token(monkeypatch):
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="GOOGLE_ACCESS_TOKEN"):
        google_sheets._get_credentials()


def test_get_credentials_omits_refresh_fields_when_absent():
    creds = google_sheets._get_credentials()

    assert creds.token == "access-token"
    assert creds.refresh_token is None


def test_get_credentials_includes_refresh_fields_when_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "refresh-token")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")

    creds = google_sheets._get_credentials()

    assert creds.refresh_token == "refresh-token"
    assert creds.client_id == "client-id"
    assert creds.client_secret == "client-secret"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("abc123", "abc123"),
        ("https://docs.google.com/spreadsheets/d/abc123/edit#gid=0", "abc123"),
        ("https://docs.google.com/spreadsheets/u/1/d/abc123/edit", "abc123"),
    ],
)
def test_resolve_spreadsheet_id_accepts_bare_id_and_url_forms(value, expected):
    assert google_sheets._resolve_spreadsheet_id(value) == expected


_URL_WITH_ID = "https://docs.google.com/spreadsheets/d/abc123/edit#gid=0"


@pytest.mark.parametrize(
    ("mock_target", "invoke"),
    [
        (
            lambda sp: sp.get,
            lambda sid: google_sheets.google_sheets_get_spreadsheet(sid),
        ),
        (
            lambda sp: sp.values.return_value.get,
            lambda sid: google_sheets.google_sheets_read_range(sid, "Sheet1!A1"),
        ),
        (
            lambda sp: sp.values.return_value.update,
            lambda sid: google_sheets.google_sheets_update_range(
                sid, "Sheet1!A1", [["x"]]
            ),
        ),
        (
            lambda sp: sp.values.return_value.append,
            lambda sid: google_sheets.google_sheets_append_rows(
                sid, "Sheet1!A1", [["x"]]
            ),
        ),
        (
            lambda sp: sp.values.return_value.clear,
            lambda sid: google_sheets.google_sheets_clear_range(sid, "Sheet1!A1"),
        ),
        (
            lambda sp: sp.batchUpdate,
            lambda sid: google_sheets.google_sheets_add_sheet(sid, "Extra"),
        ),
        (
            lambda sp: sp.batchUpdate,
            lambda sid: google_sheets.google_sheets_delete_sheet(sid, 1),
        ),
    ],
    ids=[
        "get_spreadsheet",
        "read_range",
        "update_range",
        "append_rows",
        "clear_range",
        "add_sheet",
        "delete_sheet",
    ],
)
def test_id_taking_tools_resolve_full_spreadsheet_urls(
    monkeypatch, mock_target, invoke
):
    """Every id-taking tool calls _resolve_spreadsheet_id before hitting the
    API, but the other tests here all pass an already-bare id, so none of
    them would notice if that call were deleted from a tool. Drive this
    through the real call sites with a full URL and assert the *resolved*
    id is what reached the mocked API call, for each of the 7 tools."""
    spreadsheets = Mock()
    _mock_sheets_service(monkeypatch, spreadsheets)

    invoke(_URL_WITH_ID)

    assert mock_target(spreadsheets).call_args.kwargs["spreadsheetId"] == "abc123"


def test_get_spreadsheet_defaults_missing_grid_properties(monkeypatch):
    """A non-grid sheet (e.g. a chart sheet) can omit gridProperties entirely;
    parsing must not raise KeyError/AttributeError on that shape."""
    spreadsheets = Mock()
    spreadsheets.get.return_value.execute.return_value = {
        "spreadsheetId": "sid",
        "properties": {"title": "My Sheet"},
        "spreadsheetUrl": "https://example.com/sid",
        "sheets": [{"properties": {"sheetId": 0, "title": "Sheet1"}}],
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_get_spreadsheet("sid"))

    assert result["status"] == "success"
    assert result["sheets"] == [
        {"sheet_id": 0, "title": "Sheet1", "row_count": None, "column_count": None}
    ]


def test_get_spreadsheet_returns_error_payload_on_failure(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.get.return_value.execute.side_effect = RuntimeError("boom")
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_get_spreadsheet("sid"))

    assert result["status"] == "error"
    assert "boom" in result["message"]


def test_create_spreadsheet_without_parent(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.create.return_value.execute.return_value = {
        "spreadsheetId": "sid",
        "properties": {"title": "New"},
        "spreadsheetUrl": "https://example.com/sid",
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_create_spreadsheet("New"))

    assert result["status"] == "success"
    assert result["spreadsheet_id"] == "sid"


def test_create_spreadsheet_moves_to_parent_and_removes_previous_parents(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.create.return_value.execute.return_value = {
        "spreadsheetId": "sid",
        "properties": {"title": "New"},
        "spreadsheetUrl": "https://example.com/sid",
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    drive = Mock()
    drive.files.return_value.get.return_value.execute.return_value = {
        "parents": ["root"]
    }
    monkeypatch.setattr(google_sheets, "get_drive_service", lambda: drive)

    result = json.loads(
        google_sheets.google_sheets_create_spreadsheet("New", parent_id="folder123")
    )

    assert result["status"] == "success"
    update_kwargs = drive.files.return_value.update.call_args.kwargs
    assert update_kwargs["addParents"] == "folder123"
    assert update_kwargs["removeParents"] == "root"


def test_create_spreadsheet_returns_partial_status_when_move_fails(monkeypatch):
    """A failed Drive move must not lose the id of the already-created
    spreadsheet: the tool must return it with a "partial" status so the
    caller can recover the file instead of it being silently orphaned in
    the Drive root with no id to retry against."""
    spreadsheets = Mock()
    spreadsheets.create.return_value.execute.return_value = {
        "spreadsheetId": "sid",
        "properties": {"title": "New"},
        "spreadsheetUrl": "https://example.com/sid",
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    drive = Mock()
    drive.files.return_value.get.return_value.execute.side_effect = RuntimeError(
        "insufficient permission"
    )
    monkeypatch.setattr(google_sheets, "get_drive_service", lambda: drive)

    result = json.loads(
        google_sheets.google_sheets_create_spreadsheet("New", parent_id="folder123")
    )

    assert result["status"] == "partial"
    assert result["spreadsheet_id"] == "sid"
    assert "insufficient permission" in result["message"]


def test_create_spreadsheet_returns_error_payload_when_create_fails(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.create.return_value.execute.side_effect = RuntimeError(
        "quota exceeded"
    )
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_create_spreadsheet("New"))

    assert result["status"] == "error"
    assert "quota exceeded" in result["message"]


def test_read_range_returns_values(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.values.return_value.get.return_value.execute.return_value = {
        "range": "Sheet1!A1:B2",
        "values": [["a", "b"], ["c", "d"]],
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_read_range("sid", "Sheet1!A1:B2"))

    assert result["status"] == "success"
    assert result["values"] == [["a", "b"], ["c", "d"]]
    assert result["truncated"] is False


def test_read_range_defaults_missing_values_key(monkeypatch):
    """A range with no data at all omits the "values" key entirely rather
    than sending an empty array; the tool's .get("values", []) default must
    produce an empty list, not raise."""
    spreadsheets = Mock()
    spreadsheets.values.return_value.get.return_value.execute.return_value = {
        "range": "Sheet1!A1:B2"
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_read_range("sid", "Sheet1!A1:B2"))

    assert result["status"] == "success"
    assert result["values"] == []
    assert result["truncated"] is False


def test_read_range_caps_rows_and_flags_truncated(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.values.return_value.get.return_value.execute.return_value = {
        "range": "Sheet1",
        "values": [[str(i)] for i in range(5)],
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(
        google_sheets.google_sheets_read_range("sid", "Sheet1", max_rows=3)
    )

    assert result["status"] == "success"
    assert result["values"] == [["0"], ["1"], ["2"]]
    assert result["truncated"] is True


@pytest.mark.parametrize("bad_max_rows", [0, -1])
def test_read_range_rejects_non_positive_max_rows(monkeypatch, bad_max_rows):
    """A negative max_rows would otherwise flow into values[:max_rows] and
    silently drop rows from the *end* (values[:-1]) instead of capping —
    reject it before the API call like google_analytics does for limit."""
    spreadsheets = Mock()
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(
        google_sheets.google_sheets_read_range("sid", "Sheet1", max_rows=bad_max_rows)
    )

    assert result["status"] == "error"
    assert "max_rows" in result["message"]
    spreadsheets.values.return_value.get.assert_not_called()


def test_update_range_sends_values_as_2d_array(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.values.return_value.update.return_value.execute.return_value = {
        "updatedRange": "Sheet1!A1:B2",
        "updatedCells": 4,
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(
        google_sheets.google_sheets_update_range(
            "sid", "Sheet1!A1:B2", [["a", "b"], ["c", "d"]]
        )
    )

    assert result["status"] == "success"
    assert result["updated_cells"] == 4
    body = spreadsheets.values.return_value.update.call_args.kwargs["body"]
    assert body == {"values": [["a", "b"], ["c", "d"]]}


async def test_update_range_validation_via_mcp_layer(monkeypatch):
    """The 2D shape contract moved from a manual isinstance check into the
    values: list[list[Any]] signature, so it is enforced by FastMCP's
    validation layer — which direct function calls bypass. Exercise the real
    call path: a 1D list must be rejected before the tool body runs, and a
    JSON-encoded string (a common LLM behavior) must be pre-parsed into the
    2D array rather than passed through as a string."""
    from mcp.server.fastmcp.exceptions import ToolError

    spreadsheets = Mock()
    spreadsheets.values.return_value.update.return_value.execute.return_value = {
        "updatedRange": "Sheet1!A1:B2",
        "updatedCells": 2,
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    with pytest.raises(ToolError, match="validation error"):
        await google_sheets.mcp.call_tool(
            "google_sheets_update_range",
            {"spreadsheet_id": "sid", "range_name": "Sheet1!A1", "values": ["a", "b"]},
        )
    spreadsheets.values.return_value.update.assert_not_called()

    await google_sheets.mcp.call_tool(
        "google_sheets_update_range",
        {
            "spreadsheet_id": "sid",
            "range_name": "Sheet1!A1:B2",
            "values": '[["a","b"]]',
        },
    )
    body = spreadsheets.values.return_value.update.call_args.kwargs["body"]
    assert body == {"values": [["a", "b"]]}


def test_update_range_returns_error_payload_on_failure(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.values.return_value.update.return_value.execute.side_effect = (
        RuntimeError("bad range")
    )
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(
        google_sheets.google_sheets_update_range("sid", "Sheet1!A1", [["a"]])
    )

    assert result["status"] == "error"
    assert "bad range" in result["message"]


def test_append_rows_sends_values_as_2d_array(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.values.return_value.append.return_value.execute.return_value = {
        "updates": {"updatedRange": "Sheet1!A3:B4", "updatedRows": 2},
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(
        google_sheets.google_sheets_append_rows("sid", "Sheet1!A1", [["e", "f"]])
    )

    assert result["status"] == "success"
    assert result["updated_rows"] == 2
    body = spreadsheets.values.return_value.append.call_args.kwargs["body"]
    assert body == {"values": [["e", "f"]]}


def test_append_rows_defaults_missing_updates_key(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.values.return_value.append.return_value.execute.return_value = {}
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(
        google_sheets.google_sheets_append_rows("sid", "Sheet1!A1", [["e", "f"]])
    )

    assert result["status"] == "success"
    assert result["updated_range"] == "Sheet1!A1"
    assert result["updated_rows"] == 0


def test_clear_range(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.values.return_value.clear.return_value.execute.return_value = {
        "clearedRange": "Sheet1!A1:D10"
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_clear_range("sid", "Sheet1!A1:D10"))

    assert result["status"] == "success"
    assert result["cleared_range"] == "Sheet1!A1:D10"


def test_clear_range_defaults_missing_cleared_range_key(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.values.return_value.clear.return_value.execute.return_value = {}
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_clear_range("sid", "Sheet1!A1:D10"))

    assert result["status"] == "success"
    assert result["cleared_range"] == "Sheet1!A1:D10"


def test_add_sheet_returns_new_sheet_properties(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.batchUpdate.return_value.execute.return_value = {
        "replies": [{"addSheet": {"properties": {"sheetId": 42, "title": "Extra"}}}]
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_add_sheet("sid", "Extra"))

    assert result["status"] == "success"
    assert result["sheet_id"] == 42
    assert result["title"] == "Extra"


def test_add_sheet_handles_missing_replies_without_raising(monkeypatch):
    """An unexpected/empty replies list must surface as sheet_id=None rather
    than an unguarded KeyError/IndexError bubbling out as a confusing
    {"message": "'replies'"} error payload."""
    spreadsheets = Mock()
    spreadsheets.batchUpdate.return_value.execute.return_value = {}
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_add_sheet("sid", "Extra"))

    assert result["status"] == "success"
    assert result["sheet_id"] is None
    assert result["title"] is None


def test_delete_sheet(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.batchUpdate.return_value.execute.return_value = {}
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_delete_sheet("sid", 42))

    assert result["status"] == "success"
    assert "42" in result["message"]


def test_delete_sheet_returns_error_payload_on_failure(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.batchUpdate.return_value.execute.side_effect = RuntimeError(
        "no such sheet"
    )
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_delete_sheet("sid", 42))

    assert result["status"] == "error"
    assert "no such sheet" in result["message"]
