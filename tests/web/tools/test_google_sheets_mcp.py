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


class _HttpResponse:
    def __init__(self, status: int):
        self.status = status
        self.reason = "error"


def _http_error(status: int, body: dict):
    from googleapiclient.errors import HttpError

    return HttpError(
        _HttpResponse(status),
        json.dumps(body).encode("utf-8"),
        uri="https://sheets.googleapis.com/v4/spreadsheets/sid?alt=json",
    )


def test_get_spreadsheet_maps_not_found_to_an_actionable_message(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.get.return_value.execute.side_effect = _http_error(
        404,
        {"error": {"code": 404, "message": "Requested entity was not found."}},
    )
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_get_spreadsheet("sid"))

    assert result["status"] == "error"
    message = result["message"]
    assert message.startswith("Google Sheets could not open this spreadsheet")
    assert "google_sheets_create_spreadsheet" in message
    assert message.endswith("HTTP 404 Requested entity was not found.")


def test_read_range_maps_permission_denied_to_an_actionable_message(monkeypatch):
    spreadsheets = Mock()
    spreadsheets.values.return_value.get.return_value.execute.side_effect = _http_error(
        403,
        {
            "error": {
                "code": 403,
                "message": "The caller does not have permission",
                "status": "PERMISSION_DENIED",
            }
        },
    )
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_read_range("sid", "Sheet1"))

    assert result["message"].startswith("Google Sheets could not open this spreadsheet")


def test_read_range_keeps_raw_error_for_a_scope_403(monkeypatch):
    spreadsheets = Mock()
    error = _http_error(
        403,
        {
            "error": {
                "code": 403,
                "message": "Request had insufficient authentication scopes.",
                "details": [{"reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}],
            }
        },
    )
    spreadsheets.values.return_value.get.return_value.execute.side_effect = error
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(google_sheets.google_sheets_read_range("sid", "Sheet1"))

    assert result["message"] == str(error)


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: google_sheets.google_sheets_get_spreadsheet("Budget 2026"),
        lambda: google_sheets.google_sheets_read_range("Budget 2026", "Sheet1"),
        lambda: google_sheets.google_sheets_append_rows(
            "Budget 2026", "Sheet1!A1", [["x"]]
        ),
    ],
)
def test_id_taking_tools_reject_a_title_without_calling_the_api(monkeypatch, invoke):
    get_service = Mock()
    monkeypatch.setattr(google_sheets, "get_sheets_service", get_service)

    result = json.loads(invoke())

    assert result["status"] == "error"
    message = result["message"]
    assert "'Budget 2026' is not a Google Sheets link" in message
    assert "cannot search for or list spreadsheets by name" in message
    assert "https://docs.google.com/spreadsheets/d/" in message
    get_service.assert_not_called()


def test_get_spreadsheet_rejects_a_published_link_instead_of_reading_e_as_the_id(
    monkeypatch,
):
    get_service = Mock()
    monkeypatch.setattr(google_sheets, "get_sheets_service", get_service)

    result = json.loads(
        google_sheets.google_sheets_get_spreadsheet(
            "https://docs.google.com/spreadsheets/d/e/2PACX-1vQabc123/pubhtml"
        )
    )

    assert result["status"] == "error"
    assert "is a published-to-the-web link" in result["message"]
    get_service.assert_not_called()


_VIEW_ONLY_403 = {
    "error": {
        "code": 403,
        "message": "The caller does not have permission",
        "status": "PERMISSION_DENIED",
    }
}

# (tool call, the mocked request whose execute() raises) for each tool that
# changes a spreadsheet.
_EDITING_CALLS = [
    pytest.param(
        lambda: google_sheets.google_sheets_update_range("sid", "Sheet1!A1", [["x"]]),
        lambda spreadsheets: spreadsheets.values.return_value.update.return_value,
        id="update_range",
    ),
    pytest.param(
        lambda: google_sheets.google_sheets_append_rows("sid", "Sheet1!A1", [["x"]]),
        lambda spreadsheets: spreadsheets.values.return_value.append.return_value,
        id="append_rows",
    ),
    pytest.param(
        lambda: google_sheets.google_sheets_clear_range("sid", "Sheet1!A1:B2"),
        lambda spreadsheets: spreadsheets.values.return_value.clear.return_value,
        id="clear_range",
    ),
    pytest.param(
        lambda: google_sheets.google_sheets_add_sheet("sid", "Q4"),
        lambda spreadsheets: spreadsheets.batchUpdate.return_value,
        id="add_sheet",
    ),
    pytest.param(
        lambda: google_sheets.google_sheets_delete_sheet("sid", 7),
        lambda spreadsheets: spreadsheets.batchUpdate.return_value,
        id="delete_sheet",
    ),
]


@pytest.mark.parametrize(("invoke", "request_of"), _EDITING_CALLS)
def test_editing_tools_explain_a_refused_edit(monkeypatch, invoke, request_of):
    spreadsheets = Mock()
    request_of(spreadsheets).execute.side_effect = _http_error(403, _VIEW_ONLY_403)
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(invoke())

    assert result["status"] == "error"
    message = result["message"]
    assert message.startswith("Google Sheets could not make this change")
    assert "only view or comment access" in message
    assert "can edit the spreadsheet" in message
    assert "could not open" not in message
    assert message.endswith("HTTP 403 The caller does not have permission")


@pytest.mark.parametrize(("invoke", "request_of"), _EDITING_CALLS)
def test_editing_tools_map_not_found_to_an_actionable_message(
    monkeypatch, invoke, request_of
):
    spreadsheets = Mock()
    request_of(spreadsheets).execute.side_effect = _http_error(
        404,
        {"error": {"code": 404, "message": "Requested entity was not found."}},
    )
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(invoke())

    assert result["message"].startswith("Google Sheets could not open this spreadsheet")


async def test_get_spreadsheet_description_says_drive_is_not_needed():
    tools = {tool.name: tool for tool in await google_sheets.mcp.list_tools()}

    description = " ".join(tools["google_sheets_get_spreadsheet"].description.split())
    assert "cannot search for or list spreadsheets by name" in description
    assert "Connecting Google Drive is not needed" in description
    assert (
        "use google_drive_search to find its id if that tool is available"
        in description
    )
    assert "otherwise, or if it finds nothing, ask the user to paste the link" in (
        description
    )


# What the Sheets API returns for an Excel file stored in Drive, such as one
# opened from a ".../spreadsheets/d/<id>/edit?rtpof=true" link.
_OFFICE_FILE_400 = {
    "error": {
        "code": 400,
        "message": (
            "This operation is not supported for this document. The document "
            "must not be an Office file."
        ),
        "status": "FAILED_PRECONDITION",
    }
}


def _assert_explains_an_excel_file(message):
    # Google Sheets itself opens the file (in Office compatibility mode, which
    # is where an "rtpof=true" link comes from); only its API cannot.
    assert message.startswith(
        "The Google Sheets tools cannot open this file: it is most likely an "
        "Excel file (.xlsx) stored in Google Drive rather than a Google Sheets "
        "spreadsheet. Google Sheets can open such a file in Office "
        "compatibility mode, but the API these tools use cannot."
    )
    assert "File > Save as Google Sheets" in message
    assert "is a Google Docs or Google Slides file instead" in message
    assert message.endswith(
        "Google API response: HTTP 400 This operation is not supported for this "
        "document. The document must not be an Office file."
    )


@pytest.mark.parametrize(
    ("invoke", "request_of"),
    [
        pytest.param(
            lambda: google_sheets.google_sheets_get_spreadsheet(
                "https://docs.google.com/spreadsheets/d/abc123/edit"
                "?usp=sharing&rtpof=true&sd=true"
            ),
            lambda spreadsheets: spreadsheets.get.return_value,
            id="get_spreadsheet",
        ),
        pytest.param(
            lambda: google_sheets.google_sheets_read_range("sid", "Sheet1"),
            lambda spreadsheets: spreadsheets.values.return_value.get.return_value,
            id="read_range",
        ),
    ],
)
def test_read_tools_explain_an_excel_file_stored_in_drive(
    monkeypatch, invoke, request_of
):
    spreadsheets = Mock()
    request_of(spreadsheets).execute.side_effect = _http_error(400, _OFFICE_FILE_400)
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(invoke())

    assert result["status"] == "error"
    message = result["message"]
    _assert_explains_an_excel_file(message)
    assert "download the file with google_drive_download_file" in message
    assert "read the downloaded copy with read_file" in message
    assert "works only if the Google Drive connection can access the file" in message
    assert "attach the file to their message" in message


@pytest.mark.parametrize(("invoke", "request_of"), _EDITING_CALLS)
def test_editing_tools_explain_an_excel_file_stored_in_drive(
    monkeypatch, invoke, request_of
):
    spreadsheets = Mock()
    request_of(spreadsheets).execute.side_effect = _http_error(400, _OFFICE_FILE_400)
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(invoke())

    assert result["status"] == "error"
    message = result["message"]
    _assert_explains_an_excel_file(message)
    # Reading a downloaded copy cannot complete a change, and saving as a
    # Google Sheet leaves the Excel file unchanged.
    assert "To make this change with these tools, ask the user to open it" in (message)
    assert (
        "this creates a new spreadsheet: the change will be made in that copy, "
        "and the original file will stay unchanged"
    ) in message
    assert "google_drive_download_file" not in message


# A 400 that names its cause is about the request, also for an id taken
# from a Drive open?id= or uc?id= link.
@pytest.mark.parametrize(
    "spreadsheet_id",
    [
        "sid",
        "https://drive.google.com/open?id=sid",
        "https://drive.google.com/uc?export=download&id=sid",
    ],
)
def test_read_range_keeps_raw_error_for_other_400(monkeypatch, spreadsheet_id):
    spreadsheets = Mock()
    error = _http_error(
        400,
        {
            "error": {
                "code": 400,
                "message": "Unable to parse range: Sheet9!A1",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    spreadsheets.values.return_value.get.return_value.execute.side_effect = error
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(
        google_sheets.google_sheets_read_range(spreadsheet_id, "Sheet9!A1")
    )

    get = spreadsheets.values.return_value.get
    assert get.call_args.kwargs["spreadsheetId"] == "sid"
    assert result["message"] == str(error)


# An older Drive share link or a Drive download link can name a native
# spreadsheet as well as an Excel file, so its id goes to the API, which
# decides.
def test_get_spreadsheet_opens_a_native_spreadsheet_from_a_drive_open_link(
    monkeypatch,
):
    spreadsheets = Mock()
    spreadsheets.get.return_value.execute.return_value = {
        "spreadsheetId": "abc123",
        "properties": {"title": "Budget"},
        "sheets": [{"properties": {"sheetId": 0, "title": "Sheet1"}}],
    }
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(
        google_sheets.google_sheets_get_spreadsheet(
            "https://drive.google.com/open?id=abc123"
        )
    )

    assert result["status"] == "success"
    assert spreadsheets.get.call_args.kwargs["spreadsheetId"] == "abc123"


@pytest.mark.parametrize(
    ("invoke", "request_of", "editing"),
    [
        pytest.param(
            lambda: google_sheets.google_sheets_read_range(
                "https://drive.google.com/open?id=abc123", "Sheet1"
            ),
            lambda spreadsheets: spreadsheets.values.return_value.get,
            False,
            id="read_range-open-link",
        ),
        pytest.param(
            lambda: google_sheets.google_sheets_update_range(
                "https://drive.google.com/uc?export=download&id=abc123",
                "Sheet1!A1",
                [["x"]],
            ),
            lambda spreadsheets: spreadsheets.values.return_value.update,
            True,
            id="update_range-uc-link",
        ),
    ],
)
def test_tools_explain_an_excel_file_named_by_a_drive_open_or_uc_link(
    monkeypatch, invoke, request_of, editing
):
    spreadsheets = Mock()
    request_of(spreadsheets).return_value.execute.side_effect = _http_error(
        400, _OFFICE_FILE_400
    )
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(invoke())

    assert result["status"] == "error"
    assert request_of(spreadsheets).call_args.kwargs["spreadsheetId"] == "abc123"
    message = result["message"]
    _assert_explains_an_excel_file(message)
    assert ("read the downloaded copy with read_file" in message) is not editing
    assert ("To make this change with these tools" in message) is editing


@pytest.mark.parametrize(
    "link",
    [
        "https://drive.google.com/file/d/abc123/view?usp=sharing",
        "https://docs.google.com/file/d/abc123/edit",
        "https://drive.usercontent.google.com/download?id=abc123&export=download",
    ],
)
def test_get_spreadsheet_explains_a_drive_file_link_without_calling_the_api(
    monkeypatch, link
):
    get_service = Mock()
    monkeypatch.setattr(google_sheets, "get_sheets_service", get_service)

    result = json.loads(google_sheets.google_sheets_get_spreadsheet(link))

    assert result["status"] == "error"
    message = result["message"]
    assert "is a Google Drive file link, not a Google Sheets link" in message
    assert "such as an Excel file (.xlsx)" in message
    assert "download the file with google_drive_download_file" in message
    assert "works only if the Google Drive connection can access the file" in message
    assert "File > Save as Google Sheets" in message
    assert "https://docs.google.com/spreadsheets/d/..." in message
    get_service.assert_not_called()


@pytest.mark.parametrize(
    ("invoke", "editing"),
    [
        pytest.param(
            lambda sid: google_sheets.google_sheets_read_range(sid, "Sheet1!A1"),
            False,
            id="read_range",
        ),
        pytest.param(
            lambda sid: google_sheets.google_sheets_update_range(
                sid, "Sheet1!A1", [["x"]]
            ),
            True,
            id="update_range",
        ),
        pytest.param(
            lambda sid: google_sheets.google_sheets_append_rows(
                sid, "Sheet1!A1", [["x"]]
            ),
            True,
            id="append_rows",
        ),
        pytest.param(
            lambda sid: google_sheets.google_sheets_clear_range(sid, "Sheet1!A1"),
            True,
            id="clear_range",
        ),
        pytest.param(
            lambda sid: google_sheets.google_sheets_add_sheet(sid, "Q4"),
            True,
            id="add_sheet",
        ),
        pytest.param(
            lambda sid: google_sheets.google_sheets_delete_sheet(sid, 7),
            True,
            id="delete_sheet",
        ),
    ],
)
def test_tools_give_their_own_next_steps_for_a_drive_file_link(
    monkeypatch, invoke, editing
):
    get_service = Mock()
    monkeypatch.setattr(google_sheets, "get_sheets_service", get_service)

    result = json.loads(invoke("https://drive.google.com/file/d/abc123/view"))

    assert result["status"] == "error"
    message = result["message"]
    assert "is a Google Drive file link, not a Google Sheets link" in message
    assert "File > Save as Google Sheets" in message
    # A tool that changes the file gets the same steps as for the Office-file
    # error from the API: a downloaded or attached copy can only be read, and
    # saving as a Google Sheet leaves the Excel file unchanged.
    assert ("read the downloaded copy with read_file" in message) is not editing
    assert ("attach the file to their message" in message) is not editing
    assert ("To make this change with these tools" in message) is editing
    assert (
        "this creates a new spreadsheet: the change will be made in that copy, "
        "and the original file will stay unchanged" in message
    ) is editing
    get_service.assert_not_called()


# The Sheets API answers an Excel file with the Office-file 400 (above). For
# any other 400 after a Drive open?id= or uc?id= link, which can name an
# uploaded file such as a PDF, every tool must pass its own argument to the
# error message, which then adds the next steps for an uploaded file.
@pytest.mark.parametrize(
    "link",
    [
        "https://drive.google.com/open?id=abc123",
        "https://drive.google.com/uc?export=download&id=abc123",
    ],
)
@pytest.mark.parametrize(
    ("request_of", "invoke", "editing"),
    [
        pytest.param(
            lambda sp: sp.get,
            google_sheets.google_sheets_get_spreadsheet,
            False,
            id="get_spreadsheet",
        ),
        pytest.param(
            lambda sp: sp.values.return_value.get,
            lambda sid: google_sheets.google_sheets_read_range(sid, "Sheet1!A1"),
            False,
            id="read_range",
        ),
        pytest.param(
            lambda sp: sp.values.return_value.update,
            lambda sid: google_sheets.google_sheets_update_range(
                sid, "Sheet1!A1", [["x"]]
            ),
            True,
            id="update_range",
        ),
        pytest.param(
            lambda sp: sp.values.return_value.append,
            lambda sid: google_sheets.google_sheets_append_rows(
                sid, "Sheet1!A1", [["x"]]
            ),
            True,
            id="append_rows",
        ),
        pytest.param(
            lambda sp: sp.values.return_value.clear,
            lambda sid: google_sheets.google_sheets_clear_range(sid, "Sheet1!A1"),
            True,
            id="clear_range",
        ),
        pytest.param(
            lambda sp: sp.batchUpdate,
            lambda sid: google_sheets.google_sheets_add_sheet(sid, "Extra"),
            True,
            id="add_sheet",
        ),
        pytest.param(
            lambda sp: sp.batchUpdate,
            lambda sid: google_sheets.google_sheets_delete_sheet(sid, 1),
            True,
            id="delete_sheet",
        ),
    ],
)
def test_tools_add_next_steps_to_another_400_for_a_drive_open_or_uc_link(
    monkeypatch, link, request_of, invoke, editing
):
    spreadsheets = Mock()
    request_of(spreadsheets).return_value.execute.side_effect = _http_error(
        400,
        {
            "error": {
                "code": 400,
                "message": "Request contains an invalid argument.",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    _mock_sheets_service(monkeypatch, spreadsheets)

    result = json.loads(invoke(link))

    assert result["status"] == "error"
    assert request_of(spreadsheets).call_args.kwargs["spreadsheetId"] == "abc123"
    message = result["message"]
    assert message.startswith(
        "Google Sheets rejected this request (Google API response: HTTP 400 "
        "Request contains an invalid argument.). The spreadsheet id was taken "
        "from a Google Drive link"
    )
    assert "File > Save as Google Sheets" in message
    assert ("read the downloaded copy with read_file" in message) is not editing
    assert ("To make this change with these tools" in message) is editing
    assert message.endswith("the error is about the request itself.")
