import json
from unittest.mock import Mock

import pytest
import requests

from xagent.config import get_tool_max_output_length
from xagent.web.tools.mcp import excel


class MockResponse:
    def __init__(self, json_data=None, status_code=200, content=None, url=None):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = (
            json.dumps(self._json_data).encode("utf-8") if content is None else content
        )
        self.text = self.content.decode("utf-8", errors="replace")
        self.url = url or "https://graph.microsoft.com/v1.0/example"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Error for url: {self.url}",
                response=self,
            )


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")


# ---------------------------------------------------------------------------
# path/address helpers
# ---------------------------------------------------------------------------


def test_workbook_base_defaults_to_own_onedrive():
    assert (
        excel._workbook_base("Reports/Q1.xlsx", None, None)
        == "/me/drive/root:/Reports/Q1.xlsx:/workbook"
    )


def test_workbook_base_uses_drive_id_only():
    assert (
        excel._workbook_base("book.xlsx", None, "drive-1")
        == "/drives/drive-1/root:/book.xlsx:/workbook"
    )


def test_workbook_base_uses_site_and_drive():
    assert (
        excel._workbook_base("book.xlsx", "root", "drive-1")
        == "/sites/root/drives/drive-1/root:/book.xlsx:/workbook"
    )


def test_workbook_base_uses_site_default_drive():
    assert (
        excel._workbook_base("book.xlsx", "root", None)
        == "/sites/root/drive/root:/book.xlsx:/workbook"
    )


def test_workbook_base_terminates_path_form_site_before_default_drive():
    assert (
        excel._workbook_base("book.xlsx", "contoso.sharepoint.com:/teams/finance", None)
        == "/sites/contoso.sharepoint.com:/teams/finance:/drive/root:/book.xlsx:/workbook"
    )


def test_workbook_base_terminates_path_form_site_before_specific_drive():
    assert (
        excel._workbook_base(
            "book.xlsx", "contoso.sharepoint.com:/teams/finance", "drive-1"
        )
        == "/sites/contoso.sharepoint.com:/teams/finance:/drives/drive-1/root:/book.xlsx:/workbook"
    )


def test_workbook_base_rejects_dot_segments():
    with pytest.raises(ValueError, match="must not contain"):
        excel._workbook_base("../secret.xlsx", None, None)


def test_workbook_base_rejects_trailing_slash():
    with pytest.raises(ValueError, match="must include a filename"):
        excel._workbook_base("Reports/", None, None)


def test_workbook_base_rejects_consecutive_slashes():
    with pytest.raises(ValueError, match="empty segments"):
        excel._workbook_base("Reports//Q1.xlsx", None, None)


def test_workbook_base_rejects_trailing_period_filename():
    """A trailing-dot filename can be silently normalized by Graph/SharePoint's
    backing storage to the name without the dot, so "Report.xlsx." could
    silently resolve to a real, different "Report.xlsx" workbook."""
    with pytest.raises(ValueError, match="must not end with a period"):
        excel._workbook_base("Report.xlsx.", None, None)


def test_workbook_base_rejects_malicious_site_id():
    with pytest.raises(ValueError, match="must not contain"):
        excel._workbook_base("book.xlsx", "contoso.sharepoint.com:/../etc", None)


def test_workbook_base_rejects_empty_string_site_id():
    with pytest.raises(ValueError, match="site_id is required"):
        excel._workbook_base("book.xlsx", "", None)


def test_workbook_base_rejects_whitespace_only_drive_id():
    with pytest.raises(ValueError, match="drive_id"):
        excel._workbook_base("book.xlsx", None, "   ")


def test_workbook_base_rejects_whitespace_only_site_id():
    with pytest.raises(ValueError, match="site_id"):
        excel._workbook_base("book.xlsx", "   ", "drive-1")


def test_workbook_base_rejects_trailing_period_on_intermediate_segment():
    """The trailing-period hazard applies to every path segment, not just
    the filename -- "Reports./Q1.xlsx" must be rejected even though the
    filename itself is fine."""
    with pytest.raises(ValueError, match="must not end with a period"):
        excel._workbook_base("Reports./Q1.xlsx", None, None)


def test_workbook_base_rejects_segment_with_trailing_whitespace():
    with pytest.raises(ValueError, match="whitespace"):
        excel._workbook_base("Reports /Q1.xlsx", None, None)


def test_workbook_base_rejects_segment_with_leading_whitespace():
    with pytest.raises(ValueError, match="whitespace"):
        excel._workbook_base("Reports/ Q1.xlsx", None, None)


@pytest.mark.parametrize("path", [" Reports/Q1.xlsx", "Reports/Q1.xlsx "])
def test_workbook_base_rejects_boundary_whitespace(path):
    with pytest.raises(ValueError, match="whitespace"):
        excel._workbook_base(path, None, None)


def test_workbook_base_rejects_leading_separator():
    with pytest.raises(ValueError, match="must be relative"):
        excel._workbook_base("/Reports/Q1.xlsx", None, None)


def test_site_segment_rejects_empty_segment():
    with pytest.raises(ValueError, match="empty segments"):
        excel._site_segment("contoso.sharepoint.com:/a//b")


def test_odata_key_segment_escapes_quote():
    assert (
        excel._odata_key_segment("worksheets", "O'Brien")
        == "worksheets('O%27%27Brien')"
    )


def test_odata_key_segment_preserves_padded_worksheet_name():
    assert (
        excel._odata_key_segment("worksheets", " Sheet ") == "worksheets('%20Sheet%20')"
    )


def test_odata_string_literal_escapes_quote():
    assert excel._odata_string_literal("A1:B2") == "A1%3AB2"
    assert excel._odata_string_literal("O'Brien!A1") == "O%27%27Brien%21A1"


def test_normalize_relative_path_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        excel._normalize_relative_path(123)


def test_odata_string_literal_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        excel._odata_string_literal(123)


def test_odata_string_literal_rejects_empty_string():
    with pytest.raises(ValueError, match="address is required"):
        excel._odata_string_literal("")


def test_valid_clear_apply_to_is_immutable():
    assert isinstance(excel._VALID_CLEAR_APPLY_TO, frozenset)


def test_parse_values_json_requires_array_of_arrays():
    with pytest.raises(ValueError, match="array of arrays"):
        excel._parse_values_json('["a", "b"]')


def test_parse_values_json_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        excel._parse_values_json("{not json")


def test_parse_values_json_success():
    assert excel._parse_values_json('[["a", 1], ["b", 2]]') == [["a", 1], ["b", 2]]


def test_parse_values_json_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        excel._parse_values_json(123)


# ---------------------------------------------------------------------------
# worksheets
# ---------------------------------------------------------------------------


def test_list_worksheets_success(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"value": [{"id": "1", "name": "Sheet1"}]})
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_worksheets("book.xlsx"))

    assert result["status"] == "success"
    assert result["worksheets"] == [{"id": "1", "name": "Sheet1"}]
    assert mock_request.call_args.kwargs["url"].endswith(
        "/me/drive/root:/book.xlsx:/workbook/worksheets"
    )


def test_add_worksheet_with_name(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "1", "name": "NewSheet"}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_add_worksheet("book.xlsx", name="NewSheet"))

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["url"].endswith("/workbook/worksheets/add")
    assert kwargs["json"] == {"name": "NewSheet"}


def test_add_worksheet_without_name_sends_empty_body(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "1", "name": "Sheet2"}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_add_worksheet("book.xlsx"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"] == {}


@pytest.mark.parametrize("name", ["", "   "])
def test_add_worksheet_rejects_explicit_blank_name_without_request(monkeypatch, name):
    mock_request = Mock()
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_add_worksheet("book.xlsx", name=name))

    assert result["status"] == "error"
    assert "name" in result["message"]
    mock_request.assert_not_called()


def test_list_worksheets_exposes_next_link(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [{"id": "1", "name": "Sheet1"}],
                "@odata.nextLink": "https://graph.microsoft.com/v1.0/next-page",
            }
        )
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_worksheets("book.xlsx"))

    assert result["next_link"] == "https://graph.microsoft.com/v1.0/next-page"


def test_list_worksheets_next_link_is_none_on_last_page(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"value": [{"id": "1", "name": "Sheet1"}]})
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_worksheets("book.xlsx"))

    assert result["next_link"] is None


def test_list_worksheets_consumes_valid_next_link(monkeypatch):
    next_link = (
        "https://graph.microsoft.com/v1.0/me/drive/root:/book.xlsx:/workbook/"
        "worksheets?$skiptoken=page-2"
    )
    mock_request = Mock(return_value=MockResponse({"value": [{"id": "2"}]}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_worksheets("book.xlsx", next_link=next_link))

    assert result["worksheets"] == [{"id": "2"}]
    assert mock_request.call_args.kwargs["url"] == next_link


def test_list_worksheets_rejects_next_link_for_another_collection(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(
        excel.excel_list_worksheets(
            "book.xlsx",
            next_link="https://graph.microsoft.com/v1.0/me/messages?$skiptoken=x",
        )
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_delete_worksheet_uses_odata_key_segment(monkeypatch):
    mock_request = Mock(return_value=MockResponse({}, status_code=204, content=b""))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_delete_worksheet("book.xlsx", "Sheet1"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("worksheets('Sheet1')")


# ---------------------------------------------------------------------------
# ranges
# ---------------------------------------------------------------------------


def test_get_range_with_address(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"address": "Sheet1!A1:B2"}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_get_range("book.xlsx", "Sheet1", address="A1:B2"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith(
        "worksheets('Sheet1')/range(address='A1%3AB2')"
    )


def test_get_range_without_address_omits_function_call(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"address": "Sheet1!A1:Z100"}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_get_range("book.xlsx", "Sheet1"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("worksheets('Sheet1')/range")


def test_get_range_rejects_explicit_empty_address(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_get_range("book.xlsx", "Sheet1", address=""))

    assert result["status"] == "error"
    assert "address is required" in result["message"]
    mock_request.assert_not_called()


def test_update_range_sends_parsed_values(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"address": "Sheet1!A1:B1"}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(
        excel.excel_update_range("book.xlsx", "Sheet1", "A1:B1", '[["Name", "Score"]]')
    )

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["method"] == "PATCH"
    assert kwargs["json"] == {"values": [["Name", "Score"]]}


def test_update_range_caps_oversized_response(monkeypatch):
    max_output_length = get_tool_max_output_length()
    mock_request = Mock(
        return_value=MockResponse({"values": [["x" * (max_output_length + 1000)]]})
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(
        excel.excel_update_range("book.xlsx", "Sheet1", "A1", '[["x"]]')
    )

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_update_range_rejects_invalid_values_json():
    result = json.loads(
        excel.excel_update_range("book.xlsx", "Sheet1", "A1:B1", "not json")
    )
    assert result["status"] == "error"
    assert "not valid JSON" in result["message"]


def test_clear_range_validates_apply_to():
    result = json.loads(
        excel.excel_clear_range("book.xlsx", "Sheet1", "A1:B1", apply_to="Bogus")
    )
    assert result["status"] == "error"
    assert "apply_to must be one of" in result["message"]


def test_clear_range_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({}, status_code=204, content=b""))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_clear_range("book.xlsx", "Sheet1", "A1:B1"))

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["url"].endswith("range(address='A1%3AB1')/clear")
    assert kwargs["json"] == {"applyTo": "Contents"}


def test_get_range_caps_oversized_response(monkeypatch):
    max_output_length = get_tool_max_output_length()
    oversized_row = ["x" * (max_output_length + 1000)]
    mock_request = Mock(
        return_value=MockResponse(
            {"address": "Sheet1!A1:B2", "values": [oversized_row]}
        )
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_get_range("book.xlsx", "Sheet1", address="A1:B2"))

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_get_used_range_with_values_only(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"address": "Sheet1!A1:C3"}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(
        excel.excel_get_used_range("book.xlsx", "Sheet1", values_only=True)
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("usedRange(valuesOnly=true)")


def test_get_used_range_caps_oversized_response(monkeypatch):
    max_output_length = get_tool_max_output_length()
    oversized_row = ["x" * (max_output_length + 1000)]
    mock_request = Mock(
        return_value=MockResponse(
            {"address": "Sheet1!A1:C3", "values": [oversized_row]}
        )
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_get_used_range("book.xlsx", "Sheet1"))

    assert result["status"] == "success"
    assert result["truncated"] is True


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def test_list_tables_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"value": [{"id": "1"}]}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_tables("book.xlsx"))

    assert result["status"] == "success"
    assert result["tables"] == [{"id": "1"}]
    assert result["next_link"] is None


def test_list_tables_exposes_next_link(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [{"id": "1"}],
                "@odata.nextLink": "https://graph.microsoft.com/v1.0/next-page",
            }
        )
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_tables("book.xlsx"))

    assert result["next_link"] == "https://graph.microsoft.com/v1.0/next-page"


def test_add_table_sends_address_and_has_headers(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"id": "1", "name": "Table1"}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_add_table("book.xlsx", "Sheet1!A1:C5"))

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["url"].endswith("/workbook/tables/add")
    assert kwargs["json"] == {"address": "Sheet1!A1:C5", "hasHeaders": True}


def test_list_table_rows_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"value": [{"index": 0}]}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_table_rows("book.xlsx", "Table1"))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("tables('Table1')/rows")
    assert result["next_link"] is None


def test_list_table_rows_exposes_next_link(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [{"index": 0}],
                "@odata.nextLink": "https://graph.microsoft.com/v1.0/next-page",
            }
        )
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_table_rows("book.xlsx", "Table1"))

    assert result["next_link"] == "https://graph.microsoft.com/v1.0/next-page"


def test_list_table_rows_consumes_equivalently_encoded_next_link(monkeypatch):
    next_link = (
        "https://graph.microsoft.com/v1.0/me/drive/root:/book.xlsx:/workbook/"
        "tables(%27Table1%27)/rows?$skiptoken=page-2"
    )
    mock_request = Mock(return_value=MockResponse({"value": [{"index": 1}]}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(
        excel.excel_list_table_rows("book.xlsx", "Table1", next_link=next_link)
    )

    assert result["rows"] == [{"index": 1}]
    assert mock_request.call_args.kwargs["url"] == next_link


def test_list_table_rows_caps_oversized_response_and_keeps_next_link(monkeypatch):
    max_output_length = get_tool_max_output_length()
    next_link = (
        "https://graph.microsoft.com/v1.0/me/drive/root:/book.xlsx:/workbook/"
        "tables('Table1')/rows?$skiptoken=page-2"
    )
    mock_request = Mock(
        return_value=MockResponse(
            {
                "value": [{"values": [["x" * (max_output_length + 1000)]]}],
                "@odata.nextLink": next_link,
            }
        )
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_table_rows("book.xlsx", "Table1"))

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["next_link"] == next_link


def test_add_table_rows_with_index(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"index": 0}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(
        excel.excel_add_table_rows("book.xlsx", "Table1", "[[1, 2, 3]]", index=0)
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["json"] == {"values": [[1, 2, 3]], "index": 0}


def test_add_table_rows_without_index_omits_field(monkeypatch):
    mock_request = Mock(return_value=MockResponse({"index": 5}))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    json.loads(excel.excel_add_table_rows("book.xlsx", "Table1", "[[1, 2, 3]]"))

    assert "index" not in mock_request.call_args.kwargs["json"]


def test_add_table_rows_caps_oversized_response(monkeypatch):
    max_output_length = get_tool_max_output_length()
    mock_request = Mock(
        return_value=MockResponse({"values": [["x" * (max_output_length + 1000)]]})
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_add_table_rows("book.xlsx", "Table1", '[["x"]]'))

    assert result["status"] == "success"
    assert result["truncated"] is True


def test_add_table_rows_rejects_negative_index():
    result = json.loads(
        excel.excel_add_table_rows("book.xlsx", "Table1", "[[1, 2, 3]]", index=-1)
    )
    assert result["status"] == "error"
    assert "index" in result["message"]


def test_add_table_rows_rejects_bool_index():
    """bool is a subclass of int, so an unguarded isinstance(index, int)
    check would silently accept index=True and forward it to Graph as the
    literal 1 instead of raising a clear local error."""
    result = json.loads(
        excel.excel_add_table_rows("book.xlsx", "Table1", "[[1, 2, 3]]", index=True)
    )
    assert result["status"] == "error"
    assert "index" in result["message"]


def test_add_table_rows_rejects_non_integer_index():
    result = json.loads(
        excel.excel_add_table_rows("book.xlsx", "Table1", "[[1, 2, 3]]", index="0")
    )
    assert result["status"] == "error"
    assert "index" in result["message"]


def test_delete_table_row_rejects_negative_index():
    result = json.loads(excel.excel_delete_table_row("book.xlsx", "Table1", -1))
    assert result["status"] == "error"
    assert "row_index" in result["message"]


def test_delete_table_row_rejects_bool_index():
    """bool is a subclass of int in Python, so `isinstance(True, int)` is
    True and `True < 0` is False -- without an explicit bool exclusion,
    row_index=True would silently pass validation and get interpolated into
    the URL as the literal string "True", producing a broken Graph request
    instead of a clear local validation error."""
    result = json.loads(excel.excel_delete_table_row("book.xlsx", "Table1", True))
    assert result["status"] == "error"
    assert "row_index" in result["message"]


def test_delete_table_row_rejects_non_integer():
    result = json.loads(excel.excel_delete_table_row("book.xlsx", "Table1", "3"))
    assert result["status"] == "error"
    assert "row_index" in result["message"]


def test_delete_table_row_success(monkeypatch):
    mock_request = Mock(return_value=MockResponse({}, status_code=204, content=b""))
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_delete_table_row("book.xlsx", "Table1", 3))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith("tables('Table1')/rows/3")


# ---------------------------------------------------------------------------
# error propagation
# ---------------------------------------------------------------------------


def test_graph_error_response_is_surfaced_as_error(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse({"error": {"message": "Not Found"}}, status_code=404)
    )
    monkeypatch.setattr(excel.requests, "request", mock_request)

    result = json.loads(excel.excel_list_worksheets("book.xlsx"))

    assert result["status"] == "error"
    assert "Not Found" in result["message"]


def test_missing_auth_token_is_reported(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)

    result = json.loads(excel.excel_list_worksheets("book.xlsx"))

    assert result["status"] == "error"
    assert "AUTH_TOKEN" in result["message"]
