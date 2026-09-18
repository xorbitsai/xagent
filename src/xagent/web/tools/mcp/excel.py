import json
import logging
import os
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("excel-mcp")

setup_proxy_env()

mcp = FastMCP("excel-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30

_VALID_CLEAR_APPLY_TO = frozenset({"All", "Formats", "Contents"})


class _GraphRequestError(RuntimeError):
    """Graph HTTP failure that retains its status without response parsing."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _graph_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    token = os.environ.get("AUTH_TOKEN")
    if not token:
        raise ValueError("AUTH_TOKEN environment variable is missing")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _graph_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=_graph_headers(extra_headers),
        params=params,
        json=body,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text.strip()
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
        raise _GraphRequestError(message, status_code=response.status_code) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _site_segment(site_id: str) -> str:
    """Percent-encode a caller-supplied Graph site identifier for
    interpolation into a URL path segment.

    A Graph site id is one of: the literal "root", a composite id
    ("hostname,spSiteId,spWebId"), or a "hostname:/server-relative-path"
    form. ':' and '/' stay unescaped because they're structural to the
    third shape, while a '.'/'..' segment is rejected outright -- standard
    HTTP client URL normalization (verified against requests/urllib3's own
    dot-segment collapsing) could otherwise walk the request off
    "/sites/{id}/..." and onto a different Graph endpoint under the same
    OAuth token.
    """
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError("site_id is required")
    value = site_id.strip()
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"site_id must not contain '.' or '..' segments: {site_id!r}")
    return quote(value, safe=":/,")


def _normalize_relative_path(path: str) -> str:
    """Normalize a drive-relative file path for a root:/{path}: request URL,
    rejecting '.'/'..' segments, a trailing folder separator, and a filename
    ending in a period.

    The trailing-period case matters even though this module never writes
    arbitrary file content (unlike onedrive.py/sharepoint.py's upload
    tools): Graph/SharePoint's backing storage can silently normalize a
    trailing-dot filename to the same name without the dot, so
    "Report.xlsx." could silently resolve to a real, different
    "Report.xlsx" workbook than the caller intended to address.
    """
    if not isinstance(path, str):
        raise TypeError("file_path must be a string")
    value = path.strip().strip("/")
    if not value:
        raise ValueError("file_path is required")
    if path.strip().endswith("/"):
        raise ValueError(
            "file_path must include a filename, not end with a folder separator"
        )
    if "\\" in value:
        raise ValueError("file_path must use '/' separators and must not contain '\\'")
    if any(segment in (".", "..") for segment in value.split("/")):
        raise ValueError(f"file_path must not contain '.' or '..' segments: {path!r}")
    if value.rsplit("/", 1)[-1].endswith("."):
        raise ValueError(f"file_path filename must not end with a period: {path!r}")
    return value


def _odata_key_segment(collection: str, value: str) -> str:
    """Build a "{collection}('{value}')" OData alternate-key path segment,
    escaping a literal single quote by doubling it (the standard OData
    string-literal escaping convention) before percent encoding. Used for a
    worksheet or table addressed by either its Graph id or its display name
    -- Graph accepts both interchangeably in this form.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{collection} identifier is required")
    escaped = value.strip().replace("'", "''")
    return f"{collection}('{quote(escaped, safe='')}')"


def _odata_string_literal(value: str) -> str:
    """Escape and percent-encode a string for use inside a Graph OData
    function call argument, e.g. range(address='...')."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    escaped = value.replace("'", "''")
    return quote(escaped, safe="")


def _workbook_base(file_path: str, site_id: str | None, drive_id: str | None) -> str:
    """Build the "/.../root:/{path}:/workbook" base path for a workbook
    (.xlsx) driveItem, addressed either in the caller's own OneDrive
    (default), a specific drive (drive_id only), or a SharePoint site's
    document library (site_id, optionally with drive_id for a non-default
    library).
    """
    normalized = _normalize_relative_path(file_path)
    if site_id:
        site_segment = _site_segment(site_id)
        drive_base = (
            f"/sites/{site_segment}/drives/{url_path_id(drive_id, 'drive_id')}"
            if drive_id
            else f"/sites/{site_segment}/drive"
        )
    elif drive_id:
        drive_base = f"/drives/{url_path_id(drive_id, 'drive_id')}"
    else:
        drive_base = "/me/drive"
    return f"{drive_base}/root:/{quote(normalized, safe='/')}:/workbook"


def _parse_values_json(values_json: str) -> list:
    """Parse a 2-D array of cell values from a JSON array-of-arrays string.

    Taking a JSON string (rather than a raw list parameter) matches
    google_slides_batch_update's precedent for an open-ended payload -- a
    range/table row's cell values are a caller-defined mix of strings,
    numbers, booleans, and nulls that an MCP tool schema can't usefully
    constrain further.
    """
    if not isinstance(values_json, str):
        raise TypeError("values_json must be a string")
    try:
        parsed = json.loads(values_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"values_json is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(row, list) for row in parsed):
        raise ValueError("values_json must decode to a JSON array of arrays (rows)")
    return parsed


@mcp.tool()
def excel_list_worksheets(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """List the worksheets in an Excel workbook (.xlsx file).

    file_path is the workbook's path in the document library/drive. By
    default this addresses the caller's own OneDrive; pass site_id (a Graph
    site id -- "root" for the tenant's root site, or a site's own id/path)
    to address a SharePoint site's document library instead, optionally
    with drive_id for a non-default library."""
    try:
        base = _workbook_base(file_path, site_id, drive_id)
        result = _graph_request("GET", f"{base}/worksheets")
        return _success(worksheets=result.get("value", []))
    except Exception as e:
        logger.error("Error listing worksheets for %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_add_worksheet(
    file_path: str,
    name: str | None = None,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Add a new worksheet to an Excel workbook, added at the end of the
    existing worksheets. name is optional; if omitted, Excel assigns one."""
    try:
        base = _workbook_base(file_path, site_id, drive_id)
        body = {"name": name} if name else {}
        result = _graph_request("POST", f"{base}/worksheets/add", body=body)
        return _success(worksheet=result)
    except Exception as e:
        logger.error("Error adding worksheet to %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_delete_worksheet(
    file_path: str,
    worksheet: str,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Delete a worksheet from an Excel workbook. worksheet is either the
    worksheet's Graph id or its display name."""
    try:
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("worksheets", worksheet)
        _graph_request("DELETE", f"{base}/{segment}")
        return _success(message="Worksheet deleted successfully")
    except Exception as e:
        logger.error("Error deleting worksheet %s from %s: %s", worksheet, file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_get_range(
    file_path: str,
    worksheet: str,
    address: str | None = None,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Get a cell range's values, formulas, and formatting from a worksheet.

    address is an A1-style range (e.g. "A1:C10"); if omitted, the entire
    worksheet range is returned."""
    try:
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("worksheets", worksheet)
        path = f"{base}/{segment}/range"
        if address:
            path += f"(address='{_odata_string_literal(address)}')"
        result = _graph_request("GET", path)
        return _success(range=result)
    except Exception as e:
        logger.error(
            "Error getting range %s on worksheet %s in %s: %s",
            address,
            worksheet,
            file_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def excel_update_range(
    file_path: str,
    worksheet: str,
    address: str,
    values_json: str,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Write values into a cell range on a worksheet.

    address is an A1-style range (e.g. "A1:C2"). values_json is a JSON
    array-of-arrays of cell values matching the range's shape, e.g.
    '[["Name", "Score"], ["Ada", 98]]'. A single-cell values_json is
    broadcast across the whole range (matches Excel's own CTRL+Enter fill
    behavior) when the target range is larger than one cell."""
    try:
        values = _parse_values_json(values_json)
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("worksheets", worksheet)
        path = f"{base}/{segment}/range(address='{_odata_string_literal(address)}')"
        result = _graph_request("PATCH", path, body={"values": values})
        return _success(range=result)
    except Exception as e:
        logger.error(
            "Error updating range %s on worksheet %s in %s: %s",
            address,
            worksheet,
            file_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def excel_clear_range(
    file_path: str,
    worksheet: str,
    address: str,
    apply_to: str = "Contents",
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Clear a cell range on a worksheet. apply_to is one of "All",
    "Formats", or "Contents" (default: clears cell values only, keeping
    formatting)."""
    try:
        if apply_to not in _VALID_CLEAR_APPLY_TO:
            raise ValueError(
                f"apply_to must be one of {sorted(_VALID_CLEAR_APPLY_TO)}, got {apply_to!r}"
            )
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("worksheets", worksheet)
        path = (
            f"{base}/{segment}/range(address='{_odata_string_literal(address)}')/clear"
        )
        _graph_request("POST", path, body={"applyTo": apply_to})
        return _success(message="Range cleared successfully")
    except Exception as e:
        logger.error(
            "Error clearing range %s on worksheet %s in %s: %s",
            address,
            worksheet,
            file_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def excel_get_used_range(
    file_path: str,
    worksheet: str,
    values_only: bool = False,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Get the smallest range on a worksheet that encompasses every cell
    with a value or formatting. values_only=True considers only cells with
    values (ignoring formatting-only cells)."""
    try:
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("worksheets", worksheet)
        path = f"{base}/{segment}/usedRange"
        if values_only:
            path += "(valuesOnly=true)"
        result = _graph_request("GET", path)
        return _success(range=result)
    except Exception as e:
        logger.error(
            "Error getting used range for worksheet %s in %s: %s",
            worksheet,
            file_path,
            e,
        )
        return _error(str(e))


@mcp.tool()
def excel_list_tables(
    file_path: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """List the tables (structured ranges) defined in an Excel workbook."""
    try:
        base = _workbook_base(file_path, site_id, drive_id)
        result = _graph_request("GET", f"{base}/tables")
        return _success(tables=result.get("value", []))
    except Exception as e:
        logger.error("Error listing tables in %s: %s", file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_add_table(
    file_path: str,
    address: str,
    has_headers: bool = True,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Create a table from an existing cell range. address must include the
    worksheet name, e.g. "Sheet1!A1:D5". has_headers indicates whether the
    range's first row already contains column headers."""
    try:
        base = _workbook_base(file_path, site_id, drive_id)
        body = {"address": address, "hasHeaders": has_headers}
        result = _graph_request("POST", f"{base}/tables/add", body=body)
        return _success(table=result)
    except Exception as e:
        logger.error("Error adding table %s in %s: %s", address, file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_list_table_rows(
    file_path: str, table: str, site_id: str | None = None, drive_id: str | None = None
) -> str:
    """List the rows in an Excel table. table is either the table's Graph
    id or its display name."""
    try:
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("tables", table)
        result = _graph_request("GET", f"{base}/{segment}/rows")
        return _success(rows=result.get("value", []))
    except Exception as e:
        logger.error("Error listing rows for table %s in %s: %s", table, file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_add_table_rows(
    file_path: str,
    table: str,
    values_json: str,
    index: int | None = None,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Add one or more rows to an Excel table.

    values_json is a JSON array-of-arrays, one inner array per row, e.g.
    '[["Ada", 98], ["Grace", 95]]'. index is the zero-based position to
    insert at; omit it to append at the end. Prefer batching multiple rows
    into one call over calling this repeatedly for single rows."""
    try:
        values = _parse_values_json(values_json)
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("tables", table)
        body: dict[str, Any] = {"values": values}
        if index is not None:
            body["index"] = index
        result = _graph_request("POST", f"{base}/{segment}/rows", body=body)
        return _success(row=result)
    except Exception as e:
        logger.error("Error adding rows to table %s in %s: %s", table, file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_delete_table_row(
    file_path: str,
    table: str,
    row_index: int,
    site_id: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Delete a row from an Excel table by its zero-based row index."""
    try:
        if not isinstance(row_index, int) or isinstance(row_index, bool):
            raise TypeError("row_index must be an integer")
        if row_index < 0:
            raise ValueError("row_index must be zero or a positive integer")
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("tables", table)
        _graph_request("DELETE", f"{base}/{segment}/rows/{row_index}")
        return _success(message="Table row deleted successfully")
    except Exception as e:
        logger.error(
            "Error deleting row %s from table %s in %s: %s",
            row_index,
            table,
            file_path,
            e,
        )
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
