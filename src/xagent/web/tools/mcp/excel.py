import json
import logging
import os
from typing import Annotated, Any
from urllib.parse import quote, unquote, urlsplit

import requests
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ....config import get_tool_max_output_length
from .utils import setup_proxy_env, success_with_capped_dict, url_path_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("excel-mcp")

setup_proxy_env()

mcp = FastMCP("excel-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_COLLECTION_PAGE_SIZE = 20
MAX_COLLECTION_PAGE_SIZE = 100

StrictNonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
StrictPageSize = Annotated[int, Field(strict=True, ge=1, le=MAX_COLLECTION_PAGE_SIZE)]

_VALID_CLEAR_APPLY_TO = frozenset({"All", "Formats", "Contents"})


class _GraphRequestError(RuntimeError):
    """Graph HTTP failure that retains its status without response parsing."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _GraphMutationIndeterminateError(RuntimeError):
    """A mutation may have committed even though no response was received."""


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _success_with_bounded_collection(
    field_name: str,
    values: list[Any],
    *,
    next_link: str | None,
) -> str:
    """Return a complete Graph page or fail without advancing its cursor.

    A Graph nextLink resumes after the complete server page. Locally dropping
    values while retaining that cursor would make the dropped values
    unreachable, so collection pages are never truncated here. Callers can
    request a smaller server page instead.
    """
    response = json.dumps(
        {
            "status": "success",
            field_name: values,
            "next_link": next_link,
            "truncated": False,
        },
        ensure_ascii=False,
    )
    if len(response) <= get_tool_max_output_length():
        return response
    return _error(
        "The Graph page exceeds the tool output limit; retry the collection "
        "from the beginning with a smaller page_size."
    )


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _indeterminate(message: str) -> str:
    return json.dumps(
        {
            "status": "indeterminate",
            "message": message,
            "retry_safe": False,
        },
        ensure_ascii=False,
    )


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


def _graph_mutation_request(
    path: str,
    *,
    body: dict[str, Any],
) -> Any:
    """POST a non-idempotent mutation without masking unknown outcomes."""
    try:
        return _graph_request("POST", path, body=body)
    except _GraphRequestError as exc:
        if exc.status_code >= 500:
            raise _GraphMutationIndeterminateError(
                f"Graph returned HTTP {exc.status_code} after a non-idempotent "
                "request. The mutation may already have been applied; inspect "
                "the workbook before retrying."
            ) from exc
        raise
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise _GraphMutationIndeterminateError(
            "The connection ended before Graph confirmed this non-idempotent "
            "request. The mutation may already have been applied; inspect the "
            "workbook before retrying."
        ) from exc


def _validate_page_size(page_size: int) -> int:
    if not isinstance(page_size, int) or isinstance(page_size, bool):
        raise TypeError("page_size must be an integer")
    if not 1 <= page_size <= MAX_COLLECTION_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_COLLECTION_PAGE_SIZE}")
    return page_size


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
    if not isinstance(site_id, str):
        raise TypeError("site_id must be a string")
    if not site_id:
        raise ValueError("site_id is required")
    if site_id != site_id.strip():
        raise ValueError("site_id must not have leading or trailing whitespace")
    value = site_id
    if ":/" not in value:
        if "/" in value or ":" in value:
            raise ValueError(
                "site_id must be 'root', a composite id, or a "
                "hostname:/server-relative-path value"
            )
        return quote(value, safe=",")

    if value.count(":/") != 1:
        raise ValueError("site_id contains more than one hostname/path separator")
    hostname, relative_path = value.split(":/", 1)
    terminated = relative_path.endswith(":")
    if terminated:
        relative_path = relative_path[:-1]
    if not hostname or not relative_path or ":" in hostname or ":" in relative_path:
        raise ValueError("site_id has an invalid hostname/path form")
    segments = relative_path.split("/")
    if any(segment in (".", "..", "") for segment in segments):
        raise ValueError(
            f"site_id must not contain '.', '..', or empty segments: {site_id!r}"
        )
    encoded_path = "/".join(quote(segment, safe="") for segment in segments)
    suffix = ":" if terminated else ""
    return f"{quote(hostname, safe='')}:/{encoded_path}{suffix}"


def _normalize_relative_path(path: str) -> str:
    """Normalize a drive-relative file path for a root:/{path}: request URL,
    rejecting '.'/'..' segments, an empty segment (consecutive slashes), a
    trailing folder separator, and any segment with a trailing period or
    leading/trailing whitespace.

    An empty segment (e.g. "Reports//Q1.xlsx") is rejected rather than
    collapsed: unlike a ".." segment, which requests' own PreparedRequest
    normalizes away before the request is even sent (verified directly --
    see the '.'/'..' check below), a doubled slash is sent to Graph
    exactly as given, and there's no well-defined "collapse to a single
    slash" semantic to fall back on here that wouldn't risk silently
    addressing a different path than the caller wrote.

    The trailing-period and leading/trailing-whitespace checks matter even
    though this module never writes arbitrary file content (unlike
    onedrive.py/sharepoint.py's upload tools): Graph/SharePoint's backing
    storage rejects or silently normalizes a folder/file name with a
    trailing dot or leading/trailing spaces (confirmed against Microsoft's
    own OneDrive/SharePoint restrictions doc: "Leading and trailing spaces
    in file or folder names also aren't allowed"), so e.g. "Reports./Q1.xlsx"
    or "Reports /Q1.xlsx" could silently resolve to a different path than
    the caller intended to address. This is checked on every segment, not
    just the filename, since an intermediate folder segment is exposed to
    the same hazard.
    """
    if not isinstance(path, str):
        raise TypeError("file_path must be a string")
    if not path.strip():
        raise ValueError("file_path is required")
    if path.startswith("/"):
        raise ValueError("file_path must be relative and must not start with '/'")
    if path.endswith("/"):
        raise ValueError(
            "file_path must include a filename, not end with a folder separator"
        )
    value = path
    if "\\" in value:
        raise ValueError("file_path must use '/' separators and must not contain '\\'")
    segments = value.split("/")
    if any(segment in (".", "..", "") for segment in segments):
        raise ValueError(
            f"file_path must not contain '.', '..', or empty segments: {path!r}"
        )
    for segment in segments:
        if segment != segment.strip():
            raise ValueError(
                "file_path segments must not have leading or trailing "
                f"whitespace: {path!r}"
            )
        if segment.endswith("."):
            raise ValueError(f"file_path segments must not end with a period: {path!r}")
    return value


def _odata_key_segment(collection: str, value: str) -> str:
    """Build a "{collection}('{value}')" OData alternate-key path segment,
    escaping a literal single quote by doubling it (the standard OData
    string-literal escaping convention) before percent encoding. Used for a
    worksheet or table addressed by either its Graph id or its display name
    -- Graph accepts both interchangeably in this form.
    """
    if not isinstance(value, str):
        raise TypeError(f"{collection} identifier must be a string")
    if not value.strip():
        raise ValueError(f"{collection} identifier is required")
    escaped = value.replace("'", "''")
    return f"{collection}('{quote(escaped, safe='')}')"


def _next_page_path(next_link: str, expected_path: str) -> str:
    """Validate an Excel collection continuation before reusing its token."""
    if not isinstance(next_link, str):
        raise TypeError("next_link must be a string")
    parsed = urlsplit(next_link)
    expected = urlsplit(f"{GRAPH_BASE_URL}{expected_path}")
    graph_base_path = urlsplit(GRAPH_BASE_URL).path
    if (
        parsed.scheme != expected.scheme
        or parsed.netloc != expected.netloc
        or not parsed.path.startswith(f"{graph_base_path}/")
        or unquote(parsed.path) != unquote(expected.path)
        or parsed.fragment
        or not parsed.query
    ):
        raise ValueError("next_link is not valid for this Excel collection")
    if any(segment in {".", ".."} for segment in unquote(parsed.path).split("/")):
        raise ValueError("next_link is not valid for this Excel collection")
    return f"{parsed.path[len(graph_base_path) :]}?{parsed.query}"


def _odata_string_literal(value: str) -> str:
    """Escape and percent-encode a string for use inside a Graph OData
    function call argument, e.g. range(address='...')."""
    if not isinstance(value, str):
        raise TypeError("address must be a string")
    if not value:
        raise ValueError("address is required")
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
    if site_id is not None:
        site_segment = _site_segment(site_id)
        if ":/" in site_segment and not site_segment.endswith(":"):
            site_segment += ":"
        drive_base = (
            f"/sites/{site_segment}/drives/{url_path_id(drive_id, 'drive_id')}"
            if drive_id is not None
            else f"/sites/{site_segment}/drive"
        )
    elif drive_id is not None:
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
    file_path: str,
    site_id: str | None = None,
    drive_id: str | None = None,
    next_link: str | None = None,
    page_size: StrictPageSize = DEFAULT_COLLECTION_PAGE_SIZE,
) -> str:
    """List the worksheets in an Excel workbook (.xlsx file).

    file_path is the workbook's path in the document library/drive. By
    default this addresses the caller's own OneDrive; pass site_id (a Graph
    site id -- "root" for the tenant's root site, or a site's own id/path)
    to address a SharePoint site's document library instead, optionally
    with drive_id for a non-default library. Pass a returned next_link to
    retrieve the next page. page_size bounds each server page and applies
    when starting a listing."""
    try:
        validated_page_size = _validate_page_size(page_size)
        base = _workbook_base(file_path, site_id, drive_id)
        collection_path = f"{base}/worksheets"
        path = (
            collection_path
            if next_link is None
            else _next_page_path(next_link, collection_path)
        )
        result = _graph_request(
            "GET",
            path,
            params={"$top": validated_page_size} if next_link is None else None,
        )
        return _success_with_bounded_collection(
            "worksheets",
            result.get("value", []),
            next_link=result.get("@odata.nextLink"),
        )
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
        if name is None:
            body = {}
        else:
            if not isinstance(name, str):
                raise TypeError("name must be a string")
            if not name.strip():
                raise ValueError("name must not be empty or whitespace")
            body = {"name": name}
        result = _graph_mutation_request(f"{base}/worksheets/add", body=body)
        return _success(worksheet=result)
    except _GraphMutationIndeterminateError as e:
        logger.error(
            "Worksheet creation outcome is indeterminate for %s: %s", file_path, e
        )
        return _indeterminate(str(e))
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
        if address is not None:
            path += f"(address='{_odata_string_literal(address)}')"
        result = _graph_request("GET", path)
        return success_with_capped_dict("range", result)
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
        return success_with_capped_dict("range", result)
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
        if not isinstance(apply_to, str):
            raise TypeError("apply_to must be a string")
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
        if not isinstance(values_only, bool):
            raise TypeError("values_only must be a boolean")
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("worksheets", worksheet)
        path = f"{base}/{segment}/usedRange"
        if values_only:
            path += "(valuesOnly=true)"
        result = _graph_request("GET", path)
        return success_with_capped_dict("range", result)
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
    file_path: str,
    site_id: str | None = None,
    drive_id: str | None = None,
    next_link: str | None = None,
    page_size: StrictPageSize = DEFAULT_COLLECTION_PAGE_SIZE,
) -> str:
    """List the tables (structured ranges) defined in an Excel workbook.

    Pass a returned next_link to retrieve the next page. page_size bounds
    each server page and applies when starting a listing."""
    try:
        validated_page_size = _validate_page_size(page_size)
        base = _workbook_base(file_path, site_id, drive_id)
        collection_path = f"{base}/tables"
        path = (
            collection_path
            if next_link is None
            else _next_page_path(next_link, collection_path)
        )
        result = _graph_request(
            "GET",
            path,
            params={"$top": validated_page_size} if next_link is None else None,
        )
        return _success_with_bounded_collection(
            "tables",
            result.get("value", []),
            next_link=result.get("@odata.nextLink"),
        )
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
        if not isinstance(has_headers, bool):
            raise TypeError("has_headers must be a boolean")
        base = _workbook_base(file_path, site_id, drive_id)
        body = {"address": address, "hasHeaders": has_headers}
        result = _graph_mutation_request(f"{base}/tables/add", body=body)
        return _success(table=result)
    except _GraphMutationIndeterminateError as e:
        logger.error("Table creation outcome is indeterminate for %s: %s", file_path, e)
        return _indeterminate(str(e))
    except Exception as e:
        logger.error("Error adding table %s in %s: %s", address, file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_list_table_rows(
    file_path: str,
    table: str,
    site_id: str | None = None,
    drive_id: str | None = None,
    next_link: str | None = None,
    page_size: StrictPageSize = DEFAULT_COLLECTION_PAGE_SIZE,
) -> str:
    """List the rows in an Excel table. table is either the table's Graph
    id or its display name. Pass a returned next_link to retrieve the next
    page of a large table. page_size bounds each server page and applies
    when starting a listing."""
    try:
        validated_page_size = _validate_page_size(page_size)
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("tables", table)
        collection_path = f"{base}/{segment}/rows"
        path = (
            collection_path
            if next_link is None
            else _next_page_path(next_link, collection_path)
        )
        result = _graph_request(
            "GET",
            path,
            params={"$top": validated_page_size} if next_link is None else None,
        )
        return _success_with_bounded_collection(
            "rows",
            result.get("value", []),
            next_link=result.get("@odata.nextLink"),
        )
    except Exception as e:
        logger.error("Error listing rows for table %s in %s: %s", table, file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_add_table_rows(
    file_path: str,
    table: str,
    values_json: str,
    index: StrictNonNegativeInt | None = None,
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
        if index is not None:
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("index must be an integer")
            if index < 0:
                raise ValueError("index must be zero or a positive integer")
        base = _workbook_base(file_path, site_id, drive_id)
        segment = _odata_key_segment("tables", table)
        body: dict[str, Any] = {"values": values}
        if index is not None:
            body["index"] = index
        result = _graph_mutation_request(f"{base}/{segment}/rows", body=body)
        return success_with_capped_dict("row", result)
    except _GraphMutationIndeterminateError as e:
        logger.error(
            "Table row insertion outcome is indeterminate for %s in %s: %s",
            table,
            file_path,
            e,
        )
        return _indeterminate(str(e))
    except Exception as e:
        logger.error("Error adding rows to table %s in %s: %s", table, file_path, e)
        return _error(str(e))


@mcp.tool()
def excel_delete_table_row(
    file_path: str,
    table: str,
    row_index: StrictNonNegativeInt,
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
