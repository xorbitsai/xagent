import json
import logging
import os
import re
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from mcp.server.fastmcp import FastMCP

from .utils import resolve_id_from_url, setup_proxy_env

logger = logging.getLogger("google-sheets-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-sheets-mcp")

_SPREADSHEET_URL_ID_PATTERN = re.compile(r"/spreadsheets/(?:u/\d+/)?d/([a-zA-Z0-9_-]+)")


def _get_credentials() -> Credentials:
    token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if not token:
        raise ValueError("GOOGLE_ACCESS_TOKEN environment variable is missing")

    creds_kwargs = {"token": token}
    if refresh_token and client_id and client_secret:
        creds_kwargs.update(
            {
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )

    return Credentials(**creds_kwargs)


def get_sheets_service() -> Any:
    return build("sheets", "v4", credentials=_get_credentials())


def get_drive_service() -> Any:
    return build("drive", "v3", credentials=_get_credentials())


def _resolve_spreadsheet_id(spreadsheet_id: str) -> str:
    """Accept either a bare spreadsheet id or a full Google Sheets URL."""
    return resolve_id_from_url(spreadsheet_id, _SPREADSHEET_URL_ID_PATTERN)


@mcp.tool()
def google_sheets_get_spreadsheet(spreadsheet_id: str) -> str:
    """
    Get metadata for a Google Sheets spreadsheet by id or full URL: its title
    and the list of sheets (tabs) with their sheet_id, title, and grid size.
    """
    try:
        resolved_spreadsheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        spreadsheet = (
            service.spreadsheets()
            .get(
                spreadsheetId=resolved_spreadsheet_id,
                fields="spreadsheetId,properties.title,spreadsheetUrl,sheets.properties",
            )
            .execute()
        )
        sheets = []
        for s in spreadsheet.get("sheets", []):
            properties = s.get("properties", {})
            grid_properties = properties.get("gridProperties") or {}
            sheets.append(
                {
                    "sheet_id": properties.get("sheetId"),
                    "title": properties.get("title"),
                    "row_count": grid_properties.get("rowCount"),
                    "column_count": grid_properties.get("columnCount"),
                }
            )

        return json.dumps(
            {
                "status": "success",
                "spreadsheet_id": spreadsheet.get(
                    "spreadsheetId", resolved_spreadsheet_id
                ),
                "title": spreadsheet.get("properties", {}).get("title"),
                "url": spreadsheet.get("spreadsheetUrl"),
                "sheets": sheets,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error getting spreadsheet: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_sheets_create_spreadsheet(
    title: str,
    sheet_titles: list[str] | None = None,
    parent_id: str | None = None,
) -> str:
    """
    Create a new Google Sheets spreadsheet with the given title.
    Optionally pass sheet_titles to create additional named sheets (tabs)
    beyond the default first sheet, and parent_id to place the spreadsheet
    in a specific Drive folder. Because this connector is authorized with
    the drive.file scope, parent_id only works for a folder this app has
    already created or that the user explicitly picked; other folder ids
    will typically be rejected by the Drive API.
    """
    try:
        service = get_sheets_service()
        body: dict[str, Any] = {"properties": {"title": title}}
        if sheet_titles:
            body["sheets"] = [{"properties": {"title": t}} for t in sheet_titles]

        spreadsheet = (
            service.spreadsheets()
            .create(
                body=body,
                fields="spreadsheetId,properties.title,spreadsheetUrl",
            )
            .execute()
        )
    except Exception as e:
        logger.error(f"Error creating spreadsheet: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    if parent_id:
        try:
            drive = get_drive_service()
            existing_file = (
                drive.files()
                .get(fileId=spreadsheet["spreadsheetId"], fields="parents")
                .execute()
            )
            previous_parents = existing_file.get("parents") or []
            update_kwargs: dict[str, Any] = {
                "fileId": spreadsheet["spreadsheetId"],
                "addParents": parent_id,
                "fields": "id,parents",
            }
            if previous_parents:
                update_kwargs["removeParents"] = ",".join(previous_parents)
            drive.files().update(**update_kwargs).execute()
        except Exception as e:
            logger.error(f"Error moving new spreadsheet to parent_id: {e}")
            return json.dumps(
                {
                    "status": "partial",
                    "spreadsheet_id": spreadsheet.get("spreadsheetId"),
                    "title": spreadsheet.get("properties", {}).get("title"),
                    "url": spreadsheet.get("spreadsheetUrl"),
                    "message": (
                        "Spreadsheet was created but could not be moved to "
                        f"parent_id {parent_id!r}: {e}"
                    ),
                },
                ensure_ascii=False,
            )

    return json.dumps(
        {
            "status": "success",
            "spreadsheet_id": spreadsheet.get("spreadsheetId"),
            "title": spreadsheet.get("properties", {}).get("title"),
            "url": spreadsheet.get("spreadsheetUrl"),
        },
        ensure_ascii=False,
    )


@mcp.tool()
def google_sheets_read_range(
    spreadsheet_id: str, range_name: str, max_rows: int = 1000
) -> str:
    """
    Read cell values from a range in a Google Sheets spreadsheet, e.g.
    range_name="Sheet1!A1:D10". spreadsheet_id accepts a bare id or full URL.
    A bare tab reference like range_name="Sheet1" is valid A1 notation for
    the entire sheet, so max_rows caps how many rows are returned; extra
    rows are dropped and the response sets truncated=true.
    """
    try:
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")

        resolved_spreadsheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=resolved_spreadsheet_id, range=range_name)
            .execute()
        )
        values = result.get("values", [])
        truncated = len(values) > max_rows
        if truncated:
            values = values[:max_rows]

        return json.dumps(
            {
                "status": "success",
                "range": result.get("range", range_name),
                "values": values,
                "truncated": truncated,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error reading range: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_sheets_update_range(
    spreadsheet_id: str,
    range_name: str,
    values: list[list[Any]],
    value_input_option: str = "USER_ENTERED",
) -> str:
    """
    Overwrite cell values in a range, e.g. range_name="Sheet1!A1:B2".
    values must be a 2D array of rows, e.g. [["a","b"],["c","d"]].
    value_input_option controls parsing: "USER_ENTERED" (default, parses
    formulas/numbers like typed input) or "RAW" (stores every value as a
    literal string).
    """
    try:
        resolved_spreadsheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .values()
            .update(
                spreadsheetId=resolved_spreadsheet_id,
                range=range_name,
                valueInputOption=value_input_option,
                body={"values": values},
            )
            .execute()
        )
        return json.dumps(
            {
                "status": "success",
                "updated_range": result.get("updatedRange", range_name),
                "updated_cells": result.get("updatedCells", 0),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error updating range: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_sheets_append_rows(
    spreadsheet_id: str,
    range_name: str,
    values: list[list[Any]],
    value_input_option: str = "USER_ENTERED",
) -> str:
    """
    Append rows after the last row of data found in range_name, e.g.
    range_name="Sheet1!A1". values must be a 2D array of rows, e.g.
    [["a","b"],["c","d"]].
    """
    try:
        resolved_spreadsheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .values()
            .append(
                spreadsheetId=resolved_spreadsheet_id,
                range=range_name,
                valueInputOption=value_input_option,
                insertDataOption="INSERT_ROWS",
                body={"values": values},
            )
            .execute()
        )
        updates = result.get("updates", {})
        return json.dumps(
            {
                "status": "success",
                "updated_range": updates.get("updatedRange", range_name),
                "updated_rows": updates.get("updatedRows", 0),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error appending rows: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_sheets_clear_range(spreadsheet_id: str, range_name: str) -> str:
    """
    Clear all values in a range, e.g. range_name="Sheet1!A1:D10", without
    removing cell formatting.
    """
    try:
        resolved_spreadsheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .values()
            .clear(spreadsheetId=resolved_spreadsheet_id, range=range_name)
            .execute()
        )
        return json.dumps(
            {
                "status": "success",
                "cleared_range": result.get("clearedRange", range_name),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error clearing range: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_sheets_add_sheet(
    spreadsheet_id: str, title: str, rows: int = 1000, columns: int = 26
) -> str:
    """
    Add a new sheet (tab) to an existing spreadsheet.
    """
    try:
        resolved_spreadsheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .batchUpdate(
                spreadsheetId=resolved_spreadsheet_id,
                body={
                    "requests": [
                        {
                            "addSheet": {
                                "properties": {
                                    "title": title,
                                    "gridProperties": {
                                        "rowCount": rows,
                                        "columnCount": columns,
                                    },
                                }
                            }
                        }
                    ]
                },
            )
            .execute()
        )
        new_sheet = (
            (result.get("replies") or [{}])[0].get("addSheet", {}).get("properties", {})
        )
        return json.dumps(
            {
                "status": "success",
                "sheet_id": new_sheet.get("sheetId"),
                "title": new_sheet.get("title"),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error adding sheet: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def google_sheets_delete_sheet(spreadsheet_id: str, sheet_id: int) -> str:
    """
    Delete a sheet (tab) from a spreadsheet by its numeric sheet_id (see
    google_sheets_get_spreadsheet for the sheet_id of each tab; this is not
    the spreadsheet_id string).
    """
    try:
        resolved_spreadsheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        service.spreadsheets().batchUpdate(
            spreadsheetId=resolved_spreadsheet_id,
            body={"requests": [{"deleteSheet": {"sheetId": sheet_id}}]},
        ).execute()
        return json.dumps(
            {"status": "success", "message": f"Sheet {sheet_id} deleted."},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error deleting sheet: {e}")
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()
