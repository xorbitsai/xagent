import json
import logging
import os
import re
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from mcp.server.fastmcp import FastMCP

from .utils import resolve_id_from_url, setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-sheets-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-sheets-mcp")

_SPREADSHEET_URL_ID_PATTERN = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")


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
        sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        spreadsheet = (
            service.spreadsheets()
            .get(
                spreadsheetId=sheet_id,
                fields="spreadsheetId,properties.title,spreadsheetUrl,sheets.properties",
            )
            .execute()
        )
        sheets = [
            {
                "sheet_id": s["properties"]["sheetId"],
                "title": s["properties"]["title"],
                "row_count": s["properties"].get("gridProperties", {}).get("rowCount"),
                "column_count": s["properties"]
                .get("gridProperties", {})
                .get("columnCount"),
            }
            for s in spreadsheet.get("sheets", [])
        ]

        return json.dumps(
            {
                "status": "success",
                "spreadsheet_id": spreadsheet.get("spreadsheetId", sheet_id),
                "title": spreadsheet.get("properties", {}).get("title"),
                "url": spreadsheet.get("spreadsheetUrl"),
                "sheets": sheets,
            }
        )
    except Exception as e:
        logger.error(f"Error getting spreadsheet: {e}")
        return json.dumps({"status": "error", "message": str(e)})


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
    in a specific Drive folder.
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

        if parent_id:
            drive = get_drive_service()
            drive.files().update(
                fileId=spreadsheet["spreadsheetId"],
                addParents=parent_id,
                fields="id,parents",
            ).execute()

        return json.dumps(
            {
                "status": "success",
                "spreadsheet_id": spreadsheet.get("spreadsheetId"),
                "title": spreadsheet.get("properties", {}).get("title"),
                "url": spreadsheet.get("spreadsheetUrl"),
            }
        )
    except Exception as e:
        logger.error(f"Error creating spreadsheet: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_sheets_read_range(spreadsheet_id: str, range_name: str) -> str:
    """
    Read cell values from a range in a Google Sheets spreadsheet, e.g.
    range_name="Sheet1!A1:D10". spreadsheet_id accepts a bare id or full URL.
    """
    try:
        sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=sheet_id, range=range_name)
            .execute()
        )
        return json.dumps(
            {
                "status": "success",
                "range": result.get("range", range_name),
                "values": result.get("values", []),
            }
        )
    except Exception as e:
        logger.error(f"Error reading range: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_sheets_update_range(
    spreadsheet_id: str,
    range_name: str,
    values_json: str,
    value_input_option: str = "USER_ENTERED",
) -> str:
    """
    Overwrite cell values in a range, e.g. range_name="Sheet1!A1:B2".
    values_json must be a JSON-encoded 2D array of rows, e.g.
    '[["a","b"],["c","d"]]'. value_input_option controls parsing:
    "USER_ENTERED" (default, parses formulas/numbers like typed input) or
    "RAW" (stores every value as a literal string).
    """
    try:
        sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        values = json.loads(values_json)
        if not isinstance(values, list):
            raise ValueError("values_json must be a JSON array of rows")

        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .values()
            .update(
                spreadsheetId=sheet_id,
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
            }
        )
    except Exception as e:
        logger.error(f"Error updating range: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_sheets_append_rows(
    spreadsheet_id: str,
    range_name: str,
    values_json: str,
    value_input_option: str = "USER_ENTERED",
) -> str:
    """
    Append rows after the last row of data found in range_name, e.g.
    range_name="Sheet1!A1". values_json must be a JSON-encoded 2D array of
    rows, e.g. '[["a","b"],["c","d"]]'.
    """
    try:
        sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        values = json.loads(values_json)
        if not isinstance(values, list):
            raise ValueError("values_json must be a JSON array of rows")

        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .values()
            .append(
                spreadsheetId=sheet_id,
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
            }
        )
    except Exception as e:
        logger.error(f"Error appending rows: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_sheets_clear_range(spreadsheet_id: str, range_name: str) -> str:
    """
    Clear all values in a range, e.g. range_name="Sheet1!A1:D10", without
    removing cell formatting.
    """
    try:
        sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .values()
            .clear(spreadsheetId=sheet_id, range=range_name)
            .execute()
        )
        return json.dumps(
            {
                "status": "success",
                "cleared_range": result.get("clearedRange", range_name),
            }
        )
    except Exception as e:
        logger.error(f"Error clearing range: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_sheets_add_sheet(
    spreadsheet_id: str, title: str, rows: int = 1000, columns: int = 26
) -> str:
    """
    Add a new sheet (tab) to an existing spreadsheet.
    """
    try:
        sheet_id = _resolve_spreadsheet_id(spreadsheet_id)
        service = get_sheets_service()
        result = (
            service.spreadsheets()
            .batchUpdate(
                spreadsheetId=sheet_id,
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
        new_sheet = result["replies"][0]["addSheet"]["properties"]
        return json.dumps(
            {
                "status": "success",
                "sheet_id": new_sheet.get("sheetId"),
                "title": new_sheet.get("title"),
            }
        )
    except Exception as e:
        logger.error(f"Error adding sheet: {e}")
        return json.dumps({"status": "error", "message": str(e)})


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
            {"status": "success", "message": f"Sheet {sheet_id} deleted."}
        )
    except Exception as e:
        logger.error(f"Error deleting sheet: {e}")
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    mcp.run()
