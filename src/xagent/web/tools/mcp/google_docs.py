import io
import json
import logging
import os
import re
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from googleapiclient.http import MediaIoBaseUpload  # type: ignore[import-not-found]
from mcp.server.fastmcp import FastMCP

from .utils import (
    GoogleFileKind,
    google_file_error_message,
    resolve_google_file_id,
    setup_proxy_env,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-docs-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-docs-mcp")

# "(?!e/)" skips a published-to-the-web link (".../d/e/<publish-id>/pub"),
# whose "e" segment is not the document id.
_DOCUMENT_URL_ID_PATTERN = re.compile(r"/document/(?:u/\d+/)?d/(?!e/)([a-zA-Z0-9_-]+)")
_DOCUMENT_KIND = GoogleFileKind(
    product="Google Docs",
    noun="document",
    link_example="https://docs.google.com/document/d/...",
    create_tool="google_docs_create_document",
    office_file="a Word file (.docx)",
    office_reader="read_file",
)

_HEADING_PREFIXES = {
    "TITLE": "# ",
    "HEADING_1": "# ",
    "HEADING_2": "## ",
    "HEADING_3": "### ",
    "HEADING_4": "#### ",
    "HEADING_5": "##### ",
    "HEADING_6": "###### ",
}


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


def get_docs_service() -> Any:
    return build("docs", "v1", credentials=_get_credentials())


def get_drive_service() -> Any:
    return build("drive", "v3", credentials=_get_credentials())


def _resolve_document_id(document_id: str, *, editing: bool = False) -> str:
    """Accept either a bare document id or a full Google Docs URL.

    Pass ``editing=True`` from a tool that changes the document."""
    return resolve_google_file_id(
        document_id,
        _DOCUMENT_URL_ID_PATTERN,
        "document_id",
        _DOCUMENT_KIND,
        editing=editing,
    )


def _document_error(
    exc: Exception, document_id: str, *, editing: bool = False, opened: bool = False
) -> str:
    """``document_id`` is the tool's own argument, before it was resolved.

    Pass ``opened=True`` once the tool has read the document: it is then a
    Google Doc, whatever link named it, so a 400 for a later request is
    about that request and keeps Google's own error."""
    return json.dumps(
        {
            "status": "error",
            "message": google_file_error_message(
                exc,
                _DOCUMENT_KIND,
                editing=editing,
                file_link_or_id=None if opened else document_id,
            ),
        },
        ensure_ascii=False,
    )


def _paragraph_text(paragraph: dict[str, Any]) -> str:
    text = "".join(
        element.get("textRun", {}).get("content", "")
        for element in paragraph.get("elements", [])
    )
    style = paragraph.get("paragraphStyle", {}).get("namedStyleType", "")
    prefix = _HEADING_PREFIXES.get(style, "")
    if prefix and text.strip():
        return prefix + text
    return text


def _extract_text(content: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for element in content:
        if "paragraph" in element:
            parts.append(_paragraph_text(element["paragraph"]))
        elif "table" in element:
            for row in element["table"].get("tableRows", []):
                cells = []
                for cell in row.get("tableCells", []):
                    cells.append(_extract_text(cell.get("content", [])).strip())
                parts.append(" | ".join(cells) + "\n")
        elif "tableOfContents" in element:
            parts.append(_extract_text(element["tableOfContents"].get("content", [])))
    return "".join(parts)


def _document_end_index(document: dict[str, Any]) -> int:
    content = document.get("body", {}).get("content", [])
    if not content:
        return 1
    return int(content[-1].get("endIndex", 2))


@mcp.tool()
def google_docs_get_document(document_id: str) -> str:
    """
    Read a Google Doc by document id or full document URL.
    Returns the title and the document text with headings rendered as Markdown.

    Documents are opened only by link or id; this connector cannot
    search for or list documents by name. When the user names a document
    without giving its link, use google_drive_search to find its id if
    that tool is available (with per-file Drive access it only finds
    files created through this app or granted to it); otherwise, or if
    it finds nothing, ask the user to paste the link
    (https://docs.google.com/document/d/...). Connecting Google Drive is
    not needed to open a document by its link.
    """
    try:
        doc_id = _resolve_document_id(document_id)
        service = get_docs_service()
        document = service.documents().get(documentId=doc_id).execute()

        return json.dumps(
            {
                "status": "success",
                "document_id": document.get("documentId"),
                "title": document.get("title"),
                "content": _extract_text(document.get("body", {}).get("content", [])),
            }
        )
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        return _document_error(e, document_id)


@mcp.tool()
def google_docs_create_document(
    title: str,
    content: str = "",
    content_format: str = "markdown",
    parent_id: str | None = None,
) -> str:
    """
    Create a new Google Doc with the given title and optional content.
    content_format can be "markdown" (headings, lists, bold, tables are converted
    to native Google Docs formatting) or "text" (inserted as plain text).
    Optionally pass parent_id to place the document in a specific Drive folder.
    """
    try:
        drive = get_drive_service()
        file_metadata: dict[str, Any] = {
            "name": title,
            "mimeType": "application/vnd.google-apps.document",
        }
        if parent_id:
            file_metadata["parents"] = [parent_id]

        upload_mime_type = (
            "text/markdown" if content_format == "markdown" else "text/plain"
        )
        media = MediaIoBaseUpload(
            io.BytesIO(content.encode("utf-8")),
            mimetype=upload_mime_type,
            resumable=True,
        )

        file = (
            drive.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id, name, webViewLink, mimeType",
            )
            .execute()
        )

        return json.dumps({"status": "success", "document": file})
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_docs_append_text(document_id: str, text: str) -> str:
    """
    Append plain text to the end of an existing Google Doc.
    """
    opened = False
    try:
        doc_id = _resolve_document_id(document_id, editing=True)
        service = get_docs_service()
        document = service.documents().get(documentId=doc_id).execute()
        opened = True
        insert_index = max(_document_end_index(document) - 1, 1)

        result = (
            service.documents()
            .batchUpdate(
                documentId=doc_id,
                body={
                    "requests": [
                        {
                            "insertText": {
                                "location": {"index": insert_index},
                                "text": text,
                            }
                        }
                    ]
                },
            )
            .execute()
        )

        return json.dumps(
            {
                "status": "success",
                "document_id": result.get("documentId", doc_id),
                "message": f"Appended {len(text)} characters.",
            }
        )
    except Exception as e:
        logger.error(f"Error appending text: {e}")
        return _document_error(e, document_id, editing=True, opened=opened)


@mcp.tool()
def google_docs_replace_text(
    document_id: str, find_text: str, replace_text: str, match_case: bool = True
) -> str:
    """
    Replace all occurrences of find_text with replace_text in a Google Doc.
    """
    try:
        doc_id = _resolve_document_id(document_id, editing=True)
        service = get_docs_service()
        result = (
            service.documents()
            .batchUpdate(
                documentId=doc_id,
                body={
                    "requests": [
                        {
                            "replaceAllText": {
                                "containsText": {
                                    "text": find_text,
                                    "matchCase": match_case,
                                },
                                "replaceText": replace_text,
                            }
                        }
                    ]
                },
            )
            .execute()
        )

        occurrences = 0
        for reply in result.get("replies", []):
            occurrences += reply.get("replaceAllText", {}).get("occurrencesChanged", 0)

        return json.dumps(
            {
                "status": "success",
                "document_id": result.get("documentId", doc_id),
                "occurrences_changed": occurrences,
            }
        )
    except Exception as e:
        logger.error(f"Error replacing text: {e}")
        return _document_error(e, document_id, editing=True)


@mcp.tool()
def google_docs_batch_update(document_id: str, requests_json: str) -> str:
    """
    Advanced: apply raw Google Docs API batchUpdate requests to a document.
    requests_json must be a JSON array of request objects following the Docs API
    schema (e.g. updateParagraphStyle, insertTable, updateTextStyle).
    Use this only when the simpler tools cannot express the required edit.
    """
    try:
        doc_id = _resolve_document_id(document_id, editing=True)
        requests = json.loads(requests_json)
        if not isinstance(requests, list):
            raise ValueError("requests_json must be a JSON array of request objects")

        service = get_docs_service()
        result = (
            service.documents()
            .batchUpdate(documentId=doc_id, body={"requests": requests})
            .execute()
        )

        return json.dumps(
            {
                "status": "success",
                "document_id": result.get("documentId", doc_id),
                "replies": result.get("replies", []),
            }
        )
    except Exception as e:
        logger.error(f"Error applying batch update: {e}")
        return _document_error(e, document_id, editing=True)


if __name__ == "__main__":
    mcp.run()
