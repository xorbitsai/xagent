import io
import json
import logging
import os
from typing import Any

from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from googleapiclient.http import (  # type: ignore[import-not-found]
    MediaIoBaseDownload,
    MediaIoBaseUpload,
)
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-drive-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-drive-mcp")


def get_drive_service() -> Any:
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

    credentials = Credentials(**creds_kwargs)
    return build("drive", "v3", credentials=credentials)


@mcp.tool()
def google_drive_search(query: str = "", max_results: int = 10) -> str:
    """
    Search for files in Google Drive.
    Use query parameter for Google Drive search syntax (e.g. "name contains 'meeting'").
    """
    try:
        service = get_drive_service()
        results = (
            service.files()
            .list(
                q=query if query else None,
                pageSize=max_results,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            )
            .execute()
        )
        items = results.get("files", [])

        return json.dumps({"status": "success", "files": items})
    except Exception as e:
        logger.error(f"Error searching drive: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_get_file_content(file_id: str, mime_type: str = "text/plain") -> str:
    """
    Download or export file content from Google Drive by file_id.
    If it's a Google Workspace document (Docs, Sheets), it will be exported to the requested mime_type.
    """
    try:
        service = get_drive_service()
        file_metadata = (
            service.files().get(fileId=file_id, fields="id, name, mimeType").execute()
        )
        file_mime_type = file_metadata.get("mimeType", "")

        if "application/vnd.google-apps" in file_mime_type:
            # Export Google Workspace document
            request = service.files().export_media(fileId=file_id, mimeType=mime_type)
        else:
            # Download regular file
            request = service.files().get_media(fileId=file_id)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

        return json.dumps(
            {
                "status": "success",
                "file": file_metadata,
                "content": fh.getvalue().decode("utf-8", errors="replace"),
            }
        )
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_create_file(
    name: str, content: str, mime_type: str = "text/plain", parent_id: str | None = None
) -> str:
    """
    Create a new file in Google Drive.
    If you want to create a Google Doc, use mime_type="application/vnd.google-apps.document"
    and pass plain text or HTML in the content. For normal text files, use "text/plain".
    """
    try:
        service = get_drive_service()
        file_metadata: dict[str, Any] = {"name": name, "mimeType": mime_type}
        if parent_id:
            file_metadata["parents"] = [parent_id]

        fh = io.BytesIO(content.encode("utf-8"))

        # When creating a Google Doc, the upload mime type needs to be the original content's mime type (like text/plain)
        upload_mime_type = "text/plain" if "google-apps" in mime_type else mime_type
        media = MediaIoBaseUpload(fh, mimetype=upload_mime_type, resumable=True)

        file = (
            service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id, name, webViewLink, mimeType",
            )
            .execute()
        )

        return json.dumps({"status": "success", "file": file})
    except Exception as e:
        logger.error(f"Error creating file: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_create_folder(name: str, parent_id: str | None = None) -> str:
    """
    Create a new folder in Google Drive.
    """
    try:
        service = get_drive_service()
        file_metadata: dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            file_metadata["parents"] = [parent_id]

        folder = (
            service.files()
            .create(body=file_metadata, fields="id, name, webViewLink")
            .execute()
        )

        return json.dumps({"status": "success", "folder": folder})
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_rename_file(file_id: str, new_name: str) -> str:
    """
    Rename an existing file or folder in Google Drive.
    """
    try:
        service = get_drive_service()
        file_metadata = {"name": new_name}

        updated_file = (
            service.files()
            .update(
                fileId=file_id,
                body=file_metadata,
                fields="id, name, webViewLink, mimeType",
            )
            .execute()
        )

        return json.dumps({"status": "success", "file": updated_file})
    except Exception as e:
        logger.error(f"Error renaming file: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_delete_file(file_id: str) -> str:
    """
    Delete a file or folder in Google Drive.
    Note: This skips the trash and permanently deletes the file if the user has permission.
    Otherwise, you may want to use google_drive_trash_file if needed, but this permanently deletes.
    """
    try:
        service = get_drive_service()
        try:
            service.files().delete(fileId=file_id).execute()
        except Exception as e:
            # Handle httplib2 proxy issue with 204 No Content responses causing SSL EOF
            if "UNEXPECTED_EOF_WHILE_READING" in str(e):
                logger.warning(
                    f"Ignored SSL EOF error during delete (often caused by proxy on 204 response): {e}"
                )
                # Verify if it was actually deleted
                try:
                    service.files().get(fileId=file_id).execute()
                    raise Exception(f"File was not deleted, SSL error occurred: {e}")
                except Exception as get_err:
                    if "404" in str(get_err) or "not found" in str(get_err).lower():
                        pass  # Successfully deleted
                    else:
                        raise e
            else:
                raise e

        return json.dumps(
            {
                "status": "success",
                "message": f"File/Folder {file_id} successfully deleted.",
            }
        )
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    mcp.run()
