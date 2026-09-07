import io
import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from googleapiclient.http import (  # type: ignore[import-not-found]
    MediaIoBaseDownload,
    MediaIoBaseUpload,
)
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import require_clean_identifier, resolve_id_from_url, setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-drive-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-drive-mcp")

# Matches the id segment out of any of the common Drive/Docs/Sheets/Slides
# share-link shapes, since a user (and therefore a model relaying the user's
# words) is far more likely to hand over the URL they see in the browser
# than the bare id. The optional "u/<n>/" branch tolerates the account-index
# segment Google inserts (e.g. ".../file/u/1/d/<id>/view") whenever more than
# one Google account is signed into the browser -- see google_sheets.py's
# identical (?:u/\d+/)? tolerance for the spreadsheets URL shape. The
# "[?&]id=" branch matches the older `drive.google.com/open?id=<id>` /
# `docs.google.com/open?id=<id>` share-link format, still emitted by
# DriveApp.getUrl() in Apps Script and some third-party integrations, which
# has no "/d/<id>" path segment for the first branch to match at all.
_DRIVE_URL_ID_PATTERN = re.compile(
    r"(?:/(?:file/(?:u/\d+/)?d|folders|document/(?:u/\d+/)?d"
    r"|spreadsheets/(?:u/\d+/)?d|presentation/(?:u/\d+/)?d)/|[?&]id=)"
    r"([a-zA-Z0-9_-]+)"
)

# "owner" is deliberately excluded: ownership transfer needs
# transferOwnership=True, is not reversible the way a role change is, and is
# a distinct product decision from "share this with someone" -- one this
# connector doesn't make on the model's behalf.
_SHARE_ROLES = ("reader", "commenter", "writer")


def _resolve_file_id(file_id: str) -> str:
    return resolve_id_from_url(file_id, _DRIVE_URL_ID_PATTERN, "file_id")


def _require_share_role(role: str) -> str:
    if role not in _SHARE_ROLES:
        raise ValueError(f"role must be one of {_SHARE_ROLES}")
    return role


def _capped_permissions_response(permissions: list[Any]) -> str:
    """Build the success payload, halving ``permissions`` until it fits the
    platform's output limit -- mirrors deputy.py's/salesforce.py's
    _success_with_capped_list. There is no cursor to resume from here, so
    entries dropped this way are gone for this call, not just this page.
    """
    max_output_length = get_tool_max_output_length()
    truncated = False
    response = json.dumps(
        {"status": "success", "permissions": permissions, "truncated": truncated}
    )
    while len(response) > max_output_length and permissions:
        permissions = permissions[: len(permissions) // 2]
        truncated = True
        response = json.dumps(
            {"status": "success", "permissions": permissions, "truncated": truncated}
        )
    return response


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
        resolved_file_id = _resolve_file_id(file_id)
        service = get_drive_service()
        file_metadata = (
            service.files()
            .get(fileId=resolved_file_id, fields="id, name, mimeType")
            .execute()
        )
        file_mime_type = file_metadata.get("mimeType", "")

        if "application/vnd.google-apps" in file_mime_type:
            # Export Google Workspace document
            request = service.files().export_media(
                fileId=resolved_file_id, mimeType=mime_type
            )
        else:
            # Download regular file
            request = service.files().get_media(fileId=resolved_file_id)

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
        resolved_file_id = _resolve_file_id(file_id)
        service = get_drive_service()
        file_metadata = {"name": new_name}

        updated_file = (
            service.files()
            .update(
                fileId=resolved_file_id,
                body=file_metadata,
                fields="id, name, webViewLink, mimeType",
            )
            .execute()
        )

        return json.dumps({"status": "success", "file": updated_file})
    except Exception as e:
        logger.error(f"Error renaming file: {e}")
        return json.dumps({"status": "error", "message": str(e)})


def _execute_ignoring_204_ssl_eof(
    execute: Callable[[], Any], verify_done: Callable[[], None]
) -> None:
    """Run a Drive delete-style call that returns 204 No Content, tolerating
    the SSL EOF error a proxy can raise on that empty response.

    httplib2 (via a proxy) can turn a 204 response into
    ``UNEXPECTED_EOF_WHILE_READING`` even though the delete already
    succeeded server-side. When that happens, confirm the object is
    actually gone (``verify_done`` should raise a 404-shaped error once it
    is) before treating the call as successful; any other error propagates
    unchanged.
    """
    try:
        execute()
    except Exception as e:
        if "UNEXPECTED_EOF_WHILE_READING" not in str(e):
            raise
        logger.warning(
            f"Ignored SSL EOF error (often caused by proxy on 204 response): {e}"
        )
        try:
            verify_done()
        except Exception as verify_err:
            if "404" in str(verify_err) or "not found" in str(verify_err).lower():
                return  # Successfully completed
            raise e from verify_err

        # verify_done() didn't raise, so the object is still there -- the
        # delete/permission removal genuinely did not happen.
        raise Exception(f"Operation did not complete, SSL error occurred: {e}") from e


@mcp.tool()
def google_drive_delete_file(file_id: str) -> str:
    """
    Delete a file or folder in Google Drive.
    Note: This skips the trash and permanently deletes the file if the user has permission.
    Otherwise, you may want to use google_drive_trash_file if needed, but this permanently deletes.
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        service = get_drive_service()
        _execute_ignoring_204_ssl_eof(
            lambda: (
                service.files()
                .delete(fileId=resolved_file_id, supportsAllDrives=True)
                .execute()
            ),
            lambda: (
                service.files()
                .get(fileId=resolved_file_id, supportsAllDrives=True)
                .execute()
            ),
        )

        return json.dumps(
            {
                "status": "success",
                "message": f"File/Folder {resolved_file_id} successfully deleted.",
            }
        )
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_list_permissions(file_id: str) -> str:
    """
    List who currently has access to a Drive file or folder (owner,
    editors, commenters, viewers) and their permission ids. Use the
    returned permission ids with google_drive_update_permission or
    google_drive_remove_permission.
    """
    try:
        service = get_drive_service()
        results = (
            service.permissions()
            .list(
                fileId=_resolve_file_id(file_id),
                supportsAllDrives=True,
                fields="permissions(id, type, role, emailAddress, displayName)",
            )
            .execute()
        )

        return _capped_permissions_response(results.get("permissions", []))
    except Exception as e:
        logger.error(f"Error listing permissions: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_share_file(
    file_id: str,
    email: str,
    role: str = "reader",
    send_notification: bool = True,
    message: str | None = None,
) -> str:
    """
    Grant a user (by email) access to a Drive file or folder, making it
    visible to someone outside this conversation. This is an external
    action -- confirm the target file, the email, and the role with the
    user before calling it.
    role: "reader" (can view), "commenter" (can view and comment), or
    "writer" (can edit). Sharing a folder gives that access to everything
    inside it. When send_notification is True, Google emails the person
    being added; message is included in that email if given.
    """
    try:
        _require_share_role(role)
        require_clean_identifier(email, "email")
        if "@" not in email:
            raise ValueError("email must be a valid email address")
        resolved_file_id = _resolve_file_id(file_id)

        service = get_drive_service()
        # The API rejects emailMessage outright when sendNotificationEmail is
        # false, so it's only included when a notification is actually going out.
        create_kwargs: dict[str, Any] = {
            "fileId": resolved_file_id,
            "body": {"type": "user", "role": role, "emailAddress": email},
            "sendNotificationEmail": send_notification,
            "supportsAllDrives": True,
            "fields": "id, type, role, emailAddress, displayName",
        }
        if send_notification and message is not None:
            create_kwargs["emailMessage"] = message

        permission = service.permissions().create(**create_kwargs).execute()

        return json.dumps({"status": "success", "permission": permission})
    except Exception as e:
        logger.error(f"Error sharing file: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_update_permission(file_id: str, permission_id: str, role: str) -> str:
    """
    Change an existing collaborator's role on a Drive file or folder (e.g.
    upgrade a viewer to an editor). Get permission_id from
    google_drive_list_permissions. This is an external action -- confirm
    the change with the user before calling it.
    role: "reader", "commenter", or "writer".
    """
    try:
        _require_share_role(role)
        resolved_file_id = _resolve_file_id(file_id)
        resolved_permission_id = require_clean_identifier(
            permission_id, "permission_id"
        )

        service = get_drive_service()
        permission = (
            service.permissions()
            .update(
                fileId=resolved_file_id,
                permissionId=resolved_permission_id,
                body={"role": role},
                supportsAllDrives=True,
                fields="id, type, role, emailAddress, displayName",
            )
            .execute()
        )

        return json.dumps({"status": "success", "permission": permission})
    except Exception as e:
        logger.error(f"Error updating permission: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_remove_permission(file_id: str, permission_id: str) -> str:
    """
    Revoke a collaborator's access to a Drive file or folder. Get
    permission_id from google_drive_list_permissions. This is an external
    action -- confirm who is losing access with the user before calling it.
    """
    try:
        resolved_file_id = _resolve_file_id(file_id)
        resolved_permission_id = require_clean_identifier(
            permission_id, "permission_id"
        )
        service = get_drive_service()
        _execute_ignoring_204_ssl_eof(
            lambda: (
                service.permissions()
                .delete(
                    fileId=resolved_file_id,
                    permissionId=resolved_permission_id,
                    supportsAllDrives=True,
                )
                .execute()
            ),
            lambda: (
                service.permissions()
                .get(
                    fileId=resolved_file_id,
                    permissionId=resolved_permission_id,
                    supportsAllDrives=True,
                )
                .execute()
            ),
        )

        return json.dumps(
            {
                "status": "success",
                "message": f"Permission {resolved_permission_id} successfully removed.",
            }
        )
    except Exception as e:
        logger.error(f"Error removing permission: {e}")
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    mcp.run()
