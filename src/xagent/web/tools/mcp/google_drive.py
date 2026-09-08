import io
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from google.oauth2.credentials import Credentials
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

# mime types safe to decode as UTF-8 text and return inline. Anything else
# (PDFs, Office/OOXML formats, images, etc.) must go through
# google_drive_download_file instead — decoding arbitrary binary content as
# UTF-8 with errors="replace" silently corrupts it into unusable garbage.
_TEXT_MIME_TYPES = {"application/json", "application/xml"}


def _is_text_mime_type(mime_type: str) -> bool:
    return mime_type.startswith("text/") or mime_type in _TEXT_MIME_TYPES


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.() -]")


def _output_dir() -> Path:
    """Root directory google_drive_download_file writes into: the current
    task's workspace output/ subdirectory, mirroring TaskWorkspace.output_dir
    so downloaded files show up alongside other generated deliverables."""
    raw_dirs = os.environ.get("XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS", "")
    base = next((d for raw in raw_dirs.split(",") if (d := raw.strip())), "") or str(
        Path.cwd()
    )
    output_dir = (Path(base).expanduser().resolve()) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _safe_output_filename(name: str) -> str:
    """Collapse a Drive file name into a single safe path segment so it
    can't escape the output directory (e.g. via ".." or embedded "/") — the
    name comes from user/Drive data, not a trusted constant.

    Stem and suffix are sanitized separately: sanitizing the whole string
    in one pass would let the trailing ".strip('._')" eat into the
    extension's own leading dot whenever the stem sanitizes down to nothing
    (e.g. an all-non-ASCII or all-punctuation name) — "季度报告.pdf" must
    still come out as "....pdf" (something ending in .pdf), not "pdf".
    """
    base = Path(name).name
    stem = _UNSAFE_FILENAME_CHARS.sub("_", Path(base).stem).strip("._")
    suffix = _UNSAFE_FILENAME_CHARS.sub("_", Path(base).suffix)
    return (stem or "file") + suffix


def _download_media(request: Any) -> bytes:
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    return fh.getvalue()


def _unique_output_path(output_dir: Path, filename: str) -> Path:
    """Avoid silently overwriting a same-named file already in the output
    dir (e.g. downloading the same deck twice) by appending " (1)", " (2)",
    etc. — mirrors common download-manager behavior rather than either
    clobbering data or forcing the caller to pick a unique name upfront."""
    candidate = output_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    counter = 1
    while (candidate := output_dir / f"{stem} ({counter}){suffix}").exists():
        counter += 1
    return candidate


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
    Download or export a text file's content from Google Drive by file_id,
    returned inline as a string. If it's a Google Workspace document (Docs,
    Sheets), it will be exported to the requested mime_type.

    mime_type must be a text format (e.g. "text/plain", "text/csv",
    "application/json") — this tool decodes the result as UTF-8 text, which
    would corrupt a binary format like a PDF or image into garbage. For a
    PDF, an Office format, an image, or any other binary content, use
    google_drive_download_file instead, which writes the real bytes to a
    file instead of decoding them as text.
    """
    try:
        if not _is_text_mime_type(mime_type):
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"mime_type '{mime_type}' is not a text format — this "
                        "tool would corrupt binary content by decoding it as "
                        "UTF-8. Use google_drive_download_file instead to "
                        "get the real bytes as a file."
                    ),
                }
            )

        service = get_drive_service()
        file_metadata = (
            service.files().get(fileId=file_id, fields="id, name, mimeType").execute()
        )
        file_mime_type = file_metadata.get("mimeType", "")

        if "application/vnd.google-apps" in file_mime_type:
            # Export Google Workspace document — always produces content in
            # the requested (already-validated-as-text) mime_type.
            request = service.files().export_media(fileId=file_id, mimeType=mime_type)
        else:
            # Regular file: get_media ignores mime_type entirely and
            # returns the file's own real bytes, so it's file_mime_type —
            # not the requested mime_type — that determines whether
            # decoding as UTF-8 is safe.
            if not _is_text_mime_type(file_mime_type):
                return json.dumps(
                    {
                        "status": "error",
                        "message": (
                            f"'{file_metadata.get('name', file_id)}' is not a "
                            f"text file (mimeType '{file_mime_type}') — this "
                            "tool would corrupt it by decoding as UTF-8. Use "
                            "google_drive_download_file instead."
                        ),
                    }
                )
            request = service.files().get_media(fileId=file_id)

        return json.dumps(
            {
                "status": "success",
                "file": file_metadata,
                "content": _download_media(request).decode("utf-8", errors="replace"),
            }
        )
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_drive_download_file(
    file_id: str, mime_type: str = "", filename: str = ""
) -> str:
    """
    Download or export a Google Drive file to a real file in the task
    workspace, returning its path — use this for any binary content (PDF,
    image, Office format, etc.) that google_drive_get_file_content would
    otherwise corrupt by decoding as text. The returned path can be passed
    directly to another tool that reads local files, e.g. gmail_send_messages's
    'attachments'.

    mime_type: required when file_id is a Google Workspace document (Docs,
    Sheets, Slides) — the format to export to (e.g. "application/pdf").
    Ignored for a file that already has real binary content of its own
    (mime_type is not required and has no effect there).
    filename: optional name for the written file; defaults to the Drive
    file's own name (with an extension appended if exporting to a mime_type
    whose extension doesn't already match).
    """
    try:
        service = get_drive_service()
        file_metadata = (
            service.files().get(fileId=file_id, fields="id, name, mimeType").execute()
        )
        file_mime_type = file_metadata.get("mimeType", "")
        drive_name = file_metadata.get("name") or file_id

        if "application/vnd.google-apps" in file_mime_type:
            if not mime_type:
                return json.dumps(
                    {
                        "status": "error",
                        "message": (
                            f"'{drive_name}' is a Google Workspace document "
                            "(mimeType "
                            f"'{file_mime_type}'); specify mime_type to "
                            'export it to (e.g. "application/pdf").'
                        ),
                    }
                )
            request = service.files().export_media(fileId=file_id, mimeType=mime_type)
            extension = mimetypes.guess_extension(mime_type) or ""
            safe_drive_name = _safe_output_filename(drive_name)
            default_name = (
                safe_drive_name
                if safe_drive_name.endswith(extension)
                else safe_drive_name + extension
            )
        else:
            request = service.files().get_media(fileId=file_id)
            default_name = _safe_output_filename(drive_name)

        data = _download_media(request)

        chosen_name = _safe_output_filename(filename) if filename else default_name
        output_path = _unique_output_path(_output_dir(), chosen_name)
        output_path.write_bytes(data)

        return json.dumps(
            {
                "status": "success",
                "file": file_metadata,
                "path": str(output_path),
                "size": len(data),
            }
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
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
