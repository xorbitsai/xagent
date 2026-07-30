import json
import logging
import os
import urllib.parse
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zoom-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("zoom-mcp")

ZOOM_BASE_URL = "https://api.zoom.us/v2"
DEFAULT_TIMEOUT_SECONDS = 30


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _encode_meeting_id(meeting_id: str) -> str:
    """URL-encode a Zoom meeting id or UUID for use in a path segment.

    Per Zoom's API docs, a meeting UUID that starts with a slash (``/``) or
    contains a double slash (``//``) must be double-encoded, or Zoom's
    routing mangles the path; a plain numeric meeting id only needs the
    normal single encoding. Detect on the raw value, since encoding it once
    would hide the leading-slash/double-slash shape the check depends on.
    """
    raw = str(meeting_id)
    quoted = urllib.parse.quote(raw, safe="")
    if raw.startswith("/") or "//" in raw:
        quoted = urllib.parse.quote(quoted, safe="")
    return quoted


def _headers() -> dict[str, str]:
    access_token = os.environ.get("ZOOM_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("ZOOM_ACCESS_TOKEN environment variable is missing")
    return {"Authorization": f"Bearer {access_token}"}


def _extract_error_detail(response: requests.Response) -> str | None:
    """Pull the human-readable message out of a Zoom error body.

    Zoom error responses are typically ``{"code": ..., "message": ...}``;
    returning that message alone is more useful to the LLM than the raw
    body. Returns None if the body isn't in the expected shape, so the
    caller can fall back to the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    return message if isinstance(message, str) and message else None


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{ZOOM_BASE_URL}{path}",
        headers=_headers(),
        params=params,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = str(exc)
        detail = _extract_error_detail(response)
        if detail is None:
            detail = response.text.strip()
        if detail:
            message = f"{message} - {detail}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _download_text(download_url: str) -> str:
    response = requests.get(
        download_url,
        headers=_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.text


def _find_transcript_file(recording_files: list[Any]) -> dict[str, Any] | None:
    for file_entry in recording_files:
        if isinstance(file_entry, dict) and file_entry.get("file_type") == "TRANSCRIPT":
            return file_entry
    return None


@mcp.tool()
def zoom_list_meetings(meeting_type: str = "scheduled") -> str:
    """
    List meetings for the connected Zoom user.
    meeting_type: one of "scheduled" (default, upcoming + recurring), "live", or "upcoming".
    Use this to find a meeting_id when the user refers to a meeting by name or "latest".
    """
    try:
        result = _request(
            "GET",
            "/users/me/meetings",
            params={"type": meeting_type, "page_size": 100},
        )
        meetings = result.get("meetings", []) if isinstance(result, dict) else []
        return _success(meetings=meetings)
    except Exception as e:
        logger.error(f"Error listing Zoom meetings: {e}")
        return _error(str(e))


@mcp.tool()
def zoom_get_meeting(meeting_id: str) -> str:
    """
    Get details for one meeting by its numeric id or UUID.
    Falls back to the past-meeting endpoint automatically if the meeting has already ended.
    """
    encoded_id = _encode_meeting_id(meeting_id)
    try:
        try:
            result = _request("GET", f"/meetings/{encoded_id}")
        except RuntimeError as exc:
            if "404" not in str(exc):
                raise
            result = _request("GET", f"/past_meetings/{encoded_id}")
        return _success(meeting=result)
    except Exception as e:
        logger.error(f"Error getting Zoom meeting {meeting_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zoom_list_recordings(meeting_id: str) -> str:
    """
    List cloud recording files for one meeting (audio, video, and transcript files),
    including each file's type, size, and download_url. Does not download file content —
    use zoom_get_meeting_transcript to fetch the transcript text itself.
    """
    encoded_id = _encode_meeting_id(meeting_id)
    try:
        result = _request("GET", f"/meetings/{encoded_id}/recordings")
        recording_files = (
            result.get("recording_files", []) if isinstance(result, dict) else []
        )
        return _success(recording_files=recording_files)
    except Exception as e:
        logger.error(f"Error listing Zoom recordings for meeting {meeting_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zoom_get_meeting_transcript(meeting_id: str) -> str:
    """
    Get the full text of a meeting's cloud-recording transcript (VTT captions), if one exists.
    """
    encoded_id = _encode_meeting_id(meeting_id)
    try:
        try:
            transcript = _request("GET", f"/meetings/{encoded_id}/transcript")
            download_url = (
                transcript.get("download_url") if isinstance(transcript, dict) else None
            )
        except RuntimeError as exc:
            if "404" not in str(exc):
                raise
            download_url = None

        if not download_url:
            recordings = _request("GET", f"/meetings/{encoded_id}/recordings")
            recording_files = (
                recordings.get("recording_files", [])
                if isinstance(recordings, dict)
                else []
            )
            transcript_file = _find_transcript_file(recording_files)
            if transcript_file is None:
                return _error("No transcript found for this meeting")
            download_url = transcript_file.get("download_url")

        if not download_url:
            return _error("Transcript file has no download_url")

        transcript_text = _download_text(download_url)
        return _success(transcript=transcript_text)
    except Exception as e:
        logger.error(f"Error getting Zoom transcript for meeting {meeting_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zoom_get_current_user() -> str:
    """
    Get profile info (id, email, name) for the connected Zoom account.
    """
    try:
        result = _request("GET", "/users/me")
        return _success(user=result)
    except Exception as e:
        logger.error(f"Error getting Zoom current user: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
