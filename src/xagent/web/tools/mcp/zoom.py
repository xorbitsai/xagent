import json
import logging
import os
import re
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
# Matches meta_graph.py's convention: an error body that isn't the expected
# {"message": ...} shape (e.g. an HTML gateway error page) must not be
# forwarded to the LLM/logs verbatim and unbounded.
MAX_ERROR_RESPONSE_TEXT_CHARS = 1000
# download_url is a Zoom-supplied field, not user input, but _download_text
# attaches the live Zoom bearer token to it — assert the host before sending
# credentials as defense-in-depth, since this is the only connector in the
# package that downloads from a URL rather than a fixed first-party path.
_ZOOM_DOWNLOAD_HOST_SUFFIX = ".zoom.us"

# Documented `type` values for GET /users/{userId}/meetings. Zoom staff have
# confirmed this endpoint only ever returns scheduled/live/upcoming meetings —
# there is no "previous_meetings" type; past meetings require the separate
# Reports API (a different OAuth scope this connector doesn't request). See
# https://devforum.zoom.us/t/get-users-meetings-meetings-past-instances-and-past-meeting-instances-participants/37995
MEETING_LIST_TYPES = (
    "scheduled",
    "live",
    "upcoming",
)

# Zoom always exports the full HH:MM:SS.mmm form, but WebVTT itself allows
# the hours group to be omitted (MM:SS.mmm) — match both so a cue-index
# line preceding a short-form timestamp is still recognized as scaffolding.
_VTT_TIMESTAMP_LINE = re.compile(r"^(?:\d{2}:)?\d{2}:\d{2}\.\d{3}\s+-->")


class _ZoomApiError(RuntimeError):
    """A Zoom API error carrying the HTTP status code, so callers can branch
    on 404 precisely instead of substring-matching the message (which could
    false-positive on an error body that merely mentions "404")."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


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
            if len(detail) > MAX_ERROR_RESPONSE_TEXT_CHARS:
                detail = detail[:MAX_ERROR_RESPONSE_TEXT_CHARS] + "... [truncated]"
        if detail:
            message = f"{message} - {detail}"
        raise _ZoomApiError(message, status_code=response.status_code) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _is_not_found(exc: Exception) -> bool:
    return isinstance(exc, _ZoomApiError) and exc.status_code == 404


def _download_text(download_url: str) -> str:
    host = urllib.parse.urlparse(download_url).hostname or ""
    if host != "zoom.us" and not host.endswith(_ZOOM_DOWNLOAD_HOST_SUFFIX):
        raise RuntimeError("Refusing to send Zoom credentials to an unexpected host")
    # The request itself (not just a bad status) must stay inside the
    # try/except: requests.ConnectionError embeds the full request URL —
    # including any access_token query param — in str(exc), so a connection
    # failure must be sanitized exactly like an HTTP error status is below.
    # The broader RequestException handler also covers Timeout and other
    # request-layer failures defensively, even though a plain read timeout's
    # message ("Read timed out...") doesn't itself carry the URL.
    try:
        response = requests.get(
            download_url,
            headers=_headers(),
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Transcript download failed with HTTP {exc.response.status_code}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Transcript download failed: {type(exc).__name__}") from exc
    # Decode as utf-8-sig rather than plain utf-8: for a text/* content type
    # without a charset (a plausible header for a VTT download), requests
    # falls back to ISO-8859-1, which corrupts non-ASCII (e.g. Chinese)
    # transcripts, and a BOM-prefixed file would otherwise leak into the
    # first line and fail the "WEBVTT" header check in _vtt_to_text.
    return response.content.decode("utf-8-sig", errors="replace")


def _vtt_to_text(vtt_text: str) -> str:
    """Strip WebVTT scaffolding (header, cue numbers, timestamp lines) and
    return only the spoken lines. Cue timing rarely matters for summarization
    and roughly doubles the token count of the payload handed to the LLM."""
    raw_lines = [raw_line.strip() for raw_line in vtt_text.splitlines()]
    lines: list[str] = []
    for index, line in enumerate(raw_lines):
        if not line or line == "WEBVTT":
            continue
        if _VTT_TIMESTAMP_LINE.match(line):
            continue
        # A cue-index line is digit-only *and* either the last line in the
        # file or immediately followed by a timestamp line — checking
        # structure, not just line.isdigit(), keeps a genuinely spoken
        # digit-only line (a PIN, an order number, a year read aloud) in the
        # transcript instead of silently dropping it. A trailing bare-digit
        # line can never be complete spoken content, since a real cue
        # always requires its own following timestamp and text line, so
        # it's treated as a (truncated) cue index too.
        if line.isdigit() and (
            index + 1 >= len(raw_lines)
            or _VTT_TIMESTAMP_LINE.match(raw_lines[index + 1])
        ):
            continue
        lines.append(line)
    return "\n".join(lines)


def _find_transcript_file(recording_files: list[Any]) -> dict[str, Any] | None:
    for file_entry in recording_files:
        if isinstance(file_entry, dict) and file_entry.get("file_type") == "TRANSCRIPT":
            return file_entry
    return None


@mcp.tool()
def zoom_list_meetings(meeting_type: str = "scheduled", page_token: str = "") -> str:
    """
    List meetings for the connected Zoom user.
    meeting_type: one of "scheduled" (default, unexpired scheduled meetings),
    "live", or "upcoming". This endpoint never returns meetings that have
    already ended — Zoom's List Meetings API only covers scheduled/live/
    upcoming meetings, not history. To look up a meeting that already
    happened, ask the user for its meeting id or UUID and call
    zoom_get_meeting / zoom_get_meeting_transcript directly.
    page_token: pass the next_page_token from a previous response to fetch the
    next page when the result was truncated.
    """
    if meeting_type not in MEETING_LIST_TYPES:
        return _error(
            f"Invalid meeting_type {meeting_type!r}; expected one of "
            f"{', '.join(MEETING_LIST_TYPES)}"
        )
    try:
        params: dict[str, Any] = {"type": meeting_type, "page_size": 100}
        if page_token:
            params["next_page_token"] = page_token
        result = _request("GET", "/users/me/meetings", params=params)
        meetings = result.get("meetings") or [] if isinstance(result, dict) else []
        next_page_token = (
            result.get("next_page_token", "") if isinstance(result, dict) else ""
        )
        return _success(meetings=meetings, next_page_token=next_page_token)
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
        except Exception as exc:
            if not _is_not_found(exc):
                raise
            try:
                result = _request("GET", f"/past_meetings/{encoded_id}")
            except Exception as past_exc:
                if _is_not_found(past_exc):
                    # Surface the real situation (unknown id) instead of the
                    # misleading "past_meetings lookup failed" from the second
                    # leg alone — the common case here is a typo'd meeting_id.
                    return _error(
                        f"Meeting {meeting_id} not found (checked both upcoming "
                        "and past meetings)"
                    )
                raise
        if not result:
            return _error(f"Meeting {meeting_id} returned no data")
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
            result.get("recording_files") or [] if isinstance(result, dict) else []
        )
        return _success(recording_files=recording_files)
    except Exception as e:
        logger.error(f"Error listing Zoom recordings for meeting {meeting_id}: {e}")
        return _error(str(e))


@mcp.tool()
def zoom_get_meeting_transcript(meeting_id: str) -> str:
    """
    Get the spoken text of a meeting's cloud-recording transcript, if one exists
    (WebVTT scaffolding such as timestamps and cue numbers is stripped).
    """
    encoded_id = _encode_meeting_id(meeting_id)
    try:
        restriction_reason: str | None = None
        can_download: Any = None
        try:
            transcript = _request("GET", f"/meetings/{encoded_id}/transcript")
            if isinstance(transcript, dict):
                download_url = transcript.get("download_url")
                restriction_reason = transcript.get("download_restriction_reason")
                can_download = transcript.get("can_download")
            else:
                download_url = None
        except Exception as exc:
            if not _is_not_found(exc):
                raise
            download_url = None

        if not download_url and restriction_reason:
            # NOT_READY means the transcript is still processing — a
            # retry-later condition, not a restriction like DELETED_OR_TRASHED
            # or UNSUPPORTED. Calling it "restricted" would push the agent to
            # the paste/upload fallback when waiting would have worked.
            if restriction_reason == "NOT_READY":
                return _error(
                    "Transcript is still processing on Zoom's side; try again "
                    "in a few minutes."
                )
            return _error(f"Transcript download is restricted: {restriction_reason}")

        if not download_url:
            try:
                recordings = _request("GET", f"/meetings/{encoded_id}/recordings")
            except Exception as rec_exc:
                if _is_not_found(rec_exc):
                    return _error(
                        f"Meeting {meeting_id} not found or has no cloud "
                        "recording (checked both the transcript and recordings "
                        "endpoints)"
                    )
                raise
            recording_files = (
                recordings.get("recording_files") or []
                if isinstance(recordings, dict)
                else []
            )
            transcript_file = _find_transcript_file(recording_files)
            if transcript_file is not None:
                download_url = transcript_file.get("download_url")
            elif can_download is False:
                # The transcript-shortcut endpoint said no, and the
                # independent recordings listing found no transcript file
                # either — trust can_download only once both endpoints
                # agree, rather than short-circuiting on it alone before
                # ever checking recordings.
                return _error("Transcript download is restricted")
            else:
                return _error("No transcript found for this meeting")

        if not download_url:
            return _error("Transcript file has no download_url")

        transcript_text = _vtt_to_text(_download_text(download_url))
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
        if not result:
            return _error("Zoom returned no user data")
        return _success(user=result)
    except Exception as e:
        logger.error(f"Error getting Zoom current user: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
