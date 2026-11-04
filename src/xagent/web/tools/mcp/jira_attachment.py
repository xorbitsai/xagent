"""Local-file guard and response handling for the Jira attachment upload tool.

Kept apart from jira.py so the upload-specific logic does not grow that
module. Nothing here depends on jira.py, so jira.py can import it without a
cycle.
"""

import logging
import os
from pathlib import Path
from typing import Any

from ....core.utils import public_requests as requests
from .utils import allowed_dirs_from_env

logger = logging.getLogger("jira-mcp")

UPLOAD_ALLOWED_DIRS_ENV_VAR = "XAGENT_JIRA_FILE_ALLOWED_DIRS"

# A memory-safety bound for the connector subprocess, which holds the whole
# file in memory and builds a multipart body from it (a 429 retry builds a
# second body). It is NOT Jira's own limit: that is a per-site admin setting
# (Jira Cloud's default is 1 GB), and Jira's 413 gets its own message below.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

# requests' timeout bounds the WHOLE body send, not each socket operation, so
# the 30s default would need ~7 Mbit/s of uplink for a 25 MiB file. 120s
# matches OneDrive's upload timeout and needs under 2 Mbit/s.
UPLOAD_TIMEOUT_SECONDS = 120

_ATTACHMENT_MAY_HAVE_COMPLETED = (
    "the upload may have completed -- check the issue's attachments before retrying"
)

_ATTACHMENT_STATUS_HINTS = {
    403: (
        "Jira denied the upload -- the connected account needs the Browse "
        "Projects and Create attachments permissions on this issue's project"
    ),
    404: (
        "Jira could not find the issue or the site (check issue_key and "
        "cloud_id), or the connected account cannot view the issue"
    ),
    413: (
        "Jira rejected the upload as too large -- the file exceeds the site's "
        "attachment size limit, or the issue has reached its attachment limit"
    ),
    # A gateway answers after the body was sent, so Jira may have stored it.
    502: _ATTACHMENT_MAY_HAVE_COMPLETED,
    503: _ATTACHMENT_MAY_HAVE_COMPLETED,
    504: _ATTACHMENT_MAY_HAVE_COMPLETED,
}

_ATTACHMENT_UNCONFIRMED = (
    "Jira did not confirm the attachment (no attachment id in the response); "
    "check the issue before retrying"
)

_OUTCOME_UNKNOWN_ERRORS = (
    requests.Timeout,
    requests.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.JSONDecodeError,
)


def allowed_file_dirs() -> list[Path]:
    try:
        return allowed_dirs_from_env(UPLOAD_ALLOWED_DIRS_ENV_VAR)
    except ValueError as exc:
        logger.warning("Invalid Jira upload directory configuration: %s", exc)
        raise ValueError("Upload directory configuration is invalid") from None


def _resolve(file_path: str) -> Path:
    try:
        candidate = Path(file_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve()
        # An explicit symlink is resolved strictly, so a cyclic or dangling
        # link raises here instead of passing on as an unresolved path (the
        # same handling allowed_dirs_from_env gives its configured roots).
        if candidate.is_symlink():
            resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Could not resolve Jira attachment path %r: %s", file_path, exc)
        raise ValueError("Could not resolve file_path") from exc
    return resolved


def _require_allowed(local_path: Path) -> None:
    allowed_dirs = allowed_file_dirs()
    if any(local_path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
        return
    # The absolute host path is deliberately kept out of the raised message:
    # it reaches the caller/LLM unfiltered through the tool's error payload,
    # and host filesystem layout has no business in a model transcript. Full
    # detail (including the allowed directories) is logged server-side.
    logger.warning(
        "Rejected Jira attachment path %s outside allowed directories: %s",
        local_path,
        ", ".join(str(path) for path in allowed_dirs),
    )
    raise PermissionError(
        "file_path is outside the allowed upload directories; ask the user "
        "for a file inside the task workspace or another allowed location"
    )


def read_allowed_file(file_path: str) -> tuple[str, bytes]:
    """Return ``(filename, content)`` for a file under an allowlisted directory.

    Restricts jira_add_attachment to the task workspace (and any other
    configured root), the same defense the Slack, Gmail and OneDrive upload
    tools use -- without it an agent could be tricked into exfiltrating
    arbitrary host files to Jira.

    Containment is checked before existence, so a path outside the allowlist
    fails the same way whether or not it exists: reporting "not found" for
    one and "outside" for the other would let a caller probe the host
    filesystem. The size and the bytes come from one opened handle, so the
    file cannot grow past the checked size between the check and the read.

    The allowlist check runs on the resolved path before the open, as in the
    sibling connectors, so it does not by itself bind the opened handle to
    that path: a path swapped for a symlink inside that window is not
    detected.
    """
    if not file_path.strip():
        raise ValueError("file_path must not be blank")

    local_path = _resolve(file_path)
    _require_allowed(local_path)

    try:
        exists = local_path.exists()
        is_file = exists and local_path.is_file()
    except OSError as exc:
        # An EACCES or ESTALE on stat raises with the host path in its text.
        logger.warning("Could not stat Jira attachment %s: %s", local_path, exc)
        raise ValueError("Could not read the file") from exc
    if exists and not is_file:
        raise ValueError("The given path is not a regular file")
    if not is_file:
        raise FileNotFoundError("File not found at the given path")

    try:
        with local_path.open("rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size == 0:
                raise ValueError("File is empty")
            if size > MAX_ATTACHMENT_BYTES:
                raise ValueError(
                    f"File is {size} bytes, over the local upload limit of "
                    f"{MAX_ATTACHMENT_BYTES} bytes"
                )
            data = fh.read(size)
    except OSError as exc:
        # str(OSError) embeds the host path, so it cannot be re-raised as-is.
        logger.warning("Could not read Jira attachment %s: %s", local_path, exc)
        raise ValueError("Could not read the file") from exc
    if len(data) != size:
        raise ValueError("File changed while it was being read")
    return local_path.name, data


def attachment_error_hint(exc: Exception) -> str | None:
    """An actionable hint for a failed upload request, or None.

    Only call this for an exception raised by the upload POST itself: it has
    no way to tell that apart from a failure of the site lookup that precedes
    it, and a hint about attachment permissions, or about the upload having
    completed, would be false for the lookup.

    Once the body has been sent, a failure leaves it unknown whether Jira
    stored the file, so the agent is told to check before retrying: a
    timeout or dropped connection (a send-side timeout surfaces as a
    ConnectionError, not a Timeout), a response that is cut off or is not
    JSON, and a gateway 502/503/504. A ConnectTimeout is the one failure
    known to have sent nothing, so it gets no hint. Other connection errors
    that sent nothing, such as a refused connection or a failed TLS
    handshake, cannot be told apart from a drop mid-send and get the hint
    too: telling the agent to check once is cheaper than a duplicate.
    _request_absolute raises RuntimeError(...) from the HTTPError for an HTTP
    error status, which is where the failing response lives.
    """
    if isinstance(exc, _OUTCOME_UNKNOWN_ERRORS) and not isinstance(
        exc, requests.ConnectTimeout
    ):
        return _ATTACHMENT_MAY_HAVE_COMPLETED
    response = getattr(exc.__cause__, "response", None)
    status = getattr(response, "status_code", None)
    return _ATTACHMENT_STATUS_HINTS.get(status) if isinstance(status, int) else None


def confirmed_attachment(result: Any, sent_bytes: int) -> dict[str, Any]:
    """Summarize the attachment Jira created, or raise if the response does
    not confirm one.

    Jira answers with an array of the created attachments. A 200 that carries
    no attachment id is not a success, and a size that differs from what was
    sent means something other than the bytes that were read was stored: both
    raise rather than reporting success. The size message names the
    attachment because it does exist on the issue -- an agent that retried
    blindly would add a duplicate.
    """
    entry = result[0] if isinstance(result, list) and result else None
    if not isinstance(entry, dict):
        raise ValueError(_ATTACHMENT_UNCONFIRMED)
    attachment_id = entry.get("id")
    # Presence, not truthiness: see jira_transition_issue's id check.
    if attachment_id is None or (
        isinstance(attachment_id, str)
        and (not attachment_id or attachment_id.strip() != attachment_id)
    ):
        raise ValueError(_ATTACHMENT_UNCONFIRMED)
    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size != sent_bytes:
        raise ValueError(
            f"Jira created attachment {attachment_id} but reported size "
            f"{size!r} instead of {sent_bytes} bytes; verify the attachment "
            "on the issue before retrying"
        )
    return {
        "id": attachment_id,
        "filename": entry.get("filename"),
        "size": size,
        "mime_type": entry.get("mimeType"),
    }
