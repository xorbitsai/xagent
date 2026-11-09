"""Run a Composio tool and stage a file for it.

The proxy transport carries JSON calls and relays a binary answer, but it
cannot carry a multipart or resumable upload: a Drive create or content
update sends metadata and bytes in one request. Those three Drive tools use
Composio's own tools instead, with the bytes staged through Composio's Files
API: the file is uploaded once, then named by the key the tool receives.

A staged object is not read again here, so the bytes that were sent are the
bytes the tool sees. The upload request carries the file's MD5, which is what
Composio's own deduplication keys on.

Nothing raised here names the API key, the connected account id, the user id,
or the presigned URL: a presigned URL is a bearer credential for the staged
file. A provider error Composio relays back is redacted and truncated before
it reaches a caller.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO, Protocol
from urllib.parse import urlsplit

from ....core.utils import public_requests as requests
from .composio_proxy import (
    COMPOSIO_API_KEY_ENV,
    COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
    COMPOSIO_USER_ID_ENV,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_ERROR_TEXT_CHARS,
    redact,
)

COMPOSIO_API_BASE_URL = "https://backend.composio.dev/api/v3.1"
GOOGLEDRIVE_TOOLKIT_SLUG = "googledrive"
# The two Drive tools that move bytes. A create carries metadata and content
# in one request; an update is a media-only upload. Neither fits Proxy
# Execute, which takes one JSON body and no multipart or resumable session.
GOOGLEDRIVE_CREATE_FILE = "GOOGLEDRIVE_CREATE_FILE"
GOOGLEDRIVE_UPLOAD_UPDATE_FILE = "GOOGLEDRIVE_UPLOAD_UPDATE_FILE"

# Composio's Azure-backed storage requires this header on a block-blob PUT.
_AZURE_BLOB_BACKEND = "azure_blob_storage"
_AZURE_BLOB_HEADER = "x-ms-blob-type"
_AZURE_BLOCK_BLOB = "BlockBlob"

logger = logging.getLogger(__name__)


class ReadableBody(Protocol):
    """A streaming request body: what ``requests`` reads and measures."""

    def read(self, amount: int = -1) -> bytes: ...

    def __len__(self) -> int: ...


class ComposioToolError(RuntimeError):
    """Composio itself refused or failed the call, so the tool may never have
    run. ``status`` is Composio's HTTP status, None when it was unreachable."""

    def __init__(
        self, message: str, *, status: int | None = None, slug: str | None = None
    ) -> None:
        # Redact before truncating, so a secret cut in half cannot survive.
        super().__init__(_safe_text(message))
        self.status = status
        self.slug = slug


@dataclass(frozen=True)
class ToolResult:
    """One executed Composio tool's own answer.

    ``successful`` is Composio's field, not ours: a tool that ran and failed
    (an expired grant, a Drive refusal) answers 2xx with ``successful``
    false and an ``error`` text. Only a transport or Composio API failure
    raises ``ComposioToolError``.
    """

    successful: bool
    data: Any
    error: str | None


def stage_file(
    *,
    tool_slug: str,
    filename: str,
    mimetype: str,
    md5: str,
    data: bytes | BinaryIO | ReadableBody,
    length: int,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, str]:
    """Upload ``data`` to Composio's file storage for ``tool_slug``.

    Returns the ``{name, mimetype, s3key}`` object a Drive tool takes as its
    ``file_to_upload`` argument. ``length`` is the number of bytes ``data``
    holds; it becomes the upload's Content-Length, so a stream is read once
    and not buffered.

    Composio deduplicates on ``md5``. When it answers with a key but no
    upload URL, the bytes are already stored under that key and nothing is
    uploaded again.
    """
    if not isinstance(length, int) or length < 0:
        raise ValueError("length must be a non-negative byte count")
    payload = {
        "toolkit_slug": GOOGLEDRIVE_TOOLKIT_SLUG,
        "tool_slug": tool_slug,
        "filename": filename,
        "mimetype": mimetype,
        "md5": md5,
    }
    status, answer = _call("/files/upload/request", payload, timeout)
    if not 200 <= status < 300:
        raise _api_error("Composio refused the file upload request", status, answer)
    if not isinstance(answer, Mapping):
        raise ComposioToolError("Composio returned an unusable file upload answer")
    key = answer.get("key")
    if not isinstance(key, str) or not key:
        raise ComposioToolError("Composio returned no key for the uploaded file")
    url = answer.get("new_presigned_url") or answer.get("newPresignedUrl")
    if isinstance(url, str) and url:
        backend = _storage_backend(answer)
        _put_file(
            url,
            data=data,
            length=length,
            mimetype=mimetype,
            backend=backend,
            timeout=timeout,
        )
    return {"name": filename, "mimetype": mimetype, "s3key": key}


def execute_tool(
    tool_slug: str,
    arguments: Mapping[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ToolResult:
    """Run ``tool_slug`` for the launched grant with ``arguments``.

    The connected account id and user id are those the MCP subprocess was
    launched with, the same ones every proxied call uses. Passing both keeps
    the tool off a project-wide default account.
    """
    payload = {
        "connected_account_id": _required_env(COMPOSIO_CONNECTED_ACCOUNT_ID_ENV),
        # Required: the Composio project enforces 2FA on private connections.
        "user_id": _required_env(COMPOSIO_USER_ID_ENV),
        "arguments": dict(arguments),
    }
    status, answer = _call(f"/tools/execute/{tool_slug}", payload, timeout)
    if not 200 <= status < 300:
        raise _api_error("Composio could not run the Google Drive tool", status, answer)
    if not isinstance(answer, Mapping):
        raise ComposioToolError("Composio returned an unusable tool answer")
    successful = answer.get("successful")
    if not isinstance(successful, bool):
        raise ComposioToolError("Composio returned a tool answer without a result")
    error = answer.get("error")
    return ToolResult(
        successful=successful,
        data=answer.get("data"),
        error=_safe_text(error) if isinstance(error, str) and error else None,
    )


def _storage_backend(answer: Mapping[str, Any]) -> str | None:
    metadata = answer.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    backend = metadata.get("storage_backend")
    return backend if isinstance(backend, str) else None


def _put_file(
    url: str,
    *,
    data: bytes | BinaryIO | ReadableBody,
    length: int,
    mimetype: str,
    backend: str | None,
    timeout: float,
) -> None:
    """PUT the staged bytes to Composio's presigned URL.

    Only an HTTPS URL is followed. ``public_requests`` resolves and pins the
    destination to a public address, so the URL Composio returns cannot be
    used to reach an internal host. The URL is a credential and is never put
    in a message or a log line.
    """
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ComposioToolError("Composio returned an unusable upload URL")
    headers = {"Content-Type": mimetype, "Content-Length": str(length)}
    if backend == _AZURE_BLOB_BACKEND:
        headers[_AZURE_BLOB_HEADER] = _AZURE_BLOCK_BLOB
    try:
        response = requests.request(
            "PUT",
            url,
            data=data,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
    except (requests.RequestException, OSError) as exc:
        # Type only: an exception's text can name the presigned URL. A
        # caller-supplied stream's own read error is not one of these and
        # reaches the caller unchanged.
        logger.warning(
            "Composio file upload failed (exception_type=%s)", type(exc).__name__
        )
        raise ComposioToolError(
            "the file could not be uploaded to Composio's file storage"
        ) from None
    if not 200 <= response.status_code < 300:
        raise ComposioToolError(
            f"Composio's file storage refused the upload (HTTP {response.status_code})"
        )


def _call(path: str, payload: Mapping[str, Any], timeout: float) -> tuple[int, Any]:
    api_key = _required_env(COMPOSIO_API_KEY_ENV)
    try:
        response = requests.request(
            method="POST",
            url=f"{COMPOSIO_API_BASE_URL}{path}",
            headers={
                "x-api-key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=dict(payload),
            timeout=timeout,
        )
    except Exception as exc:
        # Type only: a connection error's text can name the request.
        logger.warning(
            "Composio tool request failed (path=%s exception_type=%s)",
            path,
            type(exc).__name__,
        )
        raise ComposioToolError("Composio could not be reached") from None
    try:
        answer = response.json()
    except Exception:
        answer = None
    return int(response.status_code), answer


def _api_error(prefix: str, status: int, answer: Any) -> ComposioToolError:
    error = answer.get("error") if isinstance(answer, Mapping) else None
    slug: str | None = None
    detail: str | None = None
    if isinstance(error, Mapping):
        if isinstance(error.get("slug"), str):
            slug = error["slug"]
        if isinstance(error.get("message"), str) and error["message"]:
            detail = error["message"]
    elif isinstance(error, str) and error:
        detail = error
    message = f"{prefix} (HTTP {status})"
    if slug:
        message = f"{message} ({slug})"
    if detail:
        message = f"{message}: {detail}"
    return ComposioToolError(message, status=status, slug=slug)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} environment variable is missing")
    return value


def _safe_text(value: str) -> str:
    redacted = redact(value)
    text = redacted if isinstance(redacted, str) else str(redacted)
    if len(text) > MAX_ERROR_TEXT_CHARS:
        return f"{text[:MAX_ERROR_TEXT_CHARS]}...[truncated]"
    return text
