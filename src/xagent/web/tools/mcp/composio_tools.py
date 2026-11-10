"""Run a Composio tool and stage a file for it.

The proxy transport carries JSON calls and relays a binary answer, but it
cannot carry a multipart or resumable upload: a Drive create or content
update sends metadata and bytes in one request. A toolkit's own tool takes
the bytes instead (``google_drive_composio`` is Drive's), staged through
Composio's Files API: the file is uploaded once, then named by the key the
tool receives.

A staged object is not read again here, so the bytes that were sent are the
bytes the tool sees. The upload request carries the file's MD5, which is what
Composio's own deduplication keys on. A failed tool leaves its staged object
in Composio's storage until Composio's retention expires it (about a day in
the published documentation); this module does not call a delete endpoint, so
confirming that retention and whether such an endpoint exists is a stage
check. The object holds the user's bytes for that window.

Nothing raised here names the API key, the connected account id, the user id,
or the presigned URL: a presigned URL is a bearer credential for the staged
file. A provider error Composio relays back is redacted and truncated before
it reaches a caller.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from ....core.utils import public_requests as requests
from ....core.utils.security import PrivateNetworkHostError
from .composio_proxy import (
    _REDACTED,
    _SECRET_ASSIGNMENT_RE,
    _SECRET_MEMBER_RE,
    COMPOSIO_API_KEY_ENV,
    COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
    COMPOSIO_USER_ID_ENV,
    DEFAULT_TIMEOUT_SECONDS,
    _required_env,
    _truncated,
    redact,
)

COMPOSIO_API_BASE_URL = "https://backend.composio.dev/api/v3.1"

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

    def __init__(self, message: str, *, status: int | None = None) -> None:
        # Redact before truncating, so a secret cut in half cannot survive.
        super().__init__(_safe_text(message))
        self.status = status


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
    toolkit_slug: str,
    tool_slug: str,
    filename: str,
    mimetype: str,
    md5: str,
    data: bytes | ReadableBody,
    length: int,
    on_dedup: Callable[[], None] | None = None,
) -> dict[str, str]:
    """Upload ``data`` to Composio's file storage for ``tool_slug``.

    ``toolkit_slug`` is the Composio toolkit the tool belongs to. Returns the
    ``{name, mimetype, s3key}`` object a Drive tool takes as its
    ``file_to_upload`` argument. ``length`` is the number of bytes ``data``
    holds; it becomes the upload's Content-Length, so a stream is read once
    and not buffered.

    Composio deduplicates on ``md5``. When it answers with a key but no
    upload URL, the bytes are already stored under that key and nothing is
    uploaded again. ``on_dedup`` is then called once, so a caller whose body
    is a local file can confirm the file did not change while the request was
    in flight; on the upload path that read itself detects a change.
    """
    if type(length) is not int or length < 0:
        raise ValueError("length must be a non-negative byte count")
    payload = {
        "toolkit_slug": toolkit_slug,
        "tool_slug": tool_slug,
        "filename": filename,
        "mimetype": mimetype,
        "md5": md5,
    }
    status, answer = _call("/files/upload/request", payload)
    if not 200 <= status < 300:
        raise _api_error("Composio refused the file upload request", status, answer)
    if not isinstance(answer, Mapping):
        raise ComposioToolError("Composio returned an unusable file upload answer")
    key = answer.get("key")
    if not isinstance(key, str) or not key:
        raise ComposioToolError("Composio returned no key for the uploaded file")
    url = _upload_url(answer)
    if url is None:
        # A missing or empty URL is Composio's documented deduplication: the
        # bytes for this md5 are already stored under the returned key.
        if on_dedup is not None:
            on_dedup()
    else:
        backend = _storage_backend(answer)
        _put_file(url, data=data, length=length, mimetype=mimetype, backend=backend)
    return {"name": filename, "mimetype": mimetype, "s3key": key}


def _upload_url(answer: Mapping[str, Any]) -> str | None:
    """The presigned PUT URL in an upload-request answer, or ``None`` for a
    deduplicated upload.

    A member that is present with another type is not a deduplication: the
    bytes would silently never be uploaded, so it is refused rather than
    treated as "no URL".
    """
    if "new_presigned_url" not in answer:
        return None
    url = answer["new_presigned_url"]
    if url is None or url == "":
        return None
    if not isinstance(url, str):
        raise ComposioToolError("Composio returned an unusable upload URL")
    return url


def execute_tool(
    tool_slug: str, arguments: Mapping[str, Any], *, toolkit: str
) -> ToolResult:
    """Run ``tool_slug`` for the launched grant with ``arguments``.

    The connected account id and user id are those the MCP subprocess was
    launched with, the same ones every proxied call uses. Passing both keeps
    the tool off a project-wide default account. ``toolkit`` names the
    toolkit in an error a caller reads ("Google Drive").
    """
    payload = {
        "connected_account_id": _required_env(COMPOSIO_CONNECTED_ACCOUNT_ID_ENV),
        # Required: the Composio project enforces 2FA on private connections.
        "user_id": _required_env(COMPOSIO_USER_ID_ENV),
        "arguments": dict(arguments),
    }
    status, answer = _call(f"/tools/execute/{tool_slug}", payload)
    if not 200 <= status < 300:
        raise _api_error(f"Composio could not run the {toolkit} tool", status, answer)
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
    data: bytes | ReadableBody,
    length: int,
    mimetype: str,
    backend: str | None,
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
            timeout=DEFAULT_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except (PrivateNetworkHostError, requests.exceptions.InvalidURL):
        # The public-only client refuses a non-public destination before a
        # socket is opened, in one of two shapes: a private host literal is
        # refused while the request is prepared, so PrivateNetworkHostError
        # itself arrives here, and a host that resolves to a private address
        # is refused in the adapter's send, which reports it as an
        # InvalidURL (public_requests.py). Keep the same ComposioToolError
        # contract as any other unusable upload URL; the message names
        # neither the host nor the URL.
        raise ComposioToolError("Composio returned an unusable upload URL") from None
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


def _call(path: str, payload: Mapping[str, Any]) -> tuple[int, Any]:
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
            timeout=DEFAULT_TIMEOUT_SECONDS,
            # A redirect is not followed: a 307/308 replays the JSON body to
            # another origin, which the public-only client refuses after the
            # POST has been sent. The 3xx is reported instead, so a caller can
            # see that the request went out and the result is unknown rather
            # than that nothing was sent.
            allow_redirects=False,
        )
    except (requests.RequestException, OSError) as exc:
        # Type only: a connection error's text can name the request. A
        # programming error is not a network failure and is allowed to
        # propagate with its own message.
        logger.warning(
            "Composio tool request failed (path=%s exception_type=%s)",
            path,
            type(exc).__name__,
        )
        raise ComposioToolError("Composio could not be reached") from None
    if 300 <= response.status_code < 400:
        raise ComposioToolError(
            f"Composio answered with a redirect (HTTP {response.status_code}) "
            "that was not followed: the request was sent, and its result is "
            "not known",
            status=int(response.status_code),
        )
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
    return ComposioToolError(message, status=status)


# A presigned storage URL is a bearer credential for a staged object, and it
# is not one of the shapes ``composio_proxy`` redacts (the proxy client never
# handles one). Drop the whole URL rather than only the signature: the object
# path is a capability too, and an error has no use for it.
_SIGNED_URL_RE = re.compile(
    r"https?://[^\s\"']*"
    r"(?:X-Amz-Signature|X-Goog-Signature|[?&](?:sig|Signature|token)=)"
    r"[^\s\"']*"
)


def _safe_text(value: str) -> str:
    """``value`` with a credential or a signed URL removed, then cut.

    A vendor error can echo a credential by member name or as an assignment,
    which the launch-secret redaction does not know; the member and
    assignment patterns are ``composio_proxy``'s own, so an answer text is
    redacted the same way whichever Composio client relays it.
    """
    text = _SIGNED_URL_RE.sub("[redacted-url]", value)
    text = _SECRET_MEMBER_RE.sub(
        lambda match: f'"{match.group(1)}": "{_REDACTED}"', text
    )
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}={_REDACTED}", text
    )
    redacted = redact(text)
    return _truncated(redacted if isinstance(redacted, str) else str(redacted))
