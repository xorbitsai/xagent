"""Registry of remote MCP connectors that take files through a signed upload.

Some remote MCP servers accept a file in two steps: a tool call returns a
short-lived signed upload URL plus a server-side file id, and the caller then
sends the file bytes to that URL.  A model can do neither step on its own: it
only holds a workspace file reference, never the bytes, and the signed URL is a
credential that must not reach it.

An embedding deployment registers such a connector here, at process start, with
:func:`register_signed_upload_binding`.  This package registers nothing by
itself.  While no binding is registered, MCP tool adapters behave exactly as
they do without this module.

A binding replaces one scalar string argument of one tool.  A list value in
that argument is passed through unchanged.
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

_REMOTE_TRANSPORTS = frozenset({"streamable_http", "sse"})
_HOST_FORBIDDEN_CHARS = frozenset("*/:@")
_REMOTE_FILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
# RFC 9110 token for a header name; visible ASCII, space and tab for a value.
_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_HEADER_VALUE_PATTERN = re.compile(r"[\t\x20-\x7e]*")
_FORBIDDEN_UPLOAD_HEADERS = frozenset(
    {"authorization", "cookie", "host", "connection", "transfer-encoding"}
)
_FORBIDDEN_UPLOAD_HEADER_PREFIX = "proxy-"
_FORBIDDEN_URL_PATH_ESCAPES = ("%2e", "%2f", "%5c")


@dataclass(frozen=True)
class SignedUploadBinding:
    """One tool of one remote connector whose file argument is uploaded by the host."""

    connection_host: str  # lower-case hostname, exact match with the connection URL
    tool_name: str  # tool whose argument is replaced
    file_argument: str  # scalar argument carrying file:<id>
    issuing_tool_name: str  # tool that returns the signed upload URL
    # exact hostnames, no wildcards, no IP literals
    allowed_upload_hosts: frozenset[str]
    upload_path_prefix: str  # starts and ends with "/"
    content_types: Mapping[str, str]  # lower-case suffix with dot -> MIME


# Registered bindings keyed by (connection_host, tool_name).
# Tests, including an embedder's tests, replace this mapping to start from an
# empty registry.
_BINDINGS: dict[tuple[str, str], SignedUploadBinding] = {}


def _validate_hostname(host: Any, label: str) -> None:
    if not isinstance(host, str) or not host:
        raise ValueError(f"{label} must be a non-empty string")
    if host != host.lower():
        raise ValueError(f"{label} must be lower-case")
    if any(ch in _HOST_FORBIDDEN_CHARS or ch.isspace() for ch in host):
        raise ValueError(f"{label} must be an exact hostname")
    if host.endswith("."):
        # A URL hostname keeps a trailing dot, so "host." and "host" would be
        # two registrations for one DNS name; only the form without the dot is
        # accepted.
        raise ValueError(f"{label} must not end with a dot")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return
    raise ValueError(f"{label} must be a hostname, not an IP address")


def _validate_binding(binding: SignedUploadBinding) -> None:
    _validate_hostname(binding.connection_host, "connection_host")
    if not isinstance(binding.allowed_upload_hosts, frozenset):
        raise ValueError("allowed_upload_hosts must be a frozenset of hostnames")
    if not binding.allowed_upload_hosts:
        raise ValueError("allowed_upload_hosts must not be empty")
    for upload_host in binding.allowed_upload_hosts:
        _validate_hostname(upload_host, "allowed_upload_hosts entry")

    prefix = binding.upload_path_prefix
    if not isinstance(prefix, str) or not (
        prefix.startswith("/") and prefix.endswith("/")
    ):
        raise ValueError("upload_path_prefix must start and end with '/'")
    if ".." in prefix:
        raise ValueError("upload_path_prefix must not contain '..'")

    if not isinstance(binding.content_types, Mapping):
        raise ValueError("content_types must be a mapping")
    if not binding.content_types:
        raise ValueError("content_types must not be empty")
    for suffix, mime in binding.content_types.items():
        if not isinstance(suffix, str) or not suffix.startswith("."):
            raise ValueError("content_types keys must start with '.'")
        if suffix != suffix.lower():
            raise ValueError("content_types keys must be lower-case")
        if not isinstance(mime, str) or not mime:
            raise ValueError("content_types values must be non-empty")

    for label, value in (
        ("tool_name", binding.tool_name),
        ("file_argument", binding.file_argument),
        ("issuing_tool_name", binding.issuing_tool_name),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")

    if binding.issuing_tool_name == binding.tool_name:
        raise ValueError("issuing_tool_name must differ from tool_name")
    for other in _BINDINGS.values():
        if other.connection_host != binding.connection_host:
            continue
        if (
            other.issuing_tool_name == binding.tool_name
            or binding.issuing_tool_name == other.tool_name
        ):
            raise ValueError(
                "a tool of this connection host cannot be both an issuing tool "
                "and a replaced tool"
            )

    if (binding.connection_host, binding.tool_name) in _BINDINGS:
        raise ValueError("a binding for this connection host and tool already exists")


def register_signed_upload_binding(binding: SignedUploadBinding) -> None:
    """Register ``binding``; raise ``ValueError`` if it is invalid or a duplicate."""
    _validate_binding(binding)
    _BINDINGS[(binding.connection_host, binding.tool_name)] = binding


def _remote_connection_host(connection: Any) -> str | None:
    """Return the lower-case URL hostname of a remote connection, else ``None``."""
    if not isinstance(connection, Mapping):
        return None
    if connection.get("transport") not in _REMOTE_TRANSPORTS:
        return None
    url = connection.get("url")
    if not isinstance(url, str):
        return None
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def find_signed_upload_binding(
    connection: Any, tool_name: str
) -> SignedUploadBinding | None:
    """Return the binding registered for ``tool_name`` on ``connection``, if any."""
    host = _remote_connection_host(connection)
    if host is None:
        return None
    return _BINDINGS.get((host, tool_name))


def is_signed_upload_issuing_tool(connection: Any, tool_name: str) -> bool:
    """Whether ``tool_name`` is the issuing tool of a binding on ``connection``."""
    host = _remote_connection_host(connection)
    if host is None:
        return False
    return any(
        binding.connection_host == host and binding.issuing_tool_name == tool_name
        for binding in _BINDINGS.values()
    )


@dataclass(frozen=True)
class SignedUploadTicket:
    """A validated reply of an issuing tool."""

    file_id: str  # the connector's own id for the file
    upload_url: str  # not yet checked against a binding
    headers: Mapping[str, str]  # headers the connector asks to send with the PUT
    expires_at: datetime  # timezone-aware


class SignedUploadResponseError(ValueError):
    """An issuing tool reply that cannot be used.

    Only a fixed ``reason`` is carried, never any part of the reply: the reply
    holds a credential-bearing URL and caller-visible headers.
    """

    reason: Literal["malformed", "forbidden_header"]

    def __init__(self, reason: Literal["malformed", "forbidden_header"]) -> None:
        super().__init__(reason)
        self.reason = reason


class SignedUploadURLRejected(ValueError):
    """A signed upload URL that the binding does not allow.

    ``host`` is the parsed hostname when the URL could be parsed, else ``None``.
    The URL itself is never carried.
    """

    host: str | None

    def __init__(self, host: str | None) -> None:
        super().__init__("signed upload URL rejected")
        self.host = host


def _malformed() -> SignedUploadResponseError:
    return SignedUploadResponseError("malformed")


def _response_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    structured = result.get("structured_content")
    if isinstance(structured, dict):
        return structured
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                text = item.get("text")
                if not isinstance(text, str):
                    raise _malformed()
                try:
                    payload = json.loads(text)
                except ValueError:
                    raise _malformed() from None
                if isinstance(payload, dict):
                    return payload
                raise _malformed()
    raise _malformed()


def parse_signed_upload_response(
    result: Mapping[str, Any], *, size_bytes: int
) -> SignedUploadTicket:
    """Validate the normalized result of an issuing tool call.

    ``result`` is the shape produced by ``_normalized_mcp_call_result``.  Raises
    :class:`SignedUploadResponseError`; the error never echoes reply content.
    """
    payload = _response_payload(result)

    file_id = payload.get("file_id")
    if not isinstance(file_id, str) or not _REMOTE_FILE_ID_PATTERN.fullmatch(file_id):
        raise _malformed()
    upload_url = payload.get("upload_url")
    if not isinstance(upload_url, str):
        raise _malformed()
    method = payload.get("method")
    if not isinstance(method, str) or method.upper() != "PUT":
        raise _malformed()
    headers = payload.get("headers")
    if not isinstance(headers, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in headers.items()
    ):
        raise _malformed()
    expires_raw = payload.get("url_expires_at")
    if not isinstance(expires_raw, str):
        raise _malformed()
    try:
        expires_at = datetime.fromisoformat(expires_raw)
    except ValueError:
        raise _malformed() from None
    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
        raise _malformed()

    if not all(
        _HEADER_NAME_PATTERN.fullmatch(key) and _HEADER_VALUE_PATTERN.fullmatch(value)
        for key, value in headers.items()
    ):
        raise _malformed()

    names = [key.lower() for key in headers]
    if len(set(names)) != len(names):
        raise _malformed()
    for name, value in ((key.lower(), value) for key, value in headers.items()):
        if name in _FORBIDDEN_UPLOAD_HEADERS or name.startswith(
            _FORBIDDEN_UPLOAD_HEADER_PREFIX
        ):
            raise SignedUploadResponseError("forbidden_header")
        if name == "content-length" and value != str(size_bytes):
            raise SignedUploadResponseError("forbidden_header")

    return SignedUploadTicket(
        file_id=file_id,
        upload_url=upload_url,
        headers=MappingProxyType(dict(headers)),
        expires_at=expires_at,
    )


def check_signed_upload_url(raw: str, binding: SignedUploadBinding) -> httpx.URL:
    """Parse ``raw`` and check it against ``binding``; return the checked URL.

    The returned object is the one that must be used to send the request, so
    what was checked is exactly what is contacted.  Raises
    :class:`SignedUploadURLRejected`.
    """
    if "\\" in raw or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in raw):
        raise SignedUploadURLRejected(None)
    try:
        url = httpx.URL(raw)
        # The host is decoded lazily, so an invalid internationalized name only
        # fails here.
        host = url.host
    except (httpx.InvalidURL, ValueError):
        raise SignedUploadURLRejected(None) from None

    if (
        url.scheme != "https"
        or url.userinfo != b""
        or url.port is not None
        or host not in binding.allowed_upload_hosts
    ):
        raise SignedUploadURLRejected(host)
    try:
        path = url.raw_path.split(b"?", 1)[0].decode("ascii")
    except UnicodeDecodeError:
        raise SignedUploadURLRejected(host) from None
    if not path.startswith(binding.upload_path_prefix):
        raise SignedUploadURLRejected(host)
    lowered = path.lower()
    if any(escape in lowered for escape in _FORBIDDEN_URL_PATH_ESCAPES):
        raise SignedUploadURLRejected(host)
    return url
