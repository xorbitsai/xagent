"""Signed-upload settings of remote MCP connectors.

Some remote MCP servers accept a file in two steps: a tool call returns a
short-lived signed upload URL plus a server-side file id, and the caller then
sends the file bytes to that URL.  A model can do neither step on its own: it
only holds a workspace file reference, never the bytes, and the signed URL is a
credential that must not reach it.

The owner of a connector describes such tools in the connector's own
``signed_uploads`` setting, which reaches the adapter in the connection
mapping.  A connection without that setting behaves exactly as it does without
this module.

The settings are trusted as the connector owner's own.  On a shared connector
only the connector's owner or a platform administrator may change them.  This
module only reads the setting and does not enforce that rule; whatever writes
the setting must.  The upload-host rules below check only the form of a name:
they accept names such as ``localhost`` or a single label and never look at
the address a name resolves to.  The upload request must therefore be sent
through ``PublicHttpTransport`` (``xagent.core.utils.public_http_transport``),
which resolves the host and refuses any address that is not public.

A settings entry takes one of two forms.  A binding replaces one scalar string
argument of one tool, and a list value in that argument is passed through
unchanged.  An issuer takes over one issuing tool: the host offers the model a
tool of the same name that stages a file, runs the issuing call itself and
returns only the connector's file id.
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

# The host-run upload is wired only for these two transports; settings on any
# other transport are never read.
SIGNED_UPLOAD_TRANSPORTS = frozenset({"sse", "streamable_http"})
_MAX_SETTINGS_ENTRIES = 20
_MAX_STRING_LENGTH = 255
_MAX_HOSTNAME_LENGTH = 253
_MAX_UPLOAD_HOSTS = 20
_MAX_CONTENT_TYPES = 50
_MAX_SUFFIX_LENGTH = 16
_SETTINGS_ENTRY_KEYS = frozenset(
    {
        "tool_name",
        "file_argument",
        "issuing_tool_name",
        "allowed_upload_hosts",
        "upload_path_prefix",
        "content_types",
    }
)
_ISSUER_ENTRY_KEYS = frozenset(
    {
        "issuing_tool_name",
        "allowed_upload_hosts",
        "upload_path_prefix",
        "content_types",
    }
)
_UNREADABLE_SETTINGS_MESSAGE = "Upload settings could not be read."
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

    # Hostname of the URL of the connection the settings were read from,
    # filled in when settings are parsed.  Settings apply to a connection only
    # because that connection carries them, so this field may be used to
    # refuse a binding but never to grant one.
    connection_host: str
    tool_name: str  # tool whose argument is replaced
    file_argument: str  # scalar argument carrying file:<id>
    issuing_tool_name: str  # tool that returns the signed upload URL
    # exact hostnames, no wildcards, no IP literals
    allowed_upload_hosts: frozenset[str]
    upload_path_prefix: str  # starts and ends with "/"
    content_types: Mapping[str, str]  # lower-case one-dot suffix -> MIME


@dataclass(frozen=True)
class SignedUploadIssuer:
    """An issuing tool of one remote connector that the host runs for the model."""

    # Same meaning and same rule as SignedUploadBinding.connection_host.
    connection_host: str
    issuing_tool_name: str  # tool that returns the signed upload URL
    # exact hostnames, no wildcards, no IP literals
    allowed_upload_hosts: frozenset[str]
    upload_path_prefix: str  # starts and ends with "/"
    content_types: Mapping[str, str]  # lower-case one-dot suffix -> MIME


def _validate_hostname(host: Any, label: str) -> None:
    if not isinstance(host, str) or not host:
        raise ValueError(f"{label} must be a non-empty string")
    if host != host.lower():
        raise ValueError(f"{label} must be lower-case")
    if any(ch in _HOST_FORBIDDEN_CHARS or ch.isspace() for ch in host):
        raise ValueError(f"{label} must be an exact hostname")
    if host.endswith("."):
        # A URL hostname keeps a trailing dot, so "host." and "host" would be
        # two entries for one DNS name; only the form without the dot is
        # accepted.
        raise ValueError(f"{label} must not end with a dot")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return
    raise ValueError(f"{label} must be a hostname, not an IP address")


def _validate_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if len(value) > _MAX_STRING_LENGTH:
        raise ValueError(f"{label} must not be longer than {_MAX_STRING_LENGTH}")
    return value


def _validate_entry(
    entry: Any, position: int, connection_host: str
) -> SignedUploadBinding | SignedUploadIssuer:
    """Validate one stored settings entry; ``position`` only labels errors.

    An entry with exactly the keys of a binding is a binding, and an entry with
    exactly the keys of an issuer is an issuer.
    """
    label = f"signed_uploads[{position}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{label} must be an object")
    keys = set(entry)
    if keys == _ISSUER_ENTRY_KEYS:
        issuing_tool_name = _validate_string(
            entry["issuing_tool_name"], f"{label}.issuing_tool_name"
        )
        hosts, prefix, content_types = _validate_upload_rules(entry, label)
        return SignedUploadIssuer(
            connection_host=connection_host,
            issuing_tool_name=issuing_tool_name,
            allowed_upload_hosts=hosts,
            upload_path_prefix=prefix,
            content_types=content_types,
        )
    if keys != _SETTINGS_ENTRY_KEYS:
        raise ValueError(f"{label} must have exactly the documented keys")

    tool_name = _validate_string(entry["tool_name"], f"{label}.tool_name")
    file_argument = _validate_string(entry["file_argument"], f"{label}.file_argument")
    issuing_tool_name = _validate_string(
        entry["issuing_tool_name"], f"{label}.issuing_tool_name"
    )
    hosts, prefix, content_types = _validate_upload_rules(entry, label)
    return SignedUploadBinding(
        connection_host=connection_host,
        tool_name=tool_name,
        file_argument=file_argument,
        issuing_tool_name=issuing_tool_name,
        allowed_upload_hosts=hosts,
        upload_path_prefix=prefix,
        content_types=content_types,
    )


def _validate_upload_rules(
    entry: dict[str, Any], label: str
) -> tuple[frozenset[str], str, Mapping[str, str]]:
    """Validate the upload host, path prefix and type table shared by both entry forms.

    ``entry`` holds the three keys; returns the kept values, which no longer
    share anything with the stored value.
    """
    hosts_label = f"{label}.allowed_upload_hosts"
    hosts = entry["allowed_upload_hosts"]
    if not isinstance(hosts, list):
        raise ValueError(f"{hosts_label} must be a list of hostnames")
    if not hosts:
        raise ValueError(f"{hosts_label} must not be empty")
    if len(hosts) > _MAX_UPLOAD_HOSTS:
        raise ValueError(
            f"{hosts_label} must not have more than {_MAX_UPLOAD_HOSTS} entries"
        )
    for upload_host in hosts:
        if isinstance(upload_host, str) and len(upload_host) > _MAX_HOSTNAME_LENGTH:
            raise ValueError(
                f"{hosts_label} entry must not be longer than {_MAX_HOSTNAME_LENGTH}"
            )
        _validate_hostname(upload_host, f"{hosts_label} entry")

    prefix_label = f"{label}.upload_path_prefix"
    prefix = entry["upload_path_prefix"]
    if not isinstance(prefix, str) or not (
        prefix.startswith("/") and prefix.endswith("/")
    ):
        raise ValueError(f"{prefix_label} must start and end with '/'")
    if ".." in prefix:
        raise ValueError(f"{prefix_label} must not contain '..'")
    if len(prefix) > _MAX_STRING_LENGTH:
        raise ValueError(f"{prefix_label} must not be longer than {_MAX_STRING_LENGTH}")

    types_label = f"{label}.content_types"
    content_types = entry["content_types"]
    if not isinstance(content_types, dict):
        raise ValueError(f"{types_label} must be an object")
    if not content_types:
        raise ValueError(f"{types_label} must not be empty")
    if len(content_types) > _MAX_CONTENT_TYPES:
        raise ValueError(
            f"{types_label} must not have more than {_MAX_CONTENT_TYPES} entries"
        )
    for suffix, mime in content_types.items():
        if not isinstance(suffix, str) or not suffix.startswith("."):
            raise ValueError(f"{types_label} keys must start with '.'")
        if suffix != suffix.lower():
            raise ValueError(f"{types_label} keys must be lower-case")
        if len(suffix) > _MAX_SUFFIX_LENGTH:
            raise ValueError(
                f"{types_label} keys must not be longer than {_MAX_SUFFIX_LENGTH}"
            )
        # The upload looks a file up by its last suffix only, so a key such
        # as ".tar.gz" could never match. A key that is only a dot never
        # matches either, because a file name ending in "." has no suffix.
        if suffix.count(".") > 1:
            raise ValueError(f"{types_label} keys must contain only one '.'")
        if suffix == ".":
            raise ValueError(f"{types_label} keys must have a suffix after '.'")
        if not isinstance(mime, str) or not mime:
            raise ValueError(f"{types_label} values must be non-empty")
        if len(mime) > _MAX_STRING_LENGTH:
            raise ValueError(
                f"{types_label} values must not be longer than {_MAX_STRING_LENGTH}"
            )

    return frozenset(hosts), prefix, MappingProxyType(dict(content_types))


def _check_entries_together(
    entries: list[SignedUploadBinding | SignedUploadIssuer],
) -> None:
    """Reject settings whose entries clash with each other.

    ``entries`` is in stored order and every position in an error is the
    entry's position in that order.  The checks run in a fixed order: a
    repeated replaced tool, a repeated issuing tool of two issuers, an issuing
    tool that an issuer shares with a binding, and a tool that is both an
    issuing tool and a replaced tool.
    """
    replaced_tools: set[str] = set()
    binding_issuing_tools: set[str] = set()
    issuer_tools: set[str] = set()
    repeated_replaced_tool: int | None = None
    repeated_issuer_tool: int | None = None
    issuer_positions: list[tuple[int, str]] = []
    for position, entry in enumerate(entries):
        if isinstance(entry, SignedUploadBinding):
            if entry.tool_name in replaced_tools and repeated_replaced_tool is None:
                repeated_replaced_tool = position
            replaced_tools.add(entry.tool_name)
            binding_issuing_tools.add(entry.issuing_tool_name)
        else:
            if entry.issuing_tool_name in issuer_tools and repeated_issuer_tool is None:
                repeated_issuer_tool = position
            issuer_tools.add(entry.issuing_tool_name)
            issuer_positions.append((position, entry.issuing_tool_name))
    if repeated_replaced_tool is not None:
        raise ValueError(
            f"signed_uploads[{repeated_replaced_tool}].tool_name "
            "repeats an earlier entry"
        )
    if repeated_issuer_tool is not None:
        raise ValueError(
            f"signed_uploads[{repeated_issuer_tool}].issuing_tool_name "
            "repeats an earlier entry"
        )
    # The index names the issuer: a binding's issuing tool is by definition the
    # issuing tool of a binding, so the issuer is the entry to change.
    for position, issuing_tool_name in issuer_positions:
        if issuing_tool_name in binding_issuing_tools:
            raise ValueError(
                f"signed_uploads[{position}].issuing_tool_name "
                "is also the issuing tool of a replacement entry"
            )
    if replaced_tools & (binding_issuing_tools | issuer_tools):
        raise ValueError("a tool cannot be both an issuing tool and a replaced tool")


def parse_signed_upload_settings(
    raw: Any, *, connection_host: str | None
) -> tuple[SignedUploadBinding | SignedUploadIssuer, ...]:
    """Validate a connector's stored signed_uploads value; raise ValueError.

    ``None`` and an empty list mean no settings and give an empty tuple.  Any
    other value must be a list of entries, each with exactly the keys of a
    binding or exactly the keys of an issuer, and the connection needs a
    hostname (``connection_host``, the lower-case hostname of the connection
    URL, taken as given; ``None`` and an empty string both mean it has none).
    The result has one object per entry, in stored order.  The error text is
    fixed and never carries a stored value.
    """
    if raw is None or (isinstance(raw, list) and not raw):
        return ()
    if not connection_host:
        raise ValueError("the connection URL has no hostname")
    if not isinstance(raw, list):
        raise ValueError("signed_uploads must be a list")
    if len(raw) > _MAX_SETTINGS_ENTRIES:
        raise ValueError(
            f"signed_uploads must not have more than {_MAX_SETTINGS_ENTRIES} entries"
        )
    entries = [
        _validate_entry(entry, position, connection_host)
        for position, entry in enumerate(raw)
    ]
    _check_entries_together(entries)
    return tuple(entries)


def _url_hostname(url: Any) -> str | None:
    """Return the lower-case hostname of ``url``, or ``None`` if it has none."""
    if not isinstance(url, str):
        return None
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def signed_upload_settings_error(raw: Any, *, transport: Any, url: Any) -> str | None:
    """Why stored settings are unusable, or None when absent, inert or valid."""
    if not isinstance(transport, str) or transport not in SIGNED_UPLOAD_TRANSPORTS:
        return None
    try:
        parse_signed_upload_settings(raw, connection_host=_url_hostname(url))
    except ValueError as exc:
        return str(exc)
    except Exception:
        return _UNREADABLE_SETTINGS_MESSAGE
    return None


def connection_signed_uploads(
    connection: Any,
) -> tuple[SignedUploadBinding | SignedUploadIssuer, ...]:
    """Read and validate the settings carried by ``connection``.

    A connection that is not a mapping, or whose transport does not take part,
    has no settings.  Stored settings that are not valid raise ``ValueError``.
    """
    if not isinstance(connection, Mapping):
        return ()
    transport = connection.get("transport")
    if not isinstance(transport, str) or transport not in SIGNED_UPLOAD_TRANSPORTS:
        return ()
    raw = connection.get("signed_uploads")
    return parse_signed_upload_settings(
        raw, connection_host=_url_hostname(connection.get("url"))
    )


@dataclass(frozen=True)
class SignedUploadTicket:
    """A validated reply of an issuing tool."""

    file_id: str  # the connector's own id for the file
    upload_url: str  # not yet checked against a binding
    headers: Mapping[str, str]  # headers the connector asks to send with the PUT
    expires_at: datetime  # timezone-aware
    # When the connector reports when the file id stops working: that moment in
    # UTC.  ``None`` when the reply has no such time or it cannot be read; the
    # upload does not depend on it.
    file_id_expires_at: datetime | None = None


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

    # The file id's own expiry is optional and only informs the model, so a
    # value that cannot be read, or cannot be expressed in UTC, counts as absent.
    file_id_expires_at: datetime | None = None
    file_id_expiry_raw = payload.get("expires_at")
    if isinstance(file_id_expiry_raw, str):
        try:
            parsed = datetime.fromisoformat(file_id_expiry_raw)
            if parsed.tzinfo is not None:
                file_id_expires_at = parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            pass

    return SignedUploadTicket(
        file_id=file_id,
        upload_url=upload_url,
        headers=MappingProxyType(dict(headers)),
        expires_at=expires_at,
        file_id_expires_at=file_id_expires_at,
    )


def check_signed_upload_url(
    raw: str, binding: SignedUploadBinding | SignedUploadIssuer
) -> httpx.URL:
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
