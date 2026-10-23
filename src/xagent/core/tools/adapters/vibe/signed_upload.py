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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

_REMOTE_TRANSPORTS = frozenset({"streamable_http", "sse"})
_HOST_FORBIDDEN_CHARS = frozenset("*/:@")


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
