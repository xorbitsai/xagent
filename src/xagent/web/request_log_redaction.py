"""Keep credentials out of server request and handshake logs.

Server and proxy logs record request targets verbatim. Any credential a client
still puts in a query string (a stale client's retired ``?token=``, a
short-lived file ``ticket``, an OAuth ``code``) would otherwise be written to
disk and shipped to log storage. At debug level the WebSocket stack also logs
every handshake header -- including the ``Sec-WebSocket-Protocol`` and
``Authorization: Bearer`` credential carriers -- before the application sees
the request. Application handlers log the path only; the ASGI server's own
access, handshake and handshake-header lines go through
:class:`RequestTargetRedactionFilter`.
"""

from __future__ import annotations

import logging

from ..core.utils.security import (
    REDACTED,
    is_sensitive_param_name,
    redact_url_query_credentials,
)

_LOGGER_NAMES = ("uvicorn.access", "uvicorn.error")

# Handshake header lines of both WebSocket implementations uvicorn drives with
# the ``websockets`` package -- the legacy protocol (``--ws websockets``,
# ``websockets.legacy.server``) and the Sans-I/O one (``--ws
# websockets-sansio``, ``websockets.server``): one record per header, logged at
# debug to ``uvicorn.error`` with ``(name, value)`` args. ``<`` is the request,
# ``>`` the response.
_HEADER_LOG_FORMATS = frozenset({"< %s: %s", "> %s: %s"})
# Headers whose whole value is a credential, besides names that already read
# as credentials (``X-Api-Key``, ``X-Auth-Token``, see
# ``is_sensitive_param_name(..., url_query=True)``).
_SENSITIVE_HEADER_NAMES = frozenset({"cookie", "set-cookie"})
# ``Sec-WebSocket-Protocol`` items carrying a login credential
# (``xagent-auth-token.<jwt>``); the protocol names stay readable.
_CREDENTIAL_SUBPROTOCOL_PREFIXES = ("xagent-auth-token.",)


def redact_request_target(target: str) -> str:
    """Return ``target`` with credential-like query values replaced.

    Accepts a request target (``/path?query``) or an absolute URL. Parameter
    names and non-sensitive values are kept so logs stay useful. The query is
    parsed, encoded and masked by the same function the text and record
    redactors use (``redact_url_query_credentials``), so a request target
    cannot disagree with them about which names are sensitive, how a value is
    encoded, or what happens to a target that cannot be parsed. A target with
    nothing to redact is returned unchanged (never re-serialized), so the log
    still matches the request on the wire.
    """
    if "?" not in target:
        return target
    return redact_url_query_credentials(target)


def redact_header_value(name: str, value: str) -> str:
    """Return a header ``value`` safe to log, keeping non-secret detail.

    ``Authorization`` keeps its scheme, ``Sec-WebSocket-Protocol`` keeps its
    protocol names, and other credential headers are replaced outright.
    """
    lowered = name.strip().lower()
    if lowered in ("authorization", "proxy-authorization"):
        scheme, separator, credential = value.strip().partition(" ")
        if separator and credential:
            return f"{scheme} {REDACTED}"
        return REDACTED if value else value
    if lowered == "sec-websocket-protocol":
        items = [item.strip() for item in value.split(",")]
        if not any(item.startswith(_CREDENTIAL_SUBPROTOCOL_PREFIXES) for item in items):
            return value
        return ", ".join(
            next(
                (
                    prefix + REDACTED
                    for prefix in _CREDENTIAL_SUBPROTOCOL_PREFIXES
                    if item.startswith(prefix)
                ),
                item,
            )
            for item in items
        )
    if lowered.startswith("sec-websocket-"):
        # Key/accept nonces, version and extensions: handshake diagnostics.
        return value
    if lowered in _SENSITIVE_HEADER_NAMES or is_sensitive_param_name(
        lowered, url_query=True
    ):
        return REDACTED if value else value
    return value


class RequestTargetRedactionFilter(logging.Filter):
    """Redact credentials in server access/handshake log args."""

    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn's access and handshake lines, and the WebSocket stack's
        # header lines, always pass the request target / header as positional
        # ``%s`` arguments, never pre-formatted into ``msg``.
        args = record.args
        if (
            record.msg in _HEADER_LOG_FORMATS
            and isinstance(args, tuple)
            and len(args) == 2
        ):
            name, value = args
            name = str(name)
            text = value if isinstance(value, str) else str(value)
            safe = redact_header_value(name, text)
            if safe != text:
                record.args = (name, safe)
            return True
        if isinstance(args, tuple) and args:
            record.args = tuple(
                redact_request_target(arg)
                if isinstance(arg, str) and "?" in arg
                else arg
                for arg in args
            )
        return True


_FILTER = RequestTargetRedactionFilter()


def install_request_log_redaction() -> None:
    """Attach the filter to the ASGI server loggers (idempotent).

    Logger-level filters survive later ``dictConfig`` calls (for example the
    server applying its own logging config), which only replace handlers.
    """
    for name in _LOGGER_NAMES:
        logger = logging.getLogger(name)
        if _FILTER not in logger.filters:
            logger.addFilter(_FILTER)
