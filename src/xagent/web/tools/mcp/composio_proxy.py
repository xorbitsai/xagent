"""Send one API call through Composio's Proxy Execute endpoint (#3645).

A Composio-backed grant holds Composio ids, not a provider token: Composio
keeps the token and injects it into the call. A connector's MCP subprocess is
launched with ``COMPOSIO_API_KEY``, ``COMPOSIO_CONNECTED_ACCOUNT_ID`` and
``COMPOSIO_USER_ID`` (``tools/config.py``); a connector's runtime sends its
calls through ``proxy_request``. Facebook and Instagram keep their own
Graph-specific client (``meta_graph.py``).

Composio adds the user's token to the call, and nothing here relies on it
refusing a foreign host, so an absolute endpoint must be on one of the hosts
the caller allows. A path is resolved by Composio against the connected
account's own host.

Two kinds of failure stay apart:

- Composio itself refusing or failing (bad key, unknown account, rate limit,
  an answer that is not a proxy result) raises ``ComposioProxyError``.
- The provider's own answer, error or not, is returned as a ``ProxyResponse``
  with the provider's status, so each connector maps its own API's errors.

Nothing raised here carries the API key, the connected account id, the
Composio user id, a provider token echoed back by Composio, or a request body
Composio echoes back (a Gmail message's body and attachments travel in it).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ....core.utils import public_requests as requests

COMPOSIO_PROXY_URL = "https://backend.composio.dev/api/v3.1/tools/execute/proxy"
COMPOSIO_API_KEY_ENV = "COMPOSIO_API_KEY"
COMPOSIO_CONNECTED_ACCOUNT_ID_ENV = "COMPOSIO_CONNECTED_ACCOUNT_ID"
COMPOSIO_USER_ID_ENV = "COMPOSIO_USER_ID"
_LAUNCH_SECRET_ENVS = (
    COMPOSIO_API_KEY_ENV,
    COMPOSIO_CONNECTED_ACCOUNT_ID_ENV,
    COMPOSIO_USER_ID_ENV,
)
DEFAULT_TIMEOUT_SECONDS = 30
# The largest JSON body one Proxy Execute call may carry. Composio's API is
# served by Vercel functions (its answers can carry ``x-vercel-id``), whose
# documented request-body limit is 4.5 MB; the decimal value stays below that
# however the platform rounds it. A body above this is refused here, with both
# sizes in the message, instead of failing at Composio's edge with an opaque
# 413. This has not been confirmed against a live project.
MAX_JSON_BODY_BYTES = 4_500_000
MAX_ERROR_TEXT_CHARS = 1000
_REDACTED = "[redacted]"
# A provider token can be echoed back in an error the proxy relays (for
# example an upstream Authorization header). Its value is not a launch
# variable we know, so it is redacted by shape: a Bearer credential, or a
# Google access token's ``ya29.`` prefix.
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[^\s\"',;]+", re.IGNORECASE)
_GOOGLE_ACCESS_TOKEN_RE = re.compile(r"\bya29\.[A-Za-z0-9._-]+")
# A JSON body member that carries a message: the ``raw`` of a Gmail message
# is base64, so a vendor echoing the request body leaks the body and the
# attachments. Redacted by shape too, in case the echo re-encodes it.
_RAW_MEMBER_RE = re.compile(r'"raw"\s*:\s*"(?:[^"\\]|\\.)*"')
# A request-body string at least this long is treated as content worth
# redacting if the error text repeats it; ordinary words in an error are
# shorter than this and survive.
_MIN_ECHOED_BODY_CHARS = 32
# C0 control characters have no place in a query or header value; a bare
# CR/LF there is what request splitting needs. Tab is the one C0 exception
# HTTP allows in a header value.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")
# A path Composio resolves against the account's own host: one leading "/",
# no backslash, whitespace or control character, so no URL parser on the
# way can read it as another host ("//evil", "/\\evil").
_RELATIVE_ENDPOINT_RE = re.compile(r"/(?!/)[^\\\s\x00-\x1f\x7f]*")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProxyResponse:
    """The provider's answer as Composio relays it.

    ``headers`` are the provider response headers Composio returns (a 3xx
    carries ``location`` there). A binary answer puts its bytes in
    ``binary_data`` (``url``, ``content_type``, ``size``) and carries
    ``data=None``: Composio stores the file and returns a signed URL, it
    never relays the bytes in ``data``.
    """

    status: int
    data: Any
    headers: Mapping[str, str] | None = None
    binary_data: Mapping[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class ComposioProxyError(RuntimeError):
    """Composio itself refused or failed the call; the provider may never
    have seen it. ``status`` is Composio's HTTP status, None when Composio
    could not be reached."""

    def __init__(
        self, message: str, *, status: int | None = None, slug: str | None = None
    ) -> None:
        # Redact before truncating, so a secret cut in half cannot survive.
        super().__init__(_truncated(redact(message)))
        self.status = status
        self.slug = slug


def serves_composio_grant() -> bool:
    """Whether this connector process serves a Composio-backed grant: it is
    launched with Composio's ids instead of a provider token."""
    return bool(os.environ.get(COMPOSIO_CONNECTED_ACCOUNT_ID_ENV))


def redact(value: Any) -> Any:
    """``value`` with every launch secret and provider token replaced, in
    nested values too."""
    secrets = [os.environ.get(name) for name in _LAUNCH_SECRET_ENVS]
    return _redact(value, [secret for secret in secrets if secret])


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        return _redacted_text(value, secrets)
    if isinstance(value, Mapping):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    return value


def _redacted_text(value: str, secrets: list[str]) -> str:
    for secret in secrets:
        value = value.replace(secret, _REDACTED)
    value = _BEARER_TOKEN_RE.sub(f"Bearer {_REDACTED}", value)
    return _GOOGLE_ACCESS_TOKEN_RE.sub(_REDACTED, value)


def _redact_echoed_body(message: str, body: Any) -> str:
    """``message`` with a request body it echoes back removed.

    A vendor error sometimes repeats the request it refused. A Gmail
    message's body and attachments travel base64-encoded in ``raw`` inside
    that body, so leaving the echo intact would put them in the tool result
    and the server log. The body's long strings are replaced by value, and
    any ``raw`` member by shape, before the message reaches an error.
    """
    if body is None:
        return message
    for value in _long_strings(body):
        message = message.replace(value, _REDACTED)
    return _RAW_MEMBER_RE.sub(f'"raw": "{_REDACTED}"', message)


def _long_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if len(value) >= _MIN_ECHOED_BODY_CHARS else []
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _long_strings(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _long_strings(child)]
    return []


def _truncated(text: str) -> str:
    if len(text) > MAX_ERROR_TEXT_CHARS:
        return f"{text[:MAX_ERROR_TEXT_CHARS]}...[truncated]"
    return text


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} environment variable is missing")
    return value


def _checked_endpoint(endpoint: str, allowed_hosts: Collection[str]) -> str:
    """An HTTPS URL on an allowed host, or a path Composio resolves against
    the connected account's own host (a Salesforce org's My Domain)."""
    if _CONTROL_CHARS_RE.search(endpoint):
        raise ValueError("endpoint must not contain control characters")
    if endpoint.startswith("/"):
        if not _RELATIVE_ENDPOINT_RE.fullmatch(endpoint):
            raise ValueError(
                "a path endpoint must start with a single '/' and contain no "
                "backslash, whitespace or control character"
            )
        return endpoint
    parts = urlsplit(endpoint)
    if parts.scheme != "https" or not parts.netloc:
        raise ValueError("endpoint must be an https URL or a path starting with '/'")
    if parts.username is not None or parts.port is not None:
        raise ValueError("endpoint must not carry credentials or a port")
    if (parts.hostname or "") not in {host.lower() for host in allowed_hosts}:
        raise ValueError("endpoint host is not one this connector calls")
    return endpoint


def _checked_parameter(value: Any, kind: str) -> str:
    """One parameter as text, refusing a control character in it."""
    text = str(value)
    if _CONTROL_CHARS_RE.search(text):
        raise ValueError(f"a {kind} contains a control character")
    return text


def _proxy_payload(
    method: str,
    endpoint: str,
    *,
    query: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    headers: Mapping[str, str] | None = None,
    body: Any = None,
) -> dict[str, Any]:
    """The JSON body one Proxy Execute call would carry, without sending it."""
    pairs = query.items() if isinstance(query, Mapping) else query
    connected_account_id = _required_env(COMPOSIO_CONNECTED_ACCOUNT_ID_ENV)
    user_id = _required_env(COMPOSIO_USER_ID_ENV)

    parameters = [
        {
            "name": _checked_parameter(name, "query parameter name"),
            "value": _checked_parameter(value, "query parameter value"),
            "type": "query",
        }
        for name, value in pairs
    ] + [
        {
            "name": _checked_parameter(name, "header name"),
            "value": _checked_parameter(value, "header value"),
            "type": "header",
        }
        for name, value in (headers or {}).items()
    ]
    payload: dict[str, Any] = {
        "endpoint": endpoint,
        "method": method.upper(),
        "connected_account_id": connected_account_id,
        # Required: the Composio project enforces 2FA on private connections.
        "user_id": user_id,
    }
    if parameters:
        payload["parameters"] = parameters
    if body is not None:
        payload["body"] = body
    return payload


def _checked_payload_size(
    payload: Mapping[str, Any], max_body_bytes: int | None = None
) -> None:
    limit = MAX_JSON_BODY_BYTES if max_body_bytes is None else max_body_bytes
    encoded_payload = json.dumps(payload)
    if len(encoded_payload) > limit:
        # Same serialization requests would send, so the sizes match what the
        # proxy would have received.
        raise ValueError(
            "the request body is too large to send through Composio: "
            f"{len(encoded_payload)} bytes of JSON, above the "
            f"{limit} byte limit; send less in one request "
            "(for example a smaller attachment)"
        )


def check_json_body_size(
    method: str,
    endpoint: str,
    *,
    query: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    headers: Mapping[str, str] | None = None,
    body: Any = None,
    max_body_bytes: int | None = None,
) -> None:
    """Refuse a proxy call whose JSON payload is above
    ``MAX_JSON_BODY_BYTES``, before anything is sent.

    ``proxy_request`` performs the same check. A connector that sends several
    calls in one tool invocation (one message each) calls this for each of
    them first, so an oversized later call cannot leave the earlier ones
    already delivered. ``max_body_bytes`` overrides the shared default for a
    connector whose real limit differs.
    """
    _checked_payload_size(
        _proxy_payload(method, endpoint, query=query, headers=headers, body=body),
        max_body_bytes,
    )


def proxy_request(
    method: str,
    endpoint: str,
    *,
    allowed_hosts: Collection[str],
    query: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    headers: Mapping[str, str] | None = None,
    body: Any = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_body_bytes: int | None = None,
) -> ProxyResponse:
    """Send ``method endpoint`` through Composio as the launched grant.

    ``allowed_hosts`` are the hostnames an absolute ``endpoint`` may name.
    ``query`` is a mapping, or a sequence of pairs when a parameter repeats;
    ``body`` is sent as JSON. Raises ``ComposioProxyError`` when Composio does
    not relay the call, and ``ValueError`` before sending anything when the
    endpoint, a launch variable or a parameter is unusable, or when the JSON
    body would exceed ``MAX_JSON_BODY_BYTES``. ``max_body_bytes`` overrides
    that shared default, for a connector whose real limit differs.
    """
    if isinstance(allowed_hosts, str):
        raise TypeError("allowed_hosts must be a collection of hostnames, not a str")
    endpoint = _checked_endpoint(endpoint, allowed_hosts)
    api_key = _required_env(COMPOSIO_API_KEY_ENV)
    payload = _proxy_payload(method, endpoint, query=query, headers=headers, body=body)
    _checked_payload_size(payload, max_body_bytes)

    try:
        response = requests.request(
            method="POST",
            url=COMPOSIO_PROXY_URL,
            headers={
                "x-api-key": api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except Exception as exc:
        # Type only: a connection error's text can name the request.
        logger.warning(
            "Composio proxy request failed (exception_type=%s)", type(exc).__name__
        )
        raise ComposioProxyError("Composio proxy could not be reached") from None
    try:
        answer = response.json()
    except Exception:
        answer = None

    status = int(response.status_code)
    if not 200 <= status < 300:
        error = answer.get("error") if isinstance(answer, dict) else None
        error = error if isinstance(error, dict) else {}
        slug = error.get("slug") if isinstance(error.get("slug"), str) else None
        detail = error.get("message")
        message = f"Composio proxy request failed with HTTP {status}"
        if slug:
            message = f"{message} ({slug})"
        if isinstance(detail, str) and detail:
            message = f"{message}: {detail}"
        raise ComposioProxyError(
            _redact_echoed_body(message, body), status=status, slug=slug
        )
    if not isinstance(answer, dict) or type(answer.get("status")) is not int:
        raise ComposioProxyError(
            "Composio proxy returned an answer without the provider's status",
            status=status,
        )
    provider_status = answer["status"]
    if "data" not in answer and provider_status != 204:
        # A 2xx without a body is not a successful provider answer (R3: a
        # truncated answer used to reach a callback and fail opaquely there).
        raise ComposioProxyError(
            "Composio proxy returned an answer without the provider's data",
            status=status,
        )
    answer_headers = answer.get("headers")
    binary_data = answer.get("binary_data")
    return ProxyResponse(
        status=provider_status,
        data=answer.get("data"),
        headers=answer_headers if isinstance(answer_headers, Mapping) else None,
        binary_data=binary_data if isinstance(binary_data, Mapping) else None,
    )
