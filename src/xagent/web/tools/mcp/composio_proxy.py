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

Nothing raised here carries the API key, the connected account id or the
Composio user id.
"""

from __future__ import annotations

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
MAX_ERROR_TEXT_CHARS = 1000
_REDACTED = "[redacted]"
# A path Composio resolves against the account's own host: one leading "/",
# no backslash, whitespace or control character, so no URL parser on the
# way can read it as another host ("//evil", "/\\evil").
_RELATIVE_ENDPOINT_RE = re.compile(r"/(?!/)[^\\\s\x00-\x1f\x7f]*")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProxyResponse:
    """The provider's answer as Composio relays it."""

    status: int
    data: Any

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


def redact(value: Any) -> Any:
    """``value`` with every launch secret replaced, in nested values too."""
    secrets = [os.environ.get(name) for name in _LAUNCH_SECRET_ENVS]
    return _redact(value, [secret for secret in secrets if secret])


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, _REDACTED)
        return value
    if isinstance(value, Mapping):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    return value


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


def proxy_request(
    method: str,
    endpoint: str,
    *,
    allowed_hosts: Collection[str],
    query: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    headers: Mapping[str, str] | None = None,
    body: Any = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ProxyResponse:
    """Send ``method endpoint`` through Composio as the launched grant.

    ``allowed_hosts`` are the hostnames an absolute ``endpoint`` may name.
    ``query`` is a mapping, or a sequence of pairs when a parameter repeats;
    ``body`` is sent as JSON. Raises ``ComposioProxyError`` when Composio does
    not relay the call, and ``ValueError`` before sending anything when the
    endpoint or a launch variable is unusable.
    """
    if isinstance(allowed_hosts, str):
        raise TypeError("allowed_hosts must be a collection of hostnames, not a str")
    endpoint = _checked_endpoint(endpoint, allowed_hosts)
    pairs = query.items() if isinstance(query, Mapping) else query
    api_key = _required_env(COMPOSIO_API_KEY_ENV)
    connected_account_id = _required_env(COMPOSIO_CONNECTED_ACCOUNT_ID_ENV)
    user_id = _required_env(COMPOSIO_USER_ID_ENV)

    parameters = [
        {"name": str(name), "value": str(value), "type": "query"}
        for name, value in pairs
    ] + [
        {"name": str(name), "value": str(value), "type": "header"}
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
        raise ComposioProxyError(message, status=status, slug=slug)
    if not isinstance(answer, dict) or type(answer.get("status")) is not int:
        raise ComposioProxyError(
            "Composio proxy returned an answer without the provider's status",
            status=status,
        )
    return ProxyResponse(status=answer["status"], data=answer.get("data"))
