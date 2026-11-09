"""Create-only Jira ticket connector, authenticated with an API token.

Why this exists next to ``jira.py``: that connector runs on the connecting
user's Atlassian OAuth login and exposes every Jira tool (search, read, update,
delete, transitions, comments) across every project the login can reach. An
agent that serves other people on its owner's credentials -- for example one
embedded for guests that only needs to file tickets -- would let any visitor
search and read other reporters' tickets. This server offers exactly one tool,
``jira_ticket_create``, and fixes the site and project in the connection
configuration, so the worst a visitor can do is file a ticket in that project.

There is deliberately no read, search, get, update, comment, transition or
delete tool, and no argument selects a project, site or URL.

Configuration (per-user encrypted env, like the other key-based connectors):
``JIRA_TICKET_SITE``, ``JIRA_TICKET_EMAIL``, ``JIRA_TICKET_API_TOKEN`` and
``JIRA_TICKET_PROJECT_KEY``. Requests use HTTP Basic auth with the email and
the API token.

Scope note: written against the published Jira Cloud REST v2 contract and
covered by mocked tests; it has not been exercised against a live site.
"""

import json
import logging
import os
import re
import socket
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

from ....core.tools.core.web_content import get_trusted_proxy_url
from ....core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    reject_private_network_host,
)
from ...utils.graphql_errors import truncate_error_text
from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira-ticket-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("jira-ticket-mcp")

# Same session posture as freshdesk.py/zendesk.py: trust_env=False so an
# ambient proxy cannot bypass the address check in _base_url(), and no .netrc
# credential can follow a redirect; the CA bundle is re-applied explicitly
# because trust_env=False also turns off requests' own lookup of it.
_session = requests.Session()
_session.trust_env = False
_ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
if _ca_bundle:
    _session.verify = _ca_bundle

ATLASSIAN_DOMAIN = "atlassian.net"

# One DNS label, ASCII, internal hyphens only. Anchored with \A/\Z because $
# also matches before a trailing newline, which would reach the Host header.
_LABEL_PATTERN = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
# Jira's default project-key shape: an uppercase letter, then 1-9 uppercase
# letters, digits or underscores.
_PROJECT_KEY_PATTERN = re.compile(r"\A[A-Z][A-Z0-9_]{1,9}\Z")
_ISSUE_KEY_PATTERN = re.compile(r"\A[A-Z][A-Z0-9_]*-[0-9]+\Z")
# Credentials are used as the Basic auth pair, so they may not contain the
# separator or whitespace.
_EMAIL_PATTERN = re.compile(r"\A[^@\s:]+@[^@\s:]+\Z")

CREATE_TIMEOUT_SECONDS = 30
DEFAULT_ISSUE_TYPE = "Task"
# Jira's own limit for the summary field.
MAX_SUMMARY_CHARS = 255
# Jira's description cap is 32,767 characters; stay under it with margin.
MAX_DESCRIPTION_CHARS = 32000
MAX_LABELS = 10
MAX_ERROR_DETAILS = 5
MAX_ERROR_DETAIL_CHARS = 200
MAX_ERROR_MESSAGE_CHARS = 800


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} environment variable is missing or empty")
    return value


def _site() -> str:
    """Return the normalized ``https://<label>.atlassian.net`` site URL.

    Accepts ``acme``, ``acme.atlassian.net`` or ``https://acme.atlassian.net``
    (case and surrounding whitespace are normalized). Anything that could
    change *which host* is contacted -- another domain, extra labels, a path,
    port, userinfo, query or fragment, or a plain-http scheme -- is rejected.
    """
    raw = _env("JIRA_TICKET_SITE").lower()
    if raw.startswith("https://"):
        raw = raw[len("https://") :]
    raw = raw.removesuffix("/").removesuffix(f".{ATLASSIAN_DOMAIN}")
    if not _LABEL_PATTERN.match(raw):
        raise ValueError(
            "JIRA_TICKET_SITE must be your Jira Cloud site -- the 'acme' in "
            f"acme.{ATLASSIAN_DOMAIN} (or that full hostname or https URL), "
            "with no path, port or other host"
        )
    return f"https://{raw}.{ATLASSIAN_DOMAIN}"


def _base_url() -> str:
    """Return the site URL, refusing one whose name resolves privately.

    ``<label>.atlassian.net`` is a name in a zone only Atlassian controls, so
    a user cannot point it at an internal address. The resolve-and-check is
    kept anyway, as in freshdesk.py/zendesk.py: it is cheap, it costs no
    extra moving parts, and it keeps this connector's SSRF posture identical
    to its siblings instead of a one-off argument about why it differs.
    """
    site = _site()
    hostname = site.removeprefix("https://")
    try:
        resolved = socket.getaddrinfo(
            hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
        for *_, sockaddr in resolved:
            reject_private_network_host(str(sockaddr[0]))
    except PrivateNetworkHostError as exc:
        raise ValueError(f"JIRA_TICKET_SITE is not allowed: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"JIRA_TICKET_SITE host could not be resolved: {exc}") from exc
    return site


def _project_key() -> str:
    key = _env("JIRA_TICKET_PROJECT_KEY")
    if not _PROJECT_KEY_PATTERN.match(key):
        raise ValueError(
            "JIRA_TICKET_PROJECT_KEY must be a Jira project key such as 'SUP' "
            "(uppercase letters, digits or underscores, 2-10 characters)"
        )
    return key


def _credentials() -> tuple[str, str]:
    email = _env("JIRA_TICKET_EMAIL")
    if not _EMAIL_PATTERN.match(email):
        raise ValueError("JIRA_TICKET_EMAIL must be the Atlassian account email")
    return email, _env("JIRA_TICKET_API_TOKEN")


def _safe_text(value: Any) -> str:
    """Text safe to return to the model: redacted, token-free and bounded."""
    text = redact_sensitive_text(str(value))
    token = (os.environ.get("JIRA_TICKET_API_TOKEN") or "").strip()
    if token:
        text = text.replace(token, "[redacted]")
    return truncate_error_text(text)


def _error_details(response: requests.Response) -> str:
    """Field-level detail from a Jira 400 body, never the raw body.

    Jira reports validation failures as ``errorMessages`` (a list of strings)
    and ``errors`` (a field-id to message map). Only those, capped, are used;
    any other body shape yields no detail.
    """
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    messages = payload.get("errorMessages")
    if isinstance(messages, list):
        parts.extend(str(m) for m in messages if isinstance(m, str) and m)
    errors = payload.get("errors")
    if isinstance(errors, dict):
        parts.extend(
            f"{field}: {message}"
            for field, message in errors.items()
            if isinstance(message, str) and message
        )
    shown = [part[:MAX_ERROR_DETAIL_CHARS] for part in parts[:MAX_ERROR_DETAILS]]
    detail = "; ".join(shown)
    if len(parts) > MAX_ERROR_DETAILS:
        detail += f"; (+{len(parts) - MAX_ERROR_DETAILS} more)"
    return _safe_text(detail)[:MAX_ERROR_MESSAGE_CHARS]


def _status_message(response: requests.Response) -> str:
    """Short, body-free text for an HTTP error status.

    A 5xx never carries "try again" advice: the ticket may already exist.
    """
    status = response.status_code
    if status == 400:
        detail = _error_details(response)
        base = "Jira rejected the request (HTTP 400)"
        return f"{base}: {detail}" if detail else base
    if status == 401:
        return (
            "Jira rejected the credentials (HTTP 401) -- check "
            "JIRA_TICKET_EMAIL and JIRA_TICKET_API_TOKEN"
        )
    if status == 403:
        return (
            "Jira denied the request (HTTP 403) -- the account needs the "
            "Browse Projects and Create Issues permissions on this project"
        )
    if status == 404:
        return (
            "Jira could not find the project or site (HTTP 404) -- check "
            "JIRA_TICKET_SITE and JIRA_TICKET_PROJECT_KEY"
        )
    if status == 429:
        retry_after = response.headers.get("Retry-After", "")
        suffix = (
            f"; retry after {retry_after}s" if retry_after.strip().isdigit() else ""
        )
        return f"Jira rate limited the request (HTTP 429){suffix}"
    if status >= 500:
        # The server answered after the request was sent; the caller adds what
        # that means for a create. No "try again" advice here.
        return f"Jira is unavailable (HTTP {status})"
    return f"Jira returned an unexpected response (HTTP {status})"


def _send(
    site: str,
    method: str,
    path: str,
    *,
    json_data: dict[str, Any] | None = None,
    timeout: int = CREATE_TIMEOUT_SECONDS,
) -> requests.Response:
    """Issue one request; raise a token-free error for any failure."""
    email, token = _credentials()
    url = f"{site}/rest/api/2{path}"
    try:
        proxy_url = get_trusted_proxy_url()
    except PrivateNetworkHostError as exc:
        raise type(exc)(_safe_text(exc)) from None
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else {}

    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    try:
        response = _session.request(
            method=method,
            url=url,
            proxies=proxies,
            # Never follow a redirect with the Basic credential attached.
            allow_redirects=False,
            auth=(email, token),
            headers=headers,
            json=json_data,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        # Keep the exception class (the create path distinguishes timeouts)
        # but rebuild its text: it can embed a proxy URL with credentials.
        raise type(exc)(f"Jira request failed: {_safe_text(exc)}") from None

    if 300 <= response.status_code < 400:
        raise RuntimeError(
            f"Jira returned an unexpected redirect (HTTP {response.status_code}); "
            "refusing to follow it with credentials attached"
        )
    if response.status_code >= 400:
        # The cause carries the response so _outcome_unknown can read the
        # status, the same shape jira.py's _request_absolute raises.
        raise RuntimeError(_status_message(response)) from requests.HTTPError(
            response=response
        )
    return response


def _validated_labels(labels: list[str] | None) -> list[str]:
    if not labels:
        return []
    if not isinstance(labels, list):
        raise ValueError("labels must be a list of strings")
    if len(labels) > MAX_LABELS:
        raise ValueError(f"at most {MAX_LABELS} labels are allowed")
    cleaned: list[str] = []
    for label in labels:
        value = label.strip() if isinstance(label, str) else ""
        if not value or any(ch.isspace() for ch in value):
            raise ValueError(
                "each label must be a non-empty string without spaces "
                "(Jira labels cannot contain spaces)"
            )
        cleaned.append(value)
    return cleaned


def _validated_text(name: str, value: str, limit: int) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ValueError(f"{name} is required")
    if len(text) > limit:
        raise ValueError(f"{name} is too long ({len(text)} characters; max {limit})")
    return text


_OUTCOME_UNKNOWN_ADVICE = (
    "The ticket may have been created -- check the project before retrying"
)


def _outcome_unknown(exc: Exception) -> bool:
    """True if a failed create may nevertheless have created the ticket."""
    if isinstance(exc, requests.ConnectTimeout):
        return False  # the connection was never established: nothing was sent
    if isinstance(exc, requests.RequestException):
        # Anything else raised while sending or reading the response may come
        # after Jira committed the issue (timeouts, resets, a body that cannot
        # be decoded, ...).
        return True
    response = getattr(exc.__cause__, "response", None)
    status = getattr(response, "status_code", None)
    return isinstance(status, int) and status >= 500


@mcp.tool()
def jira_ticket_create(
    summary: str,
    description: str,
    issue_type: str = "",
    labels: list[str] | None = None,
) -> str:
    """
    Create one Jira ticket in the project this connection is configured for.
    This is the only thing this connector can do: it cannot read, search,
    update, comment on or delete tickets, and the project is fixed by the
    connection -- it cannot be chosen here.

    summary: one-line title (max 255 characters).
    description: plain-text body (max 32000 characters).
    issue_type: Jira issue type name, e.g. "Bug" or "Task"; empty means "Task".
    labels: optional labels; each is one word without spaces (max 10).

    Calling this again creates another ticket, so do not repeat a call whose
    outcome is unknown without first checking the project.

    Returns the issue key and its browse URL.
    """
    try:
        text_summary = _validated_text("summary", summary, MAX_SUMMARY_CHARS)
        text_description = _validated_text(
            "description", description, MAX_DESCRIPTION_CHARS
        )
        clean_labels = _validated_labels(labels)
        project_key = _project_key()
        site = _base_url()

        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "summary": text_summary,
            "description": text_description,
            "issuetype": {"name": (issue_type or "").strip() or DEFAULT_ISSUE_TYPE},
        }
        if clean_labels:
            fields["labels"] = clean_labels

        try:
            response = _send(site, "POST", "/issue", json_data={"fields": fields})
        except Exception as exc:
            if not _outcome_unknown(exc):
                raise
            # The advice goes first and the detail is truncated once, so a long
            # underlying message cannot cut the advice off.
            safe_message = f"{_OUTCOME_UNKNOWN_ADVICE}. Detail: {_safe_text(exc)}"
            logger.error("Error creating Jira ticket: %s", safe_message)
            return _error(safe_message)
        try:
            created = response.json()
        except ValueError:
            raise RuntimeError(
                "Jira returned a non-JSON response to the create request; the "
                "ticket may have been created -- check the project before retrying"
            ) from None
        key = created.get("key") if isinstance(created, dict) else None
        if not isinstance(key, str) or not _ISSUE_KEY_PATTERN.match(key):
            raise RuntimeError(
                "Jira did not return a usable issue key; the ticket may have "
                "been created -- check the project before retrying"
            )
    except Exception as exc:
        safe_message = _safe_text(exc)
        logger.error("Error creating Jira ticket: %s", safe_message)
        return _error(safe_message)

    return _success(key=key, url=f"{site}/browse/{key}")


if __name__ == "__main__":
    mcp.run(transport="stdio")
