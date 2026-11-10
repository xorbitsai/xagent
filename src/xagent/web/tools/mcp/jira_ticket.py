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
delete tool, and no argument selects a project, site or URL. Attachments are
only ever uploaded to the issue the same call just created.

Attachments are read only from directories named in
``XAGENT_JIRA_FILE_ALLOWED_DIRS``. The shared helper treats an unset or blank
value as "allow the working directory"; this server does not, because a launch
path that does not provide an upload allowlist would then let a relative path
such as ``.env`` be read from the server's own directory and uploaded. Unset
or blank therefore means attachments are unavailable in that run.

Configuration (per-user encrypted env, like the other key-based connectors):
``JIRA_TICKET_SITE``, ``JIRA_TICKET_EMAIL``, ``JIRA_TICKET_API_TOKEN`` and
``JIRA_TICKET_PROJECT_KEY``. Requests use HTTP Basic auth with the email and
a classic (unscoped) Atlassian API token: only that kind of token works against
``https://<site>.atlassian.net``; scoped tokens and service accounts must go
through ``api.atlassian.com/ex/jira/{cloudId}``, which this connector does not
use.

The description is sent as Atlassian Document Format (REST v3), built from the
plain text, so Jira shows it literally: the wiki markup that REST v2 would
render (links, external images, macros) is never interpreted.

Scope note: written against the published Jira Cloud REST v3 contract and
covered by mocked tests; it has not been exercised against a live site.
"""

import json
import logging
import os
import re
import socket
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, NewConnectionError

from ....core.utils import public_requests as requests
from ....core.utils.security import (
    PrivateNetworkHostError,
    redact_sensitive_text,
    reject_private_network_host,
)
from ...utils.graphql_errors import truncate_error_text
from .jira_attachment import (
    UPLOAD_ALLOWED_DIRS_ENV_VAR,
    UPLOAD_TIMEOUT_SECONDS,
    confirmed_attachment,
    read_allowed_file,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jira-ticket-mcp")

mcp = FastMCP("jira-ticket-mcp")

# The public_requests facade's Session pins every connection to the address it
# validated as public, honours only the operator-trusted egress proxy, ignores
# ambient proxies and .netrc, and keeps the CA-bundle environment variables.
_session = requests.Session()

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
    a user cannot point it at an internal address. This check only turns a
    misconfiguration into an early, clear error before anything is sent; it
    does not pin the connection. The public_requests session does that: it
    resolves again, rejects non-public addresses and connects to the address
    it validated.
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
            "Jira rejected the credentials (HTTP 401) -- JIRA_TICKET_API_TOKEN "
            "must be a classic (unscoped) Atlassian API token created by the "
            "JIRA_TICKET_EMAIL account"
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


class _NothingSentError(RuntimeError):
    """A transport failure that certainly happened before Jira got the request."""


# NameResolutionError subclasses NewConnectionError in urllib3 2.x.
_CONNECT_STAGE_ERRORS = (NewConnectionError, ConnectTimeoutError)


def _certainly_not_sent(exc: requests.RequestException) -> bool:
    """True if the request provably never reached Jira.

    Only failures that occur before a connection carries the request qualify:
    a connect timeout, a rejected URL, a proxy that could not be reached, a
    urllib3 connect-stage failure (including a DNS failure), and the
    public_requests facade's own pre-send "host could not be resolved" error.
    TLS, read and decoding failures can follow a committed create and are not
    included.
    """
    if isinstance(
        exc,
        (requests.ConnectTimeout, requests.exceptions.InvalidURL),
    ):
        return True
    if isinstance(exc, requests.exceptions.ProxyError):
        return True
    if not isinstance(exc, requests.ConnectionError):
        return False
    # public_requests raises ConnectionError(...) from the OSError that the
    # DNS lookup failed with; urllib3 failures are passed as args[0] instead.
    if isinstance(exc.__cause__, OSError):
        return True
    reason = exc.args[0] if exc.args else None
    if isinstance(reason, MaxRetryError):
        reason = reason.reason
    return isinstance(reason, _CONNECT_STAGE_ERRORS)


def _send(
    site: str,
    method: str,
    path: str,
    *,
    json_data: dict[str, Any] | None = None,
    files: dict[str, tuple[str, bytes]] | None = None,
    timeout: int = CREATE_TIMEOUT_SECONDS,
) -> requests.Response:
    """Issue one request to the full API ``path``; raise a token-free error.

    ``files`` makes it a multipart upload: requests sets the Content-Type (it
    carries the boundary) and Jira blocks an upload without the XSRF opt-out.
    """
    email, token = _credentials()
    url = f"{site}{path}"
    headers = {"Accept": "application/json"}
    if files is not None:
        headers["X-Atlassian-Token"] = "no-check"

    try:
        response = _session.request(
            method=method,
            url=url,
            # Never follow a redirect with the Basic credential attached.
            allow_redirects=False,
            auth=(email, token),
            headers=headers,
            json=json_data,
            files=files,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        # Rebuild the text (it can embed a proxy URL with credentials). Keep
        # the class unless the request certainly never left, which the create
        # path must tell apart from a failure after Jira may have committed.
        message = f"Jira request failed: {_safe_text(exc)}"
        if _certainly_not_sent(exc):
            raise _NothingSentError(message) from None
        raise type(exc)(message) from None

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
    "The ticket may already have been created; do not call jira_ticket_create "
    "again, and tell the user it may already have been filed"
)


def _adf_document(text: str) -> dict[str, Any]:
    """Plain text as an Atlassian Document Format document.

    Blank-line-separated blocks become paragraphs and single newlines become
    hard breaks. Every line is one literal text node, so nothing in it is
    interpreted as markup. Empty lines and paragraphs are skipped because ADF
    rejects empty text nodes.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    content: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", normalized):
        nodes: list[dict[str, Any]] = []
        for line in block.split("\n"):
            if not line.strip():
                continue
            if nodes:
                nodes.append({"type": "hardBreak"})
            nodes.append({"type": "text", "text": line})
        if nodes:
            content.append({"type": "paragraph", "content": nodes})
    return {"type": "doc", "version": 1, "content": content}


_UPLOAD_FAILURE_GUIDANCE = (
    "The ticket exists; this connector cannot add files to it later, so "
    "mention the file in a follow-up instead of calling jira_ticket_create again."
)


def _upload_failure_reason(exc: Exception) -> str:
    """Reason for a failed upload, from this module's own wording only.

    Raw exception text is never included: it can carry host paths, proxy URLs
    or advice written for jira.py's tools (including "try again", which with
    this connector can only mean a second create and a duplicate ticket). The
    only text that comes from outside is Jira's own capped field/error details
    on a 4xx, which ``_error_details`` redacts and bounds.

    The failure is classified by type and cause, not by message: a failed HTTP
    response reaches here as the RuntimeError ``_send`` raises from an
    ``HTTPError`` that carries the response.
    """
    check = "ask the user to check the ticket's attachments"
    response = getattr(exc.__cause__, "response", None)
    status = getattr(response, "status_code", None)
    if response is not None and isinstance(status, int):
        if status == 403:
            return (
                "Jira denied the upload: the connection's account needs the "
                "Create Attachments permission in the project"
            )
        if status == 404:
            return "Jira could not find the ticket just created to attach the file"
        if status == 413:
            return (
                "Jira rejected the file as too large for the site's attachment "
                "limit, or the ticket reached its attachment limit"
            )
        if status == 429:
            return "Jira rate-limited the upload"
        if status >= 500:
            return (
                f"Jira answered HTTP {status}; the upload may have completed -- {check}"
            )
        if 400 <= status < 500:
            detail = _error_details(response)
            base = f"Jira rejected the upload (HTTP {status})"
            return f"{base}: {detail}" if detail else base
    if isinstance(exc, _NothingSentError):
        return "The upload could not be sent to Jira"
    if isinstance(exc, requests.exceptions.JSONDecodeError) or (
        isinstance(exc, ValueError) and not isinstance(exc, requests.RequestException)
    ):
        # A 2xx whose body is not JSON, or that confirmed no matching attachment.
        return f"Jira did not confirm the attachment -- {check}"
    if isinstance(exc, requests.RequestException):
        return (
            "The upload did not complete; the file may or may not be attached "
            f"-- {check}"
        )
    return "The upload failed"


def _attachment_failure(arg: str, filename: str, exc: Exception) -> dict[str, str]:
    """A failed_attachments entry for ``filename`` (a name, never a host path).

    ``argument`` names the tool argument the file came from, so two files with
    the same name stay distinguishable.
    """
    return {
        "argument": arg,
        "path": Path(filename).name or "attachment",
        "error": f"{_upload_failure_reason(exc)}. {_UPLOAD_FAILURE_GUIDANCE}",
    }


def _read_attachments(paths: dict[str, str]) -> list[tuple[str, str, bytes]]:
    """Read every named local file up front, before any ticket exists.

    ``paths`` maps argument name to the raw path; blank ones are skipped.
    Returns (argument, file name, bytes) per file. Raises ValueError naming the argument for the first rejected file.
    Holds up to three files of MAX_ATTACHMENT_BYTES in memory at once.
    """
    named = {arg: raw.strip() for arg, raw in paths.items() if raw and raw.strip()}
    if named and not (os.environ.get(UPLOAD_ALLOWED_DIRS_ENV_VAR) or "").strip():
        raise ValueError(
            f"{next(iter(named))}: attachments are not available in this run "
            "(no upload directory allowlist was provided). No ticket was "
            "created; mention the files in the description instead"
        )
    files: list[tuple[str, str, bytes]] = []
    for arg, raw in named.items():
        try:
            filename, data = read_allowed_file(raw)
            files.append((arg, filename, data))
        except (ValueError, OSError) as exc:
            raise ValueError(
                f"{arg} could not be attached: {_safe_text(exc)}. "
                "No ticket was created."
            ) from exc
    return files


def _upload(site: str, key: str, filename: str, data: bytes) -> dict[str, Any]:
    """Upload one pre-read file to ``key``; raises on any failure."""
    response = _send(
        site,
        "POST",
        f"/rest/api/3/issue/{key}/attachments",
        files={"file": (filename, data)},
        timeout=UPLOAD_TIMEOUT_SECONDS,
    )
    return confirmed_attachment(response.json(), len(data))


def _outcome_unknown(exc: Exception) -> bool:
    """True if a failed create may nevertheless have created the ticket."""
    if isinstance(exc, _NothingSentError):
        return False  # classified in _send: the request never reached Jira
    if isinstance(exc, requests.RequestException):
        # Anything else raised while sending or reading the response may come
        # after Jira committed the issue (timeouts, resets, TLS or decoding
        # failures, ...).
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
    attachment_path: str = "",
    attachment_path_2: str = "",
    attachment_path_3: str = "",
) -> str:
    """
    Create one Jira ticket in the project this connection is configured for.
    This is the only thing this connector can do: it cannot read, search,
    update, comment on or delete tickets, and the project is fixed by the
    connection -- it cannot be chosen here.

    summary: one-line title (max 255 characters).
    description: plain text, max 32000 characters; sent as Atlassian Document
        Format so Jira shows it literally -- no markup, links or images are
        interpreted.
    issue_type: Jira issue type name, e.g. "Bug" or "Task"; empty means "Task".
    labels: optional labels; each is one word without spaces (max 10).
    attachment_path, attachment_path_2, attachment_path_3: optional files to
        attach, up to three; they take files the user uploaded in this
        conversation. They are attached only to the ticket this call creates.
        A file that is missing, not allowed or too large is rejected before
        anything is sent, so no ticket is created. If the ticket is created
        but an upload then fails, the result lists it under failed_attachments;
        this connector cannot add files to a ticket later, so mention such a
        file in a follow-up instead of calling this tool again.

    Calling this again creates another ticket, so if a call fails with the
    ticket possibly already created, do not call this tool again; tell the
    user it may already have been filed.

    Returns the issue key and its browse URL, plus attachments (the files
    uploaded to the ticket) and failed_attachments (the files that could not
    be, each with its argument name, file name and reason).
    """
    try:
        # Jira Cloud rejects a summary containing newline characters.
        text_summary = _validated_text(
            "summary",
            " ".join(summary.split()) if isinstance(summary, str) else summary,
            MAX_SUMMARY_CHARS,
        )
        text_description = _validated_text(
            "description", description, MAX_DESCRIPTION_CHARS
        )
        clean_labels = _validated_labels(labels)
        project_key = _project_key()
        site = _base_url()
        # Local files are validated before the ticket exists, so a bad path
        # cannot leave an orphan ticket that the agent then recreates.
        files = _read_attachments(
            {
                "attachment_path": attachment_path,
                "attachment_path_2": attachment_path_2,
                "attachment_path_3": attachment_path_3,
            }
        )

        fields: dict[str, Any] = {
            "project": {"key": project_key},
            "summary": text_summary,
            "description": _adf_document(text_description),
            "issuetype": {"name": (issue_type or "").strip() or DEFAULT_ISSUE_TYPE},
        }
        if clean_labels:
            fields["labels"] = clean_labels

        try:
            response = _send(
                site, "POST", "/rest/api/3/issue", json_data={"fields": fields}
            )
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
                f"Jira returned a non-JSON response to the create request. "
                f"{_OUTCOME_UNKNOWN_ADVICE}"
            ) from None
        key = created.get("key") if isinstance(created, dict) else None
        if not isinstance(key, str) or not _ISSUE_KEY_PATTERN.match(key):
            raise RuntimeError(
                f"Jira did not return a usable issue key. {_OUTCOME_UNKNOWN_ADVICE}"
            )
    except Exception as exc:
        safe_message = _safe_text(exc)
        logger.error("Error creating Jira ticket: %s", safe_message)
        return _error(safe_message)

    attachments: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for arg, filename, data in files:
        try:
            attachments.append(_upload(site, key, filename, data))
        except Exception as exc:
            failed.append(_attachment_failure(arg, filename, exc))

    return _success(
        key=key,
        url=f"{site}/browse/{key}",
        attachments=attachments,
        failed_attachments=failed,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
