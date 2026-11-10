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

import base64
import bisect
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
# attachments. Redacted by shape too, in case the echo re-encodes it. A
# vendor that truncates the echo leaves the closing quote off, so the match
# ends at the next quote or at the end of the text.
_RAW_MEMBER_RE = re.compile(r'"raw"\s*:\s*"(?:[^"\\]|\\.)*(?:"|$)')
# A request-body string at least this long is treated as content worth
# redacting if the error text repeats it; ordinary words in an error are
# shorter than this and survive.
_MIN_ECHOED_BODY_CHARS = 32
# Credentials a vendor error can name directly. Their values are not launch
# variables we hold (a refresh token and a client secret are the grant's), so
# they are matched by member name, in JSON and in form-encoded text.
_SECRET_MEMBER_RE = re.compile(
    r'(?i)"(refresh_token|client_secret|access_token|id_token|api_key)"'
    r'\s*:\s*"(?:[^"\\]|\\.)*(?:"|$)'
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(refresh_token|client_secret|access_token|id_token|api_key)"
    r"=([^\s&\"',;]+)"
)
# The mail a decoded ``raw`` carries: its body and the headers the caller
# wrote. A vendor can echo the decoded message instead of the JSON body it
# was sent, so these values are redacted too.
_DECODED_HEADER_NAMES = frozenset({"subject", "to", "cc", "bcc"})
_MIN_DECODED_CONTENT_CHARS = 8
_MIN_BASE64_MESSAGE_CHARS = 64
_MIME_MARKERS = (
    b"Subject:",
    b"MIME-Version:",
    b"Content-Type:",
    b"Content-Transfer-Encoding:",
)
# C0 control characters have no place in a query parameter's name, a
# header's name or value, or an endpoint; a bare CR/LF there is what request
# splitting needs. A query *value* may legitimately carry newlines (a
# multi-line SOQL or Gmail search query), so it is passed through for
# Composio to encode. Tab is the one C0 exception HTTP allows in a header
# value.
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
    ``binary_data`` (``url``, ``content_type``, ``size``) and ``data`` is
    null or absent: Composio stores the file and returns a signed URL, it
    never relays the bytes in ``data``. ``data`` is optional and nullable in
    Composio's schema, so ``None`` means no body; whether an empty 2xx is a
    complete answer is the provider's own API's rule, not this client's.
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
    have seen it. ``status`` is Composio's HTTP status; None means the
    failure is not one of those statuses, either because Composio could not
    be reached or because it answered and the answer was refused here."""

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
    and the server log. The body's long strings are replaced by value, any
    ``raw`` member by shape (a truncated echo included), credential-shaped
    members by name, and the content of a decoded message by value.
    """
    for value in _long_strings(body):
        message = message.replace(value, _REDACTED)
    message = _RAW_MEMBER_RE.sub(f'"raw": "{_REDACTED}"', message)
    message = _SECRET_MEMBER_RE.sub(
        lambda match: f'"{match.group(1)}": "{_REDACTED}"', message
    )
    message = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}={_REDACTED}", message
    )
    if body is None:
        return message
    # A launch secret or provider token can straddle the cut, so redact them
    # while the text is still whole. ``ComposioProxyError`` redacts again
    # before its own cut; this one keeps the fragments below matching the text
    # the error will show.
    message = redact(message)
    # The decoded content is replaced before the error text is cut: a fragment
    # that crosses the cut would otherwise keep its visible prefix, and one
    # whose token ``redact`` rewrote would no longer match at all. The pass
    # walks the original text in order and stops once it has produced the
    # text that will be kept, so a replacement cannot shorten the text in
    # front of a fragment and pull it into the kept text.
    return _redacted_decoded_fragments(message, _decoded_mime_fragments(body))


# A decoded fragment is looked up by the characters it opens with, so a pass
# over the start of the text costs one key lookup per position instead of one
# search per fragment: a 4.5 MB message decodes to tens of thousands of lines.
_FRAGMENT_KEY_CHARS = 32
# How many positions one match is looked through while it grows over the
# fragments that overlap it. Reaching the limit takes the rest of the text as
# one region: that redacts more than needed, where stopping would leave
# fragment text in view.
_MAX_REGION_SCAN_CHARS = 4 * MAX_ERROR_TEXT_CHARS


def _redacted_decoded_fragments(message: str, fragments: list[str]) -> str:
    """``message`` with every decoded fragment that can reach its kept text
    removed.

    ``ComposioProxyError`` cuts the text to ``MAX_ERROR_TEXT_CHARS`` after
    this pass, so the kept text is the first characters of the result. The
    pass copies the original text in order, replaces a fragment where it
    starts with one marker, and stops once it has produced that many
    characters: a character or fragment after that point can never reach the
    kept text, however much an earlier replacement shortens the text. A
    fragment that crosses the cut is still met whole, because the scan sees
    its start.

    Each fragment is also matched in the form ``redact`` leaves it in,
    because the text has already been redacted. Overlapping fragments are
    replaced as one region, so a shorter match cannot leave the tail of a
    longer one visible.
    """
    variants = _decoded_fragment_variants(fragments)
    if not variants:
        return message
    index = _fragment_index(variants)
    pieces: list[str] = []
    kept = 0
    position = 0
    size = len(message)
    while position < size and kept < MAX_ERROR_TEXT_CHARS:
        length = _fragment_at(message, position, index)
        if length is None:
            pieces.append(message[position])
            kept += 1
            position += 1
            continue
        end = _fragment_region_end(message, position, position + length, index)
        pieces.append(_REDACTED)
        kept += len(_REDACTED)
        position = end
    if position < size:
        # The error cuts here anyway; the rest only keeps the cut visible.
        pieces.append(message[position:])
    return "".join(pieces)


def _decoded_fragment_variants(fragments: list[str]) -> set[str]:
    """The distinct fragments to look for, including the form ``redact``
    leaves them in."""
    variants: set[str] = set()
    for fragment in set(fragments):
        if not fragment:
            continue
        variants.add(fragment)
        variants.add(str(redact(fragment)))
    return variants


@dataclass(frozen=True)
class _FragmentIndex:
    """The fragments of one decoded message, indexed for an exact lookup at
    one position.

    ``by_length`` maps a length to the fragments of that length, and
    ``lengths_by_key`` maps a fragment's opening characters to the sorted
    lengths of the fragments that open with them. Many fragments can share an
    opening -- the body and its first line, a body whose lines start with the
    same run, an attachment's base64 padding -- and they collapse into the
    lengths that share the key. A lookup is then one key lookup plus one set
    membership test per distinct length that shares the opening and still
    fits, never one per fragment: thousands of same-prefix fragments of one
    length cost one test, and the number of tests is bounded by the distinct
    line lengths of the message, not by how many fragments share the prefix.

    ``short_lengths`` holds the lengths below the key: a fragment shorter
    than ``_FRAGMENT_KEY_CHARS`` is found from its length alone, and there
    are fewer than that many possible lengths.
    """

    by_length: dict[int, set[str]]
    lengths_by_key: dict[str, tuple[int, ...]]
    short_lengths: tuple[int, ...]


def _fragment_index(variants: set[str]) -> _FragmentIndex:
    """Index the fragments by their opening characters and by length."""
    by_length: dict[int, set[str]] = {}
    keyed: dict[str, set[int]] = {}
    for value in variants:
        length = len(value)
        by_length.setdefault(length, set()).add(value)
        if length >= _FRAGMENT_KEY_CHARS:
            keyed.setdefault(value[:_FRAGMENT_KEY_CHARS], set()).add(length)
    return _FragmentIndex(
        by_length=by_length,
        lengths_by_key={key: tuple(sorted(lengths)) for key, lengths in keyed.items()},
        short_lengths=tuple(
            sorted(length for length in by_length if length < _FRAGMENT_KEY_CHARS)
        ),
    )


def _fragment_at(
    message: str,
    position: int,
    index: _FragmentIndex,
    *,
    longer_than: int = 0,
) -> int | None:
    """The length of the longest fragment starting at ``position`` and longer
    than ``longer_than``, or None."""
    remaining = len(message) - position
    if remaining <= 0:
        return None
    if remaining >= _FRAGMENT_KEY_CHARS:
        lengths = index.lengths_by_key.get(
            message[position : position + _FRAGMENT_KEY_CHARS], ()
        )
        for slot in _candidate_slots(lengths, longer_than, remaining):
            length = lengths[slot]
            if message[position : position + length] in index.by_length[length]:
                return length
    for slot in _candidate_slots(index.short_lengths, longer_than, remaining):
        length = index.short_lengths[slot]
        if message[position : position + length] in index.by_length[length]:
            return length
    return None


def _candidate_slots(
    lengths: tuple[int, ...], longer_than: int, remaining: int
) -> range:
    """The positions in ``lengths`` (sorted) whose value is in
    ``(longer_than, remaining]``, longest first, so the region scan walks
    only lengths that can reach past its end and the main loop stops at the
    longest match."""
    return range(
        bisect.bisect_right(lengths, remaining) - 1,
        bisect.bisect_right(lengths, longer_than) - 1,
        -1,
    )


def _fragment_region_end(
    message: str,
    start: int,
    end: int,
    index: _FragmentIndex,
) -> int:
    """``end`` extended over fragments that start inside the match and reach
    past it, so one region is replaced once and no tail is left behind.

    Only the last ``MAX_ERROR_TEXT_CHARS`` characters of the match are walked.
    A fragment that starts further back is longer than that window, so it can
    only be the body or one of its own long lines, and a decoded message does
    not make one of those start inside another: the body is matched whole
    where it begins. Walking a multi-megabyte body from its start would cost
    as much as the text itself.

    The walk has a fixed budget. If a text ever keeps growing the region past
    it, the rest of the text becomes one region: that redacts more than
    needed, where stopping would leave fragment text in view.
    """
    position = max(start + 1, end - MAX_ERROR_TEXT_CHARS)
    budget = _MAX_REGION_SCAN_CHARS
    while position < end and budget:
        length = _fragment_at(message, position, index, longer_than=end - position)
        if length is not None:
            end = position + length
        position += 1
        budget -= 1
    if position < end:
        return len(message)
    return end


def _long_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if len(value) >= _MIN_ECHOED_BODY_CHARS else []
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _long_strings(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _long_strings(child)]
    return []


def _decoded_mime_fragments(body: Any) -> list[str]:
    """The decoded mail content a vendor would leak by echoing it decoded."""
    fragments: list[str] = []
    for value in _long_strings(body):
        text = _decoded_message_text(value)
        if text is not None:
            fragments.extend(_message_text_fragments(text))
    return fragments


def _decoded_message_text(value: str) -> str | None:
    """A base64 ``raw`` message decoded to text, or None for anything else."""
    if len(value) < _MIN_BASE64_MESSAGE_CHARS:
        return None
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except Exception:
        return None
    if b"\n" not in decoded or not any(marker in decoded for marker in _MIME_MARKERS):
        return None
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _message_text_fragments(text: str) -> list[str]:
    """A decoded message's body and the headers the caller wrote."""
    head, separator, body = text.partition("\r\n\r\n")
    if not separator:
        head, _, body = text.partition("\n\n")
    fragments: list[str] = []
    for line in head.splitlines():
        name, colon, value = line.partition(":")
        if colon and name.strip().lower() in _DECODED_HEADER_NAMES:
            value = value.strip()
            if len(value) >= _MIN_DECODED_CONTENT_CHARS:
                fragments.append(value)
    body = body.strip()
    if len(body) >= _MIN_DECODED_CONTENT_CHARS:
        fragments.append(body)
        fragments.extend(
            line.strip()
            for line in body.splitlines()
            if len(line.strip()) >= _MIN_DECODED_CONTENT_CHARS
        )
    return fragments


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
    """A query name or header name/value as text, refusing a control
    character in it. Query values are deliberately not checked here."""
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
            # A query value is passed through as given: Composio encodes it,
            # and a multi-line SOQL or search query is valid there.
            "value": str(value),
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
    # Composio's 200 schema only requires ``status``: ``data`` is optional and
    # nullable, and a file answer carries ``binary_data`` instead. All of them
    # go back to the caller with the provider's own status and headers; each
    # connector decides what an empty 2xx means for the API it speaks.
    answer_headers = answer.get("headers")
    binary_data = answer.get("binary_data")
    if answer_headers is not None and not isinstance(answer_headers, Mapping):
        # Silently dropping them would lose a 3xx's Location.
        raise ComposioProxyError(
            "Composio proxy returned response headers in an unexpected shape",
            status=status,
        )
    if binary_data is not None and not isinstance(binary_data, Mapping):
        # Silently treating this as an empty body would turn a malformed file
        # answer into a successful 0-byte download.
        raise ComposioProxyError(
            "Composio proxy returned a binary answer in an unexpected shape",
            status=status,
        )
    return ProxyResponse(
        status=provider_status,
        data=answer.get("data"),
        headers=answer_headers,
        binary_data=binary_data,
    )
