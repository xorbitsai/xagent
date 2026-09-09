import json
import os
import re
import urllib.request
from datetime import datetime
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as _date_parser
from dateutil.rrule import rrulestr as _rrulestr

from ....config import get_tool_max_output_length


def require_clean_identifier(value: str, field_name: str) -> str:
    """Reject an empty or whitespace-padded id rather than silently fixing it.

    An id copy-pasted or concatenated by a caller with accidental whitespace
    is more likely a bug worth surfacing than a value to repair - repairing
    it would mask the bug and could send a query for a different object.
    Use this for ids that go into a JSON request body; for ids interpolated
    into a URL path, use url_path_id instead - encoding (not just rejecting
    whitespace) is what actually closes path/query injection.
    """
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(
            f"{field_name} must be a non-empty id with no surrounding whitespace"
        )
    return value


def url_path_id(value: str, field_name: str) -> str:
    """Validate then percent-encode an id for safe interpolation into a URL
    path segment.

    Percent-encoding - not a blocklist of "/", "?", "#" - is what actually
    prevents a value like "x?limit=1&foo=/reports/metrics" from escaping
    its intended path segment: any character that could do that gets
    encoded regardless of which one it is, rather than relying on an
    enumeration that could miss one. "." and ".." are the one exception
    that survives encoding unchanged (they're always-unreserved characters
    per RFC 3986, so quote() never touches them), and requests/urllib3
    normalize dot-segments out of the final URL before sending it --
    verified directly: requests.Request("GET",
    ".../sobjects/Account/..").prepare().url collapses to ".../sobjects/",
    a completely different (still valid) endpoint. Rejected explicitly
    since encoding can't close this one off.
    """
    require_clean_identifier(value, field_name)
    if value in (".", ".."):
        raise ValueError(f"{field_name} must not be '.' or '..'")
    return quote(value, safe="")


def success_with_capped_dict(field_name: str, data: Any) -> str:
    """Build a ``{"status": "success", ...}`` payload, trimming a dict
    until it fits the platform's output limit.

    A record/report/metrics response can be a dict keyed by date,
    dimension, or id depending on the endpoint, where most of the
    payload's size typically lives in one or two large nested list/dict
    values while the rest are small scalars (an "offset" or "total" field
    alongside a big "breakdowns" list, or a handful of small standard
    fields alongside one huge Long Text Area value). Dropping whole
    top-level keys to shrink such a dict can discard the entire useful
    payload on the very first step while leaving small, mostly empty
    scalar fields behind -- and there's no cursor to retry with, so that
    data is gone for this call. Phase 1 instead repeatedly finds the
    largest list/dict-valued key and halves *its* contents (recursing one
    level, not further), so small scalar keys survive untouched as long as
    there is a bigger key left to shrink first. Phase 2 is a fallback for
    the residual case -- a dict with no list/dict-valued keys at all (e.g.
    a handful of scalar keys with huge string values) -- and drops whole
    keys, exactly as phase 1 replaces; it's guaranteed to terminate at {}.
    """
    max_output_length = get_tool_max_output_length()
    response = json.dumps(
        {"status": "success", field_name: data, "truncated": False},
        ensure_ascii=False,
    )
    if not isinstance(data, dict) or len(response) <= max_output_length:
        return response

    def _build(payload: dict[str, Any], truncated: bool) -> str:
        return json.dumps(
            {"status": "success", field_name: payload, "truncated": truncated},
            ensure_ascii=False,
        )

    working = dict(data)
    truncated = False
    while len(response) > max_output_length:
        collection_keys = [
            key
            for key, value in working.items()
            if isinstance(value, (list, dict)) and len(value) > 0
        ]
        if not collection_keys:
            break
        target_key = max(
            collection_keys,
            key=lambda key: len(json.dumps(working[key], ensure_ascii=False)),
        )
        target_value = working[target_key]
        if isinstance(target_value, list):
            working[target_key] = target_value[: len(target_value) // 2]
        else:
            sub_keys = list(target_value.keys())
            working[target_key] = {
                sub_key: target_value[sub_key]
                for sub_key in sub_keys[: len(sub_keys) // 2]
            }
        truncated = True
        response = _build(working, truncated)

    keys = list(working.keys())
    while len(response) > max_output_length and keys:
        keys = keys[: len(keys) // 2]
        working = {key: working[key] for key in keys}
        truncated = True
        response = _build(working, truncated)
    return response


def resolve_zoneinfo(timezone: str) -> ZoneInfo:
    """Resolve an IANA timezone name to a stdlib ZoneInfo, raising a clean
    ValueError (naming the bad value) instead of letting ZoneInfoNotFoundError
    propagate unworded."""
    try:
        return ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown timezone: {timezone!r}") from exc


def _strip_rrule_prefix(rrule_text: str) -> str:
    """Return `rrule_text` with any leading "RRULE:" (case-insensitive)
    removed and outer whitespace trimmed - the one place this stripping
    happens, so `ensure_rrule_prefix` and `parse_rrule` can't drift apart
    on what counts as "the prefix"."""
    body = rrule_text.strip()
    if body.upper().startswith("RRULE:"):
        body = body[len("RRULE:") :]
    return body.strip()


def ensure_rrule_prefix(rrule_text: str) -> str:
    """Return `rrule_text` with a leading "RRULE:" (adding one if it
    doesn't already have one), with its body canonicalized to uppercase.

    RFC 5545's RRULE grammar has no case-sensitive free-text values - every
    token (FREQ's value, BYDAY's day codes, the "T"/"Z" markers in UNTIL,
    ...) is a fixed-case keyword - so uppercasing the whole body is always
    safe, and it matters here specifically: `parse_rrule`'s validation
    tolerates lowercase (dateutil is lenient), but a lowercase rule would
    otherwise reach Google's API as literal text at whatever case the
    caller happened to use.
    """
    return f"RRULE:{_strip_rrule_prefix(rrule_text).upper()}"


_DIGITS_ONLY_RE = re.compile(r"[0-9]+")


def parse_rrule(
    rrule_text: str,
    dtstart: str | datetime,
    timezone: str | None = None,
) -> dict[str, str]:
    """Validate an RFC 5545 RRULE string and return its components (FREQ,
    INTERVAL, BYDAY, UNTIL, COUNT, ...) as a plain dict of upper-cased keys
    to raw string values.

    ``dtstart`` anchors a validation pass through ``dateutil.rrule.rrulestr``
    so a rule that's syntactically plausible but semantically broken (e.g.
    a nonsense FREQ) is rejected here rather than being sent to
    Google/Outlook and either erroring opaquely or - worse - only ever
    landing as inert description text, which is exactly the failure mode
    reported against this connector before recurrence support existed.
    ``dtstart`` also anchors a separate, explicit check this function adds
    on top of dateutil's: an UNTIL before dtstart parses fine under
    rrulestr (it just silently produces zero occurrences), so that case is
    checked here directly rather than trusted to the library. Pass an
    RFC3339/ISO8601 string (matching what these calendar tools already
    require for start_time/start_datetime), or a `datetime` directly when
    the caller already has one on hand (e.g. after localizing a naive
    Outlook start time) - skipping the format-then-reparse round trip that
    passing `.isoformat()` back in would otherwise cost.

    ``timezone``, if given, localizes a naive ``dtstart`` (e.g. an all-day
    Google event's bare "date", or an Outlook naive dateTime with no
    embedded offset) before validation. This matters beyond cosmetics:
    RFC 5545 requires DTSTART and UNTIL to either both be timezone-aware or
    both be "floating" - a naive DTSTART paired with the (very common)
    "Z"-suffixed/aware UNTIL is otherwise rejected by dateutil with a
    confusingly-worded error, and the UNTIL-not-before-dtstart check above
    can't run at all without an aware anchor to compare against.

    The returned dict (rather than the parsed rrule object) is what
    callers actually build a provider payload from: Google takes the RRULE
    text close to verbatim, while Outlook needs these components
    translated into Graph's own pattern/range structure - neither needs
    dateutil's internal representation, just confirmation that the text is
    valid RFC 5545.
    """
    if "\n" in rrule_text or "\r" in rrule_text:
        raise ValueError(
            "recurrence rule must not contain embedded newlines (this could "
            "smuggle an extra RRULE/EXDATE/RDATE line into the calendar API "
            "request)"
        )
    body = _strip_rrule_prefix(rrule_text)
    if not body:
        raise ValueError("recurrence rule must not be empty")

    parts: dict[str, str] = {}
    for chunk in body.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"invalid recurrence rule component: {chunk!r}")
        key, _, value = chunk.partition("=")
        parts[key.strip().upper()] = value.strip()
    if "FREQ" not in parts:
        raise ValueError(
            "recurrence rule must include FREQ, e.g. "
            "'FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z'"
        )
    if "UNTIL" in parts and "COUNT" in parts:
        raise ValueError(
            "recurrence rule must not specify both UNTIL and COUNT - RFC "
            "5545 treats these as mutually exclusive ways to end a series"
        )
    # RFC 5545 defines INTERVAL/COUNT as `1*DIGIT` - plain unsigned digits,
    # nothing else. Python's int() is more permissive (a leading "+"/"-",
    # PEP 515 "_" digit separators), and `parts` holds the ORIGINAL string,
    # which reaches Google's API close to verbatim (only re-cased, not
    # reconstructed from the parsed int) - so validating only the parsed
    # value and not the string's own shape would let something like
    # "INTERVAL=1_0" or "COUNT=+5" slip through to the live API as invalid
    # RRULE text.
    if "INTERVAL" in parts and not _DIGITS_ONLY_RE.fullmatch(parts["INTERVAL"]):
        raise ValueError(
            f"invalid recurrence rule: INTERVAL must be an integer, "
            f"got {parts['INTERVAL']!r}"
        )
    if "COUNT" in parts and not _DIGITS_ONLY_RE.fullmatch(parts["COUNT"]):
        raise ValueError(
            f"invalid recurrence rule: COUNT must be an integer, got {parts['COUNT']!r}"
        )
    if "INTERVAL" in parts and int(parts["INTERVAL"]) < 1:
        raise ValueError(
            "invalid recurrence rule: INTERVAL must be a positive integer, "
            f"got {int(parts['INTERVAL'])}"
        )
    if "COUNT" in parts and int(parts["COUNT"]) < 1:
        raise ValueError(
            "invalid recurrence rule: COUNT must be a positive integer, "
            f"got {int(parts['COUNT'])}"
        )

    if isinstance(dtstart, datetime):
        anchor = dtstart
    else:
        try:
            anchor = _date_parser.isoparse(dtstart)
        except ValueError as exc:
            raise ValueError(
                f"invalid start time for recurrence rule: {dtstart}"
            ) from exc
    if anchor.tzinfo is None and timezone is not None:
        anchor = anchor.replace(tzinfo=resolve_zoneinfo(timezone))
    try:
        _rrulestr(f"RRULE:{body}", dtstart=anchor)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid recurrence rule {rrule_text!r}: {exc}") from exc

    # dateutil's rrulestr above does NOT catch this: an UNTIL before dtstart
    # parses fine and just silently yields zero occurrences - a "recurring"
    # event that never actually recurs, reported as success. Only checked
    # when both sides are timezone-aware (guaranteed for every caller that
    # passes `timezone` above for a naive dtstart, or that already had an
    # aware one); an aware-vs-naive comparison would raise TypeError rather
    # than answer the question, so it's skipped rather than guessed at.
    if "UNTIL" in parts:
        try:
            until_dt = _date_parser.isoparse(parts["UNTIL"])
        except ValueError as exc:
            raise ValueError(
                f"invalid UNTIL value in recurrence rule: {parts['UNTIL']!r}"
            ) from exc
        if (
            anchor.tzinfo is not None
            and until_dt.tzinfo is not None
            and until_dt < anchor
        ):
            raise ValueError(
                f"recurrence UNTIL ({parts['UNTIL']}) is before the start "
                "time; this recurrence would never actually happen"
            )

    return parts


def clamp_limit(limit: int, *, max_limit: int) -> int:
    """Clamp a caller-supplied pagination page size to ``[1, max_limit]``.

    An LLM caller can pass 0, a negative number, or an absurdly large value
    for a tool's ``limit`` parameter. Silently clamping (rather than
    raising) keeps a malformed value from producing a permanently-stuck,
    zero-progress page -- 0 or a negative limit always slices to an empty
    page regardless of offset, so a caller mechanically following a
    pagination contract's own has_more/next_offset would retry forever
    with no error to signal why.
    """
    return max(1, min(int(limit), max_limit))


def clamp_offset(offset: int) -> int:
    """Clamp a caller-supplied pagination offset to ``>= 0``.

    Python slicing treats a negative start index as "count from the end",
    so an unclamped negative offset would silently return items from the
    tail of the list instead of erroring or being treated as the first
    page.
    """
    return max(0, int(offset))


def resolve_id_from_url(value: str, pattern: re.Pattern[str], field_name: str) -> str:
    """Return the id captured by ``pattern`` when ``value`` is a matching URL,
    otherwise the stripped value itself.

    Guards against a non-string value the same way require_clean_identifier
    does: ``pattern.search()``/``.strip()`` would otherwise raise a raw
    TypeError/AttributeError instead of a clean, actionable ValueError.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    match = pattern.search(value)
    if match:
        return match.group(1)
    return value.strip()


def setup_proxy_env() -> None:
    """Setup proxy environment variables from system proxies if missing."""
    # Filter out empty proxy vars to prevent httplib2 hangs
    for var in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]:
        if var in os.environ and not os.environ[var]:
            del os.environ[var]

    system_proxies = urllib.request.getproxies()
    if (
        "https" in system_proxies
        and "HTTPS_PROXY" not in os.environ
        and "https_proxy" not in os.environ
    ):
        os.environ["HTTPS_PROXY"] = system_proxies["https"]
    if (
        "http" in system_proxies
        and "HTTP_PROXY" not in os.environ
        and "http_proxy" not in os.environ
    ):
        os.environ["HTTP_PROXY"] = system_proxies["http"]

    # If ALL_PROXY is set, ensure HTTPS_PROXY is also set
    if "ALL_PROXY" in os.environ and "HTTPS_PROXY" not in os.environ:
        os.environ["HTTPS_PROXY"] = os.environ["ALL_PROXY"]
