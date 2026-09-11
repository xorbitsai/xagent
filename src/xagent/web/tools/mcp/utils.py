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

_DIGITS_ONLY_RE = re.compile(r"[0-9]+")
# RFC 5545 UNTIL is always in "basic" form - no "-"/":" separators, unlike
# ISO8601's "extended" form that dateutil's isoparse also happily accepts
# (e.g. "2026-09-11T23:59:59+08:00"). dateutil.rrule.rrulestr's own RFC
# 5545 line parser expects exactly this basic shape and raises a raw,
# uninformative "too many values to unpack" if it isn't - see parse_rrule.
# The trailing "Z" is itself optional: RFC 5545 permits UNTIL as a
# floating "DATE WITH LOCAL TIME" (no "Z") whenever DTSTART is also
# floating local time, not just the aware "DATE WITH UTC TIME" form.
_UNTIL_RE = re.compile(r"[0-9]{8}(T[0-9]{6}Z?)?")


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


def is_bare_date(value: str) -> bool:
    """Whether a caller-supplied date/time string is a bare "date" (e.g.
    Google's all-day form, "2026-08-26") rather than a "dateTime".

    Checked structurally (exactly three hyphen-separated all-digit parts,
    after stripping outer whitespace) rather than by the absence of a
    "T"/"t" date/time separator: RFC3339 also permits a space in place of
    "T" for readability, so "2026-08-26 07:00:00" contains no "t" either
    and would otherwise be misclassified as a bare date.

    "all-digit" is checked via `_DIGITS_ONLY_RE` (ASCII 0-9 only), not
    `str.isdigit()`: the latter also accepts non-ASCII digit lookalikes
    (superscript, Thai, ...) that would misclassify a value as a bare
    date with no further local validation before it's written straight
    into a Google Calendar request body.
    """
    parts = value.strip().split("-")
    return len(parts) == 3 and all(_DIGITS_ONLY_RE.fullmatch(part) for part in parts)


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


def parse_rrule(
    rrule_text: str,
    dtstart: str | datetime,
    timezone: str | None = None,
) -> dict[str, str]:
    """Validate an RFC 5545 RRULE string and return its components (FREQ,
    INTERVAL, BYDAY, UNTIL, COUNT, ...) as a plain dict of upper-cased keys
    to upper-cased string values.

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

    ``timezone``, if given, localizes a naive ``dtstart``. For a genuine
    all-day event - ``dtstart`` passed as a bare-date string with no "T",
    e.g. Google's own all-day "date" field - this only happens when
    needed to compare against an aware UNTIL; a naive/bare-date UNTIL is
    correctly left floating to match the still-naive anchor instead,
    since RFC 5545's DATE value type legitimately pairs a floating DTSTART
    with a floating UNTIL. For anything else (a full dateTime string,
    aware or not, or an already-`datetime` object), the
    anchor is always localized when naive - RFC 5545's DATE-TIME value
    type requires UNTIL to be aware too, so if it isn't, that's an actual
    mismatch dateutil's own validation below correctly rejects, the same
    way it already does for a dtstart string that carries its own offset.
    Either way, the UNTIL-not-before-dtstart check
    below still runs - it compares whenever anchor and UNTIL share the
    same awareness, aware-vs-aware or naive-vs-naive alike.

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
        key = key.strip().upper()
        if key in parts:
            # A duplicate key (e.g. "FREQ=DAILY;FREQ=WEEKLY") would
            # otherwise just silently overwrite the first occurrence in
            # `parts` for local validation, while the raw text - still
            # containing BOTH occurrences - reaches Google's API close to
            # verbatim, where its behavior is unspecified rather than
            # matching whatever this function validated.
            raise ValueError(
                f"invalid recurrence rule: {key} is specified more than once"
            )
        # Uppercased for the same reason ensure_rrule_prefix uppercases the
        # whole rule text: RFC 5545's RRULE grammar has no case-sensitive
        # free-text values, and this dict is what a future Outlook
        # translator (parse_rrule's only other planned caller) would key
        # lookups against - a caller who wrote "freq=daily" shouldn't get
        # a dict with "daily" where every other caller gets "DAILY".
        parts[key] = value.strip().upper()
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
    for key in ("INTERVAL", "COUNT"):
        if key not in parts:
            continue
        if not _DIGITS_ONLY_RE.fullmatch(parts[key]):
            raise ValueError(
                f"invalid recurrence rule: {key} must be an integer, got {parts[key]!r}"
            )
        if int(parts[key]) < 1:
            raise ValueError(
                f"invalid recurrence rule: {key} must be a positive integer, "
                f"got {int(parts[key])}"
            )

    # Parsed before the anchor is localized below, since whether to
    # localize at all now depends on UNTIL's own value type.
    until_dt = None
    if "UNTIL" in parts:
        # Checked against RFC 5545's actual (basic-form) UNTIL grammar
        # before handing it to dateutil at all: isoparse below is lenient
        # enough to accept ISO8601's extended form too (dashes, colons,
        # e.g. "2026-09-11T23:59:59+08:00"), which RFC 5545 never allows
        # for UNTIL - and dateutil.rrule.rrulestr's own line parser chokes
        # on exactly that shape with a raw, uninformative "too many values
        # to unpack" once this rule reaches the validation pass further
        # down, instead of the clear error this function exists to give.
        if not _UNTIL_RE.fullmatch(parts["UNTIL"]):
            raise ValueError(
                f"invalid UNTIL value in recurrence rule: {parts['UNTIL']!r} "
                "- must be RFC 5545's basic form with no '-'/':' separators, "
                "e.g. '20260911T235959Z' or '20260911'"
            )
        try:
            until_dt = _date_parser.isoparse(parts["UNTIL"])
        except ValueError as exc:
            raise ValueError(
                f"invalid UNTIL value in recurrence rule: {parts['UNTIL']!r}"
            ) from exc

    # A bare-date dtstart string (e.g. Google's own all-day "date" field,
    # "2026-08-26") is the same signal calendar.py itself already uses to
    # detect an all-day event - RFC 5545's DATE value type, which
    # legitimately pairs with an equally bare-date (floating) UNTIL.
    # Anything else - a full dateTime string (with or without its own
    # offset) or an already-`datetime` object (as Outlook's caller always
    # passes, already localized aware) - represents DATE-TIME, which RFC
    # 5545 requires UNTIL to match: aware, not floating.
    dtstart_is_bare_date = isinstance(dtstart, str) and is_bare_date(dtstart)

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
        if dtstart_is_bare_date:
            # All-day: only localize if needed to compare against an
            # aware UNTIL - localizing unconditionally would instead
            # create a mismatch in the opposite direction for a
            # legitimately floating, bare-date UNTIL.
            if until_dt is not None and until_dt.tzinfo is not None:
                anchor = anchor.replace(tzinfo=resolve_zoneinfo(timezone))
        else:
            # Timed: always localize. If UNTIL is naive/bare-date here,
            # that's an actual RFC 5545 value-type mismatch (a DATE-TIME
            # DTSTART requires an aware UNTIL) - localizing the anchor
            # anyway makes dateutil's own validation below correctly
            # reject it, the same way it always has for an
            # already-aware-string dtstart.
            anchor = anchor.replace(tzinfo=resolve_zoneinfo(timezone))
    try:
        _rrulestr(f"RRULE:{body}", dtstart=anchor)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid recurrence rule {rrule_text!r}: {exc}") from exc

    # dateutil's rrulestr above does NOT catch this: an UNTIL before dtstart
    # parses fine and just silently yields zero occurrences - a "recurring"
    # event that never actually recurs, reported as success. Comparable
    # whenever anchor and until_dt share the same awareness (both aware, or
    # both naive/floating - a plain datetime comparison works fine either
    # way); an aware-vs-naive mismatch would raise TypeError rather than
    # answer the question, so THAT combination is skipped rather than
    # guessed at (dateutil's rrulestr call above already rejects it before
    # this point is ever reached, in fact).
    if until_dt is not None and (until_dt.tzinfo is None) == (anchor.tzinfo is None):
        if until_dt < anchor:
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
