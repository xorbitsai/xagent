import json
import os
import re
import urllib.request
from datetime import date, datetime, time, timedelta
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ....config import get_tool_max_output_length


class InsufficientScopeError(RuntimeError):
    """Raised by a connector's ``_find_conflicts`` on a whole-batch
    missing-scope error (see that function's own docstring for why this
    is raised at all rather than degrading to unchecked).

    Carries whatever ``conflicts``/``unchecked_attendees`` had already
    been confirmed before the error - a caller accumulating results
    across multiple ``_find_conflicts`` calls for one create/update (e.g.
    one call for newly-added attendees, another per delta segment for
    retained attendees) needs this to still report an already-confirmed
    real conflict (found before the error, in this call or an earlier
    one) instead of silently discarding it just because a *later*,
    unrelated check also hit the same scope problem. Reporting a known
    conflict is always safe regardless of what else couldn't be checked;
    only the "nothing confirmed yet, can't tell if this is safe" case
    should still reject the write outright.
    """

    def __init__(
        self,
        message: str,
        conflicts: list[dict[str, Any]],
        unchecked_attendees: list[str],
    ) -> None:
        super().__init__(message)
        self.conflicts = conflicts
        self.unchecked_attendees = unchecked_attendees


def merge_scope_error(
    exc: InsufficientScopeError,
    conflicts: list[dict[str, Any]],
    unchecked_attendees: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fold an `InsufficientScopeError` caught mid-accumulation into the
    running `conflicts`/`unchecked_attendees` a connector's create/update
    tool is building up (potentially across more than one `_find_conflicts`
    call) - merging the error's own already-confirmed findings in first, so
    a real conflict found before the error is never lost regardless of
    which of possibly several calls actually raised.

    Re-raises `exc` itself (preserving its original traceback/cause) when
    the merged `conflicts` is still empty: nothing was confirmed yet, so
    there is no already-known problem safe to report instead of rejecting
    the write outright.
    """
    conflicts = [*conflicts, *exc.conflicts]
    unchecked_attendees = [*unchecked_attendees, *exc.unchecked_attendees]
    if not conflicts:
        raise exc
    return conflicts, unchecked_attendees


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


def normalize_addresses(addresses: list[str] | str) -> list[str]:
    """Split/strip a comma-separated string or list of email addresses into a
    clean list, dropping anything blank and case-insensitive duplicates
    (keeping the first casing seen - email addresses are case-insensitive,
    so a caller passing the same person twice with different casing, e.g.
    from a human-written invite list, must not become two separate
    attendee entries downstream)."""
    if isinstance(addresses, str):
        raw = [address.strip() for address in addresses.split(",") if address.strip()]
    else:
        raw = [address.strip() for address in addresses if address and address.strip()]
    seen: set[str] = set()
    deduped = []
    for address in raw:
        key = address.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(address)
    return deduped


def conflict_response(
    conflicts: list[dict[str, Any]],
    unchecked_attendees: list[str],
    start: str,
    end: str,
) -> str:
    """Build the status="conflict" envelope a calendar-writing MCP tool
    returns instead of creating/updating an event, so the calling agent
    reports the conflict to the user rather than silently double-booking.

    `unchecked_attendees` is usually the self-explanatory kind (an
    attendee absent from the provider's response, or its own per-attendee
    error) - a missing-OAuth-scope 403 covering the whole call is instead
    raised as `InsufficientScopeError` before ever reaching here, since
    writing an event whose availability could never actually be checked
    would defeat the point of this feature. The one exception: a caller
    that catches `InsufficientScopeError` because `conflicts` was already
    non-empty (a real conflict was confirmed before the scope error hit)
    may pass that error's own `unchecked_attendees` through here too - at
    that point the write is already correctly blocked by the known
    conflict, so being honest about who else couldn't be checked is safe.

    `conflicts` and `unchecked_attendees` are both uncapped input (an
    organizer with a busy shared calendar or a wide window can turn up
    far more overlapping events than a caller needs to see; a missing-
    scope error partway through a large invite list can likewise mark
    an entire remaining batch unchecked) - unlike every other response
    path here, which routes through `success_with_capped_dict`, this
    shape has multiple top-level fields under a non-"success" status
    that function doesn't support, so the payload is capped locally
    instead: whichever of `conflicts`/`unchecked_attendees` is currently
    larger (by its own serialized size) is halved each step, so a
    single real conflict isn't fully dropped just to make room for a
    still-oversized `unchecked_attendees` list (unconditionally halving
    `conflicts` first would zero out a 1-item list in a single step,
    regardless of whether that was actually necessary). If the fixed
    envelope itself is too large, the response falls back to a compact,
    valid JSON object instead of relying on the framework to truncate the
    serialized JSON at an arbitrary character boundary.
    """
    payload: dict[str, Any] = {
        "status": "conflict",
        "message": f"{len(conflicts)} existing event(s) overlap {start} - {end}",
        "conflicts": conflicts,
        "unchecked_attendees": unchecked_attendees,
        "hint": (
            "Do not retry with the same time slot. Report these conflicts to "
            "the user and ask them to pick a different time or confirm they "
            "want to proceed anyway. Only call this tool again with "
            "ignore_conflicts=true after the user has explicitly confirmed "
            "they still want this slot."
        ),
        # Always present (not just when truncation actually happens),
        # matching every other capped response path in this package
        # (see success_with_capped_dict) - a caller that learned
        # "truncated is always in the payload" from those shouldn't have
        # to special-case this one.
        "truncated": False,
    }
    response = json.dumps(payload, ensure_ascii=False)
    max_output_length = get_tool_max_output_length()
    if len(response) <= max_output_length:
        return response

    remaining_conflicts = conflicts
    remaining_unchecked = unchecked_attendees
    while (remaining_conflicts or remaining_unchecked) and len(
        response
    ) > max_output_length:
        conflicts_size = len(json.dumps(remaining_conflicts, ensure_ascii=False))
        unchecked_size = len(json.dumps(remaining_unchecked, ensure_ascii=False))
        if remaining_conflicts and conflicts_size >= unchecked_size:
            remaining_conflicts = remaining_conflicts[: len(remaining_conflicts) // 2]
        else:
            remaining_unchecked = remaining_unchecked[: len(remaining_unchecked) // 2]
        payload["conflicts"] = remaining_conflicts
        payload["unchecked_attendees"] = remaining_unchecked
        payload["truncated"] = True
        response = json.dumps(payload, ensure_ascii=False)

    if len(response) > max_output_length:
        compact_payloads: tuple[dict[str, Any], ...] = (
            {
                "status": "conflict",
                "message": "Scheduling conflict detected; details truncated.",
                "conflicts": [],
                "unchecked_attendees": [],
                "truncated": True,
            },
            {"status": "conflict", "truncated": True},
        )
        for compact_payload in compact_payloads:
            compact_response = json.dumps(compact_payload, ensure_ascii=False)
            if len(compact_response) <= max_output_length:
                return compact_response
        # A configured limit smaller than this JSON object cannot preserve
        # both valid JSON and the response's required status. Keep the
        # smallest contract-preserving response rather than returning `{}`.
        return json.dumps({"status": "conflict"}, ensure_ascii=False)
    return response


def attendees_were_given(attendees: list[str] | str | None) -> bool:
    """Whether `attendees` was actually provided by the caller for an
    update, treating an empty string the same as not-provided at all -
    matching every other optional field's truthy convention here (and the
    create path's own check) - rather than as "clear every attendee".
    An explicit empty list is still considered supplied, even though the
    current additive attendee API treats it as a no-op rather than removing
    existing attendees."""
    return attendees is not None and attendees != ""


def attendees_to_add(
    attendees: list[str] | str | None, existing_attendee_emails: set[str]
) -> list[str]:
    """The newly-added addresses from an update's `attendees` argument,
    normalized and deduped, that aren't already in
    `existing_attendee_emails` - what actually needs writing when
    `attendees` only ever adds attendees and never removes any. Returns
    [] for anything `attendees_were_given` treats as not-provided (None,
    ""), matching its own convention, as well as for a caller-supplied
    list/string that turns out to name only people already on the event.
    """
    if not attendees_were_given(attendees):
        return []
    assert attendees is not None  # narrows for mypy; attendees_were_given implies this
    existing = {email.strip().lower() for email in existing_attendee_emails}
    return [
        address
        for address in normalize_addresses(attendees)
        if address.lower() not in existing
    ]


_OVERLONG_FRACTIONAL_SECONDS = re.compile(r"(\.\d{6})\d+")


def datetime_key_for_comparison(value: str | None) -> datetime | str | None:
    """Return a value suitable for equality-comparing two datetime strings
    that may be written in different but equivalent formats.

    A caller re-submitting the same instant it just read back from an API
    (or a value it typed by hand) can differ in formatting alone - "Z" vs
    "+00:00", a different (but equal-instant) zone offset, missing vs
    present fractional seconds - while meaning the same moment. Comparing
    such strings directly treats a same-instant resubmission as a real
    change, which for a scheduling-conflict check means re-querying (and,
    worse, misjudging self-conflicts against) a window that never actually
    moved.

    Parses to a `datetime` and returns it: two aware datetimes compare
    equal when they denote the same instant regardless of differing zone
    offsets, which a string/isoformat() comparison would not catch. Two
    naive datetimes (Outlook's dateTime values, which carry no offset of
    their own) compare directly, which is correct since both sides here
    always come from the same source. An aware value never equals a naive
    one, which is fine - these are simply never the same source's values
    (never worse than the plain-string comparison this replaces).

    Returns the original value unchanged if it doesn't parse (None stays
    None, and a genuinely malformed value still compares by raw string,
    same as before this normalization existed - never worse, never a
    crash).

    Outlook commonly reports 7-digit (100-nanosecond) fractional seconds
    (e.g. ".0000000"), one more digit than a `datetime` microsecond can
    hold. `fromisoformat`'s tolerance for that is a CPython-version detail
    the caller shouldn't need to know about, so any fractional-seconds run
    longer than 6 digits is truncated to 6 before parsing, rather than
    relying on the current interpreter to accept (and correctly truncate)
    the extra digits itself.
    """
    if value is None:
        return None
    normalized = value
    if normalized[-1:] in ("Z", "z"):
        # Only the trailing UTC marker, not a blanket .replace("Z", ...) -
        # a naive str.replace would also touch a "Z"/"z" anywhere else in
        # the string, which happens to never occur in a valid ISO
        # datetime today but is needless coupling to that happening to
        # stay true.
        normalized = normalized[:-1] + "+00:00"
    normalized = _OVERLONG_FRACTIONAL_SECONDS.sub(r"\1", normalized)
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return value


def reject_reversed_window(start_value: str, end_value: str) -> None:
    """Raise ValueError when `end_value` is not after `start_value` - an
    event window's basic ordering sanity, checked before forwarding it to
    the provider API as-is (both on create, and on update's effective
    window regardless of ignore_conflicts - this isn't a conflict-check
    decision a caller can opt out of).

    Deliberately permissive when either side doesn't parse to a real,
    comparable instant (stays silent rather than rejecting) - matching
    this module's other datetime comparisons, which treat "can't tell"
    as "don't block", not "assume invalid". That includes the case where
    both sides parse but one is offset-aware and the other naive (e.g. a
    ``Z``-suffixed value alongside a naive one): Python raises
    ``TypeError`` comparing those with ``<=`` even though both are real
    ``datetime`` instances, so the ``isinstance`` check alone isn't
    enough to guarantee a safe comparison - see ``window_delta_segments``,
    which guards the identical hazard the same way.
    """
    start_key = datetime_key_for_comparison(start_value)
    end_key = datetime_key_for_comparison(end_value)
    if not (isinstance(start_key, datetime) and isinstance(end_key, datetime)):
        return
    try:
        if end_key <= start_key:
            raise ValueError(
                f"end ({end_value!r}) must be after start ({start_value!r})."
            )
    except TypeError:
        return


def require_offset_datetime(value: str, field_name: str) -> None:
    """Raise ValueError when `value` parses to a real instant but doesn't
    carry a UTC offset or "Z" suffix.

    A caller-supplied boundary that's naive isn't just non-compliant with
    the documented RFC3339 contract: this module's own internal instants
    (e.g. an existing event's boundary via `_event_boundary`) are always
    offset-bearing, so a naive value compared against one raises
    `TypeError` deep inside a downstream comparison (see
    `window_delta_segments`'s aware-vs-naive guard) - which conservatively
    falls back to using the naive value as-is, so it eventually reaches
    the provider API's timeMin/timeMax and fails there with an opaque
    error instead of this clear, actionable one.

    Silent (no-op) when `value` doesn't parse to a real instant at all -
    that's a different failure a caller will already hit downstream with
    its own clear error, not this function's job to preempt.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return
    if parsed.tzinfo is None:
        raise ValueError(
            f"{field_name} ({value!r}) must include a UTC offset or 'Z' "
            "suffix (RFC3339), e.g. '2026-08-27T10:00:00+00:00' or "
            "'2026-08-27T10:00:00Z'."
        )


def resolve_zoneinfo(name: str) -> ZoneInfo:
    """Resolve a Google Calendar timeZone value (an IANA Time Zone
    Database name, per Google's own EventDateTime docs) to a real
    ``ZoneInfo``, for use anywhere a working zone is required (not just a
    best-effort comparison) - e.g. attaching a real UTC offset to a naive
    datetime string.

    Raises ``ValueError`` rather than silently defaulting to UTC when the
    name can't be resolved: a wrong silent guess here is exactly the class
    of bug (a query or write running in the wrong real-world window) this
    helper exists to prevent.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Timezone {name!r} isn't a recognized IANA zone name."
        ) from exc


def offset_datetime_string(value: str, tz_name: str) -> str:
    """Combine a naive datetime string (no embedded UTC offset - Outlook's
    dateTimeTimeZone.dateTime is always this shape, paired with a separate
    timeZone field) with its zone name into an offset-bearing ISO 8601
    string.

    Needed anywhere a naive value has to be sent to an API that (unlike
    Outlook's own structured ``{"dateTime", "timeZone"}`` request bodies)
    takes a single datetime string and infers UTC when it carries no
    offset of its own. Graph's own docs for List calendarView's
    startDateTime/endDateTime query parameters say exactly this: they're
    "interpreted using the timezone offset specified in the value" and
    "aren't impacted by the value of the Prefer ... header" - a naive
    value there is silently read as UTC regardless of what timezone the
    caller actually meant.

    Raises ``ValueError`` (via ``resolve_zoneinfo``) rather than falling
    back to UTC when ``tz_name`` can't be resolved - and also raises if
    ``value`` turns out to already carry its own offset/``Z``, rather
    than silently relabeling those same clock digits with `tz_name`'s
    offset instead of converting them (``.replace(tzinfo=...)`` changes
    what zone a datetime is interpreted in without changing the instant
    it names, e.g. turning "10:00 UTC" into "10:00 <tz_name>" - a real
    shift, not a no-op, whenever the two offsets differ). No public tool
    parameter is documented as requiring a naive value, so a caller
    passing one with an offset already attached is a real, reachable
    input here, not just a theoretical one.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        raise ValueError(
            f"{value!r} already carries a UTC offset; pass a naive "
            "datetime string (no trailing 'Z' or +HH:MM) together with "
            "its timezone name instead of embedding an offset in both."
        )
    return parsed.replace(tzinfo=resolve_zoneinfo(tz_name)).isoformat()


def calendar_day_bounds(
    date_value: str, tz_name: str, *, days: int = 1
) -> tuple[str, str]:
    """Return (start, end) offset-bearing ISO instants spanning ``days``
    full calendar day(s) starting at ``date_value``'s date, in ``tz_name``.

    ``date_value`` may be a bare "YYYY-MM-DD" or a full datetime string
    (only its date component is used). Used to widen an all-day event's
    (or an all-day toggle's) boundary into a real queryable window: an
    all-day event occupies the *calendar's own* day, not a UTC day, so a
    hardcoded "T00:00:00Z" is only correct for a UTC calendar.
    """
    if days <= 0:
        raise ValueError("days must be a positive integer")
    zone = resolve_zoneinfo(tz_name)
    day: date = datetime.fromisoformat(date_value).date()
    start = datetime.combine(day, time.min, tzinfo=zone)
    # Advance the local calendar date before attaching the timezone again.
    # Adding a timedelta to an aware datetime preserves the original
    # offset across DST transitions, which can produce the wrong local
    # midnight for the end boundary.
    end_day = day + timedelta(days=days)
    end = datetime.combine(end_day, time.min, tzinfo=zone)
    return start.isoformat(), end.isoformat()


def window_delta_segments(
    existing_start: datetime | str | None,
    existing_end: datetime | str | None,
    new_start: datetime | str | None,
    new_end: datetime | str | None,
) -> list[tuple[datetime, datetime]]:
    """The portion(s) of the half-open [new_start, new_end) window that
    fall OUTSIDE [existing_start, existing_end) - the only territory an
    attendee already on the event needs a fresh free/busy check against.

    An attendee already on the event will always show this very event's
    own busy block for any instant inside the OLD window - querying that
    overlap can't distinguish "busy because of this event" from a real
    conflict, the exact self-conflict bug this function exists to avoid.
    But a window can move in a way that's neither identical/subset (no
    new territory at all) nor fully disjoint (the whole new window is new
    territory) - a partial nudge like 10:00-10:30 -> 10:15-10:45 makes
    10:30-10:45 new territory while 10:15-10:30 is still old ground, and
    checking the retained attendee only over the confirmed-safe old
    portion (or not at all) would silently miss a genuine conflict
    sitting in that new segment. This computes exactly that new territory
    so the caller can check retained attendees against precisely it,
    while newly-added attendees (who have no footprint on this event at
    all) still get the FULL new window checked regardless of any of this
    - they need every instant in it verified, delta or not.

    ISO strings are normalized internally before comparison, so callers do
    not have to coordinate a separate parsing step.

    Returns:
    - ``[]`` when the new window is confirmed to add no territory beyond
      the old one (identical or a subset) - nothing new for a retained
      attendee to be checked against.
    - Up to two disjoint segments when the new window extends past the
      old one's start, end, or both (e.g. extending a meeting on both
      sides in one call).
    - The whole ``[new_start, new_end)`` as a single segment when the two
      windows are confirmed to not overlap at all (matching "moved
      somewhere completely disjoint -> check the whole thing"), OR when
      any value isn't a real, mutually-comparable instant (parsing
      failure, or one side aware and the other naive) - "can't confirm
      the delta is smaller than the whole window" must never silently
      shrink what gets checked, so treat it as needing the full window.
    - ``[]`` in the same can't-tell scenario if there's no valid new
      window at all to fall back to (``new_start``/``new_end`` themselves
      aren't real instants) - there is nothing meaningful to check.
    """
    existing_start = (
        datetime_key_for_comparison(existing_start)
        if isinstance(existing_start, str)
        else existing_start
    )
    existing_end = (
        datetime_key_for_comparison(existing_end)
        if isinstance(existing_end, str)
        else existing_end
    )
    new_start = (
        datetime_key_for_comparison(new_start)
        if isinstance(new_start, str)
        else new_start
    )
    new_end = (
        datetime_key_for_comparison(new_end) if isinstance(new_end, str) else new_end
    )
    if not (isinstance(new_start, datetime) and isinstance(new_end, datetime)):
        return []
    if not (
        isinstance(existing_start, datetime) and isinstance(existing_end, datetime)
    ):
        return [(new_start, new_end)]
    try:
        fully_disjoint = new_end <= existing_start or new_start >= existing_end
        if fully_disjoint:
            return [(new_start, new_end)]
        segments = []
        if new_start < existing_start:
            segments.append((new_start, existing_start))
        if new_end > existing_end:
            segments.append((existing_end, new_end))
        return segments
    except TypeError:
        # Real datetimes that still can't be compared (aware vs. naive) -
        # same "can't confirm smaller than the whole window" fallback.
        return [(new_start, new_end)]


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
