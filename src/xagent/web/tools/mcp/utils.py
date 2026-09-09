import json
import os
import re
import urllib.request
from datetime import date, datetime, time, timedelta
from datetime import timezone as dt_timezone
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    *,
    unchecked_reason: str | None = None,
) -> str:
    """Build the status="conflict" envelope a calendar-writing MCP tool
    returns instead of creating/updating an event, so the calling agent
    reports the conflict to the user rather than silently double-booking.

    `unchecked_reason`, when given, is a short human-readable explanation
    of why `unchecked_attendees` couldn't be checked (e.g. a missing OAuth
    scope) - included only when there's an actionable cause to relay,
    since most unchecked cases (an attendee absent from the provider's
    response, or its own per-attendee error) are already self-explanatory.
    """
    payload = {
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
    }
    if unchecked_reason:
        payload["unchecked_reason"] = unchecked_reason
    return json.dumps(payload, ensure_ascii=False)


def unchecked_extra(
    unchecked_attendees: list[str], unchecked_reason: str | None
) -> dict[str, Any]:
    """Extra fields for a status="success" envelope when some attendees
    ended up unchecked - empty (nothing to add) when there's nothing to
    report, matching `conflict_response`'s own "only when actionable"
    policy for `unchecked_reason`."""
    if not unchecked_attendees:
        return {}
    extra: dict[str, Any] = {"unchecked_attendees": unchecked_attendees}
    if unchecked_reason:
        extra["unchecked_reason"] = unchecked_reason
    return extra


def attendees_were_given(attendees: list[str] | str | None) -> bool:
    """Whether `attendees` was actually provided by the caller for an
    update, treating an empty string the same as not-provided at all -
    matching every other optional field's truthy convention here (and the
    create path's own check) - rather than as "clear every attendee".
    That's still expressible, just via an explicit empty list: `[]` is a
    deliberate, differently-typed value that must keep working."""
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
    return [
        address
        for address in normalize_addresses(attendees)
        if address.lower() not in existing_attendee_emails
    ]


def attendees_needing_check(
    normalized_attendees: list[str],
    existing_attendee_emails: set[str],
    *,
    moved_to_a_disjoint_window: bool,
) -> list[str]:
    """Which attendees actually need a fresh free/busy check on an update.

    An existing attendee's schedule will always show this event's own
    busy block for the window it currently occupies - querying that
    window for them can't distinguish "busy because of this very event"
    from a real conflict. Only once the window has moved somewhere that
    no longer overlaps the old one is it safe to re-check everyone;
    otherwise only the newly-added attendees (who have no such footprint
    yet) are worth checking.
    """
    if moved_to_a_disjoint_window:
        return normalized_attendees
    return [
        address
        for address in normalized_attendees
        if address.lower() not in existing_attendee_emails
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
    enough to guarantee a safe comparison - see ``windows_overlap``,
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


# Microsoft Graph timeZone values come in two shapes depending on how/where
# an event was created: IANA names (e.g. "Asia/Singapore", which Graph also
# accepts on write) and legacy Windows names (e.g. "Pacific Standard Time",
# from desktop Outlook or an Exchange org defaulting to them). zoneinfo only
# understands the former. This is the standard (CLDR) Windows-to-IANA
# mapping, restricted to the zone list Graph's own dateTimeTimeZone docs
# enumerate as supported - not exhaustive of every Windows zone that has
# ever existed, but covers every zone Graph itself claims to support.
_WINDOWS_TO_IANA: dict[str, str] = {
    "UTC": "UTC",
    "GMT Standard Time": "Europe/London",
    "Greenwich Standard Time": "Atlantic/Reykjavik",
    "W. Europe Standard Time": "Europe/Berlin",
    "Central Europe Standard Time": "Europe/Budapest",
    "Central European Standard Time": "Europe/Warsaw",
    "Romance Standard Time": "Europe/Paris",
    "E. Europe Standard Time": "Europe/Bucharest",
    "FLE Standard Time": "Europe/Kyiv",
    "Turkey Standard Time": "Europe/Istanbul",
    "Russian Standard Time": "Europe/Moscow",
    "Kaliningrad Standard Time": "Europe/Kaliningrad",
    "Arabic Standard Time": "Asia/Baghdad",
    "Syria Standard Time": "Asia/Damascus",
    "Arab Standard Time": "Asia/Riyadh",
    "Israel Standard Time": "Asia/Jerusalem",
    "Jordan Standard Time": "Asia/Amman",
    "Middle East Standard Time": "Asia/Beirut",
    "Egypt Standard Time": "Africa/Cairo",
    "South Africa Standard Time": "Africa/Johannesburg",
    "E. Africa Standard Time": "Africa/Nairobi",
    "Mauritius Standard Time": "Indian/Mauritius",
    "Iran Standard Time": "Asia/Tehran",
    "Arabian Standard Time": "Asia/Dubai",
    "Azerbaijan Standard Time": "Asia/Baku",
    "Georgian Standard Time": "Asia/Tbilisi",
    "Caucasus Standard Time": "Asia/Yerevan",
    "Afghanistan Standard Time": "Asia/Kabul",
    "Pakistan Standard Time": "Asia/Karachi",
    "West Asia Standard Time": "Asia/Tashkent",
    "India Standard Time": "Asia/Kolkata",
    "Sri Lanka Standard Time": "Asia/Colombo",
    "Nepal Standard Time": "Asia/Kathmandu",
    "Central Asia Standard Time": "Asia/Almaty",
    "Bangladesh Standard Time": "Asia/Dhaka",
    "Ekaterinburg Standard Time": "Asia/Yekaterinburg",
    "Myanmar Standard Time": "Asia/Yangon",
    "SE Asia Standard Time": "Asia/Bangkok",
    "Novosibirsk Standard Time": "Asia/Novosibirsk",
    "China Standard Time": "Asia/Shanghai",
    "North Asia Standard Time": "Asia/Krasnoyarsk",
    "Singapore Standard Time": "Asia/Singapore",
    "Taipei Standard Time": "Asia/Taipei",
    "Ulaanbaatar Standard Time": "Asia/Ulaanbaatar",
    "North Asia East Standard Time": "Asia/Irkutsk",
    "W. Australia Standard Time": "Australia/Perth",
    "Tokyo Standard Time": "Asia/Tokyo",
    "Korea Standard Time": "Asia/Seoul",
    "Cen. Australia Standard Time": "Australia/Adelaide",
    "AUS Central Standard Time": "Australia/Darwin",
    "E. Australia Standard Time": "Australia/Brisbane",
    "AUS Eastern Standard Time": "Australia/Sydney",
    "West Pacific Standard Time": "Pacific/Port_Moresby",
    "Tasmania Standard Time": "Australia/Hobart",
    "Yakutsk Standard Time": "Asia/Yakutsk",
    "Central Pacific Standard Time": "Pacific/Guadalcanal",
    "Vladivostok Standard Time": "Asia/Vladivostok",
    "New Zealand Standard Time": "Pacific/Auckland",
    "Fiji Standard Time": "Pacific/Fiji",
    "Magadan Standard Time": "Asia/Magadan",
    "Tonga Standard Time": "Pacific/Tongatapu",
    "Samoa Standard Time": "Pacific/Apia",
    "Line Islands Standard Time": "Pacific/Kiritimati",
    "Dateline Standard Time": "Etc/GMT+12",
    "Hawaiian Standard Time": "Pacific/Honolulu",
    "Alaskan Standard Time": "America/Anchorage",
    "Pacific Standard Time (Mexico)": "America/Santa_Isabel",
    "Pacific Standard Time": "America/Los_Angeles",
    "US Mountain Standard Time": "America/Phoenix",
    "Mountain Standard Time (Mexico)": "America/Chihuahua",
    "Mountain Standard Time": "America/Denver",
    "Central America Standard Time": "America/Guatemala",
    "Central Standard Time": "America/Chicago",
    "Central Standard Time (Mexico)": "America/Mexico_City",
    "Canada Central Standard Time": "America/Regina",
    "SA Pacific Standard Time": "America/Bogota",
    "Eastern Standard Time": "America/New_York",
    "US Eastern Standard Time": "America/Indiana/Indianapolis",
    "Venezuela Standard Time": "America/Caracas",
    "Paraguay Standard Time": "America/Asuncion",
    "Atlantic Standard Time": "America/Halifax",
    "Central Brazilian Standard Time": "America/Cuiaba",
    "SA Western Standard Time": "America/La_Paz",
    "Pacific SA Standard Time": "America/Santiago",
    "Newfoundland Standard Time": "America/St_Johns",
    "E. South America Standard Time": "America/Sao_Paulo",
    "Argentina Standard Time": "America/Argentina/Buenos_Aires",
    "SA Eastern Standard Time": "America/Cayenne",
    "Greenland Standard Time": "America/Godthab",
    "Montevideo Standard Time": "America/Montevideo",
    "Bahia Standard Time": "America/Bahia",
    "Azores Standard Time": "Atlantic/Azores",
    "Cape Verde Standard Time": "Atlantic/Cape_Verde",
    "Morocco Standard Time": "Africa/Casablanca",
    "Namibia Standard Time": "Africa/Windhoek",
    "W. Central Africa Standard Time": "Africa/Lagos",
}


def resolve_zone_name(name: str) -> str:
    """Map a Graph timeZone value to a name zoneinfo can load.

    Returns the input unchanged when it isn't a recognized legacy Windows
    zone name - it's then assumed to already be IANA-shaped, which
    zoneinfo can load directly.
    """
    return _WINDOWS_TO_IANA.get(name, name)


def resolve_zoneinfo(name: str) -> ZoneInfo:
    """Resolve a Graph timeZone value (Windows or IANA) to a real
    ``ZoneInfo``, for use anywhere a working zone is required (not just a
    best-effort comparison) - e.g. attaching a real UTC offset to a naive
    datetime string.

    Raises ``ValueError`` rather than silently defaulting to UTC when the
    name can't be resolved: a wrong silent guess here is exactly the class
    of bug (a query or write running in the wrong real-world window) this
    helper exists to prevent.
    """
    try:
        return ZoneInfo(resolve_zone_name(name))
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Timezone {name!r} isn't a recognized IANA or Windows zone name."
        ) from exc


def timezones_could_differ(a: str, b: str) -> bool:
    """True only when two Graph timeZone values can be POSITIVELY
    confirmed to denote different zones (compared by current UTC offset,
    not just the zone key, so e.g. a Windows name and the IANA name Graph
    also accepts for the same real zone compare equal).

    False whenever that can't be confirmed - including when either name
    fails to resolve (an unmappable legacy Windows zone) - because "can't
    tell" must never read as "these are different": the only use of this
    function is deciding whether to reject a caller-supplied timezone as
    ambiguous, and the actual write always uses Graph's own already-valid
    timeZone string regardless of this check's answer. Wrongly rejecting
    a same-zone resubmission (written in a different but equally valid
    form) is the real failure mode to avoid; wrongly allowing a genuinely
    different but unresolvable zone through is never worse than what
    happens when the caller omits the argument entirely.
    """
    if a == b:
        return False
    try:
        zone_a = resolve_zoneinfo(a)
        zone_b = resolve_zoneinfo(b)
    except ValueError:
        return False
    now = datetime.now(dt_timezone.utc)
    return now.astimezone(zone_a).utcoffset() != now.astimezone(zone_b).utcoffset()


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


def naive_day_bounds(date_value: str, *, days: int = 1) -> tuple[str, str]:
    """Return (start, end) NAIVE ISO datetime strings spanning ``days``
    full calendar day(s) starting at ``date_value``'s date - the
    zone-agnostic counterpart to ``calendar_day_bounds``, for an API like
    Outlook's dateTimeTimeZone that wants a naive clock value paired with
    a separate timeZone field rather than an embedded offset (attaching a
    zone here and stripping it back off would just be lossy round-tripping
    for no benefit, since no zone conversion is actually needed - "midnight
    of this date" is the same clock reading regardless of which zone it's
    later paired with).

    ``date_value`` may be a bare "YYYY-MM-DD" or a full datetime string
    (only its date component is used).
    """
    day: date = datetime.fromisoformat(date_value).date()
    start = datetime.combine(day, time.min)
    end = start + timedelta(days=days)
    return start.isoformat(), end.isoformat()


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
    zone = resolve_zoneinfo(tz_name)
    day: date = datetime.fromisoformat(date_value).date()
    start = datetime.combine(day, time.min, tzinfo=zone)
    end = start + timedelta(days=days)
    return start.isoformat(), end.isoformat()


def windows_overlap(
    existing_start: datetime | str | None,
    existing_end: datetime | str | None,
    new_start: datetime | str | None,
    new_end: datetime | str | None,
) -> bool:
    """Whether two half-open [start, end) intervals share any instant.

    Takes comparison keys already produced by ``datetime_key_for_comparison``
    (so an equal-instant value compares equal regardless of formatting/zone
    differences), not raw strings.

    A calendar-conflict check must tell "moved to a genuinely disjoint
    window" (safe to re-check every existing attendee - this event's own
    footprint can't appear in a window it doesn't occupy) apart from "same
    or overlapping window" (an existing attendee's free/busy would still
    show this very event's own busy block inside the overlap, which isn't
    a real conflict). This is that test.

    If any key isn't a real parsed ``datetime`` - still a raw string
    because it failed to parse, or (rarer) one side aware and the other
    naive - "can't confirm disjoint" applies, so this returns ``True``
    (overlapping) rather than assuming a safety it can't verify. A raw
    string still compares (and orders) against another string with `<`
    without raising, so this can't rely on catching ``TypeError`` alone -
    it must check that every key actually is a comparable `datetime`.
    """
    if not (
        isinstance(existing_start, datetime)
        and isinstance(existing_end, datetime)
        and isinstance(new_start, datetime)
        and isinstance(new_end, datetime)
    ):
        return True
    try:
        return existing_start < new_end and new_start < existing_end
    except TypeError:
        return True


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
