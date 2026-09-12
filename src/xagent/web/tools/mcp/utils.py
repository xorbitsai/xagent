import json
import os
import re
import urllib.request
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as _date_parser
from dateutil.rrule import rrulestr as _rrulestr

from ....config import get_tool_max_output_length

_DIGITS_ONLY_RE = re.compile(r"[0-9]+")
_BARE_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_NUMERIC_BYDAY_RE = re.compile(r"[+-]?[0-9]{1,2}(MO|TU|WE|TH|FR|SA|SU)")
# RFC 5545 UNTIL is always in "basic" form - no "-"/":" separators, unlike
# ISO8601's "extended" form that dateutil's isoparse also happily accepts
# (e.g. "2026-09-11T23:59:59+08:00"). dateutil.rrule.rrulestr's own RFC
# 5545 line parser expects exactly this basic shape and raises a raw,
# uninformative "too many values to unpack" if it isn't - see parse_rrule.
# The trailing "Z" is itself optional: RFC 5545 permits UNTIL as a
# floating "DATE WITH LOCAL TIME" (no "Z") whenever DTSTART is also
# floating local time, not just the aware "DATE WITH UTC TIME" form.
_UNTIL_RE = re.compile(r"[0-9]{8}(T[0-9]{6}Z?)?")


def allowed_dirs_from_env(env_var_name: str) -> list[Path]:
    """Parse JSON or legacy comma-separated roots.

    An unset, blank, or empty legacy value falls back to CWD for compatibility.
    An explicit JSON array is authoritative, so an empty array denies all roots.
    """
    raw_dirs = os.environ.get(env_var_name, "")
    stripped_raw_dirs = raw_dirs.strip()
    is_json_array = stripped_raw_dirs.startswith("[")
    if is_json_array:
        try:
            decoded_dirs = json.loads(stripped_raw_dirs)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{env_var_name} must contain a valid JSON array") from exc
        if not isinstance(decoded_dirs, list) or not all(
            isinstance(raw_dir, str) for raw_dir in decoded_dirs
        ):
            raise ValueError(f"{env_var_name} must contain a JSON array of paths")
        raw_dir_values = decoded_dirs
    else:
        raw_dir_values = raw_dirs.split(",")
    try:
        parsed_dirs = []
        for raw_dir in raw_dir_values:
            stripped = raw_dir.strip()
            if not stripped:
                continue
            candidate = Path(stripped).expanduser()
            resolved = candidate.resolve()
            # Python 3.13's non-strict resolve() can leave a symlink loop
            # unresolved instead of raising. Validate an explicit symlink
            # strictly so a broken or cyclic configured root cannot be
            # mistaken for a usable authorization boundary.
            if candidate.is_symlink():
                resolved = candidate.resolve(strict=True)
            parsed_dirs.append(resolved)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{env_var_name} contains an invalid path") from exc
    if is_json_array:
        return parsed_dirs
    return parsed_dirs or [Path.cwd().resolve()]


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


def success_with_capped_dict(
    field_name: str,
    data: Any,
    *,
    extra_fields: dict[str, Any] | None = None,
) -> str:
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

    ``extra_fields`` adds fixed top-level response fields that must be counted
    while trimming the dict, such as a calendar event's derived Meet link.
    Reserved envelope keys cannot be overridden.
    """
    extras = extra_fields or {}
    reserved_fields = {"status", field_name, "truncated"}
    if reserved_fields.intersection(extras):
        raise ValueError(
            "extra_fields must not override status, the capped field, or truncated"
        )

    max_output_length = get_tool_max_output_length()

    def _build(
        payload: Any,
        truncated: bool,
        *,
        with_extras: bool = True,
    ) -> str:
        return json.dumps(
            {
                "status": "success",
                field_name: payload,
                **(extras if with_extras else {}),
                "truncated": truncated,
            },
            ensure_ascii=False,
        )

    response = _build(data, False)
    if not isinstance(data, dict) or len(response) <= max_output_length:
        return response

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

    if len(response) > max_output_length:
        compact_data: dict[str, Any] = {}
        if isinstance(data.get("id"), (str, int, float, bool)):
            compact_data["id"] = data["id"]
        candidates = (
            _build(compact_data, True, with_extras=False),
            _build({}, True, with_extras=False),
            json.dumps({"status": "success", "truncated": True}, ensure_ascii=False),
        )
        for candidate in candidates:
            if len(candidate) <= max_output_length:
                return candidate
        return json.dumps({"status": "success"}, ensure_ascii=False)
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


def _halve_largest_list_fields_until_bounded(
    payload: dict[str, Any], field_names: tuple[str, ...], max_output_length: int
) -> str:
    """Serialize ``payload``, halving its largest named list until bounded."""
    response = json.dumps(payload, ensure_ascii=False)
    while len(response) > max_output_length:
        populated_fields = [name for name in field_names if payload.get(name)]
        if not populated_fields:
            break
        field_name = max(
            populated_fields,
            key=lambda name: len(json.dumps(payload[name], ensure_ascii=False)),
        )
        values = payload[field_name]
        payload[field_name] = values[: len(values) // 2]
        payload["truncated"] = True
        response = json.dumps(payload, ensure_ascii=False)
    return response


def conflict_response(
    conflicts: list[dict[str, Any]],
    unchecked_attendees: list[str],
    start: str,
    end: str,
    *,
    check_error: str | None = None,
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
    if check_error:
        payload["check_error"] = check_error
    max_output_length = get_tool_max_output_length()
    response = _halve_largest_list_fields_until_bounded(
        payload, ("conflicts", "unchecked_attendees"), max_output_length
    )

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


def incomplete_check_response(
    unchecked_attendees: list[str],
    start: str,
    end: str,
    *,
    message: str | None = None,
) -> str:
    """Return a bounded pre-write response when required calendars could
    not be checked. The caller may retry with ``ignore_conflicts=True`` only
    after the user explicitly accepts proceeding without complete checks.
    ``message`` can describe a non-calendar prerequisite, such as a malformed
    event window, while preserving the same bounded response contract.
    """
    payload: dict[str, Any] = {
        "status": "conflict_check_incomplete",
        "message": message
        or (
            f"Availability could not be checked for {len(unchecked_attendees)} "
            f"calendar(s) for {start} - {end}. No event was written."
        ),
        "unchecked_attendees": unchecked_attendees,
        "hint": (
            "Report the unchecked calendars to the user. Choose another action, "
            "or call this tool again with ignore_conflicts=true only after the "
            "user explicitly confirms they want to proceed without complete "
            "availability checks."
        ),
        "truncated": False,
    }
    max_output_length = get_tool_max_output_length()
    response = _halve_largest_list_fields_until_bounded(
        payload, ("unchecked_attendees",), max_output_length
    )
    if len(response) <= max_output_length:
        return response

    compact_payloads: tuple[dict[str, Any], ...] = (
        {
            "status": "conflict_check_incomplete",
            "message": "Availability check incomplete; details truncated.",
            "truncated": True,
        },
        {"status": "conflict_check_incomplete", "truncated": True},
    )
    for compact_payload in compact_payloads:
        compact = json.dumps(compact_payload, ensure_ascii=False)
        if len(compact) <= max_output_length:
            return compact
    return json.dumps({"status": "conflict_check_incomplete"}, ensure_ascii=False)


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
    existing = {email.strip().lower() for email in existing_attendee_emails}
    return [
        address
        for address in normalize_addresses(attendees or [])
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
        parsed: datetime = _date_parser.isoparse(normalized)
    except ValueError:
        return value
    return parsed


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
        parsed: datetime = _date_parser.isoparse(value)
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
    parsed: datetime = _date_parser.isoparse(value)
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
    day: date = _date_parser.isoparse(date_value).date()
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


def is_bare_date(value: str) -> bool:
    """Whether a caller-supplied date/time string is a bare "date" (e.g.
    Google's all-day form, "2026-08-26") rather than a "dateTime".

    Checked as an exact, calendar-valid YYYY-MM-DD value (after stripping
    outer whitespace) rather than only by the absence of a
    "T"/"t" date/time separator: RFC3339 also permits a space in place of
    "T" for readability, so "2026-08-26 07:00:00" contains no "t" either
    and would otherwise be misclassified as a bare date.

    The explicit ASCII shape check rejects abbreviated dates and Unicode
    digit lookalikes; ``date.fromisoformat`` then rejects impossible dates.
    """
    normalized = value.strip()
    if not _BARE_DATE_RE.fullmatch(normalized):
        return False
    try:
        date.fromisoformat(normalized)
    except ValueError:
        return False
    return True


def _validate_rrule_constraints(
    parts: dict[str, str], *, dtstart_is_date: bool
) -> None:
    """Enforce RFC 5545 constraints that dateutil accepts leniently."""
    freq = parts["FREQ"]
    if dtstart_is_date and any(
        key in parts for key in ("BYSECOND", "BYMINUTE", "BYHOUR")
    ):
        raise ValueError(
            "invalid recurrence rule: BYSECOND, BYMINUTE, and BYHOUR must "
            "not be used with an all-day DATE start"
        )

    byday = parts.get("BYDAY")
    has_numeric_byday = bool(
        byday
        and any(
            _NUMERIC_BYDAY_RE.fullmatch(value.strip()) for value in byday.split(",")
        )
    )
    if has_numeric_byday and freq not in {"MONTHLY", "YEARLY"}:
        raise ValueError(
            "invalid recurrence rule: numeric BYDAY values are only valid "
            "with MONTHLY or YEARLY frequency"
        )
    if has_numeric_byday and freq == "YEARLY" and "BYWEEKNO" in parts:
        raise ValueError(
            "invalid recurrence rule: numeric BYDAY must not be combined "
            "with BYWEEKNO in a YEARLY rule"
        )
    if freq == "WEEKLY" and "BYMONTHDAY" in parts:
        raise ValueError(
            "invalid recurrence rule: BYMONTHDAY must not be used with WEEKLY frequency"
        )
    if freq in {"DAILY", "WEEKLY", "MONTHLY"} and "BYYEARDAY" in parts:
        raise ValueError(
            "invalid recurrence rule: BYYEARDAY is only valid with YEARLY frequency"
        )
    if freq != "YEARLY" and "BYWEEKNO" in parts:
        raise ValueError(
            "invalid recurrence rule: BYWEEKNO is only valid with YEARLY frequency"
        )
    if "BYSETPOS" in parts and not any(
        key.startswith("BY") and key != "BYSETPOS" for key in parts
    ):
        raise ValueError(
            "invalid recurrence rule: BYSETPOS requires another BY rule part"
        )


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
    if "\n" in rrule_text or "\r" in rrule_text:
        raise ValueError("recurrence rule must not contain embedded newlines")
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

    ``timezone``, if given, localizes a naive DATE-TIME ``dtstart``. A
    genuine all-day event uses a DATE value instead and remains timezone
    independent. RFC 5545 requires UNTIL to have the same value type as
    DTSTART: a DATE for an all-day start, a floating DATE-TIME for a
    floating start, or a UTC DATE-TIME for an aware/timezone-qualified
    start. These relationships are checked explicitly rather than left to
    dateutil, which accepts several mismatched combinations. The
    UNTIL-not-before-dtstart check then compares values with matching types
    and awareness.

    The returned dict (rather than the parsed rrule object) is what
    callers actually build a provider payload from: Google takes the RRULE
    text close to verbatim, while Outlook needs these components
    translated into Graph's own pattern/range structure - neither needs
    dateutil's internal representation, just confirmation that the text is
    valid RFC 5545.
    """
    if isinstance(dtstart, str):
        dtstart = dtstart.strip()
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
    dtstart_is_bare_date = isinstance(dtstart, str) and is_bare_date(dtstart)
    _validate_rrule_constraints(parts, dtstart_is_date=dtstart_is_bare_date)
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

    if isinstance(dtstart, datetime):
        anchor = dtstart
    else:
        try:
            anchor = _date_parser.isoparse(dtstart)
        except ValueError as exc:
            raise ValueError(
                f"invalid start time for recurrence rule: {dtstart}"
            ) from exc
    resolved_timezone = resolve_zoneinfo(timezone) if timezone is not None else None
    if until_dt is not None:
        until_is_date = "T" not in parts["UNTIL"]
        if dtstart_is_bare_date != until_is_date:
            raise ValueError(
                "invalid recurrence rule: UNTIL must use the same DATE or "
                "DATE-TIME value type as the event start"
            )
        if not dtstart_is_bare_date:
            until_is_utc = parts["UNTIL"].endswith("Z")
            dtstart_has_timezone = (
                anchor.tzinfo is not None or resolved_timezone is not None
            )
            if dtstart_has_timezone != until_is_utc:
                expected = (
                    "UTC with a trailing Z"
                    if dtstart_has_timezone
                    else "floating local time"
                )
                raise ValueError(
                    f"invalid recurrence rule: UNTIL must be {expected} to match "
                    "the event start"
                )
    if (
        anchor.tzinfo is None
        and resolved_timezone is not None
        and not dtstart_is_bare_date
    ):
        anchor = anchor.replace(tzinfo=resolved_timezone)
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
