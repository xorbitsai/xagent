"""Translate RFC 5545 recurrence rules into Microsoft Graph payloads."""

import calendar as _stdlib_calendar
import re
from datetime import date, datetime, timedelta
from typing import Any

from dateutil import parser as _date_parser
from dateutil import tz as _date_tz

from .utils import parse_rrule
from .utils import resolve_zoneinfo as _resolve_zoneinfo

_RRULE_DAY_TO_GRAPH = {
    "MO": "monday",
    "TU": "tuesday",
    "WE": "wednesday",
    "TH": "thursday",
    "FR": "friday",
    "SA": "saturday",
    "SU": "sunday",
}


def _rrule_until_to_date(until: str, zone: Any, anchor: datetime) -> str:
    """Convert an RRULE UNTIL value ('20260911T235959Z' or a bare
    '20260911') into the 'YYYY-MM-DD' form Graph's recurrenceRange wants,
    expressed in `zone`.

    Graph's recurrenceRange.endDate is a calendar date interpreted in
    recurrenceTimeZone, not UTC - converting first (rather than taking the
    UTC calendar date directly) matters whenever the UTC UNTIL instant
    crosses local midnight, or a valid final local occurrence would be
    silently excluded.

    But Graph's `endDate` range is day-granular - Graph includes the
    WHOLE end_date day, not a specific instant within it - while every
    occurrence in this series actually happens at `anchor`'s own local
    time-of-day, not UNTIL's. So whenever UNTIL's local time-of-day falls
    EARLIER in the day than the series' own occurrence time, the
    occurrence that would land on that same calendar day is genuinely
    past the aware UNTIL cutoff instant, yet Graph would still include it
    since it only ever checks the calendar date - `endDate` is backed off
    by one day in that case so Graph actually excludes it, rather than
    silently running the series one occurrence past what UNTIL specified.

    `anchor` is normalized into `zone` before this helper is called, so
    its wall-clock time and the converted UNTIL value are in the same
    frame as Graph's recurrenceTimeZone.
    """
    try:
        parsed = _date_parser.isoparse(until.strip())
    except ValueError as exc:
        raise ValueError(f"invalid UNTIL value in recurrence rule: {until}") from exc
    if parsed.tzinfo is None:
        return parsed.date().isoformat()
    parsed = parsed.astimezone(zone)
    end_date = parsed.date()
    if parsed.time() < anchor.time():
        end_date -= timedelta(days=1)
    return end_date.isoformat()


_WEEKDAY_INDEX_TO_GRAPH_DAY = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _graph_day_from_date(date_str: str) -> str:
    """The Graph day-of-week name (monday/tuesday/.../sunday) for a
    'YYYY-MM-DD' date."""
    return _WEEKDAY_INDEX_TO_GRAPH_DAY[date.fromisoformat(date_str).weekday()]


_NUMBERED_BYDAY_RE = re.compile(r"^([+-]?)(\d{1,2})?(MO|TU|WE|TH|FR|SA|SU)$")
_PLAIN_BYDAY_RE = re.compile(r"^(MO|TU|WE|TH|FR|SA|SU)$")

_ORDINAL_TO_GRAPH_INDEX = {1: "first", 2: "second", 3: "third", 4: "fourth", -1: "last"}


def _parse_plain_byday(byday: str) -> list[str]:
    """Parse and deduplicate unnumbered BYDAY values."""
    days: list[str] = []
    seen_days: set[str] = set()
    for raw_code in byday.split(","):
        code = raw_code.strip().upper()
        match = _PLAIN_BYDAY_RE.fullmatch(code)
        if not match:
            raise ValueError(f"invalid day code in BYDAY: {raw_code!r}")
        day_code = match.group(1)
        if day_code not in seen_days:
            seen_days.add(day_code)
            days.append(_RRULE_DAY_TO_GRAPH[day_code])
    return days


def _parse_relative_byday(
    byday: str, bysetpos: str | None = None
) -> tuple[str, list[str]]:
    """Parse a numbered BYDAY value (e.g. "2TU" for "the second Tuesday") into
    the (index, daysOfWeek) pair Graph's relativeMonthly/relativeYearly
    pattern needs.

    Every code must carry the same numeric ordinal - Graph's `index` field
    applies to the whole pattern, not per-day, so "2TU,3WE" ("second
    Tuesday" and "third Wednesday" together) has no single-index Graph
    equivalent and must be rejected rather than silently collapsed to one
    of the two ordinals. Likewise a code with no ordinal at all (plain
    "MO") means something different under RFC 5545 (every such weekday in
    the period, not one specific occurrence) and is rejected rather than
    guessed at.

    An alternative RFC 5545 spelling uses unnumbered BYDAY values plus a
    single BYSETPOS (e.g. BYDAY=MO,TU,WE,TH,FR;BYSETPOS=1 for the first
    weekday). That form maps directly to Graph's shared index and
    daysOfWeek fields and is accepted here.

    More than one DISTINCT weekday sharing the same inline ordinal (e.g.
    "2TU,2WE", "the second Tuesday AND the second Wednesday") is rejected
    too, for a different reason: RFC 5545 means two independent
    occurrences per period (confirmed against dateutil's rrule, the
    reference implementation this codebase already validates against),
    but Microsoft's own recurrencePattern docs state that when a relative
    monthly/yearly pattern's `daysOfWeek` lists more than one day, "the
    event falls on the first day that satisfies the pattern" - a single
    occurrence, not one per listed day. Sending this rule to Graph as-is
    would silently produce a materially narrower series (missing every
    other listed weekday) with no error surfaced anywhere.
    """
    if bysetpos is not None:
        if "," in bysetpos or not re.fullmatch(r"[+-]?\d{1,3}", bysetpos):
            raise ValueError(
                "unsupported recurrence pattern: this connector only supports "
                f"a single integer value for BYSETPOS, got {bysetpos!r}"
            )
        bysetpos_ordinal = int(bysetpos)
        if bysetpos_ordinal not in _ORDINAL_TO_GRAPH_INDEX:
            raise ValueError(
                f"unsupported recurrence pattern: BYSETPOS={bysetpos_ordinal} has no "
                "Outlook equivalent; this connector only supports "
                "1, 2, 3, 4, and -1 (last)"
            )
        try:
            bysetpos_days = _parse_plain_byday(byday)
        except ValueError as exc:
            raise ValueError(
                "unsupported recurrence pattern: BYSETPOS requires "
                "unnumbered BYDAY values"
            ) from exc
        return _ORDINAL_TO_GRAPH_INDEX[bysetpos_ordinal], bysetpos_days

    ordinal: int | None = None
    days: list[str] = []
    seen_days: set[str] = set()
    for raw_code in byday.split(","):
        code = raw_code.strip().upper()
        match = _NUMBERED_BYDAY_RE.fullmatch(code)
        if not match:
            raise ValueError(f"invalid day code in BYDAY: {raw_code!r}")
        sign, number, day_code = match.groups()
        if number is None:
            raise ValueError(
                f"unsupported recurrence pattern: BYDAY={byday!r} includes "
                f"{raw_code!r} with no numeric ordinal; a relative pattern "
                "needs one on every day (e.g. '2TU' for 'the second "
                "Tuesday', not 'TU')"
            )
        this_ordinal = int(f"{sign}{number}")
        if this_ordinal not in _ORDINAL_TO_GRAPH_INDEX:
            raise ValueError(
                f"unsupported recurrence pattern: BYDAY ordinal "
                f"{this_ordinal} has no Outlook equivalent; this connector "
                "only supports 1st/2nd/3rd/4th, and -1 (last)"
            )
        if ordinal is None:
            ordinal = this_ordinal
        elif ordinal != this_ordinal:
            raise ValueError(
                f"unsupported recurrence pattern: BYDAY={byday!r} mixes "
                "different numeric ordinals; Outlook's recurrence index "
                "applies to the whole rule, not per day"
            )
        if day_code not in seen_days:
            seen_days.add(day_code)
            days.append(_RRULE_DAY_TO_GRAPH[day_code])
    if ordinal is None:
        raise RuntimeError(
            f"no BYDAY ordinal found while parsing {byday!r} - every code "
            "in the loop above either sets one or raises, so this "
            "shouldn't be reachable"
        )
    if len(days) > 1:
        raise ValueError(
            f"unsupported recurrence pattern: BYDAY={byday!r} specifies "
            "more than one distinct weekday for the same relative "
            "ordinal - Outlook's relativeMonthly/relativeYearly pattern "
            "picks only the first listed day satisfying the pattern each "
            "period, not one occurrence per day, so this would silently "
            "produce a narrower series than requested"
        )
    return _ORDINAL_TO_GRAPH_INDEX[ordinal], days


# RFC 5545's `1*DIGIT` grammar for INTERVAL/COUNT has no upper bound, but
# Graph types `recurrencePattern.interval` and `recurrenceRange.
# numberOfOccurrences` as signed Int32 - a value beyond this (valid RFC
# 5545 text the shared parser's digit-only/positivity checks happily
# accept) would otherwise reach Graph as an out-of-range JSON integer and
# get rejected remotely with an opaque error, instead of this connector's
# own clear local one.
_GRAPH_INT32_MAX = 2_147_483_647

# Components every FREQ recognizes in this connector, on top of whatever a
# specific FREQ branch below consumes - checked once a pattern is built, so
# a component this connector silently ignores (e.g. BYDAY on a plain DAILY
# rule, or BYSETPOS/BYHOUR/BYWEEKNO anywhere) is rejected instead of just
# never making it into the Graph payload.
_COMMON_RRULE_KEYS = frozenset({"FREQ", "INTERVAL", "UNTIL", "COUNT", "WKST"})
_FREQ_RECOGNIZED_KEYS = {
    "DAILY": frozenset({"BYDAY"}),
    "WEEKLY": frozenset({"BYDAY"}),
    "MONTHLY": frozenset({"BYDAY", "BYMONTHDAY", "BYSETPOS"}),
    "YEARLY": frozenset({"BYDAY", "BYMONTH", "BYMONTHDAY", "BYSETPOS"}),
}


def _int_rrule_component(
    parts: dict[str, str], key: str, default: int, low: int, high: int
) -> int:
    """Parse an RRULE numeric component (BYMONTHDAY/BYMONTH), defaulting to
    `default` - the corresponding field from DTSTART, per RFC 5545's own
    rule that an omitted BYMONTHDAY/BYMONTH is derived from the start date
    - when the component is absent. Range-checks the result and raises a
    clean error for a value `int()` can't parse (e.g. a comma-separated
    multi-value list, valid RFC 5545 but with no single-value equivalent
    in Graph's pattern) instead of letting a raw ValueError propagate.
    """
    if key not in parts:
        return default
    raw_value = parts[key]
    if "," in raw_value:
        raise ValueError(
            f"unsupported recurrence pattern: this connector only supports "
            f"a single value for {key}, got {raw_value!r}"
        )
    component_pattern = r"[+-]?\d{1,2}" if key == "BYMONTHDAY" else r"\d{1,2}"
    if not re.fullmatch(component_pattern, raw_value):
        raise ValueError(
            f"invalid recurrence rule: {key} must be a one- or two-digit "
            f"integer, got {raw_value!r}"
        )
    try:
        value = int(raw_value)
    except ValueError:
        raise ValueError(
            f"unsupported recurrence pattern: this connector only supports "
            f"a single value for {key}, got {parts[key]!r}"
        ) from None
    if not low <= value <= high:
        if key == "BYMONTHDAY" and value < low:
            # A negative BYMONTHDAY (e.g. -1 for "the last day of the
            # month") is valid RFC 5545 syntax, not a malformed rule - it's
            # rejected because this connector has no Graph equivalent for
            # it, which "invalid recurrence rule" would misleadingly imply.
            raise ValueError(
                f"unsupported recurrence pattern: BYMONTHDAY must be "
                f"between {low} and {high} for this connector, got "
                f"{value} (RFC 5545 allows negative values like -1 for "
                '"the last day of the month", but this connector doesn\'t '
                "support translating those)"
            )
        raise ValueError(
            f"invalid recurrence rule: {key} must be between {low} and "
            f"{high} for this connector, got {value}"
        )
    return value


def _validate_day_of_month(
    day_of_month: int,
    month: int | None,
    *,
    day_was_explicit: bool,
    month_was_explicit: bool = True,
) -> None:
    """Reject a BYMONTHDAY value Microsoft Graph would silently clamp to a
    different day instead of reproducing faithfully.

    Graph's documented behavior for `dayOfMonth` past a given month's
    actual length is to clamp to that month's last day - not skip the
    occurrence the way RFC 5545 itself does (BYMONTHDAY=31 on a MONTHLY
    rule skips every 30-day month entirely; BYMONTHDAY=29 on a YEARLY
    February rule skips every non-leap year). Silently letting either
    through would translate the rule into a materially different,
    recurring divergence with no warning - it's rejected here instead.

    `month` is None for `absoluteMonthly` (the pattern applies to every
    month, so only a day valid in EVERY month - including February - is
    safe) and 1-12 for `absoluteYearly` (the pattern applies to one
    specific month each year, so the check is against that month's own
    length, accounting for February's leap-year variability across
    different years).
    """
    day_description = (
        f"BYMONTHDAY={day_of_month}"
        if day_was_explicit
        else f"the start date's day of month ({day_of_month})"
    )
    if month is None:
        if day_of_month > 28:
            raise ValueError(
                "unsupported recurrence pattern: FREQ=MONTHLY with "
                f"{day_description} has no exact Outlook "
                "equivalent - Graph clamps a day past a short month's "
                "length to that month's last day instead of skipping the "
                "month the way RFC 5545 does, so only BYMONTHDAY 1-28 "
                "(valid in every month, including February) is supported "
                "here"
            )
        return
    # A leap year (2000) vs. a non-leap one (2001) bracket this month's
    # possible lengths across different years - the two only ever differ
    # for February.
    max_possible_day = _stdlib_calendar.monthrange(2000, month)[1]
    min_guaranteed_day = _stdlib_calendar.monthrange(2001, month)[1]
    if day_of_month > max_possible_day:
        raise ValueError(
            f"invalid recurrence rule: {day_description} does not exist in "
            f"month {month}"
        )
    if day_of_month > min_guaranteed_day:
        month_description = (
            f"BYMONTH={month}"
            if month_was_explicit
            else f"the start date's month ({month})"
        )
        raise ValueError(
            "unsupported recurrence pattern: FREQ=YEARLY with "
            f"{month_description} and {day_description} has no exact Outlook "
            "equivalent - Graph clamps to that month's last day in years "
            "where this day doesn't exist (e.g. Feb 29 in a non-leap "
            "year) instead of skipping that year's occurrence the way "
            "RFC 5545 does"
        )


def build_graph_recurrence(
    recurrence: str,
    start_datetime: str,
    timezone: str = "UTC",
    *,
    is_all_day: bool = False,
) -> dict[str, Any]:
    """Translate an RFC 5545 RRULE string into the pattern/range object
    Microsoft Graph's event.recurrence expects.

    Graph has no "just take this RRULE text" input the way Google Calendar
    does, so this maps the cases Toby is expected to produce: daily,
    weekly, absolute monthly/yearly (a plain BYMONTHDAY/BYMONTH, or none at
    all - RFC 5545 then derives it from the start date), and relative
    monthly/yearly via a numbered BYDAY (e.g. "2TU" for "the second
    Tuesday", "-1FR" for "the last Friday") translated into Graph's
    index+daysOfWeek. Any other component this connector doesn't translate
    for the given FREQ (e.g. BYDAY on a plain DAILY rule, or BYSETPOS/
    BYHOUR/BYWEEKNO anywhere) is rejected rather than silently dropped -
    dateutil's own RRULE validation accepts these as syntactically valid,
    but ignoring them here would translate a rule into a materially
    different, broader recurrence with no warning. Numeric components
    (INTERVAL, BYMONTHDAY, BYMONTH) are range-checked here too, rather than
    left for Graph's API to reject remotely with an opaque error.

    Outlook normally supplies start_datetime as a naive local dateTime
    paired with a separate timeZone field. Offset-bearing input is also
    accepted, but its instant is first converted into `timezone`; every
    derived field must use the same wall-clock frame Graph will apply via
    recurrenceTimeZone. RFC 5545 requires a recurrence's UNTIL to be a UTC
    value whenever DTSTART carries a timezone reference, so the localized
    anchor is passed directly into validation. The same resolved zone is
    then reused (not re-looked-up) to convert UNTIL into range.endDate.
    For an all-day event, only start_datetime's written calendar date is
    meaningful: no timezone conversion or daylight-saving validation is
    applied, and UNTIL must also be a date rather than a date-time.

    Args:
        recurrence: RFC 5545 RRULE text, with or without the ``RRULE:`` prefix.
        start_datetime: The event's first date or local/offset date-time.
        timezone: A Graph-supported IANA or Windows time zone name.
        is_all_day: Whether the event uses date-only all-day semantics.

    Returns:
        A Microsoft Graph ``patternedRecurrence`` payload.

    Raises:
        ValueError: If the rule, start, timezone, or requested recurrence
            cannot be represented faithfully by Microsoft Graph.
    """
    zone = _resolve_zoneinfo(timezone, allow_windows_names=True)
    try:
        anchor = _date_parser.isoparse(start_datetime)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid start_datetime: {start_datetime!r}") from exc
    if not is_all_day:
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=zone)
        else:
            anchor = anchor.astimezone(zone)
        if not _date_tz.datetime_exists(anchor):
            raise ValueError(
                f"{start_datetime!r} does not exist in timezone {timezone!r} "
                "because of a daylight-saving transition"
            )
        if _date_tz.datetime_ambiguous(anchor):
            raise ValueError(
                f"{start_datetime!r} is ambiguous in timezone {timezone!r} "
                "because of a daylight-saving transition"
            )
    # parse_rrule already guarantees INTERVAL is a positive integer
    # whenever it's present at all, so there's nothing left to check here
    # beyond the Graph-specific Int32 bound below.
    validation_start: str | datetime = (
        anchor.date().isoformat() if is_all_day else anchor
    )
    parts = parse_rrule(
        recurrence,
        validation_start,
        None,
    )
    freq = parts["FREQ"].upper()
    interval = int(parts.get("INTERVAL", "1"))
    start_date = anchor.date().isoformat()

    if "BYSETPOS" in parts and "BYDAY" not in parts:
        raise ValueError(
            "unsupported recurrence pattern: BYSETPOS requires BYDAY for "
            "translation into an Outlook relative recurrence"
        )

    if freq in _FREQ_RECOGNIZED_KEYS:
        leftover = set(parts) - _COMMON_RRULE_KEYS - _FREQ_RECOGNIZED_KEYS[freq]
        if leftover:
            raise ValueError(
                "unsupported recurrence pattern: this connector does not "
                f"translate {', '.join(sorted(leftover))} for FREQ={freq} "
                "into an Outlook recurrence"
            )

    pattern: dict[str, Any]
    if freq == "DAILY" and "BYDAY" in parts:
        if interval != 1:
            raise ValueError(
                "unsupported recurrence pattern: FREQ=DAILY with BYDAY is "
                "only equivalent to an Outlook weekly pattern when INTERVAL=1"
            )
        pattern = {
            "type": "weekly",
            "interval": 1,
            "daysOfWeek": _parse_plain_byday(parts["BYDAY"]),
            "firstDayOfWeek": _RRULE_DAY_TO_GRAPH[
                parts.get("WKST", "MO").strip().upper()
            ],
        }
    elif freq == "DAILY":
        pattern = {"type": "daily", "interval": interval}
    elif freq == "WEEKLY":
        byday = parts.get("BYDAY")
        if byday:
            days = _parse_plain_byday(byday)
        else:
            days = [_graph_day_from_date(start_date)]
        wkst_code = parts.get("WKST", "MO").strip().upper()
        if wkst_code not in _RRULE_DAY_TO_GRAPH:
            raise ValueError(f"invalid day code in WKST: {parts['WKST']!r}")
        pattern = {
            "type": "weekly",
            "interval": interval,
            "daysOfWeek": days,
            # RFC 5545 defaults WKST to Monday when omitted; Graph's own
            # firstDayOfWeek default is Sunday - stamping it explicitly
            # keeps interval-boundary weeks consistent with the RRULE's
            # actual (possibly implicit) semantics instead of silently
            # picking up Graph's different default.
            "firstDayOfWeek": _RRULE_DAY_TO_GRAPH[wkst_code],
        }
    elif freq in {"MONTHLY", "YEARLY"} and {
        "BYMONTHDAY",
        "BYDAY",
    }.issubset(parts):
        raise ValueError(
            f"unsupported recurrence pattern: FREQ={freq} with both "
            'BYMONTHDAY and BYDAY (e.g. "the 15th, but only if a '
            "specified weekday\") has no equivalent in Outlook's recurrence model"
        )
    elif freq == "MONTHLY" and "BYDAY" in parts:
        index, days = _parse_relative_byday(parts["BYDAY"], parts.get("BYSETPOS"))
        pattern = {
            "type": "relativeMonthly",
            "interval": interval,
            "daysOfWeek": days,
            "index": index,
        }
    elif freq == "MONTHLY":
        # BYMONTHDAY defaults to DTSTART's own day of month when omitted
        # (RFC 5545's own rule for an unqualified FREQ=MONTHLY), so this
        # covers "repeat monthly on the 15th" (BYMONTHDAY=15) as well as
        # plain "repeat monthly" (no BYMONTHDAY at all).
        day_of_month = _int_rrule_component(parts, "BYMONTHDAY", anchor.day, 1, 31)
        _validate_day_of_month(
            day_of_month,
            month=None,
            day_was_explicit="BYMONTHDAY" in parts,
        )
        pattern = {
            "type": "absoluteMonthly",
            "interval": interval,
            "dayOfMonth": day_of_month,
        }
    elif freq == "YEARLY" and "BYDAY" in parts:
        if "BYMONTH" not in parts:
            # Without BYMONTH, RFC 5545 makes this a single YEAR-WIDE
            # ordinal weekday (e.g. "the first Monday of the year" -
            # confirmed against dateutil's rrule, the reference
            # implementation this codebase already validates against) -
            # not one scoped to whatever month DTSTART happens to fall
            # in. Outlook's relativeYearly pattern always requires a
            # specific month, with no way to express "year-wide" at all,
            # so defaulting the month from the anchor would silently
            # narrow the series to a single month instead of representing
            # (or rejecting) the actual RFC 5545 semantics.
            raise ValueError(
                "unsupported recurrence pattern: FREQ=YEARLY;BYDAY=... "
                "without BYMONTH means a single year-wide ordinal weekday "
                "under RFC 5545 (e.g. 'the first Monday of the year'), "
                "which Outlook's relativeYearly pattern - always scoped "
                "to one specific month - has no way to represent; add an "
                "explicit BYMONTH to scope it to a single month instead"
            )
        index, days = _parse_relative_byday(parts["BYDAY"], parts.get("BYSETPOS"))
        month = _int_rrule_component(parts, "BYMONTH", anchor.month, 1, 12)
        pattern = {
            "type": "relativeYearly",
            "interval": interval,
            "daysOfWeek": days,
            "index": index,
            "month": month,
        }
    elif freq == "YEARLY":
        if "BYMONTHDAY" in parts and "BYMONTH" not in parts:
            # Same reasoning as the BYDAY branch above: without BYMONTH,
            # RFC 5545 expands BYMONTHDAY across EVERY month, every year
            # (confirmed against dateutil), not just DTSTART's own month.
            # Outlook's absoluteYearly pattern always requires a specific
            # month and can't represent "every month", so defaulting to
            # the anchor's month would silently narrow a 12x/year series
            # down to a single yearly occurrence with no warning.
            raise ValueError(
                "unsupported recurrence pattern: FREQ=YEARLY;BYMONTHDAY="
                "... without BYMONTH means this day of EVERY month, every "
                "year, under RFC 5545 - Outlook's absoluteYearly pattern "
                "has no way to represent that; use FREQ=MONTHLY instead "
                "for that meaning, or add an explicit BYMONTH to scope "
                "this rule to a single month"
            )
        # No BYMONTHDAY/BYDAY selector present (BYMONTH alone, or nothing
        # at all) - RFC 5545's own default derives the day (and, if
        # BYMONTH is also absent, the month) from DTSTART, which is
        # faithful here since there's no BYMONTHDAY/BYDAY selector that
        # could be scoped to the wrong month.
        month = _int_rrule_component(parts, "BYMONTH", anchor.month, 1, 12)
        day_of_month = _int_rrule_component(parts, "BYMONTHDAY", anchor.day, 1, 31)
        _validate_day_of_month(
            day_of_month,
            month=month,
            day_was_explicit="BYMONTHDAY" in parts,
            month_was_explicit="BYMONTH" in parts,
        )
        pattern = {
            "type": "absoluteYearly",
            "interval": interval,
            "dayOfMonth": day_of_month,
            "month": month,
        }
    else:
        raise ValueError(
            f"unsupported recurrence pattern (FREQ={freq}); this connector "
            "only translates DAILY, WEEKLY, MONTHLY (BYMONTHDAY or a "
            "numbered BYDAY), and YEARLY (BYMONTH/BYMONTHDAY or a numbered "
            "BYDAY) into an Outlook recurrence"
        )

    # Checked only once FREQ (and the rest of `pattern`) is confirmed
    # supported above - otherwise an unsupported FREQ combined with an
    # out-of-range INTERVAL would surface this less fundamental error
    # first, sending the caller on a second round-trip to find the real
    # problem.
    if interval > _GRAPH_INT32_MAX:
        raise ValueError(
            f"invalid recurrence rule: INTERVAL must be at most "
            f"{_GRAPH_INT32_MAX} for this connector, got {interval}"
        )

    if "UNTIL" in parts:
        end_date = _rrule_until_to_date(parts["UNTIL"], zone, anchor)
        if end_date < start_date:
            raise ValueError(
                "invalid recurrence rule: the computed recurrence end "
                f"date ({end_date}) is before its start date ({start_date})"
            )
        range_: dict[str, Any] = {
            "type": "endDate",
            "startDate": start_date,
            "endDate": end_date,
        }
    elif "COUNT" in parts:
        count = int(parts["COUNT"])
        if count > _GRAPH_INT32_MAX:
            raise ValueError(
                f"invalid recurrence rule: COUNT must be at most "
                f"{_GRAPH_INT32_MAX} for this connector, got {count}"
            )
        range_ = {
            "type": "numbered",
            "startDate": start_date,
            "numberOfOccurrences": count,
        }
    else:
        range_ = {"type": "noEnd", "startDate": start_date}

    # Graph otherwise defaults recurrenceTimeZone to the event's own
    # already-configured start time zone, which this function has no way
    # to know when start_datetime/timezone came from an update's fallback
    # GET rather than the caller. Stamping it explicitly to the same
    # `timezone` start_date/start_datetime were derived from keeps the
    # range self-consistent with the pattern, instead of leaving it to an
    # implicit default that may not match.
    range_["recurrenceTimeZone"] = timezone

    return {"pattern": pattern, "range": range_}
