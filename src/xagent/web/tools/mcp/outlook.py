import calendar as _stdlib_calendar
import json
import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote

import requests
from dateutil import parser as _date_parser
from dateutil import tz as _tz
from mcp.server.fastmcp import FastMCP
from tzlocal.windows_tz import win_tz as _WINDOWS_TZ_TO_IANA

from .utils import parse_rrule, setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outlook-mcp")

setup_proxy_env()

mcp = FastMCP("outlook-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False)


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"status": "error", "message": message}
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _graph_headers(extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    token = os.environ.get("AUTH_TOKEN")
    if not token:
        raise ValueError("AUTH_TOKEN environment variable is missing")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def _graph_request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Any:
    response = requests.request(
        method=method,
        url=f"{GRAPH_BASE_URL}{path}",
        headers=_graph_headers(extra_headers),
        params=params,
        json=body,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = response.text.strip()
        message = str(exc)
        if response_text:
            message = f"{message} - {response_text}"
        raise RuntimeError(message) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


def _normalize_addresses(addresses: list[str] | str) -> list[str]:
    if isinstance(addresses, str):
        return [address.strip() for address in addresses.split(",") if address.strip()]
    return [address.strip() for address in addresses if address and address.strip()]


def _recipient_list(addresses: list[str] | str) -> list[dict[str, Any]]:
    return [
        {"emailAddress": {"address": address}}
        for address in _normalize_addresses(addresses)
    ]


def _attendee_list(addresses: list[str] | str) -> list[dict[str, Any]]:
    return [
        {
            "emailAddress": {"address": address},
            "type": "required",
        }
        for address in _normalize_addresses(addresses)
    ]


def _message_body(content: str, content_type: str) -> dict[str, str]:
    normalized = content_type.strip().lower()
    if normalized not in {"text", "html"}:
        raise ValueError("content_type must be either 'text' or 'html'")
    return {"contentType": normalized, "content": content}


_RRULE_DAY_TO_GRAPH = {
    "MO": "monday",
    "TU": "tuesday",
    "WE": "wednesday",
    "TH": "thursday",
    "FR": "friday",
    "SA": "saturday",
    "SU": "sunday",
}


def _resolve_timezone(timezone: str) -> Any:
    """Resolve a timezone name to a dateutil tzinfo, trying it as an IANA
    name first (what this tool itself always writes) and falling back to
    the full CLDR Windows<->IANA mapping (what Graph often reports for
    events created by other clients, e.g. Outlook desktop/web) before
    giving up.

    A hand-written subset of this table (~18 entries) previously covered
    only common business timezones and left every other valid Windows
    zone (e.g. "Aleutian Standard Time") failing here - safely, with a
    clear error, but blocking a real recurrence-only update for any event
    whose creator used one of the ~120 unmapped zones. `tzlocal.windows_tz`
    is the same CLDR-derived table (~139 entries) `tzlocal` itself uses
    for Windows-local-timezone detection, already a transitive dependency
    via `celery` and now a direct one.
    """
    if not timezone.strip():
        raise ValueError("timezone must not be blank")
    zone = _tz.gettz(timezone)
    if zone is not None:
        return zone
    iana_name = _WINDOWS_TZ_TO_IANA.get(timezone)
    if iana_name is not None:
        zone = _tz.gettz(iana_name)
        if zone is not None:
            return zone
    raise ValueError(f"unknown timezone for recurrence rule: {timezone}")


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
    """
    try:
        parsed = _date_parser.isoparse(until.strip())
    except ValueError as exc:
        raise ValueError(f"invalid UNTIL value in recurrence rule: {until}") from exc
    if parsed.tzinfo is None:
        return parsed.date().isoformat()
    parsed = parsed.astimezone(zone)
    end_date = parsed.date()
    if parsed.time() < anchor.astimezone(zone).time():
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


_NUMBERED_BYDAY_RE = re.compile(r"^([+-]?)(\d+)?(MO|TU|WE|TH|FR|SA|SU)$")

_ORDINAL_TO_GRAPH_INDEX = {1: "first", 2: "second", 3: "third", 4: "fourth", -1: "last"}


def _parse_relative_byday(byday: str) -> tuple[str, list[str]]:
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

    More than one DISTINCT weekday sharing the same ordinal (e.g.
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
    ordinal: int | None = None
    days: list[str] = []
    seen_days: set[str] = set()
    for raw_code in byday.split(","):
        code = raw_code.strip().upper()
        match = _NUMBERED_BYDAY_RE.match(code)
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
_COMMON_RRULE_KEYS = {"FREQ", "INTERVAL", "UNTIL", "COUNT", "WKST"}
_FREQ_RECOGNIZED_KEYS = {
    "DAILY": set(),
    "WEEKLY": {"BYDAY"},
    "MONTHLY": {"BYDAY", "BYMONTHDAY"},
    "YEARLY": {"BYDAY", "BYMONTH", "BYMONTHDAY"},
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
    try:
        value = int(parts[key])
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


def _validate_day_of_month(day_of_month: int, month: int | None) -> None:
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
    if month is None:
        if day_of_month > 28:
            raise ValueError(
                "unsupported recurrence pattern: FREQ=MONTHLY with "
                f"BYMONTHDAY={day_of_month} has no exact Outlook "
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
            f"invalid recurrence rule: BYMONTHDAY={day_of_month} does not "
            f"exist in month {month}"
        )
    if day_of_month > min_guaranteed_day:
        raise ValueError(
            "unsupported recurrence pattern: FREQ=YEARLY;BYMONTH="
            f"{month};BYMONTHDAY={day_of_month} has no exact Outlook "
            "equivalent - Graph clamps to that month's last day in years "
            "where this day doesn't exist (e.g. Feb 29 in a non-leap "
            "year) instead of skipping that year's occurrence the way "
            "RFC 5545 does"
        )


def _build_graph_recurrence(
    recurrence: str, start_datetime: str, timezone: str = "UTC"
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

    start_datetime here is Outlook's own convention: a naive local
    dateTime paired with a separate timeZone field, unlike Google's
    RFC3339-with-offset. RFC 5545 requires a recurrence's UNTIL to be a
    UTC value whenever DTSTART carries a timezone reference - which this
    one does, just not embedded in the string - so it's localized with
    `timezone` before validation; passing the bare naive string through
    would make dateutil see a floating time and reject a (correct) UTC
    UNTIL as a mismatch. The same resolved zone is then reused (not
    re-looked-up) to convert UNTIL into `range.endDate`.
    """
    zone = _resolve_timezone(timezone)
    anchor = _date_parser.isoparse(start_datetime)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=zone)
    # parse_rrule already guarantees INTERVAL is a positive integer
    # whenever it's present at all, so there's nothing left to check here
    # beyond the Graph-specific Int32 bound below.
    parts = parse_rrule(recurrence, anchor)
    freq = parts["FREQ"].upper()
    interval = int(parts.get("INTERVAL", "1"))
    start_date = anchor.date().isoformat()

    if freq in _FREQ_RECOGNIZED_KEYS:
        leftover = set(parts) - _COMMON_RRULE_KEYS - _FREQ_RECOGNIZED_KEYS[freq]
        if leftover:
            raise ValueError(
                "unsupported recurrence pattern: this connector does not "
                f"translate {', '.join(sorted(leftover))} for FREQ={freq} "
                "into an Outlook recurrence"
            )

    pattern: dict[str, Any]
    if freq == "DAILY":
        pattern = {"type": "daily", "interval": interval}
    elif freq == "WEEKLY":
        byday = parts.get("BYDAY")
        if byday:
            days = []
            seen_days: set[str] = set()
            for code in byday.split(","):
                clean_code = code.strip().upper()
                if clean_code not in _RRULE_DAY_TO_GRAPH:
                    raise ValueError(f"invalid day code in BYDAY: {code!r}")
                if clean_code not in seen_days:
                    seen_days.add(clean_code)
                    days.append(_RRULE_DAY_TO_GRAPH[clean_code])
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
    elif freq == "MONTHLY" and "BYMONTHDAY" in parts and "BYDAY" in parts:
        raise ValueError(
            "unsupported recurrence pattern: FREQ=MONTHLY with both "
            'BYMONTHDAY and BYDAY (e.g. "the 15th, but only if a '
            "Tuesday\") has no equivalent in Outlook's recurrence model"
        )
    elif freq == "MONTHLY" and "BYDAY" in parts:
        index, days = _parse_relative_byday(parts["BYDAY"])
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
        _validate_day_of_month(day_of_month, month=None)
        pattern = {
            "type": "absoluteMonthly",
            "interval": interval,
            "dayOfMonth": day_of_month,
        }
    elif freq == "YEARLY" and "BYMONTHDAY" in parts and "BYDAY" in parts:
        raise ValueError(
            "unsupported recurrence pattern: FREQ=YEARLY with both "
            'BYMONTHDAY and BYDAY (e.g. "Nov 15th, but only if a '
            "Thursday\") has no equivalent in Outlook's recurrence model"
        )
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
        index, days = _parse_relative_byday(parts["BYDAY"])
        month = _int_rrule_component(parts, "BYMONTH", anchor.month, 1, 12)
        pattern = {
            "type": "relativeYearly",
            "interval": interval,
            "daysOfWeek": days,
            "index": index,
            "month": month,
        }
    elif freq == "YEARLY" and "BYMONTHDAY" in parts:
        if "BYMONTH" not in parts:
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
        month = _int_rrule_component(parts, "BYMONTH", anchor.month, 1, 12)
        day_of_month = _int_rrule_component(parts, "BYMONTHDAY", anchor.day, 1, 31)
        _validate_day_of_month(day_of_month, month=month)
        pattern = {
            "type": "absoluteYearly",
            "interval": interval,
            "dayOfMonth": day_of_month,
            "month": month,
        }
    elif freq == "YEARLY":
        # No BYMONTHDAY/BYDAY selector present (BYMONTH alone, or nothing
        # at all) - RFC 5545's own default derives the day (and, if
        # BYMONTH is also absent, the month) from DTSTART, which is
        # faithful here since there's no BYMONTHDAY/BYDAY selector that
        # could be scoped to the wrong month.
        month = _int_rrule_component(parts, "BYMONTH", anchor.month, 1, 12)
        day_of_month = _int_rrule_component(parts, "BYMONTHDAY", anchor.day, 1, 31)
        _validate_day_of_month(day_of_month, month=month)
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
            # start_date is derived from `anchor` in whatever tzinfo it
            # already carries (its own embedded offset, if any - only a
            # naive anchor gets `zone`), while end_date is derived by
            # projecting UNTIL into `zone` specifically. These only ever
            # disagree when start_datetime's own offset disagrees with
            # the separately-passed `timezone` - a documented edge case
            # this connector doesn't fully reconcile - but this specific
            # consequence (an inverted range Graph would reject outright)
            # is worse than the usual wrong-weekday symptom of that edge
            # case, so it's caught here with a clear local error instead
            # of reaching Graph as a malformed request.
            raise ValueError(
                "invalid recurrence rule: the computed recurrence end "
                f"date ({end_date}) is before its start date "
                f"({start_date}) - this usually means start_datetime's "
                "own UTC offset disagrees with the separately-passed "
                "timezone; pass a naive start_datetime (no embedded "
                "offset) so timezone alone determines it, or make sure "
                "the two agree"
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


@mcp.tool()
def outlook_get_profile() -> str:
    """Get the current Outlook/Microsoft 365 user profile."""
    try:
        me = _graph_request(
            "GET",
            "/me",
            params={
                "$select": (
                    "id,displayName,userPrincipalName,mail,givenName,surname,"
                    "jobTitle,department,mobilePhone,officeLocation"
                )
            },
        )
        return _success(user=me)
    except Exception as e:
        logger.error("Error getting Outlook profile: %s", e)
        return _error(str(e))


@mcp.tool()
def outlook_list_messages(
    top: int = 10,
    folder_id: str | None = None,
    search: str | None = None,
    select_fields: list[str] | None = None,
) -> str:
    """List Outlook email messages, optionally filtered by folder or search query."""
    try:
        top = max(1, min(top, 100))
        path = (
            f"/me/mailFolders/{quote(folder_id, safe='')}/messages"
            if folder_id
            else "/me/messages"
        )
        params: dict[str, Any] = {"$top": top, "$orderby": "receivedDateTime DESC"}
        if select_fields:
            params["$select"] = ",".join(select_fields)
        else:
            params["$select"] = (
                "id,subject,from,toRecipients,receivedDateTime,isRead,"
                "hasAttachments,importance,bodyPreview"
            )
        extra_headers = None
        if search:
            params["$search"] = f'"{search}"'
            extra_headers = {"ConsistencyLevel": "eventual"}

        result = _graph_request("GET", path, params=params, extra_headers=extra_headers)
        return _success(
            messages=result.get("value", []),
            next_link=result.get("@odata.nextLink"),
        )
    except Exception as e:
        logger.error("Error listing Outlook messages: %s", e)
        return _error(str(e))


@mcp.tool()
def outlook_get_message(
    message_id: str,
    body_type: str = "text",
) -> str:
    """Get a single Outlook message by message_id."""
    try:
        normalized_body_type = body_type.strip().lower()
        if normalized_body_type not in {"text", "html"}:
            raise ValueError("body_type must be either 'text' or 'html'")
        result = _graph_request(
            "GET",
            f"/me/messages/{quote(message_id, safe='')}",
            params={
                "$select": (
                    "id,subject,from,toRecipients,ccRecipients,bccRecipients,"
                    "receivedDateTime,sentDateTime,isRead,hasAttachments,"
                    "importance,body,bodyPreview"
                )
            },
            extra_headers={
                "Prefer": f'outlook.body-content-type="{normalized_body_type}"'
            },
        )
        return _success(message=result)
    except Exception as e:
        logger.error("Error getting Outlook message %s: %s", message_id, e)
        return _error(str(e))


@mcp.tool()
def outlook_send_message(
    to: list[str] | str,
    subject: str,
    body: str,
    cc: list[str] | str | None = None,
    bcc: list[str] | str | None = None,
    content_type: str = "text",
    save_to_sent_items: bool = True,
) -> str:
    """Send an Outlook email message."""
    try:
        if not _normalize_addresses(to):
            raise ValueError("at least one recipient is required")
        message: dict[str, Any] = {
            "subject": subject,
            "body": _message_body(body, content_type),
            "toRecipients": _recipient_list(to),
        }
        if cc:
            message["ccRecipients"] = _recipient_list(cc)
        if bcc:
            message["bccRecipients"] = _recipient_list(bcc)

        payload: dict[str, Any] = {
            "message": message,
            "saveToSentItems": save_to_sent_items,
        }

        _graph_request("POST", "/me/sendMail", body=payload)
        return _success(message="Message sent successfully")
    except Exception as e:
        logger.error("Error sending Outlook message: %s", e)
        return _error(str(e))


@mcp.tool()
def outlook_list_events(
    top: int = 20,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
) -> str:
    """List Outlook calendar events or view a time range if both datetimes are supplied."""
    try:
        top = max(1, min(top, 100))
        if start_datetime and end_datetime:
            path = "/me/calendarView"
            params = {
                "startDateTime": start_datetime,
                "endDateTime": end_datetime,
                "$top": top,
                "$orderby": "start/dateTime",
                "$select": (
                    "id,subject,start,end,location,organizer,attendees,"
                    "isAllDay,bodyPreview,webLink"
                ),
            }
        else:
            path = "/me/events"
            params = {
                "$top": top,
                "$orderby": "start/dateTime",
                "$select": (
                    "id,subject,start,end,location,organizer,attendees,"
                    "isAllDay,bodyPreview,webLink"
                ),
            }

        result = _graph_request("GET", path, params=params)
        return _success(
            events=result.get("value", []),
            next_link=result.get("@odata.nextLink"),
        )
    except Exception as e:
        logger.error("Error listing Outlook events: %s", e)
        return _error(str(e))


@mcp.tool()
def outlook_create_event(
    subject: str,
    start_datetime: str,
    end_datetime: str,
    timezone: str = "UTC",
    body: str | None = None,
    location: str | None = None,
    attendees: list[str] | str | None = None,
    is_all_day: bool = False,
    recurrence: str | None = None,
) -> str:
    """Create an Outlook calendar event.
    recurrence, if given, is a single RFC 5545 RRULE string describing a
    repeating series for this event (the "RRULE:" prefix is optional),
    e.g. 'FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z' for
    "every weekday until Sep 11, 2026". Graph has no direct RRULE input,
    so this is translated into its own structured recurrence - only
    DAILY, WEEKLY, MONTHLY, and YEARLY rules are supported. MONTHLY's
    BYMONTHDAY is optional, defaulting to the start date's own day per RFC
    5545 (or use a numbered BYDAY instead, e.g. 'FREQ=MONTHLY;BYDAY=2TU'
    for "the second Tuesday of every month"). YEARLY's BYMONTH/BYMONTHDAY
    default from the start date only when NEITHER BYMONTHDAY nor BYDAY is
    given at all; if either is given, BYMONTH must be given too (Graph's
    yearly patterns are always scoped to one specific month, unlike RFC
    5545's own BYMONTHDAY-without-BYMONTH, which spans every month);
    anything else is rejected with a clear error rather
    than silently producing the wrong pattern.
    """
    try:
        payload: dict[str, Any] = {
            "subject": subject,
            "start": {"dateTime": start_datetime, "timeZone": timezone},
            "end": {"dateTime": end_datetime, "timeZone": timezone},
            "isAllDay": is_all_day,
        }
        if body:
            payload["body"] = _message_body(body, "text")
        if location:
            payload["location"] = {"displayName": location}
        if attendees:
            payload["attendees"] = _attendee_list(attendees)
        if recurrence is not None:
            payload["recurrence"] = _build_graph_recurrence(
                recurrence, start_datetime, timezone
            )

        result = _graph_request("POST", "/me/events", body=payload)
        return _success(event=result)
    except Exception as e:
        logger.error("Error creating Outlook event: %s", e)
        return _error(str(e))


@mcp.tool()
def outlook_update_event(
    event_id: str,
    subject: str | None = None,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
    timezone: str = "UTC",
    body: str | None = None,
    location: str | None = None,
    attendees: list[str] | str | None = None,
    is_all_day: bool | None = None,
    recurrence: str | None = None,
) -> str:
    """Update an existing Outlook calendar event.
    recurrence works like it does in outlook_create_event: a single RFC
    5545 RRULE string turns this event into a repeating series, or
    replaces its existing one - there is no way to clear an existing
    recurrence back to a single event through this parameter.
    """
    try:
        payload: dict[str, Any] = {}
        if subject is not None:
            payload["subject"] = subject
        if start_datetime is not None:
            payload["start"] = {"dateTime": start_datetime, "timeZone": timezone}
        if end_datetime is not None:
            payload["end"] = {"dateTime": end_datetime, "timeZone": timezone}
        if body is not None:
            payload["body"] = _message_body(body, "text")
        if location is not None:
            payload["location"] = {"displayName": location}
        if attendees is not None:
            payload["attendees"] = _attendee_list(attendees)
        if is_all_day is not None:
            payload["isAllDay"] = is_all_day
        if recurrence is not None:
            effective_start = start_datetime
            effective_timezone = timezone
            if effective_start is None:
                # Without a `Prefer: outlook.timezone` header, Graph ALWAYS
                # reports start/end in UTC (both dateTime and
                # timeZone: "UTC"), regardless of the zone the event was
                # actually created in - so the calendar-date portion of
                # that dateTime (which BYDAY-from-start-date derives from,
                # and which becomes recurrenceRange.startDate) would
                # silently be the wrong local day whenever the event's true
                # zone differs from UTC. originalStartTimeZone instead
                # names the zone the event was really created in, so it's
                # read first and used to re-fetch start expressed in that
                # zone, rather than trusting the UTC-defaulted response.
                existing = _graph_request(
                    "GET",
                    f"/me/events/{quote(event_id, safe='')}",
                    params={"$select": "originalStartTimeZone"},
                )
                original_timezone = existing.get("originalStartTimeZone")
                if (
                    not original_timezone
                    or original_timezone == "tzone://Microsoft/Custom"
                ):
                    raise ValueError(
                        "could not determine the event's true creation "
                        "timezone (originalStartTimeZone is missing or a "
                        "legacy custom timezone Graph can't resolve by "
                        "name); pass start_datetime and timezone explicitly "
                        "to set a recurrence rule on this event"
                    )
                existing = _graph_request(
                    "GET",
                    f"/me/events/{quote(event_id, safe='')}",
                    params={"$select": "start"},
                    extra_headers={"Prefer": f'outlook.timezone="{original_timezone}"'},
                )
                existing_start_field = existing.get("start") or {}
                effective_start = existing_start_field.get("dateTime")
                existing_timezone = existing_start_field.get("timeZone")
                if not effective_start or not existing_timezone:
                    raise ValueError(
                        "could not determine the event's start time to "
                        "validate the recurrence rule; pass start_datetime "
                        "explicitly"
                    )
                # Graph's own documented no-Prefer-header default is
                # exactly `timeZone: "UTC"` - if the response still comes
                # back as UTC despite asking for a specific non-UTC zone,
                # that's a strong, specific signal the Prefer header above
                # wasn't honored (rather than merely echoed back in a
                # different-but-equivalent format), so this fails loudly
                # instead of silently reintroducing the UTC-day bug this
                # whole two-GET flow exists to fix.
                if (
                    existing_timezone == "UTC"
                    and original_timezone.strip().upper() != "UTC"
                ):
                    raise ValueError(
                        "could not re-fetch the event's start expressed in "
                        f"its own timezone ({original_timezone!r}) - Graph "
                        "returned it in UTC again as if the Prefer header "
                        "wasn't honored; pass start_datetime and timezone "
                        "explicitly to set a recurrence rule on this event"
                    )
                effective_timezone = existing_timezone
            payload["recurrence"] = _build_graph_recurrence(
                recurrence, effective_start, effective_timezone
            )

        if not payload:
            raise ValueError("at least one field must be provided to update the event")

        result = _graph_request(
            "PATCH",
            f"/me/events/{quote(event_id, safe='')}",
            body=payload,
        )
        return _success(event=result)
    except Exception as e:
        logger.error("Error updating Outlook event %s: %s", event_id, e)
        return _error(str(e))


@mcp.tool()
def outlook_delete_event(event_id: str) -> str:
    """Delete an Outlook calendar event by event_id."""
    try:
        _graph_request("DELETE", f"/me/events/{quote(event_id, safe='')}")
        return _success(message="Event deleted successfully")
    except Exception as e:
        logger.error("Error deleting Outlook event %s: %s", event_id, e)
        return _error(str(e))


@mcp.tool()
def outlook_list_contacts(top: int = 25, search: str | None = None) -> str:
    """List Outlook contacts for the current user, optionally filtered by search query."""
    try:
        top = max(1, min(top, 100))
        params: dict[str, Any] = {
            "$top": top,
            "$select": (
                "id,displayName,givenName,surname,emailAddresses,businessPhones,"
                "mobilePhone,companyName,jobTitle"
            ),
        }
        extra_headers = None
        if search:
            params["$search"] = f'"{search}"'
            extra_headers = {"ConsistencyLevel": "eventual"}
        result = _graph_request(
            "GET",
            "/me/contacts",
            params=params,
            extra_headers=extra_headers,
        )
        return _success(
            contacts=result.get("value", []),
            next_link=result.get("@odata.nextLink"),
        )
    except Exception as e:
        logger.error("Error listing Outlook contacts: %s", e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
