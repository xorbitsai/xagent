import json
import logging
import os
import uuid
from typing import Any, cast

from dateutil import parser as _date_parser
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore
from mcp.server.fastmcp import FastMCP

from .utils import (
    ensure_rrule_prefix,
    is_bare_date,
    parse_rrule,
    resolve_zoneinfo,
    setup_proxy_env,
    success_with_capped_dict,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calendar-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("calendar-mcp")


def _normalize_rrule(recurrence: str, dtstart: str, timezone: str | None = None) -> str:
    """Validate `recurrence` and return it as a single RRULE line, with
    the "RRULE:" prefix added if the caller omitted it.

    `timezone`, when given, localizes a naive `dtstart` (e.g. an RFC3339
    dateTime with no UTC offset) before validation, so it can be compared
    against a "Z"-suffixed UNTIL without dateutil rejecting the mismatch.
    """
    parse_rrule(recurrence, dtstart, timezone)
    return ensure_rrule_prefix(recurrence)


def _merge_recurrence(
    existing_recurrence: list[str] | None, new_rrule: str
) -> list[str]:
    """Build the `recurrence` list for an update: replace the RRULE line(s)
    with `new_rrule`, but keep any EXDATE/RDATE/EXRULE lines already on the
    event (e.g. a previously-cancelled single occurrence) intact.

    Google's `recurrence` field is a flat list of RRULE/EXRULE/RDATE/EXDATE
    lines, not just the RRULE - overwriting the whole list with only the
    new RRULE would silently resurrect any occurrence the user had already
    cancelled.

    `isinstance` checked before `line.strip()`, which would otherwise
    raise on a non-str element (e.g. a malformed fetched event's
    `recurrence` carrying a `None` or a nested list) - the same reason
    `_event_side` guards a non-dict "start"/"end" elsewhere in this file,
    rather than trusting Google always returns the documented shape.
    """
    preserved = [
        line
        for line in existing_recurrence or []
        if isinstance(line, str) and not line.strip().upper().startswith("RRULE:")
    ]
    return [new_rrule, *preserved]


def _has_own_utc_offset(dt_string: str) -> bool:
    """Whether an RFC3339 dateTime string carries its own explicit UTC
    offset or "Z" suffix, as opposed to a naive local time meant to be
    paired with a separate timeZone field. Used to avoid stamping a
    possibly-different reused timeZone onto a value that's already
    fully self-describing.
    """
    try:
        return _date_parser.isoparse(dt_string).tzinfo is not None
    except ValueError:
        return False


def _event_side(event: dict[str, Any], side: str) -> dict[str, Any]:
    """Return `event[side]` as a dict, or {} if it's absent or not
    actually a dict (a malformed fetched event could carry a truthy
    non-dict value there, e.g. a list) - `event.get(side) or {}` alone
    only guards the absent/falsy case, not a truthy non-dict one, and a
    subsequent `.get(...)` on it would crash with an unhelpful raw
    AttributeError.
    """
    value = event.get(side)
    return value if isinstance(value, dict) else {}


def _stamp_timezone(
    event: dict[str, Any],
    side: str,
    current_value: str | None,
    timezone: str,
    *,
    force: bool,
) -> None:
    """Write `timezone` onto ``event[side]["timeZone"]``, skipping only
    when doing so would be malformed or, for a non-recurring event,
    redundant.

    Skipped entirely when `current_value` is None: this side has no
    "dateTime"/"date" value at all (a malformed fetched event), and
    stamping a bare {"timeZone": ...} would send Google a structurally
    invalid EventDateTime carrying neither key.

    Otherwise skipped only when `force` is False and `current_value`
    already carries its own UTC offset, to avoid a possibly-disagreeing
    zone on a field Google only requires unconditionally for a
    *recurring* event (`force=True`, passed by the recurrence branch) -
    Google's own EventDateTime reference states timeZone is required for
    a recurring event regardless of any offset already in dateTime, since
    the offset fixes this occurrence's instant while timeZone governs how
    the recurrence itself expands (e.g. across DST transitions); they are
    not redundant. For a plain reschedule timeZone is merely optional, so
    it's skipped there instead when the value is already self-describing.
    """
    if current_value is None:
        return
    if not force and _has_own_utc_offset(current_value):
        return
    event[side] = _event_side(event, side)
    event[side]["timeZone"] = timezone


def _resolve_side(
    event: dict[str, Any], side: str, new_value: str | None
) -> tuple[str | None, bool]:
    """Apply `new_value` (a start_time/end_time argument) to
    ``event[side]`` if given, writing it under "date" or "dateTime"
    depending on its shape, then return `(current_value, is_all_day)` for
    that side as it now stands: `current_value` is the actual "dateTime"/
    "date" string on the event (whether just written or already there),
    and `is_all_day` is whether that value is a bare date.

    Applying this once per side (rather than separate near-identical
    start/end blocks) means a fix to this logic - e.g. the bare-date
    whitespace-stripping fix - only has to be made once to cover both.
    """
    if new_value:
        if is_bare_date(new_value):
            # Stripped so the payload always matches what is_bare_date
            # actually validated - it strips internally for the shape
            # check, but a raw new_value with incidental surrounding
            # whitespace would otherwise still reach Google verbatim.
            event[side] = {"date": new_value.strip()}
        else:
            event[side] = {"dateTime": new_value}
    current_value = _event_side(event, side).get("dateTime") or _event_side(
        event, side
    ).get("date")
    return current_value, current_value is not None and is_bare_date(current_value)


def get_calendar_service() -> Any:
    token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if not token:
        raise ValueError("GOOGLE_ACCESS_TOKEN environment variable is missing")

    creds_kwargs = {"token": token}
    if refresh_token and client_id and client_secret:
        creds_kwargs.update(
            {
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )

    credentials = Credentials(**creds_kwargs)
    return build("calendar", "v3", credentials=credentials)


@mcp.tool()
def google_calendar_search_events(
    query: str | None = None,
    time_min: str | None = None,
    time_max: str | None = None,
    max_results: int = 10,
) -> str:
    """
    Search and list Google Calendar events with optional query and label filters.
    Optionally filter by time_min and time_max (RFC3339 formatted, e.g., '2024-01-01T00:00:00Z').
    """
    try:
        service = get_calendar_service()
        kwargs = {
            "calendarId": "primary",
            "maxResults": max_results,
            "singleEvents": True,
            "orderBy": "startTime",
        }

        if time_min:
            kwargs["timeMin"] = time_min
        if time_max:
            kwargs["timeMax"] = time_max
        if query:
            kwargs["q"] = query

        events_result = service.events().list(**kwargs).execute()
        events = events_result.get("items", [])

        return json.dumps({"status": "success", "events": events})

    except Exception as e:
        logger.error(f"Error listing events: {e}")
        return json.dumps({"status": "error", "message": str(e)})


def _add_conference_request(event: dict[str, Any]) -> None:
    event["conferenceData"] = {
        "createRequest": {
            "requestId": uuid.uuid4().hex,
            "conferenceSolutionKey": {"type": "hangoutsMeet"},
        }
    }


def _merge_attendees(event: dict[str, Any], attendees: list[str] | None) -> None:
    """Add attendees to the event without dropping ones already on it.

    Matching against existing attendees (and within the new list) is
    case-/whitespace-insensitive, since `Alice@Example.com` and
    `alice@example.com` are the same mailbox and would otherwise be added
    as a redundant duplicate.
    """
    if not attendees:
        return
    existing = event.get("attendees") or []
    existing_emails = {
        email.strip().lower()
        for a in existing
        if isinstance(a, dict) and (email := a.get("email"))
    }

    seen = set()
    new_attendees = []
    for email in attendees:
        if not email:
            continue
        normalized = email.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in existing_emails or key in seen:
            continue
        seen.add(key)
        new_attendees.append({"email": normalized})

    if new_attendees:
        event["attendees"] = existing + new_attendees


def _create_request_status_code(conference_data: dict[str, Any]) -> str | None:
    """Safely read conferenceData.createRequest.status.statusCode, or None
    if any part of that path is absent or explicitly null.

    Shared by _needs_conference_request (decides whether to retry) and
    _event_response (decides what to tell the caller): this exact chain has
    already needed two separate null-safety fixes (a missing conferenceData,
    then a missing conferenceSolution.key -- an unrelated field, but caught
    by the same class of review), each landing in only one of the two
    call sites at a time because the logic lived in two places. Keeping one
    copy means a future defensive-parsing fix only has to land once.
    """
    create_request = conference_data.get("createRequest") or {}
    return (create_request.get("status") or {}).get("statusCode")


def _needs_conference_request(event: dict[str, Any]) -> bool:
    """True if add_google_meet should (re)send a createRequest.

    Checks two signals, in order. First, a legacy `hangoutLink` with no
    `conferenceData` at all (predates the conferenceData API) counts as
    already resolved -- False, so it isn't clobbered by a duplicate
    conference. Otherwise: True when the event has no conferenceData at
    all, or when the only conferenceData present is a createRequest whose
    status.statusCode is "failure". Everything else returns False, so it's
    left untouched: a still-*pending* createRequest (status.statusCode ==
    "pending" -- which, like a failed one, has no entryPoints/
    conferenceSolution yet, so those fields alone can't be used to tell
    "safe to retry" apart from "do not clobber the in-flight request"); an
    already-resolved conference (its createRequest, if it went through one
    at all, carries status.statusCode == "success", or there's no
    createRequest key at all for a conference attached some other way --
    e.g. a Zoom add-on).
    """
    if event.get("hangoutLink"):
        return False
    conference_data = event.get("conferenceData") or {}
    if not conference_data:
        return True
    return _create_request_status_code(conference_data) == "failure"


def _apply_conference_request(event: dict[str, Any], add_google_meet: bool) -> bool:
    """Mutates event in place: attaches a fresh Meet createRequest when
    add_google_meet=True and _needs_conference_request(event) says one isn't
    already in flight or resolved. A no-op (no error, no signal in the
    response) if the event already has a *non*-Meet conference (e.g. Zoom) --
    see the add_google_meet docstring on the public tools.

    Returns whether a createRequest was actually added to `event` this call
    -- the caller's raw add_google_meet argument isn't enough on its own to
    tell whether this specific request's body ended up carrying one (it may
    have been a no-op), which matters for _error_message: blaming a Meet
    conference request for a failure is only accurate when this call's body
    actually contained one.
    """
    if add_google_meet and _needs_conference_request(event):
        _add_conference_request(event)
        return True
    return False


def _api_call_kwargs(event: dict[str, Any], notify_attendees: bool) -> dict[str, Any]:
    """Build the conferenceDataVersion/sendUpdates kwargs for insert()/update().

    conferenceDataVersion is always sent as 1: it only declares that the caller
    understands conference data, it does not request or remove a conference by
    itself (that's driven by whether `body` carries a createRequest). Omitting it
    (or sending 0) makes Google ignore any conferenceData already in the body,
    which would silently drop an event's existing Meet link on every update that
    doesn't also pass add_google_meet=True.
    """
    return {
        "conferenceDataVersion": 1,
        "sendUpdates": "all" if notify_attendees and event.get("attendees") else "none",
    }


def _error_message(exc: Exception, requested_conference: bool) -> str:
    """Append an actionable hint when a request whose body actually included
    a Google Meet createRequest fails outright (as opposed to the conference
    itself merely failing to provision -- see
    _needs_conference_request/conference_status for that path). If the
    account/domain can't create Meet conferences at all, Google can reject
    the whole insert()/update() call, taking the entire event create/update
    down with it; that's indistinguishable here from any other request-level
    failure, so at least point the caller at the one thing in this request
    that's most likely to be the cause.

    requested_conference must be whether THIS call's body actually carried a
    createRequest (_apply_conference_request's return value), not the raw
    add_google_meet argument: add_google_meet=True is a no-op whenever the
    event already has a conference, and a failure can also happen before
    _apply_conference_request ever runs (auth setup, or update()'s
    preliminary event fetch) -- in both cases blaming a Meet conference
    request that was never actually part of this call would misdirect the
    caller at an unrelated failure's real cause.
    """
    message = str(exc)
    if requested_conference:
        message += (
            " (this request included a Google Meet conference request; if this"
            " Google account/domain cannot create Meet conferences, retry with"
            " add_google_meet=False)"
        )
    return message


def _event_response(event: dict[str, Any]) -> str:
    conference_data = event.get("conferenceData") or {}

    hangout_link = event.get("hangoutLink")
    if not hangout_link:
        # hangoutLink is a legacy convenience field that's only reliably
        # populated for Meet conferences. Only fall back to entryPoints when
        # the conference actually IS a Meet conference -- entryPoints with
        # entryPointType "video" is solution-agnostic, so a Zoom/Teams/other
        # third-party conference would otherwise get mislabeled as a
        # "hangout_link" (a name callers reasonably read as "this is Meet").
        solution_type = (
            (conference_data.get("conferenceSolution") or {}).get("key") or {}
        ).get("type")
        if solution_type == "hangoutsMeet":
            for entry_point in conference_data.get("entryPoints") or []:
                if entry_point.get("entryPointType") == "video" and entry_point.get(
                    "uri"
                ):
                    hangout_link = entry_point["uri"]
                    break

    extra: dict[str, Any] = {}
    if hangout_link:
        extra["hangout_link"] = hangout_link
    else:
        status_code = _create_request_status_code(conference_data)
        if status_code:
            # Meet link creation is asynchronous; surface the status instead of
            # implying failure when hangoutLink isn't populated yet. A status
            # other than "pending" (e.g. "failure") won't resolve on its own --
            # call create/update again with add_google_meet=True to retry,
            # rather than polling google_calendar_get_event for it to change.
            extra["conference_status"] = status_code

    # The event itself (description, attendees, recurrence rules, ...) can be
    # large enough to blow past the platform's output-length budget; cap it
    # the same way the rest of this package caps large single-record
    # responses, then splice the convenience fields back in afterwards so
    # capping never has to reason about them.
    response = json.loads(success_with_capped_dict("event", event))
    response.update(extra)
    return json.dumps(response)


@mcp.tool()
def google_calendar_create_events(
    summary: str,
    start_time: str,
    end_time: str,
    description: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
    notify_attendees: bool = False,
    add_google_meet: bool = False,
    timezone: str | None = None,
    recurrence: str | None = None,
) -> str:
    """
    Create a new event in Google Calendar.
    start_time and end_time must be RFC3339 formatted (e.g., '2024-01-01T10:00:00Z' or '2024-01-01T10:00:00-07:00'),
    or both a bare date (e.g. '2024-01-01') to create an all-day event -
    they must both be the same kind, never a mix.
    recurrence, if given, is a single RFC 5545 RRULE string describing a
    repeating series for this event (the "RRULE:" prefix is optional; it
    must not contain embedded newlines), e.g.
    'FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z' for "every
    weekday until Sep 11, 2026". It's validated before being sent to
    Google and rejected with a clear error if it can't be parsed, rather
    than silently landing as inert text with no actual recurrence. There
    is no way to clear a recurrence back to a single event once set,
    through this tool or google_calendar_update_events.
    timezone is an IANA timezone name (e.g. 'America/Los_Angeles'). Google
    requires it for a non-all-day recurring event specifically - and does
    so unconditionally, even when start_time/end_time already carry their
    own UTC offset, since timezone governs how the recurrence itself
    expands (e.g. across DST transitions) while the offset only fixes
    this one occurrence's instant - so it's required here whenever
    recurrence is set on one; an all-day recurring event doesn't need it
    and it has no effect at all when passed to an all-day event without
    recurrence either (Google doesn't attach a timeZone to a date-only
    event). Outside of recurrence, timezone also localizes a naive
    start_time/end_time (no UTC offset) for the purposes of comparing
    recurrence's UNTIL against them, and is only written to a side that
    doesn't already carry its own offset (there it's optional, so it's
    skipped rather than risk disagreeing with that offset).
    attendees is a list of email addresses to add to the event. Adding attendees does not, by
    itself, email them; set notify_attendees=True to have Google Calendar send them a native
    invite immediately. Confirm the recipient list with the user before setting notify_attendees=True.
    Set add_google_meet=True to attach a real Google Meet video-conference link to the event. The
    link is returned as hangout_link once Google finishes provisioning it; if it isn't ready yet the
    response includes conference_status instead (e.g. "pending") — call google_calendar_get_event
    shortly after to fetch the link. A conference_status other than "pending" (e.g. "failure") won't
    resolve on its own; call this tool again with add_google_meet=True to retry, rather than polling
    google_calendar_get_event for it to change. A plain "Google Meet" string in location does not
    create a link, and if the account/domain can't create Meet conferences at all, this whole call
    can fail outright rather than just skipping the link.
    """
    requested_conference = False
    try:
        start_is_all_day = is_bare_date(start_time)
        end_is_all_day = is_bare_date(end_time)
        if start_is_all_day != end_is_all_day:
            raise ValueError(
                "start_time and end_time must both be a bare date or both "
                "a dateTime, never a mix - a Google Calendar event's start "
                "and end must be the same kind"
            )
        if recurrence is not None and not start_is_all_day and not timezone:
            raise ValueError(
                "timezone is required when recurrence is set (Google expands "
                "a recurring event's occurrences in this timezone)"
            )
        if timezone:
            # Validate eagerly with a clean, actionable message - the only
            # other path that would otherwise catch a bad IANA name is
            # parse_rrule, which only runs when recurrence is also set,
            # leaving an invalid timezone on a non-recurring event to
            # surface as Google's own opaque server-side error instead.
            resolve_zoneinfo(timezone)

        service = get_calendar_service()

        # Stripped once here so the value written into the payload always
        # matches what is_bare_date actually validated - is_bare_date
        # strips internally for the shape check, but a raw start_time with
        # incidental surrounding whitespace would otherwise still reach
        # Google verbatim.
        start_value = start_time.strip() if start_is_all_day else start_time
        end_value = end_time.strip() if end_is_all_day else end_time
        event: dict[str, Any] = {
            "summary": summary,
            "start": (
                {"date": start_value} if start_is_all_day else {"dateTime": start_value}
            ),
            "end": {"date": end_value} if end_is_all_day else {"dateTime": end_value},
        }
        if timezone and not start_is_all_day:
            force = recurrence is not None
            _stamp_timezone(event, "start", start_time, timezone, force=force)
            _stamp_timezone(event, "end", end_time, timezone, force=force)

        if description:
            event["description"] = description
        if location:
            event["location"] = location
        if recurrence is not None:
            # An all-day event's naive date anchor still needs *some*
            # timezone to compare against an aware ("Z"-suffixed) UNTIL -
            # RFC 5545 requires DTSTART and UNTIL to either both be aware
            # or both floating, regardless of whether Google itself cares
            # about a timeZone for a date-only event. UTC is only used
            # here for that comparison; it's never written to the event.
            localization_timezone = timezone or ("UTC" if start_is_all_day else None)
            event["recurrence"] = [
                _normalize_rrule(recurrence, start_value, localization_timezone)
            ]
        _merge_attendees(event, attendees)
        requested_conference = _apply_conference_request(event, add_google_meet)

        request = service.events().insert(
            calendarId="primary",
            body=event,
            **_api_call_kwargs(event, notify_attendees),
        )
        created_event = request.execute()
        return _event_response(created_event)

    except Exception as e:
        logger.error(f"Error creating event: {e}")
        return json.dumps(
            {"status": "error", "message": _error_message(e, requested_conference)}
        )


@mcp.tool()
def google_calendar_get_event(event_id: str) -> str:
    """
    Get a specific event from Google Calendar. If the event has a Google Meet
    link, it's returned as hangout_link (or as conference_status if it's
    still being provisioned) -- this is the tool to call to pick up a link
    that wasn't ready yet right after create/update.
    """
    try:
        service = get_calendar_service()
        event = service.events().get(calendarId="primary", eventId=event_id).execute()
        return _event_response(event)
    except Exception as e:
        logger.error(f"Error getting event: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_calendar_update_events(
    event_id: str,
    summary: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    description: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
    notify_attendees: bool = False,
    add_google_meet: bool = False,
    timezone: str | None = None,
    recurrence: str | None = None,
) -> str:
    """
    Update an existing event in Google Calendar.
    start_time and end_time must be RFC3339 formatted if provided, or
    both a bare date (e.g. '2024-01-01') to convert the event to/from
    all-day - if only one of the two is given, it must match the kind
    (dateTime vs. bare date) the event already has on its other side.
    recurrence works like it does in google_calendar_create_events: a
    single RFC 5545 RRULE string turns this event into a repeating series,
    or replaces its existing one. Any EXDATE/RDATE lines already on the
    event (e.g. a previously-cancelled single occurrence) are kept -
    there is no way to clear an existing recurrence back to a single
    event through this parameter. event_id must be the series' own id,
    not one of its individual occurrences - google_calendar_search_events
    lists occurrences (each carrying a recurringEventId pointing at the
    actual series), and passing one of those here with recurrence set is
    rejected with a clear error rather than silently targeting the wrong
    resource.
    timezone is an IANA timezone name; Google requires one for a
    non-all-day recurring event - and does so unconditionally, even when
    start_time/end_time (or the event's existing values) already carry
    their own UTC offset. If recurrence is set and timezone is omitted,
    each of the event's start/end reuses its own existing timeZone (they
    can legitimately differ, e.g. a flight); if a side has none at all, it
    falls back to the other side's rather than being left without one, and
    if neither side has one either, this call is rejected with a clear
    error rather than sending an incomplete request to Google. Passing
    timezone without also passing start_time/end_time reinterprets the
    event's existing wall-clock time under the new zone rather than
    converting the underlying instant - e.g. an event stored as
    07:00/Asia/Manila becomes 07:00/America/New_York, a ~12h shift in
    absolute time, not a same-instant re-labeling; pass start_time/
    end_time as well to actually convert the instant. Exception: for a
    recurring event whose stored value already carries its own UTC
    offset, timezone is stamped alongside that untouched offset rather
    than reinterpreting anything (Google requires timeZone unconditionally
    for a recurring event regardless of any offset already present), so
    no wall-clock shift happens in that specific case. timezone is
    otherwise optional and, outside of recurrence, is skipped for a side
    that already carries its own offset (there it's merely optional, so
    it's skipped rather than risk disagreeing with that offset) and has
    no effect at all on an all-day event.
    attendees is a list of email addresses to add to the event; attendees already on the event are
    kept, and there is no way to remove an attendee through this parameter. Adding attendees does
    not, by itself, email anyone; set notify_attendees=True to have Google Calendar send a native
    invite/update immediately to every attendee on the event (existing and newly added). Confirm the
    recipient list with the user before setting notify_attendees=True.
    Set add_google_meet=True to attach a real Google Meet video-conference link to the event if it
    doesn't already have a conference (an existing conference is left untouched, whether it's a Meet
    conference or a non-Google one such as Zoom -- in the latter case this is a silent no-op, with no
    hangout_link/conference_status in the response, since the event already has a conference). The
    link is returned as hangout_link once Google finishes provisioning it; if it isn't ready yet the
    response includes conference_status instead (e.g. "pending") — call google_calendar_get_event
    shortly after to fetch the link. A conference_status other than "pending" (e.g. "failure") won't
    resolve on its own; call this tool again with add_google_meet=True to retry, rather than polling
    google_calendar_get_event for it to change. A plain "Google Meet" string in location does not
    create a link, and if the account/domain can't create Meet conferences at all, this whole call
    can fail outright rather than just skipping the link.
    """
    requested_conference = False
    try:
        if timezone:
            # Validate eagerly with a clean, actionable message - the
            # only other path that would otherwise catch a bad IANA name
            # is parse_rrule, which only runs when recurrence is also
            # set, leaving an invalid timezone on a plain reschedule to
            # surface as Google's own opaque server-side error instead.
            resolve_zoneinfo(timezone)

        service = get_calendar_service()

        # First get the existing event
        event = service.events().get(calendarId="primary", eventId=event_id).execute()
        if recurrence is not None and event.get("recurringEventId"):
            # A single occurrence of a recurring series (e.g. one row
            # from google_calendar_search_events, which lists occurrences
            # with singleEvents=True) carries recurringEventId pointing
            # at the master event - it isn't itself a series, so setting
            # recurrence here targets the wrong resource rather than
            # actually changing the series. Reject rather than send
            # Google a request whose effect wouldn't match the caller's
            # intent.
            raise ValueError(
                f"event {event_id!r} is a single occurrence of a recurring "
                f"series (recurringEventId={event['recurringEventId']!r}), not "
                "the series itself - call this on the series' master event id "
                "to change its recurrence"
            )
        # Captured before any of the reassignments below can overwrite
        # event["start"]/["end"] - events().update() replaces the whole
        # resource, so wholesale-replacing "start"/"end" with a bare
        # {"dateTime": ...} below would otherwise silently strip the
        # event's existing timeZone (not just for the recurrence fallback
        # further down, but for a plain reschedule with no recurrence
        # involved at all).
        existing_start_timezone = _event_side(event, "start").get("timeZone")
        existing_end_timezone = _event_side(event, "end").get("timeZone")
        # Also captured before any reassignment - whether the event was
        # all-day or timed *before* this call, so a value-type change can
        # be detected below even though _resolve_side below overwrites
        # event["start"] in place.
        original_start_value = _event_side(event, "start").get(
            "dateTime"
        ) or _event_side(event, "start").get("date")
        original_is_all_day = original_start_value is not None and is_bare_date(
            original_start_value
        )
        # Whether this event is (or, after this call, remains) a
        # recurring series - Google requires timeZone unconditionally for
        # one, regardless of this call's own recurrence argument: leaving
        # recurrence unset on an update doesn't clear it, so a plain
        # reschedule of an already-recurring event still needs a
        # correct timeZone on both sides.
        event_is_recurring = recurrence is not None or bool(event.get("recurrence"))

        if summary:
            event["summary"] = summary
        # An all-day event's start is a bare "date" (no "T"), never a
        # "dateTime" - Google doesn't attach (or require) a timeZone to a
        # whole-day occurrence. Determined from the CURRENT value (after
        # any start_time/end_time reassignment, which now correctly
        # writes "date" for a bare-date value), so moving an all-day
        # event to a specific dateTime in this same call correctly stops
        # treating it as all-day.
        current_start_value, is_all_day = _resolve_side(event, "start", start_time)
        # Google requires start and end to be the SAME kind (both a bare
        # "date" or both a "dateTime") - never a mix. Passing only one of
        # start_time/end_time to convert an all-day event to a timed one
        # (or vice versa) would otherwise silently leave the other side in
        # its old shape, producing a payload Google's API would reject.
        current_end_value, end_is_all_day = _resolve_side(event, "end", end_time)
        if (
            current_start_value is not None
            and current_end_value is not None
            and is_all_day != end_is_all_day
        ):
            raise ValueError(
                "start_time and end_time must both be provided together "
                "when converting between an all-day event and a timed "
                "event; a Google Calendar event's start and end must both "
                "be a bare date or both a dateTime, never a mix"
            )
        if recurrence is not None and not current_start_value:
            raise ValueError(
                "could not determine the event's start time or date to "
                "validate the recurrence rule; pass start_time explicitly"
            )

        # Single source of truth for what timeZone (if any) belongs on
        # each side of this event. Previously computed independently in
        # three separate places (a per-field "reuse existing" fallback on
        # start_time/end_time reassignment, an explicit-timezone stamp,
        # and the recurrence branch's own fallback) - between them they
        # missed real combinations (a plain reschedule of an
        # already-recurring event with no existing zone at all sent no
        # timeZone whatsoever). Computed per side, not one shared value:
        # start and end can legitimately have different timeZones (e.g. a
        # flight), so falling back to *start's* zone for *end* too would
        # silently clobber a genuinely-different existing end zone - each
        # side's own zone (explicit override, else its own existing
        # timeZone) only falls back to the *other* side's own zone as a
        # last resort, when it has none of its own at all (symmetric:
        # applies in both directions).
        own_start_timezone = timezone or existing_start_timezone
        own_end_timezone = timezone or existing_end_timezone
        effective_start_timezone = own_start_timezone or own_end_timezone
        effective_end_timezone = own_end_timezone or own_start_timezone

        if (
            event_is_recurring
            and not is_all_day
            and (recurrence is not None or start_time or end_time)
        ):
            # Google requires timeZone unconditionally for a recurring
            # event, regardless of any offset already in
            # current_start_value/current_end_value - the offset fixes
            # this occurrence's instant, timeZone governs how the
            # recurrence itself expands (e.g. across DST transitions).
            # This applies whenever the event already has a recurrence,
            # not just when this call's own recurrence argument sets one
            # - leaving recurrence unset on an update doesn't clear it.
            # Gated on this call actually touching recurrence/start_time/
            # end_time, not on every update to an already-recurring event
            # - a metadata-only change (summary, attendees, ...) doesn't
            # rewrite start/end at all, so it shouldn't be blocked by a
            # zone the *existing* (untouched) data happens to be missing.
            if not effective_start_timezone:
                raise ValueError(
                    "timezone is required to set a recurrence rule on this "
                    "event (Google expands a recurring event's occurrences "
                    "in this timezone, and the event doesn't already have one)"
                )
            # A side just moved to a value that already carries its own
            # UTC offset can't safely reuse the event's existing zone
            # instead - that offset may no longer match it (e.g. moving a
            # Shanghai-zoned series to a date given in Pacific time). Ask
            # for an explicit timezone rather than silently guessing
            # wrong in either direction.
            if (
                start_time
                and not timezone
                and current_start_value is not None
                and _has_own_utc_offset(current_start_value)
            ):
                raise ValueError(
                    "start_time was moved to a value with its own UTC "
                    "offset on a recurring event, but timezone wasn't "
                    "given - the event's existing timezone can't be "
                    "safely assumed to still match; pass timezone "
                    "explicitly"
                )
            if (
                end_time
                and not timezone
                and current_end_value is not None
                and _has_own_utc_offset(current_end_value)
            ):
                raise ValueError(
                    "end_time was moved to a value with its own UTC "
                    "offset on a recurring event, but timezone wasn't "
                    "given - the event's existing timezone can't be "
                    "safely assumed to still match; pass timezone "
                    "explicitly"
                )
            # cast(), not a runtime check: the raise above already
            # guarantees effective_start_timezone is set here, and
            # effective_end_timezone always falls back to it.
            _stamp_timezone(
                event,
                "start",
                current_start_value,
                cast(str, effective_start_timezone),
                force=True,
            )
            _stamp_timezone(
                event,
                "end",
                current_end_value,
                cast(str, effective_end_timezone),
                force=True,
            )
        elif timezone and not is_all_day:
            # Either not recurring, or recurring but this call doesn't
            # touch recurrence/start_time/end_time: timeZone is merely
            # optional here, so a side already carrying its own offset is
            # skipped instead of risking a disagreeing zone (see
            # _stamp_timezone).
            _stamp_timezone(event, "start", current_start_value, timezone, force=False)
            _stamp_timezone(event, "end", current_end_value, timezone, force=False)
        elif not is_all_day:
            # Same as above but with no explicit timezone either:
            # opportunistically reuse each side's own existing zone
            # (again skipped for a side already carrying its own offset).
            if existing_start_timezone:
                _stamp_timezone(
                    event,
                    "start",
                    current_start_value,
                    existing_start_timezone,
                    force=False,
                )
            if existing_end_timezone:
                _stamp_timezone(
                    event, "end", current_end_value, existing_end_timezone, force=False
                )

        if description:
            event["description"] = description
        if location:
            event["location"] = location
        if recurrence is not None:
            existing_recurrence = event.get("recurrence")
            if original_is_all_day != is_all_day and any(
                isinstance(line, str) and not line.strip().upper().startswith("RRULE:")
                for line in existing_recurrence or []
            ):
                # An existing EXDATE/RDATE line is typed to match the
                # event's OLD all-day-vs-timed kind (a DATE-TIME value for
                # a timed series, a bare-date value for an all-day one).
                # Converting the event's kind in this same call would
                # carry that now-mismatched line forward verbatim -
                # _merge_recurrence only ever replaces the RRULE line(s),
                # never revalidates or converts the rest - producing an
                # RFC 5545 value-type mismatch between DTSTART and
                # EXDATE/RDATE with no error. Converting the value type
                # and replacing recurrence safely can't both happen in
                # one call without knowing how to re-derive each
                # preserved line under the new kind; reject rather than
                # silently send Google an inconsistent recurrence.
                raise ValueError(
                    "cannot convert this event between all-day and timed "
                    "while also replacing its recurrence rule: the "
                    "existing EXDATE/RDATE line(s) are typed for the "
                    "event's current kind and would no longer match after "
                    "the conversion - change the event's kind in a "
                    "separate call first, then set recurrence"
                )
            # An all-day event's naive date anchor still needs *some*
            # timezone to compare against an aware ("Z"-suffixed) UNTIL -
            # RFC 5545 requires DTSTART and UNTIL to either both be aware or
            # both be floating, regardless of whether Google itself cares
            # about a timeZone for a date-only event. UTC is only used here
            # for that comparison; it's never written to the event.
            localization_timezone = effective_start_timezone or (
                "UTC" if is_all_day else None
            )
            # cast(): the earlier "could not determine the event's start
            # time" check already guarantees current_start_value is set
            # whenever recurrence is not None.
            new_rrule = _normalize_rrule(
                recurrence, cast(str, current_start_value), localization_timezone
            )
            event["recurrence"] = _merge_recurrence(existing_recurrence, new_rrule)
        _merge_attendees(event, attendees)
        requested_conference = _apply_conference_request(event, add_google_meet)

        request = service.events().update(
            calendarId="primary",
            eventId=event_id,
            body=event,
            **_api_call_kwargs(event, notify_attendees),
        )
        updated_event = request.execute()
        return _event_response(updated_event)

    except Exception as e:
        logger.error(f"Error updating event: {e}")
        return json.dumps(
            {"status": "error", "message": _error_message(e, requested_conference)}
        )


@mcp.tool()
def google_calendar_delete_events(event_id: str) -> str:
    """
    Delete an existing event in Google Calendar.
    """
    try:
        service = get_calendar_service()
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return json.dumps(
            {"status": "success", "message": "Event deleted successfully"}
        )
    except Exception as e:
        logger.error(f"Error deleting event: {e}")
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    mcp.run()
