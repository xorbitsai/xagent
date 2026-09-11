import json
import logging
import os
import uuid
from datetime import date
from typing import Any

from dateutil import parser as _date_parser
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore
from mcp.server.fastmcp import FastMCP

from .utils import (
    ensure_rrule_prefix,
    is_bare_date,
    parse_rrule,
    reject_reversed_window,
    resolve_zoneinfo,
    setup_proxy_env,
    success_with_capped_dict,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calendar-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("calendar-mcp")

_GOOGLE_SUPPORTED_RRULE_FREQUENCIES = frozenset(
    {"DAILY", "WEEKLY", "MONTHLY", "YEARLY"}
)


def _normalize_rrule(recurrence: str, dtstart: str, timezone: str | None = None) -> str:
    """Validate `recurrence` and return it as a single RRULE line, with
    the "RRULE:" prefix added if the caller omitted it.

    `timezone`, when given, localizes a naive `dtstart` (e.g. an RFC3339
    dateTime with no UTC offset) before validation, so it can be compared
    against a "Z"-suffixed UNTIL without dateutil rejecting the mismatch.
    """
    parts = parse_rrule(recurrence, dtstart, timezone)
    if parts["FREQ"] not in _GOOGLE_SUPPORTED_RRULE_FREQUENCIES:
        supported = ", ".join(sorted(_GOOGLE_SUPPORTED_RRULE_FREQUENCIES))
        raise ValueError(
            f"Google Calendar does not support FREQ={parts['FREQ']}; "
            f"use one of {supported}"
        )
    return ensure_rrule_prefix(recurrence)


def _classify_event_time(value: str, field_name: str) -> bool:
    """Validate an event time and return whether it is an all-day date."""
    if is_bare_date(value):
        return True
    try:
        _date_parser.isoparse(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a valid YYYY-MM-DD date or RFC3339 dateTime"
        ) from exc
    if (
        len(value) < 11
        or value[4] != "-"
        or value[7] != "-"
        or value[10] not in {"T", "t"}
    ):
        raise ValueError(
            f"{field_name} must be a valid YYYY-MM-DD date or RFC3339 dateTime"
        )
    return False


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
    or both a bare date (e.g. '2024-01-01') to create an all-day event.
    For an all-day event, end_time is exclusive and must be later than
    start_time: use the following date for a one-day event. Both values
    must be the same kind, never a mix.
    recurrence, if given, is one RFC 5545 RRULE with DAILY, WEEKLY,
    MONTHLY, or YEARLY frequency; the "RRULE:" prefix is optional and
    embedded newlines are rejected. For example,
    'FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z'. An all-day
    event must use a bare-date UNTIL such as UNTIL=20260911; a timed event
    with timezone must use a UTC UNTIL ending in Z.
    timezone is an IANA name such as 'America/Los_Angeles' and is required
    for timed recurring events, including when start_time/end_time carry
    their own offsets. It is ignored for all-day events.
    Do not use google_calendar_update_events to reschedule an all-day or
    recurring event yet; that tool cannot preserve their date/timeZone
    fields. Metadata-only updates remain safe.
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
        # Normalize both accepted input shapes before classifying or validating
        # them. Google expects exact RFC3339/date values and rejects otherwise
        # valid values that carry incidental surrounding whitespace.
        start_time = start_time.strip()
        end_time = end_time.strip()
        if timezone is not None:
            timezone = timezone.strip() or None
        start_is_all_day = _classify_event_time(start_time, "start_time")
        end_is_all_day = _classify_event_time(end_time, "end_time")
        if start_is_all_day != end_is_all_day:
            raise ValueError(
                "start_time and end_time must both be a bare date or both "
                "a dateTime, never a mix - a Google Calendar event's start "
                "and end must be the same kind"
            )
        if start_is_all_day and date.fromisoformat(end_time) <= date.fromisoformat(
            start_time
        ):
            raise ValueError(
                "end_time is exclusive for an all-day event and must be later "
                "than start_time; use the following date for a one-day event"
            )
        if not start_is_all_day:
            reject_reversed_window(start_time, end_time)
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

        normalized_recurrence = (
            _normalize_rrule(recurrence, start_time, timezone)
            if recurrence is not None
            else None
        )

        service = get_calendar_service()

        event: dict[str, Any] = {
            "summary": summary,
            "start": (
                {"date": start_time} if start_is_all_day else {"dateTime": start_time}
            ),
            "end": {"date": end_time} if end_is_all_day else {"dateTime": end_time},
        }
        if timezone and not start_is_all_day:
            force = recurrence is not None
            _stamp_timezone(event, "start", start_time, timezone, force=force)
            _stamp_timezone(event, "end", end_time, timezone, force=force)

        if description:
            event["description"] = description
        if location:
            event["location"] = location
        if normalized_recurrence is not None:
            event["recurrence"] = [normalized_recurrence]
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
) -> str:
    """
    Update an existing event in Google Calendar.
    start_time and end_time must be RFC3339 formatted if provided.
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
        service = get_calendar_service()

        # First get the existing event
        event = service.events().get(calendarId="primary", eventId=event_id).execute()

        if summary:
            event["summary"] = summary
        if start_time:
            event["start"] = {"dateTime": start_time}
        if end_time:
            event["end"] = {"dateTime": end_time}
        if description:
            event["description"] = description
        if location:
            event["location"] = location
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
