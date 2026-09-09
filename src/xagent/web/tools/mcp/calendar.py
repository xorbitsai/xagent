import json
import logging
import os
import uuid
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore
from googleapiclient.errors import HttpError  # type: ignore
from mcp.server.fastmcp import FastMCP

from .utils import attendees_needing_check as _attendees_needing_check
from .utils import attendees_to_add as _attendees_to_add
from .utils import calendar_day_bounds as _calendar_day_bounds
from .utils import conflict_response as _conflict_response
from .utils import datetime_key_for_comparison as _datetime_key_for_comparison
from .utils import normalize_addresses as _normalize_addresses
from .utils import reject_reversed_window as _reject_reversed_window
from .utils import setup_proxy_env, success_with_capped_dict
from .utils import unchecked_extra as _unchecked_extra
from .utils import windows_overlap as _windows_overlap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calendar-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("calendar-mcp")

# freebusy.query accepts at most this many calendars per call
# (Google's calendarExpansionMax).
_MAX_ATTENDEES_PER_FREEBUSY_QUERY = 50


def _is_insufficient_scope_error(exc: Any) -> bool:
    """Whether `exc` is specifically "the access token doesn't have the
    scope this call needs", not just any 403.

    Google's error body can carry this reason on either of two fields,
    and both show up in practice: the legacy `error.errors[].reason ==
    "insufficientPermissions"`, and a newer `error.details[].reason ==
    "ACCESS_TOKEN_SCOPE_INSUFFICIENT"` (an ErrorInfo entry) that a real
    Calendar API 403 for this exact case carries ALONGSIDE the legacy
    one, not instead of it. `HttpError._get_reason` picks one field per
    a fixed priority order (`detail` > `details` > `errors` > `message`)
    to build `exc.error_details`/`str(exc)` - when both are present,
    `details` wins and the legacy reason string is dropped entirely from
    the formatted message. So a plain substring check on `str(exc)` can
    silently stop matching real scope errors the moment Google's backend
    starts including both. Check the parsed body directly instead of
    trusting which one HttpError's own formatting happens to surface.
    """
    try:
        data = json.loads(exc.content.decode("utf-8"))
    except (ValueError, AttributeError):
        return False
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return False
    for field, reason in (
        ("errors", "insufficientPermissions"),
        ("details", "ACCESS_TOKEN_SCOPE_INSUFFICIENT"),
    ):
        entries = error.get(field)
        if not isinstance(entries, list):
            # A real Google error body always shapes these as arrays -
            # this is defensive against a malformed/unexpected body,
            # which is exactly the "can't tell" case this function
            # already treats as False elsewhere (e.g. non-JSON content).
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("reason") == reason:
                return True
    return False


def _find_conflicts(
    service: Any,
    time_min: str,
    time_max: str,
    attendees: list[str],
    *,
    exclude_event_id: str | None = None,
    check_organizer: bool = True,
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    """Check the organizer's own calendar (when check_organizer) plus each
    attendee's free/busy for anything overlapping [time_min, time_max).

    Delegates the actual interval-overlap comparison to the Google Calendar
    API's timeMin/timeMax semantics rather than parsing/normalizing the
    timestamps locally, so this can't reproduce a timezone-comparison bug.

    Free/busy has no concept of "exclude this event": it only returns raw
    busy time ranges, so a query against a window an attendee is *already*
    busy for (because that's the very event being updated) can't be told
    apart from a genuine conflict here. Callers that aren't moving the event
    to a new window must restrict `attendees` to only the newly-added ones
    (see google_calendar_update_events) rather than relying on this function
    to exclude the event's own footprint on attendee calendars.

    Returns (conflicts, unchecked_attendees, unchecked_reason). The third
    element is set only when attendees ended up unchecked because of a
    missing-scope 403 - the one unchecked case an LLM caller can actually
    act on (ask the user to reconnect the connector) - not for the other,
    self-explanatory unchecked cases (absent from the response, or a
    per-attendee error entry).
    """
    conflicts: list[dict[str, Any]] = []
    unchecked_attendees: list[str] = []
    unchecked_reason: str | None = None

    if check_organizer:
        organizer_events = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
            .get("items")
            or []
        )
        for item in organizer_events:
            if item.get("status") == "cancelled":
                continue
            if item.get("transparency") == "transparent":
                continue
            if exclude_event_id and item.get("id") == exclude_event_id:
                continue
            # An event the organizer personally declined still sits on
            # their calendar (declining doesn't clear transparency), but
            # it's not something they're actually busy for.
            self_response = next(
                (
                    a.get("responseStatus")
                    for a in item.get("attendees") or []
                    if a.get("self")
                ),
                None,
            )
            if self_response == "declined":
                continue
            start = item.get("start") or {}
            end = item.get("end") or {}
            conflicts.append(
                {
                    "calendar": "organizer",
                    "summary": item.get("summary") or "(no title)",
                    "start": start.get("dateTime") or start.get("date"),
                    "end": end.get("dateTime") or end.get("date"),
                }
            )

    # freebusy.query caps the number of calendars per call
    # (Google's calendarExpansionMax) - chunk rather than giving up on the
    # whole batch, so a large invite list still gets everyone it can check
    # only. `range(0, 0, N)` is empty, so this is also just a no-op loop
    # when `attendees` is empty.
    for offset in range(0, len(attendees), _MAX_ATTENDEES_PER_FREEBUSY_QUERY):
        batch = attendees[offset : offset + _MAX_ATTENDEES_PER_FREEBUSY_QUERY]
        try:
            freebusy = (
                service.freebusy()
                .query(
                    body={
                        "timeMin": time_min,
                        "timeMax": time_max,
                        "items": [{"id": email} for email in batch],
                    }
                )
                .execute()
            )
        except HttpError as exc:
            if exc.resp.status == 403 and _is_insufficient_scope_error(exc):
                # This connection's OAuth token predates the
                # calendar.freebusy scope (see the
                # 20260907_add_calendar_freebusy_scope migration) and
                # hasn't been reconnected yet. Don't abort the whole
                # booking over an availability check the token can't
                # run - report this batch as unchecked instead.
                unchecked_attendees.extend(batch)
                unchecked_reason = (
                    "Missing the calendar.freebusy permission needed to "
                    "check attendee availability - reconnect the Google "
                    "Calendar connector to grant it."
                )
                continue
            raise

        # Google's own calendarList entries are case-normalized; be
        # defensive in case freebusy echoes calendar ids back
        # differently from how the caller supplied them, rather than
        # silently dropping a real conflict/error for that attendee.
        calendars = {
            key.lower(): value
            for key, value in (freebusy.get("calendars") or {}).items()
        }
        for email in batch:
            if email.lower() not in calendars:
                # Not even present in the response - can't tell
                # whether they're free, so don't silently report them
                # as clear.
                unchecked_attendees.append(email)
                continue
            info = calendars[email.lower()] or {}
            if info.get("errors"):
                unchecked_attendees.append(email)
                continue
            for busy in info.get("busy") or []:
                conflicts.append(
                    {
                        "calendar": email,
                        "summary": None,
                        "start": busy.get("start"),
                        "end": busy.get("end"),
                    }
                )

    return conflicts, unchecked_attendees, unchecked_reason


def _primary_calendar_timezone(service: Any) -> str:
    """The primary calendar's own configured IANA timezone (falls back to
    UTC if somehow absent - Google's Calendar resource always carries a
    timeZone, so this is a defensive last resort, not the expected path)."""
    return (
        service.calendars().get(calendarId="primary").execute().get("timeZone") or "UTC"
    )


def _event_boundary(field: dict[str, Any] | None, tz_name: str) -> str | None:
    """Return a RFC3339 timestamp usable as a freebusy/events.list time
    bound for one side (start or end) of an event.

    Google represents a timed event's boundary as {"dateTime": ...} (which
    already carries its own zone offset - used as-is) and an all-day
    event's as {"date": "YYYY-MM-DD"} - timeMin/timeMax require a full
    RFC3339 timestamp with a zone offset, so a bare date needs widening.

    An all-day event's date is a day on the *calendar's own* calendar, not
    a UTC day - `tz_name` must be that calendar's configured timezone, not
    a hardcoded UTC, or the widened boundary is off by the calendar's own
    UTC offset. All-day events already store both start.date and end.date
    as the correct (exclusive-end) calendar-day range, so no day-arithmetic
    is needed here - just giving each date its own midnight in tz_name.
    """
    field = field or {}
    date_time: str | None = field.get("dateTime")
    if date_time:
        return date_time
    date_value: str | None = field.get("date")
    if date_value:
        start, _ = _calendar_day_bounds(date_value, tz_name)
        return start
    return None


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


def _event_response(event: dict[str, Any], **extra_fields: Any) -> str:
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
    response.update(extra_fields)
    return json.dumps(response, ensure_ascii=False)


@mcp.tool()
def google_calendar_create_events(
    summary: str,
    start_time: str,
    end_time: str,
    description: str | None = None,
    location: str | None = None,
    attendees: list[str] | str | None = None,
    notify_attendees: bool = False,
    add_google_meet: bool = False,
    ignore_conflicts: bool = False,
) -> str:
    """
    Create a new event in Google Calendar.
    start_time and end_time must be RFC3339 formatted (e.g., '2024-01-01T10:00:00Z' or '2024-01-01T10:00:00-07:00').
    attendees, if given, are checked for scheduling conflicts together with the organizer's own
    calendar; a conflict returns status="conflict" instead of creating the event. Pass
    ignore_conflicts=True to create it anyway once the user has explicitly confirmed a conflict is
    fine. Adding attendees does not, by itself, email them; set notify_attendees=True to have Google
    Calendar send them a native invite immediately. Confirm the recipient list with the user before
    setting notify_attendees=True.
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
        _reject_reversed_window(start_time, end_time)
        service = get_calendar_service()
        normalized_attendees = _normalize_addresses(attendees) if attendees else []

        unchecked_attendees: list[str] = []
        unchecked_reason: str | None = None
        if not ignore_conflicts:
            conflicts, unchecked_attendees, unchecked_reason = _find_conflicts(
                service, start_time, end_time, normalized_attendees
            )
            if conflicts:
                return _conflict_response(
                    conflicts,
                    unchecked_attendees,
                    start_time,
                    end_time,
                    unchecked_reason=unchecked_reason,
                )

        event: dict[str, Any] = {
            "summary": summary,
            "start": {
                "dateTime": start_time,
            },
            "end": {
                "dateTime": end_time,
            },
        }

        if description:
            event["description"] = description
        if location:
            event["location"] = location
        if normalized_attendees:
            event["attendees"] = [
                {"email": address} for address in normalized_attendees
            ]
        requested_conference = _apply_conference_request(event, add_google_meet)

        request = service.events().insert(
            calendarId="primary",
            body=event,
            **_api_call_kwargs(event, notify_attendees),
        )
        created_event = request.execute()
        return _event_response(
            created_event, **_unchecked_extra(unchecked_attendees, unchecked_reason)
        )

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
    attendees: list[str] | str | None = None,
    notify_attendees: bool = False,
    add_google_meet: bool = False,
    ignore_conflicts: bool = False,
) -> str:
    """
    Update an existing event in Google Calendar.
    start_time and end_time must be RFC3339 formatted if provided.
    If the update moves the event to a new time, or adds attendees, that change is checked for
    conflicts the same way google_calendar_create_events is; pass ignore_conflicts=True to skip the
    check once the user has explicitly confirmed a conflict is fine. Editing other fields (summary,
    description, location) without moving the event is never blocked.
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

        existing_is_all_day = "date" in (event.get("start") or {}) or "date" in (
            event.get("end") or {}
        )
        if existing_is_all_day and bool(start_time) != bool(end_time):
            # The existing event's start/end are {"date": ...}; writing
            # only one of start_time/end_time would overwrite just that
            # side with {"dateTime": ...} while the other side keeps its
            # {"date": ...} shape - Google rejects a body mixing the two
            # formats between an event's start and end.
            raise ValueError(
                "Existing event is all-day (uses date-only start/end); "
                "updating only one of start_time/end_time would mix "
                "date-only and dateTime formats. Pass both start_time and "
                "end_time together."
            )

        # The calendar's own timezone is only needed to widen an all-day
        # boundary (see _event_boundary) - fetch it lazily so a plain
        # timed-event update doesn't pay for an extra API call it has no
        # use for.
        calendar_timezone = (
            _primary_calendar_timezone(service) if existing_is_all_day else "UTC"
        )
        existing_start = _event_boundary(event.get("start"), calendar_timezone)
        existing_end = _event_boundary(event.get("end"), calendar_timezone)
        effective_start = start_time or existing_start
        effective_end = end_time or existing_end
        if effective_start and effective_end:
            _reject_reversed_window(effective_start, effective_end)
        existing_start_key = _datetime_key_for_comparison(existing_start)
        existing_end_key = _datetime_key_for_comparison(existing_end)
        effective_start_key = _datetime_key_for_comparison(effective_start)
        effective_end_key = _datetime_key_for_comparison(effective_end)
        window_changed = (effective_start_key, effective_end_key) != (
            existing_start_key,
            existing_end_key,
        )

        existing_attendees_raw = [
            a["email"]
            for a in (event.get("attendees") or [])
            if isinstance(a, dict) and a.get("email")
        ]
        existing_attendee_emails = {
            address.lower() for address in existing_attendees_raw
        }
        # attendees only ever ADDS: there's no way to remove an existing
        # attendee through this parameter (matching this tool's own
        # docstring), so the effective set for conflict-checking purposes
        # is always existing-plus-newly-added, never a caller-supplied
        # subset that could silently drop someone. `added_attendees` (the
        # ones actually new) is what the write step appends below.
        added_attendees = _attendees_to_add(attendees, existing_attendee_emails)
        # Preserve the casing already on the event - only the membership
        # check below needs to be case-insensitive, not what gets
        # queried/reported back to the caller.
        normalized_attendees = sorted(existing_attendees_raw + added_attendees)

        unchecked_attendees: list[str] = []
        unchecked_reason: str | None = None
        if not ignore_conflicts and effective_start and effective_end:
            # A partial nudge that still overlaps the old window is just
            # as unsafe to free/busy-check existing attendees against as
            # the unchanged-window case: the overlap still contains this
            # event's own busy block on their calendars, which isn't a
            # real conflict. Only a window that's moved to somewhere
            # completely disjoint from the old one is safe to check every
            # attendee against. The organizer side has no such problem
            # (excluded by event id, not by window), so it only needs
            # `window_changed`, not the stricter disjoint test.
            overlapping = _windows_overlap(
                existing_start_key,
                existing_end_key,
                effective_start_key,
                effective_end_key,
            )
            attendees_to_check = _attendees_needing_check(
                normalized_attendees,
                existing_attendee_emails,
                moved_to_a_disjoint_window=window_changed and not overlapping,
            )
            check_organizer = window_changed

            if check_organizer or attendees_to_check:
                conflicts, unchecked_attendees, unchecked_reason = _find_conflicts(
                    service,
                    effective_start,
                    effective_end,
                    attendees_to_check,
                    exclude_event_id=event_id,
                    check_organizer=check_organizer,
                )
                if conflicts:
                    return _conflict_response(
                        conflicts,
                        unchecked_attendees,
                        effective_start,
                        effective_end,
                        unchecked_reason=unchecked_reason,
                    )

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
        if added_attendees:
            # Only append the newly-added attendees - existing entries are
            # left completely untouched (dict identity and all), so their
            # RSVP state (responseStatus, optional, comment, ...) is never
            # at risk of being clobbered by a re-submission.
            event["attendees"] = (event.get("attendees") or []) + [
                {"email": address} for address in added_attendees
            ]
        requested_conference = _apply_conference_request(event, add_google_meet)

        request = service.events().update(
            calendarId="primary",
            eventId=event_id,
            body=event,
            **_api_call_kwargs(event, notify_attendees),
        )
        updated_event = request.execute()
        return _event_response(
            updated_event, **_unchecked_extra(unchecked_attendees, unchecked_reason)
        )

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
