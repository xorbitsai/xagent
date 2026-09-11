import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore
from googleapiclient.errors import HttpError  # type: ignore
from mcp.server.fastmcp import FastMCP

from ....config import get_tool_max_output_length
from .utils import InsufficientScopeError
from .utils import attendees_to_add as _attendees_to_add
from .utils import calendar_day_bounds as _calendar_day_bounds
from .utils import conflict_response as _conflict_response
from .utils import datetime_key_for_comparison as _datetime_key_for_comparison
from .utils import merge_scope_error as _merge_scope_error
from .utils import normalize_addresses as _normalize_addresses
from .utils import offset_datetime_string as _offset_datetime_string
from .utils import reject_reversed_window as _reject_reversed_window
from .utils import require_offset_datetime as _require_offset_datetime
from .utils import setup_proxy_env, success_with_capped_dict
from .utils import window_delta_segments as _window_delta_segments

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
) -> tuple[list[dict[str, Any]], list[str]]:
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

    Returns (conflicts, unchecked_attendees). A missing-scope 403 covering
    the whole call raises `InsufficientScopeError` (carrying whatever was
    already confirmed in `conflicts`/`unchecked_attendees` before the
    error) instead of degrading to unchecked and returning normally -
    that's OUR OWN credential's problem, not a per-attendee visibility
    gap, and writing an event whose availability was never actually
    checked would defeat the point of this feature. Every entry that
    does end up in `unchecked_attendees` on a normal return is the
    self-explanatory kind (absent from the response, or its own
    per-attendee error).
    """
    conflicts: list[dict[str, Any]] = []
    unchecked_attendees: list[str] = []

    if check_organizer:
        page_token: str | None = None
        while True:
            try:
                page = (
                    service.events()
                    .list(
                        calendarId="primary",
                        timeMin=time_min,
                        timeMax=time_max,
                        singleEvents=True,
                        orderBy="startTime",
                        pageToken=page_token,
                    )
                    .execute()
                )
            except HttpError as exc:
                if exc.resp.status == 403 and _is_insufficient_scope_error(exc):
                    raise InsufficientScopeError(
                        "Missing the calendar.events permission needed to "
                        "check organizer availability - reconnect the Google "
                        "Calendar connector to grant it. This is a missing "
                        "permission on our own credential, not a real "
                        "scheduling conflict to route around.",
                        conflicts,
                        unchecked_attendees + attendees,
                    ) from exc
                raise

            for item in page.get("items") or []:
                if item.get("status") == "cancelled":
                    continue
                if item.get("transparency") == "transparent":
                    continue
                # A recurring event's instances (from singleEvents=True
                # expansion) carry their OWN id, never the master's - e.g.
                # "<masterId>_<recurrenceStamp>" - so excluding only by `id`
                # never matches when `exclude_event_id` names the master
                # itself (the normal way to address a recurring series).
                # `recurringEventId` is the field Google's own Events
                # resource documents as reporting the master's id on each
                # instance; without also checking it, rescheduling or adding
                # an attendee to a recurring event's master would report the
                # event as conflicting with its own instances.
                if exclude_event_id and exclude_event_id in (
                    item.get("id"),
                    item.get("recurringEventId"),
                ):
                    continue
                # NOTE: declining an invite (attendees[].responseStatus ==
                # "declined" for the organizer's own self entry) is NOT used
                # as a busy/free signal here - Google's own Events docs treat
                # responseStatus and transparency as independent fields, with
                # no documented guarantee that declining clears transparency
                # to "transparent". An earlier version of this check skipped
                # declined events outright, which silently treated a
                # still-opaque declined event as free and permitted a real
                # double booking. `transparency` above is the one field
                # Google actually documents as the busy/free predicate.
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

            page_token = page.get("nextPageToken")
            if not page_token:
                break

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
                # hasn't been reconnected yet. This is OUR OWN
                # credential's problem, not a per-attendee visibility gap
                # (e.g. a specific attendee's calendar being unreadable,
                # which genuinely can't be fixed and still degrades to
                # unchecked below) - proceeding to write an event whose
                # availability was never actually checked would defeat
                # the entire point of this feature, so reject instead of
                # silently booking over a possible conflict. Carrying
                # `conflicts` (e.g. an organizer conflict already found
                # above, before this batch ever ran) lets a caller still
                # report it rather than silently discarding a known
                # problem just because this later, unrelated check also
                # failed.
                raise InsufficientScopeError(
                    "Missing the calendar.freebusy permission needed to "
                    "check attendee availability - reconnect the Google "
                    "Calendar connector to grant it. This is a missing "
                    "permission on our own credential, not a real "
                    "scheduling conflict to route around.",
                    conflicts,
                    # Every attendee from this failed batch onward is
                    # unchecked - every remaining batch would hit this
                    # same scope error too, so there's nothing left to
                    # gain by attempting them.
                    unchecked_attendees + attendees[offset:],
                ) from exc
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

    return conflicts, unchecked_attendees


def _primary_calendar_info(service: Any) -> tuple[str, str]:
    """The connected account's own (email, timezone) for its primary
    calendar - a single calendars().get(calendarId="primary") call serves
    both needs: `id` is the connected account's own address (used to tell
    whether the authenticated caller is actually a given event's
    organizer, since "primary" always means the caller's own calendar,
    never an arbitrary other attendee's or organizer's), and `timeZone`
    is the calendar's configured IANA zone, falling back to UTC if
    somehow absent (Google's Calendar resource always carries a
    timeZone, so this is a defensive last resort, not the expected path).

    Needs the calendar.calendars.readonly scope (see the
    20260909_add_calendar_calendars_readonly_scope migration) - neither
    calendar.events nor calendar.freebusy authorizes calendars.get, per
    Google's own scope reference for that endpoint.
    """
    try:
        calendar = service.calendars().get(calendarId="primary").execute()
    except HttpError as exc:
        if exc.resp.status == 403 and _is_insufficient_scope_error(exc):
            # This connection's OAuth token predates the
            # calendar.calendars.readonly scope. Without the calendar's
            # own identity/timezone, neither an all-day boundary nor a
            # different-organizer conflict check can be safely resolved -
            # reject rather than silently guessing (e.g. defaulting to
            # UTC, or assuming the caller is the organizer), either of
            # which could misjudge the query window or silently skip a
            # real conflict.
            # Raised as InsufficientScopeError rather than a validation
            # ValueError so broad input-validation handlers cannot
            # accidentally swallow a credential failure. Nothing has been
            # confirmed by this helper itself, so both accumulator args
            # are empty.
            raise InsufficientScopeError(
                "Missing the calendar.calendars.readonly permission "
                "needed to look up the calendar's own identity/timezone "
                "for this update - reconnect the Google Calendar "
                "connector to grant it. This is a missing permission on "
                "our own credential, not a real scheduling conflict to "
                "route around.",
                [],
                [],
            ) from exc
        raise
    return calendar.get("id") or "", calendar.get("timeZone") or "UTC"


def _event_boundary(field: dict[str, Any] | None, tz_name: str) -> str | None:
    """Return a RFC3339 timestamp usable as a freebusy/events.list time
    bound for one side (start or end) of an event.

    Google represents a timed event's boundary as {"dateTime": ...} and an
    all-day event's as {"date": "YYYY-MM-DD"} - timeMin/timeMax require a
    full RFC3339 timestamp with a zone offset, so a bare date needs
    widening.

    A timed boundary's `dateTime` USUALLY already carries its own zone
    offset, but Google's own EventDateTime docs are explicit that this
    isn't guaranteed: "A time zone offset is required unless a time zone
    is explicitly specified in timeZone" - a client (including one that
    isn't this tool) can legally write an offsetless `dateTime` paired
    with a sibling `timeZone` field instead. Comparing that offsetless
    string as if it were self-describing would compare naive against
    aware for what could be the same instant, defeating the overlap math
    this whole module exists to get right - so it must be resolved
    against its own `timeZone` (falling back to the calendar's default,
    `tz_name`, only if this specific field is missing one).

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
        try:
            is_naive = datetime.fromisoformat(date_time).tzinfo is None
        except ValueError:
            return date_time  # malformed - pass through unchanged, as before
        if not is_naive:
            return date_time
        return _offset_datetime_string(date_time, field.get("timeZone") or tz_name)
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


def _event_response(
    event: dict[str, Any], unchecked_attendees: list[str] | None = None
) -> str:
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
    if not unchecked_attendees:
        return json.dumps(response, ensure_ascii=False)

    # unchecked_attendees is appended AFTER the event has already been
    # capped to fit the output budget, so it needs its own check - an
    # attendee list has no documented maximum, and a whole-batch scope
    # error or a large invite list can leave most of it unchecked. Without
    # this, a long enough list could push the final response back over
    # budget with nothing left to shrink it, unlike conflict_response's
    # payload (which caps both of its own list fields for the same
    # reason).
    response["unchecked_attendees"] = unchecked_attendees
    max_output_length = get_tool_max_output_length()
    encoded = json.dumps(response, ensure_ascii=False)
    remaining_unchecked = unchecked_attendees
    while remaining_unchecked and len(encoded) > max_output_length:
        remaining_unchecked = remaining_unchecked[: len(remaining_unchecked) // 2]
        response["unchecked_attendees"] = remaining_unchecked
        response["truncated"] = True
        encoded = json.dumps(response, ensure_ascii=False)
    return encoded


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
    fine. This also skips the whole check outright (not just the conflict result) whenever it can't
    run at all -- e.g. a connector token missing a needed OAuth scope -- rather than failing loudly;
    only pass it once the user has actually confirmed they want to proceed without checking, not as a
    blanket way to route around an unrelated error. Adding attendees does not, by itself, email them; set notify_attendees=True to have Google
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
        # Checked before reject_reversed_window (which stays permissive,
        # not diagnostic, on a mismatched-awareness pair): two NAIVE
        # values that also happen to be reversed compare just fine (no
        # TypeError, so reject_reversed_window doesn't defer to this
        # check) and would otherwise surface the generic "must be after"
        # message instead of this more specific, actionable one.
        _require_offset_datetime(start_time, "start_time")
        _require_offset_datetime(end_time, "end_time")
        _reject_reversed_window(start_time, end_time)
        service = get_calendar_service()
        normalized_attendees = _normalize_addresses(attendees) if attendees else []

        unchecked_attendees: list[str] = []
        if not ignore_conflicts:
            try:
                conflicts, unchecked_attendees = _find_conflicts(
                    service, start_time, end_time, normalized_attendees
                )
            except InsufficientScopeError as exc:
                conflicts, unchecked_attendees = _merge_scope_error(exc, [], [])
            if conflicts:
                return _conflict_response(
                    conflicts,
                    unchecked_attendees,
                    start_time,
                    end_time,
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
        return _event_response(created_event, unchecked_attendees=unchecked_attendees)

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
    check once the user has explicitly confirmed a conflict is fine. This also skips the whole check
    outright (not just the conflict result) whenever it can't run at all -- e.g. a connector token
    missing a needed OAuth scope, including the one needed to widen an all-day event's own boundary
    -- rather than failing loudly; only pass it once the user has actually confirmed they want to
    proceed without checking, not as a blanket way to route around an unrelated error. Editing other
    fields (summary, description, location) without moving the event is never blocked.
    Moving the time or adding an attendee on a recurring event's own master (not a single occurrence)
    is refused outright rather than checked, since only that one occurrence's window can be verified
    here, not every occurrence the series will generate. Update a single occurrence directly by its
    own event id instead, or pass ignore_conflicts=True once the user has confirmed they've verified
    every occurrence themselves.
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
        if start_time is not None:
            _require_offset_datetime(start_time, "start_time")
        if end_time is not None:
            _require_offset_datetime(end_time, "end_time")
        service = get_calendar_service()

        # Lazily fetched and cached (as a one-element list, mutated
        # rather than rebound, so the closure needs no `nonlocal`) - this
        # single calendars().get(calendarId="primary") call resolves both
        # the connected account's own identity (needed below to tell
        # whether the caller is actually this event's organizer) and its
        # timezone (needed further down only for an all-day boundary),
        # so a call site that only needs one doesn't force a redundant
        # second round-trip when the other site already fetched it.
        primary_calendar_cache: list[tuple[str, str]] = []

        def primary_calendar_info() -> tuple[str, str]:
            if not primary_calendar_cache:
                primary_calendar_cache.append(_primary_calendar_info(service))
            return primary_calendar_cache[0]

        # First get the existing event
        event = service.events().get(calendarId="primary", eventId=event_id).execute()

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
        # docstring). `added_attendees` (genuinely new) and
        # `existing_attendees_raw` (retained) are checked separately
        # below - a retained attendee only needs checking against the
        # portion of a moved window that's actually new territory, while
        # a newly-added one needs the whole effective window checked.
        # Computed this early (rather than right before its first use)
        # specifically so the all-day timezone-lookup gate below can key
        # off whether anything is *actually* new, not just whether
        # `attendees` was given at all.
        added_attendees = _attendees_to_add(attendees, existing_attendee_emails)

        if (
            event.get("recurrence")
            and not ignore_conflicts
            and (bool(start_time) or bool(end_time) or added_attendees)
        ):
            # A recurring MASTER event's `recurrence` field (its RRULE/
            # EXRULE/RDATE/EXDATE lines) is only present on the master
            # itself - an individual occurrence carries `recurringEventId`
            # instead (see the exclusion logic in _find_conflicts) and is
            # unaffected by this check. Moving the master's time, or
            # adding an attendee, affects EVERY occurrence the series
            # generates, but the conflict check below only evaluates the
            # single window this call computes (in effect, just the
            # master's own occurrence) - there is no logic here to expand
            # and check each future occurrence the recurrence rule would
            # produce. Silently succeeding could let a real conflict on
            # some LATER occurrence through completely unchecked. Refuse
            # outright rather than give a false "no conflict" answer for
            # the whole series - a metadata-only edit (no start_time/
            # end_time/attendees change) is unaffected, and updating a
            # single occurrence directly by its own event id is too (that
            # event carries recurringEventId, not recurrence).
            raise ValueError(
                "This event is part of a recurring series (it has its "
                "own recurrence rule) - moving it or adding an attendee "
                "can only be conflict-checked against the single "
                "occurrence this call computes, not every occurrence the "
                "series will generate, so this write was refused rather "
                "than risk missing a real conflict on a later date. "
                "Update a single occurrence directly using its own event "
                "id instead of the series' id, make a metadata-only edit "
                "(summary/description/location) instead, or pass "
                "ignore_conflicts=True only once the user has explicitly "
                "confirmed they've verified every occurrence themselves."
            )

        # The organizer's own calendar is checked separately, via
        # check_organizer (events.list, excluded by id/recurringEventId) -
        # freebusy.query has no concept of "exclude this event", so if the
        # organizer's own email also shows up among the attendees being
        # freebusy-checked (just added now, or already on the event from
        # an earlier call), that query would always find this very
        # event's own busy block on their calendar and report a false
        # self-conflict. Excluded only from the freebusy-checked copies
        # below, never from `added_attendees` itself - a caller that
        # explicitly asked to add the organizer as an attendee must still
        # have them actually written to the event. Computed this early
        # (alongside added_attendees, rather than right before use) so
        # both the timezone-lookup gate and the organizer-check gate
        # below can key off the POST-filter attendee sets, not the raw
        # ones - gating off the raw sets would trigger an unnecessary
        # timezone lookup, or (worse) skip checking the organizer's
        # calendar entirely, in the specific case where the organizer's
        # own email was the only thing making a raw set non-empty.
        organizer = event.get("organizer") or {}
        # The organizer's REAL, event-provided email only - never
        # backfilled. Used later for the identity-fallback comparison
        # and for naming the organizer in `unchecked_attendees`; a
        # backfilled guess has no business appearing in either (the
        # identity question is already answered directly by
        # organizer_self whenever a backfill would apply, and reporting
        # a *guessed* address to the caller as "the organizer" would be
        # actively misleading).
        organizer_email = organizer.get("email")
        # Google's own documented signal for "is the organizer this
        # connected account" (per the Events resource: "Whether the
        # organizer corresponds to the calendar on which this copy of
        # the event appears") - reading it directly here is more
        # reliable than comparing email strings (no case-folding to get
        # wrong) and, in the common case where it's present, needs no
        # extra API call at all for identity purposes. None only when
        # the whole `organizer` object is absent or lacks this field -
        # not actually documented to happen for a self-organized event,
        # but handled the same as "assume self" purely as a defensive
        # fallback, not because it's expected in practice.
        organizer_self = organizer.get("self")
        # The address actually excluded from the freebusy-checked
        # attendee lists below - separate from organizer_email because
        # this one MAY fall back to the caller's own resolved address
        # when the real organizer email is missing. Conflating the two
        # would make the later identity-fallback comparison compare a
        # backfilled value against itself (always trivially True,
        # answering nothing) instead of a real, independent check.
        exclude_email = organizer_email
        if exclude_email is None and organizer_self is not False and added_attendees:
            # Whether via organizer.self=True or a wholly-missing
            # organizer object/field, the caller is (or is assumed to
            # be) the organizer here - but neither case gives us an
            # actual email to exclude from the freebusy batch below.
            # Resolved via the same calendars().get(calendarId="primary")
            # already used for caller_is_organizer/all-day widening
            # below (the closure caches it, so this never fetches
            # twice). Only resolved when there's a newly-added attendee
            # to check this against - a retained attendee's delta-
            # segment query never re-scans the event's OLD window
            # (where their own footprint would live), so it can't
            # self-conflict regardless of this exclusion, and a
            # metadata-only edit has no attendee to need it for at all.
            # Without this, adding the caller's OWN address as a "new"
            # attendee on such an event would never be excluded from
            # the freebusy batch and would always find this very
            # event's own busy block on their calendar - the exact
            # self-conflict class this exclusion exists to prevent for
            # the known-organizer-email case already.
            try:
                exclude_email = primary_calendar_info()[0]
            except (InsufficientScopeError, HttpError):
                # Not just the recognized missing-scope case -
                # _primary_calendar_info re-raises any OTHER HttpError
                # from calendars().get() unchanged (rate limiting, a
                # transient 5xx, an unrelated 403), and that's still
                # "this lookup couldn't run" every bit as much as a
                # missing scope is.
                if not ignore_conflicts:
                    raise
                # ignore_conflicts=True means the caller has already
                # decided the conflict check doesn't need to run - this
                # lookup failing for ANY reason here would only ever be
                # surfaced by that check (which is skipped entirely
                # below when ignore_conflicts is set), so leaving
                # exclude_email unresolved is safe: nothing downstream
                # that consults it runs in that case either. Without
                # this, any failure of this lookup - not just a missing
                # scope - would hard-fail a call the caller explicitly
                # opted out of the check the docstring says this
                # guards.
        # None never equals a real address's lowercased form, so this
        # filter is a no-op (keeps everything) when there's no address
        # to exclude - same effect as branching on `exclude_email`
        # explicitly, without a duplicated ternary.
        exclude_email_lower = exclude_email.lower() if exclude_email else None
        attendees_to_check = [
            a for a in added_attendees if a.lower() != exclude_email_lower
        ]
        existing_attendees_to_check = [
            a for a in existing_attendees_raw if a.lower() != exclude_email_lower
        ]
        # None never equals a real address's lowercased form either -
        # used below only for the identity-fallback comparison and for
        # naming the organizer in unchecked_attendees, both of which
        # need the REAL email, not exclude_email's backfilled guess.
        organizer_email_lower = organizer_email.lower() if organizer_email else None
        # Whether the organizer's own email was itself one of the
        # genuinely-new additions (filtered out of attendees_to_check
        # above) - if so, their availability still needs verifying, just
        # via the organizer-calendar path (which can actually exclude
        # this event) rather than freebusy (which can't). Without this,
        # adding ONLY the organizer as a new attendee - with the window
        # otherwise unchanged - would skip checking their calendar
        # entirely: `attendees_to_check` ends up empty (organizer
        # filtered out) and `check_organizer` alone wouldn't have been
        # true for an unmoved window, so no conflict check would run for
        # them at all, silently missing a real conflict on their calendar.
        # Recovered as a length comparison rather than a separate scan -
        # `added_attendees` is already case-insensitively deduplicated
        # (via normalize_addresses), so at most one entry can match
        # `exclude_email_lower` - which also keeps this fact and
        # `attendees_to_check`'s own filtering from drifting apart (both
        # the timezone-lookup gate below and the organizer-check gate
        # further down must account for this consistently).
        organizer_newly_added = len(attendees_to_check) != len(added_attendees)

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
        # boundary (see _event_boundary) so a moved window, a genuinely
        # new attendee, or the organizer-calendar check gets checked
        # against the right absolute instant - fetch it lazily so a
        # plain timed-event update, or a summary/location/description-
        # only edit of an all-day event that never touches its window or
        # attendees, doesn't pay for an extra API call (or a scope-403
        # that would otherwise block an edit that never needed timezone
        # precision at all) it has no use for. `organizer_newly_added`
        # is included alongside `attendees_to_check` (genuinely new,
        # organizer excluded) since it independently triggers
        # `check_organizer` below, which also needs a correctly-widened
        # window - omitting it here would run that check against a
        # wrong-timezone (UTC-defaulted) boundary instead of failing
        # loudly or skipping it. A resubmission that never moves the
        # window makes any retained attendee's delta segment trivially
        # empty regardless of timezone precision (both sides of that
        # comparison fall back to the same assumption consistently), so
        # that case alone still doesn't need the real zone.
        needs_real_calendar_timezone = existing_is_all_day and (
            bool(start_time)
            or bool(end_time)
            or bool(attendees_to_check)
            or organizer_newly_added
        )
        try:
            calendar_timezone = (
                primary_calendar_info()[1] if needs_real_calendar_timezone else "UTC"
            )
            # _event_boundary can itself raise ValueError (via
            # resolve_zoneinfo) when a stored event's own dateTime is
            # offsetless and its sibling timeZone field is unresolvable -
            # kept inside this same try so that case is governed by
            # ignore_conflicts too, not just a bad calendar_timezone
            # lookup. Otherwise a malformed event would hard-fail this
            # call even when the caller explicitly opted out of the
            # check the docstring says this guards.
            existing_start = _event_boundary(event.get("start"), calendar_timezone)
            existing_end = _event_boundary(event.get("end"), calendar_timezone)
        except (InsufficientScopeError, ValueError, HttpError):
            # Besides a recognized missing-scope error or malformed-event
            # ValueError from _event_boundary, a bare HttpError here means
            # _primary_calendar_info hit some OTHER failure (rate limiting,
            # a transient 5xx, an unrelated 403) that it re-raises unchanged.
            # All three mean this conflict-only lookup could not run.
            if not ignore_conflicts:
                raise
            # ignore_conflicts=True means the caller has already decided
            # the conflict check doesn't need to run - this lookup
            # failing for ANY reason here would only ever be surfaced by
            # that check (or by the reversed-window sanity check below,
            # which is fine to run a little less precisely against UTC
            # than to block a write the caller explicitly opted out of
            # checking). Leaving the existing boundaries unresolved
            # (None) is safe: the conflict-check block below is itself
            # gated on `not ignore_conflicts`, and the write payload only
            # ever uses the caller-supplied start_time/end_time, never
            # these.
            calendar_timezone = "UTC"
            existing_start = None
            existing_end = None
        effective_start = start_time or existing_start
        effective_end = end_time or existing_end
        if (start_time or end_time) and effective_start and effective_end:
            # Only worth checking when this call is actually about to
            # write a (possibly partly-existing) window - an
            # attendees/summary-only edit that never moves either
            # boundary would otherwise re-validate the event's
            # already-stored, unchanged start/end and could reject an
            # unrelated field edit over pre-existing data this call
            # never touches.
            _reject_reversed_window(effective_start, effective_end)
        existing_start_key = _datetime_key_for_comparison(existing_start)
        existing_end_key = _datetime_key_for_comparison(existing_end)
        effective_start_key = _datetime_key_for_comparison(effective_start)
        effective_end_key = _datetime_key_for_comparison(effective_end)
        window_changed = (effective_start_key, effective_end_key) != (
            existing_start_key,
            existing_end_key,
        )

        unchecked_attendees: list[str] = []
        if not ignore_conflicts and effective_start and effective_end:
            # check_organizer queries calendarId="primary", which is
            # always the AUTHENTICATED CALLER's own calendar - not
            # necessarily this event's organizer. Google copies an event
            # onto every attendee's own calendar too, and a guest with
            # edit permission (guestsCanModify) can call this tool on an
            # event they didn't organize; event["organizer"]["email"]
            # would then differ from the caller's own address. Comparing
            # against the calendar's own resolved identity (rather than
            # assuming caller==organizer, as an earlier version did)
            # catches that case rather than silently checking - and
            # excluding from freebusy - the wrong person's calendar.
            # A newly-added organizer is checked via the organizer path
            # (events.list, which can actually exclude this event by id/
            # recurringEventId) rather than freebusy - without also
            # triggering it here, adding ONLY the organizer as a new
            # attendee on an otherwise-unmoved event would leave
            # `attendees_to_check` empty (organizer filtered out above)
            # and `window_changed` false, skipping their calendar
            # entirely instead of just skipping the self-conflicting
            # freebusy path for them. Independent of whether the
            # organizer's own identity is resolvable at all - even an
            # event with no organizer info whatsoever must still trigger
            # this the same as before this identity fix.
            organizer_needs_verification = window_changed or organizer_newly_added
            # Gated on organizer_needs_verification too - caller_is_organizer
            # is only ever consulted below when the organizer needs
            # (re)verifying at all, so a metadata-only edit (summary/
            # description/location, no window or attendee change) on an
            # event with an explicit organizer field must not pay for
            # an API call when nothing downstream would use its result.
            caller_is_organizer = True
            if organizer_needs_verification:
                if organizer_self is not None:
                    # The documented, direct signal - no email
                    # comparison (and its case-folding risk) or extra
                    # API call needed when Google already told us.
                    caller_is_organizer = organizer_self
                elif organizer_email_lower is not None:
                    caller_is_organizer = (
                        primary_calendar_info()[0].lower() == organizer_email_lower
                    )
            # Only actually query "primary" for the organizer's own
            # conflicts when the caller genuinely IS the organizer -
            # otherwise "primary" is someone else's calendar entirely,
            # and mislabeling their busy blocks as the organizer's would
            # be actively wrong, not just incomplete.
            check_organizer = organizer_needs_verification and caller_is_organizer
            if (
                organizer_email
                and organizer_needs_verification
                and not caller_is_organizer
            ):
                if window_changed:
                    # freebusy.query accepts any calendar/email id, not
                    # just "primary" - the real organizer's calendar CAN
                    # be checked this way, same as any other attendee.
                    # Their own copy of this event already occupies the
                    # OLD window regardless of whether they're also
                    # listed as an attendee (they've always been the
                    # organizer), so - exactly like a retained attendee -
                    # only the delta segments beyond that old window are
                    # safe to check without self-conflicting on their own
                    # unrelated busy block for this same event.
                    existing_attendees_to_check = existing_attendees_to_check + [
                        organizer_email
                    ]
                else:
                    # window_changed is False here only because
                    # organizer_newly_added forced organizer_needs_
                    # verification - the window itself never moved, so
                    # window_delta_segments would find no new territory
                    # at all to check them against. Unlike the
                    # caller_is_organizer path (which excludes by event
                    # id via events.list and so can safely re-scan the
                    # whole unchanged window), freebusy has no id-based
                    # exclusion - there is genuinely no way to verify an
                    # unchanged window for them without risking a
                    # self-conflict. Report as unchecked rather than
                    # silently treating them as clear.
                    unchecked_attendees.append(organizer_email)
            all_conflicts: list[dict[str, Any]] = []
            # Tracks the most recent InsufficientScopeError seen across
            # EITHER block below, re-raised only once both blocks have
            # had their chance to run and nothing was confirmed by
            # either. The two blocks check disjoint attendee sets (newly
            # added vs. retained), so one block hitting a scope error
            # must not skip the other - otherwise a retained attendee's
            # delta-segment check would silently never run at all
            # (neither confirmed as conflicting nor reported as
            # unchecked) whenever the first block happens to fail with
            # an already-confirmed conflict in hand (which is exactly
            # the case that stops `_merge_scope_error` from re-raising).
            pending_scope_error: InsufficientScopeError | None = None

            def _run_and_accumulate(
                time_min: str, time_max: str, attendees: list[str], *, check_org: bool
            ) -> bool:
                """Runs one _find_conflicts call, folding its result (or,
                on a scope error, whatever it had already confirmed) into
                the shared all_conflicts/unchecked_attendees accumulators.
                Returns True on a scope error, so a segment loop can stop
                attempting further segments (every remaining one would
                hit the same error) - a `for/break`, not a re-raise,
                since the OTHER accumulation block below must still get
                its own chance to run regardless.
                """
                nonlocal pending_scope_error
                try:
                    conflicts, unchecked = _find_conflicts(
                        service,
                        time_min,
                        time_max,
                        attendees,
                        exclude_event_id=event_id,
                        check_organizer=check_org,
                    )
                    all_conflicts.extend(conflicts)
                    unchecked_attendees.extend(unchecked)
                    return False
                except InsufficientScopeError as exc:
                    all_conflicts.extend(exc.conflicts)
                    unchecked_attendees.extend(exc.unchecked_attendees)
                    pending_scope_error = exc
                    return True

            # A newly-added attendee has no footprint on this event
            # at all, so the FULL effective window is safe (and
            # necessary) to check for them - same call also covers
            # the organizer, who's excluded by event id rather than
            # by window, so `window_changed` alone (not a
            # disjoint-move test) decides whether to re-check them.
            if check_organizer or attendees_to_check:
                _run_and_accumulate(
                    effective_start,
                    effective_end,
                    attendees_to_check,
                    check_org=check_organizer,
                )

            # A retained attendee's own busy block for THIS event
            # covers the entire OLD window - querying that overlap
            # can't tell "busy because of this event" from a real
            # conflict. Only the portion of the new window that's
            # genuinely new territory (window_delta_segments - empty
            # for an unchanged/shrunk window, up to two segments for
            # a partial nudge that extends past the old window on one
            # or both sides, or the whole new window for a fully
            # disjoint move) can hide a real conflict for them.
            if existing_attendees_to_check:
                for seg_start, seg_end in _window_delta_segments(
                    existing_start_key,
                    existing_end_key,
                    effective_start_key,
                    effective_end_key,
                ):
                    hit_scope_error = _run_and_accumulate(
                        seg_start.isoformat(),
                        seg_end.isoformat(),
                        existing_attendees_to_check,
                        check_org=False,
                    )
                    # Every remaining segment would hit this same scope
                    # error too - nothing left to gain by attempting them.
                    if hit_scope_error:
                        break

            # A both-sides-widened window produces two delta segments
            # (the new territory on each side), both checked against the
            # SAME existing_attendees_to_check list - an attendee who's
            # unverifiable for a reason unrelated to which segment ran
            # (absent from every freebusy response, or their own
            # per-attendee error) would otherwise land in
            # unchecked_attendees once per segment. Dedup both
            # accumulators once, after every block above has had its
            # chance to contribute, rather than per-block (which would
            # miss a duplicate straddling the organizer/attendee split).
            deduped_unchecked: list[str] = []
            seen_unchecked: set[str] = set()
            for attendee in unchecked_attendees:
                if attendee not in seen_unchecked:
                    seen_unchecked.add(attendee)
                    deduped_unchecked.append(attendee)
            unchecked_attendees = deduped_unchecked

            deduped_conflicts: list[dict[str, Any]] = []
            seen_conflicts: set[tuple[Any, Any, Any, Any]] = set()
            for conflict in all_conflicts:
                conflict_key = (
                    conflict.get("calendar"),
                    conflict.get("summary"),
                    conflict.get("start"),
                    conflict.get("end"),
                )
                if conflict_key not in seen_conflicts:
                    seen_conflicts.add(conflict_key)
                    deduped_conflicts.append(conflict)
            all_conflicts = deduped_conflicts

            if pending_scope_error is not None and not all_conflicts:
                raise pending_scope_error

            if all_conflicts:
                return _conflict_response(
                    all_conflicts,
                    unchecked_attendees,
                    effective_start,
                    effective_end,
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
        return _event_response(updated_event, unchecked_attendees=unchecked_attendees)

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
