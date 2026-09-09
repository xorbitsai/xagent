import json
import logging
import os
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any
from urllib.parse import quote

import requests
from mcp.server.fastmcp import FastMCP

from .utils import attendees_to_add as _attendees_to_add
from .utils import attendees_were_given as _attendees_were_given
from .utils import conflict_response as _conflict_response
from .utils import datetime_key_for_comparison as _datetime_key_for_comparison
from .utils import naive_day_bounds as _naive_day_bounds
from .utils import normalize_addresses as _normalize_addresses
from .utils import offset_datetime_string as _offset_datetime_string
from .utils import reject_reversed_window as _reject_reversed_window
from .utils import resolve_zoneinfo as _resolve_zoneinfo
from .utils import setup_proxy_env
from .utils import timezones_could_differ as _timezones_could_differ
from .utils import unchecked_extra as _unchecked_extra
from .utils import window_delta_segments as _window_delta_segments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("outlook-mcp")

setup_proxy_env()

mcp = FastMCP("outlook-mcp")

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30

# getSchedule accepts at most this many schedules (users/resources) per call.
_MAX_ATTENDEES_PER_SCHEDULE_QUERY = 20


class _GraphRequestError(RuntimeError):
    """Raised by _graph_request on any HTTP error response.

    Carries status_code so a caller that needs to distinguish e.g. a 403
    (likely a missing-scope/permission issue on a /me/... call) from other
    failures doesn't have to string-match the rendered message - Graph has
    no single canonical error code for "missing scope" the way Slack does.
    Still a RuntimeError, so every existing bare `except Exception` call
    site keeps working unchanged.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


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
        raise _GraphRequestError(message, status_code=response.status_code) from exc

    if response.status_code == 204 or not response.content:
        return {}
    return response.json()


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


def _utc_field_in_zone(field: dict[str, Any], zone_name: str) -> dict[str, Any]:
    """Convert a dateTimeTimeZone field from a plain (unprefixed) GET -
    Microsoft's own docs: "By default, the start/end time is in UTC" -
    into the equivalent wall-clock value in `zone_name`, computed locally
    rather than by re-fetching with a Prefer header.

    Needed anywhere `zone_name` comes from `originalStartTimeZone` (the
    event's real configured zone) rather than from this same field's own
    "timeZone" (always "UTC" here): pairing that real zone name with the
    UTC clock value unchanged - e.g. via `_offset_datetime_string` - would
    silently mislabel a UTC instant as if it were already expressed in
    the real zone, off by the real zone's UTC offset. This also matters
    for is_all_day day-bounds widening, which operates on the naive
    clock value's own date - a UTC-denominated date can name a different
    calendar day than the instant's real local one, near a day boundary.

    Falls back to `field` unchanged when there's nothing to convert (no
    dateTime) or `zone_name` doesn't resolve - a wrong zone name here is
    caught elsewhere (`_resolve_zoneinfo` is also used on the write path),
    not silently swallowed by this being a no-op.
    """
    date_time = field.get("dateTime")
    if not date_time:
        return field
    try:
        zone = _resolve_zoneinfo(zone_name)
    except ValueError:
        return field
    utc_instant = datetime.fromisoformat(date_time).replace(tzinfo=dt_timezone.utc)
    local_instant = utc_instant.astimezone(zone).replace(tzinfo=None)
    return {"dateTime": local_instant.isoformat(), "timeZone": zone_name}


def _next_link_path(next_link: str) -> str:
    """Strip GRAPH_BASE_URL from an @odata.nextLink so it can be re-issued
    through `_graph_request` as a plain path+query (the link is always an
    absolute URL; `_graph_request` builds its own URL as
    ``f"{GRAPH_BASE_URL}{path}"``, so passing the absolute link verbatim as
    `path` would double up the host instead of following it)."""
    if next_link.startswith(GRAPH_BASE_URL):
        return next_link[len(GRAPH_BASE_URL) :]
    return next_link


# Defensive cap on calendarView pages followed for one organizer-side
# conflict check - comfortably more than any single-window query should
# ever need, just bounding the loop against an unexpected/pathological
# amount of paging rather than looping forever.
_MAX_CALENDAR_VIEW_PAGES = 20


def _find_conflicts(
    start_datetime: str,
    end_datetime: str,
    timezone: str,
    attendees: list[str],
    *,
    exclude_event_id: str | None = None,
    check_organizer: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Check the organizer's own calendar (when check_organizer) plus each
    attendee's schedule for anything overlapping [start_datetime,
    end_datetime) in the given timezone.

    `start_datetime`/`end_datetime` are naive (no embedded offset) clock
    values paired with `timezone` - the same shape Outlook's own
    dateTimeTimeZone resource uses, and getSchedule's structured
    startTime/endTime fields accept directly. calendarView's
    startDateTime/endDateTime query parameters are different: Graph's own
    docs say they're "interpreted using the timezone offset specified in
    the value" and "aren't impacted by the value of the Prefer ... header"
    - a naive value there is silently read as UTC. So only the calendarView
    call needs an explicit offset attached before it's sent.

    getSchedule has no concept of "exclude this event": it only returns raw
    busy blocks, so a query against a window an attendee is *already* busy
    for (because that's the very event being updated) can't be told apart
    from a genuine conflict here. Callers that aren't moving the event to a
    new window must restrict `attendees` to only the newly-added ones (see
    outlook_update_event) rather than relying on this function to exclude
    the event's own footprint on attendee schedules.

    Returns (conflicts, unchecked_attendees). A missing-scope 403 covering
    the whole call raises ValueError instead of degrading to unchecked -
    that's OUR OWN credential/policy problem, not a per-attendee
    visibility gap, and writing an event whose availability was never
    actually checked would defeat the point of this feature. Every entry
    that does end up in `unchecked_attendees` is the self-explanatory
    kind (absent from the response, or its own per-attendee error).
    """
    conflicts: list[dict[str, Any]] = []
    unchecked_attendees: list[str] = []

    if check_organizer:
        offset_start = _offset_datetime_string(start_datetime, timezone)
        offset_end = _offset_datetime_string(end_datetime, timezone)
        params: dict[str, Any] | None = {
            "startDateTime": offset_start,
            "endDateTime": offset_end,
            "$top": 250,
            "$select": "id,subject,start,end,isCancelled,showAs,responseStatus",
        }
        next_path: str | None = None
        for _ in range(_MAX_CALENDAR_VIEW_PAGES):
            calendar_view = _graph_request(
                "GET",
                next_path if next_path is not None else "/me/calendarView",
                params=params if next_path is None else None,
                extra_headers={"Prefer": f'outlook.timezone="{timezone}"'},
            )
            for item in calendar_view.get("value") or []:
                if item.get("isCancelled"):
                    continue
                # Graph's own showAs enum documents "unknown" as an
                # unclassified value (e.g. an event synced from a
                # third-party calendar that never set it), not
                # specifically a permission gap - it can represent a
                # genuinely busy block. Missing a real conflict is worse
                # than occasionally over-flagging one the caller can
                # dismiss with ignore_conflicts, so only skip the values
                # that are unambiguously not-busy - matching the policy
                # used for attendees' schedules below.
                if item.get("showAs") in ("free", "workingElsewhere"):
                    continue
                if exclude_event_id and item.get("id") == exclude_event_id:
                    continue
                # NOTE: declining an invite (responseStatus.response ==
                # "declined") is NOT used as a busy/free signal here -
                # Microsoft's event resource documents responseStatus and
                # showAs as independent fields, with no documented
                # guarantee that declining clears showAs to "free". An
                # earlier version of this check skipped declined events
                # outright, which silently treated a still-busy declined
                # event as free and permitted a real double booking.
                # `showAs` above is the one field Graph actually
                # documents as the busy/free predicate.
                start = item.get("start") or {}
                end = item.get("end") or {}
                conflicts.append(
                    {
                        "calendar": "organizer",
                        "summary": item.get("subject") or "(no subject)",
                        "start": start.get("dateTime"),
                        "end": end.get("dateTime"),
                    }
                )
            next_link = calendar_view.get("@odata.nextLink")
            if not next_link:
                break
            next_path = _next_link_path(next_link)

    # getSchedule accepts at most _MAX_ATTENDEES_PER_SCHEDULE_QUERY
    # schedules per call - chunk rather than giving up on the whole batch,
    # so a large invite list still gets everyone it can check checked.
    # `range(0, 0, N)` is empty, so this is also just a no-op loop when
    # `attendees` is empty.
    for offset in range(0, len(attendees), _MAX_ATTENDEES_PER_SCHEDULE_QUERY):
        batch = attendees[offset : offset + _MAX_ATTENDEES_PER_SCHEDULE_QUERY]
        try:
            schedule = _graph_request(
                "POST",
                "/me/calendar/getSchedule",
                body={
                    "schedules": batch,
                    "startTime": {"dateTime": start_datetime, "timeZone": timezone},
                    "endTime": {"dateTime": end_datetime, "timeZone": timezone},
                    "availabilityViewInterval": 30,
                },
            )
        except _GraphRequestError as exc:
            if exc.status_code == 403:
                # Graph gives no single reliable code for "this is a
                # missing-scope/permission issue" (unlike Google's
                # PERMISSION_DENIED/insufficientPermissions pairing) -
                # but a 403 on this /me/... call for the signed-in
                # user's own mailbox is not expected to have another
                # cause here. This is OUR OWN credential/policy problem,
                # not a per-attendee visibility gap (which genuinely
                # can't be fixed and still degrades to unchecked below) -
                # proceeding to write an event whose availability was
                # never actually checked would defeat the entire point
                # of this feature, so reject instead of silently booking
                # over a possible conflict.
                raise ValueError(
                    "Missing the calendars.read/schedule permission "
                    "needed to check attendee availability - "
                    "reconnecting the Outlook connector may grant it; if "
                    "the connector already has calendar access, this is "
                    "more likely an org-level policy blocking this call, "
                    "which a reconnect won't fix. Pass "
                    "ignore_conflicts=true if the user has confirmed "
                    "they want to proceed without this check."
                ) from exc
            raise

        # Keyed by Graph's own scheduleId casing for lookup, but every
        # value reported back below (conflicts and unchecked_attendees)
        # uses the caller's original attendees casing - matching
        # google_calendar's _find_conflicts, which iterates `attendees`
        # for the same reason.
        by_schedule_id = {
            (entry.get("scheduleId") or "").lower(): entry
            for entry in (schedule.get("value") or [])
        }
        for email in batch:
            entry = by_schedule_id.get(email.lower())
            if entry is None:
                # Not even present in the response - can't tell
                # whether they're free, so don't silently report them
                # as clear.
                unchecked_attendees.append(email)
                continue
            if entry.get("error"):
                unchecked_attendees.append(email)
                continue
            for item in entry.get("scheduleItems") or []:
                # See the matching comment on the organizer-side loop
                # above: "unknown" is unclassified, not confirmed-free,
                # so it's treated the same way here for consistency -
                # missing a real conflict is worse than occasionally
                # over-flagging one the caller can dismiss with
                # ignore_conflicts.
                if item.get("status") in ("busy", "tentative", "oof", "unknown"):
                    start = item.get("start") or {}
                    end = item.get("end") or {}
                    conflicts.append(
                        {
                            "calendar": email,
                            "summary": None,
                            "start": start.get("dateTime"),
                            "end": end.get("dateTime"),
                        }
                    )

    return conflicts, unchecked_attendees


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
    ignore_conflicts: bool = False,
) -> str:
    """Create an Outlook calendar event.
    attendees, if given, are invited by email (Graph emails them the invite)
    and are checked for scheduling conflicts together with the organizer's
    own calendar; a conflict returns status="conflict" instead of creating
    the event. Pass ignore_conflicts=True to create it anyway once the user
    has explicitly confirmed a conflict is fine.
    """
    try:
        # Both sides share the same `timezone`, so comparing them as naive
        # values (no offset attached) is already a valid relative
        # comparison - it doesn't matter which real zone that is.
        _reject_reversed_window(start_datetime, end_datetime)
        normalized_attendees = _normalize_addresses(attendees) if attendees else []

        unchecked_attendees: list[str] = []
        if not ignore_conflicts:
            conflicts, unchecked_attendees = _find_conflicts(
                start_datetime, end_datetime, timezone, normalized_attendees
            )
            if conflicts:
                return _conflict_response(
                    conflicts,
                    unchecked_attendees,
                    start_datetime,
                    end_datetime,
                )

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
        if normalized_attendees:
            payload["attendees"] = _attendee_list(normalized_attendees)

        result = _graph_request("POST", "/me/events", body=payload)
        return _success(event=result, **_unchecked_extra(unchecked_attendees))
    except Exception as e:
        logger.error("Error creating Outlook event: %s", e)
        return _error(str(e))


@mcp.tool()
def outlook_update_event(
    event_id: str,
    subject: str | None = None,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
    timezone: str | None = None,
    body: str | None = None,
    location: str | None = None,
    attendees: list[str] | str | None = None,
    is_all_day: bool | None = None,
    ignore_conflicts: bool = False,
) -> str:
    """Update an existing Outlook calendar event.
    If the update moves the event to a new time, or adds attendees, that
    change is checked for conflicts the same way outlook_create_event is;
    pass ignore_conflicts=True to skip the check once the user has
    explicitly confirmed a conflict is fine. Editing other fields (subject,
    body, location) without moving the event is never blocked.
    attendees, if given, fully replaces the event's attendee list: any
    address already on the event that's left out is removed, and passing
    an explicit empty list clears every attendee. Leave attendees unset to
    keep the existing list untouched.
    Passing only one of start_datetime/end_datetime nudges that boundary
    while keeping the other as-is; leave timezone unset for this case and
    the event's own existing timezone is reused automatically. Passing
    both together fully replaces the window, and timezone then describes
    both new values (defaulting to UTC if also left unset).
    """
    try:
        attendees_given = _attendees_were_given(attendees)

        touches_schedule = (
            start_datetime is not None
            or end_datetime is not None
            or attendees_given
            or is_all_day is not None
        )

        # A single boundary changing without the other means the untouched
        # one keeps its existing clock value - but Graph needs ONE timeZone
        # per boundary object, and writing the new boundary in a different
        # zone than the one the untouched boundary (and any conflict check)
        # is actually anchored to would silently shift the event by the
        # zone offset. So this case always needs the existing event's own
        # timeZone, for the PATCH body itself, not just for the conflict
        # check - unlike every other needs_existing reason below, this one
        # applies even with ignore_conflicts=True.
        single_boundary_update = (start_datetime is not None) != (
            end_datetime is not None
        )

        # Needed for the conflict check below, for resolving the timezone
        # a single-boundary update must be written in, and (when attendees
        # is given) to merge into rather than replace the existing attendee
        # array - fetch it once up front whenever any of those is actually
        # needed. ignore_conflicts only skips the check itself, not the
        # timezone-resolution or attendee-merge uses, so those still need
        # this even then; a plain is_all_day-only edit (or a full
        # start_datetime+end_datetime replace) with ignore_conflicts=True
        # needs none of them and stays a single PATCH.
        existing: dict[str, Any] = {}
        needs_existing = (
            attendees_given
            or single_boundary_update
            or (not ignore_conflicts and touches_schedule)
        )
        if needs_existing:
            existing = _graph_request(
                "GET",
                f"/me/events/{quote(event_id, safe='')}",
                params={
                    "$select": "start,end,attendees,isAllDay,originalStartTimeZone"
                },
            )

        existing_attendees_raw = [
            a["emailAddress"]["address"]
            for a in (existing.get("attendees") or [])
            if (a.get("emailAddress") or {}).get("address")
        ]
        existing_attendee_emails = {
            address.lower() for address in existing_attendees_raw
        }

        if single_boundary_update:
            # A plain GET (no Prefer header, as above) always returns
            # start/end in UTC regardless of the event's actual configured
            # zone - Microsoft's own docs for both properties: "By
            # default, the start/end time is in UTC." So start.timeZone
            # here is always "UTC", never useful as "the event's real
            # timezone". originalStartTimeZone is the one field that
            # reports the zone the event was actually created in,
            # unaffected by any Prefer header.
            existing_timezone = existing.get("originalStartTimeZone")
            if not existing_timezone:
                raise ValueError(
                    "Existing event has no originalStartTimeZone; cannot "
                    "safely update only one of start_datetime/end_datetime "
                    "without it."
                )
            if existing_timezone.startswith("tzone://"):
                # A legacy custom timezone set in desktop Outlook (Graph's
                # own docs call this out specifically for this field) -
                # not a real IANA/Windows zone name, so it can't be used as
                # a Prefer header value or a dateTimeTimeZone.timeZone on
                # write. There's no value this code could resolve to here
                # that Graph would actually accept.
                raise ValueError(
                    "Existing event uses a legacy custom timezone "
                    f"({existing_timezone!r}) that can't be reused for a "
                    "single-boundary update; pass both start_datetime and "
                    "end_datetime together with an explicit timezone."
                )
            if timezone is not None:
                try:
                    _resolve_zoneinfo(timezone)
                except ValueError as exc:
                    # `timezones_could_differ` treats "can't resolve" as
                    # "benefit of the doubt, not confirmed different" -
                    # right for a name Graph itself reported (it's always
                    # written verbatim from `existing_timezone`, never from
                    # this value), but wrong for the caller's OWN value: an
                    # unresolvable `timezone` here is a bad argument, not an
                    # ambiguous-but-plausible one, and must not be silently
                    # discarded in favor of the existing zone.
                    raise ValueError(f"Unknown timezone {timezone!r}.") from exc
            if timezone is not None and _timezones_could_differ(
                timezone, existing_timezone
            ):
                # Only one boundary is moving, and the caller explicitly
                # gave a timezone that positively denotes a different real
                # zone than the one the untouched boundary is actually
                # recorded in - there's no single timezone that correctly
                # describes both endpoints here. (An unset `timezone` is
                # not treated as a conflicting choice - it just means "use
                # whatever this event already uses"; nor is a same-zone
                # value written differently, e.g. a Windows name Graph
                # reported vs. the IANA name the caller supplied.)
                raise ValueError(
                    "Updating only start_datetime or only end_datetime "
                    f"with a timezone ({timezone!r}) that denotes a "
                    "different real zone than the existing event's "
                    f"({existing_timezone!r}) is ambiguous; pass both "
                    "start_datetime and end_datetime together, or omit "
                    "timezone to reuse the existing one."
                )
            resolved_timezone = existing_timezone
            # The first GET (no Prefer header) returned start/end in UTC,
            # not `resolved_timezone` - re-query specifically in that zone
            # so the untouched boundary's clock value is expressed the
            # same way the PATCH itself is about to write it, rather than
            # this code guessing at a UTC<->zone conversion Graph already
            # knows how to do correctly.
            existing_zoned = _graph_request(
                "GET",
                f"/me/events/{quote(event_id, safe='')}",
                params={"$select": "start,end"},
                extra_headers={"Prefer": f'outlook.timezone="{resolved_timezone}"'},
            )
            existing_start_field = existing_zoned.get("start") or {}
            existing_end_field = existing_zoned.get("end") or {}
            existing_zone = resolved_timezone
        else:
            resolved_timezone = timezone or "UTC"
            existing_start_field = existing.get("start") or {}
            existing_end_field = existing.get("end") or {}
            # A plain GET (no Prefer header) reports start/end in UTC by
            # default, so its own "timeZone" field is never useful as
            # "the event's real timezone" - same caveat as the
            # single-boundary branch above. originalStartTimeZone is the
            # field that actually reports it (used here, unlike the
            # single-boundary branch, without erroring on a legacy
            # `tzone://` custom zone or an absent value - this branch
            # doesn't always need the real zone at all, e.g. a plain
            # attendees-only edit on a timed event never widens anything
            # with it, so fall back to the UTC-normalized field, the
            # prior behavior, rather than failing a call that may not
            # need this value to be exact).
            existing_timezone = existing.get("originalStartTimeZone")
            if existing_timezone and not existing_timezone.startswith("tzone://"):
                existing_zone = existing_timezone
                # existing_start_field/existing_end_field are still the
                # plain-GET UTC values above - re-express them as the real
                # zone's own wall-clock reading (computed locally, no
                # extra round trip) so they're not a UTC clock value
                # mislabeled with a different zone's name. That mismatch
                # would misjudge a same-instant resubmission as "moved" in
                # _key() below, and - separately - would widen an
                # is_all_day toggle to the wrong calendar day whenever the
                # real zone's offset pushes the instant across a day
                # boundary from its UTC date.
                existing_start_field = _utc_field_in_zone(
                    existing_start_field, existing_timezone
                )
                existing_end_field = _utc_field_in_zone(
                    existing_end_field, existing_timezone
                )
            else:
                existing_zone = existing_start_field.get("timeZone")

        payload: dict[str, Any] = {}
        if subject is not None:
            payload["subject"] = subject
        if start_datetime is not None:
            payload["start"] = {
                "dateTime": start_datetime,
                "timeZone": resolved_timezone,
            }
        if end_datetime is not None:
            payload["end"] = {"dateTime": end_datetime, "timeZone": resolved_timezone}
        if body is not None:
            payload["body"] = _message_body(body, "text")
        if location is not None:
            payload["location"] = {"displayName": location}
        # attendees fully REPLACES the event's attendee list (matching this
        # connector's pre-existing base behavior before this tool's
        # conflict-detection support was added): an explicit [] clears
        # everyone, and any existing address left out of the new list is
        # removed. `added_attendees` (genuinely new) and
        # `retained_attendees_raw` (kept from before) are checked
        # separately below - a retained attendee only needs checking
        # against the portion of a moved window that's actually new
        # territory, while a newly-added one needs the whole query window
        # checked; a removed attendee needs no check at all.
        added_attendees = _attendees_to_add(attendees, existing_attendee_emails)
        if attendees_given:
            # Reuse each retained attendee's existing raw dict (not just
            # its address) so its RSVP state (status, type, ...) is never
            # clobbered by a re-submission - only a genuinely new address
            # gets a fresh minimal dict.
            existing_by_email = {
                a["emailAddress"]["address"].lower(): a
                for a in (existing.get("attendees") or [])
                if (a.get("emailAddress") or {}).get("address")
            }
            assert (
                attendees is not None
            )  # narrows for mypy; attendees_given implies this
            desired_addresses = _normalize_addresses(attendees)
            desired_lower = {address.lower() for address in desired_addresses}
            if desired_lower != existing_attendee_emails:
                # A byte-identical resubmission (same members, regardless
                # of order) is not a real change - leaving the field out of
                # the PATCH entirely, rather than resending a reconstructed
                # copy of it, is a stronger guarantee against wiping RSVP
                # state than relying on every entry happening to compare
                # equal.
                payload["attendees"] = [
                    existing_by_email.get(
                        address.lower(),
                        {"emailAddress": {"address": address}, "type": "required"},
                    )
                    for address in desired_addresses
                ]
            retained_attendees_raw = [
                address
                for address in existing_attendees_raw
                if address.lower() in desired_lower
            ]
        else:
            retained_attendees_raw = existing_attendees_raw
        if is_all_day is not None:
            payload["isAllDay"] = is_all_day

        if not payload and not attendees_given:
            # attendees_given alone can leave `payload` empty (the
            # resubmitted set exactly matches what's already on the event)
            # without this having been a no-arg call - the caller did
            # provide a real field, it just happens to be a no-op given the
            # event's current state.
            raise ValueError("at least one field must be provided to update the event")

        # Checked unconditionally, not gated on ignore_conflicts - this is
        # basic input sanity (a window a provider API should never be
        # asked to write), not a conflict-check decision the caller can
        # opt out of. Matches google_calendar_update_events, and this
        # tool's own create path, both of which validate regardless of
        # ignore_conflicts too.
        existing_end = existing_end_field.get("dateTime")
        existing_start = existing_start_field.get("dateTime")
        effective_start = start_datetime or existing_start
        effective_end = end_datetime or existing_end
        if effective_start and effective_end:
            _reject_reversed_window(effective_start, effective_end)

        unchecked_attendees: list[str] = []
        if not ignore_conflicts and touches_schedule:
            # Both boundaries always end up denominated in the same zone
            # here: when only one of start_datetime/end_datetime is given
            # (single_boundary_update), `existing_zone` already equals
            # `resolved_timezone`; when both are given, `resolved_timezone`
            # is used for the query. Only when NEITHER is given (an
            # attendees/is_all_day-only edit) does the query need the
            # existing event's own recorded zone. Whether that zone is
            # actually known (and, if not, whether that's fatal) is
            # resolved lazily below, only once it's clear a query will
            # really run - a same-attendees, same-window resubmission
            # must stay a no-op PATCH even if the existing event happens
            # to be missing timezone info that would only matter for a
            # query nothing here ends up needing.
            provisional_query_timezone = (
                resolved_timezone
                if (start_datetime is not None or end_datetime is not None)
                else (existing_zone or "UTC")
            )

            # existing_start/existing_end are naive Graph values - they're
            # only comparable against effective_start/effective_end (also
            # naive, but not necessarily in the same zone: existing_zone
            # can be plain UTC from an un-prefixed GET while
            # query_timezone is whatever the caller/event actually uses)
            # once both sides carry a real, matching UTC offset. Comparing
            # the naive strings directly would read a same-instant
            # resubmission written in a different zone as "the window
            # moved" - resurrecting the self-conflict bug this whole
            # timezone-resolution logic exists to prevent. This comparison
            # only decides whether to widen the attendee check, never what
            # gets written or queried, so an unknown zone falls back to
            # UTC here (the conservative direction: a wrong guess makes
            # the window look more likely to have "changed", which only
            # means checking more attendees, never fewer).
            def _key(value: str | None, tz_name: str) -> datetime | str | None:
                return _datetime_key_for_comparison(
                    _offset_datetime_string(value, tz_name) if value else None
                )

            existing_start_key = _key(existing_start, existing_zone or "UTC")
            existing_end_key = _key(existing_end, existing_zone or "UTC")
            effective_start_key = _key(effective_start, provisional_query_timezone)
            effective_end_key = _key(effective_end, provisional_query_timezone)

            # is_all_day changes the event's effective span even when the
            # literal start/end clock values don't move (e.g. turning a
            # 30-minute meeting into an all-day one) - treat that the same
            # as a moved window for deciding whether to check the
            # organizer at all (organizer conflicts are always safe to
            # re-check: excluded by event id, not by window).
            literal_window_changed = (effective_start_key, effective_end_key) != (
                existing_start_key,
                existing_end_key,
            )
            existing_is_all_day = bool(existing.get("isAllDay"))
            effective_is_all_day = (
                is_all_day if is_all_day is not None else existing_is_all_day
            )
            check_organizer = literal_window_changed or (
                effective_is_all_day != existing_is_all_day
            )

            # An all-day event/toggle occupies the *whole* calendar day(s)
            # it lands on, not just the literal clock-time slot it was
            # given - widen the actual query window to that before
            # checking, or a conflict elsewhere that day (or this event's
            # own now-irrelevant narrow slot) would be missed/misjudged.
            # Outside that case, query_start/query_end are exactly
            # effective_start/effective_end, so their keys are exactly
            # effective_start_key/effective_end_key already computed above
            # - no need to recompute them from scratch.
            query_start, query_end = effective_start, effective_end
            query_start_key, query_end_key = effective_start_key, effective_end_key
            if effective_is_all_day and effective_start and effective_end:
                query_start, _ = _naive_day_bounds(effective_start)
                end_of_its_day, next_day_start = _naive_day_bounds(effective_end)
                # effective_end may already BE a well-formed exclusive
                # day boundary (exactly midnight, e.g. an existing
                # all-day event's own end untouched by this update) -
                # naive_day_bounds always treats its input as a day that
                # needs widening to [that midnight, next midnight), so
                # applying it here unconditionally would push an
                # already-correct boundary one whole day too far,
                # falsely conflicting with a following-day event. Only
                # push to the next midnight when effective_end isn't
                # already sitting exactly on one.
                query_end = (
                    effective_end
                    if datetime.fromisoformat(effective_end)
                    == datetime.fromisoformat(end_of_its_day)
                    else next_day_start
                )
                query_start_key = _key(query_start, provisional_query_timezone)
                query_end_key = _key(query_end, provisional_query_timezone)

            all_conflicts: list[dict[str, Any]] = []

            # window_delta_segments is safe to call even when existing_zone
            # is unknown: existing_start_key/existing_end_key and
            # query_start_key/query_end_key all fall back to the same "UTC"
            # assumption in that case (see the `_key` closure above), and a
            # uniform (even if wrong) offset applied to both sides of the
            # comparison can't change which portion is genuinely new
            # territory - only the *absolute* query values sent to Graph
            # below would be wrong, which is exactly what the fatal check
            # right after this guards against, and only when a query is
            # actually about to use them.
            retained_segments = (
                _window_delta_segments(
                    existing_start_key, existing_end_key, query_start_key, query_end_key
                )
                if retained_attendees_raw
                else []
            )

            any_query_needed = (
                query_start
                and query_end
                and (check_organizer or added_attendees or retained_segments)
            )
            if any_query_needed:
                # Only now, with a query actually about to run, does a
                # missing existing-event timezone (relevant only when
                # neither start_datetime nor end_datetime was given)
                # become fatal rather than a moot point - a same-attendees,
                # same-effective-window resubmission (retained_segments
                # empty, no organizer/added-attendee query needed) must
                # stay a no-op PATCH even if the existing event happens to
                # be missing timezone info that would only matter for a
                # query nothing here ends up needing.
                if (
                    start_datetime is None
                    and end_datetime is None
                    and not existing_zone
                ):
                    raise ValueError(
                        "Existing event has no timeZone on its start "
                        "time; cannot safely check conflicts for this "
                        "update."
                    )

            if query_start and query_end and (check_organizer or added_attendees):
                # A newly-added attendee has no footprint on this event
                # at all, so the FULL query window is safe (and
                # necessary) to check for them - same call also covers
                # the organizer, who's excluded by event id rather than
                # by window.
                conflicts, unchecked = _find_conflicts(
                    query_start,
                    query_end,
                    provisional_query_timezone,
                    added_attendees,
                    exclude_event_id=event_id,
                    check_organizer=check_organizer,
                )
                all_conflicts.extend(conflicts)
                unchecked_attendees.extend(unchecked)

            # A retained attendee's own busy block for THIS event covers
            # the entire OLD (unwidened, literal) window - querying that
            # overlap can't tell "busy because of this event" from a real
            # conflict. Only the portion of the query window that's
            # genuinely new territory (window_delta_segments - empty for
            # an unchanged/shrunk window, up to two segments for a
            # partial nudge, or the whole query window for a disjoint
            # move or an is_all_day widening) can hide a real conflict
            # for them.
            if retained_segments:
                for seg_start, seg_end in retained_segments:
                    seg_start_naive = seg_start.astimezone(dt_timezone.utc).replace(
                        tzinfo=None
                    )
                    seg_end_naive = seg_end.astimezone(dt_timezone.utc).replace(
                        tzinfo=None
                    )
                    conflicts, unchecked = _find_conflicts(
                        seg_start_naive.isoformat(),
                        seg_end_naive.isoformat(),
                        "UTC",
                        retained_attendees_raw,
                        exclude_event_id=event_id,
                        check_organizer=False,
                    )
                    all_conflicts.extend(conflicts)
                    unchecked_attendees.extend(unchecked)

            if all_conflicts:
                # narrows for mypy: a conflict can only have been found by
                # a query that ran, and every query above only runs when
                # query_start/query_end (or the delta segments derived
                # from them) are real values.
                assert query_start is not None
                assert query_end is not None
                return _conflict_response(
                    all_conflicts,
                    unchecked_attendees,
                    query_start,
                    query_end,
                )

        result = _graph_request(
            "PATCH",
            f"/me/events/{quote(event_id, safe='')}",
            body=payload,
        )
        return _success(event=result, **_unchecked_extra(unchecked_attendees))
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
