import json
import logging
import os
from typing import Any
from urllib.parse import quote

import requests
from dateutil import parser as _date_parser
from mcp.server.fastmcp import FastMCP

from .utils import InsufficientScopeError
from .utils import conflict_response as _conflict_response
from .utils import merge_scope_error as _merge_scope_error
from .utils import naive_day_bounds as _naive_day_bounds
from .utils import normalize_addresses as _normalize_addresses
from .utils import offset_datetime_string as _offset_datetime_string
from .utils import reject_reversed_window as _reject_reversed_window
from .utils import setup_proxy_env
from .utils import unchecked_extra as _unchecked_extra

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
    the whole call raises `InsufficientScopeError` (carrying whatever was
    already confirmed in `conflicts`/`unchecked_attendees` before the
    error) instead of degrading to unchecked and returning normally -
    that's OUR OWN credential/policy problem, not a per-attendee
    visibility gap, and writing an event whose availability was never
    actually checked would defeat the point of this feature. Every entry
    that does end up in `unchecked_attendees` on a normal return is the
    self-explanatory kind (absent from the response, or its own
    per-attendee error).
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
        else:
            raise RuntimeError(
                "Outlook calendar conflict check exceeded the pagination limit; "
                "the event was not created because availability could not be "
                "verified completely."
            )

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
                # over a possible conflict. Carrying `conflicts` (e.g. an
                # organizer conflict already found above, before this
                # batch ever ran) lets a caller still report it rather
                # than silently discarding a known problem just because
                # this later, unrelated check also failed.
                raise InsufficientScopeError(
                    "Missing the calendars.read/schedule permission "
                    "needed to check attendee availability - "
                    "reconnecting the Outlook connector may grant it; if "
                    "the connector already has calendar access, this is "
                    "more likely an org-level policy blocking this call, "
                    "which a reconnect won't fix. Pass "
                    "ignore_conflicts=true if the user has confirmed "
                    "they want to proceed without this check.",
                    conflicts,
                    # Every attendee from this failed batch onward is
                    # unchecked - every remaining batch would hit this
                    # same scope error too, so there's nothing left to
                    # gain by attempting them.
                    unchecked_attendees + attendees[offset:],
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

        # Graph requires all-day event boundaries to be midnight in the
        # same timezone. Normalize both the availability query and the
        # eventual write, including when ignore_conflicts bypasses the query.
        effective_start, effective_end = start_datetime, end_datetime
        if is_all_day:
            effective_start, _ = _naive_day_bounds(start_datetime)
            end_of_its_day, next_day_start = _naive_day_bounds(end_datetime)
            parsed_end = _date_parser.isoparse(end_datetime)
            end_is_midnight = (
                parsed_end.hour == 0
                and parsed_end.minute == 0
                and parsed_end.second == 0
                and parsed_end.microsecond == 0
            )
            effective_end = end_of_its_day if end_is_midnight else next_day_start

        unchecked_attendees: list[str] = []
        if not ignore_conflicts:
            try:
                conflicts, unchecked_attendees = _find_conflicts(
                    effective_start, effective_end, timezone, normalized_attendees
                )
            except InsufficientScopeError as exc:
                conflicts, unchecked_attendees = _merge_scope_error(exc, [], [])
            if conflicts:
                return _conflict_response(
                    conflicts,
                    unchecked_attendees,
                    effective_start,
                    effective_end,
                )

        payload: dict[str, Any] = {
            "subject": subject,
            "start": {"dateTime": effective_start, "timeZone": timezone},
            "end": {"dateTime": effective_end, "timeZone": timezone},
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
    timezone: str = "UTC",
    body: str | None = None,
    location: str | None = None,
    attendees: list[str] | str | None = None,
    is_all_day: bool | None = None,
) -> str:
    """Update an existing Outlook calendar event."""
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
