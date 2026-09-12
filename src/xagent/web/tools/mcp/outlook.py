import json
import logging
import os
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import requests
from dateutil import parser as _date_parser
from mcp.server.fastmcp import FastMCP

from .utils import InsufficientScopeError
from .utils import conflict_response as _conflict_response
from .utils import datetime_key_for_comparison as _datetime_key_for_comparison
from .utils import incomplete_check_response as _incomplete_check_response
from .utils import merge_scope_error as _merge_scope_error
from .utils import naive_day_bounds as _naive_day_bounds
from .utils import normalize_addresses as _normalize_addresses
from .utils import offset_datetime_string as _offset_datetime_string
from .utils import reject_reversed_window as _reject_reversed_window
from .utils import resolve_zoneinfo as _resolve_zoneinfo
from .utils import setup_proxy_env
from .utils import timezones_could_differ as _timezones_could_differ

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


class _ConflictCheckIncompleteError(RuntimeError):
    """An availability scan that stopped after finding partial results."""

    def __init__(
        self,
        message: str,
        conflicts: list[dict[str, Any]],
        unchecked_attendees: list[str],
    ) -> None:
        super().__init__(message)
        self.conflicts = conflicts
        self.unchecked_attendees = unchecked_attendees


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


def _next_link_path(next_link: Any) -> str:
    """Strip GRAPH_BASE_URL from an @odata.nextLink so it can be re-issued
    through `_graph_request` as a plain path+query (the link is always an
    absolute URL; `_graph_request` builds its own URL as
    ``f"{GRAPH_BASE_URL}{path}"``, so passing the absolute link verbatim as
    `path` would double up the host instead of following it)."""
    if not isinstance(next_link, str) or not next_link.startswith(f"{GRAPH_BASE_URL}/"):
        raise ValueError("Outlook returned an invalid calendarView next link.")
    # Decode before splitting so an encoded slash cannot hide a dot segment
    # inside one raw segment (for example ``%2e%2e%2fusers``).
    if any(
        segment in {".", ".."}
        for segment in unquote(urlsplit(next_link).path).split("/")
    ):
        raise ValueError("Outlook returned an invalid calendarView next link.")
    return next_link[len(GRAPH_BASE_URL) :]


def _is_midnight_in_timezone(value: str, timezone: str) -> bool:
    """Whether ``value`` represents midnight in the event timezone."""
    parsed: datetime = _date_parser.isoparse(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(_resolve_zoneinfo(timezone))
    return not (parsed.hour or parsed.minute or parsed.second or parsed.microsecond)


def _naive_datetime_in_timezone(value: str, timezone: str) -> str:
    """Return an Outlook dateTimeTimeZone clock value in ``timezone``.

    Graph represents these values as a naive local datetime plus a separate
    timeZone field. Preserve already-naive wall-clock input, but convert an
    offset-bearing instant into the requested zone before removing its offset.
    """
    normalized = value.strip()
    if (
        len(normalized) < 19
        or normalized[4] != "-"
        or normalized[7] != "-"
        or normalized[10] not in {"T", "t"}
        or normalized[13] != ":"
        or normalized[16] != ":"
    ):
        raise ValueError(
            "Outlook event datetimes must use the extended ISO format "
            "YYYY-MM-DDTHH:MM:SS, optionally followed by fractional seconds "
            "and a UTC offset or Z suffix."
        )
    try:
        parsed: datetime = _date_parser.isoparse(normalized)
    except ValueError as exc:
        raise ValueError(
            "Outlook event datetimes must use the extended ISO format "
            "YYYY-MM-DDTHH:MM:SS, optionally followed by fractional seconds "
            "and a UTC offset or Z suffix."
        ) from exc
    if parsed.tzinfo is None:
        return normalized
    return (
        parsed.astimezone(_resolve_zoneinfo(timezone)).replace(tzinfo=None).isoformat()
    )


def _reject_invalid_create_window(
    start_datetime: str, end_datetime: str, *, is_all_day: bool
) -> None:
    try:
        _reject_reversed_window(start_datetime, end_datetime)
    except ValueError as exc:
        if is_all_day:
            raise ValueError(
                "end_datetime is exclusive for an all-day event and must "
                "be after start_datetime; use the following day's midnight "
                "as the end of a one-day event."
            ) from exc
        raise


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

    The organizer results are accepted as Graph's own overlap decision for
    this half-open window; this connector does not apply a second local
    overlap calculation at the exact start/end edges. Each returned
    boundary retains its response timezone below so organizer-local and
    getSchedule UTC values cannot be mistaken for the same clock basis.

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
            try:
                calendar_view = _graph_request(
                    "GET",
                    next_path if next_path is not None else "/me/calendarView",
                    params=params if next_path is None else None,
                    extra_headers={"Prefer": f'outlook.timezone="{timezone}"'},
                )
            except _GraphRequestError as exc:
                if exc.status_code == 403:
                    raise InsufficientScopeError(
                        "Missing the calendars.read permission needed to check "
                        "organizer availability - reconnecting the Outlook "
                        "connector may grant it; if it already has calendar "
                        "access, an org-level policy may be blocking the call. "
                        "This is a credential or policy error, not a scheduling "
                        "conflict.",
                        conflicts,
                        unchecked_attendees + attendees,
                    ) from exc
                raise
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
                        "start_timezone": start.get("timeZone") or timezone,
                        "end_timezone": end.get("timeZone") or timezone,
                    }
                )
            next_link = calendar_view.get("@odata.nextLink")
            if next_link is None:
                break
            try:
                next_path = _next_link_path(next_link)
            except ValueError as exc:
                raise _ConflictCheckIncompleteError(
                    str(exc), conflicts, unchecked_attendees + attendees
                ) from exc
        else:
            raise _ConflictCheckIncompleteError(
                "Outlook calendar conflict check exceeded the pagination limit; "
                "availability could not be verified completely.",
                conflicts,
                unchecked_attendees + attendees,
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
                extra_headers={"Prefer": f'outlook.timezone="{timezone}"'},
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
            schedule_items = entry.get("scheduleItems") or []
            found_busy_item = False
            for item in schedule_items:
                # See the matching comment on the organizer-side loop
                # above: "unknown" is unclassified, not confirmed-free,
                # so it's treated the same way here for consistency -
                # missing a real conflict is worse than occasionally
                # over-flagging one the caller can dismiss with
                # ignore_conflicts.
                if item.get("status") in ("busy", "tentative", "oof", "unknown"):
                    found_busy_item = True
                    start = item.get("start") or {}
                    end = item.get("end") or {}
                    conflicts.append(
                        {
                            "calendar": email,
                            "summary": None,
                            "start": start.get("dateTime"),
                            "end": end.get("dateTime"),
                            "start_timezone": start.get("timeZone") or timezone,
                            "end_timezone": end.get("timeZone") or timezone,
                        }
                    )
            availability_view = entry.get("availabilityView")
            if not found_busy_item and (
                not isinstance(availability_view, str)
                or not availability_view
                or any(slot not in {"0", "4"} for slot in availability_view)
            ):
                # scheduleItems can be withheld even though availabilityView
                # still reports a busy slot. Graph uses "4" for
                # workingElsewhere, which follows the same non-blocking policy
                # as detailed schedule items above. Without item boundaries for
                # any other non-free state there is
                # not enough detail to construct a normal conflict entry, but
                # treating the attendee as free would permit a double booking.
                unchecked_attendees.append(email)

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
    The organizer's calendar is always checked for scheduling conflicts.
    attendees, if given, are invited by email (Graph emails them the invite)
    and their schedules are checked too; a conflict returns status="conflict"
    instead of creating the event. Pass ignore_conflicts=True to create it
    anyway once the user has explicitly confirmed a conflict is fine.
    For an all-day event, end_datetime is an exclusive boundary: use the
    following day's midnight as the end of a one-day event.
    """
    try:
        # Timezone validity is part of the write contract, independent of
        # whether the caller explicitly bypasses availability checks.
        _resolve_zoneinfo(timezone)
        normalized_attendees = _normalize_addresses(attendees) if attendees else []

        # Retain the caller-level ordering check before all-day normalization:
        # two reversed times on the same date must not become a valid full-day
        # window merely because both are widened to date boundaries.
        _reject_invalid_create_window(
            start_datetime, end_datetime, is_all_day=is_all_day
        )

        # Graph's dateTimeTimeZone shape carries a naive wall-clock value and
        # its timezone separately. Convert offset-bearing inputs to that shape
        # before comparing, querying, or writing them. This also makes a mixed
        # aware/naive pair comparable instead of letting it bypass the ordering
        # check when Python refuses to compare the two datetime kinds.
        effective_start = _naive_datetime_in_timezone(start_datetime, timezone)
        effective_end = _naive_datetime_in_timezone(end_datetime, timezone)

        # Graph additionally requires all-day boundaries to be midnight in the
        # same timezone. Normalize both the availability query and the eventual
        # write, including when ignore_conflicts bypasses the query.
        if is_all_day:
            effective_start, _ = _naive_day_bounds(start_datetime, timezone)
            end_of_its_day, next_day_start = _naive_day_bounds(end_datetime, timezone)
            end_is_midnight = _is_midnight_in_timezone(end_datetime, timezone)
            # end_datetime is an exclusive date boundary. A same-day time
            # range still denotes one all-day event, while a later date is
            # already the exclusive boundary even if its clock is non-midnight.
            effective_end = (
                end_of_its_day
                if end_is_midnight or end_of_its_day != effective_start
                else next_day_start
            )

        # The raw comparison is deliberately unable to compare a mixed
        # aware/naive pair. Recheck after normalization so that case cannot
        # bypass the invariant.
        _reject_invalid_create_window(
            effective_start, effective_end, is_all_day=is_all_day
        )

        unchecked_attendees: list[str] = []
        check_error: str | None = None
        if not ignore_conflicts:
            try:
                conflicts, unchecked_attendees = _find_conflicts(
                    effective_start, effective_end, timezone, normalized_attendees
                )
            except InsufficientScopeError as exc:
                check_error = str(exc)
                conflicts, unchecked_attendees = _merge_scope_error(exc, [], [])
            except _ConflictCheckIncompleteError as exc:
                check_error = str(exc)
                conflicts = exc.conflicts
                unchecked_attendees = exc.unchecked_attendees
            if conflicts:
                return _conflict_response(
                    conflicts,
                    unchecked_attendees,
                    effective_start,
                    effective_end,
                    check_error=check_error,
                )
            if check_error or unchecked_attendees:
                return _incomplete_check_response(
                    unchecked_attendees,
                    effective_start,
                    effective_end,
                    message=check_error,
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
    timezone: str | None = None,
    body: str | None = None,
    location: str | None = None,
    attendees: list[str] | str | None = None,
    is_all_day: bool | None = None,
    ignore_conflicts: bool = False,
) -> str:
    """Update an existing Outlook calendar event.
    If the update moves the event to a new time or changes its all-day
    span, the signed-in calendar is checked for conflicts before the write;
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
        touches_schedule = (
            start_datetime is not None
            or end_datetime is not None
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
        needs_existing = single_boundary_update or (
            not ignore_conflicts and touches_schedule
        )
        if needs_existing:
            existing = _graph_request(
                "GET",
                f"/me/events/{quote(event_id, safe='')}",
                params={
                    "$select": "start,end,attendees,isAllDay,originalStartTimeZone"
                },
            )

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
            existing_timezone_resolves = False
            if existing_timezone and not existing_timezone.startswith("tzone://"):
                try:
                    _resolve_zoneinfo(existing_timezone)
                    existing_timezone_resolves = True
                except ValueError:
                    # An unmapped/legacy Windows zone id (the same gap
                    # `_WINDOWS_TO_IANA` can't ever fully close) - using it
                    # as existing_zone anyway would crash the later _key()
                    # comparison instead of leaving this branch's original
                    # "may not need this value to be exact" guarantee
                    # intact, so fall back to the UTC-normalized field
                    # exactly as if originalStartTimeZone were absent.
                    pass
            if existing_timezone_resolves:
                assert existing_timezone is not None  # narrows for mypy
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
        if attendees is not None:
            payload["attendees"] = _attendee_list(attendees)
        if is_all_day is not None:
            payload["isAllDay"] = is_all_day

        if not payload:
            raise ValueError("at least one field must be provided to update the event")

        # Checked unconditionally, not gated on ignore_conflicts - this is
        # basic input sanity (a window a provider API should never be
        # asked to write), not a conflict-check decision the caller can
        # opt out of. Matches google_calendar_update_events, and this
        # tool's own create path, both of which validate regardless of
        # ignore_conflicts too. Only worth checking when this call is
        # actually about to write a (possibly partly-existing) window -
        # an attendees/subject-only edit that never moves either boundary
        # would otherwise re-validate the event's already-stored,
        # unchanged start/end and could reject an unrelated field edit
        # over pre-existing data this call never touches.
        existing_end = existing_end_field.get("dateTime")
        existing_start = existing_start_field.get("dateTime")
        effective_start = start_datetime or existing_start
        effective_end = end_datetime or existing_end
        if (start_datetime or end_datetime) and effective_start and effective_end:
            _reject_reversed_window(effective_start, effective_end)

        unchecked_attendees: list[str] = []
        if not ignore_conflicts and touches_schedule:
            query_timezone = (
                resolved_timezone
                if (start_datetime is not None or end_datetime is not None)
                else (existing_zone or "UTC")
            )

            def _key(value: str | None, tz_name: str) -> datetime | str | None:
                return _datetime_key_for_comparison(
                    _offset_datetime_string(value, tz_name) if value else None
                )

            existing_start_key = _key(existing_start, existing_zone or "UTC")
            existing_end_key = _key(existing_end, existing_zone or "UTC")
            effective_start_key = _key(effective_start, query_timezone)
            effective_end_key = _key(effective_end, query_timezone)
            existing_is_all_day = bool(existing.get("isAllDay"))
            effective_is_all_day = (
                is_all_day if is_all_day is not None else existing_is_all_day
            )
            check_organizer = (effective_start_key, effective_end_key) != (
                existing_start_key,
                existing_end_key,
            ) or effective_is_all_day != existing_is_all_day

            query_start, query_end = effective_start, effective_end
            if effective_is_all_day and effective_start and effective_end:
                query_start, _ = _naive_day_bounds(effective_start, query_timezone)
                end_of_its_day, next_day_start = _naive_day_bounds(
                    effective_end, query_timezone
                )
                end_is_midnight = _is_midnight_in_timezone(
                    effective_end, query_timezone
                )
                query_end = end_of_its_day if end_is_midnight else next_day_start

            if check_organizer:
                if not query_start or not query_end:
                    raise ValueError(
                        "Existing event has no complete time window; cannot safely "
                        "check conflicts for this update."
                    )
                if (
                    start_datetime is None
                    and end_datetime is None
                    and not existing_zone
                ):
                    raise ValueError(
                        "Existing event has no timeZone on its start time; cannot "
                        "safely check conflicts for this update."
                    )
                conflicts, unchecked_attendees = _find_conflicts(
                    query_start,
                    query_end,
                    query_timezone,
                    [],
                    exclude_event_id=event_id,
                    check_organizer=True,
                )
                if conflicts:
                    return _conflict_response(
                        conflicts, unchecked_attendees, query_start, query_end
                    )

        result = _graph_request(
            "PATCH",
            f"/me/events/{quote(event_id, safe='')}",
            body=payload,
        )
        extra = (
            {"unchecked_attendees": unchecked_attendees} if unchecked_attendees else {}
        )
        return _success(event=result, **extra)
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
