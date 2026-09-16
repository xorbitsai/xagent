import json
import logging
import os
from datetime import datetime
from datetime import timezone as dt_timezone
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import requests
from dateutil import parser as _date_parser
from dateutil import tz as _date_tz
from mcp.server.fastmcp import FastMCP

from .utils import InsufficientScopeError
from .utils import attendees_were_given as _attendees_were_given
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

    Pairing a caller-supplied zone name with the UTC clock value unchanged
    would silently mislabel the instant by the zone's UTC offset. Reject a
    conversion into a repeated daylight-saving wall time because Graph's
    naive dateTime plus timeZone shape cannot preserve which fold represented
    the snapshot instant.

    The sole caller uses a plain event GET without a Prefer header, so a naive
    response value must be UTC. Reject a contradictory zone label or malformed
    timestamp instead of silently using an unconverted value downstream.
    """
    date_time = field.get("dateTime")
    if not date_time:
        return field
    zone = _resolve_zoneinfo(zone_name, allow_windows_names=True)
    try:
        parsed = _date_parser.isoparse(date_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Outlook returned an invalid event boundary datetime: {date_time!r}."
        ) from exc
    if parsed.tzinfo is None and field.get("timeZone") not in (None, "UTC"):
        raise ValueError(
            "Outlook returned a non-UTC event boundary from a plain event GET; "
            "the existing window cannot be converted safely."
        )
    utc_instant = (
        parsed.replace(tzinfo=dt_timezone.utc)
        if parsed.tzinfo is None
        else parsed.astimezone(dt_timezone.utc)
    )
    local_instant = utc_instant.astimezone(zone)
    if _date_tz.datetime_ambiguous(local_instant):
        raise ValueError(
            "Outlook returned an event boundary whose local time is ambiguous "
            f"in timezone {zone_name!r} because of a daylight-saving transition; "
            "provide both boundaries in an unambiguous timezone such as UTC."
        )
    return {
        "dateTime": local_instant.replace(tzinfo=None).isoformat(),
        "timeZone": zone_name,
    }


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
        parsed = parsed.astimezone(
            _resolve_zoneinfo(timezone, allow_windows_names=True)
        )
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
    zone = _resolve_zoneinfo(timezone, allow_windows_names=True)
    if parsed.tzinfo is None:
        localized = parsed.replace(tzinfo=zone)
        if not _date_tz.datetime_exists(localized):
            raise ValueError(
                f"{value!r} does not exist in timezone {timezone!r} because of "
                "a daylight-saving transition"
            )
        result = normalized
    else:
        localized = parsed.astimezone(zone)
        result = localized.replace(tzinfo=None).isoformat()
    if _date_tz.datetime_ambiguous(localized):
        raise ValueError(
            f"{value!r} is ambiguous in timezone {timezone!r} because of a "
            "daylight-saving transition"
        )
    return result


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


def _normalize_all_day_window(
    start_datetime: str, end_datetime: str, timezone: str
) -> tuple[str, str]:
    """Normalize an all-day window to Graph's exclusive midnight bounds."""
    effective_start, _ = _naive_day_bounds(
        start_datetime, timezone, allow_windows_names=True
    )
    end_of_its_day, next_day_start = _naive_day_bounds(
        end_datetime, timezone, allow_windows_names=True
    )
    end_is_midnight = _is_midnight_in_timezone(end_datetime, timezone)
    effective_end = (
        end_of_its_day
        if end_is_midnight or end_of_its_day != effective_start
        else next_day_start
    )
    return effective_start, effective_end


# Defensive cap on calendarView pages followed for one signed-in-calendar
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
    organizer_calendar_label: str = "organizer",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Check the signed-in calendar (when check_organizer) plus each
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

    `organizer_calendar_label` controls how the `/me/calendarView` source is
    identified in conflicts. Create uses the default because the signed-in
    user is creating the event; update uses `signed_in_calendar` because an
    event in `/me/events` may have been organized by someone else.

    The signed-in calendar results are accepted as Graph's overlap decision for
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
        offset_start = _offset_datetime_string(
            start_datetime, timezone, allow_windows_names=True
        )
        offset_end = _offset_datetime_string(
            end_datetime, timezone, allow_windows_names=True
        )
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
                        "signed-in calendar availability - reconnecting the Outlook "
                        "connector may grant it; if it already has calendar "
                        "access, an org-level policy may be blocking the call. "
                        "This is a credential or policy error, not a scheduling "
                        "conflict. Pass ignore_conflicts=true if the user has "
                        "confirmed they want to proceed without this check.",
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
                        "calendar": organizer_calendar_label,
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
                # signed-in-calendar conflict already found above, before this
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
                            "summary": item.get("subject"),
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
                or any(slot != "0" for slot in availability_view)
            ):
                # scheduleItems can be withheld even though availabilityView
                # still reports a busy slot. Graph folds workingElsewhere into
                # the documented "0" (free) availability code. Without item
                # boundaries for any non-free state there is
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
                    "isAllDay,type,seriesMasterId,bodyPreview,webLink"
                ),
            }
        else:
            path = "/me/events"
            params = {
                "$top": top,
                "$orderby": "start/dateTime",
                "$select": (
                    "id,subject,start,end,location,organizer,attendees,"
                    "isAllDay,type,seriesMasterId,bodyPreview,webLink"
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
    The organizer's primary/default calendar is always checked for scheduling
    conflicts, matching Outlook's free/busy availability semantics.
    attendees, if given, are invited by email (Graph emails them the invite)
    and their schedules are checked too; a conflict returns status="conflict"
    instead of creating the event. Pass ignore_conflicts=True to create it
    anyway once the user has explicitly confirmed a conflict is fine.
    Timed values use extended ISO format (YYYY-MM-DDTHH:MM:SS with an optional
    offset). For an all-day event, bare YYYY-MM-DD dates are also accepted and
    end_datetime is an exclusive boundary: use the following date as the end
    of a one-day event.
    """
    try:
        # Timezone validity is part of the write contract, independent of
        # whether the caller explicitly bypasses availability checks.
        _resolve_zoneinfo(timezone, allow_windows_names=True)
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
        # Graph additionally requires all-day boundaries to be midnight in the
        # same timezone. Normalize both the availability query and the eventual
        # write, including when ignore_conflicts bypasses the query.
        if is_all_day:
            effective_start, effective_end = _normalize_all_day_window(
                start_datetime, end_datetime, timezone
            )
        else:
            effective_start = _naive_datetime_in_timezone(start_datetime, timezone)
            effective_end = _naive_datetime_in_timezone(end_datetime, timezone)

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
    If the update moves the event, changes its all-day span, or adds
    attendees, the affected calendars are checked before the write;
    pass ignore_conflicts=True to skip the check once the user has
    explicitly confirmed a conflict is fine. Editing other fields (subject,
    body, location) without moving the event is never blocked.
    attendees, if given, fully replaces the event's attendee list: any
    address already on the event that's left out is removed, and passing
    an explicit empty list clears every attendee. Leave attendees unset to
    keep the existing list untouched. timezone is only valid together with
    start_datetime or end_datetime; omit it for an attendee-only update.
    Passing only one of start_datetime/end_datetime nudges a timed event
    boundary while keeping the other as-is; timezone is required and describes
    the changed boundary. Passing both together fully replaces the window, and
    timezone then describes both new values (defaulting to UTC if also left
    unset). Changing to or resizing an all-day window requires both boundaries
    and an explicit timezone because Graph does not expose a reliable current
    boundary zone; all-day boundaries are normalized to midnight values in one
    shared timezone.
    New attendees are checked across the complete effective window. Retained
    attendees are checked only across newly introduced portions of a changed
    window, avoiding the event's own existing busy block. A detected conflict
    returns status="conflict" without updating the event. An incomplete
    availability check returns status="conflict_check_incomplete", or
    status="error" for a missing OAuth scope, and also skips the update.
    The check and PATCH are separate Graph calls, so availability can still
    change between them.
    Schedule changes to recurring series masters cannot be checked safely as
    one scalar window and are rejected unless ignore_conflicts=True. Update a
    specific occurrence when possible.
    Because a plain Graph GET does not expose a reliable timezone for an
    existing all-day window, adding attendees to one requires resubmitting both
    boundaries with an explicit timezone. Moving or expanding an all-day event
    into new dates while retaining attendees fails closed unless
    ignore_conflicts=True; unchanged and shrunken date windows remain safe.
    """
    try:
        attendees_given = _attendees_were_given(attendees)
        touches_schedule = (
            start_datetime is not None
            or end_datetime is not None
            or attendees_given
            or is_all_day is not None
        )
        single_boundary_update = (start_datetime is not None) != (
            end_datetime is not None
        )
        both_boundaries_supplied = (
            start_datetime is not None and end_datetime is not None
        )
        if timezone is not None and touches_schedule:
            _resolve_zoneinfo(timezone, allow_windows_names=True)
            if start_datetime is None and end_datetime is None:
                raise ValueError(
                    "timezone can only be supplied when start_datetime or "
                    "end_datetime is also supplied; omit it for a flag-only "
                    "update so the event's existing timezone is reused."
                )

        if single_boundary_update:
            supplied_boundary = (
                start_datetime if start_datetime is not None else end_datetime
            )
            assert supplied_boundary is not None
            if timezone is None:
                raise ValueError(
                    "timezone is required when updating only start_datetime or "
                    "only end_datetime because Graph exposes original creation "
                    "zones, not the boundary's current timezone."
                )
            _naive_datetime_in_timezone(supplied_boundary, timezone)
        if both_boundaries_supplied:
            assert start_datetime is not None and end_datetime is not None
            if not start_datetime.strip() or not end_datetime.strip():
                raise ValueError(
                    "Outlook event datetimes must use the extended ISO format "
                    "YYYY-MM-DDTHH:MM:SS, optionally followed by fractional "
                    "seconds and a UTC offset or Z suffix."
                )
            _reject_invalid_create_window(
                start_datetime,
                end_datetime,
                is_all_day=is_all_day is True,
            )

        # Existing state is needed for conflict checks and for enforcing the
        # all-day and recurrence restrictions even when conflict checks are
        # explicitly bypassed.
        existing: dict[str, Any] = {}
        needs_existing = touches_schedule
        if needs_existing:
            existing = _graph_request(
                "GET",
                f"/me/events/{quote(event_id, safe='')}",
                params={"$select": "start,end,attendees,isAllDay,type"},
            )

        snapshot_start_field = existing.get("start") or {}
        snapshot_end_field = existing.get("end") or {}
        existing_attendees_raw = _normalize_addresses(
            [
                attendee["emailAddress"]["address"]
                for attendee in (existing.get("attendees") or [])
                if (attendee.get("emailAddress") or {}).get("address", "").strip()
            ]
        )
        existing_attendee_emails = {
            address.lower() for address in existing_attendees_raw
        }

        existing_is_all_day = bool(existing.get("isAllDay"))
        effective_is_all_day = (
            is_all_day if is_all_day is not None else existing_is_all_day
        )
        if effective_is_all_day and both_boundaries_supplied and timezone is None:
            raise ValueError(
                "timezone is required when replacing an all-day event window "
                "because its calendar dates cannot safely default to UTC."
            )
        if effective_is_all_day != existing_is_all_day and not both_boundaries_supplied:
            raise ValueError(
                "Changing is_all_day requires both start_datetime and "
                "end_datetime in one shared timezone; deriving boundaries from "
                "an earlier event snapshot could overwrite a concurrent schedule "
                "change."
            )
        if existing_is_all_day and single_boundary_update:
            raise ValueError(
                "Updating an existing all-day event requires both start_datetime "
                "and end_datetime with one shared timezone because Graph does "
                "not expose the boundaries' reliable current timezone."
            )

        schedule_semantics_supplied = (
            start_datetime is not None
            or end_datetime is not None
            or effective_is_all_day != existing_is_all_day
        )
        if (
            not ignore_conflicts
            and existing.get("type") == "seriesMaster"
            and schedule_semantics_supplied
        ):
            raise ValueError(
                "Cannot safely conflict-check a schedule or timezone change to "
                "a recurring series master because it can affect multiple "
                "occurrences. Update a specific occurrence, or pass "
                "ignore_conflicts=True only after the user confirms every "
                "occurrence is safe."
            )

        resolved_timezone = timezone or "UTC"
        existing_start_field = snapshot_start_field
        existing_end_field = snapshot_end_field
        if not existing_is_all_day and single_boundary_update:
            # A plain GET returns timed boundaries in UTC. Convert only the
            # untouched snapshot boundary to the caller's explicit comparison
            # zone; the supplied boundary is normalized separately below and
            # is the only one included in the PATCH.
            if start_datetime is None:
                existing_start_field = _utc_field_in_zone(
                    existing_start_field, resolved_timezone
                )
            else:
                existing_end_field = _utc_field_in_zone(
                    existing_end_field, resolved_timezone
                )
        existing_zone = existing_start_field.get("timeZone") or "UTC"

        existing_end = existing_end_field.get("dateTime")
        existing_start = existing_start_field.get("dateTime")
        if single_boundary_update and (not existing_start or not existing_end):
            raise ValueError(
                "Existing event has no complete time window; cannot safely "
                "validate a single-boundary update."
            )
        query_timezone = (
            resolved_timezone
            if (start_datetime is not None or end_datetime is not None)
            else existing_zone
        )
        if start_datetime is not None or end_datetime is not None or is_all_day is True:
            _resolve_zoneinfo(query_timezone, allow_windows_names=True)

        raw_effective_start = (
            start_datetime if start_datetime is not None else existing_start
        )
        raw_effective_end = end_datetime if end_datetime is not None else existing_end
        if (
            (start_datetime is not None or end_datetime is not None)
            and raw_effective_start
            and raw_effective_end
        ):
            _reject_invalid_create_window(
                raw_effective_start,
                raw_effective_end,
                is_all_day=effective_is_all_day,
            )
        effective_start = raw_effective_start
        effective_end = raw_effective_end
        if effective_is_all_day and raw_effective_start and raw_effective_end:
            effective_start, effective_end = _normalize_all_day_window(
                raw_effective_start, raw_effective_end, query_timezone
            )
        else:
            if start_datetime is not None:
                effective_start = _naive_datetime_in_timezone(
                    start_datetime, query_timezone
                )
            if end_datetime is not None:
                effective_end = _naive_datetime_in_timezone(
                    end_datetime, query_timezone
                )

        payload: dict[str, Any] = {}
        if subject is not None:
            payload["subject"] = subject
        if start_datetime is not None:
            payload["start"] = {
                "dateTime": effective_start,
                "timeZone": query_timezone,
            }
        if end_datetime is not None:
            payload["end"] = {
                "dateTime": effective_end,
                "timeZone": query_timezone,
            }
        if body is not None:
            payload["body"] = _message_body(body, "text")
        if location is not None:
            payload["location"] = {"displayName": location}
        desired_addresses = (
            _normalize_addresses(attendees) if attendees_given and attendees else []
        )
        added_attendees = [
            address
            for address in desired_addresses
            if address.lower() not in existing_attendee_emails
        ]
        if attendees_given:
            existing_by_email = {
                attendee["emailAddress"]["address"].strip().lower(): attendee
                for attendee in (existing.get("attendees") or [])
                if (attendee.get("emailAddress") or {}).get("address", "").strip()
            }
            desired_lower = {address.lower() for address in desired_addresses}
            if desired_lower != existing_attendee_emails:
                payload["attendees"] = [
                    existing_by_email.get(
                        address.lower(),
                        {"emailAddress": {"address": address}, "type": "required"},
                    )
                    for address in desired_addresses
                ]
            retained_attendees = [
                address
                for address in desired_addresses
                if address.lower() in existing_attendee_emails
            ]
        else:
            retained_attendees = existing_attendees_raw
        if is_all_day is not None:
            payload["isAllDay"] = is_all_day

        if not payload and attendees_given:
            return _success(event=existing, message="No attendee changes were needed")
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
        if (
            (start_datetime is not None or end_datetime is not None)
            and effective_start
            and effective_end
        ):
            _reject_reversed_window(effective_start, effective_end)

        if not ignore_conflicts and (schedule_semantics_supplied or added_attendees):
            query_start, query_end = effective_start, effective_end
            if (
                added_attendees
                and not existing_is_all_day
                and not both_boundaries_supplied
                and not single_boundary_update
            ):
                # A timed attendee-only update queries the unchanged snapshot
                # window. A plain event GET is documented to return UTC; validate
                # that contract rather than pairing contradictory response clock
                # values with their labels and querying the wrong instant.
                query_start = _utc_field_in_zone(snapshot_start_field, "UTC").get(
                    "dateTime"
                )
                query_end = _utc_field_in_zone(snapshot_end_field, "UTC").get(
                    "dateTime"
                )
                query_timezone = "UTC"
            if not query_start or not query_end:
                raise ValueError(
                    "Existing event has no complete time window; cannot safely "
                    "check conflicts for this update."
                )

            def _key(value: str, timezone_name: str) -> datetime | str | None:
                return _datetime_key_for_comparison(
                    _offset_datetime_string(
                        value, timezone_name, allow_windows_names=True
                    )
                )

            retained_segments: list[tuple[datetime, datetime]] = []
            if schedule_semantics_supplied and retained_attendees:
                if existing_is_all_day:
                    # Graph exposes existing all-day boundaries as UTC-labeled
                    # calendar-date values, not reliable instants. They are still
                    # safe to compare as date labels to determine whether the new
                    # window introduces any territory. An unchanged or shrunken
                    # window therefore needs no retained-attendee query at all.
                    retained_segments = _window_delta_segments(
                        snapshot_start_field.get("dateTime"),
                        snapshot_end_field.get("dateTime"),
                        effective_start,
                        effective_end,
                    )
                    if retained_segments:
                        raise ValueError(
                            "Outlook does not expose a reliable timezone for the "
                            "existing all-day window, so retained attendee conflicts "
                            "cannot be checked safely for newly introduced dates. "
                            "Retry with ignore_conflicts=true only after the user "
                            "confirms the attendee availability."
                        )
                else:
                    snapshot_start = _utc_field_in_zone(
                        snapshot_start_field, "UTC"
                    ).get("dateTime")
                    snapshot_end = _utc_field_in_zone(snapshot_end_field, "UTC").get(
                        "dateTime"
                    )
                    if not snapshot_start or not snapshot_end:
                        raise ValueError(
                            "Existing event has no complete time window; cannot safely "
                            "check retained attendee conflicts for this update."
                        )
                    retained_segments = _window_delta_segments(
                        _key(snapshot_start, "UTC"),
                        _key(snapshot_end, "UTC"),
                        _key(query_start, query_timezone),
                        _key(query_end, query_timezone),
                    )

            if existing_is_all_day and added_attendees and not both_boundaries_supplied:
                raise ValueError(
                    "Outlook does not expose a reliable timezone for the existing "
                    "all-day window. Provide both boundaries with an explicit "
                    "timezone so new attendee availability can be checked safely."
                )

            all_conflicts: list[dict[str, Any]] = []
            seen_conflicts: set[str] = set()
            unchecked_attendees: list[str] = []
            check_error: str | None = None
            pending_scope_error: InsufficientScopeError | None = None

            def _extend_conflicts(items: list[dict[str, Any]]) -> None:
                for item in items:
                    key = json.dumps(
                        item, ensure_ascii=False, sort_keys=True, default=str
                    )
                    if key not in seen_conflicts:
                        all_conflicts.append(item)
                        seen_conflicts.add(key)

            def _run_and_accumulate(
                time_min: str,
                time_max: str,
                timezone_name: str,
                attendees_to_check: list[str],
                *,
                check_organizer: bool,
            ) -> bool:
                nonlocal check_error, pending_scope_error
                try:
                    conflicts, unchecked = _find_conflicts(
                        time_min,
                        time_max,
                        timezone_name,
                        attendees_to_check,
                        exclude_event_id=event_id,
                        check_organizer=check_organizer,
                        organizer_calendar_label="signed_in_calendar",
                    )
                    _extend_conflicts(conflicts)
                    unchecked_attendees.extend(unchecked)
                    return False
                except InsufficientScopeError as exc:
                    check_error = check_error or str(exc)
                    pending_scope_error = exc
                    _extend_conflicts(exc.conflicts)
                    unchecked_attendees.extend(exc.unchecked_attendees)
                    return True
                except _ConflictCheckIncompleteError as exc:
                    check_error = check_error or str(exc)
                    _extend_conflicts(exc.conflicts)
                    unchecked_attendees.extend(exc.unchecked_attendees)
                    return True

            _run_and_accumulate(
                query_start,
                query_end,
                query_timezone,
                added_attendees,
                check_organizer=schedule_semantics_supplied,
            )

            for segment_start, segment_end in retained_segments:
                segment_start_utc = segment_start.astimezone(dt_timezone.utc).replace(
                    tzinfo=None
                )
                segment_end_utc = segment_end.astimezone(dt_timezone.utc).replace(
                    tzinfo=None
                )
                stopped = _run_and_accumulate(
                    segment_start_utc.isoformat(),
                    segment_end_utc.isoformat(),
                    "UTC",
                    retained_attendees,
                    check_organizer=False,
                )
                if stopped:
                    break

            unchecked_attendees = list(dict.fromkeys(unchecked_attendees))
            if pending_scope_error is not None and not all_conflicts:
                raise pending_scope_error
            if all_conflicts:
                return _conflict_response(
                    all_conflicts,
                    unchecked_attendees,
                    query_start,
                    query_end,
                    check_error=check_error,
                )
            if check_error or unchecked_attendees:
                return _incomplete_check_response(
                    unchecked_attendees,
                    query_start,
                    query_end,
                    message=check_error,
                )

        patch_headers: dict[str, str] | None = None
        if "attendees" in payload:
            event_etag = existing.get("@odata.etag")
            if not isinstance(event_etag, str) or not event_etag.strip():
                raise ValueError(
                    "Outlook did not return an event version, so attendee changes "
                    "cannot be applied safely. Read the event again and retry."
                )
            patch_headers = {"If-Match": event_etag.strip()}
        try:
            result = _graph_request(
                "PATCH",
                f"/me/events/{quote(event_id, safe='')}",
                body=payload,
                extra_headers=patch_headers,
            )
        except _GraphRequestError as exc:
            if patch_headers and exc.status_code == 412:
                raise ValueError(
                    "The event changed while attendee availability was being "
                    "checked. No update was applied; read the latest event and "
                    "retry so recent attendee or RSVP changes are preserved."
                ) from exc
            raise
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
