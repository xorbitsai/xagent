import json
import logging
import os
from datetime import date
from typing import Any
from urllib.parse import quote

import requests
from dateutil import parser as _date_parser
from dateutil import tz as _tz
from dateutil.rrule import FR as _FR
from dateutil.rrule import MO as _MO
from dateutil.rrule import SA as _SA
from dateutil.rrule import SU as _SU
from dateutil.rrule import TH as _TH
from dateutil.rrule import TU as _TU
from dateutil.rrule import WE as _WE
from mcp.server.fastmcp import FastMCP

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


_RRULE_WEEKDAYS = (_MO, _TU, _WE, _TH, _FR, _SA, _SU)

_RRULE_DAY_TO_GRAPH = {
    "MO": "monday",
    "TU": "tuesday",
    "WE": "wednesday",
    "TH": "thursday",
    "FR": "friday",
    "SA": "saturday",
    "SU": "sunday",
}


def _rrule_until_to_date(until: str) -> str:
    """Convert an RRULE UNTIL value ('20260911T235959Z' or a bare
    '20260911') into the 'YYYY-MM-DD' form Graph's recurrenceRange wants."""
    try:
        return _date_parser.isoparse(until.strip()).date().isoformat()
    except ValueError as exc:
        raise ValueError(f"invalid UNTIL value in recurrence rule: {until}") from exc


def _weekday_code_from_date(date_str: str) -> str:
    """The RRULE two-letter day code (MO/TU/.../SU) for a 'YYYY-MM-DD' date."""
    year, month, day = (int(part) for part in date_str.split("-"))
    return str(_RRULE_WEEKDAYS[date(year, month, day).weekday()])


def _build_graph_recurrence(
    recurrence: str, start_datetime: str, timezone: str = "UTC"
) -> dict[str, Any]:
    """Translate an RFC 5545 RRULE string into the pattern/range object
    Microsoft Graph's event.recurrence expects.

    Graph has no "just take this RRULE text" input the way Google Calendar
    does, so this maps the common cases Toby is expected to produce
    (daily, weekly, absolute monthly, absolute yearly) and fails loudly on
    anything else - a relative pattern like "second Tuesday of the month"
    needs an index (first/second/.../last) Graph requires but an RRULE
    without a numeric BYDAY prefix doesn't carry, so guessing here would
    silently produce the wrong days rather than the ones actually asked
    for.

    start_datetime here is Outlook's own convention: a naive local
    dateTime paired with a separate timeZone field, unlike Google's
    RFC3339-with-offset. RFC 5545 requires a recurrence's UNTIL to be a
    UTC value whenever DTSTART carries a timezone reference - which this
    one does, just not embedded in the string - so it's localized with
    `timezone` before validation; passing the bare naive string through
    would make dateutil see a floating time and reject a (correct) UTC
    UNTIL as a mismatch.
    """
    anchor = _date_parser.isoparse(start_datetime)
    if anchor.tzinfo is None:
        zone = _tz.gettz(timezone)
        if zone is None:
            raise ValueError(f"unknown timezone for recurrence rule: {timezone}")
        anchor = anchor.replace(tzinfo=zone)
    parts = parse_rrule(recurrence, anchor.isoformat())
    freq = parts["FREQ"].upper()
    interval = int(parts.get("INTERVAL", "1"))
    start_date = start_datetime.split("T", 1)[0]

    pattern: dict[str, Any]
    if freq == "DAILY":
        pattern = {"type": "daily", "interval": interval}
    elif freq == "WEEKLY":
        byday = parts.get("BYDAY")
        if byday:
            days = []
            for code in byday.split(","):
                clean_code = code.strip().upper()
                if clean_code not in _RRULE_DAY_TO_GRAPH:
                    raise ValueError(f"invalid day code in BYDAY: {code!r}")
                days.append(_RRULE_DAY_TO_GRAPH[clean_code])
        else:
            days = [_RRULE_DAY_TO_GRAPH[_weekday_code_from_date(start_date)]]
        pattern = {"type": "weekly", "interval": interval, "daysOfWeek": days}
    elif freq == "MONTHLY" and "BYMONTHDAY" in parts:
        pattern = {
            "type": "absoluteMonthly",
            "interval": interval,
            "dayOfMonth": int(parts["BYMONTHDAY"]),
        }
    elif freq == "YEARLY" and "BYMONTH" in parts and "BYMONTHDAY" in parts:
        pattern = {
            "type": "absoluteYearly",
            "interval": interval,
            "dayOfMonth": int(parts["BYMONTHDAY"]),
            "month": int(parts["BYMONTH"]),
        }
    else:
        raise ValueError(
            f"unsupported recurrence pattern (FREQ={freq}); this connector "
            "only translates DAILY, WEEKLY, MONTHLY with BYMONTHDAY, and "
            "YEARLY with BYMONTH/BYMONTHDAY into an Outlook recurrence"
        )

    if "UNTIL" in parts:
        range_: dict[str, Any] = {
            "type": "endDate",
            "startDate": start_date,
            "endDate": _rrule_until_to_date(parts["UNTIL"]),
        }
    elif "COUNT" in parts:
        range_ = {
            "type": "numbered",
            "startDate": start_date,
            "numberOfOccurrences": int(parts["COUNT"]),
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
    DAILY, WEEKLY, MONTHLY (with BYMONTHDAY), and YEARLY (with BYMONTH/
    BYMONTHDAY) rules are supported; anything else is rejected with a
    clear error rather than silently producing the wrong pattern.
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
        if recurrence:
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
    replaces its existing one.
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
        if recurrence:
            effective_start = start_datetime
            effective_timezone = timezone
            if effective_start is None:
                existing = _graph_request(
                    "GET",
                    f"/me/events/{quote(event_id, safe='')}",
                    params={"$select": "start"},
                )
                existing_start_field = existing.get("start") or {}
                effective_start = existing_start_field.get("dateTime")
                if not effective_start:
                    raise ValueError(
                        "could not determine the event's start time to "
                        "validate the recurrence rule; pass start_datetime "
                        "explicitly"
                    )
                # `timezone` only pairs with a start_datetime the caller is
                # ALSO providing, which isn't the case here - Graph's
                # dateTimeTimeZone object always reports the zone its
                # dateTime is actually expressed in (defaulting to "UTC"
                # when no Prefer header was sent, as here), so read that
                # instead of trusting the caller to have independently
                # passed a matching timezone.
                existing_timezone = existing_start_field.get("timeZone")
                if not existing_timezone:
                    raise ValueError(
                        "existing event has no timeZone on its start time; "
                        "cannot safely build a recurrence rule without an "
                        "explicit start_datetime for this update"
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
