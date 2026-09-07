import json
import logging
import os
import uuid
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calendar-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("calendar-mcp")


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


def _needs_conference_request(event: dict[str, Any]) -> bool:
    """True if add_google_meet should (re)send a createRequest: the event has
    no conferenceData yet, or its only conferenceData is a *failed*
    createRequest (status.statusCode == "failure").

    A conference that's still *pending* must not be treated the same as a
    failed one -- both lack entryPoints/conferenceSolution while Google is
    still provisioning them, so checking for those fields can't tell "failed,
    safe to retry" apart from "pending, do not clobber the in-flight
    request". Read status.statusCode directly instead (the same field
    _event_response already reads for the opposite purpose). A conference
    that has already resolved (has entryPoints or a conferenceSolution --
    Meet or otherwise) also returns False, so it isn't clobbered either.
    """
    conference_data = event.get("conferenceData") or {}
    if not conference_data:
        return True
    create_request = conference_data.get("createRequest") or {}
    status_code = (create_request.get("status") or {}).get("statusCode")
    return status_code == "failure"


def _conference_and_notify_kwargs(
    event: dict[str, Any], notify_attendees: bool, add_google_meet: bool
) -> dict[str, Any]:
    """Apply add_google_meet to event in place, and build the insert()/update() kwargs.

    conferenceDataVersion is always sent as 1: it only declares that the caller
    understands conference data, it does not request or remove a conference by
    itself (that's driven by whether `body` carries a createRequest). Omitting it
    (or sending 0) makes Google ignore any conferenceData already in the body,
    which would silently drop an event's existing Meet link on every update that
    doesn't also pass add_google_meet=True.
    """
    if add_google_meet and _needs_conference_request(event):
        _add_conference_request(event)
    return {
        "conferenceDataVersion": 1,
        "sendUpdates": "all" if notify_attendees and event.get("attendees") else "none",
    }


def _event_response(event: dict[str, Any]) -> dict[str, Any]:
    response = {"status": "success", "event": event}
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
            (conference_data.get("conferenceSolution") or {}).get("key", {}).get("type")
        )
        if solution_type == "hangoutsMeet":
            for entry_point in conference_data.get("entryPoints") or []:
                if entry_point.get("entryPointType") == "video" and entry_point.get(
                    "uri"
                ):
                    hangout_link = entry_point["uri"]
                    break

    if hangout_link:
        response["hangout_link"] = hangout_link
    else:
        create_request = conference_data.get("createRequest") or {}
        status_code = (create_request.get("status") or {}).get("statusCode")
        if status_code:
            # Meet link creation is asynchronous; surface the status instead of
            # implying failure when hangoutLink isn't populated yet.
            response["conference_status"] = status_code
    return response


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
) -> str:
    """
    Create a new event in Google Calendar.
    start_time and end_time must be RFC3339 formatted (e.g., '2024-01-01T10:00:00Z' or '2024-01-01T10:00:00-07:00').
    attendees is a list of email addresses to add to the event. Adding attendees does not, by
    itself, email them; set notify_attendees=True to have Google Calendar send them a native
    invite immediately. Confirm the recipient list with the user before setting notify_attendees=True.
    Set add_google_meet=True to attach a real Google Meet video-conference link to the event. The
    link is returned as hangout_link once Google finishes provisioning it; if it isn't ready yet the
    response includes conference_status instead (e.g. "pending") — call google_calendar_get_event
    shortly after to fetch the link. A plain "Google Meet" string in location does not create a link.
    """
    try:
        service = get_calendar_service()

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
        _merge_attendees(event, attendees)

        request_kwargs = _conference_and_notify_kwargs(
            event, notify_attendees, add_google_meet
        )
        request = service.events().insert(
            calendarId="primary",
            body=event,
            **request_kwargs,
        )
        created_event = request.execute()
        return json.dumps(_event_response(created_event))

    except Exception as e:
        logger.error(f"Error creating event: {e}")
        return json.dumps({"status": "error", "message": str(e)})


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
        return json.dumps(_event_response(event))
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
    kept. Adding attendees does not, by itself, email anyone; set notify_attendees=True to have
    Google Calendar send a native invite/update immediately to every attendee on the event (existing
    and newly added). Confirm the recipient list with the user before setting notify_attendees=True.
    Set add_google_meet=True to attach a real Google Meet video-conference link to the event if it
    doesn't already have a conference (an existing conference is left untouched). The link is
    returned as hangout_link once Google finishes provisioning it; if it isn't ready yet the response
    includes conference_status instead (e.g. "pending") — call google_calendar_get_event shortly
    after to fetch the link. A plain "Google Meet" string in location does not create a link.
    """
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

        request_kwargs = _conference_and_notify_kwargs(
            event, notify_attendees, add_google_meet
        )
        request = service.events().update(
            calendarId="primary",
            eventId=event_id,
            body=event,
            **request_kwargs,
        )
        updated_event = request.execute()
        return json.dumps(_event_response(updated_event))

    except Exception as e:
        logger.error(f"Error updating event: {e}")
        return json.dumps({"status": "error", "message": str(e)})


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
