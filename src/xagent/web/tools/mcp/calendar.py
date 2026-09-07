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


def _event_response(event: dict[str, Any]) -> dict[str, Any]:
    response = {"status": "success", "event": event}
    hangout_link = event.get("hangoutLink")
    if hangout_link:
        response["hangout_link"] = hangout_link
    return response


@mcp.tool()
def google_calendar_create_events(
    summary: str,
    start_time: str,
    end_time: str,
    description: str | None = None,
    location: str | None = None,
    attendees: list[str] | None = None,
    add_google_meet: bool = False,
) -> str:
    """
    Create a new event in Google Calendar.
    start_time and end_time must be RFC3339 formatted (e.g., '2024-01-01T10:00:00Z' or '2024-01-01T10:00:00-07:00').
    attendees is a list of email addresses to invite; passing any attendees makes Google Calendar
    email them a native invite immediately, so confirm the recipient list with the user first.
    Set add_google_meet=True to attach a real Google Meet video-conference link to the event
    (the link is returned as hangout_link); a plain "Google Meet" string in location does not do this.
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
        if attendees:
            event["attendees"] = [{"email": email} for email in attendees]
        if add_google_meet:
            _add_conference_request(event)

        request = service.events().insert(
            calendarId="primary",
            body=event,
            conferenceDataVersion=1 if add_google_meet else 0,
            sendUpdates="all" if attendees else "none",
        )
        created_event = request.execute()
        return json.dumps(_event_response(created_event))

    except Exception as e:
        logger.error(f"Error creating event: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def google_calendar_get_event(event_id: str) -> str:
    """
    Get a specific event from Google Calendar.
    """
    try:
        service = get_calendar_service()
        event = service.events().get(calendarId="primary", eventId=event_id).execute()
        return json.dumps({"status": "success", "event": event})
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
    add_google_meet: bool = False,
) -> str:
    """
    Update an existing event in Google Calendar.
    start_time and end_time must be RFC3339 formatted if provided.
    attendees is a list of email addresses to invite; passing any attendees makes Google Calendar
    email them a native invite immediately, so confirm the recipient list with the user first.
    Set add_google_meet=True to attach a real Google Meet video-conference link to the event
    (the link is returned as hangout_link); a plain "Google Meet" string in location does not do this.
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
        if attendees:
            event["attendees"] = [{"email": email} for email in attendees]
        if add_google_meet:
            _add_conference_request(event)

        request = service.events().update(
            calendarId="primary",
            eventId=event_id,
            body=event,
            conferenceDataVersion=1 if add_google_meet else 0,
            sendUpdates="all" if attendees else "none",
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
