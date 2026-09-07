import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import calendar


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")


def _fake_service(execute_result: dict, existing_event: dict | None = None):
    """A stand-in for the googleapiclient Calendar service that records the
    kwargs passed to events().insert()/update() and returns execute_result.
    existing_event is what events().get() returns, simulating an event that
    already has state (attendees, conferenceData) before this call."""
    events = Mock()
    request = Mock()
    request.execute.return_value = execute_result
    events.insert = Mock(return_value=request)
    events.update = Mock(return_value=request)
    events.get = Mock(
        return_value=Mock(execute=Mock(return_value=existing_event or {"id": "evt1"}))
    )
    service = Mock()
    service.events.return_value = events
    return service


def test_create_event_without_attendees_or_meet_sends_conference_version_and_no_invites(
    monkeypatch,
):
    service = _fake_service({"id": "evt1"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="1:1",
            start_time="2026-09-07T15:00:00+08:00",
            end_time="2026-09-07T16:00:00+08:00",
            location="Google Meet",
        )
    )

    assert result["status"] == "success"
    assert "hangout_link" not in result

    _, kwargs = service.events.return_value.insert.call_args
    assert "attendees" not in kwargs["body"]
    assert "conferenceData" not in kwargs["body"]
    assert kwargs["body"]["location"] == "Google Meet"
    # conferenceDataVersion=1 must always be sent: it only declares capability,
    # it doesn't create a conference by itself (that needs a createRequest in
    # the body), but omitting it would make Google ignore any conferenceData
    # that IS present -- see the update() tests below for why this matters.
    assert kwargs["conferenceDataVersion"] == 1
    assert kwargs["sendUpdates"] == "none"


def test_create_event_with_attendees_but_no_notify_does_not_send_invites(monkeypatch):
    execute_result = {
        "id": "evt1",
        "hangoutLink": "https://meet.google.com/abc-defg-hij",
    }
    service = _fake_service(execute_result)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="1:1",
            start_time="2026-09-07T15:00:00+08:00",
            end_time="2026-09-07T16:00:00+08:00",
            attendees=["someone@example.com"],
            add_google_meet=True,
        )
    )

    assert result["status"] == "success"
    assert result["hangout_link"] == "https://meet.google.com/abc-defg-hij"

    _, kwargs = service.events.return_value.insert.call_args
    body = kwargs["body"]
    assert body["attendees"] == [{"email": "someone@example.com"}]
    assert body["conferenceData"]["createRequest"]["conferenceSolutionKey"] == {
        "type": "hangoutsMeet"
    }
    assert kwargs["conferenceDataVersion"] == 1
    # Attendees are recorded on the event, but nobody is emailed unless
    # notify_attendees is explicitly set.
    assert kwargs["sendUpdates"] == "none"


def test_create_event_with_attendees_and_notify_sends_invites(monkeypatch):
    service = _fake_service({"id": "evt1"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_create_events(
        summary="1:1",
        start_time="2026-09-07T15:00:00+08:00",
        end_time="2026-09-07T16:00:00+08:00",
        attendees=["someone@example.com"],
        notify_attendees=True,
    )

    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["attendees"] == [{"email": "someone@example.com"}]
    assert kwargs["sendUpdates"] == "all"


def test_create_event_notify_attendees_without_attendees_does_not_send_invites(
    monkeypatch,
):
    service = _fake_service({"id": "evt1"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_create_events(
        summary="1:1",
        start_time="2026-09-07T15:00:00+08:00",
        end_time="2026-09-07T16:00:00+08:00",
        notify_attendees=True,
    )

    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["sendUpdates"] == "none"


def test_update_event_without_attendees_or_meet_sends_conference_version_and_no_invites(
    monkeypatch,
):
    service = _fake_service({"id": "evt1"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(event_id="evt1", summary="New title")
    )

    assert result["status"] == "success"
    assert "hangout_link" not in result

    _, kwargs = service.events.return_value.update.call_args
    assert "attendees" not in kwargs["body"]
    assert "conferenceData" not in kwargs["body"]
    assert kwargs["conferenceDataVersion"] == 1
    assert kwargs["sendUpdates"] == "none"


def test_update_event_preserves_existing_meet_link_when_add_google_meet_not_set(
    monkeypatch,
):
    """Regression test: conferenceDataVersion must be 1 even when
    add_google_meet isn't passed, or Google ignores the conferenceData this
    call resends from the fetched event and silently drops the Meet link."""
    existing_event = {
        "id": "evt1",
        "conferenceData": {
            "conferenceId": "abc-defg-hij",
            "entryPoints": [{"uri": "https://meet.google.com/abc-defg-hij"}],
        },
        "hangoutLink": "https://meet.google.com/abc-defg-hij",
    }
    service = _fake_service(existing_event, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(event_id="evt1", summary="New title")

    _, kwargs = service.events.return_value.update.call_args
    # The existing conferenceData must still be present in the body...
    assert kwargs["body"]["conferenceData"] == existing_event["conferenceData"]
    # ...and conferenceDataVersion must be 1, or Google would ignore it.
    assert kwargs["conferenceDataVersion"] == 1


def test_update_event_merges_new_attendees_with_existing_ones(monkeypatch):
    existing_event = {
        "id": "evt1",
        "attendees": [{"email": "alice@example.com", "responseStatus": "accepted"}],
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(
        event_id="evt1", attendees=["bob@example.com"]
    )

    _, kwargs = service.events.return_value.update.call_args
    # Alice is kept (with her existing RSVP metadata intact), Bob is added.
    assert kwargs["body"]["attendees"] == [
        {"email": "alice@example.com", "responseStatus": "accepted"},
        {"email": "bob@example.com"},
    ]


def test_update_event_does_not_duplicate_an_already_invited_attendee(monkeypatch):
    existing_event = {
        "id": "evt1",
        "attendees": [{"email": "alice@example.com", "responseStatus": "accepted"}],
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(
        event_id="evt1", attendees=["alice@example.com"]
    )

    _, kwargs = service.events.return_value.update.call_args
    # Alice's existing entry (with her RSVP status) is kept, not replaced with
    # a bare {"email": ...} dict, and she isn't duplicated.
    assert kwargs["body"]["attendees"] == [
        {"email": "alice@example.com", "responseStatus": "accepted"}
    ]


def test_update_event_dedupes_repeated_emails_in_the_same_attendees_call(monkeypatch):
    service = _fake_service({"id": "evt1"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(
        event_id="evt1",
        attendees=["bob@example.com", "bob@example.com"],
    )

    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["attendees"] == [{"email": "bob@example.com"}]


def test_update_event_with_no_new_attendees_leaves_existing_attendees_untouched(
    monkeypatch,
):
    existing_event = {
        "id": "evt1",
        "attendees": [{"email": "alice@example.com", "responseStatus": "accepted"}],
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(event_id="evt1", summary="New title")

    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["attendees"] == existing_event["attendees"]


def test_update_event_notify_attendees_notifies_existing_attendees_without_repassing_them(
    monkeypatch,
):
    """Regression test: notify_attendees must look at the event's actual
    attendees, not just the attendees param on this call, or you can never
    notify already-invited people of e.g. a reschedule."""
    existing_event = {
        "id": "evt1",
        "attendees": [{"email": "alice@example.com"}],
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(
        event_id="evt1",
        start_time="2026-09-08T15:00:00+08:00",
        notify_attendees=True,
    )

    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["sendUpdates"] == "all"


def test_update_event_notify_attendees_is_noop_when_event_has_no_attendees(
    monkeypatch,
):
    service = _fake_service({"id": "evt1"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(
        event_id="evt1", summary="New title", notify_attendees=True
    )

    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["sendUpdates"] == "none"


def test_update_event_add_google_meet_creates_conference_when_none_exists(
    monkeypatch,
):
    execute_result = {
        "id": "evt1",
        "hangoutLink": "https://meet.google.com/abc-defg-hij",
    }
    service = _fake_service(execute_result)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(event_id="evt1", add_google_meet=True)
    )

    assert result["hangout_link"] == "https://meet.google.com/abc-defg-hij"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["conferenceData"]["createRequest"][
        "conferenceSolutionKey"
    ] == {"type": "hangoutsMeet"}


def test_update_event_add_google_meet_does_not_replace_existing_conference(
    monkeypatch,
):
    """Regression test: add_google_meet=True must not orphan a Meet link
    that's already shared with attendees by generating a brand-new one."""
    existing_event = {
        "id": "evt1",
        "conferenceData": {
            "conferenceId": "abc-defg-hij",
            "entryPoints": [{"uri": "https://meet.google.com/abc-defg-hij"}],
        },
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(event_id="evt1", add_google_meet=True)

    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["conferenceData"] == existing_event["conferenceData"]
    assert "createRequest" not in kwargs["body"]["conferenceData"]


def test_response_surfaces_pending_conference_status_instead_of_hangout_link(
    monkeypatch,
):
    """Meet link creation is asynchronous; if hangoutLink isn't populated yet
    the response should say so instead of silently omitting everything."""
    execute_result = {
        "id": "evt1",
        "conferenceData": {
            "createRequest": {
                "requestId": "abc123",
                "status": {"statusCode": "pending"},
            }
        },
    }
    service = _fake_service(execute_result)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="1:1",
            start_time="2026-09-07T15:00:00+08:00",
            end_time="2026-09-07T16:00:00+08:00",
            add_google_meet=True,
        )
    )

    assert "hangout_link" not in result
    assert result["conference_status"] == "pending"


def test_response_handles_null_conference_data_without_crashing(monkeypatch):
    """Guard against event["conferenceData"] (or nested keys) being present but
    explicitly null in the API response, not just absent."""
    service = _fake_service({"id": "evt1", "conferenceData": None})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="1:1",
            start_time="2026-09-07T15:00:00+08:00",
            end_time="2026-09-07T16:00:00+08:00",
        )
    )

    assert result["status"] == "success"
    assert "hangout_link" not in result
    assert "conference_status" not in result
