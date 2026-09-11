import copy
import json
import re
from typing import Any
from unittest.mock import Mock

import pytest
from googleapiclient.errors import HttpError

from xagent.web.tools.mcp import calendar
from xagent.web.tools.mcp import utils as mcp_utils


def _fake_service(execute_result: dict, existing_event: dict | None = None):
    """A stand-in for the googleapiclient Calendar service that records the
    kwargs passed to events().insert()/update() and returns execute_result.
    existing_event is what events().get() returns, simulating an event that
    already has state (attendees, conferenceData) before this call.

    A deep copy is handed to the code under test (rather than the caller's
    own existing_event object): calendar.py mutates the dict it gets back
    from events().get().execute() in place before resending it as the update
    body, so returning the original object would let that mutation leak back
    into the test's "expected" value and make before/after comparisons a
    tautology.
    """
    events = Mock()
    request = Mock()
    request.execute.return_value = execute_result
    events.insert = Mock(return_value=request)
    events.update = Mock(return_value=request)
    events.get = Mock(
        return_value=Mock(
            execute=Mock(
                return_value=copy.deepcopy(existing_event)
                if existing_event
                else {"id": "evt1"}
            )
        )
    )
    # These Meet-focused tests don't exercise the scheduling-conflict
    # check (that's the other half of this file's test suite) - stub the
    # organizer/attendee queries it always runs (unless ignore_conflicts)
    # to report "no conflicts" so it doesn't block on an un-mocked Mock.
    events.list = Mock(return_value=Mock(execute=Mock(return_value={"items": []})))
    service = Mock()
    service.events.return_value = events
    service.freebusy.return_value = Mock(
        query=Mock(return_value=Mock(execute=Mock(return_value={"calendars": {}})))
    )
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


def test_update_event_ignores_a_whitespace_only_attendee_entry(monkeypatch):
    service = _fake_service({"id": "evt1"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(
        event_id="evt1",
        attendees=["   ", "bob@example.com"],
    )

    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["attendees"] == [{"email": "bob@example.com"}]


def test_update_event_attendee_matching_is_case_and_whitespace_insensitive(
    monkeypatch,
):
    """Regression test: `Alice@Example.com` and `alice@example.com ` are the
    same mailbox and must not produce a duplicate attendee entry."""
    existing_event = {
        "id": "evt1",
        "attendees": [{"email": "alice@example.com", "responseStatus": "accepted"}],
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(
        event_id="evt1",
        attendees=["Alice@Example.com ", " BOB@example.com", "bob@example.com"],
    )

    _, kwargs = service.events.return_value.update.call_args
    # Alice's existing entry is kept as-is (not duplicated under different
    # casing), and Bob is added once despite two differently-cased spellings.
    assert kwargs["body"]["attendees"] == [
        {"email": "alice@example.com", "responseStatus": "accepted"},
        {"email": "BOB@example.com"},
    ]


def test_update_event_tolerates_a_malformed_existing_attendee_list(monkeypatch):
    """Regression test: an existing attendee entry with no "email" key (e.g.
    a resource/room attendee), or a non-dict item, must not crash the merge
    or get treated as a real email to match/duplicate against."""
    existing_event = {
        "id": "evt1",
        "attendees": [
            {"displayName": "Conference Room A"},
            "not-a-dict",
            {"email": "alice@example.com"},
        ],
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(
        event_id="evt1", attendees=["bob@example.com"]
    )

    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["attendees"] == existing_event["attendees"] + [
        {"email": "bob@example.com"}
    ]


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


def test_update_event_add_google_meet_does_not_duplicate_a_legacy_hangout_link(
    monkeypatch,
):
    """Regression test: a legacy event can carry hangoutLink with no
    conferenceData at all (it predates the conferenceData API). Without a
    hangoutLink check, add_google_meet=True would see "no conferenceData" and
    request a second, duplicate conference for an event that already has
    one."""
    existing_event = {
        "id": "evt1",
        "hangoutLink": "https://meet.google.com/abc-defg-hij",
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(event_id="evt1", add_google_meet=True)

    _, kwargs = service.events.return_value.update.call_args
    assert "conferenceData" not in kwargs["body"]


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


def test_update_event_add_google_meet_retries_after_a_failed_create_request(
    monkeypatch,
):
    """Regression test: a failed createRequest leaves conferenceData set but
    with no entryPoints/conferenceSolution. add_google_meet=True must treat
    that as "no conference yet" and retry, not as an existing conference to
    preserve -- otherwise the event is permanently stuck with no Meet link."""
    existing_event = {
        "id": "evt1",
        "conferenceData": {
            "createRequest": {
                "requestId": "abc123",
                "status": {"statusCode": "failure"},
            }
        },
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(event_id="evt1", add_google_meet=True)

    _, kwargs = service.events.return_value.update.call_args
    new_create_request = kwargs["body"]["conferenceData"]["createRequest"]
    # A hardcoded different literal would also satisfy `!= "abc123"`; check
    # it's actually a fresh uuid4().hex (32 lowercase hex chars), not just
    # any old different string.
    assert re.fullmatch(r"[0-9a-f]{32}", new_create_request["requestId"])
    assert new_create_request["requestId"] != "abc123"
    assert new_create_request["conferenceSolutionKey"] == {"type": "hangoutsMeet"}


def test_update_event_add_google_meet_does_not_clobber_a_pending_create_request(
    monkeypatch,
):
    """Regression test: a *pending* createRequest has no entryPoints/
    conferenceSolution yet either -- the same shape as a failed one. It must
    not be treated as retryable, or a second add_google_meet=True call (e.g.
    while also changing the time) abandons the in-flight request and starts
    a new, unrelated one."""
    existing_event = {
        "id": "evt1",
        "conferenceData": {
            "createRequest": {
                "requestId": "abc123",
                "status": {"statusCode": "pending"},
            }
        },
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_update_events(event_id="evt1", add_google_meet=True)

    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["conferenceData"] == existing_event["conferenceData"]
    assert kwargs["body"]["conferenceData"]["createRequest"]["requestId"] == "abc123"


def test_event_response_falls_back_to_entry_points_when_hangout_link_is_absent(
    monkeypatch,
):
    """hangoutLink is only reliably populated for Meet conferences; fall back
    to the canonical conferenceData.entryPoints when it's missing, but only
    for an actual Meet conference (conferenceSolution.key.type ==
    "hangoutsMeet") -- see the sibling test below for the non-Meet case."""
    execute_result = {
        "id": "evt1",
        "conferenceData": {
            "conferenceSolution": {"key": {"type": "hangoutsMeet"}},
            "entryPoints": [
                {"entryPointType": "phone", "uri": "tel:+1-234-567-8900"},
                {
                    "entryPointType": "video",
                    "uri": "https://meet.google.com/abc-defg-hij",
                },
            ],
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

    assert result["hangout_link"] == "https://meet.google.com/abc-defg-hij"


def test_event_response_does_not_label_a_non_meet_conference_as_hangout_link(
    monkeypatch,
):
    """Regression test: a third-party conference (e.g. Zoom's Calendar add-on)
    also registers its join URL under entryPointType "video". Surfacing that
    as hangout_link would mislead a caller into thinking it's a Meet link."""
    execute_result = {
        "id": "evt1",
        "conferenceData": {
            "conferenceSolution": {"key": {"type": "addOn"}, "name": "Zoom Meeting"},
            "entryPoints": [
                {"entryPointType": "video", "uri": "https://zoom.us/j/123456789"},
            ],
        },
    }
    service = _fake_service(execute_result)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="1:1",
            start_time="2026-09-07T15:00:00+08:00",
            end_time="2026-09-07T16:00:00+08:00",
        )
    )

    assert "hangout_link" not in result


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


def test_response_handles_null_conference_solution_key_without_crashing(monkeypatch):
    """Guard against conferenceData.conferenceSolution.key being present but
    explicitly null, not just absent -- .get("key", {}) only defaults for a
    missing key, not a present-but-null one."""
    service = _fake_service(
        {
            "id": "evt1",
            "conferenceData": {"conferenceSolution": {"key": None}},
        }
    )
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


def test_get_event_surfaces_hangout_link_like_create_and_update_do(monkeypatch):
    """Regression test: both tool docstrings tell callers to fall back to
    google_calendar_get_event to pick up a Meet link that wasn't ready yet
    at create/update time -- it must honor that by using the same
    hangout_link/conference_status convention, not return the raw event."""
    fetched_event = {
        "id": "evt1",
        "hangoutLink": "https://meet.google.com/abc-defg-hij",
    }
    service = _fake_service({"id": "evt1"}, existing_event=fetched_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(calendar.google_calendar_get_event(event_id="evt1"))

    assert result["status"] == "success"
    assert result["hangout_link"] == "https://meet.google.com/abc-defg-hij"


def test_create_event_error_message_hints_at_add_google_meet_on_failure(monkeypatch):
    """Regression test: if the whole insert() call fails while add_google_meet
    was requested (e.g. the account/domain can't create Meet conferences at
    all), the error should point at add_google_meet as the likely cause --
    it's otherwise indistinguishable from any other request-level failure."""
    service = _fake_service({"id": "evt1"})
    service.events.return_value.insert.return_value.execute.side_effect = RuntimeError(
        "Bad Request"
    )
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="1:1",
            start_time="2026-09-07T15:00:00+08:00",
            end_time="2026-09-07T16:00:00+08:00",
            add_google_meet=True,
        )
    )

    assert result["status"] == "error"
    assert "Bad Request" in result["message"]
    assert "add_google_meet" in result["message"]


def test_update_event_error_message_hints_at_add_google_meet_on_failure(monkeypatch):
    service = _fake_service({"id": "evt1"})
    service.events.return_value.update.return_value.execute.side_effect = RuntimeError(
        "Bad Request"
    )
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(event_id="evt1", add_google_meet=True)
    )

    assert result["status"] == "error"
    assert "Bad Request" in result["message"]
    assert "add_google_meet" in result["message"]


def test_update_event_error_message_has_no_hint_when_add_google_meet_was_a_noop(
    monkeypatch,
):
    """Regression test: add_google_meet=True is a no-op when the event
    already has a resolved conference -- this call's body never actually
    carried a createRequest, so a failure has nothing to do with Meet and
    the hint would misdirect the caller."""
    existing_event = {
        "id": "evt1",
        "conferenceData": {
            "conferenceSolution": {"key": {"type": "hangoutsMeet"}},
            "entryPoints": [{"entryPointType": "video", "uri": "https://x"}],
        },
    }
    service = _fake_service({"id": "evt1"}, existing_event=existing_event)
    service.events.return_value.update.return_value.execute.side_effect = RuntimeError(
        "Bad Request"
    )
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(event_id="evt1", add_google_meet=True)
    )

    assert result["status"] == "error"
    assert result["message"] == "Bad Request"


def test_update_event_error_message_has_no_hint_when_the_preliminary_fetch_fails(
    monkeypatch,
):
    """Regression test: a failure in the preliminary events().get() fetch
    (e.g. a bad event_id) happens before _apply_conference_request ever
    runs, so this call's body never carried a createRequest either."""
    service = _fake_service({"id": "evt1"})
    service.events.return_value.get.return_value.execute.side_effect = RuntimeError(
        "Not Found"
    )
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(event_id="evt1", add_google_meet=True)
    )

    assert result["status"] == "error"
    assert result["message"] == "Not Found"


def test_create_event_error_message_has_no_hint_when_service_setup_fails(monkeypatch):
    """Regression test: a failure before any event body is even built (e.g.
    a missing/invalid credential) happens before _apply_conference_request
    ever runs, so blaming a Meet conference request would be wrong."""

    def _raise():
        raise ValueError("GOOGLE_ACCESS_TOKEN environment variable is missing")

    monkeypatch.setattr(calendar, "get_calendar_service", _raise)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="1:1",
            start_time="2026-09-07T15:00:00+08:00",
            end_time="2026-09-07T16:00:00+08:00",
            add_google_meet=True,
        )
    )

    assert result["status"] == "error"
    assert "add_google_meet" not in result["message"]


def test_create_event_error_message_has_no_hint_without_add_google_meet(monkeypatch):
    service = _fake_service({"id": "evt1"})
    service.events.return_value.insert.return_value.execute.side_effect = RuntimeError(
        "Bad Request"
    )
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="1:1",
            start_time="2026-09-07T15:00:00+08:00",
            end_time="2026-09-07T16:00:00+08:00",
        )
    )

    assert result["status"] == "error"
    assert result["message"] == "Bad Request"


def test_event_response_caps_a_large_event_like_other_tools_in_this_package(
    monkeypatch,
):
    """Regression test: the raw Google event (description, attendees,
    recurrence rules, ...) can be large enough to blow past the platform's
    output-length budget; this package's other tools already guard against
    that with success_with_capped_dict, and this response builder must too."""
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 300)
    execute_result = {
        "id": "evt1",
        "attendees": [{"email": f"person{i}@example.com"} for i in range(200)],
    }
    service = _fake_service(execute_result)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    raw = calendar.google_calendar_create_events(
        summary="1:1",
        start_time="2026-09-07T15:00:00+08:00",
        end_time="2026-09-07T16:00:00+08:00",
    )
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["event"]["attendees"]) < 200


def test_event_response_does_not_truncate_a_small_event(monkeypatch):
    service = _fake_service({"id": "evt1", "summary": "1:1"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="1:1",
            start_time="2026-09-07T15:00:00+08:00",
            end_time="2026-09-07T16:00:00+08:00",
        )
    )

    assert result["status"] == "success"
    assert result["truncated"] is False


class _Exec:
    def __init__(self, result: dict[str, Any]):
        self._result = result

    def execute(self) -> dict[str, Any]:
        return self._result


class FakeEvents:
    def __init__(
        self,
        *,
        list_result: dict[str, Any] | None = None,
        get_result: dict[str, Any] | None = None,
        insert_result: dict[str, Any] | None = None,
        update_result: dict[str, Any] | None = None,
    ):
        self._list_result = list_result if list_result is not None else {"items": []}
        # Keyed by the request's own pageToken (None for the first page) -
        # lets a test simulate a multi-page events.list response instead
        # of always returning the same page regardless of pageToken. Only
        # used when set; _list_result covers every single-page test.
        self._list_results_by_page_token: dict[str | None, dict[str, Any]] | None = None
        self._get_result = get_result or {}
        self._insert_result = insert_result or {"id": "created"}
        self._update_result = update_result or {"id": "updated"}
        self.list_calls: list[dict[str, Any]] = []
        self.insert_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []

    def list(self, **kwargs: Any) -> _Exec:
        self.list_calls.append(kwargs)
        if self._list_results_by_page_token is not None:
            return _Exec(self._list_results_by_page_token[kwargs.get("pageToken")])
        return _Exec(self._list_result)

    def get(self, **kwargs: Any) -> _Exec:
        return _Exec(self._get_result)

    def insert(self, **kwargs: Any) -> _Exec:
        self.insert_calls.append(kwargs)
        return _Exec(self._insert_result)

    def update(self, **kwargs: Any) -> _Exec:
        self.update_calls.append(kwargs)
        return _Exec(self._update_result)


class _FakeHttpResp:
    def __init__(self, status: int):
        self.status = status
        self.reason = "error"


def _insufficient_scope_error() -> HttpError:
    """A real Google API 403 for this case carries the reason on BOTH the
    legacy `errors[].reason` field and a newer `details[].reason`
    ErrorInfo entry at once - not one or the other. `HttpError`'s own
    `_get_reason()` picks `details` over `errors` whenever both are
    present, dropping the legacy reason from `str(exc)` entirely, which
    is exactly what makes a naive `"insufficientPermissions" in str(exc)`
    substring check unreliable (see `_is_insufficient_scope_error`)."""
    return HttpError(
        _FakeHttpResp(403),
        (
            b'{"error": {"code": 403, '
            b'"message": "Request had insufficient authentication scopes.", '
            b'"errors": [{"message": "Insufficient Permission", '
            b'"domain": "global", "reason": "insufficientPermissions"}], '
            b'"status": "PERMISSION_DENIED", '
            b'"details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", '
            b'"reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT", '
            b'"domain": "googleapis.com", '
            b'"metadata": {"service": "calendar-json.googleapis.com", '
            b'"method": "calendar.v3.Freebusy.Query"}}]}}'
        ),
    )


def _legacy_only_insufficient_scope_error() -> HttpError:
    """Older/less-instrumented responses may still carry only the legacy
    field - must keep matching this shape too, not just the dual one."""
    return HttpError(
        _FakeHttpResp(403),
        (
            b'{"error": {"errors": [{"reason": "insufficientPermissions"}], '
            b'"message": "Request had insufficient authentication scopes."}}'
        ),
    )


class FakeFreebusy:
    def __init__(
        self,
        result: dict[str, Any] | None = None,
        *,
        raise_error: Exception | None = None,
    ):
        self._result = result or {"calendars": {}}
        self._raise_error = raise_error
        self.query_calls: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> _Exec:
        self.query_calls.append(kwargs)
        if self._raise_error is not None:
            raise self._raise_error
        return _Exec(self._result)


class FakeCalendars:
    def __init__(
        self,
        timezone: str = "UTC",
        *,
        email: str = "me@example.com",
        raise_error: Exception | None = None,
    ):
        self._timezone = timezone
        # The connected account's own address (the "id" field
        # calendars().get returns on a real primary calendar) - defaults
        # to the same
        # "me@example.com" convention most fixtures already use for the
        # caller/organizer, so a test that never customizes this and
        # never means to simulate a different-organizer scenario keeps
        # caller_is_organizer=True without having to set this explicitly.
        self._email = email
        self._raise_error = raise_error
        self.get_calls: list[dict[str, Any]] = []

    def get(self, **kwargs: Any) -> _Exec:
        self.get_calls.append(kwargs)
        if self._raise_error is not None:
            raise self._raise_error
        return _Exec({"id": self._email, "timeZone": self._timezone})


class FakeService:
    def __init__(
        self,
        events: FakeEvents | None = None,
        freebusy: FakeFreebusy | None = None,
        calendars: FakeCalendars | None = None,
    ):
        self._events = events or FakeEvents()
        self._freebusy = freebusy or FakeFreebusy()
        self._calendars = calendars or FakeCalendars()

    def events(self) -> FakeEvents:
        return self._events

    def freebusy(self) -> FakeFreebusy:
        return self._freebusy

    def calendars(self) -> FakeCalendars:
        return self._calendars


@pytest.fixture
def fake_service(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)
    return service


def _confirmed_event(
    event_id="existing-1",
    summary="1:1 with Hazel",
    transparency=None,
    status="confirmed",
):
    event: dict[str, Any] = {
        "id": event_id,
        "summary": summary,
        "status": status,
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
    }
    if transparency:
        event["transparency"] = transparency
    return event


def test_create_events_detects_organizer_conflict_same_timezone(fake_service):
    """Reproduces the reported bug: booking a slot that overlaps an existing
    event, both in the same timezone, must be caught rather than silently
    created."""
    fake_service._events._list_result = {"items": [_confirmed_event()]}

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"
    assert len(result["conflicts"]) == 1
    assert result["conflicts"][0]["summary"] == "1:1 with Hazel"
    assert fake_service._events.insert_calls == []


def test_create_events_follows_pagination_to_find_a_conflict_on_a_later_page(
    fake_service,
):
    """Regression test: events.list() caps a single response at a limited
    page size and signals more with nextPageToken - stopping after the
    first page (as an earlier version of this check did) would silently
    place a busy event on a later page outside the check entirely,
    letting a real conflict through undetected."""
    fake_service._events._list_results_by_page_token = {
        None: {"items": [], "nextPageToken": "page-2"},
        "page-2": {"items": [_confirmed_event(event_id="other-1")]},
    }

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "1:1 with Hazel"
    page_tokens = [call.get("pageToken") for call in fake_service._events.list_calls]
    assert page_tokens == [None, "page-2"]


def test_create_events_ignores_transparent_organizer_event(fake_service):
    fake_service._events._list_result = {
        "items": [_confirmed_event(transparency="transparent")]
    }

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "success"
    assert len(fake_service._events.insert_calls) == 1


def test_create_events_ignores_cancelled_organizer_event(fake_service):
    fake_service._events._list_result = {
        "items": [_confirmed_event(status="cancelled")]
    }

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "success"


def test_create_events_detects_attendee_freebusy_conflict(fake_service):
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "chelsea@example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T10:00:00+08:00",
                            "end": "2026-08-27T10:30:00+08:00",
                        }
                    ]
                }
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "chelsea@example.com"
    assert fake_service._events.insert_calls == []


def test_create_events_unchecked_attendee_does_not_block_creation(fake_service):
    """An attendee whose calendar can't be queried (external/unshared) must
    not block booking — only be surfaced so Toby can say so."""
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "outsider@gmail.com": {"errors": [{"reason": "notFound"}]},
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["outsider@gmail.com"],
        )
    )

    assert result["status"] == "success"
    assert result["unchecked_attendees"] == ["outsider@gmail.com"]
    insert_call = fake_service._events.insert_calls[0]
    assert insert_call["body"]["attendees"] == [{"email": "outsider@gmail.com"}]
    # notify_attendees defaults to False - adding an attendee never emails
    # them without the caller explicitly opting in.
    assert insert_call["sendUpdates"] == "none"


def test_create_events_ignore_conflicts_skips_the_check_entirely(fake_service):
    fake_service._events._list_result = {"items": [_confirmed_event()]}

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    assert fake_service._events.list_calls == []
    assert len(fake_service._events.insert_calls) == 1


def test_create_events_rejects_a_reversed_window(fake_service):
    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:30:00+08:00",
            end_time="2026-08-27T10:00:00+08:00",
        )
    )

    assert result["status"] == "error"
    assert "must be after" in result["message"]
    assert fake_service._events.list_calls == []
    assert fake_service._events.insert_calls == []


def test_create_events_rejects_an_offsetless_start_time(fake_service):
    """Regression test: an offsetless start_time compared naive-against-
    aware deep inside a downstream comparison would previously fall
    through to Google's own freebusy/events.list APIs (which require an
    RFC3339 offset on timeMin/timeMax) and fail there with an opaque
    error, instead of this clear, actionable one raised up front."""
    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00",  # no offset
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "error"
    assert "start_time" in result["message"]
    assert fake_service._events.insert_calls == []


def test_create_events_prefers_the_offset_error_over_the_reversed_window_one(
    fake_service,
):
    """Regression test: two NAIVE values that also happen to be reversed
    compare just fine (no TypeError, so reject_reversed_window's
    permissive aware-vs-naive handling never kicks in) and would
    otherwise raise the generic "must be after" message before the more
    specific, actionable offset error ever got a chance to run - offset
    validation must happen first."""
    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:30:00",  # naive AND reversed
            end_time="2026-08-27T10:00:00",  # naive
        )
    )

    assert result["status"] == "error"
    assert "start_time" in result["message"]
    assert "offset" in result["message"]
    assert fake_service._events.insert_calls == []


def test_update_events_rejects_an_offsetless_start_time(fake_service):
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [{"email": "existing@example.com"}],
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00",  # no offset
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "error"
    assert "start_time" in result["message"]
    assert fake_service._events.update_calls == []
    assert fake_service._freebusy.query_calls == []


def test_update_events_excludes_the_event_being_moved_from_its_own_conflicts(
    fake_service,
):
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
    }
    fake_service._events._list_result = {
        "items": [
            {
                "id": "self-1",
                "summary": "old slot",
                "status": "confirmed",
                "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
                "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
            },
            _confirmed_event(event_id="other-1", summary="Board sync"),
        ]
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"
    assert [c["summary"] for c in result["conflicts"]] == ["Board sync"]


def test_update_events_excludes_a_recurring_events_own_instances_from_its_conflicts(
    fake_service,
):
    """Regression test: events.list(singleEvents=True) expands a recurring
    event into instances that carry their OWN id
    ("<masterId>_<recurrenceStamp>"), never the master's - rescheduling
    the master itself (the normal way to address a recurring series)
    must not report it as conflicting with its own instances just
    because `id` alone never matches `exclude_event_id`. recurringEventId
    is the field that actually names the master on each instance."""
    fake_service._events._get_result = {
        "id": "series-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
    }
    fake_service._events._list_result = {
        "items": [
            {
                "id": "series-1_20260827T010000Z",
                "recurringEventId": "series-1",
                "summary": "Weekly sync (this instance)",
                "status": "confirmed",
                "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
                "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
            },
            _confirmed_event(event_id="other-1", summary="Board sync"),
        ]
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="series-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"
    assert [c["summary"] for c in result["conflicts"]] == ["Board sync"]


def test_update_events_metadata_only_edit_never_checks_conflicts(fake_service):
    """A pure metadata edit (no time/attendee change) must never run the
    conflict check at all - not just tolerate it, since even running the
    check would spuriously self-conflict against the event's own attendees
    (see the two tests below)."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [{"email": "chelsea@example.com"}],
    }
    fake_service._events._list_result = {
        "items": [_confirmed_event(event_id="other-1")]
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            summary="New Title",
        )
    )

    assert result["status"] == "success"
    assert fake_service._events.list_calls == []
    assert fake_service._freebusy.query_calls == []
    assert fake_service._events.update_calls[0]["sendUpdates"] == "none"


def test_update_events_same_window_only_checks_newly_added_attendee(fake_service):
    """Adding an attendee without moving the event must only check the new
    attendee's free/busy - checking an existing attendee against the
    unchanged window would always find the event's own busy block on their
    calendar and falsely report a conflict."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [{"email": "old@example.com"}],
    }
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "old@example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T10:00:00+08:00",
                            "end": "2026-08-27T10:30:00+08:00",
                        }
                    ]
                },
                "new@example.com": {"busy": []},
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            attendees=["old@example.com", "new@example.com"],
        )
    )

    assert result["status"] == "success"
    assert fake_service._events.list_calls == []
    queried_emails = {
        item["id"]
        for call in fake_service._freebusy.query_calls
        for item in call["body"]["items"]
    }
    assert queried_emails == {"new@example.com"}


def test_update_events_adding_the_organizer_as_an_attendee_does_not_self_conflict(
    fake_service,
):
    """Regression test: freebusy.query has no concept of "exclude this
    event", unlike the organizer-calendar path (excluded by id/
    recurringEventId). Adding the organizer's own email as a "new"
    attendee would otherwise be checked via freebusy and always find
    this very event's own busy block on their calendar - the organizer's
    own availability is already covered by the organizer-calendar check,
    so they must be excluded from the freebusy batch."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [],
        "organizer": {"email": "me@example.com"},
    }
    fake_service._events._list_result = {"items": []}
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "me@example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T10:00:00+08:00",
                            "end": "2026-08-27T10:30:00+08:00",
                        }
                    ]
                },
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            attendees=["me@example.com"],
        )
    )

    assert result["status"] == "success"
    assert fake_service._freebusy.query_calls == []
    # The organizer's own calendar must still have been checked - just via
    # the id/recurringEventId-excluding events.list path instead of the
    # self-conflicting freebusy one - not skipped outright.
    assert len(fake_service._events.list_calls) == 1


def test_update_events_adding_the_organizer_as_an_attendee_still_catches_a_real_conflict(
    fake_service,
):
    """Regression test: excluding the organizer from the freebusy batch
    must not also skip checking their calendar altogether. Before this
    fix, adding ONLY the organizer as a new attendee (window otherwise
    unchanged) left both `attendees_to_check` empty (organizer filtered
    out) and `check_organizer` false (window unchanged), so a genuinely
    different conflicting event on the organizer's own calendar was
    silently missed entirely."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [],
        "organizer": {"email": "me@example.com"},
    }
    fake_service._events._list_result = {
        "items": [_confirmed_event(event_id="other-1", summary="Board sync")]
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            attendees=["me@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert [c["summary"] for c in result["conflicts"]] == ["Board sync"]
    assert fake_service._freebusy.query_calls == []


def test_update_events_on_someone_elses_event_reports_the_real_organizer_unchecked(
    fake_service,
):
    """Regression test: check_organizer queries calendarId="primary",
    which is always the AUTHENTICATED CALLER's own calendar, not
    necessarily this event's organizer - a guest with edit permission
    (guestsCanModify) can update an event they didn't organize. Before
    this fix, the organizer's email was unconditionally excluded from the
    freebusy-checked set on the assumption check_organizer already
    covered them, which is only true when the caller IS the organizer;
    here they differ, so the real organizer (boss@example.com) must be
    reported as unchecked rather than silently treated as clear - and
    "primary" (the caller's own calendar) must not be queried and
    mislabeled as the organizer's. With nothing else to check here, the
    write still proceeds (same as any other "can't verify this one
    attendee" gap elsewhere in this module, e.g. one absent from a
    freebusy response) - it just surfaces the gap in the response."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [{"email": "boss@example.com"}],
        "organizer": {"email": "boss@example.com"},
    }
    # Default FakeCalendars() resolves the connected account's own
    # address as "me@example.com" - different from the organizer above.

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T14:00:00+08:00",  # disjoint move
            end_time="2026-08-27T14:30:00+08:00",
        )
    )

    assert result["status"] == "success"
    assert result["unchecked_attendees"] == ["boss@example.com"]
    # Never queried "primary" (the caller's own calendar) as if it were
    # the organizer's, and never freebusy-checked the real organizer
    # either (which would have self-conflicted on their own copy of this
    # event, same as the caller-is-organizer case this exclusion was
    # originally written for).
    assert fake_service._events.list_calls == []
    assert fake_service._freebusy.query_calls == []


def test_update_events_on_someone_elses_event_still_catches_another_attendees_conflict(
    fake_service,
):
    """Companion to the test above: the real organizer being unverifiable
    must not swallow a genuinely different, checkable attendee's real
    conflict."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [{"email": "boss@example.com"}],
        "organizer": {"email": "boss@example.com"},
    }
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "coworker@example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T14:10:00+08:00",
                            "end": "2026-08-27T14:20:00+08:00",
                        }
                    ]
                }
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T14:00:00+08:00",
            end_time="2026-08-27T14:30:00+08:00",
            attendees=["coworker@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert [c["calendar"] for c in result["conflicts"]] == ["coworker@example.com"]
    assert set(result["unchecked_attendees"]) == {"boss@example.com"}


def test_update_events_treats_existing_attendee_case_insensitively(fake_service):
    """Regression test for a review finding: an existing attendee re-passed
    with different casing must still be recognized as "already there" -
    otherwise it's treated as newly-added, gets checked against the
    unchanged window, and always self-conflicts on its own busy block for
    this very event."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [{"email": "old@example.com"}],
    }
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                # This event's own busy block on the existing attendee's
                # calendar - present so the test would fail with a false
                # "conflict" if the case-insensitive exclusion regressed.
                "old@example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T10:00:00+08:00",
                            "end": "2026-08-27T10:30:00+08:00",
                        }
                    ]
                },
                "new@example.com": {"busy": []},
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            attendees=["Old@Example.com", "new@example.com"],
        )
    )

    assert result["status"] == "success"
    queried_emails = {
        item["id"]
        for call in fake_service._freebusy.query_calls
        for item in call["body"]["items"]
    }
    assert queried_emails == {"new@example.com"}


def test_update_events_reports_original_casing_in_conflicts(fake_service):
    """Regression test: falling back to the event's existing attendees (no
    attendees param passed) while moving the time must report a conflict
    using the originally-stored casing, not the lowercased form used
    internally for the case-insensitive membership check."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [{"email": "John.Smith@Example.com"}],
    }
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "john.smith@example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T10:00:00+08:00",
                            "end": "2026-08-27T10:30:00+08:00",
                        }
                    ]
                }
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "John.Smith@Example.com"


def test_update_events_moving_time_checks_organizer_and_all_attendees(fake_service):
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [{"email": "existing@example.com"}],
    }
    fake_service._events._list_result = {
        "items": [_confirmed_event(event_id="other-1")]
    }
    fake_service._freebusy = FakeFreebusy({"calendars": {}})

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"
    assert len(fake_service._events.list_calls) == 1
    queried_emails = {
        item["id"]
        for call in fake_service._freebusy.query_calls
        for item in call["body"]["items"]
    }
    assert queried_emails == {"existing@example.com"}


def test_update_events_rejects_a_reversed_window(fake_service):
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [],
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:30:00+08:00",
            end_time="2026-08-27T10:00:00+08:00",
        )
    )

    assert result["status"] == "error"
    assert "must be after" in result["message"]
    assert fake_service._events.list_calls == []
    assert fake_service._events.update_calls == []


def test_freebusy_lookup_is_case_insensitive(fake_service):
    """Not confirmed against real Google API behavior, but the lookup should
    be defensive either way: a case-mismatched key must not silently
    swallow a real conflict."""
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "Chelsea@Example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T10:00:00+08:00",
                            "end": "2026-08-27T10:30:00+08:00",
                        }
                    ]
                }
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "conflict"


def test_organizer_events_list_tolerates_null_items(fake_service):
    """Regression test for a review finding: the API can return an explicit
    "items": null instead of omitting the key, which must not crash the
    tool with a TypeError."""
    fake_service._events._list_result = {"items": None}

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "success"


def test_freebusy_tolerates_null_calendars(fake_service):
    """Regression test for a review finding: freebusy.query can return an
    explicit "calendars": null, which must not crash the tool with an
    AttributeError."""
    fake_service._freebusy = FakeFreebusy({"calendars": None})

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "success"


def test_update_events_ignore_conflicts_skips_the_check(fake_service):
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
    }
    fake_service._events._list_result = {"items": [_confirmed_event()]}

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    assert len(fake_service._events.update_calls) == 1


def test_organizer_declined_but_opaque_event_is_still_a_conflict(fake_service):
    """Google documents `responseStatus` and `transparency` as independent
    fields with no guaranteed link - declining an invite doesn't reliably
    clear its transparency, so a still-opaque declined event must still be
    treated as busy rather than assumed free."""
    event = _confirmed_event()
    event["attendees"] = [
        {"email": "me@example.com", "self": True, "responseStatus": "declined"}
    ]
    fake_service._events._list_result = {"items": [event]}

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"


def test_organizer_declined_and_transparent_event_is_not_a_conflict(fake_service):
    event = _confirmed_event(transparency="transparent")
    event["attendees"] = [
        {"email": "me@example.com", "self": True, "responseStatus": "declined"}
    ]
    fake_service._events._list_result = {"items": [event]}

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "success"


def test_organizer_accepted_event_is_still_a_conflict(fake_service):
    event = _confirmed_event()
    event["attendees"] = [
        {"email": "me@example.com", "self": True, "responseStatus": "accepted"}
    ]
    fake_service._events._list_result = {"items": [event]}

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"


def test_freebusy_batch_over_limit_is_chunked_into_multiple_calls(fake_service):
    """Regression test: an over-limit attendee list must be split into
    multiple <=50-sized freebusy.query calls and the results merged,
    rather than giving up on checking anyone at all."""
    attendees = [f"person{i}@example.com" for i in range(51)]
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "person0@example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T10:00:00+08:00",
                            "end": "2026-08-27T10:30:00+08:00",
                        }
                    ]
                },
                "person50@example.com": {"busy": []},
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="All hands",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=attendees,
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "person0@example.com"
    batch_sizes = sorted(
        len(call["body"]["items"]) for call in fake_service._freebusy.query_calls
    )
    assert batch_sizes == [1, 50]


def test_freebusy_missing_scope_rejects_the_write(fake_service):
    """A whole-batch 403 means our own credential/scope is insufficient,
    not that a particular attendee's calendar is merely invisible - proceed
    would silently skip the entire conflict check, defeating the feature.
    The write must be rejected rather than degraded to unchecked."""
    fake_service._freebusy = FakeFreebusy(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "error"
    assert "reconnect" in result["message"].lower()
    assert fake_service._events.insert_calls == []


def test_freebusy_missing_scope_rejects_the_write_even_with_ignore_conflicts_false(
    fake_service,
):
    """`ignore_conflicts` is the caller's explicit escape hatch - without it,
    a scope-insufficient 403 must reject rather than silently proceed."""
    fake_service._freebusy = FakeFreebusy(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["chelsea@example.com"],
            ignore_conflicts=False,
        )
    )

    assert result["status"] == "error"
    assert fake_service._events.insert_calls == []


def test_freebusy_missing_scope_can_be_bypassed_with_ignore_conflicts(fake_service):
    fake_service._freebusy = FakeFreebusy(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["chelsea@example.com"],
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    assert len(fake_service._events.insert_calls) == 1


def test_freebusy_missing_scope_still_reports_an_already_confirmed_conflict(
    fake_service,
):
    """Regression test: a real conflict already confirmed before the scope
    error hit (here, the organizer's own calendar, which doesn't need the
    freebusy scope at all) must not be silently discarded just because a
    later, unrelated attendee check then hit the same missing-scope 403 -
    reporting a known conflict is always safe, and blindly rejecting
    instead would tell the caller to retry with ignore_conflicts=true,
    which would then book straight over the real conflict nobody ever
    mentioned."""
    fake_service._events._list_result = {"items": [_confirmed_event()]}
    fake_service._freebusy = FakeFreebusy(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "1:1 with Hazel"
    assert result["unchecked_attendees"] == ["chelsea@example.com"]
    assert fake_service._events.insert_calls == []


def test_freebusy_missing_scope_legacy_only_body_still_rejects(fake_service):
    """The dual-format body is what a real Calendar API 403 actually
    carries, but a response with only the legacy `errors[]` field must
    still be recognized - not just the newer `details[]` shape."""
    fake_service._freebusy = FakeFreebusy(
        raise_error=_legacy_only_insufficient_scope_error()
    )

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "error"
    assert fake_service._events.insert_calls == []


def test_is_insufficient_scope_error_matches_details_only_body():
    """Regression test for the actual bug: HttpError._get_reason() prefers
    `details` over `errors` whenever a response carries both, dropping
    "insufficientPermissions" from str(exc) entirely - a plain substring
    check on str(exc) would miss this. Must check the parsed body's
    fields directly instead."""
    details_only = HttpError(
        _FakeHttpResp(403),
        (b'{"error": {"details": [{"reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}]}}'),
    )
    assert calendar._is_insufficient_scope_error(details_only) is True


def test_is_insufficient_scope_error_rejects_unrelated_403():
    other = HttpError(
        _FakeHttpResp(403),
        b'{"error": {"errors": [{"reason": "forbiddenForNonOrganizer"}]}}',
    )
    assert calendar._is_insufficient_scope_error(other) is False


def test_is_insufficient_scope_error_tolerates_malformed_body():
    malformed = HttpError(_FakeHttpResp(403), b"not json")
    assert calendar._is_insufficient_scope_error(malformed) is False


def test_is_insufficient_scope_error_tolerates_non_list_error_fields():
    """Regression test: valid JSON whose "errors"/"details" field is
    present but isn't an array (e.g. a boolean/object, from a malformed
    or unexpected error body) must not crash - `x or []` treats a truthy
    non-list value as itself, and iterating a non-iterable raises
    TypeError; must fall through to "can't tell, so False" instead."""
    non_list_fields = HttpError(
        _FakeHttpResp(403), b'{"error": {"errors": true, "details": 42}}'
    )
    assert calendar._is_insufficient_scope_error(non_list_fields) is False


def test_freebusy_other_http_errors_still_propagate(fake_service):
    """A 500 whose body happens to carry a scope-shaped reason must NOT be
    treated as the missing-scope case - the gate is `status == 403` AND a
    matching reason, not the reason alone. A body with no "errors"/
    "details" field at all (as an arbitrary 500 body typically has) would
    fail the reason check regardless of status, so it can't tell an
    intact `status == 403` gate from a broken one; this fixture's body
    would match the reason check, so it actually exercises the status
    gate."""
    other_error = HttpError(
        _FakeHttpResp(500),
        (
            b'{"error": {"message": "boom", '
            b'"errors": [{"reason": "insufficientPermissions"}]}}'
        ),
    )
    fake_service._freebusy = FakeFreebusy(raise_error=other_error)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "error"
    assert "reconnect" not in result["message"].lower()


def test_freebusy_attendee_absent_from_response_is_marked_unchecked(fake_service):
    """An attendee simply missing from calendars (no entry at all, not even
    an "errors" key) must not be silently reported as free."""
    fake_service._freebusy = FakeFreebusy({"calendars": {}})

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Kickoff",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["ghost@example.com"],
        )
    )

    assert result["status"] == "success"
    assert result["unchecked_attendees"] == ["ghost@example.com"]


def test_update_events_checks_conflicts_for_an_all_day_event(fake_service):
    """Regression test: an all-day event stores its bounds under "date", not
    "dateTime" - previously existing_start/existing_end came back None for
    it, so the whole conflict check (and thus adding attendees) silently
    skipped, even though sendUpdates="all" would still fire real invites.

    Adding an attendee without moving the (all-day) window only checks the
    newly-added attendee, same as the timed-event case - so this asserts
    the freebusy query actually ran, with the date widened into a valid
    RFC3339 bound, rather than an organizer-side conflict.
    """
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"date": "2026-08-27"},
        "end": {"date": "2026-08-28"},
        "attendees": [],
    }
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "new@example.com": {
                    "busy": [
                        {"start": "2026-08-27T09:00:00Z", "end": "2026-08-27T10:00:00Z"}
                    ]
                }
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            attendees=["new@example.com"],
        )
    )

    assert result["status"] == "conflict"
    # The fake calendar's own timezone also happens to be UTC, so a
    # skipped lookup (needs_real_calendar_timezone wrongly False, falling
    # through to the hardcoded "UTC" literal) would produce this exact
    # same timeMin/timeMax undetected - assert the lookup actually ran,
    # not just that its result looks right.
    assert fake_service._calendars.get_calls
    query_call = fake_service._freebusy.query_calls[0]
    assert query_call["body"]["timeMin"] == "2026-08-27T00:00:00+00:00"
    assert query_call["body"]["timeMax"] == "2026-08-28T00:00:00+00:00"


def test_update_events_rejects_single_boundary_change_on_an_all_day_event(
    fake_service,
):
    """Regression test: an all-day event's start/end are {"date": ...} -
    writing only one of start_time/end_time would overwrite just that side
    with {"dateTime": ...} while the other side keeps its {"date": ...}
    shape, which Google rejects as a malformed mixed body. Must fail
    loudly with an actionable message instead of sending that request."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"date": "2026-08-27"},
        "end": {"date": "2026-08-28"},
        "attendees": [],
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
        )
    )

    assert result["status"] == "error"
    assert "all-day" in result["message"].lower()
    assert fake_service._events.update_calls == []


def test_update_events_allows_replacing_both_boundaries_on_an_all_day_event(
    fake_service,
):
    """Both start_time and end_time together are a full, unambiguous
    replacement of an all-day event's bounds, so this must succeed."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"date": "2026-08-27"},
        "end": {"date": "2026-08-28"},
        "attendees": [],
    }
    fake_service._events._list_result = {"items": []}

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
        )
    )

    assert result["status"] == "success"


def test_update_events_widens_all_day_boundary_in_the_calendars_own_timezone(
    fake_service,
):
    """Regression test: an all-day event's date is a day on the
    *calendar's own* calendar, not a UTC day - hardcoding "T00:00:00Z"
    shifts the queried window by the calendar's own UTC offset instead of
    asking what timezone the calendar is actually configured in."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"date": "2026-08-27"},
        "end": {"date": "2026-08-28"},
        "attendees": [],
    }
    fake_service._calendars = FakeCalendars(timezone="Asia/Singapore")
    fake_service._freebusy = FakeFreebusy(
        {"calendars": {"new@example.com": {"busy": []}}}
    )

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            attendees=["new@example.com"],
        )
    )

    assert result["status"] == "success"
    query_call = fake_service._freebusy.query_calls[0]
    assert query_call["body"]["timeMin"] == "2026-08-27T00:00:00+08:00"
    assert query_call["body"]["timeMax"] == "2026-08-28T00:00:00+08:00"
    assert fake_service._calendars.get_calls  # actually looked it up


def test_update_events_organizer_only_new_attendee_on_all_day_event_widens_correctly(
    fake_service,
):
    """Regression test: adding ONLY the organizer as a new attendee on an
    all-day event, with the window otherwise unchanged, must still look
    up the calendar's real timezone before checking their calendar - the
    organizer-check that `organizer_newly_added` triggers on its own
    (attendees_to_check ends up empty, the organizer is filtered out of
    it) needs a correctly-widened window just as much as an ordinary new
    attendee would, not one silently defaulted to UTC."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"date": "2026-08-27"},
        "end": {"date": "2026-08-28"},
        "attendees": [],
        "organizer": {"email": "me@example.com"},
    }
    fake_service._calendars = FakeCalendars(timezone="Asia/Singapore")
    fake_service._events._list_result = {"items": []}

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            attendees=["me@example.com"],
        )
    )

    assert result["status"] == "success"
    assert fake_service._calendars.get_calls  # actually looked it up
    list_call = fake_service._events.list_calls[0]
    assert list_call["timeMin"] == "2026-08-27T00:00:00+08:00"
    assert list_call["timeMax"] == "2026-08-28T00:00:00+08:00"


def test_update_events_all_day_with_existing_attendees_resubmitted_unchanged_skips_the_timezone_lookup(
    fake_service,
):
    """Regression test: an all-day event that already has attendees,
    resubmitted with neither a new window nor new attendees (e.g. a
    summary-only edit), must not trigger the calendar's real timezone
    lookup at all - a resubmission that never moves the window makes any
    retained attendee's delta segment trivially empty regardless of
    timezone precision, so needs_real_calendar_timezone correctly stays
    False here. Pins that this genuinely common "just editing the title"
    case doesn't regress into an unnecessary API call (or a scope-403 for
    a token that predates calendar.calendars.readonly, on an edit that
    never needed timezone precision at all)."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"date": "2026-08-27"},
        "end": {"date": "2026-08-28"},
        "attendees": [{"email": "existing@example.com"}],
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            summary="Renamed",
        )
    )

    assert result["status"] == "success"
    assert fake_service._calendars.get_calls == []
    assert fake_service._events.list_calls == []
    assert fake_service._freebusy.query_calls == []


def test_update_events_calendar_timezone_lookup_missing_scope_rejects_the_write(
    fake_service,
):
    """The calendars.get call needed to widen an all-day boundary needs its
    own calendar.calendars.readonly scope - a token that predates that
    migration must reject the write outright, same as a missing
    freebusy scope, rather than silently guessing at UTC."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"date": "2026-08-27"},
        "end": {"date": "2026-08-28"},
        "attendees": [],
    }
    fake_service._calendars = FakeCalendars(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            attendees=["new@example.com"],
        )
    )

    assert result["status"] == "error"
    assert "reconnect" in result["message"].lower()
    assert fake_service._events.update_calls == []


def test_update_events_summary_only_edit_of_all_day_event_never_looks_up_calendar_timezone(
    fake_service,
):
    """Regression test: a summary/location/description-only edit of an
    all-day event never moves its window or touches attendees, so it has
    no use for the calendar's real timezone - looking it up anyway would
    needlessly block this benign edit behind a calendar.calendars.readonly
    scope-403 for every pre-migration token, a far bigger blast radius
    than the migration's stated purpose (all-day conflict checks)."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"date": "2026-08-27"},
        "end": {"date": "2026-08-28"},
        "attendees": [],
    }
    fake_service._calendars = FakeCalendars(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            summary="Renamed",
        )
    )

    assert result["status"] == "success"
    assert fake_service._calendars.get_calls == []
    assert len(fake_service._events.update_calls) == 1


def test_update_events_calendar_timezone_lookup_missing_scope_falls_back_with_ignore_conflicts(
    fake_service,
):
    """ignore_conflicts is the caller's explicit escape hatch - it must not
    be defeated by an unrelated all-day-timezone-lookup failure; the
    write still proceeds, just without a real calendar timezone."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"date": "2026-08-27"},
        "end": {"date": "2026-08-28"},
        "attendees": [],
    }
    fake_service._calendars = FakeCalendars(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    assert len(fake_service._events.update_calls) == 1


def test_update_events_timed_event_never_looks_up_calendar_timezone(fake_service):
    """The calendar-timezone lookup is only needed to widen an all-day
    boundary - a plain timed-event update must not pay for it."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [],
    }

    result = json.loads(
        calendar.google_calendar_update_events(event_id="self-1", summary="Renamed")
    )

    assert result["status"] == "success"
    assert fake_service._calendars.get_calls == []


def test_update_events_partial_overlap_nudge_only_checks_the_new_delta_segment(
    fake_service,
):
    """Regression test: nudging a meeting to a window that still overlaps
    its old one (e.g. 10:00-10:30 -> 10:15-10:45) must only check existing
    attendees against the genuinely new sliver (10:30-10:45) - the
    retained 10:15-10:30 overlap still contains this event's own busy
    block on their calendars, which isn't a real conflict. Checking the
    delta segment (rather than skipping the check outright) still catches
    a real conflict that happens to land in the newly-added time."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [{"email": "existing@example.com"}],
    }
    fake_service._events._list_result = {"items": []}
    # The fake, unlike the real freebusy.query endpoint, doesn't filter its
    # configured busy blocks down to the requested [timeMin, timeMax) - so
    # this leaves it empty and instead asserts directly on the query's
    # timeMin/timeMax below. That's the actual behavior under our control;
    # Google's own server is responsible for only returning busy blocks
    # that overlap whatever window we send it.
    fake_service._freebusy = FakeFreebusy({"calendars": {}})

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:15:00+08:00",
            end_time="2026-08-27T10:45:00+08:00",
        )
    )

    assert result["status"] == "success"
    assert len(fake_service._freebusy.query_calls) == 1
    query_call = fake_service._freebusy.query_calls[0]
    assert query_call["body"]["timeMin"] == "2026-08-27T10:30:00+08:00"
    assert query_call["body"]["timeMax"] == "2026-08-27T10:45:00+08:00"


def test_update_events_partial_overlap_nudge_still_catches_a_real_conflict(
    fake_service,
):
    """The delta segment isn't just a smaller no-op window - a genuine
    conflict that only exists in the newly-added time must still be
    caught."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [{"email": "existing@example.com"}],
    }
    fake_service._events._list_result = {"items": []}
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "existing@example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T10:35:00+08:00",
                            "end": "2026-08-27T10:40:00+08:00",
                        }
                    ]
                },
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:15:00+08:00",
            end_time="2026-08-27T10:45:00+08:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "existing@example.com"


def test_update_events_disjoint_move_checks_existing_attendees(fake_service):
    """The counterpart to the partial-overlap test: moving to a window
    that's completely disjoint from the old one IS safe to check every
    existing attendee against, since this event's own busy block can't
    show up in a window it doesn't occupy."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [{"email": "existing@example.com"}],
    }
    fake_service._events._list_result = {"items": []}
    fake_service._freebusy = FakeFreebusy(
        {
            "calendars": {
                "existing@example.com": {
                    "busy": [
                        {
                            "start": "2026-08-27T14:00:00+08:00",
                            "end": "2026-08-27T14:30:00+08:00",
                        }
                    ]
                },
            }
        }
    )

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T14:00:00+08:00",
            end_time="2026-08-27T14:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "existing@example.com"


def test_update_events_empty_string_attendees_is_treated_as_not_provided(
    fake_service,
):
    """Regression test: an accidental empty string (a stray default from a
    template, or an LLM passing "" instead of omitting the argument) must
    not be treated as a genuine attendees argument - attendees only ever
    adds people, so there'd be nothing to add either way, but this
    confirms the event's existing attendee data is left alone rather than
    triggering any attendee-processing path at all."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [{"email": "existing@example.com", "responseStatus": "accepted"}],
    }

    result = json.loads(
        calendar.google_calendar_update_events(event_id="self-1", attendees="")
    )

    assert result["status"] == "success"
    update_call = fake_service._events.update_calls[0]
    assert update_call["body"]["attendees"] == [
        {"email": "existing@example.com", "responseStatus": "accepted"}
    ]
    assert update_call["sendUpdates"] == "none"


def test_update_events_preserves_attendee_rsvp_state_on_resubmission(fake_service):
    """Re-passing the same attendee list must not wipe responseStatus etc.
    by replacing the whole array with bare {"email": ...} objects."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [
            {
                "email": "existing@example.com",
                "responseStatus": "accepted",
                "comment": "looking forward to it",
            }
        ],
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            attendees=["existing@example.com"],
        )
    )

    assert result["status"] == "success"
    update_call = fake_service._events.update_calls[0]
    assert update_call["body"]["attendees"] == [
        {
            "email": "existing@example.com",
            "responseStatus": "accepted",
            "comment": "looking forward to it",
        }
    ]
    # Byte-identical resubmission is not a real change - no notification.
    assert update_call["sendUpdates"] == "none"


def test_update_events_treats_equivalent_timestamp_formats_as_unchanged(
    fake_service,
):
    """A same-instant resubmission written in a different (but equivalent)
    format must not be treated as a real time change - that would run the
    organizer check on the same window and self-conflict."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T02:00:00+00:00"},
        "end": {"dateTime": "2026-08-27T02:30:00+00:00"},
        "attendees": [{"email": "existing@example.com"}],
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",  # same instant, "Z"-free offset form
            end_time="2026-08-27T10:30:00+08:00",
            location="Room 4",
        )
    )

    assert result["status"] == "success"
    assert fake_service._events.list_calls == []
    assert fake_service._freebusy.query_calls == []
    assert fake_service._events.update_calls[0]["sendUpdates"] == "none"


def test_update_events_summary_only_edit_never_revalidates_the_existing_window(
    fake_service,
):
    """Regression test: an update that never touches start_time/end_time
    isn't about to write any window at all, so it must not re-validate
    the event's already-stored start/end - a zero-duration or malformed
    pre-existing window (e.g. from another client/tool) would otherwise
    reject an unrelated summary edit that never asked to change the
    time."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:00:00+08:00"},  # zero-duration
        "attendees": [],
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            summary="Renamed",
        )
    )

    assert result["status"] == "success"


def test_update_events_respects_sibling_timezone_on_an_offsetless_datetime(
    fake_service,
):
    """Regression test: Google's own docs say dateTime "may" omit its
    offset when a sibling timeZone field is present instead - treating
    that offsetless value as if it belonged to some other zone (e.g. the
    calendar's, or UTC) would misjudge a same-instant resubmission as a
    real time change and run the organizer check against the event's own
    now-"moved" window. The sibling zone here (Asia/Singapore, +08:00) is
    deliberately NOT the code's own UTC fallback, so this only passes if
    the sibling timeZone field is actually read rather than ignored -
    falling back to UTC would compute a different instant and treat this
    resubmission as a real move."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "Asia/Singapore"},
        "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "Asia/Singapore"},
        "attendees": [{"email": "existing@example.com"}],
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T02:00:00+08:00",  # same instant, explicit offset
            end_time="2026-08-27T02:30:00+08:00",
            location="Room 4",
        )
    )

    assert result["status"] == "success"
    assert fake_service._events.list_calls == []
    assert fake_service._freebusy.query_calls == []
    assert fake_service._events.update_calls[0]["sendUpdates"] == "none"


def test_update_events_missing_scope_rejects_the_write(fake_service):
    """A whole-batch 403 while checking the newly-added attendee is our own
    credential's problem, not a per-attendee visibility gap - with nothing
    else already confirmed, the write must be rejected outright."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [],
    }
    fake_service._events._list_result = {"items": []}
    fake_service._freebusy = FakeFreebusy(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["outsider@gmail.com"],
        )
    )

    assert result["status"] == "error"
    assert "reconnect" in result["message"].lower()
    assert fake_service._events.update_calls == []


def test_update_events_missing_scope_still_reports_an_already_confirmed_conflict(
    fake_service,
):
    """Regression test: a real conflict already confirmed before the scope
    error hit (here, the organizer's own calendar, which doesn't need the
    freebusy scope at all) must not be silently discarded just because a
    later, unrelated attendee check then hit the same missing-scope 403 -
    reporting a known conflict is always safe, and blindly rejecting
    instead would tell the caller to retry with ignore_conflicts=true,
    which would then book straight over the real conflict nobody ever
    mentioned."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [],
    }
    fake_service._events._list_result = {
        "items": [_confirmed_event(event_id="other-1")]
    }
    fake_service._freebusy = FakeFreebusy(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["outsider@gmail.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "1:1 with Hazel"
    assert result["unchecked_attendees"] == ["outsider@gmail.com"]


def test_update_events_missing_scope_on_retained_attendees_still_reports_the_first_calls_conflict(
    fake_service,
):
    """Regression test for the multi-call accumulation path specifically
    (as opposed to a single _find_conflicts call's own internal one): a
    real conflict found by the FIRST call (organizer, via events.list,
    which needs no extra scope) must survive being merged with a SECOND,
    separate call's InsufficientScopeError (the retained-attendee delta
    segment, via freebusy.query) at google_calendar_update_events' own
    accumulation layer - not just within a single _find_conflicts call."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [{"email": "existing@example.com"}],
    }
    fake_service._events._list_result = {
        "items": [_confirmed_event(event_id="other-1")]
    }
    fake_service._freebusy = FakeFreebusy(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T14:00:00+08:00",  # disjoint move
            end_time="2026-08-27T14:30:00+08:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "1:1 with Hazel"
    assert result["unchecked_attendees"] == ["existing@example.com"]


def test_update_events_missing_scope_in_first_call_still_checks_retained_attendees(
    fake_service,
):
    """Regression test: when the FIRST _find_conflicts call (newly-added
    attendees + organizer) raises InsufficientScopeError but already
    carries an already-confirmed organizer conflict, the accumulation
    layer must still attempt the SECOND call (retained attendees' delta
    segments) rather than returning immediately - otherwise a retained
    attendee is silently dropped from the response entirely: neither
    reported as conflicting nor as unchecked, contradicting the whole
    feature's guarantee that an unverified attendee is always surfaced."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [{"email": "existing@example.com"}],
    }
    fake_service._events._list_result = {
        "items": [_confirmed_event(event_id="other-1")]
    }
    fake_service._freebusy = FakeFreebusy(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T14:00:00+08:00",  # disjoint move
            end_time="2026-08-27T14:30:00+08:00",
            attendees=["new@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "1:1 with Hazel"
    assert set(result["unchecked_attendees"]) == {
        "new@example.com",
        "existing@example.com",
    }


def test_update_events_missing_scope_can_be_bypassed_with_ignore_conflicts(
    fake_service,
):
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [],
    }
    fake_service._events._list_result = {"items": []}
    fake_service._freebusy = FakeFreebusy(raise_error=_insufficient_scope_error())

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["outsider@gmail.com"],
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    assert len(fake_service._events.update_calls) == 1


def test_update_events_conflict_response_still_reports_unchecked_attendees(
    fake_service,
):
    """A conflict found via one source (the organizer's own calendar) must
    not suppress unchecked_attendees info for a different attendee whose
    calendar couldn't be read (absent from the freebusy response, a
    per-attendee gap rather than a whole-batch 403) - a caller acting on
    the conflict still needs to know that attendee was never actually
    checked."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T09:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T09:30:00+08:00"},
        "attendees": [],
    }
    fake_service._events._list_result = {
        "items": [_confirmed_event(event_id="other-1")]
    }
    fake_service._freebusy = FakeFreebusy({"calendars": {}})

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T10:00:00+08:00",
            end_time="2026-08-27T10:30:00+08:00",
            attendees=["ghost@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["unchecked_attendees"] == ["ghost@example.com"]


def test_update_events_widening_both_boundaries_does_not_duplicate_unchecked_attendees(
    fake_service,
):
    """Regression test: widening a window on BOTH sides produces two
    delta segments (window_delta_segments), and a retained attendee who's
    unverifiable for a reason unrelated to which segment ran (here,
    absent from every freebusy response) gets checked - and appended to
    unchecked_attendees - once per segment, not once overall."""
    fake_service._events._get_result = {
        "id": "self-1",
        "start": {"dateTime": "2026-08-27T10:00:00+08:00"},
        "end": {"dateTime": "2026-08-27T10:30:00+08:00"},
        "attendees": [{"email": "ghost@example.com"}],
    }
    fake_service._events._list_result = {"items": []}
    fake_service._freebusy = FakeFreebusy({"calendars": {}})

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="self-1",
            start_time="2026-08-27T09:45:00+08:00",  # widened earlier...
            end_time="2026-08-27T10:45:00+08:00",  # ...and later
        )
    )

    assert result["status"] == "success"
    assert result["unchecked_attendees"] == ["ghost@example.com"]
