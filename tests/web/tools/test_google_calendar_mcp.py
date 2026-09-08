import copy
import json
import re
from unittest.mock import Mock

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
