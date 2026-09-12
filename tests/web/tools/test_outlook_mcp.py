import json
from typing import Any
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import outlook


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "token")


def _busy_event(
    event_id="existing-1",
    subject="1:1 with Hazel",
    show_as="busy",
    is_cancelled=False,
    response=None,
    timezone=None,
):
    event = {
        "id": event_id,
        "subject": subject,
        "isCancelled": is_cancelled,
        "showAs": show_as,
        "start": {"dateTime": "2026-08-27T10:00:00"},
        "end": {"dateTime": "2026-08-27T10:30:00"},
    }
    if timezone is not None:
        event["start"]["timeZone"] = timezone
        event["end"]["timeZone"] = timezone
    if response is not None:
        event["responseStatus"] = {"response": response}
    return event


def test_graph_request_preserves_http_status(monkeypatch):
    response = Mock(status_code=429, content=b"rate limited", text="rate limited")
    response.raise_for_status.side_effect = outlook.requests.HTTPError(
        "429 Too Many Requests"
    )
    monkeypatch.setattr(outlook.requests, "request", Mock(return_value=response))

    with pytest.raises(outlook._GraphRequestError) as exc_info:
        outlook._graph_request("GET", "/me/calendarView")

    assert exc_info.value.status_code == 429
    assert "rate limited" in str(exc_info.value)


def test_find_conflicts_can_skip_the_organizer_calendar(monkeypatch):
    graph_request = Mock(
        return_value={
            "value": [
                {
                    "scheduleId": "chelsea@example.com",
                    "availabilityView": "0",
                    "scheduleItems": [],
                }
            ]
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    conflicts, unchecked = outlook._find_conflicts(
        "2026-08-27T10:00:00",
        "2026-08-27T10:30:00",
        "UTC",
        ["chelsea@example.com"],
        check_organizer=False,
    )

    assert conflicts == []
    assert unchecked == []
    graph_request.assert_called_once()
    assert graph_request.call_args.args[:2] == ("POST", "/me/calendar/getSchedule")


def test_create_event_detects_organizer_conflict_same_timezone(monkeypatch):
    """Reproduces the reported bug: booking a slot that overlaps an existing
    event, both in the same timezone, must be caught rather than silently
    created and emailed to attendees."""
    graph_request = Mock(return_value={"value": [_busy_event()]})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            timezone="Asia/Singapore",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "1:1 with Hazel"
    assert result["conflicts"][0]["start_timezone"] == "Asia/Singapore"
    graph_request.assert_called_once()
    assert graph_request.call_args.args[:2] == ("GET", "/me/calendarView")


def test_create_event_ignores_free_and_cancelled_events(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "value": [
                    _busy_event(event_id="e1", show_as="free"),
                    _busy_event(event_id="e2", is_cancelled=True),
                ]
            },
            {"id": "created"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 2
    assert graph_request.call_args.args[:2] == ("POST", "/me/events")


def test_create_event_declined_but_busy_event_is_still_a_conflict(monkeypatch):
    """Regression test matching google_calendar's equivalent check:
    Microsoft documents `responseStatus` and `showAs` as independent
    fields with no guaranteed link - declining an invite doesn't reliably
    clear its `showAs`, so a still-busy declined event must still be
    treated as a conflict rather than assumed free."""
    graph_request = Mock(
        side_effect=[
            {"value": [_busy_event(response="declined")]},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "conflict"


def test_create_event_ignores_organizers_own_declined_and_free_event(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {"value": [_busy_event(response="declined", show_as="free")]},
            {"id": "created"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 2


def test_create_event_detects_attendee_schedule_conflict(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "chelsea@example.com",
                        "scheduleItems": [
                            {
                                "status": "busy",
                                "start": {"dateTime": "2026-08-27T10:00:00"},
                                "end": {"dateTime": "2026-08-27T10:30:00"},
                            }
                        ],
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "chelsea@example.com"
    assert result["conflicts"][0]["start_timezone"] == "UTC"
    assert graph_request.call_count == 2


def test_create_event_reports_all_conflicts_in_the_requested_timezone(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {"value": [_busy_event(timezone="Singapore Standard Time")]},
            {
                "value": [
                    {
                        "scheduleId": "chelsea@example.com",
                        "scheduleItems": [
                            {
                                "status": "busy",
                                "start": {
                                    "dateTime": "2026-08-27T10:00:00",
                                    "timeZone": "Singapore Standard Time",
                                },
                                "end": {
                                    "dateTime": "2026-08-27T10:30:00",
                                    "timeZone": "Singapore Standard Time",
                                },
                            }
                        ],
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            timezone="Asia/Singapore",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert {item["start_timezone"] for item in result["conflicts"]} == {
        "Singapore Standard Time"
    }
    assert graph_request.call_args_list[1].kwargs["extra_headers"] == {
        "Prefer": 'outlook.timezone="Asia/Singapore"'
    }


def test_create_event_accepts_attendees_as_a_comma_separated_string(monkeypatch):
    """attendees is documented as list[str] | str - a comma-separated
    string (as opposed to a list) must be end-to-end equivalent, not just
    accepted by normalize_addresses in isolation."""
    graph_request = Mock(
        side_effect=[
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "chelsea@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    },
                    {
                        "scheduleId": "dana@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    },
                ]
            },
            {"id": "created"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees="chelsea@example.com, dana@example.com",
        )
    )

    assert result["status"] == "success"
    assert "unchecked_attendees" not in result
    create_call = graph_request.call_args_list[-1]
    assert {
        a["emailAddress"]["address"] for a in create_call.kwargs["body"]["attendees"]
    } == {
        "chelsea@example.com",
        "dana@example.com",
    }


def test_create_event_conflict_reports_caller_casing_not_graphs(monkeypatch):
    """Regression test: the conflict's "calendar" field must echo back the
    casing the caller supplied (matching google_calendar's equivalent
    fix), not whatever casing Graph's own scheduleId happens to use in its
    response."""
    graph_request = Mock(
        side_effect=[
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "chelsea@example.com",
                        "scheduleItems": [
                            {
                                "status": "busy",
                                "start": {"dateTime": "2026-08-27T10:00:00"},
                                "end": {"dateTime": "2026-08-27T10:30:00"},
                            }
                        ],
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["Chelsea@Example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "Chelsea@Example.com"


def test_create_event_unchecked_attendee_blocks_creation(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "outsider@gmail.com",
                        "error": {"message": "no access"},
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["outsider@gmail.com"],
        )
    )

    assert result["status"] == "conflict_check_incomplete"
    assert result["unchecked_attendees"] == ["outsider@gmail.com"]
    assert graph_request.call_count == 2


def test_create_event_uses_availability_view_when_schedule_items_are_hidden(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "chelsea@example.com",
                        "availabilityView": "2",
                        "scheduleItems": [],
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "conflict_check_incomplete"
    assert result["unchecked_attendees"] == ["chelsea@example.com"]
    assert graph_request.call_count == 2


def test_create_event_ignore_conflicts_skips_the_check_entirely(monkeypatch):
    graph_request = Mock(return_value={"id": "created"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    graph_request.assert_called_once()
    assert graph_request.call_args.args[:2] == ("POST", "/me/events")


def test_create_event_rejects_a_reversed_window(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:30:00",
            end_datetime="2026-08-27T10:00:00",
        )
    )

    assert result["status"] == "error"
    assert "must be after" in result["message"]
    graph_request.assert_not_called()


def test_create_event_rejects_a_reversed_mixed_offset_window(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:30:00Z",
            end_datetime="2026-08-27T10:00:00",
            timezone="UTC",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "must be after" in result["message"]
    graph_request.assert_not_called()


def test_create_event_rejects_an_invalid_timezone_even_when_checks_are_bypassed(
    monkeypatch,
):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            timezone="Not/ARealZone",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "recognized" in result["message"]
    graph_request.assert_not_called()


@pytest.mark.parametrize("value", ["2026-08-27", "20260827T100000"])
def test_create_event_rejects_non_graph_datetime_shapes(monkeypatch, value):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime=value,
            end_datetime="2026-08-27T10:30:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "extended ISO format" in result["message"]
    graph_request.assert_not_called()


def test_create_event_explains_exclusive_all_day_end(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Company holiday",
            start_datetime="2026-08-27T00:00:00",
            end_datetime="2026-08-27T00:00:00",
            is_all_day=True,
        )
    )

    assert result["status"] == "error"
    assert "exclusive" in result["message"]
    assert "following day" in result["message"]
    graph_request.assert_not_called()


def test_create_event_widens_all_day_conflict_check_to_the_full_day(monkeypatch):
    """Regression test: Graph doesn't require start/end to already be
    day-aligned for isAllDay=True, so a caller passing a literal
    business-hours slot with is_all_day=True must still get the WHOLE
    day checked for conflicts - a conflict earlier or later that same
    day would otherwise be silently missed."""
    graph_request = Mock(
        side_effect=[
            {
                "value": [
                    {
                        "id": "other-1",
                        "subject": "Early standup",
                        "isCancelled": False,
                        "showAs": "busy",
                        "start": {"dateTime": "2026-08-27T08:00:00"},
                        "end": {"dateTime": "2026-08-27T08:30:00"},
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Company holiday",
            start_datetime="2026-08-27T09:00:00",
            end_datetime="2026-08-27T17:00:00",
            is_all_day=True,
        )
    )

    assert result["status"] == "conflict"
    calendar_view_call = graph_request.call_args_list[0]
    assert calendar_view_call.kwargs["params"]["startDateTime"] == (
        "2026-08-27T00:00:00+00:00"
    )
    assert calendar_view_call.kwargs["params"]["endDateTime"] == (
        "2026-08-28T00:00:00+00:00"
    )


def test_create_event_preserves_an_exclusive_z_suffixed_all_day_end(monkeypatch):
    graph_request = Mock(side_effect=[{"value": []}, {"id": "created"}])
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Company holiday",
            start_datetime="2026-08-27T00:00:00Z",
            end_datetime="2026-08-28T00:00:00Z",
            is_all_day=True,
        )
    )

    assert result["status"] == "success"
    calendar_view_call = graph_request.call_args_list[0]
    assert calendar_view_call.kwargs["params"]["startDateTime"] == (
        "2026-08-27T00:00:00+00:00"
    )
    assert calendar_view_call.kwargs["params"]["endDateTime"] == (
        "2026-08-28T00:00:00+00:00"
    )
    create_call = graph_request.call_args_list[1]
    assert create_call.kwargs["body"]["start"]["dateTime"] == "2026-08-27T00:00:00"
    assert create_call.kwargs["body"]["end"]["dateTime"] == "2026-08-28T00:00:00"


def test_create_event_converts_offset_all_day_boundaries_to_event_timezone(
    monkeypatch,
):
    graph_request = Mock(side_effect=[{"value": []}, {"id": "created"}])
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Company holiday",
            start_datetime="2026-08-27T20:00:00Z",
            end_datetime="2026-08-28T20:00:00Z",
            timezone="Asia/Singapore",
            is_all_day=True,
        )
    )

    assert result["status"] == "success"
    payload = graph_request.call_args_list[1].kwargs["body"]
    assert payload["start"]["dateTime"] == "2026-08-28T00:00:00"
    # The supplied end falls on Aug 29 in Singapore, which is already the
    # exclusive date boundary for the Aug 28 all-day event.
    assert payload["end"]["dateTime"] == "2026-08-29T00:00:00"


def test_create_event_rejects_reversed_all_day_window_after_timezone_conversion(
    monkeypatch,
):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Company holiday",
            start_datetime="2026-08-27T20:00:00Z",
            end_datetime="2026-08-27T00:00:00",
            timezone="Asia/Singapore",
            is_all_day=True,
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "exclusive" in result["message"]
    graph_request.assert_not_called()


def test_create_event_normalizes_all_day_payload_when_conflicts_are_ignored(
    monkeypatch,
):
    graph_request = Mock(return_value={"id": "created"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Company holiday",
            start_datetime="2026-08-27T09:00:00",
            end_datetime="2026-08-27T17:00:00",
            is_all_day=True,
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    create_call = graph_request.call_args
    assert create_call.args[:2] == ("POST", "/me/events")
    assert create_call.kwargs["body"]["start"]["dateTime"] == "2026-08-27T00:00:00"
    assert create_call.kwargs["body"]["end"]["dateTime"] == "2026-08-28T00:00:00"
    assert create_call.kwargs["body"]["isAllDay"] is True


def test_create_event_all_day_bounds_follow_dst_offsets(monkeypatch):
    graph_request = Mock(side_effect=[{"value": []}, {"id": "created"}])
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="DST day",
            start_datetime="2026-11-01T09:00:00",
            end_datetime="2026-11-01T17:00:00",
            timezone="America/New_York",
            is_all_day=True,
        )
    )

    assert result["status"] == "success"
    params = graph_request.call_args_list[0].kwargs["params"]
    assert params["startDateTime"] == "2026-11-01T00:00:00-04:00"
    assert params["endDateTime"] == "2026-11-02T00:00:00-05:00"


def test_create_event_batch_over_limit_is_chunked_into_multiple_calls(
    monkeypatch,
):
    """Regression test: an over-limit attendee list must be split into
    multiple <=20-sized getSchedule calls and the results merged, rather
    than giving up on checking anyone at all."""
    attendees = [f"person{i}@example.com" for i in range(21)]

    def fake_get_schedule(batch: list[str]) -> dict[str, Any]:
        return {
            "value": [
                {
                    "scheduleId": email,
                    "availabilityView": (
                        "2" if email == "person0@example.com" else "0"
                    ),
                    "scheduleItems": (
                        [{"status": "busy", "start": {}, "end": {}}]
                        if email == "person0@example.com"
                        else []
                    ),
                }
                for email in batch
            ]
        }

    calls: list[list[str]] = []

    def graph_request(method, path, **kwargs):
        if path == "/me/calendarView":
            return {"value": []}
        if path == "/me/calendar/getSchedule":
            batch = kwargs["body"]["schedules"]
            calls.append(batch)
            return fake_get_schedule(batch)
        assert (method, path) == ("POST", "/me/events")
        return {"id": "created"}

    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="All hands",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=attendees,
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "person0@example.com"
    assert sorted(len(batch) for batch in calls) == [1, 20]


def test_create_event_missing_schedule_scope_rejects_the_write(monkeypatch):
    """A whole-batch 403 means our own credential/policy is insufficient,
    not that a particular attendee's schedule is merely invisible -
    proceeding would silently skip the entire conflict check, defeating
    the feature. The write must be rejected rather than degraded to
    unchecked."""
    graph_request = Mock(
        side_effect=[
            {"value": []},  # organizer calendarView
            outlook._GraphRequestError("403 Forbidden", status_code=403),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "error"
    # The message must give an LLM caller something actionable -
    # reconnecting the connector - rather than a bare error string.
    assert "reconnect" in result["message"].lower()


def test_create_event_organizer_scope_error_is_actionable(monkeypatch):
    graph_request = Mock(
        side_effect=[outlook._GraphRequestError("403 Forbidden", status_code=403)]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "error"
    assert "calendars.read" in result["message"].lower()
    assert "reconnect" in result["message"].lower()
    graph_request.assert_called_once()


def test_create_event_missing_schedule_scope_still_reports_an_already_confirmed_conflict(
    monkeypatch,
):
    """Regression test: a real conflict already confirmed before the scope
    error hit (here, the organizer's own calendar, which doesn't need the
    schedule scope at all) must not be silently discarded just because a
    later, unrelated attendee check then hit the same missing-scope 403 -
    reporting a known conflict is always safe, and blindly rejecting
    instead would tell the caller to retry with ignore_conflicts=true,
    which would then book straight over the real conflict nobody ever
    mentioned."""
    graph_request = Mock(
        side_effect=[
            {"value": [_busy_event(event_id="other-1", subject="Board sync")]},
            outlook._GraphRequestError("403 Forbidden", status_code=403),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "Board sync"
    assert result["unchecked_attendees"] == ["chelsea@example.com"]
    assert "check_error" in result
    assert "reconnect" in result["check_error"].lower()


def test_create_event_missing_schedule_scope_can_be_bypassed_with_ignore_conflicts(
    monkeypatch,
):
    graph_request = Mock(return_value={"id": "created"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["chelsea@example.com"],
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 1


def test_create_event_other_graph_errors_still_propagate(monkeypatch):
    """Only a 403 is treated as a missing-scope signal; any other error must
    still surface as a real failure. Asserting the actual message (not just
    status="error") matters here - an exhausted mock raises its own
    StopIteration on an unexpected extra call, which this same try/except
    would also report as status="error", so a bare status check alone
    could pass for the wrong reason if a future change altered the call
    count instead of genuinely propagating this error."""
    graph_request = Mock(
        side_effect=[
            {"value": []},  # organizer calendarView
            outlook._GraphRequestError("500 boom", status_code=500),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "error"
    assert "500 boom" in result["message"]


def test_create_event_attendee_absent_from_schedule_response_is_unchecked(
    monkeypatch,
):
    """An attendee simply missing from getSchedule's response (no entry at
    all, not even an error) must not be silently reported as free."""
    graph_request = Mock(
        side_effect=[
            {"value": []},  # organizer calendarView
            {"value": []},  # getSchedule - no entry at all for the attendee
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["ghost@example.com"],
        )
    )

    assert result["status"] == "conflict_check_incomplete"
    assert result["unchecked_attendees"] == ["ghost@example.com"]
    assert graph_request.call_count == 2


def test_create_event_treats_unknown_showas_as_a_conflict(monkeypatch):
    """Graph's showAs enum documents "unknown" as simply unclassified (e.g.
    an event synced from a third-party calendar that never set it) - it can
    represent a genuinely busy block, so it must not be silently treated as
    free. Missing a real conflict is worse than an occasional over-flag the
    caller can dismiss with ignore_conflicts."""
    graph_request = Mock(
        return_value={
            "value": [
                {
                    "id": "other-1",
                    "subject": "Mystery event",
                    "isCancelled": False,
                    "showAs": "unknown",
                    "start": {"dateTime": "2026-08-27T10:00:00"},
                    "end": {"dateTime": "2026-08-27T10:30:00"},
                }
            ]
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "Mystery event"


def test_create_event_calendarview_query_carries_an_explicit_offset(monkeypatch):
    """Regression test for the critical review finding: calendarView's
    startDateTime/endDateTime query parameters are, per Graph's own docs,
    "interpreted using the timezone offset specified in the value" and
    read as UTC if none is present - unlike the request body, they are
    NOT affected by the Prefer header. A naive value there would silently
    be checked against the wrong instant whenever the caller isn't in
    UTC."""
    graph_request = Mock(return_value={"value": []})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            timezone="Asia/Singapore",
        )
    )

    assert result["status"] == "success"
    calendar_view_call = graph_request.call_args_list[0]
    assert calendar_view_call.kwargs["params"]["startDateTime"] == (
        "2026-08-27T10:00:00+08:00"
    )
    assert calendar_view_call.kwargs["params"]["endDateTime"] == (
        "2026-08-27T10:30:00+08:00"
    )


def test_create_event_converts_offset_boundaries_to_datetime_timezone_shape(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "chelsea@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            {"id": "created"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00Z",
            end_datetime="2026-08-27T10:30:00Z",
            timezone="Asia/Singapore",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "success"
    params = graph_request.call_args_list[0].kwargs["params"]
    assert params["startDateTime"] == "2026-08-27T18:00:00+08:00"
    assert params["endDateTime"] == "2026-08-27T18:30:00+08:00"
    schedule_body = graph_request.call_args_list[1].kwargs["body"]
    assert schedule_body["startTime"] == {
        "dateTime": "2026-08-27T18:00:00",
        "timeZone": "Asia/Singapore",
    }
    assert schedule_body["endTime"] == {
        "dateTime": "2026-08-27T18:30:00",
        "timeZone": "Asia/Singapore",
    }
    payload = graph_request.call_args_list[2].kwargs["body"]
    assert payload["start"] == {
        "dateTime": "2026-08-27T18:00:00",
        "timeZone": "Asia/Singapore",
    }
    assert payload["end"] == {
        "dateTime": "2026-08-27T18:30:00",
        "timeZone": "Asia/Singapore",
    }


def test_create_event_accepts_windows_timezone_names(monkeypatch):
    graph_request = Mock(side_effect=[{"value": []}, {"id": "created"}])
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            timezone="China Standard Time",
        )
    )

    assert result["status"] == "success"
    params = graph_request.call_args_list[0].kwargs["params"]
    assert params["startDateTime"] == "2026-08-27T10:00:00+08:00"


def test_create_event_calendarview_follows_pagination(monkeypatch):
    """A tenant with more events than fit on one page must have every page
    checked, not just the first - Graph signals more pages via
    @odata.nextLink rather than ever returning everything at once."""
    graph_request = Mock(
        side_effect=[
            {
                "value": [],
                "@odata.nextLink": (
                    f"{outlook.GRAPH_BASE_URL}/me/calendarView?%24skip=250"
                ),
            },
            {"value": [_busy_event(event_id="other-1", subject="Board sync")]},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "Board sync"
    assert graph_request.call_count == 2
    second_call = graph_request.call_args_list[1]
    assert second_call.args[:2] == ("GET", "/me/calendarView?%24skip=250")
    # The nextLink is a complete, already-parameterized URL - no separate
    # params dict should be re-sent alongside it.
    assert second_call.kwargs.get("params") is None


def test_create_event_fails_closed_when_calendarview_exceeds_page_limit(monkeypatch):
    graph_request = Mock(
        return_value={
            "value": [],
            "@odata.nextLink": f"{outlook.GRAPH_BASE_URL}/me/calendarView?%24skip=250",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "conflict_check_incomplete"
    assert "pagination limit" in result["message"]
    assert graph_request.call_count == outlook._MAX_CALENDAR_VIEW_PAGES


def test_create_event_pagination_failure_preserves_found_conflicts(monkeypatch):
    graph_request = Mock(
        return_value={
            "value": [_busy_event(event_id="other-1", subject="Board sync")],
            "@odata.nextLink": f"{outlook.GRAPH_BASE_URL}/me/calendarView?%24skip=250",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "Board sync"
    assert "pagination limit" in result["check_error"]


@pytest.mark.parametrize(
    "next_link",
    [
        0,
        123,
        "",
        "https://example.com/wrong-host",
        f"{outlook.GRAPH_BASE_URL}evil",
        f"{outlook.GRAPH_BASE_URL}/me/../users/calendarView",
        f"{outlook.GRAPH_BASE_URL}/me/%2e%2e/users/calendarView",
    ],
)
def test_create_event_rejects_invalid_calendarview_next_link(monkeypatch, next_link):
    graph_request = Mock(return_value={"value": [], "@odata.nextLink": next_link})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Kickoff",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "conflict_check_incomplete"
    assert "invalid" in result["message"]
    graph_request.assert_called_once()


def test_send_message_normalizes_and_dedupes_recipients(monkeypatch):
    """Regression test: outlook_send_message's to/cc/bcc go through the same
    normalize_addresses dedup as the calendar tools - a case-variant
    duplicate must collapse to one recipient (keeping the first casing
    seen), not be sent twice."""
    graph_request = Mock(return_value={})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_send_message(
            to=["Alice@Example.com", "alice@example.com", "bob@example.com"],
            subject="Hi",
            body="Hello",
            cc=["Carol@Example.com", "carol@example.com"],
        )
    )

    assert result["status"] == "success"
    message = graph_request.call_args.kwargs["body"]["message"]
    assert message["toRecipients"] == [
        {"emailAddress": {"address": "Alice@Example.com"}},
        {"emailAddress": {"address": "bob@example.com"}},
    ]
    assert message["ccRecipients"] == [
        {"emailAddress": {"address": "Carol@Example.com"}},
    ]


def test_update_event_normalizes_and_dedupes_attendees(monkeypatch):
    graph_request = Mock(return_value={"id": "event-1"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="event-1",
            attendees=["Alice@Example.com", "alice@example.com", "bob@example.com"],
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_args.kwargs["body"]["attendees"] == [
        {"emailAddress": {"address": "Alice@Example.com"}, "type": "required"},
        {"emailAddress": {"address": "bob@example.com"}, "type": "required"},
    ]
