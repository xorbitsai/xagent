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


def test_create_event_treats_working_elsewhere_availability_as_non_blocking(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "chelsea@example.com",
                        "availabilityView": "4",
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
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=["chelsea@example.com"],
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 3


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
        f"{outlook.GRAPH_BASE_URL}/me/%2e%2e%2fusers/calendarView",
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


def test_update_event_excludes_the_event_being_moved_from_its_own_conflicts(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T09:00:00"},
                "end": {"dateTime": "2026-08-27T09:30:00"},
                "attendees": [],
                "isAllDay": False,
            },
            {
                "value": [
                    _busy_event(event_id="self-1", subject="old slot"),
                    _busy_event(event_id="other-1", subject="Board sync"),
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "conflict"
    assert [c["summary"] for c in result["conflicts"]] == ["Board sync"]


def test_update_event_metadata_only_edit_never_checks_conflicts(monkeypatch):
    """A pure metadata edit (no time/attendee change) must never fetch the
    existing event or run the conflict check at all - not just tolerate it,
    since even running the check would spuriously self-conflict against the
    event's own attendees (see the test below)."""
    graph_request = Mock(return_value={"id": "updated"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            subject="New Subject",
        )
    )

    assert result["status"] == "success"
    graph_request.assert_called_once()
    assert graph_request.call_args.args[:2] == ("PATCH", "/me/events/self-1")


def test_update_event_rejects_ambiguous_single_boundary_timezone_change(monkeypatch):
    """Regression test: moving only end_datetime while leaving start
    untouched, with a timezone different from the existing event's, is
    ambiguous - there's no single timezone that correctly describes both
    the unmoved start and the newly given end. Must fail loudly rather
    than silently mis-check conflicts in the wrong timezone."""
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
            "originalStartTimeZone": "Asia/Singapore",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
            timezone="UTC",
        )
    )

    assert result["status"] == "error"
    assert "ambiguous" in result["message"].lower()


def test_update_event_single_boundary_rejects_unresolvable_caller_timezone(
    monkeypatch,
):
    """Regression test: `timezones_could_differ` treats "can't resolve" as
    "benefit of the doubt, not confirmed different" - correct for a zone
    name Graph itself reported (never written verbatim; the PATCH always
    uses `existing_timezone`), but wrong for the CALLER's own `timezone`
    argument. A typo'd/unknown zone string must be rejected outright, not
    silently discarded in favor of the existing zone with no signal to
    the caller that their argument had no effect."""
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
            "originalStartTimeZone": "Pacific Standard Time",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
            timezone="This/IsNotAZone",
        )
    )

    assert result["status"] == "error"
    assert "This/IsNotAZone" in result["message"]
    graph_request.assert_called_once()


def test_update_event_single_boundary_change_with_matching_timezone_succeeds(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "Asia/Singapore",
            },
            {
                "start": {
                    "dateTime": "2026-08-27T10:00:00",
                    "timeZone": "Asia/Singapore",
                },
                "end": {
                    "dateTime": "2026-08-27T10:30:00",
                    "timeZone": "Asia/Singapore",
                },
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
            timezone="Asia/Singapore",
        )
    )

    assert result["status"] == "success"


def test_update_event_single_boundary_change_with_unset_timezone_reuses_existing(
    monkeypatch,
):
    """Regression test: timezone's default must not look like an explicit
    "UTC" request - omitting it entirely (the common case: the caller just
    wants to nudge one boundary) must reuse the existing event's timezone
    rather than being treated as a UTC/existing-timezone mismatch.

    A plain GET (no Prefer header) always returns start/end in UTC
    regardless of the event's actual zone (Microsoft's documented default),
    so the real timezone comes from originalStartTimeZone instead - and the
    untouched boundary's clock value must be re-fetched WITH a matching
    Prefer header rather than trusting the dateTime the first (UTC) GET
    returned.
    """
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "Asia/Singapore",
            },
            {
                "start": {
                    "dateTime": "2026-08-27T10:00:00",
                    "timeZone": "Asia/Singapore",
                },
                "end": {
                    "dateTime": "2026-08-27T10:30:00",
                    "timeZone": "Asia/Singapore",
                },
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
        )
    )

    assert result["status"] == "success"
    zoned_refetch_call = graph_request.call_args_list[1]
    assert zoned_refetch_call.kwargs["extra_headers"] == {
        "Prefer": 'outlook.timezone="Asia/Singapore"'
    }
    calendar_view_call = graph_request.call_args_list[2]
    assert calendar_view_call.kwargs["extra_headers"] == {
        "Prefer": 'outlook.timezone="Asia/Singapore"'
    }
    # Regression test: the conflict check running in the existing event's
    # timezone is only meaningful if the actual PATCH is written in that
    # same timezone - writing it as "UTC" instead would silently move the
    # event by the zone offset despite the check having just verified a
    # different, correct instant.
    patch_call = graph_request.call_args_list[3]
    assert patch_call.kwargs["body"]["end"] == {
        "dateTime": "2026-08-27T11:00:00",
        "timeZone": "Asia/Singapore",
    }


def test_update_event_single_boundary_writes_the_checked_timezone_even_with_ignore_conflicts(
    monkeypatch,
):
    """The same timezone-resolution bug is reachable with ignore_conflicts=
    True too, since the PATCH write itself (not just the skipped check)
    needs the existing event's timezone for a single-boundary change."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "Asia/Singapore",
            },
            {
                "start": {
                    "dateTime": "2026-08-27T10:00:00",
                    "timeZone": "Asia/Singapore",
                },
                "end": {
                    "dateTime": "2026-08-27T10:30:00",
                    "timeZone": "Asia/Singapore",
                },
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 3
    patch_call = graph_request.call_args_list[2]
    assert patch_call.kwargs["body"]["end"] == {
        "dateTime": "2026-08-27T11:00:00",
        "timeZone": "Asia/Singapore",
    }


def test_update_event_single_boundary_fails_loudly_without_original_timezone(
    monkeypatch,
):
    """Regression test: a plain GET always reports start.timeZone as "UTC"
    (Microsoft's documented default) - that must never be mistaken for the
    event's real zone. Only originalStartTimeZone can answer that, so its
    absence must fail loudly rather than silently resolve to "UTC"."""
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
        )
    )

    assert result["status"] == "error"
    assert "originalStartTimeZone" in result["message"]


def test_update_event_single_boundary_rejects_legacy_custom_timezone(monkeypatch):
    """Regression test: a `tzone://Microsoft/Custom` originalStartTimeZone
    (Microsoft's own docs: set for events created in a legacy custom
    Outlook-desktop timezone) is not a real IANA/Windows zone name Graph
    would accept as a Prefer header or a written timeZone - must fail
    loudly rather than pass it straight through to another API call."""
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
            "originalStartTimeZone": "tzone://Microsoft/Custom",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
        )
    )

    assert result["status"] == "error"
    assert "custom timezone" in result["message"].lower()
    assert graph_request.call_count == 1


def test_update_event_both_boundaries_changed_ignores_missing_existing_timezone(
    monkeypatch,
):
    """When both start_datetime and end_datetime are supplied together,
    the existing event's own timeZone is irrelevant to the query - it
    must not be required, even if missing."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00"},
                "end": {"dateTime": "2026-08-27T10:30:00"},
                "attendees": [],
                "isAllDay": False,
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T14:00:00",
            end_datetime="2026-08-27T14:30:00",
            timezone="UTC",
        )
    )

    assert result["status"] == "success"


def test_update_event_toggling_all_day_alone_still_triggers_a_real_check(monkeypatch):
    """Regression test: is_all_day changes the event's effective span even
    when the literal start/end clock values don't move (a 30-minute meeting
    becoming an all-day event now overlaps the organizer's whole day). An
    earlier version of this fix only added is_all_day to the "should we
    even look" gate without teaching window_changed about it, so the
    literal-string comparison stayed unchanged and the check was silently
    skipped anyway - this must now actually run the organizer check."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {"value": [_busy_event(event_id="other-1", subject="Board sync")]},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            is_all_day=True,
        )
    )

    assert result["status"] == "conflict"


def test_update_event_rejects_a_reversed_window(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:30:00",
            end_datetime="2026-08-27T10:00:00",
        )
    )

    assert result["status"] == "error"
    assert "must be after" in result["message"]
    # Only the existing-event GET - no calendarView/getSchedule/PATCH.
    assert graph_request.call_count == 1


def test_all_day_events_are_no_longer_exempt_from_conflict_checks(monkeypatch):
    """Regression check: the create path used to skip the conflict check
    whenever is_all_day was True, with no equivalent skip on the Google
    side - a genuinely overlapping all-day event must now be caught too."""
    graph_request = Mock(return_value={"value": [_busy_event()]})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Offsite",
            start_datetime="2026-08-27T00:00:00",
            end_datetime="2026-08-28T00:00:00",
            is_all_day=True,
        )
    )

    assert result["status"] == "conflict"


def test_update_event_ignore_conflicts_skips_the_check(monkeypatch):
    graph_request = Mock(return_value={"id": "updated"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    graph_request.assert_called_once()
    assert graph_request.call_args.args[:2] == ("PATCH", "/me/events/self-1")


def test_update_event_rejects_a_reversed_window_even_with_ignore_conflicts(
    monkeypatch,
):
    """Regression test: the reversed-window guard is basic input sanity,
    not a conflict-check decision - it must still run when
    ignore_conflicts=True skips the actual conflict check, matching
    google_calendar_update_events and this tool's own create path."""
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:30:00",
            end_datetime="2026-08-27T10:00:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "must be after" in result["message"]
    graph_request.assert_not_called()


def test_update_event_treats_equivalent_timestamp_formats_as_unchanged(monkeypatch):
    """A same-instant resubmission written with different fractional-second
    precision must not be treated as a real time change."""
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T10:00:00.0000000", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T10:30:00.0000000", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            location="Room 4",
        )
    )

    assert result["status"] == "success"
    # Only the existing-event GET and the PATCH - no calendarView/getSchedule.
    assert graph_request.call_count == 2


def test_update_event_subject_only_edit_never_revalidates_the_existing_window(
    monkeypatch,
):
    """Regression test: an update that never touches start_datetime/
    end_datetime isn't about to write any window at all, so it must not
    re-validate the event's already-stored start/end - a zero-duration
    or malformed pre-existing window would otherwise reject an unrelated
    subject edit that never asked to change the time."""
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            subject="Renamed",
        )
    )

    assert result["status"] == "success"


def test_update_event_windows_and_iana_names_for_the_same_zone_are_not_ambiguous(
    monkeypatch,
):
    """Regression test: a single-boundary update whose timezone denotes the
    SAME real zone as the existing event's originalStartTimeZone - just
    spelled differently (a Windows name vs. the equivalent IANA name) -
    must not be rejected as ambiguous. A raw string comparison would
    reject this even though there is no real conflict in what zone either
    boundary is in."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "Pacific Standard Time",
            },
            {
                "start": {
                    "dateTime": "2026-08-27T10:00:00",
                    "timeZone": "Pacific Standard Time",
                },
                "end": {
                    "dateTime": "2026-08-27T10:30:00",
                    "timeZone": "Pacific Standard Time",
                },
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
            timezone="America/Los_Angeles",
        )
    )

    assert result["status"] == "success"


def test_update_event_moving_to_all_day_widens_query_to_the_full_day(monkeypatch):
    """Regression test: converting to an all-day event occupies the whole
    calendar day(s), not just the literal clock-time slot given - the
    conflict check must be widened to that full-day window, or a
    conflict elsewhere that day would be missed."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            is_all_day=True,
            timezone="UTC",
        )
    )

    assert result["status"] == "success"
    calendar_view_call = graph_request.call_args_list[1]
    assert calendar_view_call.kwargs["params"]["startDateTime"] == (
        "2026-08-27T00:00:00+00:00"
    )
    assert calendar_view_call.kwargs["params"]["endDateTime"] == (
        "2026-08-28T00:00:00+00:00"
    )


def test_update_event_moving_to_all_day_does_not_double_widen_a_boundary_already_at_midnight(
    monkeypatch,
):
    """Regression test: effective_end may already BE a well-formed
    exclusive day boundary (exactly midnight) - naive_day_bounds always
    treats its input as needing widening to [that midnight, next
    midnight), so applying it unconditionally here would push an
    already-correct boundary one whole day too far, falsely conflicting
    with a following-day event."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T00:00:00",
            end_datetime="2026-08-28T00:00:00",
            is_all_day=True,
            timezone="UTC",
        )
    )

    assert result["status"] == "success"
    calendar_view_call = graph_request.call_args_list[1]
    assert calendar_view_call.kwargs["params"]["startDateTime"] == (
        "2026-08-27T00:00:00+00:00"
    )
    # Must stay at the 28th, not get pushed to the 29th.
    assert calendar_view_call.kwargs["params"]["endDateTime"] == (
        "2026-08-28T00:00:00+00:00"
    )


def test_update_event_flag_only_all_day_toggle_widens_in_the_events_real_timezone(
    monkeypatch,
):
    """Regression test: a plain GET (no Prefer header) always reports
    start/end in UTC regardless of the event's actual configured zone -
    using that UTC clock value's own date as "the event's real calendar
    day" for a flag-only is_all_day toggle silently widens the query to
    the wrong day whenever the real zone's offset pushes the instant
    across a day boundary from its UTC date. originalStartTimeZone
    reports the real zone; 20:00 UTC is 04:00 the *next* day in
    Asia/Singapore (+08:00), so the correct widened day is the 28th, not
    the UTC date's 27th."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T20:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T20:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "Asia/Singapore",
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            is_all_day=True,
        )
    )

    assert result["status"] == "success"
    calendar_view_call = graph_request.call_args_list[1]
    assert calendar_view_call.kwargs["params"]["startDateTime"] == (
        "2026-08-28T00:00:00+08:00"
    )
    assert calendar_view_call.kwargs["params"]["endDateTime"] == (
        "2026-08-29T00:00:00+08:00"
    )


def test_update_event_unresolvable_original_timezone_falls_back_to_utc_instead_of_erroring(
    monkeypatch,
):
    """Regression test: originalStartTimeZone can be a legacy/exotic
    Windows zone id that isn't in the (necessarily incomplete)
    Windows-to-IANA map - using it as existing_zone anyway without
    checking it actually resolves would crash the later timezone-key
    comparison with a raw ValueError, even for a plain flag-only edit
    that never needed the real zone to be exact in the first place. Must
    gracefully fall back to the UTC-normalized field instead, exactly as
    if originalStartTimeZone were absent."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "Some Exotic Legacy Standard Time",
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            is_all_day=True,
        )
    )

    assert result["status"] == "success"
    calendar_view_call = graph_request.call_args_list[1]
    assert calendar_view_call.kwargs["params"]["startDateTime"] == (
        "2026-08-27T00:00:00+00:00"
    )
    assert calendar_view_call.kwargs["params"]["endDateTime"] == (
        "2026-08-28T00:00:00+00:00"
    )
