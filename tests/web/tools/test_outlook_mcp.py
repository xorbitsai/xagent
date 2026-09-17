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


def test_utc_field_in_zone_converts_an_aware_instant_instead_of_relabeling_it():
    assert outlook._utc_field_in_zone(
        {"dateTime": "2026-08-27T10:00:00.0000000-04:00", "timeZone": "UTC"},
        "Asia/Singapore",
    ) == {
        "dateTime": "2026-08-27T22:00:00",
        "timeZone": "Asia/Singapore",
    }


def test_utc_field_in_zone_rejects_malformed_graph_datetime():
    with pytest.raises(ValueError, match="invalid event boundary datetime"):
        outlook._utc_field_in_zone(
            {"dateTime": "not-a-datetime", "timeZone": "UTC"},
            "Asia/Singapore",
        )


def test_utc_field_in_zone_rejects_non_utc_plain_get_boundary():
    with pytest.raises(ValueError, match="non-UTC event boundary"):
        outlook._utc_field_in_zone(
            {"dateTime": "2026-08-27T10:00:00", "timeZone": "Asia/Singapore"},
            "Asia/Singapore",
        )


def test_utc_field_in_zone_rejects_ambiguous_snapshot_boundary():
    with pytest.raises(ValueError, match="local time is ambiguous"):
        outlook._utc_field_in_zone(
            {"dateTime": "2026-11-01T06:30:00", "timeZone": "UTC"},
            "America/New_York",
        )


@pytest.mark.parametrize(
    ("start_datetime", "end_datetime", "expected_path"),
    [
        (None, None, "/me/events"),
        (
            "2026-08-27T00:00:00Z",
            "2026-08-28T00:00:00Z",
            "/me/calendarView",
        ),
    ],
)
def test_list_events_exposes_recurrence_identity(
    monkeypatch, start_datetime, end_datetime, expected_path
):
    graph_request = Mock(return_value={"value": []})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_list_events(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_args.args[:2] == ("GET", expected_path)
    selected_fields = graph_request.call_args.kwargs["params"]["$select"].split(",")
    assert "type" in selected_fields
    assert "seriesMasterId" in selected_fields


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
    assert graph_request.call_args.kwargs["params"] == {
        "startDateTime": "2026-08-27T10:00:00+08:00",
        "endDateTime": "2026-08-27T10:30:00+08:00",
        "$top": 250,
        "$select": "id,subject,start,end,isCancelled,showAs,responseStatus",
    }
    assert graph_request.call_args.kwargs["extra_headers"] == {
        "Prefer": 'outlook.timezone="Asia/Singapore"'
    }


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
    assert [call.args[:2] for call in graph_request.call_args_list] == [
        ("GET", "/me/calendarView"),
        ("POST", "/me/events"),
    ]


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


@pytest.mark.parametrize("status", ["busy", "tentative", "oof", "unknown"])
def test_create_event_detects_attendee_schedule_conflict(monkeypatch, status):
    graph_request = Mock(
        side_effect=[
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "chelsea@example.com",
                        "scheduleItems": [
                            {
                                "status": status,
                                "subject": "Private focus time",
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
    assert result["conflicts"][0]["summary"] == "Private focus time"
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


def test_create_event_treats_working_elsewhere_schedule_as_non_blocking(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "chelsea@example.com",
                        # availabilityView folds workingElsewhere into the
                        # documented free code even though scheduleItems keeps
                        # the more specific status.
                        "availabilityView": "0",
                        "scheduleItems": [{"status": "workingElsewhere"}],
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


def test_create_event_rejects_undocumented_availability_view_code(monkeypatch):
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


def test_create_event_accepts_bare_dates_for_all_day_event(monkeypatch):
    graph_request = Mock(side_effect=[{"value": []}, {"id": "created"}])
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Company holiday",
            start_datetime="2026-08-27",
            end_datetime="2026-08-28",
            timezone="Singapore Standard Time",
            is_all_day=True,
        )
    )

    assert result["status"] == "success"
    payload = graph_request.call_args_list[-1].kwargs["body"]
    assert payload["start"] == {
        "dateTime": "2026-08-27T00:00:00",
        "timeZone": "Singapore Standard Time",
    }
    assert payload["end"] == {
        "dateTime": "2026-08-28T00:00:00",
        "timeZone": "Singapore Standard Time",
    }


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
    assert calls == [attendees[:20], attendees[20:]]


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


def test_create_event_second_schedule_batch_scope_error_rejects_the_write(
    monkeypatch,
):
    attendees = [f"person{i}@example.com" for i in range(21)]
    first_batch = {
        "value": [
            {
                "scheduleId": email,
                "availabilityView": "0",
                "scheduleItems": [],
            }
            for email in attendees[:20]
        ]
    }
    graph_request = Mock(
        side_effect=[
            {"value": []},
            first_batch,
            outlook._GraphRequestError("403 Forbidden", status_code=403),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="All hands",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            attendees=attendees,
        )
    )

    assert result["status"] == "error"
    assert "reconnect" in result["message"].lower()
    assert graph_request.call_count == 3


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
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {
                "value": [
                    {
                        "scheduleId": "Alice@Example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    },
                    {
                        "scheduleId": "bob@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    },
                ]
            },
            {"id": "event-1"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="event-1",
            attendees=["Alice@Example.com", "alice@example.com", "bob@example.com"],
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_args_list[-1].kwargs["body"]["attendees"] == [
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
    assert result["conflicts"][0]["calendar"] == "signed_in_calendar"


def test_update_event_rejects_series_master_schedule_change(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
            "type": "seriesMaster",
            "originalStartTimeZone": "UTC",
            "originalEndTimeZone": "UTC",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="series-master-1",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == "error"
    assert "recurring series master" in result["message"]
    assert "specific occurrence" in result["message"]
    graph_request.assert_called_once()
    assert "type" in graph_request.call_args.kwargs["params"]["$select"]


def test_update_event_allows_confirmed_series_master_schedule_change(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
                "isAllDay": False,
                "type": "seriesMaster",
            },
            {"id": "series-master-1"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="series-master-1",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 2
    assert graph_request.call_args_list[1].args[:2] == (
        "PATCH",
        "/me/events/series-master-1",
    )
    assert graph_request.call_args_list[1].kwargs["extra_headers"] == {
        "If-Match": 'W/"version-1"'
    }


def test_update_event_rejects_series_master_attendee_addition(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
            "type": "seriesMaster",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="series-master-1",
            attendees=["new@example.com"],
        )
    )

    assert result["status"] == "error"
    assert "attendee addition" in result["message"]
    assert "recurring series master" in result["message"]
    graph_request.assert_called_once()


def test_update_event_allows_confirmed_series_master_attendee_addition(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "type": "seriesMaster",
            },
            {"id": "series-master-1"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="series-master-1",
            attendees=["new@example.com"],
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    patch_call = graph_request.call_args_list[1]
    assert patch_call.kwargs["body"]["attendees"] == [
        {
            "emailAddress": {"address": "new@example.com"},
            "type": "required",
        }
    ]
    assert patch_call.kwargs["extra_headers"] == {"If-Match": 'W/"version-1"'}


def test_update_event_rejects_series_master_when_only_the_current_instant_matches(
    monkeypatch,
):
    """Equal snapshot instants do not prove recurrence timezone semantics match."""
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
            "isAllDay": False,
            "type": "seriesMaster",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="series-master-1",
            start_datetime="2026-08-27T09:00:00",
            end_datetime="2026-08-27T09:30:00",
            timezone="UTC",
        )
    )

    assert result["status"] == "error"
    assert "schedule, timezone" in result["message"]
    graph_request.assert_called_once()


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


def test_update_event_metadata_only_edit_ignores_legacy_timezone_argument(monkeypatch):
    graph_request = Mock(return_value={"id": "updated"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            subject="New Subject",
            timezone="UTC",
        )
    )

    assert result["status"] == "success"
    graph_request.assert_called_once()
    assert graph_request.call_args.args[:2] == ("PATCH", "/me/events/self-1")
    assert graph_request.call_args.kwargs["body"] == {"subject": "New Subject"}


def test_update_event_single_boundary_accepts_independent_timezone(monkeypatch):
    """Timed start/end boundaries have independent Graph timezones.

    The unchanged start keeps its stored instant while the changed end is
    written in the caller's timezone. The conflict query expresses both
    instants in that caller timezone without rewriting the untouched start.
    """
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
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
            end_datetime="2026-08-27T11:00:00",
            timezone="America/Los_Angeles",
        )
    )

    assert result["status"] == "success"
    calendar_view_call = graph_request.call_args_list[1]
    assert calendar_view_call.kwargs["params"]["startDateTime"] == (
        "2026-08-26T19:00:00-07:00"
    )
    assert calendar_view_call.kwargs["params"]["endDateTime"] == (
        "2026-08-27T11:00:00-07:00"
    )
    assert calendar_view_call.kwargs["extra_headers"] == {
        "Prefer": 'outlook.timezone="America/Los_Angeles"'
    }
    patch_body = graph_request.call_args_list[2].kwargs["body"]
    assert "start" not in patch_body
    assert patch_body["end"] == {
        "dateTime": "2026-08-27T11:00:00",
        "timeZone": "America/Los_Angeles",
    }


def test_update_event_rejects_ambiguous_untouched_snapshot_boundary(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-11-01T04:30:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-11-01T06:30:00", "timeZone": "UTC"},
            "isAllDay": False,
            "type": "singleInstance",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-11-01T00:30:00",
            timezone="America/New_York",
        )
    )

    assert result["status"] == "error"
    assert "local time is ambiguous" in result["message"]
    assert "provide both boundaries" in result["message"]
    graph_request.assert_called_once_with(
        "GET",
        "/me/events/self-1",
        params={"$select": "start,end,attendees,isAllDay,type"},
    )


def test_update_event_rejects_ambiguous_untouched_snapshot_start(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-11-01T06:30:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-11-01T08:00:00", "timeZone": "UTC"},
            "isAllDay": False,
            "type": "singleInstance",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-11-01T03:30:00",
            timezone="America/New_York",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "local time is ambiguous" in result["message"]
    graph_request.assert_called_once()


def test_update_event_rejects_single_boundary_on_existing_all_day_event(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2027-01-15T00:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2027-01-16T00:00:00", "timeZone": "UTC"},
            "isAllDay": True,
            "originalStartTimeZone": "America/Denver",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2027-01-17T00:00:00",
            timezone="America/Phoenix",
        )
    )

    assert result["status"] == "error"
    assert "existing all-day event" in result["message"]
    assert "both start_datetime and end_datetime" in result["message"]
    graph_request.assert_called_once()


def test_update_event_single_boundary_rejects_unresolvable_caller_timezone(
    monkeypatch,
):
    """A typo in the caller's explicit zone must fail before any Graph call."""
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
    graph_request.assert_not_called()


def test_update_event_flag_only_rejects_unresolvable_caller_timezone(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            is_all_day=True,
            timezone="This/IsNotAZone",
        )
    )

    assert result["status"] == "error"
    assert "This/IsNotAZone" in result["message"]
    graph_request.assert_not_called()


def test_update_event_flag_only_rejects_unused_valid_timezone(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            is_all_day=True,
            timezone="America/Los_Angeles",
        )
    )

    assert result["status"] == "error"
    assert "can only be supplied" in result["message"]
    assert "existing timezone" in result["message"]
    graph_request.assert_not_called()


def test_update_event_single_boundary_change_with_matching_timezone_succeeds(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
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
            end_datetime="2026-08-27T11:00:00",
            timezone="Asia/Singapore",
        )
    )

    assert result["status"] == "success"


def test_update_event_single_boundary_requires_explicit_current_timezone(monkeypatch):
    """Creation-time zones cannot safely stand in for the current boundary zone."""
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
        )
    )

    assert result["status"] == "error"
    assert "timezone is required" in result["message"]
    assert "current timezone" in result["message"]
    graph_request.assert_not_called()


def test_update_event_single_boundary_requires_timezone_even_when_checks_are_bypassed(
    monkeypatch,
):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-08-27T11:00:00",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "timezone is required" in result["message"]
    graph_request.assert_not_called()


def test_update_event_rejects_nonexistent_local_time_when_checks_are_bypassed(
    monkeypatch,
):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-03-08T02:30:00",
            timezone="America/Los_Angeles",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "does not exist" in result["message"]
    assert "daylight-saving transition" in result["message"]
    graph_request.assert_not_called()


def test_update_event_rejects_ambiguous_local_time_when_checks_are_bypassed(
    monkeypatch,
):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2026-11-01T01:30:00",
            timezone="America/New_York",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "ambiguous" in result["message"]
    assert "daylight-saving transition" in result["message"]
    graph_request.assert_not_called()


def test_update_event_rejects_incomplete_snapshot_when_checks_are_bypassed(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
            "isAllDay": False,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:00:00",
            timezone="UTC",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "no complete time window" in result["message"]
    assert "single-boundary update" in result["message"]
    graph_request.assert_called_once()


def test_update_event_single_boundary_with_explicit_timezone_ignores_unusable_stored_zone(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "tzone://Microsoft/Custom",
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
    assert graph_request.call_args_list[2].kwargs["body"]["end"] == {
        "dateTime": "2026-08-27T11:00:00",
        "timeZone": "America/Los_Angeles",
    }


def test_update_event_both_boundaries_changed_ignores_missing_existing_timezone(
    monkeypatch,
):
    """When both start_datetime and end_datetime are supplied together,
    the existing event's own timeZone is irrelevant to the query - it
    must not be required, even if missing."""
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
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


def test_update_event_all_day_transition_requires_caller_boundaries(monkeypatch):
    """Reject before a stale snapshot can be materialized back into the PATCH."""
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
            "isAllDay": False,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(event_id="self-1", is_all_day=True)
    )

    assert result["status"] == "error"
    assert "both start_datetime and end_datetime" in result["message"]
    assert "concurrent schedule change" in result["message"]
    graph_request.assert_called_once()


def test_update_event_all_day_to_timed_transition_requires_caller_boundaries(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T00:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-28T00:00:00", "timeZone": "UTC"},
            "isAllDay": True,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(event_id="self-1", is_all_day=False)
    )

    assert result["status"] == "error"
    assert "both start_datetime and end_datetime" in result["message"]
    graph_request.assert_called_once()


@pytest.mark.parametrize(
    ("existing_is_all_day", "requested_is_all_day", "ignore_conflicts"),
    [(True, None, False), (False, True, False), (True, None, True)],
)
def test_update_event_all_day_window_requires_explicit_timezone(
    monkeypatch, existing_is_all_day, requested_is_all_day, ignore_conflicts
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T00:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-28T00:00:00", "timeZone": "UTC"},
            "isAllDay": existing_is_all_day,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    kwargs = (
        {} if requested_is_all_day is None else {"is_all_day": requested_is_all_day}
    )

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-29T00:00:00",
            end_datetime="2026-08-30T00:00:00",
            ignore_conflicts=ignore_conflicts,
            **kwargs,
        )
    )

    assert result["status"] == "error"
    assert "timezone is required" in result["message"]
    assert "cannot safely default to UTC" in result["message"]
    graph_request.assert_called_once()


@pytest.mark.parametrize("with_conflict", [False, True])
def test_update_event_preserves_partial_results_when_conflict_scan_stops(
    monkeypatch, with_conflict
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
            "originalStartTimeZone": "UTC",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    conflicts = [_busy_event(subject="Board sync")] if with_conflict else []
    monkeypatch.setattr(
        outlook,
        "_find_conflicts",
        Mock(
            side_effect=outlook._ConflictCheckIncompleteError(
                "Outlook calendar conflict check exceeded the pagination limit",
                conflicts,
                [],
            )
        ),
    )

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:00:00",
            end_datetime="2026-08-27T10:30:00",
        )
    )

    assert result["status"] == (
        "conflict" if with_conflict else "conflict_check_incomplete"
    )
    assert "pagination limit" in result.get("check_error", result["message"])
    if with_conflict:
        assert result["conflicts"][0]["subject"] == "Board sync"
    graph_request.assert_called_once()


def test_update_event_reports_real_calendarview_pagination_limit(monkeypatch):
    monkeypatch.setattr(outlook, "_MAX_CALENDAR_VIEW_PAGES", 2)
    next_link = f"{outlook.GRAPH_BASE_URL}/me/calendarView?%24skip=250"
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
                "isAllDay": False,
            },
            {"value": [], "@odata.nextLink": next_link},
            {"value": [], "@odata.nextLink": next_link},
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

    assert result["status"] == "conflict_check_incomplete"
    assert "pagination limit" in result["message"]
    assert graph_request.call_count == 3
    assert graph_request.call_args_list[2].args[:2] == (
        "GET",
        "/me/calendarView?%24skip=250",
    )


def test_update_event_organizer_scope_error_rejects_the_write(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "UTC",
            },
            outlook._GraphRequestError("403 Forbidden", status_code=403),
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

    assert result["status"] == "error"
    assert "calendars.read" in result["message"].lower()
    assert "reconnect" in result["message"].lower()
    assert graph_request.call_count == 2


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
    # A complete caller-supplied window can be rejected before any read or write.
    graph_request.assert_not_called()


def test_update_event_rejects_an_explicit_empty_boundary(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": False,
            "originalStartTimeZone": "UTC",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="",
            timezone="UTC",
        )
    )

    assert result["status"] == "error"
    assert "extended ISO format" in result["message"]
    graph_request.assert_not_called()


def test_update_event_rejects_two_empty_boundaries_before_graph_read(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="",
            end_datetime="",
        )
    )

    assert result["status"] == "error"
    assert "extended ISO format" in result["message"]
    graph_request.assert_not_called()


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
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
                "isAllDay": False,
            },
            {"id": "updated"},
        ]
    )
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
    assert [call.args[:2] for call in graph_request.call_args_list] == [
        ("GET", "/me/events/self-1"),
        ("PATCH", "/me/events/self-1"),
    ]


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


def test_update_event_checks_resubmitted_boundaries_when_current_zone_is_unknown(
    monkeypatch,
):
    """Equal UTC instants do not prove that boundary timezone semantics match."""
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00.0000000", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00.0000000", "timeZone": "UTC"},
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
            location="Room 4",
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 3
    assert graph_request.call_args_list[1].args[:2] == ("GET", "/me/calendarView")


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


def test_update_event_timed_boundary_accepts_timezone_that_differs_on_event_date(
    monkeypatch,
):
    """A timed endpoint may use a timezone independent from the other endpoint."""
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2027-01-15T14:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2027-01-15T14:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "America/Santo_Domingo",
                "originalEndTimeZone": "America/Santo_Domingo",
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            end_datetime="2027-01-15T11:00:00",
            timezone="America/New_York",
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_args_list[2].kwargs["body"]["end"] == {
        "dateTime": "2027-01-15T11:00:00",
        "timeZone": "America/New_York",
    }


def test_update_event_normalizes_offset_boundaries_before_check_and_write(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "UTC",
            },
            {"value": []},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:00:00Z",
            end_datetime="2026-08-27T10:30:00Z",
            timezone="Asia/Singapore",
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_args_list[1].kwargs["params"]["startDateTime"] == (
        "2026-08-27T18:00:00+08:00"
    )
    payload = graph_request.call_args_list[2].kwargs["body"]
    assert payload["start"] == {
        "dateTime": "2026-08-27T18:00:00",
        "timeZone": "Asia/Singapore",
    }
    assert payload["end"] == {
        "dateTime": "2026-08-27T18:30:00",
        "timeZone": "Asia/Singapore",
    }


def test_update_event_moving_to_all_day_widens_query_to_the_full_day(monkeypatch):
    """Regression test: converting to an all-day event occupies the whole
    calendar day(s), not just the literal clock-time slot given - the
    conflict check must be widened to that full-day window, or a
    conflict elsewhere that day would be missed."""
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
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
    payload = graph_request.call_args_list[2].kwargs["body"]
    assert payload["start"] == {
        "dateTime": "2026-08-27T00:00:00",
        "timeZone": "UTC",
    }
    assert payload["end"] == {
        "dateTime": "2026-08-28T00:00:00",
        "timeZone": "UTC",
    }


def test_update_event_rejects_equal_exclusive_end_on_all_day_replacement(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27",
            end_datetime="2026-08-27",
            timezone="UTC",
            is_all_day=True,
        )
    )

    assert result["status"] == "error"
    assert "exclusive" in result["message"]
    graph_request.assert_not_called()


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
                "@odata.etag": 'W/"version-1"',
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


def test_update_event_flag_only_all_day_toggle_never_uses_creation_timezone(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T20:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T20:30:00", "timeZone": "UTC"},
            "isAllDay": False,
            "originalStartTimeZone": "Asia/Singapore",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(event_id="self-1", is_all_day=True)
    )

    assert result["status"] == "error"
    assert "both start_datetime and end_datetime" in result["message"]
    graph_request.assert_called_once()


def test_update_event_same_window_only_checks_newly_added_attendee(monkeypatch):
    """Adding an attendee without moving the event must only check the new
    attendee's schedule - checking an existing attendee against the
    unchanged window would always find the event's own busy block on their
    schedule and falsely report a conflict."""
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    {"emailAddress": {"address": "old@example.com"}},
                ],
            },
            {
                "value": [
                    {
                        "scheduleId": "new@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["old@example.com", "new@example.com"],
        )
    )

    assert result["status"] == "success"
    schedule_call = graph_request.call_args_list[1]
    assert schedule_call.args[:2] == ("POST", "/me/calendar/getSchedule")
    assert schedule_call.kwargs["body"]["schedules"] == ["new@example.com"]
    assert graph_request.call_count == 3


def test_update_event_attendee_only_update_rejects_unused_timezone(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["new@example.com"],
            timezone="America/New_York",
        )
    )

    assert result["status"] == "error"
    assert "timezone can only be supplied" in result["message"]
    graph_request.assert_not_called()


def test_update_event_adding_attendee_to_all_day_event_requires_window(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T00:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-28T00:00:00", "timeZone": "UTC"},
            "attendees": [],
            "isAllDay": True,
            "type": "singleInstance",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["new@example.com"],
        )
    )

    assert result["status"] == "error"
    assert "Provide both boundaries" in result["message"]
    graph_request.assert_called_once()


def test_update_event_all_day_move_with_retained_attendee_fails_closed(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T00:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-28T00:00:00", "timeZone": "UTC"},
            "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
            "isAllDay": True,
            "type": "singleInstance",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-28T00:00:00",
            end_datetime="2026-08-29T00:00:00",
            timezone="America/New_York",
            is_all_day=True,
        )
    )

    assert result["status"] == "error"
    assert "retained attendee conflicts cannot be checked safely" in result["message"]
    graph_request.assert_called_once()


def test_update_event_all_day_same_labels_with_retained_attendee_fails_closed(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T00:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-28T00:00:00", "timeZone": "UTC"},
            "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
            "isAllDay": True,
            "type": "singleInstance",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T00:00:00",
            end_datetime="2026-08-28T00:00:00",
            timezone="America/New_York",
            is_all_day=True,
        )
    )

    assert result["status"] == "error"
    assert "even when the submitted date labels are unchanged" in result["message"]
    graph_request.assert_called_once()


def test_update_event_all_day_narrower_labels_with_retained_attendee_fails_closed(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T00:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-29T00:00:00", "timeZone": "UTC"},
            "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
            "isAllDay": True,
            "type": "singleInstance",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T00:00:00",
            end_datetime="2026-08-28T00:00:00",
            timezone="America/New_York",
            is_all_day=True,
        )
    )

    assert result["status"] == "error"
    assert "or narrower" in result["message"]
    graph_request.assert_called_once()


def test_update_event_all_day_attendee_addition_with_retained_attendee_fails_closed(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "@odata.etag": 'W/"version-1"',
            "start": {"dateTime": "2026-08-27T00:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-28T00:00:00", "timeZone": "UTC"},
            "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
            "isAllDay": True,
            "type": "singleInstance",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T00:00:00",
            end_datetime="2026-08-28T00:00:00",
            timezone="America/New_York",
            is_all_day=True,
            attendees=["existing@example.com", "new@example.com"],
        )
    )

    assert result["status"] == "error"
    assert "retained attendee conflicts cannot be checked safely" in result["message"]
    graph_request.assert_called_once()


def test_update_event_adds_attendee_to_all_day_event_without_retained_attendees(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T00:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-28T00:00:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": True,
                "type": "singleInstance",
            },
            {"value": []},
            {
                "value": [
                    {
                        "scheduleId": "new@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T00:00:00",
            end_datetime="2026-08-28T00:00:00",
            timezone="America/New_York",
            is_all_day=True,
            attendees=["new@example.com"],
        )
    )

    assert result["status"] == "success"
    schedule_call = graph_request.call_args_list[2]
    assert schedule_call.args[:2] == ("POST", "/me/calendar/getSchedule")
    assert schedule_call.kwargs["body"]["schedules"] == ["new@example.com"]
    patch_call = graph_request.call_args_list[-1]
    assert patch_call.kwargs["extra_headers"] == {"If-Match": 'W/"version-1"'}


def test_update_event_treats_existing_attendee_case_insensitively(monkeypatch):
    """Regression test for a review finding: an existing attendee re-passed
    with different casing must still be recognized as "already there" -
    otherwise it's treated as newly-added, gets checked against the
    unchanged window, and always self-conflicts on its own busy block for
    this very event."""
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    {"emailAddress": {"address": "old@example.com"}},
                ],
            },
            {
                "value": [
                    {
                        "scheduleId": "new@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["Old@Example.com", "new@example.com"],
        )
    )

    assert result["status"] == "success"
    schedule_call = graph_request.call_args_list[1]
    assert schedule_call.kwargs["body"]["schedules"] == ["new@example.com"]


def test_update_event_skips_timezone_resolution_when_nothing_new_to_check(monkeypatch):
    """Regression test: re-passing only the attendees the event already has
    (no genuinely new ones, no time/is_all_day change) must not need to
    resolve a timezone at all - even if the existing event happens to be
    missing a timeZone on its start time - since no conflict check ends up
    running. Failing loudly here would block a call that was always a
    no-op for scheduling purposes."""
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T10:00:00"},
            "end": {"dateTime": "2026-08-27T10:30:00"},
            "attendees": [{"emailAddress": {"address": "old@example.com"}}],
            "isAllDay": False,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["old@example.com"],
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 2
    assert graph_request.call_args_list[1].args == ("GET", "/me/events/self-1")


def test_update_event_moving_time_checks_organizer_and_all_attendees(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T09:00:00"},
                "end": {"dateTime": "2026-08-27T09:30:00"},
                "attendees": [
                    {"emailAddress": {"address": "existing@example.com"}},
                ],
            },
            {"value": [_busy_event(event_id="other-1", subject="Board sync")]},
            {
                "value": [
                    {
                        "scheduleId": "existing@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
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
    assert graph_request.call_count == 3


def test_update_event_resubmitting_only_existing_attendees_does_not_touch_them(
    monkeypatch,
):
    """Re-passing only addresses already on the event adds nothing new, so
    the PATCH body must not carry an "attendees" key at all - Graph's PATCH
    is a true partial update, and simply not touching the field is a
    stronger guarantee against wiping RSVP state (status/type etc.) than
    resending a reconstructed copy of it would be."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    {
                        "emailAddress": {"address": "  existing@example.com  "},
                        "type": "required",
                        "status": {
                            "response": "accepted",
                            "time": "2026-08-20T00:00:00Z",
                        },
                    }
                ],
                "isAllDay": False,
            },
            {
                "id": "self-1",
                "subject": "Full event",
                "attendees": [
                    {
                        "emailAddress": {"address": "existing@example.com"},
                        "type": "required",
                        "status": {
                            "response": "accepted",
                            "time": "2026-08-20T00:00:00Z",
                        },
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["existing@example.com"],
        )
    )

    assert result["status"] == "success"
    assert result["message"] == "No attendee changes were needed"
    assert result["event"]["id"] == "self-1"
    assert result["event"]["subject"] == "Full event"
    assert result["event"]["attendees"][0]["status"]["response"] == "accepted"
    assert graph_request.call_count == 2
    assert graph_request.call_args_list[1].args == ("GET", "/me/events/self-1")


def test_update_event_attendees_replace_drops_addresses_left_out_of_the_new_list(
    monkeypatch,
):
    """attendees fully replaces the event's attendee list (this connector's
    pre-existing base behavior) - an existing address left out of the new
    list must actually be removed, not silently kept the way an
    append-only design would."""
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    {
                        "emailAddress": {"address": "keep@example.com"},
                        "type": "required",
                    },
                    {
                        "emailAddress": {"address": "drop@example.com"},
                        "type": "required",
                    },
                ],
                "isAllDay": False,
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["keep@example.com"],
        )
    )

    assert result["status"] == "success"
    patch_call = graph_request.call_args_list[1]
    assert {
        a["emailAddress"]["address"] for a in patch_call.kwargs["body"]["attendees"]
    } == {"keep@example.com"}


def test_update_event_attendees_empty_list_clears_every_attendee(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    {
                        "emailAddress": {"address": "existing@example.com"},
                        "type": "required",
                    },
                ],
                "isAllDay": False,
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=[],
        )
    )

    assert result["status"] == "success"
    patch_call = graph_request.call_args_list[1]
    assert patch_call.kwargs["body"]["attendees"] == []


def test_update_event_attendees_replace_preserves_rsvp_state_for_retained_entries(
    monkeypatch,
):
    """A replace that both keeps an existing attendee and adds a new one
    must reuse the retained entry's own raw dict (preserving its RSVP
    state) while giving the genuinely new address a fresh minimal one."""
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    {
                        "emailAddress": {"address": "  existing@example.com  "},
                        "type": "required",
                        "status": {
                            "response": "accepted",
                            "time": "2026-08-20T00:00:00Z",
                        },
                    }
                ],
                "isAllDay": False,
            },
            # No window/is_all_day change, so check_organizer is False here
            # - only the newly-added attendee's schedule is queried.
            {
                "value": [
                    {
                        "scheduleId": "new@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["existing@example.com", "new@example.com"],
        )
    )

    assert result["status"] == "success"
    patch_call = graph_request.call_args_list[-1]
    by_address = {
        a["emailAddress"]["address"].strip(): a
        for a in patch_call.kwargs["body"]["attendees"]
    }
    assert by_address["existing@example.com"]["status"] == {
        "response": "accepted",
        "time": "2026-08-20T00:00:00Z",
    }
    assert by_address["new@example.com"] == {
        "emailAddress": {"address": "new@example.com"},
        "type": "required",
    }
    assert patch_call.kwargs["extra_headers"] == {"If-Match": 'W/"version-1"'}


def test_update_event_attendee_patch_uses_graph_etag(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"etag-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {
                "value": [
                    {
                        "scheduleId": "new@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(event_id="self-1", attendees=["new@example.com"])
    )

    assert result["status"] == "success"
    assert graph_request.call_args_list[-1].kwargs["extra_headers"] == {
        "If-Match": 'W/"etag-1"'
    }


def test_update_event_attendee_patch_rejects_concurrent_change(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {
                "value": [
                    {
                        "scheduleId": "new@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            outlook._GraphRequestError("412 Precondition Failed", status_code=412),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(event_id="self-1", attendees=["new@example.com"])
    )

    assert result["status"] == "conflict_stale_version"
    assert "changed before this update could be applied" in result["message"]


def test_update_event_schedule_patch_rejects_concurrent_change(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {"value": []},
            outlook._GraphRequestError("412 Precondition Failed", status_code=412),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T11:00:00",
            end_datetime="2026-08-27T11:30:00",
        )
    )

    assert result["status"] == "conflict_stale_version"
    assert "changed before this update could be applied" in result["message"]
    patch_call = graph_request.call_args_list[-1]
    assert patch_call.kwargs["extra_headers"] == {"If-Match": 'W/"version-1"'}


def test_update_event_schedule_patch_requires_event_etag(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {"value": []},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T11:00:00",
            end_datetime="2026-08-27T11:30:00",
        )
    )

    assert result["status"] == "error"
    assert "did not return an event version" in result["message"]
    assert graph_request.call_count == 2


def test_update_event_attendee_patch_does_not_treat_change_key_as_etag(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "changeKey": "not-an-etag",
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {
                "value": [
                    {
                        "scheduleId": "new@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(event_id="self-1", attendees=["new@example.com"])
    )

    assert result["status"] == "error"
    assert "did not return an event version" in result["message"]
    assert graph_request.call_count == 2


def test_update_event_deduplicates_snapshot_attendees_before_delta_checks(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    {"emailAddress": {"address": "Existing@Example.com"}},
                    {"emailAddress": {"address": " existing@example.com "}},
                ],
                "isAllDay": False,
            },
            {"id": "updated"},
        ]
    )
    find_conflicts = Mock(side_effect=[([], []), ([], [])])
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", find_conflicts)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T11:00:00",
            end_datetime="2026-08-27T11:30:00",
        )
    )

    assert result["status"] == "success"
    assert find_conflicts.call_count == 2
    assert find_conflicts.call_args_list[1].args[3] == ["Existing@Example.com"]


def test_update_event_rejects_snapshot_attendee_without_an_address(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
            "attendees": [{"emailAddress": {"address": None}}],
            "isAllDay": False,
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(event_id="self-1", attendees=["new@example.com"])
    )

    assert result["status"] == "error"
    assert "attendee at index 0 has no valid email address" in result["message"]
    graph_request.assert_called_once()


def test_update_event_preserves_first_duplicate_snapshot_attendee(monkeypatch):
    first_attendee = {
        "emailAddress": {"address": "Existing@Example.com"},
        "type": "required",
        "status": {"response": "accepted"},
    }
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    first_attendee,
                    {
                        "emailAddress": {"address": "existing@example.com"},
                        "type": "optional",
                        "status": {"response": "declined"},
                    },
                ],
                "isAllDay": False,
            },
            {
                "value": [
                    {
                        "scheduleId": "new@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["Existing@Example.com", "new@example.com"],
        )
    )

    assert result["status"] == "success"
    patch_call = graph_request.call_args_list[-1]
    assert patch_call.kwargs["body"]["attendees"][0] == first_attendee


def test_update_event_partial_overlap_nudge_only_checks_the_new_delta_segment(
    monkeypatch,
):
    """Regression test: nudging a boundary to a window that still overlaps
    the event's OLD window (10:00-10:30 -> 10:15-10:45) must only check
    existing attendees against the genuinely new sliver (10:30-10:45) -
    the retained 10:15-10:30 overlap still contains this event's own busy
    block on their calendar, which isn't a real conflict."""
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
                "isAllDay": False,
            },
            {"value": []},  # organizer calendarView
            {
                "value": [
                    {
                        "scheduleId": "existing@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:15:00",
            end_datetime="2026-08-27T10:45:00",
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 4
    schedule_call = graph_request.call_args_list[2]
    assert schedule_call.args[:2] == ("POST", "/me/calendar/getSchedule")
    assert (
        schedule_call.kwargs["body"]["startTime"]["dateTime"] == "2026-08-27T10:30:00"
    )
    assert schedule_call.kwargs["body"]["endTime"]["dateTime"] == "2026-08-27T10:45:00"


def test_update_event_partial_overlap_nudge_still_catches_a_real_conflict(
    monkeypatch,
):
    """The delta segment isn't just a smaller no-op window - a genuine
    conflict that only exists in the newly-added time must still be
    caught."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
                "isAllDay": False,
            },
            {"value": []},  # organizer calendarView
            {
                "value": [
                    {
                        "scheduleId": "existing@example.com",
                        "scheduleItems": [
                            {
                                "status": "busy",
                                "start": {"dateTime": "2026-08-27T10:35:00"},
                                "end": {"dateTime": "2026-08-27T10:40:00"},
                            }
                        ],
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T10:15:00",
            end_datetime="2026-08-27T10:45:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "existing@example.com"


def test_update_event_deduplicates_conflict_across_two_delta_segments(monkeypatch):
    busy_schedule = {
        "value": [
            {
                "scheduleId": "existing@example.com",
                "availabilityView": "2",
                "scheduleItems": [
                    {
                        "status": "busy",
                        "subject": "Long-running hold",
                        "start": {"dateTime": "2026-08-27T08:30:00"},
                        "end": {"dateTime": "2026-08-27T12:30:00"},
                    }
                ],
            }
        ]
    }
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T11:00:00", "timeZone": "UTC"},
                "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
                "isAllDay": False,
                "type": "singleInstance",
            },
            {"value": []},
            busy_schedule,
            busy_schedule,
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T09:00:00",
            end_datetime="2026-08-27T12:00:00",
        )
    )

    assert result["status"] == "conflict"
    assert len(result["conflicts"]) == 1
    assert result["conflicts"][0]["summary"] == "Long-running hold"
    first_segment_call = graph_request.call_args_list[2]
    second_segment_call = graph_request.call_args_list[3]
    assert first_segment_call.kwargs["body"]["startTime"]["dateTime"] == (
        "2026-08-27T09:00:00"
    )
    assert first_segment_call.kwargs["body"]["endTime"]["dateTime"] == (
        "2026-08-27T10:00:00"
    )
    assert second_segment_call.kwargs["body"]["startTime"]["dateTime"] == (
        "2026-08-27T11:00:00"
    )
    assert second_segment_call.kwargs["body"]["endTime"]["dateTime"] == (
        "2026-08-27T12:00:00"
    )


def test_update_event_collects_distinct_conflicts_from_both_delta_segments(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T11:00:00", "timeZone": "UTC"},
            "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
            "isAllDay": False,
            "type": "singleInstance",
        }
    )
    find_conflicts = Mock(
        side_effect=[
            ([], []),
            (
                [
                    {
                        "calendar": "existing@example.com",
                        "summary": "Early conflict",
                        "start": "2026-08-27T09:15:00Z",
                        "end": "2026-08-27T09:30:00Z",
                    }
                ],
                [],
            ),
            (
                [
                    {
                        "calendar": "existing@example.com",
                        "summary": "Late conflict",
                        "start": "2026-08-27T11:15:00Z",
                        "end": "2026-08-27T11:30:00Z",
                    }
                ],
                [],
            ),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", find_conflicts)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T09:00:00",
            end_datetime="2026-08-27T12:00:00",
        )
    )

    assert result["status"] == "conflict"
    assert [conflict["summary"] for conflict in result["conflicts"]] == [
        "Early conflict",
        "Late conflict",
    ]
    assert "+00:00" in result["message"]
    assert find_conflicts.call_count == 3
    assert find_conflicts.call_args_list[1].args[0:3] == (
        "2026-08-27T09:00:00",
        "2026-08-27T10:00:00",
        "UTC",
    )
    assert find_conflicts.call_args_list[2].args[0:3] == (
        "2026-08-27T11:00:00",
        "2026-08-27T12:00:00",
        "UTC",
    )


def test_update_event_disjoint_move_checks_existing_attendees(monkeypatch):
    """A move to a window that is completely disjoint from the event's old
    one is the one case an existing attendee's schedule is safe (and
    necessary) to re-check - the old window's busy block on their
    calendar is no longer where the query is looking."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
                "isAllDay": False,
            },
            {"value": []},  # organizer calendarView
            {
                "value": [
                    {
                        "scheduleId": "existing@example.com",
                        "scheduleItems": [
                            {
                                "status": "busy",
                                "start": {"dateTime": "2026-08-27T14:00:00"},
                                "end": {"dateTime": "2026-08-27T14:30:00"},
                            }
                        ],
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T14:00:00",
            end_datetime="2026-08-27T14:30:00",
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "existing@example.com"


def test_update_event_adding_attendee_queries_the_plain_get_utc_window(
    monkeypatch,
):
    """A timed plain GET already returns the existing absolute window in UTC.

    Historical originalStartTimeZone metadata must not reinterpret it.
    """
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T02:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T02:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
                "originalStartTimeZone": "Asia/Singapore",
            },
            {
                "value": [
                    {
                        "scheduleId": "new@example.com",
                        "availabilityView": "0",
                        "scheduleItems": [],
                    }
                ]
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["new@example.com"],
        )
    )

    assert result["status"] == "success"
    schedule_call = graph_request.call_args_list[1]
    assert schedule_call.args[:2] == ("POST", "/me/calendar/getSchedule")
    assert schedule_call.kwargs["body"]["startTime"] == {
        "dateTime": "2026-08-27T02:00:00",
        "timeZone": "UTC",
    }
    assert schedule_call.kwargs["body"]["endTime"] == {
        "dateTime": "2026-08-27T02:30:00",
        "timeZone": "UTC",
    }


def test_update_event_adding_attendee_rejects_non_utc_plain_get_window(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {
                "dateTime": "2026-08-27T10:00:00",
                "timeZone": "Asia/Singapore",
            },
            "end": {
                "dateTime": "2026-08-27T10:30:00",
                "timeZone": "Asia/Singapore",
            },
            "attendees": [],
            "isAllDay": False,
            "type": "singleInstance",
        }
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["new@example.com"],
        )
    )

    assert result["status"] == "error"
    assert "non-UTC event boundary" in result["message"]
    graph_request.assert_called_once()


def test_update_event_empty_string_attendees_clears_every_attendee(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
                "isAllDay": False,
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees="",
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 2
    patch_call = graph_request.call_args_list[1]
    assert patch_call.args[:2] == ("PATCH", "/me/events/self-1")
    assert patch_call.kwargs["body"]["attendees"] == []
    assert patch_call.kwargs["extra_headers"] == {"If-Match": 'W/"version-1"'}


def test_update_event_missing_schedule_scope_rejects_the_write(monkeypatch):
    """A whole-batch 403 while checking the newly-added attendee is our own
    credential's problem, not a per-attendee visibility gap - with nothing
    else already confirmed, the write must be rejected outright."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {"value": []},
            outlook._GraphRequestError("403 Forbidden", status_code=403),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T14:00:00",
            end_datetime="2026-08-27T14:30:00",
            attendees=["outsider@gmail.com"],
        )
    )

    assert result["status"] == "error"
    assert "reconnect" in result["message"].lower()
    assert result["details"]["unchecked_attendees"] == ["outsider@gmail.com"]


def test_update_event_keeps_first_scope_error_and_all_unchecked_attendees(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
            "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
            "isAllDay": False,
        }
    )
    find_conflicts = Mock(
        side_effect=[
            outlook.InsufficientScopeError(
                "First scope error", [], ["new@example.com"]
            ),
            outlook.InsufficientScopeError(
                "Second scope error", [], ["existing@example.com"]
            ),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", find_conflicts)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T11:00:00",
            end_datetime="2026-08-27T11:30:00",
            attendees=["existing@example.com", "new@example.com"],
        )
    )

    assert result["status"] == "error"
    assert result["message"] == "First scope error"
    assert result["details"]["unchecked_attendees"] == [
        "new@example.com",
        "existing@example.com",
    ]


def test_update_event_missing_schedule_scope_still_reports_an_already_confirmed_conflict(
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
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {"value": [_busy_event(event_id="other-1", subject="Board sync")]},
            outlook._GraphRequestError("403 Forbidden", status_code=403),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T14:00:00",
            end_datetime="2026-08-27T14:30:00",
            attendees=["outsider@gmail.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["summary"] == "Board sync"
    assert result["unchecked_attendees"] == ["outsider@gmail.com"]


def test_update_event_scope_error_does_not_skip_retained_attendee_check(
    monkeypatch,
):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
            "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
            "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
            "isAllDay": False,
        }
    )
    find_conflicts = Mock(
        side_effect=[
            outlook.InsufficientScopeError(
                "Reconnect Outlook to grant schedule access.",
                [],
                ["new@example.com"],
            ),
            (
                [
                    {
                        "calendar": "existing@example.com",
                        "summary": "Busy",
                        "start": "2026-08-27T11:00:00Z",
                        "end": "2026-08-27T11:30:00Z",
                    }
                ],
                [],
            ),
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", find_conflicts)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T11:00:00",
            end_datetime="2026-08-27T11:30:00",
            attendees=["existing@example.com", "new@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["conflicts"][0]["calendar"] == "existing@example.com"
    assert result["unchecked_attendees"] == ["new@example.com"]
    assert find_conflicts.call_count == 2
    assert find_conflicts.call_args_list[1].args[3] == ["existing@example.com"]


def test_update_event_missing_schedule_scope_can_be_bypassed_with_ignore_conflicts(
    monkeypatch,
):
    graph_request = Mock(
        side_effect=[
            {
                "@odata.etag": 'W/"version-1"',
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T14:00:00",
            end_datetime="2026-08-27T14:30:00",
            attendees=["outsider@gmail.com"],
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 2
    assert graph_request.call_args_list[-1].args[:2] == ("PATCH", "/me/events/self-1")


def test_update_event_conflict_response_still_reports_unchecked_attendees(
    monkeypatch,
):
    """A conflict found via one source (the organizer's own calendar) must
    not suppress unchecked_attendees info for a different attendee whose
    schedule couldn't be read (absent from the getSchedule response, a
    per-attendee gap rather than a whole-batch 403) - a caller acting on
    the conflict still needs to know that attendee was never actually
    checked."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T09:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T09:30:00", "timeZone": "UTC"},
                "attendees": [],
                "isAllDay": False,
            },
            {"value": [_busy_event(event_id="other-1", subject="Board sync")]},
            {"value": []},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            start_datetime="2026-08-27T14:00:00",
            end_datetime="2026-08-27T14:30:00",
            attendees=["ghost@example.com"],
        )
    )

    assert result["status"] == "conflict"
    assert result["unchecked_attendees"] == ["ghost@example.com"]


def test_create_event_sends_translated_recurrence(monkeypatch):
    graph_request = Mock(return_value={"id": "created"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", Mock(return_value=([], [])))

    result = json.loads(
        outlook.outlook_create_event(
            subject="Daily Catch up with Bright",
            start_datetime="2026-08-26T07:00:00",
            end_datetime="2026-08-26T07:15:00",
            timezone="Asia/Manila",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    payload = graph_request.call_args.kwargs["body"]
    assert payload["recurrence"]["pattern"]["type"] == "weekly"
    assert payload["recurrence"]["pattern"]["daysOfWeek"] == [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
    ]
    # 2026-09-11T23:59:59Z is 2026-09-12 07:59:59 in Asia/Manila (+08:00) -
    # the correct local endDate is the 12th, not the UTC calendar date.
    assert payload["recurrence"]["range"]["endDate"] == "2026-09-12"


def test_create_recurring_event_with_attendees_skips_availability_and_invites_series(
    monkeypatch,
):
    graph_request = Mock(return_value={"id": "created"})
    find_conflicts = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", find_conflicts)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Weekly planning",
            start_datetime="2026-08-26T09:00:00",
            end_datetime="2026-08-26T09:30:00",
            attendees=["alice@example.com", "bob@example.com"],
            recurrence="FREQ=WEEKLY;BYDAY=WE;COUNT=5",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    find_conflicts.assert_not_called()
    payload = graph_request.call_args.kwargs["body"]
    assert [
        attendee["emailAddress"]["address"] for attendee in payload["attendees"]
    ] == ["alice@example.com", "bob@example.com"]
    assert payload["recurrence"]["range"]["numberOfOccurrences"] == 5


def test_create_event_rejects_invalid_recurrence_without_calling_graph(monkeypatch):
    graph_request = Mock()
    find_conflicts = Mock(return_value=([], []))
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", find_conflicts)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Standup",
            start_datetime="2026-08-26T09:00:00",
            end_datetime="2026-08-26T09:15:00",
            recurrence="FREQ=FORTNIGHTLY",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "FORTNIGHTLY" in result["message"]
    graph_request.assert_not_called()
    find_conflicts.assert_not_called()


def test_create_event_rejects_explicit_empty_recurrence(monkeypatch):
    """Match google_calendar_create_events' explicit-value precedent:
    recurrence="" must not be silently treated the same as omitting it."""
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", Mock(return_value=([], [])))

    result = json.loads(
        outlook.outlook_create_event(
            subject="Standup",
            start_datetime="2026-08-26T09:00:00",
            end_datetime="2026-08-26T09:15:00",
            recurrence="",
        )
    )

    assert result["status"] == "error"
    assert "must not be empty" in result["message"]
    graph_request.assert_not_called()


def test_create_event_without_recurrence_has_no_recurrence_key(monkeypatch):
    graph_request = Mock(return_value={"id": "created"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", Mock(return_value=([], [])))

    outlook.outlook_create_event(
        subject="One-off",
        start_datetime="2026-08-26T09:00:00",
        end_datetime="2026-08-26T09:15:00",
    )

    payload = graph_request.call_args.kwargs["body"]
    assert "recurrence" not in payload


def test_create_all_day_event_sends_date_based_recurrence(monkeypatch):
    graph_request = Mock(return_value={"id": "created"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", Mock(return_value=([], [])))

    result = json.loads(
        outlook.outlook_create_event(
            subject="Daily leave",
            start_datetime="2026-08-26",
            end_datetime="2026-08-27",
            timezone="Pacific Standard Time",
            is_all_day=True,
            recurrence="FREQ=DAILY;UNTIL=20260911",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    payload = graph_request.call_args.kwargs["body"]
    assert payload["recurrence"] == {
        "pattern": {"type": "daily", "interval": 1},
        "range": {
            "type": "endDate",
            "startDate": "2026-08-26",
            "endDate": "2026-09-11",
            "recurrenceTimeZone": "Pacific Standard Time",
        },
    }


def test_create_all_day_event_rejects_datetime_until_before_availability_check(
    monkeypatch,
):
    graph_request = Mock()
    find_conflicts = Mock(return_value=([], []))
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", find_conflicts)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Daily leave",
            start_datetime="2026-08-26",
            end_datetime="2026-08-27",
            is_all_day=True,
            recurrence="FREQ=DAILY;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "error"
    assert "same DATE or DATE-TIME value type" in result["message"]
    graph_request.assert_not_called()
    find_conflicts.assert_not_called()


def test_create_recurring_event_requires_explicit_conflict_override(monkeypatch):
    graph_request = Mock()
    find_conflicts = Mock(return_value=([], []))
    monkeypatch.setattr(outlook, "_graph_request", graph_request)
    monkeypatch.setattr(outlook, "_find_conflicts", find_conflicts)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Weekly planning",
            start_datetime="2026-08-26T09:00:00",
            end_datetime="2026-08-26T09:30:00",
            recurrence="FREQ=WEEKLY;BYDAY=WE;COUNT=5",
        )
    )

    assert result["status"] == "error"
    assert "complete series schedule is safe" in result["message"]
    graph_request.assert_not_called()
    find_conflicts.assert_not_called()
