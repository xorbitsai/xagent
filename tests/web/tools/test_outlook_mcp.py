import json
from unittest.mock import Mock

import pytest

from xagent.web.tools.mcp import outlook


def test_build_graph_recurrence_daily_with_until():
    recurrence = outlook._build_graph_recurrence(
        "FREQ=DAILY;INTERVAL=2;UNTIL=20260911T235959Z",
        "2026-08-26T07:00:00+08:00",
    )

    assert recurrence == {
        "pattern": {"type": "daily", "interval": 2},
        "range": {
            "type": "endDate",
            "startDate": "2026-08-26",
            "endDate": "2026-09-11",
            "recurrenceTimeZone": "UTC",
        },
    }


def test_build_graph_recurrence_stamps_the_given_timezone():
    """Graph otherwise defaults recurrenceTimeZone to the event's own
    already-configured start time zone - which this function has no way
    to confirm when start_datetime came from an update's fallback GET
    rather than the caller - so it must be stamped explicitly to stay
    consistent with the dates this function actually derived."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=DAILY", "2026-08-26T07:00:00", "Asia/Manila"
    )

    assert recurrence["range"]["recurrenceTimeZone"] == "Asia/Manila"


def test_build_graph_recurrence_weekly_with_explicit_byday():
    recurrence = outlook._build_graph_recurrence(
        "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
        "2026-08-26T07:00:00+08:00",
    )

    assert recurrence["pattern"] == {
        "type": "weekly",
        "interval": 1,
        "daysOfWeek": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    }
    assert recurrence["range"]["endDate"] == "2026-09-11"


def test_build_graph_recurrence_weekly_without_byday_derives_from_start_date():
    """2026-08-26 is a Wednesday; with no BYDAY given, the weekly pattern
    must repeat on the start date's own weekday rather than defaulting to
    something arbitrary (e.g. Monday)."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=WEEKLY;COUNT=5",
        "2026-08-26T07:00:00+08:00",
    )

    assert recurrence["pattern"]["daysOfWeek"] == ["wednesday"]
    assert recurrence["range"] == {
        "type": "numbered",
        "startDate": "2026-08-26",
        "numberOfOccurrences": 5,
        "recurrenceTimeZone": "UTC",
    }


def test_build_graph_recurrence_absolute_monthly():
    recurrence = outlook._build_graph_recurrence(
        "FREQ=MONTHLY;BYMONTHDAY=15",
        "2026-08-15T07:00:00+08:00",
    )

    assert recurrence["pattern"] == {
        "type": "absoluteMonthly",
        "interval": 1,
        "dayOfMonth": 15,
    }
    assert recurrence["range"] == {
        "type": "noEnd",
        "startDate": "2026-08-15",
        "recurrenceTimeZone": "UTC",
    }


def test_build_graph_recurrence_absolute_yearly():
    recurrence = outlook._build_graph_recurrence(
        "FREQ=YEARLY;BYMONTH=8;BYMONTHDAY=15",
        "2026-08-15T07:00:00+08:00",
    )

    assert recurrence["pattern"] == {
        "type": "absoluteYearly",
        "interval": 1,
        "dayOfMonth": 15,
        "month": 8,
    }


def test_build_graph_recurrence_rejects_unsupported_pattern():
    """A relative pattern ("second Tuesday of the month") needs an index
    Graph requires but plain RRULE components here don't carry - this must
    be rejected rather than silently produced with the wrong days."""
    with pytest.raises(ValueError, match="unsupported recurrence pattern"):
        outlook._build_graph_recurrence(
            "FREQ=MONTHLY;BYDAY=2TU", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_invalid_rrule():
    with pytest.raises(ValueError, match="invalid recurrence rule"):
        outlook._build_graph_recurrence("FREQ=FORTNIGHTLY", "2026-08-11T07:00:00+08:00")


def test_create_event_sends_translated_recurrence(monkeypatch):
    graph_request = Mock(return_value={"id": "created"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Daily Catch up with Bright",
            start_datetime="2026-08-26T07:00:00",
            end_datetime="2026-08-26T07:15:00",
            timezone="Asia/Manila",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
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
    assert payload["recurrence"]["range"]["endDate"] == "2026-09-11"


def test_create_event_rejects_invalid_recurrence_without_calling_graph(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_create_event(
            subject="Standup",
            start_datetime="2026-08-26T09:00:00",
            end_datetime="2026-08-26T09:15:00",
            recurrence="FREQ=FORTNIGHTLY",
        )
    )

    assert result["status"] == "error"
    graph_request.assert_not_called()


def test_create_event_without_recurrence_has_no_recurrence_key(monkeypatch):
    graph_request = Mock(return_value={"id": "created"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    outlook.outlook_create_event(
        subject="One-off",
        start_datetime="2026-08-26T09:00:00",
        end_datetime="2026-08-26T09:15:00",
    )

    payload = graph_request.call_args.kwargs["body"]
    assert "recurrence" not in payload


def test_update_event_adds_recurrence_using_the_passed_start_datetime(monkeypatch):
    """Reproduces the reported bug: an event created as a single instance
    must be convertible into a true recurring series via update, with a
    structured Graph recurrence actually reaching the PATCH body - not
    just landing in description text."""
    graph_request = Mock(return_value={"id": "updated"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="existing-1",
            start_datetime="2026-08-26T07:00:00",
            end_datetime="2026-08-26T07:15:00",
            timezone="Asia/Manila",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "success"
    graph_request.assert_called_once()
    payload = graph_request.call_args.kwargs["body"]
    assert payload["recurrence"]["pattern"]["type"] == "weekly"


def test_update_event_fetches_existing_start_when_not_provided(monkeypatch):
    """Setting recurrence without also moving the event must validate and
    translate the rule against the event's own existing start time rather
    than requiring the caller to repeat it."""
    graph_request = Mock(
        side_effect=[
            {"start": {"dateTime": "2026-08-26T07:00:00"}},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="existing-1",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "success"
    assert graph_request.call_count == 2
    get_call = graph_request.call_args_list[0]
    assert get_call.args[:2] == ("GET", "/me/events/existing-1")
    patch_call = graph_request.call_args_list[1]
    assert patch_call.kwargs["body"]["recurrence"]["range"]["startDate"] == (
        "2026-08-26"
    )
    # The fallback GET sent no Prefer header, so Graph returned this value
    # in UTC regardless of the caller's own default `timezone="UTC"` -
    # asserting it explicitly here would pass by coincidence; the real
    # requirement is that it's *not* silently left as an unrelated caller
    # timezone if this function's default ever changes.
    assert patch_call.kwargs["body"]["recurrence"]["range"]["recurrenceTimeZone"] == (
        "UTC"
    )


def test_update_event_rejects_invalid_recurrence_without_calling_graph(monkeypatch):
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="existing-1",
            start_datetime="2026-08-26T07:00:00",
            end_datetime="2026-08-26T07:15:00",
            recurrence="FREQ=FORTNIGHTLY",
        )
    )

    assert result["status"] == "error"
    graph_request.assert_not_called()
