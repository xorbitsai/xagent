import copy
import json
import re
from typing import Any
from unittest.mock import Mock

import pytest

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
    request.headers = {}
    request.execute.return_value = execute_result
    events.insert = Mock(return_value=request)
    events.update = Mock(return_value=request)
    # These Meet-focused tests do not exercise scheduling-conflict discovery.
    # Return empty availability data so they continue to isolate event writes.
    events.list = Mock(return_value=Mock(execute=Mock(return_value={"items": []})))
    fetched_event = {
        "id": "evt1",
        "start": {"dateTime": "2026-09-07T15:00:00+08:00"},
        "end": {"dateTime": "2026-09-07T16:00:00+08:00"},
    }
    if existing_event:
        fetched_event.update(copy.deepcopy(existing_event))
    events.get = Mock(return_value=Mock(execute=Mock(return_value=fetched_event)))
    service = Mock()
    service.events.return_value = events

    def freebusy_query(**kwargs: Any) -> Mock:
        calendars = {
            item["id"]: {"busy": []} for item in kwargs.get("body", {}).get("items", [])
        }
        return Mock(execute=Mock(return_value={"calendars": calendars}))

    service.freebusy.return_value = Mock(query=Mock(side_effect=freebusy_query))
    service.calendars.return_value = Mock(
        get=Mock(
            return_value=Mock(
                execute=Mock(
                    return_value={"id": "organizer@example.com", "timeZone": "UTC"}
                )
            )
        )
    )
    return service


def test_has_own_utc_offset_true_for_a_z_suffix():
    assert calendar._has_own_utc_offset("2026-08-26T07:00:00Z")


def test_has_own_utc_offset_true_for_an_explicit_offset():
    assert calendar._has_own_utc_offset("2026-08-26T07:00:00+08:00")


def test_has_own_utc_offset_false_for_a_naive_datetime():
    assert not calendar._has_own_utc_offset("2026-08-26T07:00:00")


def test_has_own_utc_offset_false_for_unparseable_input():
    assert not calendar._has_own_utc_offset("not-a-date")


def test_normalize_rrule_returns_a_prefixed_uppercased_string():
    assert (
        calendar._normalize_rrule("freq=daily;count=3", "2026-08-26T07:00:00")
        == "RRULE:FREQ=DAILY;COUNT=3"
    )


def test_create_events_sets_recurrence(monkeypatch):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Daily Catch up with Bright",
            start_time="2026-08-26T07:00:00",
            end_time="2026-08-26T07:15:00",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
            timezone="Asia/Shanghai",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["recurrence"] == [
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z"
    ]
    assert kwargs["body"]["start"]["timeZone"] == "Asia/Shanghai"
    assert kwargs["body"]["end"]["timeZone"] == "Asia/Shanghai"


def test_create_events_stamps_timezone_even_when_recurrence_has_its_own_offset(
    monkeypatch,
):
    """Google's EventDateTime reference requires timeZone unconditionally
    for a recurring event, regardless of whether start_time/end_time
    already carry their own UTC offset - the offset only fixes this one
    occurrence's instant, while timeZone governs how the recurrence
    itself expands (e.g. across DST transitions). Must NOT be skipped
    here just because the value is already self-describing."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Weekly sync",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T10:00:00+08:00",
            timezone="America/Los_Angeles",
            recurrence="FREQ=WEEKLY;COUNT=5",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["start"]["timeZone"] == "America/Los_Angeles"
    assert kwargs["body"]["end"]["timeZone"] == "America/Los_Angeles"


def test_create_events_skips_timezone_on_its_own_offset_without_recurrence(
    monkeypatch,
):
    """Without recurrence, timeZone is merely optional (per Google's own
    docs), so a side that already carries its own UTC offset is skipped
    instead of risking a disagreeing zone on an optional field."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="One-off",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T10:00:00+08:00",
            timezone="America/Los_Angeles",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert "timeZone" not in kwargs["body"]["start"]
    assert "timeZone" not in kwargs["body"]["end"]


def test_create_events_skips_timezone_independently_per_side_on_asymmetric_offsets(
    monkeypatch,
):
    """start/end are checked independently, not just with a shared
    offset-or-not assumption - a mock that always gives both sides the
    same offset shape couldn't catch _stamp_timezone's start/end
    arguments being accidentally swapped."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="One-off",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T10:00:00",
            timezone="America/Los_Angeles",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert "timeZone" not in kwargs["body"]["start"]
    assert kwargs["body"]["end"]["timeZone"] == "America/Los_Angeles"


def test_create_events_requires_timezone_when_recurrence_is_set(monkeypatch):
    """Google documents timeZone as required specifically for recurring
    events (it's what the recurrence is expanded in) - this must be
    rejected here rather than silently sent to Google without one."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Standup",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T09:15:00+08:00",
            recurrence="FREQ=DAILY",
        )
    )

    assert result["status"] == "error"
    assert "timezone is required" in result["message"]
    service.events.return_value.insert.assert_not_called()


def test_create_events_treats_empty_timezone_as_not_provided(monkeypatch):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="All day recurring",
            start_time="2026-09-01",
            end_time="2026-09-02",
            recurrence="FREQ=DAILY;COUNT=3",
            timezone="  ",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert "timeZone" not in kwargs["body"]["start"]
    assert "timeZone" not in kwargs["body"]["end"]


def test_create_events_rejects_an_invalid_timezone_even_without_recurrence(
    monkeypatch,
):
    """Confirmed bug: an invalid IANA timezone name was only ever caught
    by resolve_zoneinfo when recurrence was also set (parse_rrule is its
    only caller) - without recurrence, a bad timezone silently reached
    Google's API and surfaced as an opaque server-side error instead of
    this connector's own clean message."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Standup",
            start_time="2026-08-26T09:00:00",
            end_time="2026-08-26T09:15:00",
            timezone="Not/AZone",
        )
    )

    assert result["status"] == "error"
    assert "recognized IANA zone name" in result["message"]
    service.events.return_value.insert.assert_not_called()


def test_create_events_accepts_recurrence_with_explicit_prefix(monkeypatch):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Standup",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T09:15:00+08:00",
            recurrence="RRULE:FREQ=DAILY;COUNT=10",
            timezone="Asia/Shanghai",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["recurrence"] == ["RRULE:FREQ=DAILY;COUNT=10"]


def test_create_events_normalizes_lowercase_recurrence(monkeypatch):
    """Regression test: dateutil's validation is lenient about case, so a
    lowercase RRULE must still be canonicalized to uppercase before
    reaching Google's API - not sent verbatim in whatever case an LLM
    happened to produce."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_create_events(
        summary="Standup",
        start_time="2026-08-26T09:00:00+08:00",
        end_time="2026-08-26T09:15:00+08:00",
        recurrence="freq=daily;count=10",
        timezone="Asia/Shanghai",
        ignore_conflicts=True,
    )

    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["recurrence"] == ["RRULE:FREQ=DAILY;COUNT=10"]


def test_create_events_rejects_invalid_recurrence_without_calling_the_api(
    monkeypatch,
):
    """A rule that can't be parsed must be reported as an error, not sent
    to Google where it would either be rejected opaquely or - the
    originally reported failure mode - silently accepted as inert text
    with no actual recurrence."""
    service = _fake_service({"id": "created"})
    get_calendar_service = Mock(return_value=service)
    monkeypatch.setattr(calendar, "get_calendar_service", get_calendar_service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Standup",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T09:15:00+08:00",
            recurrence="FREQ=FORTNIGHTLY",
            timezone="Asia/Shanghai",
        )
    )

    assert result["status"] == "error"
    assert "invalid recurrence rule" in result["message"]
    get_calendar_service.assert_not_called()
    service.events.return_value.insert.assert_not_called()


@pytest.mark.parametrize("frequency", ["SECONDLY", "MINUTELY", "HOURLY"])
def test_create_events_rejects_frequencies_google_does_not_support(
    monkeypatch, frequency
):
    get_calendar_service = Mock()
    monkeypatch.setattr(calendar, "get_calendar_service", get_calendar_service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Too frequent",
            start_time="2026-09-01T09:00:00+08:00",
            end_time="2026-09-01T09:15:00+08:00",
            recurrence=f"FREQ={frequency};COUNT=3",
            timezone="Asia/Shanghai",
        )
    )

    assert result["status"] == "error"
    assert f"does not support FREQ={frequency}" in result["message"]
    get_calendar_service.assert_not_called()


def test_create_events_rejects_explicit_empty_recurrence(monkeypatch):
    """Regression test: every other optional field in this function treats
    an explicitly-passed value as meaningful; recurrence="" must not be
    silently treated the same as not mentioning recurrence at all."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Standup",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T09:15:00+08:00",
            timezone="Asia/Shanghai",
            recurrence="",
        )
    )

    assert result["status"] == "error"
    assert "must not be empty" in result["message"]
    service.events.return_value.insert.assert_not_called()


def test_create_events_without_recurrence_has_no_recurrence_key(monkeypatch):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    calendar.google_calendar_create_events(
        summary="One-off",
        start_time="2026-08-26T09:00:00+08:00",
        end_time="2026-08-26T09:15:00+08:00",
    )

    _, kwargs = service.events.return_value.insert.call_args
    assert "recurrence" not in kwargs["body"]


def test_create_events_creates_an_all_day_event_from_bare_dates(monkeypatch):
    """Confirmed bug: a bare-date start_time/end_time (e.g. "2026-09-01",
    no time component) used to be silently written under the "dateTime"
    key regardless - {"dateTime": "2026-09-01"} is not valid RFC3339 and
    Google's real API would reject it. Must be written under "date"
    instead, with no timeZone stamped onto it."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="All day thing",
            start_time="2026-09-01",
            end_time="2026-09-02",
            timezone="Asia/Shanghai",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["start"] == {"date": "2026-09-01"}
    assert kwargs["body"]["end"] == {"date": "2026-09-02"}


@pytest.mark.parametrize(
    ("start_time", "end_time"),
    [
        ("2026-02-30", "2026-03-01"),
        ("2026-2-3", "2026-02-04"),
        ("2026-13-01", "2026-13-02"),
    ],
)
def test_create_events_rejects_invalid_all_day_dates(monkeypatch, start_time, end_time):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Invalid all-day event",
            start_time=start_time,
            end_time=end_time,
        )
    )

    assert result["status"] == "error"
    assert "valid YYYY-MM-DD date or RFC3339 dateTime" in result["message"]
    service.events.return_value.insert.assert_not_called()


@pytest.mark.parametrize(
    ("start_time", "end_time"),
    [("2026-09-01", "2026-09-01"), ("2026-09-02", "2026-09-01")],
)
def test_create_events_rejects_nonpositive_all_day_ranges(
    monkeypatch, start_time, end_time
):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Invalid all-day range",
            start_time=start_time,
            end_time=end_time,
        )
    )

    assert result["status"] == "error"
    assert "end_time is exclusive" in result["message"]
    service.events.return_value.insert.assert_not_called()


def test_create_events_strips_whitespace_from_a_bare_date_before_sending_it(
    monkeypatch,
):
    """is_bare_date strips surrounding whitespace to decide the shape, but
    the raw start_time/end_time used to be written into the payload as-is
    - a bare date with incidental whitespace would reach Google as
    something other than the exact "YYYY-MM-DD" it requires."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="All day thing",
            start_time="  2026-09-01  ",
            end_time="  2026-09-02  ",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["start"] == {"date": "2026-09-01"}
    assert kwargs["body"]["end"] == {"date": "2026-09-02"}


def test_create_events_strips_whitespace_from_timed_values_before_validation_and_send(
    monkeypatch,
):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Standup",
            start_time="  2026-09-01T09:00:00+08:00  ",
            end_time="  2026-09-01T09:15:00+08:00  ",
            recurrence="FREQ=DAILY;COUNT=3",
            timezone="Asia/Shanghai",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["start"]["dateTime"] == "2026-09-01T09:00:00+08:00"
    assert kwargs["body"]["end"]["dateTime"] == "2026-09-01T09:15:00+08:00"
    assert kwargs["body"]["recurrence"] == ["RRULE:FREQ=DAILY;COUNT=3"]


def test_create_events_accepts_a_whitespace_padded_all_day_recurrence(monkeypatch):
    """Confirmed bug: the payload used the stripped bare-date value, but
    recurrence validation was passed the raw, unstripped start_time - so
    the exact same all-day start_time that succeeds without recurrence
    was rejected as an "invalid start time" the moment recurrence was
    also set."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="All day thing",
            start_time="  2026-09-01  ",
            end_time="  2026-09-02  ",
            recurrence="FREQ=DAILY;COUNT=3",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"


def test_create_events_rejects_a_space_separated_datetime(
    monkeypatch,
):
    """ISO 8601 permits a space separator, but Google's EventDateTime
    contract requires RFC3339. Reject it locally instead of forwarding a
    value the API may reject."""
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Timed with a space separator",
            start_time="2026-09-01 10:00:00",
            end_time="2026-09-01 11:00:00",
        )
    )

    assert result["status"] == "error"
    assert "RFC3339 dateTime" in result["message"]
    service.events.return_value.insert.assert_not_called()


@pytest.mark.parametrize("value", ["20260826T070000", "2026W011T070000"])
def test_create_events_rejects_non_rfc3339_datetime_shapes(monkeypatch, value):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Invalid datetime shape",
            start_time=value,
            end_time="2026-09-01T11:00:00",
        )
    )

    assert result["status"] == "error"
    assert "RFC3339 dateTime" in result["message"]
    service.events.return_value.insert.assert_not_called()


@pytest.mark.parametrize(
    ("start_time", "end_time"),
    [
        ("2026-09-01T10:00:00+08:00", "2026-09-01T10:00:00+08:00"),
        ("2026-09-01T11:00:00+08:00", "2026-09-01T10:00:00+08:00"),
    ],
)
def test_create_events_rejects_nonpositive_timed_ranges(
    monkeypatch, start_time, end_time
):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Invalid timed range",
            start_time=start_time,
            end_time=end_time,
        )
    )

    assert result["status"] == "error"
    assert "must be after start" in result["message"]
    service.events.return_value.insert.assert_not_called()


def test_create_events_rejects_mixed_bare_date_and_datetime(monkeypatch):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Mixed",
            start_time="2026-09-01",
            end_time="2026-09-02T10:00:00",
        )
    )

    assert result["status"] == "error"
    assert "must both be a bare date or both a dateTime" in result["message"]
    service.events.return_value.insert.assert_not_called()


def test_create_events_all_day_recurrence_does_not_require_timezone(monkeypatch):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="All day recurring",
            start_time="2026-09-01",
            end_time="2026-09-02",
            recurrence="FREQ=DAILY;UNTIL=20260911",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["recurrence"] == ["RRULE:FREQ=DAILY;UNTIL=20260911"]
    assert "timeZone" not in kwargs["body"]["start"]


def test_create_events_recurring_series_requires_explicit_conflict_bypass(monkeypatch):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Weekly sync",
            start_time="2026-09-01T09:00:00+08:00",
            end_time="2026-09-01T09:30:00+08:00",
            timezone="Asia/Shanghai",
            recurrence="FREQ=WEEKLY;COUNT=5",
        )
    )

    assert result["status"] == "error"
    assert "every occurrence" in result["message"]
    service.events.return_value.insert.assert_not_called()


def test_create_events_checks_all_day_window_in_primary_calendar_timezone(monkeypatch):
    service = _fake_service({"id": "created"})
    service.calendars.return_value.get.return_value.execute.return_value = {
        "id": "organizer@example.com",
        "timeZone": "Asia/Shanghai",
    }
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Conference",
            start_time="2026-09-01",
            end_time="2026-09-03",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.list.call_args
    assert kwargs["timeMin"] == "2026-09-01T00:00:00+08:00"
    assert kwargs["timeMax"] == "2026-09-03T00:00:00+08:00"


def test_create_events_normalizes_each_naive_conflict_boundary_independently(
    monkeypatch,
):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Mixed boundary representation",
            start_time="2026-09-01T09:00:00+08:00",
            end_time="2026-09-01T02:00:00",
            timezone="UTC",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.list.call_args
    assert kwargs["timeMin"] == "2026-09-01T09:00:00+08:00"
    assert kwargs["timeMax"] == "2026-09-01T02:00:00+00:00"


def test_create_events_rejects_a_reversed_mixed_offset_window_after_normalizing(
    monkeypatch,
):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Reversed",
            start_time="2026-09-01T09:00:00+08:00",
            end_time="2026-09-01T00:30:00",
            timezone="UTC",
            ignore_conflicts=True,
        )
    )

    assert result["status"] == "error"
    assert "after start" in result["message"]
    service.events.return_value.insert.assert_not_called()


def test_event_boundary_returns_none_for_a_malformed_non_dict_boundary():
    assert calendar._event_boundary(["not", "an", "object"], "UTC") is None


def test_update_with_missing_boundary_fails_closed_before_write(monkeypatch):
    service = _fake_service(
        {"id": "evt1"},
        existing_event={
            "start": ["not", "an", "object"],
            "end": {"dateTime": "2026-09-07T16:00:00+08:00"},
        },
    )
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="evt1", attendees=["new@example.com"]
        )
    )

    assert result["status"] == "conflict_check_incomplete"
    assert "complete start/end window" in result["message"]
    service.events.return_value.update.assert_not_called()


def test_update_with_real_window_checks_a_new_attendee_before_write(monkeypatch):
    service = _fake_service(
        {"id": "evt1"},
        existing_event={
            "start": {"dateTime": "2026-09-07T15:00:00+08:00"},
            "end": {"dateTime": "2026-09-07T16:00:00+08:00"},
        },
    )
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="evt1", attendees=["new@example.com"]
        )
    )

    assert result["status"] == "success"
    query_kwargs = service.freebusy.return_value.query.call_args.kwargs
    assert query_kwargs["body"]["items"] == [{"id": "new@example.com"}]
    service.events.return_value.update.assert_called_once()


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
        end_time="2026-09-08T16:00:00+08:00",
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
