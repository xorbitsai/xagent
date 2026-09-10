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


def test_create_events_sets_recurrence(monkeypatch):
    service = _fake_service({"id": "created"})
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Daily Catch up with Bright",
            start_time="2026-08-26T07:00:00+08:00",
            end_time="2026-08-26T07:15:00+08:00",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
            timezone="Asia/Shanghai",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["recurrence"] == [
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z"
    ]
    assert kwargs["body"]["start"]["timeZone"] == "Asia/Shanghai"
    assert kwargs["body"]["end"]["timeZone"] == "Asia/Shanghai"


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
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

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
    service.events.return_value.insert.assert_not_called()


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
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.insert.call_args
    assert kwargs["body"]["recurrence"] == ["RRULE:FREQ=DAILY;UNTIL=20260911"]
    assert "timeZone" not in kwargs["body"]["start"]


def test_update_events_adds_recurrence_to_a_previously_single_event(monkeypatch):
    """Reproduces the reported bug: an event created as a single instance
    must be convertible into a true recurring series via update, with the
    RRULE actually reaching Google rather than only landing in
    description text."""
    existing_event = {
        "id": "existing-1",
        "summary": "Daily Catch up with Bright",
        "start": {"dateTime": "2026-08-26T07:00:00+08:00"},
        "end": {"dateTime": "2026-08-26T07:15:00+08:00"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
            timezone="Asia/Shanghai",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["recurrence"] == [
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z"
    ]


def test_update_events_requires_timezone_when_recurrence_is_set_on_a_timed_event(
    monkeypatch,
):
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00+08:00"},
        "end": {"dateTime": "2026-08-26T07:15:00+08:00"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=DAILY",
        )
    )

    assert result["status"] == "error"
    assert "timezone is required" in result["message"]
    service.events.return_value.update.assert_not_called()


def test_update_events_reuses_the_existing_events_own_timezone(monkeypatch):
    """When timezone isn't passed to the update, the event's own existing
    timeZone must be reused rather than requiring the caller to repeat
    it."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "Asia/Manila"},
        "end": {"dateTime": "2026-08-26T07:15:00", "timeZone": "Asia/Manila"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=DAILY;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["start"]["timeZone"] == "Asia/Manila"
    assert kwargs["body"]["end"]["timeZone"] == "Asia/Manila"


def test_update_events_reuses_existing_timezone_even_when_moving_the_event(
    monkeypatch,
):
    """Regression test: passing start_time (to move the event) together
    with recurrence, but no explicit timezone, must still fall back to the
    event's own existing timeZone - not be treated as if the event has
    none just because start_time's reassignment happens to overwrite
    event["start"] before the fallback is read."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "Asia/Manila"},
        "end": {"dateTime": "2026-08-26T07:15:00", "timeZone": "Asia/Manila"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            start_time="2026-08-27T07:00:00",
            end_time="2026-08-27T07:15:00",
            recurrence="FREQ=DAILY;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["start"]["timeZone"] == "Asia/Manila"
    assert kwargs["body"]["end"]["timeZone"] == "Asia/Manila"


def test_update_events_preserves_existing_timezone_on_a_plain_reschedule(monkeypatch):
    """Regression test: events().update() replaces the whole resource, so
    rescheduling (start_time/end_time) without recurrence and without
    re-passing timezone must not silently strip the event's existing
    timeZone from Google's stored event - even though no recurrence logic
    is involved in this call at all."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "Asia/Manila"},
        "end": {"dateTime": "2026-08-26T07:15:00", "timeZone": "Asia/Manila"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            start_time="2026-08-27T07:00:00",
            end_time="2026-08-27T07:15:00",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["start"] == {
        "dateTime": "2026-08-27T07:00:00",
        "timeZone": "Asia/Manila",
    }
    assert kwargs["body"]["end"] == {
        "dateTime": "2026-08-27T07:15:00",
        "timeZone": "Asia/Manila",
    }


def test_update_events_explicit_timezone_overrides_existing_one_on_reschedule(
    monkeypatch,
):
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "Asia/Manila"},
        "end": {"dateTime": "2026-08-26T07:15:00", "timeZone": "Asia/Manila"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            start_time="2026-08-27T07:00:00",
            end_time="2026-08-27T07:15:00",
            timezone="America/New_York",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["start"]["timeZone"] == "America/New_York"
    assert kwargs["body"]["end"]["timeZone"] == "America/New_York"


def test_update_events_preserves_existing_exdate_when_replacing_the_rrule(
    monkeypatch,
):
    """Regression test: Google's `recurrence` field is a flat list of
    RRULE/EXDATE/RDATE/EXRULE lines, not just the RRULE. Overwriting the
    whole list with only the new RRULE would silently resurrect a
    previously-cancelled single occurrence."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00+08:00"},
        "end": {"dateTime": "2026-08-26T07:15:00+08:00"},
        "recurrence": [
            "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            "EXDATE:20260902T070000+08:00",
        ],
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
            timezone="Asia/Shanghai",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["recurrence"] == [
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
        "EXDATE:20260902T070000+08:00",
    ]


def test_update_events_rejects_invalid_recurrence_without_calling_the_api(
    monkeypatch,
):
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00+08:00"},
        "end": {"dateTime": "2026-08-26T07:15:00+08:00"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=FORTNIGHTLY",
        )
    )

    assert result["status"] == "error"
    service.events.return_value.update.assert_not_called()


def test_update_events_rejects_explicit_empty_recurrence(monkeypatch):
    """Regression test: recurrence="" must not be silently treated the
    same as not passing recurrence at all."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00+08:00"},
        "end": {"dateTime": "2026-08-26T07:15:00+08:00"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            timezone="Asia/Shanghai",
            recurrence="",
        )
    )

    assert result["status"] == "error"
    assert "must not be empty" in result["message"]
    service.events.return_value.update.assert_not_called()


def test_update_events_derives_recurrence_start_from_all_day_events_date_field(
    monkeypatch,
):
    """An all-day event has no "dateTime" (only "date") - recurrence must
    still be derivable from that date without requiring the caller to
    redundantly repeat start_time just because the event happens to be
    all-day."""
    existing_event = {
        "id": "existing-1",
        "start": {"date": "2026-08-26"},
        "end": {"date": "2026-08-27"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=DAILY",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["recurrence"] == ["RRULE:FREQ=DAILY"]


def test_update_events_all_day_recurrence_with_utc_until_does_not_error(monkeypatch):
    """Confirmed bug: an all-day event's naive date anchor combined with a
    "Z"-suffixed UNTIL (a very common way to write one) made dateutil
    reject the rule as an aware/naive DTSTART/UNTIL mismatch, even though
    Google itself doesn't need (or accept) a timeZone for a date-only
    event."""
    existing_event = {
        "id": "existing-1",
        "start": {"date": "2026-08-26"},
        "end": {"date": "2026-08-27"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=DAILY;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["recurrence"] == ["RRULE:FREQ=DAILY;UNTIL=20260911T235959Z"]
    assert "timeZone" not in kwargs["body"]["start"]
    assert "timeZone" not in kwargs["body"]["end"]


def test_update_events_all_day_recurrence_with_bare_date_until_does_not_error(
    monkeypatch,
):
    """Confirmed bug: a bare (no-time) UNTIL is the RFC 5545-correct value
    type for a DATE (all-day) DTSTART - RFC 5545 requires UNTIL to match
    DTSTART's own value type. The earlier fix for the "Z"-suffixed-UNTIL
    case unconditionally localized the naive all-day anchor, which broke
    this opposite, equally valid case by creating a NEW aware/naive
    mismatch instead."""
    existing_event = {
        "id": "existing-1",
        "start": {"date": "2026-08-26"},
        "end": {"date": "2026-08-27"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=DAILY;UNTIL=20260911",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["recurrence"] == ["RRULE:FREQ=DAILY;UNTIL=20260911"]


def test_update_events_rejects_bare_date_until_on_a_timed_event(monkeypatch):
    """Confirmed bug: fixing the all-day case above (a bare-date UNTIL
    must be ACCEPTED for a DATE DTSTART) by only conditionally localizing
    the anchor made a TIMED event's naive dateTime string (paired with a
    separate timeZone field, not its own embedded offset) skip
    localization too whenever UNTIL was also naive/bare-date - so the
    mismatch (a DATE-TIME DTSTART needs an aware UNTIL) silently stopped
    being caught for this one dtstart shape, even though it was still
    correctly caught for an offset-carrying dtstart string. A bare-date
    UNTIL must still be rejected for any timed event, regardless of
    whether its dtstart string carries its own offset or relies on a
    separate timeZone field."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "Asia/Manila"},
        "end": {"dateTime": "2026-08-26T07:15:00", "timeZone": "Asia/Manila"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=DAILY;UNTIL=20260911",
        )
    )

    assert result["status"] == "error"
    service.events.return_value.update.assert_not_called()


def test_update_events_explicit_timezone_on_all_day_event_does_not_send_malformed_payload(
    monkeypatch,
):
    """Confirmed bug: passing timezone while updating an all-day event
    (with or without touching recurrence) used to unconditionally stamp
    event["start"]/["end"] with timeZone before the all-day check, mixing
    "date" and "timeZone" in the same object - a malformed EventDateTime.
    timezone must still be usable to localize the UNTIL comparison for an
    all-day event's recurrence, just never written into the payload."""
    existing_event = {
        "id": "existing-1",
        "start": {"date": "2026-08-26"},
        "end": {"date": "2026-08-27"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            timezone="Asia/Shanghai",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["start"] == {"date": "2026-08-26"}
    assert kwargs["body"]["end"] == {"date": "2026-08-27"}


def test_update_events_explicit_timezone_localizes_all_day_recurrence_without_writing_it(
    monkeypatch,
):
    """Same as above, but with recurrence also set: the explicit timezone
    must still be used to correctly localize the UNTIL comparison (instead
    of falling back to a blind UTC default), while still never being
    written into the all-day event's date-only start/end."""
    existing_event = {
        "id": "existing-1",
        "start": {"date": "2026-08-26"},
        "end": {"date": "2026-08-27"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            timezone="Asia/Shanghai",
            recurrence="FREQ=DAILY;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["recurrence"] == ["RRULE:FREQ=DAILY;UNTIL=20260911T235959Z"]
    assert "timeZone" not in kwargs["body"]["start"]
    assert "timeZone" not in kwargs["body"]["end"]


def test_update_events_all_day_recurrence_actually_uses_the_given_timezone_not_utc(
    monkeypatch,
):
    """Regression test distinguishing "localizes with the given timezone"
    from "always falls back to UTC regardless of what's passed": the
    all-day anchor is 2026-09-12 and UNTIL is 2026-09-11T20:00:00Z. Read
    as Asia/Shanghai (+8), the anchor is 2026-09-11 16:00 UTC, which is
    BEFORE the 20:00 UTC UNTIL - not-before-start, so this succeeds. If
    the code silently used UTC instead of the given timezone to localize
    the anchor, it would read as 2026-09-12 00:00 UTC - AFTER UNTIL - and
    wrongly reject this as "UNTIL before the start time"."""
    existing_event = {
        "id": "existing-1",
        "start": {"date": "2026-09-12"},
        "end": {"date": "2026-09-13"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            timezone="Asia/Shanghai",
            recurrence="FREQ=DAILY;UNTIL=20260911T200000Z",
        )
    )

    assert result["status"] == "success"


def test_update_events_rejects_end_time_only_on_an_all_day_event(monkeypatch):
    """Confirmed bug: passing only end_time (a timed RFC3339 value) on an
    all-day event left start as a bare "date" while end became a
    "dateTime" - Google requires start and end to be the same kind, so
    this mismatched payload would be rejected by the real API. Caught
    here instead, with a clear error telling the caller to pass both."""
    existing_event = {
        "id": "existing-1",
        "start": {"date": "2026-08-26"},
        "end": {"date": "2026-08-27"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            end_time="2026-08-28T10:00:00Z",
        )
    )

    assert result["status"] == "error"
    assert "must both be provided together" in result["message"]
    service.events.return_value.update.assert_not_called()


def test_update_events_rejects_start_time_only_on_an_all_day_event(monkeypatch):
    existing_event = {
        "id": "existing-1",
        "start": {"date": "2026-08-26"},
        "end": {"date": "2026-08-27"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            start_time="2026-08-26T10:00:00Z",
        )
    )

    assert result["status"] == "error"
    assert "must both be provided together" in result["message"]
    service.events.return_value.update.assert_not_called()


def test_update_events_rejects_bare_date_start_time_on_a_timed_event(monkeypatch):
    """Confirmed bug: a bare date (no "T") passed as start_time on an
    already-timed event flips is_all_day true for start alone, while end
    (untouched) stays a "dateTime" - the same shape-mismatch class of bug
    as the all-day case above, just approached from the opposite
    direction and via undocumented misuse of start_time's RFC3339
    contract."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T09:00:00+08:00", "timeZone": "Asia/Shanghai"},
        "end": {"dateTime": "2026-08-26T09:15:00+08:00", "timeZone": "Asia/Shanghai"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            start_time="2026-08-27",
            timezone="Asia/Shanghai",
        )
    )

    assert result["status"] == "error"
    assert "must both be provided together" in result["message"]
    service.events.return_value.update.assert_not_called()


def test_update_events_converts_all_day_event_to_timed_when_both_are_provided(
    monkeypatch,
):
    """The one legitimate way to convert an all-day event to a timed one:
    both start_time and end_time given together, so start/end stay the
    same kind throughout."""
    existing_event = {
        "id": "existing-1",
        "start": {"date": "2026-08-26"},
        "end": {"date": "2026-08-27"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            start_time="2026-08-26T09:00:00",
            end_time="2026-08-26T10:00:00",
            timezone="Asia/Shanghai",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["start"] == {
        "dateTime": "2026-08-26T09:00:00",
        "timeZone": "Asia/Shanghai",
    }
    assert kwargs["body"]["end"] == {
        "dateTime": "2026-08-26T10:00:00",
        "timeZone": "Asia/Shanghai",
    }


def test_update_events_converts_a_timed_event_to_all_day_when_both_are_provided(
    monkeypatch,
):
    """Confirmed bug: the mirror direction of the test above. A bare-date
    start_time/end_time (converting an existing TIMED event to all-day)
    used to still be written under the "dateTime" key - producing
    {"dateTime": "2026-08-27", "timeZone": "Asia/Shanghai"}, a value
    that's neither valid RFC3339 nor a legitimate all-day EventDateTime
    (all-day events never carry timeZone). Must be written under "date",
    with no timeZone."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "Asia/Manila"},
        "end": {"dateTime": "2026-08-26T08:00:00", "timeZone": "Asia/Manila"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            start_time="2026-08-27",
            end_time="2026-08-28",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["start"] == {"date": "2026-08-27"}
    assert kwargs["body"]["end"] == {"date": "2026-08-28"}


def test_update_events_does_not_reuse_existing_timezone_when_new_value_has_own_offset(
    monkeypatch,
):
    """Confirmed bug: a new start_time/end_time that already carries its
    own UTC offset is fully self-describing - stamping a possibly-
    different reused timeZone from the existing event on top of it
    produces an internally-contradictory EventDateTime (e.g. "-07:00" in
    dateTime alongside timeZone: "Asia/Manila", UTC+8)."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "Asia/Manila"},
        "end": {"dateTime": "2026-08-26T08:00:00", "timeZone": "Asia/Manila"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            start_time="2026-08-27T09:00:00-07:00",
            end_time="2026-08-27T09:15:00-07:00",
        )
    )

    assert result["status"] == "success"
    _, kwargs = service.events.return_value.update.call_args
    assert kwargs["body"]["start"] == {"dateTime": "2026-08-27T09:00:00-07:00"}
    assert kwargs["body"]["end"] == {"dateTime": "2026-08-27T09:15:00-07:00"}


def test_update_events_recurrence_survives_a_missing_end_key(monkeypatch):
    """Confirmed bug: a malformed fetched event missing the "end" key
    entirely (Google always returns both in practice, but this is
    defensive) used to raise a raw, opaque KeyError('end') instead of
    either a clean error or a graceful degradation."""
    existing_event = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "Asia/Manila"},
    }
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=DAILY;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "success"


def test_update_events_rejects_recurrence_when_no_start_information_exists(
    monkeypatch,
):
    """Neither "dateTime" nor "date" present on the existing event (a
    malformed/unexpected shape) must still get a clear error instead of an
    opaque TypeError from parsing None as a datetime."""
    existing_event = {"id": "existing-1"}
    service = _fake_service({"id": "existing-1"}, existing_event=existing_event)
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=DAILY",
        )
    )

    assert result["status"] == "error"
    assert "start time" in result["message"]
    service.events.return_value.update.assert_not_called()


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
