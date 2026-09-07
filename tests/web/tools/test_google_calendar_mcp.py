import json
from typing import Any

import pytest

from xagent.web.tools.mcp import calendar


class _Exec:
    def __init__(self, result: dict[str, Any]):
        self._result = result

    def execute(self) -> dict[str, Any]:
        return self._result


class FakeEvents:
    def __init__(
        self,
        *,
        get_result: dict[str, Any] | None = None,
        insert_result: dict[str, Any] | None = None,
        update_result: dict[str, Any] | None = None,
    ):
        self._get_result = get_result or {}
        self._insert_result = insert_result or {"id": "created"}
        self._update_result = update_result or {"id": "updated"}
        self.insert_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []

    def get(self, **kwargs: Any) -> _Exec:
        return _Exec(self._get_result)

    def insert(self, **kwargs: Any) -> _Exec:
        self.insert_calls.append(kwargs)
        return _Exec(self._insert_result)

    def update(self, **kwargs: Any) -> _Exec:
        self.update_calls.append(kwargs)
        return _Exec(self._update_result)


class FakeService:
    def __init__(self, events: FakeEvents | None = None):
        self._events = events or FakeEvents()

    def events(self) -> FakeEvents:
        return self._events


@pytest.fixture
def fake_service(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(calendar, "get_calendar_service", lambda: service)
    return service


def test_create_events_sets_recurrence(fake_service):
    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Daily Catch up with Bright",
            start_time="2026-08-26T07:00:00+08:00",
            end_time="2026-08-26T07:15:00+08:00",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "success"
    insert_body = fake_service._events.insert_calls[0]["body"]
    assert insert_body["recurrence"] == [
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z"
    ]


def test_create_events_accepts_recurrence_with_explicit_prefix(fake_service):
    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Standup",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T09:15:00+08:00",
            recurrence="RRULE:FREQ=DAILY;COUNT=10",
        )
    )

    assert result["status"] == "success"
    insert_body = fake_service._events.insert_calls[0]["body"]
    assert insert_body["recurrence"] == ["RRULE:FREQ=DAILY;COUNT=10"]


def test_create_events_rejects_invalid_recurrence_without_calling_the_api(
    fake_service,
):
    """A rule that can't be parsed must be reported as an error, not sent
    to Google where it would either be rejected opaquely or - the
    originally reported failure mode - silently accepted as inert text
    with no actual recurrence."""
    result = json.loads(
        calendar.google_calendar_create_events(
            summary="Standup",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T09:15:00+08:00",
            recurrence="FREQ=FORTNIGHTLY",
        )
    )

    assert result["status"] == "error"
    assert "invalid recurrence rule" in result["message"]
    assert fake_service._events.insert_calls == []


def test_create_events_without_recurrence_has_no_recurrence_key(fake_service):
    json.loads(
        calendar.google_calendar_create_events(
            summary="One-off",
            start_time="2026-08-26T09:00:00+08:00",
            end_time="2026-08-26T09:15:00+08:00",
        )
    )

    insert_body = fake_service._events.insert_calls[0]["body"]
    assert "recurrence" not in insert_body


def test_update_events_adds_recurrence_to_a_previously_single_event(fake_service):
    """Reproduces the reported bug: an event created as a single instance
    must be convertible into a true recurring series via update, with the
    RRULE actually reaching Google rather than only landing in
    description text."""
    fake_service._events._get_result = {
        "id": "existing-1",
        "summary": "Daily Catch up with Bright",
        "start": {"dateTime": "2026-08-26T07:00:00+08:00"},
        "end": {"dateTime": "2026-08-26T07:15:00+08:00"},
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
        )
    )

    assert result["status"] == "success"
    update_body = fake_service._events.update_calls[0]["body"]
    assert update_body["recurrence"] == [
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z"
    ]


def test_update_events_rejects_invalid_recurrence_without_calling_the_api(
    fake_service,
):
    fake_service._events._get_result = {
        "id": "existing-1",
        "start": {"dateTime": "2026-08-26T07:00:00+08:00"},
        "end": {"dateTime": "2026-08-26T07:15:00+08:00"},
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=FORTNIGHTLY",
        )
    )

    assert result["status"] == "error"
    assert fake_service._events.update_calls == []


def test_update_events_rejects_recurrence_on_all_day_event_without_start_time(
    fake_service,
):
    """An all-day event has no "dateTime" (only "date"), so a caller
    setting recurrence without also passing start_time must get a clear
    error instead of an opaque TypeError from parsing None as a
    datetime."""
    fake_service._events._get_result = {
        "id": "existing-1",
        "start": {"date": "2026-08-26"},
        "end": {"date": "2026-08-27"},
    }

    result = json.loads(
        calendar.google_calendar_update_events(
            event_id="existing-1",
            recurrence="FREQ=DAILY",
        )
    )

    assert result["status"] == "error"
    assert "start time" in result["message"]
    assert fake_service._events.update_calls == []
