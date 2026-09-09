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
):
    event = {
        "id": event_id,
        "subject": subject,
        "isCancelled": is_cancelled,
        "showAs": show_as,
        "start": {"dateTime": "2026-08-27T10:00:00"},
        "end": {"dateTime": "2026-08-27T10:30:00"},
    }
    if response is not None:
        event["responseStatus"] = {"response": response}
    return event


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


def test_create_event_ignores_organizers_own_declined_event(monkeypatch):
    """Regression test matching google_calendar's equivalent check: an
    event the organizer personally declined still sits on their calendar
    (declining doesn't clear showAs), but it's not something they're
    actually busy for. `/me/calendarView` reports the signed-in user's own
    response via `responseStatus`."""
    graph_request = Mock(
        side_effect=[
            {"value": [_busy_event(response="declined")]},
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
    assert graph_request.call_count == 2


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


def test_create_event_unchecked_attendee_does_not_block_creation(monkeypatch):
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
            {"id": "created"},
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

    assert result["status"] == "success"
    assert result["unchecked_attendees"] == ["outsider@gmail.com"]
    create_call = graph_request.call_args_list[-1]
    assert create_call.args[:2] == ("POST", "/me/events")
    assert create_call.kwargs["body"]["attendees"] == [
        {"emailAddress": {"address": "outsider@gmail.com"}, "type": "required"}
    ]


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


def test_update_event_same_window_only_checks_newly_added_attendee(monkeypatch):
    """Adding an attendee without moving the event must only check the new
    attendee's schedule - checking an existing attendee against the
    unchanged window would always find the event's own busy block on their
    schedule and falsely report a conflict."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    {"emailAddress": {"address": "old@example.com"}},
                ],
            },
            {"value": [{"scheduleId": "new@example.com", "scheduleItems": []}]},
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


def test_update_event_treats_existing_attendee_case_insensitively(monkeypatch):
    """Regression test for a review finding: an existing attendee re-passed
    with different casing must still be recognized as "already there" -
    otherwise it's treated as newly-added, gets checked against the
    unchanged window, and always self-conflicts on its own busy block for
    this very event."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [
                    {"emailAddress": {"address": "old@example.com"}},
                ],
            },
            {"value": [{"scheduleId": "new@example.com", "scheduleItems": []}]},
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


def test_update_event_reuses_existing_timezone_when_time_is_unchanged(monkeypatch):
    """Regression test for a review finding: when only attendees change
    (start_datetime not passed), the conflict check must use the existing
    event's own timeZone, not whatever `timezone` the caller happened to
    pass. `timezone` only describes a start_datetime/end_datetime the
    caller is ALSO providing; existing_start/end came back from Graph
    as-is (no Prefer header sent), so they're only meaningful paired with
    the timeZone Graph actually reported for them."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {
                    "dateTime": "2026-08-27T10:00:00",
                    "timeZone": "Asia/Singapore",
                },
                "end": {
                    "dateTime": "2026-08-27T10:30:00",
                    "timeZone": "Asia/Singapore",
                },
                "attendees": [],
            },
            {"value": [{"scheduleId": "new@example.com", "scheduleItems": []}]},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            attendees=["new@example.com"],
            timezone="UTC",
        )
    )

    assert result["status"] == "success"
    schedule_call = graph_request.call_args_list[1]
    assert schedule_call.kwargs["body"]["startTime"]["timeZone"] == "Asia/Singapore"
    assert schedule_call.kwargs["body"]["endTime"]["timeZone"] == "Asia/Singapore"


def test_update_event_fails_loudly_when_existing_event_has_no_timezone(monkeypatch):
    graph_request = Mock(
        return_value={
            "start": {"dateTime": "2026-08-27T10:00:00"},
            "end": {"dateTime": "2026-08-27T10:30:00"},
            "attendees": [],
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
    assert "timeZone" in result["message"]


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


def test_update_event_toggling_all_day_alone_does_not_self_conflict_on_attendees(
    monkeypatch,
):
    """Regression test: toggling is_all_day alone doesn't move the literal
    start/end clock values, so querying an EXISTING attendee's schedule for
    that same unmoved window would always find this very event's own busy
    block there. The organizer check may legitimately run (see the test
    above), but existing attendees must not be re-checked against an
    unmoved window just because is_all_day flipped."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
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
            is_all_day=True,
        )
    )

    assert result["status"] == "success"
    # Only the organizer's calendarView should have been queried (2nd
    # call) - no getSchedule call for the existing attendee.
    assert graph_request.call_count == 3
    assert graph_request.call_args_list[0].args[:2] == ("GET", "/me/events/self-1")
    assert graph_request.call_args_list[1].args[:2] == ("GET", "/me/calendarView")


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
            {"value": [{"scheduleId": "existing@example.com", "scheduleItems": []}]},
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


def test_create_event_missing_schedule_scope_degrades_to_unchecked(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {"value": []},  # organizer calendarView
            outlook._GraphRequestError("403 Forbidden", status_code=403),
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
    assert result["unchecked_attendees"] == ["chelsea@example.com"]
    # The reason must give an LLM caller something actionable - reconnecting
    # the connector - rather than a bare error string.
    assert "reconnect" in result["unchecked_reason"].lower()


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
            {"id": "created"},
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

    assert result["status"] == "success"
    assert result["unchecked_attendees"] == ["ghost@example.com"]


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
                        "emailAddress": {"address": "existing@example.com"},
                        "type": "required",
                        "status": {
                            "response": "accepted",
                            "time": "2026-08-20T00:00:00Z",
                        },
                    }
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
            attendees=["existing@example.com"],
        )
    )

    assert result["status"] == "success"
    patch_call = graph_request.call_args_list[1]
    assert "attendees" not in patch_call.kwargs["body"]


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


def test_update_event_partial_overlap_nudge_does_not_self_conflict(monkeypatch):
    """Regression test: nudging a boundary to a window that still overlaps
    the event's OLD window is just as unsafe to check existing attendees
    against as an unchanged window - the overlap still contains this
    event's own busy block on their calendar. Only a genuinely disjoint
    move is safe to re-check everyone."""
    graph_request = Mock(
        side_effect=[
            {
                "start": {"dateTime": "2026-08-27T10:00:00", "timeZone": "UTC"},
                "end": {"dateTime": "2026-08-27T10:30:00", "timeZone": "UTC"},
                "attendees": [{"emailAddress": {"address": "existing@example.com"}}],
                "isAllDay": False,
            },
            {"value": []},  # organizer calendarView
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
    # Existing-event GET + organizer calendarView + PATCH - no getSchedule
    # call for the existing attendee.
    assert graph_request.call_count == 3
    assert graph_request.call_args_list[-1].args[:2] == ("PATCH", "/me/events/self-1")


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


def test_update_event_empty_string_attendees_is_treated_as_not_provided(monkeypatch):
    """An empty string for attendees (e.g. an accidental default) must be
    treated the same as not passing attendees at all - not as "clear
    every attendee" - matching every other optional field's truthy
    convention here and outlook_create_event's own check."""
    graph_request = Mock(return_value={"id": "updated"})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="self-1",
            subject="New Subject",
            attendees="",
        )
    )

    assert result["status"] == "success"
    # A subject-only edit (attendees="" not counting as given) never needs
    # the existing event - just the one PATCH.
    graph_request.assert_called_once()
    patch_call = graph_request.call_args
    assert patch_call.args[:2] == ("PATCH", "/me/events/self-1")
    assert "attendees" not in patch_call.kwargs["body"]


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


def test_update_event_unchecked_reason_appears_alongside_a_conflict(monkeypatch):
    """unchecked_attendees/unchecked_reason must still surface even when
    the response status is "conflict" (from the organizer or another
    attendee), not just on a clean "success" - a caller acting on the
    conflict still needs to know some attendees were never actually
    checked."""
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
    assert result["unchecked_attendees"] == ["outsider@gmail.com"]
    assert "reconnect" in result["unchecked_reason"].lower()
