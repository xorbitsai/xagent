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


def test_build_graph_recurrence_until_backs_off_end_date_when_occurrence_lands_later():
    """Confirmed bug: Graph's recurrenceRange(type=endDate) includes the
    WHOLE endDate day regardless of time-of-day, but every occurrence in
    the series happens at the anchor's own local time (09:00 Shanghai
    here), not UNTIL's. 2026-09-11T00:00:00Z is 2026-09-11 08:00 Shanghai
    local - earlier in the day than the series' 09:00 occurrence time, so
    that Friday's occurrence is genuinely past the true UTC cutoff and
    must be excluded by backing endDate off to the 10th, or Graph would
    silently run the series one occurrence past what UNTIL specified."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=WEEKLY;UNTIL=20260911T000000Z",
        "2026-08-14T09:00:00",
        "Asia/Shanghai",
    )
    assert recurrence["range"]["endDate"] == "2026-09-10"


def test_build_graph_recurrence_until_keeps_end_date_when_occurrence_lands_earlier():
    """Same scenario, but UNTIL's local time-of-day (20:00 Shanghai) falls
    AFTER the series' own occurrence time (09:00) on the same calendar day
    - that day's occurrence is still within the true cutoff, so no
    back-off is needed."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=WEEKLY;UNTIL=20260911T120000Z",
        "2026-08-14T09:00:00",
        "Asia/Shanghai",
    )
    assert recurrence["range"]["endDate"] == "2026-09-11"


def test_build_graph_recurrence_weekly_with_explicit_byday():
    recurrence = outlook._build_graph_recurrence(
        "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
        "2026-08-26T07:00:00+08:00",
    )

    assert recurrence["pattern"] == {
        "type": "weekly",
        "interval": 1,
        "daysOfWeek": ["monday", "tuesday", "wednesday", "thursday", "friday"],
        "firstDayOfWeek": "monday",
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


def test_build_graph_recurrence_weekly_byday_tolerates_lowercase():
    """An LLM/user-supplied BYDAY isn't guaranteed to be clean uppercase
    RFC 5545 tokens - dateutil's own RRULE validation already accepts
    lowercase day codes (it's whitespace inside the list, not case, that
    it rejects outright before this ever runs), so a raw dict lookup with
    the unnormalized text would still KeyError on "mo"/"tu"."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=WEEKLY;BYDAY=mo,tu", "2026-08-26T07:00:00+08:00"
    )

    assert recurrence["pattern"]["daysOfWeek"] == ["monday", "tuesday"]


def test_build_graph_recurrence_rejects_invalid_byday_code():
    """FREQ=WEEKLY combined with a numbered BYDAY (e.g. "2TU", meaning
    "the second Tuesday" - normally a MONTHLY/YEARLY construct) is
    syntactically valid RRULE that dateutil's own validation accepts, but
    it has no numbered-occurrence concept in Graph's plain
    weekly/daysOfWeek pattern - it must be rejected here rather than
    silently truncated to a bare weekday or crashing on a raw dict
    lookup."""
    with pytest.raises(ValueError, match="invalid day code in BYDAY"):
        outlook._build_graph_recurrence(
            "FREQ=WEEKLY;BYDAY=2TU", "2026-08-26T07:00:00+08:00"
        )


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


def test_build_graph_recurrence_monthly_without_bymonthday_defaults_to_start_date():
    """RFC 5545: an unqualified FREQ=MONTHLY (no BYMONTHDAY) repeats on
    DTSTART's own day of the month - the most natural way to say "repeat
    monthly" must actually work, not be rejected as unsupported."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=MONTHLY", "2026-08-15T07:00:00+08:00"
    )

    assert recurrence["pattern"] == {
        "type": "absoluteMonthly",
        "interval": 1,
        "dayOfMonth": 15,
    }


def test_build_graph_recurrence_yearly_without_bymonth_defaults_to_start_date():
    """Same RFC 5545 default as MONTHLY: an unqualified FREQ=YEARLY
    repeats on DTSTART's own month and day."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=YEARLY", "2026-08-15T07:00:00+08:00"
    )

    assert recurrence["pattern"] == {
        "type": "absoluteYearly",
        "interval": 1,
        "dayOfMonth": 15,
        "month": 8,
    }


def test_build_graph_recurrence_rejects_multi_value_bymonthday():
    """BYMONTHDAY=15,20 is valid RFC 5545 (multiple days per month), but
    Graph's absoluteMonthly pattern only accepts a single dayOfMonth - a
    raw ValueError from int("15,20") must not leak through unworded."""
    with pytest.raises(ValueError, match="only supports a single value"):
        outlook._build_graph_recurrence(
            "FREQ=MONTHLY;BYMONTHDAY=15,20", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_bymonthday_with_byday_on_monthly():
    """FREQ=MONTHLY with both BYMONTHDAY and BYDAY (e.g. "the 15th, but
    only if a Tuesday") is valid RFC 5545 - dateutil accepts it - but
    Graph's absoluteMonthly pattern has no way to express that
    intersection. Silently keeping only BYMONTHDAY would translate it into
    a materially different, broader "every 15th" recurrence with no
    warning."""
    with pytest.raises(ValueError, match="BYMONTHDAY and BYDAY"):
        outlook._build_graph_recurrence(
            "FREQ=MONTHLY;BYMONTHDAY=15;BYDAY=2TU", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_bymonthday_with_byday_on_yearly():
    """Same gap as the MONTHLY case above, for YEARLY: BYMONTH+BYMONTHDAY+
    BYDAY together (e.g. "Nov 15th, but only if a Thursday") is valid RFC
    5545 but has no relativeYearly/absoluteYearly equivalent - silently
    keeping only BYMONTH+BYDAY would drop the BYMONTHDAY constraint and
    translate it into a materially different "4th Thursday of November"
    recurrence with no warning."""
    with pytest.raises(ValueError, match="BYMONTHDAY and BYDAY"):
        outlook._build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=11;BYMONTHDAY=15;BYDAY=4TH",
            "2026-08-15T07:00:00+08:00",
        )


def test_build_graph_recurrence_rejects_out_of_range_bymonthday():
    with pytest.raises(ValueError, match="BYMONTHDAY must be between 1 and 31"):
        outlook._build_graph_recurrence(
            "FREQ=MONTHLY;BYMONTHDAY=32", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_negative_bymonthday():
    """RRULE allows BYMONTHDAY=-1 ("the last day of the month") as valid
    syntax, but Graph's dayOfMonth field only accepts 1-31 - this must be
    rejected as unsupported-by-this-connector (not as malformed RRULE
    syntax, which "invalid recurrence rule" would misleadingly imply)
    rather than sent to Graph as a bare -1."""
    with pytest.raises(
        ValueError,
        match="unsupported recurrence pattern: BYMONTHDAY must be between 1 and 31",
    ):
        outlook._build_graph_recurrence(
            "FREQ=MONTHLY;BYMONTHDAY=-1", "2026-08-15T07:00:00+08:00"
        )


@pytest.mark.parametrize("day", [29, 30, 31])
def test_build_graph_recurrence_rejects_bymonthday_graph_would_clamp_on_monthly(day):
    """Confirmed via Microsoft's own docs: Graph's absoluteMonthly clamps a
    dayOfMonth past a short month's length to that month's last day (e.g.
    31 in April becomes April 30) instead of skipping the month the way
    RFC 5545 does - so any day not valid in EVERY month (including
    February) must be rejected rather than silently diverging."""
    with pytest.raises(ValueError, match="only BYMONTHDAY 1-28"):
        outlook._build_graph_recurrence(
            f"FREQ=MONTHLY;BYMONTHDAY={day}", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_accepts_bymonthday_28_on_monthly():
    """28 is valid in every month (including non-leap February), so it
    must still be accepted - the fix above must not overreach."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=MONTHLY;BYMONTHDAY=28", "2026-08-15T07:00:00+08:00"
    )
    assert recurrence["pattern"]["dayOfMonth"] == 28


def test_build_graph_recurrence_rejects_bymonthday_29_on_yearly_february():
    """FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=29 ("Feb 29 every year") would
    have Graph clamp to Feb 28 in every non-leap year, silently dropping
    the "only on a leap year" semantics RFC 5545 actually specifies."""
    with pytest.raises(ValueError, match="Feb 29 in a non-leap year"):
        outlook._build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=29", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_bymonthday_that_never_exists_in_month():
    """BYMONTH=4;BYMONTHDAY=31 has no equivalent in ANY year (April never
    has 31 days) - a different, more clear-cut error than the "sometimes
    valid" Feb 29 case above."""
    with pytest.raises(ValueError, match="does not exist in month 4"):
        outlook._build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=4;BYMONTHDAY=31", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_accepts_bymonthday_30_on_yearly_april():
    """April always has exactly 30 days in every year - no leap-year-style
    divergence is possible, so this must still be accepted."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=YEARLY;BYMONTH=4;BYMONTHDAY=30", "2026-08-15T07:00:00+08:00"
    )
    assert recurrence["pattern"]["dayOfMonth"] == 30


def test_build_graph_recurrence_rejects_out_of_range_bymonth():
    with pytest.raises(ValueError, match="BYMONTH must be between 1 and 12"):
        outlook._build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=13;BYMONTHDAY=1", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="INTERVAL must be a positive integer"):
        outlook._build_graph_recurrence(
            "FREQ=DAILY;INTERVAL=0", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_resolves_windows_style_timezone():
    """Graph commonly reports Windows-style timezone identifiers (e.g. for
    events created via Outlook desktop/web rather than this tool), which
    dateutil.tz.gettz can't resolve directly - a small common-cases
    mapping must translate it rather than raising for an otherwise valid,
    pre-existing event."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=DAILY", "2026-08-26T07:00:00", "Pacific Standard Time"
    )
    assert recurrence["range"]["recurrenceTimeZone"] == "Pacific Standard Time"


def test_build_graph_recurrence_rejects_unmapped_timezone():
    with pytest.raises(ValueError, match="unknown timezone"):
        outlook._build_graph_recurrence(
            "FREQ=DAILY", "2026-08-26T07:00:00", "Not A Real Timezone"
        )


def test_build_graph_recurrence_rejects_unsupported_freq():
    """FREQ=HOURLY is valid RFC 5545 that dateutil accepts, but this
    connector has no Graph pattern type to translate it into."""
    with pytest.raises(ValueError, match="unsupported recurrence pattern"):
        outlook._build_graph_recurrence("FREQ=HOURLY", "2026-08-11T07:00:00+08:00")


def test_build_graph_recurrence_rejects_invalid_rrule():
    with pytest.raises(ValueError, match="invalid recurrence rule"):
        outlook._build_graph_recurrence("FREQ=FORTNIGHTLY", "2026-08-11T07:00:00+08:00")


def test_build_graph_recurrence_relative_monthly():
    """FREQ=MONTHLY;BYDAY=2TU ("the second Tuesday of every month") has no
    BYMONTHDAY equivalent but does map onto Graph's relativeMonthly type."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=MONTHLY;BYDAY=2TU", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"] == {
        "type": "relativeMonthly",
        "interval": 1,
        "daysOfWeek": ["tuesday"],
        "index": "second",
    }


def test_build_graph_recurrence_relative_monthly_last_weekday():
    recurrence = outlook._build_graph_recurrence(
        "FREQ=MONTHLY;BYDAY=-1FR", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"] == {
        "type": "relativeMonthly",
        "interval": 1,
        "daysOfWeek": ["friday"],
        "index": "last",
    }


def test_build_graph_recurrence_relative_yearly():
    recurrence = outlook._build_graph_recurrence(
        "FREQ=YEARLY;BYMONTH=11;BYDAY=4TH", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"] == {
        "type": "relativeYearly",
        "interval": 1,
        "daysOfWeek": ["thursday"],
        "index": "fourth",
        "month": 11,
    }


def test_build_graph_recurrence_relative_pattern_rejects_mixed_ordinals():
    with pytest.raises(ValueError, match="mixes different numeric ordinals"):
        outlook._build_graph_recurrence(
            "FREQ=MONTHLY;BYDAY=2TU,3WE", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_relative_pattern_rejects_missing_ordinal():
    """A plain (non-numbered) BYDAY on MONTHLY means something different
    under RFC 5545 (every such weekday in the month) - it must not be
    silently treated as a relative single-occurrence pattern."""
    with pytest.raises(ValueError, match="no numeric ordinal"):
        outlook._build_graph_recurrence(
            "FREQ=MONTHLY;BYDAY=TU", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_relative_pattern_rejects_unsupported_ordinal():
    with pytest.raises(ValueError, match="has no Outlook equivalent"):
        outlook._build_graph_recurrence(
            "FREQ=MONTHLY;BYDAY=5TU", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_leftover_byday_on_daily():
    """BYDAY on a plain DAILY rule is syntactically valid RRULE but this
    connector's daily pattern has no way to honor it - it must be rejected
    rather than silently ignored, which would translate the rule into a
    materially broader "every day" recurrence with no warning."""
    with pytest.raises(ValueError, match="does not translate BYDAY"):
        outlook._build_graph_recurrence(
            "FREQ=DAILY;BYDAY=MO", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_leftover_bysetpos():
    with pytest.raises(ValueError, match="BYSETPOS"):
        outlook._build_graph_recurrence(
            "FREQ=WEEKLY;BYDAY=MO;BYSETPOS=1", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_weekly_dedupes_byday():
    recurrence = outlook._build_graph_recurrence(
        "FREQ=WEEKLY;BYDAY=MO,MO,TU", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"]["daysOfWeek"] == ["monday", "tuesday"]


def test_build_graph_recurrence_weekly_defaults_first_day_of_week_to_monday():
    """RFC 5545 defaults WKST to Monday when omitted, but Graph's own
    firstDayOfWeek default is Sunday - the RRULE's own default must be
    stamped explicitly rather than picking up Graph's different one."""
    recurrence = outlook._build_graph_recurrence(
        "FREQ=WEEKLY;BYDAY=MO,TU", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"]["firstDayOfWeek"] == "monday"


def test_build_graph_recurrence_weekly_honors_explicit_wkst():
    recurrence = outlook._build_graph_recurrence(
        "FREQ=WEEKLY;BYDAY=MO,TU;WKST=SU", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"]["firstDayOfWeek"] == "sunday"


def test_resolve_timezone_rejects_blank_string():
    with pytest.raises(ValueError, match="must not be blank"):
        outlook._resolve_timezone("")


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
    # 2026-09-11T23:59:59Z is 2026-09-12 07:59:59 in Asia/Manila (+08:00) -
    # the correct local endDate is the 12th, not the UTC calendar date.
    assert payload["recurrence"]["range"]["endDate"] == "2026-09-12"


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


def test_create_event_rejects_explicit_empty_recurrence(monkeypatch):
    """Regression test: outlook_create_event now aligns with
    outlook_update_event's `is not None` semantic for recurrence -
    recurrence="" must not be silently treated the same as omitting it."""
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

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
    than requiring the caller to repeat it. Without a Prefer header Graph
    always reports start in UTC regardless of the event's true zone, so
    this must first read originalStartTimeZone and re-fetch start
    expressed in that zone rather than trusting the UTC-defaulted GET."""
    graph_request = Mock(
        side_effect=[
            {"originalStartTimeZone": "UTC"},
            {"start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "UTC"}},
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
    assert graph_request.call_count == 3
    tz_get_call = graph_request.call_args_list[0]
    assert tz_get_call.args[:2] == ("GET", "/me/events/existing-1")
    assert tz_get_call.kwargs["params"] == {"$select": "originalStartTimeZone"}
    start_get_call = graph_request.call_args_list[1]
    assert start_get_call.args[:2] == ("GET", "/me/events/existing-1")
    assert start_get_call.kwargs["extra_headers"] == {
        "Prefer": 'outlook.timezone="UTC"'
    }
    patch_call = graph_request.call_args_list[2]
    assert patch_call.kwargs["body"]["recurrence"]["range"]["startDate"] == (
        "2026-08-26"
    )
    assert patch_call.kwargs["body"]["recurrence"]["range"]["recurrenceTimeZone"] == (
        "UTC"
    )


def test_update_event_uses_the_existing_events_own_timezone_not_the_default(
    monkeypatch,
):
    """Regression test: a caller setting recurrence without also moving the
    event (so `timezone` is left at its "UTC" default, unrelated to this
    event) must get the recurrence built against the existing event's own
    timeZone - not silently UTC, and not the caller's unrelated default -
    or a weekly pattern with no BYDAY can be generated for the wrong day
    whenever the event's local day differs from its UTC day."""
    graph_request = Mock(
        side_effect=[
            {"originalStartTimeZone": "Asia/Manila"},
            {"start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "Asia/Manila"}},
            {"id": "updated"},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="existing-1",
            recurrence="FREQ=WEEKLY;COUNT=5",
        )
    )

    assert result["status"] == "success"
    patch_call = graph_request.call_args_list[2]
    recurrence = patch_call.kwargs["body"]["recurrence"]
    assert recurrence["range"]["recurrenceTimeZone"] == "Asia/Manila"
    # 2026-08-26 is a Wednesday; deriving the weekday from the wrong
    # timezone (e.g. treating this UTC-adjacent dateTime as UTC) would
    # shift it to the wrong day.
    assert recurrence["pattern"]["daysOfWeek"] == ["wednesday"]


def test_update_event_fails_loudly_when_original_timezone_is_unavailable(monkeypatch):
    graph_request = Mock(return_value={})
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="existing-1",
            recurrence="FREQ=WEEKLY;COUNT=5",
        )
    )

    assert result["status"] == "error"
    assert "originalStartTimeZone" in result["message"]
    graph_request.assert_called_once()


def test_update_event_fails_loudly_on_legacy_custom_timezone(monkeypatch):
    """Graph's documented placeholder for an unresolvable legacy custom
    timezone (set in desktop Outlook) - there's no name to resolve it by,
    so this must be rejected rather than passed to Graph's Prefer header
    as a literal, meaningless value."""
    graph_request = Mock(
        return_value={"originalStartTimeZone": "tzone://Microsoft/Custom"}
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="existing-1",
            recurrence="FREQ=WEEKLY;COUNT=5",
        )
    )

    assert result["status"] == "error"
    assert "originalStartTimeZone" in result["message"]
    graph_request.assert_called_once()


def test_update_event_fails_loudly_when_prefer_header_get_has_no_start(monkeypatch):
    graph_request = Mock(
        side_effect=[
            {"originalStartTimeZone": "Asia/Manila"},
            {"start": {}},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="existing-1",
            recurrence="FREQ=WEEKLY;COUNT=5",
        )
    )

    assert result["status"] == "error"
    assert "could not determine the event's start time" in result["message"]


def test_update_event_fails_loudly_when_prefer_header_is_silently_ignored(
    monkeypatch,
):
    """Graph's own documented no-Prefer-header default is exactly
    timeZone: "UTC" - if the second GET (which asked for a specific
    non-UTC zone via the Prefer header) still comes back as UTC, that's a
    strong signal the header wasn't honored. Silently trusting it anyway
    would reintroduce the exact BYDAY-from-UTC-day bug this two-GET flow
    exists to fix - it must fail loudly instead."""
    graph_request = Mock(
        side_effect=[
            {"originalStartTimeZone": "Asia/Manila"},
            {"start": {"dateTime": "2026-08-26T07:00:00", "timeZone": "UTC"}},
        ]
    )
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="existing-1",
            recurrence="FREQ=WEEKLY;COUNT=5",
        )
    )

    assert result["status"] == "error"
    assert "Prefer header wasn't honored" in result["message"]


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


def test_update_event_rejects_explicit_empty_recurrence(monkeypatch):
    """Regression test: every other optional field in this function uses
    an `is not None` check so an explicitly-passed value is never silently
    ignored; recurrence must behave the same way rather than treating
    recurrence="" identically to not mentioning it at all."""
    graph_request = Mock()
    monkeypatch.setattr(outlook, "_graph_request", graph_request)

    result = json.loads(
        outlook.outlook_update_event(
            event_id="existing-1",
            start_datetime="2026-08-26T07:00:00",
            end_datetime="2026-08-26T07:15:00",
            recurrence="",
        )
    )

    assert result["status"] == "error"
    assert "must not be empty" in result["message"]
    graph_request.assert_not_called()
