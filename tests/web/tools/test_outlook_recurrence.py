import pytest

from xagent.web.tools.mcp import outlook_recurrence


def test_build_graph_recurrence_daily_with_until():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY;INTERVAL=2;UNTIL=20260911T235959Z",
        "2026-08-26T07:00:00",
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
    recurrence = outlook_recurrence.build_graph_recurrence(
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
    recurrence = outlook_recurrence.build_graph_recurrence(
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
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=WEEKLY;UNTIL=20260911T120000Z",
        "2026-08-14T09:00:00",
        "Asia/Shanghai",
    )
    assert recurrence["range"]["endDate"] == "2026-09-11"


def test_build_graph_recurrence_normalizes_offset_anchor_to_recurrence_timezone():
    """Graph interprets startDate, daysOfWeek, and recurrenceTimeZone as
    one local-time frame. An offset-bearing start therefore has to be
    converted into the separately supplied recurrence timezone before
    any calendar fields are derived."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=WEEKLY;COUNT=2",
        "2026-08-26T00:30:00+08:00",
        "America/Los_Angeles",
    )

    assert recurrence["pattern"]["daysOfWeek"] == ["tuesday"]
    assert recurrence["range"] == {
        "type": "numbered",
        "startDate": "2026-08-25",
        "numberOfOccurrences": 2,
        "recurrenceTimeZone": "America/Los_Angeles",
    }


def test_build_graph_recurrence_weekly_with_explicit_byday():
    recurrence = outlook_recurrence.build_graph_recurrence(
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
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=WEEKLY;COUNT=5",
        "2026-08-26T07:00:00",
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
    recurrence = outlook_recurrence.build_graph_recurrence(
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
    with pytest.raises(ValueError, match="numeric BYDAY values are only valid"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=WEEKLY;BYDAY=2TU", "2026-08-26T07:00:00+08:00"
        )


def test_build_graph_recurrence_absolute_monthly():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=MONTHLY;BYMONTHDAY=15",
        "2026-08-15T07:00:00",
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
    recurrence = outlook_recurrence.build_graph_recurrence(
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
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=MONTHLY", "2026-08-15T07:00:00"
    )

    assert recurrence["pattern"] == {
        "type": "absoluteMonthly",
        "interval": 1,
        "dayOfMonth": 15,
    }


def test_build_graph_recurrence_yearly_without_bymonth_defaults_to_start_date():
    """Same RFC 5545 default as MONTHLY: an unqualified FREQ=YEARLY
    repeats on DTSTART's own month and day."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=YEARLY", "2026-08-15T07:00:00"
    )

    assert recurrence["pattern"] == {
        "type": "absoluteYearly",
        "interval": 1,
        "dayOfMonth": 15,
        "month": 8,
    }


def test_build_graph_recurrence_yearly_bymonth_derives_day_from_start_date():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=YEARLY;BYMONTH=3", "2026-04-30T07:00:00"
    )

    assert recurrence["pattern"] == {
        "type": "absoluteYearly",
        "interval": 1,
        "dayOfMonth": 30,
        "month": 3,
    }


def test_build_graph_recurrence_rejects_multi_value_bymonthday():
    """BYMONTHDAY=15,20 is valid RFC 5545 (multiple days per month), but
    Graph's absoluteMonthly pattern only accepts a single dayOfMonth - a
    raw ValueError from int("15,20") must not leak through unworded."""
    with pytest.raises(ValueError, match="only supports a single value"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=MONTHLY;BYMONTHDAY=15,20", "2026-08-15T07:00:00+08:00"
        )


@pytest.mark.parametrize(
    "rule",
    [
        "FREQ=MONTHLY;BYMONTHDAY=1_5",
        "FREQ=YEARLY;BYMONTH=1_1",
        "FREQ=MONTHLY;BYDAY=0002TU",
    ],
)
def test_build_graph_recurrence_rejects_non_rfc_numeric_spellings(rule):
    with pytest.raises(ValueError, match="invalid recurrence rule|invalid day code"):
        outlook_recurrence.build_graph_recurrence(rule, "2026-08-15T07:00:00")


@pytest.mark.parametrize("byday", ["+2TU", "02TU"])
def test_build_graph_recurrence_accepts_valid_numbered_byday_spellings(byday):
    recurrence = outlook_recurrence.build_graph_recurrence(
        f"FREQ=MONTHLY;BYDAY={byday}", "2026-08-15T07:00:00"
    )

    assert recurrence["pattern"]["index"] == "second"


def test_build_graph_recurrence_rejects_bymonthday_with_byday_on_monthly():
    """FREQ=MONTHLY with both BYMONTHDAY and BYDAY (e.g. "the 15th, but
    only if a Tuesday") is valid RFC 5545 - dateutil accepts it - but
    Graph's absoluteMonthly pattern has no way to express that
    intersection. Silently keeping only BYMONTHDAY would translate it into
    a materially different, broader "every 15th" recurrence with no
    warning."""
    with pytest.raises(ValueError, match="BYMONTHDAY and BYDAY"):
        outlook_recurrence.build_graph_recurrence(
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
        outlook_recurrence.build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=11;BYMONTHDAY=15;BYDAY=4TH",
            "2026-08-15T07:00:00+08:00",
        )


def test_build_graph_recurrence_rejects_yearly_bymonthday_without_bymonth():
    """Confirmed bug: FREQ=YEARLY;BYMONTHDAY=15 with no BYMONTH means "the
    15th of every month, every year" under RFC 5545 (confirmed against
    dateutil) - Outlook's absoluteYearly pattern always requires a single
    specific month and has no way to represent that, so defaulting the
    month from DTSTART silently narrowed a 12x/year series down to a
    single yearly occurrence with no warning."""
    with pytest.raises(
        ValueError, match="without BYMONTH means this day of EVERY month"
    ):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=YEARLY;BYMONTHDAY=15", "2026-06-01T09:00:00"
        )


def test_build_graph_recurrence_rejects_yearly_byday_without_bymonth():
    """Confirmed bug: FREQ=YEARLY;BYDAY=1MO with no BYMONTH means a single
    year-wide ordinal weekday under RFC 5545 (e.g. "the first Monday of
    the year" - confirmed against dateutil), not an ordinal scoped to
    whatever month DTSTART happens to fall in. Outlook's relativeYearly
    pattern always requires a specific month and has no way to represent
    "year-wide", so defaulting the month from DTSTART silently produced a
    different (and wrong) date with no warning."""
    with pytest.raises(ValueError, match="without BYMONTH means a single year-wide"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=YEARLY;BYDAY=1MO", "2026-06-01T09:00:00"
        )


def test_build_graph_recurrence_accepts_yearly_selectors_with_explicit_bymonth():
    """The fix above must not overreach - an explicit BYMONTH still works
    for both the BYMONTHDAY and BYDAY yearly variants."""
    r1 = outlook_recurrence.build_graph_recurrence(
        "FREQ=YEARLY;BYMONTH=6;BYMONTHDAY=15", "2026-06-01T09:00:00"
    )
    assert r1["pattern"] == {
        "type": "absoluteYearly",
        "interval": 1,
        "dayOfMonth": 15,
        "month": 6,
    }
    r2 = outlook_recurrence.build_graph_recurrence(
        "FREQ=YEARLY;BYMONTH=6;BYDAY=1MO", "2026-06-01T09:00:00"
    )
    assert r2["pattern"] == {
        "type": "relativeYearly",
        "interval": 1,
        "daysOfWeek": ["monday"],
        "index": "first",
        "month": 6,
    }


def test_build_graph_recurrence_rejects_out_of_range_bymonthday():
    with pytest.raises(ValueError, match="BYMONTHDAY must be between 1 and 31"):
        outlook_recurrence.build_graph_recurrence(
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
        outlook_recurrence.build_graph_recurrence(
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
        outlook_recurrence.build_graph_recurrence(
            f"FREQ=MONTHLY;BYMONTHDAY={day}", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_derived_monthly_day_error_names_start_date():
    with pytest.raises(ValueError, match=r"start date's day of month \(31\)"):
        outlook_recurrence.build_graph_recurrence("FREQ=MONTHLY", "2026-01-31T07:00:00")


def test_build_graph_recurrence_derived_yearly_day_error_names_start_date():
    with pytest.raises(ValueError, match=r"start date's day of month \(29\)"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=2", "2028-03-29T07:00:00"
        )


def test_build_graph_recurrence_accepts_bymonthday_28_on_monthly():
    """28 is valid in every month (including non-leap February), so it
    must still be accepted - the fix above must not overreach."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=MONTHLY;BYMONTHDAY=28", "2026-08-15T07:00:00+08:00"
    )
    assert recurrence["pattern"]["dayOfMonth"] == 28


def test_build_graph_recurrence_rejects_bymonthday_29_on_yearly_february():
    """FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=29 ("Feb 29 every year") would
    have Graph clamp to Feb 28 in every non-leap year, silently dropping
    the "only on a leap year" semantics RFC 5545 actually specifies."""
    with pytest.raises(ValueError, match="Feb 29 in a non-leap year"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=2;BYMONTHDAY=29", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_bymonthday_that_never_exists_in_month():
    """BYMONTH=4;BYMONTHDAY=31 has no equivalent in ANY year (April never
    has 31 days) - a different, more clear-cut error than the "sometimes
    valid" Feb 29 case above."""
    with pytest.raises(ValueError, match="does not exist in month 4"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=4;BYMONTHDAY=31", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_accepts_bymonthday_30_on_yearly_april():
    """April always has exactly 30 days in every year - no leap-year-style
    divergence is possible, so this must still be accepted."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=YEARLY;BYMONTH=4;BYMONTHDAY=30", "2026-08-15T07:00:00+08:00"
    )
    assert recurrence["pattern"]["dayOfMonth"] == 30


def test_build_graph_recurrence_rejects_out_of_range_bymonth():
    with pytest.raises(ValueError, match="BYMONTH must be between 1 and 12"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=13;BYMONTHDAY=1", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="INTERVAL must be a positive integer"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=DAILY;INTERVAL=0", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_interval_beyond_graph_int32():
    """Confirmed bug: RFC 5545's INTERVAL grammar has no upper bound, but
    Graph types recurrencePattern.interval as a signed Int32 - a value
    beyond that (valid RFC 5545 text the shared parser's digit-only/
    positivity checks accept) used to reach Graph as-is and fail remotely
    with an opaque error instead of a clear local one."""
    with pytest.raises(ValueError, match="INTERVAL must be at most 2147483647"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=DAILY;INTERVAL=2147483648", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_count_beyond_graph_int32():
    with pytest.raises(ValueError, match="COUNT must be at most 2147483647"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=DAILY;COUNT=2147483648", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_accepts_interval_at_graph_int32_boundary():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY;INTERVAL=2147483647", "2026-08-15T07:00:00+08:00"
    )
    assert recurrence["pattern"]["interval"] == 2147483647


def test_build_graph_recurrence_accepts_count_at_graph_int32_boundary():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY;COUNT=2147483647", "2026-08-15T07:00:00+08:00"
    )
    assert recurrence["range"]["numberOfOccurrences"] == 2147483647


def test_build_graph_recurrence_reports_unsupported_freq_before_int32_bound():
    """An unsupported FREQ combined with an out-of-range INTERVAL must
    surface the more fundamental "unsupported FREQ" error first - fixing
    an out-of-range INTERVAL alone would still leave the caller with an
    unsupported rule, so reporting the Int32 bound first would send them
    on a second round-trip to find the real problem."""
    with pytest.raises(ValueError, match="unsupported recurrence pattern"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=HOURLY;INTERVAL=2147483648", "2026-08-15T07:00:00+08:00"
        )


def test_build_graph_recurrence_resolves_windows_style_timezone():
    """Graph commonly reports Windows-style timezone identifiers (e.g. for
    events created via Outlook desktop/web rather than this tool), which
    zoneinfo.ZoneInfo can't resolve directly (it only knows IANA names) -
    the CLDR Windows<->IANA mapping must translate it rather than raising
    for an otherwise valid, pre-existing event."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY", "2026-08-26T07:00:00", "Pacific Standard Time"
    )
    assert recurrence["range"]["recurrenceTimeZone"] == "Pacific Standard Time"


def test_build_graph_recurrence_resolves_a_previously_unmapped_windows_timezone():
    """Confirmed bug: the old hand-written map only covered ~18 common
    business timezones and left every other valid Windows zone name
    (e.g. "Aleutian Standard Time", a real, Microsoft-documented Graph
    timezone) failing here, blocking a real recurrence-only update for
    any event created in one of the ~120 unmapped zones."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY", "2026-08-26T07:00:00", "Aleutian Standard Time"
    )
    assert recurrence["range"]["recurrenceTimeZone"] == "Aleutian Standard Time"


def test_build_graph_recurrence_rejects_unmapped_timezone():
    with pytest.raises(ValueError, match="isn't a recognized IANA zone name"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=DAILY", "2026-08-26T07:00:00", "Not A Real Timezone"
        )


def test_build_graph_recurrence_rejects_unsupported_freq():
    """FREQ=HOURLY is valid RFC 5545 that dateutil accepts, but this
    connector has no Graph pattern type to translate it into."""
    with pytest.raises(ValueError, match="unsupported recurrence pattern"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=HOURLY", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_rejects_invalid_rrule():
    with pytest.raises(ValueError, match="invalid recurrence rule"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=FORTNIGHTLY", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_wraps_invalid_start_datetime():
    with pytest.raises(ValueError, match="invalid start_datetime: 'not-a-date'"):
        outlook_recurrence.build_graph_recurrence("FREQ=DAILY", "not-a-date")


def test_build_graph_recurrence_normalizes_offset_before_until_range_dates():
    """Normalizing both DTSTART and UNTIL into recurrenceTimeZone avoids
    producing an inverted range when the input offset names the next local
    calendar day."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY;UNTIL=20260825T230100Z",
        "2026-08-26T07:00:00+08:00",
    )

    assert recurrence["range"]["startDate"] == "2026-08-25"
    assert recurrence["range"]["endDate"] == "2026-08-25"


@pytest.mark.parametrize(
    ("start_datetime", "message"),
    [
        ("2026-03-08T02:30:00", "does not exist"),
        ("2026-11-01T01:30:00", "is ambiguous"),
    ],
)
def test_build_graph_recurrence_rejects_dst_transition_wall_times(
    start_datetime, message
):
    with pytest.raises(ValueError, match=message):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=DAILY", start_datetime, "America/Los_Angeles"
        )


def test_build_graph_recurrence_relative_monthly():
    """FREQ=MONTHLY;BYDAY=2TU ("the second Tuesday of every month") has no
    BYMONTHDAY equivalent but does map onto Graph's relativeMonthly type."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=MONTHLY;BYDAY=2TU", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"] == {
        "type": "relativeMonthly",
        "interval": 1,
        "daysOfWeek": ["tuesday"],
        "index": "second",
    }


def test_build_graph_recurrence_relative_monthly_last_weekday():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=MONTHLY;BYDAY=-1FR", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"] == {
        "type": "relativeMonthly",
        "interval": 1,
        "daysOfWeek": ["friday"],
        "index": "last",
    }


def test_build_graph_recurrence_relative_monthly_with_bysetpos_weekdays():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=MONTHLY;BYDAY=MO,TU,WE,TH,FR;BYSETPOS=1",
        "2026-08-11T07:00:00",
    )

    assert recurrence["pattern"] == {
        "type": "relativeMonthly",
        "interval": 1,
        "daysOfWeek": ["monday", "tuesday", "wednesday", "thursday", "friday"],
        "index": "first",
    }


def test_build_graph_recurrence_relative_yearly_with_bysetpos():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=YEARLY;BYMONTH=11;BYDAY=SA,SU;BYSETPOS=-1",
        "2026-08-11T07:00:00",
    )

    assert recurrence["pattern"] == {
        "type": "relativeYearly",
        "interval": 1,
        "daysOfWeek": ["saturday", "sunday"],
        "index": "last",
        "month": 11,
    }


@pytest.mark.parametrize(
    "rule",
    [
        "FREQ=MONTHLY;BYDAY=MO;BYSETPOS=1,2",
        "FREQ=MONTHLY;BYDAY=2MO;BYSETPOS=1",
        "FREQ=MONTHLY;BYSETPOS=1",
        "FREQ=MONTHLY;BYDAY=MO;BYSETPOS=5",
    ],
)
def test_build_graph_recurrence_rejects_unrepresentable_bysetpos(rule):
    with pytest.raises(ValueError, match="BYSETPOS"):
        outlook_recurrence.build_graph_recurrence(rule, "2026-08-11T07:00:00")


def test_build_graph_recurrence_relative_yearly():
    recurrence = outlook_recurrence.build_graph_recurrence(
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
        outlook_recurrence.build_graph_recurrence(
            "FREQ=MONTHLY;BYDAY=2TU,3WE", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_relative_pattern_rejects_same_ordinal_multi_day():
    """Confirmed bug: RFC 5545's BYDAY=2TU,2WE means "the second Tuesday
    AND the second Wednesday" (two independent occurrences per month,
    confirmed against dateutil's rrule), but Microsoft's own
    recurrencePattern docs state that a relative pattern with more than
    one daysOfWeek value "falls on the first day that satisfies the
    pattern" - a single occurrence. Silently sending this to Graph would
    produce a materially narrower series with no error raised anywhere,
    so it must be rejected instead."""
    with pytest.raises(ValueError, match="more than one distinct weekday"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=MONTHLY;BYDAY=2TU,2WE", "2026-08-11T07:00:00+08:00"
        )
    with pytest.raises(ValueError, match="more than one distinct weekday"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=YEARLY;BYMONTH=11;BYDAY=2TU,2WE", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_relative_pattern_allows_duplicate_same_day():
    """A same-day duplicate (e.g. "2TU,2TU") isn't a multi-day pattern
    once deduped - must still be accepted, not conflated with the
    multi-day rejection above."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=MONTHLY;BYDAY=2TU,2TU", "2026-08-11T07:00:00+08:00"
    )
    assert recurrence["pattern"]["daysOfWeek"] == ["tuesday"]


def test_build_graph_recurrence_relative_pattern_rejects_missing_ordinal():
    """A plain (non-numbered) BYDAY on MONTHLY means something different
    under RFC 5545 (every such weekday in the month) - it must not be
    silently treated as a relative single-occurrence pattern."""
    with pytest.raises(ValueError, match="no numeric ordinal"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=MONTHLY;BYDAY=TU", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_relative_pattern_rejects_unsupported_ordinal():
    with pytest.raises(ValueError, match="has no Outlook equivalent"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=MONTHLY;BYDAY=5TU", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_daily_byday_maps_to_equivalent_weekly_pattern():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY;BYDAY=MO,WE,FR", "2026-08-11T07:00:00"
    )

    assert recurrence["pattern"] == {
        "type": "weekly",
        "interval": 1,
        "daysOfWeek": ["monday", "wednesday", "friday"],
        "firstDayOfWeek": "monday",
    }


def test_build_graph_recurrence_rejects_daily_byday_with_larger_interval():
    with pytest.raises(ValueError, match="only equivalent.*INTERVAL=1"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=DAILY;INTERVAL=2;BYDAY=MO,WE,FR", "2026-08-11T07:00:00"
        )


def test_build_graph_recurrence_rejects_leftover_bysetpos():
    with pytest.raises(ValueError, match="BYSETPOS"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=WEEKLY;BYDAY=MO;BYSETPOS=1", "2026-08-11T07:00:00+08:00"
        )


def test_build_graph_recurrence_weekly_dedupes_byday():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=WEEKLY;BYDAY=MO,MO,TU", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"]["daysOfWeek"] == ["monday", "tuesday"]


def test_build_graph_recurrence_weekly_defaults_first_day_of_week_to_monday():
    """RFC 5545 defaults WKST to Monday when omitted, but Graph's own
    firstDayOfWeek default is Sunday - the RRULE's own default must be
    stamped explicitly rather than picking up Graph's different one."""
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=WEEKLY;BYDAY=MO,TU", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"]["firstDayOfWeek"] == "monday"


def test_build_graph_recurrence_derived_weekday_stamps_first_day_of_week():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=WEEKLY", "2026-08-11T07:00:00"
    )

    assert recurrence["pattern"] == {
        "type": "weekly",
        "interval": 1,
        "daysOfWeek": ["tuesday"],
        "firstDayOfWeek": "monday",
    }


def test_build_graph_recurrence_weekly_honors_explicit_wkst():
    recurrence = outlook_recurrence.build_graph_recurrence(
        "FREQ=WEEKLY;BYDAY=MO,TU;WKST=SU", "2026-08-11T07:00:00+08:00"
    )

    assert recurrence["pattern"]["firstDayOfWeek"] == "sunday"


def test_build_graph_recurrence_accepts_date_until_for_all_day_event():
    result = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY;UNTIL=20260911",
        "2026-08-26T00:00:00",
        "Pacific Standard Time",
        is_all_day=True,
    )

    assert result == {
        "pattern": {"type": "daily", "interval": 1},
        "range": {
            "type": "endDate",
            "startDate": "2026-08-26",
            "endDate": "2026-09-11",
            "recurrenceTimeZone": "Pacific Standard Time",
        },
    }


def test_build_graph_recurrence_all_day_keeps_written_date_across_timezones():
    result = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY;COUNT=2",
        "2026-08-26T00:00:00+08:00",
        "America/Los_Angeles",
        is_all_day=True,
    )

    assert result["range"]["startDate"] == "2026-08-26"


def test_build_graph_recurrence_all_day_skips_dst_wall_time_validation():
    result = outlook_recurrence.build_graph_recurrence(
        "FREQ=DAILY",
        "2026-03-08T00:00:00",
        "America/Havana",
        is_all_day=True,
    )

    assert result["range"]["startDate"] == "2026-03-08"


@pytest.mark.parametrize(
    ("rule", "expected_pattern", "range_type"),
    [
        (
            "FREQ=WEEKLY;BYDAY=MO;COUNT=3",
            {"type": "weekly", "daysOfWeek": ["monday"]},
            "numbered",
        ),
        (
            "FREQ=MONTHLY;BYMONTHDAY=15",
            {"type": "absoluteMonthly", "dayOfMonth": 15},
            "noEnd",
        ),
        (
            "FREQ=YEARLY;BYMONTH=8;BYMONTHDAY=15",
            {"type": "absoluteYearly", "month": 8, "dayOfMonth": 15},
            "noEnd",
        ),
    ],
)
def test_build_graph_recurrence_all_day_patterns(rule, expected_pattern, range_type):
    result = outlook_recurrence.build_graph_recurrence(
        rule, "2026-08-15", "Asia/Shanghai", is_all_day=True
    )

    for key, value in expected_pattern.items():
        assert result["pattern"][key] == value
    assert result["pattern"]["interval"] == 1
    assert result["range"]["type"] == range_type


def test_build_graph_recurrence_rejects_datetime_until_for_all_day_event():
    with pytest.raises(ValueError, match="same DATE or DATE-TIME value type"):
        outlook_recurrence.build_graph_recurrence(
            "FREQ=DAILY;UNTIL=20260911T235959Z",
            "2026-08-26T00:00:00",
            is_all_day=True,
        )
