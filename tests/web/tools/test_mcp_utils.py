import json
import os
import re
from pathlib import Path

import pytest
import requests

from xagent.web.tools.mcp import utils


def test_naive_day_bounds_accepts_a_z_suffixed_datetime():
    assert utils.naive_day_bounds("2026-08-27T23:30:00Z") == (
        "2026-08-27T00:00:00",
        "2026-08-28T00:00:00",
    )


def test_naive_day_bounds_converts_an_instant_before_selecting_the_day():
    assert utils.naive_day_bounds("2026-08-27T20:00:00Z", "Asia/Singapore") == (
        "2026-08-28T00:00:00",
        "2026-08-29T00:00:00",
    )


def test_require_clean_identifier_rejects_empty_and_whitespace():
    with pytest.raises(ValueError, match="record_id"):
        utils.require_clean_identifier("", "record_id")
    with pytest.raises(ValueError, match="record_id"):
        utils.require_clean_identifier(" 001xx ", "record_id")
    assert utils.require_clean_identifier("001xx", "record_id") == "001xx"


def test_require_clean_identifier_rejects_non_string():
    """A truthy non-str (e.g. an int) previously slipped past `not value`
    and crashed on `.strip()` with a raw AttributeError instead of a clean
    ValueError."""
    with pytest.raises(ValueError, match="record_id"):
        utils.require_clean_identifier(12345, "record_id")


def test_require_clean_text_rejects_empty_and_whitespace():
    with pytest.raises(ValueError, match="summary"):
        utils.require_clean_text("", "summary")
    with pytest.raises(ValueError, match="summary"):
        utils.require_clean_text("  padded  ", "summary")
    assert utils.require_clean_text("Fix the bug", "summary") == "Fix the bug"


def test_require_clean_text_rejects_non_string():
    with pytest.raises(ValueError, match="summary"):
        utils.require_clean_text(12345, "summary")


def test_require_clean_text_message_does_not_call_the_field_an_id():
    # The whole point of this helper (vs. require_clean_identifier) is a
    # message phrased for a human-facing field like a title or name, not an
    # id -- regressing back to "id" wording defeats that.
    with pytest.raises(ValueError) as excinfo:
        utils.require_clean_text("", "summary")
    assert "id" not in str(excinfo.value)


def test_url_path_id_percent_encodes_reserved_characters():
    # A literal ".." blocklist misses "/" and "?", which redirect the
    # request to a different endpoint or inject query params without ever
    # containing "..". Percent-encoding closes off all of them at once.
    assert utils.url_path_id("Account/001abc", "sobject_type") == ("Account%2F001abc")
    assert utils.url_path_id("001x?fields=Id", "record_id") == ("001x%3Ffields%3DId")
    with pytest.raises(ValueError):
        utils.url_path_id("", "record_id")


def test_url_path_id_rejects_exact_dot_segments():
    # "." and ".." are always-unreserved characters -- quote() never
    # touches them, and requests/urllib3 collapse dot-segments out of the
    # final URL before sending it, so percent-encoding alone can't close
    # this off the way it does for "/" and "?".
    with pytest.raises(ValueError, match="record_id"):
        utils.url_path_id("..", "record_id")
    with pytest.raises(ValueError, match="record_id"):
        utils.url_path_id(".", "record_id")


@pytest.mark.parametrize(
    "limit,expected",
    [
        (50, 50),  # within range, passed through unchanged
        (1, 1),  # lower boundary, passed through unchanged
        (200, 200),  # exactly max_limit, passed through unchanged
        (201, 200),  # just above max_limit, clamped down
        (10**9, 200),  # extreme, clamped down the same as a mild overage
        (0, 1),  # zero would slice to an empty page forever -- clamped up
        (-1, 1),  # mild negative, clamped up
        (-(10**9), 1),  # extreme negative, clamped up the same as mild
    ],
)
def test_clamp_limit_boundaries(limit, expected):
    assert utils.clamp_limit(limit, max_limit=200) == expected


@pytest.mark.parametrize(
    "offset,expected",
    [
        (0, 0),
        (5, 5),
        (-1, 0),  # mild negative -- would slice from the end unclamped
        (-(10**9), 0),  # extreme negative, clamped the same as mild
    ],
)
def test_clamp_offset_boundaries(offset, expected):
    assert utils.clamp_offset(offset) == expected


_TEST_URL_ID_PATTERN = re.compile(r"/d/([a-zA-Z0-9_-]+)")


def test_resolve_id_from_url_extracts_id_from_matching_url():
    assert (
        utils.resolve_id_from_url(
            "https://docs.google.com/document/d/abc123/edit",
            _TEST_URL_ID_PATTERN,
            "document_id",
        )
        == "abc123"
    )


def test_resolve_id_from_url_passes_through_bare_id():
    assert (
        utils.resolve_id_from_url(" abc123 ", _TEST_URL_ID_PATTERN, "document_id")
        == "abc123"
    )


def test_resolve_id_from_url_rejects_non_string():
    """Mirrors require_clean_identifier's fix: pattern.search() would
    otherwise raise a raw TypeError on a non-string value instead of a
    clean ValueError."""
    with pytest.raises(ValueError, match="document_id"):
        utils.resolve_id_from_url(12345, _TEST_URL_ID_PATTERN, "document_id")


def test_parse_rrule_extracts_components():
    parts = utils.parse_rrule(
        "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z",
        "2026-08-26T07:00:00+08:00",
    )
    assert parts == {
        "FREQ": "WEEKLY",
        "BYDAY": "MO,TU,WE,TH,FR",
        "UNTIL": "20260911T235959Z",
    }


def test_parse_rrule_accepts_and_strips_rrule_prefix():
    parts = utils.parse_rrule("RRULE:FREQ=DAILY;COUNT=5", "2026-08-26T07:00:00+08:00")
    assert parts == {"FREQ": "DAILY", "COUNT": "5"}


def test_ensure_rrule_prefix_adds_prefix_and_uppercases():
    assert utils.ensure_rrule_prefix("FREQ=DAILY;COUNT=5") == "RRULE:FREQ=DAILY;COUNT=5"


def test_ensure_rrule_prefix_keeps_existing_prefix_and_uppercases():
    assert (
        utils.ensure_rrule_prefix("rrule:freq=weekly;byday=mo,tu")
        == "RRULE:FREQ=WEEKLY;BYDAY=MO,TU"
    )


def test_ensure_rrule_prefix_rejects_embedded_newlines():
    with pytest.raises(ValueError, match="embedded newlines"):
        utils.ensure_rrule_prefix("FREQ=DAILY\nEXDATE:20260902")


def test_is_bare_date_accepts_a_bare_date():
    assert utils.is_bare_date("2026-08-26")


def test_is_bare_date_rejects_a_datetime():
    assert not utils.is_bare_date("2026-08-26T07:00:00")


def test_is_bare_date_rejects_a_space_separated_datetime():
    assert not utils.is_bare_date("2026-08-26 07:00:00")


def test_is_bare_date_accepts_surrounding_whitespace():
    assert utils.is_bare_date("  2026-08-26  ")


@pytest.mark.parametrize(
    "value",
    ["2026-2-3", "2026-02-30", "2026-13-01", "0000-01-01"],
)
def test_is_bare_date_rejects_noncanonical_or_impossible_dates(value):
    assert not utils.is_bare_date(value)


def test_is_bare_date_rejects_non_ascii_digit_lookalikes():
    """Confirmed bug: str.isdigit() also accepts non-ASCII digit
    lookalikes (superscript, Thai, fullwidth, ...), unlike the ASCII-only
    _DIGITS_ONLY_RE this file already uses elsewhere for the identical
    risk (see parse_rrule's INTERVAL/COUNT validation). A misclassified
    value here reaches a Google Calendar request body's "date" field with
    no further local validation, unlike parse_rrule's own path."""
    assert not utils.is_bare_date("²⁰²⁶-08-26")
    assert not utils.is_bare_date("๒๐๒๖-08-26")


def test_ensure_rrule_prefix_normalizes_lowercase_rrule():
    """RFC 5545's RRULE grammar has no case-sensitive free-text values, so
    a fully lowercase rule (which parse_rrule's dateutil-backed validation
    tolerates) must still reach the calendar API canonicalized to
    uppercase - not sent verbatim in whatever case an LLM happened to
    produce."""
    assert (
        utils.ensure_rrule_prefix(
            "freq=weekly;byday=mo,tu,we,th,fr;until=20260911t235959z"
        )
        == "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20260911T235959Z"
    )


def test_parse_rrule_rejects_empty_string():
    with pytest.raises(ValueError, match="must not be empty"):
        utils.parse_rrule("", "2026-08-26T07:00:00+08:00")
    with pytest.raises(ValueError, match="must not be empty"):
        utils.parse_rrule("RRULE:", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_missing_freq():
    with pytest.raises(ValueError, match="FREQ"):
        utils.parse_rrule("BYDAY=MO,TU", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_malformed_component():
    with pytest.raises(ValueError, match="invalid recurrence rule component"):
        utils.parse_rrule("FREQ=DAILY;BOGUS", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_duplicate_keys():
    """A duplicate key (e.g. FREQ specified twice) would otherwise
    silently keep only the last occurrence in the returned dict for local
    validation, while the raw text - still containing BOTH occurrences -
    reaches Google's API close to verbatim, where its behavior is
    unspecified rather than matching whatever this function validated."""
    with pytest.raises(ValueError, match="FREQ is specified more than once"):
        utils.parse_rrule("FREQ=DAILY;FREQ=WEEKLY", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_unparseable_rule():
    """A syntactically plausible but semantically invalid rule (an unknown
    FREQ value) must be rejected, not silently accepted as valid RFC 5545
    - this is exactly the class of bug the connector shipped with before:
    a recurrence rule that only ever looks valid, never actually works."""
    with pytest.raises(ValueError, match="invalid recurrence rule"):
        utils.parse_rrule("FREQ=FORTNIGHTLY", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_bad_dtstart():
    with pytest.raises(ValueError, match="invalid start time"):
        utils.parse_rrule("FREQ=DAILY", "not-a-date")


def test_parse_rrule_rejects_an_extended_iso8601_until_with_a_clean_message():
    """Confirmed bug: dateutil's own isoparse is lenient enough to accept
    ISO8601's extended form for UNTIL (dashes/colons, e.g.
    "2026-09-11T23:59:59+08:00"), which RFC 5545 never permits - UNTIL
    must be RFC 5545's basic form. dateutil.rrule.rrulestr's own RFC 5545
    line parser chokes on the extended form with a raw, uninformative
    "too many values to unpack" instead of this function's own clean
    error - must be caught here directly."""
    with pytest.raises(ValueError, match="must be RFC 5545's basic form"):
        utils.parse_rrule(
            "FREQ=DAILY;UNTIL=2026-09-11T23:59:59+08:00",
            "2026-09-01T07:00:00+08:00",
        )


def test_parse_rrule_accepts_a_floating_local_time_until():
    """Confirmed bug: the previous fix for the extended-ISO8601 UNTIL
    problem was too strict - RFC 5545 permits UNTIL as a floating "DATE
    WITH LOCAL TIME" (no "Z") whenever DTSTART is also floating local
    time, not only the aware "DATE WITH UTC TIME" form. This combination
    was wrongly rejected with no test catching the regression."""
    parts = utils.parse_rrule("FREQ=DAILY;UNTIL=20260911T235959", "2026-08-26T07:00:00")
    assert parts["UNTIL"] == "20260911T235959"


def test_parse_rrule_uppercases_component_values_not_just_keys():
    """Confirmed bug: only the FREQ/UNTIL/etc. *keys* were uppercased, not
    their values - a lowercase "freq=daily" produced {"FREQ": "daily"}.
    RFC 5545's RRULE grammar has no case-sensitive free-text values (the
    same reasoning ensure_rrule_prefix already applies to the whole rule
    text), and this returned dict is what a future Outlook translator
    would key lookups against."""
    parts = utils.parse_rrule("freq=daily;count=3", "2026-08-26T07:00:00")
    assert parts == {"FREQ": "DAILY", "COUNT": "3"}


def test_parse_rrule_accepts_a_datetime_object_directly():
    """A caller that already has a datetime (e.g. after localizing a naive
    Outlook start time) shouldn't need to format it back into a string
    just to have it reparsed here."""
    from datetime import datetime, timezone

    parts = utils.parse_rrule(
        "FREQ=DAILY;UNTIL=20260911T235959Z",
        datetime(2026, 8, 26, 7, 0, 0, tzinfo=timezone.utc),
    )
    assert parts == {"FREQ": "DAILY", "UNTIL": "20260911T235959Z"}


def test_parse_rrule_rejects_until_before_dtstart():
    """dateutil's own rrulestr validation does NOT catch this - an UNTIL
    before dtstart parses fine and just silently yields zero occurrences,
    which would otherwise report status=success for a "recurring" event
    that never actually recurs. This must be checked explicitly."""
    from datetime import datetime, timezone

    with pytest.raises(ValueError, match="before the start time"):
        utils.parse_rrule(
            "FREQ=DAILY;UNTIL=20260101T000000Z",
            datetime(2026, 6, 1, tzinfo=timezone.utc),
        )


def test_parse_rrule_allows_until_after_dtstart():
    from datetime import datetime, timezone

    parts = utils.parse_rrule(
        "FREQ=DAILY;UNTIL=20260601T000000Z",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert parts["UNTIL"] == "20260601T000000Z"


def test_parse_rrule_naive_dtstart_with_utc_until_is_rejected_by_dateutil_itself():
    """A naive dtstart combined with a "Z"-suffixed (UTC/aware) UNTIL is
    already rejected by dateutil's own rrulestr validation before this
    function's explicit UNTIL-vs-dtstart check ever runs - documented here
    so the anchor.tzinfo guard in that check (added for exactly this kind
    of aware/naive mismatch) isn't mistaken for dead code: dateutil catches
    this shape first, with its own message."""
    from datetime import datetime

    with pytest.raises(ValueError, match="invalid recurrence rule"):
        utils.parse_rrule(
            "FREQ=DAILY;UNTIL=20260101T000000Z",
            datetime(2026, 6, 1),
        )


def test_parse_rrule_all_day_bare_date_until_is_accepted():
    """Confirmed bug: a bare (no-time) UNTIL is the RFC 5545-correct value
    type for a DATE (all-day) DTSTART, but localizing a naive anchor
    whenever `timezone` was given - added to fix the "Z"-suffixed-UNTIL
    case above - broke this opposite case by forcing the anchor aware
    while UNTIL stayed naive, a new mismatch. The anchor must only be
    localized when UNTIL itself is aware."""
    parts = utils.parse_rrule(
        "FREQ=DAILY;UNTIL=20260911", "2026-08-26", timezone="Asia/Shanghai"
    )
    assert parts["UNTIL"] == "20260911"


def test_parse_rrule_all_day_bare_date_until_before_start_is_still_rejected():
    """The before-start guard must still catch this now-naive-vs-naive
    comparison, not just the aware-vs-aware case."""
    with pytest.raises(ValueError, match="before the start time"):
        utils.parse_rrule(
            "FREQ=DAILY;UNTIL=20260101", "2026-08-26", timezone="Asia/Shanghai"
        )


@pytest.mark.parametrize(
    "until",
    ["20260911T235959", "20260911T235959Z"],
)
def test_parse_rrule_rejects_datetime_until_for_all_day_start(until):
    with pytest.raises(ValueError, match="same DATE or DATE-TIME value type"):
        utils.parse_rrule(
            f"FREQ=DAILY;UNTIL={until}",
            "2026-08-26",
            timezone="Asia/Shanghai",
        )


def test_parse_rrule_rejects_date_until_for_floating_timed_start():
    with pytest.raises(ValueError, match="same DATE or DATE-TIME value type"):
        utils.parse_rrule(
            "FREQ=DAILY;UNTIL=20260911",
            "2026-08-26T07:00:00",
        )


@pytest.mark.parametrize("part", ["BYSECOND=10", "BYMINUTE=30", "BYHOUR=9"])
def test_parse_rrule_rejects_time_parts_for_all_day_start(part):
    with pytest.raises(ValueError, match="all-day DATE start"):
        utils.parse_rrule(f"FREQ=DAILY;{part};COUNT=3", "2026-08-26")


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ("FREQ=DAILY;BYDAY=1MO;COUNT=3", "numeric BYDAY"),
        ("FREQ=YEARLY;BYWEEKNO=1;BYDAY=1MO;COUNT=3", "numeric BYDAY"),
        ("FREQ=WEEKLY;BYMONTHDAY=1;COUNT=3", "BYMONTHDAY"),
        ("FREQ=DAILY;BYYEARDAY=1;COUNT=3", "BYYEARDAY"),
        ("FREQ=MONTHLY;BYWEEKNO=1;COUNT=3", "BYWEEKNO"),
        ("FREQ=MONTHLY;BYSETPOS=1;COUNT=3", "BYSETPOS"),
    ],
)
def test_parse_rrule_rejects_invalid_frequency_combinations(rule, message):
    with pytest.raises(ValueError, match=message):
        utils.parse_rrule(rule, "2026-08-26T07:00:00")


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (
            "FREQ=MONTHLY;BYDAY=1MO;COUNT=3",
            {"FREQ": "MONTHLY", "BYDAY": "1MO", "COUNT": "3"},
        ),
        (
            "FREQ=YEARLY;BYWEEKNO=1;BYDAY=MO;COUNT=3",
            {"FREQ": "YEARLY", "BYWEEKNO": "1", "BYDAY": "MO", "COUNT": "3"},
        ),
        (
            "FREQ=YEARLY;BYYEARDAY=1;COUNT=3",
            {"FREQ": "YEARLY", "BYYEARDAY": "1", "COUNT": "3"},
        ),
        (
            "FREQ=MONTHLY;BYMONTHDAY=1;BYSETPOS=1;COUNT=3",
            {
                "FREQ": "MONTHLY",
                "BYMONTHDAY": "1",
                "BYSETPOS": "1",
                "COUNT": "3",
            },
        ),
    ],
)
def test_parse_rrule_accepts_valid_frequency_combinations(rule, expected):
    assert utils.parse_rrule(rule, "2026-08-26T07:00:00") == expected


def test_parse_rrule_strips_string_dtstart_before_parsing():
    assert utils.parse_rrule("FREQ=DAILY;COUNT=3", "  2026-08-26T07:00:00  ") == {
        "FREQ": "DAILY",
        "COUNT": "3",
    }


def test_parse_rrule_reports_invalid_byday_without_calling_it_numeric():
    with pytest.raises(ValueError, match="invalid recurrence rule") as exc_info:
        utils.parse_rrule("FREQ=FORTNIGHTLY;BYDAY=MOO", "2026-08-26T07:00:00")
    assert "numeric BYDAY" not in str(exc_info.value)


def test_parse_rrule_rejects_space_separated_dtstart_paired_with_bare_date_until():
    """Confirmed bug: RFC3339 permits a space in place of "T" as the
    date/time separator, so a check that only looked for the absence of
    "t" misclassified a timed dtstart like "2026-08-26 07:00:00" as an
    all-day DATE. That let it silently pair with a floating (bare-date)
    UNTIL instead of being rejected as the DATE-TIME-dtstart/floating-
    UNTIL mismatch it actually is."""
    with pytest.raises(ValueError, match="invalid recurrence rule"):
        utils.parse_rrule(
            "FREQ=DAILY;UNTIL=20260911",
            "2026-08-26 07:00:00",
            timezone="Asia/Shanghai",
        )


def test_parse_rrule_timezone_localizes_naive_dtstart_for_utc_until():
    """Confirmed bug: unlike the previous test (no timezone given), passing
    a resolvable IANA timezone must localize a naive dtstart so it can be
    compared against a "Z"-suffixed UNTIL instead of dateutil rejecting the
    aware/naive mismatch."""
    parts = utils.parse_rrule(
        "FREQ=DAILY;UNTIL=20260911T235959Z",
        "2026-08-26T07:00:00",
        timezone="Asia/Shanghai",
    )
    assert parts["UNTIL"] == "20260911T235959Z"


def test_parse_rrule_timezone_still_rejects_until_before_dtstart():
    with pytest.raises(ValueError, match="before the start time"):
        utils.parse_rrule(
            "FREQ=DAILY;UNTIL=20260101T000000Z",
            "2026-08-26T07:00:00",
            timezone="Asia/Shanghai",
        )


def test_parse_rrule_rejects_unknown_timezone():
    with pytest.raises(ValueError, match="recognized IANA zone name"):
        utils.parse_rrule(
            "FREQ=DAILY;UNTIL=20260911T235959Z",
            "2026-08-26T07:00:00",
            timezone="Not/ARealZone",
        )


def test_parse_rrule_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="INTERVAL must be a positive integer"):
        utils.parse_rrule("FREQ=DAILY;INTERVAL=0", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_non_integer_interval():
    with pytest.raises(ValueError, match="INTERVAL must be an integer"):
        utils.parse_rrule("FREQ=DAILY;INTERVAL=abc", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_non_positive_count():
    with pytest.raises(ValueError, match="COUNT must be a positive integer"):
        utils.parse_rrule("FREQ=DAILY;COUNT=0", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_non_integer_count():
    with pytest.raises(ValueError, match="COUNT must be an integer"):
        utils.parse_rrule("FREQ=DAILY;COUNT=abc", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_signed_interval():
    """RFC 5545 defines INTERVAL as `1*DIGIT` - plain unsigned digits only.
    Python's int() is more permissive than that grammar (accepts a leading
    "+"/"-"), and the raw string (not the parsed int) is what reaches
    Google's API almost verbatim - so a value int() accepts but RFC 5545
    doesn't must still be rejected here, not just range-checked once
    parsed."""
    with pytest.raises(ValueError, match="INTERVAL must be an integer"):
        utils.parse_rrule("FREQ=DAILY;INTERVAL=+2", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_rejects_underscore_separated_count():
    """Python's int() accepts PEP 515 "_" digit separators ("1_0" == 10),
    which isn't valid RFC 5545 RRULE syntax and would reach Google's API as
    literal, malformed text."""
    with pytest.raises(ValueError, match="COUNT must be an integer"):
        utils.parse_rrule("FREQ=DAILY;COUNT=1_0", "2026-08-26T07:00:00+08:00")


def test_parse_rrule_accepts_interval_with_leading_zeros():
    """Unlike a sign or underscore separator, leading zeros ARE valid under
    RFC 5545's `1*DIGIT` grammar and must still be accepted."""
    parts = utils.parse_rrule("FREQ=DAILY;INTERVAL=007", "2026-08-26T07:00:00+08:00")
    assert parts["INTERVAL"] == "007"


def test_parse_rrule_rejects_until_and_count_together():
    """RFC 5545 treats UNTIL and COUNT as mutually exclusive ways to end a
    series - dateutil's own rrulestr validation does not catch this
    (COUNT simply wins), so it must be checked explicitly."""
    with pytest.raises(ValueError, match="must not specify both UNTIL and COUNT"):
        utils.parse_rrule(
            "FREQ=DAILY;UNTIL=20260911T235959Z;COUNT=5",
            "2026-08-26T07:00:00+08:00",
        )


def test_parse_rrule_rejects_embedded_newline():
    """A newline could smuggle an extra RRULE/EXDATE/RDATE line into the
    calendar API request past this connector's single-RRULE validation."""
    with pytest.raises(ValueError, match="embedded newlines"):
        utils.parse_rrule(
            "FREQ=DAILY\nEXDATE:20260101T000000Z",
            "2026-08-26T07:00:00+08:00",
        )


def test_resolve_zoneinfo_returns_zoneinfo_for_valid_iana_name():
    from zoneinfo import ZoneInfo

    assert utils.resolve_zoneinfo("Asia/Shanghai") == ZoneInfo("Asia/Shanghai")


@pytest.mark.parametrize(
    ("windows_name", "iana_name"),
    [
        ("China Standard Time", "Asia/Shanghai"),
        ("Central Asia Standard Time", "Asia/Bishkek"),
        ("E. Europe Standard Time", "Europe/Chisinau"),
        ("Mountain Standard Time (Mexico)", "America/Mazatlan"),
        ("Aleutian Standard Time", "America/Adak"),
        ("UTC-11", "Etc/GMT+11"),
        ("Yukon Standard Time", "America/Whitehorse"),
    ],
)
def test_resolve_zoneinfo_accepts_windows_timezone_names_when_enabled(
    windows_name, iana_name
):
    from zoneinfo import ZoneInfo

    assert utils.resolve_zoneinfo(windows_name, allow_windows_names=True) == ZoneInfo(
        iana_name
    )


def test_resolve_zoneinfo_rejects_windows_timezone_names_by_default():
    with pytest.raises(ValueError, match="recognized IANA zone name"):
        utils.resolve_zoneinfo("Eastern Standard Time")


def test_resolve_zoneinfo_rejects_unknown_timezone():
    with pytest.raises(ValueError, match="recognized IANA zone name"):
        utils.resolve_zoneinfo("Not/ARealZone")


def test_url_path_id_output_survives_requests_url_normalization():
    """Confirms the actual exploit this guards against: a naively
    interpolated ".." collapses the path via requests' own URL
    normalization to a completely different (still valid) endpoint."""
    prepared = requests.Request(
        "GET", "https://acme.my.salesforce.com/services/data/v59.0/sobjects/Account/.."
    ).prepare()
    assert (
        prepared.url == "https://acme.my.salesforce.com/services/data/v59.0/sobjects/"
    )

    with pytest.raises(ValueError):
        utils.url_path_id("..", "record_id")


_TEST_ALLOWED_DIRS_ENV_VAR = "XAGENT_TEST_FILE_ALLOWED_DIRS"


def test_allowed_dirs_from_env_falls_back_to_cwd_when_unset(monkeypatch):
    monkeypatch.delenv(_TEST_ALLOWED_DIRS_ENV_VAR, raising=False)
    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        Path.cwd().resolve()
    ]


def test_allowed_dirs_from_env_falls_back_to_cwd_when_blank(monkeypatch):
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, "   ")
    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        Path.cwd().resolve()
    ]


@pytest.mark.parametrize("raw_value", [",", " , ", ",,,"])
def test_allowed_dirs_from_env_denies_entryless_legacy_value(monkeypatch, raw_value):
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, raw_value)
    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == []


def test_allowed_dirs_from_env_parses_multiple_dirs_with_whitespace(
    monkeypatch, tmp_path
):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, f" {dir_a} , {dir_b} ")

    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        dir_a.resolve(),
        dir_b.resolve(),
    ]


def test_allowed_dirs_from_env_parses_json_paths_containing_commas(
    monkeypatch, tmp_path
):
    directory = tmp_path / "reports,final"
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, json.dumps([str(directory)]))

    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        directory.resolve()
    ]


@pytest.mark.parametrize("raw_value", ["[]", '[""]', '["   "]'])
def test_allowed_dirs_from_env_treats_empty_json_paths_as_deny_all(
    monkeypatch, raw_value
):
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, raw_value)

    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == []


@pytest.mark.parametrize("raw_value", ["[", '["ok", 42]'])
def test_allowed_dirs_from_env_rejects_invalid_json(monkeypatch, raw_value):
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, raw_value)

    with pytest.raises(ValueError, match="JSON array"):
        utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges")
def test_allowed_dirs_from_env_rejects_unresolvable_path(monkeypatch, tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, json.dumps([str(loop)]))

    with pytest.raises(ValueError, match="invalid path") as exc_info:
        utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR)
    assert str(loop) not in str(exc_info.value)


def test_allowed_dirs_from_env_expands_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, "~/workspace")

    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        (tmp_path / "workspace").resolve()
    ]


def test_datetime_key_for_comparison_truncates_seven_digit_fractional_seconds():
    """Outlook commonly reports 100-nanosecond (7-digit) fractional
    seconds, one more digit than a microsecond can hold - this must not
    depend on whichever CPython version happens to run it."""
    key = utils.datetime_key_for_comparison("2026-08-27T10:00:00.1234567")
    assert key == utils.datetime_key_for_comparison("2026-08-27T10:00:00.123456")


def test_datetime_key_for_comparison_treats_equal_instants_as_equal_across_offsets():
    z_form = utils.datetime_key_for_comparison("2026-08-27T02:00:00Z")
    offset_form = utils.datetime_key_for_comparison("2026-08-27T10:00:00+08:00")
    assert z_form == offset_form


def test_datetime_key_for_comparison_falls_back_to_raw_string_on_malformed_input():
    assert utils.datetime_key_for_comparison("not-a-date") == "not-a-date"


def test_datetime_key_for_comparison_passes_none_through():
    assert utils.datetime_key_for_comparison(None) is None


def test_normalize_addresses_drops_case_insensitive_duplicates():
    """Regression test: email addresses are case-insensitive, so the same
    person listed twice with different casing must not become two separate
    attendee entries downstream - keeps the first casing seen."""
    assert utils.normalize_addresses(
        ["Chelsea@Example.com", "chelsea@example.com", "new@example.com"]
    ) == ["Chelsea@Example.com", "new@example.com"]


def test_normalize_addresses_dedup_works_for_comma_separated_string_input():
    assert utils.normalize_addresses("a@x.com, A@X.com, b@x.com") == [
        "a@x.com",
        "b@x.com",
    ]


def test_attendees_were_given_treats_empty_string_as_not_provided():
    assert utils.attendees_were_given(None) is False
    assert utils.attendees_were_given("") is False


def test_attendees_were_given_treats_empty_list_as_provided():
    """An explicit [] is a deliberate "clear everyone" - distinct from
    not-provided, unlike an empty string."""
    assert utils.attendees_were_given([]) is True
    assert utils.attendees_were_given(["a@x.com"]) is True
    assert utils.attendees_were_given("a@x.com") is True


def test_attendees_to_add_filters_out_existing_and_normalizes():
    assert utils.attendees_to_add(
        ["Old@Example.com", "new@example.com"], {"OLD@EXAMPLE.COM"}
    ) == ["new@example.com"]


def test_attendees_to_add_returns_empty_for_not_provided_or_empty():
    assert utils.attendees_to_add(None, {"old@example.com"}) == []
    assert utils.attendees_to_add("", {"old@example.com"}) == []
    assert utils.attendees_to_add([], {"old@example.com"}) == []


def test_merge_scope_error_reraises_when_no_conflict_is_known():
    error = utils.InsufficientScopeError(
        "reconnect required", [], ["unchecked@example.com"]
    )

    with pytest.raises(utils.InsufficientScopeError) as raised:
        utils.merge_scope_error(error, [], [])

    assert raised.value is error


def test_merge_scope_error_preserves_confirmed_results():
    existing_conflict = {"calendar": "organizer"}
    error_conflict = {"calendar": "attendee@example.com"}
    error = utils.InsufficientScopeError(
        "reconnect required", [error_conflict], ["unchecked@example.com"]
    )

    conflicts, unchecked = utils.merge_scope_error(
        error, [existing_conflict], ["already-unchecked@example.com"]
    )

    assert conflicts == [existing_conflict, error_conflict]
    assert unchecked == [
        "already-unchecked@example.com",
        "unchecked@example.com",
    ]


def _key(value: str):
    return utils.datetime_key_for_comparison(value)


def test_window_delta_segments_empty_for_unchanged_or_shrunk_window():
    """A retained attendee's own busy block already covers everything
    inside the old window - a new window that's identical, or a strict
    subset of it, adds no territory worth checking them against."""
    old_start, old_end = (
        _key("2026-08-27T10:00:00+00:00"),
        _key("2026-08-27T10:30:00+00:00"),
    )
    assert utils.window_delta_segments(old_start, old_end, old_start, old_end) == []

    shrunk_start = _key("2026-08-27T10:05:00+00:00")
    shrunk_end = _key("2026-08-27T10:25:00+00:00")
    assert (
        utils.window_delta_segments(old_start, old_end, shrunk_start, shrunk_end) == []
    )


def test_window_delta_segments_returns_the_new_only_portion_of_a_partial_nudge():
    """10:00-10:30 nudged to 10:15-10:45 - only 10:30-10:45 is new
    territory a retained attendee could have a genuine conflict in;
    10:15-10:30 is still covered by their busy block for this event."""
    old_start, old_end = (
        _key("2026-08-27T10:00:00+00:00"),
        _key("2026-08-27T10:30:00+00:00"),
    )
    new_start, new_end = (
        _key("2026-08-27T10:15:00+00:00"),
        _key("2026-08-27T10:45:00+00:00"),
    )

    segments = utils.window_delta_segments(old_start, old_end, new_start, new_end)

    assert segments == [(old_end, new_end)]


def test_window_delta_segments_normalizes_iso_strings_internally():
    assert utils.window_delta_segments(
        "2026-08-27T10:00:00+00:00",
        "2026-08-27T10:30:00+00:00",
        "2026-08-27T10:15:00+00:00",
        "2026-08-27T10:45:00+00:00",
    ) == [
        (
            utils.datetime_key_for_comparison("2026-08-27T10:30:00+00:00"),
            utils.datetime_key_for_comparison("2026-08-27T10:45:00+00:00"),
        )
    ]


def test_window_delta_segments_returns_two_segments_when_widened_on_both_sides():
    """Extending a meeting earlier AND later in the same call creates new
    territory on both sides of the old window."""
    old_start, old_end = (
        _key("2026-08-27T10:00:00+00:00"),
        _key("2026-08-27T10:30:00+00:00"),
    )
    new_start, new_end = (
        _key("2026-08-27T09:45:00+00:00"),
        _key("2026-08-27T10:45:00+00:00"),
    )

    segments = utils.window_delta_segments(old_start, old_end, new_start, new_end)

    assert len(segments) == 2
    assert segments[0] == (new_start, old_start)
    assert segments[1] == (old_end, new_end)


def test_window_delta_segments_returns_the_whole_window_for_a_disjoint_move():
    """A move to somewhere with zero overlap with the old window means the
    entire new window is new territory - this test also confirms the
    move-by-15-minutes example from the actual reported bug is caught:
    the delta segment here is the same shape as the disjoint case."""
    old_start, old_end = (
        _key("2026-08-27T10:00:00+00:00"),
        _key("2026-08-27T10:30:00+00:00"),
    )
    new_start, new_end = (
        _key("2026-08-27T14:00:00+00:00"),
        _key("2026-08-27T14:30:00+00:00"),
    )

    assert utils.window_delta_segments(old_start, old_end, new_start, new_end) == [
        (new_start, new_end)
    ]


def test_window_delta_segments_is_conservative_on_unparseable_or_mismatched_keys():
    """ "Can't confirm the delta is smaller than the whole window" must
    never silently shrink what gets checked - falls back to the whole new
    window, both for a key that never parsed to a real datetime, and for
    two real datetimes that can't be compared (aware vs. naive raises
    TypeError on `<`)."""
    new_start, new_end = (
        _key("2026-08-27T10:00:00+00:00"),
        _key("2026-08-27T10:30:00+00:00"),
    )
    assert utils.window_delta_segments(
        "not-a-date", "also-not", new_start, new_end
    ) == [(new_start, new_end)]

    naive_start = _key("2026-08-27T09:00:00")
    naive_end = _key("2026-08-27T11:00:00")
    assert utils.window_delta_segments(naive_start, naive_end, new_start, new_end) == [
        (new_start, new_end)
    ]


def test_window_delta_segments_empty_when_new_window_itself_is_unparseable():
    """No valid new window at all means nothing meaningful to check,
    regardless of the old window."""
    old_start, old_end = (
        _key("2026-08-27T10:00:00+00:00"),
        _key("2026-08-27T10:30:00+00:00"),
    )
    assert (
        utils.window_delta_segments(old_start, old_end, "not-a-date", "also-not") == []
    )


def test_window_delta_segments_empty_when_only_one_of_the_new_window_parses():
    """Regression test: the guard is `new_start AND new_end` both being
    real datetimes - a MIXED pair (one parses, one doesn't) must still
    return [], not fall through and try to build a segment out of a
    half-valid window. A prior test only exercised BOTH sides failing
    together, which can't tell `and` apart from a mistakenly-broadened
    `or` in that guard - this fixture, with exactly one side invalid,
    can."""
    old_start, old_end = (
        _key("2026-08-27T10:00:00+00:00"),
        _key("2026-08-27T10:30:00+00:00"),
    )
    new_start = _key("2026-08-27T14:00:00+00:00")
    assert utils.window_delta_segments(old_start, old_end, new_start, "also-not") == []
    assert (
        utils.window_delta_segments(old_start, old_end, "not-a-date", new_start) == []
    )


def test_reject_reversed_window_raises_when_end_is_not_after_start():
    with pytest.raises(ValueError, match="must be after"):
        utils.reject_reversed_window(
            "2026-08-27T10:30:00+00:00", "2026-08-27T10:00:00+00:00"
        )
    with pytest.raises(ValueError, match="must be after"):
        utils.reject_reversed_window(
            "2026-08-27T10:00:00+00:00", "2026-08-27T10:00:00+00:00"
        )


def test_reject_reversed_window_allows_a_forward_window():
    utils.reject_reversed_window(
        "2026-08-27T10:00:00+00:00", "2026-08-27T10:30:00+00:00"
    )


def test_reject_reversed_window_is_permissive_on_unparseable_input():
    """Can't-tell must never read as "reject" - only a confirmed reversal
    should raise."""
    utils.reject_reversed_window("not-a-date", "also-not-a-date")


def test_reject_reversed_window_is_permissive_when_aware_and_naive_are_mixed():
    """Regression test: both sides parse to real `datetime` instances, but
    comparing an offset-aware one against a naive one with `<=` raises
    `TypeError` in Python - the `isinstance` check alone doesn't guard
    against this, only checking that both are `datetime` instances of the
    SAME awareness does. Must stay permissive here, not crash with a raw
    TypeError, matching `window_delta_segments`'s handling of the
    identical hazard."""
    utils.reject_reversed_window("2026-08-27T10:30:00", "2026-08-27T10:30:00Z")
    utils.reject_reversed_window("2026-08-27T10:30:00Z", "2026-08-27T10:00:00")


def test_require_offset_datetime_rejects_a_naive_value():
    with pytest.raises(ValueError, match="start_time"):
        utils.require_offset_datetime("2026-08-27T10:30:00", "start_time")


def test_require_offset_datetime_accepts_an_offset_or_z_suffixed_value():
    utils.require_offset_datetime("2026-08-27T10:30:00+08:00", "start_time")
    utils.require_offset_datetime("2026-08-27T10:30:00Z", "start_time")


def test_require_offset_datetime_is_permissive_on_unparseable_input():
    """A value that doesn't even parse is a different failure a caller
    will already hit downstream with its own clear error - not this
    function's job to preempt with a possibly-confusing offset-specific
    message."""
    utils.require_offset_datetime("not-a-date", "start_time")


def test_calendar_day_bounds_spans_a_short_day_across_a_dst_spring_forward():
    """2026-03-08 is when America/New_York springs forward (clocks skip
    02:00-03:00), so the calendar day is only 23 hours long. The end
    boundary must be constructed from the next local calendar date so the
    timezone can apply the new UTC offset; adding 24 elapsed hours would
    land at 01:00 on the following day and mis-widen the query window."""
    start, end = utils.calendar_day_bounds("2026-03-08", "America/New_York")
    assert start == "2026-03-08T00:00:00-05:00"
    assert end == "2026-03-09T00:00:00-04:00"


def test_calendar_day_bounds_spans_a_long_day_across_a_dst_fall_back():
    start, end = utils.calendar_day_bounds("2026-11-01", "America/New_York")
    assert start == "2026-11-01T00:00:00-04:00"
    assert end == "2026-11-02T00:00:00-05:00"


@pytest.mark.parametrize("days", [0, -1])
def test_calendar_day_bounds_rejects_non_positive_days(days):
    with pytest.raises(ValueError, match="positive"):
        utils.calendar_day_bounds("2026-08-27", "UTC", days=days)


@pytest.mark.parametrize("days", [0, -1])
def test_naive_day_bounds_rejects_non_positive_days(days):
    with pytest.raises(ValueError, match="positive"):
        utils.naive_day_bounds("2026-08-27", "UTC", days=days)


def test_resolve_zoneinfo_reports_missing_name_as_value_error():
    with pytest.raises(ValueError, match="recognized IANA"):
        utils.resolve_zoneinfo(None)  # type: ignore[arg-type]


def test_offset_datetime_string_attaches_the_zone_offset_to_a_naive_value():
    assert (
        utils.offset_datetime_string("2026-08-27T10:00:00", "Asia/Singapore")
        == "2026-08-27T10:00:00+08:00"
    )


def test_offset_datetime_string_rejects_input_that_already_carries_an_offset():
    """Regression test: no public tool parameter is documented as requiring
    a naive value, so a caller passing one with a trailing 'Z' or an
    explicit offset is a real, reachable mistake - not a theoretical one.
    `.replace(tzinfo=...)` doesn't convert an aware datetime, it just
    relabels the same clock digits under a different zone, silently
    shifting the real instant by however much the two offsets differ
    (e.g. "10:00:00Z" relabeled as Asia/Shanghai reads as 10:00 Shanghai
    time - actually 8 hours earlier). Must fail loudly instead."""
    with pytest.raises(ValueError, match="already carries a UTC offset"):
        utils.offset_datetime_string("2026-08-27T10:00:00Z", "Asia/Shanghai")
    with pytest.raises(ValueError, match="already carries a UTC offset"):
        utils.offset_datetime_string("2026-08-27T10:00:00+00:00", "Asia/Shanghai")


def test_offset_datetime_string_rejects_a_nonexistent_dst_gap_time():
    with pytest.raises(ValueError, match="does not exist.*daylight-saving"):
        utils.offset_datetime_string("2026-03-08T02:30:00", "America/Los_Angeles")


def test_conflict_response_uncapped_when_it_fits(monkeypatch):
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "50000")
    conflicts = [{"calendar": "organizer", "summary": "1:1", "start": "a", "end": "b"}]
    response = json.loads(
        utils.conflict_response(
            conflicts, [], "2026-08-27T10:00:00", "2026-08-27T10:30:00"
        )
    )
    assert response["conflicts"] == conflicts
    assert response["truncated"] is False


def test_conflict_response_caps_an_oversized_conflicts_list(monkeypatch):
    """Regression test: unlike every other response path in this module,
    conflict_response used to return an uncapped payload - a busy shared
    calendar or wide window can turn up far more overlapping events than
    fit the platform's output budget. Only `conflicts` (the field that
    can actually grow large) should be halved to fit; small fields like
    `hint`/`unchecked_attendees` must survive untouched."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "2000")
    conflicts = [
        {
            "calendar": f"person{i}@example.com",
            "summary": "Busy block " + "x" * 50,
            "start": "2026-08-27T10:00:00+00:00",
            "end": "2026-08-27T10:30:00+00:00",
        }
        for i in range(100)
    ]
    response = json.loads(
        utils.conflict_response(
            conflicts,
            ["unreachable@example.com"],
            "2026-08-27T10:00:00",
            "2026-08-27T10:30:00",
        )
    )

    assert response["status"] == "conflict"
    assert response["truncated"] is True
    assert len(response["conflicts"]) < len(conflicts)
    assert response["conflicts"] == conflicts[: len(response["conflicts"])]
    assert response["unchecked_attendees"] == ["unreachable@example.com"]
    assert len(json.dumps(response, ensure_ascii=False)) <= 2000


def test_conflict_response_caps_unchecked_attendees_when_conflicts_alone_is_not_enough(
    monkeypatch,
):
    """Regression test: a single real conflict combined with a huge
    unchecked_attendees list (e.g. every remaining attendee in a large
    invite batch, marked unchecked after a mid-batch scope error) can
    still exceed the output budget even after `conflicts` has been
    shrunk to nothing - unchecked_attendees must also be capped, not
    left untouched forever."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "2000")
    conflicts = [
        {
            "calendar": "organizer",
            "summary": "1:1 with Hazel",
            "start": "2026-08-27T10:00:00+00:00",
            "end": "2026-08-27T10:30:00+00:00",
        }
    ]
    unchecked_attendees = [f"person{i}@example.com" for i in range(200)]

    response = json.loads(
        utils.conflict_response(
            conflicts,
            unchecked_attendees,
            "2026-08-27T10:00:00",
            "2026-08-27T10:30:00",
        )
    )

    assert response["status"] == "conflict"
    assert response["truncated"] is True
    assert response["conflicts"] == conflicts
    assert len(response["unchecked_attendees"]) < len(unchecked_attendees)
    assert (
        response["unchecked_attendees"]
        == unchecked_attendees[: len(response["unchecked_attendees"])]
    )
    assert len(json.dumps(response, ensure_ascii=False)) <= 2000


def test_conflict_response_uses_valid_compact_json_below_fixed_envelope(
    monkeypatch,
):
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "300")
    response_text = utils.conflict_response(
        [
            {
                "calendar": "organizer",
                "summary": "Busy",
                "start": "2026-08-27T10:00:00+00:00",
                "end": "2026-08-27T10:30:00+00:00",
            }
        ],
        ["unreachable@example.com"],
        "2026-08-27T10:00:00+00:00",
        "2026-08-27T10:30:00+00:00",
    )

    response = json.loads(response_text)
    assert response["status"] == "conflict"
    assert response["truncated"] is True
    assert len(response_text) <= 300


# ---------------------------------------------------------------------------
# success_with_capped_dict - dict-collapse sharp edge and last-resort extras
# ---------------------------------------------------------------------------


def test_success_with_capped_dict_degrades_a_single_key_nested_dict_to_a_marker(
    monkeypatch,
):
    """Regression test: phase 1 shrinks a dict-valued field by halving its
    KEY COUNT, mirroring how a list halves its elements. That works for a
    list at any length, but floors to zero for a dict with only one key,
    collapsing the whole field to {} in a single step instead of degrading
    gradually. The field should become a small, non-empty truncation
    marker instead, so a caller can tell something was dropped rather than
    reading {} and being unable to distinguish "truncated" from "always
    empty"."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "200")

    raw = utils.success_with_capped_dict(
        "record", {"id": "rec1", "metrics": {"total": "x" * 5000}}
    )
    result = json.loads(raw)

    assert len(raw) <= 200
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["record"]["id"] == "rec1"
    assert result["record"]["metrics"] == {"truncated": True}


def test_success_with_capped_dict_degrades_a_two_key_nested_dict_to_a_marker(
    monkeypatch,
):
    """Regression test: a 2-key dict reaches the same floor as the 1-key
    case one step later -- halving its key count first drops to 1 key, and
    a *subsequent* halving of that 1-key remainder is what collapses it to
    {}. The field must still end up as a non-empty marker, not {}, once
    fully exhausted."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "150")

    raw = utils.success_with_capped_dict(
        "record",
        {"id": "rec1", "metrics": {"total": "x" * 3000, "avg": "y" * 3000}},
    )
    result = json.loads(raw)

    assert len(raw) <= 150
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["record"]["id"] == "rec1"
    assert result["record"]["metrics"] == {"truncated": True}


def test_success_with_capped_dict_skips_the_marker_when_it_would_grow_the_field(
    monkeypatch,
):
    """Regression test: phase 1 floors an exhausted dict field to {} first
    (never installing the marker directly), then only upgrades it to the
    {"truncated": true} marker (19 bytes) if the *actual rebuilt response*
    with that marker installed still fits. Here upgrading "tiny" would cost
    17 bytes over its floored {}, and "name"/"note" are still oversized
    enough at that point that there's no surplus to spend -- so the
    upgrade must be skipped, matching the pre-marker (base) result exactly:
    "name" and "note" still get dropped by phase 2 afterward, same as
    always. Checked end-to-end (not just via the size-comparison unit test
    below) so a regression that reintroduced eager marker installation --
    which could still land on the same final shape by chance if a later
    pass coincidentally downgraded it back -- would be caught here too."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "85")

    raw = utils.success_with_capped_dict(
        "record",
        {"tiny": {"a": 1}, "id": "r1", "name": "Bob", "note": "n" * 30},
    )
    result = json.loads(raw)

    assert len(raw) <= 85
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["record"] == {"tiny": {}, "id": "r1"}


def test_success_with_capped_dict_top_level_single_key_survives_phase_two(
    monkeypatch,
):
    """Regression test: phase 2's top-level key-drop halves the key COUNT
    the same way phase 1 does for a nested dict, which floors to zero once
    only one top-level key is left -- the exact gap `mixpanel.py`'s
    `_success_with_capped_list` had to work around locally, since it
    wraps a bare list as `{key: items}` before calling this function.
    Once down to a single top-level key that still doesn't fit on its
    own, phase 2 must stop rather than empty `data` to {} and lose a
    field (here, "id") that the final compact-data fallback could
    otherwise have recovered."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "70")

    # "notes" sorts first (dict insertion order) so phase 2's "keep the
    # first half of keys" step drops "id" before "notes" -- if phase 2 were
    # then allowed to empty the single remaining "notes" key down to {},
    # "id" would never be recovered even though it's small enough to fit.
    raw = utils.success_with_capped_dict("record", {"notes": "x" * 5000, "id": "rec1"})
    result = json.loads(raw)

    assert len(raw) <= 70
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["record"] == {"id": "rec1"}


def test_success_with_capped_dict_last_resort_degrades_extras_before_dropping_them(
    monkeypatch,
):
    """Regression test: once `data` itself is fully truncated, the last
    resort used to drop every `extra_fields` entry outright the moment the
    full-extras candidate didn't fit. A caller-registered extra field
    (e.g. a per-field truncation flag) should keep its key -- even
    degraded to a generic placeholder -- for as long as there's room,
    instead of the whole `extra_fields` dict vanishing in one step. Uses a
    lowercase `id` and a limit distinct from
    test_success_with_capped_dict_last_resort_fallback_keeps_capitalized_id
    in test_deputy_mcp.py, which pins a different contract (capitalized
    `Id` survives) via the same shared helper -- so the two tests don't
    duplicate the same call."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "80")

    raw = utils.success_with_capped_dict(
        "record",
        {"id": "REC1", "Notes": "x" * 5000},
        extra_fields={"note": "y" * 60},
    )
    result = json.loads(raw)

    assert len(raw) <= 80
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["record"] == {"id": "REC1"}
    assert "note" in result
    assert result["note"] is True


def test_success_with_capped_dict_last_resort_falls_back_to_dropping_extras(
    monkeypatch,
):
    """When even a fully-degraded extras dict (every value replaced by
    `True`) still doesn't fit, the last resort must still fall back to
    dropping extras entirely rather than getting stuck or erroring -- while
    still preserving the compact `Id` the drop-extras rungs exist to keep
    (asserted explicitly so a regression that lost the id along with the
    extras, e.g. by falling all the way to the id-less rung instead of the
    id-with-no-extras one, would be caught here)."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "80")

    raw = utils.success_with_capped_dict(
        "record",
        {"Id": 123, "Notes": "x" * 5000},
        extra_fields={"note_one": "y" * 60, "note_two": "z" * 60},
    )
    result = json.loads(raw)

    assert len(raw) <= 80
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["record"] == {"Id": 123}
    assert "note_one" not in result
    assert "note_two" not in result


def test_success_with_capped_dict_last_resort_degrades_the_larger_extra_first(
    monkeypatch,
):
    """Pins the "largest first" claim in the degradation-order docstring
    with two extras of genuinely different sizes (unlike the
    same-size-extras fallback test above, which can't distinguish "largest
    first" from any other tie-break). At a limit that fits the id plus one
    intact extra but not both intact, the larger extra ("big") must be the
    one degraded to `True` while the smaller ("small") stays intact -- a
    regression that degraded by iteration/insertion order instead of size,
    or picked the smallest first, would flip this."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "110")

    raw = utils.success_with_capped_dict(
        "record",
        {"Id": 123, "Notes": "x" * 5000},
        extra_fields={"big": "b" * 80, "small": "s" * 20},
    )
    result = json.loads(raw)

    assert len(raw) <= 110
    assert result["truncated"] is True
    assert result["record"] == {"Id": 123}
    assert result["big"] is True
    assert result["small"] == "s" * 20


def test_halve_dict_or_mark_halves_while_more_than_one_key_remains():
    """The shared per-step helper both success_with_capped_dict's phase 1
    and shopify.py's local tail loop delegate to: drops the trailing half of
    the keys and reports the field is not yet at its floor."""
    shrunk, floored = utils.halve_dict_or_mark({"a": 1, "b": 2, "c": 3, "d": 4})

    assert shrunk == {"a": 1, "b": 2}
    assert floored is False


def test_halve_dict_or_mark_uses_the_marker_at_the_floor_when_smaller():
    shrunk, floored = utils.halve_dict_or_mark({"total": "x" * 5000})

    assert shrunk == {"truncated": True}
    assert floored is True


def test_halve_dict_or_mark_falls_back_to_empty_when_the_marker_would_grow_it():
    """Regression test for the review finding that motivated gating the
    marker on size at all: a tiny value (here, {"a": 1}, 8 bytes) is
    smaller than the marker itself (19 bytes) -- installing the marker
    would grow the payload instead of shrinking it, so this must fall
    back to {} instead, and still report the floor was reached."""
    shrunk, floored = utils.halve_dict_or_mark({"a": 1})

    assert shrunk == {}
    assert floored is True


def test_halve_dict_or_mark_uses_the_marker_on_an_exact_size_tie():
    """When the marker and the value it would replace serialize to the
    exact same length, the marker must still win: installing it costs
    nothing extra here, so there's no reason to throw away the
    "truncated" signal for a byte-neutral swap. `{"num": 1234567890}` and
    `{"truncated": true}` both serialize to 19 bytes."""
    value = {"num": 1234567890}
    assert len(json.dumps(value, ensure_ascii=False)) == len(
        json.dumps({"truncated": True}, ensure_ascii=False)
    )

    shrunk, floored = utils.halve_dict_or_mark(value)

    assert shrunk == {"truncated": True}
    assert floored is True


_MARKER_OVERHEAD_RECORD = {
    "id": "r1",
    "a": {"k": "x" * 500},
    "b": {"k": "y" * 500},
    "c": {"k": "z" * 500},
    "meta": {"left": "L", "right": "R"},
    "name": "Bob",
}


def test_success_with_capped_dict_floors_exactly_like_base_before_any_marker(
    monkeypatch,
):
    """Regression test: several dict fields hitting the one-key floor in
    the same call each cost a marker's worth of bytes over {}. If markers
    were installed eagerly during phase 1 (before every field's final size
    is known), that overhead could push the response over budget while
    OTHER fields are still being shrunk -- evicting a nested value phase 1
    itself would otherwise have kept intact, exactly unlike the pre-marker
    (base) code. Checked directly against base's own output at this cap:
    three same-size fields (a, b, c) reach the floor, and base still has
    to halve the fourth ("meta") once (to {"left": "L"}) before everything
    fits -- matched here byte-for-byte, since phase 1 floors to {} exactly
    like base until every field's final size is known, before any marker
    is even considered."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "140")

    raw = utils.success_with_capped_dict("record", _MARKER_OVERHEAD_RECORD)
    result = json.loads(raw)

    assert len(raw) <= 140
    assert result["truncated"] is True
    assert set(result["record"]) == {"id", "a", "b", "c", "meta", "name"}
    assert result["record"]["id"] == "r1"
    assert result["record"]["name"] == "Bob"
    # Matches base exactly at this cap: "meta" still needs one halving
    # pass even without any marker in the picture.
    assert result["record"]["meta"] == {"left": "L"}


def test_success_with_capped_dict_upgrades_a_marker_only_from_leftover_budget(
    monkeypatch,
):
    """Regression test: with the same record as the limit=140 case above
    but 20 more bytes of budget, base's own halving of "meta" is no
    longer needed -- it fits fully intact ({"left": "L", "right": "R"})
    once a/b/c are {}. A regression that installs a/b/c's markers
    immediately (spending budget before "meta" is ever considered) would
    force "meta" through phase 1's own halving anyway, permanently losing
    real data that base, and this fixed version, both keep intact --
    markers should only ever be paid for out of budget left over *after*
    matching base, never budget base itself needed. The full key set and
    "id"/"name" are asserted here too (not just the fields the
    marker-upgrade pass touches), so a regression that dropped an
    unrelated top-level key specifically in this branch would also be
    caught."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "160")

    raw = utils.success_with_capped_dict("record", _MARKER_OVERHEAD_RECORD)
    result = json.loads(raw)

    assert len(raw) <= 160
    assert result["truncated"] is True
    assert set(result["record"]) == {"id", "a", "b", "c", "meta", "name"}
    assert result["record"]["id"] == "r1"
    assert result["record"]["name"] == "Bob"
    assert result["record"]["meta"] == {"left": "L", "right": "R"}
    # The 20 extra bytes of budget over the base-equivalent 140-byte state
    # are exactly enough for one marker upgrade (17 bytes) -- spent on "a",
    # the first field floored, rather than wasted or spent unsafely early.
    assert result["record"]["a"] == {"truncated": True}
    assert result["record"]["b"] == {}
    assert result["record"]["c"] == {}


def test_success_with_capped_dict_marks_a_one_key_dict_without_recursing_into_it(
    monkeypatch,
):
    """Phase 1 recurses one level, not further: a one-key dict whose sole
    value is a huge list is replaced by the marker wholesale rather than
    having that inner list halved. Pinned here so the behavior reads as
    deliberate now that the marker exists, not as an oversight."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "150")

    raw = utils.success_with_capped_dict(
        "record",
        {"id": "r1", "wrap": {"items": [{"k": "x" * 100} for _ in range(50)]}},
    )
    result = json.loads(raw)

    assert len(raw) <= 150
    assert result["record"]["id"] == "r1"
    assert result["record"]["wrap"] == {"truncated": True}


def test_success_with_capped_dict_last_resort_keeps_extras_that_fit_beside_an_empty_record(
    monkeypatch,
):
    """Regression test: the last-resort ladder went from "id + extras"
    straight to "id, no extras", never trying "empty record + extras". A
    long id that doesn't fit beside the extras therefore cost the extras
    entirely, even when they'd have fit beside {} -- which is how a
    calendar event's Meet link could vanish under truncation although it
    had room. The pre-PR code kept it (its phase 2 collapsed the record to
    {} with extras still attached), so this was a regression."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "66")

    raw = utils.success_with_capped_dict(
        "record",
        {"id": "LONGISHIDVALUE1234", "Notes": "x" * 5000},
        extra_fields={"f": True},
    )
    result = json.loads(raw)

    assert len(raw) <= 66
    assert result["truncated"] is True
    assert result["record"] == {}
    assert result["f"] is True


def test_success_with_capped_dict_last_resort_never_degrades_an_extra_that_would_grow(
    monkeypatch,
):
    """Regression test: degrading an already-tiny extra field (here, `1`,
    1 byte) to `True` (4 bytes) would grow the payload, so the ladder must
    skip that step -- and, since the extras fit beside an empty record,
    keep the field with its real value rather than a bloated `true`."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "70")

    raw = utils.success_with_capped_dict(
        "record",
        {"id": "rec1", "Notes": "x" * 5000},
        extra_fields={"tiny": 1},
    )
    result = json.loads(raw)

    assert len(raw) <= 70
    assert result["truncated"] is True
    assert result["record"] == {}
    assert result["tiny"] == 1


def test_success_with_capped_dict_last_resort_never_flips_a_boolean_extra(
    monkeypatch,
):
    """Regression test: the extras-degradation loop's size gate only
    excluded an already-`True` value, not a real `False` -- since
    json.dumps(False) (5 bytes) is bigger than json.dumps(True) (4
    bytes), a real `False` extra field passed the "does degrading this
    shrink it" check and got silently flipped to `True`. That's not
    truncation, it's data corruption: a caller reading the response sees
    the wrong value with no signal anything happened. Booleans must never
    be treated as a degrade target at all, regardless of which one they
    currently hold."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "90")

    raw = utils.success_with_capped_dict(
        "record",
        {"id": "r1", "Notes": "x" * 5000},
        extra_fields={"is_recurring": False, "n": "N"},
    )
    result = json.loads(raw)

    assert len(raw) <= 90
    assert result["truncated"] is True
    assert result["is_recurring"] is False
    assert result["n"] == "N"


@pytest.mark.parametrize("field_name", ["status", "truncated"])
def test_success_with_capped_dict_rejects_a_field_name_that_shadows_the_envelope(
    field_name,
):
    """Regression test: `reserved_fields` validated that `extra_fields`
    couldn't collide with `field_name`/`status`/`truncated`, but never
    validated that `field_name` itself isn't one of those reserved keys.
    `field_name: payload` and the envelope's own hardcoded key would then
    be the literal same dict entry, and whichever appears later in the
    dict literal wins -- silently discarding the real payload (field_name
    == "status") or the envelope's own success/truncation markers
    (field_name == "truncated"), with no error and no size-capping ever
    attempted (the corrupted response is tiny, so the initial fit check
    trivially passes)."""
    with pytest.raises(ValueError, match="reserved envelope key"):
        utils.success_with_capped_dict(field_name, {"a": 1, "b": 2})


def test_success_with_capped_dict_last_resort_ranks_a_degraded_extra_below_the_id(
    monkeypatch,
):
    """Pins the fifth ladder rung and its position: an empty record beside a
    *degraded* extra ranks below the id alone -- only intact extras are
    worth giving the id up for. Here nothing containing the id fits and the
    extra doesn't fit intact beside {} either, so the degraded-extra rung is
    the first that fits and the id is (correctly) already gone; swapping
    rungs 4 and 5, or dropping rung 5, would change this result."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "75")

    raw = utils.success_with_capped_dict(
        "record",
        {"id": "LONGISHIDVALUE1234", "Notes": "x" * 5000},
        extra_fields={"note": "y" * 60},
    )
    result = json.loads(raw)

    assert len(raw) <= 75
    assert result["truncated"] is True
    assert result["record"] == {}
    assert result["note"] is True


def test_success_with_capped_dict_last_resort_prefers_an_intact_url_over_the_id(
    monkeypatch,
):
    """Regression test for a reviewer-reported case: a calendar-shaped event
    (a 26-character id plus a long description) with a real, still-intact
    Meet link as `extra_fields`. Before the ladder was reordered, `id +
    degraded extras` (the id beside a `True` placeholder for the link) was
    tried -- and fit -- before `{} + intact extras` got a chance, so the
    usable URL was thrown away in favor of an id next to a flag that only
    says "there was a link". The reordered ladder tries the empty record
    with the link still intact before ever degrading that link, so the
    real, followable URL wins over the id."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "109")

    raw = utils.success_with_capped_dict(
        "event",
        {"id": "e" * 26, "description": "d" * 5000},
        extra_fields={"hangout_link": "https://meet.google.com/abc-defg-hij"},
    )
    result = json.loads(raw)

    assert len(raw) <= 109
    assert result["truncated"] is True
    assert result["event"] == {}
    assert result["hangout_link"] == "https://meet.google.com/abc-defg-hij"


# ---------------------------------------------------------------------------
# _halve_largest_list_fields_until_bounded - list-collapse sharp edge
# ---------------------------------------------------------------------------
#
# Halving a list's element count works at any length above one, but floors
# to [] in one step for a single-item list: `conflict_response`'s
# `conflicts`/`unchecked_attendees` and `incomplete_check_response`'s
# `unchecked_attendees` could silently lose a genuine last item, becoming
# indistinguishable from "none found" even though `truncated` is True.


def test_conflict_response_marks_a_last_conflict_truncated_instead_of_emptying(
    monkeypatch,
):
    """Regression test: with a single real conflict too large to fit
    alongside the fixed envelope, halving it (`values[: len(values) // 2]`)
    used to collapse it straight to `[]` in one step -- indistinguishable
    from "no conflicts found" even though `truncated` is True. It should
    become a one-item truncation marker instead, as long as there's budget
    left over once nothing else can be shrunk further (see
    `_halve_largest_list_fields_until_bounded`)."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "700")
    conflicts = [
        {
            "calendar": "organizer",
            "summary": "Busy block " + "x" * 400,
            "start": "2026-08-27T10:00:00+00:00",
            "end": "2026-08-27T10:30:00+00:00",
        }
    ]

    response_text = utils.conflict_response(
        conflicts, [], "2026-08-27T10:00:00", "2026-08-27T10:30:00"
    )
    response = json.loads(response_text)

    assert len(response_text) <= 700
    assert response["status"] == "conflict"
    assert response["truncated"] is True
    assert response["conflicts"] == [{"truncated": True}]


def test_conflict_response_marks_the_last_unchecked_attendee_truncated_instead_of_emptying(
    monkeypatch,
):
    """Same sharp edge as above, for `unchecked_attendees`. Uses a small,
    realistic `conflicts` list (every real caller only reaches
    `conflict_response` with at least one conflict already confirmed -
    see `calendar.py`/`outlook.py`'s `if conflicts:` guards) that fits
    untouched, isolating the marker behavior to `unchecked_attendees`
    alone. The marker for a `list[str]` field is a sentinel string rather
    than the dict marker used for `conflicts`' event objects, since the
    field must stay a well-formed list of its own element type."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "600")
    conflicts = [
        {
            "calendar": "organizer",
            "summary": "1:1 with Hazel",
            "start": "2026-08-27T10:00:00+00:00",
            "end": "2026-08-27T10:30:00+00:00",
        }
    ]
    unchecked_attendees = ["person" + "y" * 300 + "@example.com"]

    response_text = utils.conflict_response(
        conflicts, unchecked_attendees, "2026-08-27T10:00:00", "2026-08-27T10:30:00"
    )
    response = json.loads(response_text)

    assert len(response_text) <= 600
    assert response["status"] == "conflict"
    assert response["truncated"] is True
    assert response["conflicts"] == conflicts
    assert response["unchecked_attendees"] == ["<truncated>"]


def test_conflict_response_marks_one_field_while_the_other_keeps_shrinking(
    monkeypatch,
):
    """Regression test: with both `conflicts` and `unchecked_attendees`
    populated, `conflicts` (one oversized event) reaches its one-item
    floor and gets marked well before `unchecked_attendees` (200 small
    entries) is done shrinking - the marker for the first field must not
    stop the second from continuing to shrink toward the budget on later
    loop iterations."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "900")
    conflicts = [
        {
            "calendar": "organizer",
            "summary": "Busy block " + "x" * 300,
            "start": "2026-08-27T10:00:00+00:00",
            "end": "2026-08-27T10:30:00+00:00",
        }
    ]
    unchecked_attendees = [f"person{i}@example.com" for i in range(100)]

    response_text = utils.conflict_response(
        conflicts, unchecked_attendees, "2026-08-27T10:00:00", "2026-08-27T10:30:00"
    )
    response = json.loads(response_text)

    assert len(response_text) <= 900
    assert response["status"] == "conflict"
    assert response["truncated"] is True
    assert response["conflicts"] == [{"truncated": True}]
    assert 0 < len(response["unchecked_attendees"]) < len(unchecked_attendees)
    assert (
        response["unchecked_attendees"]
        == unchecked_attendees[: len(response["unchecked_attendees"])]
    )


def test_conflict_response_upgrades_only_the_first_floored_field_when_budget_is_tight(
    monkeypatch,
):
    """Regression test: once both `conflicts` and `unchecked_attendees`
    have floored to [], the marker-upgrade pass spends whatever budget is
    left over one field at a time, in the order they floored. `conflicts`
    floors first here (its single event is larger than the single
    unchecked attendee), so its marker is installed; there's no budget
    left for `unchecked_attendees`'s own marker, which must stay []
    rather than push the response over the limit."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "475")
    conflicts = [
        {
            "calendar": "organizer",
            "summary": "Busy block " + "x" * 300,
            "start": "2026-08-27T10:00:00+00:00",
            "end": "2026-08-27T10:30:00+00:00",
        }
    ]
    unchecked_attendees = ["person" + "y" * 250 + "@example.com"]

    response_text = utils.conflict_response(
        conflicts, unchecked_attendees, "2026-08-27T10:00:00", "2026-08-27T10:30:00"
    )
    response = json.loads(response_text)

    assert len(response_text) <= 475
    assert response["status"] == "conflict"
    assert response["truncated"] is True
    assert response["conflicts"] == [{"truncated": True}]
    assert response["unchecked_attendees"] == []


def test_conflict_response_still_empties_a_last_conflict_when_no_budget_for_the_marker(
    monkeypatch,
):
    """Regression test: the marker upgrade only happens if it actually
    fits (checked against the real rebuilt response, not just estimated) --
    a limit too tight for the marker must still fall back to `[]` rather
    than exceed `max_output_length`. The limit is chosen so the *floored*
    envelope (conflicts=[], original message/hint intact) already fits on
    its own - proving this is the marker-upgrade pass declining to install
    a marker that doesn't fit, not `conflict_response`'s separate, much
    more aggressive compact-JSON fallback for a limit too tight for the
    fixed envelope itself (see
    test_conflict_response_uses_valid_compact_json_below_fixed_envelope)."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "460")
    conflicts = [
        {
            "calendar": "organizer",
            "summary": "Busy block " + "x" * 400,
            "start": "2026-08-27T10:00:00+00:00",
            "end": "2026-08-27T10:30:00+00:00",
        }
    ]

    response_text = utils.conflict_response(
        conflicts, [], "2026-08-27T10:00:00", "2026-08-27T10:30:00"
    )
    response = json.loads(response_text)

    assert len(response_text) <= 460
    assert response["status"] == "conflict"
    assert response["truncated"] is True
    assert response["conflicts"] == []
    assert response["message"] == (
        "1 existing event(s) overlap 2026-08-27T10:00:00 - 2026-08-27T10:30:00"
    )


def test_incomplete_check_response_uncapped_when_it_fits(monkeypatch):
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "50000")
    response = json.loads(
        utils.incomplete_check_response(
            ["unreachable@example.com"],
            "2026-08-27T10:00:00",
            "2026-08-27T10:30:00",
        )
    )
    assert response["status"] == "conflict_check_incomplete"
    assert response["unchecked_attendees"] == ["unreachable@example.com"]
    assert response["truncated"] is False


def test_incomplete_check_response_marks_the_last_unchecked_attendee_truncated_instead_of_emptying(
    monkeypatch,
):
    """Same sharp edge as `conflict_response`'s, for
    `incomplete_check_response`'s own (sole) `unchecked_attendees`
    field -- the last unchecked calendar must not silently vanish to `[]`
    under a tight limit."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "500")
    unchecked_attendees = ["person" + "x" * 300 + "@example.com"]

    response_text = utils.incomplete_check_response(
        unchecked_attendees, "2026-08-27T10:00:00", "2026-08-27T10:30:00"
    )
    response = json.loads(response_text)

    assert len(response_text) <= 500
    assert response["status"] == "conflict_check_incomplete"
    assert response["truncated"] is True
    assert response["unchecked_attendees"] == ["<truncated>"]


def test_incomplete_check_response_still_empties_when_no_budget_for_the_marker(
    monkeypatch,
):
    """Same distinction as `conflict_response`'s: at this limit the
    floored envelope (unchecked_attendees=[], original message intact)
    already fits, so a response reaching that state proves the
    marker-upgrade pass itself declined a marker that didn't fit - not
    the separate, much more aggressive compact-JSON fallback for a limit
    too tight for the fixed envelope itself."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "455")
    unchecked_attendees = ["person" + "x" * 300 + "@example.com"]

    response_text = utils.incomplete_check_response(
        unchecked_attendees, "2026-08-27T10:00:00", "2026-08-27T10:30:00"
    )
    response = json.loads(response_text)

    assert len(response_text) <= 455
    assert response["status"] == "conflict_check_incomplete"
    assert response["truncated"] is True
    assert response["unchecked_attendees"] == []
    assert response["message"] == (
        "Availability could not be checked for 1 calendar(s) for "
        "2026-08-27T10:00:00 - 2026-08-27T10:30:00. No event was written."
    )


def test_success_with_capped_dict_last_resort_fallback_keeps_capitalized_id(
    monkeypatch,
):
    """The severe-truncation last resort used to check only lowercase "id",
    so Deputy's capitalized "Id" (and MYOB's "Uid") was silently dropped
    once a record got truncated all the way down -- exercised directly
    against the shared utility (not through a deputy_* tool) because
    reaching this exact branch also requires extra_fields to push the
    unstripped candidate over the limit, which no deputy_* call site uses."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "100")

    raw = utils.success_with_capped_dict(
        "record",
        {"Id": 123, "Notes": "x" * 5000},
        extra_fields={"note": "y" * 60},
    )
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["record"] == {"Id": 123}


def test_success_with_capped_dict_critical_fields_survive_id_only_candidate(
    monkeypatch,
):
    """Unlike extra_fields (previous test), critical_fields must NOT be
    dropped once the ladder is down to its id-only candidate -- this is
    what _verify_created_record relies on so an unconfirmed-create warning
    is never silently lost to truncation, which would reproduce the exact
    "confident-looking success" failure this mechanism exists to catch.
    (At this budget the id-only candidate already fits, so this doesn't
    reach _fit_critical_fields -- see the dedicated tests for that.)"""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "140")

    raw = utils.success_with_capped_dict(
        "record",
        {"Id": 123, "Notes": "x" * 5000},
        critical_fields={"warning": "y" * 60},
    )
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["record"] == {"Id": 123}
    assert result["warning"] == "y" * 60


def test_success_with_capped_dict_critical_fields_survive_tiny_requested_budget(
    monkeypatch,
):
    """Even with an aggressively low requested budget (floored back up to
    _MIN_OUTPUT_LENGTH=64), a critical value short enough to already fit
    within that floor's own fixed envelope overhead must come through
    completely untouched -- this is satisfied by the ladder's own
    bare-status-plus-critical candidate at this size, not by
    _fit_critical_fields itself (see the dedicated tests below for that
    function's own guarantee, including for values too long to fit)."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "10")

    raw = utils.success_with_capped_dict(
        "record",
        {"Id": 123, "Notes": "x" * 5000},
        critical_fields={"warning": "unconf"},
    )
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["warning"] == "unconf"


def test_success_with_capped_dict_fits_the_exact_reviewer_reported_budget(
    monkeypatch,
):
    """Verified against rogercloud's own PR #2592 review numbers exactly:
    XAGENT_TOOL_MAX_OUTPUT_LENGTH=64, an Id=1 "returned no record" warning.
    At this budget there isn't even room left for the truncation marker
    once the fixed envelope overhead is accounted for -- the response must
    still fit, even if that means the surviving warning content is just a
    character or two; a stdio MCP child's own budget is mirrored into a
    *separate*, JSON-unaware truncation the parent process applies to this
    exact string, and an oversized response here would be blindly sliced
    mid-structure by that parent-side filter, producing invalid JSON
    instead of a valid-but-abbreviated one."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "64")
    warning = (
        "Deputy reported this Roster create as successful (Id 1), but "
        "reading it back returned no record. Treat this as unconfirmed -- "
        "verify in Deputy directly before relying on it."
    )
    assert len(warning) > 64  # the scenario only arises when it doesn't fit whole

    raw = utils.success_with_capped_dict(
        "record",
        {"Id": 1, "Notes": "x" * 5000},
        critical_fields={"warning": warning},
    )

    assert len(raw) <= 64
    result = json.loads(raw)
    assert result["status"] == "success"
    assert result["truncated"] is True


def test_success_with_capped_dict_shortens_an_oversized_critical_field(monkeypatch):
    """With a bit more room than the tightest possible budget, shortening
    must keep a real prefix of the original content plus the truncation
    marker, not just whatever survives is dropped down to nothing."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "100")
    warning = (
        "Deputy reported this Roster create as successful (Id 1), but "
        "reading it back returned no record. Treat this as unconfirmed -- "
        "verify in Deputy directly before relying on it."
    )
    assert len(warning) > 100

    raw = utils.success_with_capped_dict(
        "record",
        {"Id": 1, "Notes": "x" * 5000},
        critical_fields={"warning": warning},
    )

    assert len(raw) <= 100
    result = json.loads(raw)
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["warning"].endswith("... [truncated]")
    assert warning.startswith(result["warning"].removesuffix("... [truncated]"))


def test_success_with_capped_dict_never_inflates_a_critical_field_below_marker_size(
    monkeypatch,
):
    """A critical field whose value is already SHORTER than the truncation
    marker must be left alone, not replaced by something bigger -- without
    a size gate matching _truncation_marker_or_empty's own ("only install
    the marker if it's no bigger than what it replaces"), a long key name
    paired with a short value could get "shortened" into a larger, still
    oversized response with nothing left to shrink afterward."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "64")

    raw = utils.success_with_capped_dict(
        "record",
        {"Id": 1, "Notes": "x" * 5000},
        critical_fields={"a_very_long_critical_field_key_name_here": "x"},
    )

    assert len(raw) <= 64
    result = json.loads(raw)
    assert result["status"] == "success"
    assert result["truncated"] is True
    # Even the bare key name alone ("a_very_long_critical_field_key_name_here",
    # 41 chars) plus the envelope skeleton already exceeds this 64-char
    # budget regardless of its value, so this field is always dropped here
    # -- never left alone at "x" -- which is what should be asserted,
    # rather than an OR that would also pass if shrinking somehow mangled
    # the value into something other than "x".
    assert "a_very_long_critical_field_key_name_here" not in result


def test_success_with_capped_dict_drops_a_critical_field_that_cannot_shrink(
    monkeypatch,
):
    """A non-string critical value can't be shortened the way a string
    can -- once every shrinkable string is exhausted and the envelope is
    still oversized, the field must be dropped outright rather than the
    function ever handing back something wider than max_output_length."""
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", "64")

    raw = utils.success_with_capped_dict(
        "record",
        {"Id": 1, "Notes": "x" * 5000},
        critical_fields={"stats": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}},
    )

    assert len(raw) <= 64
    result = json.loads(raw)
    assert result["status"] == "success"
    assert result["truncated"] is True


def test_success_with_capped_dict_rejects_overlapping_extra_and_critical_fields():
    with pytest.raises(ValueError, match="must not share a key"):
        utils.success_with_capped_dict(
            "record",
            {"Id": 1},
            extra_fields={"note": "a"},
            critical_fields={"note": "b"},
        )
