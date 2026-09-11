import json
import re

import pytest
import requests

from xagent.web.tools.mcp import utils


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
