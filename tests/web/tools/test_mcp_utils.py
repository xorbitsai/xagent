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
    with pytest.raises(ValueError, match="unknown timezone"):
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
    with pytest.raises(ValueError, match="unknown timezone"):
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
