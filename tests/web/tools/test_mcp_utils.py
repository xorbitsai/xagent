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
    key = utils.datetime_key_for_comparison("2026-08-27T10:00:00.0000000")
    assert key == utils.datetime_key_for_comparison("2026-08-27T10:00:00")


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
        ["Old@Example.com", "new@example.com"], {"old@example.com"}
    ) == ["new@example.com"]


def test_attendees_to_add_returns_empty_for_not_provided_or_empty():
    assert utils.attendees_to_add(None, {"old@example.com"}) == []
    assert utils.attendees_to_add("", {"old@example.com"}) == []
    assert utils.attendees_to_add([], {"old@example.com"}) == []


def test_unchecked_extra_empty_when_nothing_unchecked():
    assert utils.unchecked_extra([], None) == {}
    assert utils.unchecked_extra([], "some reason") == {}


def test_unchecked_extra_includes_reason_only_when_given():
    assert utils.unchecked_extra(["a@x.com"], None) == {
        "unchecked_attendees": ["a@x.com"]
    }
    assert utils.unchecked_extra(["a@x.com"], "reconnect the connector") == {
        "unchecked_attendees": ["a@x.com"],
        "unchecked_reason": "reconnect the connector",
    }


def test_attendees_needing_check_disjoint_window_checks_everyone():
    assert utils.attendees_needing_check(
        ["old@x.com", "new@x.com"],
        {"old@x.com"},
        moved_to_a_disjoint_window=True,
    ) == ["old@x.com", "new@x.com"]


def test_attendees_needing_check_same_or_overlapping_window_checks_only_new():
    """An existing attendee's schedule always shows this event's own busy
    block for a window it still occupies - only newly-added attendees are
    safe to check there."""
    assert utils.attendees_needing_check(
        ["old@x.com", "new@x.com"],
        {"old@x.com"},
        moved_to_a_disjoint_window=False,
    ) == ["new@x.com"]


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


def test_resolve_zone_name_covers_graphs_additional_time_zones():
    """Regression test: `_WINDOWS_TO_IANA` was missing several of the
    Windows names for zones Microsoft's own dateTimeTimeZone docs list
    under "Additional time zones" (e.g. Kaliningrad, Ekaterinburg) - an
    event whose originalStartTimeZone happened to be one of these
    previously hard-failed every single-boundary update and conflict
    check outright."""
    assert utils.resolve_zone_name("Kaliningrad Standard Time") == "Europe/Kaliningrad"
    assert utils.resolve_zone_name("Ekaterinburg Standard Time") == "Asia/Yekaterinburg"
    assert utils.resolve_zone_name("Vladivostok Standard Time") == "Asia/Vladivostok"


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
