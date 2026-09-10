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
    02:00-03:00), so the calendar day is only 23 hours long. `start +
    timedelta(days=1)` on an aware datetime only advances the wall-clock
    date/time components (per datetime's documented semantics) and
    re-derives the UTC offset lazily via ZoneInfo - a naive
    `timedelta(hours=24)` would instead land on 01:00 the following day,
    silently mis-widening an all-day event's boundary by an hour on every
    DST-transition day in a zone that observes it."""
    start, end = utils.calendar_day_bounds("2026-03-08", "America/New_York")
    assert start == "2026-03-08T00:00:00-05:00"
    assert end == "2026-03-09T00:00:00-04:00"


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
