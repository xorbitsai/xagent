from __future__ import annotations

import asyncio
import zoneinfo
from collections.abc import Iterator
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from xagent.core.tools.adapters.vibe.current_time_tool import (
    ValidateLocalTimeTool,
    validate_local_time,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

# The sibling suite owns this stub; both tools go through the same pipeline.
from .test_current_time_tool import _FakeConfig


def _rows(local_time: str, zone: str) -> list[tuple[str, str, str, str]]:
    result = validate_local_time(local_time, zone)
    return [(m.local, m.abbreviation, m.utc_offset, m.utc) for m in result.mappings]


def test_skipped_local_time_does_not_exist() -> None:
    # Sydney springs forward at 02:00 on 4 Oct 2026, so 02:30 never occurs.
    result = validate_local_time("2026-10-04 02:30", "Australia/Sydney")

    assert result.local_time_status == "nonexistent"
    assert result.mappings == []
    assert result.timezone == "Australia/Sydney"


def test_repeated_local_time_reports_both_instants() -> None:
    # Sydney goes back at 03:00 on 5 Apr 2026, so 02:30 occurs twice. The
    # abbreviation and offset on each row are what the model must quote
    # instead of recalling which one applies in a given season.
    result = validate_local_time("2026-04-05 02:30", "Australia/Sydney")

    assert result.local_time_status == "ambiguous"
    assert _rows("2026-04-05 02:30", "Australia/Sydney") == [
        ("2026-04-05 02:30:00", "AEDT", "+11:00", "2026-04-04 15:30:00"),
        ("2026-04-05 02:30:00", "AEST", "+10:00", "2026-04-04 16:30:00"),
    ]


def test_ordinary_local_time_maps_to_one_instant() -> None:
    assert _rows("2026-06-01 09:00", "Australia/Sydney") == [
        ("2026-06-01 09:00:00", "AEST", "+10:00", "2026-05-31 23:00:00")
    ]


@pytest.fixture
def bundled_tzdata_only() -> Iterator[None]:
    """Run against the tzdata wheel, as in a container with no system zoneinfo.

    Required, not incidental: the fold-validation this fixture exercises is
    indistinguishable from trusting one fold under a tzdata release where no
    zone has the Nuuk shape (macOS ships 2026c, which has none).
    """
    original = zoneinfo.TZPATH
    zoneinfo.reset_tzpath([])
    ZoneInfo.clear_cache()
    try:
        yield
    finally:
        zoneinfo.reset_tzpath(original)
        ZoneInfo.clear_cache()


def test_a_reading_a_fold_reports_but_cannot_convert_back_is_dropped(
    bundled_tzdata_only: None,
) -> None:
    # Nuuk's fold=0 reading claims -01:00, an offset it does not have at the
    # resulting instant; fold=1 claims -02:00 and converts back exactly.
    # Trusting one fold's offset instead of validating each candidate reports
    # this real, single instant as ambiguous with a fabricated second reading.
    result = validate_local_time("2023-10-28 23:30", "America/Nuuk")

    assert result.local_time_status == "unique"
    assert result.mappings[0].utc == "2023-10-29 01:30:00"
    assert result.mappings[0].utc_offset == "-02:00"


@pytest.mark.parametrize(
    "zone",
    ["Australia/Sydney", "Australia/Lord_Howe", "Pacific/Apia", "America/St_Johns"],
)
@pytest.mark.parametrize(
    "local_time", ["2026-10-04 02:15", "2026-04-05 02:30", "2011-12-30 12:00"]
)
def test_every_reported_mapping_round_trips(zone: str, local_time: str) -> None:
    # The one invariant behind all three classifications, asserted without
    # fixed expectations so it holds under any tzdata release: a reported
    # instant must convert back to exactly the wall time that was asked about.
    asked = datetime.fromisoformat(local_time)

    for mapping in validate_local_time(local_time, zone).mappings:
        instant = datetime.fromisoformat(mapping.utc).replace(tzinfo=timezone.utc)
        assert instant.astimezone(ZoneInfo(zone)).replace(tzinfo=None) == asked


def test_half_hour_offset_is_reported_with_its_minutes() -> None:
    # Lord Howe has no lettered abbreviation, so the model has to quote the
    # numeric one it is given. Pinned on standard time, where the offset
    # actually carries the half hour.
    assert _rows("2026-06-01 12:00", "Australia/Lord_Howe") == [
        ("2026-06-01 12:00:00", "+1030", "+10:30", "2026-06-01 01:30:00")
    ]


def test_a_half_hour_shift_keeps_both_readings_distinct() -> None:
    assert _rows("2026-04-05 01:45", "Australia/Lord_Howe") == [
        ("2026-04-05 01:45:00", "+11", "+11:00", "2026-04-04 14:45:00"),
        ("2026-04-05 01:45:00", "+1030", "+10:30", "2026-04-04 15:15:00"),
    ]


def test_sub_minute_offset_keeps_its_seconds() -> None:
    # Liberia ran -00:44:30 until 1972. Truncating to whole minutes would make
    # utc_offset contradict the local/utc pair in the same row, and the
    # description tells the model to quote that offset.
    assert _rows("1971-01-01 12:00", "Africa/Monrovia") == [
        ("1971-01-01 12:00:00", "MMT", "-00:44:30", "1971-01-01 12:44:30")
    ]


@pytest.mark.parametrize(
    "zone",
    [
        "EST",  # a fixed -05:00 that never observes DST
        "MST",
        "GMT",
        "CET",
        "Etc/GMT+10",  # POSIX sign inversion: actually -10:00
        "Factory",
    ],
)
def test_a_name_that_does_not_mean_a_place_is_rejected(zone: str) -> None:
    # These all resolve, so the offset would be returned with no signal that
    # it is not the zone the caller meant. Here the offset is the answer.
    with pytest.raises(ValueError, match="Region/City"):
        validate_local_time("2026-07-15 12:00", zone)


@pytest.mark.parametrize("zone", ["UTC", "utc", "America/New_York", "us/eastern"])
def test_a_name_that_does_mean_a_place_is_accepted(zone: str) -> None:
    assert validate_local_time("2026-07-15 12:00", zone).mappings


def test_zone_name_is_case_insensitive() -> None:
    assert validate_local_time("2026-06-01 09:00", "australia/sydney").timezone == (
        "Australia/Sydney"
    )


@pytest.mark.parametrize("zone", ["Not/AZone", "", "   ", "/etc/passwd"])
def test_unknown_zone_is_rejected(zone: str) -> None:
    with pytest.raises(ValueError, match="Unknown IANA timezone"):
        validate_local_time("2026-06-01 09:00", zone)


@pytest.mark.parametrize(
    "bad",
    [
        "not a time",
        "",
        "2026-10-04",  # a bare date would silently mean midnight
        "2026-06-01 09:00+10:00",
        "2026-06-01T09:00:00Z",
        "2026-06-01 09:00:00.5",
        "2026-6-1 9:00",
        "20260601T0900",
        "2026-06-01 09:00\x00",
        "٢٠٢٦-٠٦-٠١ ٠٩:٠٠",  # \d would admit non-ASCII digits
        "2026-06-01 0٩:00",
        "2026-06-01 09:60",
    ],
)
def test_malformed_local_time_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="local_time must be"):
        validate_local_time(bad, "Australia/Sydney")


@pytest.mark.parametrize("bad", ["2026-10-03 24:00", "2026-10-03 24:00:00"])
def test_hour_24_is_rejected_rather_than_rolled_to_the_next_day(bad: str) -> None:
    # Python 3.14's fromisoformat reads '24:00' as the next day's midnight,
    # which would answer for a date the caller never named. requires-python
    # allows 3.14, so the bound lives in the pattern.
    with pytest.raises(ValueError, match="local_time must be"):
        validate_local_time(bad, "Australia/Sydney")


def test_well_formed_but_unreal_date_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a real date and time"):
        validate_local_time("2026-02-29 09:00", "Australia/Sydney")


def test_conversion_leaving_the_representable_range_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="representable date range"):
        validate_local_time("9999-12-31 23:59", "America/New_York")


def test_year_below_1000_is_zero_padded() -> None:
    # An eastward offset reaches year 999 from a 4-digit local year. glibc's
    # strftime("%Y") would render it unpadded; this can only fail there, since
    # macOS pads either way.
    only = validate_local_time("1000-01-01 09:00", "Australia/Sydney").mappings[0]

    assert (only.local, only.utc) == ("1000-01-01 09:00:00", "0999-12-31 22:55:08")


def test_tool_runs_through_the_json_surface() -> None:
    tool = ValidateLocalTimeTool()

    assert tool.run_json_sync(
        {"local_time": "2026-10-04 02:30", "timezone": "Australia/Sydney"}
    ) == {
        "local_time_status": "nonexistent",
        "mappings": [],
        "timezone": "Australia/Sydney",
    }
    assert asyncio.run(
        tool.run_json_async(
            {"local_time": "2026-06-01 09:00", "timezone": "Australia/Sydney"}
        )
    ) == {
        "local_time_status": "unique",
        "mappings": [
            {
                "local": "2026-06-01 09:00:00",
                "abbreviation": "AEST",
                "utc_offset": "+10:00",
                "utc": "2026-05-31 23:00:00",
            }
        ],
        "timezone": "Australia/Sydney",
    }


def test_json_surface_rejects_a_missing_field() -> None:
    with pytest.raises(ValidationError):
        ValidateLocalTimeTool().run_json_sync({"timezone": "Australia/Sydney"})


def test_status_field_avoids_the_tool_result_control_channel() -> None:
    # A top-level 'status' of 'error' fails the whole tool call, and
    # 'waiting_for_user' suspends the turn (core/agent/result.py).
    payload = ValidateLocalTimeTool().run_json_sync(
        {"local_time": "2026-06-01 09:00", "timezone": "Australia/Sydney"}
    )

    assert "status" not in payload


def test_description_hands_off_from_the_current_time_tool() -> None:
    from xagent.core.tools.adapters.vibe.current_time_tool import CurrentTimeTool

    # The system prompt names only get_current_time, so the hand-off has to
    # come from its description.
    assert "validate_local_time" in CurrentTimeTool().description


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (ToolSelectionSpec.from_raw(tool_categories=None), 1),  # ALL
        (ToolSelectionSpec.from_raw(tool_categories=["web_search"]), 1),  # non-basic
        (ToolSelectionSpec.from_raw(tool_categories=[]), 0),  # explicit NONE
    ],
)
async def test_intrinsic_tool_is_assembled_for_non_none_specs(
    spec: ToolSelectionSpec, expected: int
) -> None:
    tools = await ToolFactory.create_all_tools(
        _FakeConfig(spec), apply_user_override_filter=False
    )

    assert [t.name for t in tools].count("validate_local_time") == expected


async def test_intrinsic_creator_is_skipped_for_explicit_none() -> None:
    # Pinned at the registry level: the post-build name filter is a second
    # guard that would otherwise mask a missing selection_gate.
    tools = await ToolRegistry.create_registered_tools(
        _FakeConfig(ToolSelectionSpec.from_raw(tool_categories=[]))
    )

    assert "validate_local_time" not in [getattr(t, "name", None) for t in tools]
