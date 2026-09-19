from __future__ import annotations

import asyncio
import itertools
import re
from datetime import datetime, timezone
from typing import Callable, Optional

import pytest
from pydantic import ValidationError

from tests.core.tools.test_current_time_tool import _FakeConfig

# The sibling suite owns this fixture; both tools read the same tz database.
from tests.core.tools.test_validate_local_time_tool import (  # noqa: F401
    bundled_tzdata_only,
)
from xagent.core.tools.adapters.vibe import current_time_tool as module
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.current_time_tool import (
    GRAMMAR_FORMS,
    RESOLUTION_REASONS,
    CurrentTimeTool,
    ResolveDatetimeResult,
    ResolveDatetimeTool,
    ValidateLocalTimeTool,
    resolve_datetime,
    validate_local_time,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec

# Tuesday 2026-09-15 12:30 in Sydney (AEST), 10:30 in Shanghai.
FROZEN = datetime(2026, 9, 15, 2, 30, 0, tzinfo=timezone.utc)

SYDNEY = "Australia/Sydney"
SHANGHAI = "Asia/Shanghai"

# (phrase, zone, expected) where expected is (resolved, has_time) on success
# or ("REFUSED", resolution) on refusal.
CASES: list[tuple[str, str, tuple[str, object]]] = [
    ("today", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    ("tomorrow", SYDNEY, ("2026-09-16T00:00:00+10:00", False)),
    ("tomorrow at 3pm", SYDNEY, ("2026-09-16T15:00:00+10:00", True)),
    ("yesterday", SYDNEY, ("2026-09-14T00:00:00+10:00", False)),
    ("day after tomorrow", SYDNEY, ("2026-09-17T00:00:00+10:00", False)),
    ("next Monday", SYDNEY, ("2026-09-21T00:00:00+10:00", False)),
    ("next tuesday", SYDNEY, ("2026-09-22T00:00:00+10:00", False)),
    ("last Friday", SYDNEY, ("2026-09-11T00:00:00+10:00", False)),
    ("Friday at 10:30", SYDNEY, ("2026-09-18T10:30:00+10:00", True)),
    ("Friday 10am", SYDNEY, ("2026-09-18T10:00:00+10:00", True)),
    ("in 3 days", SYDNEY, ("2026-09-18T00:00:00+10:00", False)),
    ("in 2 hours", SYDNEY, ("2026-09-15T14:30:00+10:00", True)),
    ("in 1 week", SYDNEY, ("2026-09-22T00:00:00+10:00", False)),
    ("15 Sep 2026 10:00", SYDNEY, ("2026-09-15T10:00:00+10:00", True)),
    ("15 Sep 2026", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    ("1 Jan 1990", SYDNEY, ("1990-01-01T00:00:00+11:00", False)),
    ("1990-01-01", SYDNEY, ("1990-01-01T00:00:00+11:00", False)),
    ("2026-10-04", SYDNEY, ("2026-10-04T00:00:00+10:00", False)),
    ("2026-09-15T10:00:00+10:00", SHANGHAI, ("2026-09-15T08:00:00+08:00", True)),
    ("01/02/1990", SYDNEY, ("REFUSED", "ambiguous_date")),
    ("今天", SHANGHAI, ("2026-09-15T00:00:00+08:00", False)),
    ("明天下午三点", SHANGHAI, ("2026-09-16T15:00:00+08:00", True)),
    ("后天", SHANGHAI, ("2026-09-17T00:00:00+08:00", False)),
    ("昨天", SHANGHAI, ("2026-09-14T00:00:00+08:00", False)),
    ("下周三", SHANGHAI, ("2026-09-23T00:00:00+08:00", False)),
    ("上周五", SHANGHAI, ("2026-09-11T00:00:00+08:00", False)),
    ("周五上午十点半", SHANGHAI, ("2026-09-18T10:30:00+08:00", True)),
    ("3天后", SHANGHAI, ("2026-09-18T00:00:00+08:00", False)),
    ("两小时后", SHANGHAI, ("2026-09-15T12:30:00+08:00", True)),
    ("1990年1月1日", SHANGHAI, ("1990-01-01T00:00:00+08:00", False)),
    ("2026年9月20日 下午两点", SHANGHAI, ("2026-09-20T14:00:00+08:00", True)),
    ("soon", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("next month", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("the first Monday of October", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("Bibek khadka", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("2026-10-04 02:30", SYDNEY, ("REFUSED", "nonexistent_local_time")),
    ("2026-04-05 02:30", SYDNEY, ("REFUSED", "ambiguous_local_time")),
    ("tomorrow", "EST", ("REFUSED", "invalid_timezone")),
    ("3pm", SYDNEY, ("2026-09-15T15:00:00+10:00", True)),
    ("10:30", SYDNEY, ("2026-09-15T10:30:00+10:00", True)),
    ("3", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("下午三点", SHANGHAI, ("2026-09-15T15:00:00+08:00", True)),
    ("十点半", SHANGHAI, ("2026-09-15T10:30:00+08:00", True)),
    ("noon", SYDNEY, ("REFUSED", "unsupported_expression")),
    # Hour twelve with a half-day word is refused rather than picked one of
    # two ways: it could be midnight (today ending or tomorrow beginning)
    # or, read as bare digits, noon.
    ("晚上12点", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("12 pm", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("上午12点", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("12 am", SYDNEY, ("REFUSED", "unsupported_expression")),
    # A supported form with extra words around it is the whole phrase
    # failing to match, not the supported word inside it being found.
    ("call me back tomorrow if you can", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("我昨天见过他", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("tomorrow at 3pm please", SYDNEY, ("REFUSED", "unsupported_expression")),
    # "this <weekday>" stays within the current calendar week, including
    # when today already is that weekday.
    ("this friday", SYDNEY, ("2026-09-18T00:00:00+10:00", False)),
    ("this tuesday", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    # The minute half of "N hours/minutes later" (English and Chinese).
    ("in 30 minutes", SYDNEY, ("2026-09-15T13:00:00+10:00", True)),
    ("30分钟后", SHANGHAI, ("2026-09-15T11:00:00+08:00", True)),
    # The alternate character for Sunday folds to the same weekday
    # position as the other spelling.
    ("周天", SHANGHAI, ("2026-09-20T00:00:00+08:00", False)),
    # "中午" (noon) only names an hour within one of noon: 11, 12, or 13.
    # Any other hour contradicts the period word instead of merely leaving
    # it ambiguous, so it is refused the same way hour twelve with a
    # half-day word is. There is no frozen-design reading for "中午1点",
    # so it is refused rather than guessed.
    ("中午10点", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("中午12点", SHANGHAI, ("2026-09-15T12:00:00+08:00", True)),
    ("中午1点", SHANGHAI, ("REFUSED", "unsupported_expression")),
    # A supported form with extra words around it, for every fullmatch
    # point that isn't already covered above: the whole phrase failing to
    # match, not the supported word inside it being found.
    (
        "please renew by 2026-09-20 thanks",
        SYDNEY,
        ("REFUSED", "unsupported_expression"),
    ),
    ("合同到2026年9月20日为止", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("下周三我请假", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("我们大概3天后见", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("我们两小时后再聊", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("我们下午三点在楼下见吧", SHANGHAI, ("REFUSED", "unsupported_expression")),
    ("see you next friday ok", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("let's meet in 3 days ok", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("call me in 2 hours please", SYDNEY, ("REFUSED", "unsupported_expression")),
    # An English calendar date carrying a time is parsed whole by dateutil,
    # which reads 12 pm as noon and 12 am as midnight on its own; the same
    # refusal that a bare "12 pm" gets must reach this path too.
    ("15 Sep 2026 12 pm", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15 Sep 2026 12 am", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15 Sep 2026 12:30 pm", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15 Sep 2026 12:30 am", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("Sep 15 2026 12pm", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("January 1st, 1990 12 am", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("1 Jan 1990 12:00:00 pm", SYDNEY, ("REFUSED", "unsupported_expression")),
    # Other spellings of the same hour-twelve-with-am/pm combination: a "."
    # minute separator, the dotted "p.m." form, and a leading-zero hour. The
    # calendar path must catch these by reading the hour dateutil parsed, not
    # by matching only the spelling the grammar's own examples use.
    ("15 Sep 2026 12.30 pm", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15 Sep 2026 12 p.m.", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15 Sep 2026 012 pm", SYDNEY, ("REFUSED", "unsupported_expression")),
    # The digit 12 elsewhere in the phrase -- a day, an unrelated hour, or no
    # am/pm at all -- does not trip the hour-twelve refusal.
    ("12 Dec 2026 3 pm", SYDNEY, ("2026-12-12T15:00:00+11:00", True)),
    ("12 Sep 2026 11am", SYDNEY, ("2026-09-12T11:00:00+10:00", True)),
    ("12 January 2026", SYDNEY, ("2026-01-12T00:00:00+11:00", False)),
    ("15 Sep 2026 12:00", SYDNEY, ("2026-09-15T12:00:00+10:00", True)),
    ("15 Sep 2026 10am", SYDNEY, ("2026-09-15T10:00:00+10:00", True)),
    ("15 Sep 2026 11:59 pm", SYDNEY, ("2026-09-15T23:59:00+10:00", True)),
    ("15 Sep 2026 3:12 pm", SYDNEY, ("2026-09-15T15:12:00+10:00", True)),
    ("15 Sep 2026 11 p.m.", SYDNEY, ("2026-09-15T23:00:00+10:00", True)),
    # The frozen design's other three calendar-date shapes: month-name-first,
    # an ordinal day, and both date separators.
    ("Jan 1 1990", SYDNEY, ("1990-01-01T00:00:00+11:00", False)),
    ("January 1st, 1990", SYDNEY, ("1990-01-01T00:00:00+11:00", False)),
    ("25/12/1990", SYDNEY, ("1990-12-25T00:00:00+11:00", False)),
    ("15-09-2026", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    # Shapes dateutil would accept on its own but the whole-phrase pattern
    # does not declare: a year written first, an eight-digit run, a "." date
    # separator, a "T" separator, and an "h" time separator.
    ("2026 Sep 15", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("20260915", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15.09.2026", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("2026/09/15", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15 Sep 2026 T 10:00", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15 Sep 2026 10h30", SYDNEY, ("REFUSED", "unsupported_expression")),
    # The 24-hour clock's own optional seconds group, previously unpinned.
    ("15 Sep 2026 10:00:00", SYDNEY, ("2026-09-15T10:00:00+10:00", True)),
    # A year under four digits, or a four-digit year starting with zero,
    # would otherwise let dateutil substitute a century from the machine's
    # real clock instead of the phrase (see _EN_CALENDAR_DATE's year group).
    ("1 Jan 90", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("1 Jan 0001", SYDNEY, ("REFUSED", "unsupported_expression")),
    # "Sept" is one of dateutil's own three September tokens, and a trailing
    # period on any month name is a spelling dateutil already accepted, in
    # either the day-first or the month-first branch.
    ("15 Sept 2026", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    ("15 Sep. 2026", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    ("Sep. 15 2026", SYDNEY, ("2026-09-15T00:00:00+10:00", False)),
    # The day-first branch's own comma, a shape a person ordinarily writes.
    ("1 Jan, 1990", SYDNEY, ("1990-01-01T00:00:00+11:00", False)),
    # The four-digit-year rule holds in the slash/dash branch too, not only
    # in the two month-name branches. "25/12/90" is the same century-
    # substitution hole closed above: without the rule, dateutil reads the
    # two-digit "90" as 1990 off the machine's real clock. "15-09-0055" is a
    # different failure the same rule also blocks: the year is already four
    # digits, so there is no substitution -- without the rule dateutil takes
    # it literally and returns the year 0055.
    ("25/12/90", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15-09-0055", SYDNEY, ("REFUSED", "unsupported_expression")),
    # The general am/pm alternative admits only hours one through eleven; a
    # zero-padded 24-hour value such as "023" is not one of them, so dateutil
    # never gets the chance to read it as hour 23 and discard the am/pm word.
    ("15 Sep 2026 023 am", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("15 Sep 2026 013 pm", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("023 am", SYDNEY, ("REFUSED", "unsupported_expression")),
    ("013 pm", SYDNEY, ("REFUSED", "unsupported_expression")),
]


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_now", lambda: FROZEN)


# This also pins the wall-time-to-instants extraction: the same error text.
def test_validate_local_time_unchanged_after_helper_extraction() -> None:
    with pytest.raises(ValueError) as exc:
        validate_local_time("2026-10-04T02:30", "EST")

    assert str(exc.value) == (
        "timezone must be a Region/City IANA name such as "
        "'Australia/Sydney' or 'UTC', not 'EST'"
    )


def test_resolve_datetime_returns_the_validated_fold(
    bundled_tzdata_only: None,  # noqa: F811
) -> None:
    """America/Nuuk has no daylight-saving change on 2023-10-28, so -02:00 is
    the only offset it has at this instant; the fold-0 reading claims -01:00,
    an offset the zone never has, and does not convert back to 23:30. The
    resolver must return the reading that does."""
    result = resolve_datetime("2023-10-28 23:30", "America/Nuuk")

    assert result == {
        "resolved": "2023-10-28T23:30:00-02:00",
        "has_time": True,
        "timezone": "America/Nuuk",
    }


def test_grammar_table_has_every_case() -> None:
    assert len(CASES) == 109


@pytest.mark.parametrize(
    ("phrase", "zone", "expected"), CASES, ids=[c[0] for c in CASES]
)
def test_resolve_datetime_grammar(
    phrase: str, zone: str, expected: tuple[str, object]
) -> None:
    result = resolve_datetime(phrase, zone)

    if expected[0] == "REFUSED":
        reason = expected[1]
        assert result["success"] is False
        assert result["tool_name"] == "resolve_datetime"
        assert result["resolution"] == reason
        assert reason in RESOLUTION_REASONS
        assert result["error"]
        assert "status" not in result
        assert ("supported" in result) == (reason == "unsupported_expression")
    else:
        resolved, has_time = expected
        assert result == {"resolved": resolved, "has_time": has_time, "timezone": zone}


def test_resolve_result_has_no_quotable_clock() -> None:
    assert set(ResolveDatetimeResult.model_fields) == {
        "resolved",
        "has_time",
        "timezone",
    }

    supported = resolve_datetime("soon", "UTC")["supported"]
    for line in supported:
        assert not re.search(r"\d{4}-\d{2}-\d{2}", line)
        assert not re.search(r"\d{4}", line)
    # What a refusal sends is exactly this published list, in this order;
    # test_reader_inventory_matches_the_grammar_lines below is what checks the
    # reader inventory and its count against this same list, which this
    # equality cannot.
    assert supported == list(GRAMMAR_FORMS)

    assert set(RESOLUTION_REASONS) == {
        "unsupported_expression",
        "ambiguous_date",
        "nonexistent_local_time",
        "ambiguous_local_time",
        "invalid_timezone",
    }


# Every reader function the module is expected to have: the seven functions
# in _READERS, the six in _LOWERCASE_READERS, and the calendar-date fallback
# the phrase reaches when none of those match -- fourteen in total.
_ALL_READERS: frozenset[Callable[[str, datetime], object]] = frozenset(
    {
        module._read_iso,
        module._read_en_calendar_date,
        module._read_zh_date,
        module._read_en_relative_day,
        module._read_zh_relative_day,
        module._read_en_shifted_weekday,
        module._read_en_weekday,
        module._read_zh_weekday,
        module._read_en_days_later,
        module._read_en_hours_later,
        module._read_zh_days_later,
        module._read_zh_hours_later,
        module._read_en_time,
        module._read_zh_time,
    }
)


def test_reader_inventory_matches_the_grammar_lines() -> None:
    """Checks two properties of the reader inventory, not which GRAMMAR_FORMS
    line a given reader owns -- that correspondence is not checked here.
    First, _READERS, _LOWERCASE_READERS and the calendar-date fallback the
    phrase reaches when none of those match together equal the fixed set
    _ALL_READERS, so a reader can be added or removed only by editing that
    set too. Second, the reader count is len(GRAMMAR_FORMS) + 1."""
    all_readers = (
        set(module._READERS)
        | set(module._LOWERCASE_READERS)
        | {module._read_en_calendar_date}
    )
    assert all_readers == _ALL_READERS
    # +1 because the bare-time grammar line is the only line served by two
    # readers, one per language: _read_en_time and _read_zh_time.
    assert len(_ALL_READERS) == len(GRAMMAR_FORMS) + 1


def test_resolve_datetime_rejects_sub_minute_offset() -> None:
    """Africa/Monrovia used a -00:44:30 offset before it standardised in
    1972, which RFC 3339 / JSON Schema 'date-time' cannot represent (only a
    ±HH:MM offset is legal). The value is refused rather than truncated or
    rounded into an approximation."""
    result = resolve_datetime("1 Jan 1970", "Africa/Monrovia")

    assert result["success"] is False
    assert result["resolution"] == "unsupported_expression"
    assert "fractional minute" in result["error"]
    assert "supported" not in result


@pytest.mark.parametrize(
    ("phrase", "zone", "expected"),
    [
        # The conversion to the UTC instant itself leaves the range: an hour
        # before 0001-01-01T00:00 is year zero, an hour after
        # 9999-12-31T23:59:59 is year ten thousand. No target zone avoids it.
        ("0001-01-01T00:00:00+01:00", "UTC", None),
        ("9999-12-31T23:59:59-01:00", "UTC", None),
        # The UTC instant is representable, but rendering it in the target
        # zone is not: Pacific/Kiritimati was 10:29:20 behind UTC in year one
        # and is 14:00 ahead today, so each end of the calendar falls out on
        # the far side.
        ("0001-01-01T00:00:00Z", "Pacific/Kiritimati", None),
        ("9999-12-31T23:59:59+01:00", "Pacific/Kiritimati", None),
        # Neither end is about the year written in the phrase: the same year
        # one and year 9999 phrases resolve when the instant they name stays
        # inside the range.
        ("0001-01-01T00:00:00-01:00", "UTC", "0001-01-01T01:00:00+00:00"),
        ("9999-12-31T23:59:59+01:00", "UTC", "9999-12-31T22:59:59+00:00"),
    ],
)
def test_resolve_datetime_refuses_instants_it_cannot_represent(
    phrase: str, zone: str, expected: object
) -> None:
    result = resolve_datetime(phrase, zone)

    if expected is None:
        assert result["success"] is False
        assert result["resolution"] == "unsupported_expression"
        # The phrase itself matched the grammar, so listing the grammar would
        # only misdirect a caller into rewriting an already-supported phrase.
        assert "supported" not in result
        assert result["error"]
    else:
        assert result["resolved"] == expected
        assert result["has_time"] is True


def test_both_conversion_paths_refuse_an_unrepresentable_instant_alike() -> None:
    """The offset-bearing path and the wall-clock path convert different
    things and fail in different places, but agree on the reason code: one
    failure class, one code. They differ on the grammar list: the
    offset-bearing refusal was ruled to withhold it, while the wall-clock
    path keeps the list it has always carried for this failure. Both phrases
    match the grammar, so the difference is not a property of the phrase --
    it is an asymmetry left in place because changing the wall-clock side
    would alter the public output of 269 reachable zone/date pairs in a
    review-fix commit."""
    aware = resolve_datetime("0001-01-01T00:00:00+01:00", "UTC")
    naive = resolve_datetime("0001-01-01", "Africa/Addis_Ababa")

    assert aware["resolution"] == naive["resolution"] == "unsupported_expression"
    assert "supported" not in aware
    assert "supported" in naive
    assert aware["success"] is naive["success"] is False


# The half-day-word compatibility table, typed out as literal hours rather
# than derived from the module: the test states the ruling, the module states
# the implementation, and the parametrization compares them cell by cell for
# all twenty-four hours of every period, including the no-word row.
# (period, hours the word shifts past noon, hours it keeps as written)
ZH_PERIOD_TABLE: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = [
    ("上午", (), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)),
    ("早上", (), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)),
    ("中午", (), (11, 12, 13)),
    ("下午", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), (13, 14, 15, 16, 17, 18)),
    ("晚上", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), (18, 19, 20, 21, 22, 23)),
    ("", (), tuple(range(24))),
]
# (period, hour as written, hour the tool must resolve to, or None to refuse)
ZH_PERIOD_CASES: list[tuple[str, int, object]] = [
    (
        period,
        hour,
        hour + 12 if hour in shifted else hour if hour in kept else None,
    )
    for period, shifted, kept in ZH_PERIOD_TABLE
    for hour in range(24)
]


def test_zh_period_table_has_every_cell() -> None:
    assert len(ZH_PERIOD_CASES) == 144


@pytest.mark.parametrize(
    ("period", "hour", "expected_hour"),
    ZH_PERIOD_CASES,
    ids=[f"{p or 'bare'}{h}" for p, h, _ in ZH_PERIOD_CASES],
)
def test_zh_half_day_word_hour_compatibility(
    period: str, hour: int, expected_hour: object
) -> None:
    result = resolve_datetime(f"{period}{hour}点", SHANGHAI)

    if expected_hour is None:
        assert result["success"] is False
        assert result["resolution"] == "unsupported_expression"
        assert result["error"]
    else:
        assert result == {
            "resolved": f"2026-09-15T{expected_hour:02d}:00:00+08:00",
            "has_time": True,
            "timezone": SHANGHAI,
        }


@pytest.mark.parametrize(
    ("phrase", "must_contain"),
    [
        ("上午0点", "write 0点 without a half-day word"),
        ("上午13点", "which names hours 1 to 11"),
        ("下午20点", "which names hours 1 to 11 or 13 to 18"),
        ("晚上15点", "which names hours 1 to 11 or 18 to 23"),
        # Hour twelve keeps its own wording, which names the three readings
        # it cannot choose between rather than a range of admitted hours.
        # Without all four cells the twelve-hour check can lose any subset of
        # its members unnoticed: those inputs then fall through to the
        # compatibility table, which also refuses them, so every reason-code
        # assertion stays green while the reason the caller is given changes.
        # The four words split into two guard branches (上午/早上 vs 下午/晚上),
        # so both must be pinned for either direction of that split to show.
        ("下午12点", "could mean today ending at"),
        ("晚上12点", "could mean today ending at"),
        ("上午12点", "could mean today ending at"),
        ("早上12点", "could mean today ending at"),
    ],
)
def test_zh_half_day_refusal_names_the_hours_the_word_admits(
    phrase: str, must_contain: str
) -> None:
    """The model reads this text: a refusal that does not say which hours the
    word admits leaves it guessing at a second wrong phrasing."""
    result = resolve_datetime(phrase, SHANGHAI)

    assert result["resolution"] == "unsupported_expression"
    assert must_contain in result["error"]


@pytest.mark.parametrize(
    ("phrase", "expected_resolution"),
    [
        ("周三上午13点", "unsupported_expression"),
        ("2026年9月20日 晚上0点", "unsupported_expression"),
    ],
)
def test_zh_half_day_rule_reaches_rows_other_than_a_bare_time(
    phrase: str, expected_resolution: str
) -> None:
    """The rule lives in the Chinese time sub-grammar, which the calendar-date,
    relative-day, weekday and bare-time rows all read their time through, so a
    contradictory hour is refused however the day was named."""
    result = resolve_datetime(phrase, SHANGHAI)

    assert result["success"] is False
    assert result["resolution"] == expected_resolution


def test_resolve_datetime_dst_and_zone() -> None:
    skipped = resolve_datetime("2026-10-04 02:30", SYDNEY)
    assert skipped["success"] is False
    assert skipped["resolution"] == "nonexistent_local_time"

    repeated = resolve_datetime("2026-04-05 02:30", SYDNEY)
    assert repeated["success"] is False
    assert repeated["resolution"] == "ambiguous_local_time"

    utc = resolve_datetime("tomorrow", "UTC")
    assert utc["resolved"] == "2026-09-16T00:00:00+00:00"
    assert utc["timezone"] == "UTC"


# (zone, day, what the zone does to that day's midnight,
#  date-only expectation, expectation when the phrase also names 00:30,
#  a substring the date-only refusal's error message must contain, or None
#  to check the reason code alone -- a row carrying None makes no claim at
#  all about the message text)
MIDNIGHT_TABLE: list[tuple[str, str, str, object, object, Optional[str]]] = [
    (
        "Australia/Sydney",
        "2026-09-15",
        "has it once",
        "2026-09-15T00:00:00+10:00",
        "2026-09-15T00:30:00+10:00",
        None,
    ),
    (
        "America/Santiago",
        "2026-09-06",
        "skips it",
        "2026-09-06T00:00:00-03:00",
        ("REFUSED", "nonexistent_local_time"),
        None,
    ),
    (
        "America/Havana",
        "2026-11-01",
        "repeats it",
        "2026-11-01T00:00:00-04:00",
        ("REFUSED", "ambiguous_local_time"),
        None,
    ),
    (
        # The scan that finds the day's first existing minute runs the whole
        # day, not a short prefix of it: this gap is 420 minutes, well past
        # an hour, so a scan cut short would refuse this day while every
        # other row here stayed green.
        "Antarctica/Vostok",
        "1994-11-01",
        "skips a gap longer than an hour",
        "1994-11-01T00:00:00+07:00",
        ("REFUSED", "nonexistent_local_time"),
        None,
    ),
    (
        # The gap here is 600 minutes, the longest one anywhere in tzdata
        # over any zone and year the scan can see. The Vostok row above
        # already catches a scan cut down to 60 minutes, but a scan bound
        # anywhere from 421 through 600 minutes would still pass that row
        # while silently refusing this one, so this is the row that catches
        # that remaining range.
        "Antarctica/Macquarie",
        "1948-03-25",
        "skips the longest midnight gap in tzdata",
        "1948-03-25T00:00:00+10:00",
        ("REFUSED", "nonexistent_local_time"),
        None,
    ),
    (
        "Pacific/Kiritimati",
        "1994-12-31",
        "skips the whole day",
        ("REFUSED", "nonexistent_local_time"),
        ("REFUSED", "nonexistent_local_time"),
        "moved across the date line and skipped the whole day",
    ),
    (
        "Africa/Johannesburg",
        "0001-01-01",
        "cannot be converted at all",
        ("REFUSED", "unsupported_expression"),
        ("REFUSED", "unsupported_expression"),
        None,
    ),
]


@pytest.mark.parametrize(
    (
        "zone",
        "day",
        "behavior",
        "date_only_expected",
        "with_time_expected",
        "date_only_error_substring",
    ),
    MIDNIGHT_TABLE,
    ids=[f"{zone}:{behavior}" for zone, _, behavior, _, _, _ in MIDNIGHT_TABLE],
)
def test_date_only_phrase_across_midnight_behavior(
    zone: str,
    day: str,
    behavior: str,
    date_only_expected: object,
    with_time_expected: object,
    date_only_error_substring: Optional[str],
) -> None:
    """A phrase naming only a day is answered for that day whichever way the
    zone treats its midnight -- has it once, skips it, repeats it, or (a
    date-line crossing) skips the whole calendar day -- or is refused
    unsupported_expression where the conversion cannot be done at all. A
    phrase that also names 00:30 on the same day keeps the wall-clock guard's
    existing refusals unchanged: only the date-only reading is new."""
    date_only_result = resolve_datetime(day, zone)
    with_time_result = resolve_datetime(f"{day} 00:30", zone)

    if isinstance(date_only_expected, tuple):
        assert date_only_result["success"] is False, behavior
        assert date_only_result["resolution"] == date_only_expected[1]
        if date_only_error_substring is not None:
            assert date_only_error_substring in date_only_result["error"]
    else:
        assert date_only_result["resolved"] == date_only_expected
        assert date_only_result["has_time"] is False
        # The time part of a date-only answer is this tool's own filler, not
        # something the user wrote, whichever branch produced the instant.
        assert date_only_result["resolved"][10:].startswith("T00:00:00")

    if isinstance(with_time_expected, tuple):
        assert with_time_result["success"] is False
        assert with_time_result["resolution"] == with_time_expected[1]
    else:
        assert with_time_result["resolved"] == with_time_expected
        assert with_time_result["has_time"] is True


_SANTIAGO = "America/Santiago"


# This zone and day are the same cell as the America/Santiago row in
# MIDNIGHT_TABLE above, but the two pins check different things. That row
# checks what the zone does to a midnight it skips, reached through the ISO
# literal alone. The parametrized test below checks that every date-only
# grammar form -- not just the ISO literal -- reaches this same cell, so a
# reader that mis-parses one phrasing does not go unnoticed just because the
# ISO literal still works. Both pins stay.
@pytest.mark.parametrize(
    ("phrase", "now_utc"),
    [
        ("tomorrow", datetime(2026, 9, 5, 16, 0, 0, tzinfo=timezone.utc)),
        ("next sunday", datetime(2026, 9, 5, 16, 0, 0, tzinfo=timezone.utc)),
        ("sunday", datetime(2026, 9, 5, 16, 0, 0, tzinfo=timezone.utc)),
        ("in 3 days", datetime(2026, 9, 3, 16, 0, 0, tzinfo=timezone.utc)),
        ("明天", datetime(2026, 9, 5, 16, 0, 0, tzinfo=timezone.utc)),
        ("3天后", datetime(2026, 9, 3, 16, 0, 0, tzinfo=timezone.utc)),
        ("6 Sep 2026", datetime(2026, 9, 6, 16, 0, 0, tzinfo=timezone.utc)),
        ("2026-09-06", datetime(2026, 9, 6, 16, 0, 0, tzinfo=timezone.utc)),
        ("2026年9月6日", datetime(2026, 9, 6, 16, 0, 0, tzinfo=timezone.utc)),
    ],
)
def test_every_date_only_grammar_line_answers_a_skipped_midnight(
    monkeypatch: pytest.MonkeyPatch, phrase: str, now_utc: datetime
) -> None:
    """America/Santiago skips 2026-09-06's midnight at a daylight-saving
    change. Each of the nine date-only grammar rows reaches that day through
    a different reader (ISO, English or Chinese calendar date, a relative
    day, a weekday, or an N-days-later count), so this pins all nine rather
    than only the ISO literal, which is the row the underlying defect would
    otherwise leave unexercised."""
    monkeypatch.setattr(module, "_now", lambda: now_utc)

    result = resolve_datetime(phrase, _SANTIAGO)

    assert result == {
        "resolved": "2026-09-06T00:00:00-03:00",
        "has_time": False,
        "timezone": _SANTIAGO,
    }


# Unlike FROZEN, this clock's own seconds and microseconds are non-zero: the
# form this pins (a duration counted from now) must drop them, which a clock
# permanently parked on :00 could never catch.
FROZEN_WITH_SECONDS = datetime(2026, 9, 15, 2, 30, 37, 123456, tzinfo=timezone.utc)


def test_duration_later_forms_land_on_whole_minute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'in N hours/minutes' and its Chinese equivalents name a duration from
    now, not a second within this minute, so the answer must land on :00
    even when the clock read at call time carries live seconds. A phrase
    this form does not touch (3pm) is unaffected by the same clock."""
    monkeypatch.setattr(module, "_now", lambda: FROZEN_WITH_SECONDS)

    for phrase in ("in 2 hours", "in 30 minutes", "两小时后", "30分钟后"):
        result = resolve_datetime(phrase, SYDNEY)
        assert result["resolved"][17:19] == "00", (phrase, result["resolved"])

    unaffected = resolve_datetime("3pm", SYDNEY)
    assert unaffected["resolved"] == "2026-09-15T15:00:00+10:00"


def test_refusal_rejects_a_code_outside_the_published_set() -> None:
    """RESOLUTION_REASONS is the published contract a caller can act on: a
    refusal code outside it would put a word into the model's context that
    no caller can do anything with, so _refusal fails loudly on one instead
    of shipping it, while every published reason still builds normally."""
    with pytest.raises(ValueError):
        module._refusal("bogus_code", "x")
    for reason in RESOLUTION_REASONS:
        assert module._refusal(reason, "x")["resolution"] == reason


@pytest.mark.parametrize("zone", ["EST", "Sydney", "Etc/GMT+10"])
def test_resolve_datetime_rejects_non_region_city_zone(zone: str) -> None:
    result = resolve_datetime("tomorrow", zone)

    assert result["success"] is False
    assert result["resolution"] == "invalid_timezone"
    assert "supported" not in result


def test_phrase_zone_name_is_refused() -> None:
    result = resolve_datetime("15 Sep 2026 10:00 EST", SYDNEY)

    assert result["success"] is False
    assert result["resolution"] == "unsupported_expression"


def test_hour_twelve_refusal_is_the_same_with_and_without_a_date() -> None:
    """The refusal is a property of writing hour twelve with am or pm, not of
    which reader saw the phrase: the calendar path and the bare-time path
    must answer with the same reason code and the same words, in either
    letter case dateutil accepts. On the calendar-date path, the same
    wording must also survive the other spellings _EN_HOUR_TWELVE_RE
    recognises -- but only for the spellings that reach that guard at all: a
    spelling the whole-phrase calendar pattern does not admit ("12h30 pm")
    never gets there. That phrase fails the whole-phrase match itself and is
    refused generically at the door, not by falling through the hour-twelve
    guard unrecognised. Those out-of-grammar spellings have no bare-path
    reading to match either: the bare grammar requires a colon, so it never
    parses an hour from them at all."""
    bare = resolve_datetime("12 pm", SYDNEY)
    dated = resolve_datetime("15 Sep 2026 12 pm", SYDNEY)
    dated_upper = resolve_datetime("15 Sep 2026 12 PM", SYDNEY)

    assert bare["resolution"] == dated["resolution"] == "unsupported_expression"
    assert bare["error"] == dated["error"] == dated_upper["error"]

    for spelling in ("12.30 pm", "12 p.m.", "012 pm"):
        assert (
            resolve_datetime(f"15 Sep 2026 {spelling}", SYDNEY)["error"]
            == "'12pm' names both midnight and noon"
        )

    # "12h30 pm" never reaches dateutil at all: the "h" separator is not one
    # of the shapes the whole-phrase calendar pattern admits, so the phrase
    # fails that match and is refused at the door, generically -- the
    # hour-twelve wording above is not promised for every spelling dateutil
    # would otherwise have accepted.
    assert (
        resolve_datetime("15 Sep 2026 12h30 pm", SYDNEY)["error"]
        == "unsupported date or time expression: '15 Sep 2026 12h30 pm'"
    )


# All eleven spellings the ruling names as having to keep the *named*
# hour-twelve wording, not just its reason code: (phrase, expected error).
# Pinning only the reason code (as CASES does) would let any of these
# silently drop to the generic wording with a green suite -- confirmed by
# mutating away the ":SS" group of the 24-hour clock branch, which does
# exactly that to the seventh entry below.
EN_HOUR_TWELVE_NAMED_WORDING_CASES: list[tuple[str, str]] = [
    ("15 Sep 2026 12 pm", "'12pm' names both midnight and noon"),
    ("15 Sep 2026 12 am", "'12am' names both midnight and noon"),
    ("15 Sep 2026 12:30 pm", "'12pm' names both midnight and noon"),
    ("15 Sep 2026 12:30 am", "'12am' names both midnight and noon"),
    ("Sep 15 2026 12pm", "'12pm' names both midnight and noon"),
    ("January 1st, 1990 12 am", "'12am' names both midnight and noon"),
    ("1 Jan 1990 12:00:00 pm", "'12pm' names both midnight and noon"),
    ("15 Sep 2026 12.30 pm", "'12pm' names both midnight and noon"),
    ("15 Sep 2026 12 p.m.", "'12pm' names both midnight and noon"),
    ("15 Sep 2026 012 pm", "'12pm' names both midnight and noon"),
    ("15 Sep 2026 12 PM", "'12pm' names both midnight and noon"),
]


@pytest.mark.parametrize(
    ("phrase", "expected_error"),
    EN_HOUR_TWELVE_NAMED_WORDING_CASES,
    ids=[phrase for phrase, _ in EN_HOUR_TWELVE_NAMED_WORDING_CASES],
)
def test_hour_twelve_named_wording_for_every_ruled_spelling(
    phrase: str, expected_error: str
) -> None:
    result = resolve_datetime(phrase, SYDNEY)

    assert result["error"] == expected_error


@pytest.mark.parametrize(
    "phrase",
    ["15 Sep 2026 10 A.M", "15 Sep 2026 11 P.M."],
)
def test_zone_misread_am_pm_spelling_is_refused_at_every_hour(phrase: str) -> None:
    """The zone-name callback, not the whole-phrase pattern, is what refuses
    these two spellings: dateutil reads the isolated uppercase "M" as a
    candidate timezone name at any hour, not only at twelve, so
    _reject_zone_in_phrase raises regardless of what hour precedes it. Their
    lowercase twins ("10 a.m", "11 p.m.") resolve normally -- this is the
    same phrase, capitalisation only, reaching a different outcome."""
    result = resolve_datetime(phrase, SYDNEY)

    assert result["success"] is False
    assert result["error"] == f"unsupported date or time expression: {phrase!r}"


@pytest.mark.parametrize(
    "phrase",
    [
        "15 Sep 2026 10.30 pm",
        "15 Sep 2026 10.05 am",
        "15 Sep 2026 10:30.15 pm",
    ],
)
def test_calendar_date_dot_minute_separator_is_refused_off_twelve(
    phrase: str,
) -> None:
    """The "." minute separator survives only for a literal twelve (the
    third alternative of _EN_CALENDAR_CLOCK): at any other hour dateutil
    reads the digits after the dot as something other than a minute count --
    "10.30 pm" would silently drop the 30, "10:30.15 pm" would silently drop
    the 15 -- so the whole phrase is refused off twelve rather than risk
    returning a value it never wrote."""
    result = resolve_datetime(phrase, SYDNEY)

    assert result["success"] is False
    assert result["error"] == f"unsupported date or time expression: {phrase!r}"


def test_calendar_date_dot_minute_separator_still_names_hour_twelve() -> None:
    result = resolve_datetime("15 Sep 2026 12.30 pm", SYDNEY)

    assert result["error"] == "'12pm' names both midnight and noon"


def test_implicit_pm_hour_zero_refuses_with_the_same_shape_as_the_bare_form() -> None:
    """dateutil folds "0:30 am" to hour zero -- the same folded hour an
    hour-twelve reading produces, but this phrase never spells a twelve. The
    bare time sub-grammar already refuses it generically (_en_time's
    `if not 1 <= hour <= 12: return None`), so the calendar-date reader must
    land on that same generic shape -- reason unsupported_expression, the
    grammar list attached -- instead of naming an hour the phrase never
    wrote."""
    for phrase in ("0:30 am", "15 Sep 2026 0:30 am"):
        result = resolve_datetime(phrase, SYDNEY)

        assert result["resolution"] == "unsupported_expression"
        assert result["error"] == f"unsupported date or time expression: {phrase!r}"
        assert "supported" in result


def _en_period_spellings() -> list[str]:
    """Every dot/space/letter combination the am/pm half of an English time
    can be spelled, doubled for letter case: a/p, an optional ".", optional
    whitespace, an optional "m", and another optional "."."""
    letters = ("a", "p")
    seen = {
        f"{letter}{dot1}{space}{m}{dot2}"
        for letter, dot1, space, m, dot2 in itertools.product(
            letters, ("", "."), ("", " "), ("", "m"), ("", ".")
        )
    }
    return sorted(seen | {spelling.upper() for spelling in seen})


EN_PERIOD_SPELLINGS = _en_period_spellings()
# The spellings that write the letter and the "m" with no whitespace between
# them: the same set _EN_HOUR_TWELVE_RE recognises, and (with one verified
# exception below) the ones that reach the hour-twelve guard rather than the
# whole-phrase door refusing them first.
_EN_PERIOD_CONNECTED_RE = re.compile(r"[ap]\.?m\.?", re.IGNORECASE)
# Four of the sixteen connected spellings never reach the hour-twelve guard
# at all: dateutil.parser.parse reads a solitary uppercase "M" split from its
# letter by a period as a candidate military-timezone name rather than the
# second half of a meridian marker -- lowercase "m" is never read that way,
# confirmed directly by calling dateutil with a tzinfos probe ("A.M" invokes
# it with name="M"; "a.m" invokes it with (None, None)). That routes these
# four through the pre-existing _reject_zone_in_phrase rejection, which
# resolve_datetime's generic `except (ValueError, OverflowError)` turns into
# the same generic refusal every out-of-grammar spelling gets, before
# dateutil ever finishes parsing an hour to check. The phrase is still
# refused either way; only the wording differs.
_EN_PERIOD_ZONE_MISREAD = {"A.M", "A.M.", "P.M", "P.M."}
EN_PERIOD_CASES: list[tuple[str, str]] = [
    (sep, spelling) for spelling in EN_PERIOD_SPELLINGS for sep in ("", " ")
]


def test_en_period_spellings_table_has_every_cell() -> None:
    assert len(EN_PERIOD_SPELLINGS) == 60
    assert len(EN_PERIOD_CASES) == 120


@pytest.mark.parametrize(
    ("sep", "spelling"),
    EN_PERIOD_CASES,
    ids=[f"{spelling}|sep={sep!r}" for sep, spelling in EN_PERIOD_CASES],
)
def test_calendar_hour_twelve_am_pm_spelling_never_succeeds(
    sep: str, spelling: str
) -> None:
    """Hour twelve with am or pm is refused for every dot/space/case spelling
    of the period word, not only the ones the frozen-design examples use:
    the refusal must not depend on which sixteen spellings anyone thought to
    write down. The sixteen spellings that write the letter and the "m" with
    no whitespace between them reach the hour-twelve guard and get its named
    wording, except the four in _EN_PERIOD_ZONE_MISREAD, which are
    intercepted earlier by the pre-existing zone-name rejection and get its
    generic wording instead (pinned exactly, not merely skipped, so a
    dateutil upgrade that changed the quirk would be noticed); every other
    spelling never reaches the guard at all -- the whole-phrase calendar
    pattern does not admit it, so it gets the generic refusal instead -- but
    still never succeeds."""
    phrase = f"15 Sep 2026 12{sep}{spelling}"
    result = resolve_datetime(phrase, SYDNEY)

    assert result["success"] is False
    if spelling in _EN_PERIOD_ZONE_MISREAD:
        assert result["error"] == f"unsupported date or time expression: {phrase!r}"
    elif _EN_PERIOD_CONNECTED_RE.fullmatch(spelling):
        period = "pm" if spelling[0].lower() == "p" else "am"
        assert result["error"] == f"'12{period}' names both midnight and noon"


_EN_WEEKDAY_CALENDAR_WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_EN_WEEKDAY_CALENDAR_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_EN_WEEKDAY_CALENDAR_YEARS = (2025, 2026, 2027)
EN_WEEKDAY_CALENDAR_PHRASES: list[str] = [
    f"{weekday} {month} {year}"
    for weekday in _EN_WEEKDAY_CALENDAR_WEEKDAYS
    for month in _EN_WEEKDAY_CALENDAR_MONTHS
    for year in _EN_WEEKDAY_CALENDAR_YEARS
]


def test_en_weekday_calendar_phrases_table_has_every_cell() -> None:
    assert len(EN_WEEKDAY_CALENDAR_PHRASES) == 252


@pytest.mark.parametrize("phrase", EN_WEEKDAY_CALENDAR_PHRASES)
def test_calendar_date_with_a_weekday_name_never_resolves(phrase: str) -> None:
    """A weekday name is not one of the day-month-year / month-day-year /
    slash-or-dash calendar shapes this grammar row declares. Without the
    whole-phrase gate, dateutil would fold the day the phrase never wrote
    onto the nearest date matching that weekday relative to whichever
    default it was parsed against, and the two defaults' folded days would
    then agree with each other by construction -- so the missing-component
    check that would normally catch a phrase with no day never fires."""
    result = resolve_datetime(phrase, SYDNEY)

    assert "resolved" not in result


@pytest.mark.parametrize(
    "phrase",
    [
        "Monday 15 Sep 2026",
        "Tuesday 15 Sep 2026",
        "Mon 15 Sep 2026",
        "15 Sep 2026 Monday",
        "Monday, 15 Sep 2026",
    ],
)
def test_calendar_date_with_a_weekday_name_and_a_day_is_refused(phrase: str) -> None:
    result = resolve_datetime(phrase, SYDNEY)

    assert result["success"] is False
    assert result["resolution"] == "unsupported_expression"


def test_calendar_date_weekday_only_does_not_invent_the_seventh() -> None:
    """Without the whole-phrase gate, "Monday Sep 2026" parses against
    _DEFAULT_A (2000-01-01) to 2026-09-07, the Monday nearest that default's
    own day -- a date the phrase itself never wrote. The gate must keep that
    date out of the result entirely, not merely out of a success reading."""
    result = resolve_datetime("Monday Sep 2026", SYDNEY)

    assert "2026-09-07" not in str(result)


@pytest.mark.parametrize(
    "phrase",
    [
        "15 Sep 2026 10:00:00.5",
        "15 Sep 2026 10:00:00,5",
        "15 Sep 2026 10:00:00.123456",
        "15 Sep 2026 10:00:00.",
    ],
)
def test_calendar_date_fractional_seconds_are_refused(phrase: str) -> None:
    """A fractional-seconds component is a component the phrase itself
    carried; dropping it silently would misreport the instant, so the whole
    phrase is refused rather than rounded or truncated."""
    result = resolve_datetime(phrase, SYDNEY)

    assert result["success"] is False
    assert result["resolution"] == "unsupported_expression"
    assert result["error"] == f"unsupported date or time expression: {phrase!r}"


def test_mixed_separator_numeric_dates_read_like_consistent_ones() -> None:
    """`12/05-2026` passes the whole-phrase gate: the numeric date branch
    writes its two separators as independent character classes. No value is
    wrong -- a mixed-separator phrase lands on exactly the same answer as the
    same phrase written with one separator -- so this pins the acceptance
    rather than narrowing it."""
    assert (
        resolve_datetime("25/12-1990", "UTC")["resolved"]
        == resolve_datetime("25/12/1990", "UTC")["resolved"]
        == "1990-12-25T00:00:00+00:00"
    )
    assert (
        resolve_datetime("25-12/1990", "UTC")["resolved"] == "1990-12-25T00:00:00+00:00"
    )
    assert (
        resolve_datetime("12/05-2026", "UTC")["resolution"]
        == resolve_datetime("12/05/2026", "UTC")["resolution"]
        == "ambiguous_date"
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (ToolSelectionSpec.from_raw(tool_categories=None), 1),  # ALL
        (ToolSelectionSpec.from_raw(tool_categories=["web_search"]), 1),  # non-basic
        (ToolSelectionSpec.from_raw(tool_categories=[]), 0),  # explicit NONE
    ],
)
async def test_resolve_datetime_is_intrinsic(
    spec: ToolSelectionSpec, expected: int
) -> None:
    """The full factory pipeline assembles exactly one usable
    resolve_datetime for any non-NONE selection -- including one that
    never picks the basic category -- and none for an explicit NONE."""
    tools = await ToolFactory.create_all_tools(
        _FakeConfig(spec), apply_user_override_filter=False
    )

    assert [t.name for t in tools].count("resolve_datetime") == expected


async def test_resolve_datetime_creator_is_skipped_for_explicit_none() -> None:
    """The registry gate must not even build the intrinsic tool for an
    explicit zero-tools agent: the NONE contract wins over always-on."""
    spec = ToolSelectionSpec.from_raw(tool_categories=[])

    tools = await ToolRegistry.create_registered_tools(_FakeConfig(spec))

    assert "resolve_datetime" not in [getattr(t, "name", None) for t in tools]


def test_tool_declares_read_only_other_identity() -> None:
    metadata = ResolveDatetimeTool().metadata

    assert metadata.name == "resolve_datetime"
    assert metadata.read_only is True
    assert metadata.concurrency_safe is True
    assert metadata.category is ToolCategory.OTHER


def test_time_tools_cross_reference() -> None:
    description = ResolveDatetimeTool().description

    assert "get_current_time" in description
    assert "validate_local_time" in description
    assert "resolve_datetime" in CurrentTimeTool().description
    assert "resolve_datetime" in ValidateLocalTimeTool().description
    # The description promises only what the engine does at this point: no
    # verbatim-span refusal and no instruction to quote the result as a source.
    assert "refused when that text" not in description
    assert "Quote the result" not in description


def test_tool_runs_through_the_json_surface() -> None:
    tool = ResolveDatetimeTool()

    assert tool.run_json_sync({"phrase": "tomorrow at 3pm", "timezone": SYDNEY}) == {
        "resolved": "2026-09-16T15:00:00+10:00",
        "has_time": True,
        "timezone": SYDNEY,
    }

    refused = asyncio.run(tool.run_json_async({"phrase": "soon", "timezone": "UTC"}))
    assert refused["resolution"] == "unsupported_expression"

    with pytest.raises(ValidationError):
        tool.run_json_sync({"phrase": "tomorrow"})
