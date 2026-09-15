from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

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
    assert len(CASES) == 86


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
    assert len(supported) == len(GRAMMAR_FORMS)

    assert set(RESOLUTION_REASONS) == {
        "unsupported_expression",
        "ambiguous_date",
        "nonexistent_local_time",
        "ambiguous_local_time",
        "invalid_timezone",
    }


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
    letter case dateutil accepts. On the calendar-date path, which is the
    only path that pattern is consulted on, the same wording must also
    survive the other spellings the widened _EN_HOUR_TWELVE_RE recognises;
    a spelling it does not recognise ("12h30 pm") must fall back to the
    generic refusal there rather than silently losing its wording. Those
    out-of-grammar spellings have no bare-path reading to match: the bare
    grammar requires a colon, so it never parses an hour from them at
    all."""
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

    # "12h30 pm" folds to hour 12 and spells a twelve, but the "h" separator
    # is not one of the spellings _EN_HOUR_TWELVE_RE recognises, so it gets
    # the generic refusal instead: the hour-twelve wording above is not
    # promised for every spelling dateutil accepts.
    assert (
        resolve_datetime("15 Sep 2026 12h30 pm", SYDNEY)["error"]
        == "unsupported date or time expression: '15 Sep 2026 12h30 pm'"
    )


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
