import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Callable, Literal, Mapping, NoReturn, Optional, Type
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from dateutil import parser as dateutil_parser
from pydantic import BaseModel, Field

from .....web.tools.config import WebToolConfig
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool

_STAMP = "%Y-%m-%d %H:%M:%S"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CurrentTimeArgs(BaseModel):
    # Field name is the wire contract; it locally shadows the datetime.timezone
    # import, which this class body does not use.
    timezone: Optional[str] = Field(
        default=None,
        description=(
            "IANA timezone name to report local time in, for example "
            "'Australia/Melbourne'. Copy it from the zone named in the system "
            "prompt's date-and-time line. Omit it when that line reports UTC "
            "only. An unresolvable name falls back to UTC, and the 'timezone' "
            "field of the result says which zone was actually used."
        ),
    )


class CurrentTimeResult(BaseModel):
    utc: str = Field(description="Current UTC time, as YYYY-MM-DD HH:MM:SS.")
    local: str = Field(
        description=(
            "Current time in the reported zone, as YYYY-MM-DD HH:MM:SS. Equal "
            "to 'utc' when the zone is UTC."
        )
    )
    # Field name is the wire contract; it locally shadows the datetime.timezone
    # import, which this class body does not use.
    timezone: str = Field(
        description=(
            "Zone the 'local' field is expressed in. 'UTC' when none was "
            "supplied or the supplied name could not be resolved."
        )
    )
    utc_offset: str = Field(
        description="Offset of the reported zone from UTC, for example '+10:00'."
    )


@lru_cache(maxsize=1)
def _zone_by_lowercase() -> dict[str, str]:
    return {name.lower(): name for name in available_timezones()}


def _resolve_zone(name: Optional[str]) -> Optional[ZoneInfo]:
    # Always resolve through the canonical map rather than ZoneInfo(name)
    # directly: ZoneInfo.key echoes whatever string it was given, and
    # tzdata lookup is case-sensitive on Linux but not on macOS, so the
    # reported zone would otherwise vary by input case and filesystem.
    # The map guarantees a canonical key (e.g. "australia/melbourne" ->
    # "Australia/Melbourne") and identical behavior across platforms.
    # An unusable name comes from the model, so it degrades to UTC.
    if not isinstance(name, str) or not name.strip():
        return None
    canonical = _zone_by_lowercase().get(name.strip().lower())
    if canonical is None:
        return None
    try:
        return ZoneInfo(canonical)
    except (ZoneInfoNotFoundError, ValueError, OSError, KeyError):
        return None


def _format_offset(offset: timedelta) -> str:
    # Seconds are shown only when non-zero: dropping them would contradict the
    # local/utc pair reported beside this string. Liberia ran -00:44:30 until
    # 1972, so it is reachable from an ordinary historical date, not just LMT.
    sign = "-" if offset < timedelta(0) else "+"
    hours, remainder = divmod(abs(offset), timedelta(hours=1))
    minutes, remainder = divmod(remainder, timedelta(minutes=1))
    seconds = remainder // timedelta(seconds=1)
    stamp = f"{sign}{hours:02d}:{minutes:02d}"
    return f"{stamp}:{seconds:02d}" if seconds else stamp


def current_time(timezone_name: Optional[str] = None) -> CurrentTimeResult:
    """Read the wall clock now, in UTC and optionally in a named zone."""
    now_utc = _now()
    zone = _resolve_zone(timezone_name)
    local = now_utc.astimezone(zone) if zone is not None else now_utc
    # local is always aware, so utcoffset() is never None; the fallback only
    # satisfies the type checker.
    offset = local.utcoffset()
    return CurrentTimeResult(
        utc=now_utc.strftime(_STAMP),
        local=local.strftime(_STAMP),
        timezone=zone.key if zone is not None else "UTC",
        utc_offset=_format_offset(offset if offset is not None else timedelta(0)),
    )


class CurrentTimeTool(AbstractBaseTool):
    """Answers 'what time is it now', which the system prompt cannot."""

    # OTHER (not BASIC) keeps it out of the builder's category picker: it is
    # intrinsic (selection_gate="intrinsic"), always on for any non-NONE agent,
    # so presenting a togglable "basic" category that cannot actually disable it
    # would mislead. OTHER is in AGENT_CONFIG_UNASSIGNABLE_CATEGORIES.
    category = ToolCategory.OTHER
    read_only = True  # reads a clock ⇒ concurrency-safe

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return (
            "Return the real current time. The date and time in the system "
            "prompt is stamped once when the turn begins and does not advance "
            "while the turn runs, so call this whenever the answer depends on "
            "the time now rather than on when the turn started: measuring how "
            "long something took, checking whether a deadline has passed, or "
            "resolving a relative date during a turn that may have crossed "
            "midnight. Pass the timezone named in the system prompt's "
            "date-and-time line to get local time alongside UTC. This reports "
            "only the present moment: to turn some other local date and time "
            "into UTC, or to check whether one exists in a zone that changes "
            "its clocks, call the validate_local_time tool if it is available. "
            "To turn a date or time the user wrote in words into an exact "
            "date-time, call the resolve_datetime tool if it is available."
        )

    def args_type(self) -> Type[BaseModel]:
        return CurrentTimeArgs

    def return_type(self) -> Type[BaseModel]:
        return CurrentTimeResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        parsed = CurrentTimeArgs.model_validate(args)
        return current_time(parsed.timezone).model_dump()

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return self.run_json_sync(args)


@register_tool(selection_gate="intrinsic")
async def create_current_time_tool(config: WebToolConfig) -> list[AbstractBaseTool]:
    """Create the current-time tool."""
    return [CurrentTimeTool()]


class ValidateLocalTimeArgs(BaseModel):
    local_time: str = Field(
        description=(
            "The local wall-clock date-time to check, as 'YYYY-MM-DD HH:MM' or "
            "'YYYY-MM-DD HH:MM:SS' (a space or 'T' between date and time). A "
            "reading of the clock in the named zone, so it carries no offset "
            "and no trailing 'Z'. The time of day is required."
        )
    )
    # Field name is the wire contract; it locally shadows the datetime.timezone
    # import, which this class body does not use.
    timezone: str = Field(
        description=(
            "IANA zone name in Region/City form, for example "
            "'Australia/Sydney'. When the question does not name a zone, copy "
            "the one from the system prompt's date-and-time line rather than "
            "the example here. An abbreviation ('EST'), a bare city "
            "('Sydney'), an 'Etc/*' name, or an unknown name is an error, not "
            "a fallback to UTC."
        )
    )


class LocalTimeMapping(BaseModel):
    local: str = Field(description="The local wall-clock time.")
    abbreviation: str = Field(
        description="Zone abbreviation at this instant, e.g. 'AEST' or '+1030'."
    )
    utc_offset: str = Field(description="Offset from UTC at this instant.")
    utc: str = Field(description="The matching UTC time.")


class ValidateLocalTimeResult(BaseModel):
    # Not named 'status': that key is the framework's tool-result control
    # channel (see core/agent/result.py), where 'error' fails the call.
    local_time_status: Literal["unique", "nonexistent", "ambiguous"] = Field(
        description="Whether the local time exists once, not at all, or twice."
    )
    mappings: list[LocalTimeMapping] = Field(
        description="Valid UTC instants for this local time, earliest first."
    )
    # Field name is the wire contract; it locally shadows the datetime.timezone
    # import, which this class body does not use.
    timezone: str = Field(description="The resolved (case-normalized) zone name.")


def _stamp(moment: datetime) -> str:
    # isoformat, not strftime: %Y is platform-delegated and glibc renders a
    # year below 1000 unpadded, which an eastward offset reaches in 'utc'
    # even for a 4-digit local year.
    return moment.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


def _mapping(local: datetime, utc: datetime) -> LocalTimeMapping:
    offset = local.utcoffset()
    return LocalTimeMapping(
        local=_stamp(local),
        abbreviation=local.tzname() or "",
        utc_offset=_format_offset(offset if offset is not None else timedelta(0)),
        utc=_stamp(utc),
    )


def _instants_for_wall_time(
    naive: datetime, zone: ZoneInfo, wall_text: str
) -> list[datetime]:
    """Every instant this wall time names in this zone, earliest first.

    PEP 495: fold 0 and 1 are the two candidate readings. Each is kept only
    if it converts back to the requested wall time -- a candidate that does
    not is the zone telling us this reading does not exist. Trusting one
    fold's offset instead would misreport zones whose fold=0 reading returns
    an offset they never have (America/Nuuk under tzdata 2025c).

    The list is empty for a wall time the zone skips at a daylight-saving
    change, holds one entry for an ordinary wall time, and two where the
    zone repeats it. Each entry carries the zone and therefore the offset
    that applies at its own instant, so a caller needs no second conversion
    to learn it.

    A wall time at the edge of the calendar has no instant to name: the
    conversion leaves the range datetime can represent, and this raises
    ValueError naming wall_text, which is how the caller spelled the wall
    time. The caller passes that spelling rather than formatting it here
    because callers spell it differently: validate_local_time's caller wrote
    it, resolve_datetime's wall-clock branch formats its own moment, and
    _day_midnight_instants forwards that same formatted spelling on every
    minute it tries, not a fresh one per minute.
    """
    by_instant: dict[datetime, datetime] = {}
    for fold in (0, 1):
        local = naive.replace(fold=fold, tzinfo=zone)
        try:
            utc = local.astimezone(timezone.utc)
            round_trip = utc.astimezone(zone).replace(tzinfo=None)
        except (OverflowError, OSError) as exc:
            raise ValueError(
                f"local_time converts outside the representable date range: {wall_text!r}"
            ) from exc
        if round_trip == naive:
            by_instant.setdefault(utc, local)
    return [by_instant[utc] for utc in sorted(by_instant)]


def _require_region_city_zone(timezone_name: str) -> ZoneInfo:
    """Resolve a zone whose offset is part of the answer: Region/City or UTC only."""
    zone = _resolve_zone(timezone_name)
    if zone is None:
        raise ValueError(f"Unknown IANA timezone: {timezone_name!r}")
    # Region/City only (plus UTC). These are all real tzdata entries that do
    # not mean what a caller naming them means: 'EST' is a fixed -05:00 that
    # never observes DST, 'Etc/GMT+10' inverts the sign to -10:00, 'Factory'
    # is a placeholder. Here the offset IS the answer, so guessing wrong is
    # worse than refusing. get_current_time keeps accepting them: there a bad
    # zone only skews a displayed clock.
    if zone.key != "UTC" and ("/" not in zone.key or zone.key.startswith("Etc/")):
        raise ValueError(
            "timezone must be a Region/City IANA name such as "
            f"'Australia/Sydney' or 'UTC', not {timezone_name!r}"
        )
    return zone


def validate_local_time(local_time: str, timezone_name: str) -> ValidateLocalTimeResult:
    """Resolve a wall-clock time in a zone to every UTC instant it names."""
    zone = _require_region_city_zone(timezone_name)
    # Matched strictly rather than left to fromisoformat, which also accepts
    # forms this contract does not offer: a bare date would silently be read
    # as midnight, and on 3.14 '24:00' as the next day's midnight, answering
    # for a date the caller never named.
    text = local_time.strip()
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T]"
        r"(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?",
        text,
    ):
        raise ValueError(
            "local_time must be 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD HH:MM:SS', "
            "with no offset or 'Z'"
        )
    try:
        naive = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"local_time is not a real date and time: {text!r}") from exc

    instants = _instants_for_wall_time(naive, zone, text)
    mappings = [_mapping(local, local.astimezone(timezone.utc)) for local in instants]
    status: Literal["unique", "nonexistent", "ambiguous"] = (
        "nonexistent"
        if not mappings
        else "unique"
        if len(mappings) == 1
        else "ambiguous"
    )
    return ValidateLocalTimeResult(
        local_time_status=status, mappings=mappings, timezone=zone.key
    )


class ValidateLocalTimeTool(AbstractBaseTool):
    """Resolves a given wall-clock time to UTC from tzdata, which the model
    must not infer: daylight-saving gaps, overlaps, offsets and abbreviations
    are all read from the zone database."""

    category = ToolCategory.OTHER
    read_only = True  # pure tzdata lookup ⇒ concurrency-safe

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "validate_local_time"

    @property
    def description(self) -> str:
        return (
            "Check whether a specific local date-time exists in an IANA "
            "timezone and which UTC instants it maps to. Call this instead of "
            "working out a daylight-saving change yourself. When clocks spring "
            "forward the local time is skipped and no UTC instant matches "
            "(local_time_status 'nonexistent', no mappings); when they go back "
            "it occurs twice ('ambiguous', two mappings). Quote the offset and "
            "abbreviation from each mapping rather than recalling which one the "
            "zone uses in a given season, and note that only the local clock "
            "jumps at such a change -- UTC itself runs continuously. For the "
            "time right now use get_current_time instead. To resolve a spoken "
            "or relative expression such as 'tomorrow at 3pm' use "
            "resolve_datetime."
        )

    def args_type(self) -> Type[BaseModel]:
        return ValidateLocalTimeArgs

    def return_type(self) -> Type[BaseModel]:
        return ValidateLocalTimeResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        parsed = ValidateLocalTimeArgs.model_validate(args)
        return validate_local_time(parsed.local_time, parsed.timezone).model_dump()

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return self.run_json_sync(args)


@register_tool(selection_gate="intrinsic")
async def create_validate_local_time_tool(
    config: WebToolConfig,
) -> list[AbstractBaseTool]:
    """Create the local-time validation tool."""
    return [ValidateLocalTimeTool()]


RESOLUTION_REASONS: frozenset[str] = frozenset(
    {
        "unsupported_expression",
        "ambiguous_date",
        "nonexistent_local_time",
        "ambiguous_local_time",
        "invalid_timezone",
    }
)

# One line per grammar form, listed in a refusal so the caller can rephrase.
# Form names only: no date literal and no four-digit number may appear here,
# because the refusal is rendered to the model and must not offer a value
# it could quote as a date.
GRAMMAR_FORMS: tuple[str, ...] = (
    "ISO date YYYY-MM-DD, optionally with HH:MM[:SS] and a UTC offset",
    "an English calendar date, optionally with a time (e.g. day month year)",
    "a Chinese calendar date: <year>年<month>月<day>日, optionally with a "
    "Chinese time of day",
    "today / tomorrow / yesterday / day after tomorrow, optionally at a time",
    "今天 / 明天 / 后天 / 昨天 / 前天, optionally with a Chinese time of day",
    "next / last / this <weekday>, optionally at a time",
    "<weekday>, optionally at a time",
    "下周X / 上周X / 周X (also 星期, 礼拜), optionally with a Chinese time of day",
    "in N days / in N weeks",
    "in N hours / in N minutes",
    "N天后 / N周后",
    "N小时后 / N分钟后",
    "a bare time of day: H:MM, Ham/pm, or a Chinese time of day",
)


class ResolveDatetimeResult(BaseModel):
    resolved: str = Field(
        description=(
            "ISO 8601 date-time with UTC offset, e.g. YYYY-MM-DDTHH:MM:SS+HH:MM; "
            "the first ten characters are the calendar date."
        )
    )
    has_time: bool = Field(
        description=(
            "False when the phrase named a day only; resolved is then that "
            "day's 00:00:00 and its time part is not something the user said."
        )
    )
    # Field name is the wire contract; it locally shadows the datetime.timezone
    # import, which this class body does not use.
    timezone: str = Field(
        description="Canonical IANA zone key the offset was taken from."
    )


# A reading of the phrase before the zone is applied: either a naive local
# wall-clock (validated against the zone afterwards) or an aware instant
# (already exact, only converted into the zone). The flag says whether the
# phrase named a time of day.
_Reading = tuple[datetime, bool]

_WEEKDAYS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_WEEKDAY = "|".join(_WEEKDAYS)
_ZH_WEEKDAYS = "一二三四五六日天"
_ZH_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_ZH_NUMERAL = r"[一二两三四五六七八九十]{1,3}"
_ZH_TIME = (
    r"(?P<zh_period>上午|早上|中午|下午|晚上)?"
    rf"(?P<zh_hour>[0-9]{{1,2}}|{_ZH_NUMERAL})(?:点|时)"
    rf"(?:(?P<zh_half>半)|(?P<zh_minute>[0-9]{{1,2}}|{_ZH_NUMERAL})分)?"
)
_ZH_COUNT = rf"(?P<count>[0-9]+|{_ZH_NUMERAL})"
_EN_TIME = (
    r"(?P<en_hour>[0-9]{1,2})(?::(?P<en_minute>[0-9]{2}))? ?(?P<en_period>am|pm)?"
)
_EN_TIME_TAIL = rf"(?: (?:at )?{_EN_TIME})?"
_ZH_TIME_TAIL = rf"(?: ?{_ZH_TIME})?"

_ISO_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"(?:[ T](?P<clock>[0-9]{2}:[0-9]{2}(?::[0-9]{2})?)(?P<offset>Z|[+-][0-9]{2}:[0-9]{2})?)?"
)
_ZH_DATE_RE = re.compile(
    r"(?P<year>[0-9]{4})年(?P<month>[0-9]{1,2})月(?P<day>[0-9]{1,2})(?:日|号)"
    + _ZH_TIME_TAIL
)
_ZH_RELATIVE_DAYS = {
    "今天": 0,
    "今日": 0,
    "明天": 1,
    "明日": 1,
    "后天": 2,
    "昨天": -1,
    "昨日": -1,
    "前天": -2,
}
_ZH_RELATIVE_RE = re.compile(
    r"(?P<day>" + "|".join(_ZH_RELATIVE_DAYS) + ")" + _ZH_TIME_TAIL
)
_ZH_WEEKDAY_RE = re.compile(
    r"(?P<shift>下|上|这|本)?(?:周|星期|礼拜)(?P<weekday>[一二三四五六日天])"
    + _ZH_TIME_TAIL
)
_ZH_DAYS_LATER_RE = re.compile(
    _ZH_COUNT + r"(?P<unit>天|日|周|个星期|星期)(?:后|之后|以后)"
)
_ZH_HOURS_LATER_RE = re.compile(
    _ZH_COUNT + r"(?P<unit>小时|个小时|分钟)(?:后|之后|以后)"
)
_ZH_TIME_RE = re.compile(_ZH_TIME)
_EN_RELATIVE_DAYS = {
    "today": 0,
    "tomorrow": 1,
    "yesterday": -1,
    "day after tomorrow": 2,
}
_EN_RELATIVE_RE = re.compile(
    r"(?P<day>" + "|".join(_EN_RELATIVE_DAYS) + ")" + _EN_TIME_TAIL
)
_EN_SHIFTED_WEEKDAY_RE = re.compile(
    rf"(?P<shift>next|last|this) (?P<weekday>{_WEEKDAY})" + _EN_TIME_TAIL
)
_EN_WEEKDAY_RE = re.compile(rf"(?:on )?(?P<weekday>{_WEEKDAY})" + _EN_TIME_TAIL)
_EN_DAYS_LATER_RE = re.compile(r"in (?P<count>[0-9]+) (?P<unit>day|days|week|weeks)")
_EN_HOURS_LATER_RE = re.compile(
    r"in (?P<count>[0-9]+) (?P<unit>hour|hours|minute|minutes)"
)
_EN_TIME_RE = re.compile(_EN_TIME)

# Grammar row 2 (an English calendar date, optionally with a time) is matched
# whole here, the way the other twelve rows match theirs. dateutil then does
# the date arithmetic for a phrase this pattern has already accepted; which
# strings pass this gate is decided here, by the pattern, not by dateutil --
# though, as the paragraph below explains, dateutil's own zone-name heuristic
# still decides whether four particular am/pm spellings succeed once past the
# gate. Written out, the accepted shapes are
#   <day> <month name>[.][,] <year>   |   <month name>[.] <day>[,] <year>
#   <day>/<month>/<year> or <day>-<month>-<year>
# the year always four digits and never starting with zero, in every branch,
# each optionally followed by one space and a time, which is one of
#   HH:MM[:SS]                                     (24-hour, no am/pm)
#   H[:MM[:SS]] am/pm, hour one through eleven      (am/pm mandatory here)
#   a literal twelve, zero to two ":" or "." groups, am/pm   (mandatory too)
# The am/pm half of the second and third alternatives admits the sixteen
# spellings that write the letter and the "m" together (am, a.m, am., a.m.
# and the p forms, in either letter case) -- the same set _EN_HOUR_TWELVE_RE
# covers. Twelve of those sixteen resolve normally at hours one through
# eleven and reach the hour-twelve guard below, getting its named wording, at
# hour twelve. The other four ("A.M", "A.M.", "P.M", "P.M.") never resolve at
# any hour and never reach that guard either, at hour twelve or any other:
# dateutil reads a solitary uppercase "M" split from its letter by a period
# as a candidate timezone name rather than the second half of a meridian
# marker, and the tzinfos callback below raises for it inside
# dateutil_parser.parse itself, before this function ever computes an hour to
# check -- "15 Sep 2026 11 p.m." resolves, "15 Sep 2026 11 P.M." does not,
# same phrase, capitalisation only, and the second phrase is refused before
# the guard runs, not through it.
# The third alternative exists solely so the hour-twelve spellings that must
# keep the named wording still reach the guard: it matches nothing but a
# literal twelve, so nothing it admits can ever resolve (verified over
# leading zeros zero through five, both group counts zero through two across
# the full two-digit range, all sixteen period spellings, and both
# separators, across nine date branches, not by sampling -- 0*12 admits
# unboundedly many leading-zero counts, so this is a wide explicit range
# rather than literally every one).
# What the door refuses that dateutil would otherwise have accepted includes,
# at least: a weekday name, fractional seconds, an eight-digit run, a "."
# date separator, a year written first, a year under four digits or starting
# with zero, a "T" or "h" time separator, a space-separated numeric date, a
# bare hour with no am/pm, a single-digit minute or second, a "." between an
# hour and its minutes anywhere but the literal-twelve spelling, a leading
# "on", trailing punctuation, full-width digits, any am/pm spelling whose
# letters are split by whitespace or reduced to one letter, and an hour
# written with three digits and paired with am/pm -- the general branch
# admits one or two digits, one through eleven, only, so both a 24-hour
# value like "023" and a padded twelve-hour value like "010" are refused;
# only a literal twelve may be written with extra leading zeros, and that
# spelling never resolves anyway.
_EN_MONTH = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)
_EN_CALENDAR_DAY = r"[0-9]{1,2}(?:st|nd|rd|th)?"
_EN_CALENDAR_DATE = (
    rf"(?:{_EN_CALENDAR_DAY} (?:{_EN_MONTH})\.?,? [1-9][0-9]{{3}}"
    rf"|(?:{_EN_MONTH})\.? {_EN_CALENDAR_DAY},? [1-9][0-9]{{3}}"
    r"|[0-9]{1,2}[/-][0-9]{1,2}[/-][1-9][0-9]{3})"
)
_EN_CALENDAR_CLOCK = (
    r"(?:[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?"
    r"|(?:0?[1-9]|1[01])(?::[0-9]{2}){0,2} ?(?:a|p)\.?m\.?"
    r"|0*12(?:[:.][0-9]{2}){0,2} ?(?:a|p)\.?m\.?)"
)
_EN_CALENDAR_RE = re.compile(
    rf"{_EN_CALENDAR_DATE}(?: {_EN_CALENDAR_CLOCK})?", re.IGNORECASE
)

# Two defaults that differ in every field a phrase may leave out (year, month,
# day, hour). A field that follows the default was not in the phrase.
_DEFAULT_A = datetime(2000, 1, 1, 0, 0, 0)
_DEFAULT_B = datetime(2001, 2, 2, 1, 0, 0)


class _AmbiguousDate(ValueError):
    """The phrase reads as two different dates depending on day/month order."""


class _AmbiguousHour(ValueError):
    """A spoken hour paired with a half-day word that does not settle on,
    or flatly contradicts, one correct time of day.

    Three shapes raise this. Hour twelve paired with any half-day word
    (English am/pm, Chinese 上午/早上/下午/晚上) names both endpoints of a
    day: '12' is the hour at which the label flips, without telling you
    which side of midnight or noon it is on, so a period word attached to
    it is read one way by some speakers and the other way by others.
    '中午' (noon) paired with any hour other than 11, 12, or 13 has the
    opposite problem: instead of leaving the hour ambiguous, it names an
    hour nowhere near noon, contradicting the period word it came with.
    The third shape generalises the second to the other Chinese words: each
    of 上午/早上/下午/晚上 admits only its own half of the day, in either
    the twelve-hour or the twenty-four-hour spelling, so an hour outside
    both -- including hour zero, which belongs to no half-day word --
    contradicts the word it came with.
    Guessing a reading in any of the three cases can land the result hours
    or a full day off, so the phrase is refused instead of resolved to one.
    """


def _reject_zone_in_phrase(name: Optional[str], offset: Optional[int]) -> None:
    # The zone comes from the timezone argument only. dateutil calls this for
    # every parse, with (None, None) when the phrase names no zone; a zone
    # name or offset inside the phrase would otherwise be dropped silently
    # or turned into an offset the caller never asked for. The whole-phrase
    # calendar pattern in _EN_CALENDAR_RE refuses every spelling that
    # deliberately names a zone (e.g. "EST"), but that does not make this
    # callback unreachable: dateutil reads a solitary uppercase "M" split
    # from its letter by a period (as in "12 A.M") as a candidate zone name
    # rather than the second half of a meridian marker, so a phrase spelling
    # am/pm that way is called in here with name="M", not (None, None), and
    # still raises below.
    if name is None and offset is None:
        return None
    raise ValueError(f"time zone inside the phrase is not supported: {name!r}")


def _parse_number(token: str) -> Optional[int]:
    """Read an Arabic numeral or a Chinese numeral from one to ninety-nine."""
    if token.isascii() and token.isdigit():
        return int(token)
    if token == "两":
        return 2
    match = re.fullmatch(
        r"(?P<tens>[一二三四五六七八九])?(?P<ten>十)?(?P<units>[一二三四五六七八九])?",
        token,
    )
    if match is None:
        return None
    tens, ten, units = match.group("tens"), match.group("ten"), match.group("units")
    if ten is None:
        # A single digit; two digits without the tens marker are not a numeral.
        if tens is not None and units is not None:
            return None
        digit = tens or units
        return _ZH_DIGITS[digit] if digit else None
    value = 10 * (_ZH_DIGITS[tens] if tens else 1)
    return value + (_ZH_DIGITS[units] if units else 0)


# On the calendar-date path this pattern only chooses which refusal message
# to raise, once the parsed hour has already decided that a refusal is due
# (see _read_en_calendar_date): does the phrase spell a twelve at all, in
# one of the ways this pattern covers? The whole-phrase calendar pattern
# above (_EN_CALENDAR_RE) now refuses most of what used to reach here
# unrecognised before dateutil ever saw it -- "12h30 pm" and "12,30 pm" are
# refused at that door, not here. What still reaches this point without a
# literal twelve in the text is a folded hour spelled some other way, such
# as "0:30 am" folding to hour zero: that case falls through to the generic
# refusal rather than being named, which is the existing, intended behaviour
# (see _read_en_calendar_date's docstring). A leading-zero run, a "." minute
# separator, and the dotted "a.m."/"p.m." form are all safe to recognise
# here, because being generous only affects the wording, never whether the
# phrase is refused.
_EN_HOUR_TWELVE_RE = re.compile(
    r"(?<![0-9:.])0*12(?:[:.][0-9]{2}){0,2} ?(?:a|p)\.?m\.?(?![a-z])",
    re.IGNORECASE,
)


def _refuse_hour_twelve_with_period(period: str) -> NoReturn:
    """Refuse an English hour twelve carrying am or pm, whichever reader saw it.

    Both English readers reach this: the time sub-grammar, which has the hour
    parsed, and the calendar-date reader, where dateutil parses the time
    itself and would otherwise accept the combination the sub-grammar
    refuses. Keeping the decision and its wording here is what makes a
    phrase with a date and a phrase without return the same refusal.
    """
    raise _AmbiguousHour(f"'12{period.lower()}' names both midnight and noon")


def _en_time(match: re.Match[str]) -> Optional[tuple[int, int]]:
    """Hour and minute from an English time; None when the branch does not hold.

    Raises _AmbiguousHour for '12am' / '12pm', via _refuse_hour_twelve_with_period:
    am/pm otherwise disambiguates the hour, but at twelve it does not (some
    speakers take 12pm as noon and 12am as midnight; others get the two
    backwards), so this one hour value is refused rather than picked either way.
    """
    hour = int(match.group("en_hour"))
    minute_text = match.group("en_minute")
    period = match.group("en_period")
    if period is None:
        # Without am/pm the minutes are required: a bare number is not a time.
        if minute_text is None or not 0 <= hour <= 23:
            return None
    else:
        if not 1 <= hour <= 12:
            return None
        if hour == 12:
            _refuse_hour_twelve_with_period(period)
        if period == "pm":
            hour += 12
    minute = int(minute_text) if minute_text is not None else 0
    if not 0 <= minute <= 59:
        return None
    return hour, minute


# Each half-day word admits two spellings of its own half of the day: the
# twelve-hour hours, which the word shifts past noon, and the twenty-four-hour
# spelling of the same clock positions, which it keeps as written. An hour in
# neither range contradicts the word instead of qualifying it. Hour zero is in
# no word's ranges: midnight is written 0点 on its own.
_ZH_HALF_DAY_HOURS: dict[str, tuple[range, range]] = {
    "上午": (range(0, 0), range(1, 12)),
    "早上": (range(0, 0), range(1, 12)),
    "下午": (range(1, 12), range(13, 19)),
    "晚上": (range(1, 12), range(18, 24)),
}


def _zh_time(match: re.Match[str]) -> Optional[tuple[int, int]]:
    """Hour and minute from a Chinese time; None when the branch does not hold.

    Three shapes raise _AmbiguousHour. Hour twelve paired with 上午, 早上,
    下午, or 晚上 (morning, morning, afternoon, or evening): the period word
    tells you which half of the day the hour is in for every other hour, but
    at twelve the half-day boundary itself is what's in question (an evening
    or afternoon reading could mean tonight's midnight, i.e. today ending,
    or tomorrow beginning; a morning reading could mean midnight or, taken
    as bare digits, noon). The noon period word (中午) is unaffected by
    that check: it already names hour twelve unambiguously. It raises the
    same exception for a narrower reason instead: paired with any hour
    other than 11, 12, or 13 it names an hour nowhere near noon (e.g.
    '中午10点' names 10 in the morning), so those combinations are refused
    too. Third, each of 上午/早上/下午/晚上 admits only the hours in
    _ZH_HALF_DAY_HOURS for that word (its own twelve-hour hours, shifted past
    noon, and the matching twenty-four-hour hours, kept as written); an hour
    outside both -- including hour zero, which belongs to no half-day word --
    contradicts the word and is refused, naming the hours it does admit.
    """
    hour = _parse_number(match.group("zh_hour"))
    if hour is None:
        return None
    if match.group("zh_half") is not None:
        minute: Optional[int] = 30
    elif match.group("zh_minute") is not None:
        minute = _parse_number(match.group("zh_minute"))
    else:
        minute = 0
    if minute is None:
        return None
    period = match.group("zh_period")
    if hour == 12 and period in ("上午", "早上", "下午", "晚上"):
        raise _AmbiguousHour(
            f"'{period}12点' could mean today ending at "
            "midnight, tomorrow beginning at midnight, or noon"
        )
    if period == "中午" and hour not in (11, 12, 13):
        raise _AmbiguousHour(
            f"'中午{hour}点' names an hour nowhere near noon: only 11, 12, or 13 do"
        )
    if period in _ZH_HALF_DAY_HOURS:
        shifted, kept = _ZH_HALF_DAY_HOURS[period]
        if hour in shifted:
            hour += 12
        elif hour == 0:
            raise _AmbiguousHour(
                f"'{period}0点' pairs a half-day word with hour zero: "
                "write 0点 without a half-day word for midnight"
            )
        elif hour not in kept:
            spans = " or ".join(
                f"{span.start} to {span.stop - 1}" for span in (shifted, kept) if span
            )
            raise _AmbiguousHour(
                f"'{period}{hour}点' contradicts {period}, which names hours {spans}"
            )
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return hour, minute


def _wall(day: date, clock: Optional[tuple[int, int]]) -> _Reading:
    if clock is None:
        return datetime(day.year, day.month, day.day), False
    return datetime(day.year, day.month, day.day, clock[0], clock[1]), True


_ClockOf = Callable[[re.Match[str]], Optional[tuple[int, int]]]


def _day_reading(
    match: re.Match[str], day: date, clock_of: _ClockOf
) -> Optional[_Reading]:
    """Combine a day with the optional time group of the match.

    Without a time group the reading is that day's midnight; with one it is
    the day at that time, or None when the group is present but not a time.
    """
    groups = match.groupdict()
    if groups.get("en_hour") is None and groups.get("zh_hour") is None:
        return _wall(day, None)
    clock = clock_of(match)
    return _wall(day, clock) if clock is not None else None


def _read_iso(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _ISO_RE.fullmatch(text)
    if match is None:
        return None
    # fromisoformat is deliberately given the whole phrase, not
    # match.group(0): the two calls are equal today because the fullmatch
    # above already requires the whole phrase to match, but this keeps a
    # second, independent whole-phrase check in place should that
    # fullmatch above ever be loosened.
    moment = datetime.fromisoformat(text)
    return moment, match.group("clock") is not None


def _read_zh_date(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _ZH_DATE_RE.fullmatch(text)
    if match is None:
        return None
    day = date(
        int(match.group("year")), int(match.group("month")), int(match.group("day"))
    )
    return _day_reading(match, day, _zh_time)


def _read_zh_relative_day(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _ZH_RELATIVE_RE.fullmatch(text)
    if match is None:
        return None
    day = now_local.date() + timedelta(days=_ZH_RELATIVE_DAYS[match.group("day")])
    return _day_reading(match, day, _zh_time)


def _read_zh_weekday(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _ZH_WEEKDAY_RE.fullmatch(text)
    if match is None:
        return None
    today = now_local.date()
    # Weeks run Monday to Sunday: the next-week prefix moves a week ahead,
    # the previous-week prefix a week back, and no prefix (or either
    # this-week prefix) stays in the week containing today.
    monday = today - timedelta(days=today.weekday())
    # Both spellings of Sunday sit at the end of the table and share index 6.
    weekday = min(_ZH_WEEKDAYS.index(match.group("weekday")), 6)
    shift = {"下": 7, "上": -7}.get(match.group("shift") or "", 0)
    day = monday + timedelta(days=shift + weekday)
    return _day_reading(match, day, _zh_time)


def _read_zh_days_later(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _ZH_DAYS_LATER_RE.fullmatch(text)
    if match is None:
        return None
    count = _parse_number(match.group("count"))
    if count is None:
        return None
    days = count * (7 if match.group("unit") in ("周", "个星期", "星期") else 1)
    return _wall(now_local.date() + timedelta(days=days), None)


def _duration_later(count: int, minutes: bool, now_local: datetime) -> _Reading:
    """`now` plus a duration, as an exact instant on a whole minute.

    Added in UTC: adding to the local reading would shift the wall clock
    across a daylight-saving change instead of the instant. The clock's own
    seconds are dropped first -- the phrase said how long from now, not which
    second -- so this form lands on a whole minute like every other form in
    the grammar.
    """
    delta = timedelta(minutes=count) if minutes else timedelta(hours=count)
    return now_local.astimezone(timezone.utc).replace(
        second=0, microsecond=0
    ) + delta, True


def _read_zh_hours_later(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _ZH_HOURS_LATER_RE.fullmatch(text)
    if match is None:
        return None
    count = _parse_number(match.group("count"))
    if count is None:
        return None
    return _duration_later(count, match.group("unit") == "分钟", now_local)


def _read_zh_time(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _ZH_TIME_RE.fullmatch(text)
    if match is None:
        return None
    return _day_reading(match, now_local.date(), _zh_time)


def _read_en_relative_day(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _EN_RELATIVE_RE.fullmatch(text)
    if match is None:
        return None
    day = now_local.date() + timedelta(days=_EN_RELATIVE_DAYS[match.group("day")])
    return _day_reading(match, day, _en_time)


def _read_en_shifted_weekday(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _EN_SHIFTED_WEEKDAY_RE.fullmatch(text)
    if match is None:
        return None
    today = now_local.date()
    weekday = _WEEKDAYS.index(match.group("weekday"))
    shift = match.group("shift")
    if shift == "next":
        # The coming one; when today is that weekday, a full week ahead.
        day = today + timedelta(days=(weekday - today.weekday()) % 7 or 7)
    elif shift == "last":
        # The previous one; when today is that weekday, a full week back.
        day = today - timedelta(days=(today.weekday() - weekday) % 7 or 7)
    else:
        day = today - timedelta(days=today.weekday()) + timedelta(days=weekday)
    return _day_reading(match, day, _en_time)


def _read_en_weekday(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _EN_WEEKDAY_RE.fullmatch(text)
    if match is None:
        return None
    today = now_local.date()
    weekday = _WEEKDAYS.index(match.group("weekday"))
    # The coming one, today included when today is that weekday.
    day = today + timedelta(days=(weekday - today.weekday()) % 7)
    return _day_reading(match, day, _en_time)


def _read_en_days_later(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _EN_DAYS_LATER_RE.fullmatch(text)
    if match is None:
        return None
    count = int(match.group("count"))
    days = count * (7 if match.group("unit").startswith("week") else 1)
    return _wall(now_local.date() + timedelta(days=days), None)


def _read_en_hours_later(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _EN_HOURS_LATER_RE.fullmatch(text)
    if match is None:
        return None
    count = int(match.group("count"))
    return _duration_later(count, match.group("unit").startswith("minute"), now_local)


def _read_en_time(text: str, now_local: datetime) -> Optional[_Reading]:
    match = _EN_TIME_RE.fullmatch(text)
    if match is None:
        return None
    return _day_reading(match, now_local.date(), _en_time)


def _read_en_calendar_date(text: str, now_local: datetime) -> Optional[_Reading]:
    """Read an English calendar date, letting dateutil do only the arithmetic.

    The whole phrase must first match _EN_CALENDAR_RE; a phrase that does not
    is not this grammar row at all, and dateutil never sees it. Past that
    gate, the phrase is parsed eight times: against two defaults and, for
    each, all four day-first / year-first combinations. dateutil only
    computes the date and time from a phrase the pattern above has already
    accepted, and every shape that pattern admits keeps that computation
    equal to what the phrase wrote: the year is four digits not starting
    with zero, so dateutil's own century substitution for a shorter year is
    never reached; the am/pm alternative admits only hours one through
    eleven, so dateutil can never read a zero-padded 24-hour value (e.g.
    "023") and silently discard the am/pm word that came with it; and the
    pattern's only "." time spelling is restricted to a literal twelve,
    which never resolves (verified over a wide, explicit range of leading
    zeros, dot-group counts, and period spellings, not literally every
    hour-string -- 0*12 admits unboundedly many). Readings that differ
    across the four combinations are an ambiguous date. A date that follows
    the default is missing a component. An hour that follows the default
    means the phrase named no time of day. An hour folded onto 0 or 12 with
    am/pm and matching _EN_HOUR_TWELVE_RE is refused through the same rule
    and the same wording as the bare time sub-grammar
    (_refuse_hour_twelve_with_period); a period word paired with that folded
    hour but no spelled-out twelve (e.g. "0:30 am") is refused too, but
    generically: the bare time sub-grammar already refuses that same reading
    (_en_time's `if not 1 <= hour <= 12: return None`), so this reader must
    not resolve it either. A microsecond in the parsed result is refused
    rather than dropped, as a second, independent check: the pattern above
    already refuses fractional seconds, so this is unreachable today, the
    same way _read_iso keeps its own second whole-phrase check. A zone name
    or offset the phrase deliberately writes (e.g. "EST") is refused by the
    tzinfos callback below, and the pattern above already refuses every such
    spelling on its own; that callback is still reachable for an unrelated
    reason, though, since dateutil separately reads a solitary uppercase "M"
    split from its letter by a period (as in "12 A.M") as a candidate zone
    name rather than the second half of a meridian marker, so that spelling
    reaches the same callback and the same generic refusal instead of the
    hour-twelve guard's wording.
    """
    if _EN_CALENDAR_RE.fullmatch(text) is None:
        return None
    readings: dict[datetime, list[datetime]] = {}
    for default in (_DEFAULT_A, _DEFAULT_B):
        for dayfirst in (False, True):
            for yearfirst in (False, True):
                readings.setdefault(default, []).append(
                    dateutil_parser.parse(
                        text,
                        default=default,
                        dayfirst=dayfirst,
                        yearfirst=yearfirst,
                        fuzzy=False,
                        tzinfos=_reject_zone_in_phrase,
                    )
                )
    for parsed in readings.values():
        if any(reading != parsed[0] for reading in parsed[1:]):
            raise _AmbiguousDate(
                f"date reads two ways (day-first or month-first): {text!r}"
            )
    first, second = readings[_DEFAULT_A][0], readings[_DEFAULT_B][0]
    if first.tzinfo is not None or second.tzinfo is not None:
        return None
    if first.date() != second.date():
        return None
    if first.hour != second.hour:
        return _wall(first.date(), None)
    marker = re.search(
        r"(?<![a-z])(?P<period>a|p)\.?m\.?(?![a-z])", text, re.IGNORECASE
    )
    if marker is not None and first.hour in (0, 12):
        # dateutil reads 12 pm as noon and 12 am as midnight without ever
        # asking whether the writer meant that. Its own parsed hour is the
        # reliable witness: an hour written as 1 through 11 never folds
        # onto 0 or 12, so a folded 0 or 12 means the writer wrote a twelve
        # or a zero, and neither can be placed on a twelve-hour clock.
        if _EN_HOUR_TWELVE_RE.search(text) is not None:
            _refuse_hour_twelve_with_period(marker.group("period") + "m")
        # The wording pattern does not recognise a twelve here because the
        # phrase wrote a zero, not a twelve ("0:30 am" folds to hour 0, and
        # "0:30 pm" folds to hour 12 the same way "12" would). A spelling of
        # twelve the pattern itself does not cover, such as "12h30 pm", no
        # longer reaches this line at all: the whole-phrase gate above
        # refuses it before dateutil ever parses it. Naming an hour the
        # message cannot vouch for would misdirect, so fall through to the
        # generic refusal, matching what the bare time sub-grammar already
        # does with this same folded hour.
        return None
    if first.microsecond:
        # Unreachable while the whole-phrase pattern above refuses fractional
        # seconds; kept as a second, independent check should that pattern
        # ever be loosened, the same way _read_iso keeps its own second
        # whole-phrase check. Replacing the microsecond with zero here would
        # drop a component the phrase itself carried.
        return None
    return first, True


_READERS = (
    _read_iso,
    _read_zh_date,
    _read_zh_relative_day,
    _read_zh_weekday,
    _read_zh_days_later,
    _read_zh_hours_later,
    _read_zh_time,
)
_LOWERCASE_READERS = (
    _read_en_relative_day,
    _read_en_shifted_weekday,
    _read_en_weekday,
    _read_en_days_later,
    _read_en_hours_later,
    _read_en_time,
)


def _read_phrase(text: str, now_local: datetime) -> Optional[_Reading]:
    """Match the phrase against the grammar; the first form that holds wins."""
    for reader in _READERS:
        reading = reader(text, now_local)
        if reading is not None:
            return reading
    lowered = text.lower()
    for reader in _LOWERCASE_READERS:
        reading = reader(lowered, now_local)
        if reading is not None:
            return reading
    return _read_en_calendar_date(text, now_local)


_MINUTES_PER_DAY = 24 * 60


def _day_midnight_instants(
    midnight: datetime, zone: ZoneInfo, wall_text: str
) -> list[datetime]:
    """Midnight on this calendar day, carrying an offset the zone really used.

    A date-only phrase names a day, not a wall-clock time: the 00:00:00 in
    `midnight` is this function's own filler, not something the user wrote,
    so the daylight-saving guard that protects a written clock reading must
    not refuse the day. This collapses the day to at most one reading --
    which is what the caller's three-way table expects -- in three ways:

    * the zone has that midnight once: that instant, unchanged;
    * the zone repeats it: the earlier of the two, which is where the day
      starts;
    * the zone skips it: midnight stamped with the offset in force at the
      first minute of the day the zone did not skip, found by asking
      `_instants_for_wall_time` minute by minute. The offset therefore comes
      from a reading the zone check accepted, never from attaching the zone
      to a wall time it rejected.

    The list is empty only when no minute of the calendar day exists at all,
    which happens where a zone moved across the date line and dropped a whole
    day (Pacific/Kiritimati 1994-12-31). A conversion that leaves the range
    datetime can represent still raises ValueError out of the helper it calls,
    exactly as it does for a wall-clock time.

    On a day the zone skips, the offset this returns is one the zone really
    uses, but the instant it produces reads back, in this same zone, as a
    moment on the previous calendar day (America/Santiago 2026-09-06 returns
    2026-09-06T00:00:00-03:00, which is 2026-09-05T23:00:00-04:00 in
    Santiago). Callers are expected to use the returned value as text, where
    the requested date is its first ten characters, rather than convert it
    back into a local time.
    """
    instants = _instants_for_wall_time(midnight, zone, wall_text)
    if instants:
        return instants[:1]
    for minutes in range(1, _MINUTES_PER_DAY):
        later = _instants_for_wall_time(
            midnight + timedelta(minutes=minutes), zone, wall_text
        )
        if later:
            offset = later[0].utcoffset()
            # utcoffset() is None only for a naive value; every entry here
            # carries the zone.
            assert offset is not None
            return [midnight.replace(tzinfo=timezone(offset))]
    return []


def _refusal(
    reason: str, error: str, *, include_grammar: bool = True
) -> dict[str, Any]:
    if reason not in RESOLUTION_REASONS:
        # The reason set is the tool's published contract: a refusal carrying
        # a code outside it would put a word into the model's context that no
        # caller can act on. Nothing reachable passes one -- every call site
        # writes a literal from the set -- so this fails loudly rather than
        # shipping an unknown code.
        raise ValueError(f"unknown resolution reason: {reason!r}")
    # success=False routes the call through the framework's failure branch;
    # the reason is carried under 'resolution' because 'status' is the
    # framework's control channel.
    refusal: dict[str, Any] = {
        "success": False,
        "tool_name": "resolve_datetime",
        "error": error,
        "resolution": reason,
    }
    if reason == "unsupported_expression" and include_grammar:
        refusal["supported"] = list(GRAMMAR_FORMS)
    return refusal


def resolve_datetime(phrase: str, timezone_name: str) -> dict[str, Any]:
    """Turn a spoken date or time into an exact ISO date-time in a zone.

    The result depends only on the phrase, the zone, the clock and tzdata:
    the same three inputs under the same clock reading always give the same
    result. A phrase outside the grammar, a date that reads two ways, an
    hour twelve whose half-day word does not settle which side of midnight
    or noon it is on, a noon word (中午) paired with an hour outside 11, 12,
    or 13, a half-day word contradicted by the hour it came with, a written
    wall-clock time the zone skips or repeats, a calendar day the zone
    skipped entirely at a date-line change, a resulting UTC offset with a
    fractional minute, or an instant outside the representable date range is
    refused with a reason rather than guessed or approximated.

    A phrase that named a day only is answered for the day even where that
    day's midnight is skipped or repeated: the 00:00:00 is this tool's own
    filler, so it carries the offset of the first minute of the day the zone
    did have, and the time part stays 00:00:00.
    """
    try:
        zone = _require_region_city_zone(timezone_name)
    except ValueError as exc:
        return _refusal("invalid_timezone", str(exc))
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", phrase)).strip()
    if not text:
        return _refusal("unsupported_expression", "phrase is empty")
    now_local = _now().astimezone(zone)
    try:
        reading = _read_phrase(text, now_local)
    except _AmbiguousDate as exc:
        return _refusal("ambiguous_date", str(exc))
    except _AmbiguousHour as exc:
        return _refusal("unsupported_expression", str(exc))
    except (ValueError, OverflowError):
        reading = None
    if reading is None:
        return _refusal(
            "unsupported_expression",
            f"unsupported date or time expression: {text!r}",
        )
    moment, has_time = reading
    if moment.tzinfo is not None:
        try:
            aware = moment.astimezone(zone)
        except (OverflowError, OSError):
            # Two conversions can leave the representable range: subtracting
            # an offset to reach the UTC instant, which no target zone can
            # avoid, and rendering that instant in the target zone, which a
            # zone far from UTC reaches at either end of the calendar. The
            # phrase itself matched the grammar, so the grammar list would
            # only misdirect a caller into rewriting an already-supported
            # phrase.
            return _refusal(
                "unsupported_expression",
                f"{text!r} names an instant outside the range this tool can "
                "represent, which ends at year 1 and year 9999 in "
                f"{zone.key} and in UTC",
                include_grammar=False,
            )
    else:
        stamp = moment.isoformat(sep=" ", timespec="seconds")
        try:
            # A phrase that named a time of day is answered for that exact
            # wall clock; a phrase that named a day only is answered for the
            # day, because its 00:00:00 came from this tool, not the user.
            instants = (
                _instants_for_wall_time(moment, zone, stamp)
                if has_time
                else _day_midnight_instants(moment, zone, stamp)
            )
        except ValueError as exc:
            return _refusal("unsupported_expression", str(exc))
        if not instants:
            return _refusal(
                "nonexistent_local_time",
                f"local time {stamp} does not exist in {zone.key}: "
                "clocks skip it at a daylight-saving change"
                if has_time
                else f"{zone.key} has no {moment.date().isoformat()}: it "
                "moved across the date line and skipped the whole day",
            )
        if len(instants) > 1:
            return _refusal(
                "ambiguous_local_time",
                f"local time {stamp} occurs twice in {zone.key} "
                "at a daylight-saving change",
            )
        aware = instants[0]
    # A handful of zones carried a sub-minute UTC offset before their
    # jurisdiction standardised its clock (e.g. Africa/Monrovia until 1972,
    # -00:44:30). RFC 3339 / JSON Schema 'date-time' only allow a ±HH:MM
    # offset, so such an instant has no valid representation here; rather
    # than truncate or round it into an approximation, refuse it.
    offset = aware.utcoffset()
    if offset is None or offset.total_seconds() % 60:
        # The phrase itself matched the grammar; the problem is the zone's
        # historical offset, so the grammar list would only mislead a
        # caller into rewriting an already-supported phrase.
        return _refusal(
            "unsupported_expression",
            f"{zone.key} used a UTC offset with a fractional minute at "
            "this date, which is not representable as a date-time offset",
            include_grammar=False,
        )
    resolved = aware.isoformat(timespec="seconds")
    return ResolveDatetimeResult(
        resolved=resolved, has_time=has_time, timezone=zone.key
    ).model_dump()


class ResolveDatetimeArgs(BaseModel):
    phrase: str = Field(
        description=(
            "The user's exact words for the date or time, copied verbatim from "
            "a user message: for example 'tomorrow at 3pm', 'next Friday', "
            "'1 Jan 1990', '下周三上午十点'. Do not rephrase, translate, or "
            "shorten it."
        )
    )
    # Field name is the wire contract; it locally shadows the datetime.timezone
    # import, which this class body does not use.
    timezone: str = Field(
        description=(
            "IANA zone name in Region/City form the phrase should be read in, "
            "for example 'Australia/Sydney', or 'UTC'. Copy the zone named in "
            "the system prompt's date-and-time line unless the user named "
            "another zone. An abbreviation ('EST'), a bare city ('Sydney'), or "
            "an 'Etc/*' name is an error, not a fallback to UTC."
        )
    )


class ResolveDatetimeTool(AbstractBaseTool):
    """Turns a date or time the user wrote in words into an exact ISO
    date-time with a UTC offset, deterministically from the clock and tzdata;
    anything the grammar cannot read is refused rather than guessed."""

    category = ToolCategory.OTHER
    read_only = True  # reads a clock and tzdata ⇒ concurrency-safe

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "resolve_datetime"

    @property
    def description(self) -> str:
        return (
            "Turn a date or time the user wrote in words into an exact ISO "
            "8601 date-time with a UTC offset, computed from the real clock "
            "and the IANA zone database. Use it before filling a date or "
            "date-time field from a relative or spoken expression such as "
            "'tomorrow at 3pm', 'next Friday', 'in 3 days', '3pm', "
            "'1 Jan 1990', '下周三上午十点'. Pass the user's exact words as "
            "phrase. Pass the zone named in the system prompt's date-and-time "
            "line as timezone. An expression outside the supported forms, a "
            "date that reads two ways (day-first or month-first), a time "
            "of day that does not exist or occurs twice at a daylight-saving "
            "change, or a calendar day a zone skipped entirely at a "
            "date-line change, is refused with a reason instead of guessed: "
            "then ask the user. For the time right now use get_current_time; "
            "to check a specific wall-clock time in a zone use "
            "validate_local_time."
        )

    def args_type(self) -> Type[BaseModel]:
        return ResolveDatetimeArgs

    def return_type(self) -> Type[BaseModel]:
        # The success shape only. A refusal is a framework-shaped failure
        # mapping (success/tool_name/error/resolution), not a narrower
        # ResolveDatetimeResult, and nothing turns this class into a schema
        # the engine validates results against: the two wrappers that read
        # return_type() only pass it through, and the one consumer that would
        # validate against it belongs to FunctionTool, which this tool does
        # not go through.
        return ResolveDatetimeResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        parsed = ResolveDatetimeArgs.model_validate(args)
        # Already a dict: a success is the result model dumped, a refusal is
        # the framework-shaped failure mapping. validate_local_time raises
        # instead, because every way it can fail is a caller mistake; this
        # tool's refusals are answers about the phrase the model must be able
        # to read and act on, so they come back as a result with a reason.
        return resolve_datetime(parsed.phrase, parsed.timezone)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return self.run_json_sync(args)


@register_tool(selection_gate="intrinsic")
async def create_resolve_datetime_tool(config: WebToolConfig) -> list[AbstractBaseTool]:
    """Create the date-time resolution tool."""
    return [ResolveDatetimeTool()]
