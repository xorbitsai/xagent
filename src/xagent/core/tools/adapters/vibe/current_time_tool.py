import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Literal, Mapping, Optional, Type
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

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
            "its clocks, call the validate_local_time tool if it is available."
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


def validate_local_time(local_time: str, timezone_name: str) -> ValidateLocalTimeResult:
    """Resolve a wall-clock time in a zone to every UTC instant it names."""
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

    # PEP 495: fold 0 and 1 are the two candidate readings. Each is kept only
    # if it converts back to the requested wall time -- a candidate that does
    # not is the zone telling us this reading does not exist. Trusting one
    # fold's offset instead would misreport zones whose fold=0 reading returns
    # an offset they never have (America/Nuuk under tzdata 2025c).
    by_instant: dict[datetime, datetime] = {}
    for fold in (0, 1):
        local = naive.replace(fold=fold, tzinfo=zone)
        try:
            utc = local.astimezone(timezone.utc)
            round_trip = utc.astimezone(zone).replace(tzinfo=None)
        except (OverflowError, OSError) as exc:
            raise ValueError(
                f"local_time converts outside the representable date range: {text!r}"
            ) from exc
        if round_trip == naive:
            by_instant.setdefault(utc, local)

    mappings = [_mapping(by_instant[utc], utc) for utc in sorted(by_instant)]
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
            "time right now use get_current_time instead."
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
