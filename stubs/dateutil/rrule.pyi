"""Type stubs for dateutil.rrule (only the pieces this repo uses)."""

from datetime import datetime
from typing import Any, Iterator

class weekday:
    def __init__(self, wkday: int, n: int | None = ...) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

MO: weekday
TU: weekday
WE: weekday
TH: weekday
FR: weekday
SA: weekday
SU: weekday

YEARLY: int
MONTHLY: int
WEEKLY: int
DAILY: int
HOURLY: int
MINUTELY: int
SECONDLY: int

class rrulebase:
    def __iter__(self) -> Iterator[datetime]: ...

class rrule(rrulebase):
    def __init__(
        self,
        freq: int,
        dtstart: datetime | None = ...,
        interval: int = ...,
        wkst: Any = ...,
        count: int | None = ...,
        until: datetime | None = ...,
        bysetpos: Any = ...,
        bymonth: Any = ...,
        bymonthday: Any = ...,
        byyearday: Any = ...,
        byeaster: Any = ...,
        byweekno: Any = ...,
        byweekday: Any = ...,
        byhour: Any = ...,
        byminute: Any = ...,
        bysecond: Any = ...,
        cache: bool = ...,
    ) -> None: ...

class rruleset(rrulebase):
    def __init__(self, cache: bool = ...) -> None: ...

def rrulestr(
    s: str,
    dtstart: datetime | None = ...,
    cache: bool = ...,
    unfold: bool = ...,
    forceset: bool = ...,
    compatible: bool = ...,
    ignoretz: bool = ...,
    tzids: Any = ...,
    tzinfos: Any = ...,
) -> rrulebase: ...
