"""Frozen clock and boundary-straddling instants for the daily-window tests.

Shared by the SQLite suite (``test_monitor_daily_windows.py``) and the
PostgreSQL one (``test_monitor_postgresql.py``) so both pin the same
boundary against the same instants -- the whole point of covering two
dialects is that they disagree about a naive literal, not about which rows
were seeded.

Not a ``conftest.py``: these are plain values and a class, imported by name
where they are used, matching how the rest of ``tests/web/api/`` shares
helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone

# 18:00 UTC on a UTC+8 server: the local wall clock reads 02:00 on the *next*
# calendar day. A "today" boundary built from naive local time is therefore
# 2026-08-18 00:00 while the correct UTC boundary is 2026-08-17 00:00 -- a
# full-day gap no seeded row can straddle by accident, so the pre-fix failure
# is a decisive count of zero rather than an off-by-one.
NOW_UTC = datetime(2026, 8, 17, 18, 0, tzinfo=timezone.utc)
NOW_LOCAL_NAIVE = datetime(2026, 8, 18, 2, 0)

# The session timezone the PostgreSQL case sets, chosen to match the +08:00
# offset the frozen clock models so both halves of that test tell the same
# story.
PG_SESSION_TIME_ZONE = "Asia/Shanghai"

UTC_TODAY_EARLY = datetime(2026, 8, 17, 0, 1, tzinfo=timezone.utc)
UTC_TODAY_LATE = datetime(2026, 8, 17, 17, 0, tzinfo=timezone.utc)
UTC_YESTERDAY_LATE = datetime(2026, 8, 16, 23, 59, tzinfo=timezone.utc)


class FrozenDatetime(datetime):
    """``datetime`` whose ``now()`` is pinned to :data:`NOW_UTC`.

    The naive branch returns what a UTC+8 server's wall clock would show at
    that instant, so the pre-fix call sites reproduce the skew
    deterministically instead of only when the test happens to run between
    16:00 and 24:00 UTC.
    """

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is None:
            return NOW_LOCAL_NAIVE
        return NOW_UTC.astimezone(tz)
