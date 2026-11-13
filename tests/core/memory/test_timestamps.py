"""Tests for UTC memory note timestamps coexisting with naive legacy rows."""

from datetime import datetime, timedelta, timezone

from xagent.core.memory.base import comparable_timestamp
from xagent.core.memory.core import MemoryNote


def test_new_memory_note_timestamp_is_utc_aware() -> None:
    note = MemoryNote(content="x")

    assert note.timestamp.tzinfo is not None
    assert note.timestamp.utcoffset() == timedelta(0)


def test_comparable_timestamp_orders_naive_local_against_aware_utc() -> None:
    # Legacy rows carry naive server-local time; new rows carry aware UTC.
    aware_now = datetime.now(timezone.utc)
    naive_earlier = (aware_now - timedelta(minutes=5)).astimezone().replace(tzinfo=None)
    naive_later = (aware_now + timedelta(minutes=5)).astimezone().replace(tzinfo=None)

    ordered = sorted([naive_later, aware_now, naive_earlier], key=comparable_timestamp)

    assert ordered == [naive_earlier, aware_now, naive_later]
    assert comparable_timestamp(naive_earlier).tzinfo is not None
    assert comparable_timestamp("not a datetime") == "not a datetime"
