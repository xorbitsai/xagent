"""Tests for UTC memory note timestamps coexisting with naive legacy rows."""

import os
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import pytest

from xagent.core.memory.base import comparable_timestamp
from xagent.core.memory.core import MemoryNote
from xagent.core.memory.in_memory import InMemoryMemoryStore

UTC = timezone.utc


@pytest.fixture
def new_york_tz() -> Iterator[None]:
    """Pin the process timezone so naive legacy timestamps have a known
    meaning (EST is UTC-5 in January) regardless of the CI host's zone."""
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    yield
    if previous is None:
        del os.environ["TZ"]
    else:
        os.environ["TZ"] = previous
    time.tzset()


def test_new_memory_note_timestamp_is_utc_aware() -> None:
    note = MemoryNote(content="x")

    assert note.timestamp.tzinfo is not None
    assert note.timestamp.utcoffset() == timedelta(0)


@pytest.mark.usefixtures("new_york_tz")
def test_comparable_timestamp_reads_naive_as_server_local_time() -> None:
    # A legacy naive 09:30 written by a New York server is 14:30 UTC.
    assert comparable_timestamp(datetime(2026, 1, 15, 9, 30)) == datetime(
        2026, 1, 15, 14, 30, tzinfo=UTC
    )
    aware = datetime(2026, 1, 15, 9, 30, tzinfo=timezone(timedelta(hours=8)))
    assert comparable_timestamp(aware) == datetime(2026, 1, 15, 1, 30, tzinfo=UTC)
    assert comparable_timestamp("not a datetime") == "not a datetime"


@pytest.mark.usefixtures("new_york_tz")
def test_comparable_timestamp_orders_naive_legacy_against_aware_utc() -> None:
    naive_earlier = datetime(2026, 1, 15, 9, 0)  # 14:00 UTC
    aware_middle = datetime(2026, 1, 15, 14, 15, tzinfo=UTC)
    naive_later = datetime(2026, 1, 15, 9, 30)  # 14:30 UTC

    ordered = sorted(
        [naive_later, aware_middle, naive_earlier], key=comparable_timestamp
    )

    assert ordered == [naive_earlier, aware_middle, naive_later]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime.min, datetime.min.replace(tzinfo=UTC)),
        (datetime.max, datetime.max.replace(tzinfo=UTC)),
        (
            datetime.min.replace(tzinfo=timezone(timedelta(hours=5))),
            datetime.min.replace(tzinfo=UTC),
        ),
        (
            datetime.max.replace(tzinfo=timezone(timedelta(hours=-5))),
            datetime.max.replace(tzinfo=UTC),
        ),
    ],
    ids=("naive-min", "naive-max", "aware-min", "aware-max"),
)
@pytest.mark.usefixtures("new_york_tz")
def test_comparable_timestamp_clamps_unrepresentable_extremes(
    value: datetime, expected: datetime
) -> None:
    assert comparable_timestamp(value) == expected


@pytest.mark.usefixtures("new_york_tz")
def test_extreme_date_filters_mean_unbounded() -> None:
    store = InMemoryMemoryStore()
    store.add(MemoryNote(id="legacy", content="a", timestamp=datetime(2026, 1, 1)))
    store.add(MemoryNote(id="current", content="b"))

    filters = {"date_from": datetime.min, "date_to": datetime.max}

    assert {note.id for note in store.list_all(filters=filters)} == {
        "legacy",
        "current",
    }
    assert {note.id for note in store.search("", k=10, filters=filters)} == {
        "legacy",
        "current",
    }
