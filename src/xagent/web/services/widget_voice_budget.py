"""Daily usage budgets for widget voice, whose callers are not signed in.

Synthesis is counted in characters and transcription in seconds, each in
three buckets: per visitor (the widget entity plus the visitor key), per
widget entity, and per site. Each bucket is a fixed UTC calendar-day window:
the key carries the date, so a new day starts from zero.

A reservation adds the amount to one bucket at a time and compares after each
addition; when a bucket ends over its limit, the buckets after it are left
alone, everything added so far is subtracted again, and the request is
refused. A transcription session reserves its longest possible duration up
front and refunds what it did not use when it ends, on the same day keys it
reserved under; a refund that fails only leaves the day over-counted. With
``XAGENT_REDIS_URL`` set the counters live only in
Redis (keys expire after 48 hours), and a client that cannot be built or any
Redis error refuses the request: there is no fallback to memory. Without it
the counters live in this process: each web process counts on its own, so N
processes admit up to N times each limit, and the counts start over when the
process restarts.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Protocol, TypeVar

from redis import Redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from ...config import (
    get_redis_url,
    get_widget_voice_asr_entity_daily_seconds,
    get_widget_voice_asr_site_daily_seconds,
    get_widget_voice_asr_visitor_daily_seconds,
    get_widget_voice_tts_entity_daily_chars,
    get_widget_voice_tts_site_daily_chars,
    get_widget_voice_tts_visitor_daily_chars,
)
from .db_runtime import await_task_settlement
from .writer_workers import run_tracked

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_KEY_TTL_SECONDS = 48 * 3600
# The process-memory store keeps the same days a Redis key lives: the
# incoming key's UTC day and the day before it.
_MEMORY_KEEP_DAYS = _KEY_TTL_SECONDS // (24 * 3600)


class VoiceBudgetExhausted(Exception):
    """A budget bucket would go over its daily limit."""


class VoiceBudgetUnavailable(Exception):
    """The budget storage failed, so the request cannot be counted."""


@dataclass(frozen=True)
class BudgetBucket:
    name: str
    # None: the bucket has no limit.
    limit: Callable[[], int | None]


TTS_BUCKETS = (
    BudgetBucket("tts-visitor", get_widget_voice_tts_visitor_daily_chars),
    BudgetBucket("tts-entity", get_widget_voice_tts_entity_daily_chars),
    BudgetBucket("tts-site", get_widget_voice_tts_site_daily_chars),
)
ASR_BUCKETS = (
    BudgetBucket("asr-visitor", get_widget_voice_asr_visitor_daily_seconds),
    BudgetBucket("asr-entity", get_widget_voice_asr_entity_daily_seconds),
    BudgetBucket("asr-site", get_widget_voice_asr_site_daily_seconds),
)


class _Store(Protocol):
    def incrby(self, key: str, amount: int) -> int: ...

    def decrby(self, key: str, amount: int) -> None: ...


class _MemoryStore:
    """Per-process counters for the incoming key's UTC day and the day before
    it, the two days a Redis key lives. Older days are dropped; a later day is
    never dropped, so a reservation that took its day just before midnight
    cannot erase the new day's counts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def incrby(self, key: str, amount: int) -> int:
        day = key.split(":", 3)[2]
        oldest_kept = (
            date.fromisoformat(day) - timedelta(days=_MEMORY_KEEP_DAYS - 1)
        ).strftime("%Y%m%d")
        with self._lock:
            for stale in [k for k in self._counts if k.split(":", 3)[2] < oldest_kept]:
                del self._counts[stale]
            self._counts[key] = self._counts.get(key, 0) + amount
            return self._counts[key]

    def decrby(self, key: str, amount: int) -> None:
        with self._lock:
            if key in self._counts:
                self._counts[key] -= amount


class _RedisStore:
    def __init__(self, client: Redis) -> None:
        self._client = client

    def incrby(self, key: str, amount: int) -> int:
        pipeline = self._client.pipeline()
        pipeline.incrby(key, amount)
        pipeline.expire(key, _KEY_TTL_SECONDS)
        value, _ = pipeline.execute()
        return int(value)

    def decrby(self, key: str, amount: int) -> None:
        self._client.decrby(key, amount)


_memory_store = _MemoryStore()


@lru_cache(maxsize=2)
def _redis_client(url: str) -> Redis:
    return Redis.from_url(
        url, socket_connect_timeout=2, socket_timeout=2, retry=Retry(NoBackoff(), 0)
    )


def _store() -> _Store:
    url = get_redis_url()
    if not url:
        return _memory_store
    return _RedisStore(_redis_client(url))


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _reserve(
    buckets: tuple[BudgetBucket, ...], scopes: tuple[str, ...], amount: int
) -> tuple[str, ...]:
    """Add ``amount`` to every bucket and return the keys it was added to.

    Raises :class:`VoiceBudgetExhausted` when a bucket would go over its
    limit and :class:`VoiceBudgetUnavailable` when the storage fails; either
    way nothing stays counted (unless the subtraction itself fails).
    """
    day = _utc_day()
    added: list[str] = []
    store: _Store | None = None
    try:
        store = _store()
        for bucket, scope in zip(buckets, scopes, strict=True):
            key = f"xagent:widget-voice:{day}:{bucket.name}:{scope}"
            value = store.incrby(key, amount)
            added.append(key)
            # A bucket without a limit is still counted.
            limit = bucket.limit()
            if limit is not None and value > limit:
                raise VoiceBudgetExhausted(bucket.name)
        return tuple(added)
    except BaseException as exc:
        # Best effort: a failed subtraction only leaves the day over-counted.
        for key in added:
            try:
                assert store is not None
                store.decrby(key, amount)
            except Exception:
                logger.warning("Widget voice budget rollback failed", exc_info=True)
        if isinstance(exc, (VoiceBudgetExhausted, VoiceBudgetUnavailable)):
            raise
        if isinstance(exc, Exception):
            logger.warning("Widget voice budget storage failed: %s", exc)
            raise VoiceBudgetUnavailable() from None
        raise


def reserve_tts_sync(entity_key: str, visitor_key: str, chars: int) -> None:
    """Count ``chars`` against the three synthesis buckets, or refuse.

    Raises like :func:`_reserve`.
    """
    _reserve(TTS_BUCKETS, (f"{entity_key}|{visitor_key}", entity_key, "all"), chars)


async def _run_settled(function: Callable[..., _T], *args: object) -> _T:
    """``function`` in a tracked worker thread that a cancellation of the
    caller cannot cut short.

    The worker task is shielded from the caller's cancellation, so the tracked
    thread always starts and finishes; the cancellation is re-raised only after
    the work has settled."""
    worker = asyncio.get_running_loop().create_task(run_tracked(function, *args))
    result, cancellation = await await_task_settlement(worker)
    if cancellation is not None:
        raise cancellation
    return result


async def reserve_tts(entity_key: str, visitor_key: str, chars: int) -> None:
    """:func:`reserve_tts_sync` off the event loop, counted as a tracked writer
    and drained on cancellation so a reservation is never left half-applied."""
    await _run_settled(reserve_tts_sync, entity_key, visitor_key, chars)


@dataclass(frozen=True)
class Reservation:
    """The counter keys one reservation added ``amount`` to."""

    keys: tuple[str, ...]
    amount: int


def reserve_asr_sync(entity_key: str, visitor_key: str, seconds: int) -> Reservation:
    """Count ``seconds`` against the three transcription buckets, or refuse.

    Raises like :func:`_reserve`. The returned reservation is what
    :func:`refund_sync` gives back from.
    """
    keys = _reserve(
        ASR_BUCKETS, (f"{entity_key}|{visitor_key}", entity_key, "all"), seconds
    )
    return Reservation(keys, seconds)


def refund_sync(reservation: Reservation, amount: int) -> None:
    """Subtract ``amount`` (at most the reserved amount) from every key the
    reservation counted on. Never raises: a failed subtraction is logged and
    only leaves that day over-counted."""
    amount = min(max(amount, 0), reservation.amount)
    if not amount:
        return
    try:
        store = _store()
        for key in reservation.keys:
            store.decrby(key, amount)
    except Exception:
        logger.warning("Widget voice budget refund failed", exc_info=True)


async def reserve_asr(entity_key: str, visitor_key: str, seconds: int) -> Reservation:
    """:func:`reserve_asr_sync` off the event loop. A cancellation that
    arrives while it runs waits for it, gives a completed reservation back,
    and is then raised, so nothing stays counted."""
    worker = asyncio.get_running_loop().create_task(
        run_tracked(reserve_asr_sync, entity_key, visitor_key, seconds)
    )
    reservation, cancellation = await await_task_settlement(worker)
    if cancellation is not None:
        await _run_settled(refund_sync, reservation, reservation.amount)
        raise cancellation
    return reservation


async def refund(reservation: Reservation, amount: int) -> None:
    """:func:`refund_sync` as a tracked writer. A cancellation that arrives
    while it runs waits for the refund to finish and is then raised."""
    await _run_settled(refund_sync, reservation, amount)
