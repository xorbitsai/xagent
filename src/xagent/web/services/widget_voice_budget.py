"""Daily usage budgets for widget voice, whose callers are not signed in.

Synthesis is counted in characters in three buckets: per visitor (the widget
entity plus the visitor key), per widget entity, and per site. Each bucket is
a fixed UTC calendar-day window: the key carries the date, so a new day
starts from zero.

A reservation adds the amount to one bucket at a time and compares after each
addition; when a bucket ends over its limit, the buckets after it are left
alone, everything added so far is subtracted again, and the request is
refused. With ``XAGENT_REDIS_URL`` set the counters live only in
Redis (keys expire after 48 hours), and a client that cannot be built or any
Redis error refuses the request: there is no fallback to memory. Without it
the counters live in this process and start over when the process restarts.
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
    get_widget_voice_tts_entity_daily_chars,
    get_widget_voice_tts_max_text_chars,
    get_widget_voice_tts_site_daily_chars,
    get_widget_voice_tts_visitor_daily_chars,
)
from .db_runtime import await_task_settlement
from .writer_workers import run_tracked

logger = logging.getLogger(__name__)

_KEY_TTL_SECONDS = 48 * 3600
# The process-memory store keeps the same days a Redis key lives: the
# incoming key's UTC day and the day before it.
_MEMORY_KEEP_DAYS = _KEY_TTL_SECONDS // (24 * 3600)
_MAX_TEXT_CHARS_DEFAULT = 1000
# The text bound of TTSStreamRequest; a larger setting could never be sent.
_MAX_TEXT_CHARS_CEILING = 5000


class VoiceBudgetExhausted(Exception):
    """A budget bucket would go over its daily limit."""


class VoiceBudgetUnavailable(Exception):
    """The budget storage failed, so the request cannot be counted."""


@dataclass(frozen=True)
class BudgetBucket:
    name: str
    setting: Callable[[], str]
    # None: the bucket has no limit unless its setting gives one.
    default: int | None


TTS_BUCKETS = (
    BudgetBucket("tts-visitor", get_widget_voice_tts_visitor_daily_chars, None),
    BudgetBucket("tts-entity", get_widget_voice_tts_entity_daily_chars, None),
    BudgetBucket("tts-site", get_widget_voice_tts_site_daily_chars, 1_000_000),
)


_DefaultT = TypeVar("_DefaultT", bound="int | None")


def _parse_amount(
    raw: str, default: _DefaultT, name: str, ceiling: int | None
) -> int | _DefaultT:
    """0 is a real limit (refuse all); empty means the default; a negative,
    unparsable, or over-ceiling value falls back to the default with a
    warning. A ``None`` default means no limit."""
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0 or (ceiling is not None and value > ceiling):
        logger.warning(
            "Invalid widget voice budget %s=%r; falling back to %s",
            name,
            raw,
            "no limit" if default is None else default,
        )
        return default
    return value


def bucket_limit(bucket: BudgetBucket) -> int | None:
    """The bucket's daily limit, or ``None`` when it has no limit."""
    return _parse_amount(bucket.setting(), bucket.default, bucket.name, None)


def tts_max_text_chars() -> int:
    """The longest text one widget voice synthesis request may carry."""
    return _parse_amount(
        get_widget_voice_tts_max_text_chars(),
        _MAX_TEXT_CHARS_DEFAULT,
        "tts-max-text",
        _MAX_TEXT_CHARS_CEILING,
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
) -> None:
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
            limit = bucket_limit(bucket)
            if limit is not None and value > limit:
                raise VoiceBudgetExhausted(bucket.name)
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

    Raises :class:`VoiceBudgetExhausted` when a bucket would go over its
    limit and :class:`VoiceBudgetUnavailable` when the storage fails; either
    way nothing stays counted (unless the subtraction itself fails).
    """
    _reserve(TTS_BUCKETS, (f"{entity_key}|{visitor_key}", entity_key, "all"), chars)


async def reserve_tts(entity_key: str, visitor_key: str, chars: int) -> None:
    """:func:`reserve_tts_sync` off the event loop, counted as a tracked writer
    and drained on cancellation so a reservation is never left half-applied.

    The worker task is shielded from the caller's cancellation, so the tracked
    thread always starts and finishes; the cancellation is re-raised only after
    the reservation has settled."""
    worker = asyncio.get_running_loop().create_task(
        run_tracked(reserve_tts_sync, entity_key, visitor_key, chars)
    )
    _, cancellation = await await_task_settlement(worker)
    if cancellation is not None:
        raise cancellation
