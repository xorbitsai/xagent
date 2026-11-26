"""Backend generation identity and the serving generation record.

While two backend generations overlap during a release, exactly one of them may
claim new singleton work (triggers, auto-resume, lease recovery, chat intake and
the maintenance loops). The record names it; a process compares it with its own
identity, ``XAGENT_GENERATION``.

- ``XAGENT_GENERATION`` unset or empty: generations are off and every check
  passes, which is the behaviour before this module existed.
- No ``serving_generation`` row: every check passes (nothing has been handed
  over yet).
- Otherwise a process serves only while the row names its own generation.

Fail direction: a database error never reads as "serving". ``is_serving()``
returns ``False`` and logs, so a singleton skips the tick and tries again on the
next one. ``tick_allowed()`` is the guard the loops call. Singleton loops start
even when another generation serves, then remain dormant at each tick.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy import CursorResult, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

GENERATION_ENV = "XAGENT_GENERATION"
DORMANT_START = True
_RECORD_ID = 1
_cache_lock = threading.Lock()
_cached_generation: str | None = None
_cached_serving = False
_cached_at = float("-inf")
_serving_changed: asyncio.Event | None = None


def current_generation() -> str | None:
    """This process's generation, or ``None`` when generations are off."""
    value = os.getenv(GENERATION_ENV, "").strip()
    return value or None


def serving_generation() -> str | None:
    """The generation the record names, or ``None`` when the row is missing.

    Raises on a database error; callers decide the fail direction.
    """
    from ..models.database import get_session_local
    from ..models.serving_generation import ServingGeneration

    with get_session_local()() as db:
        row = db.get(ServingGeneration, _RECORD_ID)
        return str(row.generation) if row is not None else None


def set_serving_generation(new: str, expected: str | None) -> bool:
    """Move the record from ``expected`` to ``new`` in one conditional write.

    ``expected=None`` means the row must be missing (it is then created).
    Returns ``False`` when the record did not hold ``expected``, including a lost
    race on the first insert; it never overwrites a value it did not expect.
    """
    from ..models.database import get_session_local
    from ..models.serving_generation import ServingGeneration

    if not new:
        raise ValueError("a serving generation must be non-empty")
    with get_session_local()() as db:
        if expected is None:
            db.add(ServingGeneration(id=_RECORD_ID, generation=new))
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                return False
            return True
        result = db.execute(
            update(ServingGeneration)
            .where(
                ServingGeneration.id == _RECORD_ID,
                ServingGeneration.generation == expected,
            )
            .values(generation=new, updated_at=func.now())
        )
        db.commit()
        return bool(cast(CursorResult[Any], result).rowcount == 1)


def is_serving() -> bool:
    """Whether this process may claim new singleton work right now.

    ``True`` when generations are off, the row is missing, or the row names this
    process's generation. ``False`` on a database error (fail closed).
    """
    mine = current_generation()
    if mine is None:
        return True
    try:
        serving = serving_generation()
    except Exception:  # noqa: BLE001
        logger.warning(
            "component=serving-generation could not read the record; "
            "treating this process as not serving",
            exc_info=True,
        )
        return False
    return serving is None or serving == mine


def _read_serving(mine: str) -> bool:
    try:
        serving = serving_generation()
    except Exception:  # noqa: BLE001
        logger.warning(
            "could not read serving generation; refusing new work", exc_info=True
        )
        return False
    return serving is None or serving == mine


def is_serving_cached(max_age: float = 2.0) -> bool:
    """Bound command-claim reads; errors expire a previous positive result."""
    mine = current_generation()
    if mine is None:
        return True
    global _cached_generation, _cached_serving, _cached_at
    with _cache_lock:
        now = time.monotonic()
        if _cached_generation != mine or now - _cached_at >= max_age:
            _cached_serving = _read_serving(mine)
            _cached_generation = mine
            _cached_at = now
        return _cached_serving


async def refresh_serving() -> bool:
    """Read the record now and wake dormant loops when this generation serves."""
    mine = current_generation()
    if mine is None:
        return True
    serving = await asyncio.to_thread(_read_serving, mine)
    global _cached_generation, _cached_serving, _cached_at, _serving_changed
    with _cache_lock:
        _cached_generation = mine
        _cached_serving = serving
        _cached_at = time.monotonic()
    if serving:
        event = _serving_changed
        _serving_changed = asyncio.Event()
        if event is not None:
            event.set()
    return serving


async def wait_tick(
    component: str, seconds: float, *, stop_event: asyncio.Event | None = None
) -> None:
    """Wait for the next tick or an explicit serving refresh, whichever is first."""
    if current_generation() is None:
        if stop_event is None:
            await asyncio.sleep(seconds)
        else:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=seconds)
            except TimeoutError:
                pass
        return
    global _serving_changed
    if _serving_changed is None:
        _serving_changed = asyncio.Event()
    event = _serving_changed
    # Capture the event before reading. A serve notification between the last
    # failed guard and this wait must not be lost behind a fresh event object.
    if (stop_event is not None and stop_event.is_set()) or await asyncio.to_thread(
        is_serving
    ):
        return
    waits = [asyncio.create_task(event.wait())]
    if stop_event is not None:
        waits.append(asyncio.create_task(stop_event.wait()))
    try:
        await asyncio.wait(waits, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waits:
            waiter.cancel()
        await asyncio.gather(*waits, return_exceptions=True)


def serving_status() -> dict[str, object]:
    """Expose a fresh serving view for the lifecycle status endpoint."""
    mine = current_generation()
    checked_at = datetime.now(timezone.utc).isoformat()
    if mine is None:
        return {"generation": None, "serving": True, "checked_at": checked_at}
    try:
        serving = serving_generation()
    except Exception:  # noqa: BLE001
        logger.warning("could not read serving generation for status", exc_info=True)
        return {"generation": mine, "serving": None, "checked_at": checked_at}
    return {
        "generation": mine,
        "serving": serving is None or serving == mine,
        "checked_at": checked_at,
    }


def _definitely_not_serving() -> bool:
    """``True`` only when the record is readable and names another generation."""
    mine = current_generation()
    if mine is None:
        return False
    try:
        serving = serving_generation()
    except Exception:  # noqa: BLE001
        logger.warning(
            "component=serving-generation could not read the record at start; "
            "starting and letting the per-tick guard decide",
            exc_info=True,
        )
        return False
    return serving is not None and serving != mine


async def tick_allowed(component: str) -> bool:
    """Per-tick guard for a singleton loop; ``False`` means skip this tick."""
    if current_generation() is None:
        return True
    allowed = await asyncio.to_thread(is_serving)
    if not allowed:
        logger.debug(
            "component=%s skipped a tick: not the serving generation", component
        )
    return allowed


async def other_generation_serves(component: str) -> bool:
    """``True`` only when the record is readable and names another generation.

    For callers that must close something already running: an unreadable record
    is not enough to tear down a live intake, so it answers ``False`` there
    (and the caller simply opens nothing new, via ``tick_allowed``).
    """
    if current_generation() is None:
        return False
    other = await asyncio.to_thread(_definitely_not_serving)
    if other:
        logger.info("component=%s: another generation is serving", component)
    return other


def start_allowed(component: str) -> bool:
    """Start every singleton loop; its per-tick guard controls work."""
    if _definitely_not_serving():
        logger.info(
            "component=%s starting dormant: another generation serves", component
        )
    return True
