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
next one. ``tick_allowed()`` is the guard the loops call; ``start_allowed()`` is
the start-time form, which only refuses on a definite answer (the row names
another generation) so that a restart during a database blip still starts the
loop and lets the per-tick guard hold it back until the database answers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, cast

from sqlalchemy import CursorResult, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

GENERATION_ENV = "XAGENT_GENERATION"
_RECORD_ID = 1


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


def start_allowed(component: str) -> bool:
    """Start-time guard: ``False`` only when another generation definitely serves."""
    if _definitely_not_serving():
        logger.info(
            "component=%s not started: another generation is serving", component
        )
        return False
    return True
