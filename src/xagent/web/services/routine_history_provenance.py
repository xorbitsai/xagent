"""Routine-run transcript provenance.

The initiating turn identifies what teammates may read, while the task run id
keeps that identity scoped to the execution that the routine actually started.
Later owner conversations reuse the Task but must keep their own turn ids.
"""

from __future__ import annotations

from typing import Any

TRIGGER_TURN_ID_CONFIG_KEY = "trigger_turn_id"
TRIGGER_TASK_RUN_ID_CONFIG_KEY = "trigger_task_run_id"


def routine_history_turn_id(
    task: Any,
    fallback_turn_id: str | None,
    *,
    expected_run_id: str | None,
) -> str | None:
    """Return the shared-history turn for one execution of ``task``.

    Both identities are required. Failing closed to the current turn prevents
    a legacy/malformed trigger Task from exposing a later private conversation.
    """

    if getattr(task, "source", None) != "trigger":
        return fallback_turn_id
    config = dict(getattr(task, "agent_config", None) or {})
    initiating_turn_id = config.get(TRIGGER_TURN_ID_CONFIG_KEY)
    routine_run_id = config.get(TRIGGER_TASK_RUN_ID_CONFIG_KEY)
    if (
        isinstance(initiating_turn_id, str)
        and initiating_turn_id
        and isinstance(routine_run_id, str)
        and routine_run_id
        and isinstance(expected_run_id, str)
        and expected_run_id == routine_run_id
    ):
        return initiating_turn_id
    return fallback_turn_id


def load_routine_history_turn_id(
    task_id: int,
    fallback_turn_id: str | None,
    *,
    expected_run_id: str | None,
) -> str | None:
    """Resolve provenance in a short worker-owned database session."""

    from ..models.database import get_session_local
    from ..models.task import Task

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        if task is None:
            return fallback_turn_id
        return routine_history_turn_id(
            task,
            fallback_turn_id,
            expected_run_id=expected_run_id,
        )
