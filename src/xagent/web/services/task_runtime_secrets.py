"""Run-scoped encrypted storage shared by request and execution processes."""

from __future__ import annotations

import json
from typing import Any, cast

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import delete, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from ...config import get_task_runtime_secrets_encryption_key
from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ERROR_RUNTIME_SECRET_UNAVAILABLE,
    RUNTIME_SECRET_REASON_STORE_LOST,
    ConnectorRef,
    ConnectorRuntimeError,
)
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_runtime_secret import TaskRuntimeSecret
from ..models.user import User


def _unavailable() -> ConnectorRuntimeError:
    return ConnectorRuntimeError(
        ERROR_RUNTIME_SECRET_UNAVAILABLE,
        "Connector runtime values are unavailable.",
        details={"reason": RUNTIME_SECRET_REASON_STORE_LOST},
        status_code=503,
    )


def _runtime_cipher() -> Fernet:
    key = get_task_runtime_secrets_encryption_key()
    if key is not None:
        try:
            return Fernet(key.encode())
        except (ValueError, UnicodeError):
            pass
    # Do not reuse get_cipher(): another feature may have cached the public
    # development fallback. Never include the configured key in this error.
    raise ConnectorRuntimeError(
        ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
        "Connector runtime storage requires a valid, non-default ENCRYPTION_KEY.",
        details={"reason": "encryption_key_unavailable"},
        status_code=503,
    )


def stage_runtime_values(
    db: Session,
    *,
    task_id: int,
    turn_id: str,
    values_by_ref: dict[ConnectorRef, dict[str, dict[str, Any]]],
) -> None:
    """Stage encrypted inputs in the caller's acceptance transaction.

    Bind them to the accepted run with ``bind_runtime_values_to_run`` before
    committing this same transaction. Unbound inputs must not be committed.
    Terminal settlement and the recovery sweep remove finished or replaced
    runs; paused and waiting runs retain their inputs without an independent TTL.
    """
    if not values_by_ref:
        return
    cipher = _runtime_cipher()
    owner_subject = db.execute(
        select(User.actor_subject)
        .join(Task, Task.user_id == User.id)
        .where(Task.id == task_id)
    ).scalar_one()
    body = {
        "task_id": task_id,
        "turn_id": turn_id,
        "owner_subject": owner_subject,
        "values": {ref.storage_key: values for ref, values in values_by_ref.items()},
    }
    ciphertext = cipher.encrypt(json.dumps(body, allow_nan=False).encode()).decode()
    db.add(
        TaskRuntimeSecret(
            task_id=task_id,
            turn_id=turn_id,
            owner_subject=owner_subject,
            ciphertext=ciphertext,
        )
    )
    db.flush()


def bind_runtime_values_to_run(
    db: Session, *, task_id: int, turn_id: str, run_id: str
) -> bool:
    result = db.execute(
        update(TaskRuntimeSecret)
        .where(
            TaskRuntimeSecret.task_id == task_id, TaskRuntimeSecret.turn_id == turn_id
        )
        .values(run_id=run_id)
    )
    return cast(CursorResult, result).rowcount == 1


def load_runtime_values(
    db: Session,
    *,
    task: Task,
    turn_id: str | None = None,
    required: bool = False,
) -> dict[str, Any] | None:
    query = select(TaskRuntimeSecret).where(TaskRuntimeSecret.task_id == task.id)
    if turn_id is not None:
        query = query.where(TaskRuntimeSecret.turn_id == turn_id)
    else:
        # Reply transcript IDs change; the accepted inputs belong to the run.
        query = query.where(TaskRuntimeSecret.run_id == task.run_id)
    row = db.execute(query).scalar_one_or_none()
    if row is None:
        if turn_id is None and task.run_id is not None:
            from ..models.task_command import TaskExecutionCommand

            accepted = db.execute(
                select(TaskExecutionCommand.payload).where(
                    TaskExecutionCommand.task_id == task.id,
                    TaskExecutionCommand.target_run_id == task.run_id,
                    TaskExecutionCommand.kind == "start",
                )
            ).scalar_one_or_none()
            required = required or bool(accepted and accepted.get("runtime_values_ref"))
        if required:
            raise _unavailable()
        return None
    owner = db.execute(
        select(User.actor_subject).where(User.id == task.user_id)
    ).scalar_one_or_none()
    if owner != row.owner_subject or row.run_id != task.run_id:
        raise _unavailable()
    try:
        body = json.loads(_runtime_cipher().decrypt(row.ciphertext.encode()))
    except (InvalidToken, ValueError, UnicodeError):
        raise _unavailable() from None
    if (
        not isinstance(body, dict)
        or body.get("task_id") != task.id
        or body.get("turn_id") != row.turn_id
        or body.get("owner_subject") != owner
        or not isinstance(body.get("values"), dict)
    ):
        raise _unavailable()
    return cast(dict[str, Any], body["values"])


def delete_runtime_values_no_commit(
    db: Session,
    *,
    task_id: int,
    turn_id: str | None = None,
    run_id: str | None = None,
) -> None:
    statement = delete(TaskRuntimeSecret).where(TaskRuntimeSecret.task_id == task_id)
    if turn_id is not None:
        statement = statement.where(TaskRuntimeSecret.turn_id == turn_id)
    if run_id is not None:
        statement = statement.where(TaskRuntimeSecret.run_id == run_id)
    db.execute(statement)


def delete_runtime_values(*, task_id: int, turn_id: str) -> None:
    with get_session_local()() as db:
        delete_runtime_values_no_commit(db, task_id=task_id, turn_id=turn_id)
        db.commit()


def clean_finished_runtime_values(
    *, task_id: int | None = None, run_id: str | None = None
) -> None:
    """Compensate interrupted cleanup without expiring queued or active inputs.

    Accepted inputs already have a run binding at commit; another session
    cannot observe the intermediate unbound rows in the acceptance transaction.
    Paused and waiting runs retain their accepted inputs for resumption on
    any worker. Only terminal or replaced runs are removed. Optional scope
    keeps an old execution's cleanup from deleting a newer run's inputs.
    """
    statement = (
        select(TaskRuntimeSecret.id)
        .join(Task)
        .where(
            TaskRuntimeSecret.run_id.is_distinct_from(Task.run_id)
            | (
                Task.runner_id.is_(None)
                & Task.status.in_((TaskStatus.COMPLETED, TaskStatus.FAILED))
            )
        )
    )
    if task_id is not None:
        statement = statement.where(TaskRuntimeSecret.task_id == task_id)
    if run_id is not None:
        statement = statement.where(TaskRuntimeSecret.run_id == run_id)
    with get_session_local()() as db:
        rows = db.execute(statement).scalars().all()
        if rows:
            db.execute(delete(TaskRuntimeSecret).where(TaskRuntimeSecret.id.in_(rows)))
        db.commit()
