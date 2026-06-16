from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.agent import Agent
from ..models.background_job import BackgroundJob, BackgroundJobType
from ..models.task import Task, TaskStatus
from ..models.trigger import AgentTrigger, TriggerRun, TriggerRunStatus, TriggerType
from .background_jobs import create_background_job, enqueue_background_job
from .task_orchestrator import (
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)

logger = logging.getLogger(__name__)

_TRIGGER_SCOPE_PAYLOAD_KEYS = (
    "integration_id",
    "account_id",
    "mailbox_id",
    "channel_id",
    "tenant_id",
)

_PAYLOAD_PREVIEW_LIMIT = 64_000
_WEBHOOK_SECRET_BCRYPT_COST = 12
_TRIGGER_NAME_MAX_LENGTH = 200


class TriggerServiceError(ValueError):
    """Validation or state error raised by trigger service helpers."""


class TriggerNotFoundError(LookupError):
    """Raised when a trigger is missing or not owned by the caller."""


class TriggerSecretError(PermissionError):
    """Raised when a webhook secret does not match."""


@dataclass(frozen=True)
class _PreparedTriggerStart:
    run_id: int
    trigger_id: int
    task_id: int
    task_owner_user_id: int
    prompt: str
    trigger_type: str
    test: bool


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _payload_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = _json_dumps(payload)
    if len(encoded) <= _PAYLOAD_PREVIEW_LIMIT:
        return payload
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return {
        "truncated": True,
        "sha256": digest,
        "preview": encoded[:_PAYLOAD_PREVIEW_LIMIT],
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _hash_secret(secret: str) -> str:
    return bcrypt.hashpw(
        secret.encode("utf-8"),
        bcrypt.gensalt(rounds=_WEBHOOK_SECRET_BCRYPT_COST),
    ).decode("utf-8")


def verify_webhook_secret(trigger: AgentTrigger, provided_secret: str | None) -> None:
    expected = trigger.secret_hash
    if not expected:
        return
    if not provided_secret:
        raise TriggerSecretError("Missing webhook secret")
    try:
        matched = bcrypt.checkpw(
            provided_secret.encode("utf-8"),
            str(expected).encode("utf-8"),
        )
    except (TypeError, ValueError):
        matched = False
    if not matched:
        raise TriggerSecretError("Invalid webhook secret")


def _new_webhook_token() -> str:
    return secrets.token_urlsafe(32)


def _new_webhook_secret() -> str:
    return secrets.token_urlsafe(32)


def _normalize_trigger_type(trigger_type: str) -> str:
    try:
        normalized = TriggerType(trigger_type).value
    except ValueError as exc:
        raise TriggerServiceError(f"Unsupported trigger type: {trigger_type}") from exc
    return normalized


def _default_trigger_name(trigger_type: str) -> str:
    if trigger_type == TriggerType.WEBHOOK.value:
        return "Webhook trigger"
    if trigger_type == TriggerType.SCHEDULED.value:
        return "Scheduled trigger"
    return "Agent trigger"


def _normalize_trigger_name(name: str | None, *, default: str | None = None) -> str:
    resolved = default if name is None else name
    value = str(resolved or "").strip()
    if not value:
        raise TriggerServiceError("Trigger name must not be empty")
    if len(value) > _TRIGGER_NAME_MAX_LENGTH:
        raise TriggerServiceError(
            f"Trigger name must be at most {_TRIGGER_NAME_MAX_LENGTH} characters"
        )
    return value


def _compute_next_run_at(
    config: dict[str, Any],
    *,
    from_time: datetime | None = None,
    previous_due_at: datetime | None = None,
    include_explicit: bool = True,
) -> datetime | None:
    """Compute the next scheduled fire time for the supported MVP config."""
    base = _coerce_utc(previous_due_at) or from_time or _now()
    base = _coerce_utc(base) or _now()

    if include_explicit:
        explicit_next = config.get("next_run_at")
        if isinstance(explicit_next, str) and explicit_next.strip():
            try:
                return _coerce_utc(datetime.fromisoformat(explicit_next))
            except ValueError as exc:
                raise TriggerServiceError("Invalid next_run_at") from exc

    interval = config.get("interval_seconds")
    if interval is None:
        return None
    try:
        interval_seconds = int(interval)
    except (TypeError, ValueError) as exc:
        raise TriggerServiceError("interval_seconds must be an integer") from exc
    if interval_seconds <= 0:
        raise TriggerServiceError("interval_seconds must be positive")

    candidate = base + timedelta(seconds=interval_seconds)
    now = from_time or _now()
    now = _coerce_utc(now) or _now()
    if candidate <= now:
        elapsed_seconds = (now - base).total_seconds()
        steps = int(elapsed_seconds // interval_seconds) + 1
        candidate = base + timedelta(seconds=steps * interval_seconds)
    return candidate


def _validate_config(trigger_type: str, config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise TriggerServiceError("config must be an object")
    if trigger_type == TriggerType.SCHEDULED.value:
        if "interval_seconds" not in config and "next_run_at" not in config:
            raise TriggerServiceError(
                "scheduled trigger requires interval_seconds or next_run_at"
            )
        _compute_next_run_at(config)


def get_owned_agent(db: Session, *, user_id: int, agent_id: int) -> Agent | None:
    return (
        db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == user_id).first()
    )


def get_owned_trigger(
    db: Session,
    *,
    user_id: int,
    agent_id: int,
    trigger_id: int,
) -> AgentTrigger | None:
    return (
        db.query(AgentTrigger)
        .filter(
            AgentTrigger.id == trigger_id,
            AgentTrigger.agent_id == agent_id,
            AgentTrigger.user_id == user_id,
        )
        .first()
    )


def create_agent_trigger(
    db: Session,
    *,
    user_id: int,
    agent_id: int,
    trigger_type: str,
    name: str | None = None,
    enabled: bool = True,
    config: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    secret: str | None = None,
) -> tuple[AgentTrigger, str | None]:
    agent = get_owned_agent(db, user_id=user_id, agent_id=agent_id)
    if agent is None:
        raise TriggerNotFoundError("Agent not found")

    resolved_type = _normalize_trigger_type(trigger_type)
    resolved_config = dict(config or {})
    _validate_config(resolved_type, resolved_config)

    plain_secret: str | None = None
    webhook_token: str | None = None
    secret_hash: str | None = None
    if resolved_type == TriggerType.WEBHOOK.value:
        webhook_token = _new_webhook_token()
        plain_secret = secret or _new_webhook_secret()
        secret_hash = _hash_secret(plain_secret)

    next_run_at = None
    if resolved_type == TriggerType.SCHEDULED.value and enabled:
        next_run_at = _compute_next_run_at(resolved_config)

    trigger = AgentTrigger(
        user_id=user_id,
        agent_id=agent_id,
        type=resolved_type,
        name=_normalize_trigger_name(
            name, default=_default_trigger_name(resolved_type)
        ),
        enabled=enabled,
        config=resolved_config,
        prompt_template=prompt_template,
        webhook_token=webhook_token,
        secret_hash=secret_hash,
        next_run_at=next_run_at,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger, plain_secret


def update_agent_trigger(
    db: Session,
    *,
    user_id: int,
    agent_id: int,
    trigger_id: int,
    updates: dict[str, Any],
) -> tuple[AgentTrigger, str | None]:
    trigger = get_owned_trigger(
        db, user_id=user_id, agent_id=agent_id, trigger_id=trigger_id
    )
    if trigger is None:
        raise TriggerNotFoundError("Trigger not found")

    plain_secret: str | None = None
    if "name" in updates and updates["name"] is not None:
        setattr(trigger, "name", _normalize_trigger_name(str(updates["name"])))
    if "enabled" in updates and updates["enabled"] is not None:
        setattr(trigger, "enabled", bool(updates["enabled"]))
    if "prompt_template" in updates:
        setattr(trigger, "prompt_template", updates["prompt_template"])
    if "config" in updates and updates["config"] is not None:
        config = dict(updates["config"])
        _validate_config(str(trigger.type), config)
        setattr(trigger, "config", config)
    if "secret" in updates and updates["secret"]:
        plain_secret = str(updates["secret"])
        setattr(trigger, "secret_hash", _hash_secret(plain_secret))
    elif updates.get("rotate_secret"):
        plain_secret = _new_webhook_secret()
        setattr(trigger, "secret_hash", _hash_secret(plain_secret))

    if trigger.type == TriggerType.SCHEDULED.value:
        if trigger.enabled:
            setattr(
                trigger,
                "next_run_at",
                _compute_next_run_at(dict(trigger.config or {})),
            )
        else:
            setattr(trigger, "next_run_at", None)

    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger, plain_secret


def delete_agent_trigger(
    db: Session,
    *,
    user_id: int,
    agent_id: int,
    trigger_id: int,
) -> None:
    trigger = get_owned_trigger(
        db, user_id=user_id, agent_id=agent_id, trigger_id=trigger_id
    )
    if trigger is None:
        raise TriggerNotFoundError("Trigger not found")
    db.delete(trigger)
    db.commit()


def render_trigger_prompt(
    trigger: AgentTrigger,
    *,
    event_payload: dict[str, Any],
    source_event_id: str | None = None,
    test: bool = False,
) -> str:
    payload_json = json.dumps(event_payload, ensure_ascii=False, indent=2, default=str)
    template = (trigger.prompt_template or "").strip()
    if template:
        replacements = {
            "{{payload}}": payload_json,
            "{{trigger_type}}": str(trigger.type),
            "{{source_event_id}}": source_event_id or "",
            "{{test}}": "true" if test else "false",
        }
        rendered = template
        for key, value in replacements.items():
            rendered = rendered.replace(key, value)
        return rendered

    label = "test " if test else ""
    return (
        f"Handle this {label}{trigger.type} trigger event.\n\n"
        f"Trigger: {trigger.name}\n"
        f"Source event ID: {source_event_id or 'none'}\n\n"
        f"Event payload:\n{payload_json}"
    )


def _event_source_id(event_payload: dict[str, Any], source_event_id: str | None) -> str:
    if source_event_id:
        return source_event_id
    for key in ("id", "event_id", "message_id"):
        value = event_payload.get(key)
        if value:
            return str(value)
    return f"payload:{_payload_hash(event_payload)}"


def _trigger_run_idempotency_key(
    trigger: AgentTrigger,
    *,
    event_payload: dict[str, Any],
    source_event_id: str | None,
    test: bool,
) -> str:
    if test:
        return f"trigger-run:test:{trigger.id}:{secrets.token_urlsafe(16)}"
    event_identity = _event_source_id(event_payload, source_event_id)
    return f"trigger-run:{trigger.id}:{event_identity}"


def _get_or_create_trigger_run(
    db: Session,
    *,
    trigger: AgentTrigger,
    event_payload: dict[str, Any],
    source_event_id: str | None,
    background_job_id: str | None,
    test: bool,
) -> tuple[TriggerRun, bool]:
    idempotency_key = _trigger_run_idempotency_key(
        trigger,
        event_payload=event_payload,
        source_event_id=source_event_id,
        test=test,
    )
    existing = (
        db.query(TriggerRun)
        .filter(TriggerRun.idempotency_key == idempotency_key)
        .first()
    )
    if existing is not None:
        return existing, False

    run = TriggerRun(
        trigger_id=int(trigger.id),
        background_job_id=background_job_id,
        status=TriggerRunStatus.PENDING.value,
        source_event_id=source_event_id,
        payload_snapshot=_payload_snapshot(event_payload),
        idempotency_key=idempotency_key,
    )
    db.add(run)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(TriggerRun)
            .filter(TriggerRun.idempotency_key == idempotency_key)
            .first()
        )
        if existing is not None:
            return existing, False
        raise
    db.refresh(run)
    return run, True


def _mark_run_failed(
    db: Session,
    *,
    trigger: AgentTrigger,
    run: TriggerRun,
    error_message: str,
) -> None:
    setattr(run, "status", TriggerRunStatus.FAILED.value)
    setattr(run, "error_message", error_message)
    setattr(run, "finished_at", _now())
    setattr(trigger, "last_error", error_message)
    db.add(run)
    db.add(trigger)
    db.commit()


def _trigger_task_title(trigger: AgentTrigger, prompt: str) -> str:
    title = f"{trigger.name}: {prompt[:50]}"
    if len(title) > 80:
        title = title[:77] + "..."
    return title


def _trigger_execution_context(
    *,
    trigger: AgentTrigger,
    run: TriggerRun,
    test: bool,
) -> dict[str, Any]:
    return {
        "trigger_id": int(trigger.id),
        "trigger_run_id": int(run.id),
        "trigger_type": str(trigger.type),
        "trigger_test": test,
    }


def _attach_task_to_trigger_run(
    db: Session,
    *,
    trigger: AgentTrigger,
    run: TriggerRun,
    event_payload: dict[str, Any],
    source_event_id: str | None,
    test: bool,
) -> TriggerRun:
    if run.task_id is not None:
        return run

    prompt = render_trigger_prompt(
        trigger,
        event_payload=event_payload,
        source_event_id=source_event_id,
        test=test,
    )
    agent = db.query(Agent).filter(Agent.id == trigger.agent_id).first()
    task = Task(
        user_id=int(trigger.user_id),
        title=_trigger_task_title(trigger, prompt),
        description=prompt,
        status=TaskStatus.PENDING,
        agent_id=int(trigger.agent_id),
        execution_mode=getattr(agent, "execution_mode", None) or "balanced",
        source="trigger",
        is_visible=False,
        input=prompt,
        agent_config=_trigger_execution_context(
            trigger=trigger,
            run=run,
            test=test,
        ),
    )
    db.add(task)
    db.flush()
    run.task_id = int(task.id)
    run.status = TriggerRunStatus.PENDING.value
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def prepare_trigger_run(
    db: Session,
    *,
    trigger: AgentTrigger,
    event_payload: dict[str, Any],
    source_event_id: str | None = None,
    background_job_id: str | None = None,
    test: bool = False,
) -> tuple[TriggerRun, bool]:
    """Persist a trigger run and hidden task without starting agent execution."""
    if not test and not trigger.enabled:
        raise TriggerServiceError("Trigger is disabled")

    run, created = _get_or_create_trigger_run(
        db,
        trigger=trigger,
        event_payload=event_payload,
        source_event_id=source_event_id,
        background_job_id=background_job_id,
        test=test,
    )
    if not created and run.task_id is not None:
        return run, False

    try:
        run = _attach_task_to_trigger_run(
            db,
            trigger=trigger,
            run=run,
            event_payload=event_payload,
            source_event_id=source_event_id,
            test=test,
        )
        return run, created
    except Exception as exc:
        db.rollback()
        error_message = f"{type(exc).__name__}: {exc}"
        _mark_run_failed(db, trigger=trigger, run=run, error_message=error_message)
        logger.exception("Trigger run %s failed to prepare task", run.id)
        return run, True


def _with_session() -> Session:
    from ..models.database import get_session_local

    return get_session_local()()


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


def _claim_pending_trigger_run(db: Session, run_id: int) -> bool:
    claim_time = _now()
    result = db.execute(
        update(TriggerRun)
        .where(TriggerRun.id == run_id)
        .where(TriggerRun.status == TriggerRunStatus.PENDING.value)
        .values(
            status=TriggerRunStatus.RUNNING.value,
            started_at=claim_time,
            error_message=None,
        )
        .execution_options(synchronize_session=False)
    )
    db.commit()
    return _rowcount(result) == 1


def _load_prepared_trigger_start(run_id: int) -> _PreparedTriggerStart | None:
    db = _with_session()
    try:
        if not _claim_pending_trigger_run(db, run_id):
            return None
        run = db.query(TriggerRun).filter(TriggerRun.id == run_id).first()
        if run is None:
            return None
        trigger = (
            db.query(AgentTrigger)
            .filter(AgentTrigger.id == int(run.trigger_id))
            .first()
        )
        if run.task_id is None:
            if trigger is not None:
                _mark_run_failed(
                    db,
                    trigger=trigger,
                    run=run,
                    error_message="Trigger run has no prepared task",
                )
            return None

        task = db.query(Task).filter(Task.id == int(run.task_id)).first()
        if task is None or trigger is None:
            if trigger is not None:
                _mark_run_failed(
                    db,
                    trigger=trigger,
                    run=run,
                    error_message="Prepared trigger task or trigger is missing",
                )
            return None

        if task.status == TaskStatus.RUNNING:
            setattr(run, "status", TriggerRunStatus.RUNNING.value)
            setattr(run, "started_at", run.started_at or _now())
            db.add(run)
            db.commit()
            return None
        if task.status == TaskStatus.COMPLETED:
            setattr(run, "status", TriggerRunStatus.COMPLETED.value)
            setattr(run, "error_message", None)
            setattr(run, "finished_at", run.finished_at or _now())
            db.add(run)
            db.commit()
            return None
        if task.status == TaskStatus.FAILED:
            setattr(run, "status", TriggerRunStatus.FAILED.value)
            setattr(run, "error_message", task.error_message)
            setattr(run, "finished_at", run.finished_at or _now())
            db.add(run)
            db.commit()
            return None
        if task.status != TaskStatus.PENDING:
            return None

        task_config = dict(task.agent_config or {})
        return _PreparedTriggerStart(
            run_id=int(run.id),
            trigger_id=int(trigger.id),
            task_id=int(task.id),
            task_owner_user_id=int(task.user_id),
            prompt=str(task.input or task.description or ""),
            trigger_type=str(trigger.type),
            test=bool(task_config.get("trigger_test")),
        )
    finally:
        db.close()


def _mark_trigger_run_started(start: _PreparedTriggerStart) -> None:
    db = _with_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == start.run_id).first()
        trigger = (
            db.query(AgentTrigger).filter(AgentTrigger.id == start.trigger_id).first()
        )
        if run is None or trigger is None:
            return
        started_at = run.started_at or _now()
        setattr(run, "status", TriggerRunStatus.RUNNING.value)
        setattr(run, "started_at", started_at)
        setattr(run, "error_message", None)
        setattr(trigger, "last_run_at", started_at)
        setattr(trigger, "last_error", None)
        db.add(run)
        db.add(trigger)
        db.commit()
    finally:
        db.close()


def _mark_trigger_run_failed_by_id(run_id: int, error_message: str) -> None:
    db = _with_session()
    try:
        run = db.query(TriggerRun).filter(TriggerRun.id == run_id).first()
        if run is None:
            return
        trigger = (
            db.query(AgentTrigger)
            .filter(AgentTrigger.id == int(run.trigger_id))
            .first()
        )
        if trigger is None:
            return
        _mark_run_failed(db, trigger=trigger, run=run, error_message=error_message)
    finally:
        db.close()


def _mark_trigger_run_running_if_task_running(run_id: int, task_id: int) -> bool:
    db = _with_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        run = db.query(TriggerRun).filter(TriggerRun.id == run_id).first()
        if task is None or run is None or task.status != TaskStatus.RUNNING:
            return False
        setattr(run, "status", TriggerRunStatus.RUNNING.value)
        setattr(run, "started_at", run.started_at or _now())
        db.add(run)
        db.commit()
        return True
    finally:
        db.close()


def _finish_trigger_run_after_task(start: _PreparedTriggerStart) -> None:
    db = _with_session()
    try:
        task = db.query(Task).filter(Task.id == start.task_id).first()
        run = db.query(TriggerRun).filter(TriggerRun.id == start.run_id).first()
        if task is None or run is None:
            return
        if task.status == TaskStatus.COMPLETED:
            setattr(run, "status", TriggerRunStatus.COMPLETED.value)
            setattr(run, "error_message", None)
        elif task.status == TaskStatus.FAILED:
            setattr(run, "status", TriggerRunStatus.FAILED.value)
            setattr(run, "error_message", task.error_message)
        setattr(run, "finished_at", _now())
        db.add(run)
        db.commit()
    finally:
        db.close()


async def _start_prepared_trigger_run_id(
    run_id: int,
    *,
    wait_for_completion: bool = False,
) -> bool:
    """Start one prepared trigger task from the backend process."""
    start = await asyncio.to_thread(_load_prepared_trigger_start, run_id)
    if start is None:
        return False

    context = {
        "trigger_id": start.trigger_id,
        "trigger_run_id": start.run_id,
        "trigger_type": start.trigger_type,
        "trigger_test": start.test,
    }
    try:
        started = await TaskTurnOrchestrator.begin_turn(
            task_id=start.task_id,
            task_owner_user_id=start.task_owner_user_id,
            payload=TaskTurnPayload(transcript_message=start.prompt),
            kind=TurnKind.CREATE,
            force_fresh=False,
            context=context,
            actor_user_id=start.task_owner_user_id,
        )
    except TaskTurnError as exc:
        marked_running = await asyncio.to_thread(
            _mark_trigger_run_running_if_task_running,
            start.run_id,
            start.task_id,
        )
        if marked_running:
            return False
        await asyncio.to_thread(
            _mark_trigger_run_failed_by_id,
            start.run_id,
            f"TaskTurnError: {exc.reason}",
        )
        logger.info("Trigger run %s was not started: %s", start.run_id, exc.reason)
        return False
    except TaskTurnNotFoundError as exc:
        await asyncio.to_thread(
            _mark_trigger_run_failed_by_id,
            start.run_id,
            f"{type(exc).__name__}: {exc}",
        )
        return False
    except Exception as exc:
        await asyncio.to_thread(
            _mark_trigger_run_failed_by_id,
            start.run_id,
            f"{type(exc).__name__}: {exc}",
        )
        logger.exception("Trigger run %s failed to start task", start.run_id)
        return False

    await asyncio.to_thread(_mark_trigger_run_started, start)

    if wait_for_completion and asyncio.isfuture(started.background_task):
        await started.background_task
        await asyncio.to_thread(_finish_trigger_run_after_task, start)

    return True


async def start_prepared_trigger_run(
    db: Session,
    *,
    run: TriggerRun,
    wait_for_completion: bool = False,
) -> bool:
    """Start one prepared trigger task from the backend process."""
    return await _start_prepared_trigger_run_id(
        int(run.id),
        wait_for_completion=wait_for_completion,
    )


async def fire_trigger(
    db: Session,
    *,
    trigger: AgentTrigger,
    event_payload: dict[str, Any],
    source_event_id: str | None = None,
    background_job_id: str | None = None,
    test: bool = False,
    wait_for_completion: bool = False,
) -> tuple[TriggerRun, bool]:
    """Prepare a trigger event and start it in the current backend process."""
    run, created = prepare_trigger_run(
        db,
        trigger=trigger,
        event_payload=event_payload,
        source_event_id=source_event_id,
        background_job_id=background_job_id,
        test=test,
    )
    if created:
        await start_prepared_trigger_run(
            db,
            run=run,
            wait_for_completion=wait_for_completion,
        )
        db.refresh(run)
    return run, created


def _get_pending_trigger_run_ids(limit: int) -> list[int]:
    """Fetch pending run ids using a thread-local database session."""
    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        rows = (
            db.query(TriggerRun.id)
            .join(Task, TriggerRun.task_id == Task.id)
            .filter(
                TriggerRun.status == TriggerRunStatus.PENDING.value,
                Task.status == TaskStatus.PENDING,
            )
            .order_by(TriggerRun.created_at.asc(), TriggerRun.id.asc())
            .limit(limit)
            .all()
        )
        return [int(row[0]) for row in rows]
    finally:
        db.close()


async def dispatch_pending_trigger_runs(
    db: Session,
    *,
    limit: int = 20,
    wait_for_completion: bool = False,
) -> int:
    """Start prepared trigger tasks from the backend process."""
    pending_run_ids = await asyncio.to_thread(
        _get_pending_trigger_run_ids,
        max(1, min(limit, 100)),
    )
    if not pending_run_ids:
        return 0

    started_count = 0
    for run_id in pending_run_ids:
        if await _start_prepared_trigger_run_id(
            run_id,
            wait_for_completion=wait_for_completion,
        ):
            started_count += 1
    return started_count


def find_webhook_trigger(db: Session, webhook_token: str) -> AgentTrigger | None:
    return (
        db.query(AgentTrigger)
        .filter(
            AgentTrigger.webhook_token == webhook_token,
            AgentTrigger.type == TriggerType.WEBHOOK.value,
        )
        .first()
    )


def scan_due_scheduled_triggers(
    db: Session,
    *,
    now: datetime | None = None,
    limit: int = 100,
) -> list[TriggerRun]:
    """Prepare due scheduled triggers; backend dispatcher starts the tasks."""
    scan_time = _coerce_utc(now) or _now()
    due_triggers = (
        db.query(AgentTrigger)
        .filter(
            AgentTrigger.type == TriggerType.SCHEDULED.value,
            AgentTrigger.enabled.is_(True),
            AgentTrigger.next_run_at.is_not(None),
            AgentTrigger.next_run_at <= scan_time,
        )
        .order_by(AgentTrigger.next_run_at.asc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
    runs: list[TriggerRun] = []
    for trigger in due_triggers:
        due_at = _coerce_utc(getattr(trigger, "next_run_at", None)) or scan_time
        payload = {
            "trigger_id": int(trigger.id),
            "scheduled_at": scan_time.isoformat(),
            "due_at": due_at.isoformat(),
        }
        source_event_id = f"scheduled:{trigger.id}:{due_at.isoformat()}"
        run, _created = prepare_trigger_run(
            db,
            trigger=trigger,
            event_payload=payload,
            source_event_id=source_event_id,
            background_job_id=None,
            test=False,
        )

        config = dict(trigger.config or {})
        next_run_at = _compute_next_run_at(
            config,
            from_time=scan_time,
            previous_due_at=due_at,
            include_explicit=False,
        )
        setattr(trigger, "next_run_at", next_run_at)
        if next_run_at is None:
            setattr(trigger, "enabled", False)
        db.add(trigger)
        db.commit()
        runs.append(run)
    return runs


def _trigger_idempotency_scope(event_payload: dict[str, Any]) -> str:
    for key in _TRIGGER_SCOPE_PAYLOAD_KEYS:
        value = event_payload.get(key)
        if value is not None:
            return f"{key}:{value}"
    return "default"


def enqueue_trigger_event_job(
    db: Session,
    *,
    user_id: int,
    source_type: str,
    event_type: str,
    event_payload: dict[str, Any],
    source_event_id: str | None = None,
    trigger_id: int | None = None,
) -> BackgroundJob:
    """Persist and enqueue a trigger event job.

    Generic source_type/event_type payloads remain supported for the existing
    background-job tests. New agent-trigger callers can include trigger_id.
    """
    idempotency_key = (
        f"trigger:{user_id}:{source_type}:"
        f"{_trigger_idempotency_scope(event_payload)}:{source_event_id}"
        if source_event_id
        else None
    )
    job = create_background_job(
        db,
        user_id=user_id,
        job_type=BackgroundJobType.TRIGGER_EVENT,
        payload={
            "user_id": user_id,
            "trigger_id": trigger_id,
            "source_type": source_type,
            "event_type": event_type,
            "source_event_id": source_event_id,
            "event_payload": event_payload,
        },
        idempotency_key=idempotency_key,
    )
    return enqueue_background_job(db, job)
