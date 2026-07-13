from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import ValidationError
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_RUNTIME_SECRET_UNAVAILABLE,
    ERROR_SCHEDULED_SECRET_UNAVAILABLE,
    ConnectorRuntimeError,
)
from ...core.utils.encryption import decrypt_value, encrypt_value
from ..models.agent import Agent
from ..models.background_job import BackgroundJob, BackgroundJobType
from ..models.task import Task, TaskStatus
from ..models.trigger import AgentTrigger, TriggerRun, TriggerRunStatus, TriggerType
from ..models.user_oauth import UserOAuth
from .agent_team_scope import get_agent_team_scope, owned_agent_clause
from .background_jobs import create_background_job, enqueue_background_job
from .connector_runtime import (
    persist_create_connector_runtime_context,
    prepare_create_connector_runtime,
    reject_ephemeral_connector_runtime_payload,
)
from .task_orchestrator import (
    TaskTurnError,
    TaskTurnNotFoundError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)
from .trigger_providers.base import TriggerConfigError
from .trigger_providers.registry import maybe_get_trigger_provider
from .trigger_providers.schemas import parse_trigger_config

logger = logging.getLogger(__name__)

_TRIGGER_SCOPE_PAYLOAD_KEYS = (
    "integration_id",
    "account_id",
    "mailbox_id",
    "channel_id",
    "tenant_id",
)

_TRIGGER_NAME_MAX_LENGTH = 200


class TriggerServiceError(ValueError):
    """Validation or state error raised by trigger service helpers."""


class TriggerNotFoundError(LookupError):
    """Raised when a trigger is missing or not owned by the caller."""


class TriggerSecretError(PermissionError):
    """Raised when a webhook secret does not match."""


class TriggerRunPreparationError(TriggerServiceError):
    """A trigger run was recorded but its task could not be prepared.

    The run row exists (marked FAILED with no task attached), so a redelivery
    of the same event resolves to it via the idempotency key and retries the
    task attachment. Callers that gate acknowledgement or cursor advancement
    on successful processing must treat this as a failure so the source
    redelivers instead of silently dropping the event.
    """

    def __init__(self, message: str, *, run: TriggerRun) -> None:
        super().__init__(message)
        self.run = run


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


def _store_full_payload_enabled(trigger: AgentTrigger) -> bool:
    config: dict[str, Any] = trigger.config if isinstance(trigger.config, dict) else {}
    return bool(config.get("store_full_payload"))


def _payload_snapshot(
    trigger: AgentTrigger,
    payload: dict[str, Any],
    *,
    source_event_id: str | None,
    event_type: str | None,
    resource_id: str | None,
    received_at: datetime | None = None,
) -> dict[str, Any]:
    """Conservative trigger-run snapshot: stable hash plus allow-listed metadata.

    Event content (e.g. Gmail sender/subject/snippet/body/headers) is never
    stored by default. Full payload content is stored only when the trigger
    explicitly opts in via store_full_payload, and then only encrypted.
    """
    snapshot: dict[str, Any] = {
        "payload_sha256": _payload_hash(payload),
        "metadata": {
            "source_event_id": source_event_id,
            "event_type": event_type,
            "resource_id": resource_id,
            "received_at": (received_at or _now()).isoformat(),
        },
    }
    if _store_full_payload_enabled(trigger):
        snapshot["encrypted_payload"] = encrypt_value(_json_dumps(payload))
    return snapshot


def decrypt_trigger_run_payload(run: TriggerRun) -> Any:
    """Return the decrypted original payload of a trigger run.

    Raises TriggerServiceError when the run did not store an encrypted full
    payload (full payload storage was not enabled at event time).
    """
    snapshot = run.payload_snapshot
    if not isinstance(snapshot, dict) or "encrypted_payload" not in snapshot:
        raise TriggerServiceError(
            "Full payload storage was not enabled for this trigger run"
        )
    decrypted = decrypt_value(str(snapshot["encrypted_payload"]))
    try:
        return json.loads(decrypted)
    except ValueError as exc:
        raise TriggerServiceError("Failed to decrypt trigger run payload") from exc


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _new_callback_id() -> str:
    return secrets.token_urlsafe(24)


def _new_webhook_secret() -> str:
    return secrets.token_urlsafe(32)


def find_webhook_trigger(db: Session, webhook_token: str) -> AgentTrigger | None:
    """Resolve a legacy webhook trigger by its pre-pipeline webhook token.

    Deprecated: only serves triggers created before the unified callback
    pipeline. New triggers carry a callback_id instead of a webhook token.
    """
    return (
        db.query(AgentTrigger)
        .filter(
            AgentTrigger.webhook_token == webhook_token,
            AgentTrigger.type == TriggerType.WEBHOOK.value,
        )
        .first()
    )


def verify_webhook_secret(trigger: AgentTrigger, provided_secret: str | None) -> None:
    """Verify the legacy bcrypt-hashed webhook secret.

    Deprecated alongside find_webhook_trigger. Unlike the historical
    behavior, a trigger without a stored secret hash is rejected instead of
    accepted, so the legacy route can never run unauthenticated.
    """
    import bcrypt

    expected = trigger.secret_hash
    if not expected:
        raise TriggerSecretError("Webhook trigger has no legacy secret")
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
    if trigger_type == TriggerType.GMAIL.value:
        return "Gmail trigger"
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


def _typed_config_error(trigger_type: str, exc: ValidationError) -> TriggerServiceError:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(
            str(item) for item in error.get("loc", []) if item != "config"
        )
        message = str(error.get("msg", "invalid value"))
        message = message.removeprefix("Value error, ")
        parts.append(f"{location}: {message}" if location else message)
    detail = "; ".join(parts) or "invalid config"
    return TriggerServiceError(f"{trigger_type} trigger config invalid: {detail}")


def _resolve_gmail_resource(
    db: Session, *, user_id: int, oauth_account_id: int | None
) -> str:
    """Validate the bound Gmail account and return the normalized mailbox."""
    if oauth_account_id is None:
        raise TriggerServiceError("gmail trigger requires oauth_account_id")
    account = db.query(UserOAuth).filter(UserOAuth.id == int(oauth_account_id)).first()
    if account is None or int(account.user_id) != int(user_id):
        raise TriggerServiceError("Gmail account not found")
    if str(account.provider) != "gmail":
        raise TriggerServiceError("Selected account is not a Gmail account")
    email = str(account.email or "").strip().lower()
    if not email:
        raise TriggerServiceError("Gmail account has no email address")
    return email


def _validate_config(
    db: Session,
    *,
    user_id: int,
    trigger_type: str,
    config: dict[str, Any],
) -> str | None:
    """Validate config against the typed schema; return the resource identity.

    Callback-backed trigger types dispatch through their registered
    ``TriggerProvider.validate_config``; types without a provider (scheduled)
    validate against the typed schema directly. The stored config keeps the
    caller-provided JSON shape; validation is performed on the typed model
    without normalizing persisted fields.
    """
    if not isinstance(config, dict):
        raise TriggerServiceError("config must be an object")
    provider = maybe_get_trigger_provider(trigger_type)
    try:
        if provider is not None:
            typed = provider.validate_config(config)
        else:
            typed = parse_trigger_config(trigger_type, config)
    except TriggerConfigError as exc:
        cause = exc.__cause__
        if isinstance(cause, ValidationError):
            raise _typed_config_error(trigger_type, cause) from exc
        raise TriggerServiceError(str(exc)) from exc
    except ValidationError as exc:
        raise _typed_config_error(trigger_type, exc) from exc

    if trigger_type == TriggerType.SCHEDULED.value:
        _compute_next_run_at(config)
    _validate_persisted_connector_runtime_config(config)
    if trigger_type == TriggerType.GMAIL.value:
        return _resolve_gmail_resource(
            db,
            user_id=user_id,
            oauth_account_id=getattr(typed, "oauth_account_id", None),
        )
    return None


def _trigger_connector_runtime_payload(config: dict[str, Any] | None) -> Any:
    if not isinstance(config, dict):
        return None
    return config.get("connector_runtime_context")


def _validate_persisted_connector_runtime_config(config: dict[str, Any]) -> None:
    try:
        reject_ephemeral_connector_runtime_payload(
            _trigger_connector_runtime_payload(config)
        )
    except ConnectorRuntimeError as exc:
        raise TriggerServiceError(exc.safe_message) from exc


def _run_provider_coro(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async provider call to completion from sync CRUD code.

    CRUD helpers normally run in worker threads without an event loop
    (routes wrap them in asyncio.to_thread), where asyncio.run suffices.
    Callers that invoke CRUD from a thread already running a loop get the
    coroutine executed on a private loop in a helper thread instead; either
    way the call blocks until provisioning finishes, matching the
    previously-synchronous behavior.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _register_trigger_with_provider(db: Session, trigger: AgentTrigger) -> None:
    """Provision provider-side delivery resources for an enabled trigger."""
    if not bool(trigger.enabled):
        return
    provider = maybe_get_trigger_provider(str(trigger.type))
    if provider is None:
        return
    _run_provider_coro(provider.register(db, trigger, trigger.config))
    db.refresh(trigger)


def _unregister_trigger_binding(
    db: Session,
    trigger: AgentTrigger,
    *,
    trigger_type: str,
    config: dict[str, Any],
) -> None:
    """Tear down the delivery binding described by a trigger's previous config.

    The trigger row may already hold a different binding or be deleted, so
    the previous config is passed explicitly; providers resolve the binding
    from it alone and no-op when other triggers still reference it.
    """
    provider = maybe_get_trigger_provider(trigger_type)
    if provider is None:
        return
    _run_provider_coro(provider.unregister(db, trigger, config))


def get_owned_agent(db: Session, *, user_id: int, agent_id: int) -> Agent | None:
    return (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            owned_agent_clause(user_id, get_agent_team_scope(db, user_id)),
        )
        .first()
    )


def get_owned_trigger(
    db: Session,
    *,
    user_id: int,
    agent_id: int,
    trigger_id: int,
) -> AgentTrigger | None:
    # Visibility follows the trigger's agent, not its creator: a teammate/team
    # admin who can manage the (co-owned) agent can also read/update/delete
    # triggers others created on it. Confirm the caller manages the agent first.
    if get_owned_agent(db, user_id=user_id, agent_id=agent_id) is None:
        return None
    return (
        db.query(AgentTrigger)
        .filter(
            AgentTrigger.id == trigger_id,
            AgentTrigger.agent_id == agent_id,
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
    resource_id = _validate_config(
        db,
        user_id=user_id,
        trigger_type=resolved_type,
        config=resolved_config,
    )

    plain_secret: str | None = None
    callback_id: str | None = None
    secret_encrypted: str | None = None
    if resolved_type == TriggerType.WEBHOOK.value:
        callback_id = _new_callback_id()
        plain_secret = secret or _new_webhook_secret()
        secret_encrypted = encrypt_value(plain_secret)

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
        provider=resolved_type,
        callback_id=callback_id,
        resource_id=resource_id,
        secret_encrypted=secret_encrypted,
        next_run_at=next_run_at,
    )
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    _register_trigger_with_provider(db, trigger)
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

    old_type = str(trigger.type)
    old_enabled = bool(trigger.enabled)
    old_config = dict(trigger.config or {})

    plain_secret: str | None = None
    if "name" in updates and updates["name"] is not None:
        setattr(trigger, "name", _normalize_trigger_name(str(updates["name"])))
    if "enabled" in updates and updates["enabled"] is not None:
        setattr(trigger, "enabled", bool(updates["enabled"]))
    if "prompt_template" in updates:
        setattr(trigger, "prompt_template", updates["prompt_template"])
    if "config" in updates and updates["config"] is not None:
        config = dict(updates["config"])
        resource_id = _validate_config(
            db,
            user_id=user_id,
            trigger_type=str(trigger.type),
            config=config,
        )
        setattr(trigger, "config", config)
        setattr(trigger, "resource_id", resource_id)
    if trigger.provider is None:
        setattr(trigger, "provider", str(trigger.type))
    if str(trigger.type) == TriggerType.WEBHOOK.value and trigger.callback_id is None:
        setattr(trigger, "callback_id", _new_callback_id())
    if "secret" in updates and updates["secret"]:
        plain_secret = str(updates["secret"])
        setattr(trigger, "secret_encrypted", encrypt_value(plain_secret))
    elif updates.get("rotate_secret"):
        plain_secret = _new_webhook_secret()
        setattr(trigger, "secret_encrypted", encrypt_value(plain_secret))

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
    # Unregister on any config change, not just a binding change: provider
    # unregister is reference-counted teardown, so it no-ops while the
    # (possibly unchanged) binding is still referenced by an enabled trigger.
    new_config = dict(trigger.config or {})
    if old_enabled and (not bool(trigger.enabled) or old_config != new_config):
        _unregister_trigger_binding(
            db, trigger, trigger_type=old_type, config=old_config
        )
    _register_trigger_with_provider(db, trigger)
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
    trigger_type = str(trigger.type)
    binding_config = dict(trigger.config or {})
    db.delete(trigger)
    db.commit()
    _unregister_trigger_binding(
        db, trigger, trigger_type=trigger_type, config=binding_config
    )


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
    event_type: str | None = None,
    resource_id: str | None = None,
    received_at: datetime | None = None,
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
        payload_snapshot=_payload_snapshot(
            trigger,
            event_payload,
            source_event_id=source_event_id,
            event_type=event_type,
            resource_id=resource_id
            if resource_id is not None
            else (str(trigger.resource_id) if trigger.resource_id else None),
            received_at=received_at,
        ),
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
    if agent is None:
        raise TriggerServiceError("Agent not found")
    missing_secret_error_code = (
        ERROR_SCHEDULED_SECRET_UNAVAILABLE
        if str(trigger.type) == TriggerType.SCHEDULED.value
        else ERROR_RUNTIME_SECRET_UNAVAILABLE
    )
    runtime_plan = prepare_create_connector_runtime(
        db=db,
        agent=agent,
        payload_items=_trigger_connector_runtime_payload(trigger.config),
        allow_ephemeral=False,
        missing_ephemeral_error_code=missing_secret_error_code,
    )
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
        connector_runtime_selected_refs=[
            ref.to_wire() for ref in runtime_plan.selected_refs
        ],
    )
    db.add(task)
    db.flush()
    persist_create_connector_runtime_context(
        db=db, task_id=int(task.id), plan=runtime_plan
    )
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
    event_type: str | None = None,
    resource_id: str | None = None,
    received_at: datetime | None = None,
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
        event_type=event_type,
        resource_id=resource_id,
        received_at=received_at,
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
        raise TriggerRunPreparationError(error_message, run=run) from exc


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

    # Count one billable action for the trigger firing itself (webhook /
    # scheduled). Best-effort; never let metering break a trigger run.
    try:
        from .quota_hooks import record_trigger

        record_trigger(start.task_owner_user_id)
    except Exception:
        logger.debug("Trigger quota record failed", exc_info=True)

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
    event_type: str | None = None,
    resource_id: str | None = None,
    received_at: datetime | None = None,
) -> tuple[TriggerRun, bool]:
    """Prepare a trigger event and start it in the current backend process."""
    run, created = prepare_trigger_run(
        db,
        trigger=trigger,
        event_payload=event_payload,
        source_event_id=source_event_id,
        background_job_id=background_job_id,
        test=test,
        event_type=event_type,
        resource_id=resource_id,
        received_at=received_at,
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
        try:
            run, _created = prepare_trigger_run(
                db,
                trigger=trigger,
                event_payload=payload,
                source_event_id=source_event_id,
                background_job_id=None,
                test=False,
                event_type="scheduled",
            )
        except TriggerRunPreparationError as exc:
            # Scheduled events have no redelivery; the FAILED run is the
            # record. Keep advancing next_run_at so the schedule stays live.
            run = exc.run

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
