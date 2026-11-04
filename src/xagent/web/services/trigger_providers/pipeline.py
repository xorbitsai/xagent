"""Unified trigger callback pipeline.

One code path processes every provider callback: provider lookup, challenge
short-circuit, trigger resolution, verification, event parsing, event-type
filtering, resource authorization, trigger execution, auditing, and
acknowledgement. Providers plug in behavior; the pipeline owns ordering and
the audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from ...models.trigger import AgentTrigger, TriggerAuditOutcome, TriggerRun
from .audit import record_trigger_audit, record_trigger_audit_best_effort
from .base import CallbackRequestContext, TriggerEventParseError
from .registry import UnknownTriggerProviderError, get_trigger_provider
from .schemas import ChallengeResponse, NormalizedEvent

logger = logging.getLogger(__name__)


@dataclass
class CallbackResult:
    """Terminal state of one callback request through the pipeline."""

    status_code: int
    outcome: TriggerAuditOutcome | None
    detail: str | None = None
    challenge: ChallengeResponse | None = None
    runs: list[TriggerRun] = field(default_factory=list)
    duplicates: int = 0
    rejected_events: int = 0
    filtered_events: int = 0


def _allowed_event_types(trigger: AgentTrigger) -> list[str] | None:
    config: dict[str, Any] = trigger.config if isinstance(trigger.config, dict) else {}
    allow_list = config.get("event_types")
    if isinstance(allow_list, list) and allow_list:
        return [str(item) for item in allow_list]
    return None


def _event_allowed_for_trigger(trigger: AgentTrigger, event: NormalizedEvent) -> bool:
    allow_list = _allowed_event_types(trigger)
    return allow_list is None or event.event_type in allow_list


def _resolve_event_triggers(
    db: Session, default_trigger: AgentTrigger, events: list[NormalizedEvent]
) -> dict[int, AgentTrigger]:
    """Resolve every sibling trigger targeted by a callback's events at once.

    One IN() query per callback regardless of event count. Targets that are
    missing, owned by another user, or of a different type/provider are
    simply absent from the map, which rejects them exactly like the previous
    per-event lookup did.
    """
    resolved: dict[int, AgentTrigger] = {int(default_trigger.id): default_trigger}
    target_ids = {
        int(event.target_trigger_id)
        for event in events
        if event.target_trigger_id is not None
    } - set(resolved)
    if not target_ids:
        return resolved
    rows = (
        db.query(AgentTrigger)
        .filter(
            AgentTrigger.id.in_(target_ids),
            AgentTrigger.user_id == int(default_trigger.user_id),
            AgentTrigger.type == default_trigger.type,
            AgentTrigger.provider == default_trigger.provider,
        )
        .all()
    )
    for row in rows:
        resolved[int(row.id)] = row
    return resolved


def _attested_resource_matches(
    trigger: AgentTrigger, attested_resource_id: str | None
) -> bool:
    """Enforce persisted trigger resource identity against attested identity.

    This check belongs to the pipeline, not to providers: a trigger bound to
    a resource may only fire for events whose resource identity was proven by
    the provider's trust model. Payload-claimed identity never reaches here.
    Triggers without a bound resource (e.g. webhooks) skip the check.
    """
    if trigger.resource_id is None:
        return True
    if attested_resource_id is None:
        return False
    return (
        attested_resource_id.strip().lower() == str(trigger.resource_id).strip().lower()
    )


async def process_trigger_callback(
    db: Session,
    *,
    context: CallbackRequestContext,
    raw_body: bytes,
) -> CallbackResult:
    """Run one inbound callback through the unified provider pipeline."""
    provider_name = context.provider
    callback_id = context.callback_id

    try:
        provider = get_trigger_provider(provider_name)
    except UnknownTriggerProviderError:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.UNKNOWN_PROVIDER,
            provider=provider_name,
            callback_id=callback_id,
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=404,
            outcome=TriggerAuditOutcome.UNKNOWN_PROVIDER,
            detail="Unknown trigger provider",
        )

    ack = provider.ack_policy

    challenge = provider.handle_challenge(context, raw_body)
    if challenge is not None:
        return CallbackResult(
            status_code=challenge.status_code, outcome=None, challenge=challenge
        )

    trigger = provider.locate_trigger(db, callback_id)
    if trigger is None:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.UNKNOWN_CALLBACK,
            provider=provider_name,
            callback_id=callback_id,
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.not_found_status,
            outcome=TriggerAuditOutcome.UNKNOWN_CALLBACK,
            detail="Unknown callback",
        )

    try:
        verification = await provider.verify(
            context, db=db, trigger=trigger, raw_body=raw_body
        )
    except Exception as exc:
        # Verification errored without proving the request invalid (e.g. a
        # transient failure fetching the provider's signing keys). Answer with
        # the failure status so redelivery-based sources retry; rejecting here
        # would let providers that ack rejections drop the event permanently.
        logger.exception("Provider %s failed to verify callback", provider_name)
        record_trigger_audit_best_effort(
            db,
            outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            detail={"stage": "verify", "error": f"{type(exc).__name__}: {exc}"},
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.failure_status,
            outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
            detail="Callback verification errored",
        )
    if not verification.verified:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.REJECTED_SIGNATURE,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            detail={"reason": verification.reason},
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.rejected_status,
            outcome=TriggerAuditOutcome.REJECTED_SIGNATURE,
            detail=verification.reason or "Verification failed",
        )

    if not trigger.enabled:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.REJECTED_DISABLED,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.disabled_status,
            outcome=TriggerAuditOutcome.REJECTED_DISABLED,
            detail="Trigger is disabled",
        )

    try:
        events = await provider.parse_events(
            context, db=db, trigger=trigger, raw_body=raw_body
        )
    except TriggerEventParseError as exc:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            detail={"stage": "parse", "error": str(exc)},
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.parse_failure_status,
            outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
            detail=str(exc) or "Malformed callback payload",
        )
    except Exception as exc:
        # Transient ingestion failures (e.g. upstream API errors while
        # expanding events) get a controlled failure status so providers
        # with redelivery semantics can retry, and an audit trail either way.
        logger.exception("Provider %s failed to ingest callback events", provider_name)
        record_trigger_audit_best_effort(
            db,
            outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            detail={"stage": "ingest", "error": f"{type(exc).__name__}: {exc}"},
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.failure_status,
            outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
            detail="Event ingestion failed",
        )

    runs: list[TriggerRun] = []
    duplicates = 0
    rejected_events = 0
    rejected_resource_events = 0
    rejected_disabled_events = 0
    filtered_count = 0
    failures: list[str] = []
    event_triggers = _resolve_event_triggers(db, trigger, events)
    for event in events:
        target_trigger_id = (
            int(event.target_trigger_id)
            if event.target_trigger_id is not None
            else int(trigger.id)
        )
        event_trigger = event_triggers.get(target_trigger_id)
        if event_trigger is None:
            rejected_events += 1
            rejected_resource_events += 1
            record_trigger_audit(
                db,
                outcome=TriggerAuditOutcome.REJECTED_RESOURCE,
                provider=provider_name,
                callback_id=callback_id,
                trigger_id=int(trigger.id),
                detail={
                    "reason": "unknown_event_trigger",
                    "target_trigger_id": event.target_trigger_id,
                    "attested_resource_id": verification.attested_resource_id,
                    "event_type": event.event_type,
                },
                remote_ip=context.remote_ip,
            )
            continue

        if not event_trigger.enabled:
            rejected_events += 1
            rejected_disabled_events += 1
            record_trigger_audit(
                db,
                outcome=TriggerAuditOutcome.REJECTED_DISABLED,
                provider=provider_name,
                callback_id=callback_id,
                trigger_id=int(event_trigger.id),
                detail={"event_type": event.event_type},
                remote_ip=context.remote_ip,
            )
            continue

        if not _event_allowed_for_trigger(event_trigger, event):
            filtered_count += 1
            continue

        resource_matches = _attested_resource_matches(
            event_trigger, verification.attested_resource_id
        )
        if not resource_matches or not provider.authorize_resource(
            event_trigger, verification.attested_resource_id, event
        ):
            rejected_events += 1
            rejected_resource_events += 1
            record_trigger_audit(
                db,
                outcome=TriggerAuditOutcome.REJECTED_RESOURCE,
                provider=provider_name,
                callback_id=callback_id,
                trigger_id=int(event_trigger.id),
                detail={
                    "reason": (
                        "no attested identity"
                        if verification.attested_resource_id is None
                        else "mismatch"
                    ),
                    "claimed_resource_id": event.resource_id,
                    "attested_resource_id": verification.attested_resource_id,
                    "trigger_resource_id": event_trigger.resource_id,
                    "event_type": event.event_type,
                },
                remote_ip=context.remote_ip,
            )
            continue

        try:
            run, created = await _fire_event(db, trigger=event_trigger, event=event)
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            logger.exception(
                "Trigger %s failed to execute callback event", event_trigger.id
            )
            record_trigger_audit_best_effort(
                db,
                outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
                provider=provider_name,
                callback_id=callback_id,
                trigger_id=int(event_trigger.id),
                detail={"stage": "fire", "error": f"{type(exc).__name__}: {exc}"},
                remote_ip=context.remote_ip,
            )
            continue

        if created:
            runs.append(run)
        else:
            duplicates += 1

    if not failures:
        try:
            await _finalize_provider_callback(
                provider,
                db=db,
                context=context,
                trigger=trigger,
                events=events,
                raw_body=raw_body,
            )
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            logger.exception("Provider %s failed to finalize callback", provider_name)
            record_trigger_audit_best_effort(
                db,
                outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
                provider=provider_name,
                callback_id=callback_id,
                trigger_id=int(trigger.id),
                detail={
                    "stage": "finalize",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                remote_ip=context.remote_ip,
            )

    if failures:
        # Any fire/finalize failure means the callback was not fully
        # processed: report the provider failure status so redelivery-based
        # providers retry the whole delivery. Runs already created are
        # protected by their idempotency keys, and finalize was skipped so
        # cursors did not advance past the failed event.
        return CallbackResult(
            status_code=ack.failure_status,
            outcome=TriggerAuditOutcome.EXECUTION_FAILURE,
            detail="; ".join(failures),
            runs=runs,
            duplicates=duplicates,
            rejected_events=rejected_events,
            filtered_events=filtered_count,
        )

    if runs or duplicates or not rejected_events:
        record_trigger_audit(
            db,
            outcome=TriggerAuditOutcome.ACCEPTED,
            provider=provider_name,
            callback_id=callback_id,
            trigger_id=int(trigger.id),
            detail=_accepted_detail(
                runs=runs,
                duplicates=duplicates,
                rejected_events=rejected_events,
                filtered_events=filtered_count,
            ),
            remote_ip=context.remote_ip,
        )
        return CallbackResult(
            status_code=ack.accepted_status,
            outcome=TriggerAuditOutcome.ACCEPTED,
            runs=runs,
            duplicates=duplicates,
            rejected_events=rejected_events,
            filtered_events=filtered_count,
        )

    if rejected_resource_events == 0 and rejected_disabled_events > 0:
        # Every rejection was a disabled target trigger (sibling triggers
        # selected via target_trigger_id); mirror the per-event audit outcome
        # instead of mislabeling the callback as a resource rejection.
        return CallbackResult(
            status_code=ack.disabled_status,
            outcome=TriggerAuditOutcome.REJECTED_DISABLED,
            detail="Trigger is disabled",
            rejected_events=rejected_events,
            filtered_events=filtered_count,
        )

    return CallbackResult(
        status_code=ack.rejected_resource_status,
        outcome=TriggerAuditOutcome.REJECTED_RESOURCE,
        detail="Event resource does not match this trigger",
        rejected_events=rejected_events,
        filtered_events=filtered_count,
    )


def _accepted_detail(
    *,
    runs: list[TriggerRun],
    duplicates: int,
    rejected_events: int,
    filtered_events: int,
) -> dict[str, Any]:
    return {
        "run_ids": [int(run.id) for run in runs],
        "duplicates": duplicates,
        "rejected_events": rejected_events,
        "filtered_events": filtered_events,
    }


async def _fire_event(
    db: Session,
    *,
    trigger: AgentTrigger,
    event: NormalizedEvent,
) -> tuple[TriggerRun, bool]:
    from ..triggers import fire_trigger

    return await fire_trigger(
        db,
        trigger=trigger,
        event_payload=event.payload,
        source_event_id=event.source_event_id,
        event_type=event.event_type,
        resource_id=event.resource_id,
        received_at=event.received_at,
    )


async def _finalize_provider_callback(
    provider: Any,
    *,
    db: Session,
    context: CallbackRequestContext,
    trigger: AgentTrigger,
    events: list[NormalizedEvent],
    raw_body: bytes,
) -> None:
    await provider.finalize_callback(
        db=db,
        context=context,
        trigger=trigger,
        events=events,
        raw_body=raw_body,
    )
