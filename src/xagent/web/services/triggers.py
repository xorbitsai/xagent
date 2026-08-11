from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import logging
import secrets
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.connector_runtime import (
    ERROR_RUNTIME_SECRET_UNAVAILABLE,
    ERROR_SCHEDULED_SECRET_UNAVAILABLE,
    ConnectorRuntimeError,
)
from ...core.utils.encryption import decrypt_value, encrypt_value
from ..models.agent import Agent, AgentOrigin, is_workforce_generated_manager_agent
from ..models.background_job import BackgroundJob, BackgroundJobType
from ..models.task import Task, TaskStatus
from ..models.trigger import (
    AgentTrigger,
    TriggerProvisioningStatus,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from ..models.user import User
from ..models.user_oauth import UserOAuth
from ..models.workforce import Workforce
from .agent_team_scope import get_agent_team_scope, owned_agent_clause
from .background_jobs import create_background_job, enqueue_background_job
from .connector_runtime import (
    bind_create_connector_runtime_plan,
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
from .trigger_providers.schemas import (
    normalize_day_of_month,
    normalize_schedule_timezone,
    normalize_time_of_day,
    normalize_weekdays,
    parse_trigger_config,
)

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


def _wrap_schedule_error(normalize_fn: Callable[[Any], Any], value: Any) -> Any:
    """Run a normalize_* schedule field function, wrapping its ValueError as
    the service-layer TriggerServiceError."""
    try:
        return normalize_fn(value)
    except ValueError as exc:
        raise TriggerServiceError(str(exc)) from exc


def _schedule_tzinfo(config: dict[str, Any]) -> tzinfo:
    """The timezone that time_of_day/weekdays/day_of_month are expressed in.

    Configs written before timezone support (or by API clients that omit it)
    keep the historical UTC interpretation.
    """
    name = config.get("timezone")
    if not name:
        return timezone.utc
    try:
        return normalize_schedule_timezone(name)
    except ValueError as exc:
        raise TriggerServiceError(str(exc)) from exc


def _localize(naive_local: datetime, tz: tzinfo) -> datetime:
    """Attach `tz` to a naive local datetime, normalizing a nonexistent local
    time (a DST spring-forward gap — e.g. 2:30 AM on the day a zone jumps
    from 2:00 to 3:00) by shifting forward past the gap instead of silently
    resolving to whichever UTC instant `fold=0` happens to pick.

    A fall-back-ambiguous local time (repeated once when a zone jumps back,
    e.g. 1:30 AM occurring twice) is NOT specially handled: `roundtrip ==
    naive_local` holds trivially for both occurrences, so the correction
    branch never fires and Python's default `fold=0` — the first, pre-DST
    occurrence — wins. That's a real, if arbitrary, choice, not just an
    unhandled case; see the pinning test for the exact instant it picks.
    """
    aware = naive_local.replace(tzinfo=tz)
    roundtrip = aware.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None)
    if roundtrip != naive_local:
        # `naive_local` doesn't exist in `tz`. `roundtrip` (what `fold=0`
        # actually resolved to) IS the nearest realizable instant — the
        # shift-by-delta arithmetic this replaced (`naive_local + (roundtrip
        # - naive_local)`) reduces to exactly `roundtrip`.
        aware = roundtrip.replace(tzinfo=tz)
    return aware


def _next_weekly_occurrence(
    base: datetime, weekdays: set[int], time_of_day: time, tz: tzinfo
) -> datetime:
    """Earliest datetime strictly after `base` combining a weekday in
    `weekdays` (0=Mon..6=Sun) with `time_of_day`, both interpreted in `tz`
    (the user's schedule timezone). Returned as UTC."""
    local_base = base.astimezone(tz)
    for offset in range(8):
        candidate_date = local_base.date() + timedelta(days=offset)
        if candidate_date.weekday() not in weekdays:
            continue
        candidate = _localize(datetime.combine(candidate_date, time_of_day), tz)
        if candidate > base:
            return candidate.astimezone(timezone.utc)
    # Unreachable: offset 7 always recurs on the same weekday as offset 0,
    # 7 days later — guaranteed strictly after base — so offset 0..7 always
    # yields a match.
    raise TriggerServiceError("Unable to compute next weekly occurrence")


def _next_monthly_occurrence(
    base: datetime, day_of_month: int, time_of_day: time, tz: tzinfo
) -> datetime:
    """Earliest datetime strictly after `base` on `day_of_month` (clamped to
    the last day of short months) at `time_of_day`, both interpreted in `tz`.
    Returned as UTC."""
    local_base = base.astimezone(tz)
    year, month = local_base.year, local_base.month
    for _ in range(24):  # 24 months is far more than enough headroom
        last_day = calendar.monthrange(year, month)[1]
        day = min(day_of_month, last_day)
        candidate = _localize(datetime.combine(date(year, month, day), time_of_day), tz)
        if candidate > base:
            return candidate.astimezone(timezone.utc)
        month += 1
        if month > 12:
            month = 1
            year += 1
    raise TriggerServiceError("Unable to compute next monthly occurrence")


def _compute_next_run_at(
    config: dict[str, Any],
    *,
    from_time: datetime | None = None,
    previous_due_at: datetime | None = None,
    include_explicit: bool = True,
    allow_past_explicit: bool = True,
) -> datetime | None:
    """Compute the next scheduled fire time for the supported MVP config."""
    base = _coerce_utc(previous_due_at) or _coerce_utc(from_time) or _now()
    now = _coerce_utc(from_time) or _now()

    recurrence = config.get("recurrence")
    if recurrence in ("daily", "weekly", "monthly"):
        time_of_day = _wrap_schedule_error(
            normalize_time_of_day, config.get("time_of_day")
        )
        tz = _schedule_tzinfo(config)
        # Only the first computation (no previous fire yet) may be pushed out
        # by an explicit start_at; subsequent recomputations always advance
        # from the last fire time.
        if previous_due_at is None:
            explicit_start = config.get("start_at")
            if isinstance(explicit_start, str) and explicit_start.strip():
                try:
                    start_date = datetime.fromisoformat(explicit_start).date()
                except ValueError as exc:
                    raise TriggerServiceError("Invalid start_at") from exc
                # Only the calendar DATE is authoritative — the clock time of
                # the first occurrence is always time_of_day in `tz`, never
                # whatever time component `start_at` happens to carry. A
                # client (the schedule editor included) has no reliable way
                # to express "this wall-clock time, in this zone" as a bare
                # ISO instant without knowing which zone the RECEIVING
                # process will interpret it in — sending just a date sidesteps
                # that entirely.
                start = _localize(
                    datetime.combine(start_date, time_of_day), tz
                ).astimezone(timezone.utc)
                if start > base:
                    # Subtract a second so a start date that itself matches
                    # the recurrence still qualifies as its own first fire.
                    base = start - timedelta(seconds=1)
        # Never schedule in the past: a stale previous_due_at (downtime,
        # long-disabled trigger) skips the missed occurrences instead of
        # firing a catch-up burst, and this same clamp makes it always safe
        # to recompute from an unchanged config (see _apply_trigger_updates)
        # without ever re-arming a past instant.
        base = max(base, now)
        if recurrence == "weekly":
            weekdays = _wrap_schedule_error(normalize_weekdays, config.get("weekdays"))
            return _next_weekly_occurrence(base, weekdays, time_of_day, tz)
        if recurrence == "monthly":
            day_of_month = _wrap_schedule_error(
                normalize_day_of_month, config.get("day_of_month")
            )
            return _next_monthly_occurrence(base, day_of_month, time_of_day, tz)
        # "daily" is "weekly on every day" — _next_weekly_occurrence with the
        # full weekday set produces byte-identical results (including every
        # DST edge case) with no separate implementation to keep in sync.
        return _next_weekly_occurrence(base, set(range(7)), time_of_day, tz)

    if include_explicit:
        explicit_next = config.get("next_run_at")
        if isinstance(explicit_next, str) and explicit_next.strip():
            try:
                explicit = _coerce_utc(datetime.fromisoformat(explicit_next))
            except ValueError as exc:
                raise TriggerServiceError("Invalid next_run_at") from exc
            if explicit is not None:
                if allow_past_explicit or explicit > now:
                    # Honored verbatim, whether in the future (a genuine
                    # start anchor) or the past (the trigger is already due
                    # and the next scan should catch it up once — the same
                    # semantics as enabling a cron job whose scheduled time
                    # already passed). Only true at creation, or when
                    # re-enabling a trigger that had no prior armed
                    # schedule — see _apply_trigger_updates.
                    return explicit
                # A recompute triggered by an intentional schedule EDIT on an
                # already-armed trigger: the resubmitted config still carries
                # the ORIGINAL creation-time next_run_at (it's never advanced
                # by scans, only the DB column is). For the interval
                # mechanism, arm the next interval-aligned instant from that
                # stale anchor instead of clamping to `now` verbatim —
                # clamping to exactly `now` is treated as already-due by the
                # next scan tick (`next_run_at <= scan_time`), firing an
                # unwanted extra execution on every no-op schedule resend. A
                # genuine one-shot (no interval_seconds to align to) has
                # nothing to fall back on, so it keeps the original "catch up
                # once" clamp.
                if config.get("interval_seconds") is None:
                    return now
                base = explicit

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
    if candidate <= now:
        elapsed_seconds = (now - base).total_seconds()
        steps = int(elapsed_seconds // interval_seconds) + 1
        candidate = base + timedelta(seconds=steps * interval_seconds)
    return candidate


_SCHEDULE_RELEVANT_CONFIG_FIELDS = (
    "recurrence",
    "time_of_day",
    "weekdays",
    "day_of_month",
    "timezone",
    "interval_seconds",
    "next_run_at",
    "start_at",
)


def _schedule_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    """The subset of a scheduled trigger's config that actually determines
    its fire times, normalized so two configs that mean the same schedule
    compare equal even if their raw JSON doesn't byte-match. Used to decide
    whether an update genuinely changed the schedule (and must recompute
    next_run_at) versus resending an unrelated field alongside an unchanged
    schedule (and must not) — comparing raw, unnormalized values is too
    literal: a stored trigger genuinely irrelevant to scheduling (name,
    prompt_template, event_types, ...) never differs here, so only a real
    schedule edit trips the recompute path.

    Several normalizations matter in practice: `_validate_config` persists the
    caller-provided config verbatim (never rewrites stored JSON — see its
    docstring), so an un-padded `time_of_day` ("9:5") is never canonicalized
    at rest; `weekdays` is a set of days with no meaningful order, so
    `[0, 2]` and `[2, 0]` must compare equal; `start_at`'s calendar DATE is
    the only part `_compute_next_run_at` reads, so a bare date and a
    midnight-instant ISO string for that same date must compare equal; and
    `interval_seconds` may arrive as an int or a numeral string. All would
    otherwise register as a schedule "change" for reasons that have nothing
    to do with the schedule actually differing.
    """
    normalized: list[Any] = []
    for field in _SCHEDULE_RELEVANT_CONFIG_FIELDS:
        value = config.get(field)
        if field == "time_of_day" and value is not None:
            if isinstance(value, str) and not value.strip():
                # A present-but-blank time_of_day (accepted for hourly/custom
                # since the F6 fix — see ScheduledTriggerConfig's model
                # validator) must compare equal to an ABSENT time_of_day key,
                # same as start_at's blank/missing handling just below —
                # otherwise a direct API client resending "" where the stored
                # config has no time_of_day at all trips a spurious signature
                # mismatch and an unwanted recompute (PR #1051 review, N-follow-up).
                value = None
            else:
                try:
                    value = normalize_time_of_day(value).strftime("%H:%M")
                except ValueError:
                    pass  # Malformed values still compare (in)equal on the raw input.
        elif field == "weekdays" and isinstance(value, list):
            try:
                value = sorted(normalize_weekdays(value))
            except ValueError:
                pass
        elif field == "start_at" and isinstance(value, str) and value.strip():
            # _compute_next_run_at only ever looks at the calendar DATE of
            # start_at (see its docstring) — "2026-01-01" and
            # "2026-01-01T00:00:00" must compare equal here too, or resaving
            # an un-normalized stored value through the new UI would
            # misfire a recompute for a schedule that hasn't actually changed.
            try:
                value = datetime.fromisoformat(value).date().isoformat()
            except ValueError:
                pass
        elif field == "interval_seconds" and value is not None:
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        normalized.append(value)
    return tuple(normalized)


def _is_benign_start_at_backfill(
    old_config: dict[str, Any], new_config: dict[str, Any]
) -> bool:
    """True when the only schedule-relevant difference between an existing
    trigger's stored config and a freshly resubmitted one is that `start_at`
    went from wholly absent to today's date (in the schedule's own
    timezone) — the shape the editor's F5 default produces for a calendar
    trigger with no stored anchor, even on a Save that touches nothing else
    schedule-related.

    Without this, _apply_trigger_updates's "did the schedule actually
    change" check (comparing _schedule_signature) sees a genuine diff
    (None -> a date) and recomputes next_run_at from today, which can move
    it EARLIER than the schedule's already-computed next occurrence, purely
    as a side effect of what looks like a no-op Save (PR #1051 review, N8).
    """
    old_signature = _schedule_signature(old_config)
    new_signature = _schedule_signature(new_config)
    if old_signature == new_signature:
        return False  # not this case; the caller already treats it as unchanged
    start_at_index = _SCHEDULE_RELEVANT_CONFIG_FIELDS.index("start_at")
    old_start_at = old_signature[start_at_index]
    # A stored start_at of "" (present key, blank value) must count as
    # "wholly absent" here too, same as _compute_next_run_at's and
    # _schedule_signature's own .strip() convention for this field — a bare
    # `is not None` check misses a literal empty string, wrongly concluding a
    # real anchor already existed and skipping the benign-backfill detection
    # (PR #1051 review, N8 follow-up).
    if old_start_at is not None and not (
        isinstance(old_start_at, str) and not old_start_at.strip()
    ):
        return False  # a real anchor already existed; not the F5 backfill shape
    if (
        old_signature[:start_at_index] != new_signature[:start_at_index]
        or old_signature[start_at_index + 1 :] != new_signature[start_at_index + 1 :]
    ):
        return False  # something else also changed; a genuine schedule edit
    new_start_at = new_signature[start_at_index]
    if not isinstance(new_start_at, str):
        return False
    try:
        tz = _schedule_tzinfo(new_config)
    except TriggerServiceError:
        return False
    return new_start_at == _now().astimezone(tz).date().isoformat()


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


def _is_legacy_non_calendar_timezone_config(config: dict[str, Any]) -> bool:
    """True for a scheduled config combining a non-calendar recurrence
    (hourly/custom/a bare next_run_at one-shot) with a `timezone` field —
    allowed before this PR added schema validation, now rejected by
    ScheduledTriggerConfig's model validator."""
    if not isinstance(config, dict):
        return False
    recurrence = config.get("recurrence")
    is_non_calendar = recurrence not in ("daily", "weekly", "monthly")
    return is_non_calendar and config.get("timezone") is not None


def _tolerate_legacy_timezone_field(
    config: dict[str, Any], existing_config: dict[str, Any] | None
) -> dict[str, Any]:
    """Drop a stray `timezone` from a non-calendar scheduled config before
    strict typed validation — but only when this trigger's OWN previously
    stored config already carried that exact legacy shape AND the same
    `timezone` VALUE is being resent unchanged (PR #1051 review, N7). That
    signal distinguishes a read-then-write round-trip of already-stored
    legacy data (allowed before this PR's schema validation) from a client
    freshly authoring a new, contradictory config: a brand-new trigger has no
    `existing_config` and is always validated strictly, and an update whose
    OLD config didn't already mix these fields is too — only a trigger that
    already had both fields set gets the pass, and only for as long as it
    keeps resending that same legacy shape.

    Checking only the SHAPE (non-calendar recurrence + a timezone present)
    on both sides isn't enough: it would let a trigger with genuine legacy
    config `{recurrence: "hourly", timezone: "Asia/Shanghai"}` be resaved
    with `{recurrence: "custom", timezone: "Pacific/Auckland"}` and still
    slip through, since both configs independently satisfy the loose shape
    even though the actual timezone value changed. Requiring the OLD and NEW
    `timezone` values to match exactly closes that: a client can keep
    resending its own already-stored legacy pair indefinitely, but can't use
    the leniency to author a new, different one.

    The stripped field is used for validation only; the caller's original
    `config` (still carrying `timezone`) is what actually gets persisted,
    consistent with this module never rewriting stored JSON.
    """
    if not _is_legacy_non_calendar_timezone_config(config):
        return config
    if not _is_legacy_non_calendar_timezone_config(existing_config or {}):
        return config
    if (existing_config or {}).get("timezone") != config.get("timezone"):
        return config
    tolerant = dict(config)
    tolerant.pop("timezone", None)
    return tolerant


def _validate_config(
    db: Session,
    *,
    user_id: int,
    trigger_type: str,
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
) -> str | None:
    """Validate config against the typed schema; return the resource identity.

    Callback-backed trigger types dispatch through their registered
    ``TriggerProvider.validate_config``; types without a provider (scheduled)
    validate against the typed schema directly. The stored config keeps the
    caller-provided JSON shape; validation is performed on the typed model
    without normalizing persisted fields.

    ``existing_config`` (an update's previous stored config, absent on
    create) enables narrow backward-compat leniency for legacy scheduled
    configs — see ``_tolerate_legacy_timezone_field``.
    """
    if not isinstance(config, dict):
        raise TriggerServiceError("config must be an object")
    validated_config = config
    if trigger_type == TriggerType.SCHEDULED.value:
        validated_config = _tolerate_legacy_timezone_field(config, existing_config)
    provider = maybe_get_trigger_provider(trigger_type)
    try:
        if provider is not None:
            typed = provider.validate_config(validated_config)
        else:
            typed = parse_trigger_config(trigger_type, validated_config)
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


def _reject_workforce_connector_runtime(config: dict[str, Any] | None) -> None:
    """Reject connector-runtime payloads on workforce-owned triggers.

    Workforce runs are created through ``create_workforce_run_record``, whose
    connector-runtime handling is selection-only (see
    ``prepare_connector_runtime_selection_snapshot``): the non-/v1 creation
    path does not accept per-invocation runtime payloads. The shared trigger
    ``config`` schema would otherwise let such a payload validate and then be
    silently dropped at fire time, so reject it up front instead.
    """
    if _trigger_connector_runtime_payload(config):
        raise TriggerServiceError(
            "Workforce triggers do not support connector_runtime_context; "
            "the workforce run resolves connectors from its manager agent."
        )


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


def unregister_deleted_trigger_bindings(
    teardowns: list[tuple[AgentTrigger, str, dict[str, Any]]],
) -> None:
    """Best-effort provider teardown for trigger rows deleted outside the
    trigger CRUD path (workforce hard delete cascades them away).

    Runs on its own session(s) rather than accepting one from the caller:
    this is invoked via ``asyncio.to_thread`` from an async route, and the
    caller's request-scoped ``Session`` is not safe to hand to a background
    thread -- same reasoning as ``pause_workforce_tasks_after_archive``,
    which this mirrors by opening a fresh session per item so one binding's
    failure can't roll back another's still-pending work.

    Mirrors ``_delete_trigger``'s tail otherwise: the rows are already gone
    and committed, so a teardown failure is logged rather than surfaced -- it
    must not turn an already-successful delete into an error. Each
    ``(trigger, trigger_type, config)`` tuple must have been captured BEFORE
    the delete committed (the detached rows' attributes are expired).
    """
    if not teardowns:
        return

    from ..models.database import get_session_local

    SessionLocal = get_session_local()
    for trigger, trigger_type, config in teardowns:
        try:
            with SessionLocal() as db:
                _unregister_trigger_binding(
                    db, trigger, trigger_type=trigger_type, config=config
                )
        except Exception:
            logger.exception(
                "Failed to unregister binding for cascade-deleted trigger; "
                "the trigger row is gone but its provider-side binding may "
                "be leaked (type=%s)",
                trigger_type,
            )


def get_owned_agent(db: Session, *, user_id: int, agent_id: int) -> Agent | None:
    # Workforce-generated manager agents are private implementation details;
    # they must not be addressable through trigger management, matching the
    # exclusion every other external channel (share/widget/api-keys) applies.
    return (
        db.query(Agent)
        .filter(
            Agent.id == agent_id,
            Agent.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
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


def get_workforce_trigger(
    db: Session,
    *,
    workforce_id: int,
    trigger_id: int,
) -> AgentTrigger | None:
    """Resolve a trigger owned by a workforce.

    Workforce access (view/edit, including admin override) is the route
    layer's responsibility via ``ensure_workforce_access``; this helper only
    scopes the lookup to the already-authorized workforce.
    """
    return (
        db.query(AgentTrigger)
        .filter(
            AgentTrigger.id == trigger_id,
            AgentTrigger.workforce_id == workforce_id,
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
    return _create_trigger(
        db,
        user_id=user_id,
        agent_id=agent_id,
        trigger_type=trigger_type,
        name=name,
        enabled=enabled,
        config=config,
        prompt_template=prompt_template,
        secret=secret,
    )


def create_workforce_trigger(
    db: Session,
    *,
    user_id: int,
    workforce_id: int,
    trigger_type: str,
    name: str | None = None,
    enabled: bool = True,
    config: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    secret: str | None = None,
) -> tuple[AgentTrigger, str | None]:
    """Create a workforce-owned trigger (workforce access checked by caller).

    Note on execution identity: unlike agent triggers (which only their agent's
    owner can create), a workforce trigger can be created by anyone with
    workforce ``edit`` access, and ``user_id`` — the creator — is persisted as
    the identity the eventual workforce run executes as. This is not an
    escalation: the firing path re-checks ``ensure_workforce_access(action=
    "run")`` in ``create_workforce_run_record``, so a creator who later loses
    access is rejected at fire time.
    """
    return _create_trigger(
        db,
        user_id=user_id,
        workforce_id=workforce_id,
        trigger_type=trigger_type,
        name=name,
        enabled=enabled,
        config=config,
        prompt_template=prompt_template,
        secret=secret,
    )


def _create_trigger(
    db: Session,
    *,
    user_id: int,
    agent_id: int | None = None,
    workforce_id: int | None = None,
    trigger_type: str,
    name: str | None = None,
    enabled: bool = True,
    config: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    secret: str | None = None,
) -> tuple[AgentTrigger, str | None]:
    if (agent_id is None) == (workforce_id is None):
        raise TriggerServiceError(
            "Trigger must be owned by exactly one of agent or workforce"
        )

    resolved_type = _normalize_trigger_type(trigger_type)
    resolved_config = dict(config or {})
    if workforce_id is not None:
        _reject_workforce_connector_runtime(resolved_config)
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
        workforce_id=workforce_id,
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
    return _apply_trigger_updates(db, trigger, user_id=user_id, updates=updates)


def update_workforce_trigger(
    db: Session,
    *,
    user_id: int,
    workforce_id: int,
    trigger_id: int,
    updates: dict[str, Any],
) -> tuple[AgentTrigger, str | None]:
    """Update a workforce-owned trigger (workforce access checked by caller)."""
    trigger = get_workforce_trigger(
        db, workforce_id=workforce_id, trigger_id=trigger_id
    )
    if trigger is None:
        raise TriggerNotFoundError("Trigger not found")
    return _apply_trigger_updates(db, trigger, user_id=user_id, updates=updates)


def _apply_trigger_updates(
    db: Session,
    trigger: AgentTrigger,
    *,
    user_id: int,
    updates: dict[str, Any],
) -> tuple[AgentTrigger, str | None]:
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
        if trigger.workforce_id is not None:
            _reject_workforce_connector_runtime(config)
        resource_id = _validate_config(
            db,
            user_id=user_id,
            trigger_type=str(trigger.type),
            config=config,
            existing_config=old_config,
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
        # Keyed on whether the SCHEDULE ITSELF actually changed (comparing
        # the schedule-relevant subset of the config, not whether the
        # request payload happened to include a `config` key, and not the
        # whole config dict) rather than on `"config" in updates`: the
        # editor's full-form Save always resends the complete config,
        # including hourly/custom's flat next_run_at/interval_seconds anchor
        # — which was computed once at creation and never advances (only the
        # DB column does, via scans) — alongside fields genuinely irrelevant
        # to scheduling, like `name` or `prompt_template` (not part of
        # `config` at all — see below) or a trigger's `event_types`. Gating
        # on payload shape or whole-config equality both made this branch
        # fire on saves that touch nothing schedule-related. A resubmitted
        # `timezone` that DOES differ is a genuine schedule edit — see
        # _schedule_signature — and correctly forces a recompute.
        new_config = dict(trigger.config or {})
        if not trigger.enabled:
            setattr(trigger, "next_run_at", None)
        elif not old_enabled:
            # Re-enabled. NOT treated like a fresh creation: the stored
            # config's `next_run_at`/`start_at` is whatever was last
            # intentionally set, possibly from long before this trigger was
            # ever disabled — honoring it verbatim here would either fire
            # immediately (if never consumed) or, worse, silently no-op (if
            # the disable happened after it already fired once, since the
            # scan's idempotency key is derived from the due instant and
            # would collide with the already-consumed run). Clamp the same
            # way an intentional schedule edit does.
            #
            # A bare `{"enabled": true}` PATCH (no "config" key) reaches here
            # WITHOUT ever going through `_validate_config` above — including
            # its `interval_seconds` upper-bound check — so a stale stored
            # config predating that cap (or otherwise malformed) can still
            # make `_compute_next_run_at` raise here. Unlike the scheduled
            # scan loop (an unattended background process, where the right
            # move is to silently disable the trigger and keep going), this
            # is a synchronous request a user is waiting on, so turn the
            # failure into a clean, actionable `TriggerServiceError` — the
            # API layer already maps that to a 4xx — instead of letting a
            # bare ValueError/OverflowError escape as a generic 500 (PR
            # #1051 review, third round).
            try:
                recomputed_next_run_at = _compute_next_run_at(
                    new_config, allow_past_explicit=False
                )
            except (ValueError, ArithmeticError) as exc:
                raise TriggerServiceError(
                    "Cannot re-enable this trigger: its stored schedule "
                    f"config is invalid ({exc}). Edit the schedule and try "
                    "again."
                ) from exc
            setattr(trigger, "next_run_at", recomputed_next_run_at)
        elif _schedule_signature(new_config) != _schedule_signature(
            old_config
        ) and not (
            # The benign-backfill heuristic is specifically tuned to the
            # agent-triggers dialog's F5 auto-fill default (see
            # _is_benign_start_at_backfill's docstring) — it has no way to
            # distinguish that from a workforce-trigger owner deliberately
            # setting start_at to today via a real edit, since this function
            # is shared by both update_agent_trigger and
            # update_workforce_trigger. `trigger.workforce_id is None` is the
            # same existing signal _apply_trigger_updates already uses just
            # above (_reject_workforce_connector_runtime) to distinguish an
            # agent-owned trigger from a workforce-owned one — reused here
            # rather than threading a new parameter through both callers
            # (PR #1051 review, N8 follow-up).
            trigger.workforce_id is None
            and _is_benign_start_at_backfill(old_config, new_config)
        ):
            # The schedule was intentionally changed. Recompute, but don't
            # let a still-past explicit next_run_at — unchanged from a much
            # earlier creation, now stale — rewind an already-armed
            # schedule; only a trigger's first-ever computation gets that
            # benefit of the doubt.
            setattr(
                trigger,
                "next_run_at",
                _compute_next_run_at(new_config, allow_past_explicit=False),
            )
        # else: already enabled, schedule fields unchanged (or the only
        # change is _is_benign_start_at_backfill's harmless start_at
        # backfill, see N8) — leave next_run_at as its current, possibly
        # scan-advanced value. Blindly recomputing here on every edit that
        # resends an unchanged schedule (rename, prompt tweak, secret
        # rotation, or an incidental timezone re-derivation) would re-read
        # the same stored explicit anchor and re-arm an already-progressed
        # schedule back to a stale due time.

    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    # Unregister on any config change, not just a binding change: provider
    # unregister is reference-counted teardown, so it no-ops while the
    # (possibly unchanged) binding is still referenced by an enabled trigger.
    new_config = dict(trigger.config or {})
    unregister_failed = False
    if old_enabled and (not bool(trigger.enabled) or old_config != new_config):
        try:
            _unregister_trigger_binding(
                db, trigger, trigger_type=old_type, config=old_config
            )
        except Exception:
            # The new binding above is already committed. Letting a teardown
            # failure (e.g. a DB error in the reference-counted release
            # lookup) propagate here would skip _register_trigger_with_provider
            # below entirely, leaving the trigger pointing at its new config
            # with no working watch on either the old or the new binding — a
            # silent "never fires again" state. Surface it on the trigger
            # instead of hiding it, and still attempt to register the new
            # binding.
            logger.exception(
                "Failed to unregister previous binding for trigger %s while "
                "updating it; continuing to register the new binding",
                trigger.id,
            )
            unregister_failed = True
            # A genuine DB-level failure (as opposed to a plain external-API
            # error, which release_gmail_mailbox_if_unused already catches
            # and warns on internally) leaves this session's transaction
            # unusable until rolled back — without this, the commit just
            # below can itself raise and defeat the whole point of this
            # except block.
            db.rollback()
            setattr(
                trigger, "provisioning_status", TriggerProvisioningStatus.FAILED.value
            )
            setattr(
                trigger,
                "provisioning_error",
                "Failed to release the previous trigger binding; verify manually.",
            )
            db.add(trigger)
            db.commit()
    _register_trigger_with_provider(db, trigger)
    if (
        unregister_failed
        and trigger.provisioning_status == TriggerProvisioningStatus.ACTIVE.value
    ):
        # The new binding just registered fine, which would otherwise
        # silently erase the FAILED marker set above — this trigger is
        # genuinely usable now, but the OLD binding may still be an orphaned
        # leak (its teardown failed). Keep that visible in
        # provisioning_error instead of letting a clean "active" status hide
        # it entirely.
        setattr(
            trigger,
            "provisioning_error",
            "New binding is active, but releasing the previous binding failed; "
            "verify it manually.",
        )
        db.add(trigger)
        db.commit()
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
    _delete_trigger(db, trigger)


def delete_workforce_trigger(
    db: Session,
    *,
    workforce_id: int,
    trigger_id: int,
) -> None:
    """Delete a workforce-owned trigger (workforce access checked by caller)."""
    trigger = get_workforce_trigger(
        db, workforce_id=workforce_id, trigger_id=trigger_id
    )
    if trigger is None:
        raise TriggerNotFoundError("Trigger not found")
    _delete_trigger(db, trigger)


def _delete_trigger(db: Session, trigger: AgentTrigger) -> None:
    trigger_type = str(trigger.type)
    trigger_id = trigger.id
    binding_config = dict(trigger.config or {})
    db.delete(trigger)
    db.commit()
    try:
        _unregister_trigger_binding(
            db, trigger, trigger_type=trigger_type, config=binding_config
        )
    except Exception:
        # The delete itself already succeeded and is committed above — a
        # teardown failure here (same DB-error mode _apply_trigger_updates
        # guards against) must not surface as a failed delete to the caller,
        # or the client is told the delete failed when it actually worked.
        logger.exception(
            "Failed to unregister binding for deleted trigger %s; the "
            "trigger row is gone but its provider-side binding may be leaked",
            trigger_id,
        )
        db.rollback()


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


def _attach_workforce_task_to_trigger_run(
    db: Session,
    *,
    trigger: AgentTrigger,
    run: TriggerRun,
    prompt: str,
    test: bool,
) -> TriggerRun:
    """Prepare a workforce run + pending task for a workforce trigger.

    Routes through ``create_workforce_run_record`` so every guard that
    protects interactive runs (``ensure_workforce_access``,
    ``validate_workforce_for_run``, snapshot/fingerprint pinning) applies to
    trigger firings too; a plain Task bound to the manager agent would
    silently lose all delegation ability. Failures propagate to
    ``prepare_trigger_run``, which marks the run FAILED for idempotent retry.
    """
    # Local import: workforce_runs pulls in the orchestrator stack, which
    # would otherwise risk an import cycle with this module.
    from .workforce_runs import create_workforce_run_record

    user = db.get(User, int(trigger.user_id))
    if user is None:
        raise TriggerServiceError("Trigger owner not found")
    workforce = db.get(Workforce, int(trigger.workforce_id))
    if workforce is None:
        raise TriggerServiceError("Workforce not found")
    record = create_workforce_run_record(
        db,
        user,
        workforce,
        message=prompt,
        is_visible=False,
        source="trigger",
        # TriggerRun-level dedup already guarantees one run per event; this
        # workforce-level key makes a retried attachment after a partial
        # failure replay the same workforce run instead of creating another.
        # (If the replayed run's task was hard-deleted in between, the record
        # helper raises 409 and the run is marked FAILED — replay is
        # impossible, so failing is correct rather than starting a fresh run.)
        idempotency_key=f"trigger:{run.id}",
    )
    # Idempotent re-attach. create_workforce_run_record commits the
    # WorkforceRun+Task before this function's own commit below, so a crash in
    # between leaves run.task_id unset; a retry replays the same workforce run
    # (record.created is False) and re-enters here. Only apply what is still
    # missing and never knock a run that already advanced back to PENDING.
    task = record.task
    task_id = int(task.id)
    merged_config = {
        **dict(task.agent_config or {}),
        **_trigger_execution_context(trigger=trigger, run=run, test=test),
    }
    if merged_config != dict(task.agent_config or {}):
        setattr(task, "agent_config", merged_config)
        db.add(task)
    if run.task_id is None or int(run.task_id) != task_id:
        setattr(run, "task_id", task_id)
        db.add(run)
    # Arm only a run that has not moved past preparation; a replay must not
    # regress a RUNNING/COMPLETED run, and re-writing an already-PENDING run
    # is a no-op we skip.
    if str(run.status) not in (
        TriggerRunStatus.RUNNING.value,
        TriggerRunStatus.COMPLETED.value,
        TriggerRunStatus.PENDING.value,
    ):
        setattr(run, "status", TriggerRunStatus.PENDING.value)
        db.add(run)
    db.commit()
    db.refresh(run)
    return run


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
    if trigger.workforce_id is not None:
        return _attach_workforce_task_to_trigger_run(
            db, trigger=trigger, run=run, prompt=prompt, test=test
        )
    agent = db.get(Agent, trigger.agent_id)
    if agent is None:
        raise TriggerServiceError("Agent not found")
    if is_workforce_generated_manager_agent(agent):
        # Defense in depth (#950): even if a row bound to a generated manager
        # agent slips past ownership resolution and the migration cleanup, it
        # must never construct a plain Task — that path drops delegation.
        raise TriggerServiceError(
            "Trigger is bound to a workforce manager agent and cannot fire"
        )
    missing_secret_error_code = (
        ERROR_SCHEDULED_SECRET_UNAVAILABLE
        if str(trigger.type) == TriggerType.SCHEDULED.value
        else ERROR_RUNTIME_SECRET_UNAVAILABLE
    )
    task_source = "trigger"
    task_owner_user_id = int(trigger.user_id)
    runtime_plan = prepare_create_connector_runtime(
        db=db,
        agent=agent,
        task_source=task_source,
        connector_user_id=task_owner_user_id,
        payload_items=_trigger_connector_runtime_payload(trigger.config),
        allow_ephemeral=False,
        missing_ephemeral_error_code=missing_secret_error_code,
    )
    task = Task(
        user_id=task_owner_user_id,
        title=_trigger_task_title(trigger, prompt),
        description=prompt,
        status=TaskStatus.PENDING,
        agent_id=int(trigger.agent_id),
        execution_mode=getattr(agent, "execution_mode", None) or "balanced",
        source=task_source,
        is_visible=False,
        input=prompt,
        agent_config=_trigger_execution_context(
            trigger=trigger,
            run=run,
            test=test,
        ),
    )
    bind_create_connector_runtime_plan(task=task, plan=runtime_plan)
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


# Counter tracking consecutive prepare_trigger_run failures per trigger (PR
# #1051 review, N follow-up; persisted per a later review round). A single
# failure here is expected and routinely transient (see the except clause
# below), so it must not disable the trigger or surface anything — but a
# trigger that fails EVERY tick for a while is no longer just an unlucky
# race, and was previously retried forever with zero user-visible signal.
#
# Persisted on AgentTrigger.consecutive_prepare_failures rather than an
# in-process dict: this scan runs from at least two genuinely separate OS
# processes concurrently in the documented deployment topology (the backend's
# in-process asyncio dispatcher, see _run_trigger_dispatcher in web/app.py,
# AND a separate Celery beat/worker scan, see web/jobs/trigger_tasks.py) —
# a per-process dict can split one trigger's failures across processes such
# that NEITHER ever reaches the threshold, and — worse — the recovery-clear
# check only seeing its own process's dict means a later successful prepare
# handled by a DIFFERENT process than the one that set the failed badge would
# never clear it, permanently misleading the user. Reading/writing the
# trigger's own row makes both the increment and the clear correct across
# processes: the increment uses a single atomic `UPDATE ... SET
# consecutive_prepare_failures = COALESCE(...) + 1` (not a Python
# read-then-write) so a lost update between two concurrent processes is far
# less likely than with the old per-process counter, and the clear reads the
# row fresh (db.refresh) so it always sees the latest cross-process state.
_PREPARE_FAILURE_SURFACE_THRESHOLD = 5


def _increment_consecutive_prepare_failures(db: Session, trigger: AgentTrigger) -> int:
    """Atomically bump the trigger's persisted failure counter; return the
    new, authoritative value (shared across every process scanning this
    trigger, not just this one)."""
    db.execute(
        update(AgentTrigger)
        .where(AgentTrigger.id == trigger.id)
        .values(
            consecutive_prepare_failures=func.coalesce(
                AgentTrigger.consecutive_prepare_failures, 0
            )
            + 1
        )
        .execution_options(synchronize_session=False)
    )
    db.commit()
    db.refresh(trigger)
    return int(trigger.consecutive_prepare_failures or 0)


def _clear_consecutive_prepare_failures_if_recovered(
    db: Session, trigger: AgentTrigger
) -> None:
    """After a genuinely clean prepare_trigger_run call this tick, clear the
    persisted counter and any stale failed badge this same guard set on a
    previous tick — reading the trigger's own row fresh (not an in-process
    dict) so this is correct even when the failures being cleared were
    accumulated by a different process than the one running this tick."""
    db.refresh(trigger)
    if trigger.consecutive_prepare_failures is None:
        return
    setattr(trigger, "consecutive_prepare_failures", None)
    if trigger.provisioning_status == TriggerProvisioningStatus.FAILED.value:
        setattr(trigger, "provisioning_status", None)
        setattr(trigger, "provisioning_error", None)
    db.add(trigger)
    db.commit()


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
        prepare_run_failed_this_tick = False
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
            # This is still a failure for THIS tick (just a different failure
            # mode than the except clause below) — flagged so the
            # recovered-so-clear-the-badge block further down doesn't mistake
            # "didn't hit the other except clause" for "genuinely succeeded".
            run = exc.run
            prepare_run_failed_this_tick = True
        except (ValueError, IntegrityError, OperationalError):
            # `TriggerServiceError` (a `ValueError` subclass) here is
            # realistically either "Trigger is disabled" (prepare_trigger_run's
            # own guard) — a legitimate race if another request disabled this
            # trigger between this scan's due-triggers query and this call —
            # or a bare `ValueError` raised by `_get_encryption_key` when
            # `ENCRYPTION_KEY` is unset in a non-development environment and
            # this trigger's config has `store_full_payload: true`
            # (encrypt_value -> _payload_snapshot -> _get_or_create_trigger_run,
            # see core/utils/encryption.py). `IntegrityError`/`OperationalError`
            # cover a bare insert-conflict escaping _get_or_create_trigger_run's
            # own retry, or a transient DB/connection hiccup on its commit.
            # None of these have a TriggerRun to keep advancing. Triggers are
            # scanned in next_run_at ASC order, so letting this propagate
            # would abort the whole batch and starve every trigger ordered
            # after this one — roll back and move on to the next due trigger
            # instead; this one is retried on the next scan tick since its
            # next_run_at is left untouched. Widened from just
            # `TriggerServiceError` to bare `ValueError` (still meaningfully
            # narrower than a bare `except Exception`, so a genuinely
            # unexpected exception type — e.g. AttributeError/KeyError/TypeError
            # from a real bug — still propagates and aborts the batch loudly
            # instead of being silently absorbed) (PR #1051 review, N follow-up).
            logger.exception(
                "Failed to prepare trigger run for trigger %s; skipping it this tick",
                trigger.id,
            )
            db.rollback()
            failure_count = _increment_consecutive_prepare_failures(db, trigger)
            if failure_count >= _PREPARE_FAILURE_SURFACE_THRESHOLD:
                # A one-off failure above is expected and silently retried by
                # design — but this many consecutive scan ticks failing for
                # the SAME trigger is no longer just an unlucky race.
                # Surface it (same field the Gmail provider and the
                # recompute-failure guard below use) WITHOUT disabling the
                # trigger: unlike a bad schedule config (permanent until
                # edited), a run-preparation failure is commonly a transient
                # infrastructure issue, so silently killing the schedule over
                # something that may self-resolve would trade one silent
                # failure mode for a worse one. It keeps retrying every tick;
                # this is purely a visibility improvement.
                logger.warning(
                    "Trigger %s has failed to prepare a run %s times in a row",
                    trigger.id,
                    failure_count,
                )
                setattr(
                    trigger,
                    "provisioning_status",
                    TriggerProvisioningStatus.FAILED.value,
                )
                setattr(
                    trigger,
                    "provisioning_error",
                    "Repeated failures preparing a run "
                    f"({failure_count} consecutive attempts); still "
                    "retrying automatically.",
                )
                db.add(trigger)
                db.commit()
            continue

        if not prepare_run_failed_this_tick:
            # Only a genuinely clean prepare_trigger_run call this tick counts
            # as "recovered" — a TriggerRunPreparationError above still means
            # the trigger is failing (just via a different failure mode), so
            # that branch sets prepare_run_failed_this_tick and must not let
            # this cleanup fire despite not going through the except clause
            # just above that increments the counter.
            _clear_consecutive_prepare_failures_if_recovered(db, trigger)

        config = dict(trigger.config or {})
        disable_reason: str | None = None
        try:
            next_run_at = _compute_next_run_at(
                config,
                from_time=scan_time,
                previous_due_at=due_at,
                include_explicit=False,
            )
        except (ValueError, ArithmeticError) as exc:
            # Triggers are scanned in next_run_at ASC order, so an unguarded
            # raise here would leave this trigger's next_run_at unadvanced
            # and permanently first in line — wedging every trigger ordered
            # after it on every subsequent scan. Disable it instead, the same
            # way an unschedulable (next_run_at is None) trigger already is
            # just below. `ValueError` covers TriggerServiceError (a
            # ValueError subclass — every malformed-field case
            # _compute_next_run_at itself raises) and `ArithmeticError`
            # covers the bare OverflowError a pathological config (e.g. an
            # absurd interval_seconds predating this PR's upper-bound cap)
            # can raise out of the alignment arithmetic. Deliberately NOT a
            # bare `except Exception`: that would also swallow a genuine,
            # unrelated programming bug (e.g. a future AttributeError) as if
            # it were just a bad config, silently disabling the trigger
            # instead of surfacing the bug loudly (PR #1051 review, N follow-up).
            logger.exception(
                "Failed to recompute next_run_at for trigger %s; disabling it",
                trigger.id,
            )
            next_run_at = None
            disable_reason = (
                f"Disabled automatically: failed to compute the next run "
                f"time ({type(exc).__name__}: {exc})."
            )
        setattr(trigger, "next_run_at", next_run_at)
        if next_run_at is None:
            setattr(trigger, "enabled", False)
        if disable_reason is not None:
            # Surface the failure on the trigger itself (the same field the
            # Gmail provider uses to report provisioning failures) — a plain
            # logger.exception is invisible to the user, who otherwise just
            # sees a schedule that silently stopped firing.
            setattr(
                trigger, "provisioning_status", TriggerProvisioningStatus.FAILED.value
            )
            setattr(trigger, "provisioning_error", disable_reason)
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
