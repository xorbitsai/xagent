"""Common schemas shared by all trigger providers.

These models define the provider-agnostic vocabulary of the unified callback
pipeline: normalized events, verification results, acknowledgement policy,
registration results, and the typed trigger config union.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator, model_validator

from ...models.trigger import TriggerProvisioningStatus, TriggerType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NormalizedEvent(BaseModel):
    """One provider event normalized into the shared pipeline shape."""

    event_type: str
    source_event_id: str | None = None
    target_trigger_id: int | None = None
    """Optional trigger selected by provider-specific ingestion.

    Most providers address one trigger per callback. Shared-delivery providers
    such as Gmail can fan one callback out to multiple matching triggers while
    still letting the shared pipeline own authorization, idempotency, and audit.
    """
    resource_id: str | None = None
    """Payload-claimed resource identity. Never trusted for authorization."""
    payload: dict[str, Any] = Field(default_factory=dict)
    received_at: datetime = Field(default_factory=_utcnow)


class VerificationResult(BaseModel):
    """Outcome of provider-specific callback authentication."""

    verified: bool
    attested_resource_id: str | None = None
    """Resource identity proven by the provider's trust model (for example a
    mailbox derived from a verified OIDC token), as opposed to any identity
    claimed inside the payload."""
    reason: str | None = None

    @classmethod
    def ok(cls, *, attested_resource_id: str | None = None) -> "VerificationResult":
        return cls(verified=True, attested_resource_id=attested_resource_id)

    @classmethod
    def reject(cls, reason: str) -> "VerificationResult":
        return cls(verified=False, reason=reason)


class ChallengeResponse(BaseModel):
    """Immediate response to a provider handshake/challenge request."""

    status_code: int = 200
    body: str = ""
    media_type: str = "text/plain"


class AckPolicy(BaseModel):
    """HTTP acknowledgement behavior, decoupled from audit outcome.

    Providers with aggressive redelivery (for example Pub/Sub push) can map
    terminal rejections to 2xx to stop redelivery while the audit trail still
    records the real outcome.
    """

    accepted_status: int = 200
    not_found_status: int = 404
    rejected_status: int = 401
    rejected_resource_status: int = 403
    disabled_status: int = 409
    parse_failure_status: int = 400
    """Malformed payloads are permanent: redelivery-heavy providers map this
    to 2xx so the same broken message is not redelivered forever."""
    failure_status: int = 500


class RegistrationResult(BaseModel):
    """Result of provisioning provider-side delivery for a trigger."""

    status: TriggerProvisioningStatus
    resource_id: str | None = None
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class BaseTriggerConfig(BaseModel):
    """Fields shared by every typed trigger config."""

    event_types: list[str] | None = None
    """Optional allow-list of normalized event types this trigger fires on."""
    store_full_payload: bool = False
    """Opt-in to encrypted full-payload snapshots on trigger runs."""


# Shared normalization for schedule fields. Single source of truth used both
# by the pydantic validators below (API-time validation) and by the service
# layer (triggers.py) when recomputing schedules from stored configs — the
# same rules, one implementation, two error-wrapping styles.


def normalize_weekdays(value: Any) -> set[int]:
    """Validate and coerce a weekdays list (0=Mon..6=Sun) to a set."""
    if isinstance(value, (str, int, bool)):
        # A bare scalar (e.g. weekdays=3) means a single day, not a sequence
        # to iterate — coerce it to a one-element list first so it isn't
        # split character-by-character (str) or rejected as non-iterable
        # (int/bool).
        value = [str(value)]
    try:
        weekday_set = {int(day) for day in (value or [])}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "weekdays must be a non-empty list of integers 0-6 (Mon-Sun)"
        ) from exc
    if not weekday_set or not weekday_set.issubset(range(7)):
        raise ValueError("weekdays must be a non-empty list of integers 0-6 (Mon-Sun)")
    return weekday_set


def normalize_day_of_month(value: Any) -> int:
    """Validate and coerce day_of_month (1-31)."""
    try:
        day = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("day_of_month must be between 1 and 31") from exc
    if not (1 <= day <= 31):
        raise ValueError("day_of_month must be between 1 and 31")
    return day


def normalize_schedule_timezone(value: Any) -> ZoneInfo:
    """Validate an IANA timezone name and return its ZoneInfo."""
    try:
        return ZoneInfo(str(value))
    except (ZoneInfoNotFoundError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"unknown timezone: {value!r}") from exc


class WebhookTriggerConfig(BaseTriggerConfig):
    type: Literal["webhook"] = "webhook"


class ScheduledTriggerConfig(BaseTriggerConfig):
    type: Literal["scheduled"] = "scheduled"
    interval_seconds: int | None = None
    next_run_at: str | None = None
    recurrence: Literal["hourly", "daily", "weekly", "monthly", "custom"] | None = None
    """Recurrence family driving the schedule UI. Hourly/daily/custom keep the
    flat interval_seconds/next_run_at mechanism below (interval_seconds is
    still what the scheduler advances by); weekly/monthly require real
    calendar math and are computed from weekdays/day_of_month instead."""
    time_of_day: str | None = None
    """"HH:MM" (24h), the time-of-day component for daily/weekly/monthly."""
    weekdays: list[int] | None = None
    """0=Monday..6=Sunday. Required when recurrence == "weekly"."""
    day_of_month: int | None = None
    """1-31, clamped to the last day of short months. Required when
    recurrence == "monthly"."""
    start_at: str | None = None
    """ISO date/datetime; optional anchor before which weekly/monthly won't
    fire their first run."""

    @field_validator("interval_seconds")
    @classmethod
    def _positive_interval(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("interval_seconds must be positive")
        return value

    timezone: str | None = None
    """IANA timezone name (e.g. "Asia/Shanghai") that time_of_day and
    weekdays/day_of_month are expressed in. Defaults to UTC when absent
    (legacy configs)."""

    @field_validator("weekdays")
    @classmethod
    def _valid_weekdays(cls, value: list[int] | None) -> list[int] | None:
        if value is not None:
            normalize_weekdays(value)
        return value

    @field_validator("day_of_month")
    @classmethod
    def _valid_day_of_month(cls, value: int | None) -> int | None:
        if value is not None:
            normalize_day_of_month(value)
        return value

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str | None) -> str | None:
        if value is not None:
            normalize_schedule_timezone(value)
        return value

    @model_validator(mode="after")
    def _require_schedule(self) -> "ScheduledTriggerConfig":
        if self.recurrence == "weekly":
            if not self.weekdays:
                raise ValueError("weekly schedule requires weekdays")
            return self
        if self.recurrence == "monthly":
            if self.day_of_month is None:
                raise ValueError("monthly schedule requires day_of_month")
            return self
        if self.interval_seconds is None and not (self.next_run_at or "").strip():
            raise ValueError(
                "scheduled trigger requires interval_seconds or next_run_at"
            )
        return self


class GmailTriggerConfig(BaseTriggerConfig):
    type: Literal["gmail"] = "gmail"
    oauth_account_id: int | None = None
    """Connected Gmail OAuth account this trigger is bound to. Optional at the
    schema level during rollout; API-level enforcement lands with the typed
    trigger config slice."""
    watch_label: str
    sender_filter: str | None = None
    subject_keyword: str | None = None

    @field_validator("watch_label")
    @classmethod
    def _non_empty_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("gmail trigger requires watch_label")
        return value


TriggerConfig = Annotated[
    Union[WebhookTriggerConfig, ScheduledTriggerConfig, GmailTriggerConfig],
    Field(discriminator="type"),
]


class _TriggerConfigEnvelope(BaseModel):
    config: TriggerConfig


def parse_trigger_config(trigger_type: str, config: dict[str, Any]) -> Any:
    """Validate a raw config dict against the typed schema for trigger_type.

    The discriminator lives on the trigger row rather than inside the stored
    config JSON, so it is injected here before validation.
    """
    normalized_type = TriggerType(trigger_type).value
    payload = {**config, "type": normalized_type}
    return _TriggerConfigEnvelope(config=payload).config


def dump_trigger_config(config: BaseTriggerConfig) -> dict[str, Any]:
    """Serialize a typed config back to the stored JSON shape (no type key)."""
    data = config.model_dump(exclude_none=True)
    data.pop("type", None)
    return data
