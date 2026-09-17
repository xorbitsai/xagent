"""Webhook trigger provider: HMAC-signed requests on the unified callback route.

The callback id in the URL is a non-secret locator. Authentication is an
HMAC-SHA256 signature over ``{timestamp}.{raw_body}`` using the trigger's
encrypted secret, carried in the ``x-xagent-signature`` and
``x-xagent-timestamp`` headers, with a replay window.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Mapping

from pydantic import ValidationError
from sqlalchemy.orm import Session

from ....core.utils.encryption import decrypt_value
from ...models.trigger import AgentTrigger, TriggerType
from .base import (
    CallbackRequestContext,
    TriggerConfigError,
    TriggerEventParseError,
)
from .registry import register_trigger_provider
from .schemas import (
    AckPolicy,
    ChallengeResponse,
    NormalizedEvent,
    RegistrationResult,
    VerificationResult,
    parse_trigger_config,
)

SIGNATURE_HEADER = "x-xagent-signature"
TIMESTAMP_HEADER = "x-xagent-timestamp"
EVENT_TYPE_HEADER = "x-xagent-event-type"
EVENT_ID_HEADERS = ("x-xagent-event-id", "x-event-id", "x-request-id")
REPLAY_WINDOW_SECONDS = 300


def sign_webhook_payload(secret: str, timestamp: str, raw_body: bytes) -> str:
    """Compute the expected webhook signature for a request."""
    message = timestamp.encode("utf-8") + b"." + raw_body
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


class WebhookProvider:
    """First real implementation of the unified TriggerProvider pipeline."""

    name = TriggerType.WEBHOOK.value
    ack_policy = AckPolicy()

    def validate_config(self, config: Mapping[str, Any]) -> Any:
        try:
            return parse_trigger_config(self.name, dict(config))
        except ValidationError as exc:
            raise TriggerConfigError(str(exc)) from exc

    def locate_trigger(self, db: Session, callback_id: str) -> AgentTrigger | None:
        return (
            db.query(AgentTrigger)
            .filter(
                AgentTrigger.callback_id == callback_id,
                AgentTrigger.type == self.name,
            )
            .first()
        )

    def handle_challenge(
        self, context: CallbackRequestContext, raw_body: bytes
    ) -> ChallengeResponse | None:
        return None

    def authorize_resource(
        self,
        trigger: AgentTrigger,
        attested_resource_id: str | None,
        event: NormalizedEvent,
    ) -> bool:
        # A valid signature already proves the caller holds this trigger's
        # secret; webhooks carry no separate provider-attested resource.
        return True

    async def verify(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> VerificationResult:
        if trigger is None or not trigger.secret_encrypted:
            return VerificationResult.reject("Webhook trigger has no signing secret")

        signature = (context.header(SIGNATURE_HEADER) or "").strip()
        timestamp = (context.header(TIMESTAMP_HEADER) or "").strip()
        if not signature or not timestamp:
            return VerificationResult.reject(
                "Missing webhook signature or timestamp header"
            )

        try:
            timestamp_seconds = int(timestamp)
        except ValueError:
            return VerificationResult.reject("Invalid webhook timestamp")
        if abs(time.time() - timestamp_seconds) > REPLAY_WINDOW_SECONDS:
            return VerificationResult.reject(
                "Webhook timestamp outside the replay window"
            )

        secret = decrypt_value(str(trigger.secret_encrypted))
        expected = sign_webhook_payload(secret, timestamp, raw_body)
        provided = signature.removeprefix("sha256=")
        if not hmac.compare_digest(expected, provided):
            return VerificationResult.reject("Invalid webhook signature")
        return VerificationResult.ok()

    async def register(
        self, db: Session, trigger: AgentTrigger, config: Any
    ) -> RegistrationResult:
        from ...models.trigger import TriggerProvisioningStatus

        return RegistrationResult(status=TriggerProvisioningStatus.ACTIVE)

    async def unregister(
        self,
        db: Session,
        trigger: AgentTrigger,
        config: Any,
        *,
        resource_id: str | None = None,
    ) -> None:
        return None

    async def parse_events(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> list[NormalizedEvent]:
        if not raw_body:
            payload: dict[str, Any] = {}
        else:
            try:
                decoded = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise TriggerEventParseError(
                    "Webhook payload must be a JSON document"
                ) from exc
            if isinstance(decoded, dict):
                payload = decoded
            else:
                payload = {"value": decoded}

        source_event_id = None
        for header in EVENT_ID_HEADERS:
            source_event_id = context.header(header)
            if source_event_id:
                break
        if not source_event_id:
            for key in ("id", "event_id"):
                value = payload.get(key)
                if value:
                    source_event_id = str(value)
                    break

        event_type = (
            context.header(EVENT_TYPE_HEADER)
            or str(payload.get("event_type") or "")
            or "webhook"
        )
        return [
            NormalizedEvent(
                event_type=event_type,
                source_event_id=source_event_id,
                payload=payload,
            )
        ]

    async def finalize_callback(
        self,
        *,
        db: Session,
        context: CallbackRequestContext,
        trigger: AgentTrigger | None,
        events: list[NormalizedEvent],
        raw_body: bytes,
    ) -> None:
        """Webhook delivery has no cursor state to advance."""


register_trigger_provider(WebhookProvider(), replace=True)
