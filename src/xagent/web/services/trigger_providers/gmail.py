"""Gmail trigger provider for the unified callback pipeline."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from google.auth.exceptions import TransportError as GoogleTransportError
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token
from pydantic import ValidationError
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from ....config import (
    get_gmail_callback_base_url,
    get_gmail_pubsub_project_id,
    get_gmail_pubsub_push_service_account,
    get_gmail_watch_enabled,
)
from ...models.gmail_watch import GmailWatchState
from ...models.trigger import AgentTrigger, TriggerProvisioningStatus, TriggerType
from ...models.user_oauth import UserOAuth
from ..gmail_provisioning import (
    GMAIL_WATCH_DISABLED_ERROR,
    gmail_callback_url,
    provision_gmail_trigger,
    release_gmail_mailbox_if_unused,
)
from ..gmail_triggers import (
    gmail_binding_id,
    is_legacy_gmail_binding,
    ordinary_gmail_triggers,
)
from ..ops_signals import (
    GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED,
    GMAIL_WATCH_REGISTRATION_DISABLED,
    clear_degradation,
    register_degradation,
)
from ..user_oauth import (
    get_scoped_user_oauth_account,
    is_ordinary_gmail,
    ordinary_gmail_clause,
)
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

if TYPE_CHECKING:
    from ..gmail_triggers import GmailPubsubNotification, GmailServiceFactory

logger = logging.getLogger(__name__)

OidcVerifier = Callable[[str, str], Mapping[str, Any]]

GOOGLE_OIDC_ISSUERS = frozenset({"https://accounts.google.com", "accounts.google.com"})


def warn_if_gmail_oidc_verification_degraded() -> None:
    """Startup config-drift check for Gmail OIDC verification.

    Provisioning requires the push service account, so a configured Pub/Sub
    project without one means inbound Gmail callbacks will verify with only
    issuer/audience/signature checks. Registering the degradation at startup
    surfaces the drift on /health before the first callback arrives; the
    verify path keeps the signal current afterwards.
    """
    if not get_gmail_pubsub_project_id() or get_gmail_pubsub_push_service_account():
        return
    register_degradation(
        GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED,
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT is not configured; "
        "Gmail OIDC verification is running without service-account "
        "email checks",
    )
    logger.warning(
        "XAGENT_GMAIL_PUBSUB_PROJECT_ID is set but "
        "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT is not; Gmail OIDC "
        "verification will skip service-account email checks"
    )


def warn_if_gmail_watch_registration_degraded() -> None:
    """Startup config-drift check for Gmail watch registration.

    A configured Pub/Sub project without the watch feature enabled means
    Gmail watch registration and renewal are disabled: Gmail triggers report
    failed provisioning and existing watches expire unrenewed. Registering
    the degradation at startup surfaces the drift on /health.
    """
    if not get_gmail_pubsub_project_id() or get_gmail_watch_enabled():
        return
    message = (
        "XAGENT_GMAIL_PUBSUB_PROJECT_ID is set but XAGENT_GMAIL_WATCH_ENABLED "
        "is not; Gmail watch registration and renewal are disabled, so Gmail "
        "triggers report failed provisioning and existing watches expire "
        "unrenewed"
    )
    register_degradation(GMAIL_WATCH_REGISTRATION_DISABLED, message)
    logger.warning(message)


def _accepted_callback_audiences(
    state: GmailWatchState, callback_id: str
) -> tuple[str, ...]:
    """Return the audiences valid during a callback endpoint transition.

    Reconciliation must update Pub/Sub before it can persist the corresponding
    audience locally. During that short interval, the stored old audience and
    the configured new audience cover both sides of the cloud-before-database
    transition. After the commit, those values collapse to the new URL while
    the durable previous audience remains accepted until its bounded grace
    period expires.
    """
    stored_audience = str(state.push_audience or "").strip()
    callback_base_url = get_gmail_callback_base_url()
    configured_audience = (
        gmail_callback_url(callback_base_url, callback_id) if callback_base_url else ""
    )
    previous_audience = str(state.previous_push_audience or "").strip()
    previous_audience_expires_at = cast(
        datetime | None, state.previous_push_audience_expires_at
    )
    if previous_audience_expires_at is None or _as_utc(
        previous_audience_expires_at
    ) <= datetime.now(timezone.utc):
        previous_audience = ""
    return tuple(
        dict.fromkeys(
            audience
            for audience in (
                stored_audience,
                configured_audience,
                previous_audience,
            )
            if audience
        )
    )


def _as_utc(value: datetime) -> datetime:
    """Normalize SQLite's timezone-naive datetime round trips to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _watch_state_for_callback(db: Session, callback_id: str) -> GmailWatchState | None:
    """Resolve a callback only through a same-user ordinary Gmail account."""
    return (
        db.query(GmailWatchState)
        .join(UserOAuth, UserOAuth.id == GmailWatchState.oauth_account_id)
        .filter(
            GmailWatchState.callback_id == callback_id,
            UserOAuth.user_id == GmailWatchState.user_id,
            ordinary_gmail_clause(),
        )
        .first()
    )


def _normalized_email(value: object) -> str:
    return str(value or "").strip().lower()


def _bearer_token(context: CallbackRequestContext) -> str | None:
    authorization = context.header("authorization") or ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _claim_audience_matches(claim_value: object, expected: str) -> bool:
    if isinstance(claim_value, str):
        return claim_value == expected
    if isinstance(claim_value, list):
        return expected in {str(item) for item in claim_value}
    return False


def _claim_email_verified(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def _history_cursor_advances(current: object, incoming: str) -> bool:
    """True when incoming historyId moves the watch cursor forward.

    Gmail history ids are monotonically increasing integers. Rejecting
    non-advancing ids keeps out-of-order Pub/Sub redeliveries - and stale
    notifications processed right after an expired-history re-registration
    reset the cursor - from rolling the cursor backwards.
    """
    try:
        return int(incoming) > int(str(current or "0") or "0")
    except (TypeError, ValueError):
        return True


# Shared transport so Google's signing certs benefit from connection reuse
# instead of a fresh session (and TLS handshake) per callback.
_GOOGLE_AUTH_REQUEST = GoogleAuthRequest()

# Pub/Sub push and this host may drift by a few seconds; without tolerance a
# fresh token can fail iat validation and the rejection would be ACKed.
_OIDC_CLOCK_SKEW_SECONDS = 10


def verify_google_oidc_token(token: str, audience: str) -> Mapping[str, Any]:
    """Verify a Google OIDC token signature and audience.

    Performs blocking HTTP (Google cert fetch); call via asyncio.to_thread
    from async code.
    """
    claims = id_token.verify_oauth2_token(
        token,
        _GOOGLE_AUTH_REQUEST,
        audience,
        clock_skew_in_seconds=_OIDC_CLOCK_SKEW_SECONDS,
    )
    return claims if isinstance(claims, Mapping) else {}


def _decode_pubsub_notification(
    raw_body: bytes, *, attested_email: str
) -> GmailPubsubNotification:
    from ..gmail_triggers import GmailPubsubNotification

    try:
        envelope = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TriggerEventParseError(
            "Gmail Pub/Sub envelope is not valid JSON"
        ) from exc
    if not isinstance(envelope, dict):
        raise TriggerEventParseError("Gmail Pub/Sub envelope must be an object")

    message = envelope.get("message")
    if not isinstance(message, dict):
        raise TriggerEventParseError("Gmail Pub/Sub envelope missing message")

    data = message.get("data")
    if not isinstance(data, str) or not data.strip():
        raise TriggerEventParseError("Gmail Pub/Sub message missing data")

    try:
        padded = data + ("=" * (-len(data) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TriggerEventParseError("Gmail Pub/Sub message data is invalid") from exc
    if not isinstance(payload, dict):
        raise TriggerEventParseError("Gmail Pub/Sub message data must be an object")

    history_id = payload.get("historyId")
    if history_id in (None, ""):
        raise TriggerEventParseError("Gmail Pub/Sub notification missing historyId")

    pubsub_message_id = message.get("messageId") or message.get("message_id")
    return GmailPubsubNotification(
        email_address=attested_email,
        history_id=str(history_id),
        pubsub_message_id=str(pubsub_message_id) if pubsub_message_id else None,
    )


class GmailProvider:
    """Gmail provider for per-mailbox Pub/Sub push callbacks."""

    name = TriggerType.GMAIL.value
    ack_policy = AckPolicy(
        not_found_status=200,
        rejected_status=200,
        rejected_resource_status=200,
        disabled_status=200,
        # A malformed Pub/Sub message fails identically on every redelivery;
        # ack it so it does not loop for the retention window. Transient
        # ingestion errors raise GmailTriggerError instead, which maps to
        # failure_status=500 and keeps redelivery semantics.
        parse_failure_status=200,
    )

    def __init__(
        self,
        *,
        service_factory: GmailServiceFactory | None = None,
        oidc_verifier: OidcVerifier | None = None,
    ) -> None:
        self.service_factory = service_factory
        self.oidc_verifier = oidc_verifier or verify_google_oidc_token

    def validate_config(self, config: Mapping[str, Any]) -> Any:
        try:
            return parse_trigger_config(self.name, dict(config))
        except ValidationError as exc:
            raise TriggerConfigError(str(exc)) from exc

    def locate_trigger(self, db: Session, callback_id: str) -> AgentTrigger | None:
        state = _watch_state_for_callback(db, callback_id)
        if state is None:
            return None
        # Prefer this mailbox. Explicit bindings to another account fail
        # closed; only unbound legacy triggers use mailbox matching.
        mailbox_matches = case(
            (
                func.lower(AgentTrigger.resource_id) == _normalized_email(state.email),
                0,
            ),
            else_=1,
        )
        candidates = (
            db.query(AgentTrigger)
            .filter(
                AgentTrigger.user_id == int(state.user_id),
                AgentTrigger.type == self.name,
                AgentTrigger.provider == self.name,
            )
            .order_by(
                AgentTrigger.enabled.desc(),
                mailbox_matches,
                AgentTrigger.id.asc(),
            )
            .all()
        )
        ordinary = ordinary_gmail_triggers(
            triggers=candidates,
            oauth_account_id=int(state.oauth_account_id),
            mailbox=_normalized_email(state.email),
        )
        # An invalid binding is acknowledged as unknown before parsing. The
        # cursor stays unchanged until an ordinary trigger can consume it.
        return ordinary[0] if ordinary else None

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
        if not trigger.resource_id or not attested_resource_id:
            return False
        return (
            str(trigger.resource_id).strip().lower()
            == attested_resource_id.strip().lower()
        )

    async def verify(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> VerificationResult:
        state = _watch_state_for_callback(db, context.callback_id)
        if state is None:
            return VerificationResult.reject("Unknown Gmail callback")

        audiences = _accepted_callback_audiences(state, context.callback_id)
        if not audiences:
            return VerificationResult.reject(
                "Gmail callback audience is not configured"
            )

        token = _bearer_token(context)
        if token is None:
            return VerificationResult.reject("Missing Gmail OIDC bearer token")

        claims: Mapping[str, Any] | None = None
        verification_error: Exception | None = None
        for audience in audiences:
            try:
                candidate_claims = await asyncio.to_thread(
                    self.oidc_verifier, token, audience
                )
            except GoogleTransportError:
                # Fetching Google's JWKS certs failed; nothing about the token
                # was proven invalid. Propagate so the pipeline answers with
                # failure_status and Pub/Sub redelivers, instead of rejecting -
                # Gmail's rejected_status=200 would drop the event permanently.
                raise
            except Exception as exc:
                verification_error = exc
                continue

            if _claim_audience_matches(candidate_claims.get("aud"), audience):
                claims = candidate_claims
                break

        if claims is None:
            if verification_error is not None:
                return VerificationResult.reject(
                    "Gmail OIDC token verification failed: "
                    f"{type(verification_error).__name__}"
                )
            return VerificationResult.reject("Gmail OIDC audience does not match")

        issuer = str(claims.get("iss") or "")
        if issuer not in GOOGLE_OIDC_ISSUERS:
            return VerificationResult.reject("Gmail OIDC issuer is not trusted")

        expected_service_account = get_gmail_pubsub_push_service_account()
        if expected_service_account:
            clear_degradation(GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED)
            claim_email = _normalized_email(claims.get("email"))
            expected_email = expected_service_account.strip().lower()
            if claim_email != expected_email or not _claim_email_verified(
                claims.get("email_verified")
            ):
                return VerificationResult.reject(
                    "Gmail OIDC email claim must match configured service account "
                    "and email_verified must be true"
                )
        else:
            # Provisioning requires this env var, so its absence here means
            # config drift between the provisioning and callback processes;
            # only issuer/audience/signature checks remain in that case. The
            # degradation registry keeps this visible to monitoring (via
            # /health) instead of only to log readers.
            register_degradation(
                GMAIL_OIDC_SERVICE_ACCOUNT_UNVERIFIED,
                "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT is not configured; "
                "Gmail OIDC verification is running without service-account "
                "email checks",
            )
            logger.warning(
                "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT is not configured; "
                "skipping Gmail OIDC service-account email verification for "
                "callback %s",
                context.callback_id,
            )

        attested_email = _normalized_email(state.email)
        if not attested_email:
            return VerificationResult.reject("Gmail watch state email is empty")
        return VerificationResult.ok(attested_resource_id=attested_email)

    async def register(
        self, db: Session, trigger: AgentTrigger, config: Any
    ) -> RegistrationResult:
        status = await asyncio.to_thread(provision_gmail_trigger, db, trigger)
        return RegistrationResult(
            status=TriggerProvisioningStatus(status),
            resource_id=str(trigger.resource_id) if trigger.resource_id else None,
            error=trigger.provisioning_error,
        )

    async def unregister(
        self,
        db: Session,
        trigger: AgentTrigger,
        config: Any,
        *,
        resource_id: str | None = None,
    ) -> None:
        oauth_account_id = gmail_binding_id(config)
        if oauth_account_id is None:
            if not is_legacy_gmail_binding(config):
                return
            mailbox = _normalized_email(resource_id)
            if not mailbox:
                return
            matches = (
                db.query(UserOAuth)
                .filter(
                    UserOAuth.user_id == int(trigger.user_id),
                    ordinary_gmail_clause(),
                    func.lower(UserOAuth.email) == mailbox,
                )
                .order_by(UserOAuth.id)
                .limit(2)
                .all()
            )
            if len(matches) != 1:
                return
            oauth_account_id = int(matches[0].id)
        else:
            oauth_account = get_scoped_user_oauth_account(
                db,
                user_id=int(trigger.user_id),
                account_id=oauth_account_id,
                resource_owner_key=None,
            )
            if oauth_account is None or not is_ordinary_gmail(oauth_account):
                return

        await asyncio.to_thread(release_gmail_mailbox_if_unused, db, oauth_account_id)

    async def parse_events(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> list[NormalizedEvent]:
        state = _watch_state_for_callback(db, context.callback_id)
        if state is None:
            raise TriggerEventParseError("Gmail callback state was not found")

        attested_email = _normalized_email(state.email)
        notification = _decode_pubsub_notification(
            raw_body,
            attested_email=attested_email,
        )
        from ..gmail_triggers import build_gmail_service, collect_gmail_pubsub_events

        collection = await collect_gmail_pubsub_events(
            db,
            notification,
            state=state,
            service_factory=self.service_factory or build_gmail_service,
        )
        return [
            NormalizedEvent(
                event_type=event.event_type,
                source_event_id=event.source_event_id,
                target_trigger_id=event.trigger_id,
                resource_id=event.resource_id,
                payload=event.payload,
            )
            for event in collection.events
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
        _ = (trigger, events)
        state = _watch_state_for_callback(db, context.callback_id)
        if state is None:
            return
        if (
            str(state.status) == TriggerProvisioningStatus.FAILED.value
            and not get_gmail_watch_enabled()
            and str(state.last_error or "") == GMAIL_WATCH_DISABLED_ERROR
        ):
            # While registration is disabled, parse_events already acked this
            # callback (skipped, no history advance) instead of raising; the
            # pipeline still calls finalize_callback for the acked event. Do
            # not let a successful-looking finalize clear the FAILED/disabled
            # marking that collect_gmail_pubsub_events just recorded.
            #
            # The guard keys on the exact disabled marking, not just
            # status+flag: a row failed for a transient reason (e.g. a prior
            # message batch error) while the flag happens to be off must keep
            # advancing its cursor when a valid push later arrives, since the
            # only writers of this precise last_error string are the
            # choke-point gate (_ensure_gmail_mailbox_provisioned_locked) and
            # the webhook disabled-ack site (collect_gmail_pubsub_events).
            return
        notification = _decode_pubsub_notification(
            raw_body,
            attested_email=_normalized_email(state.email),
        )
        if not _history_cursor_advances(state.history_id, notification.history_id):
            return
        ordinary_account_exists = (
            db.query(UserOAuth.id)
            .filter(
                UserOAuth.id == GmailWatchState.oauth_account_id,
                UserOAuth.user_id == GmailWatchState.user_id,
                ordinary_gmail_clause(),
            )
            .exists()
        )
        updated = (
            db.query(GmailWatchState)
            .filter(
                GmailWatchState.id == int(state.id),
                GmailWatchState.oauth_account_id == int(state.oauth_account_id),
                GmailWatchState.user_id == int(state.user_id),
                ordinary_account_exists,
            )
            .update(
                {
                    GmailWatchState.history_id: notification.history_id,
                    GmailWatchState.last_error: None,
                },
                synchronize_session=False,
            )
        )
        if updated == 0:
            db.rollback()
            return
        db.commit()


register_trigger_provider(GmailProvider(), replace=True)
