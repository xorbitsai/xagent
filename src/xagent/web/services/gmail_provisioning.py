"""Per-mailbox Gmail Pub/Sub provisioning state machine.

Each connected Gmail mailbox that backs an enabled Gmail trigger gets its own
deterministic Pub/Sub topic and push subscription plus a Gmail watch. The
watch state row records an observable pending/active/failed status with a
clear last_error, converging through idempotent re-registration, periodic
sweeps, and reference-counted teardown.

Google credentials come from Application Default Credentials (ADC) or
GOOGLE_APPLICATION_CREDENTIALS; no xagent-specific credential store exists.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator

from sqlalchemy import String, and_
from sqlalchemy import cast as sql_cast
from sqlalchemy import func, insert, literal, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import (
    get_gmail_callback_base_url,
    get_gmail_pubsub_project_id,
    get_gmail_pubsub_push_service_account,
    get_gmail_pubsub_subscription_prefix,
    get_gmail_pubsub_topic_prefix,
    get_gmail_pubsub_transport,
    get_gmail_registration_timeout_seconds,
    get_gmail_watch_enabled,
)
from ..models.gmail_watch import GmailWatchState
from ..models.trigger import (
    AgentTrigger,
    TriggerProvisioningStatus,
    TriggerType,
)
from ..models.user_oauth import UserOAuth
from .gmail_triggers import gmail_binding_id, is_legacy_gmail_binding
from .time_utils import coerce_utc as _coerce_utc
from .user_oauth import (
    GMAIL_OAUTH_PROVIDER,
    get_scoped_user_oauth_account,
    get_user_oauth_account_by_id,
    is_ordinary_gmail,
    ordinary_gmail_clause,
    scoped_user_oauth_query,
)

logger = logging.getLogger(__name__)

GMAIL_PUSH_PUBLISHER = "gmail-api-push@system.gserviceaccount.com"
GMAIL_WATCH_LABEL_IDS = ["INBOX"]
# Pub/Sub can reuse an authenticated push token for up to one hour:
# https://docs.cloud.google.com/pubsub/docs/authenticate-push-subscriptions
# Retain five additional minutes for rollout propagation and ordinary clock
# skew so an old, still-valid audience is never rejected during rotation.
GMAIL_CALLBACK_AUDIENCE_GRACE_PERIOD = timedelta(hours=1, minutes=5)
GMAIL_WATCH_TRANSITION_LOCK_NAMESPACE = 0x58414754
_LOCAL_GMAIL_WATCH_TRANSITION_LOCK = threading.RLock()

PublisherFactory = Callable[[], Any]
SubscriberFactory = Callable[[], Any]

GMAIL_WATCH_DISABLED_ERROR = (
    "Gmail watch registration is disabled "
    "(set XAGENT_GMAIL_WATCH_ENABLED=true to enable)"
)
GMAIL_ACCOUNT_UNAVAILABLE_ERROR = "Gmail account is unavailable"
GMAIL_INVALID_OAUTH_ACCOUNT_BINDING_ERROR = (
    "Gmail trigger has an invalid OAuth account binding"
)
GMAIL_LEGACY_MAILBOX_BINDING_ERROR = "Gmail trigger has no legacy mailbox binding"


def gmail_watch_disabled_error() -> str:
    """Trigger-facing disabled message, extended with the project-id
    prerequisite when that is also missing (default deployments would
    otherwise discover the two requirements one error at a time)."""
    if get_gmail_pubsub_project_id():
        return GMAIL_WATCH_DISABLED_ERROR
    return (
        "Gmail watch registration is disabled "
        "(set XAGENT_GMAIL_WATCH_ENABLED=true and "
        "XAGENT_GMAIL_PUBSUB_PROJECT_ID to enable)"
    )


class GmailProvisioningError(RuntimeError):
    """Raised when per-mailbox Gmail provisioning cannot proceed."""


@contextmanager
def _gmail_watch_transition_lock(
    db: Session,
    oauth_account_id: int,
) -> Iterator[Session]:
    """Serialize one mailbox's cloud-plus-database provisioning transition.

    PostgreSQL deployments use a session advisory lock keyed by OAuth account,
    and bind the transition Session to that same physical connection. Commits
    can therefore expose intermediate state without releasing the advisory
    lock or checking out a second pooled connection during remote API calls.
    SQLite deployments use the caller Session under an in-process lock; SQLite
    already serializes database writers, and XAgent's background provisioning
    and reconciliation workers share one process.
    """
    bind = db.get_bind()
    engine = bind.engine
    if engine.dialect.name != "postgresql":
        with _LOCAL_GMAIL_WATCH_TRANSITION_LOCK:
            yield db
        return

    parameters = {
        "namespace": GMAIL_WATCH_TRANSITION_LOCK_NAMESPACE,
        "oauth_account_id": int(oauth_account_id),
    }
    # The caller's read transaction may already own a pooled connection.
    # Commit it before acquiring the lock connection; the transition Session
    # below performs all database work on that one lock-owning connection.
    db.commit()
    with engine.connect() as lock_connection:
        lock_connection.execute(
            text("SELECT pg_advisory_lock(:namespace, :oauth_account_id)"),
            parameters,
        )
        lock_connection.commit()
        try:
            with Session(bind=lock_connection, expire_on_commit=False) as transition_db:
                yield transition_db
        finally:
            if lock_connection.in_transaction():
                lock_connection.rollback()
            lock_connection.execute(
                text("SELECT pg_advisory_unlock(:namespace, :oauth_account_id)"),
                parameters,
            )
            lock_connection.commit()


@dataclass(frozen=True)
class GmailPushEndpointReconciliation:
    """Outcome of auditing or applying regional Gmail push endpoints."""

    scanned: int
    changed: int
    unchanged: int
    failed: int
    skipped: int = 0
    errors: tuple[str, ...] = ()


def _default_publisher() -> Any:
    if get_gmail_pubsub_transport() == "rest":
        from google.pubsub_v1 import PublisherClient

        return PublisherClient(transport="rest")
    from google.cloud import pubsub_v1  # type: ignore[import-untyped]

    return pubsub_v1.PublisherClient()


def _default_subscriber() -> Any:
    if get_gmail_pubsub_transport() == "rest":
        from google.pubsub_v1 import SubscriberClient

        return SubscriberClient(transport="rest")
    from google.cloud import pubsub_v1

    return pubsub_v1.SubscriberClient()


def _default_gmail_service(db: Session, oauth_account: UserOAuth) -> Any:
    from .gmail_triggers import build_gmail_service

    return build_gmail_service(db, oauth_account)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _trigger_facing_status(state: GmailWatchState) -> tuple[str, str | None]:
    """Derive the status a trigger should report from its mailbox watch state.

    Four derivations beyond a plain pass-through of the stored status and
    last_error:

    - An ``active`` row whose ``watch_expiration`` has passed is reported as
      failed: Gmail has already dropped the watch, so the row only looks
      healthy while nothing is delivered.
    - An ``active`` row with no recorded expiration is reported as failed too
      while ``XAGENT_GMAIL_WATCH_ENABLED`` is off: with the flag on the
      renewal scan treats a null expiration as due and renews it imminently,
      so it is only a problem when nothing will ever renew it.
    - With the flag off, a ``pending`` row is reported as failed — the sweep
      that would converge it is gated by the same flag, so it never
      resolves.
    - A row already stored as ``failed`` keeps its status, but while the flag
      is off its error is annotated to note that automatic retry is disabled
      too, distinguishing it from a failure the sweep would otherwise retry.

    Both ``provision_gmail_trigger`` and the reconcile paths derive through
    this one helper so the reported status cannot flap between them.
    """
    status = str(state.status or TriggerProvisioningStatus.PENDING.value)
    last_error = getattr(state, "last_error", None)
    error = str(last_error) if last_error else None
    watch_enabled = get_gmail_watch_enabled()
    if status == TriggerProvisioningStatus.ACTIVE.value:
        expiration = _coerce_utc(getattr(state, "watch_expiration", None))
        if expiration is not None and expiration <= _now():
            status = TriggerProvisioningStatus.FAILED.value
            error = (
                f"Gmail watch expired at {expiration.isoformat()} and was not renewed"
            )
        elif expiration is None and not watch_enabled:
            status = TriggerProvisioningStatus.FAILED.value
            error = (
                "Gmail watch has no recorded expiration and cannot be renewed "
                "while watch registration is disabled "
                "(set XAGENT_GMAIL_WATCH_ENABLED=true to enable)"
            )
    elif status == TriggerProvisioningStatus.PENDING.value and not watch_enabled:
        status = TriggerProvisioningStatus.FAILED.value
        error = error or gmail_watch_disabled_error()
    elif status == TriggerProvisioningStatus.FAILED.value and not watch_enabled:
        if not error:
            error = gmail_watch_disabled_error()
        elif "XAGENT_GMAIL_WATCH_ENABLED" not in error:
            error = (
                f"{error} (automatic retry is disabled; "
                "set XAGENT_GMAIL_WATCH_ENABLED=true to enable)"
            )
    return status, error


def mailbox_slug(email: str) -> str:
    """Deterministic, resource-name-safe identifier for one mailbox."""
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:16]


def gmail_topic_path(project_id: str, email: str) -> str:
    return f"projects/{project_id}/topics/{get_gmail_pubsub_topic_prefix()}-{mailbox_slug(email)}"


def gmail_subscription_path(project_id: str, email: str) -> str:
    return (
        f"projects/{project_id}/subscriptions/"
        f"{get_gmail_pubsub_subscription_prefix()}-{mailbox_slug(email)}"
    )


def _new_callback_id() -> str:
    return secrets.token_urlsafe(24)


def gmail_callback_url(base_url: str, callback_id: str) -> str:
    """Build the canonical Gmail push endpoint and OIDC audience.

    Pub/Sub's push endpoint, its signed OIDC audience, the persisted watch
    audience, and callback verification must use this exact value. Keeping the
    path construction here prevents those independently configured surfaces
    from drifting apart.
    """
    return f"{base_url}/api/triggers/callback/gmail/{callback_id}"


def _is_already_exists(exc: Exception) -> bool:
    try:
        # gRPC raises AlreadyExists; the REST transport maps HTTP 409 to its
        # parent class Conflict. Match exactly those two — Aborted is also a
        # Conflict subclass but signals a transient concurrency error, not
        # "already exists", and must propagate.
        from google.api_core.exceptions import AlreadyExists, Conflict

        return isinstance(exc, AlreadyExists) or type(exc) is Conflict
    except ImportError:  # pragma: no cover - google libs are a core dep
        return type(exc).__name__ in ("AlreadyExists", "Conflict")


def _is_not_found(exc: Exception) -> bool:
    try:
        from google.api_core.exceptions import NotFound

        return isinstance(exc, NotFound)
    except ImportError:  # pragma: no cover
        return type(exc).__name__ == "NotFound"


def _validate_provisioning_config() -> tuple[str, str, str]:
    """Return (project_id, callback_base_url, push_service_account) or raise."""
    project_id = get_gmail_pubsub_project_id()
    if not project_id:
        raise GmailProvisioningError(
            "XAGENT_GMAIL_PUBSUB_PROJECT_ID is required for Gmail provisioning"
        )
    base_url = get_gmail_callback_base_url()
    if not base_url:
        raise GmailProvisioningError(
            "XAGENT_S2S_API_BASE_URL, XAGENT_TRIGGER_CALLBACK_BASE_URL, or "
            "XAGENT_PUBLIC_API_BASE_URL is required for Gmail push registration"
        )
    push_service_account = get_gmail_pubsub_push_service_account()
    if not push_service_account:
        raise GmailProvisioningError(
            "XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT is required for "
            "OIDC-verified Gmail push delivery"
        )
    return project_id, base_url, push_service_account


def _referenced_gmail_oauth_account_ids(
    db: Session,
    accounts: Sequence[tuple[int, int, str]],
) -> set[int]:
    """Resolve enabled Gmail trigger bindings for a bounded owner-account batch.

    Modern triggers bind to ``config.oauth_account_id``. Only an absent
    binding key uses the legacy mailbox fallback. Both forms must belong to
    the same user as the candidate OAuth account.
    """
    account_users = {
        int(account_id): int(user_id) for account_id, user_id, _email in accounts
    }
    account_ids_by_user_email: dict[tuple[int, str], set[int]] = {}
    for account_id, user_id, raw_email in accounts:
        email = str(raw_email or "").strip().lower()
        if email:
            account_ids_by_user_email.setdefault((int(user_id), email), set()).add(
                int(account_id)
            )
    if not account_users:
        return set()

    binding_text = sql_cast(
        AgentTrigger.config["oauth_account_id"].as_string(),
        String,
    )
    # Normalize accepted decimal strings without casting malformed JSON values.
    normalized_binding_text = func.coalesce(
        func.nullif(func.ltrim(binding_text, "0"), ""),
        "0",
    )
    # Do not add DISTINCT. PostgreSQL cannot compare the projected JSON config.
    candidate_rows = (
        db.query(
            AgentTrigger.user_id,
            AgentTrigger.config,
            AgentTrigger.resource_id,
        )
        .filter(
            AgentTrigger.type == TriggerType.GMAIL.value,
            AgentTrigger.enabled.is_(True),
            or_(
                normalized_binding_text.in_(
                    {str(account_id) for account_id in account_users}
                ),
                func.lower(AgentTrigger.resource_id).in_(
                    {email for _user_id, email in account_ids_by_user_email}
                ),
            ),
        )
        .all()
    )

    referenced: set[int] = set()
    for trigger_user_id, raw_config, raw_resource_id in candidate_rows:
        user_id = int(trigger_user_id)
        bound_account_id = gmail_binding_id(raw_config)
        if (
            bound_account_id is not None
            and account_users.get(bound_account_id) == user_id
        ):
            referenced.add(bound_account_id)
            continue
        if bound_account_id is not None or not is_legacy_gmail_binding(raw_config):
            continue
        resource_id = str(raw_resource_id or "").strip().lower()
        referenced.update(account_ids_by_user_email.get((user_id, resource_id), ()))
    return referenced


def _fail_invalid_release_bindings(
    db: Session,
    *,
    user_id: int,
    email: str,
) -> bool:
    """Fail closed before teardown when a persisted binding needs repair."""
    if not email:
        return False

    invalid_triggers = [
        trigger
        for trigger in (
            db.query(AgentTrigger)
            .filter(
                AgentTrigger.type == TriggerType.GMAIL.value,
                AgentTrigger.enabled.is_(True),
                AgentTrigger.user_id == user_id,
                func.lower(AgentTrigger.resource_id) == email,
            )
            .all()
        )
        if (
            gmail_binding_id(trigger.config) is None
            and not is_legacy_gmail_binding(trigger.config)
        )
    ]
    if not invalid_triggers:
        return False

    for trigger in invalid_triggers:
        setattr(trigger, "provisioning_status", TriggerProvisioningStatus.FAILED.value)
        setattr(
            trigger,
            "provisioning_error",
            GMAIL_INVALID_OAUTH_ACCOUNT_BINDING_ERROR,
        )
        db.add(trigger)
    db.commit()
    return True


def _get_or_create_watch_state(
    db: Session, oauth_account: UserOAuth, email: str
) -> GmailWatchState:
    # Probe without a row lock so concurrent first-time creators can still
    # race on the unique constraint. Existing rows use OAuth-before-watch
    # ordering, matching release and credential deletion.
    account_id = int(oauth_account.id)
    user_id = int(oauth_account.user_id)

    def current_state() -> GmailWatchState | None:
        return (
            db.query(GmailWatchState)
            .filter(GmailWatchState.oauth_account_id == account_id)
            .first()
        )

    def locked_state() -> GmailWatchState | None:
        return (
            db.query(GmailWatchState)
            .filter(GmailWatchState.oauth_account_id == account_id)
            .with_for_update()
            .first()
        )

    def locked_ordinary_account() -> UserOAuth | None:
        return (
            db.query(UserOAuth)
            .filter(
                UserOAuth.id == account_id,
                UserOAuth.user_id == user_id,
                ordinary_gmail_clause(),
            )
            .with_for_update()
            .one_or_none()
        )

    def ordinary_account_exists() -> Any:
        # Lock this exact owner only while guarded PENDING DML is in progress.
        return (
            db.query(UserOAuth.id)
            .filter(
                UserOAuth.id == account_id,
                UserOAuth.user_id == user_id,
                ordinary_gmail_clause(),
            )
            .with_for_update()
            .exists()
        )

    def pending_values(state: GmailWatchState) -> dict[Any, Any]:
        values: dict[Any, Any] = {
            GmailWatchState.email: email,
            GmailWatchState.status: TriggerProvisioningStatus.PENDING.value,
        }
        if not state.callback_id:
            values[GmailWatchState.callback_id] = _new_callback_id()
        return values

    def persist_pending(state: GmailWatchState) -> None:
        updated = (
            db.query(GmailWatchState)
            .filter(
                GmailWatchState.id == int(state.id),
                GmailWatchState.oauth_account_id == account_id,
                GmailWatchState.user_id == user_id,
                ordinary_account_exists(),
            )
            .update(pending_values(state), synchronize_session=False)
        )
        if updated == 0:
            db.rollback()
            raise GmailProvisioningError("ordinary Gmail account not found")
        db.commit()

    state = current_state()
    if state is not None:
        if int(state.user_id) != user_id:
            raise GmailProvisioningError(
                "Gmail watch and OAuth account users do not match"
            )
        if locked_ordinary_account() is None:
            db.rollback()
            raise GmailProvisioningError("ordinary Gmail account not found")
        state = locked_state()
        if state is not None:
            if int(state.user_id) != user_id:
                raise GmailProvisioningError(
                    "Gmail watch and OAuth account users do not match"
                )
            persist_pending(state)
            db.refresh(state)
            return state

    callback_id = _new_callback_id()
    try:
        inserted = db.execute(
            insert(GmailWatchState).from_select(
                [
                    GmailWatchState.user_id,
                    GmailWatchState.oauth_account_id,
                    GmailWatchState.email,
                    GmailWatchState.history_id,
                    GmailWatchState.topic_name,
                    GmailWatchState.callback_id,
                    GmailWatchState.status,
                ],
                select(
                    literal(user_id),
                    literal(account_id),
                    literal(email),
                    literal(""),
                    literal(""),
                    literal(callback_id),
                    literal(TriggerProvisioningStatus.PENDING.value),
                ).where(ordinary_account_exists()),
            )
        )
        if int(getattr(inserted, "rowcount", 0) or 0) != 1:
            db.rollback()
            raise GmailProvisioningError("ordinary Gmail account not found")
        db.commit()
    except IntegrityError:
        # FOR UPDATE takes no lock when the row does not exist yet, so two
        # concurrent first-time enables for the same account can both reach
        # the insert path; the loser trips the oauth_account_id unique
        # constraint and adopts the winner's row instead of erroring out.
        db.rollback()
        if locked_ordinary_account() is None:
            raise GmailProvisioningError("ordinary Gmail account not found")
        state = locked_state()
        if state is None:  # pragma: no cover - row deleted between retries
            raise
        if int(state.user_id) != user_id:
            raise GmailProvisioningError(
                "Gmail watch and OAuth account users do not match"
            )
        persist_pending(state)
    else:
        state = locked_state()
        if state is None:  # pragma: no cover - row deleted after insert
            raise GmailProvisioningError("Gmail watch state was deleted")

    db.refresh(state)
    return state


def _ensure_topic(publisher: Any, topic_path: str) -> None:
    """Create the topic and grant the Gmail push identity permission to publish.

    IAM failures are never swallowed. The caller marks the watch ``ACTIVE`` on
    return, so a swallowed ``get_iam_policy``/``set_iam_policy`` failure would
    record success on a topic Gmail may be unable to publish to — the trigger
    would silently never fire. Letting the failure propagate lands the watch in
    ``FAILED`` for the retry sweep instead, matching ``_sync_push_config``.
    Unlike ``_sync_push_config``, the read cannot fall through to an
    unconditional write:
    granting the role is a read-modify-write of the policy, so a failed read
    must propagate as well.
    """
    try:
        publisher.create_topic(request={"name": topic_path})
    except Exception as exc:
        if not _is_already_exists(exc):
            raise
    # Gmail publishes watch notifications as this Google-owned identity.
    policy = publisher.get_iam_policy(request={"resource": topic_path})
    member = f"serviceAccount:{GMAIL_PUSH_PUBLISHER}"
    for binding in policy.bindings:
        if binding.role == "roles/pubsub.publisher":
            if member in binding.members:
                return
            binding.members.append(member)
            break
    else:
        policy.bindings.add(role="roles/pubsub.publisher", members=[member])
    publisher.set_iam_policy(request={"resource": topic_path, "policy": policy})


def _ensure_push_subscription(
    subscriber: Any,
    *,
    subscription_path: str,
    topic_path: str,
    push_audience: str,
    push_service_account: str,
) -> None:
    push_config = _build_push_config(
        push_audience=push_audience,
        push_service_account=push_service_account,
    )
    try:
        subscriber.create_subscription(
            request={
                "name": subscription_path,
                "topic": topic_path,
                "push_config": push_config,
            }
        )
    except Exception as exc:
        if not _is_already_exists(exc):
            raise
        # The deterministic name survives config changes; make sure an
        # existing subscription still pushes to the current audience
        # (e.g. after XAGENT_S2S_API_BASE_URL was changed).
        _sync_push_config(
            subscriber,
            subscription_path=subscription_path,
            push_config=push_config,
        )


def _build_push_config(
    *, push_audience: str, push_service_account: str
) -> dict[str, Any]:
    """Build the endpoint and matching OIDC audience as one atomic contract."""
    return {
        "push_endpoint": push_audience,
        "oidc_token": {
            "service_account_email": push_service_account,
            "audience": push_audience,
        },
    }


def _sync_push_config(
    subscriber: Any,
    *,
    subscription_path: str,
    push_config: dict[str, Any],
) -> None:
    """Reconcile an existing subscription's push config with the desired one.

    Inspecting the live subscription is only an optimization to skip a
    redundant patch: on a successful read the full
    ``(push_endpoint, service_account_email, audience)`` tuple is compared
    against ``push_config`` and the patch is skipped when they already match.
    A read failure (e.g. a narrower IAM role without
    ``pubsub.subscriptions.get``, or a transient blip) is logged and falls
    through to an unconditional, idempotent ``modify_push_config`` rather than
    aborting.

    The patch itself is never swallowed. The caller persists the new audience
    and marks the watch ``ACTIVE`` on return, so a swallowed patch failure
    would record success on a subscription that was never updated. Letting it
    propagate lands the watch in ``FAILED`` for the retry sweep instead.
    """
    try:
        existing = subscriber.get_subscription(
            request={"subscription": subscription_path}
        )
    except Exception as exc:
        logger.warning(
            "Could not inspect existing subscription %s; re-applying push "
            "config unconditionally: %s",
            subscription_path,
            exc,
        )
    else:
        # Compare the full push config, not just the endpoint: the OIDC service
        # account can change (e.g. XAGENT_GMAIL_PUBSUB_PUSH_SERVICE_ACCOUNT)
        # while the endpoint/audience stay the same, and it still needs
        # re-syncing.
        if _push_config_matches(existing, push_config):
            return
    subscriber.modify_push_config(
        request={
            "subscription": subscription_path,
            "push_config": push_config,
        }
    )


def _push_config_matches(subscription: Any, expected: dict[str, Any]) -> bool:
    """Return whether Pub/Sub exposes the complete expected push contract."""
    current_push = getattr(subscription, "push_config", None)
    current_oidc = getattr(current_push, "oidc_token", None)
    expected_oidc = expected["oidc_token"]
    return bool(
        str(getattr(current_push, "push_endpoint", "") or "")
        == expected["push_endpoint"]
        and str(getattr(current_oidc, "audience", "") or "")
        == expected_oidc["audience"]
        and str(getattr(current_oidc, "service_account_email", "") or "")
        == expected_oidc["service_account_email"]
    )


def _reconcile_gmail_push_endpoint(
    db: Session,
    *,
    state_row: Any,
    base_url: str,
    push_service_account: str,
    subscriber: Any,
    execute: bool,
) -> str:
    """Reconcile one snapshotted watch and return changed/unchanged/skipped.

    Execute mode takes the same mailbox transition lock as provisioning, then
    re-reads and row-locks the state. The snapshot may have become stale while
    waiting for another worker; using the refreshed metadata prevents a cloud
    update based on an earlier provisioning generation.
    """
    (
        state_id,
        oauth_account_id,
        _raw_email,
        raw_callback_id,
        raw_subscription_name,
        raw_push_audience,
    ) = state_row
    transition = (
        _gmail_watch_transition_lock(db, int(oauth_account_id))
        if execute
        else nullcontext(db)
    )
    with transition as transition_db:
        if execute:
            current_row = (
                transition_db.query(
                    GmailWatchState.id,
                    GmailWatchState.oauth_account_id,
                    GmailWatchState.email,
                    GmailWatchState.callback_id,
                    GmailWatchState.subscription_name,
                    GmailWatchState.push_audience,
                    GmailWatchState.previous_push_audience,
                )
                .join(
                    UserOAuth,
                    and_(
                        UserOAuth.id == GmailWatchState.oauth_account_id,
                        UserOAuth.user_id == GmailWatchState.user_id,
                    ),
                )
                .filter(
                    GmailWatchState.id == int(state_id),
                    GmailWatchState.oauth_account_id == int(oauth_account_id),
                    GmailWatchState.status == TriggerProvisioningStatus.ACTIVE.value,
                    ordinary_gmail_clause(),
                )
                .with_for_update(of=GmailWatchState)
                .one_or_none()
            )
            if current_row is None:
                transition_db.rollback()
                return "skipped"
            (
                state_id,
                oauth_account_id,
                _raw_email,
                raw_callback_id,
                raw_subscription_name,
                raw_push_audience,
                raw_previous_push_audience,
            ) = current_row
        else:
            raw_previous_push_audience = None

        callback_id = str(raw_callback_id or "").strip()
        subscription_path = str(raw_subscription_name or "").strip()
        if not callback_id or not subscription_path:
            raise GmailProvisioningError(
                f"Active Gmail watch {state_id} has incomplete push metadata"
            )
        expected_audience = gmail_callback_url(base_url, callback_id)
        expected_push_config = _build_push_config(
            push_audience=expected_audience,
            push_service_account=push_service_account,
        )
        try:
            existing = subscriber.get_subscription(
                request={"subscription": subscription_path}
            )
        except Exception:
            # Audit mode cannot claim convergence without observing cloud
            # state. Execute mode can still honor a deliberately update-only
            # service identity because modify_push_config is idempotent.
            if not execute:
                raise
            cloud_is_current = False
        else:
            cloud_is_current = _push_config_matches(existing, expected_push_config)
        database_is_current = str(raw_push_audience or "") == expected_audience
        if cloud_is_current and database_is_current:
            if execute:
                transition_db.rollback()
            return "unchanged"

        if execute:
            if not cloud_is_current:
                subscriber.modify_push_config(
                    request={
                        "subscription": subscription_path,
                        "push_config": expected_push_config,
                    }
                )

            # Keep the stored cloud audience authoritative until Pub/Sub has
            # accepted the new endpoint. Callback verification also derives
            # the configured audience, so both sides remain valid if the
            # process stops after the cloud call but before this commit.
            transition_values: dict[Any, Any] = {
                GmailWatchState.push_audience: expected_audience,
            }
            grace_deadline = _now() + GMAIL_CALLBACK_AUDIENCE_GRACE_PERIOD
            previous_audience = str(raw_push_audience or "").strip()
            if not database_is_current:
                transition_values.update(
                    {
                        GmailWatchState.previous_push_audience: (
                            previous_audience or None
                        ),
                        GmailWatchState.previous_push_audience_expires_at: (
                            grace_deadline if previous_audience else None
                        ),
                    }
                )
            elif not cloud_is_current:
                # Older releases could persist the new audience before a
                # failed cloud patch. Start a fresh grace window only after a
                # later retry actually moves Pub/Sub.
                durable_previous = str(raw_previous_push_audience or "").strip()
                if durable_previous:
                    transition_values.update(
                        {
                            GmailWatchState.previous_push_audience: durable_previous,
                            GmailWatchState.previous_push_audience_expires_at: grace_deadline,
                        }
                    )

            # Besides releasing the row lock, this guarded update detects a
            # concurrent SQLite teardown after the cloud call (PostgreSQL's
            # FOR UPDATE lock prevents that race directly).
            ordinary_account_exists = (
                transition_db.query(UserOAuth.id)
                .filter(
                    UserOAuth.id == GmailWatchState.oauth_account_id,
                    UserOAuth.user_id == GmailWatchState.user_id,
                    ordinary_gmail_clause(),
                )
                .exists()
            )
            updated = (
                transition_db.query(GmailWatchState)
                .filter(
                    GmailWatchState.id == int(state_id),
                    GmailWatchState.oauth_account_id == int(oauth_account_id),
                    GmailWatchState.status == TriggerProvisioningStatus.ACTIVE.value,
                    ordinary_account_exists,
                )
                .update(
                    transition_values,
                    synchronize_session=False,
                )
            )
            if updated == 0:
                raise GmailProvisioningError(
                    f"Active Gmail watch {state_id} changed ownership or "
                    "disappeared during the callback audience transition"
                )
            transition_db.commit()
        return "changed"


def reconcile_gmail_push_endpoints(
    db: Session,
    *,
    execute: bool = False,
    subscriber_factory: SubscriberFactory | None = None,
    batch_size: int = 100,
) -> GmailPushEndpointReconciliation:
    """Audit or migrate active Gmail subscriptions to the configured S2S URL.

    Only active watch states still referenced by an enabled Gmail trigger are
    candidates. Applying a change first modifies the Pub/Sub push endpoint and
    OIDC audience, then persists the new and previous accepted audiences. The
    configured and stored audiences keep callbacks verifiable across that
    transition. Reconciliation deliberately does not call ``users.watch`` or
    alter the callback identifier, history cursor, watch expiration,
    provisioning status, or error state.

    The default is a dry run. Re-running after a partial failure is safe:
    already-converged rows are skipped and each successful cloud update is
    committed independently.
    """
    _, base_url, push_service_account = _validate_provisioning_config()
    subscriber_factory = subscriber_factory or _default_subscriber
    subscriber: Any | None = None
    scanned = 0
    changed = 0
    unchanged = 0
    skipped = 0
    failed = 0
    errors: list[str] = []

    page_size = max(1, batch_size)
    last_state_id = 0
    while True:
        # Scalar rows remain valid across the per-watch commits below, unlike
        # ORM instances whose attributes are expired and implicitly refreshed.
        state_rows = (
            db.query(
                GmailWatchState.id,
                GmailWatchState.oauth_account_id,
                GmailWatchState.user_id,
                GmailWatchState.email,
                GmailWatchState.callback_id,
                GmailWatchState.subscription_name,
                GmailWatchState.push_audience,
            )
            .join(
                UserOAuth,
                and_(
                    UserOAuth.id == GmailWatchState.oauth_account_id,
                    UserOAuth.user_id == GmailWatchState.user_id,
                ),
            )
            .filter(
                GmailWatchState.status == TriggerProvisioningStatus.ACTIVE.value,
                GmailWatchState.id > last_state_id,
                ordinary_gmail_clause(),
            )
            .order_by(GmailWatchState.id.asc())
            .limit(page_size)
            .all()
        )
        if not state_rows:
            break
        last_state_id = int(state_rows[-1].id)
        referenced_account_ids = _referenced_gmail_oauth_account_ids(
            db,
            [
                (
                    int(row.oauth_account_id),
                    int(row.user_id),
                    str(row.email or ""),
                )
                for row in state_rows
            ],
        )

        for (
            state_id,
            oauth_account_id,
            _state_user_id,
            raw_email,
            raw_callback_id,
            raw_subscription_name,
            raw_push_audience,
        ) in state_rows:
            email = str(raw_email or "").strip().lower()
            if int(oauth_account_id) not in referenced_account_ids:
                continue

            scanned += 1
            try:
                if subscriber is None:
                    subscriber = subscriber_factory()
                outcome = _reconcile_gmail_push_endpoint(
                    db,
                    state_row=(
                        state_id,
                        oauth_account_id,
                        raw_email,
                        raw_callback_id,
                        raw_subscription_name,
                        raw_push_audience,
                    ),
                    base_url=base_url,
                    push_service_account=push_service_account,
                    subscriber=subscriber,
                    execute=execute,
                )
                if outcome == "unchanged":
                    unchanged += 1
                    continue
                if outcome == "skipped":
                    skipped += 1
                    continue
                changed += 1
            except Exception as exc:
                db.rollback()
                failed += 1
                errors.append(f"watch {state_id} ({email}): {exc}")

    return GmailPushEndpointReconciliation(
        scanned=scanned,
        changed=changed,
        unchanged=unchanged,
        failed=failed,
        skipped=skipped,
        errors=tuple(errors),
    )


def _register_gmail_watch(service: Any, topic_path: str) -> tuple[str, datetime | None]:
    response = (
        service.users()
        .watch(
            userId="me",
            body={"topicName": topic_path, "labelIds": GMAIL_WATCH_LABEL_IDS},
        )
        .execute()
    )
    history_id = response.get("historyId")
    if history_id is None:
        raise GmailProvisioningError("Gmail watch response did not include historyId")
    expiration = response.get("expiration")
    watch_expiration: datetime | None = None
    if expiration not in (None, ""):
        try:
            watch_expiration = datetime.fromtimestamp(
                int(str(expiration)) / 1000, tz=timezone.utc
            )
        except (TypeError, ValueError, OSError):
            watch_expiration = None
    return str(history_id), watch_expiration


def ensure_gmail_mailbox_provisioned(
    db: Session,
    oauth_account: UserOAuth,
    *,
    service_factory: Callable[[Session, UserOAuth], Any] | None = None,
    publisher_factory: PublisherFactory | None = None,
    subscriber_factory: SubscriberFactory | None = None,
) -> GmailWatchState:
    """Provision one ordinary Gmail mailbox without crossing owner namespaces.

    Provider and ownership errors stop before a watch row or remote resource is
    created. Other provisioning failures retain the existing failed-state retry
    behavior.
    """
    if not is_ordinary_gmail(oauth_account):
        raise GmailProvisioningError(
            "Gmail watch provisioning requires an ordinary Gmail account"
        )

    oauth_account_id = int(oauth_account.id)
    oauth_account_user_id = int(oauth_account.user_id)
    with _gmail_watch_transition_lock(db, oauth_account_id) as transition_db:
        transition_account = (
            oauth_account
            if transition_db is db
            else get_scoped_user_oauth_account(
                transition_db,
                user_id=oauth_account_user_id,
                account_id=oauth_account_id,
                resource_owner_key=None,
            )
        )
        if transition_account is None or not is_ordinary_gmail(transition_account):
            raise GmailProvisioningError("ordinary Gmail account not found")
        state = _ensure_gmail_mailbox_provisioned_locked(
            transition_db,
            transition_account,
            service_factory=service_factory,
            publisher_factory=publisher_factory,
            subscriber_factory=subscriber_factory,
        )
    if transition_db is not db:
        db.expire_all()
    return state


def _ensure_gmail_mailbox_provisioned_locked(
    db: Session,
    oauth_account: UserOAuth,
    *,
    service_factory: Callable[[Session, UserOAuth], Any] | None = None,
    publisher_factory: PublisherFactory | None = None,
    subscriber_factory: SubscriberFactory | None = None,
) -> GmailWatchState:
    """Provision one mailbox while its cross-worker transition lock is held."""
    service_factory = service_factory or _default_gmail_service
    publisher_factory = publisher_factory or _default_publisher
    subscriber_factory = subscriber_factory or _default_subscriber
    email = str(oauth_account.email or "").strip().lower()
    if not email:
        raise GmailProvisioningError("Gmail account email is required")

    if not get_gmail_watch_enabled():
        # Defense in depth: every production caller of this function already
        # checks the flag before reaching here (provision_gmail_trigger,
        # sweep_gmail_provisioning, best_effort_provision_gmail_watches_for_user,
        # _renew_watch_for_account), so this only fires for a future ungated
        # caller. Keep the failure state inside the ordinary owner fence.
        state = _get_or_create_watch_state(db, oauth_account, email)
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
            .filter(GmailWatchState.id == int(state.id), ordinary_account_exists)
            .update(
                {
                    GmailWatchState.status: TriggerProvisioningStatus.FAILED.value,
                    GmailWatchState.last_error: GMAIL_WATCH_DISABLED_ERROR,
                },
                synchronize_session=False,
            )
        )
        if updated == 0:
            db.rollback()
            raise GmailProvisioningError("ordinary Gmail ownership changed")
        db.commit()
        db.refresh(state)
        return state

    state = _get_or_create_watch_state(db, oauth_account, email)
    state_id = int(state.id)
    try:
        project_id, base_url, push_service_account = _validate_provisioning_config()
        topic_path = gmail_topic_path(project_id, email)
        subscription_path = gmail_subscription_path(project_id, email)
        push_audience = gmail_callback_url(base_url, str(state.callback_id))

        previous_audience = str(state.push_audience or "").strip()

        publisher = publisher_factory()
        _ensure_topic(publisher, topic_path)
        _ensure_push_subscription(
            subscriber_factory(),
            subscription_path=subscription_path,
            topic_path=topic_path,
            push_audience=push_audience,
            push_service_account=push_service_account,
        )
        service = service_factory(db, oauth_account)
        history_id, watch_expiration = _register_gmail_watch(service, topic_path)
    except Exception as exc:
        db.rollback()
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
            .filter(GmailWatchState.id == state_id, ordinary_account_exists)
            .update(
                {
                    GmailWatchState.status: TriggerProvisioningStatus.FAILED.value,
                    GmailWatchState.last_error: str(exc),
                },
                synchronize_session=False,
            )
        )
        if updated == 0:
            db.rollback()
            raise GmailProvisioningError("ordinary Gmail ownership changed") from exc
        db.commit()
        state = db.query(GmailWatchState).filter(GmailWatchState.id == state_id).one()
        logger.warning("Gmail provisioning failed for %s: %s", email, exc)
        return state

    transition_values: dict[Any, Any] = {
        GmailWatchState.topic_name: topic_path,
        GmailWatchState.subscription_name: subscription_path,
        GmailWatchState.history_id: history_id,
        GmailWatchState.watch_expiration: watch_expiration,
        GmailWatchState.status: TriggerProvisioningStatus.ACTIVE.value,
        GmailWatchState.last_error: None,
    }
    if previous_audience != push_audience:
        transition_values[GmailWatchState.push_audience] = push_audience
        if previous_audience:
            transition_values.update(
                {
                    GmailWatchState.previous_push_audience: previous_audience,
                    GmailWatchState.previous_push_audience_expires_at: (
                        _now() + GMAIL_CALLBACK_AUDIENCE_GRACE_PERIOD
                    ),
                }
            )

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
            GmailWatchState.id == state_id,
            GmailWatchState.oauth_account_id == oauth_account.id,
            GmailWatchState.user_id == oauth_account.user_id,
            ordinary_account_exists,
        )
        .update(transition_values, synchronize_session=False)
    )
    if updated == 0:
        db.rollback()
        raise GmailProvisioningError(
            "ordinary Gmail ownership changed during provisioning"
        )
    db.commit()
    db.refresh(state)
    return state


def _provision_in_fresh_session(oauth_account_id: int) -> None:
    from ..models.database import get_session_local

    db = get_session_local()()
    try:
        oauth_account = get_user_oauth_account_by_id(
            db,
            account_id=oauth_account_id,
            resource_owner_key=None,
        )
        if oauth_account is None or not is_ordinary_gmail(oauth_account):
            logger.warning(
                "Cannot provision Gmail watch without ordinary account %s",
                oauth_account_id,
            )
            return
        ensure_gmail_mailbox_provisioned(db, oauth_account)
    except Exception:
        logger.exception(
            "Background Gmail provisioning failed for account %s", oauth_account_id
        )
    finally:
        db.close()


def _provisioning_account(
    db: Session,
    trigger: AgentTrigger,
) -> UserOAuth | None:
    account_id = gmail_binding_id(trigger.config)
    if account_id is not None:
        account = get_scoped_user_oauth_account(
            db,
            user_id=int(trigger.user_id),
            account_id=account_id,
            resource_owner_key=None,
        )
        return account if account is not None and is_ordinary_gmail(account) else None

    if not is_legacy_gmail_binding(trigger.config):
        return None

    mailbox = str(trigger.resource_id or "").strip().lower()
    if not mailbox:
        return None

    matches = (
        scoped_user_oauth_query(
            db,
            user_id=int(trigger.user_id),
            resource_owner_key=None,
        )
        .filter(
            UserOAuth.provider == GMAIL_OAUTH_PROVIDER,
            func.lower(UserOAuth.email) == mailbox,
        )
        .order_by(UserOAuth.id)
        .limit(2)
        .all()
    )
    return matches[0] if len(matches) == 1 else None


def provision_gmail_trigger(
    db: Session,
    trigger: AgentTrigger,
    *,
    timeout_seconds: int | None = None,
    run_in_thread: Callable[[int], threading.Thread] | None = None,
) -> str:
    """Provision the mailbox bound to a Gmail trigger; reflect status on it.

    Runs provisioning in a background thread and waits up to the configured
    registration timeout. When the cloud side is slow, the API observes a
    pending state while the thread converges to active or failed on its own.
    Returns the trigger provisioning status.

    With ``XAGENT_GMAIL_WATCH_ENABLED`` off no watch is registered: the
    renewal scan that keeps watches alive is gated by the same flag, so a
    watch created here would silently expire. The trigger reports failed
    with an explicit disabled error instead of a healthy-looking state,
    unless the mailbox still has a watch state row from when the flag was
    on, in which case its derived status is reported.
    """
    legacy_binding = is_legacy_gmail_binding(trigger.config)
    bound_account_id = gmail_binding_id(trigger.config)
    oauth_account = _provisioning_account(db, trigger)
    if oauth_account is None:
        status = TriggerProvisioningStatus.FAILED.value
        if legacy_binding:
            binding_error = (
                GMAIL_ACCOUNT_UNAVAILABLE_ERROR
                if str(trigger.resource_id or "").strip()
                else GMAIL_LEGACY_MAILBOX_BINDING_ERROR
            )
        elif bound_account_id is not None:
            binding_error = GMAIL_ACCOUNT_UNAVAILABLE_ERROR
        else:
            binding_error = GMAIL_INVALID_OAUTH_ACCOUNT_BINDING_ERROR
        setattr(trigger, "provisioning_status", status)
        setattr(trigger, "provisioning_error", binding_error)
        db.add(trigger)
        db.commit()
        return status

    oauth_account_id = int(oauth_account.id)

    if not get_gmail_watch_enabled():
        state = (
            db.query(GmailWatchState)
            .filter(
                GmailWatchState.oauth_account_id == int(oauth_account_id),
                GmailWatchState.user_id == int(trigger.user_id),
            )
            .first()
        )
        if state is None:
            status = TriggerProvisioningStatus.FAILED.value
            error: str | None = gmail_watch_disabled_error()
        else:
            status, error = _trigger_facing_status(state)
        setattr(trigger, "provisioning_status", status)
        setattr(trigger, "provisioning_error", error)
        db.add(trigger)
        db.commit()
        db.refresh(trigger)
        return status

    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else get_gmail_registration_timeout_seconds()
    )

    if run_in_thread is None:

        def run_in_thread(account_id: int) -> threading.Thread:
            thread = threading.Thread(
                target=_provision_in_fresh_session,
                args=(account_id,),
                daemon=True,
                name=f"gmail-provision-{account_id}",
            )
            thread.start()
            return thread

    thread = run_in_thread(int(oauth_account_id))
    thread.join(timeout)

    db.expire_all()
    oauth_account = get_scoped_user_oauth_account(
        db,
        user_id=int(trigger.user_id),
        account_id=int(oauth_account_id),
        resource_owner_key=None,
    )
    if oauth_account is None or not is_ordinary_gmail(oauth_account):
        status = TriggerProvisioningStatus.FAILED.value
        error = GMAIL_ACCOUNT_UNAVAILABLE_ERROR
    else:
        state = (
            db.query(GmailWatchState)
            .filter(
                GmailWatchState.oauth_account_id == int(oauth_account_id),
                GmailWatchState.user_id == int(trigger.user_id),
            )
            .first()
        )
        if thread.is_alive() or state is None:
            status = TriggerProvisioningStatus.PENDING.value
            error = None
        else:
            status, error = _trigger_facing_status(state)
    setattr(trigger, "provisioning_status", status)
    setattr(trigger, "provisioning_error", error)
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return status


def reconcile_gmail_trigger_provisioning(
    db: Session,
    triggers: Sequence[AgentTrigger] | None = None,
    *,
    batch_size: int = 100,
) -> int:
    """Refresh Gmail triggers' provisioning status from their watch states.

    The background provisioning thread and the periodic sweep converge
    GmailWatchState without touching AgentTrigger, so the status the API
    reports would otherwise stay frozen at whatever the original synchronous
    create/update call observed. Each batch joins its enabled Gmail triggers
    to their mailboxes' watch states with one IN() lookup and copies
    status/last_error over when they diverge. Returns the number of triggers
    updated.

    Without an explicit trigger set (the sweep path), candidates are walked
    in id-keyset pages of ``batch_size`` so every query stays bounded no
    matter how many Gmail triggers exist system-wide.
    """
    if triggers is not None:
        return _reconcile_gmail_trigger_batch(
            db,
            [
                trigger
                for trigger in triggers
                if str(trigger.type) == TriggerType.GMAIL.value
                and bool(trigger.enabled)
            ],
        )

    page_size = max(1, batch_size)
    updated = 0
    last_id = 0
    while True:
        page = (
            db.query(AgentTrigger)
            .filter(
                AgentTrigger.type == TriggerType.GMAIL.value,
                AgentTrigger.enabled.is_(True),
                AgentTrigger.id > last_id,
            )
            .order_by(AgentTrigger.id.asc())
            .limit(page_size)
            .all()
        )
        if not page:
            return updated
        last_id = int(page[-1].id)
        updated += _reconcile_gmail_trigger_batch(db, page)
        if len(page) < page_size:
            return updated


def _bound_gmail_oauth_account_id(trigger: AgentTrigger) -> int | None:
    """Read a valid explicit Gmail OAuth account id from a trigger."""
    return gmail_binding_id(trigger.config)


def _legacy_gmail_resource_id(trigger: AgentTrigger) -> str | None:
    """Return the mailbox key for an unbound legacy trigger."""
    resource_id = str(trigger.resource_id or "").strip().lower()
    return resource_id or None


def _reconcile_gmail_trigger_batch(
    db: Session, candidates: Sequence[AgentTrigger]
) -> int:
    """Copy diverged watch-state status onto one bounded candidate batch.

    Looks up each trigger's watch state by its bound OAuth account id
    (``config.oauth_account_id``) when one is present and valid, falling back
    to the legacy ``(user_id, resource_id-email)`` key only when the binding
    key is absent. This mirrors ``_referenced_gmail_oauth_account_ids``'s
    precedence: a trigger's
    ``resource_id`` mailbox email can go stale (the connected Google account
    changed email and a reconnect refreshed ``GmailWatchState.email``), and
    matching by the durable account id instead of the stale email avoids a
    spurious miss that would otherwise clobber an active trigger to
    failed/disabled once the flag-off None-state behavior below kicks in.
    """
    if not candidates:
        return 0

    account_ids: set[int] = set()
    emails: set[str] = set()
    for trigger in candidates:
        bound_account_id = _bound_gmail_oauth_account_id(trigger)
        if bound_account_id is not None:
            account_ids.add(bound_account_id)
            continue
        if is_legacy_gmail_binding(trigger.config):
            resource_id = _legacy_gmail_resource_id(trigger)
            if resource_id:
                emails.add(resource_id)

    filters = []
    if account_ids:
        filters.append(GmailWatchState.oauth_account_id.in_(account_ids))
    if emails:
        filters.append(func.lower(GmailWatchState.email).in_(emails))
    states = (
        db.query(GmailWatchState)
        .join(
            UserOAuth,
            and_(
                UserOAuth.id == GmailWatchState.oauth_account_id,
                UserOAuth.user_id == GmailWatchState.user_id,
            ),
        )
        .filter(
            ordinary_gmail_clause(),
            or_(*filters),
        )
        .all()
        if filters
        else []
    )

    states_by_account_id = {int(state.oauth_account_id): state for state in states}
    states_by_key = {
        (int(state.user_id), str(state.email or "").strip().lower()): state
        for state in states
    }

    unresolved_bound_account_ids: set[int] = set()
    for trigger in candidates:
        bound_account_id = _bound_gmail_oauth_account_id(trigger)
        if bound_account_id is None:
            continue
        state = states_by_account_id.get(bound_account_id)
        if state is None or int(state.user_id) != int(trigger.user_id):
            unresolved_bound_account_ids.add(bound_account_id)

    ordinary_accounts_by_key: dict[tuple[int, int], UserOAuth] = {}
    if unresolved_bound_account_ids:
        ordinary_accounts_by_key = {
            (int(account.user_id), int(account.id)): account
            for account in (
                db.query(UserOAuth)
                .filter(
                    UserOAuth.id.in_(unresolved_bound_account_ids),
                    ordinary_gmail_clause(),
                )
                .all()
            )
        }

    updated = 0
    for trigger in candidates:
        bound_account_id = _bound_gmail_oauth_account_id(trigger)
        legacy_binding = is_legacy_gmail_binding(trigger.config)
        legacy_resource_id = (
            _legacy_gmail_resource_id(trigger) if legacy_binding else None
        )
        error: str | None
        if bound_account_id is not None:
            state = states_by_account_id.get(bound_account_id)
        elif legacy_binding and legacy_resource_id:
            state = states_by_key.get((int(trigger.user_id), legacy_resource_id))
        elif legacy_binding:
            state = None
            status = TriggerProvisioningStatus.FAILED.value
            error = GMAIL_LEGACY_MAILBOX_BINDING_ERROR
        else:
            state = None
            status = TriggerProvisioningStatus.FAILED.value
            error = GMAIL_INVALID_OAUTH_ACCOUNT_BINDING_ERROR

        has_mismatched_watch_owner = state is not None and int(state.user_id) != int(
            trigger.user_id
        )
        if has_mismatched_watch_owner:
            state = None
        if state is None and bound_account_id is not None:
            oauth_account = ordinary_accounts_by_key.get(
                (int(trigger.user_id), bound_account_id)
            )
            if has_mismatched_watch_owner or oauth_account is None:
                status = TriggerProvisioningStatus.FAILED.value
                error = GMAIL_ACCOUNT_UNAVAILABLE_ERROR
            elif get_gmail_watch_enabled():
                continue
            else:
                status = TriggerProvisioningStatus.FAILED.value
                error = gmail_watch_disabled_error()
        elif state is None and legacy_binding:
            if legacy_resource_id is None:
                pass
            elif get_gmail_watch_enabled():
                continue
            else:
                status = TriggerProvisioningStatus.FAILED.value
                error = gmail_watch_disabled_error()
        elif state is not None:
            status, error = _trigger_facing_status(state)
        if (
            str(trigger.provisioning_status or "") == status
            and (trigger.provisioning_error or None) == error
        ):
            continue
        setattr(trigger, "provisioning_status", status)
        setattr(trigger, "provisioning_error", error)
        db.add(trigger)
        updated += 1
    if updated:
        db.commit()
    return updated


def release_gmail_mailbox_if_unused(
    db: Session,
    oauth_account_id: int,
    *,
    service_factory: Callable[[Session, UserOAuth], Any] | None = None,
    publisher_factory: PublisherFactory | None = None,
    subscriber_factory: SubscriberFactory | None = None,
) -> bool:
    """Reference-counted teardown of one mailbox's delivery resources.

    Locks the OAuth row before its watch state. This matches the FK-cascade
    order for credential deletion and prevents an OAuth/watch lock cycle.
    When no trigger references the mailbox, stops the Gmail watch and deletes
    the per-mailbox subscription, topic, and watch state. Returns True when
    resources were released.
    """
    service_factory = service_factory or _default_gmail_service
    publisher_factory = publisher_factory or _default_publisher
    subscriber_factory = subscriber_factory or _default_subscriber

    def ordinary_account() -> UserOAuth | None:
        return (
            db.query(UserOAuth)
            .filter(
                UserOAuth.id == int(oauth_account_id),
                ordinary_gmail_clause(),
            )
            .with_for_update()
            .one_or_none()
        )

    oauth_account = ordinary_account()
    if oauth_account is None:
        db.rollback()
        return False

    state = (
        db.query(GmailWatchState)
        .filter(GmailWatchState.oauth_account_id == int(oauth_account_id))
        .with_for_update()
        .first()
    )
    if state is None:
        db.commit()
        return False

    state_id = int(state.id)
    state_user_id = int(state.user_id)
    state_oauth_account_id = int(state.oauth_account_id)
    if state_user_id != int(oauth_account.user_id):
        logger.warning(
            "Skipping Gmail watch release for state %s because account %s is "
            "not owned by user %s",
            state_id,
            state_oauth_account_id,
            state_user_id,
        )
        db.rollback()
        return False

    email = str(state.email or "").strip().lower()
    if _fail_invalid_release_bindings(
        db,
        user_id=state_user_id,
        email=email,
    ):
        return False

    referenced_account_ids = _referenced_gmail_oauth_account_ids(
        db,
        [(state_oauth_account_id, state_user_id, email)],
    )
    if state_oauth_account_id in referenced_account_ids:
        db.commit()
        return False

    service = None
    try:
        service = service_factory(db, oauth_account)
    except Exception as exc:
        logger.warning("Failed to build Gmail service for %s: %s", email, exc)

    # Service construction can commit or roll back. Rebuild the complete lock
    # boundary and release decision before any remote teardown.
    oauth_account = ordinary_account()
    if oauth_account is None:
        db.rollback()
        return False
    state = (
        db.query(GmailWatchState)
        .filter(
            GmailWatchState.id == state_id,
            GmailWatchState.oauth_account_id == state_oauth_account_id,
            GmailWatchState.user_id == state_user_id,
        )
        .with_for_update()
        .one_or_none()
    )
    if state is None:
        db.rollback()
        return False

    email = str(state.email or "").strip().lower()
    if _fail_invalid_release_bindings(db, user_id=state_user_id, email=email):
        return False
    referenced_account_ids = _referenced_gmail_oauth_account_ids(
        db,
        [(state_oauth_account_id, state_user_id, email)],
    )
    if state_oauth_account_id in referenced_account_ids:
        db.commit()
        return False

    if service is not None:
        try:
            service.users().stop(userId="me").execute()
        except Exception as exc:
            logger.warning("Failed to stop Gmail watch for %s: %s", email, exc)

    # Do not delete Pub/Sub resources after the credential owner changes.
    if ordinary_account() is None:
        db.rollback()
        return False

    project_id = get_gmail_pubsub_project_id()
    if project_id:
        subscription_path = str(
            state.subscription_name or gmail_subscription_path(project_id, email)
        )
        topic_path = str(state.topic_name or gmail_topic_path(project_id, email))
        try:
            subscriber_factory().delete_subscription(
                request={"subscription": subscription_path}
            )
        except Exception as exc:
            if not _is_not_found(exc):
                logger.warning(
                    "Failed to delete subscription %s: %s", subscription_path, exc
                )
        # Recheck between the two remote deletes. An actor transition after
        # subscription deletion must not continue into topic deletion.
        if ordinary_account() is None:
            db.rollback()
            return False
        try:
            publisher_factory().delete_topic(request={"topic": topic_path})
        except Exception as exc:
            if not _is_not_found(exc):
                logger.warning("Failed to delete topic %s: %s", topic_path, exc)

    # Delete local state only while its current account remains ordinary.
    ordinary_account_exists = (
        db.query(UserOAuth.id)
        .filter(
            UserOAuth.id == GmailWatchState.oauth_account_id,
            UserOAuth.user_id == GmailWatchState.user_id,
            ordinary_gmail_clause(),
        )
        .exists()
    )
    deleted = (
        db.query(GmailWatchState)
        .filter(
            GmailWatchState.id == state_id,
            GmailWatchState.oauth_account_id == state_oauth_account_id,
            GmailWatchState.user_id == state_user_id,
            ordinary_account_exists,
        )
        .delete(synchronize_session="fetch")
    )
    if deleted == 0:
        db.rollback()
        return False
    db.commit()
    return True


def sweep_gmail_provisioning(
    db: Session,
    *,
    now: datetime | None = None,
    stale_pending_seconds: int = 300,
    limit: int = 100,
    service_factory: Callable[[Session, UserOAuth], Any] | None = None,
    publisher_factory: PublisherFactory | None = None,
    subscriber_factory: SubscriberFactory | None = None,
) -> int:
    """Retry stale pending and failed Gmail registrations.

    Only mailboxes still referenced by an enabled Gmail trigger are retried.
    Returns the number of registration attempts. Registration retries are
    gated on ``XAGENT_GMAIL_WATCH_ENABLED`` (like the renewal scan), so this
    function re-registers nothing while the feature is switched off.
    Trigger-status reconciliation runs either way, though: it does not touch
    the cloud, and skipping it while disabled would leave disabled/expired
    statuses frozen instead of observable through the trigger API.
    """
    if not get_gmail_watch_enabled():
        reconcile_gmail_trigger_provisioning(db, batch_size=max(1, min(limit, 500)))
        return 0

    scan_time = now or _now()
    stale_before = scan_time - timedelta(seconds=stale_pending_seconds)
    candidates = (
        db.query(GmailWatchState)
        .join(
            UserOAuth,
            and_(
                UserOAuth.id == GmailWatchState.oauth_account_id,
                UserOAuth.user_id == GmailWatchState.user_id,
            ),
        )
        .filter(
            ordinary_gmail_clause(),
            (GmailWatchState.status == TriggerProvisioningStatus.FAILED.value)
            | (
                (GmailWatchState.status == TriggerProvisioningStatus.PENDING.value)
                & (GmailWatchState.updated_at <= stale_before)
            ),
        )
        .order_by(GmailWatchState.updated_at.asc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
    referenced_account_ids = _referenced_gmail_oauth_account_ids(
        db,
        [
            (
                int(state.oauth_account_id),
                int(state.user_id),
                str(state.email or ""),
            )
            for state in candidates
        ],
    )

    attempts = 0
    for state in candidates:
        if int(state.oauth_account_id) not in referenced_account_ids:
            continue
        oauth_account = get_scoped_user_oauth_account(
            db,
            user_id=int(state.user_id),
            account_id=int(state.oauth_account_id),
            resource_owner_key=None,
        )
        if oauth_account is None or not is_ordinary_gmail(oauth_account):
            continue
        state_id = int(state.id)
        try:
            ensure_gmail_mailbox_provisioned(
                db,
                oauth_account,
                service_factory=service_factory,
                publisher_factory=publisher_factory,
                subscriber_factory=subscriber_factory,
            )
        except Exception as exc:
            db.rollback()
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
                .filter(GmailWatchState.id == state_id, ordinary_account_exists)
                .update(
                    {
                        GmailWatchState.status: TriggerProvisioningStatus.FAILED.value,
                        GmailWatchState.last_error: str(exc),
                    },
                    synchronize_session=False,
                )
            )
            if updated:
                db.commit()
            logger.error(
                "Gmail provisioning sweep failed for watch %s: %s",
                state_id,
                exc,
                exc_info=True,
            )
            continue
        attempts += 1
    # Watch states that converged in a background thread (pending -> active)
    # are not sweep candidates, so the trigger-facing status is reconciled
    # here unconditionally, paged by the sweep's own limit.
    reconcile_gmail_trigger_provisioning(db, batch_size=max(1, min(limit, 500)))
    return attempts


def best_effort_provision_gmail_watches_for_user(
    db: Session,
    *,
    user_id: int,
    context: str,
) -> None:
    """Provision watches for a user's Gmail accounts referenced by triggers.

    Used after OAuth (re)connects a Gmail account: any enabled Gmail trigger
    already bound to that mailbox gets its delivery resources re-ensured.
    Failures are recorded on the watch state, never raised; a failure before
    any account resolves has no watch state to land on and is only logged.

    Rolls back ``db`` on failure, so callers must not hold uncommitted work on
    that session across this call.

    Gated on ``XAGENT_GMAIL_WATCH_ENABLED``: with the flag off nothing renews
    a watch, so registering one here would leave it to expire silently.
    """
    if not get_gmail_watch_enabled():
        logger.debug(
            "Gmail watch registration is disabled; skipping provisioning "
            "for user %s %s",
            user_id,
            context,
        )
        return

    # The account lookup and the binding resolution are inside the guard, not
    # just the per-account loop: the caller runs this after committing the
    # OAuth token, so a raise here would turn an already-successful connect
    # into an error page.
    #
    # A mailbox missed here is recovered by ``scan_due_gmail_watch_renewals``,
    # which selects Gmail accounts that back an enabled Gmail trigger and have
    # no watch state row. (``sweep_gmail_provisioning`` never covers it: it
    # only retries mailboxes that already have such a row.) Both scans and
    # this function share the ``XAGENT_GMAIL_WATCH_ENABLED`` gate.
    try:
        accounts = (
            scoped_user_oauth_query(
                db,
                user_id=int(user_id),
                resource_owner_key=None,
            )
            .filter(UserOAuth.provider == GMAIL_OAUTH_PROVIDER)
            .all()
        )
        referenced_account_ids = _referenced_gmail_oauth_account_ids(
            db,
            [
                (
                    int(account.id),
                    int(account.user_id),
                    str(account.email or ""),
                )
                for account in accounts
            ],
        )
    except Exception as exc:
        db.rollback()
        logger.warning(
            "Failed to resolve Gmail accounts for user %s %s: %s",
            user_id,
            context,
            exc,
            exc_info=True,
        )
        return

    for account in accounts:
        email = str(account.email or "").strip().lower()
        if not email:
            continue
        if int(account.id) not in referenced_account_ids:
            continue
        try:
            ensure_gmail_mailbox_provisioned(db, account)
        except Exception as exc:
            db.rollback()
            logger.warning(
                "Failed to provision Gmail watch for %s %s: %s",
                email,
                context,
                exc,
                exc_info=True,
            )
