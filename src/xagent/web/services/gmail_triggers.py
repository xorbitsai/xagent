from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, cast
from urllib.parse import quote

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import (
    AuthorizedSession,
    Request,
)
from google.oauth2.credentials import Credentials
from sqlalchemy import case, or_
from sqlalchemy.orm import Session

from ...config import (
    get_gmail_watch_enabled,
    get_gmail_watch_renewal_lead_seconds,
)
from ...core.utils.encryption import decrypt_value
from ..models.gmail_watch import GmailWatchState
from ..models.oauth_provider import OAuthProvider
from ..models.trigger import AgentTrigger, TriggerProvisioningStatus, TriggerType
from ..models.user_oauth import UserOAuth
from .time_utils import coerce_utc as _coerce_utc
from .user_oauth import get_scoped_user_oauth_account, user_oauth_owner_clause

logger = logging.getLogger(__name__)

GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
GMAIL_API_ROOT = "https://gmail.googleapis.com/gmail/v1"
DEFAULT_GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

GmailServiceFactory = Callable[[Session, UserOAuth], Any]


class GmailTriggerError(RuntimeError):
    """Base error for Gmail trigger integration failures."""


class GmailWatchConfigurationError(GmailTriggerError):
    """Raised when Gmail watch cannot be configured for the deployment."""


class GmailWatchDisabledError(GmailWatchConfigurationError):
    """Raised when watch (re-)registration is blocked by XAGENT_GMAIL_WATCH_ENABLED."""


class _GmailApiRequest:
    def __init__(
        self,
        session: Any,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> None:
        self._session = session
        self._method = method
        self._url = url
        self._kwargs = kwargs

    def execute(self) -> dict[str, Any]:
        response = self._session.request(
            self._method,
            self._url,
            timeout=10,
            **self._kwargs,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        payload = response.json()
        return payload if isinstance(payload, dict) else {}


class _GmailMessagesResource:
    def __init__(self, session: Any, user_id: str) -> None:
        self._session = session
        self._user_id = quote(user_id, safe="")

    def get(self, **kwargs: Any) -> _GmailApiRequest:
        message_id = quote(str(kwargs.pop("id")), safe="")
        return _GmailApiRequest(
            self._session,
            "GET",
            f"{GMAIL_API_ROOT}/users/{self._user_id}/messages/{message_id}",
            params=kwargs,
        )


class _GmailHistoryResource:
    def __init__(self, session: Any, user_id: str) -> None:
        self._session = session
        self._user_id = quote(user_id, safe="")

    def list(self, **kwargs: Any) -> _GmailApiRequest:
        return _GmailApiRequest(
            self._session,
            "GET",
            f"{GMAIL_API_ROOT}/users/{self._user_id}/history",
            params=kwargs,
        )


class _GmailUsersResource:
    def __init__(self, session: Any) -> None:
        self._session = session
        self._user_id = "me"

    def watch(self, *, userId: str, body: dict[str, Any]) -> _GmailApiRequest:
        user_id = quote(userId, safe="")
        return _GmailApiRequest(
            self._session,
            "POST",
            f"{GMAIL_API_ROOT}/users/{user_id}/watch",
            json=body,
        )

    def history(self) -> _GmailHistoryResource:
        return _GmailHistoryResource(self._session, self._user_id)

    def messages(self) -> _GmailMessagesResource:
        return _GmailMessagesResource(self._session, self._user_id)


class _GmailApiService:
    def __init__(self, session: Any) -> None:
        self._session = session

    def users(self) -> _GmailUsersResource:
        return _GmailUsersResource(self._session)


@dataclass(frozen=True)
class GmailPubsubNotification:
    email_address: str
    history_id: str
    pubsub_message_id: str | None = None


@dataclass(frozen=True)
class GmailCollectedEvent:
    trigger_id: int
    payload: dict[str, Any]
    source_event_id: str
    event_type: str
    resource_id: str


@dataclass(frozen=True)
class GmailPubsubEventCollection:
    events: list[GmailCollectedEvent]
    skipped: int = 0


def _get_google_oauth_config(db: Session) -> tuple[str | None, str | None]:
    env_client_id = os.environ.get("GOOGLE_CLIENT_ID")
    env_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    provider = (
        db.query(OAuthProvider).filter(OAuthProvider.provider_name == "google").first()
    )
    if not provider:
        return (env_client_id, env_client_secret)

    client_id = decrypt_value(str(provider.client_id))
    client_secret = decrypt_value(str(provider.client_secret))
    return client_id or env_client_id, client_secret or env_client_secret


def _gmail_oauth_scopes(oauth_account: UserOAuth) -> list[str]:
    scopes = [
        scope for scope in str(oauth_account.scope or "").split(" ") if scope.strip()
    ]
    return scopes or DEFAULT_GMAIL_SCOPES


def _credentials_expiry(value: datetime | None) -> datetime | None:
    """google-auth requires Credentials.expiry to be a naive UTC datetime.

    Timezone-aware databases (PostgreSQL) return aware datetimes for
    UserOAuth.expires_at; passing one through makes creds.expired raise
    "can't compare offset-naive and offset-aware datetimes".
    """
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def build_gmail_service(db: Session, oauth_account: UserOAuth) -> Any:
    """Build Gmail trigger access for one ordinary connected account."""
    if oauth_account.resource_owner_key is not None:
        raise GmailWatchConfigurationError(
            "actor-owned OAuth credentials cannot back Gmail triggers"
        )
    client_id, client_secret = _get_google_oauth_config(db)
    if not client_id or not client_secret:
        raise GmailWatchConfigurationError("Google OAuth configuration missing")

    creds = Credentials(
        token=str(oauth_account.access_token),
        refresh_token=oauth_account.refresh_token,
        token_uri=GOOGLE_TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=_gmail_oauth_scopes(oauth_account),
        # cast: legacy Column[...] typing on UserOAuth; runtime value is the
        # datetime (or None) loaded from the row.
        expiry=_credentials_expiry(cast("datetime | None", oauth_account.expires_at)),
    )
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            db.rollback()
            raise GmailWatchConfigurationError(
                "Gmail credential refresh failed"
            ) from exc
        setattr(oauth_account, "access_token", creds.token)
        if creds.expiry:
            # google-auth hands back naive UTC; store it timezone-aware.
            setattr(oauth_account, "expires_at", _coerce_utc(creds.expiry))
        db.commit()

    return _GmailApiService(AuthorizedSession(creds))


def _exception_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        resp = getattr(exc, "resp", None)
        status_code = getattr(resp, "status", None)
    if status_code is None:
        return None
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


# Message-level failures in this set are permanent for a given message id
# (deleted/inaccessible, malformed, or unauthorized) and should be skipped
# rather than held for Pub/Sub redelivery, which would otherwise re-hit the
# same error on every retry and wedge the history cursor.
#
# 429 is deliberately excluded: it is a transient rate limit, not a
# permanent per-message failure. Treating it as non-retriable would drop the
# message for good; instead it falls through to the "hold cursor, let
# Pub/Sub redeliver" path so its built-in backoff can retry the batch.
_NON_RETRIABLE_MESSAGE_STATUS_CODES = frozenset({400, 403, 404, 410})


def _is_non_retriable_message_error(exc: Exception) -> bool:
    return _exception_status_code(exc) in _NON_RETRIABLE_MESSAGE_STATUS_CODES


def _record_watch_state_error(
    db: Session,
    *,
    state_id: int,
    error_message: str,
    mark_failed: bool = False,
) -> None:
    state = db.query(GmailWatchState).filter(GmailWatchState.id == state_id).first()
    if state is None:
        return
    if mark_failed:
        setattr(state, "status", TriggerProvisioningStatus.FAILED.value)
    setattr(state, "last_error", error_message)
    db.add(state)
    db.commit()


def _renew_watch_for_account(
    db: Session,
    oauth_account: UserOAuth,
    *,
    service_factory: GmailServiceFactory,
) -> GmailWatchState:
    """Renew one mailbox watch through the per-mailbox provisioning machine.

    The legacy shared-token global-topic registration path has been removed;
    a deployment without per-mailbox Pub/Sub configuration converges to a
    failed watch state with a clear last_error instead.

    Gated on ``XAGENT_GMAIL_WATCH_ENABLED`` like the renewal scan that calls
    this directly: without the gate here too, the webhook stale-history path
    (``collect_gmail_pubsub_events``) would reach this function ungated and
    re-register a watch while the flag is off, recreating the
    silently-expiring-watch bug (#1231).
    """
    from .gmail_provisioning import (
        GMAIL_WATCH_DISABLED_ERROR,
        ensure_gmail_mailbox_provisioned,
    )

    if not get_gmail_watch_enabled():
        raise GmailWatchDisabledError(GMAIL_WATCH_DISABLED_ERROR)

    state = ensure_gmail_mailbox_provisioned(
        db,
        oauth_account,
        service_factory=service_factory,
    )
    if str(state.status or "") != "active":
        raise GmailTriggerError(
            str(state.last_error or "Gmail per-mailbox provisioning failed")
        )
    return state


def scan_due_gmail_watch_renewals(
    db: Session,
    *,
    now: datetime | None = None,
    service_factory: GmailServiceFactory = build_gmail_service,
    limit: int = 500,
) -> int:
    if not get_gmail_watch_enabled():
        return 0

    scan_time = _coerce_utc(now) or datetime.now(timezone.utc)
    renew_before = scan_time + timedelta(seconds=get_gmail_watch_renewal_lead_seconds())
    batch_size = max(1, min(int(limit), 500))
    enabled_gmail_users = (
        db.query(AgentTrigger.user_id.label("user_id"))
        .filter(
            AgentTrigger.type == TriggerType.GMAIL.value,
            AgentTrigger.enabled.is_(True),
        )
        .distinct()
        .subquery()
    )
    rows = (
        db.query(UserOAuth, GmailWatchState)
        .join(enabled_gmail_users, enabled_gmail_users.c.user_id == UserOAuth.user_id)
        .outerjoin(
            GmailWatchState,
            GmailWatchState.oauth_account_id == UserOAuth.id,
        )
        .filter(
            UserOAuth.provider == "gmail",
            user_oauth_owner_clause(None),
        )
        .filter(
            or_(
                GmailWatchState.id.is_(None),
                GmailWatchState.watch_expiration.is_(None),
                GmailWatchState.watch_expiration <= renew_before,
            )
        )
        .order_by(
            case((GmailWatchState.watch_expiration.is_(None), 0), else_=1),
            GmailWatchState.watch_expiration,
            UserOAuth.id,
        )
        .limit(batch_size)
        .all()
    )

    renewed = 0
    for oauth_account, state in rows:
        user_id = int(oauth_account.user_id)

        try:
            _renew_watch_for_account(
                db,
                oauth_account,
                service_factory=service_factory,
            )
            renewed += 1
        except Exception as exc:
            logger.error(
                "Failed to renew Gmail watch for user %s, oauth_account %s: %s",
                user_id,
                oauth_account.id,
                exc,
                exc_info=True,
            )
            db.rollback()
            if state is None:
                continue
            state = (
                db.query(GmailWatchState)
                .filter(GmailWatchState.oauth_account_id == int(oauth_account.id))
                .first()
            )
            if state is None:
                continue
            setattr(state, "last_error", str(exc))
            db.add(state)
            try:
                db.commit()
            except Exception as commit_exc:
                db.rollback()
                logger.warning(
                    "Failed to save Gmail watch renewal error for %s: %s",
                    oauth_account.id,
                    commit_exc,
                )

    return renewed


def _header_value(message: dict[str, Any], name: str) -> str:
    headers = (
        message.get("payload", {}).get("headers", [])
        if isinstance(message.get("payload"), dict)
        else []
    )
    for header in headers:
        if not isinstance(header, dict):
            continue
        if str(header.get("name", "")).lower() == name.lower():
            return str(header.get("value") or "")
    return ""


def _message_payload(
    message: dict[str, Any], *, notification: GmailPubsubNotification
) -> dict[str, Any]:
    label_ids = [str(label_id) for label_id in message.get("labelIds", [])]
    return {
        "message_id": str(message.get("id") or ""),
        "thread_id": str(message.get("threadId") or ""),
        "history_id": notification.history_id,
        "pubsub_message_id": notification.pubsub_message_id,
        "from": _header_value(message, "From"),
        "subject": _header_value(message, "Subject"),
        "snippet": str(message.get("snippet") or ""),
        "label_ids": label_ids,
    }


# Gmail's own non-incoming categories: a message the user sent, drafted, or
# that landed in spam/trash. `history.list` is called with no `labelId`
# filter, so these show up alongside real incoming mail and must be excluded
# explicitly — the UI's "watch all incoming emails" promise otherwise means
# "watch literally everything, sent mail included."
_NON_INCOMING_LABELS = {"sent", "draft", "spam", "trash"}


def _trigger_matches_message(trigger: AgentTrigger, payload: dict[str, Any]) -> bool:
    config = dict(trigger.config or {})
    label_ids = {str(label_id).lower() for label_id in payload.get("label_ids", [])}
    # Strip BEFORE falling back to the default: a whitespace-only stored
    # value (e.g. " ") is truthy pre-strip, so `... or "inbox"` alone would
    # skip the default and strip down to "", which matches neither the
    # wildcard branch below nor a real label — silently disabling all
    # filtering instead of falling back to INBOX.
    watch_label = str(config.get("watch_label") or "").strip().lower() or "inbox"
    # Gmail's own non-incoming categories are excluded regardless of which
    # branch below matches — a custom label manually applied to an
    # already-sent/draft/spam/trash message must not fire either, not just
    # the wildcard "watch everything incoming" case.
    if label_ids & _NON_INCOMING_LABELS:
        return False
    if watch_label not in {"*", "all"} and watch_label not in label_ids:
        return False

    sender_filter = str(config.get("sender_filter") or "").strip().lower()
    if sender_filter and sender_filter not in str(payload.get("from") or "").lower():
        return False

    subject_keyword = str(config.get("subject_keyword") or "").strip().lower()
    if (
        subject_keyword
        and subject_keyword not in str(payload.get("subject") or "").lower()
    ):
        return False

    return True


def _added_message_ids_from_history(history_response: dict[str, Any]) -> list[str]:
    message_ids: list[str] = []
    for history_item in history_response.get("history", []) or []:
        if not isinstance(history_item, dict):
            continue
        for added in history_item.get("messagesAdded", []) or []:
            if not isinstance(added, dict):
                continue
            message = added.get("message")
            if isinstance(message, dict) and message.get("id"):
                message_ids.append(str(message["id"]))
    return message_ids


def _list_added_message_ids(service: Any, *, start_history_id: str) -> list[str]:
    message_ids: list[str] = []
    page_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {
            "userId": "me",
            "startHistoryId": start_history_id,
            "historyTypes": ["messageAdded"],
        }
        if page_token:
            kwargs["pageToken"] = page_token
        response = service.users().history().list(**kwargs).execute()
        if isinstance(response, dict):
            message_ids.extend(_added_message_ids_from_history(response))
            page_token = response.get("nextPageToken")
        else:
            page_token = None
        if not page_token:
            return message_ids


def _get_gmail_message(service: Any, message_id: str) -> dict[str, Any]:
    response = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )
    return response if isinstance(response, dict) else {}


async def collect_gmail_pubsub_events(
    db: Session,
    notification: GmailPubsubNotification,
    *,
    state: GmailWatchState,
    service_factory: GmailServiceFactory = build_gmail_service,
) -> GmailPubsubEventCollection:
    """Decode Gmail history into provider-normalized events without firing.

    The unified TriggerProvider pipeline owns authorization, idempotency,
    audit, and execution. This helper keeps Gmail-specific history traversal
    and trigger filter matching in the Gmail module. It never advances the
    watch cursor: ``GmailProvider.finalize_callback`` moves ``history_id``
    only after every collected event fired successfully, so a failed batch
    stays redeliverable.

    ``state`` must be the watch state the callback was addressed to (resolved
    by callback id). GmailWatchState.email is not unique — two accounts can
    watch the same mailbox — so resolving by email here could read another
    account's cursor and OAuth token while the caller advances this one's.
    """
    email_address = notification.email_address.strip().lower()
    if not email_address or not notification.history_id:
        return GmailPubsubEventCollection(events=[], skipped=1)

    oauth_account = get_scoped_user_oauth_account(
        db,
        user_id=int(state.user_id),
        account_id=int(state.oauth_account_id),
        resource_owner_key=None,
    )
    if oauth_account is None:
        return GmailPubsubEventCollection(events=[], skipped=1)

    state_id = int(state.id)
    try:
        service = await asyncio.to_thread(service_factory, db, oauth_account)
    except GmailTriggerError as exc:
        db.rollback()
        _record_watch_state_error(
            db,
            state_id=state_id,
            error_message=str(exc),
        )
        raise

    start_history_id = str(state.history_id)
    try:
        message_ids = await asyncio.to_thread(
            _list_added_message_ids,
            service,
            start_history_id=start_history_id,
        )
    except Exception as exc:
        if _exception_status_code(exc) not in (400, 404):
            raise
        logger.warning(
            "Gmail startHistoryId %s is too old or expired for %s; "
            "re-registering watch: %s",
            start_history_id,
            email_address,
            exc,
        )
        db.rollback()
        try:
            await asyncio.to_thread(
                _renew_watch_for_account,
                db,
                oauth_account,
                service_factory=service_factory,
            )
        except Exception as watch_exc:
            logger.error(
                "Failed to re-register Gmail watch for %s: %s",
                email_address,
                watch_exc,
                exc_info=True,
            )
            db.rollback()
            from .gmail_provisioning import GMAIL_WATCH_DISABLED_ERROR

            watch_error = str(watch_exc).strip()
            disabled = isinstance(watch_exc, GmailWatchDisabledError)
            error_message = (
                GMAIL_WATCH_DISABLED_ERROR
                if disabled
                else (
                    "Gmail history expired and re-registration failed"
                    + (f": {watch_error}" if watch_error else "")
                )
            )
            # The disabled case is a permanent condition, not a transient
            # failure: retrying re-registration will fail identically until an
            # operator flips XAGENT_GMAIL_WATCH_ENABLED, and the renewal scan
            # already retries by expiration once it is. Mark the row failed
            # and ack (200) instead of raising, so Pub/Sub does not redeliver
            # with backoff for the retention window; a transient failure keeps
            # last_error only (no status flip) and still raises so the
            # pipeline's failure_status=500 preserves redelivery semantics.
            _record_watch_state_error(
                db,
                state_id=state_id,
                error_message=error_message,
                mark_failed=disabled,
            )
            if disabled:
                return GmailPubsubEventCollection(events=[], skipped=1)
            raise GmailTriggerError(error_message) from watch_exc
        return GmailPubsubEventCollection(events=[], skipped=1)

    triggers = (
        db.query(AgentTrigger)
        .filter(
            AgentTrigger.user_id == int(state.user_id),
            AgentTrigger.type == TriggerType.GMAIL.value,
            AgentTrigger.enabled.is_(True),
        )
        .all()
    )

    events: list[GmailCollectedEvent] = []
    skipped = 0
    failed_message_ids: list[str] = []
    for message_id in message_ids:
        try:
            try:
                message = await asyncio.to_thread(
                    _get_gmail_message, service, message_id
                )
            except Exception as exc:
                if _is_non_retriable_message_error(exc):
                    logger.warning(
                        "Skipping inaccessible Gmail message %s for %s: %s",
                        message_id,
                        email_address,
                        exc,
                    )
                    skipped += 1
                    continue
                raise

            payload = _message_payload(message, notification=notification)
            payload["message_id"] = payload["message_id"] or message_id
            matched = False
            for trigger in triggers:
                if not _trigger_matches_message(trigger, payload):
                    continue
                matched = True
                events.append(
                    GmailCollectedEvent(
                        trigger_id=int(trigger.id),
                        payload=dict(payload),
                        source_event_id=f"gmail:{message_id}",
                        event_type="gmail.message",
                        resource_id=email_address,
                    )
                )
            if not matched:
                skipped += 1
        except Exception as exc:
            logger.error(
                "Failed to collect Gmail message %s for %s: %s",
                message_id,
                email_address,
                exc,
                exc_info=True,
            )
            db.rollback()
            failed_message_ids.append(message_id)
            skipped += 1

    if failed_message_ids:
        error_message = "Failed to process Gmail message(s): " + ", ".join(
            failed_message_ids
        )
        _record_watch_state_error(
            db,
            state_id=state_id,
            error_message=error_message,
        )
        raise GmailTriggerError(error_message)

    return GmailPubsubEventCollection(events=events, skipped=skipped)
