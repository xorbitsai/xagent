"""GitHub trigger provider: repository webhooks on the unified callback route.

Each trigger owns one repository webhook, created with the trigger owner's
connected GitHub account. Deliveries are authenticated with GitHub's
``X-Hub-Signature-256`` HMAC over the raw body using the trigger's encrypted
secret, and deduplicated on ``X-GitHub-Delivery``. The provider is
channel-agnostic: it only produces normalized events for the shared pipeline.

Event types are ``<event>.<action>`` for pull_request, issues, issue_comment
and pull_request_review (e.g. ``pull_request.opened``), and the bare event
name for push, create and delete.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import threading
import weakref
from collections.abc import AsyncIterator
from typing import Any, Mapping

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ....config import get_trigger_callback_base_url
from ....core.utils.encryption import decrypt_value
from ...models.trigger import AgentTrigger, TriggerProvisioningStatus, TriggerType
from ..github_provisioning import (
    GithubProvisioningError,
    create_hook,
    delete_hook,
    ensure_fresh_token,
    join_binding,
    resolve_github_account,
    split_binding,
    update_hook,
)
from .base import (
    CallbackRequestContext,
    TriggerConfigError,
    TriggerEventParseError,
)
from .registry import register_trigger_provider
from .schemas import (
    GITHUB_ACTION_EVENTS,
    GITHUB_HOOK_EVENTS,
    AckPolicy,
    ChallengeResponse,
    NormalizedEvent,
    RegistrationResult,
    VerificationResult,
    parse_trigger_config,
)

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "x-hub-signature-256"
EVENT_HEADER = "x-github-event"
DELIVERY_HEADER = "x-github-delivery"
HOOK_ID_HEADER = "x-github-hook-id"
# GitHub caps webhook payloads at 25 MB; anything larger is not from GitHub.
MAX_BODY_BYTES = 25 * 1024 * 1024
_TEXT_LIMIT = 4000
_COMMIT_LIMIT = 20

CALLBACK_BASE_URL_MISSING_ERROR = (
    "Trigger callback base URL is not configured (XAGENT_PUBLIC_API_BASE_URL)"
)


def sign_github_payload(secret: str, raw_body: bytes) -> str:
    """Compute the ``X-Hub-Signature-256`` header value for a body."""
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def github_callback_url(base_url: str, callback_id: str) -> str:
    return f"{base_url.rstrip('/')}/api/triggers/callback/github/{callback_id}"


def hook_events_for(config: Mapping[str, Any]) -> list[str]:
    """GitHub hook-level events to subscribe to for a trigger config."""
    allow_list = config.get("event_types")
    if not isinstance(allow_list, list) or not allow_list:
        return list(GITHUB_HOOK_EVENTS)
    wanted = {str(item).partition(".")[0] for item in allow_list}
    return [event for event in GITHUB_HOOK_EVENTS if event in wanted]


def _obj(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _stored_config(trigger: AgentTrigger) -> dict[str, Any]:
    config = trigger.config
    return dict(config) if isinstance(config, dict) else {}


_PROVISIONING_LOCK_TIMEOUT_SECONDS = 30
_PG_ROW_LOCK_TIMEOUT = "5s"
_provisioning_locks: "weakref.WeakValueDictionary[int, threading.Lock]" = (
    weakref.WeakValueDictionary()
)
_provisioning_locks_guard = threading.Lock()


@contextlib.asynccontextmanager
async def _trigger_provisioning_lock(trigger_id: int) -> AsyncIterator[None]:
    """In-process per-trigger lock (the only serialization on SQLite)."""
    with _provisioning_locks_guard:
        lock = _provisioning_locks.get(trigger_id)
        if lock is None:
            lock = threading.Lock()
            _provisioning_locks[trigger_id] = lock
    acquired = await asyncio.to_thread(
        lock.acquire, True, _PROVISIONING_LOCK_TIMEOUT_SECONDS
    )
    if not acquired:
        raise GithubProvisioningError(
            "Another update of this trigger is in progress; retry shortly"
        )
    try:
        yield
    finally:
        lock.release()


def _lock_trigger_row(db: Session, trigger_id: int) -> AgentTrigger | None:
    """Reload the trigger under a row lock (PostgreSQL; a no-op on SQLite).

    The lock is held until the caller commits, so concurrent processes also
    serialize their hook writes. The wait is bounded by ``lock_timeout``.
    """
    try:
        if db.get_bind().dialect.name == "postgresql":
            db.execute(text(f"SET LOCAL lock_timeout = '{_PG_ROW_LOCK_TIMEOUT}'"))
        return (
            db.query(AgentTrigger)
            .filter(AgentTrigger.id == trigger_id)
            .populate_existing()
            .with_for_update()
            .first()
        )
    except OperationalError as exc:
        db.rollback()
        raise GithubProvisioningError(
            "Another update of this trigger is in progress; retry shortly"
        ) from exc


def _text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)[:_TEXT_LIMIT]


def _user(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value.get("login"):
        return None
    return {"login": value.get("login"), "url": value.get("html_url")}


def _pull_request(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    base = _obj(value.get("base"))
    head = _obj(value.get("head"))
    return {
        "number": value.get("number"),
        "title": _text(value.get("title")),
        "url": value.get("html_url"),
        "state": value.get("state"),
        "draft": value.get("draft"),
        "merged": value.get("merged"),
        "author": _user(value.get("user")),
        "base_ref": base.get("ref"),
        "head_ref": head.get("ref"),
        "body": _text(value.get("body")),
    }


def _issue(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_labels = value.get("labels")
    labels: list[Any] = raw_labels if isinstance(raw_labels, list) else []
    return {
        "number": value.get("number"),
        "title": _text(value.get("title")),
        "url": value.get("html_url"),
        "state": value.get("state"),
        "author": _user(value.get("user")),
        "labels": [
            str(label.get("name")) for label in labels if isinstance(label, dict)
        ],
        "is_pull_request": "pull_request" in value,
        "body": _text(value.get("body")),
    }


def _comment_like(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "id": value.get("id"),
        "url": value.get("html_url"),
        "state": value.get("state"),
        "author": _user(value.get("user")),
        "body": _text(value.get("body")),
    }


def _commits(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {
            "id": commit.get("id"),
            "message": _text(commit.get("message")),
            "url": commit.get("url"),
            "author": (commit.get("author") or {}).get("name"),
        }
        for commit in value[:_COMMIT_LIMIT]
        if isinstance(commit, dict)
    ]


def normalize_github_event(
    event: str, payload: dict[str, Any], *, delivery_id: str | None
) -> NormalizedEvent | None:
    """Normalize one verified delivery; None for events this provider ignores."""
    if event not in GITHUB_HOOK_EVENTS:
        return None
    action = payload.get("action")
    if event in GITHUB_ACTION_EVENTS:
        if not isinstance(action, str) or not action:
            return None
        event_type = f"{event}.{action}"
    else:
        event_type = event

    repository = payload.get("repository")
    repo = repository if isinstance(repository, dict) else {}
    full_name = repo.get("full_name")
    body: dict[str, Any] = {
        "provider": "github",
        "event": event,
        "action": action if isinstance(action, str) else None,
        "delivery_id": delivery_id,
        "repository": {
            "full_name": full_name,
            "url": repo.get("html_url"),
            "private": repo.get("private"),
        },
        "sender": _user(payload.get("sender")),
    }
    if event == "pull_request":
        body["pull_request"] = _pull_request(payload.get("pull_request"))
    elif event == "issues":
        body["issue"] = _issue(payload.get("issue"))
    elif event == "issue_comment":
        body["issue"] = _issue(payload.get("issue"))
        body["comment"] = _comment_like(payload.get("comment"))
    elif event == "pull_request_review":
        body["pull_request"] = _pull_request(payload.get("pull_request"))
        body["review"] = _comment_like(payload.get("review"))
    elif event == "push":
        head_commit = payload.get("head_commit")
        pusher = payload.get("pusher")
        all_commits = payload.get("commits")
        body.update(
            {
                "ref": payload.get("ref"),
                "before": payload.get("before"),
                "after": payload.get("after"),
                "compare_url": payload.get("compare"),
                "created": payload.get("created"),
                "deleted": payload.get("deleted"),
                "forced": payload.get("forced"),
                "pusher": pusher.get("name") if isinstance(pusher, dict) else None,
                "head_commit": _commits([head_commit])[0]
                if isinstance(head_commit, dict)
                else None,
                "commit_count": len(all_commits)
                if isinstance(all_commits, list)
                else 0,
                "commits": _commits(all_commits),
            }
        )
    else:  # create / delete
        body.update({"ref": payload.get("ref"), "ref_type": payload.get("ref_type")})
    return NormalizedEvent(
        event_type=event_type,
        source_event_id=delivery_id,
        resource_id=str(full_name) if full_name else None,
        payload=body,
    )


class GithubProvider:
    """GitHub repository-webhook provider for the unified callback pipeline."""

    name = TriggerType.GITHUB.value
    # GitHub does not retry deliveries; a rejected one is simply visible in its
    # delivery log, so use the default (non-2xx) statuses.
    ack_policy = AckPolicy()
    max_body_bytes = MAX_BODY_BYTES

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
        # ``ping`` is answered after signature verification (see parse_events)
        # so an unauthenticated caller cannot probe callback ids.
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
        if trigger is None or not trigger.secret_encrypted:
            return VerificationResult.reject("GitHub trigger has no signing secret")
        signature = (context.header(SIGNATURE_HEADER) or "").strip()
        if not signature.startswith("sha256="):
            return VerificationResult.reject("Missing GitHub signature header")
        secret = decrypt_value(str(trigger.secret_encrypted))
        expected = sign_github_payload(secret, raw_body)
        if not hmac.compare_digest(expected, signature):
            return VerificationResult.reject("Invalid GitHub signature")
        if not (context.header(DELIVERY_HEADER) or "").strip():
            return VerificationResult.reject("Missing GitHub delivery id")

        # The body is authentic from here on, so its repository is attested.
        # The hook id binds the delivery to the hook this trigger registered.
        full_name = ""
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
            repository = (
                decoded.get("repository") if isinstance(decoded, dict) else None
            )
            if isinstance(repository, dict):
                full_name = str(repository.get("full_name") or "")
        except (UnicodeDecodeError, ValueError):
            pass
        hook_id = (context.header(HOOK_ID_HEADER) or "").strip()
        attested = (
            join_binding(full_name, int(hook_id))
            if (full_name and hook_id.isdecimal())
            else None
        )
        return VerificationResult.ok(attested_resource_id=attested)

    async def register(
        self, db: Session, trigger: AgentTrigger, config: Any
    ) -> RegistrationResult:
        trigger_id = int(trigger.id)
        trigger_config = self.validate_config(_stored_config(trigger))
        try:
            base_url = get_trigger_callback_base_url()
            if not base_url:
                raise GithubProvisioningError(CALLBACK_BASE_URL_MISSING_ERROR)
            # Token refresh commits, so it runs before the row lock is taken.
            account = resolve_github_account(
                db,
                user_id=int(trigger.user_id),
                oauth_account_id=trigger_config.oauth_account_id,
            )
            token = await ensure_fresh_token(db, account)
            # Serialize read secret -> hook write -> persist per trigger, so a
            # stale request cannot reinstall an older secret after a newer
            # one was committed and installed.
            async with _trigger_provisioning_lock(trigger_id):
                fresh = _lock_trigger_row(db, trigger_id)
                if fresh is None or not bool(fresh.enabled):
                    return RegistrationResult(status=TriggerProvisioningStatus.PENDING)
                if not fresh.callback_id or not fresh.secret_encrypted:
                    raise GithubProvisioningError(
                        "GitHub trigger has no callback secret"
                    )
                current = self.validate_config(_stored_config(fresh))
                repository = str(current.repository)
                _, hook_id = split_binding(
                    str(fresh.resource_id) if fresh.resource_id else None
                )
                callback_url = github_callback_url(base_url, str(fresh.callback_id))
                secret = decrypt_value(str(fresh.secret_encrypted))
                events = hook_events_for(_stored_config(fresh))
                hook = None
                if hook_id is not None:
                    hook = await asyncio.to_thread(
                        update_hook,
                        token,
                        repository,
                        hook_id,
                        callback_url=callback_url,
                        secret=secret,
                        events=events,
                    )
                if hook is None:
                    hook = await asyncio.to_thread(
                        create_hook,
                        token,
                        repository,
                        callback_url=callback_url,
                        secret=secret,
                        events=events,
                    )
                binding = join_binding(repository.lower(), hook.hook_id)
                setattr(fresh, "resource_id", binding)
                setattr(
                    fresh, "provisioning_status", TriggerProvisioningStatus.ACTIVE.value
                )
                setattr(fresh, "provisioning_error", None)
                db.add(fresh)
                db.commit()
                return RegistrationResult(
                    status=TriggerProvisioningStatus.ACTIVE, resource_id=binding
                )
        except GithubProvisioningError as exc:
            logger.warning(
                "GitHub hook provisioning failed for trigger %s: %s", trigger_id, exc
            )
            db.rollback()
            setattr(
                trigger, "provisioning_status", TriggerProvisioningStatus.FAILED.value
            )
            setattr(trigger, "provisioning_error", str(exc))
            db.add(trigger)
            db.commit()
            return RegistrationResult(
                status=TriggerProvisioningStatus.FAILED, error=str(exc)
            )

    async def unregister(
        self,
        db: Session,
        trigger: AgentTrigger,
        config: Any,
        *,
        resource_id: str | None = None,
    ) -> None:
        repository, hook_id = split_binding(resource_id)
        if repository is None or hook_id is None:
            return  # never provisioned; nothing exists on GitHub
        # Read before any commit: the trigger row may already be deleted.
        user_id = int(trigger.user_id)
        raw_account_id = (
            config.get("oauth_account_id") if isinstance(config, dict) else None
        )
        try:
            account = resolve_github_account(
                db,
                user_id=user_id,
                oauth_account_id=raw_account_id
                if isinstance(raw_account_id, int)
                else None,
            )
            token = await ensure_fresh_token(db, account)
            await asyncio.to_thread(delete_hook, token, repository, hook_id)
        except GithubProvisioningError as exc:
            # Raised so trigger CRUD logs the leak; the row is already gone or
            # re-bound, so there is nothing to roll back here.
            raise RuntimeError(
                f"Could not delete GitHub webhook {hook_id} on {repository}: {exc}"
            ) from exc

    async def parse_events(
        self,
        context: CallbackRequestContext,
        *,
        db: Session,
        trigger: AgentTrigger | None,
        raw_body: bytes,
    ) -> list[NormalizedEvent]:
        event = (context.header(EVENT_HEADER) or "").strip()
        if event == "ping":
            return []  # verified handshake: acknowledged, nothing to fire
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise TriggerEventParseError("GitHub payload must be JSON") from exc
        if not isinstance(payload, dict):
            raise TriggerEventParseError("GitHub payload must be a JSON object")
        normalized = normalize_github_event(
            event,
            payload,
            delivery_id=(context.header(DELIVERY_HEADER) or "").strip() or None,
        )
        return [normalized] if normalized is not None else []

    async def finalize_callback(
        self,
        *,
        db: Session,
        context: CallbackRequestContext,
        trigger: AgentTrigger | None,
        events: list[NormalizedEvent],
        raw_body: bytes,
    ) -> None:
        """GitHub delivery has no cursor state to advance."""


register_trigger_provider(GithubProvider(), replace=True)
