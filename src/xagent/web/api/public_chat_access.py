"""Shared runtime helpers for public widget/share chat access."""

from __future__ import annotations

import json
import logging
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Protocol, TypeGuard, TypeVar

from fastapi import Depends, HTTPException, Query, UploadFile, WebSocket
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...core.task_runtime import (
    FILE_OPERATION_ACCESS_VERSION,
    FILE_OPERATION_ACCESS_VERSION_KEY,
)
from ..auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from ..jwt_validation import (
    has_matching_temporal_claim_conversion_failure,
    is_exact_integer_bindable,
)
from ..models.agent import Agent, AgentStatus, is_workforce_generated_manager_agent
from ..models.database import get_db, get_session_local, release_db_connection_if_clean
from ..models.deployment import DeploymentOwnerType
from ..models.task import Task, TaskStatus
from ..models.user import User
from ..models.user_channel import UserChannel
from ..models.workforce import Workforce, WorkforceRun
from ..schemas.chat import TaskCreateRequest, TaskCreateResponse
from ..schemas.connector_runtime import ConnectorRuntimeRequirementsModel
from ..services.client_error_messages import ClientErrorCode, client_error_message
from ..services.connector_runtime import (
    bind_connector_runtime_selection_snapshot,
    prepare_connector_runtime_selection_snapshot,
)
from ..services.db_runtime import run_db_io_cancellation_safe
from ..services.deployments import get_deployment
from ..services.orphan_upload_gc import TASKLESS_SHARE_UPLOAD_SOURCE
from ..services.share_rate_limit import (
    entity_rate_limit_key,
    get_share_rate_limiter,
    remote_ip_from_request,
)
from ..services.task_interaction_service import (
    InteractionPrincipal,
    public_chat_identity_matches,
    task_is_owned_by_public_principal,
)
from ..services.task_runtime import sanitize_client_agent_config
from ..services.workforce_runs import create_workforce_run
from ..utils.db_timezone import format_datetime_for_api
from .files import store_uploaded_files
from .websocket import (
    WebSocketPrincipal,
    handle_chat_message,
    handle_execute_task,
    handle_intervention,
    handle_status_request,
    manager,
    send_message_delivery,
)
from .websocket_auth import send_websocket_authentication_infrastructure_failure

logger = logging.getLogger(__name__)
db_session_context = contextmanager(get_db)

# Cap on files per task-less share upload request. This path predates any
# task/owner association (workforce first-turn attachments), so it is the one
# public-share write surface with no downstream owner scoping; the cap blocks
# the worst single-request abuse. Broader quota + orphan GC tracked in #973.
MAX_TASKLESS_SHARE_UPLOAD_FILES = 10

# Every public create path in this module answers this field the same way:
# it never resolves connector-runtime requirements for its caller, because
# the widget/share guest must not see connector key names. Kept as one
# named constant so the four call sites cannot drift apart.
_NO_CONNECTOR_RUNTIME_REQUIREMENTS: ConnectorRuntimeRequirementsModel | None = None


class PublicChatAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    agent_id: int | None = None
    agent_name: str | None = None
    agent_logo: str | None = None
    agent_description: str | None = None
    suggested_prompts: list[str] = []
    # Set instead of ``agent_id`` when the share token exposes a workforce.
    workforce_id: int | None = None


@dataclass(frozen=True)
class PublicChatAccessContext:
    user: User
    channel_id: int | None
    guest_id: str
    auth_mode: str = "widget"
    widget_agent_id: int | None = None
    # Set instead of ``widget_agent_id`` when the widget key exposes a
    # workforce. Exactly one of the two is populated for a widget guest.
    # Structural enforcement is tracked in #1288 (the share twin is already
    # enforced below, #1225).
    widget_workforce_id: int | None = None


@dataclass(frozen=True)
class ShareChatAccessContext:
    """Validated share-guest identity: exactly one of ``agent`` /
    ``workforce`` is set, depending on which kind of share token the guest
    presented — enforced at construction by ``__post_init__`` (#1225).
    ``user`` is always the owner the shared entity runs as.

    ``guest_id`` is the per-guest isolation credential (#973): a share link is
    public, so many anonymous visitors share the same owner + entity id, and
    ``guest_id`` is the only thing distinguishing one visitor's tasks from
    another's. It is server-minted at auth time and always non-empty here —
    :func:`get_share_chat_user` rejects tokens that lack it, so downstream
    equality checks never compare ``None == None``.
    """

    user: User
    share_token: str
    guest_id: str
    agent: Agent | None = None
    workforce: Workforce | None = None

    def __post_init__(self) -> None:
        # Exactly-one guard (#1225): the run quota keys off these markers and
        # entity_rate_limit_key prefers workforce when both ids are set, so a
        # both-set context would silently charge an agent-share run to a
        # workforce bucket; a neither-set one has no billing bucket at all.
        # Compared against None, not truthiness — tests construct the context
        # with arbitrary stand-in objects. A violation is a programming
        # defect, never a bad credential: get_share_chat_user only rewrites
        # _PublicTokenRejected into a 401 and passes HTTPException through
        # unchanged (e.g. the 403s from ensure_share_*_available), so this
        # ValueError propagates loudly (#1214). Both production construction
        # sites set exactly one entity; the guard defends future call sites.
        if (self.agent is None) == (self.workforce is None):
            raise ValueError(
                "ShareChatAccessContext requires exactly one of 'agent' or 'workforce'"
            )


def mint_share_guest_id() -> str:
    """Mint a server-owned, high-entropy guest id for a new share session.

    Unlike the widget path (which signs a client-supplied ``guest_id``), share
    links have no secondary credential such as an embed ticket or widget key,
    so the guest id is the *sole* isolation credential. It must therefore be
    generated server-side and never accepted from the client — otherwise a
    visitor could impersonate another by choosing their id.
    """
    return secrets.token_urlsafe(32)


class _ChatAccessContext(Protocol):
    @property
    def user(self) -> User: ...

    @property
    def guest_id(self) -> str: ...


_ChatAccessContextT = TypeVar("_ChatAccessContextT", bound=_ChatAccessContext)


def _is_strict_int(value: object) -> TypeGuard[int]:
    """Whether a JWT claim is a real ``int`` — a row id we may query on.

    Every id claim on a widget/share guest token goes through here rather than a
    bare ``isinstance(..., int)`` (#992). ``bool`` subclasses ``int``, so a
    ``true`` claim would pass isinstance, and SQLAlchemy renders it as ``= 1``:
    it resolves to row id 1 *and* then compares equal to an owner id of 1 in
    Python (``1 == True``), which would admit a guest as the first user. The
    behaviour is also backend-dependent — SQLite binds and matches where
    PostgreSQL typically raises — so a bare isinstance check makes an auth
    decision that varies by database.

    These tokens are server-minted and signed, so no caller can present such a
    claim today; this keeps every id claim failing closed regardless.
    """
    return isinstance(value, int) and not isinstance(value, bool)


class _PublicTokenRejected(Exception):
    """An expected widget/share credential rejection."""

    def __init__(self, reason: "_PublicTokenRejectionReason") -> None:
        super().__init__()
        self.reason = reason


class _PublicTokenRejectionReason(Enum):
    INVALID_TOKEN = auto()
    INVALID_CLAIMS = auto()
    INVALID_ENTITY = auto()


def _project_public_token_failure(
    exc: Exception, *, invalid_detail: str
) -> HTTPException | None:
    """Project expected public-token failures while preserving operations."""
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, _PublicTokenRejected):
        logger.info("Public chat credential rejected reason=%s", exc.reason.name)
        return HTTPException(status_code=401, detail=invalid_detail)
    return None


def _decode_public_token(token: str) -> dict[str, Any]:
    """Verify one public token without folding decoder defects into credentials."""
    if token.startswith("Bearer "):
        token = token[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_TOKEN) from None
    except (TypeError, OverflowError) as exc:
        if has_matching_temporal_claim_conversion_failure(token, exc):
            raise _PublicTokenRejected(
                _PublicTokenRejectionReason.INVALID_CLAIMS
            ) from None
        raise
    if type(payload) is not dict:
        raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_TOKEN)
    return payload


def _require_bindable_public_id(value: object, dialect_name: str) -> int:
    """Return an exact query id accepted by the supported database dialect."""
    if not is_exact_integer_bindable(value, dialect_name):
        raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_CLAIMS)
    return value


def create_public_chat_access_token(data: dict[str, Any]) -> str:
    """Create JWT access token for widget/share guests."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=30)
    to_encode.update({"exp": expire, "type": "widget"})
    encoded_jwt: str = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def ensure_widget_agent_available(
    db: Session,
    widget_agent_id: int,
    user_id: int,
    *,
    expected_widget_key: str | None = None,
) -> Agent:
    """Resolve a widget-guest agent id to a live, widget-enabled agent.

    Mirrors :func:`ensure_share_agent_available` for the widget channel. The
    guest JWT is long-lived (30-day TTL), so access is re-derived from the
    agent's *current* widget state on every request rather than trusted from
    the token alone. Three owner-side levers cut off outstanding guest tokens
    on their next request: disabling the widget (``widget_enabled = False``),
    rotating ``widget_key``, and unpublishing the agent (#1055).

    Publication is checked here rather than by having ``unpublish_agent`` clear
    ``widget_enabled``, so that the gate is re-derived per request and covers
    every code path that takes an agent out of ``PUBLISHED`` — and so that
    re-publishing restores the widget without the owner re-deploying it.

    ``expected_widget_key`` is the key carried by the guest JWT; when present
    it must still match the agent's live key. Tokens minted before the key was
    embedded carry no key and skip that comparison — they stay gated on
    ``widget_enabled`` and publication so those paths still revoke them.
    """
    agent = db.query(Agent).filter(Agent.id == widget_agent_id).first()
    if (
        not agent
        or is_workforce_generated_manager_agent(agent)
        or agent.user_id != user_id
        or not agent.widget_enabled
        or not agent.widget_key
        or agent.status != AgentStatus.PUBLISHED
        or (expected_widget_key is not None and agent.widget_key != expected_widget_key)
    ):
        raise HTTPException(status_code=403, detail="Widget is unavailable")
    return agent


def ensure_share_agent_available(
    db: Session,
    share_agent_id: int,
    user_id: int,
    *,
    expected_share_token: str | None = None,
) -> Agent:
    agent = db.query(Agent).filter(Agent.id == share_agent_id).first()
    if (
        not agent
        or is_workforce_generated_manager_agent(agent)
        or agent.user_id != user_id
        or not agent.share_enabled
        or not agent.share_token
        or agent.status != AgentStatus.PUBLISHED
        or (
            expected_share_token is not None
            and agent.share_token != expected_share_token
        )
    ):
        raise HTTPException(status_code=403, detail="Share link is unavailable")
    return agent


def ensure_share_workforce_available(
    db: Session,
    share_workforce_id: int,
    user_id: int,
    *,
    expected_share_token: str | None = None,
) -> Workforce:
    """Resolve a share-guest workforce id to a live, shareable workforce.

    Mirrors :func:`ensure_share_agent_available`: the share credential lives
    in the workforce's ``deployments`` row, and external access requires the
    workforce to still be published (``status == "active"``) — archiving or
    unpublishing revokes every outstanding guest token.
    """
    workforce = db.query(Workforce).filter(Workforce.id == share_workforce_id).first()
    deployment = get_deployment(db, DeploymentOwnerType.WORKFORCE, share_workforce_id)
    if (
        not workforce
        or workforce.owner_user_id != user_id
        or workforce.status != "active"
        or deployment is None
        or not deployment.share_enabled
        or not deployment.share_token
        or (
            expected_share_token is not None
            and deployment.share_token != expected_share_token
        )
    ):
        raise HTTPException(status_code=403, detail="Share link is unavailable")
    return workforce


def ensure_widget_workforce_available(
    db: Session,
    widget_workforce_id: int,
    user_id: int,
    *,
    expected_widget_key: str | None = None,
) -> Workforce:
    """Resolve a widget-guest workforce id to a live, widget-enabled workforce.

    Mirrors :func:`ensure_share_workforce_available` for the widget channel:
    the credential lives in the workforce's ``deployments`` row, so disabling
    the widget or rotating its key (``expected_widget_key`` mismatch) revokes
    every outstanding guest token on its next request. External access requires
    the workforce to still be published (``status == "active"``).
    """
    workforce = db.query(Workforce).filter(Workforce.id == widget_workforce_id).first()
    deployment = get_deployment(db, DeploymentOwnerType.WORKFORCE, widget_workforce_id)
    if (
        not workforce
        or workforce.owner_user_id != user_id
        or workforce.status != "active"
        or deployment is None
        or not deployment.widget_enabled
        or not deployment.widget_key
        or (
            expected_widget_key is not None
            and deployment.widget_key != expected_widget_key
        )
    ):
        raise HTTPException(status_code=403, detail="Widget is unavailable")
    return workforce


def get_public_chat_user(
    token: str,
    db: Session,
    *,
    expected_auth_mode: str | None = None,
) -> PublicChatAccessContext:
    """Get public chat access context from a widget/share token."""
    try:
        payload = _decode_public_token(token)
        if payload.get("type") != "widget":
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_TOKEN)

        user_id = payload.get("user_id")
        channel_id = payload.get("channel_id")
        guest_id = payload.get("guest_id")
        auth_mode = payload.get("auth_mode") or "widget"
        widget_agent_id = payload.get("widget_agent_id")
        widget_workforce_id = payload.get("widget_workforce_id")
        widget_key = payload.get("widget_key")
        if expected_auth_mode and auth_mode != expected_auth_mode:
            raise HTTPException(status_code=403, detail="Access denied")

        if auth_mode != "widget":
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_TOKEN)

        # Defense in depth (#992): the same guard get_share_chat_user applies.
        # Without it the two widget branches below disagreed about a string
        # user_id — the agent branch failed closed on an int/str owner
        # comparison, while the workforce branch coerced it and admitted the
        # request. Establishing the type here is what lets both branches pass
        # ``user_id`` straight to their ``ensure_widget_*_available`` helper.
        if (
            not _is_strict_int(user_id)
            or not user_id
            or not isinstance(guest_id, str)
            or not guest_id
        ):
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_CLAIMS)
        if channel_id is not None and not _is_strict_int(channel_id):
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_CLAIMS)

        if _is_strict_int(widget_workforce_id):
            selected_agent_id = None
            selected_workforce_id = widget_workforce_id
        elif _is_strict_int(widget_agent_id):
            selected_agent_id = widget_agent_id
            selected_workforce_id = None
        else:
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_ENTITY)

        dialect_name = str(db.get_bind().dialect.name)
        user_id = _require_bindable_public_id(user_id, dialect_name)
        if channel_id is not None:
            channel_id = _require_bindable_public_id(channel_id, dialect_name)
        if selected_workforce_id is not None:
            selected_workforce_id = _require_bindable_public_id(
                selected_workforce_id, dialect_name
            )
        else:
            selected_agent_id = _require_bindable_public_id(
                selected_agent_id, dialect_name
            )

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_CLAIMS)

        if selected_workforce_id is not None:
            # Workforce widget: re-validate the deployment on every use so
            # disabling the widget or rotating its key cuts off live guests,
            # mirroring the workforce share path.
            ensure_widget_workforce_available(
                db,
                selected_workforce_id,
                user_id,
                expected_widget_key=widget_key
                if isinstance(widget_key, str) and widget_key
                else None,
            )
            return PublicChatAccessContext(
                user=user,
                channel_id=channel_id,
                guest_id=guest_id,
                auth_mode=auth_mode,
                widget_workforce_id=selected_workforce_id,
            )

        # Agent widget: re-validate against the live agent every request so
        # disabling the widget or rotating its key revokes outstanding guest
        # tokens (mirrors the share path). The per-message WS revalidation
        # calls back through here, so live sessions drop on the next inbound
        # message too.
        if selected_agent_id is None:
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_ENTITY)
        ensure_widget_agent_available(
            db,
            selected_agent_id,
            user_id,
            expected_widget_key=widget_key
            if isinstance(widget_key, str) and widget_key
            else None,
        )

        return PublicChatAccessContext(
            user=user,
            channel_id=channel_id,
            guest_id=guest_id,
            auth_mode=auth_mode,
            widget_agent_id=selected_agent_id,
        )
    except Exception as exc:
        projected = _project_public_token_failure(
            exc, invalid_detail="Invalid widget token"
        )
        if projected is not None:
            raise projected from None
        raise


def get_share_chat_user(token: str, db: Session) -> ShareChatAccessContext:
    """Get share chat access context from a share token."""
    try:
        payload = _decode_public_token(token)
        if payload.get("type") != "widget":
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_TOKEN)

        user_id = payload.get("user_id")
        auth_mode = payload.get("auth_mode")
        share_agent_id = payload.get("share_agent_id")
        share_workforce_id = payload.get("share_workforce_id")
        share_token = payload.get("share_token")
        guest_id = payload.get("guest_id")

        if auth_mode != "share":
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_TOKEN)
        if not _is_strict_int(user_id):
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_CLAIMS)
        if not isinstance(share_token, str) or not share_token:
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_CLAIMS)
        # Fail closed on tokens minted before per-guest isolation (#973): they
        # carry no guest_id, so they cannot be scoped to a single guest and are
        # rejected rather than silently granted the old no-isolation behavior.
        # A whitespace-only id is treated as absent (it could never match a
        # server-minted token_urlsafe value and must not pass as a real guest).
        if not isinstance(guest_id, str) or not guest_id.strip():
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_CLAIMS)

        if _is_strict_int(share_workforce_id):
            selected_agent_id = None
            selected_workforce_id = share_workforce_id
        elif _is_strict_int(share_agent_id):
            selected_agent_id = share_agent_id
            selected_workforce_id = None
        else:
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_ENTITY)

        dialect_name = str(db.get_bind().dialect.name)
        user_id = _require_bindable_public_id(user_id, dialect_name)
        if selected_workforce_id is not None:
            selected_workforce_id = _require_bindable_public_id(
                selected_workforce_id, dialect_name
            )
        else:
            selected_agent_id = _require_bindable_public_id(
                selected_agent_id, dialect_name
            )

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_CLAIMS)

        if selected_workforce_id is not None:
            workforce = ensure_share_workforce_available(
                db,
                selected_workforce_id,
                user_id,
                expected_share_token=share_token,
            )
            return ShareChatAccessContext(
                user=user,
                share_token=share_token,
                guest_id=guest_id,
                workforce=workforce,
            )

        if selected_agent_id is None:
            raise _PublicTokenRejected(_PublicTokenRejectionReason.INVALID_ENTITY)
        agent = ensure_share_agent_available(
            db,
            selected_agent_id,
            user_id,
            expected_share_token=share_token,
        )
        return ShareChatAccessContext(
            user=user, share_token=share_token, guest_id=guest_id, agent=agent
        )
    except Exception as exc:
        projected = _project_public_token_failure(
            exc, invalid_detail="Invalid share token"
        )
        if projected is not None:
            raise projected from None
        raise


security = HTTPBearer()


def build_public_chat_dependency(
    expected_auth_mode: str,
) -> Callable[..., PublicChatAccessContext]:
    def dependency(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db),
    ) -> PublicChatAccessContext:
        return get_public_chat_user(
            credentials.credentials, db, expected_auth_mode=expected_auth_mode
        )

    return dependency


def build_share_chat_dependency() -> Callable[..., ShareChatAccessContext]:
    def dependency(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db),
    ) -> ShareChatAccessContext:
        return get_share_chat_user(credentials.credentials, db)

    return dependency


def _get_task_for_workforce_widget_context(
    db: Session, task_id: int, access_context: PublicChatAccessContext
) -> Task:
    """Scope a widget guest to a workforce-widget task.

    Mirrors :func:`_get_task_for_workforce_share_context` but retains the
    widget channel's per-guest isolation: the task must belong to this guest
    (``guest_id``), target this workforce, and be backed by a real
    ``WorkforceRun`` row (so a guest cannot address an arbitrary task id).
    """
    widget_workforce_id = access_context.widget_workforce_id
    # Callers today only reach here behind an `if widget_workforce_id is not
    # None` guard, but raise explicitly rather than assert: `python -O` strips
    # asserts, and this is an auth boundary — a future unguarded caller must
    # fail closed, not fall through with widget_workforce_id=None.
    if widget_workforce_id is None:
        raise HTTPException(status_code=403, detail="Widget is unavailable")
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.user_id == access_context.user.id,
        )
        .first()
    )
    if not task:
        raise HTTPException(status_code=403, detail="Task not found or access denied")
    principal = InteractionPrincipal(
        kind="guest",
        user_id=int(access_context.user.id),
        is_admin=False,
        auth_mode="widget",
        widget_workforce_id=widget_workforce_id,
        guest_id=access_context.guest_id,
    )
    if not task_is_owned_by_public_principal(task, principal):
        if public_chat_identity_matches(task, principal):
            raise HTTPException(status_code=403, detail="Widget is unavailable")
        raise HTTPException(status_code=403, detail="Task not found or access denied")
    workforce_run_id = task.agent_config.get("workforce_run_id")
    if not isinstance(workforce_run_id, int):
        raise HTTPException(status_code=403, detail="Widget is unavailable")
    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task.id),
            WorkforceRun.workforce_id == widget_workforce_id,
        )
        .first()
    )
    if run is None:
        raise HTTPException(status_code=403, detail="Widget is unavailable")
    return task


def get_task_for_public_context(
    db: Session, task_id: int, access_context: PublicChatAccessContext
) -> Task:
    # Deliberately NOT routed through task_is_owned_by_public_principal
    # (task_interaction_service.py): the widget-agent branch below has no
    # auth_mode check today, unlike the other three ownership entry points
    # in this file, which the shared predicate does check on every
    # direction. Routing this branch through the predicate would be a
    # behavior change to a production authorization path -- a candidate
    # security issue (a widget guest whose self-chosen guest_id happens to
    # equal a share guest's server-minted one can otherwise reach a
    # share-mode task through this branch) is logged against that gap, not
    # fixed here (#1304): "old-path extract, don't refactor" -- fixing it
    # belongs to a change scoped to that behavior, not to this predicate
    # extraction.
    if access_context.widget_workforce_id is not None:
        return _get_task_for_workforce_widget_context(db, task_id, access_context)
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.user_id == access_context.user.id,
            Task.channel_id.is_(access_context.channel_id)
            if access_context.channel_id is None
            else Task.channel_id == access_context.channel_id,
        )
        .first()
    )
    if not task:
        raise HTTPException(status_code=403, detail="Task not found or access denied")
    if (
        not task.agent_config
        or task.agent_config.get("guest_id") != access_context.guest_id
    ):
        raise HTTPException(status_code=403, detail="Task not found or access denied")
    if (
        access_context.widget_agent_id is not None
        and int(task.agent_id or 0) != access_context.widget_agent_id
    ):
        raise HTTPException(status_code=403, detail="Widget access is unavailable")
    return task


def _get_task_for_workforce_share_context(
    db: Session, task_id: int, access_context: ShareChatAccessContext
) -> Task:
    workforce = access_context.workforce
    # Callers today only reach here behind an `if workforce is not None`
    # guard, but raise explicitly rather than assert: `python -O` strips
    # asserts, and this is an auth boundary — a future unguarded caller must
    # fail closed, not fall through with workforce=None.
    if workforce is None:
        raise HTTPException(status_code=403, detail="Share link is unavailable")
    workforce_id = int(workforce.id)
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.user_id == access_context.user.id,
        )
        .first()
    )
    if not task:
        raise HTTPException(status_code=403, detail="Task not found or access denied")
    principal = InteractionPrincipal(
        kind="guest",
        user_id=int(access_context.user.id),
        is_admin=False,
        auth_mode="share",
        share_workforce_id=workforce_id,
        guest_id=access_context.guest_id,
    )
    if not task_is_owned_by_public_principal(task, principal):
        if public_chat_identity_matches(task, principal):
            raise HTTPException(status_code=403, detail="Share link is unavailable")
        raise HTTPException(status_code=403, detail="Task not found or access denied")
    workforce_run_id = task.agent_config.get("workforce_run_id")
    if not isinstance(workforce_run_id, int):
        raise HTTPException(status_code=403, detail="Share link is unavailable")
    run = (
        db.query(WorkforceRun)
        .filter(
            WorkforceRun.id == workforce_run_id,
            WorkforceRun.task_id == int(task.id),
            WorkforceRun.workforce_id == workforce_id,
        )
        .first()
    )
    if run is None:
        raise HTTPException(status_code=403, detail="Share link is unavailable")
    return task


def get_task_for_share_context(
    db: Session, task_id: int, access_context: ShareChatAccessContext
) -> Task:
    if access_context.workforce is not None:
        return _get_task_for_workforce_share_context(db, task_id, access_context)
    agent = access_context.agent
    if agent is None:
        raise HTTPException(status_code=403, detail="Share link is unavailable")
    task = (
        db.query(Task)
        .filter(
            Task.id == task_id,
            Task.user_id == access_context.user.id,
            Task.agent_id == int(agent.id),
        )
        .first()
    )
    if not task:
        raise HTTPException(status_code=403, detail="Task not found or access denied")
    principal = InteractionPrincipal(
        kind="guest",
        user_id=int(access_context.user.id),
        is_admin=False,
        auth_mode="share",
        share_agent_id=int(agent.id),
        guest_id=access_context.guest_id,
    )
    if not task_is_owned_by_public_principal(task, principal):
        if public_chat_identity_matches(task, principal):
            raise HTTPException(status_code=403, detail="Share link is unavailable")
        raise HTTPException(status_code=403, detail="Task not found or access denied")
    return task


def _enforce_public_upload_storage_gate(db: Session, owner: User) -> None:
    """Refuse a public upload when the owner is at their storage limit (#973).

    Mirrors the KB ingest gate: the hook is a no-op in stock xagent and only
    the cloud app layer registers it. Charged to the share/widget entity
    OWNER — the same attribution as the run gate and the on-disk write, since
    the file consumes the owner's storage. Fails open on any error so quota
    infra problems never block uploads; raises 402 on a truthy reason.
    """
    try:
        from ..services.quota_hooks import check_storage_gate

        reason = check_storage_gate(db, getattr(owner, "id", None))
    except Exception:
        reason = None
    if reason:
        raise HTTPException(status_code=402, detail=reason)


async def upload_public_chat_files(
    *,
    file: UploadFile | None,
    files: list[UploadFile] | None,
    task_type: str,
    message: str,
    task_id: str | None,
    folder: str | None,
    access_context: PublicChatAccessContext,
    db: Session,
) -> Any:
    del message
    upload_items: list[UploadFile] = []
    if file is not None:
        upload_items.append(file)
    if files:
        upload_items.extend(files)

    if not upload_items:
        raise HTTPException(status_code=422, detail="No files provided")

    user_id = int(access_context.user.id)
    _enforce_public_upload_storage_gate(db, access_context.user)

    if not task_id:
        # A workforce widget session starts its first turn inside task
        # creation, so its opening-message attachments must be uploaded BEFORE
        # any task exists and then threaded in as selected_file_ids. Allow that
        # task-less upload only for workforce widget guests; the agent widget
        # path still requires an existing task (files ride the first WS
        # message), preserving its task_id-required contract. Mirrors
        # ``upload_share_chat_files``.
        if access_context.widget_workforce_id is None:
            raise HTTPException(status_code=400, detail="task_id is required")
        # Reachable by anyone holding the widget credential before any task
        # (hence any owner association) exists, so cap the per-request file
        # count to blunt the worst abuse case cheaply — mirroring the share
        # task-less path.
        if len(upload_items) > MAX_TASKLESS_SHARE_UPLOAD_FILES:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Too many files in one request "
                    f"(max {MAX_TASKLESS_SHARE_UPLOAD_FILES})"
                ),
            )
        if not release_db_connection_if_clean(db):
            raise HTTPException(
                status_code=503,
                detail="Upload authorization could not be finalized",
            )
        # Stamp the task-less provenance marker so orphan GC (#973) can reap
        # these rows if the guest never completes task creation, without a
        # coarse task_id-IS-NULL sweep touching other paths' unbound drafts.
        return await store_uploaded_files(
            upload_items=upload_items,
            task_type=task_type,
            task_id=None,
            folder=folder,
            user_id=user_id,
            single_file_mode=file is not None and (not files),
            upload_source=TASKLESS_SHARE_UPLOAD_SOURCE,
        )

    try:
        parsed_task_id = int(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid task_id") from exc
    get_task_for_public_context(db, parsed_task_id, access_context)
    if not release_db_connection_if_clean(db):
        raise HTTPException(
            status_code=503,
            detail="Upload authorization could not be finalized",
        )

    return await store_uploaded_files(
        upload_items=upload_items,
        task_type=task_type,
        task_id=task_id,
        folder=folder,
        user_id=user_id,
        single_file_mode=file is not None and (not files),
    )


async def upload_share_chat_files(
    *,
    file: UploadFile | None,
    files: list[UploadFile] | None,
    task_type: str,
    message: str,
    task_id: str | None,
    folder: str | None,
    access_context: ShareChatAccessContext,
    db: Session,
) -> Any:
    del message
    upload_items: list[UploadFile] = []
    if file is not None:
        upload_items.append(file)
    if files:
        upload_items.extend(files)

    if not upload_items:
        raise HTTPException(status_code=422, detail="No files provided")

    user_id = int(access_context.user.id)
    _enforce_public_upload_storage_gate(db, access_context.user)

    if not task_id:
        # A workforce share session starts its first turn inside task
        # creation, so its opening-message attachments must be uploaded
        # BEFORE any task exists and then threaded in as selected_file_ids.
        # Allow that task-less upload only for workforce guests; the agent
        # path still requires an existing task (files ride the first WS
        # message), preserving its task_id-required contract.
        if access_context.workforce is None:
            raise HTTPException(status_code=400, detail="task_id is required")
        # This branch is reachable by anyone holding the public share link
        # before any task (hence any owner association) exists, so cap the
        # per-request file count to blunt the worst abuse case cheaply.
        if len(upload_items) > MAX_TASKLESS_SHARE_UPLOAD_FILES:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Too many files in one request "
                    f"(max {MAX_TASKLESS_SHARE_UPLOAD_FILES})"
                ),
            )
        if not release_db_connection_if_clean(db):
            raise HTTPException(
                status_code=503,
                detail="Upload authorization could not be finalized",
            )
        # Stamp the task-less-share provenance marker so orphan GC (#973) can
        # reap these rows if the guest never completes task creation, without
        # a coarse task_id-IS-NULL sweep touching other paths' unbound drafts.
        return await store_uploaded_files(
            upload_items=upload_items,
            task_type=task_type,
            task_id=None,
            folder=folder,
            user_id=user_id,
            single_file_mode=file is not None and (not files),
            upload_source=TASKLESS_SHARE_UPLOAD_SOURCE,
        )

    try:
        parsed_task_id = int(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid task_id") from exc
    get_task_for_share_context(db, parsed_task_id, access_context)
    if not release_db_connection_if_clean(db):
        raise HTTPException(
            status_code=503,
            detail="Upload authorization could not be finalized",
        )

    return await store_uploaded_files(
        upload_items=upload_items,
        task_type=task_type,
        task_id=task_id,
        folder=folder,
        user_id=user_id,
        single_file_mode=file is not None and (not files),
    )


async def _create_workforce_widget_chat_task(
    *,
    request: TaskCreateRequest,
    access_context: PublicChatAccessContext,
    db: Session,
    client_ip: str | None,
) -> TaskCreateResponse:
    """Guest task creation for a widget-embedded workforce.

    Like the workforce share path, a workforce session must enter through
    ``create_workforce_run`` (which pins the config snapshot, creates the
    ``WorkforceRun`` record, and begins the first turn) rather than the agent
    widget path's bare PENDING task. ``guest_id`` is threaded into the config
    so the widget channel keeps its per-guest task isolation.
    """
    widget_workforce_id = access_context.widget_workforce_id
    # Auth boundary: fail closed rather than assert (`python -O` strips
    # asserts), matching the share-channel sibling.
    if widget_workforce_id is None:
        raise HTTPException(status_code=403, detail="Widget is unavailable")
    if request.agent_id is not None:
        raise HTTPException(status_code=403, detail="Widget is unavailable")

    workforce = db.query(Workforce).filter(Workforce.id == widget_workforce_id).first()
    if workforce is None:
        raise HTTPException(status_code=403, detail="Widget is unavailable")

    result = await create_workforce_run(
        db,
        access_context.user,
        workforce,
        message=request.description or "",
        selected_file_ids=request.files,
        timezone=request.timezone,
        source="widget",
        is_visible=False,
        extra_agent_config={
            FILE_OPERATION_ACCESS_VERSION_KEY: FILE_OPERATION_ACCESS_VERSION,
            "auth_mode": "widget",
            "widget_workforce_id": int(workforce.id),
            # Null the agent marker for symmetry with the agent path (#1108):
            # entity_rate_limit_key prefers workforce, so a stray agent id
            # would be ignored anyway, but stamping both keeps the run-quota
            # key unambiguous and independent of merge ordering.
            "widget_agent_id": None,
            "guest_id": access_context.guest_id,
            # Server-observed creator IP (#1108): the per-abuser key for the
            # widget run quota. Stamped by the backend, never client-supplied.
            "widget_client_ip": client_ip,
        },
    )
    task = result.task
    return TaskCreateResponse(
        task_id=task.id,
        title=task.title,
        status=task.status.value,
        created_at=format_datetime_for_api(task.created_at)
        if task.created_at
        else None,
        channel_id=task.channel_id,
        channel_name=task.channel_name,
        connector_runtime_requirements=_NO_CONNECTOR_RUNTIME_REQUIREMENTS,
    )


async def create_public_chat_task(
    *,
    request: TaskCreateRequest,
    access_context: PublicChatAccessContext,
    db: Session,
    default_channel_name: str,
    # Required (no default): the server-observed creator IP is the widget run
    # quota's per-abuser key (#1108), so a caller must consciously pass it —
    # ``None`` only for a genuinely IP-less path, never by omission. A silent
    # default would degrade the quota to entity-only without anything failing.
    client_ip: str | None,
) -> TaskCreateResponse:
    if request.runtime_extensions:
        raise HTTPException(
            status_code=400,
            detail="Task runtime extensions are not supported for public widget tasks.",
        )
    if access_context.widget_workforce_id is not None:
        return await _create_workforce_widget_chat_task(
            request=request, access_context=access_context, db=db, client_ip=client_ip
        )

    task_description = request.description or ""

    channel = (
        db.query(UserChannel)
        .filter(UserChannel.id == access_context.channel_id)
        .first()
    )
    channel_name = channel.channel_name if channel else default_channel_name

    # Sanitize the client dict *before* the server keys go on: the strip can
    # then never reach a server-assigned value, and the server keys still win
    # on collision.
    agent_config = sanitize_client_agent_config(request.agent_config)
    agent_config["guest_id"] = access_context.guest_id
    agent_config["auth_mode"] = "widget"
    agent_config[FILE_OPERATION_ACCESS_VERSION_KEY] = FILE_OPERATION_ACCESS_VERSION
    # Server-observed creator IP (#1108): the per-abuser key for the widget
    # run quota. Stamped by the backend, never client-supplied.
    agent_config["widget_client_ip"] = client_ip
    # Stamp BOTH entity markers from the validated access context, writing the
    # inapplicable one as None (#1108). The run quota keys off these and
    # entity_rate_limit_key prefers workforce, so a client-injected
    # widget_workforce_id must never survive to redirect the bucket. The
    # sanitizer already strips both, but stamping explicitly keeps this correct
    # independent of the sanitizer's reserved-key set.
    agent_config["widget_agent_id"] = access_context.widget_agent_id
    agent_config["widget_workforce_id"] = access_context.widget_workforce_id

    agent_id = request.agent_id
    if agent_id is None and channel and channel.config:
        agent_id = channel.config.get("agent_id")
    if access_context.widget_agent_id is not None:
        if agent_id is None:
            agent_id = access_context.widget_agent_id
        elif agent_id != access_context.widget_agent_id:
            raise HTTPException(status_code=403, detail="Widget access is unavailable")
    agent = db.query(Agent).filter(Agent.id == agent_id).first() if agent_id else None
    if agent is None and access_context.widget_agent_id is not None:
        logger.info(
            "Widget task create could not load agent %s; connector runtime "
            "selection will be empty",
            access_context.widget_agent_id,
        )
    task_title = request.title or task_description or "Untitled Task"
    if task_title and len(task_title) > 50:
        task_title = task_title[:50] + "..."

    task = Task(
        user_id=access_context.user.id,
        title=task_title,
        description=task_description,
        status=TaskStatus.PENDING,
        channel_id=access_context.channel_id,
        channel_name=channel_name,
        agent_id=agent_id,
        agent_config=agent_config,
        source="widget",
        is_visible=False,
    )
    selected_refs = prepare_connector_runtime_selection_snapshot(
        db=db,
        agent=agent,
        connector_user_id=int(access_context.user.id),
    )
    bind_connector_runtime_selection_snapshot(task=task, selected_refs=selected_refs)

    db.add(task)
    db.commit()
    db.refresh(task)

    return TaskCreateResponse(
        task_id=task.id,
        title=task.title,
        status=task.status.value,
        created_at=format_datetime_for_api(task.created_at)
        if task.created_at
        else None,
        channel_id=task.channel_id,
        channel_name=task.channel_name,
        connector_runtime_requirements=_NO_CONNECTOR_RUNTIME_REQUIREMENTS,
    )


async def _create_workforce_share_chat_task(
    *,
    request: TaskCreateRequest,
    access_context: ShareChatAccessContext,
    db: Session,
) -> TaskCreateResponse:
    """Guest task creation for a shared workforce.

    Unlike the agent share path (which creates a bare PENDING task and lets
    the first WS chat message start the turn), a workforce session must enter
    through ``create_workforce_run``: it pins the config snapshot, creates the
    ``WorkforceRun`` record, and begins the first turn with the guest's
    message — so ``request.description`` doubles as the opening message.

    Opening-message attachments are uploaded task-lessly by the client first
    (there is no task yet), then passed here as ``request.files`` so the run
    binds them to its task and the first turn actually sees them.
    """
    workforce = access_context.workforce
    # Auth boundary: fail closed rather than assert (stripped under -O).
    if workforce is None:
        raise HTTPException(status_code=403, detail="Share link is unavailable")
    if request.agent_id is not None:
        raise HTTPException(status_code=403, detail="Share link is unavailable")

    result = await create_workforce_run(
        db,
        access_context.user,
        workforce,
        message=request.description or "",
        selected_file_ids=request.files,
        timezone=request.timezone,
        source="shared_link",
        is_visible=False,
        extra_agent_config={
            FILE_OPERATION_ACCESS_VERSION_KEY: FILE_OPERATION_ACCESS_VERSION,
            "auth_mode": "share",
            "share_workforce_id": int(workforce.id),
            # Null the agent marker for symmetry with the agent path (#1132),
            # mirroring the widget workforce branch: entity_rate_limit_key
            # prefers workforce, so a stray agent id would be ignored anyway,
            # but stamping both keeps the run-quota key unambiguous. The None
            # is a literal by the context's exactly-one invariant, enforced in
            # ShareChatAccessContext.__post_init__ (#1225). Safe also because
            # the snapshot-built config never sets either marker -- not
            # because merge order is irrelevant: _merge_agent_config returns
            # {**extra_agent_config, **task_config}, so the built config wins
            # every collision and this dict loses.
            "share_agent_id": None,
            # Per-guest isolation (#973). extra_agent_config is overlaid under
            # the snapshot-built config, which never sets guest_id, so this is
            # preserved; the run-critical workforce_run_id is added by the
            # snapshot config and still wins on any collision.
            "guest_id": access_context.guest_id,
        },
    )
    task = result.task
    return TaskCreateResponse(
        task_id=task.id,
        title=task.title,
        status=task.status.value,
        created_at=format_datetime_for_api(task.created_at)
        if task.created_at
        else None,
        channel_id=task.channel_id,
        channel_name=task.channel_name,
        connector_runtime_requirements=_NO_CONNECTOR_RUNTIME_REQUIREMENTS,
    )


async def create_share_chat_task(
    *,
    request: TaskCreateRequest,
    access_context: ShareChatAccessContext,
    db: Session,
    default_channel_name: str,
) -> TaskCreateResponse:
    if request.runtime_extensions:
        raise HTTPException(
            status_code=400,
            detail="Task runtime extensions are not supported for shared-link tasks.",
        )
    if access_context.workforce is not None:
        return await _create_workforce_share_chat_task(
            request=request, access_context=access_context, db=db
        )
    if access_context.agent is None:
        raise HTTPException(status_code=403, detail="Share link is unavailable")

    task_description = request.description or ""

    agent_id = request.agent_id
    share_agent_id = int(access_context.agent.id)
    if agent_id is None:
        agent_id = share_agent_id
    elif agent_id != share_agent_id:
        raise HTTPException(status_code=403, detail="Share link is unavailable")

    # The client dict is sanitized of server-owned reserved keys first, then
    # the server keys are assigned on top so they win on collision — a
    # client-supplied guest_id can never override the server-minted one carried
    # on the validated access context (#973).
    agent_config = sanitize_client_agent_config(request.agent_config)
    agent_config["auth_mode"] = "share"
    agent_config[FILE_OPERATION_ACCESS_VERSION_KEY] = FILE_OPERATION_ACCESS_VERSION
    agent_config["guest_id"] = access_context.guest_id
    # Stamp BOTH entity markers from the validated access context, writing the
    # inapplicable one as None — the share-path counterpart of the widget
    # stamping above (#1132). The share run quota keys off these at the
    # execute_task chokepoint (chat.py) and entity_rate_limit_key prefers
    # workforce, so a client-injected share_workforce_id must never survive to
    # redirect an agent-share task's bucket onto a workforce. The sanitizer
    # already strips both, but stamping explicitly keeps this correct
    # independent of the sanitizer's reserved-key set. The workforce marker is
    # a literal None by the context's exactly-one invariant, enforced in
    # ShareChatAccessContext.__post_init__ (#1225).
    agent_config["share_agent_id"] = share_agent_id
    agent_config["share_workforce_id"] = None

    task_title = request.title or task_description or "Untitled Task"
    if task_title and len(task_title) > 50:
        task_title = task_title[:50] + "..."

    task = Task(
        user_id=access_context.user.id,
        title=task_title,
        description=task_description,
        status=TaskStatus.PENDING,
        channel_id=None,
        channel_name=default_channel_name,
        agent_id=share_agent_id,
        agent_config=agent_config,
        source="shared_link",
        is_visible=False,
    )
    selected_refs = prepare_connector_runtime_selection_snapshot(
        db=db,
        agent=access_context.agent,
        connector_user_id=int(access_context.user.id),
    )
    bind_connector_runtime_selection_snapshot(task=task, selected_refs=selected_refs)

    db.add(task)
    db.commit()
    db.refresh(task)

    return TaskCreateResponse(
        task_id=task.id,
        title=task.title,
        status=task.status.value,
        created_at=format_datetime_for_api(task.created_at)
        if task.created_at
        else None,
        channel_id=task.channel_id,
        channel_name=task.channel_name,
        connector_runtime_requirements=_NO_CONNECTOR_RUNTIME_REQUIREMENTS,
    )


def widget_entity_key(context: PublicChatAccessContext) -> str | None:
    """Rate-limit key for the widget entity a guest is scoped to (#1056).

    Matches the ``allow_widget_upload`` keying: the widget ``guest_id`` is
    client-supplied (rotatable at will), so widget throttles key on the
    embedded agent/workforce instead. Shared with the widget HTTP endpoints
    (#1108) so every gate derives the key from the access context the same
    way.
    """
    return entity_rate_limit_key(context.widget_agent_id, context.widget_workforce_id)


def _authorize_chat_websocket_sync(
    *,
    load_access_context: Callable[[Session], _ChatAccessContextT],
    authorize_task: Callable[[Session, _ChatAccessContextT], object],
    widget_entity_key: Callable[[_ChatAccessContextT], str | None] | None = None,
) -> WebSocketPrincipal:
    """Authorize one public WebSocket operation in a worker-owned Session."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        access_context = load_access_context(db)
        authorize_task(db, access_context)
        user_id = access_context.user.id
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid user")
        return WebSocketPrincipal(
            id=int(user_id),
            is_admin=bool(access_context.user.is_admin),
            guest_id=access_context.guest_id,
            widget_entity_key=widget_entity_key(access_context)
            if widget_entity_key is not None
            else None,
        )


async def _authorize_public_chat_websocket(
    *,
    token: str,
    task_id: int,
    expected_auth_mode: str,
) -> WebSocketPrincipal:
    """Validate widget access without blocking the asyncio event loop."""

    def load_access_context(db: Session) -> PublicChatAccessContext:
        return get_public_chat_user(
            token,
            db,
            expected_auth_mode=expected_auth_mode,
        )

    def authorize_task(db: Session, context: PublicChatAccessContext) -> object:
        return get_task_for_public_context(db, task_id, context)

    return await run_db_io_cancellation_safe(
        lambda: _authorize_chat_websocket_sync(
            load_access_context=load_access_context,
            authorize_task=authorize_task,
            widget_entity_key=widget_entity_key,
        )
    )


async def _authorize_share_chat_websocket(
    *,
    token: str,
    task_id: int,
) -> WebSocketPrincipal:
    """Validate share-link access without blocking the asyncio event loop."""

    def load_access_context(db: Session) -> ShareChatAccessContext:
        return get_share_chat_user(token, db)

    def authorize_task(db: Session, context: ShareChatAccessContext) -> object:
        return get_task_for_share_context(db, task_id, context)

    return await run_db_io_cancellation_safe(
        lambda: _authorize_chat_websocket_sync(
            load_access_context=load_access_context,
            authorize_task=authorize_task,
        )
    )


_WS_CLOSE_REASON_MAX_BYTES = 123


def _ws_close_reason(detail: object) -> str:
    """Coerce an ``HTTPException.detail`` into a valid WS close reason.

    ``detail`` is typed ``Any`` in FastAPI while the WS protocol caps close
    reasons at 123 UTF-8 bytes — an over-long or non-string value would blow
    up mid-close and swallow the denial the frontend recovery keys on. Every
    detail reachable today is a short string literal; this is insurance for
    the future one that isn't. Truncates on a codepoint boundary.
    """
    text = str(detail) if detail is not None else ""
    encoded = text.encode("utf-8")
    if len(encoded) <= _WS_CLOSE_REASON_MAX_BYTES:
        return text
    return encoded[:_WS_CLOSE_REASON_MAX_BYTES].decode("utf-8", errors="ignore")


async def _reject_rate_limited_turn(websocket: WebSocket, message_data: dict) -> None:
    """Reject one rate-limited run-starting turn, keeping the socket open.

    A rate limit is transient, so the turn is refused in-band (the client
    surfaces it and can retry) rather than closing the connection.
    """
    client_message_id = message_data.get("client_message_id")
    rate_limited_code = ClientErrorCode.MESSAGE_RATE_LIMITED
    rate_limited_message = client_error_message(rate_limited_code)
    await send_message_delivery(
        websocket,
        client_message_id=client_message_id,
        turn_id=str(client_message_id or ""),
        accepted=False,
        message=rate_limited_message,
        error_code=rate_limited_code.value,
        rejection_outcome="not_accepted",
    )
    # send_message_delivery no-ops without a client_message_id, so a client
    # that didn't tag the turn would otherwise get zero feedback and see the
    # message silently dropped. Fall back to a generic error so the throttle
    # is always surfaced.
    if client_message_id is None:
        await manager.send_personal_message(
            {
                "type": "error",
                "error_code": rate_limited_code.value,
                "message": rate_limited_message,
            },
            websocket,
        )


async def public_chat_websocket_endpoint(
    *,
    websocket: WebSocket,
    task_id: int,
    token: str = Query(..., description="Authentication token"),
    expected_auth_mode: str,
) -> None:
    """Serve widget websocket chat with per-message revalidation."""
    # Handshake budget (#1056): mirror of the share connect gate — a widget
    # key is embedded in a public page, so connection attempts are anonymous
    # and need a per-IP ceiling. Checked *pre-accept* on purpose: an over-limit
    # probe is refused as a plain HTTP 403 with no upgrade performed at all,
    # which is the cheapest rejection — and unlike the auth denials below, a
    # rate-limited connect carries no reason the frontend recovery flow needs
    # to read. The IP is captured once — it cannot change for the lifetime of
    # the socket — and reused by the per-turn gate in the loop below.
    remote_ip = remote_ip_from_request(websocket)
    if not get_share_rate_limiter().allow_widget_ws_connect(remote_ip):
        await websocket.close(code=4008, reason="Too many connection attempts")
        return

    # Accept the handshake *before* the auth/access checks (#1057), matching the
    # share path. uvicorn only preserves a close code + reason for a *post-accept*
    # close; a pre-accept ``websocket.close()`` collapses into a bare HTTP 403
    # (browser sees onclose code=1006, reason=""), discarding the denial. Accepting
    # first makes the 4003 below a real close whose reason reaches the widget
    # client, so a returning guest whose widget was revoked can tell an access
    # denial apart from a transport failure and recover.
    await websocket.accept()
    try:
        principal = await _authorize_public_chat_websocket(
            token=token,
            task_id=task_id,
            expected_auth_mode=expected_auth_mode,
        )
    except HTTPException as exc:
        await websocket.close(code=4003, reason=_ws_close_reason(exc.detail))
        return
    except Exception as exc:
        await send_websocket_authentication_infrastructure_failure(websocket, exc)
        return

    # Already accepted above; register the live socket without re-accepting.
    manager.register_connection(websocket, task_id)

    try:
        await handle_status_request(websocket, task_id, principal)

        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            try:
                current_principal = await _authorize_public_chat_websocket(
                    token=token,
                    task_id=task_id,
                    expected_auth_mode=expected_auth_mode,
                )
            except HTTPException as exc:
                await websocket.close(code=4003, reason=_ws_close_reason(exc.detail))
                return
            except Exception as exc:
                await send_websocket_authentication_infrastructure_failure(
                    websocket, exc
                )
                return

            message_data["user_id"] = current_principal.id
            message_data["user"] = current_principal

            message_type = message_data.get("type")
            # Abuse control (#1056): follow-up turns bypass any HTTP throttle
            # and each starts an owner-billed run, so rate-limit the
            # run-starting turn types here — the widget mirror of the share
            # turn gate. Keyed on the widget entity + caller IP, NOT the
            # guest: the widget guest_id is client-supplied (rotatable at
            # will). Interventions are control messages, not new runs, so
            # they are not gated. A missing entity key (unreachable for a
            # validated widget token, which always carries exactly one entity
            # id) degrades to the limiter's shared "unknown" bucket rather
            # than skipping the gate — the per-IP bucket needs no entity, so
            # the throttle must not fail open on the key.
            if message_type in (
                "chat",
                "execute_task",
            ) and not get_share_rate_limiter().allow_widget_ws_turn(
                current_principal.widget_entity_key, remote_ip
            ):
                await _reject_rate_limited_turn(websocket, message_data)
                continue

            if message_type == "chat":
                await handle_chat_message(websocket, task_id, message_data)
            elif message_type == "execute_task":
                await handle_execute_task(websocket, task_id, message_data)
            elif message_type == "intervention":
                await handle_intervention(websocket, task_id, message_data)
    except Exception as exc:
        from fastapi import WebSocketDisconnect

        if isinstance(exc, WebSocketDisconnect):
            logger.info("Public chat WebSocket disconnected: %s", exc)
        else:
            logger.error("Public chat WebSocket error: %s", exc)
    finally:
        manager.disconnect(websocket)


async def share_chat_websocket_endpoint(
    *,
    websocket: WebSocket,
    task_id: int,
    token: str = Query(..., description="Authentication token"),
) -> None:
    """Serve share websocket chat with per-message revalidation."""
    # Handshake budget (#973): accept-before-auth (below) means every connect
    # attempt — valid or garbage token — completes a full 101 upgrade before
    # rejection, so the attempts themselves need a per-IP ceiling. Checked
    # *pre-accept* on purpose: an over-limit probe is refused as a plain HTTP
    # 403 with no upgrade performed at all, which is the cheapest rejection —
    # unlike the auth denials below, a rate-limited connect carries no reason
    # the frontend recovery flow needs to read.
    if not get_share_rate_limiter().allow_ws_connect(remote_ip_from_request(websocket)):
        await websocket.close(code=4008, reason="Too many connection attempts")
        return

    # Accept the handshake *before* the auth/access checks. uvicorn only
    # preserves a close code + reason on the wire for a *post-accept* close: a
    # pre-accept ``websocket.close()`` is collapsed into a bare HTTP 403 with an
    # empty body, so the browser sees a rejected Upgrade (onerror + onclose
    # code=1006, reason="") and the code/reason never arrive. Accepting first
    # makes the 4003 below a real close whose reason reaches the client — the
    # same post-accept behavior the per-message revalidation loop already relies
    # on — so the frontend can tell an access denial (e.g. a returning guest
    # whose persisted task belongs to another guest, or a pre-#973 legacy task)
    # apart from a transport failure and recover (#973).
    await websocket.accept()
    try:
        principal = await _authorize_share_chat_websocket(
            token=token,
            task_id=task_id,
        )
    except HTTPException as exc:
        await websocket.close(code=4003, reason=_ws_close_reason(exc.detail))
        return
    except Exception as exc:
        await send_websocket_authentication_infrastructure_failure(websocket, exc)
        return

    # Already accepted above; register the live socket without re-accepting.
    manager.register_connection(websocket, task_id)

    try:
        await handle_status_request(websocket, task_id, principal)

        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            try:
                current_principal = await _authorize_share_chat_websocket(
                    token=token,
                    task_id=task_id,
                )
            except HTTPException as exc:
                await websocket.close(code=4003, reason=_ws_close_reason(exc.detail))
                return
            except Exception as exc:
                await send_websocket_authentication_infrastructure_failure(
                    websocket, exc
                )
                return

            message_data["user_id"] = current_principal.id
            message_data["user"] = current_principal

            message_type = message_data.get("type")
            # Abuse control (#973): follow-up turns bypass the HTTP task-create
            # limiter and each starts an owner-billed run, so rate-limit the
            # run-starting turn types per guest here. Reject the turn (the
            # client surfaces it and can retry) rather than closing — a rate
            # limit is transient. Interventions are control messages, not new
            # runs, so they are not gated.
            if (
                message_type in ("chat", "execute_task")
                and current_principal.guest_id
                and not get_share_rate_limiter().allow_ws_turn(
                    current_principal.guest_id
                )
            ):
                await _reject_rate_limited_turn(websocket, message_data)
                continue

            if message_type == "chat":
                await handle_chat_message(websocket, task_id, message_data)
            elif message_type == "execute_task":
                await handle_execute_task(websocket, task_id, message_data)
            elif message_type == "intervention":
                await handle_intervention(websocket, task_id, message_data)
    except Exception as exc:
        from fastapi import WebSocketDisconnect

        if isinstance(exc, WebSocketDisconnect):
            logger.info("Share chat WebSocket disconnected: %s", exc)
        else:
            logger.error("Share chat WebSocket error: %s", exc)
    finally:
        manager.disconnect(websocket)
