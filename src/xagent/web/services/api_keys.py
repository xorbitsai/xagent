"""API key services for SDK runtime and management credentials."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, NamedTuple

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from ...core.utils.api_key import ApiKeyKind, generate_api_key
from ..models.agent import Agent, AgentOrigin
from ..models.agent_api_key import AgentApiKey
from ..models.user_api_key import UserApiKey
from ..schemas.agent_api_key import (
    AgentApiKeyListItem,
    AgentApiKeyStats,
    APIKeyGenerateResponse,
    APIKeyMetadataResponse,
    APIKeyRevokeResponse,
)
from ..schemas.user_api_key import (
    PersonalAPIKeyCreateResponse,
    PersonalAPIKeyMetadata,
    PersonalAPIKeyRevokeResponse,
)
from .agent_team_scope import get_agent_team_scope, owned_agent_clause

logger = logging.getLogger(__name__)


class KeyRotationConflict(RuntimeError):
    """Raised when a concurrent key rotation wins the active-key race."""


def _key_status(row: AgentApiKey) -> str:
    if row.revoked_at is not None:
        return "revoked"
    if row.paused_at is not None:
        return "paused"
    return "active"


def _masked_key(row: AgentApiKey) -> str:
    return f"xag_{row.key_prefix}_••••••••"


class AgentApiKeyService:
    """Owns agent runtime API key rotation and metadata."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def stage_rotated_key(self, agent_id: int) -> tuple[AgentApiKey, str]:
        """Revoke the active key (if any) and stage a new one, then flush.

        Does NOT commit -- the caller owns the transaction boundary. This
        is the composable building block (mirror of
        ``AgentStore.add_agent``): single-step callers go through
        :meth:`rotate_key` which commits, while multi-step workflows
        (create-agent-with-key) stage this plus other writes and commit
        once at the outer boundary.

        Returns the staged ORM row and the one-shot plaintext key. The
        row's ``created_at`` is only populated after the caller commits
        and refreshes.
        """
        now = datetime.now(timezone.utc)
        # Bulk-revoke rather than ``.filter(...).first()``: an agent can now
        # hold more than one simultaneously-active key (via the multi-key
        # admin endpoints), so "rotate" must invalidate all of them, not
        # just the first row this query happens to return.
        revoked_count = (
            self.db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
            )
            .update(
                {AgentApiKey.revoked_at: now, AgentApiKey.updated_at: now},
                synchronize_session=False,
            )
        )

        full_key, key_prefix, key_hash = generate_api_key(
            self.db, kind=ApiKeyKind.AGENT
        )
        new_row = AgentApiKey(
            agent_id=agent_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
        )
        self.db.add(new_row)
        self.db.flush()
        logger.info(
            "Staged runtime API key for agent %s (prefix=%s, revoked=%d)",
            agent_id,
            key_prefix,
            revoked_count,
        )
        return new_row, full_key

    def rotate_key(self, agent_id: int) -> APIKeyGenerateResponse:
        """Single-step rotate: stage + commit. Used by the JWT-gated
        ``/api/agents/{id}/api-key`` endpoint, which owns its own
        transaction: returns a one-shot key, commits itself, and maps a
        unique-index race to :class:`KeyRotationConflict`.

        Staging and commit share one ``try`` so an IntegrityError is
        translated whether it surfaces at the staging flush or at commit.
        The only remaining trigger is a ``key_prefix`` collision (the
        old ``uq_agent_api_keys_agent_active`` partial unique index this
        used to also catch was dropped when multi-key support landed --
        an agent may now hold more than one active key, so concurrent
        rotations of the *same* agent no longer race on that constraint;
        each just revokes-then-inserts independently).
        """
        try:
            new_row, full_key = self.stage_rotated_key(agent_id)
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            raise KeyRotationConflict(str(exc)) from exc

        self.db.refresh(new_row)
        return APIKeyGenerateResponse(
            full_key=full_key,
            key_prefix=new_row.key_prefix,
            created_at=new_row.created_at,
        )

    def get_metadata(self, agent_id: int) -> APIKeyMetadataResponse | None:
        # An agent can now have more than one active key (via the
        # multi-key admin endpoints), so this legacy "the active key"
        # view needs its own tiebreak: most-recently-created wins, and a
        # paused key is excluded so it's never surfaced here as "active"
        # (mirrors the auth dependency's paused == invalid treatment).
        row = (
            self.db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
                AgentApiKey.paused_at.is_(None),
            )
            .order_by(AgentApiKey.created_at.desc(), AgentApiKey.id.desc())
            .first()
        )
        if row is None:
            return None
        return APIKeyMetadataResponse(
            key_prefix=row.key_prefix,
            masked_key=f"xag_{row.key_prefix}_••••••••",
            created_at=row.created_at,
        )

    def revoke_key(self, agent_id: int) -> APIKeyRevokeResponse:
        now = datetime.now(timezone.utc)
        updated_count = (
            self.db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
            )
            .update(
                {AgentApiKey.revoked_at: now, AgentApiKey.updated_at: now},
                synchronize_session=False,
            )
        )
        if updated_count == 0:
            return APIKeyRevokeResponse(revoked=False, revoked_at=None)

        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        logger.info("Revoked runtime API key for agent %s", agent_id)
        return APIKeyRevokeResponse(revoked=True, revoked_at=now)

    # ===== Multi-key admin operations (centralized "API Keys" page) =====
    #
    # Unlike the rotate/get/revoke trio above -- which enforce "at most one
    # active key" by construction -- these let a caller hold any number of
    # simultaneously-active keys per agent. All are scoped by
    # ``owned_agent_clause`` so a caller can only list/mutate keys for agents
    # they own or co-own within their team, mirroring
    # ``_get_owned_agent_or_404`` in ``api/agents.py``.

    def _owned_agents_query(self, user_id: int) -> Any:
        return self.db.query(Agent.id).filter(
            owned_agent_clause(user_id, get_agent_team_scope(self.db, user_id)),
            Agent.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        )

    def create_key(
        self, user_id: int, agent_id: int, label: str | None
    ) -> tuple[AgentApiKeyListItem, str] | None:
        """Add a new key for ``agent_id`` without touching existing ones.

        Returns ``(list_item, full_key)`` where ``full_key`` is the
        one-shot plaintext, or ``None`` if the agent doesn't exist / isn't
        owned by ``user_id`` (the caller should map that to 404).
        """
        agent = (
            self.db.query(Agent)
            .filter(Agent.id == agent_id)
            .filter(Agent.id.in_(self._owned_agents_query(user_id)))
            .first()
        )
        if agent is None:
            return None

        full_key, key_prefix, key_hash = generate_api_key(
            self.db, kind=ApiKeyKind.AGENT
        )
        row = AgentApiKey(
            agent_id=agent_id,
            label=label,
            key_prefix=key_prefix,
            key_hash=key_hash,
        )
        self.db.add(row)
        try:
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            raise KeyRotationConflict(str(exc)) from exc
        self.db.refresh(row)
        logger.info(
            "Created API key for agent %s (prefix=%s, label=%r)",
            agent_id,
            key_prefix,
            label,
        )
        return self._to_list_item(row, str(agent.name)), full_key

    def _to_list_item(self, row: AgentApiKey, agent_name: str) -> AgentApiKeyListItem:
        return AgentApiKeyListItem(
            id=int(row.id),
            agent_id=int(row.agent_id),
            agent_name=agent_name,
            label=row.label,
            key_prefix=row.key_prefix,
            masked_key=_masked_key(row),
            status=_key_status(row),
            last_used_at=row.last_used_at,
            created_at=row.created_at,
        )

    def list_keys_for_user(
        self, user_id: int, agent_id: int | None = None
    ) -> list[AgentApiKeyListItem]:
        query = (
            self.db.query(AgentApiKey, Agent.name)
            .join(Agent, Agent.id == AgentApiKey.agent_id)
            .filter(Agent.id.in_(self._owned_agents_query(user_id)))
        )
        if agent_id is not None:
            query = query.filter(AgentApiKey.agent_id == agent_id)
        rows = query.order_by(
            AgentApiKey.created_at.desc(), AgentApiKey.id.desc()
        ).all()
        return [self._to_list_item(row, agent_name) for row, agent_name in rows]

    def get_stats_for_user(self, user_id: int) -> AgentApiKeyStats:
        # Aggregate in SQL rather than loading every row into memory --
        # revoked keys are kept forever as an audit trail, so this table
        # only grows for an active user.
        owned = self._owned_agents_query(user_id)
        current_month = datetime.now(timezone.utc).strftime("%Y-%m")
        base_filter = AgentApiKey.agent_id.in_(owned)

        total_keys = (
            self.db.query(func.count(AgentApiKey.id)).filter(base_filter).scalar() or 0
        )
        active_keys = (
            self.db.query(func.count(AgentApiKey.id))
            .filter(
                base_filter,
                AgentApiKey.revoked_at.is_(None),
                AgentApiKey.paused_at.is_(None),
            )
            .scalar()
            or 0
        )
        calls_this_month = (
            self.db.query(func.sum(AgentApiKey.usage_month_calls))
            .filter(base_filter, AgentApiKey.usage_month == current_month)
            .scalar()
            or 0
        )
        last_api_call = (
            self.db.query(func.max(AgentApiKey.last_used_at))
            .filter(base_filter)
            .scalar()
        )
        return AgentApiKeyStats(
            total_keys=total_keys,
            active_keys=active_keys,
            calls_this_month=calls_this_month,
            last_api_call=last_api_call,
        )

    def _find_owned_key(self, user_id: int, key_id: int) -> AgentApiKey | None:
        # joinedload avoids an N+1 lazy-load: every caller of this method
        # (pause_key/resume_key/regenerate_key) reads ``row.agent.name``
        # to build the response.
        return (
            self.db.query(AgentApiKey)
            .options(joinedload(AgentApiKey.agent))
            .filter(AgentApiKey.id == key_id)
            .filter(AgentApiKey.agent_id.in_(self._owned_agents_query(user_id)))
            .first()
        )

    def pause_key(self, user_id: int, key_id: int) -> AgentApiKeyListItem | None:
        row = self._find_owned_key(user_id, key_id)
        if row is None:
            return None
        if row.paused_at is None and row.revoked_at is None:
            row.paused_at = datetime.now(timezone.utc)
            self.db.commit()
            self.db.refresh(row)
        return self._to_list_item(row, row.agent.name)

    def resume_key(self, user_id: int, key_id: int) -> AgentApiKeyListItem | None:
        row = self._find_owned_key(user_id, key_id)
        if row is None:
            return None
        if row.paused_at is not None:
            row.paused_at = None  # type: ignore[assignment]
            self.db.commit()
            self.db.refresh(row)
        return self._to_list_item(row, row.agent.name)

    def regenerate_key(
        self, user_id: int, key_id: int
    ) -> tuple[AgentApiKeyListItem, str] | None:
        """Issue a new secret for an existing key row, keeping id/label/status.

        Returns ``(list_item, full_key)`` where ``full_key`` is the
        one-shot plaintext, or ``None`` if the key doesn't exist, isn't
        owned by ``user_id``, or has been revoked (a revoked key is a
        dead row -- regenerating it would hand back a secret that the
        auth dependency, which excludes revoked keys, would still 401 on).
        """
        row = self._find_owned_key(user_id, key_id)
        if row is None or row.revoked_at is not None:
            return None

        full_key, key_prefix, key_hash = generate_api_key(
            self.db, kind=ApiKeyKind.AGENT
        )
        row.key_prefix = key_prefix
        row.key_hash = key_hash
        # Deliberately NOT touching paused_at -- regenerate swaps the
        # secret only, it doesn't implicitly resume a paused key.
        try:
            self.db.commit()
        except IntegrityError as exc:
            self.db.rollback()
            raise KeyRotationConflict(str(exc)) from exc
        self.db.refresh(row)
        logger.info("Regenerated API key %s (prefix=%s)", key_id, key_prefix)
        return self._to_list_item(row, row.agent.name), full_key

    def delete_key(self, user_id: int, key_id: int) -> bool:
        """Soft-revoke a key by id. Returns False if not found/not owned."""
        row = self._find_owned_key(user_id, key_id)
        if row is None:
            return False
        if row.revoked_at is None:
            now = datetime.now(timezone.utc)
            row.revoked_at = now
            row.updated_at = now
            self.db.commit()
        logger.info("Deleted (revoked) API key %s", key_id)
        return True


class PersonalKeySecret(NamedTuple):
    full_key: str
    key_prefix: str
    key_hash: str


class UserApiKeyService:
    """Owns personal management API keys."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_key(self, user_id: int) -> PersonalAPIKeyCreateResponse:
        full_key, key_prefix, key_hash = generate_api_key(
            self.db, kind=ApiKeyKind.PERSONAL
        )
        row = UserApiKey(
            user_id=user_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
        )
        self.db.add(row)
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self.db.refresh(row)
        logger.info(
            "Created personal API key for user %s (prefix=%s)", user_id, key_prefix
        )
        return PersonalAPIKeyCreateResponse(
            id=int(row.id),
            full_key=full_key,
            key_prefix=row.key_prefix,
            created_at=row.created_at,
            expires_at=row.expires_at,
        )

    def list_keys(self, user_id: int) -> list[PersonalAPIKeyMetadata]:
        rows = (
            self.db.query(UserApiKey)
            .filter(UserApiKey.user_id == user_id)
            .order_by(UserApiKey.created_at.desc())
            .all()
        )
        return [
            PersonalAPIKeyMetadata(
                id=int(row.id),
                key_prefix=row.key_prefix,
                masked_key=f"xag_personal_{row.key_prefix}_••••••••",
                revoked_at=row.revoked_at,
                expires_at=row.expires_at,
                created_at=row.created_at,
            )
            for row in rows
        ]

    def revoke_key(self, user_id: int, key_id: int) -> PersonalAPIKeyRevokeResponse:
        row = (
            self.db.query(UserApiKey)
            .filter(UserApiKey.id == key_id, UserApiKey.user_id == user_id)
            .first()
        )
        if row is None:
            return PersonalAPIKeyRevokeResponse(revoked=False, revoked_at=None)
        if row.revoked_at is not None:
            return PersonalAPIKeyRevokeResponse(
                revoked=False, revoked_at=row.revoked_at
            )

        now = datetime.now(timezone.utc)
        row.revoked_at = now
        row.updated_at = now
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        self.db.refresh(row)
        logger.info("Revoked personal API key %s for user %s", key_id, user_id)
        return PersonalAPIKeyRevokeResponse(revoked=True, revoked_at=row.revoked_at)
