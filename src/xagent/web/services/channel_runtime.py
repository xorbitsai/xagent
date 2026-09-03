"""Detached database boundaries shared by asynchronous chat transports.

Telegram, Feishu, and Slack spend most of a turn awaiting network, file,
sandbox, and agent work. A transport must therefore never retain a SQLAlchemy
``Session`` or an attached ORM row for the lifetime of that turn. This module
owns the short worker-side transactions required by these transports and
returns only frozen primitive snapshots.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ...config import get_default_task_execution_mode
from ...core.file_storage.keys import build_task_output_storage_key
from ...core.workspace import WorkspaceFileRegistration
from ..models.agent import Agent, AgentOrigin
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..models.user_channel import UserChannel
from .agent_team_scope import get_agent_team_scope, owned_agent_clause
from .chat_history_service import persist_user_message
from .connector_runtime import (
    bind_connector_runtime_selection_snapshot,
    prepare_connector_runtime_selection_snapshot,
)
from .db_runtime import await_task_settlement, run_db_io_cancellation_safe
from .managed_task_lease import (
    ManagedTaskLease,
    finalize_managed_task_lease_result,
    start_managed_task_lease,
)
from .mcp_runtime import MCPBuiltinOAuthActorPolicyRequiredError
from .task_lease_service import TaskLease, acquire_task_lease_no_commit, utc_now
from .task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_IDENTITY_KEY,
    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY,
)
from .task_runtime import (
    mcp_runtime_authorization_policy_required as task_requires_mcp_actor_policy,
)
from .uploaded_file_store import (
    StagedUploadedFile,
    UploadedFileStore,
    compensate_staged_uploaded_files,
    stage_uploaded_file_from_local_path,
)
from .workforce_runtime import sync_workforce_run_status

logger = logging.getLogger(__name__)

TELEGRAM_TASK_LIST_LIMIT = 50
ACTOR_TASK_SOURCE = "external"


class ChannelTaskMode(StrEnum):
    """The trusted task-selection contract for one channel turn."""

    DEFAULT = "default"
    ACTOR_INTERACTION = "actor_interaction"


class ChannelConfigurationError(RuntimeError):
    """The configured channel has no resolvable owner."""


class ChannelAuthorizationError(RuntimeError):
    """The external sender is not permitted by the channel configuration."""


@dataclass(frozen=True)
class ChannelOwnerSnapshot:
    """Detached owner identity after channel authorization."""

    user_id: int


@dataclass(frozen=True)
class ChannelAgentSnapshot:
    """Detached Agent Builder identity selectable from a channel."""

    agent_id: int
    name: str


@dataclass(frozen=True)
class ClaimedChannelTask:
    """Detached channel turn identity with an already-managed exact lease."""

    user_id: int
    task_id: int
    is_new_task: bool
    managed_lease: ManagedTaskLease
    requested_agent_missing: bool = False


@dataclass(frozen=True)
class _ChannelTaskClaimSnapshot:
    """Worker result before the event loop starts the lease heartbeat."""

    user_id: int
    task_id: int
    is_new_task: bool
    lease: TaskLease
    requested_agent_missing: bool = False
    # The status this claim took the task away from, for a claim that is
    # resuming an existing run rather than starting a new one. None for every
    # fresh claim.
    #
    # Deliberately not derived from ``is_new_task``: an actor interaction that
    # passes no resume run id also reuses an existing row, so that flag cannot
    # tell "resuming a waiting run" from "re-claiming a row for a new run".
    # Compensation reads this to decide whether an abandoned claim should be
    # put back where it was or failed -- see ``_compensate_channel_task_claim_sync``.
    resumed_from_status: TaskStatus | None = None


@dataclass(frozen=True)
class DownloadedChannelFile:
    """A transport download that has not yet been registered in the database."""

    name: str
    path: Path
    mime_type: str
    size: int
    source_id: str | None = None


@dataclass(frozen=True)
class ChannelUploadedFile:
    """Detached result of durable upload and workspace registration."""

    file_id: str
    name: str
    path: str
    mime_type: str
    size: int
    source_id: str | None = None

    def to_file_info(self, *, source_key: str | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "file_id": self.file_id,
            "name": self.name,
            "path": self.path,
            "type": self.mime_type,
            "size": self.size,
        }
        if source_key is not None and self.source_id is not None:
            info[source_key] = self.source_id
        return info


@dataclass(frozen=True)
class ChannelOutputFile:
    """Detached metadata required to send a generated file to a transport."""

    file_id: str
    filename: str
    storage_path: str
    mime_type: str


@dataclass(frozen=True)
class TelegramChannelTaskSnapshot:
    """Detached Telegram task metadata safe to render outside a Session."""

    task_id: int
    title: str
    status: str
    created_at: Any
    updated_at: Any
    # The agent this task is bound to, or None for the default assistant.
    # A switch must adopt it: prepare_channel_task() discards a task whose
    # agent_id does not match the caller's selection.
    agent_id: int | None = None


@dataclass(frozen=True)
class ChannelConfigSnapshot:
    """Detached active-channel configuration consumed by bot managers."""

    channel_id: int
    channel_name: str
    config_items: tuple[tuple[str, str], ...]

    def config_value(self, key: str) -> str | None:
        return dict(self.config_items).get(key)


def _load_active_channel_configs_sync(
    *,
    channel_type: str,
    required_config_keys: Sequence[str],
    optional_config_keys: Sequence[str] = (),
) -> tuple[ChannelConfigSnapshot, ...]:
    SessionLocal = get_session_local()
    snapshots: list[ChannelConfigSnapshot] = []
    with SessionLocal() as db:
        channels = (
            db.query(UserChannel)
            .filter(
                UserChannel.channel_type == channel_type,
                UserChannel.is_active.is_(True),
            )
            # Deterministic ordering so downstream keyed maps (for example
            # Slack's team_id routing table) resolve duplicates stably
            # instead of by arbitrary row order.
            .order_by(UserChannel.id)
            .all()
        )
        for channel in channels:
            config = channel.config
            resolved = {
                key: str(config.get(key))
                for key in required_config_keys
                if config.get(key)
            }
            if len(resolved) != len(required_config_keys):
                continue
            for key in optional_config_keys:
                if config.get(key):
                    resolved[key] = str(config.get(key))
            snapshots.append(
                ChannelConfigSnapshot(
                    channel_id=int(channel.id),
                    channel_name=str(channel.channel_name),
                    config_items=tuple(sorted(resolved.items())),
                )
            )
    return tuple(snapshots)


async def load_active_channel_configs(
    *,
    channel_type: str,
    required_config_keys: Sequence[str],
    optional_config_keys: Sequence[str] = (),
) -> tuple[ChannelConfigSnapshot, ...]:
    """Load active bot credentials without blocking the asyncio event loop."""

    return await run_db_io_cancellation_safe(
        lambda: _load_active_channel_configs_sync(
            channel_type=channel_type,
            required_config_keys=required_config_keys,
            optional_config_keys=optional_config_keys,
        )
    )


def deactivate_channel_sync(
    *,
    channel_id: int,
    clear_config_keys: Sequence[str] = (),
) -> bool:
    """Deactivate one channel row, optionally dropping dead credentials.

    Used when the remote platform reports the integration is gone (for
    example Slack's ``app_uninstalled``/``tokens_revoked`` events), so the
    row stops advertising a connection that no longer works.
    """
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        channel = db.query(UserChannel).filter(UserChannel.id == channel_id).first()
        if channel is None:
            return False
        changed = False
        if bool(channel.is_active):
            channel.is_active = False  # type: ignore[assignment]
            changed = True
        if clear_config_keys:
            config = dict(channel.config)
            for key in clear_config_keys:
                if config.pop(key, None) is not None:
                    changed = True
            channel.config = config
        if changed:
            db.commit()
        return changed


def _load_channel_owner_sync(
    db: Any,
    *,
    channel_id: int | None,
    external_user_id: str,
) -> ChannelOwnerSnapshot:
    if channel_id is None:
        raise ChannelConfigurationError("Channel owner is not configured")

    channel = (
        db.query(UserChannel)
        .filter(
            UserChannel.id == int(channel_id),
            UserChannel.is_active.is_(True),
        )
        .first()
    )
    if channel is None:
        raise ChannelConfigurationError("Channel owner is not configured")

    owner_id = int(channel.user_id)
    owner_exists = db.query(User.id).filter(User.id == owner_id).first() is not None
    if not owner_exists:
        raise ChannelConfigurationError("Channel owner is not configured")

    allowed_users = channel.config.get("allowed_users")
    if allowed_users is not None:
        # config is an unconstrained JSON dict, so the allowlist may be a bare
        # string. Iterating one yields its characters, which would authorize
        # "1" and deny the intended "101" -- treat it as a single entry.
        if isinstance(allowed_users, str):
            allowed_users = [allowed_users]
        elif isinstance(allowed_users, dict):
            # Pre-PR the `in` check matched dict keys; keep that working
            # rather than locking out a channel configured that way.
            allowed_users = list(allowed_users)
        elif not isinstance(allowed_users, (list, tuple, set)):
            raise ChannelConfigurationError(
                "Channel allowed_users must be a list of sender ids"
            )
        # Ids may be numeric in config while sender ids are always strings.
        allowed_user_ids = {str(value) for value in allowed_users}
        if external_user_id not in allowed_user_ids:
            raise ChannelAuthorizationError("Channel sender is not authorized")
    return ChannelOwnerSnapshot(user_id=owner_id)


def _claim_legacy_active_telegram_task_sync(
    db: Any,
    *,
    channel: UserChannel,
    owner_id: int,
    external_user_id: str,
    active_task_id: int | None,
) -> bool:
    """Claim the legacy active task whose per-sender mapping proves ownership."""

    if active_task_id is None or active_task_id <= 0:
        return False
    active_task = (
        db.query(Task)
        .filter(
            Task.id == active_task_id,
            Task.user_id == owner_id,
            Task.channel_id == channel.id,
            Task.telegram_user_id.is_(None),
        )
        .first()
    )
    if active_task is None:
        return False
    active_task.telegram_user_id = external_user_id
    return True


def _resolve_telegram_sender_scope_sync(
    db: Any,
    *,
    channel_id: int | None,
    external_user_id: str,
    active_task_id: int | None,
) -> tuple[int, UserChannel]:
    """Authorize the sender and settle any legacy claim before task queries.

    Shared by the list and single-task loaders, which differ only in the query
    they run afterwards. Returns the owner id and the resolved channel.

    This is a *write* path: a pre-migration task whose ownership the sender's
    active-task mapping proves is stamped and committed here. Read-only-looking
    callers such as /list still need it, because without the stamp the sender's
    own history stays invisible to every subsequent query -- so the write is
    what makes the read correct.
    """

    owner = _load_channel_owner_sync(
        db,
        channel_id=channel_id,
        external_user_id=external_user_id,
    )
    channel = (
        db.query(UserChannel)
        .filter(
            UserChannel.id == channel_id,
            UserChannel.channel_type == "telegram",
            UserChannel.is_active.is_(True),
        )
        .first()
    )
    if channel is None:
        raise ChannelConfigurationError("Telegram channel is not configured")

    claimed_legacy_task = _claim_legacy_active_telegram_task_sync(
        db,
        channel=channel,
        owner_id=owner.user_id,
        external_user_id=external_user_id,
        active_task_id=active_task_id,
    )
    if claimed_legacy_task:
        db.commit()
    return owner.user_id, channel


def _telegram_task_snapshot(task: Task) -> TelegramChannelTaskSnapshot:
    return TelegramChannelTaskSnapshot(
        task_id=int(task.id),
        title=str(task.title or "Untitled Task"),
        status=str(getattr(task.status, "value", task.status) or "unknown"),
        created_at=task.created_at,
        updated_at=task.updated_at,
        agent_id=int(task.agent_id) if task.agent_id is not None else None,
    )


def _load_telegram_channel_tasks_sync(
    *,
    channel_id: int | None,
    external_user_id: str,
    active_task_id: int | None,
) -> tuple[TelegramChannelTaskSnapshot, ...]:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        # Writes: claims the legacy task the mapping proves, so the sender's
        # pre-migration history becomes visible to the query below.
        owner_id, channel = _resolve_telegram_sender_scope_sync(
            db,
            channel_id=channel_id,
            external_user_id=external_user_id,
            active_task_id=active_task_id,
        )
        rows = (
            db.query(Task)
            .filter(
                Task.user_id == owner_id,
                Task.channel_id == channel.id,
                Task.telegram_user_id == external_user_id,
                Task.is_visible.is_(True),
            )
            .order_by(Task.updated_at.desc(), Task.created_at.desc(), Task.id.desc())
            .limit(TELEGRAM_TASK_LIST_LIMIT)
            .all()
        )
        return tuple(_telegram_task_snapshot(task) for task in rows)


async def load_telegram_channel_tasks(
    *,
    channel_id: int | None,
    external_user_id: str,
    active_task_id: int | None,
) -> tuple[TelegramChannelTaskSnapshot, ...]:
    """List one Telegram sender's visible tasks without leaking ORM state."""

    return await run_db_io_cancellation_safe(
        lambda: _load_telegram_channel_tasks_sync(
            channel_id=channel_id,
            external_user_id=external_user_id,
            active_task_id=active_task_id,
        )
    )


def _load_telegram_channel_task_sync(
    *,
    channel_id: int | None,
    external_user_id: str,
    task_id: int,
    active_task_id: int | None,
) -> TelegramChannelTaskSnapshot | None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        # Writes: claims the legacy task the mapping proves, so the sender's
        # pre-migration history becomes visible to the query below.
        owner_id, channel = _resolve_telegram_sender_scope_sync(
            db,
            channel_id=channel_id,
            external_user_id=external_user_id,
            active_task_id=active_task_id,
        )
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.user_id == owner_id,
                Task.channel_id == channel.id,
                Task.telegram_user_id == external_user_id,
                Task.is_visible.is_(True),
            )
            .first()
        )
        return _telegram_task_snapshot(task) if task is not None else None


async def load_telegram_channel_task(
    *,
    channel_id: int | None,
    external_user_id: str,
    task_id: int,
    active_task_id: int | None,
) -> TelegramChannelTaskSnapshot | None:
    """Resolve a switch target inside the Telegram sender boundary."""

    return await run_db_io_cancellation_safe(
        lambda: _load_telegram_channel_task_sync(
            channel_id=channel_id,
            external_user_id=external_user_id,
            task_id=task_id,
            active_task_id=active_task_id,
        )
    )


def _authorize_channel_sender_sync(
    *,
    channel_id: int | None,
    external_user_id: str,
) -> ChannelOwnerSnapshot:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return _load_channel_owner_sync(
            db,
            channel_id=channel_id,
            external_user_id=external_user_id,
        )


async def authorize_channel_sender(
    *,
    channel_id: int | None,
    external_user_id: str,
) -> ChannelOwnerSnapshot:
    return await run_db_io_cancellation_safe(
        lambda: _authorize_channel_sender_sync(
            channel_id=channel_id,
            external_user_id=external_user_id,
        )
    )


def _owned_channel_agents_query(db: Any, owner_id: int) -> Any:
    # Mirrors AgentStore.list_agent_items: agents the owner manages, with
    # workforce-generated manager agents excluded like every other external
    # channel (share/widget/api-keys/triggers).
    return db.query(Agent).filter(
        owned_agent_clause(owner_id, get_agent_team_scope(db, owner_id)),
        Agent.origin != AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
    )


def _list_channel_owner_agents_sync(
    *,
    channel_id: int | None,
    external_user_id: str,
) -> tuple[ChannelAgentSnapshot, ...]:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        owner = _load_channel_owner_sync(
            db,
            channel_id=channel_id,
            external_user_id=external_user_id,
        )
        agents = (
            _owned_channel_agents_query(db, owner.user_id)
            .order_by(Agent.created_at.desc())
            .all()
        )
        return tuple(
            ChannelAgentSnapshot(agent_id=int(agent.id), name=str(agent.name))
            for agent in agents
        )


async def list_channel_owner_agents(
    *,
    channel_id: int | None,
    external_user_id: str,
) -> tuple[ChannelAgentSnapshot, ...]:
    """List the channel owner's selectable Agent Builder agents."""

    return await run_db_io_cancellation_safe(
        lambda: _list_channel_owner_agents_sync(
            channel_id=channel_id,
            external_user_id=external_user_id,
        )
    )


def _get_channel_owner_agent_sync(
    *,
    channel_id: int | None,
    external_user_id: str,
    agent_id: int,
) -> ChannelAgentSnapshot | None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        owner = _load_channel_owner_sync(
            db,
            channel_id=channel_id,
            external_user_id=external_user_id,
        )
        agent = (
            _owned_channel_agents_query(db, owner.user_id)
            .filter(Agent.id == int(agent_id))
            .first()
        )
        if agent is None:
            return None
        return ChannelAgentSnapshot(agent_id=int(agent.id), name=str(agent.name))


async def get_channel_owner_agent(
    *,
    channel_id: int | None,
    external_user_id: str,
    agent_id: int,
) -> ChannelAgentSnapshot | None:
    """Resolve one selectable agent, or ``None`` when not owner-visible."""

    return await run_db_io_cancellation_safe(
        lambda: _get_channel_owner_agent_sync(
            channel_id=channel_id,
            external_user_id=external_user_id,
            agent_id=agent_id,
        )
    )


def _actor_interaction_task_matches(
    task: Task | None,
    agent: Agent | None,
    *,
    owner_id: int,
    agent_id: int,
    status: TaskStatus,
) -> bool:
    return bool(
        agent is not None
        and task is not None
        and int(task.user_id) == owner_id
        and task.agent_id is not None
        and int(task.agent_id) == agent_id
        and int(agent.id) == agent_id
        and task.source == ACTOR_TASK_SOURCE
        and task.status == status
        and task.is_visible is False
        and task_requires_mcp_actor_policy(task.agent_config)
    )


def _actor_interaction_claim_predicates(
    *,
    owner_id: int,
    agent_id: int,
) -> tuple[Any, ...]:
    now = utc_now()
    return (
        Task.user_id == owner_id,
        Task.agent_id == agent_id,
        Task.source == ACTOR_TASK_SOURCE,
        Task.is_visible.is_(False),
        Task.agent_config[MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY]
        .as_boolean()
        .is_(True),
        # WAITING_FOR_USER can be visible before the prior run releases its
        # lease. Do not let the same process overwrite that live lease.
        or_(
            Task.runner_id.is_(None),
            Task.lease_expires_at.is_(None),
            Task.lease_expires_at < now,
        ),
    )


def _load_actor_interaction_task(
    db: Any,
    *,
    owner_id: int,
    active_task_id: int,
    agent_id: int,
) -> tuple[Task, Agent]:
    """Load one exact waiting actor task without a fresh-task fallback."""

    agent = (
        _owned_channel_agents_query(db, owner_id).filter(Agent.id == agent_id).first()
    )
    task = (
        db.query(Task)
        .filter(
            Task.id == active_task_id,
            Task.user_id == owner_id,
        )
        .first()
    )
    if not _actor_interaction_task_matches(
        task,
        agent,
        owner_id=owner_id,
        agent_id=agent_id,
        status=TaskStatus.WAITING_FOR_USER,
    ):
        raise ChannelAuthorizationError("The actor interaction task is unavailable")

    assert task is not None
    assert agent is not None
    return task, agent


def _prepare_channel_task_sync(
    *,
    channel_id: int | None,
    external_user_id: str,
    active_task_id: int | None,
    text: str,
    channel_name: str | None,
    expected_owner_user_id: int | None,
    agent_id: int | None = None,
    new_task_is_visible: bool = True,
    mcp_runtime_authorization_policy_required: bool = False,
    mcp_runtime_authorization_policy_identity: str | None = None,
    task_mode: ChannelTaskMode = ChannelTaskMode.DEFAULT,
    resume_run_id: str | None = None,
) -> _ChannelTaskClaimSnapshot | None:
    if not isinstance(task_mode, ChannelTaskMode):
        raise ValueError("Unsupported channel task mode")
    if task_mode is ChannelTaskMode.ACTOR_INTERACTION:
        if active_task_id is None or active_task_id <= 0:
            raise ValueError("Actor interaction requires an active task")
        if expected_owner_user_id is None:
            raise ValueError("Actor interaction requires an expected owner")
        if agent_id is None:
            raise ValueError("Actor interaction requires an agent")
        if not mcp_runtime_authorization_policy_required:
            raise ValueError("Actor interaction requires an actor policy")

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        try:
            owner = _load_channel_owner_sync(
                db,
                channel_id=channel_id,
                external_user_id=external_user_id,
            )
            owner_id = owner.user_id
            channel = db.query(UserChannel).filter(UserChannel.id == channel_id).first()
            if channel is None:
                raise ChannelConfigurationError("Channel owner is not configured")
            if (
                expected_owner_user_id is not None
                and owner_id != expected_owner_user_id
            ):
                raise ChannelAuthorizationError(
                    "Channel owner changed during sender authorization"
                )

            is_telegram = str(channel.channel_type) == "telegram"
            if is_telegram and task_mode is ChannelTaskMode.DEFAULT:
                claimed_legacy_task = _claim_legacy_active_telegram_task_sync(
                    db,
                    channel=channel,
                    owner_id=owner_id,
                    external_user_id=external_user_id,
                    active_task_id=active_task_id,
                )
                if claimed_legacy_task:
                    # Sessions run with autoflush=False, so the resume query
                    # below cannot see the pending sender stamp without this
                    # flush and would abandon the legacy task for a new one.
                    # Flush rather than commit: the claim must stay atomic with
                    # this turn's task creation so both roll back together.
                    db.flush()

            task = None
            agent_row = None
            requested_agent_missing = False
            if task_mode is ChannelTaskMode.ACTOR_INTERACTION:
                assert active_task_id is not None
                assert agent_id is not None
                task, agent_row = _load_actor_interaction_task(
                    db,
                    owner_id=owner_id,
                    active_task_id=active_task_id,
                    agent_id=agent_id,
                )
            else:
                # Actor-owned channel turns are fresh-only. Ignoring an active
                # id on the default marked path prevents it from becoming an
                # accidental continuation API.
                if mcp_runtime_authorization_policy_required is True:
                    active_task_id = None

            if (
                task_mode is ChannelTaskMode.DEFAULT
                and active_task_id is not None
                and active_task_id != -1
            ):
                active_config = (
                    db.query(Task.agent_config)
                    .filter(
                        Task.id == active_task_id,
                        Task.user_id == owner_id,
                    )
                    .scalar()
                )
                if task_requires_mcp_actor_policy(active_config):
                    raise MCPBuiltinOAuthActorPolicyRequiredError(
                        f"Task {active_task_id} is actor-marked; channel reuse is unsupported"
                    )
                query = db.query(Task).filter(
                    Task.id == active_task_id,
                    Task.user_id == owner_id,
                )
                if is_telegram:
                    query = query.filter(
                        Task.channel_id == channel_id,
                        Task.telegram_user_id == external_user_id,
                        # Same visibility boundary as /list and /switch: a task
                        # hidden in the web UI must stop resuming here too,
                        # rather than silently continuing to execute.
                        #
                        # Deliberately Telegram-only: it pairs with the
                        # sender-scoped /list + /switch surface that only
                        # Telegram has. Extending it to Feishu would change
                        # Feishu's resume behavior for hidden tasks in a
                        # Telegram feature change; do that consciously with the
                        # generic channel identity work, not here.
                        Task.is_visible.is_(True),
                    )
                task = query.first()
                if task is not None and task_requires_mcp_actor_policy(
                    task.agent_config
                ):
                    raise MCPBuiltinOAuthActorPolicyRequiredError(
                        f"Task {int(task.id)} is actor-marked; channel reuse is unsupported"
                    )

            # Revalidate the requested selection on every ordinary turn. Actor
            # interaction mode uses the stricter exact-task check above.
            if task_mode is ChannelTaskMode.DEFAULT and agent_id is not None:
                agent_row = (
                    _owned_channel_agents_query(db, owner_id)
                    .filter(Agent.id == int(agent_id))
                    .first()
                )
                requested_agent_missing = agent_row is None
                if task is not None:
                    if agent_row is None:
                        # Evict to a clean default task; never fail the turn.
                        task = None
                    elif task.agent_id is None or int(task.agent_id) != int(
                        agent_row.id
                    ):
                        task = None

            is_new_task = task is None
            if task is None:
                task_agent_id = int(agent_row.id) if agent_row is not None else None
                task_title = text or "Untitled Task"
                if len(task_title) > 50:
                    task_title = f"{task_title[:50]}..."
                task = Task(
                    user_id=owner_id,
                    title=task_title,
                    description=text,
                    status=TaskStatus.PENDING,
                    execution_mode=get_default_task_execution_mode(
                        agent_id=task_agent_id
                    ),
                    channel_id=channel_id,
                    channel_name=channel_name,
                    telegram_user_id=external_user_id if is_telegram else None,
                    agent_id=task_agent_id,
                    is_visible=(
                        False
                        if mcp_runtime_authorization_policy_required is True
                        else new_task_is_visible
                    ),
                    agent_config=(
                        {
                            MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True,
                            **(
                                {
                                    MCP_RUNTIME_AUTHORIZATION_POLICY_IDENTITY_KEY: (
                                        mcp_runtime_authorization_policy_identity
                                    )
                                }
                                if mcp_runtime_authorization_policy_identity
                                else {}
                            ),
                        }
                        if mcp_runtime_authorization_policy_required is True
                        else None
                    ),
                )
                selected_refs = prepare_connector_runtime_selection_snapshot(
                    db=db,
                    agent=agent_row,
                    connector_user_id=owner_id,
                )
                bind_connector_runtime_selection_snapshot(
                    task=task, selected_refs=selected_refs
                )
                db.add(task)
                db.flush()

            task_id = int(task.id)
            claim_predicates = ()
            if task_mode is ChannelTaskMode.ACTOR_INTERACTION:
                assert agent_id is not None
                claim_predicates = _actor_interaction_claim_predicates(
                    owner_id=owner_id,
                    agent_id=agent_id,
                )

            # A resume has to land in the *same* run the waiting checkpoint
            # was written under. `new_run=True` mints a fresh run id and nulls
            # both checkpoint pointer columns (acquire_task_lease_no_commit),
            # which leaves the pending question unreadable in the new run's
            # partition -- the agent then replans from scratch instead of
            # handing the answer back to the tool call that asked. Claiming
            # with the caller's own run id keeps the pointers, which is what
            # the websocket resume path already relies on.
            #
            # Only ever the run id the caller *read off the waiting task*: this
            # is a resume of an existing run, so a caller that cannot name that
            # run has nothing to resume and must take the fresh-run path.
            resuming = resume_run_id is not None
            # Read before the claim, which flips the row to RUNNING: after it,
            # the status this claim interrupted is no longer on the row to
            # read. Only captured for a resume -- a fresh claim compensates as
            # FAILED and has nothing to restore.
            resumed_from_status = (
                _resumable_prior_status(db, task_id) if resuming else None
            )
            lease = acquire_task_lease_no_commit(
                db,
                task_id,
                expected_status=(
                    TaskStatus.WAITING_FOR_USER
                    if task_mode is ChannelTaskMode.ACTOR_INTERACTION
                    else None
                ),
                expected_run_id=resume_run_id,
                new_run=not resuming,
                claim_predicates=claim_predicates,
            )
            if lease is None:
                db.rollback()
                return None

            # Keep task state, workforce projection, and the exact run lease in
            # one transaction. No transport can observe the newly-created
            # PENDING row before its RUNNING owner is durable.
            db.expire_all()
            claimed_task = db.query(Task).filter(Task.id == task_id).one()
            if task_mode is ChannelTaskMode.ACTOR_INTERACTION:
                assert agent_id is not None
                claimed_agent = (
                    _owned_channel_agents_query(db, owner_id)
                    .filter(Agent.id == agent_id)
                    .with_for_update()
                    .first()
                )
                if not _actor_interaction_task_matches(
                    claimed_task,
                    claimed_agent,
                    owner_id=owner_id,
                    agent_id=agent_id,
                    status=TaskStatus.RUNNING,
                ):
                    db.rollback()
                    return None
            sync_workforce_run_status(db, claimed_task, TaskStatus.RUNNING)
            db.commit()
            return _ChannelTaskClaimSnapshot(
                user_id=owner_id,
                task_id=task_id,
                is_new_task=is_new_task,
                lease=lease,
                requested_agent_missing=requested_agent_missing,
                resumed_from_status=resumed_from_status,
            )
        except Exception:
            db.rollback()
            raise


async def prepare_channel_task(
    *,
    channel_id: int | None,
    external_user_id: str,
    active_task_id: int | None,
    text: str,
    channel_name: str | None,
    expected_owner_user_id: int | None = None,
    agent_id: int | None = None,
    new_task_is_visible: bool = True,
    mcp_runtime_authorization_policy_required: bool = False,
    mcp_runtime_authorization_policy_identity: str | None = None,
    task_mode: ChannelTaskMode = ChannelTaskMode.DEFAULT,
    resume_run_id: str | None = None,
) -> ClaimedChannelTask | None:
    """Authorize, resolve or create, and claim one channel run atomically.

    The trusted actor path sets both ``new_task_is_visible=False`` and
    ``mcp_runtime_authorization_policy_required=True``. That marker is written
    in the creation transaction before the returned lease can be executed.

    ``resume_run_id`` claims an existing run instead of starting a new one, so
    the run's checkpoint pointers survive the claim and a waiting execution can
    actually be resumed. Pass the run id read off the waiting task; omitting it
    keeps the default fresh-run claim, which is what every non-resume turn
    wants.
    """

    worker = asyncio.create_task(
        asyncio.to_thread(
            _prepare_channel_task_sync,
            channel_id=channel_id,
            external_user_id=external_user_id,
            active_task_id=active_task_id,
            text=text,
            channel_name=channel_name,
            expected_owner_user_id=expected_owner_user_id,
            agent_id=agent_id,
            new_task_is_visible=new_task_is_visible,
            mcp_runtime_authorization_policy_required=(
                mcp_runtime_authorization_policy_required
            ),
            mcp_runtime_authorization_policy_identity=(
                mcp_runtime_authorization_policy_identity
            ),
            task_mode=task_mode,
            resume_run_id=resume_run_id,
        )
    )
    snapshot, cancellation = await await_task_settlement(worker)
    if cancellation is not None:
        if snapshot is not None:
            try:
                compensated = await run_db_io_cancellation_safe(
                    lambda: _compensate_channel_task_claim_sync(snapshot)
                )
                if not compensated:
                    logger.warning(
                        "Channel task %s claim changed before cancelled "
                        "preparation could settle it; preserving current state",
                        snapshot.task_id,
                    )
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "Failed to compensate channel task %s after cancelled "
                    "preparation; retaining TTL recovery",
                    snapshot.task_id,
                )
        raise cancellation

    if snapshot is None:
        return None

    try:
        managed_lease = start_managed_task_lease(snapshot.lease)
    except Exception:
        try:
            await run_db_io_cancellation_safe(
                lambda: _compensate_channel_task_claim_sync(snapshot)
            )
        except Exception:
            logger.exception(
                "Failed to compensate channel task %s after heartbeat startup "
                "failed; retaining TTL recovery",
                snapshot.task_id,
            )
        raise

    return ClaimedChannelTask(
        user_id=snapshot.user_id,
        task_id=snapshot.task_id,
        is_new_task=snapshot.is_new_task,
        managed_lease=managed_lease,
        requested_agent_missing=snapshot.requested_agent_missing,
    )


def _resumable_prior_status(db: Session, task_id: int) -> TaskStatus | None:
    """Return the status a resume claim should restore, or None.

    Only WAITING_FOR_USER is restorable. Any other status means the row was
    not parked on a question when this claim reached it -- there is no
    pending interaction to hand back to, so an abandoned claim there is an
    ordinary failed turn rather than a resume to undo.
    """
    row = db.query(Task.status).filter(Task.id == task_id).first()
    if row is None:
        return None
    status = row[0]
    return status if status is TaskStatus.WAITING_FOR_USER else None


def _compensate_channel_task_claim_sync(
    snapshot: _ChannelTaskClaimSnapshot,
) -> bool:
    """Settle the exact committed claim or leave its lease to TTL recovery.

    A fresh claim that is abandoned before execution is a turn that never
    ran, and FAILED is the honest record of it.

    A resumed claim is not: the task was parked on a question, and the answer
    was never injected, so nothing about the pending interaction has changed.
    Failing it there would be destructive rather than merely inaccurate --
    the loader and the next claim both require WAITING_FOR_USER, so a task
    failed here can never be resumed again and the approval it was waiting on
    becomes permanently unanswerable, checkpoint and all. Put it back where
    the claim found it instead, which is what every other resume path
    (websocket, A2A, V1) does on the same cancellation.
    """

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return finalize_managed_task_lease_result(
            db,
            snapshot.lease,
            status=snapshot.resumed_from_status or TaskStatus.FAILED,
        )


def _register_channel_uploaded_files_sync(
    *,
    workspace: Any,
    task_id: int,
    user_id: int,
    files: Sequence[DownloadedChannelFile],
) -> tuple[ChannelUploadedFile, ...]:
    SessionLocal = get_session_local()
    registered: list[ChannelUploadedFile] = []
    scope_segments = tuple(getattr(workspace, "scope_segments", ()))
    for downloaded in files:
        staged_file: StagedUploadedFile | None = None
        metadata_committed = False
        try:
            registration: WorkspaceFileRegistration = (
                workspace.describe_file_registration(str(downloaded.path))
            )
            file_id = str(uuid4())
            storage_key = build_task_output_storage_key(
                user_id,
                task_id,
                file_id,
                registration.relative_path,
                scope_segments=scope_segments,
            )

            # Stage checksums and durable bytes before a Session exists. Object
            # storage latency must never consume a pooled database connection.
            staged_file = stage_uploaded_file_from_local_path(
                local_path=registration.path,
                user_id=user_id,
                task_id=task_id,
                file_id=file_id,
                filename=downloaded.name,
                mime_type=downloaded.mime_type,
                storage_key=storage_key,
                workspace_relative_path=registration.relative_path,
                workspace_category=registration.category,
            )

            # The only connection-owning phase is the short metadata
            # transaction. It also revalidates the task ownership boundary.
            with SessionLocal() as db:
                task_exists = (
                    db.query(Task.id)
                    .filter(Task.id == task_id, Task.user_id == user_id)
                    .first()
                    is not None
                )
                if not task_exists:
                    raise ValueError(
                        f"Task {task_id} is not owned by channel user {user_id}"
                    )
                UploadedFileStore(db).add_already_durable(staged_file.to_record())
                db.commit()
                metadata_committed = True

            workspace.bind_already_durable_file(
                registration,
                file_id=file_id,
            )
            registered.append(
                ChannelUploadedFile(
                    file_id=file_id,
                    name=downloaded.name,
                    path=str(registration.path),
                    mime_type=downloaded.mime_type,
                    size=int(downloaded.size),
                    source_id=downloaded.source_id,
                )
            )
        except Exception:
            if staged_file is not None and not metadata_committed:
                failed_file_ids = compensate_staged_uploaded_files((staged_file,))
                if failed_file_ids:
                    logger.warning(
                        "Retained durable channel file %s because reference or "
                        "deletion state was unknown",
                        downloaded.name,
                    )
            logger.exception(
                "Failed to register channel file %s for task %s",
                downloaded.name,
                task_id,
            )
    return tuple(registered)


async def register_channel_uploaded_files(
    *,
    workspace: Any,
    task_id: int,
    user_id: int,
    files: Sequence[DownloadedChannelFile],
) -> tuple[ChannelUploadedFile, ...]:
    """Register completed downloads using one worker-owned short Session."""

    if not files:
        return ()
    return await run_db_io_cancellation_safe(
        lambda: _register_channel_uploaded_files_sync(
            workspace=workspace,
            task_id=task_id,
            user_id=user_id,
            files=files,
        )
    )


def _update_channel_task_fields_sync(
    *,
    task_id: int,
    user_id: int,
    description: str | None,
    title: str | None,
) -> None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        task = (
            db.query(Task).filter(Task.id == task_id, Task.user_id == user_id).first()
        )
        if task is None:
            return
        if description is not None:
            setattr(task, "description", description)
        if title is not None:
            setattr(task, "title", title)
        db.commit()


async def update_channel_task_fields(
    *,
    task_id: int,
    user_id: int,
    description: str | None = None,
    title: str | None = None,
) -> None:
    await run_db_io_cancellation_safe(
        lambda: _update_channel_task_fields_sync(
            task_id=task_id,
            user_id=user_id,
            description=description,
            title=title,
        )
    )


def _persist_channel_user_message_sync(
    *,
    task_id: int,
    user_id: int,
    content: str,
    attachments: list[dict[str, Any]] | None,
    turn_id: str | None,
) -> None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        persist_user_message(
            db=db,
            task_id=task_id,
            user_id=user_id,
            content=content,
            attachments=attachments,
            turn_id=turn_id,
        )


async def persist_channel_user_message(
    *,
    task_id: int,
    user_id: int,
    content: str,
    attachments: list[dict[str, Any]] | None = None,
    turn_id: str | None = None,
) -> None:
    await run_db_io_cancellation_safe(
        lambda: _persist_channel_user_message_sync(
            task_id=task_id,
            user_id=user_id,
            content=content,
            attachments=attachments,
            turn_id=turn_id,
        )
    )


def _load_channel_output_files_sync(
    *,
    file_ids: Iterable[str],
    task_id: int,
    user_id: int,
) -> tuple[ChannelOutputFile, ...]:
    ordered_ids = tuple(dict.fromkeys(str(file_id) for file_id in file_ids))
    if not ordered_ids:
        return ()

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        rows = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.file_id.in_(ordered_ids),
                UploadedFile.user_id == user_id,
                UploadedFile.task_id == task_id,
            )
            .all()
        )
        by_id = {
            str(row.file_id): ChannelOutputFile(
                file_id=str(row.file_id),
                filename=str(row.filename or "file"),
                storage_path=str(row.storage_path),
                mime_type=str(row.mime_type or ""),
            )
            for row in rows
        }
        return tuple(by_id[file_id] for file_id in ordered_ids if file_id in by_id)


async def load_channel_output_files(
    *,
    file_ids: Iterable[str],
    task_id: int,
    user_id: int,
) -> tuple[ChannelOutputFile, ...]:
    return await run_db_io_cancellation_safe(
        lambda: _load_channel_output_files_sync(
            file_ids=file_ids,
            task_id=task_id,
            user_id=user_id,
        )
    )
