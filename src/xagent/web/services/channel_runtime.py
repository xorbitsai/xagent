"""Detached database boundaries shared by asynchronous chat transports.

Telegram and Feishu spend most of a turn awaiting network, file, sandbox, and
agent work.  A transport must therefore never retain a SQLAlchemy ``Session``
or an attached ORM row for the lifetime of that turn.  This module owns the
short worker-side transactions required by both transports and returns only
frozen primitive snapshots.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

from ...config import get_default_task_execution_mode
from ...core.file_storage.keys import build_task_output_storage_key
from ...core.workspace import WorkspaceFileRegistration
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.uploaded_file import UploadedFile
from ..models.user import User
from ..models.user_channel import UserChannel
from .chat_history_service import persist_user_message
from .db_runtime import await_task_settlement, run_db_io_cancellation_safe
from .managed_task_lease import (
    ManagedTaskLease,
    finalize_managed_task_lease_result,
    start_managed_task_lease,
)
from .task_lease_service import TaskLease, acquire_task_lease_no_commit
from .uploaded_file_store import (
    StagedUploadedFile,
    UploadedFileStore,
    compensate_staged_uploaded_files,
    stage_uploaded_file_from_local_path,
)
from .workforce_runtime import sync_workforce_run_status

logger = logging.getLogger(__name__)


class ChannelConfigurationError(RuntimeError):
    """The configured channel has no resolvable owner."""


class ChannelAuthorizationError(RuntimeError):
    """The external sender is not permitted by the channel configuration."""


@dataclass(frozen=True)
class ChannelOwnerSnapshot:
    """Detached owner identity after channel authorization."""

    user_id: int


@dataclass(frozen=True)
class ClaimedChannelTask:
    """Detached channel turn identity with an already-managed exact lease."""

    user_id: int
    task_id: int
    is_new_task: bool
    managed_lease: ManagedTaskLease


@dataclass(frozen=True)
class _ChannelTaskClaimSnapshot:
    """Worker result before the event loop starts the lease heartbeat."""

    user_id: int
    task_id: int
    is_new_task: bool
    lease: TaskLease


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
            snapshots.append(
                ChannelConfigSnapshot(
                    channel_id=int(channel.id),
                    channel_name=str(channel.channel_name),
                    config_items=tuple(
                        (key, resolved[key]) for key in required_config_keys
                    ),
                )
            )
    return tuple(snapshots)


async def load_active_channel_configs(
    *,
    channel_type: str,
    required_config_keys: Sequence[str],
) -> tuple[ChannelConfigSnapshot, ...]:
    """Load active bot credentials without blocking the asyncio event loop."""

    return await run_db_io_cancellation_safe(
        lambda: _load_active_channel_configs_sync(
            channel_type=channel_type,
            required_config_keys=required_config_keys,
        )
    )


def _load_channel_owner_sync(
    db: Any,
    *,
    channel_id: int | None,
    external_user_id: str,
) -> ChannelOwnerSnapshot:
    if channel_id is None:
        raise ChannelConfigurationError("Channel owner is not configured")

    channel = db.query(UserChannel).filter(UserChannel.id == int(channel_id)).first()
    if channel is None:
        raise ChannelConfigurationError("Channel owner is not configured")

    owner_id = int(channel.user_id)
    owner_exists = db.query(User.id).filter(User.id == owner_id).first() is not None
    if not owner_exists:
        raise ChannelConfigurationError("Channel owner is not configured")

    allowed_users = channel.config.get("allowed_users")
    if allowed_users is not None and external_user_id not in allowed_users:
        raise ChannelAuthorizationError("Channel sender is not authorized")
    return ChannelOwnerSnapshot(user_id=owner_id)


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


def _prepare_channel_task_sync(
    *,
    channel_id: int | None,
    external_user_id: str,
    active_task_id: int | None,
    text: str,
    channel_name: str | None,
    expected_owner_user_id: int | None,
) -> _ChannelTaskClaimSnapshot | None:
    SessionLocal = get_session_local()
    with SessionLocal() as db:
        try:
            owner = _load_channel_owner_sync(
                db,
                channel_id=channel_id,
                external_user_id=external_user_id,
            )
            owner_id = owner.user_id
            if (
                expected_owner_user_id is not None
                and owner_id != expected_owner_user_id
            ):
                raise ChannelAuthorizationError(
                    "Channel owner changed during sender authorization"
                )

            task = None
            if active_task_id is not None and active_task_id != -1:
                task = (
                    db.query(Task)
                    .filter(Task.id == active_task_id, Task.user_id == owner_id)
                    .first()
                )

            is_new_task = task is None
            if task is None:
                task_title = text or "Untitled Task"
                if len(task_title) > 50:
                    task_title = f"{task_title[:50]}..."
                task = Task(
                    user_id=owner_id,
                    title=task_title,
                    description=text,
                    status=TaskStatus.PENDING,
                    execution_mode=get_default_task_execution_mode(),
                    channel_id=channel_id,
                    channel_name=channel_name,
                    connector_runtime_selected_refs=[],
                )
                db.add(task)
                db.flush()

            task_id = int(task.id)
            lease = acquire_task_lease_no_commit(db, task_id, new_run=True)
            if lease is None:
                db.rollback()
                return None

            # Keep task state, workforce projection, and the exact run lease in
            # one transaction. No transport can observe the newly-created
            # PENDING row before its RUNNING owner is durable.
            db.expire_all()
            claimed_task = db.query(Task).filter(Task.id == task_id).one()
            sync_workforce_run_status(db, claimed_task, TaskStatus.RUNNING)
            db.commit()
            return _ChannelTaskClaimSnapshot(
                user_id=owner_id,
                task_id=task_id,
                is_new_task=is_new_task,
                lease=lease,
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
) -> ClaimedChannelTask | None:
    """Authorize, resolve/create, and claim one exact channel run atomically."""

    worker = asyncio.create_task(
        asyncio.to_thread(
            _prepare_channel_task_sync,
            channel_id=channel_id,
            external_user_id=external_user_id,
            active_task_id=active_task_id,
            text=text,
            channel_name=channel_name,
            expected_owner_user_id=expected_owner_user_id,
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
    )


def _compensate_channel_task_claim_sync(
    snapshot: _ChannelTaskClaimSnapshot,
) -> bool:
    """Settle the exact committed claim or leave its lease to TTL recovery."""

    SessionLocal = get_session_local()
    with SessionLocal() as db:
        return finalize_managed_task_lease_result(
            db,
            snapshot.lease,
            status=TaskStatus.FAILED,
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
