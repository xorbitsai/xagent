"""Durable channel input identity, before task selection or attachment binding."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ...core.runtime_performance import increment_counter
from ...core.workspace import scoped_user_root
from ..models.database import get_session_local
from ..models.task import Task
from ..models.task_command import TaskExecutionCommand
from ..models.task_input_receipt import TaskInputReceipt
from .channel_runtime import (
    SelectedChannelTask,
    _load_channel_owner_sync,
    prepare_channel_task_no_commit,
)
from .shared_channel_execution import SharedChannelTurn, accept_channel_turn_no_commit
from .task_command_transport import (
    _resolve_actor_subject,
    command_identity_matches_task,
)
from .task_orchestrator import TaskTurnError, TaskTurnPayload
from .uploaded_file_store import StagedUploadedFile, UploadedFileStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelInput:
    channel_id: int
    external_user_id: str
    source: str
    scope: tuple[str, ...]
    message_id: str
    text: str
    source_file_ids: tuple[str, ...]
    destination: dict[str, Any]

    def payload_hash(self) -> str:
        return _hash([self.text, self.source_file_ids, self.destination])


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _identity(db: Session, incoming: ChannelInput) -> tuple[int, str]:
    owner = _load_channel_owner_sync(
        db, channel_id=incoming.channel_id, external_user_id=incoming.external_user_id
    )
    subject = _resolve_actor_subject(db, owner.user_id)
    if subject is None:
        raise TaskTurnError("input_unavailable")
    return owner.user_id, _hash(
        [
            "channel/input/v1",
            subject,
            incoming.channel_id,
            incoming.external_user_id,
            incoming.source,
            incoming.scope,
            incoming.message_id,
        ]
    )


@dataclass(frozen=True)
class AcceptedChannelInput:
    task_id: int
    command_db_id: int
    command_id: str
    run_id: str
    user_id: int
    replayed: bool
    selection: SelectedChannelTask

    def as_turn(self) -> SharedChannelTurn:
        return SharedChannelTurn(
            self.selection,
            workspace=None,
            run_id=self.run_id,
            command_id=self.command_id,
            accepted=True,
            command_db_id=self.command_db_id,
        )


def _replay(
    db: Session, receipt: TaskInputReceipt, incoming: ChannelInput, owner_id: int
) -> AcceptedChannelInput:
    task = db.get(Task, receipt.task_id) if receipt.task_id is not None else None
    command = (
        db.get(TaskExecutionCommand, receipt.command_db_id)
        if receipt.command_db_id is not None
        else None
    )
    if (
        task is None
        or command is None
        or task.user_id != owner_id
        or task.channel_id != incoming.channel_id
        or command.task_id != task.id
        or not command_identity_matches_task(db, task, command)
    ):
        raise TaskTurnError("input_unavailable")
    if receipt.payload_hash != incoming.payload_hash():
        raise TaskTurnError("input_conflict")
    return AcceptedChannelInput(
        int(task.id),
        int(command.id),
        str(command.command_id),
        str(command.target_run_id),
        owner_id,
        True,
        SelectedChannelTask(
            owner_id,
            int(task.id),
            False,
            incoming.channel_id,
            incoming.external_user_id,
            cast(str | None, task.run_id),
            int(task.state_version),
        ),
    )


def lookup_channel_input(
    incoming: ChannelInput,
) -> tuple[int, AcceptedChannelInput | None]:
    """Authorize before downloading, without selecting or creating a task."""
    with get_session_local()() as db:
        owner_id, identity = _identity(db, incoming)
        receipt = db.get(TaskInputReceipt, identity)
        result = (
            _replay(db, receipt, incoming, owner_id) if receipt is not None else None
        )
        # Persist legacy owner subject initialization after a successful lookup.
        db.commit()
        return owner_id, result


class ChannelInputBatchChanged(Exception):
    """The batch no longer matches durable state; re-run the partition lookup."""


@dataclass(frozen=True)
class RejectedChannelInput:
    incoming: ChannelInput
    reason: str


def _validate_batch(incoming: tuple[ChannelInput, ...]) -> None:
    if not incoming:
        raise TaskTurnError("input_batch_invalid")
    first = incoming[0]
    if any(
        item.channel_id != first.channel_id
        or item.external_user_id != first.external_user_id
        or item.source != first.source
        or item.scope != first.scope
        for item in incoming[1:]
    ):
        raise TaskTurnError("input_batch_invalid")


def _is_receipt_duplicate(error: IntegrityError) -> bool:
    # PostgreSQL names the primary key; SQLite names its column.
    original = error.orig
    sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    if sqlstate == "23505":
        return (
            getattr(getattr(original, "diag", None), "constraint_name", None)
            == "task_input_receipts_pkey"
        )
    return (
        str(original) == "UNIQUE constraint failed: task_input_receipts.identity_hash"
    )


def lookup_channel_inputs(
    incoming: tuple[ChannelInput, ...],
) -> tuple[
    int,
    tuple[ChannelInput, ...],
    tuple[AcceptedChannelInput, ...],
    tuple[RejectedChannelInput, ...],
]:
    """Partition physical inputs before file IO; replay each original command once."""
    _validate_batch(incoming)
    owner_id: int | None = None
    pending: dict[str, ChannelInput] = {}
    accepted: dict[int, AcceptedChannelInput] = {}
    seen: dict[str, str] = {}
    rejected: list[RejectedChannelInput] = []
    with get_session_local()() as db:
        for item in incoming:
            item_owner, identity = _identity(db, item)
            if owner_id is None:
                owner_id = item_owner
            elif item_owner != owner_id:
                raise TaskTurnError("owner_changed")
            digest = item.payload_hash()
            if identity in seen:
                if seen[identity] == digest:
                    continue
                if identity in pending:
                    raise TaskTurnError("input_conflict")
            seen[identity] = digest
            receipt = db.get(TaskInputReceipt, identity)
            if receipt is None:
                pending[identity] = item
            else:
                try:
                    replay = _replay(db, receipt, item, owner_id)
                except TaskTurnError as error:
                    if error.reason not in {"input_unavailable", "input_conflict"}:
                        raise
                    rejected.append(RejectedChannelInput(item, error.reason))
                else:
                    accepted[replay.command_db_id] = replay
        db.commit()
    return (
        cast(int, owner_id),
        tuple(pending.values()),
        tuple(accepted.values()),
        tuple(rejected),
    )


def accept_channel_input(
    incoming: ChannelInput,
    *,
    owner_id: int,
    active_task_id: int | None,
    channel_name: str | None,
    payload: TaskTurnPayload,
    staged_files: tuple[StagedUploadedFile, ...],
    host_id: str,
    additional_inputs: tuple[ChannelInput, ...] = (),
    agent_id: int | None = None,
    task_title: str | None = None,
    task_description: str | None = None,
) -> AcceptedChannelInput:
    """Commit receipt, selection, files, transcript, START and reply mapping once."""
    _validate_batch((incoming, *additional_inputs))
    with get_session_local()() as db:
        current_owner_id, identity = _identity(db, incoming)
        if current_owner_id != owner_id:
            raise TaskTurnError("owner_changed")
        identities = {identity: incoming}
        for item in additional_inputs:
            item_owner, item_identity = _identity(db, item)
            if item_owner != owner_id:
                raise TaskTurnError("owner_changed")
            if (
                item_identity in identities
                and identities[item_identity].payload_hash() != item.payload_hash()
            ):
                raise TaskTurnError("input_conflict")
            identities[item_identity] = item
        existing = []
        for key, item in identities.items():
            saved = db.get(TaskInputReceipt, key)
            if saved is not None:
                try:
                    existing.append(_replay(db, saved, item, owner_id))
                except TaskTurnError as error:
                    if additional_inputs and error.reason in {
                        "input_unavailable",
                        "input_conflict",
                    }:
                        raise ChannelInputBatchChanged() from error
                    raise
        if existing:
            if (
                len(existing) == len(identities)
                and len({row.command_db_id for row in existing}) == 1
            ):
                return existing[0]
            raise ChannelInputBatchChanged()
        # Consistent acquisition order avoids deadlocks across overlapping batches.
        receipts = [
            TaskInputReceipt(
                identity_hash=key, payload_hash=identities[key].payload_hash()
            )
            for key in sorted(identities)
        ]
        db.add_all(receipts)
        try:
            db.flush()
        except IntegrityError as error:
            db.rollback()
            if not _is_receipt_duplicate(error):
                raise
            if additional_inputs:
                raise ChannelInputBatchChanged() from error
            saved = db.get(TaskInputReceipt, identity)
            if saved is None:
                raise
            # A competing request accepted this input; normal replay authorization applies.
            current_owner_id, _ = _identity(db, incoming)
            return _replay(db, saved, incoming, current_owner_id)
        selection = prepare_channel_task_no_commit(
            db,
            channel_id=incoming.channel_id,
            external_user_id=incoming.external_user_id,
            active_task_id=active_task_id,
            text=payload.transcript_message,
            channel_name=channel_name,
            expected_owner_user_id=owner_id,
            defer_execution=True,
            agent_id=agent_id,
        )
        if selection is None:
            raise TaskTurnError("busy")
        selection = cast(SelectedChannelTask, selection)
        if selection.is_new_task and (
            task_title is not None or task_description is not None
        ):
            task = cast(Task, db.get(Task, selection.task_id))
            if task_title is not None:
                setattr(task, "title", task_title)
            if task_description is not None:
                setattr(task, "description", task_description)
        for staged in staged_files:
            if staged.user_id != owner_id or staged.task_id is not None:
                raise TaskTurnError("file_unavailable")
            record = staged.to_record()
            setattr(record, "task_id", selection.task_id)
            # Persist a stable worker cache path, independent of the staging path.
            setattr(
                record,
                "storage_path",
                str(
                    scoped_user_root(get_uploads_dir(), owner_id)
                    / staged.file_id
                    / staged.filename
                ),
            )
            UploadedFileStore(db).add_already_durable(record)
        db.flush()
        turn = SharedChannelTurn(selection, workspace=None)
        turn.delivery_destination = dict(
            (additional_inputs[-1] if additional_inputs else incoming).destination
        )
        command_id = accept_channel_turn_no_commit(db, turn, payload, host_id)
        for receipt in receipts:
            receipt.task_id = selection.task_id
            receipt.command_db_id = command_id
        try:
            db.commit()
        except Exception as error:
            db.close()
            with get_session_local()() as check:
                saved = check.get(TaskInputReceipt, identity)
                if saved is None:
                    raise
                command = (
                    check.get(TaskExecutionCommand, saved.command_db_id)
                    if saved.command_db_id is not None
                    else None
                )
                # SQLite can reuse rolled-back row IDs; also match the command UUID.
                own_commit = (
                    saved.task_id == selection.task_id
                    and saved.command_db_id == command_id
                    and command is not None
                    and command.command_id == turn.command_id
                )
                if not own_commit:
                    logger.warning(
                        "Channel input recovery found competing acceptance command_db_id=%s",
                        saved.command_db_id,
                    )
                    increment_counter("xagent.channel.acceptance.competing_commit")
                    if additional_inputs:
                        raise ChannelInputBatchChanged() from error
                    # A competing receipt grants no replay authority.
                    current_owner_id, _ = _identity(check, incoming)
                    return _replay(check, saved, incoming, current_owner_id)
                # Confirm our durable acceptance without converting later channel
                # changes into a rejection. Keep the original selection/result.
                logger.warning(
                    "Channel acceptance recovered after uncertain commit task_id=%s command_id=%s",
                    selection.task_id,
                    turn.command_id,
                )
                increment_counter("xagent.channel.acceptance.commit_recovered")
        return AcceptedChannelInput(
            selection.task_id,
            command_id,
            turn.command_id,
            turn.run_id,
            owner_id,
            False,
            selection,
        )
