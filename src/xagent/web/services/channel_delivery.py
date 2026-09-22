"""Retry final platform sends without retrying accepted Agent execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import case, false, or_, select, update
from sqlalchemy.orm import Session

from ...core.runtime_performance import increment_counter
from ..models.database import get_session_local
from ..models.task import Task, TaskStatus
from ..models.task_channel_delivery import TaskChannelDelivery
from ..models.task_command import TaskExecutionCommand
from .channel_runtime import (
    ChannelAuthorizationError,
    ChannelConfigurationError,
    _load_channel_owner_sync,
)
from .db_runtime import run_db_io_cancellation_safe
from .task_lease_service import TaskLeaseLostError

logger = logging.getLogger(__name__)
_DELIVERY_LEASE_SECONDS = 60
_DELIVERY_RETRY_SECONDS = 30
_DELIVERY_MAX_FAILURES = 10

PENDING_CHANNEL_RESULT: dict[str, Any] = {
    "success": True,
    "status": "accepted",
    "output": "Your request has been accepted and is still being processed. I will send the result here when it is ready.",
}


@dataclass(frozen=True)
class ChannelDelivery:
    command_id: int
    task_id: int
    user_id: int
    destination: dict[str, Any]
    claim_token: str


ChannelSender = Callable[[ChannelDelivery, dict[str, Any]], Awaitable[None]]


def _claim(
    command_id: int, *, progress: bool = False
) -> tuple[ChannelDelivery, dict[str, Any] | None] | None:
    from .shared_channel_execution import _read_channel_result

    now = datetime.now(timezone.utc)
    token = uuid4().hex
    with get_session_local()() as db:
        claimed = db.execute(
            update(TaskChannelDelivery)
            .where(
                TaskChannelDelivery.command_id == command_id,
                TaskChannelDelivery.status == "pending",
                or_(
                    TaskChannelDelivery.available_at.is_(None),
                    TaskChannelDelivery.available_at <= now,
                    (
                        TaskChannelDelivery.claim_token.is_(None)
                        & (TaskChannelDelivery.failure_count == 0)
                    )
                    if progress
                    else false(),
                ),
            )
            .values(
                claim_token=token,
                available_at=now + timedelta(seconds=_DELIVERY_LEASE_SECONDS),
            )
            .returning(TaskChannelDelivery.command_id)
        ).scalar_one_or_none()
        if claimed is None:
            return None
        row = db.execute(
            select(TaskChannelDelivery).where(
                TaskChannelDelivery.command_id == command_id
            )
        ).scalar_one()
        command = db.execute(
            select(TaskExecutionCommand).where(TaskExecutionCommand.id == command_id)
        ).scalar_one()
        channel = cast(dict[str, Any], command.payload)["channel"]
        delivery = ChannelDelivery(
            command_id,
            cast(int, command.task_id),
            cast(int, command.actor_user_id),
            dict(row.destination),
            token,
        )
        try:
            owner = _load_channel_owner_sync(
                db,
                channel_id=cast(int, row.channel_id),
                external_user_id=channel["external_user_id"],
            )
        except ChannelAuthorizationError:
            status = _settle_no_commit(db, delivery, status="discarded")
            db.commit()
            _record_delivery_outcome(delivery, status, reason="sender_not_authorized")
            return None
        except ChannelConfigurationError:
            # A disabled/misconfigured channel can be repaired. Keep the reply,
            # but use the same bounded retry budget as a failed platform send.
            status = _settle_no_commit(db, delivery, failed=True)
            db.commit()
            _record_delivery_outcome(
                delivery, status, reason="configuration_unavailable"
            )
            return None
        if owner.user_id != command.actor_user_id:
            status = _settle_no_commit(db, delivery, status="discarded")
            db.commit()
            _record_delivery_outcome(delivery, status, reason="owner_changed")
            return None
        run_id = str(command.target_run_id)
        db.commit()
    # The result reader owns a separate short session after the claim commits.
    try:
        result = _read_channel_result(command_id, run_id)
    except TaskLeaseLostError:
        result = {"success": True, "status": "interrupted"}
    return delivery, result


def _settle_no_commit(
    db: Session,
    delivery: ChannelDelivery,
    *,
    status: str = "pending",
    failed: bool = False,
    retry_only: bool = False,
    progress_sent: bool = False,
) -> str | None:
    now = datetime.now(timezone.utc)
    retry_at = now + timedelta(seconds=_DELIVERY_RETRY_SECONDS)
    values: dict[str, Any] = {
        "destination": delivery.destination,
        "status": status,
        # Keep the token as a retry fence: progress bypasses only unclaimed delays.
        "claim_token": delivery.claim_token if retry_only else None,
        "available_at": retry_at if status == "pending" and not progress_sent else None,
        "delivered_at": now if status == "delivered" else None,
    }
    if failed:
        failures = TaskChannelDelivery.failure_count + 1
        exhausted = failures >= _DELIVERY_MAX_FAILURES
        values.update(
            failure_count=failures,
            status=case((exhausted, "failed"), else_="pending"),
            available_at=case((exhausted, None), else_=retry_at),
        )
    return db.execute(
        update(TaskChannelDelivery)
        .where(
            TaskChannelDelivery.command_id == delivery.command_id,
            TaskChannelDelivery.claim_token == delivery.claim_token,
            TaskChannelDelivery.status == "pending",
        )
        .values(**values)
        .returning(TaskChannelDelivery.status)
    ).scalar_one_or_none()


def _record_delivery_outcome(
    delivery: ChannelDelivery, status: str | None, *, reason: str
) -> None:
    if status is None:
        return
    logger.warning(
        "Channel delivery settled task_id=%s command_id=%s status=%s reason=%s",
        delivery.task_id,
        delivery.command_id,
        status,
        reason,
    )
    increment_counter(
        "xagent.channel.delivery",
        attributes={"outcome": status, "error.type": reason},
    )


def _settle(
    delivery: ChannelDelivery,
    *,
    status: str = "pending",
    failed: bool = False,
    retry_only: bool = False,
    progress_sent: bool = False,
) -> None:
    with get_session_local()() as db:
        settled = _settle_no_commit(
            db,
            delivery,
            status=status,
            failed=failed,
            retry_only=retry_only,
            progress_sent=progress_sent,
        )
        db.commit()
    if failed:
        _record_delivery_outcome(delivery, settled, reason="delivery_error")


def _renew(delivery: ChannelDelivery) -> bool:
    with get_session_local()() as db:
        updated = db.execute(
            update(TaskChannelDelivery)
            .where(
                TaskChannelDelivery.command_id == delivery.command_id,
                TaskChannelDelivery.claim_token == delivery.claim_token,
                TaskChannelDelivery.status == "pending",
            )
            .values(
                available_at=datetime.now(timezone.utc)
                + timedelta(seconds=_DELIVERY_LEASE_SECONDS)
            )
            .returning(TaskChannelDelivery.command_id)
        ).scalar_one_or_none()
        db.commit()
        return updated is not None


async def deliver_channel_result(
    command_id: int,
    sender: ChannelSender,
    *,
    pending_notice: bool = False,
    progress: bool = False,
) -> bool:
    """Return whether a final send completed; never change Agent/task status."""
    delivery = None
    heartbeat = None
    sending: asyncio.Future[None] | None = None
    progress_pending = False
    try:
        claimed = await run_db_io_cancellation_safe(
            lambda: _claim(command_id, progress=progress)
        )
        if claimed is None:
            return False
        delivery, result = claimed
        progress_pending = progress and result is None
        if result is None and not pending_notice:
            await run_db_io_cancellation_safe(lambda: _settle(delivery))
            return False

        async def renew() -> None:
            while True:
                await asyncio.sleep(_DELIVERY_LEASE_SECONDS / 3)
                if not await run_db_io_cancellation_safe(lambda: _renew(delivery)):
                    raise ConnectionError("Channel delivery claim changed")

        heartbeat = asyncio.create_task(renew())
        sending = asyncio.ensure_future(
            sender(delivery, result or dict(PENDING_CHANNEL_RESULT))
        )
        done, _ = await asyncio.wait(
            (heartbeat, sending), return_when=asyncio.FIRST_COMPLETED
        )
        if heartbeat in done:
            await heartbeat
        await sending
        status = "delivered" if result is not None else "pending"
        await run_db_io_cancellation_safe(
            lambda: (
                _settle(delivery, progress_sent=True)
                if progress_pending
                else _settle(delivery, status=status)
            )
        )
        return result is not None
    except Exception:
        logger.exception("Channel result delivery deferred command_id=%s", command_id)
        # If the outcome of a DB commit is unknown, the persisted claim expires.
        # Retry only the platform send; never stage another execution command.
        if delivery is not None:
            try:
                await run_db_io_cancellation_safe(
                    lambda: (
                        _settle(delivery, retry_only=True)
                        if progress_pending
                        else _settle(delivery, failed=True)
                    )
                )
            except Exception:
                logger.exception(
                    "Channel delivery claim retained command_id=%s", command_id
                )
        return False
    finally:
        for child in (sending, heartbeat):
            if child is not None and not child.done():
                child.cancel()
        await asyncio.gather(
            *(child for child in (sending, heartbeat) if child is not None),
            return_exceptions=True,
        )


def _pending(channel_id: int) -> list[int]:
    now = datetime.now(timezone.utc)
    with get_session_local()() as db:
        return list(
            db.execute(
                select(TaskChannelDelivery.command_id)
                .join(
                    TaskExecutionCommand,
                    TaskExecutionCommand.id == TaskChannelDelivery.command_id,
                )
                .join(Task, Task.id == TaskExecutionCommand.task_id)
                .where(
                    TaskChannelDelivery.channel_id == channel_id,
                    TaskChannelDelivery.status == "pending",
                    or_(
                        TaskChannelDelivery.available_at.is_(None),
                        TaskChannelDelivery.available_at <= now,
                    ),
                    TaskExecutionCommand.status.in_(("completed", "failed")),
                    or_(
                        TaskExecutionCommand.status == "failed",
                        TaskExecutionCommand.result["channel_result"]
                        .as_string()
                        .is_not(None),
                        Task.run_id.is_distinct_from(
                            TaskExecutionCommand.target_run_id
                        ),
                        (
                            Task.runner_id.is_(None)
                            & Task.status.not_in(
                                (TaskStatus.RUNNING, TaskStatus.PENDING)
                            )
                        ),
                    ),
                )
                # Never let old failed destinations monopolize every batch.
                .order_by(
                    TaskChannelDelivery.available_at.asc().nulls_first(),
                    TaskChannelDelivery.command_id,
                )
                .limit(10)
            ).scalars()
        )


async def recover_channel_results(channel_id: int, sender: ChannelSender) -> None:
    try:
        commands = await run_db_io_cancellation_safe(lambda: _pending(channel_id))
        for command_id in commands:
            await deliver_channel_result(command_id, sender)
    except Exception:
        logger.exception(
            "Channel result recovery unavailable channel_id=%s", channel_id
        )
