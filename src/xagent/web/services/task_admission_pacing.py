"""Transactional startup pacing with a bounded initial burst and no window reset."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from sqlalchemy import Float, exists, select
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.functions import GenericFunction

from ..models.task_admission import TaskAdmissionTicket
from ..models.task_admission_pacing import TaskAdmissionPacing


@dataclass(frozen=True)
class StartupPacing:
    interval_seconds: float
    burst: int
    lane: Literal["batch", "interactive", "default"] = "default"

    def __post_init__(self) -> None:
        if not isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("Startup interval must be finite and positive")
        if (
            not isinstance(self.burst, int)
            or isinstance(self.burst, bool)
            or self.burst < 1
        ):
            raise ValueError("Startup burst must be a positive integer")
        if self.lane not in {"batch", "interactive", "default"}:
            raise ValueError("Unsupported admission metrics lane")


class _DatabaseTime(GenericFunction[float]):
    type = Float()
    inherit_cache = True


@compiles(_DatabaseTime, "postgresql")
def _postgres_time(element: _DatabaseTime, compiler: SQLCompiler, **kwargs: Any) -> str:
    return "EXTRACT(EPOCH FROM clock_timestamp())"


@compiles(_DatabaseTime, "sqlite")
def _sqlite_time(element: _DatabaseTime, compiler: SQLCompiler, **kwargs: Any) -> str:
    return "((julianday('now') - 2440587.5) * 86400.0)"


def database_time() -> ColumnElement[float]:
    """Use the shared database clock rather than independently skewed worker clocks."""
    return _DatabaseTime()


def stage_startup_pacing(
    db: Session, bucket: str, policy: StartupPacing | None
) -> None:
    """Caller holds the bucket lock; configuration is immutable until drained."""
    row = db.get(TaskAdmissionPacing, bucket)
    if row is not None:
        if policy is None or (row.interval_seconds, row.burst, row.lane) != (
            policy.interval_seconds,
            policy.burst,
            policy.lane,
        ):
            raise ValueError(
                "Drain the admission bucket before changing startup pacing"
            )
    elif policy is not None:
        db.add(
            TaskAdmissionPacing(
                bucket_key=bucket,
                interval_seconds=policy.interval_seconds,
                burst=policy.burst,
                lane=policy.lane,
                next_start_at=0.0,
            )
        )
        db.flush()


def startup_eligible() -> ColumnElement[bool]:
    return ~exists(
        select(1)
        .where(
            TaskAdmissionPacing.bucket_key == TaskAdmissionTicket.bucket_key,
            TaskAdmissionPacing.next_start_at
            - (TaskAdmissionPacing.burst - 1) * TaskAdmissionPacing.interval_seconds
            > database_time(),
        )
        .correlate(TaskAdmissionTicket)
    )


def reserve_startup(db: Session, bucket: str) -> bool:
    """Consume a GCRA start only with a new slot, under the same bucket lock."""
    row = db.get(TaskAdmissionPacing, bucket, populate_existing=True)
    if row is None:
        return True
    now = float(db.execute(select(database_time())).scalar_one())
    interval, burst = float(row.interval_seconds), int(row.burst)
    if now < float(row.next_start_at) - (burst - 1) * interval:
        return False
    setattr(row, "next_start_at", max(now, float(row.next_start_at)) + interval)
    db.flush()
    return True
