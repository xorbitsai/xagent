"""Shared fixtures for the TaskInteractionRequest constraint-behavior suite.

Also serves test_task_interaction_schema_gate.py and
test_task_deletion_checkpoint_pointer.py, which import
tables_excluding_interaction_requests() below to build the same
filtered-schema shape those suites need.

Split across test_task_interaction_schema.py (SQLite, always runs) and
test_task_interaction_schema_postgresql.py (PostgreSQL, skips without
XAGENT_TEST_POSTGRES_URL), following the split used for
task_status_storage_shared.py / test_task_status_storage.py /
test_task_status_storage_postgresql.py, so the PostgreSQL half's real
Postgres dependency does not leak into every local test run. Row builders
and the two assertion helpers below are written once here and called from
both files, so both backends are pinned to the identical constraint
inventory.

Both backends report a violated CHECK the same way but with a different
message shape -- SQLite: "CHECK constraint failed: <name>"; PostgreSQL:
'violates check constraint "<name>"' -- so assert_rejected only checks
that the constraint's name is a substring of str(exception), never
anything about the rest of the message shape.

UNIQUE violations are not symmetric the same way: PostgreSQL's message
carries the constraint's name ('violates unique constraint "<name>"'),
but SQLite's does not -- it reports the qualified column list instead
("UNIQUE constraint failed: task_interaction_requests.task_id,
task_interaction_requests.active_slot"), never the name given to the
constraint. This was verified against the real model (not assumed): see
_UNIQUE_CONSTRAINT_COLUMNS below, which assert_rejected uses to check
column names on SQLite for the two UNIQUE constraints and falls back to
name-substring matching everywhere else (including on PostgreSQL, where
the name is always present).

On PostgreSQL a failed statement poisons the rest of the transaction
(InFailedSqlTransaction) until a rollback; assert_rejected always rolls
back immediately after catching IntegrityError, mirroring the same
recovery step task_status_storage_shared.py's PostgreSQL callers rely on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import count

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from xagent.web.models.database import Base
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.user import User

_key_counter = count()
_username_counter = count()

# SQLite's UNIQUE violation message names columns, not the constraint --
# see the module docstring. Column order here matches declaration order in
# TaskInteractionRequest.__table_args__, which is also the order SQLite
# lists them in.
_UNIQUE_CONSTRAINT_COLUMNS: dict[str, tuple[str, ...]] = {
    "uq_task_interaction_active_slot": ("task_id", "active_slot"),
    "uq_task_interaction_request_identity": (
        "task_id",
        "run_id",
        "request_idempotency_key",
    ),
}


def tables_excluding_interaction_requests() -> list:
    """Every table this repo's models declare, except the interaction table.

    Building a database from this filtered set reproduces the shape of a
    deployment upgraded through every migration except the (unmerged) one
    that creates task_interaction_requests -- the table has no inbound
    foreign keys, so excluding it cannot break create order (verified in the
    audit: 52 of 53 tables created, zero inbound FKs to the target).
    """
    return [
        table
        for table in Base.metadata.sorted_tables
        if table.name != TaskInteractionRequest.__tablename__
    ]


def make_user(db: Session) -> int:
    user = User(
        username=f"interaction-schema-user-{next(_username_counter)}",
        password_hash="hash",
        is_admin=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return int(user.id)


def make_task(db: Session, *, user_id: int) -> int:
    task = Task(user_id=user_id, title="interaction schema fixture task")
    db.add(task)
    db.commit()
    db.refresh(task)
    return int(task.id)


def make_trace_event(db: Session, *, task_id: int) -> int:
    event = TraceEvent(
        task_id=task_id,
        event_id=f"interaction-schema-event-{next(_key_counter)}",
        event_type="agent_execution_checkpoint",
        timestamp=datetime.now(timezone.utc),
        data={},
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return int(event.id)


def make_row(
    *,
    task_id: int,
    resume_trace_event_id: int | None,
    status: str = "active",
    **overrides: object,
) -> dict[str, object]:
    """Return a column-value dict for one legal TaskInteractionRequest row.

    Defaults build a self-consistent row for ``status`` ("active" by
    default): "active" gets active_slot=1, protocol_version=1, and a
    non-null anchor; "terminated" gets active_slot=None, a valid
    terminal_reason, and terminated_at; "answered" gets active_slot=None,
    a response_payload, responded_at and responder_identity. Any keyword in
    ``overrides`` replaces the corresponding default outright, including
    fields the ``status``-specific defaults above set -- callers isolating
    a single CHECK only need to pass the field(s) that CHECK inspects and
    rely on these defaults to keep every *other* CHECK satisfied for the
    row's status.

    request_idempotency_key gets a fresh value on every call so identity
    tests (T-uq-4/T-uq-5) only collide when a test deliberately repeats one
    via ``overrides``.
    """
    now = datetime.now(timezone.utc)
    row: dict[str, object] = {
        "task_id": task_id,
        "run_id": "run-a",
        "kind": "clarification",
        "protocol_version": 1,
        "status": status,
        "active_slot": None,
        "origin": "internal",
        "request_payload": {"prompt": "example"},
        "response_payload": None,
        "request_idempotency_key": f"req-{next(_key_counter)}",
        "resume_trace_event_id": resume_trace_event_id,
        "resume_event_id": "resume-event-1",
        "resume_execution_id": "resume-execution-1",
        "resume_locator_format": "trace_event_pk_v1",
        "resume_checkpoint_type": "agent_execution_checkpoint",
        "resume_run_partition": "partition-1",
        "responder_user_id": None,
        "responder_identity": None,
        "terminal_reason": None,
        "expires_at": now + timedelta(minutes=15),
        "responded_at": None,
        "terminated_at": None,
    }
    if status == "active":
        row["active_slot"] = 1
    elif status == "terminated":
        row["terminal_reason"] = "deadline_elapsed"
        row["terminated_at"] = now
    elif status == "answered":
        row["response_payload"] = {"answer": "example"}
        row["responded_at"] = now
        row["responder_identity"] = "user:1"
    row.update(overrides)
    return row


def assert_accepted(db: Session, row: dict[str, object]) -> TaskInteractionRequest:
    """Insert ``row``, commit, and confirm it reads back.

    The positive half of every constraint pair pinned in this suite --
    without at least one accepted case per pairing, a test suite made
    entirely of rejections could pass even if the CHECK were made
    unsatisfiable by every row (see the class docstring's `answered` full
    pairing positive control).
    """
    obj = TaskInteractionRequest(**row)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def _assert_constraint_named_in_error(
    exc: IntegrityError, constraint_name: str, dialect_name: str | None
) -> None:
    """Shared match logic between ``assert_rejected`` and
    ``assert_update_rejected``: SQLite names columns instead of the
    constraint for a UNIQUE violation (see the module docstring), so that
    case checks column names on SQLite and falls back to name-substring
    matching everywhere else.
    """
    text = str(exc)
    sqlite_columns = _UNIQUE_CONSTRAINT_COLUMNS.get(constraint_name)
    if sqlite_columns is not None and dialect_name == "sqlite":
        for column in sqlite_columns:
            qualified = f"task_interaction_requests.{column}"
            assert qualified in text, (
                f"expected column {qualified!r} in the SQLite UNIQUE "
                f"violation text for {constraint_name!r}, got: {text}"
            )
    else:
        assert constraint_name in text, (
            f"expected {constraint_name!r} in the IntegrityError text, got: {text}"
        )


def assert_rejected(db: Session, row: dict[str, object], constraint_name: str) -> None:
    """Insert ``row``, expect IntegrityError naming ``constraint_name``, and
    roll back immediately so the session stays usable for the caller's next
    statement (see the module docstring for why this matters on
    PostgreSQL)."""
    obj = TaskInteractionRequest(**row)
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        dialect_name = db.bind.dialect.name if db.bind is not None else None
        _assert_constraint_named_in_error(exc, constraint_name, dialect_name)
    else:
        raise AssertionError(
            f"expected IntegrityError naming {constraint_name!r}; insert succeeded"
        )
    finally:
        db.rollback()


def assert_update_accepted(
    db: Session, row_id: int, values: dict[str, object]
) -> TaskInteractionRequest:
    """Apply ``values`` to one existing row in a single UPDATE statement,
    commit, and return the refreshed row.

    The positive half of the UPDATE-path pairing tests, mirroring
    ``assert_accepted``'s role for the INSERT path: without at least one
    accepted transition, a suite made entirely of ``assert_update_rejected``
    calls could pass even if every UPDATE were unconditionally rejected.
    """
    db.execute(
        update(TaskInteractionRequest)
        .where(TaskInteractionRequest.id == row_id)
        .values(**values)
    )
    db.commit()
    obj = db.get(TaskInteractionRequest, row_id)
    assert obj is not None
    db.refresh(obj)
    return obj


def assert_update_rejected(
    db: Session, row_id: int, values: dict[str, object], constraint_name: str
) -> None:
    """Apply ``values`` to one existing row in a single UPDATE statement and
    expect ``constraint_name`` to reject it.

    One statement, one violated CHECK: when a single UPDATE violates more
    than one, the two backends do not agree on which name surfaces
    (measured: transitioning an ``answered`` row straight to ``terminated``
    in one UPDATE, without clearing ``response_payload`` or
    ``responded_at``, violates both
    ``ck_task_interaction_requests_response_pairs_status`` and
    ``ck_task_interaction_requests_responded_at_pairs_status`` at once --
    SQLite reports ``response_pairs_status`` where PostgreSQL reports
    ``responded_at_pairs_status`` for the identical statement), and a
    name-matching assertion would flap depending on which one the caller
    happened to construct.
    """
    try:
        db.execute(
            update(TaskInteractionRequest)
            .where(TaskInteractionRequest.id == row_id)
            .values(**values)
        )
        db.commit()
    except IntegrityError as exc:
        dialect_name = db.bind.dialect.name if db.bind is not None else None
        _assert_constraint_named_in_error(exc, constraint_name, dialect_name)
    else:
        raise AssertionError(
            f"expected IntegrityError naming {constraint_name!r}; update succeeded"
        )
    finally:
        db.rollback()


# ---------------------------------------------------------------------------
# Hand-written shape literals for the create_all structural tests (T-shape-*).
# Written independently of the model, per the judgment pinned in
# task_status_storage_shared.py's TASK_STATUS_STORAGE_NAMES: deriving the
# expectation from the column/constraint definitions would make the pin
# agree with any change made to them instead of catching drift.
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS: set[str] = {
    "id",
    "task_id",
    "run_id",
    "kind",
    "protocol_version",
    "status",
    "active_slot",
    "origin",
    "request_payload",
    "response_payload",
    "request_idempotency_key",
    "resume_trace_event_id",
    "resume_event_id",
    "resume_execution_id",
    "resume_locator_format",
    "resume_checkpoint_type",
    "resume_run_partition",
    "responder_user_id",
    "responder_identity",
    "terminal_reason",
    "created_at",
    "updated_at",
    "expires_at",
    "responded_at",
    "terminated_at",
}

# column -> nullable
EXPECTED_NULLABLE: dict[str, bool] = {
    "id": False,
    "task_id": False,
    "run_id": False,
    "kind": False,
    "protocol_version": False,
    "status": False,
    "active_slot": True,
    "origin": False,
    "request_payload": False,
    "response_payload": True,
    "request_idempotency_key": False,
    "resume_trace_event_id": True,
    "resume_event_id": False,
    "resume_execution_id": False,
    "resume_locator_format": False,
    "resume_checkpoint_type": False,
    "resume_run_partition": False,
    "responder_user_id": True,
    "responder_identity": True,
    "terminal_reason": True,
    "created_at": False,
    "updated_at": False,
    "expires_at": False,
    "responded_at": True,
    "terminated_at": True,
}

# column -> VARCHAR length. Every string column on the table, closed by the
# set assertion in test_created_columns_match_the_frozen_shape: iterating a
# partial dict pins only what it happens to list.
EXPECTED_STRING_LENGTHS: dict[str, int] = {
    "run_id": 64,
    "kind": 32,
    "status": 32,
    "origin": 20,
    "request_idempotency_key": 64,
    "resume_event_id": 255,
    "resume_execution_id": 255,
    "resume_locator_format": 32,
    "resume_checkpoint_type": 64,
    "resume_run_partition": 64,
    "responder_identity": 255,
    "terminal_reason": 32,
}

TIMESTAMP_COLUMNS: tuple[str, ...] = (
    "created_at",
    "updated_at",
    "expires_at",
    "responded_at",
    "terminated_at",
)

# constraint name -> sorted column tuple
EXPECTED_UNIQUE_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    "uq_task_interaction_active_slot": ("active_slot", "task_id"),
    "uq_task_interaction_request_identity": (
        "request_idempotency_key",
        "run_id",
        "task_id",
    ),
}

EXPECTED_CHECK_CONSTRAINT_NAMES: set[str] = {
    "ck_task_interaction_requests_status",
    "ck_task_interaction_requests_kind",
    "ck_task_interaction_requests_origin",
    "ck_task_interaction_requests_resume_checkpoint_type",
    "ck_task_interaction_requests_resume_locator_format",
    "ck_task_interaction_requests_terminal_reason",
    "ck_task_interaction_requests_protocol_version_floor",
    "ck_task_interaction_requests_active_slot_value",
    "ck_task_interaction_requests_active_slot_pairs_status",
    "ck_task_interaction_requests_active_anchor",
    "ck_task_interaction_requests_active_protocol",
    "ck_task_interaction_requests_terminal_pairs_status",
    "ck_task_interaction_requests_terminated_at_pairs_status",
    "ck_task_interaction_requests_response_pairs_status",
    "ck_task_interaction_requests_responded_at_pairs_status",
    "ck_task_interaction_requests_responder_pairs_responded_at",
    "ck_task_interaction_requests_expiry_after_creation",
    "ck_task_interaction_requests_run_id_nonempty",
    "ck_task_interaction_requests_resume_event_id_nonempty",
    "ck_task_interaction_requests_resume_execution_id_nonempty",
    "ck_task_interaction_requests_resume_run_partition_nonempty",
    "ck_task_interaction_requests_request_idempotency_key_nonempty",
    "ck_task_interaction_requests_responder_identity_nonempty",
}

# constraint name -> (referred_table, referred_columns, ondelete)
EXPECTED_FOREIGN_KEYS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "fk_task_interaction_requests_task_id": ("tasks", ("id",), "CASCADE"),
    "fk_task_interaction_requests_resume_trace_event_id": (
        "trace_events",
        ("id",),
        "SET NULL",
    ),
    "fk_task_interaction_requests_responder_user_id": (
        "users",
        ("id",),
        "SET NULL",
    ),
}

# index name -> sorted column tuple, restricted to the *non-unique* subset.
# PostgreSQL's get_indexes() also reports the two UNIQUE constraints' backing
# indexes (unique=True); SQLite's does not. Filtering to unique=False first
# is what makes this literal apply to both backends' reflection.
EXPECTED_NONUNIQUE_INDEXES: dict[str, tuple[str, ...]] = {
    "ix_task_interaction_requests_task_status": ("status", "task_id"),
    "ix_task_interaction_requests_resume_trace_event_id": ("resume_trace_event_id",),
}
