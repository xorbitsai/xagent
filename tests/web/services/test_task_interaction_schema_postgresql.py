"""TaskInteractionRequest constraint-behavior and shape sentinels (PostgreSQL half).

Companion to test_task_interaction_schema.py; see that file's and
task_interaction_schema_shared.py's module docstrings for the full
rationale behind each invariant pinned here, the backend split, and the
UNIQUE-violation message asymmetry. Docstrings below are intentionally
short and point back to the SQLite half rather than repeat it. Fixture
pattern copied from test_task_status_storage_postgresql.py /
test_runtime_key_transition_postgres.py (skip-if-unset via
XAGENT_TEST_POSTGRES_URL).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from tests.web.services.task_interaction_schema_shared import (
    EXPECTED_CHECK_CONSTRAINT_NAMES,
    EXPECTED_COLUMNS,
    EXPECTED_FOREIGN_KEYS,
    EXPECTED_NONUNIQUE_INDEXES,
    EXPECTED_NULLABLE,
    EXPECTED_STRING_LENGTHS,
    EXPECTED_UNIQUE_CONSTRAINTS,
    TIMESTAMP_COLUMNS,
    assert_accepted,
    assert_rejected,
    make_row,
    make_task,
    make_trace_event,
    make_user,
)
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.user import User


@pytest.fixture()
def db_session():
    """Isolated TaskInteractionRequest rows in a real PostgreSQL test
    database. Fails closed rather than pointing at whatever XAGENT_TEST_
    POSTGRES_URL happens to resolve to: this is the same drop_all/create_all
    of the *entire* Base.metadata that every PostgreSQL suite following this
    pattern already does, so the URL must name a database whose full schema
    this suite is allowed to drop and recreate.
    """
    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def fixtures(db_session):
    """A (task_id, anchor_trace_event_id) pair shared by most tests below."""
    user_id = make_user(db_session)
    task_id = make_task(db_session, user_id=user_id)
    anchor_id = make_trace_event(db_session, task_id=task_id)
    return task_id, anchor_id


# --------------------------------------------------------------------------
# T-uq: uniqueness (3 active-slot + 2 identity)
# --------------------------------------------------------------------------


def test_active_slot_unique_rejects_second_active_row(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    assert_accepted(
        db_session, make_row(task_id=task_id, resume_trace_event_id=anchor_id)
    )
    assert_rejected(
        db_session,
        make_row(task_id=task_id, resume_trace_event_id=anchor_id),
        "uq_task_interaction_active_slot",
    )


def test_active_slot_null_is_distinct_across_terminal_rows(
    db_session, fixtures
) -> None:
    """Also the sentinel against NULLS NOT DISTINCT ever being added to
    uq_task_interaction_active_slot -- see the SQLite half and the model's
    class docstring.
    """
    task_id, anchor_id = fixtures
    for _ in range(3):
        assert_accepted(
            db_session,
            make_row(
                task_id=task_id, resume_trace_event_id=anchor_id, status="terminated"
            ),
        )


def test_active_slot_unique_is_scoped_per_task(db_session) -> None:
    user_id = make_user(db_session)
    task_a = make_task(db_session, user_id=user_id)
    task_b = make_task(db_session, user_id=user_id)
    anchor_a = make_trace_event(db_session, task_id=task_a)
    anchor_b = make_trace_event(db_session, task_id=task_b)
    assert_accepted(
        db_session, make_row(task_id=task_a, resume_trace_event_id=anchor_a)
    )
    assert_accepted(
        db_session, make_row(task_id=task_b, resume_trace_event_id=anchor_b)
    )


def test_identity_unique_rejects_same_task_run_key(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    assert_accepted(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            run_id="run-dup",
            request_idempotency_key="key-dup",
        ),
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            run_id="run-dup",
            request_idempotency_key="key-dup",
        ),
        "uq_task_interaction_request_identity",
    )


def test_identity_unique_is_run_scoped(db_session, fixtures) -> None:
    """Run-scoping: the same (task_id, key) is allowed across different
    run_id values."""
    task_id, anchor_id = fixtures
    assert_accepted(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            run_id="run-1",
            request_idempotency_key="key-shared",
        ),
    )
    assert_accepted(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            run_id="run-2",
            request_idempotency_key="key-shared",
        ),
    )


# --------------------------------------------------------------------------
# T-ck-1..6: vocabulary CHECKs
# --------------------------------------------------------------------------


def test_status_rejects_unknown_literal(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=anchor_id,
        status="bogus",
        active_slot=None,
    )
    assert_rejected(db_session, row, "ck_task_interaction_requests_status")


def test_kind_rejects_unknown_literal(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id, resume_trace_event_id=anchor_id, kind="unknown_kind"
    )
    assert_rejected(db_session, row, "ck_task_interaction_requests_kind")


def test_origin_accepts_a2a(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=anchor_id, origin="a2a")
    assert_accepted(db_session, row)


def test_origin_rejects_unknown_literal(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=anchor_id, origin="email")
    assert_rejected(db_session, row, "ck_task_interaction_requests_origin")


def test_resume_checkpoint_type_and_locator_format_reject_unknown(
    db_session, fixtures
) -> None:
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            resume_checkpoint_type="unknown_checkpoint_type",
        ),
        "ck_task_interaction_requests_resume_checkpoint_type",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            resume_locator_format="unknown_locator_format",
        ),
        "ck_task_interaction_requests_resume_locator_format",
    )


def test_terminal_reason_vocabulary(db_session, fixtures) -> None:
    """Rename regression guard -- see the SQLite half."""
    task_id, anchor_id = fixtures
    for reason in ("deadline_elapsed", "run_superseded", "answered_via_legacy_resume"):
        assert_accepted(
            db_session,
            make_row(
                task_id=task_id,
                resume_trace_event_id=anchor_id,
                status="terminated",
                terminal_reason=reason,
            ),
        )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            terminal_reason="superseded_by_legacy_resume",
        ),
        "ck_task_interaction_requests_terminal_reason",
    )


# --------------------------------------------------------------------------
# T-ck-7..10: slot admission CHECKs
# --------------------------------------------------------------------------


def test_active_slot_value_rejects_other_than_one(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=anchor_id, active_slot=2)
    assert_rejected(db_session, row, "ck_task_interaction_requests_active_slot_value")


def test_active_status_and_slot_are_paired_both_ways(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(task_id=task_id, resume_trace_event_id=anchor_id, active_slot=None),
        "ck_task_interaction_requests_active_slot_pairs_status",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            active_slot=1,
        ),
        "ck_task_interaction_requests_active_slot_pairs_status",
    )


def test_active_row_requires_anchor(db_session, fixtures) -> None:
    """Also what turns the anchor FK's ON DELETE SET NULL into an effective
    RESTRICT for active rows -- see T-fk-1 below and the SQLite half.
    """
    task_id, _anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=None)
    assert_rejected(db_session, row, "ck_task_interaction_requests_active_anchor")


def test_active_row_requires_protocol_version_one(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(task_id=task_id, resume_trace_event_id=anchor_id, protocol_version=2),
        "ck_task_interaction_requests_active_protocol",
    )
    assert_accepted(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            protocol_version=2,
        ),
    )


# --------------------------------------------------------------------------
# T-ck-11..17: paired CHECKs, plus the full-pairing positive control
# --------------------------------------------------------------------------


def test_terminated_status_and_terminal_reason_are_paired_both_ways(
    db_session, fixtures
) -> None:
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            terminal_reason=None,
        ),
        "ck_task_interaction_requests_terminal_pairs_status",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            terminal_reason="deadline_elapsed",
        ),
        "ck_task_interaction_requests_terminal_pairs_status",
    )


def test_answered_requires_response_payload(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=anchor_id,
        status="answered",
        response_payload=None,
    )
    assert_rejected(
        db_session, row, "ck_task_interaction_requests_answered_pairs_response"
    )


def test_unanswered_rejects_response_payload(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=anchor_id,
        response_payload={"stray": "value"},
    )
    assert_rejected(
        db_session, row, "ck_task_interaction_requests_unanswered_has_no_response"
    )


def test_answered_requires_responded_at(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=anchor_id,
        status="answered",
        responded_at=None,
        responder_identity=None,
    )
    assert_rejected(
        db_session, row, "ck_task_interaction_requests_answered_pairs_responded_at"
    )


def test_unanswered_rejects_responded_at(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=anchor_id,
        responded_at=datetime.now(timezone.utc),
        responder_identity="user:1",
    )
    assert_rejected(
        db_session, row, "ck_task_interaction_requests_unanswered_has_no_responded_at"
    )


def test_responder_identity_pairs_responded_at_both_ways(db_session, fixtures) -> None:
    """Renamed from ck_task_interaction_requests_responder_identity_pairs_
    responded_at (66 chars, over PostgreSQL's own 63-char identifier limit)
    -- see the SQLite half and
    test_no_constraint_name_exceeds_the_postgres_identifier_limit there.
    """
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="answered",
            responder_identity=None,
        ),
        "ck_task_interaction_requests_responder_pairs_responded_at",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            responder_identity="user:1",
        ),
        "ck_task_interaction_requests_responder_pairs_responded_at",
    )


def test_expiry_must_be_after_creation(db_session, fixtures) -> None:
    """created_at and expires_at are both bound explicitly here so the
    equal/before cases do not race the server clock -- see the SQLite half
    for why."""
    task_id, anchor_id = fixtures
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            created_at=fixed,
            expires_at=fixed,
        ),
        "ck_task_interaction_requests_expiry_after_creation",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            created_at=fixed,
            expires_at=fixed - timedelta(minutes=1),
        ),
        "ck_task_interaction_requests_expiry_after_creation",
    )


def test_answered_row_with_full_pairing_is_accepted(db_session, fixtures) -> None:
    """Positive control -- see the SQLite half."""
    task_id, anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=anchor_id, status="answered")
    assert_accepted(db_session, row)


# --------------------------------------------------------------------------
# T-empty: empty-string CHECKs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column",
    [
        "run_id",
        "resume_event_id",
        "resume_execution_id",
        "resume_run_partition",
        "request_idempotency_key",
    ],
)
def test_empty_string_is_rejected(db_session, fixtures, column) -> None:
    """NOT NULL alone does not stop a caller writing "" -- see the SQLite
    half."""
    task_id, anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=anchor_id, **{column: ""})
    assert_rejected(db_session, row, f"ck_task_interaction_requests_{column}_nonempty")


def test_run_id_may_differ_from_resume_run_partition(db_session, fixtures) -> None:
    """Negative control for the deleted CHECK -- see the SQLite half."""
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=anchor_id,
        run_id="run-one-value",
        resume_run_partition="a-different-partition-value",
    )
    assert_accepted(db_session, row)


# --------------------------------------------------------------------------
# T-fk: foreign-key delete matrix
# --------------------------------------------------------------------------


def test_deleting_anchor_of_active_row_is_blocked(db_session, fixtures) -> None:
    """resume_trace_event_id is ON DELETE SET NULL, but
    ck_task_interaction_requests_active_anchor requires a non-null anchor on
    an active row -- the delete's SET NULL fails that CHECK instead of
    going through -- see the SQLite half.
    """
    task_id, anchor_id = fixtures
    assert_accepted(
        db_session, make_row(task_id=task_id, resume_trace_event_id=anchor_id)
    )

    anchor = db_session.get(TraceEvent, anchor_id)
    db_session.delete(anchor)
    try:
        db_session.commit()
    except IntegrityError as exc:
        assert "ck_task_interaction_requests_active_anchor" in str(exc), (
            f"expected the active-anchor CHECK name in the error, got: {exc}"
        )
    else:
        raise AssertionError("deleting an active row's anchor must be blocked")
    finally:
        db_session.rollback()


def test_deleting_anchor_of_terminal_row_sets_null(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = assert_accepted(
        db_session,
        make_row(task_id=task_id, resume_trace_event_id=anchor_id, status="terminated"),
    )
    row_id = row.id

    anchor = db_session.get(TraceEvent, anchor_id)
    db_session.delete(anchor)
    db_session.commit()

    reloaded = db_session.get(TaskInteractionRequest, row_id)
    assert reloaded.resume_trace_event_id is None


def test_deleting_unrelated_trace_event_is_unaffected(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = assert_accepted(
        db_session, make_row(task_id=task_id, resume_trace_event_id=anchor_id)
    )
    row_id = row.id
    other_event_id = make_trace_event(db_session, task_id=task_id)

    other_event = db_session.get(TraceEvent, other_event_id)
    db_session.delete(other_event)
    db_session.commit()

    reloaded = db_session.get(TaskInteractionRequest, row_id)
    assert reloaded.resume_trace_event_id == anchor_id


def test_deleting_task_cascades_interaction_rows(db_session) -> None:
    """Uses a dedicated task with no trace_events pointing at it -- see the
    SQLite half for why."""
    user_id = make_user(db_session)
    task_id = make_task(db_session, user_id=user_id)
    assert_accepted(
        db_session,
        make_row(task_id=task_id, resume_trace_event_id=None, status="terminated"),
    )

    task = db_session.get(Task, task_id)
    db_session.delete(task)
    db_session.commit()

    remaining = (
        db_session.query(TaskInteractionRequest)
        .filter(TaskInteractionRequest.task_id == task_id)
        .count()
    )
    assert remaining == 0


def test_deleting_responder_user_sets_null_and_keeps_identity(
    db_session, fixtures
) -> None:
    """responder_identity and responded_at are untouched by the SET NULL --
    see the SQLite half and the model's class docstring."""
    task_id, anchor_id = fixtures
    responder_id = make_user(db_session)
    row = assert_accepted(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="answered",
            responder_user_id=responder_id,
            responder_identity="user:responder",
        ),
    )
    row_id = row.id
    original_responded_at = row.responded_at

    responder = db_session.get(User, responder_id)
    db_session.delete(responder)
    db_session.commit()

    reloaded = db_session.get(TaskInteractionRequest, row_id)
    assert reloaded.responder_user_id is None
    assert reloaded.responder_identity == "user:responder"
    assert reloaded.responded_at == original_responded_at


# --------------------------------------------------------------------------
# T-store: storage-form sentinels
# --------------------------------------------------------------------------


def test_response_payload_none_lands_as_sql_null(db_session, fixtures) -> None:
    """JSON(none_as_null=True): a Python None payload lands as SQL NULL,
    while a JSON null nested inside a payload does not -- see the SQLite
    half."""
    task_id, anchor_id = fixtures
    none_row = assert_accepted(
        db_session, make_row(task_id=task_id, resume_trace_event_id=anchor_id)
    )
    is_null = db_session.execute(
        text(
            "SELECT response_payload IS NULL FROM task_interaction_requests WHERE id = :id"
        ),
        {"id": none_row.id},
    ).scalar_one()
    assert bool(is_null) is True

    nested_null_row = assert_accepted(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="answered",
            response_payload={"answer": None},
        ),
    )
    is_null_nested = db_session.execute(
        text(
            "SELECT response_payload IS NULL FROM task_interaction_requests WHERE id = :id"
        ),
        {"id": nested_null_row.id},
    ).scalar_one()
    assert bool(is_null_nested) is False
    assert nested_null_row.response_payload == {"answer": None}


def test_expires_at_round_trips_as_utc(db_session, fixtures) -> None:
    """PostgreSQL normalizes a non-UTC-but-aware value to the correct UTC
    instant instead of storing it verbatim (contrast the SQLite half, where
    the same input is stored as wrong-instant local wall-clock time)."""
    task_id, anchor_id = fixtures
    fixed_created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    utc_value = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    utc_row = assert_accepted(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            created_at=fixed_created_at,
            expires_at=utc_value,
        ),
    )
    assert utc_row.expires_at == utc_value

    plus_eight_value = datetime(
        2026, 6, 1, 20, 0, 0, tzinfo=timezone(timedelta(hours=8))
    )
    non_utc_row = assert_accepted(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            created_at=fixed_created_at,
            expires_at=plus_eight_value,
        ),
    )
    # The +08:00 instant and utc_value are the same wall-clock moment
    # (12:00 UTC); PostgreSQL normalizes on write, so this must round-trip
    # to that shared instant rather than to 20:00.
    assert non_utc_row.expires_at == utc_value


# --------------------------------------------------------------------------
# T-shape: create_all shape
# --------------------------------------------------------------------------


def test_table_is_registered_in_metadata() -> None:
    import xagent.web.models as models

    assert "task_interaction_requests" in models.Base.metadata.tables


def test_created_columns_match_the_frozen_shape(db_session) -> None:
    inspector = inspect(get_engine())
    columns = {c["name"]: c for c in inspector.get_columns("task_interaction_requests")}
    assert set(columns) == EXPECTED_COLUMNS
    for name, expected_nullable in EXPECTED_NULLABLE.items():
        assert columns[name]["nullable"] == expected_nullable, name
    for name, expected_length in EXPECTED_STRING_LENGTHS.items():
        assert columns[name]["type"].length == expected_length, name


def test_all_timestamp_columns_are_timezone_aware(db_session) -> None:
    """Unlike the SQLite half, PostgreSQL's reflection actually reports
    this: str(column["type"]) prints "TIMESTAMP" either way, but
    column["type"].timezone distinguishes timestamp with time zone from
    timestamp without time zone (verified against information_schema)."""
    inspector = inspect(get_engine())
    columns = {c["name"]: c for c in inspector.get_columns("task_interaction_requests")}
    for name in TIMESTAMP_COLUMNS:
        assert columns[name]["type"].timezone is True, name


def test_constraint_names_match_the_frozen_inventory(db_session) -> None:
    inspector = inspect(get_engine())

    unique = {
        uc["name"]: tuple(sorted(uc["column_names"]))
        for uc in inspector.get_unique_constraints("task_interaction_requests")
    }
    assert unique == EXPECTED_UNIQUE_CONSTRAINTS

    checks = {
        c["name"] for c in inspector.get_check_constraints("task_interaction_requests")
    }
    assert checks == EXPECTED_CHECK_CONSTRAINT_NAMES

    fks = {
        fk["name"]: (
            fk["referred_table"],
            tuple(fk["referred_columns"]),
            (fk["options"] or {}).get("ondelete"),
        )
        for fk in inspector.get_foreign_keys("task_interaction_requests")
    }
    assert fks == EXPECTED_FOREIGN_KEYS

    # PostgreSQL also reports the two UNIQUE constraints' backing indexes
    # here (unique=True); filtering to unique=False first is what makes
    # this literal apply to both backends -- see EXPECTED_NONUNIQUE_INDEXES.
    nonunique_indexes = {
        idx["name"]: tuple(sorted(idx["column_names"]))
        for idx in inspector.get_indexes("task_interaction_requests")
        if not idx["unique"]
    }
    assert nonunique_indexes == EXPECTED_NONUNIQUE_INDEXES


def test_no_constraint_name_exceeds_the_postgres_identifier_limit() -> None:
    """Structural guard for a real bug on this exact backend: SQLAlchemy
    raises IdentifierError at create_all time for an explicitly given name
    over 63 characters rather than truncating it -- see the SQLite half.
    """
    table = TaskInteractionRequest.__table__
    names = [c.name for c in table.constraints if c.name is not None]
    names += [ix.name for ix in table.indexes if ix.name is not None]
    too_long = [name for name in names if len(name) > 63]
    assert too_long == [], (
        f"constraint/index name(s) exceed PostgreSQL's 63-char identifier limit: {too_long}"
    )
