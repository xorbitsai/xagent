"""TaskInteractionRequest constraint-behavior and shape sentinels (SQLite half).

Companion to test_task_interaction_schema_postgresql.py; see
task_interaction_schema_shared.py's module docstring for why the suite is
split by backend and for the UNIQUE-violation message asymmetry between the
two. Every CHECK, both UniqueConstraints, and the anchor/task/responder
foreign-key delete behavior are pinned here against a real SQLite database
with foreign-key enforcement on (required -- see
test_sqlite_foreign_keys_pragma_is_on below); the create_all shape tests at
the bottom pin the reflected column, constraint and index inventory against
hand-written literals.
"""

from __future__ import annotations

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
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'task-interaction-schema.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture()
def fixtures(db_session):
    """A (task_id, anchor_trace_event_id) pair shared by most tests below."""
    user_id = make_user(db_session)
    task_id = make_task(db_session, user_id=user_id)
    anchor_id = make_trace_event(db_session, task_id=task_id)
    return task_id, anchor_id


# --------------------------------------------------------------------------
# SQLite FK-enforcement precondition
# --------------------------------------------------------------------------


def test_sqlite_foreign_keys_pragma_is_on(db_session) -> None:
    """Pins SQLite's per-connection foreign-key enforcement pragma to ON.

    Verified directly by temporarily flipping the pragma to OFF and
    rerunning the suite: every foreign-key delete test below already goes
    red on its own without it -- each one's assertion is tied straight to
    the ON DELETE action actually firing, so a missing pragma isn't
    silently swallowed by any of them. What this sentinel buys isn't
    catching an otherwise-invisible bug; it's turning a pragma regression
    into one test with an unambiguous, on-topic failure message, instead of
    four separate tests failing for reasons ("delete unexpectedly
    succeeded", "row count is 1 not 0", "responder_user_id is not None")
    that only make sense once you already know foreign-key enforcement is
    off.
    """
    value = db_session.execute(text("PRAGMA foreign_keys")).scalar_one()
    assert value == 1, "SQLite foreign-key enforcement must be on for this suite"


# --------------------------------------------------------------------------
# T-uq: uniqueness (3 active-slot + 2 identity)
# --------------------------------------------------------------------------


def test_active_slot_unique_rejects_second_active_row(db_session, fixtures) -> None:
    """UNIQUE(task_id, active_slot) caps a task at one active row."""
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
    """Sentinel that terminal rows (active_slot NULL) coexist without bound
    -- this is also the sentinel against NULLS NOT DISTINCT ever being added
    to uq_task_interaction_active_slot (see the model's class docstring):
    that option would collapse this test to at most one accepted row.
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
    """The active-slot uniqueness is per task, not global."""
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
    """UNIQUE(task_id, run_id, request_idempotency_key) is the idempotency
    identity. Both rows are built as terminated (active_slot NULL) so the
    active-slot constraint cannot also fire and blur which constraint this
    test is actually pinning.
    """
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
    """The same (task_id, request_idempotency_key) is allowed across
    different run_id values: a new run reusing an idempotency key from a
    previous run is not a collision.
    """
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
    """status is a closed vocabulary; active_slot is nulled here so the
    slot-pairing CHECK cannot also fire for an unrecognised status value.
    """
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id,
        resume_trace_event_id=anchor_id,
        status="bogus",
        active_slot=None,
    )
    assert_rejected(db_session, row, "ck_task_interaction_requests_status")


def test_kind_rejects_unknown_literal(db_session, fixtures) -> None:
    """kind is a closed vocabulary: storage fails closed for an unknown
    interaction kind.
    """
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id, resume_trace_event_id=anchor_id, kind="unknown_kind"
    )
    assert_rejected(db_session, row, "ck_task_interaction_requests_kind")


@pytest.mark.parametrize(
    "origin", ["internal", "sdk", "a2a", "trigger", "widget", "shared_link"]
)
def test_origin_accepts_every_vocabulary_member(db_session, fixtures, origin) -> None:
    """Every member of the origin vocabulary is accepted. Parametrized
    rather than a single case because only 'a2a' and the 'internal' default
    were exercised before -- a typo in any other IN-list entry would have
    passed both suites.
    """
    task_id, anchor_id = fixtures
    assert_accepted(
        db_session,
        make_row(task_id=task_id, resume_trace_event_id=anchor_id, origin=origin),
    )


def test_origin_rejects_unknown_literal(db_session, fixtures) -> None:
    task_id, anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=anchor_id, origin="email")
    assert_rejected(db_session, row, "ck_task_interaction_requests_origin")


def test_resume_checkpoint_type_and_locator_format_reject_unknown(
    db_session, fixtures
) -> None:
    """The two single-valued resume-locator vocabularies, each checked
    independently against its own row so the two rejections are not
    conflated.
    """
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
    """The three current terminal_reason members are each individually
    accepted, and the old name superseded_by_legacy_resume is rejected --
    this is a rename with nothing to migrate (the table has zero
    production rows), and this test is the regression guard against the
    old name quietly coming back.
    """
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
    """active_slot only ever takes 1 (or NULL)."""
    task_id, anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=anchor_id, active_slot=2)
    assert_rejected(db_session, row, "ck_task_interaction_requests_active_slot_value")


def test_active_status_and_slot_are_paired_both_ways(db_session, fixtures) -> None:
    """status='active' iff active_slot IS NOT NULL, both directions."""
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
    """An active row must carry a resume anchor -- this CHECK is also what
    turns the anchor FK's ON DELETE SET NULL into an effective RESTRICT for
    active rows (see T-fk-1 and the model's class docstring).
    """
    task_id, _anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=None)
    assert_rejected(db_session, row, "ck_task_interaction_requests_active_anchor")


def test_active_row_requires_protocol_version_one(db_session, fixtures) -> None:
    """protocol_version is pinned to 1 only while a row is active; a
    terminated row with protocol_version=2 is accepted, proving this CHECK
    does not reach terminal rows at all.
    """
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


def test_protocol_version_floor_rejects_nonsense_on_terminal_rows(
    db_session, fixtures
) -> None:
    """ck_task_interaction_requests_protocol_version_floor rejects values
    that are not version numbers at all (-1, 0) on every row, not just
    active ones -- unlike ck_task_interaction_requests_active_protocol
    above, which only reaches active rows.

    The positive control (terminated, protocol_version=2) is the design's
    evolution guarantee: a future-version row may exist historically, and
    that acceptance is intentional -- do not "fix" it into a rejection.
    """
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            protocol_version=-1,
        ),
        "ck_task_interaction_requests_protocol_version_floor",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            protocol_version=0,
        ),
        "ck_task_interaction_requests_protocol_version_floor",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="answered",
            protocol_version=-1,
        ),
        "ck_task_interaction_requests_protocol_version_floor",
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


def test_terminated_status_and_terminated_at_are_paired_both_ways(
    db_session, fixtures
) -> None:
    """status='terminated' iff terminated_at IS NOT NULL. The answered case
    is the one worth pinning: answered and terminated are distinct terminal
    statuses, and nothing stamps terminated_at on an answered row.
    """
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="terminated",
            terminated_at=None,
        ),
        "ck_task_interaction_requests_terminated_at_pairs_status",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            terminated_at=datetime.now(timezone.utc),
        ),
        "ck_task_interaction_requests_terminated_at_pairs_status",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="answered",
            terminated_at=datetime.now(timezone.utc),
        ),
        "ck_task_interaction_requests_terminated_at_pairs_status",
    )


def test_answered_status_and_response_payload_are_paired_both_ways(
    db_session, fixtures
) -> None:
    """status='answered' iff response_payload IS NOT NULL, both directions."""
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="answered",
            response_payload=None,
        ),
        "ck_task_interaction_requests_response_pairs_status",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            response_payload={"stray": "value"},
        ),
        "ck_task_interaction_requests_response_pairs_status",
    )


def test_answered_status_and_responded_at_are_paired_both_ways(
    db_session, fixtures
) -> None:
    """status='answered' iff responded_at IS NOT NULL, both directions."""
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="answered",
            responded_at=None,
            responder_identity=None,
        ),
        "ck_task_interaction_requests_responded_at_pairs_status",
    )
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            responded_at=datetime.now(timezone.utc),
            responder_identity="user:1",
        ),
        "ck_task_interaction_requests_responded_at_pairs_status",
    )


def test_responder_identity_pairs_responded_at_both_ways(db_session, fixtures) -> None:
    """(responded_at IS NULL) = (responder_identity IS NULL), both
    directions. Renamed from
    ck_task_interaction_requests_responder_identity_pairs_responded_at (66
    chars, over PostgreSQL's 63-character identifier limit); see the
    model's __table_args__ comment and
    test_no_constraint_name_exceeds_the_postgres_identifier_limit below.
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
    """The CHECK pins the sign of the TTL, not row freshness. Both
    created_at and expires_at are bound explicitly here, which makes the
    equal/before cases deterministic and keeps the assertion independent
    of the server-default format: a server-written created_at carries no
    fractional seconds on SQLite while a Python bind does, and that
    mixed-format comparison admits an inversion of up to one second --
    see the model's class docstring for the per-backend slack.
    """
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
    """Positive control: without this, T-ck-12..16 would all be rejection
    paths and nothing would prove that 'answered' is actually writable.
    """
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
    """NOT NULL alone does not stop a caller writing "" instead of leaving a
    column NULL -- checkpoint_execution_id has exactly this "or ''" shape
    elsewhere in the codebase, so these five columns each get an explicit
    non-empty CHECK instead of relying on NOT NULL.
    """
    task_id, anchor_id = fixtures
    row = make_row(task_id=task_id, resume_trace_event_id=anchor_id, **{column: ""})
    assert_rejected(db_session, row, f"ck_task_interaction_requests_{column}_nonempty")


def test_responder_identity_rejects_empty_string(db_session, fixtures) -> None:
    """'' satisfies the responder pairing CHECK but names no responder:
    the identity format is a namespaced "user:{id}" / "guest:{guest_id}".
    """
    task_id, anchor_id = fixtures
    assert_rejected(
        db_session,
        make_row(
            task_id=task_id,
            resume_trace_event_id=anchor_id,
            status="answered",
            responder_identity="",
        ),
        "ck_task_interaction_requests_responder_identity_nonempty",
    )


def test_run_id_may_differ_from_resume_run_partition(db_session, fixtures) -> None:
    """Negative control for the deleted CHECK
    (ck_task_interaction_requests_run_partition_matches): forcing equality
    would turn the corruption #1071 is meant to detect into a row that
    cannot be written at all. This proves that deleted CHECK has not
    quietly come back.
    """
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
    an active row, so the SET NULL the delete would trigger fails that
    CHECK instead of going through -- the effective behavior is RESTRICT,
    but the error surface is a CHECK violation, not a foreign key
    violation.
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
    """A terminated row's anchor may be cleared -- acceptable historical
    wear, per the model's class docstring."""
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
    """Uses a dedicated task with no trace_events pointing at it -- fixtures'
    shared task has one (its anchor), and trace_events.task_id has no
    ondelete action of its own, so deleting a task with a live trace_event
    would be blocked by that unrelated foreign key before this table's own
    CASCADE is ever reached.
    """
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
    """responder_user_id is ON DELETE SET NULL; responder_identity and
    responded_at are not touched by that action -- they are the durable
    audit trail, independent of whether the user account still exists (see
    the model's class docstring on why these two columns are absent from
    every two-way paired CHECK).
    """
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
    """response_payload uses JSON(none_as_null=True): Python None on
    response_payload must land as SQL NULL, checkable with IS NULL -- and a
    JSON null *nested inside* a payload (as opposed to the payload itself
    being None) must not be affected by that flag.
    """
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


def test_request_payload_none_is_rejected(db_session, fixtures) -> None:
    """request_payload is NOT NULL, and JSON(none_as_null=True) is what
    makes that constraint actually fire on a Python None -- without the
    flag, serialization runs before binding and None would reach the column
    as the JSON text 'null', which NOT NULL does not reject (see the model's
    class docstring). Not routed through assert_rejected: this is a column
    NOT NULL violation, not a named CHECK, so there is no constraint name
    for that helper to match against.
    """
    task_id, anchor_id = fixtures
    row = make_row(
        task_id=task_id, resume_trace_event_id=anchor_id, request_payload=None
    )
    obj = TaskInteractionRequest(**row)
    db_session.add(obj)
    with pytest.raises(IntegrityError, match="NOT NULL constraint failed"):
        db_session.commit()
    db_session.rollback()


def test_expires_at_round_trips_as_utc(db_session, fixtures) -> None:
    """A caller-bound aware UTC expires_at must round-trip to the same wall-
    clock instant.

    SQLite-specific wrinkle #1 (covered only here, not portable to
    PostgreSQL): DateTime(timezone=True) drops tzinfo on the *read* side
    too, not just on bind -- a value bound as aware UTC reads back as a
    naive datetime with the same digits, verified directly here rather
    than assumed. This is why the comparison below strips tzinfo from the
    expected value instead of comparing two aware datetimes.

    Wrinkle #2: a non-UTC-but-aware value is silently stored as local
    wall-clock time while every CHECK still passes -- the row is accepted
    with the wrong instant. This is why expires_at must always be bound as
    aware UTC by the caller (see the model's class docstring).
    """
    # created_at is bound explicitly rather than left to server_default=
    # func.now(), so expires_at > created_at holds regardless of the actual
    # wall-clock date the suite runs on.
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
    assert utc_row.expires_at == utc_value.replace(tzinfo=None)

    plus_eight_value = datetime(
        2026, 6, 1, 20, 0, 0, tzinfo=timezone(timedelta(hours=8))
    )
    # status="terminated" here: the first row above already holds this
    # task's one active slot (uq_task_interaction_active_slot), and this
    # test's concern is expires_at's storage form, not active/terminated
    # semantics.
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
    raw_stored = db_session.execute(
        text("SELECT expires_at FROM task_interaction_requests WHERE id = :id"),
        {"id": non_utc_row.id},
    ).scalar_one()
    # Hazard pin, not a contract. The +08:00 instant is 12:00 UTC, and the
    # PostgreSQL twin (same test name) asserts exactly that -- it reads back
    # 12:00. SQLite's bind path drops tzinfo instead of converting, so the
    # same value lands as the bare local wall-clock digits and the row is
    # accepted with the wrong instant. The divergence between the two suites
    # is measured, not accidental.
    #
    # If this assertion ever fails, SQLite bind-side normalization has been
    # introduced: delete this block, drop the corresponding wrinkle from this
    # test's docstring and from the model docstring's aware-non-UTC
    # paragraph, and narrow the caller's UTC obligation there accordingly.
    assert raw_stored.startswith("2026-06-01 20:00:00"), (
        f"the SQLite tzinfo-dropping hazard is no longer reproducing "
        f"(expected the local wall-clock digits '20:00:00'); got: {raw_stored!r}"
    )


# --------------------------------------------------------------------------
# T-shape: create_all shape
# --------------------------------------------------------------------------


def test_table_is_registered_in_metadata() -> None:
    """Registration is an import side effect (xagent/web/models/__init__.py
    must import the module for Base.metadata to carry the table) -- this
    only imports the package, not the module directly, to pin that path.
    """
    import xagent.web.models as models

    assert "task_interaction_requests" in models.Base.metadata.tables


def test_none_as_null_is_confined_to_this_table() -> None:
    """Pins the class docstring's claim that JSON(none_as_null=True) is
    confined to this table, carried by both of its JSON columns.

    Imports xagent.web.models first (same reason as
    test_table_is_registered_in_metadata above) so every model's table is
    actually registered on Base.metadata before this walks it. Only JSON
    type instances carry a none_as_null attribute at all, so the getattr
    default guards every other column type instead of assuming JSON. This is
    a guard, not a preference: no OTHER table may quietly adopt the flag
    without this test catching it.
    """
    import xagent.web.models as models

    none_as_null_columns = {
        (table.name, column.name)
        for table in models.Base.metadata.tables.values()
        for column in table.columns
        if getattr(column.type, "none_as_null", False)
    }
    assert none_as_null_columns == {
        ("task_interaction_requests", "request_payload"),
        ("task_interaction_requests", "response_payload"),
    }


def test_created_columns_match_the_frozen_shape(db_session) -> None:
    inspector = inspect(get_engine())
    columns = {c["name"]: c for c in inspector.get_columns("task_interaction_requests")}
    assert set(columns) == EXPECTED_COLUMNS
    for name, expected_nullable in EXPECTED_NULLABLE.items():
        assert columns[name]["nullable"] == expected_nullable, name
    string_columns = {
        name
        for name, column in columns.items()
        if getattr(column["type"], "length", None) is not None
    }
    assert set(EXPECTED_STRING_LENGTHS) == string_columns
    for name, expected_length in EXPECTED_STRING_LENGTHS.items():
        assert columns[name]["type"].length == expected_length, name


def test_all_timestamp_columns_are_timezone_aware() -> None:
    """Checked against the *declared* model column, not SQLite's reflection.

    On PostgreSQL, str(column["type"]) prints "TIMESTAMP" for both
    timezone-aware and naive reflected columns -- only the type object's
    .timezone attribute actually distinguishes them there (see the
    PostgreSQL half of this suite, which asserts it via
    inspect(engine).get_columns() the way the class docstring describes).

    SQLite has no comparable reflected signal at all: its DDL for
    DateTime(timezone=True) is the bare text "DATETIME" (confirmed via
    sqlite_master), identical to what
    DateTime(timezone=False) would emit, so
    inspect(engine).get_columns()["type"].timezone is False for every
    column on this backend regardless of what the model declared -- it is
    not observable through reflection on SQLite at all. This checks the
    same invariant (all five timestamp columns are declared timezone-aware)
    against TaskInteractionRequest.__table__ directly instead, which is
    what actually governs bind/result timezone handling at runtime.
    """
    table = TaskInteractionRequest.__table__
    for name in TIMESTAMP_COLUMNS:
        assert table.c[name].type.timezone is True, name


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

    # PostgreSQL's get_indexes() also reports the two UNIQUE constraints'
    # backing indexes; SQLite's does not (see EXPECTED_NONUNIQUE_INDEXES's
    # comment). Filtering to unique=False first makes this assertion apply
    # to both backends' reflection.
    nonunique_indexes = {
        idx["name"]: tuple(sorted(idx["column_names"]))
        for idx in inspector.get_indexes("task_interaction_requests")
        if not idx["unique"]
    }
    assert nonunique_indexes == EXPECTED_NONUNIQUE_INDEXES


def test_no_constraint_name_exceeds_the_postgres_identifier_limit() -> None:
    """Structural guard for a real bug: SQLAlchemy does not truncate an
    explicitly given constraint name that is too long for PostgreSQL's
    63-character identifier limit -- it raises IdentifierError at
    create_all time instead. ck_task_interaction_requests_responder_
    identity_pairs_responded_at (66 chars) hit exactly this and was
    renamed; this test pins the limit structurally so a future added
    column or constraint cannot silently regress past it again.
    """
    table = TaskInteractionRequest.__table__
    names = [c.name for c in table.constraints if c.name is not None]
    names += [ix.name for ix in table.indexes if ix.name is not None]
    too_long = [name for name in names if len(name) > 63]
    assert too_long == [], (
        f"constraint/index name(s) exceed PostgreSQL's 63-char identifier limit: {too_long}"
    )
