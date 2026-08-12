"""Unit coverage for ``task_interaction_service``: the outcome vocabularies,
the ``create()`` typed seam, and the compatibility materialization view.

This module accumulates coverage across every deliverable this service
ships except the shared public-chat ownership predicate (covered directly
by ``tests/web/api/test_public_chat_ownership_helper.py``, since it is
extracted from and tested alongside ``public_chat_access.py``) and the
production-caller gate (its own file,
``test_task_interaction_service_production_gate.py``).

RespondOutcome's failure matrix, and what this delivery does and does not
do with it: the matrix enumerates 25 triggering cells across
``respond()``'s not-yet-implemented logic, producing 21 distinct (outcome
type, reason) pairs -- fewer than 25 because several cells share a pair
(five different "principal does not own this task" cells all produce
``(RespondUnauthorized, not_task_principal)``; two "same idempotency key,
different actor" cells both produce ``(RespondConflict,
idempotency_key_reused)``; one cell, kind/version validation, is
parametrized over two reasons on its own). The full cell-to-pair mapping:

    OK        -> (Accepted, None)                                    1
    V1        -> (ValidationRejected, unknown_kind)                  }  2
                 (ValidationRejected, unknown_protocol_version)      }
    V2        -> (ValidationRejected, malformed_idempotency_key)     1
    V3        -> (ValidationRejected, invalid_values)                1
    V5        -> (ValidationRejected, kind_version_mismatch)         1
    A1..A5    -> (Unauthorized, not_task_principal)      1 (5 cells share it)
    U1        -> (Unavailable, task_missing)                         1
    U2        -> (Unavailable, interaction_missing)                  1
    U3        -> (Unavailable, checkpoint_unavailable)                1
    R1        -> (Replayed, None)                                    1
    C1        -> (Conflict, already_answered)                        1
    C2,C3     -> (Conflict, idempotency_key_reused)      1 (2 cells share it)
    S1..S7    -> seven distinct (Stale, *) pairs                     7
    X1        -> (OutcomeUnknown, None)                              1
    -----------------------------------------------------------------
    25 cells; 21 distinct pairs (19 single-reason cells + V1's own 2)

This delivery does **not** write the 25 cell-by-cell tests this matrix
implies -- ``respond()`` itself is not implemented here (see the module
docstring in ``task_interaction_service.py``), so there is no behavior yet
for those tests to pin. What this delivery does write is the vocabulary
guard below (``test_respond_outcome_vocabulary_has_exactly_21_pairs``),
which is a different assertion from either the cell-by-cell tests or a
mapping meta-test, per the three-way division of labor between the
assertion layers:

| Assertion | Checks | Catches | Misses |
|---|---|---|---|
| Vocabulary closure (this file, written) | The set of (outcome, reason) pairs RespondOutcome can produce == the 21-pair vocabulary | A new reason constant or outcome variant added without updating the vocabulary | Cell-level coverage |
| Cell-by-cell tests (not written here; land with respond()) | One test per of the 25 cells, asserting outcome + reason + zero side effects | A regression in one specific cell's behavior | A forgotten test |
| Mapping meta-test (not written here; lands with respond()) | Each of the 21 pairs is produced by >= 1 cell's test (19 singles + V1's 2) | A new cell that produces a new pair with no test written for it; V1's parametrization missing a reason | A new cell that produces no *new* pair (e.g. a sixth not_task_principal scenario) -- caught by review, not this meta-test |
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_interaction_schema_shared import make_task, make_user
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import task_interaction_service as svc
from xagent.web.services.ops_signals import (
    CHECKPOINT_LOAD_UNAVAILABLE,
    CHECKPOINT_PK_ANCHOR_DANGLING,
    clear_degradation,
)
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD


@pytest.fixture(autouse=True)
def _clean_degradation_registry():
    """The anchor resolver registers process-global degradation signals on
    its failure paths; clear this module's two signals around every test so
    they cannot leak into tests that read the shared registry (the /health
    suite asserts exact payloads and fails on any leftover entry)."""
    for signal in (CHECKPOINT_PK_ANCHOR_DANGLING, CHECKPOINT_LOAD_UNAVAILABLE):
        clear_degradation(signal)
    yield
    for signal in (CHECKPOINT_PK_ANCHOR_DANGLING, CHECKPOINT_LOAD_UNAVAILABLE):
        clear_degradation(signal)


# ---------------------------------------------------------------------------
# Vocabulary guards (three pinned numbers -- do not recompute them here):
#
#   - RespondOutcome: 21 (outcome type, reason) pairs.
#   - CreateOutcome reason word list: 13 words total, across both delivery
#     periods.
#   - CreateOutcome pairs producible in this delivery specifically: 7.
#
# These guards prove the vocabulary stays closed at exactly these counts.
# They do NOT prove every pair has a test written against it -- several
# reasons are reachable from more than one triggering condition and are
# indistinguishable at the (type, reason) level alone (see each dict's own
# comment in the source for which ones collapse).
# ---------------------------------------------------------------------------


def test_respond_outcome_vocabulary_has_exactly_21_pairs() -> None:
    total = sum(
        len(reasons) for reasons in svc.RESPOND_OUTCOME_REASON_VOCABULARY.values()
    )
    assert total == 21


def test_respond_outcome_vocabulary_covers_all_eight_variants() -> None:
    assert set(svc.RESPOND_OUTCOME_REASON_VOCABULARY) == {
        "RespondAccepted",
        "RespondValidationRejected",
        "RespondUnauthorized",
        "RespondUnavailable",
        "RespondReplayed",
        "RespondConflict",
        "RespondStale",
        "RespondOutcomeUnknown",
    }


def test_create_outcome_reason_word_list_has_exactly_13_words() -> None:
    assert len(svc.CREATE_OUTCOME_REASON_WORDS) == 13


def test_create_outcome_producible_pairs_in_this_delivery_are_exactly_7() -> None:
    total = sum(
        len(reasons) for reasons in svc.CREATE_OUTCOME_REASON_VOCABULARY.values()
    )
    assert total == 7


def test_create_outcome_producible_reasons_are_a_subset_of_the_full_word_list() -> None:
    producible = {
        reason
        for reasons in svc.CREATE_OUTCOME_REASON_VOCABULARY.values()
        for reason in reasons
    }
    assert producible <= svc.CREATE_OUTCOME_REASON_WORDS


def test_create_outcome_this_period_covers_exactly_the_four_producible_variants() -> (
    None
):
    """CreateCreated, CreateConflict, and CreateStale are not producible
    until a later change fills create()'s call body --
    create() never stages a row in this delivery, so nothing that requires
    a staged row can be returned yet."""

    assert set(svc.CREATE_OUTCOME_REASON_VOCABULARY) == {
        "CreateValidationRejected",
        "CreateUnauthorized",
        "CreateUnavailable",
        "CreateNotWired",
    }


def test_locator_mismatch_reason_constant_does_not_exist_in_source() -> None:
    """The reason 'locator_mismatch' is deliberately not part of this
    vocabulary. Asserting its absence guards against it surviving as a
    dead string constant that would mislead a future reader into thinking
    that path is still live."""

    import inspect

    source = inspect.getsource(svc)
    assert "locator_mismatch" not in source


# ---------------------------------------------------------------------------
# build_v1_request_payload(): its output must always pass the identical
# JSON-serializability probe stage_interaction_request runs before its
# own INSERT.
# ---------------------------------------------------------------------------


def test_build_v1_request_payload_output_passes_the_json_serializability_probe() -> (
    None
):
    parsed = svc.parse_v1_request_payload(_valid_values())
    payload = svc.build_v1_request_payload(parsed)
    # The identical probe stage_interaction_request runs; does not raise.
    json.dumps(payload, allow_nan=False)


def test_build_v1_request_payload_rejects_nan_default_value() -> None:
    values = {
        "message": "Pick a number",
        "interactions": [
            {
                "type": "number_input",
                "field": "n",
                "label": "N",
                "default_value": float("nan"),
            }
        ],
    }
    parsed = svc.parse_v1_request_payload(values)
    with pytest.raises(ValueError):
        svc.build_v1_request_payload(parsed)


def test_create_rejects_nan_default_value_as_invalid_values(
    _db: Session, _seeded_task: int
) -> None:
    values = {
        "message": "Pick a number",
        "interactions": [
            {
                "type": "number_input",
                "field": "n",
                "label": "N",
                "default_value": float("inf"),
            }
        ],
    }
    envelope = _valid_envelope(values=values)
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=_owning_principal(1), envelope=envelope
    )
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")


# ---------------------------------------------------------------------------
# create(): the six (outcome, reason) pairs producible in this delivery.
# CC1 (slot_taken / idempotency_key_reused), CS1 (anchor_dangling /
# run_ended), and CU2 (checkpoint_unavailable / anchor_run_mismatch) are not
# producible -- create() never stages a row, so nothing that requires one
# can happen. Those three pairs become producible only once the wiring
# batch fills create()'s call body.
# ---------------------------------------------------------------------------


@pytest.fixture
def _engine(tmp_path: Path):
    db_path = tmp_path / "task_interaction_service.db"
    engine = create_engine(f"sqlite:///{db_path}")
    apply_sqlite_concurrency_pragmas(engine)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def _session_factory(_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=_engine)


@pytest.fixture
def _seeded_task(_session_factory) -> int:
    db = _session_factory()
    try:
        user_id = make_user(db)
        task_id = make_task(db, user_id=user_id)
        # run_id="run-a" matches every fixture row below by default --
        # the active-row predicate requires TaskInteractionRequest.run_id
        # == Task.run_id, so a task with no run_id would make every active
        # row invisible regardless of the scenario under test.
        task = db.query(Task).filter(Task.id == task_id).first()
        task.run_id = "run-a"
        db.commit()
        return task_id
    finally:
        db.close()


@pytest.fixture
def _db(_session_factory) -> Session:
    db = _session_factory()
    try:
        yield db
    finally:
        db.close()


def _valid_values() -> dict[str, Any]:
    return {
        "message": "Which environment?",
        "interactions": [
            {"type": "text_input", "field": "env", "label": "Environment"}
        ],
    }


def _valid_envelope(**overrides: Any) -> svc.CreateInteractionEnvelope:
    defaults: dict[str, Any] = {
        "kind": "clarification",
        "protocol_version": 1,
        "request_idempotency_key": "create-key-1",
        "values": _valid_values(),
        "ttl_seconds": None,
    }
    defaults.update(overrides)
    return svc.CreateInteractionEnvelope(**defaults)


def _owning_principal(task_user_id: int) -> svc.InteractionPrincipal:
    return svc.InteractionPrincipal(
        kind="user",
        user_id=task_user_id,
        is_admin=False,
        channel_id=None,
        auth_mode=None,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"kind": "not_a_real_kind"}, id="unknown_kind"),
        pytest.param({"protocol_version": 2}, id="unknown_protocol_version"),
    ],
)
def test_cv1_unknown_kind_or_protocol_version_is_rejected(
    _db: Session, _seeded_task: int, overrides: dict[str, Any]
) -> None:
    envelope = _valid_envelope(**overrides)
    outcome = svc.create(
        _db,
        task_id=_seeded_task,
        principal=_owning_principal(1),
        envelope=envelope,
    )
    assert isinstance(outcome, svc.CreateValidationRejected)
    expected = "unknown_kind" if "kind" in overrides else "unknown_protocol_version"
    assert outcome.reason == expected


@pytest.mark.parametrize(
    "bad_kind",
    [
        pytest.param(["clarification"], id="list"),
        pytest.param({"clarification": True}, id="dict"),
    ],
)
def test_cv1_unhashable_kind_is_rejected_without_raising(
    _db: Session, _seeded_task: int, bad_kind: Any
) -> None:
    """A ``kind`` that is not a str -- in particular one that is unhashable,
    like a list or a dict -- must be caught by an isinstance guard before the
    ``in _KIND_VOCABULARY`` membership check ever runs. ``_KIND_VOCABULARY``
    is a frozenset, so testing membership of an unhashable value raises
    ``TypeError: unhashable type``, not a typed outcome. (A bare ``set`` is
    deliberately not used here: CPython's set/frozenset ``__contains__`` has
    a special case for a set-typed probe value and hashes it as if it were a
    frozenset instead of raising, so it would not reproduce the bug this
    test pins.) This mirrors the isinstance-first discipline
    ``request_idempotency_key`` already gets (see
    ``test_cv2_non_string_idempotency_key_is_rejected_without_raising``
    above)."""
    envelope = _valid_envelope(kind=bad_kind)
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=_owning_principal(1), envelope=envelope
    )
    assert outcome == svc.CreateValidationRejected(reason="unknown_kind")


@pytest.mark.parametrize(
    "bad_version",
    [
        pytest.param(True, id="bool_true_equals_one"),
        pytest.param(1.0, id="float_equals_one"),
    ],
)
def test_cv1_protocol_version_type_confusable_values_are_rejected(
    _db: Session, _seeded_task: int, bad_version: Any
) -> None:
    """``protocol_version != INTERACTION_PROTOCOL_VERSION`` alone is not
    enough: ``True == 1`` and ``1.0 == 1`` both hold in Python, so a bare
    ``!=`` check lets a bool or a float through as if it were the int ``1``.
    The check must reject any non-``int`` (bools included, since ``bool`` is
    a subclass of ``int``) the same way the existing ``ttl_seconds`` check a
    few lines below already does."""
    envelope = _valid_envelope(protocol_version=bad_version)
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=_owning_principal(1), envelope=envelope
    )
    assert outcome == svc.CreateValidationRejected(reason="unknown_protocol_version")


def test_cv2_malformed_idempotency_key_is_rejected(
    _db: Session, _seeded_task: int
) -> None:
    envelope = _valid_envelope(request_idempotency_key="has a space")
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=_owning_principal(1), envelope=envelope
    )
    assert outcome == svc.CreateValidationRejected(reason="malformed_idempotency_key")


def test_cv3_values_not_shaped_like_v1_payload_is_rejected(
    _db: Session, _seeded_task: int
) -> None:
    envelope = _valid_envelope(values={"not": "a valid payload"})
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=_owning_principal(1), envelope=envelope
    )
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")


def test_cv3_ttl_out_of_policy_range_is_rejected_not_clamped(
    _db: Session, _seeded_task: int
) -> None:
    envelope = _valid_envelope(ttl_seconds=1)
    assert envelope.ttl_seconds < svc._MIN_INTERACTION_TTL_SECONDS
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=_owning_principal(1), envelope=envelope
    )
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")


@pytest.mark.parametrize(
    "ttl_seconds",
    [
        pytest.param(604801, id="one_above_max_rejected"),
        # True is also rejected via the range check below on its own (it
        # compares equal to 1, under the 60-second minimum), independent of
        # the isinstance(..., bool) branch above it -- confirmed by mutation
        # testing: deleting that isinstance(bool) exclusion from create()
        # cannot turn any case red under the current bounds (both bool
        # values fall below the floor). Kept anyway because it still pins a
        # real, correct outcome (a bool ttl_seconds must be rejected), just
        # not specifically through the bool-exclusion branch.
        pytest.param(True, id="bool_true_rejected"),
        pytest.param("60", id="numeric_string_rejected_not_coerced"),
    ],
)
def test_cv3_ttl_invalid_values_are_rejected(
    _db: Session, _seeded_task: int, ttl_seconds: Any
) -> None:
    envelope = _valid_envelope(ttl_seconds=ttl_seconds)
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=_owning_principal(1), envelope=envelope
    )
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")


@pytest.mark.parametrize(
    "ttl_seconds",
    [
        pytest.param(60, id="min_boundary_passes"),
        pytest.param(604800, id="max_boundary_passes"),
    ],
)
def test_cv3_ttl_at_policy_boundary_reaches_create_not_wired(
    _db: Session, _seeded_task: int, ttl_seconds: int
) -> None:
    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    envelope = _valid_envelope(ttl_seconds=ttl_seconds)
    outcome = svc.create(
        _db,
        task_id=_seeded_task,
        principal=_owning_principal(task.user_id),
        envelope=envelope,
    )
    assert outcome == svc.CreateNotWired(reason="seam_not_wired")


def test_ca1_principal_not_owning_the_task_is_unauthorized(
    _db: Session, _seeded_task: int
) -> None:
    envelope = _valid_envelope()
    outcome = svc.create(
        _db,
        task_id=_seeded_task,
        principal=_owning_principal(999999),
        envelope=envelope,
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")


def test_cu1_missing_task_is_unavailable(_db: Session) -> None:
    envelope = _valid_envelope()
    outcome = svc.create(
        _db, task_id=999999999, principal=_owning_principal(1), envelope=envelope
    )
    assert outcome == svc.CreateUnavailable(reason="task_missing")


def test_cw1_fully_valid_call_returns_not_wired(
    _db: Session, _seeded_task: int
) -> None:
    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    envelope = _valid_envelope()
    outcome = svc.create(
        _db,
        task_id=_seeded_task,
        principal=_owning_principal(task.user_id),
        envelope=envelope,
    )
    assert outcome == svc.CreateNotWired(reason="seam_not_wired")


def test_create_never_touches_staging_or_stages_a_row(
    _db: Session, _seeded_task: int
) -> None:
    """create() must not call stage_interaction_request -- confirmed here
    by asserting the table it would write to stays empty across a
    successful (CreateNotWired) call."""

    envelope = _valid_envelope()
    outcome = svc.create(
        _db,
        task_id=_seeded_task,
        principal=svc.InteractionPrincipal(
            kind="user",
            user_id=None,
            is_admin=True,
            channel_id=None,
            auth_mode=None,
        ),
        envelope=envelope,
    )
    assert isinstance(outcome, svc.CreateNotWired)
    assert _db.query(TaskInteractionRequest).count() == 0


@pytest.mark.parametrize(
    "bad_key",
    [
        pytest.param(None, id="none"),
        pytest.param(7, id="int"),
        pytest.param(b"create-key-1", id="bytes"),
    ],
)
def test_cv2_non_string_idempotency_key_is_rejected_without_raising(
    _db: Session, _seeded_task: int, bad_key: Any
) -> None:
    """A non-string request_idempotency_key must be caught by the isinstance
    guard before _normalize_command_id is ever called -- none of these three
    types would raise ValueError from that function (None/int/bytes each
    fail differently, some not at all: _normalize_command_id calls
    .strip() then a regex fullmatch, and a bytes object has its own
    .strip() that would not raise), so relying on a broadened except clause
    to catch them would either miss some or swallow unrelated bugs. All
    three must produce the same typed rejection with no exception
    escaping."""

    envelope = _valid_envelope(request_idempotency_key=bad_key)
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=_owning_principal(1), envelope=envelope
    )
    assert outcome == svc.CreateValidationRejected(reason="malformed_idempotency_key")


# ---------------------------------------------------------------------------
# Guest-principal authorization coverage for create()'s CA1 branch. The
# "user" kind is covered above (test_ca1_principal_not_owning_the_task_is_
# unauthorized); these cover the "guest" branch and the two fail-closed
# branches (a malformed principal, and an unrecognized kind) that branch
# sits between.
# ---------------------------------------------------------------------------


def _widget_workforce_task(db: Session, *, user_id: int, workforce_id: int) -> int:
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.agent_config = {
        "auth_mode": "widget",
        "guest_id": "guest-1",
        "widget_workforce_id": workforce_id,
    }
    db.commit()
    return task_id


def _widget_workforce_guest_principal(
    *, user_id: int, workforce_id: int, guest_id: str = "guest-1"
) -> svc.InteractionPrincipal:
    return svc.InteractionPrincipal(
        kind="guest",
        user_id=user_id,
        is_admin=False,
        channel_id=None,
        auth_mode="widget",
        widget_workforce_id=workforce_id,
        guest_id=guest_id,
    )


def test_ca1_guest_principal_is_authorized_on_its_own_task(
    _db: Session, _session_factory
) -> None:
    db = _session_factory()
    user_id = make_user(db)
    task_id = _widget_workforce_task(db, user_id=user_id, workforce_id=9)
    db.close()

    principal = _widget_workforce_guest_principal(user_id=user_id, workforce_id=9)
    outcome = svc.create(
        _db, task_id=task_id, principal=principal, envelope=_valid_envelope()
    )
    assert isinstance(outcome, svc.CreateNotWired)


def test_ca1_guest_principal_is_rejected_on_a_non_matching_task(
    _db: Session, _session_factory
) -> None:
    db = _session_factory()
    user_id = make_user(db)
    task_id = _widget_workforce_task(db, user_id=user_id, workforce_id=9)
    db.close()

    # Same owner, same auth_mode, but a different workforce_id -- the
    # entity-binding conjunct must reject this, not the (correctly
    # matching) owner or auth_mode conjuncts.
    principal = _widget_workforce_guest_principal(user_id=user_id, workforce_id=999)
    outcome = svc.create(
        _db, task_id=task_id, principal=principal, envelope=_valid_envelope()
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")


def test_ca1_guest_principal_with_two_populated_directions_is_unauthorized_not_raised(
    _db: Session, _seeded_task: int
) -> None:
    """A malformed principal that populates more than one of the four
    entity-binding fields makes task_is_owned_by_public_principal raise
    ValueError; create() must catch exactly that and translate it to
    Unauthorized(not_task_principal), not let it escape as an unhandled
    exception."""

    principal = svc.InteractionPrincipal(
        kind="guest",
        user_id=1,
        is_admin=False,
        channel_id=None,
        auth_mode="widget",
        widget_agent_id=1,
        widget_workforce_id=1,
        guest_id="guest-1",
    )
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=principal, envelope=_valid_envelope()
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")


def test_ca1_guest_principal_with_zero_populated_directions_is_unauthorized_not_raised(
    _db: Session, _seeded_task: int
) -> None:
    principal = svc.InteractionPrincipal(
        kind="guest",
        user_id=1,
        is_admin=False,
        channel_id=None,
        auth_mode="widget",
        guest_id="guest-1",
    )
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=principal, envelope=_valid_envelope()
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")


def test_ca1_unknown_principal_kind_is_always_unauthorized(
    _db: Session, _seeded_task: int
) -> None:
    """A principal.kind that is neither "user" nor "guest" must be
    rejected -- there is no third branch that defaults to allow."""

    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    principal = svc.InteractionPrincipal(
        kind="robot",
        user_id=task.user_id,
        is_admin=True,
        channel_id=None,
        auth_mode=None,
    )
    outcome = svc.create(
        _db, task_id=_seeded_task, principal=principal, envelope=_valid_envelope()
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")


def test_ca1_entity_binding_with_non_int_convertible_config_value_is_rejected_not_raised(
    _db: Session, _session_factory
) -> None:
    """agent_config is untrusted JSON another writer controls. A
    non-int-convertible widget_workforce_id (a non-numeric string here)
    must make the entity-binding conjunct fail closed, not raise -- the
    old int(x or 0) shape this replaces would raise ValueError on this
    exact input, since "not-a-number" is truthy and int("not-a-number")
    is not a valid conversion."""

    db = _session_factory()
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.agent_config = {
        "auth_mode": "widget",
        "guest_id": "guest-1",
        "widget_workforce_id": "not-a-number",
    }
    db.commit()
    db.close()

    principal = _widget_workforce_guest_principal(user_id=user_id, workforce_id=9)
    outcome = svc.create(
        _db, task_id=task_id, principal=principal, envelope=_valid_envelope()
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")


# ---------------------------------------------------------------------------
# materialize_compatibility_view(): the three-tier compatibility read.
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_trace_event(
    db: Session,
    *,
    task_id: int,
    run_partition: str = "run-a",
    execution_id: str = "exec-1",
    event_type: str = str(CHECKPOINT_EVENT_TYPE),
    checkpoint_type: str = "agent_execution_checkpoint",
    build_id: str | None = None,
) -> int:
    event = TraceEvent(
        task_id=task_id,
        event_id=f"trace-event-{task_id}",
        event_type=event_type,
        timestamp=_now(),
        build_id=build_id,
        data={
            TASK_RUN_ID_TRACE_FIELD: run_partition,
            "checkpoint_type": checkpoint_type,
            "execution_id": execution_id,
        },
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return int(event.id)


def _make_active_interaction_row(
    db: Session,
    *,
    task_id: int,
    run_id: str = "run-a",
    resume_trace_event_id: int | None,
    resume_run_partition: str = "run-a",
    resume_execution_id: str = "exec-1",
    protocol_version: int = 1,
    request_payload: dict[str, Any] | None = None,
) -> TaskInteractionRequest:
    now = _now()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id=run_id,
        kind="clarification",
        protocol_version=protocol_version,
        status="active",
        active_slot=1,
        origin="internal",
        request_payload=request_payload
        if request_payload is not None
        else {
            "message": "Which environment?",
            "interactions": [
                {"type": "text_input", "field": "env", "label": "Environment"}
            ],
        },
        request_idempotency_key=f"key-{task_id}",
        resume_trace_event_id=resume_trace_event_id,
        resume_event_id="resume-event-1",
        resume_execution_id=resume_execution_id,
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition=resume_run_partition,
        created_at=now,
        expires_at=now + timedelta(minutes=15),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _make_answered_interaction_row(db: Session, *, task_id: int, run_id: str) -> None:
    now = _now()
    row = TaskInteractionRequest(
        task_id=task_id,
        run_id=run_id,
        kind="clarification",
        protocol_version=1,
        status="answered",
        active_slot=None,
        origin="internal",
        request_payload={"message": "old question", "interactions": []},
        response_payload={"env": "prod"},
        request_idempotency_key=f"answered-key-{task_id}",
        resume_trace_event_id=None,
        resume_event_id="resume-event-2",
        resume_execution_id="exec-2",
        resume_locator_format="trace_event_pk_v1",
        resume_checkpoint_type="agent_execution_checkpoint",
        resume_run_partition=run_id,
        created_at=now,
        expires_at=now + timedelta(minutes=15),
        responder_identity="user:1",
        responded_at=now,
    )
    db.add(row)
    db.commit()


def test_t1_falls_back_to_legacy_when_the_table_does_not_exist(
    _db: Session, _seeded_task: int
) -> None:
    TaskInteractionRequest.__table__.drop(bind=_db.get_bind())
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "legacy"
    assert view.reason is None


def test_t1_falls_back_to_legacy_when_there_is_no_active_row(
    _db: Session, _seeded_task: int
) -> None:
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "legacy"
    assert view.question is None
    assert view.interactions is None


def test_t1_falls_back_to_legacy_when_task_run_id_is_null(_db: Session) -> None:
    """``_active_native_row_criteria()`` joins on
    ``TaskInteractionRequest.run_id == Task.run_id``. SQL NULL never
    compares equal to anything, including another NULL, so a task whose
    ``run_id`` is ``None`` cannot match any interaction row's ``run_id`` --
    active or not. This pins that as the current, deliberate behavior (see
    the ``_seeded_task`` fixture's own comment above): a task that has not
    started a run yet has no native interaction visible through this seam
    and always falls back to the legacy view, even with an active row
    sitting in the table."""

    user_id = make_user(_db)
    task_id = make_task(_db, user_id=user_id)
    task = _db.query(Task).filter(Task.id == task_id).first()
    assert task.run_id is None

    trace_event_id = _make_trace_event(_db, task_id=task_id)
    _make_active_interaction_row(
        _db, task_id=task_id, resume_trace_event_id=trace_event_id
    )

    view = svc.materialize_compatibility_view(_db, task_id)
    assert view.tier == "legacy"


def test_t1_falls_back_to_legacy_when_protocol_version_is_unrecognized(
    _db: Session, _seeded_task: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ck_task_interaction_requests_active_protocol`` pins every active
    row's protocol_version to 1 today, so this branch cannot be reached
    through any real write against this schema -- it defends against a
    future protocol version whose active-row CHECK has not been written
    yet. Monkeypatching the row lookup is how this delivery tests a branch
    the schema itself does not yet allow to be constructed."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    row = _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )
    row.protocol_version = 2

    def _fake_active_row(db: Session, task_id: int) -> TaskInteractionRequest:
        return row

    monkeypatch.setattr(svc, "_active_native_row", _fake_active_row)
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "legacy"


def test_t1_falls_back_to_legacy_when_request_payload_does_not_parse(
    _db: Session, _seeded_task: int
) -> None:
    """The active row's request_payload is a JSON column with no
    AskUserQuestionArgs-shape CHECK -- a row can carry any JSON dict
    that satisfies NOT NULL, so this branch is reachable through a real
    write, unlike the protocol_version branch above. A missing "message"
    field is enough to fail parse_v1_request_payload's pydantic
    validation."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db,
        task_id=_seeded_task,
        resume_trace_event_id=trace_event_id,
        request_payload={"not": "a valid v1 payload"},
    )
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "legacy"


def test_t2_native_projection_when_the_anchor_resolves(
    _db: Session, _seeded_task: int
) -> None:
    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "native"
    assert view.question == "Which environment?"
    assert view.interactions == [
        {
            "type": "text_input",
            "field": "env",
            "label": "Environment",
            "options": None,
            "placeholder": None,
            "multiline": False,
            "min": None,
            "max": None,
            "default_value": None,
            "accept": None,
            "multiple": False,
        }
    ]
    assert view.reason is None


def _force_dangling_pointer(db: Session, *, interaction_id: int) -> None:
    """Point an already-committed active row's anchor at a trace_events id
    that does not exist. This state cannot arise through any write this
    schema's own CHECK + FK constraints allow -- an INSERT with a bad
    pointer is rejected by the FK, and deleting the pointed-to row while
    the interaction row is still active is rejected by
    ck_task_interaction_requests_active_anchor -- so simulating it for a
    defensive-path test means bypassing FK enforcement for one raw write,
    the same way a real corruption (an out-of-band DB intervention, a
    migration bug) would bypass the ORM layer that normally enforces it.

    Uses an independent sqlite3 connection to the same file rather than the
    session's own connection: SQLite ignores a ``PRAGMA foreign_keys``
    change issued while a transaction is already open on that connection,
    and disturbing the session's own transaction state here would leak
    into the assertions that follow.
    """

    db_path = str(db.get_bind().url.database)
    raw = sqlite3.connect(db_path)
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        raw.execute(
            "UPDATE task_interaction_requests SET resume_trace_event_id = ? "
            "WHERE id = ?",
            (999999999, interaction_id),
        )
        raw.commit()
    finally:
        raw.close()
    db.expire_all()


def test_t3_anchor_dangling_when_the_pointer_names_no_row(
    _db: Session, _seeded_task: int
) -> None:
    """One of this delivery's two mutation-test guards: folding the T3
    branch back into the T1 legacy fallback must turn this test red, while
    the T1 tests above stay green -- proving the suite actually
    distinguishes "there is an active row but it cannot be answered right
    now" from "there is no active row at all"."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    row = _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )
    _force_dangling_pointer(_db, interaction_id=row.id)
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert view.reason == "anchor_dangling"
    assert view.question == "Which environment?"
    assert view.interactions is None


def test_t3_prime_anchor_dangling_when_the_row_fails_validation(
    _db: Session, _seeded_task: int
) -> None:
    """T3': same reason code as a missing row -- a pointer that resolves to
    an invalid row and a pointer that resolves to nothing are the same fact
    from this reader's side (see _resolve_read_direction_anchor's
    docstring for why the registration surface is deliberately wider than
    trace_handlers')."""

    trace_event_id = _make_trace_event(
        _db, task_id=_seeded_task, run_partition="a-different-run"
    )
    _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert view.reason == "anchor_dangling"


def test_t3_does_not_fall_back_to_legacy(_db: Session, _seeded_task: int) -> None:
    """A T3 result must never present
    as "no active row" -- it must always be the unanswerable tier, never
    the legacy tier, even though get_latest_waiting_question would also
    return (None, None) for this same task if it were consulted."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    row = _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )
    _force_dangling_pointer(_db, interaction_id=row.id)
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier != "legacy"


def test_t3_checkpoint_unavailable_when_the_anchor_fetch_raises(
    _db: Session, _seeded_task: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T3's second reason: the anchor row fetch itself raises (a session
    or query-layer failure), distinct from anchor_dangling -- that reason
    covers the pointer naming a missing or invalid row, not the read
    infrastructure failing before it can even answer that question."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )

    real_get = _db.get

    def _raising_get(model: Any, pk: Any, *args: Any, **kwargs: Any) -> Any:
        if model is TraceEvent:
            raise RuntimeError("simulated session failure")
        return real_get(model, pk, *args, **kwargs)

    monkeypatch.setattr(_db, "get", _raising_get)
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert view.reason == "checkpoint_unavailable"


def test_stale_run_active_row_is_invisible(_db: Session, _session_factory) -> None:
    """A5-P2, task 2: the active row was staged under a run the task has
    since moved past. Falls back to legacy, exactly like "no active row"."""

    db = _session_factory()
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-current"
    db.commit()
    trace_event_id = _make_trace_event(db, task_id=task_id, run_partition="run-old")
    _make_active_interaction_row(
        db,
        task_id=task_id,
        run_id="run-old",
        resume_trace_event_id=trace_event_id,
        resume_run_partition="run-old",
    )
    view = svc.materialize_compatibility_view(db, task_id)
    assert view.tier == "legacy"
    db.close()


def test_answered_row_is_invisible(_db: Session, _seeded_task: int) -> None:
    """A5-P2, task 3: an answered row is not an active row and must not be
    projected as one."""

    _make_answered_interaction_row(_db, task_id=_seeded_task, run_id="run-a")
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "legacy"


def test_list_returns_only_the_active_row_not_the_answered_one(
    _db: Session, _session_factory
) -> None:
    db = _session_factory()
    user_id = make_user(db)
    task_id = make_task(db, user_id=user_id)
    task = db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-a"
    db.commit()
    trace_event_id = _make_trace_event(db, task_id=task_id)
    active = _make_active_interaction_row(
        db, task_id=task_id, resume_trace_event_id=trace_event_id
    )
    _make_answered_interaction_row(db, task_id=task_id, run_id="run-a")

    rows = svc.list_active(db, task_id=task_id)
    assert [row.id for row in rows] == [active.id]
    db.close()


def test_get_scopes_by_task_id_not_by_interaction_id_alone(
    _db: Session, _session_factory
) -> None:
    db = _session_factory()
    user_id = make_user(db)
    task_a = make_task(db, user_id=user_id)
    task_b = make_task(db, user_id=user_id)
    trace_event_id = _make_trace_event(db, task_id=task_a)
    row = _make_active_interaction_row(
        db, task_id=task_a, resume_trace_event_id=trace_event_id
    )

    assert svc.get(db, task_id=task_a, interaction_id=row.id) is not None
    assert svc.get(db, task_id=task_b, interaction_id=row.id) is None
    db.close()


# ---------------------------------------------------------------------------
# Structural guards: raw SQL and the _rowcount idiom.
# ---------------------------------------------------------------------------


def test_module_issues_zero_sa_text_calls() -> None:
    """Every statement in this module goes through Core/ORM query-building
    (``db.query(...)``, ``db.get(...)``), never a raw ``sa.text(...)``
    string. AST-based rather than a source-text grep, for the same reason
    the production-caller gate is AST-based: a substring scan would also
    match this assertion's own docstring and any future prose mention of
    ``sa.text`` in a comment.

    This module does not perform any rowcount-based writes in this
    delivery (create() validates and returns; it does not stage a row), so
    there is no ``_rowcount``-idiom call to guard here yet -- that
    obligation applies once a write path lands."""

    import ast
    import inspect

    source = inspect.getsource(svc)
    tree = ast.parse(source)
    text_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "text":
            text_calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == "text":
            text_calls.append(node)
    assert text_calls == []
