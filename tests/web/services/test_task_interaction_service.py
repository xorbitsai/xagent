"""Unit coverage for ``task_interaction_service``: the outcome vocabularies,
the ``create()`` typed seam, and the compatibility materialization view.

This module accumulates coverage across every deliverable this service
ships except the shared public-chat ownership predicate (covered directly
by ``tests/web/api/test_public_chat_ownership_helper.py``, since it is
extracted from and tested alongside ``public_chat_access.py``) and the
``create()`` zero-production-caller gate (its own file,
``test_task_interaction_service_create_gate.py``).

RespondOutcome's failure matrix, and what this delivery does and does not
do with it: this build's ``RespondConflictReason``/``RespondStaleReason``
``Literal``s are narrower than a build that also classifies why the
answer fence's UPDATE matched zero rows and reconciles an ambiguous
commit against the durable graph would need (see ``RespondOutcomeUnknown``
's own docstring in ``task_interaction_service.py``) -- six triggering
scenarios that such a build would classify into five distinct ``Stale``
reasons plus ``Conflict(already_answered)`` collapse onto this build's
single ``(OutcomeUnknown, None)`` pair instead, alongside the four other
triggers (an unclassified staging ``IntegrityError``, the two
staging-race doors, and an unreconciled commit exception) this build
already reports the same way. The matrix
below enumerates this build's own 27 triggering cells, producing 14
distinct (outcome type, reason) pairs -- fewer than 27 because several
cells share a pair (six "principal does not own this task" cells all
produce ``(RespondUnauthorized, not_task_principal)``; two "same
idempotency key, different actor" cells both produce ``(RespondConflict,
idempotency_key_reused)``; six distinct triggers collapse onto
``(OutcomeUnknown, None)``; one cell, kind/version validation, is
parametrized over two reasons on its own). The full cell-to-pair mapping:

    (Cell ids are this build's subset of the full matrix an end-to-end
    delivery would enumerate; the gaps -- there is no C1 or S1-S5
    here -- are cells only a build that stages rows and classifies fence
    misses can reach.)

    OK,OK2    -> (Accepted, None)      1 (2 cells share it -- the plain
                 accepted path, and a commit that succeeds but whose
                 post-commit dispatcher notify raises)
    V1        -> (ValidationRejected, unknown_kind)                  }  2
                 (ValidationRejected, unknown_protocol_version)      }
    V2        -> (ValidationRejected, malformed_idempotency_key)     1
    V3,V4     -> (ValidationRejected, invalid_values)      1 (2 cells share
                 it -- a non-dict ``values`` payload, and a dict
                 ``values`` payload that cannot be rendered as JSON)
    V5        -> (ValidationRejected, kind_version_mismatch)         1
    A1..A6    -> (Unauthorized, not_task_principal)      1 (6 cells share
                 it -- a user principal that does not own the task, the
                 authorization-before-idempotency ordering guard, a guest
                 principal on a non-matching task, a guest principal with
                 two populated entity-binding directions, an unknown
                 principal kind, and a guest principal with zero
                 populated entity-binding directions)
    U1        -> (Unavailable, task_missing)                         1
    U2        -> (Unavailable, interaction_missing)                  1
    U3        -> (Unavailable, checkpoint_unavailable)                1
    R1,R2     -> (Replayed, None)      1 (2 cells share it -- the plain
                 replay, and a replay of an already-answered row whose
                 resume anchor was pruned before the retry)
    C2,C3     -> (Conflict, idempotency_key_reused)      1 (2 cells share it)
    S6        -> (Stale, anchor_dangling)                             1
    X1..X6    -> (OutcomeUnknown, None)      1 (6 cells share it -- a
                 fence miss with no further classification, a guest whose
                 fence-level mismatch this build cannot label, a staging
                 IntegrityError, a commit exception, and two staging-race
                 doors -- a raced row found already staged with a
                 mismatched payload, and one found with a matching
                 payload)
    -----------------------------------------------------------------
    27 cells; 14 distinct pairs (12 single-reason cells + V1's own 2)

The cell-by-cell tests this matrix implies, and the mapping meta-test that
checks their coverage against the vocabulary, are both in this file now,
alongside ``respond()``'s own implementation. The vocabulary itself is
enforced by the type system -- each outcome's reason is its own
``Literal`` -- backed by a union-membership test confirming
``RespondOutcome`` still has exactly its eight known variants, which
leaves a two-way division of labor between the remaining assertion
layers:

| Assertion | Checks | Catches | Misses |
|---|---|---|---|
| Union-membership guard (this file, written) | ``RespondOutcome`` has exactly its eight known member classes | A variant added or removed without updating this list | Reason-level coverage |
| Cell-by-cell tests (this file, written) | One test per of the 27 cells, asserting outcome + reason + zero side effects | A regression in one specific cell's behavior | A forgotten test |
| Mapping meta-test (this file, written) | Each of the 14 pairs is produced by >= 1 cell's test (12 singles + one cell's own 2) | A new cell that produces a new pair with no test written for it; the two-reason cell's parametrization missing a reason | A new cell that produces no *new* pair (e.g. a seventh not_task_principal scenario) -- caught by review, not this meta-test |
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_interaction_schema_shared import make_task, make_user
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
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
# CreateOutcome's vocabulary guards (two pinned numbers, still plain dicts
# in the source -- do not recompute them here):
#
#   - CreateOutcome reason word list: 13 words total, across both delivery
#     periods.
#   - CreateOutcome pairs producible in this delivery specifically: 7.
#
# These guards prove CreateOutcome's vocabulary stays closed at exactly
# these counts. They do NOT prove every pair has a test written against
# it -- several reasons are reachable from more than one triggering
# condition and are indistinguishable at the (type, reason) level alone
# (see each dict's own comment in the source for which ones collapse).
# RespondOutcome has no equivalent dict -- see the comment immediately
# below this one for how its vocabulary is guarded instead.
# ---------------------------------------------------------------------------


# RespondOutcome's reason vocabulary has no separate dict guard: each
# outcome that carries a reason declares it as a ``Literal`` directly on
# the dataclass field (see task_interaction_service.py), so the type
# itself is the single source of the word list -- there is nothing left
# for a count-guard test to protect against drifting out of sync with.
# The two assertions below read that type back rather than duplicating it:
# the first confirms the Union has exactly the eight known member classes,
# the second (further down, next to the mapping meta-test) confirms the
# 14 (outcome, reason) pairs it derives from those classes' own Literal
# annotations still match what every test in this file actually produces.


def test_respond_outcome_union_has_exactly_the_eight_known_variants() -> None:
    import typing

    assert {cls.__name__ for cls in typing.get_args(svc.RespondOutcome)} == {
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
# create(): the seven (outcome, reason) pairs producible in this delivery.
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


def _owning_principal(user_id: int) -> svc.InteractionPrincipal:
    return svc.InteractionPrincipal(
        kind="user",
        user_id=user_id,
        is_admin=False,
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


# ---------------------------------------------------------------------------
# T3', the other five conditions: _resolve_read_direction_anchor's row-
# validity judgment is six self-consistency conditions ANDed together (task
# id, event type, build id, checkpoint type, run partition, execution
# identity -- see that function's own docstring). The run-partition cell is
# covered above; each of these five breaks exactly one of the remaining
# conditions, following the same one-condition-per-cell shape
# test_task_interaction_anchor.py's own _CONDITION_BREAKS table uses for the
# sibling resolver it covers.
# ---------------------------------------------------------------------------

_T3_ANCHOR_VALIDATION_BREAKS: dict[str, dict[str, Any]] = {
    "task_id": {"cross_task": True},
    "event_type": {"event_type": "system_update_partial"},
    "build_id": {"build_id": "build-x"},
    "checkpoint_type": {"checkpoint_type": "not_a_checkpoint_type"},
    "execution_id": {"mismatched_resume_execution_id": "exec-mismatch"},
}


@pytest.mark.parametrize("condition", sorted(_T3_ANCHOR_VALIDATION_BREAKS))
def test_t3_prime_anchor_dangling_for_each_remaining_validity_condition(
    _db: Session, _seeded_task: int, condition: str
) -> None:
    """Every other cell in this file leaves all six conditions passing (or,
    for the run-partition cell above, breaks exactly one). Deleting any one
    of the five conditions exercised here from
    _resolve_read_direction_anchor's boolean guard must turn exactly this
    cell red and leave every other cell -- including the four remaining
    parametrizations of this same test -- green."""

    overrides = dict(_T3_ANCHOR_VALIDATION_BREAKS[condition])
    trace_task_id = _seeded_task
    if overrides.pop("cross_task", False):
        other_user_id = make_user(_db)
        trace_task_id = make_task(_db, user_id=other_user_id)
    # The trace side's execution_id stays at its non-empty default ("exec-1")
    # for every cell: an empty trace-side execution_id short-circuits the
    # comparison to "matches" regardless of the row side, so leaving it
    # non-empty is what makes the execution_id cell's mismatch reachable at
    # all, and leaving it non-empty (and equal to the row's own default) for
    # the other four cells is what keeps this condition passing everywhere
    # else.
    resume_execution_id = overrides.pop("mismatched_resume_execution_id", "exec-1")

    trace_event_id = _make_trace_event(_db, task_id=trace_task_id, **overrides)
    _make_active_interaction_row(
        _db,
        task_id=_seeded_task,
        resume_trace_event_id=trace_event_id,
        resume_execution_id=resume_execution_id,
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
# The answer fence: compile-time assertions against the predicate alone,
# independent of ``respond()`` (the fence statement's own execution is
# covered end to end by the full accepted-path and durable-graph-landed
# tests further down, not by a standalone execution test here -- see the
# fence functions' own docstrings for what they are reused by).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TaskStatusPredicate structural assertion. The active-row query
# ``_active_native_row_criteria()`` builds never references ``Task.status``
# at all (see that function's own docstring for why -- "is the task
# WAITING_FOR_USER" is a concern the answer fence adds, not part of "which
# row is the live one"), so this is the tripwire for the change that does
# add a ``Task.status`` conjunct to a query built from this same predicate
# (the answer fence, or the write-side reclaim statement): whichever lands
# must keep this passing only because its new conjunct goes through
# ``TaskStatusPredicate`` rather than a bare ``TaskStatus`` member-name
# string.
#
# Walks the statement's own bind parameters rather than comparing
# substrings of its compiled SQL text. A substring check here is
# satisfiable by construction -- there is nothing in this query for it to
# ever have found, since the query never touches ``Task.status`` at all --
# so a version of this test written that way could never actually go red
# on a real regression: it would still pass even if ``TaskStatusPredicate``
# were dropped entirely and every ``Task.status`` comparison were rewritten
# to compare bare enum members directly, because that comparison still
# would not appear as a substring of *this* unrelated query's SQL text.
# Reading the actual TaskStatus-typed bind values out of the unbuilt
# ClauseElement tree instead means a future author who adds a TaskStatus
# literal to this exact query turns this test red regardless of how
# SQLAlchemy renders it. Verified with a real mutation: adding
# ``Task.status == TaskStatus.WAITING_FOR_USER`` to the statement below
# turns up one offending bind parameter; reverting it returns to zero.
#
# Needs no database connection -- unlike the fence-predicate compile test
# just below, which needs a SQLite dialect object to compile against but
# not an actual database either.
# ---------------------------------------------------------------------------


def test_active_row_query_uses_zero_taskstatus_bind_parameters() -> None:
    from sqlalchemy.sql.elements import BindParameter
    from sqlalchemy.sql.visitors import iterate

    stmt = (
        sa.select(TaskInteractionRequest)
        .join(Task, Task.id == TaskInteractionRequest.task_id)
        .where(
            TaskInteractionRequest.task_id == 1,
            *svc._active_native_row_criteria(),
        )
    )
    taskstatus_binds = [
        node
        for node in iterate(stmt)
        if isinstance(node, BindParameter) and isinstance(node.value, TaskStatus)
    ]
    assert taskstatus_binds == []


def test_answer_fence_predicate_compiles_without_any_taskstatus_literal_string() -> (
    None
):
    principal = _owning_principal(1)
    stmt = sa.select(TaskInteractionRequest).where(
        TaskInteractionRequest.id == 1,
        TaskInteractionRequest.task_id == 1,
        Task.id == 1,
        *svc._active_native_row_criteria(),
        *svc._answer_fence_task_predicate(principal),
    )
    import sqlalchemy.dialects.sqlite

    compiled = str(
        stmt.compile(
            dialect=sqlalchemy.dialects.sqlite.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    for member in TaskStatus:
        assert member.name.lower() not in compiled
    assert "WAITING_FOR_USER" in compiled


def test_answer_fence_predicate_guest_branch_adds_a_json_lookup_term() -> None:
    user_terms = svc._answer_fence_task_predicate(_owning_principal(1))
    guest_principal = svc.InteractionPrincipal(
        kind="guest",
        user_id=1,
        is_admin=False,
        auth_mode="widget",
        guest_id="guest-1",
    )
    guest_terms = svc._answer_fence_task_predicate(guest_principal)
    assert len(guest_terms) == len(user_terms) + 1


# ---------------------------------------------------------------------------
# Structural guards: raw SQL.
# ---------------------------------------------------------------------------


def test_module_issues_zero_sa_text_calls() -> None:
    """Every statement in this module goes through Core/ORM query-building
    (``db.query(...)``, ``db.get(...)``), never a raw ``sa.text(...)``
    string. AST-based rather than a source-text grep, for the same reason
    the production-caller gate is AST-based: a substring scan would also
    match this assertion's own docstring and any future prose mention of
    ``sa.text`` in a comment.

    ``respond()``'s answer fence and its rowcount-based classification are
    both Core statements too (``sa.update(...)``, ``.with_for_update(...)``)
    -- their absence from this scan is what proves them clean, not an
    exemption."""

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


# ---------------------------------------------------------------------------
# respond(): the answer-side entry point. respond() owns and retires its own
# session (see its docstring), so every test below patches
# ``xagent.web.models.database.get_session_local`` to hand back this file's
# own file-backed SQLite session factory -- a bare ``:memory:`` database
# cannot be shared across the separate connections respond()'s own session
# and this test's setup/verification sessions each open.
# ---------------------------------------------------------------------------

import xagent.web.models.database as _database_module  # noqa: E402
from xagent.web.models.task_command import TaskExecutionCommand  # noqa: E402
from xagent.web.services.task_execution_controller import (  # noqa: E402
    TaskControlState,
)


@pytest.fixture
def _respond_db(monkeypatch: pytest.MonkeyPatch, _session_factory):
    monkeypatch.setattr(_database_module, "get_session_local", lambda: _session_factory)
    return _session_factory


def _waiting_task(
    session_factory,
    *,
    agent_config: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """A task parked in WAITING_FOR_USER, the state every respond() test
    starts from -- the answer fence's task-side predicate requires it."""

    db = session_factory()
    try:
        user_id = make_user(db)
        task_id = make_task(db, user_id=user_id)
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = TaskStatus.WAITING_FOR_USER
        task.control_state = TaskControlState.WAITING_FOR_USER.value
        task.run_id = "run-a"
        task.state_version = 5
        task.channel_id = None
        task.agent_id = None
        if agent_config is not None:
            task.agent_config = agent_config
        db.commit()
        return user_id, task_id
    finally:
        db.close()


def _active_row_ready_for_respond(
    session_factory,
    *,
    task_id: int,
    anchor_run_partition: str | None = None,
) -> int:
    """An active interaction row with a resolvable anchor -- the state every
    respond() test that expects to reach the fence (step 6 onward) starts
    from. ``anchor_run_partition``, when different from ``run_id``, is what
    ``test_respond_reports_stale_when_the_anchor_points_at_a_different_run_partition``
    below uses to force the anchor resolver's own partition check to fail
    without touching the interaction row's ``run_id`` (the fence's own,
    separate run comparison)."""

    db = session_factory()
    try:
        trace_event_id = _make_trace_event(
            db, task_id=task_id, run_partition=anchor_run_partition or "run-a"
        )
        row = _make_active_interaction_row(
            db,
            task_id=task_id,
            run_id="run-a",
            resume_trace_event_id=trace_event_id,
            resume_run_partition="run-a",
        )
        return int(row.id)
    finally:
        db.close()


def _answered_row_with_valid_anchor(
    session_factory,
    *,
    task_id: int,
    run_id: str,
    responder_identity: str,
    response_payload: dict[str, Any],
) -> int:
    """A row this service already answered in some earlier, successful call
    -- its resume anchor is still valid (nothing has pruned the checkpoint
    it points at). A live anchor is still what makes the fence-miss tests
    below reachable past step 5.5; the already-answered replay tests no
    longer need it, since step 5's pre-read now recognizes the replay
    before anchor resolution runs (the pruned-anchor replay test below is
    what pins that down)."""

    db = session_factory()
    try:
        trace_event_id = _make_trace_event(db, task_id=task_id, run_partition=run_id)
        row = _make_active_interaction_row(
            db,
            task_id=task_id,
            run_id=run_id,
            resume_trace_event_id=trace_event_id,
            resume_run_partition=run_id,
        )
        now = _now()
        row.status = "answered"
        row.active_slot = None
        row.response_payload = response_payload
        row.responded_at = now
        row.responder_identity = responder_identity
        row.request_idempotency_key = "prior-answer-key"
        db.commit()
        return int(row.id)
    finally:
        db.close()


def _stage_matching_command(
    session_factory,
    *,
    task_id: int,
    actor_user_id: int | None,
    command_id: str,
    payload: dict[str, Any],
) -> int:
    db = session_factory()
    try:
        command = TaskExecutionCommand(
            task_id=task_id,
            actor_user_id=actor_user_id,
            command_id=command_id,
            kind=svc.TaskCommandKind.RESUME.value,
            payload=payload,
            status="completed",
        )
        db.add(command)
        db.commit()
        db.refresh(command)
        return int(command.id)
    finally:
        db.close()


def _respond_envelope(**overrides: Any) -> svc.RespondEnvelope:
    defaults: dict[str, Any] = {
        "kind": "clarification",
        "protocol_version": 1,
        "values": {"env": "prod"},
        "idempotency_key": "respond-key-1",
    }
    defaults.update(overrides)
    return svc.RespondEnvelope(**defaults)


def _graph_snapshot(
    session_factory, *, task_id: int, interaction_id: int
) -> dict[str, Any]:
    """A comparable snapshot of the three tables respond() can touch, for
    the "zero side effects" half of every rejection-path assertion below."""

    db = session_factory()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        ir = (
            db.query(TaskInteractionRequest)
            .filter(TaskInteractionRequest.id == interaction_id)
            .first()
        )
        commands = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == task_id)
            .count()
        )
        return {
            "task_state_version": task.state_version if task is not None else None,
            "task_control_state": task.control_state if task is not None else None,
            "task_run_id": task.run_id if task is not None else None,
            "ir_status": ir.status if ir is not None else None,
            "ir_active_slot": ir.active_slot if ir is not None else None,
            "ir_response_payload": ir.response_payload if ir is not None else None,
            "ir_responder_identity": ir.responder_identity if ir is not None else None,
            "ir_responder_user_id": ir.responder_user_id if ir is not None else None,
            "ir_responded_at": ir.responded_at if ir is not None else None,
            "ir_updated_at": ir.updated_at if ir is not None else None,
            "commands_count": commands,
        }
    finally:
        db.close()


@contextlib.contextmanager
def _asserts_no_side_effects(
    session_factory, *, task_id: int, interaction_id: int
) -> Any:
    """Wrap a rejection-path ``respond()`` call and confirm it left the
    task/interaction/command graph exactly as it found it. Snapshots on
    entry, yields to the body (which calls ``svc.respond()`` and asserts on
    its outcome), and compares snapshots on exit -- the "zero side effects"
    half of every rejection-path test below, previously written out as a
    repeated ``before = _graph_snapshot(...)`` / ``assert ... == before``
    pair at each call site."""

    before = _graph_snapshot(
        session_factory, task_id=task_id, interaction_id=interaction_id
    )
    yield
    after = _graph_snapshot(
        session_factory, task_id=task_id, interaction_id=interaction_id
    )
    assert after == before


def _conflict_counter() -> int:
    """The current value of the response-conflict counter, the same
    process-local registry ``respond()`` itself increments through
    (``xagent.web.services.interaction_rollout``)."""

    from xagent.web.services import interaction_rollout as rollout_module

    return rollout_module.counters_snapshot().get(
        svc.COUNTER_LIFECYCLE_RESPONSE_CONFLICT, 0
    )


# ---------------------------------------------------------------------------
# The pure-read path: envelope validation, task/interaction existence,
# authorization, and anchor resolution -- steps 1 through 5.5. Twelve cells,
# each asserting outcome, reason, and zero side effects.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"kind": "not_a_real_kind"}, id="unknown_kind"),
        pytest.param({"protocol_version": 2}, id="unknown_protocol_version"),
        # Type-before-value: a non-str kind must be rejected by an isinstance
        # check before it ever reaches the vocabulary membership test, which
        # raises TypeError on an unhashable value (a list, a dict) instead of
        # returning a typed outcome if the type check is skipped.
        pytest.param({"kind": ["clarification"]}, id="kind_is_a_list"),
        pytest.param({"kind": {"clarification": 1}}, id="kind_is_a_dict"),
        # Type-before-value: bool is a subclass of int (True == 1) and a
        # float compares equal to an int of the same value (1.0 == 1), so
        # both must be rejected by an isinstance check before the equality
        # comparison, or they would silently pass validation.
        pytest.param({"protocol_version": True}, id="protocol_version_is_a_bool"),
        pytest.param({"protocol_version": 1.0}, id="protocol_version_is_a_float"),
    ],
)
def test_respond_rejects_an_envelope_outside_the_known_kind_or_version_vocabulary(
    _respond_db, overrides: dict[str, Any]
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(**overrides),
        )

        if "kind" in overrides:
            assert outcome == svc.RespondValidationRejected(reason="unknown_kind")
        else:
            assert outcome == svc.RespondValidationRejected(
                reason="unknown_protocol_version"
            )


def test_respond_rejects_an_idempotency_key_that_is_not_url_safe(_respond_db) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(idempotency_key="has a space"),
        )

        assert outcome == svc.RespondValidationRejected(
            reason="malformed_idempotency_key"
        )


def test_respond_rejects_answer_values_that_are_not_a_dict(_respond_db) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(values="not-a-dict"),
        )

        assert outcome == svc.RespondValidationRejected(reason="invalid_values")


@pytest.mark.parametrize(
    "values",
    [
        pytest.param({"a": datetime.now(timezone.utc)}, id="datetime"),
        pytest.param({"a": {1, 2}}, id="set"),
        pytest.param({"a": b"x"}, id="bytes"),
        pytest.param({"a": float("nan")}, id="nan_float"),
        pytest.param({"a": "x", 1: "y"}, id="mixed_int_str_keys"),
    ],
)
def test_respond_rejects_values_that_cannot_be_rendered_as_json(
    _respond_db, values: dict[str, Any]
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(values=values),
        )

        assert outcome == svc.RespondValidationRejected(reason="invalid_values")


def test_respond_rejects_when_the_stored_row_disagrees_with_the_envelope_on_protocol_version(
    _respond_db,
) -> None:
    """The row's own protocol_version can only differ from 1 once it is no
    longer active (``ck_task_interaction_requests_active_protocol`` pins an
    active row's protocol_version to 1), so this cell is built on a row that
    has already reached a terminal state under an older protocol."""

    user_id, task_id = _waiting_task(_respond_db)
    now = _now()
    db = _respond_db()
    try:
        row = TaskInteractionRequest(
            task_id=task_id,
            run_id="run-a",
            kind="clarification",
            protocol_version=2,
            status="terminated",
            active_slot=None,
            origin="internal",
            request_payload={"message": "q", "interactions": []},
            request_idempotency_key="protocol-mismatch-key",
            resume_trace_event_id=None,
            resume_event_id="resume-event-1",
            resume_execution_id="exec-1",
            resume_locator_format="trace_event_pk_v1",
            resume_checkpoint_type="agent_execution_checkpoint",
            resume_run_partition="run-a",
            terminal_reason="deadline_elapsed",
            terminated_at=now,
            created_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        interaction_id = int(row.id)
    finally:
        db.close()

    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(protocol_version=1),
        )

        assert outcome == svc.RespondValidationRejected(reason="kind_version_mismatch")


def test_respond_rejects_a_user_principal_that_does_not_own_the_task(
    _respond_db,
) -> None:
    owner_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    intruder = svc.InteractionPrincipal(
        kind="user",
        user_id=owner_id + 987654,
        is_admin=False,
        auth_mode=None,
    )
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=intruder,
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondUnauthorized(reason="not_task_principal")


def test_respond_rejects_a_guest_whose_bindings_match_but_principal_user_id_does_not(
    _respond_db,
) -> None:
    """A guest principal whose ``guest_id`` and entity binding both match
    the task -- so step 3's ``task_is_owned_by_public_principal`` passes,
    since that predicate never reads ``principal.user_id`` at all -- but
    whose ``user_id`` field does not match the task's real owner. Step 3
    has nothing to catch this with; the fence's own ``Task.user_id ==
    principal.user_id`` term (present for both principal kinds, not only
    the guest-specific JSON check) is what refuses it, on both backends,
    since this is a plain mismatch present from the start, not a
    concurrent change to catch only via SQLite's missing lock. The refusal
    lands as a zero-rowcount fence miss, which this build reports as
    ``OutcomeUnknown`` rather than the fine-grained ``Unauthorized`` a more
    detailed classification would give it (see respond()'s own docstring,
    step 6) -- the security property this test exists to pin (the guest
    never gets an answer accepted) holds either way; only the label does
    not."""

    owner_id, task_id = _waiting_task(
        _respond_db,
        agent_config={
            "auth_mode": "widget",
            "guest_id": "guest-1",
            "widget_workforce_id": 10,
        },
    )
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    setup_db = _respond_db()
    try:
        wrong_user_id = make_user(setup_db)
    finally:
        setup_db.close()
    guest = svc.InteractionPrincipal(
        kind="guest",
        user_id=wrong_user_id,
        is_admin=False,
        auth_mode="widget",
        widget_workforce_id=10,
        guest_id="guest-1",
    )
    assert wrong_user_id != owner_id
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=guest,
            envelope=_respond_envelope(),
        )

        assert isinstance(outcome, svc.RespondOutcomeUnknown)


def test_respond_rejects_a_guest_principal_on_a_non_matching_task(
    _respond_db,
) -> None:
    """Same owner, same auth_mode, but a different widget_workforce_id --
    the entity-binding conjunct must reject this at step 3, not the
    (correctly matching) owner or auth_mode conjuncts. Mirrors
    ``test_ca1_guest_principal_is_rejected_on_a_non_matching_task`` on the
    create() side."""

    owner_id, task_id = _waiting_task(
        _respond_db,
        agent_config={
            "auth_mode": "widget",
            "guest_id": "guest-1",
            "widget_workforce_id": 10,
        },
    )
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    guest = svc.InteractionPrincipal(
        kind="guest",
        user_id=owner_id,
        is_admin=False,
        auth_mode="widget",
        widget_workforce_id=999,
        guest_id="guest-1",
    )
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=guest,
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondUnauthorized(reason="not_task_principal")


def test_respond_rejects_a_guest_principal_with_two_populated_directions(
    _respond_db,
) -> None:
    """A malformed principal that populates more than one of the four
    entity-binding fields makes ``task_is_owned_by_public_principal`` raise
    ``ValueError``; respond() must catch exactly that and translate it to
    ``Unauthorized(not_task_principal)``, not let it escape. Mirrors
    ``test_ca1_guest_principal_with_two_populated_directions_is_unauthorized_not_raised``
    on the create() side."""

    owner_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    guest = svc.InteractionPrincipal(
        kind="guest",
        user_id=owner_id,
        is_admin=False,
        auth_mode="widget",
        widget_agent_id=1,
        widget_workforce_id=1,
        guest_id="guest-1",
    )
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=guest,
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondUnauthorized(reason="not_task_principal")


def test_respond_rejects_an_unknown_principal_kind(
    _respond_db,
) -> None:
    """A principal.kind that is neither "user" nor "guest" must be
    rejected -- there is no third branch that defaults to allow. Mirrors
    ``test_ca1_unknown_principal_kind_is_always_unauthorized`` on the
    create() side."""

    owner_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = svc.InteractionPrincipal(
        kind="robot",
        user_id=owner_id,
        is_admin=True,
        auth_mode=None,
    )
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondUnauthorized(reason="not_task_principal")


def test_respond_rejects_a_guest_principal_with_zero_populated_directions(
    _respond_db,
) -> None:
    """Mirrors
    ``test_ca1_guest_principal_with_zero_populated_directions_is_unauthorized_not_raised``
    on the create() side: a guest principal that populates none of the
    four entity-binding fields makes the ownership predicate raise
    ``ValueError``, which respond() must translate to
    ``Unauthorized(not_task_principal)`` rather than let escape."""

    owner_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    guest = svc.InteractionPrincipal(
        kind="guest",
        user_id=owner_id,
        is_admin=False,
        auth_mode="widget",
        guest_id="guest-1",
    )
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=guest,
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondUnauthorized(reason="not_task_principal")


def test_respond_reports_unavailable_when_the_task_row_does_not_exist(
    _respond_db,
) -> None:
    outcome = svc.respond(
        interaction_id=1,
        task_id=999_999_999,
        principal=_owning_principal(1),
        envelope=_respond_envelope(),
    )

    assert outcome == svc.RespondUnavailable(reason="task_missing")


def test_respond_reports_unavailable_when_the_interaction_row_does_not_exist(
    _respond_db,
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    with _asserts_no_side_effects(_respond_db, task_id=task_id, interaction_id=999_999):
        outcome = svc.respond(
            interaction_id=999_999,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondUnavailable(reason="interaction_missing")


def test_respond_reports_unavailable_when_the_anchor_row_fetch_raises(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        from sqlalchemy.orm import Session as OrmSession

        original_get = OrmSession.get

        def _raising_get(
            self: Any, model: Any, pk: Any, *args: Any, **kwargs: Any
        ) -> Any:
            if model is TraceEvent:
                raise RuntimeError("simulated session failure")
            return original_get(self, model, pk, *args, **kwargs)

        monkeypatch.setattr(OrmSession, "get", _raising_get)

        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondUnavailable(reason="checkpoint_unavailable")


def test_respond_reports_stale_when_the_anchor_points_at_a_different_run_partition(
    _respond_db,
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(
        _respond_db, task_id=task_id, anchor_run_partition="a-different-partition"
    )
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondStale(reason="anchor_dangling")


# ---------------------------------------------------------------------------
# Idempotency, the version short-circuit, and the answer fence's
# zero-rowcount classification -- steps 5, 5.5, and 6. Eleven cells plus the
# authorization-before-idempotency ordering guard.
# ---------------------------------------------------------------------------


def test_respond_returns_the_original_receipt_for_a_matching_replay(
    _respond_db,
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    principal = _owning_principal(user_id)
    values = {"env": "prod"}
    command_id = "replay-key-1"
    interaction_id = _answered_row_with_valid_anchor(
        _respond_db,
        task_id=task_id,
        run_id="run-a",
        responder_identity=principal.identity_string(),
        response_payload=values,
    )
    payload = svc._respond_command_payload(
        interaction_id=interaction_id, principal=principal, values=values
    )
    _stage_matching_command(
        _respond_db,
        task_id=task_id,
        actor_user_id=principal.user_id,
        command_id=command_id,
        payload=payload,
    )
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_respond_envelope(idempotency_key=command_id, values=values),
        )

        assert isinstance(outcome, svc.RespondReplayed)
        assert outcome.receipt.responder_identity == principal.identity_string()
        assert outcome.receipt.idempotency_key == command_id


def test_respond_replays_an_answered_row_whose_anchor_was_pruned(
    _respond_db,
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    principal = _owning_principal(user_id)
    values = {"env": "prod"}
    command_id = "replay-key-pruned-anchor"
    interaction_id = _answered_row_with_valid_anchor(
        _respond_db,
        task_id=task_id,
        run_id="run-a",
        responder_identity=principal.identity_string(),
        response_payload=values,
    )
    payload = svc._respond_command_payload(
        interaction_id=interaction_id, principal=principal, values=values
    )
    _stage_matching_command(
        _respond_db,
        task_id=task_id,
        actor_user_id=principal.user_id,
        command_id=command_id,
        payload=payload,
    )

    # Simulate the checkpoint retention pruner: the row's anchor is gone,
    # the way ON DELETE SET NULL leaves it once the checkpoint it pointed
    # at is pruned.
    db = _respond_db()
    try:
        row = db.get(TaskInteractionRequest, interaction_id)
        row.resume_trace_event_id = None
        db.commit()
    finally:
        db.close()

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=_respond_envelope(idempotency_key=command_id, values=values),
    )

    assert isinstance(outcome, svc.RespondReplayed)
    assert outcome.receipt.idempotency_key == command_id


def test_respond_receipt_refuses_a_row_that_carries_no_answer() -> None:
    """``_respond_receipt`` may only ever see an answered row: its one
    caller is the idempotent-replay branch, and the paired CHECK
    constraints make an answered row with a NULL ``responded_at`` or
    ``responder_identity`` impossible. That reasoning spans two modules
    with nothing else pinning it, so the builder raises loudly on a row
    with no answer rather than coercing ``None`` into the string
    ``'None'`` (or a falsy ``""``) inside an audit-bearing receipt."""

    from types import SimpleNamespace

    unanswered = SimpleNamespace(
        id=7,
        task_id=11,
        responded_at=None,
        responder_identity=None,
    )
    task = SimpleNamespace(state_version=1, control_state="waiting_for_user")
    with pytest.raises(RuntimeError, match="carries no answer"):
        svc._respond_receipt(
            interaction=unanswered,  # type: ignore[arg-type]
            task=task,  # type: ignore[arg-type]
            command_db_id=1,
            idempotency_key="key-1",
        )


def test_respond_reports_outcome_unknown_when_the_fence_misses_and_leaves_no_residue(
    _respond_db,
) -> None:
    """This build does not classify why the answer fence's UPDATE matched
    zero rows (see ``RespondOutcomeUnknown``'s own docstring) -- every
    fine-grained fence-miss scenario (already answered, terminated, wrong
    task state, foreign run, a concurrent ownership change) collapses onto
    this single conservative outcome instead of the outcome/reason such a
    build's more detailed sibling would report. Picks one concrete trigger
    -- the row already answered by someone else, under a fresh idempotency
    key this call has never seen -- and proves both the collapse and the
    safety property that makes it acceptable: this call changes nothing.
    The pre-existing answer, its already-staged command, and the conflict
    counter are all untouched -- a conservative miss must never be
    misreported as a conflict, since this build never confirmed it was
    one."""

    user_id, task_id = _waiting_task(_respond_db)
    principal = _owning_principal(user_id)
    interaction_id = _answered_row_with_valid_anchor(
        _respond_db,
        task_id=task_id,
        run_id="run-a",
        responder_identity=principal.identity_string(),
        response_payload={"env": "prod"},
    )
    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_respond_envelope(idempotency_key="never-seen-before"),
        )

        assert isinstance(outcome, svc.RespondOutcomeUnknown)
    assert _conflict_counter() == before_counter


def test_respond_logs_the_reread_row_state_when_the_fence_misses(
    _respond_db, caplog: pytest.LogCaptureFixture
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    principal = _owning_principal(user_id)
    interaction_id = _answered_row_with_valid_anchor(
        _respond_db,
        task_id=task_id,
        run_id="run-a",
        responder_identity=principal.identity_string(),
        response_payload={"env": "prod"},
    )
    with caplog.at_level(logging.WARNING):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_respond_envelope(idempotency_key="never-seen-before-logged"),
        )

    assert isinstance(outcome, svc.RespondOutcomeUnknown)
    matching = [
        record
        for record in caplog.records
        if "answer fence matched zero rows" in record.getMessage()
    ]
    assert len(matching) == 1
    assert "status=answered" in matching[0].getMessage()


def test_respond_reports_conflict_for_the_same_key_with_a_different_payload(
    _respond_db,
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    principal = _owning_principal(user_id)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    command_id = "shared-key-1"
    staged_payload = svc._respond_command_payload(
        interaction_id=interaction_id, principal=principal, values={"env": "staging"}
    )
    _stage_matching_command(
        _respond_db,
        task_id=task_id,
        actor_user_id=principal.user_id,
        command_id=command_id,
        payload=staged_payload,
    )
    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_respond_envelope(
                idempotency_key=command_id, values={"env": "prod"}
            ),
        )

        assert outcome == svc.RespondConflict(reason="idempotency_key_reused")
    assert _conflict_counter() == before_counter + 1


def test_respond_reports_conflict_when_a_guest_and_the_owner_share_one_key(
    _respond_db,
) -> None:
    """The same idempotency key and the same answer values, submitted once
    as a guest and once as the owning user. Without ``responder_identity``
    in the staged payload this would misclassify as a replay -- see
    ``_respond_command_payload``'s own docstring."""

    values = {"env": "prod"}
    owner_id, task_id = _waiting_task(
        _respond_db, agent_config={"auth_mode": "widget", "guest_id": "guest-1"}
    )
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    command_id = "shared-key-guest-owner"
    guest = svc.InteractionPrincipal(
        kind="guest",
        user_id=owner_id,
        is_admin=False,
        auth_mode="widget",
        widget_agent_id=None,
        guest_id="guest-1",
    )
    guest_payload = svc._respond_command_payload(
        interaction_id=interaction_id, principal=guest, values=values
    )
    _stage_matching_command(
        _respond_db,
        task_id=task_id,
        actor_user_id=guest.user_id,
        command_id=command_id,
        payload=guest_payload,
    )
    owner = _owning_principal(owner_id)
    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=owner,
            envelope=_respond_envelope(idempotency_key=command_id, values=values),
        )

        assert outcome == svc.RespondConflict(reason="idempotency_key_reused")
    assert _conflict_counter() == before_counter + 1


def test_respond_checks_authorization_before_the_idempotency_prequery(
    _respond_db,
) -> None:
    """An unauthorized caller must never be able to use a guessed
    idempotency key to read back someone else's receipt."""

    owner_id, task_id = _waiting_task(_respond_db)
    principal = _owning_principal(owner_id)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    command_id = "someone-elses-key"
    values = {"env": "prod"}
    payload = svc._respond_command_payload(
        interaction_id=interaction_id, principal=principal, values=values
    )
    _stage_matching_command(
        _respond_db,
        task_id=task_id,
        actor_user_id=owner_id,
        command_id=command_id,
        payload=payload,
    )
    intruder = svc.InteractionPrincipal(
        kind="user",
        user_id=owner_id + 42_424_242,
        is_admin=False,
        auth_mode=None,
    )

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=intruder,
        envelope=_respond_envelope(idempotency_key=command_id, values=values),
    )

    assert outcome == svc.RespondUnauthorized(reason="not_task_principal")


# ---------------------------------------------------------------------------
# The write path: the Task CAS, staging the command, commit-or-reconcile,
# and dispatcher notification -- steps 7 through 10.
# ---------------------------------------------------------------------------


def test_respond_accepts_a_fully_valid_answer_and_fills_every_receipt_field(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    dispatched: list[bool] = []

    def _record_dispatch() -> None:
        dispatched.append(True)

    monkeypatch.setattr(svc, "notify_task_command_dispatcher", _record_dispatch)
    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=_respond_envelope(idempotency_key="ok-path-key"),
    )

    assert isinstance(outcome, svc.RespondAccepted)
    receipt = outcome.receipt
    assert receipt.interaction_id == interaction_id
    assert receipt.task_id == task_id
    assert receipt.run_id == "run-a"
    assert receipt.status == "answered"
    assert receipt.responded_at is not None
    assert receipt.responder_identity == principal.identity_string()
    assert receipt.idempotency_key == "ok-path-key"
    assert receipt.command_db_id > 0
    assert receipt.task_state_version == 6
    assert receipt.task_control_state == TaskControlState.RESUME_REQUESTED.value
    assert dispatched == [True]

    after = _graph_snapshot(_respond_db, task_id=task_id, interaction_id=interaction_id)
    assert after["ir_status"] == "answered"
    assert after["ir_active_slot"] is None
    assert after["ir_response_payload"] == {"env": "prod"}
    assert after["task_state_version"] == 6
    assert after["commands_count"] == 1


def test_respond_returns_accepted_when_the_dispatcher_notify_fails(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    def _raising_notify() -> None:
        raise RuntimeError("Event loop is closed")

    monkeypatch.setattr(svc, "notify_task_command_dispatcher", _raising_notify)

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=_respond_envelope(idempotency_key="notify-fails-key"),
    )

    assert isinstance(outcome, svc.RespondAccepted)

    after = _graph_snapshot(_respond_db, task_id=task_id, interaction_id=interaction_id)
    assert after["ir_status"] == "answered"
    assert after["ir_response_payload"] == {"env": "prod"}


def test_respond_receipt_fields_do_not_touch_the_session_after_commit(
    _respond_db,
) -> None:
    """Every value on a returned receipt is a plain Python value captured
    before commit -- reading it after the caller (here, the test itself)
    expires every object on the session must not re-issue any SQL."""

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=_respond_envelope(idempotency_key="expire-all-check"),
    )
    assert isinstance(outcome, svc.RespondAccepted)
    receipt = outcome.receipt

    verify_db = _respond_db()
    try:
        verify_db.expire_all()
        query_count = 0

        def _count_queries(*_args: Any, **_kwargs: Any) -> None:
            nonlocal query_count
            query_count += 1

        from sqlalchemy import event

        event.listen(verify_db.get_bind(), "before_cursor_execute", _count_queries)
        try:
            _ = (
                receipt.interaction_id,
                receipt.task_id,
                receipt.run_id,
                receipt.status,
                receipt.responded_at,
                receipt.responder_identity,
                receipt.idempotency_key,
                receipt.command_db_id,
                receipt.task_state_version,
                receipt.task_control_state,
            )
        finally:
            event.remove(verify_db.get_bind(), "before_cursor_execute", _count_queries)
        assert query_count == 0
    finally:
        verify_db.close()


def test_respond_reports_outcome_unknown_when_commit_raises_and_leaves_no_residue(
    _respond_db, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A raised exception at commit does not, by itself, mean the write
    failed -- the acknowledgment could have been lost after the server
    applied it -- but this build does not attempt to reconcile that
    against the durable graph (a fine-grained reconciliation via a fresh
    session's own read, distinguishing a landed write from a lost one, is
    not delivered here; see respond()'s own docstring, step 9): it reports
    the ambiguity unconditionally. Simulates a commit that never actually
    reaches the database and proves that in this build's conservative
    handling nothing landed either -- the interaction row, the task row,
    and the command table are all exactly as they were before the call.
    Also asserts the ambiguity is logged: unlike the fence-miss branch,
    this door used to leave no trace for an operator to find."""

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    from sqlalchemy.orm import Session as OrmSession

    def _failing_commit(self: Any) -> None:
        raise RuntimeError("simulated lost commit acknowledgment")

    monkeypatch.setattr(OrmSession, "commit", _failing_commit)

    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        with caplog.at_level(logging.WARNING):
            outcome = svc.respond(
                interaction_id=interaction_id,
                task_id=task_id,
                principal=principal,
                envelope=_respond_envelope(idempotency_key="commit-raises"),
            )

        assert isinstance(outcome, svc.RespondOutcomeUnknown)
        matching = [
            record
            for record in caplog.records
            if "commit failed while answering" in record.getMessage()
        ]
        assert len(matching) == 1
        assert matching[0].levelno == logging.WARNING


# ---------------------------------------------------------------------------
# Step 8's own IntegrityError catch, reached when stage_task_command's own
# insert collides with a UNIQUE or FOREIGN KEY constraint (the real trigger
# is a second writer racing this call for the same idempotency key -- only
# reproducible against a real PostgreSQL server, see
# test_task_interaction_service_postgresql.py; simulated directly here).
# This build does not classify what an IntegrityError there means (a
# fine-grained classification via classify_task_command_conflict,
# distinguishing a genuine replay from a real conflict, is not delivered --
# see respond()'s own docstring, step 8): the whole transaction rolls back
# and this call reports the ambiguity.
# ---------------------------------------------------------------------------


def test_respond_reports_outcome_unknown_when_staging_the_command_raises_and_leaves_no_residue(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import IntegrityError as _IntegrityError

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    def _raising_stage_task_command(*args: Any, **kwargs: Any) -> Any:
        raise _IntegrityError(
            "INSERT", {}, Exception("simulated raced duplicate command_id")
        )

    monkeypatch.setattr(svc, "stage_task_command", _raising_stage_task_command)

    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_respond_envelope(idempotency_key="staging-raises"),
        )

        assert isinstance(outcome, svc.RespondOutcomeUnknown)
    # A conservative miss must not be misreported as a conflict either --
    # the counter only increments for an outcome this build actually
    # confirmed was a conflict.
    assert _conflict_counter() == before_counter


@pytest.mark.parametrize(
    "payload_matches",
    [False, True],
    ids=["payload-mismatch", "payload-matches"],
)
def test_respond_reports_outcome_unknown_when_staging_finds_a_raced_row(
    _respond_db, monkeypatch: pytest.MonkeyPatch, payload_matches: bool
) -> None:
    """A ``created=False`` staging result -- the row was already committed
    and visible by the time this call's own staging statement ran -- is
    reported as ``OutcomeUnknown`` and leaves no residue, regardless of
    whether the raced row's payload happens to match this call's own
    envelope. A matching payload is deliberately not treated as a replay:
    replay recognition happens once, at step 5's idempotency pre-read,
    before the staging statement runs. A raced hit that only becomes
    visible after that pre-read is a race this build cannot distinguish
    from a genuine conflict, not a replay it missed, so both the
    mismatched and the matching case collapse onto the same conservative
    outcome."""

    from xagent.web.services.task_command_transport import StagedTaskCommand

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    def _racing_stage_task_command(*args: Any, **kwargs: Any) -> StagedTaskCommand:
        return StagedTaskCommand(
            staged_db_id=4242,
            client_command_id=kwargs.get("command_id", "staging-race"),
            created=False,
            payload_matches=payload_matches,
            status="pending",
        )

    monkeypatch.setattr(svc, "stage_task_command", _racing_stage_task_command)

    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_respond_envelope(idempotency_key="staging-race"),
        )

        assert isinstance(outcome, svc.RespondOutcomeUnknown)


# ---------------------------------------------------------------------------
# The mapping meta-test. For every one of the 14 (outcome, reason) pairs in
# the vocabulary, at least one test above must produce it. This is
# deliberately not an arithmetic comparison against the total cell count
# (see the module docstring's three-way division of labor table) -- several
# triggering conditions collapse onto the same pair (six distinct
# "principal does not own this task" scenarios all produce
# ``not_task_principal``; two distinct "same idempotency key, different
# submitter" scenarios both produce ``idempotency_key_reused``), and one
# validation scenario is parametrized over two reasons on its own.
# ---------------------------------------------------------------------------


def _expected_respond_outcome_vocabulary() -> set[tuple[str, str | None]]:
    """The (outcome type, reason) pairs ``RespondOutcome`` can produce,
    read directly off each member class's own ``reason`` field -- a
    ``Literal`` for the five outcomes that carry one, absent entirely for
    the three that do not (``RespondAccepted`` / ``RespondReplayed`` /
    ``RespondOutcomeUnknown``, which contribute the reason-less pair
    instead). This is the vocabulary itself, not a copy of it: there is no
    separate dict for this function, or the tests that use it, to drift
    out of sync with.
    """

    import typing

    expected: set[tuple[str, str | None]] = set()
    for cls in typing.get_args(svc.RespondOutcome):
        hints = typing.get_type_hints(cls)
        reason_hint = hints.get("reason")
        if reason_hint is None:
            expected.add((cls.__name__, None))
            continue
        for literal_value in typing.get_args(reason_hint):
            expected.add((cls.__name__, literal_value))
    return expected


def test_every_vocabulary_pair_is_produced_by_at_least_one_cell_test() -> None:
    """AST-based, not a hand-maintained checklist: scans this module's own
    source for every ``svc.Respond<Type>(reason=...)`` construction and
    every ``isinstance(outcome, svc.Respond<Type>)`` check the cell tests
    above use, and cross-checks the resulting (type, reason) set against
    the vocabulary. Deliberately not ``len(produced) == len(vocabulary)`` or
    any other arithmetic against the 27-cell count -- six
    not_task_principal cells and two idempotency_key_reused cells
    legitimately collapse onto one pair each; this only asserts that no
    vocabulary pair is left with zero producing cells. Its blind spot: a
    new cell that produces no *new* pair -- for example a seventh
    not_task_principal scenario -- adds nothing this scan would notice
    missing, so that gap is caught by review, not by this test."""

    import ast
    import inspect

    module = inspect.getmodule(
        test_every_vocabulary_pair_is_produced_by_at_least_one_cell_test
    )
    tree = ast.parse(inspect.getsource(module))

    reasonless_types = {"RespondAccepted", "RespondReplayed", "RespondOutcomeUnknown"}
    produced: set[tuple[str, str | None]] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "svc"
            and node.func.attr.startswith("Respond")
        ):
            reason_value: str | None = None
            for kw in node.keywords:
                if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                    reason_value = kw.value.value
            produced.add((node.func.attr, reason_value))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
        ):
            type_arg = node.args[1]
            if (
                isinstance(type_arg, ast.Attribute)
                and isinstance(type_arg.value, ast.Name)
                and type_arg.value.id == "svc"
                and type_arg.attr in reasonless_types
            ):
                produced.add((type_arg.attr, None))

    expected = _expected_respond_outcome_vocabulary()
    missing = expected - produced
    assert not missing, f"vocabulary pairs with no covering test: {sorted(missing)}"


# ---------------------------------------------------------------------------
# The lock-strength static guard: every `with_for_update(...)` call this
# module issues against `tasks` must pass `key_share=True`. Writing a bare
# `with_for_update()` here compiles and passes every SQLite test in this
# file (the dialect drops the clause entirely -- see
# `_answer_fence_task_predicate`'s own docstring), so nothing short of an
# AST scan of the source itself would ever catch the regression: it is a
# production deadlock waiting for the first concurrent PostgreSQL writer,
# invisible to this module's entire SQLite-backed unit suite.
# ---------------------------------------------------------------------------


def _with_for_update_calls_missing_key_share(source: str | None = None) -> list:
    """AST-scan ``source`` (the real module's own source by default) for
    every ``with_for_update(...)`` call missing ``key_share=True``. Takes an
    optional source string so the guard test below can exercise this exact
    function against a fabricated snippet, as its own positive verification,
    instead of re-implementing its walk inline against a hardcoded string --
    a second copy of the walk logic could drift from this one and still
    pass its own test while the real guard silently stopped working.
    """

    import ast as ast_module
    import inspect

    if source is None:
        source = inspect.getsource(svc)
    tree = ast_module.parse(source)
    offenders = []
    for node in ast_module.walk(tree):
        if (
            isinstance(node, ast_module.Call)
            and isinstance(node.func, ast_module.Attribute)
            and node.func.attr == "with_for_update"
        ):
            has_true_key_share = any(
                kw.arg == "key_share"
                and isinstance(kw.value, ast_module.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            if not has_true_key_share:
                offenders.append(node)
    return offenders


def test_every_with_for_update_call_passes_key_share_true() -> None:
    """`key_share=True` compiles to PostgreSQL's `FOR NO KEY UPDATE`, which
    lets a concurrent child-row insert still take its required `KEY SHARE`
    lock on the same `tasks` row. A bare `with_for_update()` compiles to the
    stronger `FOR UPDATE`, which blocks that insert and closes a lock cycle
    with any concurrent stager -- a real `DeadlockDetected` on PostgreSQL.
    This is the static half of that regression's coverage; the dynamic half
    is the PostgreSQL concurrency test this same mutation also turns red.

    The zero-offenders assertion below, against the real module, would also
    pass vacuously if the scanner itself were broken -- either its node
    matching never recognizing a ``with_for_update`` call at all, or its
    ``key_share=True`` check being vacuously true regardless of what a call
    actually passes -- since a scanner that never flags anything reports
    zero offenders on real, compliant code too. The second and third
    assertions below rule both failure modes out by calling
    ``_with_for_update_calls_missing_key_share`` itself (not a second,
    reimplemented copy of its walk, which could drift from the scanner it
    is supposed to be verifying and still pass on its own) against two
    fabricated snippets: one missing ``key_share=True`` and one carrying
    it. Confirmed by mutation: breaking the scanner's node match, or
    hardcoding ``has_true_key_share = True``, turns either assertion red."""

    offenders = _with_for_update_calls_missing_key_share()
    assert offenders == []

    bare_source = (
        "import sqlalchemy as sa\n"
        "stmt = sa.select(Task).where(Task.id == 1).with_for_update()\n"
    )
    assert len(_with_for_update_calls_missing_key_share(bare_source)) == 1

    guarded_source = (
        "import sqlalchemy as sa\n"
        "stmt = sa.select(Task).where(Task.id == 1)"
        ".with_for_update(key_share=True)\n"
    )
    assert _with_for_update_calls_missing_key_share(guarded_source) == []
