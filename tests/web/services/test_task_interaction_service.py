"""Unit coverage for ``task_interaction_service``: the outcome vocabularies,
the ``create()`` typed seam, and the compatibility materialization view.

This module accumulates coverage across every deliverable this service
ships except the shared public-chat ownership predicate (covered directly
by ``tests/web/api/test_public_chat_ownership_helper.py``, since it is
extracted from and tested alongside ``public_chat_access.py``) and the
``create()`` zero-production-caller gate (its own file,
``test_task_interaction_service_create_gate.py``).

RespondOutcome's failure matrix, and what this delivery does and does not
do with it: this build classifies every zero-rowcount answer fence miss
specifically (step 6 -- see ``respond()``'s own docstring), so the six
triggering scenarios its conservative sibling collapsed onto
``(OutcomeUnknown, None)`` now split back out into five distinct ``Stale``
reasons plus ``Conflict(already_answered)``, and the guest whose
fence-level mismatch that sibling could not label now reports
``Unauthorized(not_task_principal)``. Steps 8 and 9 are classified as
well. Step 8's staging race is classified through both of its doors: an
``IntegrityError`` through ``classify_task_command_conflict``, a
``created=False`` result through the ``payload_matches`` verdict
``stage_task_command`` already computed, and a winner's row carrying this
call's own payload is a ``Replayed`` while a mismatched one is a
``Conflict``. Step 9's commit exception is reconciled against the durable
graph in a fresh session before falling back to ``OutcomeUnknown``. So the
only ``OutcomeUnknown``-producing cells left are the two durable-graph
reconciliation failures: an ambiguous commit with nothing in the graph to
find, and one that lands under a different identity. The matrix
below enumerates this build's own 37 triggering cells, producing 20
distinct (outcome type, reason) pairs -- fewer than 37 because several
cells share a pair (seven "principal does not own this task" cells all
produce ``(RespondUnauthorized, not_task_principal)``; four "same
idempotency key, different actor" cells all produce ``(RespondConflict,
idempotency_key_reused)``; two "the task is no longer waiting" cells both
produce ``(RespondStale, run_ended)``; two distinct triggers collapse
onto ``(OutcomeUnknown, None)``; one cell, kind/version validation, is
parametrized over two reasons on its own). ``respond()`` takes no
caller-supplied optimistic-concurrency token, so there is no cell for a
stale one -- the ``Stale`` row has six pairs, not seven, for that reason.
The full cell-to-pair mapping:

    (The C1 and S1-S5 cells its conservative sibling left empty -- the
    ones only a build that classifies fence misses can reach -- are
    filled in here. Step 8's own ``UNRELATED`` classification is not a
    cell at all: it re-raises rather than returning a ``RespondOutcome``,
    and its test lives with the escape-surface tests instead.)

    OK1..OK4  -> (Accepted, None)      1 (4 cells share it -- the plain
                 accepted path, a commit that succeeds but whose
                 post-commit dispatcher notify raises, a commit whose
                 acknowledgment was lost but whose write landed, and the
                 same with the resume coordinator having already advanced
                 state_version past what this call wrote)
    V1        -> (ValidationRejected, unknown_kind)                  }  2
                 (ValidationRejected, unknown_protocol_version)      }
    V2        -> (ValidationRejected, malformed_idempotency_key)     1
    V3,V4     -> (ValidationRejected, invalid_values)      1 (2 cells share
                 it -- a non-dict ``values`` payload, and a dict
                 ``values`` payload that cannot be rendered as JSON)
    V5        -> (ValidationRejected, kind_version_mismatch)         1
    A1..A7    -> (Unauthorized, not_task_principal)      1 (7 cells share
                 it -- a user principal that does not own the task, the
                 authorization-before-idempotency ordering guard, a guest
                 principal on a non-matching task, a guest principal with
                 two populated entity-binding directions, a guest principal
                 with zero populated entity-binding directions, a guest
                 whose bindings match but whose principal.user_id is not
                 the owner's, and an ownership change racing the fence)
    U1        -> (Unavailable, task_missing)                         1
    U2        -> (Unavailable, interaction_missing)                  1
    U3        -> (Unavailable, checkpoint_unavailable)                1
    R1..R4    -> (Replayed, None)      1 (4 cells share it -- the plain
                 replay, a replay of an already-answered row whose resume
                 anchor was pruned before the retry, a staging race whose
                 winner's row carries this call's own payload, and a
                 replay recognized only at the fence-miss reread because
                 the racing command lands under this call's own key after
                 step 5's own pre-read already ran and found nothing)
    C1        -> (Conflict, already_answered)                        1
    C2..C5    -> (Conflict, idempotency_key_reused)      1 (4 cells share
                 it -- the two step-5 pre-read cells, a staging race whose
                 winner's row carries a different payload, and a
                 fence-miss reread finding a same-key command staged after
                 step 5's pre-read whose payload does not match this
                 call's own)
    S1        -> (Stale, expired)                                    1
    S2        -> (Stale, run_superseded)                             1
    S3        -> (Stale, answered_via_chat)                          1
    S4,S4'    -> (Stale, run_ended)      1 (2 cells share it -- the task
                 no longer waiting, and the overlap where it is also on a
                 different run, which pins that the status guard runs
                 before the run guard)
    S5        -> (Stale, foreign_run)                                1
    S6        -> (Stale, anchor_dangling)                            1
    X1,X2     -> (OutcomeUnknown, None)      1 (2 cells share it -- a
                 commit whose durable-graph reconciliation never finds a
                 landed write, and one whose reconciliation finds a
                 landed row under a different identity)
    -----------------------------------------------------------------
    37 cells; 20 distinct pairs (18 single-reason cells + V1's own 2)

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
| Cell-by-cell tests (this file, written) | One test per of the 37 cells, asserting outcome + reason + zero side effects | A regression in one specific cell's behavior | A forgotten test |
| Mapping meta-test (this file, written) | Each of the 20 pairs is produced by >= 1 cell's test (18 singles + one cell's own 2) | A new cell that produces a new pair with no test written for it; the two-reason cell's parametrization missing a reason | A new cell that produces no *new* pair (e.g. a ninth not_task_principal scenario) -- caught by review, not this meta-test |
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_interaction_schema_shared import (
    anchor_event_id,
    make_task,
    make_user,
)
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus, TraceEvent
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.models.user import User
from xagent.web.services import task_interaction_service as svc
from xagent.web.services.ops_signals import (
    CHECKPOINT_LOAD_UNAVAILABLE,
    CHECKPOINT_PK_ANCHOR_DANGLING,
    INTERACTION_READ_PAYLOAD_UNREADABLE,
    INTERACTION_READ_PROTOCOL_UNRECOGNIZED,
    active_degradations,
    clear_degradation,
)
from xagent.web.services.task_clarification_draft import CLARIFICATION_REQUEST_TTL
from xagent.web.services.task_interaction_staging import InteractionAnchor
from xagent.web.services.task_lease_service import TASK_RUN_ID_TRACE_FIELD, TaskLease

_DEGRADATION_SIGNALS_UNDER_TEST = (
    CHECKPOINT_PK_ANCHOR_DANGLING,
    CHECKPOINT_LOAD_UNAVAILABLE,
    INTERACTION_READ_PROTOCOL_UNRECOGNIZED,
    INTERACTION_READ_PAYLOAD_UNREADABLE,
)


@pytest.fixture(autouse=True)
def _clean_degradation_registry():
    """The anchor resolver and materialize_compatibility_view register
    process-global degradation signals on their failure paths; clear this
    module's four signals around every test so they cannot leak into tests
    that read the shared registry (the /health suite asserts exact
    payloads and fails on any leftover entry)."""
    for signal in _DEGRADATION_SIGNALS_UNDER_TEST:
        clear_degradation(signal)
    yield
    for signal in _DEGRADATION_SIGNALS_UNDER_TEST:
        clear_degradation(signal)


# ---------------------------------------------------------------------------
# CreateOutcome's vocabulary guards (two pinned numbers, still plain dicts
# in the source -- do not recompute them here):
#
#   - CreateOutcome reason word list: 12 words total (seam_not_wired was
#     deleted along with CreateNotWired once this seam's call body landed).
#   - CreateOutcome pairs this function body has a code path that returns:
#     9. Producible here means exactly that -- a code path exists in
#     create()'s own body -- not that the path is reachable from any wired
#     production caller (see CREATE_OUTCOME_PRODUCIBLE_REASONS's own
#     docstring for the two entries that stay in this set despite being
#     unreachable from create()'s one wired caller shape today).
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
# 20 (outcome, reason) pairs it derives from those classes' own Literal
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


def test_create_outcome_reason_word_list_has_exactly_12_words() -> None:
    assert len(svc.CREATE_OUTCOME_REASON_WORDS) == 12


def test_create_outcome_producible_pairs_are_exactly_9() -> None:
    total = sum(
        len(reasons) for reasons in svc.CREATE_OUTCOME_PRODUCIBLE_REASONS.values()
    )
    assert total == 9


def test_create_outcome_producible_reasons_are_a_subset_of_the_full_word_list() -> None:
    producible = {
        reason
        for reasons in svc.CREATE_OUTCOME_PRODUCIBLE_REASONS.values()
        for reason in reasons
    }
    assert producible <= svc.CREATE_OUTCOME_REASON_WORDS


def test_create_outcome_covers_exactly_the_five_reason_carrying_variants() -> None:
    assert set(svc.CREATE_OUTCOME_PRODUCIBLE_REASONS) == {
        svc.CreateValidationRejected,
        svc.CreateUnauthorized,
        svc.CreateUnavailable,
        svc.CreateConflict,
        svc.CreateStale,
    }


def test_producible_reasons_keys_are_the_outcome_types_themselves() -> None:
    """Keyed by type, not by class-name string: a rename of any outcome
    class must break loudly here rather than silently orphan its entry.

    Mutation: switching any key back to its name string turns this red."""

    for key in svc.CREATE_OUTCOME_PRODUCIBLE_REASONS:
        assert isinstance(key, type), key
        assert key.__module__ == svc.__name__, key


def test_create_not_wired_and_seam_not_wired_no_longer_exist() -> None:
    """CreateNotWired and its reason constant were deleted together, along
    with the change that fills create()'s call body -- not extended in
    place, per the word-list dict's own contract comment."""

    assert not hasattr(svc, "CreateNotWired")
    assert "seam_not_wired" not in svc.CREATE_OUTCOME_REASON_WORDS


def test_create_outcome_producible_words_minus_word_list_leaves_exactly_three() -> None:
    """checkpoint_unavailable, anchor_dangling, and run_ended stay in the
    closed word list without a producing code path in create() today (see
    CREATE_OUTCOME_PRODUCIBLE_REASONS's own docstring for why each one)."""

    producible = {
        reason
        for reasons in svc.CREATE_OUTCOME_PRODUCIBLE_REASONS.values()
        for reason in reasons
    }
    assert svc.CREATE_OUTCOME_REASON_WORDS - producible == {
        "checkpoint_unavailable",
        "anchor_dangling",
        "run_ended",
    }


def test_create_outcome_union_has_exactly_the_six_known_variants() -> None:
    import typing

    assert {cls.__name__ for cls in typing.get_args(svc.CreateOutcome)} == {
        "CreateCreated",
        "CreateValidationRejected",
        "CreateUnauthorized",
        "CreateUnavailable",
        "CreateConflict",
        "CreateStale",
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
    _db: Session, _system_call_ctx: dict[str, Any]
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
    outcome = _system_create(_db, _system_call_ctx, values=values)
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
    _db: Session, _system_call_ctx: dict[str, Any], overrides: dict[str, Any]
) -> None:
    outcome = _system_create(_db, _system_call_ctx, **overrides)
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
    _db: Session, _system_call_ctx: dict[str, Any], bad_kind: Any
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
    outcome = _system_create(_db, _system_call_ctx, kind=bad_kind)
    assert outcome == svc.CreateValidationRejected(reason="unknown_kind")


@pytest.mark.parametrize(
    "bad_version",
    [
        pytest.param(True, id="bool_true_equals_one"),
        pytest.param(1.0, id="float_equals_one"),
    ],
)
def test_cv1_protocol_version_type_confusable_values_are_rejected(
    _db: Session, _system_call_ctx: dict[str, Any], bad_version: Any
) -> None:
    """``protocol_version != INTERACTION_PROTOCOL_VERSION`` alone is not
    enough: ``True == 1`` and ``1.0 == 1`` both hold in Python, so a bare
    ``!=`` check lets a bool or a float through as if it were the int ``1``.
    The check must reject any non-``int`` (bools included, since ``bool`` is
    a subclass of ``int``) the same way the existing ``ttl_seconds`` check a
    few lines below already does."""
    outcome = _system_create(_db, _system_call_ctx, protocol_version=bad_version)
    assert outcome == svc.CreateValidationRejected(reason="unknown_protocol_version")


def test_cv2_malformed_idempotency_key_is_rejected(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    outcome = _system_create(
        _db, _system_call_ctx, request_idempotency_key="has a space"
    )
    assert outcome == svc.CreateValidationRejected(reason="malformed_idempotency_key")


def test_cv3_values_not_shaped_like_v1_payload_is_rejected(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    outcome = _system_create(_db, _system_call_ctx, values={"not": "a valid payload"})
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")


def _interaction(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"type": "text_input", "field": "env", "label": "Env"}
    item.update(overrides)
    return item


def _values(interactions: list[dict[str, Any]], message: str = "Which one?") -> Any:
    return {"message": message, "interactions": interactions}


# The write side's admissibility rules, one row per rule. Every row here is
# shape-valid per AskUserQuestionArgs and JSON-serializable, so the only
# thing that can reject it is validate_v1_write_payload.
@pytest.mark.parametrize(
    "values",
    [
        pytest.param(_values([], message="   "), id="blank_message"),
        pytest.param(_values([], message=""), id="empty_message"),
        pytest.param(
            _values([_interaction(type="carrier_pigeon")]), id="unrenderable_type"
        ),
        pytest.param(
            _values([_interaction(field="env"), _interaction(field="env")]),
            id="duplicated_field",
        ),
        pytest.param(
            _values([_interaction(type="select_one", options=None)]),
            id="select_one_without_options",
        ),
        pytest.param(
            _values([_interaction(type="select_one", options=[])]),
            id="select_one_with_an_empty_option_list",
        ),
        pytest.param(
            _values([_interaction(type="select_multiple", options=None)]),
            id="select_multiple_without_options",
        ),
        pytest.param(
            _values([_interaction(type="action_cards", options=None)]),
            id="action_cards_without_options",
        ),
        pytest.param(
            _values(
                [_interaction(options=[{"label": "Yes", "value": "yes"}])],
            ),
            id="text_input_with_options",
        ),
        pytest.param(
            _values(
                [
                    _interaction(
                        type="confirm", options=[{"label": "Yes", "value": "yes"}]
                    )
                ],
            ),
            id="confirm_with_options",
        ),
        pytest.param(
            _values([_interaction(type="number_input", min=10, max=3)]),
            id="min_greater_than_max",
        ),
        pytest.param(
            _values(
                [_interaction(type="select_one", options=[{"label": "", "value": "a"}])]
            ),
            id="option_with_a_blank_label",
        ),
        pytest.param(
            _values(
                [_interaction(type="select_one", options=[{"label": "A", "value": ""}])]
            ),
            id="option_with_a_blank_value",
        ),
        pytest.param(
            _values(
                [
                    _interaction(
                        type="action_cards",
                        options=[
                            {"label": "A", "value": "a"},
                            {"label": "", "value": ""},
                        ],
                    )
                ]
            ),
            id="one_blank_option_among_usable_ones",
        ),
    ],
)
def test_cv4_write_side_payload_rules_reject_the_envelope(
    _db: Session, _system_call_ctx: dict[str, Any], values: Any
) -> None:
    outcome = _system_create(_db, _system_call_ctx, values=values)
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(
            _values(
                [
                    _interaction(
                        type="select_one", options=[{"label": "A", "value": "a"}]
                    ),
                    _interaction(
                        type="select_multiple",
                        field="tags",
                        options=[{"label": "B", "value": "b"}],
                    ),
                    _interaction(type="text_input", field="notes"),
                    _interaction(type="file_upload", field="doc"),
                    _interaction(type="confirm", field="agree"),
                    _interaction(type="number_input", field="count", min=1, max=9),
                    _interaction(
                        type="action_cards",
                        field="action",
                        options=[{"label": "C", "value": "c"}],
                    ),
                ]
            ),
            id="one_item_of_every_v1_type",
        ),
        pytest.param(_values([]), id="prose_only_question_with_no_form"),
        pytest.param(
            _values([_interaction(type="number_input", field="n", min=3, max=3)]),
            id="min_equal_to_max",
        ),
    ],
)
def test_cv4_write_side_payload_rules_accept_a_legal_payload(
    _db: Session, _seeded_task: int, values: Any
) -> None:
    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    # A legal payload clears every validation rule and reaches the
    # write-point fence -- proven here by the fence itself firing, since a
    # user principal never gets past it (see _assert_write_point_admissible).
    # A rejected payload would instead return CreateValidationRejected
    # before ever reaching that fence.
    with pytest.raises(svc.InteractionWritePointUnfenced):
        svc.create(
            _db,
            task_id=_seeded_task,
            principal=_owning_principal(task.user_id),
            envelope=_valid_envelope(values=values),
        )


@pytest.mark.parametrize(
    "field",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="all_whitespace"),
    ],
)
def test_validate_rejects_a_blank_interaction_field(
    _db: Session, _seeded_task: int, field: str
) -> None:
    values = _values([_interaction(field=field)])
    parsed = svc.parse_v1_request_payload(values)
    with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
        svc.validate_v1_write_payload(parsed)
    assert excinfo.value.refusal == svc.InteractionWriteRefusal(
        rule="field_blank", position="request_payload.interactions[0].field"
    )


def test_validate_rejects_a_field_with_surrounding_whitespace(
    _db: Session, _seeded_task: int
) -> None:
    """Non-blank once stripped, but the stored key still would not be the
    key a strip-agnostic answer-side comparison would need: ``" a "`` never
    equal-matches an answer keyed ``"a"``, the same key-integrity reason a
    blank field is refused."""

    values = _values([_interaction(field=" a ")])
    parsed = svc.parse_v1_request_payload(values)
    with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
        svc.validate_v1_write_payload(parsed)
    assert excinfo.value.refusal == svc.InteractionWriteRefusal(
        rule="field_whitespace", position="request_payload.interactions[0].field"
    )


def test_validate_accepts_a_field_with_no_surrounding_whitespace(
    _db: Session, _seeded_task: int
) -> None:
    values = _values([_interaction(field="a")])
    parsed = svc.parse_v1_request_payload(values)
    svc.validate_v1_write_payload(parsed)


def test_validate_reports_a_blank_field_as_blank_not_as_duplicated(
    _db: Session, _seeded_task: int
) -> None:
    """Two blank fields would also be equal to each other, so the blank
    check has to run before the duplicate check reaches them -- otherwise
    the caller learns "duplicated" for a payload whose real problem is that
    neither interaction names a field at all."""

    values = _values([_interaction(field=""), _interaction(field="", label="Second")])
    parsed = svc.parse_v1_request_payload(values)
    with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
        svc.validate_v1_write_payload(parsed)
    assert excinfo.value.refusal.rule == "field_blank"


def test_validate_rejects_duplicate_option_values_within_one_interaction(
    _db: Session, _seeded_task: int
) -> None:
    values = _values(
        [
            _interaction(
                type="select_one",
                options=[
                    {"label": "First", "value": "a"},
                    {"label": "Second", "value": "a"},
                ],
            )
        ]
    )
    parsed = svc.parse_v1_request_payload(values)
    with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
        svc.validate_v1_write_payload(parsed)
    assert excinfo.value.refusal == svc.InteractionWriteRefusal(
        rule="option_value_duplicate",
        position="request_payload.interactions[0].options[1].value",
    )


def test_validate_accepts_duplicate_option_labels(
    _db: Session, _seeded_task: int
) -> None:
    """Labels may repeat -- the renderer resolves a submitted answer back to
    an option by matching on value, so two options sharing a label are only
    confusing to look at; the answer still names exactly one of them."""

    values = _values(
        [
            _interaction(
                type="select_one",
                options=[
                    {"label": "Same", "value": "a"},
                    {"label": "Same", "value": "b"},
                ],
            )
        ]
    )
    parsed = svc.parse_v1_request_payload(values)
    svc.validate_v1_write_payload(parsed)


def test_validate_accepts_a_blank_interaction_label(
    _db: Session, _seeded_task: int
) -> None:
    """This is the decision that a blank ``label`` is not refused, pinned
    down: ``clarification-form.tsx`` renders ``interaction.label ||
    interaction.field`` (line 492), so the field name stands in for a blank
    label, and ``_normalize_ask_user_interactions`` (``react.py``) repairs a
    blank ``field`` on every ``ask_user_question`` payload but never touches
    ``label``, so a model that emits ``label=""`` reaches
    ``build_clarification_payload`` with it. Refusing it would refuse a
    shape the second producer really emits."""

    values = _values([_interaction(label="")])
    parsed = svc.parse_v1_request_payload(values)
    svc.validate_v1_write_payload(parsed)


def test_validate_accepts_an_empty_accept_list_on_file_upload(
    _db: Session, _seeded_task: int
) -> None:
    values = _values([_interaction(type="file_upload", field="doc", accept=[])])
    parsed = svc.parse_v1_request_payload(values)
    svc.validate_v1_write_payload(parsed)


def test_validate_accepts_min_and_max_on_a_non_number_type(
    _db: Session, _seeded_task: int
) -> None:
    """Only number_input reads min/max; on every other type they are an
    ignored hint, and the question still asks exactly what it asks."""

    values = _values([_interaction(type="text_input", min=1, max=5)])
    parsed = svc.parse_v1_request_payload(values)
    svc.validate_v1_write_payload(parsed)


def test_validate_accepts_an_inverted_min_max_on_a_non_number_type(
    _db: Session, _seeded_task: int
) -> None:
    """The min > max rule is scoped to the one type that reads the pair.
    ``clarification-form.tsx`` passes min and max to the rendered control
    only in its number_input branch, so on a text_input an inverted range
    never reaches the user: it is a hint nobody reads, not a question
    nobody can answer, and the write is not refused for it."""

    values = _values([_interaction(type="text_input", min=10, max=3)])
    parsed = svc.parse_v1_request_payload(values)
    svc.validate_v1_write_payload(parsed)


def test_cv4_the_read_direction_parser_still_accepts_what_the_write_side_rejects(
    _db: Session, _seeded_task: int
) -> None:
    """The two directions have different failure policies for the same
    payload. parse_v1_request_payload is what the read surface calls on an
    already-persisted row, and it must keep accepting every payload it
    accepts today -- widening it would turn readable-but-odd rows into
    unanswerable ones. Only the write side refuses."""

    values = _values([_interaction(type="select_one", options=None)])
    parsed = svc.parse_v1_request_payload(values)
    assert parsed.message == "Which one?"
    assert parsed.interactions[0].type == "select_one"
    with pytest.raises(ValueError):
        svc.validate_v1_write_payload(parsed)


def test_cv3_ttl_out_of_policy_range_is_rejected_not_clamped(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    assert 1 < svc._MIN_INTERACTION_TTL_SECONDS
    outcome = _system_create(_db, _system_call_ctx, ttl_seconds=1)
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")


@pytest.mark.parametrize(
    "ttl_seconds",
    [
        pytest.param(59, id="one_below_min_rejected"),
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
    _db: Session, _system_call_ctx: dict[str, Any], ttl_seconds: Any
) -> None:
    outcome = _system_create(_db, _system_call_ctx, ttl_seconds=ttl_seconds)
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")


@pytest.mark.parametrize(
    "ttl_seconds",
    [
        pytest.param(60, id="min_boundary_passes"),
        pytest.param(604800, id="max_boundary_passes"),
    ],
)
def test_cv3_ttl_at_policy_boundary_reaches_the_write_point(
    _db: Session, _seeded_task: int, ttl_seconds: int
) -> None:
    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    envelope = _valid_envelope(ttl_seconds=ttl_seconds)
    with pytest.raises(svc.InteractionWritePointUnfenced):
        svc.create(
            _db,
            task_id=_seeded_task,
            principal=_owning_principal(task.user_id),
            envelope=envelope,
        )


def test_the_published_ttl_falls_inside_the_override_interval() -> None:
    """The two constants describe different quantities -- one the value
    the publication path writes into expires_at, the other the range a
    caller's own override has to fall inside -- and are deliberately not
    unified. The one relationship that does have to hold between them is
    pinned here, so that moving either number alone cannot leave the
    published TTL outside the range this facade would accept for it."""

    published_ttl_seconds = CLARIFICATION_REQUEST_TTL.total_seconds()
    assert (
        svc._MIN_INTERACTION_TTL_SECONDS
        <= published_ttl_seconds
        <= svc._MAX_INTERACTION_TTL_SECONDS
    )


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


def _admin_principal(user_id: int) -> svc.InteractionPrincipal:
    return svc.InteractionPrincipal(
        kind="user",
        user_id=user_id,
        is_admin=True,
        auth_mode=None,
    )


# The id create() is asked about when a scenario wants no matching row.
_ABSENT_TASK_ID = 999999999


# create()'s task lookup, one row per (principal branch x ownership x
# whether the task exists). The three branches load differently on
# purpose: a non-admin "user" and a guest both carry the same owner
# predicate (Task.user_id == principal.user_id) into the lookup, while an
# admin loads by id alone. That difference is what decides which of the
# two "no row" outcomes each branch can return, so the table below is the
# single place all of it is asserted.
#
# The guest branch's positive cell is not in this table: it needs a task
# whose agent_config carries the widget binding, which this table's
# _seeded_task fixture does not build. It lives in
# test_ca1_guest_principal_is_authorized_on_its_own_task.
#
# The three guest rows below all end in the same outcome and get there by
# three different routes, which is the point of listing them separately:
#
#   guest_on_an_absent_task            no row for that id at all
#   guest_of_another_owner_...         a row exists, the owner term in the
#                                      lookup excludes it, so nothing loads
#   guest_on_an_existing_non_matching  the owner term admits the row, and
#                                      the post-load Python predicate
#                                      (task_is_owned_by_public_principal)
#                                      refuses it on agent_config
#
# None of the three separates the owner term from the Python predicate on
# its own -- _seeded_task carries no agent_config, so the second row would
# still be refused with the owner term removed. The cell that does
# separate them needs a task whose agent_config matches in full and whose
# user_id does not, and it lives in
# test_ca1_guest_principal_is_rejected_on_another_owners_matching_task.
#
# _WRITE_POINT_UNFENCED marks a cell that clears every authorization and
# validation check -- a user/guest/admin principal fully entitled to
# create() -- and is then refused by _assert_write_point_admissible, which
# raises rather than returning an outcome (see that function's own
# docstring). These cells prove authorization succeeded up to that fence,
# not that create() reports a well-formed request as "not wired": that
# outcome no longer exists (CreateNotWired was deleted with the change
# that filled create()'s call body).
_WRITE_POINT_UNFENCED = object()


@pytest.mark.parametrize(
    ("make_principal", "task_exists", "expected"),
    [
        pytest.param(
            lambda owner_id: _owning_principal(owner_id),
            True,
            _WRITE_POINT_UNFENCED,
            id="owner_user_on_its_own_task",
        ),
        pytest.param(
            lambda owner_id: _owning_principal(owner_id + 1000),
            True,
            svc.CreateUnauthorized(reason="not_task_principal"),
            id="foreign_user_on_an_existing_task",
        ),
        pytest.param(
            lambda owner_id: _owning_principal(owner_id),
            False,
            svc.CreateUnauthorized(reason="not_task_principal"),
            id="user_on_an_absent_task",
        ),
        pytest.param(
            lambda owner_id: _admin_principal(owner_id + 1000),
            True,
            _WRITE_POINT_UNFENCED,
            id="admin_on_someone_elses_task",
        ),
        pytest.param(
            lambda owner_id: _admin_principal(owner_id + 1000),
            False,
            svc.CreateUnavailable(reason="task_missing"),
            id="admin_on_an_absent_task",
        ),
        pytest.param(
            lambda owner_id: _widget_workforce_guest_principal(
                user_id=owner_id, workforce_id=9
            ),
            False,
            svc.CreateUnauthorized(reason="not_task_principal"),
            id="guest_on_an_absent_task",
        ),
        pytest.param(
            lambda owner_id: _widget_workforce_guest_principal(
                user_id=owner_id + 1000, workforce_id=9
            ),
            True,
            svc.CreateUnauthorized(reason="not_task_principal"),
            id="guest_of_another_owner_on_an_existing_task",
        ),
        pytest.param(
            lambda owner_id: _widget_workforce_guest_principal(
                user_id=owner_id, workforce_id=9
            ),
            True,
            svc.CreateUnauthorized(reason="not_task_principal"),
            id="guest_on_an_existing_non_matching_task",
        ),
    ],
)
def test_ca2_task_lookup_is_owner_scoped_for_every_branch_but_admin(
    _db: Session,
    _seeded_task: int,
    make_principal: Any,
    task_exists: bool,
    expected: Any,
) -> None:
    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    task_id = _seeded_task if task_exists else _ABSENT_TASK_ID
    if expected is _WRITE_POINT_UNFENCED:
        with pytest.raises(svc.InteractionWritePointUnfenced):
            svc.create(
                _db,
                task_id=task_id,
                principal=make_principal(task.user_id),
                envelope=_valid_envelope(),
            )
        return
    outcome = svc.create(
        _db,
        task_id=task_id,
        principal=make_principal(task.user_id),
        envelope=_valid_envelope(),
    )
    assert outcome == expected


def test_ca2_a_non_admin_user_cannot_tell_a_foreign_task_from_an_absent_one(
    _db: Session, _seeded_task: int
) -> None:
    """The owner predicate lives in the lookup's WHERE clause, so a
    non-admin "user" principal gets one empty result set for both "this
    task belongs to someone else" and "there is no such task". Both must
    produce the identical outcome object, or the pair is an existence
    oracle for a principal not entitled to one."""

    principal = _owning_principal(999999)
    on_a_foreign_task = svc.create(
        _db, task_id=_seeded_task, principal=principal, envelope=_valid_envelope()
    )
    on_an_absent_task = svc.create(
        _db, task_id=_ABSENT_TASK_ID, principal=principal, envelope=_valid_envelope()
    )
    assert on_a_foreign_task == on_an_absent_task
    assert on_a_foreign_task == svc.CreateUnauthorized(reason="not_task_principal")


@pytest.mark.parametrize("is_admin", [False, True], ids=["plain_user", "admin"])
@pytest.mark.parametrize("task_exists", [True, False], ids=["real_task", "absent_task"])
def test_ca3_create_rejects_a_user_principal_carrying_no_user_id(
    _db: Session, _seeded_task: int, is_admin: bool, task_exists: bool
) -> None:
    """is_admin authorizes without ownership, but not without an identity.
    Rejected before the lookup on both branches: the owner predicate would
    otherwise render as Task.user_id IS NULL and match every ownerless
    task, and an admin passing on the flag alone would reach the write
    point with nothing to record as who acted."""

    principal = svc.InteractionPrincipal(
        kind="user",
        user_id=None,
        is_admin=is_admin,
        auth_mode=None,
    )
    outcome = svc.create(
        _db,
        task_id=_seeded_task if task_exists else _ABSENT_TASK_ID,
        principal=principal,
        envelope=_valid_envelope(),
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")


def test_ca4_an_unauthorized_principal_learns_nothing_about_the_payload(
    _db: Session, _seeded_task: int
) -> None:
    """A caller with no claim on the task must not learn which envelope
    shapes this service accepts. Authorization now runs before any envelope
    check, so a malformed envelope from an unauthorized caller is still
    reported as unauthorized, not as a validation rejection."""

    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    outcome = svc.create(
        _db,
        task_id=_seeded_task,
        principal=_owning_principal(task.user_id + 1000),
        envelope=_valid_envelope(kind="not_a_kind"),
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")


def test_ca4_an_absent_task_is_reported_before_the_payload_is_judged(
    _db: Session, _seeded_task: int
) -> None:
    """Mirrors the unauthorized case for the other id-only outcome: an
    admin against a task that does not exist learns task_missing, never a
    validation reason, regardless of how malformed the envelope is."""

    outcome = svc.create(
        _db,
        task_id=_ABSENT_TASK_ID,
        principal=_admin_principal(1),
        envelope=_valid_envelope(kind="not_a_kind"),
    )
    assert outcome == svc.CreateUnavailable(reason="task_missing")


@pytest.mark.parametrize(
    ("envelope_overrides", "expected_reason"),
    [
        pytest.param({"kind": "not_a_kind"}, "unknown_kind", id="bad_kind"),
        pytest.param(
            {"protocol_version": 999},
            "unknown_protocol_version",
            id="bad_protocol_version",
        ),
        pytest.param(
            {"request_idempotency_key": "not url safe!"},
            "malformed_idempotency_key",
            id="bad_idempotency_key",
        ),
        pytest.param(
            {"values": {"not": "a valid payload"}},
            "invalid_values",
            id="bad_values",
        ),
    ],
)
def test_cv_non_system_principal_is_fenced_before_validation_ever_runs(
    _db: Session,
    _seeded_task: int,
    envelope_overrides: dict[str, Any],
    expected_reason: str,
) -> None:
    """The write-point fence now runs immediately after authorization, ahead
    of every envelope check -- so a non-system principal that clears
    authorization never reaches CreateValidationRejected at all, for any of
    these four malformed shapes: InteractionWritePointUnfenced fires first,
    every time, payload untouched. This retires the earlier contract this
    same parametrization pinned (authorization moving ahead of validation
    must not change the four reasons an authorized caller sees) -- that
    contract no longer holds by design, per the reviewed decision to fence
    ahead of validation rather than after it.

    ``expected_reason`` is unused by the assertion below; kept on the
    parametrization only so this test and its retired predecessor share one
    parameter list, which is itself part of the point -- the same four
    malformed shapes that used to reach validation now never do.

    Mutation: moving ``_assert_write_point_admissible``'s call back to
    after payload validation turns this red -- each case would instead
    return CreateValidationRejected(reason=expected_reason)."""

    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    with pytest.raises(svc.InteractionWritePointUnfenced):
        svc.create(
            _db,
            task_id=_seeded_task,
            principal=_owning_principal(task.user_id),
            envelope=_valid_envelope(**envelope_overrides),
        )


def test_create_logs_the_refused_payload_diagnostic(
    _db: Session, _system_call_ctx: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    values = _values(
        [_interaction(type="select_one", options=[{"label": "", "value": "a"}])]
    )
    with caplog.at_level(logging.WARNING):
        outcome = _system_create(
            _db, _system_call_ctx, request_idempotency_key="sys-key-diag", values=values
        )

    assert outcome == svc.CreateValidationRejected(reason="invalid_values")
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "rule=option_blank" in caplog.text
    assert "position=request_payload.interactions[0].options[0]" in caplog.text


def test_cw1_fully_valid_call_reaches_the_write_point_fence(
    _db: Session, _seeded_task: int
) -> None:
    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    envelope = _valid_envelope()
    with pytest.raises(svc.InteractionWritePointUnfenced):
        svc.create(
            _db,
            task_id=_seeded_task,
            principal=_owning_principal(task.user_id),
            envelope=envelope,
        )


def test_create_never_touches_staging_or_stages_a_row_for_a_non_system_principal(
    _db: Session, _seeded_task: int
) -> None:
    """A user/guest principal never reaches the handoff at all -- confirmed
    here by asserting the table it would write to stays empty across a
    call that clears every check up to the write-point fence and is then
    refused there.

    The principal here is a foreign admin, not the admin-with-no-user-id
    this test used to carry: create() now rejects the latter before the
    lookup, so it can no longer reach the staging seam this test is about.
    That input did not lose its coverage -- it moved to
    test_ca3_create_rejects_a_user_principal_carrying_no_user_id, whose four
    cells (is_admin x task_exists) pin the rejection itself. What this test
    still owns is the seam: a call that clears every check up to the write
    point still writes no interaction row, because the write point itself
    refuses it."""

    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    envelope = _valid_envelope()
    with pytest.raises(svc.InteractionWritePointUnfenced):
        svc.create(
            _db,
            task_id=_seeded_task,
            principal=svc.InteractionPrincipal(
                kind="user",
                user_id=task.user_id + 1000,
                is_admin=True,
                auth_mode=None,
            ),
            envelope=envelope,
        )
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
    _db: Session, _system_call_ctx: dict[str, Any], bad_key: Any
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

    outcome = _system_create(_db, _system_call_ctx, request_idempotency_key=bad_key)
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
    with pytest.raises(svc.InteractionWritePointUnfenced):
        svc.create(
            _db, task_id=task_id, principal=principal, envelope=_valid_envelope()
        )


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


def test_ca1_guest_principal_is_rejected_on_another_owners_matching_task(
    _db: Session, _session_factory
) -> None:
    """Every conjunct task_is_owned_by_public_principal evaluates matches
    -- auth_mode, the workforce binding, and guest_id are all the task's
    own values -- and the task still belongs to a different user. The
    predicate deliberately does not carry the Task.user_id term (see its
    docstring: the four public-chat entry points enforce it as a filter on
    the query that loads the task, never as a post-load check), so the
    only thing that can refuse this call is create()'s own owner-scoped
    lookup. Drop Task.user_id from that lookup and this call is
    authorized against another user's task."""

    db = _session_factory()
    owner_id = make_user(db)
    other_user_id = make_user(db)
    task_id = _widget_workforce_task(db, user_id=owner_id, workforce_id=9)
    db.close()

    assert other_user_id != owner_id
    principal = _widget_workforce_guest_principal(user_id=other_user_id, workforce_id=9)
    outcome = svc.create(
        _db, task_id=task_id, principal=principal, envelope=_valid_envelope()
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")


def test_ca3_create_rejects_a_guest_principal_carrying_no_user_id(
    _db: Session, _session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guest lookup is owner-scoped, so a guest with no user_id would
    compile to Task.user_id IS NULL and reject only because the column is
    NOT NULL. create() rejects before the lookup instead, and the outcome
    object cannot tell the two apart -- both are
    Unauthorized(not_task_principal). What separates them is whether the
    query was built at all, so that is what this test asserts: with the
    pre-lookup guard in place db.query is never called; delete the guard
    and it is, even though the outcome stays the same."""

    db = _session_factory()
    owner_id = make_user(db)
    task_id = _widget_workforce_task(db, user_id=owner_id, workforce_id=9)
    db.close()

    queried: list[Any] = []
    real_query = _db.query

    def _recording_query(*args: Any, **kwargs: Any) -> Any:
        queried.append(args)
        return real_query(*args, **kwargs)

    monkeypatch.setattr(_db, "query", _recording_query)

    principal = svc.InteractionPrincipal(
        kind="guest",
        user_id=None,
        is_admin=False,
        auth_mode="widget",
        widget_workforce_id=9,
        guest_id="guest-1",
    )
    outcome = svc.create(
        _db, task_id=task_id, principal=principal, envelope=_valid_envelope()
    )
    assert outcome == svc.CreateUnauthorized(reason="not_task_principal")
    assert queried == []


def test_ca1_guest_principal_with_two_populated_directions_is_unauthorized_not_raised(
    _db: Session, _seeded_task: int
) -> None:
    """A malformed principal that populates more than one of the four
    entity-binding fields makes task_is_owned_by_public_principal raise
    ValueError; create() must catch exactly that and translate it to
    Unauthorized(not_task_principal), not let it escape as an unhandled
    exception."""

    # The guest lookup is owner-scoped, so this has to be the task's real
    # owner: a mismatched user_id would return the empty result set and
    # reject before the predicate is ever called, and this test would pass
    # without exercising the ValueError translation it exists for.
    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    principal = svc.InteractionPrincipal(
        kind="guest",
        user_id=task.user_id,
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
    # Owner-scoped for the same reason as the two-directions test above.
    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    principal = svc.InteractionPrincipal(
        kind="guest",
        user_id=task.user_id,
        is_admin=False,
        auth_mode="widget",
        guest_id="guest-1",
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
    resume_trace_event_id: int,
    resume_run_partition: str = "run-a",
    resume_execution_id: str = "exec-1",
    resume_event_id: str | None = None,
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
        resume_event_id=resume_event_id
        if resume_event_id is not None
        else anchor_event_id(db, resume_trace_event_id),
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


def test_unreadable_protocol_version_is_unanswerable_not_legacy(
    _db: Session, _seeded_task: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ck_task_interaction_requests_active_protocol`` pins every active
    row's protocol_version to 1 today, so this branch cannot be reached
    through any real write against this schema -- it is a second line of
    defense behind that database constraint, for a future protocol version
    whose active-row CHECK has not been written yet, or an older SQLite
    file that predates the constraint. Monkeypatching the row lookup is
    how this delivery tests a branch the schema itself does not yet allow
    to be constructed.

    An active row holds this task's answer slot regardless of whether its
    protocol_version is one this reader recognizes, so this can no longer
    fold back into the legacy tier -- doing so would let the caller offer
    a transcript question whose answer would land in a slot this row has
    already claimed."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    row = _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )
    row.protocol_version = 2

    def _fake_active_row(db: Session, task_id: int) -> TaskInteractionRequest:
        return row

    monkeypatch.setattr(svc, "_active_native_row", _fake_active_row)
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert view.question is None
    assert view.interactions is None
    assert view.reason == "protocol_version_unrecognized"


def test_unreadable_payload_is_unanswerable_not_legacy(
    _db: Session, _seeded_task: int
) -> None:
    """The active row's request_payload is a JSON column with no
    AskUserQuestionArgs-shape CHECK -- a row can carry any JSON dict
    that satisfies NOT NULL, so this branch is reachable through a real
    write, unlike the protocol_version branch above. A missing "message"
    field is enough to fail parse_v1_request_payload's pydantic
    validation.

    Same rule as the protocol-version branch: an active row holds the
    answer slot even when its payload cannot be parsed, so this reports
    unanswerable rather than folding back to the legacy transcript."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db,
        task_id=_seeded_task,
        resume_trace_event_id=trace_event_id,
        request_payload={"not": "a valid v1 payload"},
    )
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert view.question is None
    assert view.interactions is None
    assert view.reason == "payload_unreadable"


def test_unrecognized_protocol_version_raises_the_ops_signal_and_a_warning(
    _db: Session, _seeded_task: int, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """The unrecognized-protocol-version branch must not be silent --
    it registers a named degradation and logs a WARNING. Mutation: delete
    the register_degradation() call in that branch and this test turns
    red (INTERACTION_READ_PROTOCOL_UNRECOGNIZED never appears)."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    row = _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )
    row.protocol_version = 2

    def _fake_active_row(db: Session, task_id: int) -> TaskInteractionRequest:
        return row

    monkeypatch.setattr(svc, "_active_native_row", _fake_active_row)
    with caplog.at_level(
        logging.WARNING, logger="xagent.web.services.task_interaction_service"
    ):
        svc.materialize_compatibility_view(_db, _seeded_task)

    from xagent.web.services.ops_signals import INTERACTION_READ_PROTOCOL_UNRECOGNIZED

    assert INTERACTION_READ_PROTOCOL_UNRECOGNIZED in active_degradations()
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_unreadable_payload_raises_the_ops_signal_and_a_warning(
    _db: Session, _seeded_task: int, caplog
) -> None:
    """Same requirement as the unrecognized-protocol-version test above,
    for the payload-unreadable branch. Mutation: delete that branch's
    register_degradation() call and this test turns red."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db,
        task_id=_seeded_task,
        resume_trace_event_id=trace_event_id,
        request_payload={"not": "a valid v1 payload"},
    )
    with caplog.at_level(
        logging.WARNING, logger="xagent.web.services.task_interaction_service"
    ):
        svc.materialize_compatibility_view(_db, _seeded_task)

    from xagent.web.services.ops_signals import INTERACTION_READ_PAYLOAD_UNREADABLE

    assert INTERACTION_READ_PAYLOAD_UNREADABLE in active_degradations()
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_both_slots_empty_shape_on_unrecognized_protocol_version(
    _db: Session, _seeded_task: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One of two cells for the exact tuple shape both empty-slot
    branches must produce."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    row = _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )
    row.protocol_version = 2

    def _fake_active_row(db: Session, task_id: int) -> TaskInteractionRequest:
        return row

    monkeypatch.setattr(svc, "_active_native_row", _fake_active_row)
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert (view.question, view.interactions, view.reason) == (
        None,
        None,
        "protocol_version_unrecognized",
    )


def test_both_slots_empty_shape_on_unreadable_payload(
    _db: Session, _seeded_task: int
) -> None:
    """The other cell: same shape assertion for the payload-unreadable
    branch."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db,
        task_id=_seeded_task,
        resume_trace_event_id=trace_event_id,
        request_payload={"not": "a valid v1 payload"},
    )
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert (view.question, view.interactions, view.reason) == (
        None,
        None,
        "payload_unreadable",
    )


def test_unreadable_payload_warning_never_logs_the_rejected_question_text(
    _db: Session, _seeded_task: int, caplog
) -> None:
    """The privacy line: the validation-failure log line must
    never carry the payload's own content. pydantic's str(ValidationError)
    embeds ``input_value=``, and for this payload the input is the
    question text an end user wrote -- logging it verbatim would leak
    that text into the ops log. Mutation: swap
    ``_validation_error_summary(exc)`` back for ``str(exc)[:500]`` in the
    warning's ``extra`` and this test turns red, because the canary string
    below would then appear in a log record's ``extra``.

    Also asserts the warning was actually emitted, not just absent of the
    canary: a caplog scan over zero records passes vacuously if the log
    line is deleted or downgraded below WARNING, which would silently
    defeat every assertion above. Filtering to this module's logger at
    WARNING and requiring exactly one match, then pinning that record's
    ``validation_errors`` to the real summary this payload produces,
    closes that hole. Mutation: delete the ``logger.warning(...)`` call,
    or downgrade it to ``logger.debug(...)``, and the record-count
    assertion turns red because zero matching records survive the
    filter."""

    canary = "SENSITIVE-CANARY-9f2a"
    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db,
        task_id=_seeded_task,
        resume_trace_event_id=trace_event_id,
        # "message" must be a string per the v1 shape; an int forces a
        # validation failure while carrying the canary text in the payload
        # a real end-user question would occupy.
        request_payload={"message": 12345, "marker_text": canary},
    )
    with caplog.at_level(
        logging.WARNING, logger="xagent.web.services.task_interaction_service"
    ):
        svc.materialize_compatibility_view(_db, _seeded_task)

    warning_records = [
        record
        for record in caplog.records
        if record.name == "xagent.web.services.task_interaction_service"
        and record.levelno == logging.WARNING
    ]
    assert len(warning_records) == 1
    (warning_record,) = warning_records
    assert warning_record.validation_errors == [
        "message:string_type",
        "interactions:missing",
    ]

    for record in caplog.records:
        assert canary not in record.getMessage()
        assert canary not in repr(record.args)
        extra_values = {
            key: value
            for key, value in vars(record).items()
            if key not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__
        }
        assert canary not in repr(extra_values)


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
# T3', the other six conditions: _resolve_read_direction_anchor's row-
# validity judgment is seven self-consistency conditions ANDed together
# (task id, event type, build id, checkpoint type, run partition, execution
# identity, anchor event id -- see that function's own docstring). The
# run-partition cell is covered above; each of these six breaks exactly one
# of the remaining conditions, following the same one-condition-per-cell
# shape test_task_interaction_anchor.py's own _CONDITION_BREAKS table uses
# for the sibling resolver it covers.
# ---------------------------------------------------------------------------

_T3_ANCHOR_VALIDATION_BREAKS: dict[str, dict[str, Any]] = {
    "task_id": {"cross_task": True},
    "event_type": {"event_type": "system_update_partial"},
    "build_id": {"build_id": "build-x"},
    "checkpoint_type": {"checkpoint_type": "not_a_checkpoint_type"},
    "execution_id": {"mismatched_resume_execution_id": "exec-mismatch"},
    "resume_event_id": {"mismatched_resume_event_id": "resume-event-mismatch"},
}


@pytest.mark.parametrize("condition", sorted(_T3_ANCHOR_VALIDATION_BREAKS))
def test_t3_prime_anchor_dangling_for_each_remaining_validity_condition(
    _db: Session, _seeded_task: int, condition: str
) -> None:
    """Every other cell in this file leaves all seven conditions passing
    (or, for the run-partition cell above, breaks exactly one). Deleting any
    one of the six conditions exercised here from
    _resolve_read_direction_anchor's boolean guard must turn exactly this
    cell red and leave every other cell -- including the five remaining
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
    # None leaves the row builder to take the anchor's event id from the
    # trace row it points at, which is what makes every other cell here
    # break exactly one condition.
    resume_event_id = overrides.pop("mismatched_resume_event_id", None)

    trace_event_id = _make_trace_event(_db, task_id=trace_task_id, **overrides)
    _make_active_interaction_row(
        _db,
        task_id=_seeded_task,
        resume_trace_event_id=trace_event_id,
        resume_execution_id=resume_execution_id,
        resume_event_id=resume_event_id,
    )

    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert view.reason == "anchor_dangling"
    assert CHECKPOINT_PK_ANCHOR_DANGLING in active_degradations()


def test_t3_row_validity_failure_raises_checkpoint_pk_anchor_dangling(
    _db: Session, _seeded_task: int
) -> None:
    """The row-invalid branch of
    _resolve_read_direction_anchor's seven-condition guard must register
    CHECKPOINT_PK_ANCHOR_DANGLING, same as the missing-row branch above.
    Mutation: delete that register_degradation() call and this test turns
    red."""

    trace_event_id = _make_trace_event(
        _db, task_id=_seeded_task, checkpoint_type="not_a_checkpoint_type"
    )
    _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )

    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert CHECKPOINT_PK_ANCHOR_DANGLING in active_degradations()


def test_t2_empty_trace_side_execution_id_is_treated_as_a_match(
    _db: Session, _seeded_task: int
) -> None:
    """A checkpoint row whose own execution_id is empty
    short-circuits the execution-identity comparison to "matches"
    regardless of the interaction row's resume_execution_id, so the
    anchor still resolves and the view still reaches T2. Mutation: delete
    the ``not row_execution_id or ...`` short-circuit (comparing execution
    ids unconditionally instead) and this test turns red, because an
    empty trace-side id would then never equal a non-empty
    resume_execution_id."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task, execution_id="")
    _make_active_interaction_row(
        _db,
        task_id=_seeded_task,
        resume_trace_event_id=trace_event_id,
        resume_execution_id="exec-1",
    )

    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "native"


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
    infrastructure failing before it can even answer that question.

    Raises a SQLAlchemy error specifically, not a bare RuntimeError: the
    fetch's except clause is scoped to a whitelist of transient
    infrastructure failures, not ``sa.exc.SQLAlchemyError`` as a whole
    (see the parametrized whitelist/blacklist tests below for the full
    boundary, and test_anchor_fetch_non_sqlalchemy_error_propagates_uncaught
    for the negative case that scoping exists to draw), so this cell has
    to raise something the except tuple actually catches to keep testing
    "a session or query-layer failure", not "any Python exception"."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )

    real_get = _db.get

    def _raising_get(model: Any, pk: Any, *args: Any, **kwargs: Any) -> Any:
        if model is TraceEvent:
            raise sa.exc.OperationalError(
                "SELECT 1", {}, Exception("simulated session failure")
            )
        return real_get(model, pk, *args, **kwargs)

    monkeypatch.setattr(_db, "get", _raising_get)
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert view.reason == "checkpoint_unavailable"


_TRANSIENT_ANCHOR_FETCH_ERRORS: list[Any] = [
    pytest.param(
        lambda: sa.exc.OperationalError(
            "SELECT 1", {}, Exception("simulated connection loss")
        ),
        id="OperationalError",
    ),
    pytest.param(
        lambda: sa.exc.InterfaceError(
            "SELECT 1", {}, Exception("simulated DBAPI interface failure")
        ),
        id="InterfaceError",
    ),
    pytest.param(
        lambda: sa.exc.DisconnectionError("simulated pool-detected disconnect"),
        id="DisconnectionError",
    ),
    pytest.param(
        lambda: sa.exc.TimeoutError("simulated pool checkout timeout"),
        id="TimeoutError",
    ),
]


@pytest.mark.parametrize("exc_factory", _TRANSIENT_ANCHOR_FETCH_ERRORS)
def test_t3_checkpoint_unavailable_for_each_transient_infrastructure_error(
    _db: Session,
    _seeded_task: int,
    monkeypatch: pytest.MonkeyPatch,
    exc_factory: Callable[[], Exception],
) -> None:
    """The fallback whitelist, one cell per class: each of these four is a
    transient, recoverable infrastructure failure (see the except
    clause's own comment on ``_resolve_read_direction_anchor`` for the
    classification), and each on its own degrades the read to
    tier="unanswerable"/reason="checkpoint_unavailable" -- not only in
    combination with the others. Mutation: delete any one of these four
    from the except tuple and that case's cell turns red, because that
    exception then propagates uncaught instead of degrading."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )

    exc = exc_factory()
    real_get = _db.get

    def _raising_get(model: Any, pk: Any, *args: Any, **kwargs: Any) -> Any:
        if model is TraceEvent:
            raise exc
        return real_get(model, pk, *args, **kwargs)

    monkeypatch.setattr(_db, "get", _raising_get)
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert view.reason == "checkpoint_unavailable"


_NON_TRANSIENT_ANCHOR_FETCH_ERRORS: list[Any] = [
    pytest.param(
        lambda: sa.exc.ProgrammingError(
            "stmt", {}, Exception("simulated malformed statement")
        ),
        id="ProgrammingError",
    ),
    pytest.param(
        lambda: sa.exc.ArgumentError("simulated bad argument"),
        id="ArgumentError",
    ),
    pytest.param(
        lambda: sa.exc.CompileError("simulated compile failure"),
        id="CompileError",
    ),
    pytest.param(
        lambda: sa.exc.InvalidRequestError("simulated invalid request"),
        id="InvalidRequestError",
    ),
    pytest.param(
        lambda: sa.exc.NoResultFound("simulated no-result-found"),
        id="NoResultFound",
    ),
    pytest.param(
        lambda: sa.exc.PendingRollbackError(
            "simulated mid-transaction session failure"
        ),
        id="PendingRollbackError",
    ),
]


@pytest.mark.parametrize("exc_factory", _NON_TRANSIENT_ANCHOR_FETCH_ERRORS)
def test_anchor_fetch_non_transient_sqlalchemy_error_propagates_uncaught(
    _db: Session,
    _seeded_task: int,
    monkeypatch: pytest.MonkeyPatch,
    exc_factory: Callable[[], Exception],
) -> None:
    """The fallback blacklist, mirroring the whitelist test above one
    class at a time: each of these six is a ``sa.exc.SQLAlchemyError``
    subclass -- the old, wide ``except sa.exc.SQLAlchemyError`` used to
    swallow every one of them -- but none names a transient
    infrastructure failure the except tuple is meant to catch, so each
    must propagate to the caller instead of being misclassified as
    "checkpoint unavailable". ``PendingRollbackError`` sits here rather
    than in the whitelist above because its source is mixed -- sometimes
    a connection failure, sometimes a prior flush failure that left the
    session itself unrecoverable -- and because it is unreachable at this
    call site on either entry path: materialize_compatibility_view runs
    ``interaction_requests_table_exists``'s own ``db.connection()`` call
    first, and respond(), which reaches the resolver directly without
    that check, has already run several statements on the session it
    owns, so either way a session broken enough to raise it raises
    earlier than this fetch. Mutation:
    widen the except clause back to ``except sa.exc.SQLAlchemyError`` and
    every case in this parametrization turns red, because all six would
    then be swallowed and reported as tier="unanswerable" instead of
    raising."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )

    exc = exc_factory()

    def _raising_get(model: Any, pk: Any, *args: Any, **kwargs: Any) -> Any:
        if model is TraceEvent:
            raise exc
        raise AssertionError(f"unexpected db.get({model!r}, {pk!r})")

    monkeypatch.setattr(_db, "get", _raising_get)
    with pytest.raises(type(exc)):
        svc.materialize_compatibility_view(_db, _seeded_task)


def test_anchor_fetch_non_sqlalchemy_error_propagates_uncaught(
    _db: Session, _seeded_task: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anchor fetch's except clause only catches
    ``sa.exc.SQLAlchemyError`` subclasses at all (and only a whitelisted
    subset of those -- see the parametrized tests above) -- a programming
    error that is not even a SQLAlchemy error (a TypeError, here) must
    propagate to the caller rather than being misclassified as a
    checkpoint that has become unavailable. Mutation: widen the except
    clause back to ``except Exception`` and this test turns red, because
    the TypeError would then be swallowed and reported as
    tier="unanswerable" instead of raising."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )

    def _raising_get(model: Any, pk: Any, *args: Any, **kwargs: Any) -> Any:
        if model is TraceEvent:
            raise TypeError("not a SQLAlchemy error")
        raise AssertionError(f"unexpected db.get({model!r}, {pk!r})")

    monkeypatch.setattr(_db, "get", _raising_get)
    with pytest.raises(TypeError, match="not a SQLAlchemy error"):
        svc.materialize_compatibility_view(_db, _seeded_task)


def test_the_session_survives_a_failed_anchor_fetch_with_no_rollback(
    _db: Session, _seeded_task: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "no db.rollback()" definitive regression test: the anchor
    fetch's except clause deliberately does
    not roll back: the session it reads through belongs to the caller,
    so disposing of a failed transaction is the session owner's call,
    not this read helper's. (The same module's respond() owns a session
    of its own and does commit and roll back on it; that ownership
    boundary is exactly the point. See the except clause's comment.)

    Proves this end to end in one real session: a write the caller had
    already staged, uncommitted, before the failing read survives it and
    can still be committed; the same session can still run a plain query
    afterward; and the same session can still complete a whole second
    materialize_compatibility_view() call -- a full response construction
    -- once the transient failure clears.

    Mutation: add ``db.rollback()`` to the except clause and the first
    assertion below -- the caller's staged write surviving -- turns red,
    because the rollback discards it along with anything else the caller
    had pending."""

    trace_event_id = _make_trace_event(_db, task_id=_seeded_task)
    _make_active_interaction_row(
        _db, task_id=_seeded_task, resume_trace_event_id=trace_event_id
    )

    # The caller's own staged, uncommitted write -- simulates a caller
    # that has already modified something in this same session before
    # asking the read surface for the pending question.
    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    task.title = "staged-before-the-failing-read"

    real_get = _db.get
    raise_once = {"armed": True}

    def _raising_get(model: Any, pk: Any, *args: Any, **kwargs: Any) -> Any:
        if model is TraceEvent and raise_once["armed"]:
            raise_once["armed"] = False
            raise sa.exc.OperationalError(
                "SELECT 1", {}, Exception("simulated session failure")
            )
        return real_get(model, pk, *args, **kwargs)

    monkeypatch.setattr(_db, "get", _raising_get)
    view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert view.tier == "unanswerable"
    assert view.reason == "checkpoint_unavailable"

    # (a) the caller's own pending write is still staged and committable --
    # a db.rollback() in the except clause would have discarded it.
    _db.commit()
    reloaded = _db.query(Task).filter(Task.id == _seeded_task).first()
    assert reloaded.title == "staged-before-the-failing-read"

    # (b) the same session can still run a plain query...
    still_readable = (
        _db.query(TaskInteractionRequest)
        .filter(TaskInteractionRequest.task_id == _seeded_task)
        .first()
    )
    assert still_readable is not None

    # ...and complete a whole second response construction: the transient
    # failure was armed for exactly one call, so the real db.get resumes
    # and the anchor now resolves normally.
    second_view = svc.materialize_compatibility_view(_db, _seeded_task)
    assert second_view.tier == "native"


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
    it points at). A live anchor is still what makes the fence-miss
    classification tests below reachable past step 5.5; the
    already-answered replay tests no longer need it, since step 5's
    pre-read now recognizes the replay before anchor resolution runs (the
    pruned-anchor replay test below is what pins that down)."""

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


def _terminated_row_with_valid_anchor(
    session_factory,
    *,
    task_id: int,
    run_id: str,
    terminal_reason: str,
) -> int:
    """The terminated counterpart to ``_answered_row_with_valid_anchor``:
    a row some earlier writer closed out, its resume anchor still live so
    the call under test reaches the fence (step 6) and misses there
    rather than being turned away at step 5.5's anchor resolution."""

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
        row.status = "terminated"
        row.active_slot = None
        row.terminal_reason = terminal_reason
        row.terminated_at = now
        row.request_idempotency_key = "terminated-row-key"
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
            actor_subject=_actor_subject(db, actor_user_id),
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


def _actor_subject(db: Session, actor_user_id: int | None) -> str | None:
    if actor_user_id is None:
        return None
    subject = db.query(User.actor_subject).filter(User.id == actor_user_id).scalar()
    assert subject is not None
    return str(subject)


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


@pytest.mark.parametrize("is_admin", [False, True], ids=["plain_user", "admin"])
def test_respond_rejects_a_user_principal_carrying_no_user_id(
    _respond_db, is_admin: bool
) -> None:
    """is_admin authorizes without ownership, but not without an identity:
    a principal that passed on the flag alone would reach the write point
    with nothing to record as who answered."""

    _owner_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = svc.InteractionPrincipal(
        kind="user",
        user_id=None,
        is_admin=is_admin,
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
    concurrent change to catch only via SQLite's missing lock.

    The label comes from step 6's reread: the row is still ``active`` and
    the task is still waiting on this same run, so the only fence term
    left that can have failed is the ownership conjunction, and this build
    reports that as ``Unauthorized(not_task_principal)`` rather than the
    unlabelled ``OutcomeUnknown`` its conservative sibling gave it. The
    security property this test exists to pin -- the guest never gets an
    answer accepted -- held either way; only the label changed."""

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

        assert outcome == svc.RespondUnauthorized(reason="not_task_principal")


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


@pytest.mark.parametrize(
    ("make_principal", "agent_config"),
    [
        pytest.param(
            lambda owner_id: svc.InteractionPrincipal(
                kind="user", user_id=None, is_admin=True, auth_mode=None
            ),
            None,
            id="user_without_an_id",
        ),
        pytest.param(
            lambda owner_id: svc.InteractionPrincipal(
                kind="guest",
                user_id=owner_id,
                is_admin=False,
                auth_mode="widget",
                widget_workforce_id=9,
                guest_id="",
            ),
            # Every conjunct ahead of the guest_id pair matches -- the
            # auth_mode and the workforce binding -- and the task's own
            # guest_id is blank too, so the equality below the guard
            # ("" == "") would pass as well. The one thing left to refuse
            # this call is the guard that requires principal.guest_id to
            # be non-empty before the comparison runs. Without an
            # agent_config the task carries none at all, the auth_mode
            # conjunct does the refusing, and the guard is never the
            # reason the assertion holds.
            {
                "auth_mode": "widget",
                "widget_workforce_id": 9,
                "guest_id": "",
            },
            id="guest_with_a_blank_guest_id",
        ),
    ],
)
def test_respond_rejects_every_principal_identity_string_cannot_name(
    _respond_db, make_principal: Any, agent_config: dict[str, Any] | None
) -> None:
    """``identity_string()`` raises for exactly two principal shapes, and
    ``respond()`` calls it at four points with no guard of its own. What
    keeps those four calls safe is the authorization gate above them, which
    happens to require the same fields -- a real coupling that nothing
    enforced until this test. Each shape below must come back as
    ``RespondUnauthorized(reason="not_task_principal")``, never as a raised
    ``ValueError`` escaping the function."""

    owner_id, task_id = _waiting_task(_respond_db, agent_config=agent_config)
    principal = make_principal(owner_id)

    with pytest.raises(ValueError):
        principal.identity_string()

    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
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
    """_resolve_read_direction_anchor's fetch is caught on a whitelist of
    transient infrastructure failures (``OperationalError``,
    ``InterfaceError``, ``DisconnectionError``, ``TimeoutError``), not on
    ``sa.exc.SQLAlchemyError`` as a whole, so this cell raises a
    whitelisted class to keep testing "a session or query-layer failure",
    not "any Python exception" -- see that resolver's own except clause
    for why the whitelist exists and is scoped that narrowly."""

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
                raise sa.exc.OperationalError(
                    "SELECT 1", {}, Exception("simulated session failure")
                )
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


def test_respond_rejects_a_legacy_command_from_a_reused_numeric_actor_id(
    _respond_db,
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    principal = _owning_principal(user_id)
    values = {"env": "prod"}
    command_id = "legacy-actor-replay-key"
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
    command_db_id = _stage_matching_command(
        _respond_db,
        task_id=task_id,
        actor_user_id=principal.user_id,
        command_id=command_id,
        payload=payload,
    )
    db = _respond_db()
    try:
        command = db.get(TaskExecutionCommand, command_db_id)
        assert command is not None
        command.actor_subject = f"legacy-user-id:{user_id}"
        assert _actor_subject(db, user_id) != command.actor_subject
        db.commit()
    finally:
        db.close()

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=_respond_envelope(idempotency_key=command_id, values=values),
    )

    assert outcome == svc.RespondConflict(reason="idempotency_key_reused")


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
    """``_respond_receipt`` may only ever see an answered row: its two
    callers each guarantee that on their own terms -- the idempotent-replay
    pre-read branch finds the row by this call's own idempotency key, which
    only matches an already-staged (and therefore already-answered) RESUME
    command, and ``_verify_respond_durable_graph`` only reaches this call
    after its own checks already confirmed ``status == "answered"`` on the
    row it is holding -- and the paired CHECK constraints make an answered
    row with a NULL ``responded_at`` or ``responder_identity`` impossible
    either way. That reasoning spans two modules with nothing else pinning
    it, so the builder raises loudly on a row with no answer rather than
    coercing ``None`` into the string ``'None'`` (or a falsy ``""``) inside
    an audit-bearing receipt."""

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


def test_respond_reports_conflict_for_an_already_answered_row_under_a_fresh_key(
    _respond_db,
) -> None:
    """The fence misses because the row is already answered, and this
    build classifies that miss instead of collapsing it onto
    ``OutcomeUnknown``: a fresh idempotency key this call has never seen
    means step 5's pre-read finds nothing, so the reread at step 6 is the
    first place the pre-existing answer becomes visible. The classified
    ``Conflict`` is what increments the response-conflict counter -- the
    conservative sibling left the counter untouched here, because it had
    never confirmed the miss was a conflict. Everything else is
    unchanged: the pre-existing answer and its already-staged command are
    left exactly as they were."""

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

        assert outcome == svc.RespondConflict(reason="already_answered")
    assert _conflict_counter() == before_counter + 1


def test_respond_logs_the_reread_row_state_when_the_fence_misses(
    _respond_db, caplog: pytest.LogCaptureFixture
) -> None:
    """The reread-state log survives this build's classification rather
    than being replaced by it. The two carry different information: the
    caller gets ``Conflict(already_answered)``, which names neither who
    answered the row nor which run it belonged to, while the operator's
    log line carries ``status``, ``active_slot``, ``terminal_reason``,
    ``run_id`` and ``responder_identity`` -- columns no ``RespondOutcome``
    variant has a field for. Asserting the log here, on a cell whose
    outcome is now specific, is what keeps the classification from being
    read as a licence to drop it."""

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

    assert outcome == svc.RespondConflict(reason="already_answered")
    matching = [
        record
        for record in caplog.records
        if "answer fence matched zero rows" in record.getMessage()
    ]
    assert len(matching) == 1
    assert "status=answered" in matching[0].getMessage()


def _racing_fence_stmt_that_answers_with_command(
    session_factory,
    *,
    interaction_id: int,
    task_id: int,
    responder_identity: str,
    response_payload: dict[str, Any],
    command_id: str,
    actor_user_id: int | None,
    command_payload: dict[str, Any],
    race_result: dict[str, Any],
) -> Any:
    """Build a monkeypatch replacement for ``svc._answer_fence_stmt`` that
    races a concurrent answer plus a same-idempotency-key RESUME command
    onto the row in the one window that can produce it: after step 5's
    idempotency pre-read has already run and found nothing under this key,
    but before step 6's own fence UPDATE (and its zero-rowcount reread)
    sees the row. Staging the same race as ordinary test setup *before*
    calling ``respond()`` -- the way the C2/C3 pre-read tests below do --
    would make step 5's own pre-read find it first and answer from there,
    never reaching the reread branch this pair of tests exists to cover.

    Records the post-race graph snapshot and the racer's own staged
    ``command_db_id`` into ``race_result`` so the two tests using this can
    assert, respectively, that the receipt names the racer's row and that
    this call's own losing attempt adds nothing on top of the race."""

    real_fence_stmt = svc._answer_fence_stmt

    def _racing_fence_stmt(*args: Any, **kwargs: Any) -> Any:
        race_db = session_factory()
        try:
            race_db.query(TaskInteractionRequest).filter(
                TaskInteractionRequest.id == interaction_id
            ).update(
                {
                    "status": "answered",
                    "active_slot": None,
                    "response_payload": response_payload,
                    "responded_at": _now(),
                    "responder_identity": responder_identity,
                }
            )
            command = TaskExecutionCommand(
                task_id=task_id,
                actor_user_id=actor_user_id,
                actor_subject=_actor_subject(race_db, actor_user_id),
                command_id=command_id,
                kind=svc.TaskCommandKind.RESUME.value,
                payload=command_payload,
                status="completed",
            )
            race_db.add(command)
            race_db.commit()
            race_db.refresh(command)
            race_result["command_db_id"] = int(command.id)
        finally:
            race_db.close()
        race_result["snapshot"] = _graph_snapshot(
            session_factory, task_id=task_id, interaction_id=interaction_id
        )
        return real_fence_stmt(*args, **kwargs)

    return _racing_fence_stmt


def test_respond_reports_replay_when_a_racing_command_lands_between_the_preread_and_the_fence(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fence-miss reread branch (step 6), not step 5's own pre-read,
    is what recognizes this replay: the racing command is staged in the
    window between the two, so step 5 finds nothing and this call only
    discovers its own answer already landed once it rereads the row after
    losing the fence. The receipt it returns must name the racing
    command's own row, not synthesize one of this call's own."""

    user_id, task_id = _waiting_task(_respond_db)
    principal = _owning_principal(user_id)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    envelope = _respond_envelope(idempotency_key="racing-replay-key")
    command_payload = svc._respond_command_payload(
        interaction_id=interaction_id, principal=principal, values=envelope.values
    )
    race_result: dict[str, Any] = {}
    monkeypatch.setattr(
        svc,
        "_answer_fence_stmt",
        _racing_fence_stmt_that_answers_with_command(
            _respond_db,
            interaction_id=interaction_id,
            task_id=task_id,
            responder_identity=principal.identity_string(),
            response_payload=envelope.values,
            command_id=envelope.idempotency_key,
            actor_user_id=principal.user_id,
            command_payload=command_payload,
            race_result=race_result,
        ),
    )

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=envelope,
    )

    assert isinstance(outcome, svc.RespondReplayed)
    assert outcome.receipt.command_db_id == race_result["command_db_id"]


def test_respond_reports_conflict_when_a_racing_command_with_a_different_payload_lands_between_the_preread_and_the_fence(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same race as the replay test above, except the command that lands
    under this call's idempotency key carries different answer values --
    a second submitter reusing the key, not this call's own retry. The
    fence-miss reread branch must draw the same already-answered/idempotency
    distinction step 5's own pre-read draws: a same-key command that does
    not match this call's payload is ``idempotency_key_reused``, not
    ``already_answered`` (which is reserved for a miss where no command
    exists under this key at all). This call's own losing attempt must add
    nothing beyond what the race itself committed -- checked directly
    against the race's own post-commit snapshot, since the standard
    ``_asserts_no_side_effects`` helper's pre-call snapshot would predate
    the race and always show a diff that is the race's doing, not this
    call's."""

    user_id, task_id = _waiting_task(_respond_db)
    principal = _owning_principal(user_id)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    envelope = _respond_envelope(idempotency_key="racing-conflict-key")
    mismatched_payload = svc._respond_command_payload(
        interaction_id=interaction_id, principal=principal, values={"env": "staging"}
    )
    race_result: dict[str, Any] = {}
    monkeypatch.setattr(
        svc,
        "_answer_fence_stmt",
        _racing_fence_stmt_that_answers_with_command(
            _respond_db,
            interaction_id=interaction_id,
            task_id=task_id,
            responder_identity=principal.identity_string(),
            response_payload={"env": "staging"},
            command_id=envelope.idempotency_key,
            actor_user_id=principal.user_id,
            command_payload=mismatched_payload,
            race_result=race_result,
        ),
    )
    before_counter = _conflict_counter()

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=envelope,
    )

    assert outcome == svc.RespondConflict(reason="idempotency_key_reused")
    assert _conflict_counter() == before_counter + 1
    after = _graph_snapshot(_respond_db, task_id=task_id, interaction_id=interaction_id)
    assert after == race_result["snapshot"]


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


@pytest.mark.parametrize(
    "terminal_reason,expected_reason",
    [
        pytest.param("deadline_elapsed", "expired", id="expired"),
        pytest.param("run_superseded", "run_superseded", id="run_superseded"),
        pytest.param(
            "answered_via_legacy_resume", "answered_via_chat", id="answered_via_chat"
        ),
    ],
)
def test_respond_reports_stale_for_a_terminated_row(
    _respond_db, terminal_reason: str, expected_reason: str
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _terminated_row_with_valid_anchor(
        _respond_db, task_id=task_id, run_id="run-a", terminal_reason=terminal_reason
    )
    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(idempotency_key="never-seen-before-terminal"),
        )

        # Written out as three literal constructions rather than the one
        # line ``svc.RespondStale(reason=expected_reason)`` this branch
        # collapses to, kept apart deliberately: the mapping meta-test at
        # the bottom of this file finds a covered pair by AST-scanning for
        # ``svc.Respond*(reason=<constant>)``, and a variable in that slot
        # is invisible to that scan. Written as a literal here is what
        # makes each of the three reasons visible to it; collapsing to the
        # variable form would not fail silently -- the meta-test would
        # turn red, reporting these same three ``Stale`` pairs as having
        # zero covering cells.
        if expected_reason == "expired":
            assert outcome == svc.RespondStale(reason="expired")
        elif expected_reason == "run_superseded":
            assert outcome == svc.RespondStale(reason="run_superseded")
        else:
            assert outcome == svc.RespondStale(reason="answered_via_chat")
    # Reverse assertion: a Stale outcome must never increment the
    # response-conflict counter -- only the three Conflict cells do.
    assert _conflict_counter() == before_counter


def test_respond_reports_stale_when_the_task_has_moved_on_from_waiting(
    _respond_db,
) -> None:
    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    db = _respond_db()
    try:
        db.query(Task).filter(Task.id == task_id).update({"status": TaskStatus.RUNNING})
        db.commit()
    finally:
        db.close()
    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondStale(reason="run_ended")
    assert _conflict_counter() == before_counter


def test_respond_prefers_run_ended_over_foreign_run_when_both_hold(
    _respond_db,
) -> None:
    """The overlap: the task is no longer ``WAITING_FOR_USER`` *and* it has
    moved to a different ``run_id`` than the interaction row's. This is the
    ordinary production shape, not a corner -- one run ends and the next
    one starts, so ``status`` and ``run_id`` change together -- and it is
    the only cell in which the order of the two guards is observable at
    all. Either guard alone is already pinned by the two tests around this
    one; neither of them can tell which runs first.

    The contract this pins is that order: with both conditions true the
    caller gets ``run_ended``, because "this task is not waiting for an
    answer any more" is the more accurate and the less leaky of the two
    answers. It is more accurate because it describes the task's own
    current state rather than a relationship between two rows, and a
    caller told ``foreign_run`` would reasonably conclude that answering
    the *current* run's question would still work, which is false here.
    It is less leaky because ``foreign_run`` implicitly reports that some
    other run exists and that this row belongs to an older one -- a fact
    about the task's history that a caller who is simply too late does not
    need. Swapping the two ``if`` statements in ``respond()``'s step 6
    classification must turn this test red; nothing else in the suite
    notices the swap."""

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    db = _respond_db()
    try:
        db.query(Task).filter(Task.id == task_id).update(
            {"status": TaskStatus.RUNNING, "run_id": "run-next"}
        )
        db.commit()
    finally:
        db.close()
    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondStale(reason="run_ended")
    assert _conflict_counter() == before_counter


def test_respond_reports_stale_when_the_task_is_waiting_on_a_different_run(
    _respond_db,
) -> None:
    """The interaction row belongs to the run it was created under; the
    task has since moved on to a new one. Both helpers build their rows on
    ``run-a``, so the divergence is introduced afterwards by advancing the
    task alone -- the same shape as the ``run_ended`` test above, and the
    reason neither helper carries a ``run_id`` parameter."""

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    db = _respond_db()
    try:
        db.query(Task).filter(Task.id == task_id).update({"run_id": "run-current"})
        db.commit()
    finally:
        db.close()
    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=_owning_principal(user_id),
            envelope=_respond_envelope(),
        )

        assert outcome == svc.RespondStale(reason="foreign_run")
    assert _conflict_counter() == before_counter


def _racing_fence_stmt_that_changes_ownership(
    session_factory, *, task_id: int, intruder_owner_id: int
) -> Any:
    """Build a monkeypatch replacement for ``svc._answer_fence_stmt`` that
    races a concurrent ownership change onto the task row between step 2's
    read and the fence statement's own execution -- the SQLite-only TOCTOU
    window (SQLite's dialect drops ``FOR UPDATE`` entirely; PostgreSQL's row
    lock closes it, see ``_answer_fence_task_predicate``'s own docstring) the
    test below needs to reproduce it."""

    real_fence_stmt = svc._answer_fence_stmt

    def _racing_fence_stmt(*args: Any, **kwargs: Any) -> Any:
        race_db = session_factory()
        try:
            race_db.query(Task).filter(Task.id == task_id).update(
                {"user_id": intruder_owner_id}
            )
            race_db.commit()
        finally:
            race_db.close()
        return real_fence_stmt(*args, **kwargs)

    return _racing_fence_stmt


def test_respond_reports_unauthorized_when_ownership_changes_between_the_check_and_the_write(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SQLite-only TOCTOU window between step 3's authorization read and
    step 6's write-point fence (see ``_answer_fence_task_predicate``'s own
    docstring for why PostgreSQL's row lock closes this window and SQLite's
    dialect does not)."""

    owner_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    intruder_owner_id = make_user(_respond_db())

    monkeypatch.setattr(
        svc,
        "_answer_fence_stmt",
        _racing_fence_stmt_that_changes_ownership(
            _respond_db, task_id=task_id, intruder_owner_id=intruder_owner_id
        ),
    )

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=_owning_principal(owner_id),
        envelope=_respond_envelope(),
    )

    assert outcome == svc.RespondUnauthorized(reason="not_task_principal")


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


def test_respond_reports_outcome_unknown_when_commit_ack_is_lost_and_the_graph_never_lands(
    _respond_db, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Injects a commit that raises without ever reaching the database, so
    the durable-graph reconciliation has nothing to find and all three of
    its attempts fail. Keeps both of the assertions the conservative
    sibling made on this same door, because reconciling does not weaken
    either: nothing landed (the interaction row, the task row and the
    command table are exactly as they were), and the lost acknowledgment
    is logged. The log fires before the reconciliation runs and regardless
    of what it concludes -- an operator needs the record that an
    acknowledgment went missing even on the runs where the write is later
    confirmed."""

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    from sqlalchemy.orm import Session as OrmSession

    original_commit = OrmSession.commit
    call_count = {"n": 0}

    def _failing_commit(self: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated lost commit acknowledgment")
        return original_commit(self)

    monkeypatch.setattr(OrmSession, "commit", _failing_commit)
    monkeypatch.setattr(svc, "_RESPOND_DURABLE_GRAPH_RETRY_SLEEP_SECONDS", 0.0)

    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        with caplog.at_level(logging.WARNING):
            outcome = svc.respond(
                interaction_id=interaction_id,
                task_id=task_id,
                principal=principal,
                envelope=_respond_envelope(idempotency_key="ambiguous-commit-no-graph"),
            )

        assert isinstance(outcome, svc.RespondOutcomeUnknown)
        matching = [
            record
            for record in caplog.records
            if "commit failed while answering" in record.getMessage()
        ]
        assert len(matching) == 1
        assert matching[0].levelno == logging.WARNING


def test_respond_reports_accepted_when_commit_ack_is_lost_but_the_graph_landed(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same injected failure as the OutcomeUnknown case above, except this
    time the underlying commit genuinely succeeded at the database layer
    (only the acknowledgment back to this process was lost) -- proving the
    reconciliation path returns Accepted, not a false negative, once the
    complete graph is actually there to find."""

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    from sqlalchemy.orm import Session as OrmSession

    original_commit = OrmSession.commit
    call_count = {"n": 0}

    def _lost_ack_but_committed(self: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            original_commit(self)
            raise RuntimeError("simulated lost commit acknowledgment")
        return original_commit(self)

    monkeypatch.setattr(OrmSession, "commit", _lost_ack_but_committed)
    monkeypatch.setattr(svc, "_RESPOND_DURABLE_GRAPH_RETRY_SLEEP_SECONDS", 0.0)

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=_respond_envelope(idempotency_key="ambiguous-commit-landed"),
    )

    assert isinstance(outcome, svc.RespondAccepted)
    assert outcome.receipt.responder_identity == principal.identity_string()


def test_respond_durable_graph_check_is_monotone_when_the_coordinator_has_already_advanced_state(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resume coordinator re-issues the same RESUME_REQUESTED
    transition once it applies the command this call staged, bumping
    state_version a second time -- the durable-graph check must still
    report Accepted against a state_version that has moved past what this
    call itself wrote, and must not compare control_state (see respond()'s
    own docstring)."""

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    from sqlalchemy.orm import Session as OrmSession

    original_commit = OrmSession.commit
    call_count = {"n": 0}

    def _lost_ack_then_coordinator_advances(self: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            original_commit(self)
            # Simulate the coordinator re-applying its own transition on a
            # separate connection before this call's reconciliation runs.
            side_db = _respond_db()
            try:
                side_db.query(Task).filter(Task.id == task_id).update(
                    {
                        "state_version": Task.state_version + 1,
                        "control_state": "running",
                    }
                )
                side_db.commit()
            finally:
                side_db.close()
            raise RuntimeError("simulated lost commit acknowledgment")
        return original_commit(self)

    monkeypatch.setattr(OrmSession, "commit", _lost_ack_then_coordinator_advances)
    monkeypatch.setattr(svc, "_RESPOND_DURABLE_GRAPH_RETRY_SLEEP_SECONDS", 0.0)

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=_respond_envelope(idempotency_key="monotone-check"),
    )

    assert isinstance(outcome, svc.RespondAccepted)


def test_respond_reports_outcome_unknown_when_the_landed_row_answers_a_different_identity(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same lost-commit-acknowledgment injection as the two cases above, but
    this time the row the durable-graph check reads back was answered under
    a *different* ``responder_identity`` than the one this call's own
    principal carries. A mismatched identity, not an absent one, is the
    only way an answered row can fail to attest to this call: two CHECK
    constraints chain to rule out a null identity on an answered row --
    ``ck_task_interaction_requests_responded_at_pairs_status`` ties
    ``status = 'answered'`` to ``responded_at IS NOT NULL``, and
    ``ck_task_interaction_requests_responder_pairs_responded_at`` ties
    ``responded_at IS NOT NULL`` to ``responder_identity IS NOT NULL`` --
    so an answered row always carries *some* identity, just not
    necessarily this call's own. See ``_verify_respond_durable_graph``'s
    own docstring for why the comparison exists. The reconciliation must
    not treat "an answer landed" as "my answer landed": with nothing tying
    the row to this principal, every attempt has to keep failing and the
    call must report OutcomeUnknown, never Accepted.
    """

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    from sqlalchemy.orm import Session as OrmSession

    original_commit = OrmSession.commit
    call_count = {"n": 0}

    def _lost_ack_but_landed_under_another_identity(self: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            original_commit(self)
            # The row genuinely committed (columns and all), but stamped
            # with a different identity than this call's own principal --
            # standing in for the row this call's ambiguous commit might
            # have raced against, not a corruption of this call's own
            # write.
            side_db = _respond_db()
            try:
                side_db.query(TaskInteractionRequest).filter(
                    TaskInteractionRequest.id == interaction_id
                ).update({"responder_identity": "user:999999"})
                side_db.commit()
            finally:
                side_db.close()
            raise RuntimeError("simulated lost commit acknowledgment")
        return original_commit(self)

    monkeypatch.setattr(
        OrmSession, "commit", _lost_ack_but_landed_under_another_identity
    )
    monkeypatch.setattr(svc, "_RESPOND_DURABLE_GRAPH_RETRY_SLEEP_SECONDS", 0.0)

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=_respond_envelope(idempotency_key="ambiguous-commit-wrong-identity"),
    )

    assert isinstance(outcome, svc.RespondOutcomeUnknown)


# ---------------------------------------------------------------------------
# classify_task_command_conflict's IntegrityError branch (step 8), reached
# when stage_task_command's own insert (inside this call's own
# `db.begin_nested()`) collides with a UNIQUE or FOREIGN KEY constraint.
# This repo already has real coverage for classify_task_command_conflict
# itself (test_task_command_transport.py) and for the pre-existing-command
# shortcut at step 5 (the idempotency_key_reused cells above), but never for
# this specific call site -- respond() reaching an IntegrityError on its own
# insert has no test anywhere in the repo before this delivery.
#
# Of TaskCommandConflictKind's three values, only RACED_DUPLICATE is
# reachable through a real race here, and only on PostgreSQL. The other
# two are not reachable through respond() at all, verified against the
# real code rather than assumed:
#
# - UNRELATED needs the staged row's actor_user_id foreign key to fail --
#   the referenced user deleted out from under a still-pending insert. In
#   respond(), actor_user_id is always `principal.user_id`, and the answer
#   fence's own task-side predicate (`_answer_fence_task_predicate`)
#   requires `Task.user_id == principal.user_id` unconditionally for
#   *both* principal kinds (guest included -- its "owning user" is the
#   value in this same term) as a condition of the fence UPDATE actually
#   matching a row. Confirmed by running an admin principal whose user_id
#   does not equal the task's owner through respond(): the fence's own
#   ownership term makes its UPDATE match zero rows, and this build
#   returns at step 6's rowcount=0 branch long before step 8 --
#   `principal.is_admin` only bypasses the earlier, separate Python check
#   at step 3, never the fence's own SQL. So the only user_id that can ever
#   reach step 8 is the task's actual owner, and that user cannot be
#   deleted while this same transaction still holds the task row it owns
#   (`tasks.user_id` has no `ondelete`, i.e. FK `RESTRICT` -- deleting it
#   while a referencing task row exists is rejected at the database level,
#   independent of any lock this transaction holds). No real race can
#   produce UNRELATED, so it gets no race test -- but the branch that
#   handles it is not left uncovered: the first test below injects an
#   IntegrityError with no duplicate row behind it, which is exactly what
#   classify_task_command_conflict calls UNRELATED, and pins that
#   respond() re-raises it instead of folding a database-level failure
#   into a typed outcome.
# - TASK_MISSING needs the task to disappear between step 2 and step 8.
#   respond() holds the tasks row through step 2's `with_for_update` for
#   the whole transaction, and by step 8 has already written on this same
#   connection at steps 6 and 7 -- on PostgreSQL that is the row lock
#   itself blocking a concurrent DELETE; on SQLite (where `with_for_update`
#   compiles to nothing, see
#   `test_every_with_for_update_call_passes_key_share_true`'s own
#   docstring) it is SQLite's whole-database writer lock, already held by
#   this connection since step 6's fence UPDATE. Either way a concurrent
#   delete of this task cannot land before this call's own insert attempt,
#   so classify_task_command_conflict can never observe the task gone
#   here. TASK_MISSING gets no test either.
#
# RACED_DUPLICATE needs a second writer to commit a row for the same
# (task_id, command_id) strictly between stage_task_command's own
# existing-row check and its own insert flush -- a real second connection
# genuinely open at that instant, not a pre-committed row (which either of
# respond()'s own two earlier existing-command checks, at step 5 and inside
# stage_task_command itself, would catch first and never reach the flush
# at all). SQLite cannot reproduce this: by the time respond() reaches
# step 8 it has already written the fence UPDATE (step 6) and the CAS
# (step 7) on this same connection, which already holds SQLite's one
# whole-database writer lock (the same fact
# `postgres_task_command_sessions`'s own docstring in
# test_task_command_transport.py documents for the sibling raced-insert
# test on enqueue_task_command) -- a second writer's commit cannot land
# inside that window until this transaction ends. PostgreSQL's row lock,
# held by step 2's `with_for_update` only on the specific `tasks` row,
# does not extend to the `task_execution_commands` table, so a second
# writer can complete there while this transaction is still open. Both of
# RACED_DUPLICATE's sub-branches (payload_matches True and False -- they
# produce different RespondOutcomes and only one increments the conflict
# counter) live in test_task_interaction_service_postgresql.py for that
# reason. What this file covers at this call site is the other door and
# the escape: the created=False result, whose payload_matches verdict this
# build decides on exactly like the IntegrityError door's, and the
# UNRELATED re-raise.
# ---------------------------------------------------------------------------


def test_respond_reraises_an_unrelated_staging_integrity_error_and_leaves_no_residue(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``IntegrityError`` with no duplicate row behind it and the task
    still present is what ``classify_task_command_conflict`` calls
    ``UNRELATED``: some constraint other than this call's own idempotency
    key failed. ``respond()`` must let it out rather than fold a
    database-level failure into ``OutcomeUnknown`` -- a caller that cannot
    tell "the database rejected this row" from "your answer was ambiguous"
    retries the first one forever (see respond()'s own docstring on what
    it lets escape). The zero-residue half is unchanged from the
    conservative sibling, which reported this same injected error as
    ``OutcomeUnknown``: the whole transaction still rolls back, and the
    conflict counter is still untouched, because nothing here was ever
    confirmed to be a conflict."""

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
        with pytest.raises(_IntegrityError):
            svc.respond(
                interaction_id=interaction_id,
                task_id=task_id,
                principal=principal,
                envelope=_respond_envelope(idempotency_key="staging-raises"),
            )

    assert _conflict_counter() == before_counter


def test_respond_reports_conflict_when_staging_finds_a_raced_row_with_another_payload(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``created=False`` staging result -- the row was already committed
    and visible by the time this call's own staging statement ran -- whose
    payload does not match this call's own envelope. Two different answers
    landed under one idempotency key, which is the same fact step 5's
    pre-read reports as ``idempotency_key_reused``; the only difference is
    that the losing writer's row became visible one statement later. The
    whole transaction rolls back, so this call's own fence UPDATE and CAS
    leave no residue, and the conflict counter moves because this build
    did confirm the conflict rather than guessing at it -- the
    conservative sibling reported ``OutcomeUnknown`` here and left the
    counter alone precisely because it could not."""

    from xagent.web.services.task_command_transport import StagedTaskCommand

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    def _racing_stage_task_command(*args: Any, **kwargs: Any) -> StagedTaskCommand:
        return StagedTaskCommand(
            staged_db_id=4242,
            client_command_id=kwargs.get("command_id", "staging-race"),
            created=False,
            payload_matches=False,
            status="pending",
        )

    monkeypatch.setattr(svc, "stage_task_command", _racing_stage_task_command)

    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        outcome = svc.respond(
            interaction_id=interaction_id,
            task_id=task_id,
            principal=principal,
            envelope=_respond_envelope(idempotency_key="staging-race"),
        )

        assert outcome == svc.RespondConflict(reason="idempotency_key_reused")
    assert _conflict_counter() == before_counter + 1


def test_respond_replays_when_staging_finds_a_raced_row_with_a_matching_payload(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same ``created=False`` door, with the winner's row carrying this
    call's own actor, kind and canonical payload. This build reports
    ``Replayed`` rather than the conservative sibling's ``OutcomeUnknown``,
    and the reversal is deliberate: that sibling's argument was that it
    could not tell a race from a replay, and ``payload_matches`` -- the
    same ``_matches_existing`` verdict ``classify_task_command_conflict``
    computes on the other door -- is exactly the fact it was missing. With
    the winner's row proven to carry this answer, the RESUME that executes
    is this answer, so this call commits its own fence UPDATE and CAS
    instead of rolling them back: the interaction row must end up answered
    and the task advanced, or the command that runs would be answering a
    row that never recorded an answer. The receipt names the winner's row,
    not one this call staged, which is why the outcome is ``Replayed`` and
    not ``Accepted``. The conflict counter must not move -- a confirmed
    replay is not a conflict."""

    from xagent.web.services.task_command_transport import StagedTaskCommand

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)
    before = _graph_snapshot(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    )

    def _racing_stage_task_command(*args: Any, **kwargs: Any) -> StagedTaskCommand:
        return StagedTaskCommand(
            staged_db_id=4242,
            client_command_id=kwargs.get("command_id", "staging-race"),
            created=False,
            payload_matches=True,
            status="pending",
        )

    monkeypatch.setattr(svc, "stage_task_command", _racing_stage_task_command)

    before_counter = _conflict_counter()
    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=_respond_envelope(idempotency_key="staging-race"),
    )

    assert isinstance(outcome, svc.RespondReplayed)
    # The winner's row id, taken straight from the staging result -- this
    # build does not re-query for it on this door.
    assert outcome.receipt.command_db_id == 4242
    assert outcome.receipt.responder_identity == principal.identity_string()
    assert _conflict_counter() == before_counter

    after = _graph_snapshot(_respond_db, task_id=task_id, interaction_id=interaction_id)
    assert before["ir_status"] == "active"
    assert after["ir_status"] == "answered"
    assert after["ir_responder_identity"] == principal.identity_string()
    assert after["task_state_version"] == before["task_state_version"] + 1
    assert after["task_control_state"] == TaskControlState.RESUME_REQUESTED.value


def test_respond_reports_replayed_when_the_replay_branch_commit_ack_is_lost_but_the_graph_landed(
    _respond_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay branch's own commit (the raced-duplicate door's ``Commit
    the fence UPDATE and the CAS`` statement) can lose its acknowledgment
    the same way step 9's does, and until this delivery it had no
    reconciliation at all -- a lost ack there fell straight through to a
    raised exception.

    ``stage_task_command`` is monkeypatched to report the same bare
    ``created=False, payload_matches=True`` race
    ``test_respond_replays_when_staging_finds_a_raced_row_with_a_matching_payload``
    injects -- no row backs it yet, because any row inserted from inside
    that mock would live inside the ``db.begin_nested()`` savepoint the
    real door-two branch unconditionally rolls back right after staging
    returns, and a genuinely concurrent second writer cannot land on
    SQLite here either: by the time ``respond()`` reaches step 8 it has
    already issued the fence UPDATE on this same connection, which
    already holds SQLite's one whole-database writer lock (see the
    module's own RACED_DUPLICATE commentary above -- the reason that
    race's real PostgreSQL coverage lives in a separate file). Instead,
    the winning row is inserted by the ``Session.commit`` monkeypatch
    itself, immediately before it defers to the real commit -- by that
    point door two's ``savepoint.rollback()`` has already run and the
    only savepoint standing between the row and the outer transaction is
    gone, so the row commits durably alongside this call's own fence
    UPDATE and CAS when the real commit underneath the injected failure
    actually runs. The underlying commit genuinely lands -- only the
    acknowledgment back to this process is lost -- so reconciliation must
    recover a receipt naming the winning row and report ``Replayed``, not
    ``Accepted``: the command that runs is the winner's, not the bare
    ``staged_db_id=4242`` this call's own staging mock returned."""

    from xagent.web.services.task_command_transport import StagedTaskCommand

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    idempotency_key = "replay-commit-lands"
    envelope = _respond_envelope(idempotency_key=idempotency_key)
    command_payload = svc._respond_command_payload(
        interaction_id=interaction_id, principal=principal, values=envelope.values
    )

    def _racing_stage_task_command(*args: Any, **kwargs: Any) -> StagedTaskCommand:
        return StagedTaskCommand(
            staged_db_id=4242,
            client_command_id=kwargs.get("command_id", idempotency_key),
            created=False,
            payload_matches=True,
            status="pending",
        )

    monkeypatch.setattr(svc, "stage_task_command", _racing_stage_task_command)

    from sqlalchemy.orm import Session as OrmSession

    original_commit = OrmSession.commit
    call_count = {"n": 0}
    winner: dict[str, int] = {}

    def _lost_ack_but_committed(self: Any) -> None:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Stand in for the other writer whose row this call's own
            # race doors detected: inserted here, outside any savepoint,
            # so it commits durably with the real commit below instead of
            # being wiped out by door two's savepoint.rollback().
            command = TaskExecutionCommand(
                task_id=task_id,
                actor_user_id=principal.user_id,
                actor_subject=_actor_subject(self, principal.user_id),
                command_id=idempotency_key,
                kind=svc.TaskCommandKind.RESUME.value,
                payload=command_payload,
                status="completed",
            )
            self.add(command)
            self.flush()
            winner["command_db_id"] = int(command.id)
            original_commit(self)
            raise RuntimeError("simulated lost commit acknowledgment")
        return original_commit(self)

    monkeypatch.setattr(OrmSession, "commit", _lost_ack_but_committed)
    monkeypatch.setattr(svc, "_RESPOND_DURABLE_GRAPH_RETRY_SLEEP_SECONDS", 0.0)

    outcome = svc.respond(
        interaction_id=interaction_id,
        task_id=task_id,
        principal=principal,
        envelope=envelope,
    )

    assert isinstance(outcome, svc.RespondReplayed)
    assert outcome.receipt.command_db_id == winner["command_db_id"]
    assert outcome.receipt.command_db_id != 4242
    assert outcome.receipt.responder_identity == principal.identity_string()

    after = _graph_snapshot(_respond_db, task_id=task_id, interaction_id=interaction_id)
    assert after["ir_status"] == "answered"
    assert after["ir_responder_identity"] == principal.identity_string()
    assert after["task_control_state"] == TaskControlState.RESUME_REQUESTED.value
    # Exactly one command row for this idempotency key -- the winner's
    # row, created by the mock itself -- not a second one this call's own
    # staging never actually inserted (door two's mock is a bare return
    # value).
    assert after["commands_count"] == 1


def test_respond_reports_outcome_unknown_when_the_replay_branch_commit_fails_and_the_graph_never_lands(
    _respond_db, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Same injected failure as the replay branch's ``Replayed`` sibling
    above, except this time the commit never reaches the database at all --
    standing in for a commit that genuinely failed rather than one whose
    acknowledgment alone was lost. With nothing landed, the durable-graph
    reconciliation has nothing to find on any of its three attempts, so the
    call reports ``OutcomeUnknown`` and leaves the fence UPDATE and CAS this
    call attempted with no residue -- same shape as step 9's own
    conservative sibling, now proven on the replay branch's door too."""

    from xagent.web.services.task_command_transport import StagedTaskCommand

    user_id, task_id = _waiting_task(_respond_db)
    interaction_id = _active_row_ready_for_respond(_respond_db, task_id=task_id)
    principal = _owning_principal(user_id)

    def _racing_stage_task_command(*args: Any, **kwargs: Any) -> StagedTaskCommand:
        return StagedTaskCommand(
            staged_db_id=4242,
            client_command_id=kwargs.get("command_id", "replay-commit-fails"),
            created=False,
            payload_matches=True,
            status="pending",
        )

    monkeypatch.setattr(svc, "stage_task_command", _racing_stage_task_command)

    from sqlalchemy.orm import Session as OrmSession

    call_count = {"n": 0}

    def _failing_commit(self: Any) -> None:
        call_count["n"] += 1
        raise RuntimeError("simulated lost commit acknowledgment")

    monkeypatch.setattr(OrmSession, "commit", _failing_commit)
    monkeypatch.setattr(svc, "_RESPOND_DURABLE_GRAPH_RETRY_SLEEP_SECONDS", 0.0)

    before_counter = _conflict_counter()
    with _asserts_no_side_effects(
        _respond_db, task_id=task_id, interaction_id=interaction_id
    ):
        with caplog.at_level(logging.WARNING):
            outcome = svc.respond(
                interaction_id=interaction_id,
                task_id=task_id,
                principal=principal,
                envelope=_respond_envelope(idempotency_key="replay-commit-fails"),
            )

        assert isinstance(outcome, svc.RespondOutcomeUnknown)
        matching = [
            record
            for record in caplog.records
            if "commit failed while answering" in record.getMessage()
        ]
        assert len(matching) == 1
        assert matching[0].levelno == logging.WARNING
    assert _conflict_counter() == before_counter
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# The mapping meta-test. For every one of the 20 (outcome, reason) pairs in
# the vocabulary, at least one test above must produce it. This is
# deliberately not an arithmetic comparison against the total cell count
# (see the module docstring's three-way division of labor table) -- several
# triggering conditions collapse onto the same pair (eight distinct
# "principal does not own this task" scenarios all produce
# ``not_task_principal``; four distinct "same idempotency key, different
# submitter" scenarios all produce ``idempotency_key_reused``; two
# distinct "the task is no longer waiting" scenarios both produce
# ``run_ended``), and one validation scenario is parametrized over two
# reasons on its own.
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
    any other arithmetic against the 38-cell count -- eight
    not_task_principal cells, four idempotency_key_reused cells and two
    run_ended cells legitimately collapse onto one pair each; this only
    asserts that no vocabulary pair is left with zero producing cells. Its
    blind spot: a new cell that produces no *new* pair -- for example a
    ninth not_task_principal scenario, or the guard-order overlap cell
    this build adds to ``run_ended`` -- adds nothing this scan would
    notice missing, so that gap is caught by review, not by this test."""

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


# ---------------------------------------------------------------------------
# InteractionPrincipalKind: the closed enum, its construction-time checks,
# and the system principal's identity rendering.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "user_id", "expect_constructs"),
    [
        pytest.param("user", None, True, id="user_none"),
        pytest.param("user", 7, True, id="user_positive"),
        pytest.param("user", 0, False, id="user_zero"),
        pytest.param("user", -1, False, id="user_negative"),
        pytest.param("user", True, False, id="user_bool_true"),
        pytest.param("user", "3", False, id="user_string"),
        pytest.param("guest", None, True, id="guest_none"),
        pytest.param("guest", 7, True, id="guest_positive"),
        pytest.param("guest", -1, False, id="guest_negative"),
        pytest.param("system", None, True, id="system_none"),
        pytest.param("system", 7, True, id="system_positive_still_valid_shape"),
        pytest.param("robot", None, False, id="unrecognized_kind"),
        pytest.param("", None, False, id="empty_kind"),
    ],
)
def test_i_a_1_kind_and_user_id_domain_at_construction(
    kind: Any, user_id: Any, expect_constructs: bool
) -> None:
    """Any kind outside the closed enum, or a user_id outside its
    domain (None, or a positive non-bool int), can never be constructed."""

    kwargs = dict(kind=kind, user_id=user_id, is_admin=False, auth_mode=None)
    if expect_constructs:
        principal = svc.InteractionPrincipal(**kwargs)
        assert isinstance(principal.kind, svc.InteractionPrincipalKind)
    else:
        with pytest.raises(ValueError):
            svc.InteractionPrincipal(**kwargs)


def test_i_a_1_mutation_deleting_kind_check_lets_garbage_through() -> None:
    """Variant of test_i_a_1 pinned directly against __post_init__: if the
    kind-coercion line were deleted, this construction would succeed
    instead of raising -- this is the mutation the matrix above is pinned
    against."""

    with pytest.raises(ValueError):
        svc.InteractionPrincipal(
            kind="not-a-real-kind", user_id=None, is_admin=False, auth_mode=None
        )


def test_i_a_2_identity_string_never_renders_a_nonexistent_person() -> None:
    """responder_identity never renders as user:None / user:True /
    guest: / an empty string, for any constructible principal -- including
    the system principal, whose rendering is a fixed, non-empty constant
    distinct from both namespaces."""

    system_principal = svc.InteractionPrincipal(
        kind=svc.InteractionPrincipalKind.SYSTEM,
        user_id=None,
        is_admin=False,
        auth_mode=None,
    )
    identity = system_principal.identity_string()
    assert identity
    assert "None" not in identity
    assert "True" not in identity
    assert not identity.startswith("user:")
    assert not identity.startswith("guest:")
    assert identity == "system:finalizer"


def test_identity_string_refuses_a_principal_built_around_post_init() -> None:
    """The defense-in-depth branch identity_string's own docstring names:
    "a frozen dataclass can still be produced by means that skip
    __post_init__ (object.__new__ plus direct attribute assignment, or a
    future subclass)". The kind coercion __post_init__ now does made the
    previous string-kind construction route illegal, and the tests that
    used to reach this branch that way (constructing with kind="robot"
    directly) were removed along with that route -- this test is the
    replacement, reaching the same branch the only way still possible:
    around the constructor entirely.

    Mutation: deleting the `self.kind != "user"` branch's raise turns this
    red -- the unrecognized kind would fall through to the user branch and
    render as `user:7`, exactly the audit corruption identity_string's
    docstring says this branch exists to stop."""

    smuggled = object.__new__(svc.InteractionPrincipal)
    object.__setattr__(smuggled, "kind", "robot")
    object.__setattr__(smuggled, "user_id", 7)
    object.__setattr__(smuggled, "is_admin", False)
    object.__setattr__(smuggled, "auth_mode", None)
    for optional in (
        "widget_agent_id",
        "widget_workforce_id",
        "share_agent_id",
        "share_workforce_id",
        "guest_id",
    ):
        object.__setattr__(smuggled, optional, None)

    with pytest.raises(ValueError, match="has no identity namespace"):
        smuggled.identity_string()


def test_i_a_3_system_principal_never_gets_authorized_through_ownership(
    _db: Session, _seeded_task: int
) -> None:
    """The system principal never reaches the write point by owning
    the task -- _assert_write_point_admissible is the only gate, and it
    checks kind alone, never user_id/ownership."""

    task = _db.query(Task).filter(Task.id == _seeded_task).first()
    system_principal_not_owning_anything = svc.InteractionPrincipal(
        kind=svc.InteractionPrincipalKind.SYSTEM,
        user_id=None,
        is_admin=False,
        auth_mode=None,
    )
    # No exception: _assert_write_point_admissible admits it regardless of
    # task ownership, because the system principal owns nothing to check.
    svc._assert_write_point_admissible(system_principal_not_owning_anything, task)


# ---------------------------------------------------------------------------
# _assert_write_point_admissible: the write-point fence, and its five test
# cells (system passes; user, admin, and guest are all refused the same way
# -- the fence checks kind, not ownership or the admin flag).
# ---------------------------------------------------------------------------


def test_write_point_admissible_system_principal_passes() -> None:
    principal = svc.InteractionPrincipal(
        kind=svc.InteractionPrincipalKind.SYSTEM,
        user_id=None,
        is_admin=False,
        auth_mode=None,
    )
    svc._assert_write_point_admissible(principal, Task(id=1))


def test_write_point_admissible_user_principal_is_refused() -> None:
    principal = svc.InteractionPrincipal(
        kind="user", user_id=7, is_admin=False, auth_mode=None
    )
    with pytest.raises(svc.InteractionWritePointUnfenced):
        svc._assert_write_point_admissible(principal, Task(id=1))


def test_write_point_admissible_admin_principal_is_refused_too() -> None:
    """The entire point of this definition: is_admin=True does not create a
    second way through. An admin user principal is refused exactly like a
    non-admin one."""

    principal = svc.InteractionPrincipal(
        kind="user", user_id=7, is_admin=True, auth_mode=None
    )
    with pytest.raises(svc.InteractionWritePointUnfenced):
        svc._assert_write_point_admissible(principal, Task(id=1))


def test_write_point_admissible_guest_principal_is_refused() -> None:
    principal = svc.InteractionPrincipal(
        kind="guest", user_id=7, is_admin=False, auth_mode="widget", guest_id="g-1"
    )
    with pytest.raises(svc.InteractionWritePointUnfenced):
        svc._assert_write_point_admissible(principal, Task(id=1))


def test_write_point_admissible_message_names_the_task_and_kind() -> None:
    principal = svc.InteractionPrincipal(
        kind="guest", user_id=7, is_admin=False, auth_mode="widget", guest_id="g-1"
    )
    with pytest.raises(svc.InteractionWritePointUnfenced) as excinfo:
        svc._assert_write_point_admissible(principal, Task(id=42))
    assert "42" in str(excinfo.value)


def test_write_point_admissible_static_call_is_not_conditional() -> None:
    """Static half of the write-point guard's own coverage: the call to
    _assert_write_point_admissible inside create() must not be reachable
    only through a branch -- pinned the same way the handoff-entry guard's
    static half is (test_create_validates_before_entering_the_handoff_static,
    test_interaction_handoff_production_surface.py)."""

    import ast
    import inspect
    import textwrap

    source = inspect.getsource(svc.create)
    tree = ast.parse(textwrap.dedent(source))
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    found_at_top_level = False
    for stmt in func_def.body:
        forbidden = [
            node
            for node in ast.walk(stmt)
            if isinstance(node, (ast.If, ast.For, ast.While))
        ]
        calls_here = [
            node
            for node in ast.walk(stmt)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_assert_write_point_admissible"
        ]
        if calls_here:
            assert not forbidden, (
                "the write-point assertion is reachable only conditionally"
            )
            found_at_top_level = True
    assert found_at_top_level


# ---------------------------------------------------------------------------
# The system-principal write path end to end: T3's zero-Task-query
# invariant, T2's three exception mappings, T4's event_id/BOM/unknown-key
# rules, and the terminal idempotency-key-normalization fix.
# ---------------------------------------------------------------------------


@pytest.fixture
def _system_call_ctx(_db: Session) -> dict[str, Any]:
    """A fully-loaded, lock-consistent context for the system-principal
    write path: a task whose lease_attempt_id matches the lease, a resume
    anchor backed by a real trace_events row, and a lease naming the same
    run the task and anchor agree on."""

    user_id = make_user(_db)
    task_id = make_task(_db, user_id=user_id)
    task = _db.query(Task).filter(Task.id == task_id).first()
    task.run_id = "run-a"
    task.source = "widget"
    task.lease_attempt_id = "attempt-1"
    _db.commit()
    _db.refresh(task)

    trace_event_id = _make_trace_event(_db, task_id=task_id)
    anchor = InteractionAnchor(
        trace_event_id=trace_event_id,
        resume_event_id=anchor_event_id(_db, trace_event_id),
        resume_execution_id="exec-1",
        resume_run_partition="run-a",
    )
    lease = TaskLease(
        task_id=task_id, runner_id="runner-1", run_id="run-a", attempt_id="attempt-1"
    )
    now = datetime.now(timezone.utc)
    expires_at = now + CLARIFICATION_REQUEST_TTL
    principal = svc.InteractionPrincipal(
        kind=svc.InteractionPrincipalKind.SYSTEM,
        user_id=None,
        is_admin=False,
        auth_mode=None,
    )
    return {
        "task": task,
        "anchor": anchor,
        "lease": lease,
        "now": now,
        "expires_at": expires_at,
        "principal": principal,
    }


def _system_create(
    db: Session, ctx: dict[str, Any], **envelope_overrides: Any
) -> svc.CreateOutcome:
    envelope = _valid_envelope(**envelope_overrides)
    return svc.create(
        db,
        task_id=int(ctx["task"].id),
        principal=ctx["principal"],
        envelope=envelope,
        system_context=svc.SystemWriteContext(
            task=ctx["task"],
            lease=ctx["lease"],
            anchor=ctx["anchor"],
            now=ctx["now"],
            expires_at=ctx["expires_at"],
        ),
    )


def test_system_write_context_rejects_a_blank_run_id(
    _system_call_ctx: dict[str, Any],
) -> None:
    """The lease preconditions are judged where the caller builds the
    context, not deep inside interaction_handoff. Before this, an empty
    run_id escaped create() as a bare ValueError from
    _validate_request_fields (task_interaction_staging.py), outside the
    typed-outcome contract entirely.

    Mutation: deleting the run_id branch of __post_init__ turns this red."""

    ctx = _system_call_ctx
    with pytest.raises(ValueError, match="non-empty str"):
        svc.SystemWriteContext(
            task=ctx["task"],
            lease=TaskLease(
                task_id=int(ctx["task"].id), runner_id="runner-1", run_id=""
            ),
            anchor=ctx["anchor"],
            now=ctx["now"],
            expires_at=ctx["expires_at"],
        )


@pytest.mark.parametrize(
    "run_id",
    [
        pytest.param(None, id="none"),
        pytest.param(12345, id="int"),
        pytest.param(b"run-a", id="bytes"),
    ],
)
def test_system_write_context_rejects_a_non_str_run_id(
    _system_call_ctx: dict[str, Any], run_id: Any
) -> None:
    """`isinstance(..., str)`, not just `is not None`: create()'s own
    comment claims every downstream use of context.lease.run_id is provably
    a str, and only this check makes that claim true. An int run_id
    previously passed create()'s `is None` precheck and failed far
    downstream with `run_id must be a str, got int`.

    Mutation: narrowing the check back to `is None` turns the int/bytes
    cases red."""

    ctx = _system_call_ctx
    with pytest.raises(ValueError, match="non-empty str"):
        svc.SystemWriteContext(
            task=ctx["task"],
            lease=TaskLease(
                task_id=int(ctx["task"].id), runner_id="runner-1", run_id=run_id
            ),
            anchor=ctx["anchor"],
            now=ctx["now"],
            expires_at=ctx["expires_at"],
        )


def test_system_write_context_rejects_a_lease_naming_another_task(
    _system_call_ctx: dict[str, Any],
) -> None:
    """Previously escaped create() as a bare ValueError raised inside
    InteractionHandoff.stage() ("lease names task N but this handoff was
    given task M") -- after the savepoint had already been opened.

    Mutation: deleting the task_id branch of __post_init__ turns this red."""

    ctx = _system_call_ctx
    with pytest.raises(ValueError, match="lease names task"):
        svc.SystemWriteContext(
            task=ctx["task"],
            lease=TaskLease(
                task_id=int(ctx["task"].id) + 1000,
                runner_id="runner-1",
                run_id="run-a",
            ),
            anchor=ctx["anchor"],
            now=ctx["now"],
            expires_at=ctx["expires_at"],
        )


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        pytest.param(
            "expires_at_naive", "expires_at must be an aware UTC", id="expires_at_naive"
        ),
        pytest.param(
            "expires_at_non_utc",
            "expires_at must be UTC",
            id="expires_at_non_utc",
        ),
        pytest.param(
            "expires_at_in_the_past",
            "expires_at must be after now",
            id="expires_at_past",
        ),
        pytest.param("now_naive", "now must be an aware UTC", id="now_naive"),
        pytest.param("now_non_utc", "now must be UTC", id="now_non_utc"),
    ],
)
def test_create_surfaces_staging_time_preconditions_as_valueerror(
    _db: Session, _system_call_ctx: dict[str, Any], bad: str, match: str
) -> None:
    """R-7's coverage gap, pinned as it stands rather than changed: these
    five remain bare ValueError by design (create()'s documented
    programming-error pattern, matching interaction_handoff's own
    _validate_request_fields). The test exists so the classification is a
    decision on record, not an untested accident -- if a later change turns
    any of them into a typed outcome, this is where it must be argued.

    Each also asserts zero persisted rows: a rejected precondition must
    leave nothing behind. SystemWriteContext.__post_init__ deliberately does
    not re-check now/expires_at (only the lease preconditions above), so
    each of these five still reaches _validate_request_fields, deep inside
    the with-block, unchanged by this delivery."""

    ctx = _system_call_ctx
    now = ctx["now"]
    expires_at = ctx["expires_at"]
    if bad == "expires_at_naive":
        expires_at = expires_at.replace(tzinfo=None)
    elif bad == "expires_at_non_utc":
        expires_at = expires_at.astimezone(timezone(timedelta(hours=1)))
    elif bad == "expires_at_in_the_past":
        expires_at = now - timedelta(hours=1)
    elif bad == "now_naive":
        now = now.replace(tzinfo=None)
    elif bad == "now_non_utc":
        now = now.astimezone(timezone(timedelta(hours=1)))

    with pytest.raises(ValueError, match=match):
        svc.create(
            _db,
            task_id=int(ctx["task"].id),
            principal=ctx["principal"],
            envelope=_valid_envelope(request_idempotency_key="sys-key-time-precond"),
            system_context=svc.SystemWriteContext(
                task=ctx["task"],
                lease=ctx["lease"],
                anchor=ctx["anchor"],
                now=now,
                expires_at=expires_at,
            ),
        )
    assert _db.query(TaskInteractionRequest).count() == 0


def test_system_principal_creates_a_fresh_row(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    outcome = _system_create(_db, _system_call_ctx, request_idempotency_key="sys-key-1")
    assert isinstance(outcome, svc.CreateCreated)
    assert outcome.receipt.task_id == int(_system_call_ctx["task"].id)
    assert outcome.receipt.run_id == "run-a"
    assert outcome.receipt.active_slot == 1
    assert _db.query(TaskInteractionRequest).count() == 1


def test_i_a_5_system_path_issues_no_task_query() -> None:
    """Static half: on the system-principal path, create() must
    never issue a Task query of its own. Walks the actual AST subtree of
    the `if principal.kind == InteractionPrincipalKind.SYSTEM:` branch's
    own body (its True side, never its `else`) and fails if any
    ``.query(...)`` call is reachable there at all -- not scoped to
    ``Task`` specifically, since the system branch has no legitimate
    reason to query anything through `db.query` at all."""

    import ast
    import inspect
    import textwrap

    source = inspect.getsource(svc.create)
    tree = ast.parse(textwrap.dedent(source))
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    def _is_system_branch_if(node: ast.AST) -> bool:
        if not isinstance(node, ast.If):
            return False
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and test.left.attr == "kind"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        ):
            return False
        (comparator,) = test.comparators
        return isinstance(comparator, ast.Attribute) and comparator.attr == "SYSTEM"

    system_if_nodes = [
        node for node in ast.walk(func_def) if _is_system_branch_if(node)
    ]
    assert len(system_if_nodes) == 1, (
        f"expected exactly one `principal.kind == ...SYSTEM` branch, found "
        f"{len(system_if_nodes)}"
    )
    system_if = system_if_nodes[0]

    # Walk only the True-branch statement list (system_if.body), never
    # system_if.orelse -- ast.walk over a bare list of statements requires
    # wrapping each one individually, since ast.walk expects a single node.
    query_calls_in_system_branch = [
        node
        for stmt in system_if.body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "query"
    ]
    assert query_calls_in_system_branch == [], (
        "a .query(...) call is reachable from the system-principal branch: "
        f"{ast.dump(query_calls_in_system_branch[0]) if query_calls_in_system_branch else ''}"
    )


def test_i_a_6_task_handed_to_handoff_is_the_caller_supplied_object(
    _db: Session, _system_call_ctx: dict[str, Any], _session_factory
) -> None:
    """Behavioral: the exact object create() passes to
    interaction_handoff is the caller's own task, never a re-derived one.

    Loads ``task`` through a second, independent session (not ``_db``, the
    one ``create()`` itself is given) before ever calling ``create()``.
    This is load-bearing, not incidental: SQLAlchemy's identity map means
    a same-session ``db.query(Task).filter(Task.id == task_id).first()``
    returns the *identical* Python object already loaded on that session,
    so an ``is`` comparison against a same-session task cannot tell "the
    caller-supplied object was used" apart from "the object was silently
    re-queried and happened to come back identical" -- a re-introduced
    query would pass such a check by accident. A second session has no
    such identity map to share, so only the genuinely caller-supplied
    object can satisfy the ``is`` check below."""

    ctx = _system_call_ctx
    second_session = _session_factory()
    try:
        cross_session_task = (
            second_session.query(Task).filter(Task.id == ctx["task"].id).first()
        )
        assert cross_session_task is not ctx["task"]  # confirms the setup itself

        real_interaction_handoff = svc.interaction_handoff
        seen: dict[str, Any] = {}

        def _spy(db: Any, lease: Any, *, task: Any, anchor: Any, now: Any) -> Any:
            seen["task"] = task
            return real_interaction_handoff(
                db, lease, task=task, anchor=anchor, now=now
            )

        with mock.patch(
            "xagent.web.services.task_interaction_service.interaction_handoff",
            side_effect=_spy,
        ):
            svc.create(
                _db,
                task_id=int(ctx["task"].id),
                principal=ctx["principal"],
                envelope=_valid_envelope(request_idempotency_key="sys-key-identity"),
                system_context=svc.SystemWriteContext(
                    task=cross_session_task,
                    lease=ctx["lease"],
                    anchor=ctx["anchor"],
                    now=ctx["now"],
                    expires_at=ctx["expires_at"],
                ),
            )
    finally:
        second_session.close()

    assert seen["task"] is cross_session_task
    assert seen["task"] is not ctx["task"]


def test_i_a_6_uncommitted_in_memory_task_state_is_what_gets_staged(
    _db: Session, _system_call_ctx: dict[str, Any], _session_factory
) -> None:
    """A second discriminator against the same mutation, independent of
    object identity: mutates ``task.source`` in memory only, on a
    second-session object, without ever committing it. Only the
    caller-supplied object carries this uncommitted value -- a fresh
    ``db.query(Task)`` on ``create()``'s own session would see the real,
    unmutated, committed row instead. ``interaction_handoff``'s own
    ``stage()`` derives the persisted row's ``origin`` from
    ``task.source``, so the row this test reads back afterward proves
    which object's data was actually used."""

    ctx = _system_call_ctx
    second_session = _session_factory()
    try:
        cross_session_task = (
            second_session.query(Task).filter(Task.id == ctx["task"].id).first()
        )
        cross_session_task.source = "sdk"  # in-memory only, never committed
        assert cross_session_task.source != ctx["task"].source

        outcome = svc.create(
            _db,
            task_id=int(ctx["task"].id),
            principal=ctx["principal"],
            envelope=_valid_envelope(request_idempotency_key="sys-key-uncommitted"),
            system_context=svc.SystemWriteContext(
                task=cross_session_task,
                lease=ctx["lease"],
                anchor=ctx["anchor"],
                now=ctx["now"],
                expires_at=ctx["expires_at"],
            ),
        )
    finally:
        second_session.close()

    assert isinstance(outcome, svc.CreateCreated)
    row = (
        _db.query(TaskInteractionRequest)
        .filter(TaskInteractionRequest.id == outcome.receipt.interaction_id)
        .first()
    )
    assert row.origin == "sdk"


def test_anchor_corrupt_maps_to_validation_rejected_invalid_values(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """InteractionAnchorCorrupt, swallowed by interaction_handoff and
    recorded on handoff.degraded_as, maps to
    CreateValidationRejected(invalid_values) -- never
    anchor_dangling/checkpoint_unavailable, which name the database-read
    half of anchor resolution this seam does not do."""

    ctx = _system_call_ctx
    corrupt_anchor = InteractionAnchor(
        trace_event_id=ctx["anchor"].trace_event_id,
        resume_event_id="",  # blank -> InteractionAnchorCorrupt
        resume_execution_id="exec-1",
        resume_run_partition="run-a",
    )
    outcome = svc.create(
        _db,
        task_id=int(ctx["task"].id),
        principal=ctx["principal"],
        envelope=_valid_envelope(request_idempotency_key="sys-key-anchor"),
        system_context=svc.SystemWriteContext(
            task=ctx["task"],
            lease=ctx["lease"],
            anchor=corrupt_anchor,
            now=ctx["now"],
            expires_at=ctx["expires_at"],
        ),
    )
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")
    assert _db.query(TaskInteractionRequest).count() == 0


def test_anchor_corrupt_mutation_wrong_reason_word() -> None:
    """Pin against the mutation of mapping InteractionAnchorCorrupt to
    anchor_dangling/checkpoint_unavailable instead: those two words are not
    in CREATE_OUTCOME_PRODUCIBLE_REASONS[CreateValidationRejected]."""

    assert (
        "anchor_dangling"
        not in svc.CREATE_OUTCOME_PRODUCIBLE_REASONS[svc.CreateValidationRejected]
    )
    assert (
        "invalid_values"
        in svc.CREATE_OUTCOME_PRODUCIBLE_REASONS[svc.CreateValidationRejected]
    )


def test_run_partition_mismatch_maps_to_stale_anchor_run_mismatch(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """InteractionRunPartitionMismatch, raised by
    stage_interaction_request's own validation when lease.run_id does not
    match anchor.resume_run_partition, swallowed by interaction_handoff and
    recorded on handoff.degraded_as, maps to
    CreateStale(anchor_run_mismatch)."""

    ctx = _system_call_ctx
    mismatched_lease = TaskLease(
        task_id=int(ctx["task"].id),
        runner_id="runner-1",
        run_id="run-b",
        attempt_id="attempt-1",
    )
    outcome = svc.create(
        _db,
        task_id=int(ctx["task"].id),
        principal=ctx["principal"],
        envelope=_valid_envelope(request_idempotency_key="sys-key-mismatch"),
        system_context=svc.SystemWriteContext(
            task=ctx["task"],
            lease=mismatched_lease,
            anchor=ctx["anchor"],
            now=ctx["now"],
            expires_at=ctx["expires_at"],
        ),
    )
    assert outcome == svc.CreateStale(reason="anchor_run_mismatch")
    assert _db.query(TaskInteractionRequest).count() == 0


def test_degraded_as_subclass_maps_to_the_parents_outcome(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """A future subclass of a mapped swallowed type must classify as its
    parent does, not fall through to the slot_taken default.
    InteractionRunPartitionMismatch is the discriminating choice: its
    outcome (CreateStale) is the one the default can never produce, so an
    exact-type lookup's failure is visible in the outcome itself.

    Mutation: restoring `_DEGRADED_AS_OUTCOME.get(handoff.degraded_as)`
    turns this red with CreateConflict(slot_taken)."""

    from xagent.web.services import task_interaction_staging as staging_module
    from xagent.web.services.task_interaction_staging import (
        InteractionRunPartitionMismatch,
    )

    class _FutureRunPartitionMismatch(InteractionRunPartitionMismatch):
        pass

    ctx = _system_call_ctx
    real_stage = staging_module.stage_interaction_request

    def _raise_subclass(*args: Any, **kwargs: Any) -> Any:
        raise _FutureRunPartitionMismatch("forced subclass degradation")

    with mock.patch.object(
        staging_module, "stage_interaction_request", side_effect=_raise_subclass
    ):
        outcome = _system_create(_db, ctx, request_idempotency_key="sys-key-subclass")

    assert outcome == svc.CreateStale(reason="anchor_run_mismatch")
    assert real_stage is staging_module.stage_interaction_request
    assert _db.query(TaskInteractionRequest).count() == 0


def test_swallowed_exception_types_are_mutually_unrelated() -> None:
    """The linear issubclass scan in create()'s degraded_as handling is
    order-independent only while no swallowed type subclasses another.
    Pins that premise so a future hierarchy change surfaces here rather
    than as a misclassification."""

    from xagent.web.services import task_interaction_staging as staging_module

    types = staging_module._SWALLOWED
    for first in types:
        for second in types:
            if first is not second:
                assert not issubclass(first, second), (
                    f"{first.__name__} subclasses {second.__name__}; the "
                    "issubclass scan in create() is no longer order-independent"
                )


def test_closed_idempotency_key_maps_to_conflict_reused(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """A request_idempotency_key already naming a closed
    (answered/terminated) row on this run maps to
    CreateConflict(idempotency_key_reused). stage_interaction_request's own
    step-3 pre-read raises InteractionRequestClosed inside the handoff;
    interaction_handoff swallows it and records
    handoff.degraded_as = InteractionRequestClosed, which create() reads
    after the `with` block and maps to this outcome -- no separate
    pre-check of its own runs ahead of the handoff."""

    ctx = _system_call_ctx
    first = _system_create(_db, ctx, request_idempotency_key="sys-key-closed")
    assert isinstance(first, svc.CreateCreated)
    row = (
        _db.query(TaskInteractionRequest)
        .filter(TaskInteractionRequest.id == first.receipt.interaction_id)
        .first()
    )
    row.status = "terminated"
    row.active_slot = None
    row.terminated_at = ctx["now"]
    row.terminal_reason = "deadline_elapsed"
    _db.commit()

    outcome = _system_create(_db, ctx, request_idempotency_key="sys-key-closed")
    assert outcome == svc.CreateConflict(reason="idempotency_key_reused")
    # No second row: the closed key on this run refuses the write, it does
    # not reclaim or replace the terminated row.
    assert _db.query(TaskInteractionRequest).count() == 1


def test_active_replay_returns_create_created_not_a_distinct_outcome(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """A replay hit whose row is still active is not
    preempted -- stage_interaction_request's own step-3 pre-read returns
    it (created=False), and create() reports CreateCreated the same as a
    fresh insert, never a distinct outcome."""

    ctx = _system_call_ctx
    first = _system_create(_db, ctx, request_idempotency_key="sys-key-replay")
    second = _system_create(_db, ctx, request_idempotency_key="sys-key-replay")
    assert isinstance(first, svc.CreateCreated)
    assert isinstance(second, svc.CreateCreated)
    assert first.receipt.interaction_id == second.receipt.interaction_id
    assert _db.query(TaskInteractionRequest).count() == 1


def test_slot_taken_maps_to_conflict_slot_taken(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """InteractionSlotTaken, swallowed by interaction_handoff and recorded
    on handoff.degraded_as, maps to CreateConflict(slot_taken). Forced
    directly (a genuine INSERT race is not reproducible in a single-threaded
    test) by making the staging primitive's own INSERT step raise it, the
    same way it would after losing a real race against a concurrent
    session's INSERT."""

    from xagent.web.services import task_interaction_staging as staging_module

    ctx = _system_call_ctx
    real_stage = staging_module.stage_interaction_request

    def _stage_and_raise_slot_taken(*args: Any, **kwargs: Any) -> Any:
        raise staging_module.InteractionSlotTaken(
            "forced for test_slot_taken_maps_to_conflict_slot_taken"
        )

    with mock.patch.object(
        staging_module,
        "stage_interaction_request",
        side_effect=_stage_and_raise_slot_taken,
    ):
        outcome = _system_create(_db, ctx, request_idempotency_key="sys-key-slot-taken")
    assert outcome == svc.CreateConflict(reason="slot_taken")
    assert real_stage is staging_module.stage_interaction_request  # patch released
    assert _db.query(TaskInteractionRequest).count() == 0


def test_handoff_degraded_as_is_set_directly_on_a_slot_taken_swallow(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """Unlike the outcome-level test above, this checks
    InteractionHandoff.degraded_as itself, at the staging primitive's own
    layer -- discriminating power the outcome-level test lacks for this one
    exception, since CreateConflict(slot_taken) is also this function's
    fallback for an unset/unrecognized degraded_as, so an outcome-only
    check cannot tell "correctly mapped" from "fell through to the
    default". Deleting the `handoff.degraded_as = type(exc)` assignment in
    interaction_handoff's except block turns this test red without
    changing the outcome-level test's result at all."""

    from xagent.web.services.task_interaction_staging import InteractionSlotTaken

    ctx = _system_call_ctx
    with svc.interaction_handoff(
        _db, ctx["lease"], task=ctx["task"], anchor=ctx["anchor"], now=ctx["now"]
    ) as handoff:
        raise InteractionSlotTaken("forced for degraded_as inspection")
    assert handoff.staged is None
    assert handoff.degraded_as is InteractionSlotTaken


def test_receipt_on_replay_reports_the_existing_rows_expires_at(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """The receipt-widening format cell this test exists for: on replay the
    receipt must report the EXISTING row's
    expires_at, not the value this call proposed. Forces the two apart by
    advancing `now`/`expires_at` between the two calls."""

    ctx = _system_call_ctx
    first = _system_create(_db, ctx, request_idempotency_key="sys-key-expiry")
    # No tzinfo workaround any more: the replay path normalizes the stored
    # value back to aware UTC (task_interaction_staging._replay_or_raise_closed),
    # so both paths hand back the same shape and these compare directly.
    first_expires_at = first.receipt.expires_at

    later_ctx = dict(ctx)
    later_ctx["now"] = ctx["now"] + timedelta(hours=1)
    later_ctx["expires_at"] = ctx["expires_at"] + timedelta(hours=1)
    second = _system_create(_db, later_ctx, request_idempotency_key="sys-key-expiry")

    assert isinstance(second, svc.CreateCreated)
    assert second.receipt.expires_at == first_expires_at
    assert second.receipt.expires_at != later_ctx["expires_at"]


def test_receipt_on_fresh_insert_carries_the_proposed_expires_at(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """The mirror of the replay cell above: a fresh insert's receipt
    reports the value this call proposed."""

    ctx = _system_call_ctx
    outcome = _system_create(_db, ctx, request_idempotency_key="sys-key-fresh-expiry")
    assert outcome.receipt.expires_at == ctx["expires_at"]


def test_receipt_expires_at_is_aware_utc_on_both_paths(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """The format invariant behind the two value cells above: a consumer
    doing `receipt.expires_at > datetime.now(timezone.utc)` must not hit a
    TypeError on one path and succeed on the other.

    Mutation: reverting to `expires_at=row.expires_at` turns the replay
    half red (tzinfo is None -> the comparison raises TypeError)."""

    ctx = _system_call_ctx
    fresh = _system_create(_db, ctx, request_idempotency_key="sys-key-tz")
    replay = _system_create(_db, ctx, request_idempotency_key="sys-key-tz")

    for label, outcome in (("fresh", fresh), ("replay", replay)):
        assert isinstance(outcome, svc.CreateCreated), label
        expires_at = outcome.receipt.expires_at
        assert expires_at.tzinfo is not None, label
        assert expires_at.utcoffset() == timedelta(0), label
        # The arithmetic the reviewer named: must not raise on either path.
        assert expires_at > datetime.now(timezone.utc), label


def _leading_keyword(statement: str) -> str:
    """The statement's own leading SQL keyword, uppercased -- a stable,
    parameter-independent shape marker for the assertions below (bound
    values and column lists change; the keyword and the statement's
    position in the sequence do not)."""

    stripped = statement.strip()
    first_word = stripped.split(None, 1)[0].upper()
    if stripped.upper().startswith("RELEASE SAVEPOINT"):
        return "RELEASE SAVEPOINT"
    return first_word


def _capture_statement_sequence(
    db: Session, ctx: dict[str, Any], key: str
) -> list[str]:
    """Every statement create() issues (through interaction_handoff and
    stage_interaction_request) for one call, as leading-keyword shapes, in
    the order they actually ran. Touches every ctx["task"] attribute
    create() itself reads before attaching the hook, so a lazy-reload
    SQLAlchemy issues for an object this test's own fixture setup expired
    (via an unrelated commit) is not mistaken for a statement create()
    itself is responsible for."""

    task = ctx["task"]
    _ = (task.id, task.user_id, task.source, task.run_id, task.lease_attempt_id)

    statements: list[str] = []

    def _before_cursor_execute(
        conn, cursor, statement, parameters, context, executemany
    ):
        statements.append(statement)

    engine = db.get_bind()
    sa.event.listen(engine, "before_cursor_execute", _before_cursor_execute)
    try:
        _system_create(db, ctx, request_idempotency_key=key)
    finally:
        sa.event.remove(engine, "before_cursor_execute", _before_cursor_execute)
    return [_leading_keyword(s) for s in statements]


def test_statement_sequence_is_unchanged_by_the_receipt_widening(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """Widening _identity_lookup_stmt's column list (this delivery's own
    change) adds no new statement and reorders nothing: a fresh insert
    still runs exactly this sequence -- the SQLite-only dummy UPDATE, the
    outer savepoint interaction_handoff opens, the identity SELECT
    stage_interaction_request's step 3 runs (now widened, same statement),
    the reclaim UPDATE (step 4), the inner savepoint
    stage_interaction_request opens for its own INSERT (step 5), the
    INSERT itself, and the two savepoints releasing in reverse order.
    Asserted by leading-keyword shape, in order -- not just a count --
    against a real, captured sequence, not an estimate."""

    sequence = _capture_statement_sequence(
        _db, _system_call_ctx, "sys-key-stmt-sequence-fresh"
    )
    assert sequence == [
        "UPDATE",  # SQLite-only dummy DML (interaction_handoff)
        "SAVEPOINT",  # outer savepoint (interaction_handoff)
        "SELECT",  # identity pre-read, step 3 (stage_interaction_request)
        "UPDATE",  # reclaim stale/superseded slot, step 4
        "SAVEPOINT",  # inner savepoint, step 5
        "INSERT",  # the new row, step 5
        "RELEASE SAVEPOINT",  # inner savepoint commits
        "RELEASE SAVEPOINT",  # outer savepoint commits
    ]


def test_statement_sequence_on_replay_is_unchanged_by_the_receipt_widening(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """The replay path's own sequence, shorter than the fresh-insert one
    because it returns from the identity pre-read directly, without ever
    opening the inner savepoint an INSERT would need."""

    ctx = _system_call_ctx
    _system_create(_db, ctx, request_idempotency_key="sys-key-stmt-sequence-replay")
    sequence = _capture_statement_sequence(_db, ctx, "sys-key-stmt-sequence-replay")
    assert sequence == [
        "UPDATE",  # SQLite-only dummy DML (interaction_handoff)
        "SAVEPOINT",  # outer savepoint (interaction_handoff)
        "SELECT",  # identity pre-read, step 3 -- hits the active row
        "RELEASE SAVEPOINT",  # outer savepoint commits; no insert attempted
    ]


def test_event_id_and_metadata_keys_are_not_rejected_as_unknown(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """event_id, and the other keys build_clarification_payload
    may add beyond message/interactions, must never be treated as unknown
    fields -- rejecting event_id specifically would refuse every
    legitimate payload, since every real caller carries it."""

    ctx = _system_call_ctx
    values = {
        "event_id": "evt-legit-1",
        "message": "Which environment?",
        "interactions": [
            {"type": "text_input", "field": "env", "label": "Environment"}
        ],
        "message_type": "question",
        "source": "widget",
        "requests": [],
    }
    outcome = _system_create(
        _db, ctx, request_idempotency_key="sys-key-metadata", values=values
    )
    assert isinstance(outcome, svc.CreateCreated)


def test_unrecognized_envelope_key_is_rejected(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    ctx = _system_call_ctx
    values = {
        "message": "Which environment?",
        "interactions": [
            {"type": "text_input", "field": "env", "label": "Environment"}
        ],
        "totally_unrecognized_key": "x",
    }
    outcome = _system_create(
        _db, ctx, request_idempotency_key="sys-key-unknown", values=values
    )
    assert outcome == svc.CreateValidationRejected(reason="invalid_values")


def test_unrecognized_envelope_key_mutation_removing_the_check() -> None:
    """Pin against deleting _reject_unknown_envelope_keys's call site: a
    payload carrying a bogus top-level key must be caught before
    parse_v1_request_payload silently drops it."""

    with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
        svc._reject_unknown_envelope_keys(
            {"message": "x", "interactions": [], "bogus": 1}
        )
    assert excinfo.value.refusal.rule == "unknown_field"
    assert excinfo.value.refusal.position == "request_payload"


def test_unknown_envelope_key_never_reaches_the_refusal_or_the_log(
    _db: Session, _system_call_ctx: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """InteractionWriteRefusal promises position never carries a value out
    of the payload (#1314 item 3), and this refusal is logged at WARNING.
    A key chosen to be unmistakable if it leaked -- and to be a log-injection
    payload if it reached an unsanitized %s -- must appear in neither.

    Mutation: restoring ``position=f"request_payload.{...}"`` turns both
    assertions below red."""

    ctx = _system_call_ctx
    hostile_key = "leaked\nWARNING forged log line\r\n" + "A" * 500
    values = {
        "message": "Which environment?",
        "interactions": [
            {"type": "text_input", "field": "env", "label": "Environment"}
        ],
        hostile_key: "x",
    }
    with caplog.at_level(logging.WARNING):
        outcome = _system_create(
            _db, ctx, request_idempotency_key="sys-key-hostile", values=values
        )

    assert outcome == svc.CreateValidationRejected(reason="invalid_values")
    assert "rule=unknown_field" in caplog.text
    assert "position=request_payload" in caplog.text
    assert "leaked" not in caplog.text
    assert "forged log line" not in caplog.text


def test_unknown_envelope_key_with_mixed_key_types_does_not_raise() -> None:
    """The TypeError guard the previous `sorted(..., key=str)` provided is
    now structural: no ordering is computed at all. A payload whose
    top-level keys mix str and int must still produce the refusal, not a
    TypeError.

    Mutation: reintroducing any `sorted(...)` over the unknown key set
    without `key=str` turns this red."""

    with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
        svc._reject_unknown_envelope_keys(
            {"message": "x", "interactions": [], "bogus": 1, 7: "int key"}
        )
    assert excinfo.value.refusal.rule == "unknown_field"
    assert excinfo.value.refusal.position == "request_payload"


@pytest.mark.parametrize(
    "field",
    [
        pytest.param("\ufeff", id="bom_only"),
        pytest.param("\ufeffenv", id="bom_prefixed"),
        pytest.param("env\ufeff", id="bom_suffixed"),
        pytest.param(" \t\ufeff ", id="mixed_ascii_and_bom"),
    ],
)
def test_field_bom_forms_are_rejected_or_pinned_to_the_shared_table(
    field: str,
) -> None:
    """field-name blankness/whitespace judgment must use the same
    30-code-point table react.py's producer uses, not bare str.strip()
    (which does not treat U+FEFF as whitespace). A BOM-only field is
    blank; a BOM-affixed non-blank field carries "surrounding whitespace"
    by this table's definition, both refused -- neither would have been
    caught by the old plain str.strip() check."""

    values = _values([_interaction(field=field)])
    parsed = svc.parse_v1_request_payload(values)
    with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
        svc.validate_v1_write_payload(parsed)
    assert excinfo.value.refusal.rule in ("field_blank", "field_whitespace")


def test_option_label_bom_only_is_rejected_strip_aware() -> None:
    """#1314 comment 3: option label/value emptiness must be strip-aware
    against the same table, not a bare falsy check -- a BOM-only label
    used to pass the old `not option.label` test."""

    values = _values(
        [
            _interaction(
                type="select_one",
                options=[{"label": "\ufeff", "value": "a"}],
            )
        ]
    )
    parsed = svc.parse_v1_request_payload(values)
    with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
        svc.validate_v1_write_payload(parsed)
    assert excinfo.value.refusal.rule == "option_blank"


def test_bom_mutation_reverting_to_str_strip_lets_bom_only_field_through() -> None:
    """Direct pin on the shared-table requirement: str.strip() alone does
    not treat U+FEFF as whitespace, so a mutation reverting
    _normalize_interaction_text to str.strip() would let this BOM-only
    field pass where the shared table refuses it."""

    assert "\ufeff".strip() != ""  # str.strip() alone: NOT blank
    from xagent.core.agent.pattern.react.react import _normalize_interaction_text

    assert _normalize_interaction_text("\ufeff") == ""  # shared table: blank


def test_message_bom_only_is_blank() -> None:
    values = _values([_interaction()])
    values["message"] = "\ufeff"
    parsed = svc.parse_v1_request_payload(values)
    with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
        svc.validate_v1_write_payload(parsed)
    assert excinfo.value.refusal.rule == "message_blank"


# ---------------------------------------------------------------------------
# Input-space exhaustion for min/max/default_value/accept, crossed against
# all seven interaction types and a set of bad shapes -- the deliverable
# table itself, not just its passing tests. Every cell states its own
# judgment (accept, or reject under a named rule) so a reader does not have
# to re-derive it from the assertion alone.
#
# min/max are InteractionArg's own Optional[int] fields -- not strings, so
# the string bad-shapes (empty/whitespace/BOM) that apply to field/label/
# option values do not apply here at all; a caller cannot even construct a
# BOM-shaped int. What varies instead is the numeric relationship between
# the two: inverted (min > max), equal, and each set alone. Today's rule
# (validate_v1_write_payload) rejects only an inverted pair, and only on
# number_input -- the one type that reads the pair at all (see that
# function's own docstring on the three deliberately-accepted shapes). On
# every other type the pair is an unread hint; an inverted one there is
# still accepted, because nothing renders it.
#
# default_value is Union[str, bool, int, float, None] -- the four string
# bad-shapes apply when it is a string. No rule anywhere checks
# default_value's content against anything (not its own type, not the
# interaction's type, not blankness) -- it is a UI-only pre-fill hint the
# renderer either uses or ignores, and the answer-side field schema (#1368)
# is what would eventually validate a submitted answer, not this seam.
# Every cell here is therefore an "accept" cell; the judgment is "no rule
# exists to fire", stated once here rather than per cell.
#
# accept is Optional[list[str]], meaningful only to file_upload's renderer
# (clarification-form.tsx passes it straight to the file input's own
# `accept` attribute) -- the model does not restrict which type may carry
# it, and neither does this validator. The four bad shapes apply to a list
# element. No rule checks accept's element content either (an empty or
# BOM-only extension string reaches the browser's own accept attribute
# unfiltered, which is a rendering-quality question, not a write-integrity
# one this seam is responsible for) -- every cell here is an "accept" cell
# for the same reason default_value's are.
# ---------------------------------------------------------------------------


def _base_interaction_for_type(
    interaction_type: str, **overrides: Any
) -> dict[str, Any]:
    item = _interaction(type=interaction_type, **overrides)
    if (
        interaction_type in svc._V1_TYPES_REQUIRING_OPTIONS
        and "options" not in overrides
    ):
        item["options"] = [{"label": "Option A", "value": "a"}]
    return item


_ALL_SEVEN_TYPES = sorted(svc._V1_INTERACTION_TYPES)

_MIN_MAX_CELLS = [
    (interaction_type, min_value, max_value)
    for interaction_type in _ALL_SEVEN_TYPES
    for min_value, max_value in [
        (5, 3),  # inverted
        (5, 5),  # equal
        (5, None),  # min alone
        (None, 5),  # max alone
    ]
]


@pytest.mark.parametrize(
    ("interaction_type", "min_value", "max_value"),
    _MIN_MAX_CELLS,
    ids=[f"{t}-min={mn}-max={mx}" for t, mn, mx in _MIN_MAX_CELLS],
)
def test_min_max_matrix(
    interaction_type: str, min_value: int | None, max_value: int | None
) -> None:
    values = _values(
        [_base_interaction_for_type(interaction_type, min=min_value, max=max_value)]
    )
    parsed = svc.parse_v1_request_payload(values)
    inverted = min_value is not None and max_value is not None and min_value > max_value
    if interaction_type == "number_input" and inverted:
        with pytest.raises(svc.InteractionWritePayloadRejected) as excinfo:
            svc.validate_v1_write_payload(parsed)
        assert excinfo.value.refusal.rule == "number_range_inverted"
    else:
        svc.validate_v1_write_payload(parsed)  # must not raise


_BAD_STRING_SHAPES = [
    ("empty", ""),
    ("whitespace", "   "),
    ("bom_only", "\ufeff"),
    ("bom_wrapped", "\ufeffvalue\ufeff"),
]

_DEFAULT_VALUE_CELLS = [
    (interaction_type, shape_id, shape_value)
    for interaction_type in _ALL_SEVEN_TYPES
    for shape_id, shape_value in _BAD_STRING_SHAPES
]


@pytest.mark.parametrize(
    ("interaction_type", "shape_id", "shape_value"),
    _DEFAULT_VALUE_CELLS,
    ids=[f"{t}-{sid}" for t, sid, _ in _DEFAULT_VALUE_CELLS],
)
def test_default_value_matrix(
    interaction_type: str, shape_id: str, shape_value: str
) -> None:
    values = _values(
        [_base_interaction_for_type(interaction_type, default_value=shape_value)]
    )
    parsed = svc.parse_v1_request_payload(values)
    svc.validate_v1_write_payload(parsed)  # must not raise -- see module note above


@pytest.mark.parametrize(
    "default_value",
    [True, False, 0, 1, 3.5, None],
    ids=["bool_true", "bool_false", "int_zero", "int_one", "float", "none"],
)
def test_default_value_non_string_shapes_are_accepted(default_value: Any) -> None:
    values = _values([_interaction(default_value=default_value)])
    parsed = svc.parse_v1_request_payload(values)
    svc.validate_v1_write_payload(parsed)  # must not raise


_ACCEPT_LIST_SHAPES = [
    ("empty_list", []),
    ("single_empty_element", [""]),
    ("single_whitespace_element", ["   "]),
    ("single_bom_only_element", ["\ufeff"]),
    ("bom_wrapped_extension", ["\ufeff.csv"]),
    ("mixed_good_and_bad", [".csv", ""]),
]

_ACCEPT_CELLS = [
    (interaction_type, shape_id, shape_value)
    for interaction_type in _ALL_SEVEN_TYPES
    for shape_id, shape_value in _ACCEPT_LIST_SHAPES
]


@pytest.mark.parametrize(
    ("interaction_type", "shape_id", "shape_value"),
    _ACCEPT_CELLS,
    ids=[f"{t}-{sid}" for t, sid, _ in _ACCEPT_CELLS],
)
def test_accept_matrix(
    interaction_type: str, shape_id: str, shape_value: list[str]
) -> None:
    values = _values([_base_interaction_for_type(interaction_type, accept=shape_value)])
    parsed = svc.parse_v1_request_payload(values)
    svc.validate_v1_write_payload(parsed)  # must not raise -- see module note above


def test_receipt_idempotency_key_is_the_normalized_form_not_the_envelope_original(
    _db: Session, _system_call_ctx: dict[str, Any]
) -> None:
    """The receipt's request_idempotency_key must be
    _normalize_command_id's return value, never the envelope's original
    string -- a key padded with surrounding whitespace must come back
    identical, byte for byte, to what is actually stored."""

    ctx = _system_call_ctx
    padded_key = "  sys-key-padded  "
    outcome = _system_create(_db, ctx, request_idempotency_key=padded_key)
    assert isinstance(outcome, svc.CreateCreated)
    assert outcome.receipt.request_idempotency_key != padded_key

    row = (
        _db.query(TaskInteractionRequest)
        .filter(TaskInteractionRequest.id == outcome.receipt.interaction_id)
        .first()
    )
    assert outcome.receipt.request_idempotency_key == row.request_idempotency_key


def test_receipt_idempotency_key_mutation_using_envelope_original() -> None:
    """Direct pin: _normalize_command_id's return value must not be
    discarded. Confirms the normalizer itself actually changes a padded
    key (proving the mutation -- filling the receipt from
    envelope.request_idempotency_key instead -- would be observable)."""

    from xagent.web.services.task_command_transport import _normalize_command_id

    padded = "  sys-key-padded  "
    normalized = _normalize_command_id(padded)
    assert normalized != padded
