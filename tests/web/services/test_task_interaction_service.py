"""Unit coverage for ``task_interaction_service``: the outcome vocabularies,
the ``create()`` typed seam, and the compatibility materialization view.

This module accumulates coverage across every deliverable this service
ships except the shared public-chat ownership predicate (covered directly
by ``tests/web/api/test_public_chat_ownership_helper.py``, since it is
extracted from and tested alongside ``public_chat_access.py``) and the
production-caller gate (its own file,
``test_task_interaction_service_production_gate.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from tests.web.services.task_interaction_schema_shared import make_task, make_user
from xagent.db.sqlite import apply_sqlite_concurrency_pragmas
from xagent.web.models.database import Base
from xagent.web.models.task import Task
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services import task_interaction_service as svc

# ---------------------------------------------------------------------------
# Vocabulary guards (three numbers, pinned by the frozen design's own
# line-by-line count -- do not recompute them here):
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


def test_create_outcome_this_period_covers_exactly_the_six_producible_variants() -> (
    None
):
    """CreateCreated, CreateConflict, and CreateStale are not producible
    until the wiring batch's ignition PR fills create()'s call body --
    create() never stages a row in this delivery, so nothing that requires
    a staged row can be returned yet."""

    assert set(svc.CREATE_OUTCOME_REASON_VOCABULARY) == {
        "CreateValidationRejected",
        "CreateUnauthorized",
        "CreateUnavailable",
        "CreateNotWired",
    }


def test_locator_mismatch_reason_constant_does_not_exist_in_source() -> None:
    """A design revision voided the reason 'locator_mismatch' before this
    delivery began. Asserting its absence guards against it surviving as a
    dead string constant that would mislead a future reader into thinking
    that path is still live."""

    import inspect

    source = inspect.getsource(svc)
    assert "locator_mismatch" not in source


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
        return make_task(db, user_id=user_id)
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
