"""Protocol marker pairing after a finalizer publishes, on PostgreSQL.

Each of the three finalizers that settle a waiting round publishes a
native clarification with the rollout policy allowing it; a fresh session
then checks that the task's protocol marker and the run's active
interaction row are present together and agree
(``assert_marker_residue`` and ``assert_interaction_marker_invariant``).
The round set-up and the finalizer calls are the SQLite suite's
(``test_finalizer_publication.py``); the channel finalizer runs as the
coordinator that owns its lease.

Skipped unless ``XAGENT_TEST_POSTGRES_URL`` is set.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_session_factory
from tests.web.services.interaction_marker_shared import (
    assert_interaction_marker_invariant,
    assert_marker_residue,
)
from tests.web.services.test_finalizer_publication import (
    FINALIZERS,
    SECOND_EVENT_ID,
    _finalize,
    _native,
    _prepare,
    _resume_same_run,
    _rows,
    _second_question_result,
    _seed_round,
    _waiting_result,
)
from xagent.web.models.task import Task
from xagent.web.models.task_interaction import TaskInteractionRequest
from xagent.web.services.task_interaction_close import (
    close_legacy_resume_interaction,
    close_legacy_resume_interaction_sync,
)

pytestmark = pytest.mark.postgresql


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch) -> Iterator[sessionmaker[Session]]:
    with disposable_session_factory("xagent_3b5_", monkeypatch) as factory:
        yield factory


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_finalizer_publication_pairs_the_marker_with_its_row(
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    finalizer: str,
) -> None:
    _native(monkeypatch)
    round_ = _seed_round()
    _prepare(finalizer, round_)

    _finalize(finalizer, round_, _waiting_result())

    with sessions() as observer:
        assert_marker_residue(
            observer, task_id=round_.task_id, run_id=round_.lease.run_id
        )
        assert_interaction_marker_invariant(observer, task_id=round_.task_id)
        rows = (
            observer.query(TaskInteractionRequest.status)
            .filter(TaskInteractionRequest.task_id == round_.task_id)
            .all()
        )
        assert [status for (status,) in rows] == ["active"]
        assert observer.get(Task, round_.task_id).interaction_protocol_version == 1


@pytest.mark.parametrize("finalizer", FINALIZERS)
def test_the_next_question_retires_the_first_on_postgresql(
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    finalizer: str,
) -> None:
    """A message that read no question leaves the first one active; the
    resumed run's next published question retires it under the table's
    CHECK constraints, and the marker still pairs with the one active row."""
    _native(monkeypatch)
    first = _seed_round()
    _prepare(finalizer, first)
    _finalize(finalizer, first, _waiting_result())

    resumed = _resume_same_run(first, close_id=None)
    _prepare("resumed", resumed)
    _finalize("resumed", resumed, _second_question_result(resumed))

    with sessions() as observer:
        rows = (
            observer.query(
                TaskInteractionRequest.status,
                TaskInteractionRequest.terminal_reason,
                TaskInteractionRequest.request_idempotency_key,
                TaskInteractionRequest.active_slot,
            )
            .filter(TaskInteractionRequest.task_id == first.task_id)
            .order_by(TaskInteractionRequest.id)
            .all()
        )
        assert [(status, reason, slot) for status, reason, _, slot in rows] == [
            ("terminated", "run_superseded", None),
            ("active", None, 1),
        ]
        assert rows[1][2] == SECOND_EVENT_ID
        assert_marker_residue(
            observer, task_id=first.task_id, run_id=first.lease.run_id
        )
        assert_interaction_marker_invariant(observer, task_id=first.task_id)


def test_a_close_that_lost_to_the_next_question_on_postgresql(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The close bound to the first question's id runs after the next
    question has retired that row: it matches nothing and leaves the new
    question and the marker alone."""
    _native(monkeypatch)
    first = _seed_round()
    _prepare("fresh", first)
    _finalize("fresh", first, _waiting_result())
    ((first_id, _, _, _),) = _rows(first.task_id)

    resumed = _resume_same_run(first, close_id=None)
    _prepare("resumed", resumed)
    _finalize("resumed", resumed, _second_question_result(resumed))
    rowcount = close_legacy_resume_interaction_sync(
        task_id=first.task_id, run_id=str(first.lease.run_id), interaction_id=first_id
    )

    assert rowcount == 0
    rows = _rows(first.task_id)
    assert [(status, reason) for _, status, reason, _ in rows] == [
        ("terminated", "run_superseded"),
        ("active", None),
    ]
    with sessions() as observer:
        assert_marker_residue(
            observer, task_id=first.task_id, run_id=first.lease.run_id
        )
        assert_interaction_marker_invariant(observer, task_id=first.task_id)


def test_a_committed_close_lets_the_blocked_publication_find_nothing_to_retire(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The close holds the task row's lock when the next question's
    publication arrives; the publication waits. Once the close commits, the
    first question is already answered, the publication's reclaim finds
    nothing, and the new question becomes the active row."""
    _native(monkeypatch)
    first = _seed_round()
    _prepare("fresh", first)
    _finalize("fresh", first, _waiting_result())
    ((first_id, _, _, _),) = _rows(first.task_id)
    resumed = _resume_same_run(first, close_id=None)
    _prepare("resumed", resumed)
    result = _second_question_result(resumed)

    failures: list[BaseException] = []

    def publish() -> None:
        try:
            _finalize("resumed", resumed, result)
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)

    holder = sessions()
    worker = threading.Thread(target=publish)
    try:
        holder.execute(
            sa.select(Task.id)
            .where(Task.id == first.task_id)
            .with_for_update(key_share=True)
        ).all()
        close_legacy_resume_interaction(
            holder,
            task_id=first.task_id,
            run_id=str(first.lease.run_id),
            interaction_id=first_id,
        )
        worker.start()
        with sessions() as probe:
            for _ in range(100):
                waiting = probe.execute(
                    sa.text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE wait_event_type = 'Lock' "
                        "AND datname = current_database()"
                    )
                ).scalar_one()
                if waiting >= 1:
                    break
                worker.join(0.1)
            else:
                pytest.fail("the publication never waited on the task row lock")
        holder.commit()
    finally:
        holder.close()
        worker.join(30)
    assert not worker.is_alive()
    assert failures == []

    rows = _rows(first.task_id)
    assert [(status, reason) for _, status, reason, _ in rows] == [
        ("terminated", "answered_via_legacy_resume"),
        ("active", None),
    ]
    with sessions() as observer:
        assert observer.get(Task, first.task_id).interaction_protocol_version == 1
        assert_marker_residue(
            observer, task_id=first.task_id, run_id=first.lease.run_id
        )
