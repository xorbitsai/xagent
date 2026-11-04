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

from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_session_factory
from tests.web.services.interaction_marker_shared import (
    assert_interaction_marker_invariant,
    assert_marker_residue,
)
from tests.web.services.test_finalizer_publication import (
    FINALIZERS,
    _finalize,
    _native,
    _prepare,
    _seed_round,
    _waiting_result,
)
from xagent.web.models.task import Task
from xagent.web.models.task_interaction import TaskInteractionRequest

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
