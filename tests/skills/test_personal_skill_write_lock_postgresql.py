"""Real-PostgreSQL coverage for the row lock personal-skill edits and deletes take.

``FOR UPDATE`` is a no-op on SQLite, so no SQLite-backed test can tell a
second writer that is genuinely held back from a lock statement that does
nothing. This file runs the real statements on two real connections: a delete
that has locked the row but not yet committed, and a second writer -- an edit,
or another delete -- that arrives in that window. The second writer has to
wait for the delete, and once the delete has committed it has to find nothing
and answer 404 instead of writing to, or deleting, a row that no longer
exists.

The database comes from ``disposable_database_factory``, which reads
``XAGENT_TEST_POSTGRES_URL`` and skips the module when it is unset.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Callable

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.api.skill_hub import (
    _delete_personal_skill,
    _update_personal_skill_md,
    _write_personal_skill,
)
from xagent.web.models.database import Base
from xagent.web.models.skill import UserSkill, UserSkillFile
from xagent.web.models.user import User

pytestmark = pytest.mark.postgresql

SKILL_MD = b"# Alpha\n\n## Description\nA test skill.\n"


@pytest.fixture()
def session_factory():
    with disposable_database_factory("xagent_personal_skill_lock") as make_database:
        engine = make_database("skill_lock")
        Base.metadata.create_all(
            bind=engine,
            tables=[User.__table__, UserSkill.__table__, UserSkillFile.__table__],
        )
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def owner_id(session_factory) -> int:
    """One owner holding one committed personal skill named ``alpha``."""
    with session_factory() as db:
        owner = User(username="skill-lock-owner", password_hash="x", is_admin=False)
        db.add(owner)
        db.flush()
        owner_row_id = int(owner.id)
        _write_personal_skill(
            db=db, user=owner, name="alpha", files={"SKILL.md": SKILL_MD}
        )
    return owner_row_id


def _second_writer_against_a_held_delete(
    session_factory,
    owner_id: int,
    second_writer: Callable[[Session, SimpleNamespace], None],
) -> HTTPException | None:
    """Run ``second_writer`` on its own connection while a delete of ``alpha``
    holds the row lock and has not committed.

    Asserts the second writer is still waiting a second later, then lets the
    delete commit and returns the ``HTTPException`` the second writer raised,
    or ``None`` if it returned normally.
    """
    owner = SimpleNamespace(id=owner_id)
    delete_holds_the_row = threading.Event()
    release_the_delete = threading.Event()
    second_finished = threading.Event()

    delete_session = session_factory()
    second_session = session_factory()
    real_commit = delete_session.commit

    def paced_commit() -> None:
        # Reached after the delete's own lock statement and before its commit:
        # the row is locked and the delete is still open.
        delete_holds_the_row.set()
        assert release_the_delete.wait(timeout=DB_PROGRESS_TIMEOUT), (
            "the delete was never released"
        )
        real_commit()

    delete_session.commit = paced_commit  # type: ignore[method-assign]

    def run_delete() -> None:
        _delete_personal_skill(db=delete_session, user=owner, name="alpha")

    def run_second() -> HTTPException | None:
        try:
            second_writer(second_session, owner)
            return None
        except HTTPException as exc:
            return exc
        finally:
            second_finished.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            delete = executor.submit(run_delete)
            assert delete_holds_the_row.wait(timeout=DB_PROGRESS_TIMEOUT), (
                "the delete never reached its commit"
            )

            second = executor.submit(run_second)
            try:
                assert not second_finished.wait(timeout=1.0), (
                    "the second writer finished while the delete still held "
                    "the row -- the lock did not block it"
                )
            finally:
                # Released even when the assertion fails, so a failure does not
                # also spend the delete's full wait before it is reported.
                release_the_delete.set()
            delete.result(timeout=DB_PROGRESS_TIMEOUT)
            return second.result(timeout=DB_PROGRESS_TIMEOUT)
    finally:
        delete_session.close()
        second_session.close()


def _assert_the_skill_is_gone(session_factory) -> None:
    with session_factory() as fresh:
        assert fresh.query(UserSkill).count() == 0
        assert fresh.query(UserSkillFile).count() == 0


def test_an_edit_waits_for_a_concurrent_delete_and_then_finds_nothing(
    session_factory, owner_id
) -> None:
    outcome = _second_writer_against_a_held_delete(
        session_factory,
        owner_id,
        lambda db, owner: _update_personal_skill_md(
            db=db, user=owner, name="alpha", skill_md="# Alpha\n\nEdited.\n"
        ),
    )

    assert isinstance(outcome, HTTPException), (
        "the edit should have found the deleted skill gone"
    )
    assert outcome.status_code == 404
    _assert_the_skill_is_gone(session_factory)


def test_a_second_delete_waits_for_the_first_and_then_finds_nothing(
    session_factory, owner_id
) -> None:
    outcome = _second_writer_against_a_held_delete(
        session_factory,
        owner_id,
        lambda db, owner: _delete_personal_skill(db=db, user=owner, name="alpha"),
    )

    assert isinstance(outcome, HTTPException), (
        "the second delete should have found the skill gone"
    )
    assert outcome.status_code == 404
    _assert_the_skill_is_gone(session_factory)
