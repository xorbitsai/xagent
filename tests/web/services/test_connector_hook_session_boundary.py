"""The root-transaction-end count reads the same on both backends.

The count comes from a ``Session``-level SQLAlchemy event
(``after_transaction_end``) and never goes through a dialect, so agreement
between backends is a property of the mechanism rather than something that
needs re-proving on every driver -- but the connector hook seam
(``services/connector_team_scope.py``) refuses requests on this count, so a
default-backend (SQLite) deployment and a PostgreSQL one must not disagree
about which hook broke the contract.

The PostgreSQL half is marker-gated: the main CI legs deselect
``-m postgresql`` (see ``ci.yml``), so this file is also registered as a
step in ``.github/workflows/test-migrations.yml``, which is the only place
that runs it with ``-m postgresql``. The SQLite half carries no marker and
runs in the ordinary suite.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web.models import Base, User
from xagent.web.models.database import (
    release_db_connection_if_clean,
    root_transaction_end_count,
)


def _sqlite_session_factory() -> tuple[sessionmaker[Session], Engine]:
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine), engine


@pytest.fixture(
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgresql", id="postgresql", marks=pytest.mark.postgresql),
    ]
)
def session_factory(request: pytest.FixtureRequest) -> Iterator[sessionmaker[Session]]:
    if request.param == "sqlite":
        factory, engine = _sqlite_session_factory()
        try:
            yield factory
        finally:
            engine.dispose()
        return
    with disposable_database_factory("xagent_hook_session_boundary") as make_database:
        engine = make_database("boundary")
        Base.metadata.create_all(bind=engine)
        yield sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _open_transaction(db: Session) -> None:
    """Put ``db`` in a root transaction without writing anything, matching
    ``test_connector_team_scope.py``'s helper of the same name -- a cell
    that says "in a transaction on entry" must not depend on unrelated
    setup calls to get there."""
    db.execute(select(1))


def _noop(db: Session) -> None:
    pass


def _one_orm_query(db: Session) -> None:
    db.query(User).all()


def _rollback(db: Session) -> None:
    db.rollback()


def _rollback_then_text(db: Session) -> None:
    db.rollback()
    db.execute(text("select 1"))


def _rollback_then_own_write(db: Session) -> None:
    db.rollback()
    db.add(User(username="t14-rollback-then-write", password_hash="x"))
    db.flush()


def _commit(db: Session) -> None:
    db.commit()


def _close(db: Session) -> None:
    db.close()


def _savepoint_commit(db: Session) -> None:
    nested = db.begin_nested()
    db.add(User(username="t14-savepoint-commit", password_hash="x"))
    db.flush()
    nested.commit()


def _savepoint_rollback(db: Session) -> None:
    nested = db.begin_nested()
    db.add(User(username="t14-savepoint-rollback", password_hash="x"))
    db.flush()
    nested.rollback()


def _write_and_flush(db: Session) -> None:
    db.add(User(username="t14-write-and-flush", password_hash="x"))
    db.flush()


def _read_then_release(db: Session) -> None:
    # Entering with no transaction open, a bare read starts one implicitly.
    # ``release_db_connection_if_clean`` rolls that transaction back because
    # nothing was written, ending a transaction the caller never opened.
    db.query(User).all()
    release_db_connection_if_clean(db)


def _rollback_twice(db: Session) -> None:
    db.rollback()
    db.rollback()


# (id, hook body, enter in a transaction, expected count delta)
_GRID: list[tuple[str, Callable[[Session], None], bool, int]] = [
    ("noop", _noop, True, 0),
    ("one-orm-query", _one_orm_query, True, 0),
    ("rollback", _rollback, True, 1),
    ("rollback-then-text", _rollback_then_text, True, 1),
    ("rollback-then-own-write", _rollback_then_own_write, True, 1),
    ("commit", _commit, True, 1),
    ("close", _close, True, 1),
    ("savepoint-commit", _savepoint_commit, True, 0),
    ("savepoint-rollback", _savepoint_rollback, True, 0),
    ("write-and-flush", _write_and_flush, True, 0),
    ("no-txn-on-entry-rollback", _rollback, False, 0),
    ("no-txn-on-entry-read-then-release", _read_then_release, False, 1),
    ("rollback-twice-not-double-counted", _rollback_twice, True, 1),
]


@pytest.mark.parametrize(
    "hook_body,enter_in_transaction,expected_delta",
    [pytest.param(body, txn, delta, id=name) for name, body, txn, delta in _GRID],
)
def test_the_counter_reads_the_same_on_both_backends(
    session_factory: sessionmaker[Session],
    hook_body: Callable[[Session], None],
    enter_in_transaction: bool,
    expected_delta: int,
) -> None:
    db = session_factory()
    try:
        if enter_in_transaction:
            _open_transaction(db)
        before = root_transaction_end_count(db)
        hook_body(db)
        after = root_transaction_end_count(db)
        assert after is not None and before is not None
        assert after - before == expected_delta
    finally:
        db.close()


def test_two_calls_in_one_session_are_counted_one_at_a_time(
    session_factory: sessionmaker[Session],
) -> None:
    """The grid above exercises one call per fresh session. This is the
    accumulation case: a second call's delta must be measured against the
    count left by the first, not against a session-wide zero -- the same
    property the seam's own ``test_two_hook_calls_in_one_request_are_
    compared_one_at_a_time`` (``test_connector_team_scope.py``) pins for the
    full gate, checked here at the counter's own level on both backends."""
    db = session_factory()
    try:
        _open_transaction(db)
        first_before = root_transaction_end_count(db)
        _noop(db)
        first_after = root_transaction_end_count(db)
        assert first_after == first_before

        _open_transaction(db)
        second_before = root_transaction_end_count(db)
        _rollback(db)
        second_after = root_transaction_end_count(db)
        assert second_after == second_before + 1
    finally:
        db.close()


def test_a_duck_typed_object_reports_no_count_on_either_backend() -> None:
    """Backend-independent by construction -- there is no session at all --
    kept in this file because it is part of the same parity claim: neither
    backend's session shape changes what ``root_transaction_end_count``
    does with an object that has no ``.info`` mapping."""

    class _Duck:
        pass

    assert root_transaction_end_count(_Duck()) is None
