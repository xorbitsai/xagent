"""PostgreSQL concurrency coverage for the preferences update lock.

Mirrors test_runtime_key_transition_postgres.py's pattern for the other
dual-dialect fence in this codebase (api/api_keys.py's
acquire_runtime_key_transition_fence): a real row-level lock is only
provable against a real PostgreSQL server, since SQLite's writer
serialization is a whole-database-file guarantee rather than a per-row
``FOR UPDATE`` lock.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from xagent.web.api import auth as auth_api
from xagent.web.models import database as database_module
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.user import User


@pytest.fixture()
def postgres_user():
    """An isolated user row in a real PostgreSQL test database."""
    import os

    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    # init_db rebinds the module-global engine/session factory - save and
    # restore in finally, matching test_auth_api.py's test_db fixture,
    # or this leaks into whichever test runs next in the same process.
    previous_engine = database_module._engine
    previous_session_local = database_module._SessionLocal
    init_db(db_url=url)
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    SessionLocal = get_session_local()
    try:
        with SessionLocal() as db:
            user = User(
                username="preferences-lock-owner",
                password_hash="hash",
                preferences={"department": "Sales"},
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = int(user.id)
        yield SessionLocal, user_id
    finally:
        Base.metadata.drop_all(bind=engine)
        database_module._engine = previous_engine
        database_module._SessionLocal = previous_session_local


def test_concurrent_disjoint_field_updates_both_survive(postgres_user) -> None:
    """Two sessions PATCHing disjoint fields must not lose either one:
    the second session's lock acquisition blocks until the first
    commits, then it merges on top of the first's already-persisted
    change rather than a stale read.

    The second side calls _merge_user_preferences_locked directly - the
    actual function the PATCH endpoint runs through
    run_db_io_cancellation_safe - rather than hand-rolling an equivalent
    lock+read+merge+commit sequence, so a regression in that function's
    own ordering (e.g. reading before the lock, or moving the fetch
    off the locked session) fails this test instead of leaving it green
    against a mirror that no longer matches production."""
    SessionLocal, user_id = postgres_user

    first_locked = threading.Event()
    release_first = threading.Event()
    first_committed = threading.Event()
    second_returned = threading.Event()

    def run_first_update() -> None:
        # Hand-rolled, not the production function: this side only needs
        # to hold the row lock open on a controlled schedule, to prove
        # the second call - which does go through the real production
        # path below - actually blocks behind it.
        with SessionLocal() as db:
            auth_api._lock_user_row_for_preferences_update(db, user_id)
            first_locked.set()
            assert release_first.wait(timeout=10)
            user = db.get(User, user_id)
            preferences = dict(user.preferences or {})
            preferences["industry"] = "Real estate"
            user.preferences = preferences
            db.commit()
        first_committed.set()

    def run_second_update() -> None:
        assert first_locked.wait(timeout=10)
        result = auth_api._merge_user_preferences_locked(user_id, {"voice": "warm"})
        assert result is not None
        second_returned.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_first_update)
        assert first_locked.wait(timeout=10)
        second = executor.submit(run_second_update)
        # The second call's lock attempt must block behind the first
        # session's still-open transaction - the whole call can't return
        # (lock, fetch, merge, and commit are all fast once acquired)
        # until the first session releases the lock.
        assert not second_returned.wait(timeout=0.2)
        release_first.set()
        first.result(timeout=10)
        second.result(timeout=10)

    assert first_committed.is_set()
    assert second_returned.is_set()
    with SessionLocal() as db:
        user = db.get(User, user_id)
        assert user.preferences == {
            "department": "Sales",
            "industry": "Real estate",
            "voice": "warm",
        }
