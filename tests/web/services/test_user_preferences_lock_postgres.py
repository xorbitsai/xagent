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

from xagent.web.api.auth import _lock_user_row_for_preferences_update
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.user import User


@pytest.fixture()
def postgres_user():
    """An isolated user row in a real PostgreSQL test database."""
    import os

    url = os.getenv("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
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


def test_concurrent_disjoint_field_updates_both_survive(postgres_user) -> None:
    """Two sessions PATCHing disjoint fields must not lose either one:
    the second session's lock acquisition blocks until the first
    commits, then it merges on top of the first's already-persisted
    change rather than a stale read."""
    SessionLocal, user_id = postgres_user

    first_locked = threading.Event()
    release_first = threading.Event()
    first_committed = threading.Event()
    second_locked = threading.Event()

    def run_first_update() -> None:
        with SessionLocal() as db:
            _lock_user_row_for_preferences_update(db, user_id)
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
        with SessionLocal() as db:
            _lock_user_row_for_preferences_update(db, user_id)
            second_locked.set()
            user = db.get(User, user_id)
            preferences = dict(user.preferences or {})
            preferences["voice"] = "warm"
            user.preferences = preferences
            db.commit()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_first_update)
        assert first_locked.wait(timeout=10)
        second = executor.submit(run_second_update)
        # The second session's lock attempt must block behind the first
        # session's still-open transaction.
        assert not second_locked.wait(timeout=0.2)
        release_first.set()
        first.result(timeout=10)
        second.result(timeout=10)

    assert first_committed.is_set()
    with SessionLocal() as db:
        user = db.get(User, user_id)
        assert user.preferences == {
            "department": "Sales",
            "industry": "Real estate",
            "voice": "warm",
        }
