"""Shopify per-store locking on PostgreSQL (#2810).

SQLite has one writer for the whole database, so the SQLite tests in
test_shopify_oauth.py cannot tell a lock on one store from a lock on every
store, and never meet PostgreSQL's row locks. These run the same flows on
PostgreSQL.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import timedelta

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from tests.web import test_shopify_oauth as shop
from xagent.web import oauth_account_hosts
from xagent.web.api import auth as auth_api
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services.shopify_store_locks import lock_shopify_store
from xagent.web.tools import config as tool_config

pytestmark = pytest.mark.postgresql


@pytest.fixture
def shopify_pg(monkeypatch) -> Iterator[tuple[Session, User]]:
    monkeypatch.setattr(
        oauth_account_hosts.socket,
        "getaddrinfo",
        shop._resolve_to(shop.PUBLIC_ADDRESS),
    )
    # The shared factory skips only when this explicit release-gate URL is absent.
    with disposable_database_factory("xagent_shopify_store") as make:
        engine = make("locks")

        # A regression that waits for a lock on the event loop would
        # otherwise hang the test instead of failing it.
        @event.listens_for(engine, "connect")
        def bound_lock_waits(dbapi_connection, _record):
            with dbapi_connection.cursor() as cursor:
                cursor.execute("SET lock_timeout = '5s'")

        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
        with factory() as db:
            yield db, shop.seed_shopify_oauth(db)
            db.rollback()


def _lock_is_free(db: Session, store: str) -> bool:
    """Whether another connection could lock ``store`` right now."""
    with sessionmaker(bind=db.get_bind())() as other:
        other.execute(text("SET lock_timeout = '300ms'"))
        try:
            lock_shopify_store(other, provider="shopify", store=store)
        except OperationalError:
            return False
        finally:
            other.rollback()
        return True


def test_a_store_lock_holds_only_that_store(shopify_pg):
    """The first lock of a store inserts its row; a second connection waits
    for that insert to commit, while other stores stay free."""
    db, _user = shopify_pg

    lock_shopify_store(db, provider="Shopify", store="Acme")

    assert _lock_is_free(db, "acme") is False
    assert _lock_is_free(db, "other") is True
    db.commit()
    assert _lock_is_free(db, "acme") is True


def test_the_callback_holds_its_store_through_the_code_exchange(
    shopify_pg, monkeypatch
):
    db, user = shopify_pg
    shop._mock_shopify_http(monkeypatch)
    fake_post = auth_api.requests.post
    during_exchange: list[tuple[bool, bool]] = []

    def exchange_while_probing(url, **kwargs):
        if url == shop.TOKEN_URL:
            during_exchange.append(
                (_lock_is_free(db, "acme"), _lock_is_free(db, "other"))
            )
        return fake_post(url, **kwargs)

    monkeypatch.setattr(auth_api.requests, "post", exchange_while_probing)

    response = shop._callback(db, shop._login_state(db, user))

    assert response.status_code == 200
    assert during_exchange == [(False, True)]
    assert _lock_is_free(db, "acme") is True


def test_two_owners_refreshing_one_store_take_turns_off_the_event_loop(
    shopify_pg, monkeypatch
):
    """The second owner waits for the store lock off the event loop, so the
    first one's refresh finishes and the loop keeps running meanwhile."""
    db, user = shopify_pg

    resolutions, most_in_flight, ticks = shop.refresh_two_owners_at_once(
        db, user, monkeypatch, timeout=10
    )

    assert [r.access_token for r in resolutions] == ["rotated-access"] * 2
    assert most_in_flight == 1
    assert ticks >= 20


def test_an_actor_refresh_locks_the_store_before_its_grant_row(shopify_pg, monkeypatch):
    """The callback locks the store, then replaces the actor's grant row; a
    refresh taking them the other way round could deadlock with it."""
    db, user = shopify_pg
    shop._held_grant(
        db, user, resource_owner_key="toby:w1:u1", expires_in=-timedelta(minutes=1)
    )
    posted = shop._fake_refresh_client(monkeypatch)
    statements: list[str] = []

    def record(conn, cursor, statement, *args):
        statements.append(" ".join(statement.split()).lower())

    engine = db.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        config = tool_config.WebToolConfig(
            db=None,
            request=None,
            db_factory=sessionmaker(bind=engine, autoflush=False),
            user_id=user.id,
        )
        resolution = asyncio.run(
            config._resolve_legacy_oauth_access_token(
                provider_name="shopify",
                app_id=shop.TEST_APP_ID,
                resource_owner_key="toby:w1:u1",
            )
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert resolution.access_token == "rotated-access"
    assert len(posted) == 1
    store_lock = next(
        i for i, s in enumerate(statements) if "into shopify_store_locks" in s
    )
    row_lock = next(
        i
        for i, s in enumerate(statements)
        if "from user_oauth" in s and s.endswith("for update")
    )
    assert store_lock < row_lock


def test_an_actor_refresh_whose_grant_moved_store_never_locks_a_store_after_it(
    shopify_pg, monkeypatch
):
    """The actor reconnects to another store between the refresh reading the
    store and locking the grant row. Locking the new store with the row held
    could deadlock with that store's callback; the refresh starts over."""
    from xagent.web.services import shopify_store_locks

    db, user = shopify_pg
    grant = shop._held_grant(
        db, user, resource_owner_key="toby:w1:u1", expires_in=-timedelta(minutes=1)
    )
    grant_id = grant.id
    posted = shop._fake_refresh_client(monkeypatch)
    real_lock = shopify_store_locks.lock_shopify_store
    moved: list[bool] = []

    def lock_after_a_move(session, **kwargs):
        if not moved:
            moved.append(True)
            with sessionmaker(bind=db.get_bind())() as other:
                other.get(UserOAuth, grant_id).instance_url = "beta"
                other.commit()
        return real_lock(session, **kwargs)

    monkeypatch.setattr(shopify_store_locks, "lock_shopify_store", lock_after_a_move)
    events, remove = shop._record_lock_order(db.get_bind())
    try:
        resolution = shop._resolve_actor_shopify(db, user.id)
    finally:
        remove()

    shop._assert_no_new_store_lock_after_a_row_lock(events)
    assert resolution.access_token == "rotated-access"
    assert len(posted) == 1
