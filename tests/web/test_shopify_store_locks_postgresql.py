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


def test_concurrent_refreshes_in_one_event_loop_take_turns(shopify_pg, monkeypatch):
    """The second resolution waits for the store off the event loop, so the
    first one's refresh can finish and the second uses its token."""
    db, user = shopify_pg
    shop._held_grant(db, user, expires_in=-timedelta(minutes=1))
    posted: list[str] = []

    class SlowAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, **kwargs):
            posted.append(url)
            await asyncio.sleep(0.2)
            return shop._Response(
                {
                    "access_token": "rotated-access",
                    "refresh_token": "rotated-refresh",
                    "expires_in": 3600,
                }
            )

    monkeypatch.setattr(tool_config, "create_public_async_client", SlowAsyncClient)
    factory = sessionmaker(bind=db.get_bind(), autoflush=False, autocommit=False)

    async def resolve_twice():
        config = tool_config.WebToolConfig(
            db=None, request=None, db_factory=factory, user_id=user.id
        )
        return await asyncio.wait_for(
            asyncio.gather(
                *(
                    config._resolve_legacy_oauth_access_token(
                        provider_name="shopify", app_id=shop.TEST_APP_ID
                    )
                    for _ in range(2)
                )
            ),
            timeout=10,
        )

    resolutions = asyncio.run(resolve_twice())

    assert [r.access_token for r in resolutions] == ["rotated-access"] * 2
    assert len(posted) == 1


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
