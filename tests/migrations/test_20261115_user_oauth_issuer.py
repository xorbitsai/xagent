"""Tests for migration 20261115_user_oauth_issuer.

The pre-migration shape of ``user_oauth`` and the catalog tables the
backfill reads is built directly, then only the migration under test runs
against it. Every backfill case runs on SQLite and, under
``@pytest.mark.postgresql``, on a real PostgreSQL server in a throwaway
schema, so the JSON audit snapshots are decoded by each dialect.
"""

from __future__ import annotations

import importlib.util
import os
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

TARGET_REVISION = "20261115_user_oauth_issuer"


def _migration_module() -> ModuleType:
    import xagent.migrations as migrations_pkg

    migrations_dir = Path(next(iter(migrations_pkg.__path__)))
    path = migrations_dir / "versions" / f"{TARGET_REVISION}.py"
    spec = importlib.util.spec_from_file_location(TARGET_REVISION, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


migration = _migration_module()


@pytest.fixture
def sqlite_engine(tmp_path: Path) -> Iterator[sa.Engine]:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    yield engine
    engine.dispose()


@pytest.fixture
def postgresql_engine() -> Iterator[sa.Engine]:
    """An engine whose default schema is a fresh, dropped-afterwards schema,
    so the shared test database's own ``user_oauth`` is never touched."""
    url = os.environ.get("XAGENT_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    schema = f"oauth_issuer_{uuid.uuid4().hex[:12]}"
    admin = sa.create_engine(url)
    with admin.begin() as conn:
        conn.execute(sa.text(f"CREATE SCHEMA {schema}"))
    engine = sa.create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(sa.text(f"DROP SCHEMA {schema} CASCADE"))
        admin.dispose()


@pytest.fixture(
    params=[
        "sqlite",
        pytest.param("postgresql", marks=pytest.mark.postgresql),
    ]
)
def engine(request: pytest.FixtureRequest) -> sa.Engine:
    return request.getfixturevalue(f"{request.param}_engine")


T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _at(hours: int | None) -> datetime | None:
    return None if hours is None else T0 + timedelta(hours=hours)


def _create_tables(conn: sa.Connection) -> None:
    """The columns the migration reads, as they stand at its parent revision."""
    conn.execute(
        sa.text(
            "CREATE TABLE user_oauth (id INTEGER PRIMARY KEY, "
            "user_id INTEGER NOT NULL, provider VARCHAR(50) NOT NULL, "
            "access_token VARCHAR NOT NULL, token_type VARCHAR(50), "
            "created_at TIMESTAMP WITH TIME ZONE)"
        )
    )
    conn.execute(
        sa.text(
            "CREATE TABLE public_mcp_apps (id INTEGER PRIMARY KEY, "
            "app_id VARCHAR(100) NOT NULL, provider_name VARCHAR(50))"
        )
    )
    conn.execute(
        sa.text(
            "CREATE TABLE public_mcp_app_audits (id INTEGER PRIMARY KEY, "
            "action VARCHAR(16) NOT NULL, app_id VARCHAR(100) NOT NULL, "
            "before_values JSON, after_values JSON, "
            "created_at TIMESTAMP WITH TIME ZONE NOT NULL)"
        )
    )
    conn.execute(
        sa.text(
            "CREATE TABLE google_oauth_grants (id INTEGER PRIMARY KEY, "
            "oauth_account_id INTEGER NOT NULL)"
        )
    )


_audits = sa.table(
    "public_mcp_app_audits",
    sa.column("id", sa.Integer),
    sa.column("action", sa.String),
    sa.column("app_id", sa.String),
    sa.column("before_values", sa.JSON),
    sa.column("after_values", sa.JSON),
    sa.column("created_at", sa.DateTime(timezone=True)),
)
_rows = sa.table(
    "user_oauth",
    sa.column("id", sa.Integer),
    sa.column("user_id", sa.Integer),
    sa.column("provider", sa.String),
    sa.column("access_token", sa.String),
    sa.column("token_type", sa.String),
    sa.column("created_at", sa.DateTime(timezone=True)),
)

Snapshot = dict[str, object] | None


def _snap(provider_name: str | None) -> dict[str, object]:
    """An admin audit snapshot as ``_public_mcp_app_values`` records it."""
    return {"name": "App", "transport": "oauth", "provider_name": provider_name}


def _seed(
    conn: sa.Connection,
    *,
    rows: list[tuple[str, str | None, int | None]],
    apps: tuple[tuple[str, str | None], ...] = (),
    audits: tuple[tuple[str, str, Snapshot, Snapshot, int], ...] = (),
    grant_account_ids: tuple[int, ...] = (),
) -> None:
    """Insert ``user_oauth`` rows ``(provider key, token_type, created hour)``
    with ids 1..n, current catalog rows, audit rows ``(action, app_id,
    before_values, after_values, hour)`` in id order, and shared-grant links.
    Hours count from ``T0``; ``None`` leaves ``created_at`` NULL."""
    for index, (app_id, provider_name) in enumerate(apps, start=1):
        conn.execute(
            sa.text("INSERT INTO public_mcp_apps VALUES (:id, :app, :provider)"),
            {"id": index, "app": app_id, "provider": provider_name},
        )
    for index, (action, app_id, before, after, hour) in enumerate(audits, start=1):
        conn.execute(
            sa.insert(_audits).values(
                id=index,
                action=action,
                app_id=app_id,
                before_values=before,
                after_values=after,
                created_at=_at(hour),
            )
        )
    for index, (key, token_type, hour) in enumerate(rows, start=1):
        conn.execute(
            sa.insert(_rows).values(
                id=index,
                user_id=1,
                provider=key,
                access_token="tok",
                token_type=token_type,
                created_at=_at(hour),
            )
        )
    for index, account_id in enumerate(grant_account_ids, start=1):
        conn.execute(
            sa.text("INSERT INTO google_oauth_grants VALUES (:id, :account)"),
            {"id": index, "account": account_id},
        )


def _upgrade(conn: sa.Connection) -> None:
    with patch.object(migration, "op", Operations(MigrationContext.configure(conn))):
        migration.upgrade()


def _issuers(conn: sa.Connection) -> dict[int, str | None]:
    return dict(
        conn.execute(sa.text("SELECT id, oauth_provider FROM user_oauth")).all()
    )


def _columns(conn: sa.Connection) -> set[str]:
    return {column["name"] for column in sa.inspect(conn).get_columns("user_oauth")}


def test_builtin_app_ids_get_their_pinned_provider(engine: sa.Engine) -> None:
    """Built-in providers are normalized and protected, so a built-in key --
    with or without a catalog row or audits -- resolves to its canonical
    provider; that includes the bare ``github`` key."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("gmail", None, 1), ("github", "bearer", None), ("slack", None, 1)],
            apps=(("gmail", "google"), ("github", "github")),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: "google", 2: "github", 3: "slack"}


def test_a_builtin_id_whose_history_names_another_provider_stays_unknown(
    engine: sa.Engine,
) -> None:
    """A row for a built-in id that still names a non-canonical provider (a
    custom app that took the id before the built-in shipped) is not pinned."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(conn, rows=[("gmail", None, 1)], apps=(("gmail", "github"),))
        _upgrade(conn)
        assert _issuers(conn) == {1: None}


def test_a_custom_app_audited_from_creation_gets_its_provider(
    engine: sa.Engine,
) -> None:
    """A complete history naming one provider: edited without changing it,
    deleted afterwards, or created without a provider first."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[
                ("acme-mail", "Bearer", 2),
                ("gone-mail", None, 2),
                ("late", None, 3),
            ],
            apps=(("acme-mail", "google"), ("late", "google")),
            audits=(
                ("create", "acme-mail", None, _snap("google"), 1),
                ("update", "acme-mail", _snap("google"), _snap("google"), 3),
                ("create", "gone-mail", None, _snap("google"), 1),
                ("delete", "gone-mail", _snap("google"), None, 3),
                ("create", "late", None, _snap(None), 1),
                ("update", "late", _snap(None), _snap("google"), 2),
            ),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: "google", 2: "google", 3: "google"}


def test_a_custom_app_that_predates_auditing_stays_unknown(engine: sa.Engine) -> None:
    """Without its ``create`` audit an app may have been retargeted before
    auditing began, so a single provider visible today proves nothing: a
    pre-audit Google -> GitHub retarget shows only ``github``."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("retargeted", None, 0), ("edited", None, 0)],
            apps=(("retargeted", "github"), ("edited", "google")),
            audits=(("update", "edited", _snap("google"), _snap("google"), 5),),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: None, 2: None}


def test_a_row_older_than_its_apps_create_audit_stays_unknown(
    engine: sa.Engine,
) -> None:
    """It may belong to a pre-audit incarnation of the same app id; a row
    without ``created_at`` cannot be placed in the history either."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("acme-mail", None, 1), ("acme-mail", None, None)],
            apps=(("acme-mail", "google"),),
            audits=(("create", "acme-mail", None, _snap("google"), 5),),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: None, 2: None}


def test_a_history_that_changed_provider_stays_unknown(engine: sa.Engine) -> None:
    """A retarget and a re-create for another provider are both ambiguous."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("retargeted", None, 2), ("recreated", None, 2)],
            apps=(("retargeted", "github"), ("recreated", "github")),
            audits=(
                ("create", "retargeted", None, _snap("google"), 1),
                ("update", "retargeted", _snap("google"), _snap("github"), 3),
                ("create", "recreated", None, _snap("google"), 1),
                ("delete", "recreated", _snap("google"), None, 3),
                ("create", "recreated", None, _snap("github"), 4),
            ),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: None, 2: None}


def test_bare_non_builtin_keys_need_a_shared_grant(engine: sa.Engine) -> None:
    """A bare ``google`` key may have been written by an app whose id is
    ``google`` before auditing; only a shared Google grant proves Google.
    An unknown key stays unknown."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("google", None, 1), ("google", None, 1), ("mystery-app", None, 1)],
            grant_account_ids=(2,),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: None, 2: "google", 3: None}


def test_a_table_without_token_type_is_backfilled(engine: sa.Engine) -> None:
    """An old ``user_oauth`` carried through the chain may predate
    ``token_type``; it holds no Composio rows, so every row is native."""
    with engine.begin() as conn:
        _create_tables(conn)
        conn.execute(sa.text("ALTER TABLE user_oauth DROP COLUMN token_type"))
        conn.execute(
            sa.text(
                "INSERT INTO user_oauth (id, user_id, provider, access_token) "
                "VALUES (1, 1, 'gmail', 'tok'), (2, 1, 'mystery-app', 'tok')"
            )
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: "google", 2: None}


def test_composio_rows_are_left_untouched(engine: sa.Engine) -> None:
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("gmail", "composio", 1), ("google", "composio", 1)],
            apps=(("gmail", "google"),),
            grant_account_ids=(2,),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: None, 2: None}


def test_rerun_keeps_recorded_issuers(engine: sa.Engine) -> None:
    """A second run only touches rows still NULL, so an issuer a callback
    recorded in between is never overwritten by catalog evidence."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(conn, rows=[("gmail", None, 1)])
        _upgrade(conn)
        conn.execute(sa.text("UPDATE user_oauth SET oauth_provider = 'github'"))
        _upgrade(conn)
        assert _issuers(conn) == {1: "github"}


def test_downgrade_removes_the_column(engine: sa.Engine) -> None:
    with engine.begin() as conn:
        _create_tables(conn)
        _upgrade(conn)
        assert "oauth_provider" in _columns(conn)
        with patch.object(
            migration, "op", Operations(MigrationContext.configure(conn))
        ):
            migration.downgrade()
        assert "oauth_provider" not in _columns(conn)


def test_upgrade_without_user_oauth_is_a_no_op(sqlite_engine: sa.Engine) -> None:
    """A fresh install creates user_oauth from the models after Alembic."""
    with sqlite_engine.begin() as conn:
        _upgrade(conn)
        assert not sa.inspect(conn).has_table("user_oauth")


def test_resolve_issuer_accepts_only_durable_evidence() -> None:
    complete = migration.AppHistory(providers={"google"}, audited_since=_at(1))
    resolve = migration.resolve_issuer
    assert resolve("gmail", None, history=None) == "google"
    assert resolve("gmail", None, history=migration.AppHistory({"github"})) is None
    assert resolve("acme", _at(2), history=complete) == "google"
    assert resolve("acme", _at(0), history=complete) is None
    assert resolve("acme", None, history=complete) is None
    assert resolve("acme", _at(2), history=migration.AppHistory({"google"})) is None
    two = migration.AppHistory(providers={"google", "github"}, audited_since=_at(1))
    assert resolve("acme", _at(2), history=two) is None
    assert resolve("google", _at(2), history=None) is None
