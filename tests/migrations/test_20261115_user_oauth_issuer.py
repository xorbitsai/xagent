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


def _create_tables(conn: sa.Connection) -> None:
    """The columns the migration reads, as they stand at its parent revision."""
    conn.execute(
        sa.text(
            "CREATE TABLE user_oauth (id INTEGER PRIMARY KEY, "
            "user_id INTEGER NOT NULL, provider VARCHAR(50) NOT NULL, "
            "access_token VARCHAR NOT NULL, token_type VARCHAR(50))"
        )
    )
    conn.execute(
        sa.text(
            "CREATE TABLE oauth_providers (id INTEGER PRIMARY KEY, "
            "provider_name VARCHAR(50) NOT NULL)"
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
            "before_values JSON, after_values JSON)"
        )
    )


_audits = sa.table(
    "public_mcp_app_audits",
    sa.column("id", sa.Integer),
    sa.column("action", sa.String),
    sa.column("app_id", sa.String),
    sa.column("before_values", sa.JSON),
    sa.column("after_values", sa.JSON),
)

Snapshot = dict[str, object] | None


def _snap(provider_name: str | None) -> dict[str, object]:
    """An admin audit snapshot as ``_public_mcp_app_values`` records it."""
    return {"name": "App", "transport": "oauth", "provider_name": provider_name}


def _seed(
    conn: sa.Connection,
    *,
    rows: list[tuple[str, str | None]],
    providers: tuple[str, ...] = ("google", "github"),
    apps: tuple[tuple[str, str | None], ...] = (),
    audits: tuple[tuple[str, str, Snapshot, Snapshot], ...] = (),
) -> None:
    """Insert ``user_oauth`` rows as ``(provider key, token_type)`` with ids
    1..n, configured providers, current catalog rows and audit rows
    ``(action, app_id, before_values, after_values)``."""
    for index, name in enumerate(providers, start=1):
        conn.execute(
            sa.text("INSERT INTO oauth_providers VALUES (:id, :name)"),
            {"id": index, "name": name},
        )
    for index, (app_id, provider_name) in enumerate(apps, start=1):
        conn.execute(
            sa.text("INSERT INTO public_mcp_apps VALUES (:id, :app, :provider)"),
            {"id": index, "app": app_id, "provider": provider_name},
        )
    for index, (action, app_id, before, after) in enumerate(audits, start=1):
        conn.execute(
            sa.insert(_audits).values(
                id=index,
                action=action,
                app_id=app_id,
                before_values=before,
                after_values=after,
            )
        )
    for index, (key, token_type) in enumerate(rows, start=1):
        conn.execute(
            sa.text(
                "INSERT INTO user_oauth (id, user_id, provider, access_token, "
                "token_type) VALUES (:id, 1, :key, 'tok', :token_type)"
            ),
            {"id": index, "key": key, "token_type": token_type},
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


def test_bare_provider_keys_get_their_own_provider(engine: sa.Engine) -> None:
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(conn, rows=[("google", None), ("github", "bearer")])
        _upgrade(conn)
        assert _issuers(conn) == {1: "google", 2: "github"}


def test_an_app_with_a_single_provider_history_gets_that_provider(
    engine: sa.Engine,
) -> None:
    """Builtin seeded rows with no audits, a custom app edited without ever
    changing its provider, and an app deleted (only its audit trail left)
    all name one provider."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("gmail", None), ("acme-mail", "Bearer"), ("gone-mail", None)],
            apps=(("gmail", "google"), ("acme-mail", "google")),
            audits=(
                ("create", "acme-mail", None, _snap("google")),
                ("update", "acme-mail", _snap("google"), _snap("google")),
                ("create", "gone-mail", None, _snap("google")),
                ("delete", "gone-mail", _snap("google"), None),
            ),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: "google", 2: "google", 3: "google"}


def test_a_history_that_changed_provider_stays_unknown(engine: sa.Engine) -> None:
    """A retarget, a re-create for another provider, and an app id that is
    also another provider's name are all ambiguous."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("retargeted", None), ("recreated", None), ("github", None)],
            apps=(
                ("retargeted", "github"),
                ("recreated", "github"),
                ("github", "google"),
            ),
            audits=(
                ("create", "retargeted", None, _snap("google")),
                ("update", "retargeted", _snap("google"), _snap("github")),
                ("create", "recreated", None, _snap("google")),
                ("delete", "recreated", _snap("google"), None),
                ("create", "recreated", None, _snap("github")),
            ),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: None, 2: None, 3: None}


def test_an_unknown_key_stays_unknown(engine: sa.Engine) -> None:
    """No catalog row, no audit trail, and not a configured provider."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(conn, rows=[("mystery-app", None)])
        _upgrade(conn)
        assert _issuers(conn) == {1: None}


def test_a_provider_less_history_entry_is_not_a_candidate(engine: sa.Engine) -> None:
    """An app with no OAuth provider at some point could not have driven a
    native callback then, so it does not compete with the one it had."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("late-oauth", None)],
            apps=(("late-oauth", "google"),),
            audits=(
                ("create", "late-oauth", None, _snap(None)),
                ("update", "late-oauth", _snap(None), _snap("google")),
            ),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: "google"}


def test_composio_rows_are_left_untouched(engine: sa.Engine) -> None:
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(
            conn,
            rows=[("gmail", "composio"), ("google", "composio")],
            apps=(("gmail", "google"),),
        )
        _upgrade(conn)
        assert _issuers(conn) == {1: None, 2: None}


def test_rerun_keeps_recorded_issuers(engine: sa.Engine) -> None:
    """A second run only touches rows still NULL, so an issuer a callback
    recorded in between is never overwritten by catalog evidence."""
    with engine.begin() as conn:
        _create_tables(conn)
        _seed(conn, rows=[("acme-mail", None)], apps=(("acme-mail", "google"),))
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


def test_resolve_issuers_needs_exactly_one_candidate() -> None:
    resolved = migration.resolve_issuers(
        ["google", "gmail", "acme", "github", "nothing"],
        provider_names={"google", "github"},
        app_history={
            "gmail": {"google"},
            "acme": {"google", "github"},
            "github": {"github"},
        },
    )
    assert resolved == {"google": "google", "gmail": "google", "github": "github"}
