"""Tests for narrowing the Google Calendar connector's OAuth scope."""

import importlib.util
import json
import os
import sqlite3
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import text

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "src/xagent/migrations/versions/20260817_narrow_google_calendar_scope.py"
)
REGISTRY_PATH = (
    Path(__file__).parent.parent.parent / "src/xagent/web/builtin_mcp_registry.py"
)

OLD_SCOPES = ["https://www.googleapis.com/auth/calendar"]
NEW_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "narrow_google_calendar_scope_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_registry_module():
    # Loaded from this checkout's file, not `import xagent...`: an editable
    # install can point at a different checkout (e.g. another git worktree),
    # which would silently compare the migration against the wrong tree.
    spec = importlib.util.spec_from_file_location(
        "builtin_mcp_registry_under_test", REGISTRY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection) -> Operations:
    return Operations(MigrationContext.configure(connection))


def _public_mcp_apps(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("app_id", sa.String(100), nullable=False, unique=True),
        sa.Column("oauth_scopes", sa.JSON),
    )


def _scopes_by_app_id(connection, table: sa.Table) -> dict[str, object]:
    rows = connection.execute(sa.select(table.c.app_id, table.c.oauth_scopes))
    return {row.app_id: row.oauth_scopes for row in rows}


def _user_oauth(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "user_oauth",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("scope", sa.String, nullable=True),
    )


def _remaining_ids_and_providers(connection, table: sa.Table) -> dict[int, str]:
    rows = connection.execute(sa.select(table.c.id, table.c.provider))
    return {row.id: row.provider for row in rows}


def test_migration_target_scope_matches_the_live_registry() -> None:
    """Guard against registry/migration drift (the bug this migration fixes).

    Reverting the scope in ``builtin_mcp_registry.py`` without updating this
    migration's ``NEW_SCOPES`` would otherwise leave every other test in this
    file green while the app requests a scope the migration never seeded.
    """
    migration = _load_migration_module()
    registry = _load_registry_module()
    registry_row = next(
        row
        for row in registry.get_builtin_public_mcp_app_rows()
        if row["app_id"] == migration.APP_ID
    )

    assert registry_row["oauth_scopes"] == migration.NEW_SCOPES


def test_upgrade_narrows_google_calendar_scope_without_touching_other_apps(
    tmp_path,
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            [
                {"app_id": "google-calendar", "oauth_scopes": OLD_SCOPES},
                {"app_id": "gmail", "oauth_scopes": ["gmail-scope"]},
            ],
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        scopes = _scopes_by_app_id(connection, table)

    assert scopes["google-calendar"] == NEW_SCOPES
    assert scopes["gmail"] == ["gmail-scope"]


def test_upgrade_is_idempotent(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {"app_id": "google-calendar", "oauth_scopes": OLD_SCOPES},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        scopes = _scopes_by_app_id(connection, table)

    assert scopes["google-calendar"] == NEW_SCOPES


def test_upgrade_skips_when_row_is_absent(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        scopes = _scopes_by_app_id(connection, table)

    assert scopes == {}


def test_upgrade_skips_when_catalog_table_is_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()


def test_upgrade_skips_when_oauth_scopes_column_is_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    table = sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("app_id", sa.String(100), primary_key=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(sa.insert(table), {"app_id": "google-calendar"})

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        stored = connection.execute(sa.select(table)).mappings().one()

    assert stored["app_id"] == "google-calendar"


def test_downgrade_restores_the_full_calendar_scope(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _public_mcp_apps(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {"app_id": "google-calendar", "oauth_scopes": OLD_SCOPES},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        scopes = _scopes_by_app_id(connection, table)

    assert scopes["google-calendar"] == OLD_SCOPES


def test_upgrade_revokes_grants_carrying_the_old_calendar_scope(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _user_oauth(metadata)
    metadata.create_all(engine)

    old_scope = migration.OLD_SCOPE
    new_scope = migration.NEW_SCOPE

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            [
                # App-scoped grant carrying the old, broad scope alongside
                # the identity scopes every Google grant also carries.
                {
                    "id": 1,
                    "provider": "google-calendar",
                    "scope": f"{old_scope} https://www.googleapis.com/auth/userinfo.email",
                },
                # A bare provider-level grant that happens to carry the old
                # scope too -- config.py's app-scoped-grant restriction
                # doesn't cover google-calendar, so this is also a live
                # Calendar credential and must be revoked the same way.
                {"id": 2, "provider": "google", "scope": old_scope},
                # Already narrowed (e.g. a user who reconnected since this
                # migration first ran) -- must survive untouched.
                {"id": 3, "provider": "google-calendar", "scope": new_scope},
                # A different connector's grant; must never be touched even
                # though its provider column also starts with "google".
                {
                    "id": 4,
                    "provider": "gmail",
                    "scope": "https://www.googleapis.com/auth/gmail.modify",
                },
                # No scope recorded at all -- must not raise on a null scope.
                {"id": 5, "provider": "google-drive", "scope": None},
            ],
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining = _remaining_ids_and_providers(connection, table)

    assert remaining == {3: "google-calendar", 4: "gmail", 5: "google-drive"}


def test_upgrade_revoke_is_idempotent(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _user_oauth(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {"id": 1, "provider": "google-calendar", "scope": migration.OLD_SCOPE},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()

        remaining = _remaining_ids_and_providers(connection, table)

    assert remaining == {}


def test_downgrade_does_not_restore_revoked_grants(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _user_oauth(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {"id": 1, "provider": "google-calendar", "scope": migration.OLD_SCOPE},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()

        remaining = _remaining_ids_and_providers(connection, table)

    assert remaining == {}


def test_upgrade_skips_revoke_when_user_oauth_table_is_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()


def test_upgrade_skips_revoke_when_scope_column_is_absent() -> None:
    migration = _load_migration_module()
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    table = sa.Table(
        "user_oauth",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("provider", sa.String(50), nullable=False),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(sa.insert(table), {"id": 1, "provider": "google-calendar"})

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining = _remaining_ids_and_providers(connection, table)

    assert remaining == {1: "google-calendar"}


def test_offline_postgresql_upgrade_emits_literal_update_only_sql() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps SET") == 1
    assert "oauth_scopes=" in sql
    assert "public_mcp_apps.app_id = 'google-calendar'" in sql
    assert "calendar.events" in sql
    assert "INSERT INTO public_mcp_apps" not in sql
    assert "DELETE FROM public_mcp_apps" not in sql
    assert "%(" not in sql
    # The revoke step needs a live SELECT to judge each row's granted scope,
    # which offline (--sql) generation can't do; it must not emit anything
    # touching user_oauth.
    assert "user_oauth" not in sql


def test_offline_postgresql_downgrade_emits_literal_update_only_sql() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.downgrade()

    sql = output.getvalue()
    assert sql.count("UPDATE public_mcp_apps SET") == 1
    assert "oauth_scopes=" in sql
    assert "public_mcp_apps.app_id = 'google-calendar'" in sql
    assert "https://www.googleapis.com/auth/calendar" in sql
    assert "calendar.events" not in sql
    assert "INSERT INTO public_mcp_apps" not in sql
    assert "DELETE FROM public_mcp_apps" not in sql
    assert "%(" not in sql


def test_offline_sqlite_downgrade_round_trips_json_scope_value() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.downgrade()

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE public_mcp_apps (app_id TEXT PRIMARY KEY, oauth_scopes JSON)"
    )
    connection.execute(
        "INSERT INTO public_mcp_apps (app_id, oauth_scopes) VALUES (?, ?)",
        ("google-calendar", json.dumps(NEW_SCOPES)),
    )
    connection.executescript(output.getvalue())

    stored = connection.execute(
        "SELECT oauth_scopes FROM public_mcp_apps WHERE app_id = 'google-calendar'"
    ).fetchone()
    connection.close()

    assert stored is not None
    assert json.loads(stored[0]) == OLD_SCOPES


def test_offline_sqlite_upgrade_round_trips_json_scope_value() -> None:
    migration = _load_migration_module()
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite",
        opts={"as_sql": True, "output_buffer": output},
    )

    with Operations.context(context):
        migration.upgrade()

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE public_mcp_apps (app_id TEXT PRIMARY KEY, oauth_scopes JSON)"
    )
    connection.execute(
        "INSERT INTO public_mcp_apps (app_id, oauth_scopes) VALUES (?, ?)",
        ("google-calendar", json.dumps(OLD_SCOPES)),
    )
    connection.executescript(output.getvalue())

    stored = connection.execute(
        "SELECT oauth_scopes FROM public_mcp_apps WHERE app_id = 'google-calendar'"
    ).fetchone()
    connection.close()

    assert stored is not None
    assert json.loads(stored[0]) == NEW_SCOPES


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20260817_narrow_google_calendar_scope"
    assert migration.down_revision == "20260813_trace_json_columns_to_jsonb"


# --- PostgreSQL-only: the sqlite tests above never exercise the real
# postgresql/JSON cast path (`_offline_scopes_literal`'s postgresql branch),
# and JSON-vs-JSONB is precisely a distinction sqlite has no concept of. Only
# a real server can confirm the offline-generated SQL is not merely
# plausible-looking text but actually executes and stores the right column
# type, mirroring the pattern in
# tests/migrations/test_20260813_trace_json_columns_to_jsonb.py (the parent
# revision, which converts these same tables' JSON columns to JSONB).


def _postgres_url() -> str | None:
    return os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
        "POSTGRES_TEST_DATABASE_URL"
    )


@pytest.fixture
def postgres_engine():
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS public_mcp_apps CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS user_oauth CASCADE"))
        conn.execute(
            text(
                "CREATE TABLE public_mcp_apps ("
                "id SERIAL PRIMARY KEY, "
                "app_id VARCHAR(100) NOT NULL UNIQUE, "
                "oauth_scopes JSON)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE user_oauth ("
                "id SERIAL PRIMARY KEY, "
                "provider VARCHAR(50) NOT NULL, "
                "scope VARCHAR)"
            )
        )
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS public_mcp_apps CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS user_oauth CASCADE"))
    engine.dispose()


def _oauth_scopes_column_type(engine: sa.engine.Engine) -> str:
    with engine.begin() as conn:
        return conn.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'public_mcp_apps' AND column_name = 'oauth_scopes'"
            )
        ).scalar_one()


@pytest.mark.postgresql
def test_postgresql_online_upgrade_narrows_scope_and_revokes_grants(
    postgres_engine,
) -> None:
    migration = _load_migration_module()

    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, oauth_scopes) "
                "VALUES ('google-calendar', :scopes)"
            ),
            {"scopes": json.dumps(OLD_SCOPES)},
        )
        connection.execute(
            text("INSERT INTO user_oauth (id, provider, scope) VALUES (1, :p, :s)"),
            {"p": "google-calendar", "s": migration.OLD_SCOPE},
        )
        connection.execute(
            text("INSERT INTO user_oauth (id, provider, scope) VALUES (2, :p, :s)"),
            {"p": "gmail", "s": "https://www.googleapis.com/auth/gmail.modify"},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        stored_scopes = connection.execute(
            text(
                "SELECT oauth_scopes FROM public_mcp_apps "
                "WHERE app_id = 'google-calendar'"
            )
        ).scalar_one()
        remaining_ids = {
            row.id for row in connection.execute(text("SELECT id FROM user_oauth"))
        }

    assert stored_scopes == NEW_SCOPES
    assert remaining_ids == {2}
    assert _oauth_scopes_column_type(postgres_engine) == "json"


@pytest.mark.postgresql
def test_postgresql_offline_upgrade_sql_executes_and_stores_json_type(
    postgres_engine,
) -> None:
    migration = _load_migration_module()

    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, oauth_scopes) "
                "VALUES ('google-calendar', :scopes)"
            ),
            {"scopes": json.dumps(OLD_SCOPES)},
        )

    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        migration.upgrade()

    with postgres_engine.begin() as connection:
        connection.execute(text(output.getvalue()))
        stored_scopes = connection.execute(
            text(
                "SELECT oauth_scopes FROM public_mcp_apps "
                "WHERE app_id = 'google-calendar'"
            )
        ).scalar_one()

    assert stored_scopes == NEW_SCOPES
    assert _oauth_scopes_column_type(postgres_engine) == "json"


@pytest.mark.postgresql
def test_postgresql_offline_downgrade_sql_executes_and_restores_old_scope(
    postgres_engine,
) -> None:
    migration = _load_migration_module()

    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, oauth_scopes) "
                "VALUES ('google-calendar', :scopes)"
            ),
            {"scopes": json.dumps(NEW_SCOPES)},
        )

    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        migration.downgrade()

    with postgres_engine.begin() as connection:
        connection.execute(text(output.getvalue()))
        stored_scopes = connection.execute(
            text(
                "SELECT oauth_scopes FROM public_mcp_apps "
                "WHERE app_id = 'google-calendar'"
            )
        ).scalar_one()

    assert stored_scopes == OLD_SCOPES
    assert _oauth_scopes_column_type(postgres_engine) == "json"
