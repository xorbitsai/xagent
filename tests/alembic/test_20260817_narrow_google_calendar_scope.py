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
        sa.Column("user_id", sa.Integer, nullable=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("scope", sa.String, nullable=True),
    )


def _mcp_servers(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "mcp_servers",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
    )


def _user_mcpservers(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "user_mcpservers",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, nullable=False),
        sa.Column("mcpserver_id", sa.Integer, nullable=False),
    )


def _remaining_ids_and_providers(connection, table: sa.Table) -> dict[int, str]:
    rows = connection.execute(sa.select(table.c.id, table.c.provider))
    return {row.id: row.provider for row in rows}


# Shared by every test that exercises "which user_oauth rows qualify for
# revocation" (the online sqlite path, the offline sqlite path, and the
# offline postgres path): one row set and one expected-survivors set, so the
# claim that all three produce identical results is structural -- checked
# against the same fixture -- rather than three independently hand-copied
# datasets that could quietly drift apart.
REVOKE_FIXTURE_ROWS = [
    # App-scoped grant carrying the old, broad scope alongside the identity
    # scopes every Google grant also carries.
    {
        "id": 1,
        "user_id": 101,
        "provider": "google-calendar",
        "scope": f"{OLD_SCOPES[0]} https://www.googleapis.com/auth/userinfo.email",
    },
    # A combined grant: include_granted_scopes=true can echo back both the
    # old and the new scope on reconnect. Presence of the old scope alone
    # must still trigger revocation -- the new scope also being present is
    # not evidence of a narrowed credential.
    {
        "id": 2,
        "user_id": 102,
        "provider": "google-calendar",
        "scope": f"{NEW_SCOPES[0]} {OLD_SCOPES[0]}",
    },
    # Already narrowed (e.g. a user who reconnected since this migration
    # first ran) -- must survive untouched.
    {"id": 3, "user_id": 103, "provider": "google-calendar", "scope": NEW_SCOPES[0]},
    # A different connector's grant; must never be touched even though its
    # provider column also starts with "google".
    {
        "id": 4,
        "user_id": 104,
        "provider": "gmail",
        "scope": "https://www.googleapis.com/auth/gmail.modify",
    },
    # No scope recorded at all -- must not raise on a null scope.
    {"id": 5, "user_id": 105, "provider": "google-drive", "scope": None},
    # A bare provider-level grant that happens to carry the old scope too --
    # deliberately left untouched: other Google connectors may be relying on
    # this exact row as their own fallback credential (config.py's
    # _resolve_legacy_oauth_access_token), so revoking it on a scope-content
    # match alone would risk collateral damage beyond Calendar. mcp_apps.py's
    # APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT closes the resulting gap
    # architecturally instead -- see test_legacy_oauth_provider_scoping.py.
    {"id": 6, "user_id": 106, "provider": "google", "scope": OLD_SCOPES[0]},
    # A google-calendar row with no scope recorded at all. Per RFC 6749 5.1
    # a provider may omit ``scope`` when it exactly matches what was
    # requested; the only thing ever requested under this app_id before this
    # migration was the old, broad scope, so a missing scope here is
    # evidence of that grant, not proof of a narrow one -- must be revoked,
    # not skipped.
    {"id": 7, "user_id": 107, "provider": "google-calendar", "scope": None},
    # Same reasoning for an empty string, which is what the callback
    # persists when the provider's response omits ``scope`` outright
    # (web/api/auth.py's ``token_data.get("scope", "")``).
    {"id": 8, "user_id": 108, "provider": "google-calendar", "scope": ""},
]
REVOKE_FIXTURE_SURVIVING_IDS = {3, 4, 5, 6}


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

    with engine.begin() as connection:
        connection.execute(sa.insert(table), REVOKE_FIXTURE_ROWS)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining_ids = set(connection.execute(sa.select(table.c.id)).scalars().all())

    assert remaining_ids == REVOKE_FIXTURE_SURVIVING_IDS


def test_upgrade_revoke_does_not_touch_gmail_watch_state(tmp_path) -> None:
    """Regression for a real collateral-damage bug: gmail_watch_states has
    ON DELETE CASCADE on user_oauth.id, so revoking a row by scope content
    alone (rather than by provider) could have silently deleted a user's
    Gmail push-notification watch state as a side effect of narrowing
    Calendar's scope."""
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    oauth_table = _user_oauth(metadata)
    watch_table = sa.Table(
        "gmail_watch_states",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "user_oauth_id",
            sa.Integer,
            sa.ForeignKey("user_oauth.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(oauth_table),
            {
                "id": 1,
                "provider": "gmail",
                # A Gmail credential that, via include_granted_scopes=true,
                # also carries the old calendar scope from a prior separate
                # Calendar connection under the same OAuth client.
                "scope": (
                    "https://www.googleapis.com/auth/gmail.modify "
                    f"{migration.OLD_SCOPE}"
                ),
            },
        )
        connection.execute(sa.insert(watch_table), {"id": 1, "user_oauth_id": 1})

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining_oauth = _remaining_ids_and_providers(connection, oauth_table)
        remaining_watch = connection.execute(sa.select(watch_table.c.id)).fetchall()

    assert remaining_oauth == {1: "gmail"}
    assert len(remaining_watch) == 1


def test_upgrade_revoke_handles_a_large_number_of_matching_rows(tmp_path) -> None:
    """The revoke DELETE is a single, unbatched statement (see the module
    docstring on why paging buys nothing under this migration's transaction
    model) -- this proves a large row count still revokes every matching
    row and doesn't silently truncate."""
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _user_oauth(metadata)
    metadata.create_all(engine)

    row_count = 1000
    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            [
                {
                    "id": i,
                    "user_id": i,
                    "provider": "google-calendar",
                    "scope": migration.OLD_SCOPE,
                }
                for i in range(1, row_count + 1)
            ],
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining = _remaining_ids_and_providers(connection, table)

    assert remaining == {}


def _calendar_server_setup(metadata: sa.MetaData):
    return (
        _user_oauth(metadata),
        _mcp_servers(metadata),
        _user_mcpservers(metadata),
    )


def test_upgrade_removes_orphaned_calendar_server_row_for_bare_google_only_user(
    tmp_path,
) -> None:
    """The runtime half of the C1/F2 fix's other shoe: a user who only ever
    connected Calendar through the app_id-less "Connect Google" batch flow
    has a user_mcpservers row backed solely by a bare provider='google' row.
    APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT (mcp_apps.py) means that bare row
    no longer counts as a Calendar credential, so this stale association
    must be cleaned up too -- left alone, the agent runtime would pick it up
    directly and fail at token resolution on every Calendar tool call."""
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    oauth_table, servers_table, user_servers_table = _calendar_server_setup(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(oauth_table),
            {"id": 1, "user_id": 100, "provider": "google", "scope": None},
        )
        connection.execute(
            sa.insert(servers_table), {"id": 1, "name": "Google Calendar"}
        )
        connection.execute(
            sa.insert(user_servers_table),
            {"id": 1, "user_id": 100, "mcpserver_id": 1},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining_servers = connection.execute(
            sa.select(user_servers_table.c.id)
        ).fetchall()

    assert remaining_servers == []


def test_upgrade_keeps_calendar_server_row_for_properly_connected_user(
    tmp_path,
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    oauth_table, servers_table, user_servers_table = _calendar_server_setup(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(oauth_table),
            {
                "id": 1,
                "user_id": 200,
                "provider": "google-calendar",
                "scope": migration.NEW_SCOPE,
            },
        )
        connection.execute(
            sa.insert(servers_table), {"id": 1, "name": "Google Calendar"}
        )
        connection.execute(
            sa.insert(user_servers_table),
            {"id": 1, "user_id": 200, "mcpserver_id": 1},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining_servers = {
            row.user_id
            for row in connection.execute(sa.select(user_servers_table.c.user_id))
        }

    assert remaining_servers == {200}


def test_upgrade_removes_calendar_server_row_for_a_user_revoked_this_run(
    tmp_path,
) -> None:
    """The orphan cleanup runs after the revoke step, against the
    post-revoke state: a user whose only google-calendar row is revoked in
    this very migration run must also lose their stale server association,
    not just the ones who never had one."""
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    oauth_table, servers_table, user_servers_table = _calendar_server_setup(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(oauth_table),
            {
                "id": 1,
                "user_id": 300,
                "provider": "google-calendar",
                "scope": migration.OLD_SCOPE,
            },
        )
        connection.execute(
            sa.insert(servers_table), {"id": 1, "name": "Google Calendar"}
        )
        connection.execute(
            sa.insert(user_servers_table),
            {"id": 1, "user_id": 300, "mcpserver_id": 1},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining_oauth = connection.execute(sa.select(oauth_table.c.id)).fetchall()
        remaining_servers = connection.execute(
            sa.select(user_servers_table.c.id)
        ).fetchall()

    assert remaining_oauth == []
    assert remaining_servers == []


def test_upgrade_orphan_cleanup_ignores_other_servers_and_users(tmp_path) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    oauth_table, servers_table, user_servers_table = _calendar_server_setup(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(servers_table),
            [
                {"id": 1, "name": "Google Calendar"},
                {"id": 2, "name": "Gmail"},
            ],
        )
        connection.execute(
            sa.insert(user_servers_table),
            [
                # Bare-google-only Calendar user -- removed.
                {"id": 1, "user_id": 100, "mcpserver_id": 1},
                # Same user also has a Gmail server association, unrelated
                # to Calendar entirely -- must survive.
                {"id": 2, "user_id": 100, "mcpserver_id": 2},
            ],
        )
        connection.execute(
            sa.insert(oauth_table),
            {"id": 1, "user_id": 100, "provider": "google", "scope": None},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining_server_ids = {
            row.mcpserver_id
            for row in connection.execute(sa.select(user_servers_table.c.mcpserver_id))
        }

    assert remaining_server_ids == {2}


def test_upgrade_skips_orphan_cleanup_when_calendar_server_row_is_absent(
    tmp_path,
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    oauth_table, servers_table, user_servers_table = _calendar_server_setup(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise with no rows anywhere


def test_upgrade_skips_orphan_cleanup_when_user_mcpservers_table_is_absent(
    tmp_path,
) -> None:
    migration = _load_migration_module()
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _user_oauth(metadata)
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(
            sa.insert(table),
            {
                "id": 1,
                "user_id": 100,
                "provider": "google-calendar",
                "scope": migration.OLD_SCOPE,
            },
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise; only user_oauth exists

        remaining = _remaining_ids_and_providers(connection, table)

    assert remaining == {}


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


def test_offline_postgresql_upgrade_emits_literal_update_and_revoke_sql() -> None:
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
    # The revoke step has no live connection to page through, so it emits
    # one literal DELETE with the same predicate the online path builds
    # (_revoke_predicate) -- see test_offline_*_upgrade_revokes_matching_*
    # for proof it round-trips the same set of rows the online path does.
    assert sql.count("DELETE FROM user_oauth") == 1
    assert "user_oauth.provider = 'google-calendar'" in sql
    assert "user_oauth.scope IS NULL" in sql
    assert "https://www.googleapis.com/auth/calendar" in sql
    # The orphaned-server cleanup needs a live SELECT (which mcp_servers row
    # is Calendar's, which users still have a credential) that offline
    # generation can't do -- it must not emit anything touching either
    # server table.
    assert "mcp_servers" not in sql
    assert "user_mcpservers" not in sql


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
    # The script also carries the revoke DELETE now (see
    # test_offline_sqlite_upgrade_revokes_matching_user_oauth_rows for that
    # behavior); this test only cares about the catalog scope, so an empty
    # table is enough to let the script run.
    connection.execute(
        "CREATE TABLE user_oauth (id INTEGER PRIMARY KEY, provider TEXT, scope TEXT)"
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


def test_offline_sqlite_upgrade_revokes_matching_user_oauth_rows() -> None:
    """The offline (--sql) path emits one literal DELETE built from the same
    _revoke_predicate the online path uses -- this executes that DELETE
    against the shared fixture and checks it matches exactly the same rows
    test_upgrade_revokes_grants_carrying_the_old_calendar_scope's online
    path does."""
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
        "CREATE TABLE user_oauth (id INTEGER PRIMARY KEY, user_id INTEGER, "
        "provider TEXT, scope TEXT)"
    )
    connection.execute(
        "INSERT INTO public_mcp_apps (app_id, oauth_scopes) VALUES (?, ?)",
        ("google-calendar", json.dumps(OLD_SCOPES)),
    )
    connection.executemany(
        "INSERT INTO user_oauth (id, user_id, provider, scope) VALUES (?, ?, ?, ?)",
        [
            (row["id"], row["user_id"], row["provider"], row["scope"])
            for row in REVOKE_FIXTURE_ROWS
        ],
    )
    connection.executescript(output.getvalue())

    remaining_ids = {row[0] for row in connection.execute("SELECT id FROM user_oauth")}
    connection.close()

    assert remaining_ids == REVOKE_FIXTURE_SURVIVING_IDS


def test_revision_metadata() -> None:
    migration = _load_migration_module()

    assert migration.revision == "20260817_narrow_google_calendar_scope"
    assert migration.down_revision == "20260813_trace_json_columns_to_jsonb"


# --- PostgreSQL-only: the sqlite tests above never exercise the real
# postgresql/JSON cast path (`_offline_scopes_literal`'s postgresql branch),
# and JSON-vs-JSONB is precisely a distinction sqlite has no concept of. Only
# a real server can confirm the offline-generated SQL is not merely
# plausible-looking text but actually executes and stores the right column
# type, mirroring the local-postgres-fixture pattern in
# tests/migrations/test_20260813_trace_json_columns_to_jsonb.py (the parent
# revision -- it converts a different set of tables, trace_events,
# trace_message_blobs, and trace_checkpoint_blobs, from JSON to JSONB;
# public_mcp_apps.oauth_scopes stays plain JSON here).


def _postgres_url() -> str | None:
    return os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
        "POSTGRES_TEST_DATABASE_URL"
    )


_POSTGRES_TABLES = (
    "user_mcpservers",
    "mcp_servers",
    "user_oauth",
    "public_mcp_apps",
)


@pytest.fixture
def postgres_engine():
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        for table_name in _POSTGRES_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
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
                "user_id INTEGER, "
                "provider VARCHAR(50) NOT NULL, "
                "scope VARCHAR)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE mcp_servers ("
                "id SERIAL PRIMARY KEY, "
                "name VARCHAR(100) NOT NULL UNIQUE)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE user_mcpservers ("
                "id SERIAL PRIMARY KEY, "
                "user_id INTEGER NOT NULL, "
                "mcpserver_id INTEGER NOT NULL)"
            )
        )
    yield engine
    with engine.begin() as conn:
        for table_name in _POSTGRES_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
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
    metadata = sa.MetaData()
    oauth_table = _user_oauth(metadata)

    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, oauth_scopes) "
                "VALUES ('google-calendar', :scopes)"
            ),
            {"scopes": json.dumps(OLD_SCOPES)},
        )
        connection.execute(sa.insert(oauth_table), REVOKE_FIXTURE_ROWS)

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        stored_scopes = connection.execute(
            text(
                "SELECT oauth_scopes FROM public_mcp_apps "
                "WHERE app_id = 'google-calendar'"
            )
        ).scalar_one()
        remaining_ids = set(
            connection.execute(sa.select(oauth_table.c.id)).scalars().all()
        )

    assert stored_scopes == NEW_SCOPES
    assert remaining_ids == REVOKE_FIXTURE_SURVIVING_IDS
    assert _oauth_scopes_column_type(postgres_engine) == "json"


@pytest.mark.postgresql
def test_postgresql_online_upgrade_removes_orphaned_calendar_server_row(
    postgres_engine,
) -> None:
    migration = _load_migration_module()
    metadata = sa.MetaData()
    oauth_table = _user_oauth(metadata)
    servers_table = _mcp_servers(metadata)
    user_servers_table = _user_mcpservers(metadata)

    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, oauth_scopes) "
                "VALUES ('google-calendar', :scopes)"
            ),
            {"scopes": json.dumps(OLD_SCOPES)},
        )
        connection.execute(
            sa.insert(oauth_table),
            {"id": 1, "user_id": 100, "provider": "google", "scope": None},
        )
        connection.execute(
            sa.insert(servers_table), {"id": 1, "name": "Google Calendar"}
        )
        connection.execute(
            sa.insert(user_servers_table),
            {"id": 1, "user_id": 100, "mcpserver_id": 1},
        )

        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        remaining_servers = connection.execute(
            sa.select(user_servers_table.c.id)
        ).fetchall()

    assert remaining_servers == []


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


@pytest.mark.postgresql
def test_postgresql_offline_upgrade_revokes_matching_user_oauth_rows(
    postgres_engine,
) -> None:
    """The offline (--sql) DELETE, executed against a real server, must
    match exactly the same rows the online path's test does -- both built
    from the same _revoke_predicate and the same fixture."""
    migration = _load_migration_module()
    metadata = sa.MetaData()
    oauth_table = _user_oauth(metadata)

    with postgres_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, oauth_scopes) "
                "VALUES ('google-calendar', :scopes)"
            ),
            {"scopes": json.dumps(OLD_SCOPES)},
        )
        connection.execute(sa.insert(oauth_table), REVOKE_FIXTURE_ROWS)

    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": output},
    )
    with Operations.context(context):
        migration.upgrade()

    with postgres_engine.begin() as connection:
        connection.execute(text(output.getvalue()))
        remaining_ids = set(
            connection.execute(sa.select(oauth_table.c.id)).scalars().all()
        )

    assert remaining_ids == REVOKE_FIXTURE_SURVIVING_IDS
