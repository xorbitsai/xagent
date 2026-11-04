"""Tests for the provenance-safe Fireflies remote-MCP connector seed."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, select, text

APP_ID = "fireflies"


def _load_migration():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20260922_seed_fireflies_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location("seed_fireflies_migration", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_apps_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                icon VARCHAR(1000),
                transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                provider_name VARCHAR(50),
                category VARCHAR(100),
                oauth_scopes JSON,
                is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                launch_config JSON
            )
            """
        )
    )


def _app_ids(connection):
    return set(connection.execute(text("SELECT app_id FROM public_mcp_apps")).scalars())


def _run(connection, migration, *steps):
    with patch.object(migration, "op", _operations(connection)):
        for step in steps:
            getattr(migration, step)()


def test_upgrade_inserts_row_matching_seed_snapshot(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        # Full-row comparison through the migration's own typed table object:
        # JSON columns deserialize and booleans come back as real bools, so
        # the persisted row compares to ROW directly (including the
        # builtin_provenance marker inside launch_config).
        row = (
            connection.execute(
                select(migration.PUBLIC_MCP_APPS_TABLE).where(
                    migration.PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID
                )
            )
            .mappings()
            .first()
        )
        assert row is not None
        assert dict(row) == migration.ROW


def test_upgrade_is_idempotent_for_provenance_owned_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade", "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
    assert count == 1


def test_upgrade_is_idempotent_even_with_an_unrelated_later_collision(tmp_path):
    """A later, unrelated custom app that happens to normalize to the same
    display name must not retroactively block every future `alembic upgrade
    head` over the already-seeded, correctly-owned row."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, is_visible_in_connector) "
                "VALUES ('unrelated-app', :name, 'stdio', 1)"
            ),
            {"name": migration.ROW["name"]},
        )
        _run(connection, migration, "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
    assert count == 1


def test_upgrade_accepts_owned_row_from_an_older_provenance_version(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        launch_config = dict(migration.ROW["launch_config"])
        marker = dict(migration.BUILTIN_PROVENANCE)
        marker["version"] = 0
        launch_config["builtin_provenance"] = marker
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, launch_config) "
                "VALUES (:app_id, :name, 'streamable_http', :launch_config)"
            ),
            {
                "app_id": APP_ID,
                "name": migration.ROW["name"],
                "launch_config": json.dumps(launch_config),
            },
        )
        _run(connection, migration, "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
    assert count == 1


def test_upgrade_refuses_custom_catalog_collision(tmp_path):
    """A pre-existing custom row under this app_id (e.g. hand-created by an
    operator via POST /admin/mcp/apps before this migration deployed) has no
    provenance marker, so this must fail closed rather than adopt the row for
    the builtin execution overlay to misidentify later."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES (:app_id, 'Operator Row', 'hand-made', 'streamable_http', 0)"
            ),
            {"app_id": APP_ID},
        )
        with pytest.raises(RuntimeError, match="public_mcp_apps"):
            _run(connection, migration, "upgrade")
        row = connection.execute(
            text("SELECT name FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
    assert row == "Operator Row"


@pytest.mark.parametrize(
    "app_id,name",
    [
        (f" {APP_ID.upper()} ", "Unrelated"),
        ("other", None),  # None -> the seeded display name, padded
        (APP_ID.capitalize(), "Unrelated"),
    ],
)
def test_upgrade_skips_seeding_on_a_name_only_collision(tmp_path, app_id, name):
    """None of these rows claim the exact app_id the overlay keys off of, so
    a display-name (or look-alike app_id) collision under a *different*
    app_id must not abort the whole `alembic upgrade head` run. It only means
    the builtin row is not seeded this run."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    if name is None:
        name = f" {migration.ROW['name'].lower()} "
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, is_visible_in_connector) "
                "VALUES (:app_id, :name, 'oauth', 0)"
            ),
            {"app_id": app_id, "name": name},
        )
        _run(connection, migration, "upgrade")  # must not raise
        assert APP_ID not in _app_ids(connection)
        row = connection.execute(
            text("SELECT app_id, name FROM public_mcp_apps WHERE app_id = :app_id"),
            {"app_id": app_id},
        ).one()
        assert row == (app_id, name)


def test_upgrade_skips_columns_missing_from_a_reduced_schema(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    launch_config JSON
                )
                """
            )
        )
        _run(connection, migration, "upgrade")
        assert APP_ID in _app_ids(connection)


def test_upgrade_raises_when_launch_config_column_is_missing(tmp_path):
    """launch_config is where the provenance marker lives; without the
    column there is no way to prove ownership of any existing row."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth'
                )
                """
            )
        )
        with pytest.raises(RuntimeError, match="launch_config"):
            _run(connection, migration, "upgrade")
        assert APP_ID not in _app_ids(connection)


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == APP_ID
    )
    assert migration.ROW == registry_row
    assert registry_row["is_visible_in_connector"] is True


def test_seed_row_classifies_as_mcp_oauth():
    """The seeded shape must classify as a remote-MCP OAuth connector -- an
    "unconnectable" classification would make the catalog entry dead on
    arrival (no connect endpoint accepts it). The provenance marker inside
    launch_config must not disturb that classification."""
    from xagent.web.mcp_apps import classify_app_auth

    migration = _load_migration()
    assert (
        classify_app_auth(migration.ROW["transport"], migration.ROW["launch_config"])
        == "mcp_oauth"
    )


def test_downgrade_only_deletes_provenance_owned_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        # A sentinel row unrelated to this migration must survive the
        # downgrade -- otherwise the seeded app_id missing from _app_ids could
        # equally mean the whole table was wiped.
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport)"
                " VALUES ('sentinel', 'Sentinel', 'oauth')"
            )
        )
        _run(connection, migration, "downgrade")
        assert _app_ids(connection) == {"sentinel"}


def test_downgrade_preserves_a_preexisting_operator_row(tmp_path):
    """A pre-existing operator row makes upgrade() raise (it never gets
    adopted), so downgrade() must never delete it either."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport) "
                "VALUES (:app_id, :name, 'hand-made', 'streamable_http')"
            ),
            {"app_id": APP_ID, "name": migration.ROW["name"]},
        )
        _run(connection, migration, "downgrade")
        row = connection.execute(
            text("SELECT name, description FROM public_mcp_apps WHERE app_id=:app_id"),
            {"app_id": APP_ID},
        ).first()
        assert row == (migration.ROW["name"], "hand-made")


def test_downgrade_is_a_noop_when_launch_config_column_is_missing(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth'
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, transport) "
                "VALUES (:app_id, :name, 'streamable_http')"
            ),
            {"app_id": APP_ID, "name": migration.ROW["name"]},
        )
        _run(connection, migration, "downgrade")
        assert APP_ID in _app_ids(connection)


def test_upgrade_and_downgrade_no_op_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _run(connection, migration, "upgrade", "downgrade")
        table_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).scalars()
        )
        assert "public_mcp_apps" not in table_names


@pytest.mark.parametrize(
    "column,value",
    [
        ("description", "Edited by an administrator"),
        ("icon", "https://cdn.example/custom.png"),
        ("category", "Support"),
        ("is_visible_in_connector", 0),
    ],
)
def test_downgrade_preserves_a_provenance_owned_row_an_admin_edited(
    tmp_path, column, value
):
    """Admin PATCH may edit the presentation fields while the provenance
    marker stays intact. Downgrade must delete only a row that still matches
    the frozen seed snapshot; an edited row is supported configuration."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        connection.execute(
            text(
                f"UPDATE public_mcp_apps SET {column} = :value WHERE app_id = :app_id"
            ),
            {"value": value, "app_id": APP_ID},
        )
        _run(connection, migration, "downgrade")
        assert APP_ID in _app_ids(connection)
        stored = connection.execute(
            text(f"SELECT {column} FROM public_mcp_apps WHERE app_id = :app_id"),
            {"app_id": APP_ID},
        ).scalar_one()
        assert stored == value


# --- mcp_servers reconciliation is wired in (matrix lives in
# tests/alembic/test_seed_helpers_remote_identity.py) ---


def _create_server_tables(connection):
    connection.execute(
        text(
            """
            CREATE TABLE mcp_servers (
                id INTEGER PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                transport VARCHAR(50),
                url VARCHAR(500),
                auth JSON
            )
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE TABLE user_mcpservers (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                mcpserver_id INTEGER NOT NULL,
                is_owner BOOLEAN NOT NULL DEFAULT 0,
                is_active BOOLEAN NOT NULL DEFAULT 1
            )
            """
        )
    )


def test_upgrade_skips_when_a_user_owned_server_squats_the_identity(tmp_path, caplog):
    """A pre-seed custom server under this app_id must keep the seed from
    claiming the identity: no catalog row, server row untouched, ERROR logged.
    The full collision matrix is the shared helper's test file."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection)
        connection.execute(
            text(
                "INSERT INTO mcp_servers (id, name, transport, url, auth) VALUES "
                "(1, :name, 'streamable_http', 'https://evil.example/mcp', :auth)"
            ),
            {"name": APP_ID, "auth": json.dumps({"type": "mcp_oauth"})},
        )
        connection.execute(
            text(
                "INSERT INTO user_mcpservers (user_id, mcpserver_id, is_owner, is_active)"
                " VALUES (7, 1, 1, 1)"
            )
        )
        _run(connection, migration, "upgrade")  # must not raise
        assert APP_ID not in _app_ids(connection)
        assert connection.execute(text("SELECT url FROM mcp_servers")).scalar_one() == (
            "https://evil.example/mcp"
        )
        messages = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert any("mcp_servers" in m and APP_ID in m for m in messages), messages


def test_upgrade_accepts_the_connect_created_row_on_a_round_trip(tmp_path):
    """upgrade -> a user connects (our connect path creates the shared row) ->
    downgrade -> upgrade must succeed: that row is ours, not a squatter."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _create_server_tables(connection)
        _run(connection, migration, "upgrade")
        connection.execute(
            text(
                "INSERT INTO mcp_servers (id, name, transport, url, auth) VALUES "
                "(1, :name, 'streamable_http', :url, :auth)"
            ),
            {
                "name": APP_ID,
                "url": migration.ROW["launch_config"]["url"],
                "auth": json.dumps(migration.ROW["launch_config"]["auth"]),
            },
        )
        connection.execute(
            text(
                "INSERT INTO user_mcpservers (user_id, mcpserver_id, is_owner, is_active)"
                " VALUES (7, 1, 0, 1)"
            )
        )
        _run(connection, migration, "downgrade")
        assert APP_ID not in _app_ids(connection)
        _run(connection, migration, "upgrade")
        assert APP_ID in _app_ids(connection)
