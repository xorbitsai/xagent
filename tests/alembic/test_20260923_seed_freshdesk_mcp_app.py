"""Tests for the Freshdesk MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260923_seed_freshdesk_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_freshdesk_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection):
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


def test_upgrade_inserts_freshdesk(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "freshdesk" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, is_visible_in_connector, "
                "launch_config FROM public_mcp_apps WHERE app_id='freshdesk'"
            )
        ).first()
        assert row[0] == "stdio"
        assert row[1] is None
        # Seeded hidden: nothing here has been verified against a live tenant
        # and the connector can email a requester. Matches zendesk/intercom.
        assert row[2] == 0
        assert "xagent.web.tools.mcp.freshdesk" in str(row[3])
        # Both halves of the per-user configuration must be declared: a
        # subdomain without a key cannot authenticate, and a key without a
        # subdomain has no tenant to authenticate against.
        assert "FRESHDESK_SUBDOMAIN" in str(row[3])
        assert "FRESHDESK_API_KEY" in str(row[3])


def test_seed_row_carries_no_secret():
    """The shared catalog row must never carry a credential -- only the names
    of the env vars each user fills in through the connect flow.
    """
    migration = _load_migration_module()
    launch_config = migration.ROW["launch_config"]
    assert set(launch_config) == {"command", "args", "required_env"}
    assert launch_config["required_env"] == [
        "FRESHDESK_SUBDOMAIN",
        "FRESHDESK_API_KEY",
    ]
    # No env/headers/auth key at all: a value here would be shared by every
    # user of the row, which is exactly what the per-user env path exists to
    # avoid.
    assert "env" not in launch_config
    assert "headers" not in launch_config
    assert "auth" not in launch_config


def test_upgrade_warns_and_still_inserts_when_column_missing(tmp_path, caplog):
    """If a table predates one of ROW's keys (shouldn't happen here, but the
    column-filter exists defensively), the row must still be inserted, and
    the drop must be logged rather than silent -- app_id then already
    exists, so a later run can never self-heal the missing column."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
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
        with patch.object(migration, "op", _operations(connection)):
            with caplog.at_level("WARNING"):
                migration.upgrade()
        assert "freshdesk" in _app_ids(connection)

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "description" in message and "missing columns" in message
        for message in messages
    )


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='freshdesk'")
        ).scalar()
        assert rows == 1


def _create_table_without_visibility_column(connection):
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
                launch_config JSON
            )
            """
        )
    )


def test_upgrade_raises_if_visibility_column_is_missing(tmp_path):
    """Without is_visible_in_connector the insert would fall back to the
    table default (TRUE) and seed the row visible, so upgrade() must fail
    instead of seeding."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table_without_visibility_column(connection)
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="is_visible_in_connector"):
                migration.upgrade()
        assert "freshdesk" not in _app_ids(connection)


def test_upgrade_raises_on_a_preexisting_row_if_visibility_column_is_missing(
    tmp_path,
):
    """The visibility-column guard runs before the collision branch, so a
    pre-existing freshdesk row on a table without the column also fails with
    RuntimeError and is left unchanged."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table_without_visibility_column(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps (app_id, name, description, transport) "
                "VALUES ('freshdesk', 'Operator Freshdesk', 'hand-made', 'stdio')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="is_visible_in_connector"):
                migration.upgrade()
        rows = connection.execute(
            text(
                "SELECT name, description, transport, launch_config "
                "FROM public_mcp_apps WHERE app_id='freshdesk'"
            )
        ).all()
        assert [tuple(r) for r in rows] == [
            ("Operator Freshdesk", "hand-made", "stdio", None)
        ]


def test_upgrade_forces_hidden_on_a_preexisting_colliding_row(tmp_path):
    """A pre-existing visible freshdesk row would otherwise stay visible, and
    the builtin registry overlays the real launch config onto any row with
    this app_id, so the collision branch must hide it without inserting a
    duplicate or touching the operator's other fields."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('freshdesk', 'Operator Freshdesk', 'hand-made', "
                "'stdio', 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        row = connection.execute(
            text(
                "SELECT COUNT(*), MIN(is_visible_in_connector), MIN(name), "
                "MIN(description), MIN(transport), MIN(launch_config) "
                "FROM public_mcp_apps WHERE app_id='freshdesk'"
            )
        ).first()
        assert row[0] == 1  # no duplicate inserted
        assert row[1] == 0  # forced hidden
        # Other fields left alone.
        assert row[2] == "Operator Freshdesk"
        assert row[3] == "hand-made"
        assert row[4] == "stdio"
        assert row[5] is None


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    freshdesk row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "freshdesk"
    )
    assert migration.ROW == registry_row


def test_seed_row_classifies_api_key():
    """The Freshdesk entry must classify as "api_key" -- an "unconnectable"
    classification would make the catalog entry dead on arrival in the
    connector UI, which is the whole point of issue #1409."""
    from xagent.web.mcp_apps import classify_app_auth

    migration = _load_migration_module()
    assert (
        classify_app_auth(migration.ROW["transport"], migration.ROW["launch_config"])
        == "api_key"
    )


def test_downgrade_leaves_an_operator_modified_row_alone(tmp_path):
    """downgrade() removes only a row that still matches the seed snapshot.
    upgrade() adopts a pre-existing freshdesk row (forcing it hidden) instead
    of inserting one, so an unconditional delete by app_id would drop a row
    this migration never created -- an operator's own freshdesk entry -- or
    one edited after the seed, e.g. by a later migration.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        # Someone flips the row visible after the fact -- exactly what the
        # follow-up verification migration will do.
        connection.execute(
            text(
                "UPDATE public_mcp_apps SET is_visible_in_connector=1 "
                "WHERE app_id='freshdesk'"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "freshdesk" in _app_ids(connection), (
            "a row that no longer matches this migration's seed snapshot "
            "must survive its downgrade"
        )


def test_downgrade_preserves_an_adopted_preexisting_row(tmp_path):
    """upgrade() adopts a pre-existing operator row by flipping only
    is_visible_in_connector, so downgrade() must leave that row in place
    rather than delete it. It stays hidden: downgrade does not restore the
    visibility it had before upgrade()."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('freshdesk', 'Operator Freshdesk', 'hand-made', "
                "'stdio', 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        row = connection.execute(
            text(
                "SELECT name, is_visible_in_connector FROM public_mcp_apps "
                "WHERE app_id='freshdesk'"
            )
        ).first()
        # Adopted and forced hidden by upgrade(), then left in place by
        # downgrade() -- destroyed would mean row is None.
        assert row is not None
        assert row[0] == "Operator Freshdesk"
        assert row[1] == 0


def test_downgrade_removes_freshdesk(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "freshdesk" not in _app_ids(connection)


def test_upgrade_and_downgrade_no_op_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        table_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).scalars()
        )
        assert "public_mcp_apps" not in table_names
