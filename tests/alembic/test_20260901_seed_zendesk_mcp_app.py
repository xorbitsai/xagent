"""Tests for the Zendesk MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260901_seed_zendesk_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_zendesk_migration", migration_file
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


def test_upgrade_inserts_zendesk(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "zendesk" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, launch_config, "
                "is_visible_in_connector FROM public_mcp_apps"
                " WHERE app_id='zendesk'"
            )
        ).first()
        assert row[0] == "stdio"
        assert row[1] is None
        assert "xagent.web.tools.mcp.zendesk" in str(row[2])
        for env_key in ("ZENDESK_SUBDOMAIN", "ZENDESK_EMAIL", "ZENDESK_API_TOKEN"):
            assert env_key in str(row[2])
        # Assert the persisted column value, not just dict agreement
        # (test_seed_row_matches_registry): the table DDL defaults this
        # column to 1, so a dropped/mistyped key would ship the connector
        # visible.
        assert row[3] == 0


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='zendesk'")
        ).scalar()
        assert rows == 1


def test_upgrade_still_inserts_when_non_visibility_column_missing(tmp_path):
    """A column other than is_visible_in_connector missing (e.g.
    description) must not block seeding -- only is_visible_in_connector's
    absence is load-bearing enough to fail loudly."""
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
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                    launch_config JSON
                )
                """
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "zendesk" in _app_ids(connection)


def test_upgrade_raises_if_visibility_column_is_missing(tmp_path):
    """is_visible_in_connector is load-bearing for the hidden-rollout gate
    this row ships behind, so its absence must fail loudly rather than let
    the column-filter silently drop it and fall back to the table's
    visible-by-default."""
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
        with patch.object(migration, "op", _operations(connection)):
            try:
                migration.upgrade()
                raised = False
            except RuntimeError as exc:
                raised = True
                assert "is_visible_in_connector" in str(exc)
        assert raised, "upgrade() must raise when the visibility column is missing"
        assert "zendesk" not in _app_ids(connection)


def test_upgrade_forces_hidden_on_a_preexisting_colliding_row(tmp_path):
    """N1: a hand-created row with this app_id (visible by the table
    default) would otherwise survive the collision branch untouched -- and
    the builtin registry overlays the real launch config onto any row
    sharing the app_id at read time, silently yielding a visible, working
    Zendesk connector and defeating the hidden-rollout gate. The collision
    branch must enforce hidden instead of returning early, without
    inserting a duplicate or touching the operator's other fields."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('zendesk', 'Operator Zendesk', 'hand-made', "
                "'stdio', 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        row = connection.execute(
            text(
                "SELECT COUNT(*), MIN(is_visible_in_connector), MIN(name) "
                "FROM public_mcp_apps WHERE app_id='zendesk'"
            )
        ).first()
        assert row[0] == 1  # no duplicate inserted
        assert row[1] == 0  # forced hidden
        assert row[2] == "Operator Zendesk"  # other fields left alone


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    zendesk row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "zendesk"
    )
    assert migration.ROW == registry_row


def test_downgrade_removes_zendesk(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "zendesk" not in _app_ids(connection)


def test_downgrade_is_a_noop_when_a_guard_column_is_missing(tmp_path):
    """The downgrade() guard matches on name/description/transport to avoid
    destroying an adopted operator row -- but a reduced-schema table
    missing one of those columns (e.g. mid-migration-chain, same
    precondition upgrade() already tolerates via its column-filter) must
    not make the DELETE reference a nonexistent column and raise.

    It must also not fall back to deleting on whichever guard columns
    remain: fewer guards is a weaker match, which reopens the exact
    coincidental-match risk the three-column guard exists to rule out (see
    test_downgrade_preserves_an_adopted_row_on_a_reduced_schema below).
    No-op and leave a hidden orphan row behind instead, same tradeoff as
    the admin-edited-description case, mirroring the chrome seed
    migration's identical guard."""
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
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                    launch_config JSON
                )
                """
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()  # must not raise: no `description` column
        # No-op, not a delete: the migration's own row is left behind as a
        # hidden orphan rather than risk a weaker, coincidence-prone match.
        assert "zendesk" in _app_ids(connection)


def test_downgrade_preserves_an_adopted_row_on_a_reduced_schema(tmp_path):
    """On a reduced schema missing a guard column, a weaker fix would drop
    that column's predicate and delete on whatever guards remained -- e.g.
    with `description` absent, matching on just name+transport. A
    hand-made operator row is plausibly named 'Zendesk' with transport
    'stdio' (the natural values for this connector), so that weaker match
    could delete an adopted operator row, not just the migration's own
    row. Pin that such a row survives both upgrade()'s adoption and
    downgrade()'s no-op."""
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
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                    launch_config JSON
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, is_visible_in_connector) "
                "VALUES ('zendesk', 'Zendesk', 'stdio', 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        row = connection.execute(
            text(
                "SELECT is_visible_in_connector FROM public_mcp_apps "
                "WHERE app_id='zendesk'"
            )
        ).first()
        # Adopted (forced hidden) by upgrade(), then left in place by
        # downgrade()'s no-op -- destroyed would mean row is None.
        assert row is not None
        assert row[0] == 0


def test_downgrade_preserves_an_adopted_preexisting_row(tmp_path):
    """Mi3: the collision branch in upgrade() adopts a hand-created row
    (e.g. an operator who created one before this migration deployed) by
    flipping only is_visible_in_connector -- it does not overwrite
    name/description/transport. An unconditional DELETE-by-app_id on
    downgrade would then destroy the operator's own row, not "remove the
    entry this migration owns." Pin that upgrade (adopt) -> downgrade
    (restore, not destroy) sequence, mirroring the chrome seed migration's
    identical guard."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('zendesk', 'Operator Zendesk', 'hand-made', "
                "'stdio', 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        row = connection.execute(
            text(
                "SELECT name, is_visible_in_connector FROM public_mcp_apps "
                "WHERE app_id='zendesk'"
            )
        ).first()
        # Adopted and forced hidden by upgrade(), then left in place by
        # downgrade() -- destroyed would mean row is None.
        assert row is not None
        assert row[0] == "Operator Zendesk"
        assert row[1] == 0


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
