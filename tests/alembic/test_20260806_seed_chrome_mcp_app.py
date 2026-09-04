"""Tests for the Chrome MCP connector seed migration."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260806_seed_chrome_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_chrome_migration", migration_file
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


def test_upgrade_inserts_chrome(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "chrome-devtools" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, launch_config, is_visible_in_connector "
                "FROM public_mcp_apps WHERE app_id='chrome-devtools'"
            )
        ).first()
        assert row[0] == "stdio"
        assert "chrome-devtools-mcp" in str(row[1])
        # Assert the persisted column value, not just dict agreement
        # (test_seed_row_matches_registry): the table DDL defaults this
        # column to 1, so a dropped/mistyped key would ship the connector
        # visible — and it must stay hidden until persistent stdio MCP
        # sessions land.
        assert row[2] == 0


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='chrome-devtools'")
        ).scalar()
        assert rows == 1


def test_upgrade_forces_hidden_on_a_preexisting_colliding_row(tmp_path):
    """Round-8 N1: a hand-created row with this app_id (visible by the table
    default) would otherwise survive the collision branch untouched — and the
    builtin registry overlays the real launch config onto any row sharing the
    app_id at read time, silently yielding a visible, working Chrome
    connector and defeating the hidden-rollout gate. The collision branch
    must enforce hidden instead of returning early, without inserting a
    duplicate or touching the operator's other fields."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('chrome-devtools', 'Operator Chrome', 'hand-made', "
                "'stdio', 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        row = connection.execute(
            text(
                "SELECT COUNT(*), MIN(is_visible_in_connector), MIN(name) "
                "FROM public_mcp_apps WHERE app_id='chrome-devtools'"
            )
        ).first()
        assert row[0] == 1  # no duplicate inserted
        assert row[1] == 0  # forced hidden
        assert row[2] == "Operator Chrome"  # other fields left alone


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    chrome row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "chrome-devtools"
    )
    assert migration.ROW == registry_row


def test_seed_row_classifies_keyless():
    """The Chrome entry must classify as "keyless" — an "unconnectable"
    classification would make the catalog entry dead on arrival in the
    connector UI."""
    from xagent.web.mcp_apps import classify_app_auth

    migration = _load_migration_module()
    assert (
        classify_app_auth(migration.ROW["transport"], migration.ROW["launch_config"])
        == "keyless"
    )


def test_downgrade_removes_chrome(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "chrome-devtools" not in _app_ids(connection)


def test_downgrade_preserves_an_adopted_preexisting_row(tmp_path):
    """Round-9: the collision branch in upgrade() adopts a hand-created row
    (e.g. an operator who created one before this migration deployed, per
    #1143) by flipping only is_visible_in_connector -- it does not overwrite
    name/description/transport. An unconditional DELETE-by-app_id on
    downgrade would then destroy the operator's own row, not "remove the
    entry this migration owns." Pin that upgrade (adopt) -> downgrade
    (restore, not destroy) sequence."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('chrome-devtools', 'Operator Chrome', 'hand-made', "
                "'stdio', 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        row = connection.execute(
            text(
                "SELECT name, is_visible_in_connector FROM public_mcp_apps "
                "WHERE app_id='chrome-devtools'"
            )
        ).first()
        # Adopted and forced hidden by upgrade(), then left in place by
        # downgrade() -- destroyed would mean row is None.
        assert row is not None
        assert row[0] == "Operator Chrome"
        assert row[1] == 0


def test_downgrade_removes_chrome_when_guard_columns_are_missing(tmp_path):
    """The downgrade() guard matches on name/description/transport to avoid
    destroying an adopted operator row (test_downgrade_preserves_an_adopted_
    preexisting_row) -- but a reduced-schema table missing one of those
    columns (e.g. mid-migration-chain, same precondition upgrade() already
    tolerates via its column-filter) must not make the DELETE reference a
    nonexistent column and raise. It should still remove the migration's own
    row using whichever guard columns actually exist."""
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
            migration.upgrade()
            migration.downgrade()  # must not raise: no `description` column
        assert "chrome-devtools" not in _app_ids(connection)


def test_downgrade_then_upgrade_round_trip(tmp_path):
    """Round-6 MINOR-4: downgrade().upgrade() had no coverage. A downgrade
    only deletes the catalog row (leftover MCPServer/UserMCPServer rows are
    intentionally left in place per the downgrade docstring), so a
    subsequent upgrade must cleanly re-seed it rather than hitting the
    existing-row early return or a uniqueness conflict.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.upgrade()
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='chrome-devtools'")
        ).scalar()
        assert rows == 1
        assert "chrome-devtools" in _app_ids(connection)


def test_upgrade_and_downgrade_are_no_ops_without_the_table(tmp_path):
    """Round-6 MINOR-4: the early-return branch (both directions) when
    public_mcp_apps doesn't exist yet -- e.g. a fresh database mid-migration
    chain, before the migration that creates the table has run. Must not
    raise; must not create the table itself.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        existing_tables = set(sa.inspect(connection).get_table_names())
        assert "public_mcp_apps" not in existing_tables


def test_upgrade_raises_if_visibility_column_is_missing(tmp_path):
    """Round-6 MINOR-4/nit: exercises the RuntimeError path directly (the
    migration test suite runs without -O, so the assert-vs-raise distinction
    is otherwise never actually executed by this suite). A table missing
    is_visible_in_connector must fail loudly rather than seed the chrome row
    visible via the column-filter's silent drop.
    """
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
        assert "chrome-devtools" not in _app_ids(connection)


def test_dockerfile_npx_cache_pin_matches_registry():
    """The chrome-devtools-mcp version is pinned in three places: the
    registry launch args, the migration snapshot (tied to the registry by
    test_seed_row_matches_registry), and the Dockerfile's npx cache-warm
    line. The first two are covered by dict equality; this closes the third
    side of the triangle — an unsynchronized bump would silently reopen the
    per-launch registry fetch the cache warm exists to prevent (the warmed
    version would no longer match the version the launch args request).
    """
    import re

    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "chrome-devtools"
    )
    registry_specs = {
        arg
        for arg in registry_row["launch_config"]["args"]
        if arg.startswith("chrome-devtools-mcp@")
    }
    assert len(registry_specs) == 1

    dockerfile = (
        Path(__file__).parent.parent.parent / "docker/Dockerfile.backend"
    ).read_text()
    dockerfile_specs = set(re.findall(r"chrome-devtools-mcp@[\w.\-]+", dockerfile))
    assert dockerfile_specs == registry_specs

    # Round-8 N8: pin the whole npx resolution prefix, not just the version.
    # The cache key npx resolves against is determined by the command +
    # npx-level flags + package spec; if the warm line and the runtime launch
    # args drift on that prefix (e.g. one gains --prefer-offline and the
    # other doesn't), CI stays green while the warmed cache no longer
    # matches what the runtime actually resolves. Tool-level flags after the
    # package spec (--headless etc.) are intentionally excluded: the warm
    # line runs --help instead of starting a server.
    spec = next(iter(registry_specs))
    args = registry_row["launch_config"]["args"]
    runtime_prefix = " ".join(
        [registry_row["launch_config"]["command"]] + args[: args.index(spec) + 1]
    )
    assert runtime_prefix in dockerfile, (
        f"Dockerfile cache-warm line no longer matches the runtime npx "
        f"resolution prefix {runtime_prefix!r}"
    )


def test_seed_builtin_apps_raises_when_visibility_column_missing(tmp_path):
    """Round-9: seed_builtin_oauth_and_public_mcp_apps (the fresh-database
    path invoked from database.py, as opposed to this file's migration
    upgrade()) carries the identical missing-visibility-column guard for the
    identical reason -- but the only test touching its caller
    (tests/migration/test_migration.py) monkeypatches the whole function
    out, leaving this copy of the RuntimeError unexercised. Drive it
    directly instead of through that caller."""
    from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE public_mcp_apps (app_id VARCHAR(100) NOT NULL UNIQUE)")
        )
        with pytest.raises(RuntimeError, match="is_visible_in_connector"):
            seed_builtin_oauth_and_public_mcp_apps(connection)
