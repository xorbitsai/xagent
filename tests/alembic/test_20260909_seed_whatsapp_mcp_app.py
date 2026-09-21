"""Tests for the provenance-safe WhatsApp Business MCP connector seed."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def _load_migration():
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20260909_seed_whatsapp_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location("seed_whatsapp_migration", path)
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


def test_upgrade_inserts_whatsapp_with_provenance(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        assert "whatsapp" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, oauth_scopes, launch_config, "
                "is_visible_in_connector, category FROM public_mcp_apps"
                " WHERE app_id='whatsapp'"
            )
        ).first()
        assert row[0] == "oauth"
        assert row[1] == "meta"
        for scope in (
            "business_management",
            "whatsapp_business_management",
            "whatsapp_business_messaging",
        ):
            assert scope in str(row[2])
        assert "xagent.web.tools.mcp.whatsapp" in str(row[3])
        assert "META_ACCESS_TOKEN" in str(row[3])
        assert '"builtin_provenance"' in str(row[3])
        assert row[4] == 1
        assert row[5] == "Communication"


def test_upgrade_is_idempotent_for_provenance_owned_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade", "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='whatsapp'")
        ).scalar_one()
    assert count == 1


def test_upgrade_is_idempotent_even_with_an_unrelated_later_collision(tmp_path):
    """A later, wholly unrelated custom app that happens to normalize to the
    same identity (e.g. an admin literally names a different connector
    "WhatsApp") must not retroactively block every future `alembic upgrade
    head` over the already-seeded, correctly-owned whatsapp row -- that
    ambiguity belongs to the unrelated row's own creation path, not to an
    idempotent re-run of this migration."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade")
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, transport, is_visible_in_connector) "
                "VALUES ('unrelated-app', 'WhatsApp', 'stdio', 1)"
            )
        )
        _run(connection, migration, "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='whatsapp'")
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
                "VALUES ('whatsapp', 'WhatsApp Business', 'oauth', :launch_config)"
            ),
            {"launch_config": json.dumps(launch_config)},
        )
        _run(connection, migration, "upgrade")
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='whatsapp'")
        ).scalar_one()
    assert count == 1


def test_upgrade_refuses_custom_catalog_collision(tmp_path):
    """A pre-existing custom app_id='whatsapp' row (e.g. hand-created by an
    operator via POST /admin/mcp/apps before this migration deployed) has no
    provenance marker, so this must fail closed rather than silently leaving
    the row in place for the builtin execution overlay to potentially
    misidentify later."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('whatsapp', 'Operator WhatsApp', 'hand-made', 'oauth', 0)"
            )
        )
        with pytest.raises(RuntimeError, match="public_mcp_apps"):
            _run(connection, migration, "upgrade")
        row = connection.execute(
            text("SELECT name FROM public_mcp_apps WHERE app_id='whatsapp'")
        ).scalar_one()
    assert row == "Operator WhatsApp"


@pytest.mark.parametrize(
    "app_id,name",
    [
        (" WHATSAPP ", "Unrelated"),
        ("other", " whatsapp business "),
        ("WhAtSaPp", "Unrelated"),
    ],
)
def test_upgrade_skips_seeding_on_a_name_only_collision(tmp_path, app_id, name):
    """None of these rows claim app_id="whatsapp" itself -- the identifier
    the builtin execution overlay actually keys off of -- so a display-name
    (or look-alike app_id) collision under a *different* app_id must not
    abort the whole `alembic upgrade head` run the way an exact app_id
    collision does. It only means the builtin row doesn't get seeded this
    run; nothing else in the migration chain is affected."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
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
        assert "whatsapp" not in _app_ids(connection)
        # The unrelated row itself is left completely untouched.
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
        assert "whatsapp" in _app_ids(connection)


def test_upgrade_raises_when_launch_config_column_is_missing(tmp_path):
    """launch_config is where the provenance marker lives; without the
    column there is no way to prove ownership of any existing row, so this
    must fail loudly instead of seeding (or skipping) silently."""
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
        assert "whatsapp" not in _app_ids(connection)


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    whatsapp row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "whatsapp"
    )
    assert migration.ROW == registry_row
    assert registry_row["is_visible_in_connector"] is True


def test_downgrade_only_deletes_provenance_owned_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        _run(connection, migration, "upgrade", "downgrade")
        assert "whatsapp" not in _app_ids(connection)


def test_downgrade_preserves_a_preexisting_operator_row(tmp_path):
    """A pre-existing operator row makes upgrade() raise (it never gets
    inserted/adopted), so downgrade() must never delete it either."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration()
    with engine.begin() as connection:
        _create_apps_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, provider_name) "
                "VALUES ('whatsapp', 'WhatsApp Business', 'hand-made', "
                "'oauth', 'meta')"
            )
        )
        _run(connection, migration, "downgrade")
        row = connection.execute(
            text(
                "SELECT name, description FROM public_mcp_apps WHERE app_id='whatsapp'"
            )
        ).first()
        assert row == ("WhatsApp Business", "hand-made")


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
                "VALUES ('whatsapp', 'WhatsApp Business', 'oauth')"
            )
        )
        _run(connection, migration, "downgrade")
        assert "whatsapp" in _app_ids(connection)


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


def test_fresh_registry_seed_rejects_normalized_custom_server_collision(tmp_path):
    """seed_builtin_oauth_and_public_mcp_apps's protected_server_identities
    guard now covers whatsapp -- pin it the same way excel's is already
    pinned, so a future edit can't silently reopen the fresh-install
    collision hole for this connector specifically."""
    from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
    from xagent.web.models.database import Base
    from xagent.web.models.mcp import MCPServer

    engine = create_engine(f"sqlite:///{tmp_path / 'fresh-seed.sqlite'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    db.add(
        MCPServer(
            name=" WhatsApp Business ",
            managed="external",
            transport="stdio",
            command="custom",
        )
    )
    db.commit()
    db.close()

    with engine.begin() as connection:
        with pytest.raises(RuntimeError, match="custom mcp_servers identity"):
            seed_builtin_oauth_and_public_mcp_apps(connection)
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='whatsapp'")
        ).scalar_one()

    assert count == 0
