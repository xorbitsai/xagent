"""Tests for the create-only Jira ticket connector seed migration."""

import importlib.util
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20261109_seed_jira_ticket_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_jira_ticket_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


_COLUMNS = """
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
"""


def _create_table(connection, columns: str = _COLUMNS):
    connection.execute(text(f"CREATE TABLE public_mcp_apps ({columns})"))


def _app_ids(connection):
    return set(connection.execute(text("SELECT app_id FROM public_mcp_apps")).scalars())


def _insert_custom(
    connection,
    *,
    app_id: str,
    name: str,
    launch_config: dict | None = None,
    visible: int = 0,
):
    connection.execute(
        text(
            "INSERT INTO public_mcp_apps (app_id, name, transport, "
            "is_visible_in_connector, launch_config) "
            "VALUES (:app_id, :name, 'stdio', :visible, :launch)"
        ),
        {
            "app_id": app_id,
            "name": name,
            "visible": visible,
            "launch": json.dumps(launch_config or {"command": "operator-cmd"}),
        },
    )


def _upgrade(migration, connection):
    with patch.object(migration, "op", _operations(connection)):
        migration.upgrade()


def test_upgrade_inserts_the_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _upgrade(migration, connection)
        row = (
            connection.execute(
                text(
                    "SELECT name, transport, provider_name, is_visible_in_connector, "
                    "launch_config FROM public_mcp_apps WHERE app_id='jira-ticket'"
                )
            )
            .mappings()
            .one()
        )
    assert row["name"] == "Jira Ticket"
    assert row["transport"] == "stdio"
    assert row["provider_name"] is None
    # Hidden until verified against a live site (xorbitsai/xagent#3661).
    assert row["is_visible_in_connector"] == 0
    launch = json.loads(row["launch_config"])
    assert launch["args"] == ["-m", "xagent.web.tools.mcp.jira_ticket"]
    assert launch["builtin_provenance"] == {
        "registry": "xagent",
        "app_id": "jira-ticket",
        "version": 1,
    }


def test_seed_row_carries_no_secret():
    migration = _load_migration_module()
    launch_config = migration.ROW["launch_config"]
    assert set(launch_config) == {
        "command",
        "args",
        "required_env",
        "builtin_provenance",
    }
    assert launch_config["required_env"] == [
        "JIRA_TICKET_SITE",
        "JIRA_TICKET_EMAIL",
        "JIRA_TICKET_API_TOKEN",
        "JIRA_TICKET_PROJECT_KEY",
    ]
    for forbidden in ("env", "headers", "auth"):
        assert forbidden not in launch_config


def test_seed_row_matches_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "jira-ticket"
    )
    assert migration.ROW == registry_row


def test_seed_row_classifies_api_key():
    from xagent.web.mcp_apps import classify_app_auth

    migration = _load_migration_module()
    assert (
        classify_app_auth(migration.ROW["transport"], migration.ROW["launch_config"])
        == "api_key"
    )


def test_builtin_server_identity_is_the_registry_app_id():
    """The catalog connect flow names the server row after the app_id, and the
    runtime identity checks key on the canonical identity of that
    name -- so the app_id must canonicalize to the same identity as the name."""
    from xagent.builtin_identity import canonicalize_builtin_identity

    migration = _load_migration_module()
    assert canonicalize_builtin_identity(migration.APP_ID) == "jira-ticket"
    assert canonicalize_builtin_identity(migration.ROW["name"]) == "jira-ticket"


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _upgrade(migration, connection)
        _upgrade(migration, connection)
        count = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='jira-ticket'")
        ).scalar()
    assert count == 1


def test_missing_launch_config_column_fails_loudly(tmp_path):
    """The provenance marker lives in launch_config: dropping it silently
    would seed a row the registry overlay and the connect flow cannot trust."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    columns = _COLUMNS.replace("launch_config JSON", "extra TEXT")
    with engine.begin() as connection:
        _create_table(connection, columns)
        with pytest.raises(RuntimeError, match="launch_config"):
            _upgrade(migration, connection)
        assert _app_ids(connection) == set()


def test_missing_optional_column_warns_and_still_inserts(tmp_path, caplog):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    columns = "\n".join(
        line for line in _COLUMNS.splitlines() if "description" not in line
    )
    with engine.begin() as connection:
        _create_table(connection, columns)
        with caplog.at_level(logging.WARNING):
            _upgrade(migration, connection)
        assert "jira-ticket" in _app_ids(connection)
    assert any("missing columns" in r.getMessage() for r in caplog.records)


def test_existing_row_with_matching_provenance_is_left_alone(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _upgrade(migration, connection)
        connection.execute(text("UPDATE public_mcp_apps SET is_visible_in_connector=1"))
        _upgrade(migration, connection)
        rows = connection.execute(
            text(
                "SELECT is_visible_in_connector FROM public_mcp_apps "
                "WHERE app_id = 'jira-ticket'"
            )
        ).all()
    # The seed ships hidden; an operator who showed the row keeps it shown.
    assert [row[0] for row in rows] == [1]


def test_collision_with_a_custom_app_id_row_is_preserved_and_logged(tmp_path, caplog):
    """A hand-made row under this app_id has no provenance marker. It is not
    overwritten, not adopted (no marker is added, so the registry overlay keeps
    ignoring it), and the skip is logged rather than silent."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert_custom(
            connection, app_id="jira-ticket", name="Operator Jira", visible=1
        )
        with caplog.at_level(logging.WARNING):
            _upgrade(migration, connection)
        row = (
            connection.execute(
                text(
                    "SELECT name, is_visible_in_connector, launch_config "
                    "FROM public_mcp_apps WHERE app_id='jira-ticket'"
                )
            )
            .mappings()
            .one()
        )
    assert row["name"] == "Operator Jira"
    assert row["is_visible_in_connector"] == 1
    assert json.loads(row["launch_config"]) == {"command": "operator-cmd"}
    assert any(
        "jira-ticket" in r.getMessage() and "builtin_provenance" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize(
    "app_id, name",
    [
        ("custom-jira", "Jira Ticket"),
        ("custom-jira", "  jira   ticket "),
        ("Jira-Ticket", "Something Else"),
    ],
)
def test_normalized_identity_collision_with_another_row_skips_the_seed(
    tmp_path, caplog, app_id, name
):
    """The connect flow rejects a catalog whose normalized id/name collides
    with another row, so seeding next to one would only create a connector
    nobody can connect. Skip it, loudly, and leave the other row untouched."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert_custom(connection, app_id=app_id, name=name)
        with caplog.at_level(logging.WARNING):
            _upgrade(migration, connection)
        assert _app_ids(connection) == {app_id}
    assert any("collides" in r.getMessage() for r in caplog.records)


def test_downgrade_removes_the_seeded_row(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _upgrade(migration, connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "jira-ticket" not in _app_ids(connection)


def test_downgrade_leaves_a_row_without_provenance_alone(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _insert_custom(connection, app_id="jira-ticket", name="Operator Jira")
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        assert "jira-ticket" in _app_ids(connection)


def test_upgrade_and_downgrade_no_op_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _upgrade(migration, connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        tables = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).scalars()
        )
    assert "public_mcp_apps" not in tables


def test_registry_row_matches_the_seeded_visibility() -> None:
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    (row,) = [
        row
        for row in get_builtin_public_mcp_app_rows()
        if row["app_id"] == "jira-ticket"
    ]
    assert row["is_visible_in_connector"] is False
    assert _load_migration_module().ROW["is_visible_in_connector"] is False
