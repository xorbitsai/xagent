import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    select,
)

VERSIONS = Path(__file__).parent.parent.parent / "src/xagent/migrations/versions"


def _load(file_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, VERSIONS / file_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_migration():
    return _load(
        "20261019_zendesk_marketplace_headers.py", "zendesk_marketplace_migration"
    )


def _oauth_launch_config() -> dict:
    """The launch_config 20261012_zendesk_oauth left on the catalog row."""
    oauth = _load("20261012_migrate_zendesk_to_oauth.py", "zendesk_oauth_migration")
    return copy.deepcopy(oauth.OAUTH_LAUNCH_CONFIG)


@pytest.fixture()
def database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'zendesk.db'}")
    metadata = MetaData()
    apps = Table(
        "public_mcp_apps",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("app_id", String, unique=True),
        Column("launch_config", JSON),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        yield connection, apps
    engine.dispose()


def _run(connection, step: str):
    migration = _load_migration()
    with patch.object(
        migration, "op", Operations(MigrationContext.configure(connection))
    ):
        getattr(migration, step)()
    return migration


def _launch_config(connection, apps, app_id: str = "zendesk"):
    return connection.execute(
        select(apps.c.launch_config).where(apps.c.app_id == app_id)
    ).scalar_one()


def test_upgrade_forwards_the_marketplace_settings_to_the_connector(database):
    connection, apps = database
    connection.execute(
        apps.insert().values(app_id="zendesk", launch_config=_oauth_launch_config())
    )

    migration = _run(connection, "upgrade")

    launch_config = _launch_config(connection, apps)
    assert launch_config["static_env"] == migration.MARKETPLACE_STATIC_ENV
    assert {
        key: value for key, value in launch_config.items() if key != "static_env"
    } == _oauth_launch_config()


def test_upgrade_target_matches_the_registry_row(database):
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app

    connection, apps = database
    connection.execute(
        apps.insert().values(app_id="zendesk", launch_config=_oauth_launch_config())
    )

    _run(connection, "upgrade")

    registry_row = get_builtin_public_mcp_app("zendesk")
    assert registry_row is not None
    assert _launch_config(connection, apps) == registry_row["launch_config"]


def test_upgrade_keeps_other_static_env_entries(database):
    connection, apps = database
    launch_config = _oauth_launch_config()
    launch_config["static_env"] = {"EXTRA": "EXTRA_HOST_VAR"}
    connection.execute(
        apps.insert().values(app_id="zendesk", launch_config=launch_config)
    )

    migration = _run(connection, "upgrade")

    assert _launch_config(connection, apps)["static_env"] == {
        "EXTRA": "EXTRA_HOST_VAR",
        **migration.MARKETPLACE_STATIC_ENV,
    }


def test_upgrade_is_idempotent(database):
    connection, apps = database
    connection.execute(
        apps.insert().values(app_id="zendesk", launch_config=_oauth_launch_config())
    )

    _run(connection, "upgrade")
    once = _launch_config(connection, apps)
    _run(connection, "upgrade")

    assert _launch_config(connection, apps) == once


@pytest.mark.parametrize(
    "launch_config",
    [
        # The legacy API-token row, which never had Xagent provenance.
        {
            "command": "python",
            "args": ["-m", "xagent.web.tools.mcp.zendesk"],
            "required_env": ["ZENDESK_SUBDOMAIN", "ZENDESK_EMAIL"],
        },
        # A row an operator pointed at another registry.
        {
            "command": "python",
            "args": ["-m", "custom.zendesk"],
            "builtin_provenance": {
                "registry": "custom",
                "app_id": "zendesk",
                "version": 1,
            },
        },
        None,
    ],
)
def test_upgrade_leaves_a_row_xagent_does_not_own(database, launch_config):
    connection, apps = database
    connection.execute(
        apps.insert().values(app_id="zendesk", launch_config=launch_config)
    )

    _run(connection, "upgrade")

    assert _launch_config(connection, apps) == launch_config


def test_upgrade_and_downgrade_tolerate_a_missing_row(database):
    connection, apps = database
    connection.execute(
        apps.insert().values(app_id="intercom", launch_config={"command": "python"})
    )

    _run(connection, "upgrade")
    _run(connection, "downgrade")

    assert _launch_config(connection, apps, "intercom") == {"command": "python"}


def test_downgrade_restores_the_previous_launch_config(database):
    connection, apps = database
    connection.execute(
        apps.insert().values(app_id="zendesk", launch_config=_oauth_launch_config())
    )

    _run(connection, "upgrade")
    _run(connection, "downgrade")

    assert _launch_config(connection, apps) == _oauth_launch_config()


def test_downgrade_keeps_other_static_env_entries(database):
    connection, apps = database
    launch_config = _oauth_launch_config()
    launch_config["static_env"] = {"EXTRA": "EXTRA_HOST_VAR"}
    connection.execute(
        apps.insert().values(app_id="zendesk", launch_config=launch_config)
    )

    _run(connection, "upgrade")
    _run(connection, "downgrade")

    assert _launch_config(connection, apps) == launch_config
