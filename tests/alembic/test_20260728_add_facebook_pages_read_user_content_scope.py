"""Tests for adding the pages_read_user_content Facebook scope."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260728_add_facebook_pages_read_user_content_scope.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_facebook_scope_migration", migration_file
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
                oauth_scopes JSON
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO public_mcp_apps (app_id, oauth_scopes) "
            'VALUES (\'facebook\', \'["pages_show_list", "pages_read_engagement", '
            '"pages_manage_posts"]\')'
        )
    )


def _scopes(connection):
    import json

    row = connection.execute(
        text("SELECT oauth_scopes FROM public_mcp_apps WHERE app_id='facebook'")
    ).first()
    value = row[0]
    return json.loads(value) if isinstance(value, str) else value


def _create_user_oauth_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE user_oauth (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                provider VARCHAR(50) NOT NULL,
                access_token VARCHAR NOT NULL
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO user_oauth (user_id, provider, access_token) VALUES "
            "(1, 'facebook', 'old-facebook-token'), "
            "(1, 'instagram', 'old-instagram-token')"
        )
    )


def _access_tokens(connection):
    return dict(
        connection.execute(text("SELECT provider, access_token FROM user_oauth")).all()
    )


def test_upgrade_adds_pages_read_user_content(tmp_path, monkeypatch):
    monkeypatch.delenv("META_CONFIG_ID", raising=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _scopes(connection) == [
            "pages_show_list",
            "pages_read_engagement",
            "pages_manage_posts",
            "pages_read_user_content",
        ]


def test_upgrade_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv("META_CONFIG_ID", raising=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        assert _scopes(connection) == [
            "pages_show_list",
            "pages_read_engagement",
            "pages_manage_posts",
            "pages_read_user_content",
        ]


def test_downgrade_removes_pages_read_user_content(tmp_path, monkeypatch):
    monkeypatch.delenv("META_CONFIG_ID", raising=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert _scopes(connection) == [
            "pages_show_list",
            "pages_read_engagement",
            "pages_manage_posts",
        ]


def test_upgrade_invalidates_existing_facebook_grant_only(tmp_path, monkeypatch):
    monkeypatch.delenv("META_CONFIG_ID", raising=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        tokens = _access_tokens(connection)
        assert tokens["facebook"] == ""
        assert tokens["instagram"] == "old-instagram-token"


def test_upgrade_without_user_oauth_table_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("META_CONFIG_ID", raising=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when user_oauth doesn't exist


def test_upgrade_leaves_grants_connected_under_meta_config_id(tmp_path, monkeypatch):
    """Under META_CONFIG_ID, invalidating tokens can't be remedied by
    reconnecting (see upgrade()'s guard), so existing grants must be left
    alone rather than force-disconnected for nothing.
    """
    monkeypatch.setenv("META_CONFIG_ID", "1234567890")
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        tokens = _access_tokens(connection)
        assert tokens["facebook"] == "old-facebook-token"
        assert tokens["instagram"] == "old-instagram-token"
        # The scope column is still kept in sync regardless.
        assert _scopes(connection) == [
            "pages_show_list",
            "pages_read_engagement",
            "pages_manage_posts",
            "pages_read_user_content",
        ]


def test_migration_scopes_match_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "facebook"
    )
    assert migration.CURRENT_SCOPES == registry_row["oauth_scopes"]
