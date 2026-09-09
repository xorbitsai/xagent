"""Xero seeding preserves operator configuration and actor launch settings."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from xagent.web.builtin_mcp_registry import (
    get_builtin_oauth_provider_rows,
    get_builtin_public_mcp_app,
)
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.public_mcp import PublicMCPApp


@pytest.fixture
def seeded(tmp_path):
    path = (
        Path(__file__).parents[2]
        / "src/xagent/migrations/versions/20260908_seed_xero_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location("seed_xero", path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'xero.db'}")
    OAuthProvider.__table__.create(engine)
    PublicMCPApp.__table__.create(engine)
    try:
        with engine.begin() as connection:
            with patch.object(
                migration, "op", Operations(MigrationContext.configure(connection))
            ):
                yield migration, connection
    finally:
        engine.dispose()


def test_xero_seed_matches_registry(seeded, monkeypatch):
    migration, connection = seeded
    monkeypatch.setenv("XERO_CLIENT_ID", "test-client")
    monkeypatch.setenv("XERO_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("XERO_REDIRECT_URI", "https://app.example/callback")
    migration.upgrade()
    migration.upgrade()
    app = connection.execute(sa.select(PublicMCPApp.__table__)).mappings().one()
    provider = connection.execute(sa.select(OAuthProvider.__table__)).mappings().one()
    expected_app = get_builtin_public_mcp_app("xero")
    assert expected_app is not None
    assert {key: app[key] for key in expected_app} == expected_app
    expected_provider = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "xero"
    )
    assert {key: provider[key] for key in expected_provider} == expected_provider


def test_xero_seed_preserves_configuration(seeded):
    migration, connection = seeded
    app = get_builtin_public_mcp_app("xero")
    assert app is not None
    app.update(is_visible_in_connector=False, description="Operator description")
    provider = next(
        row
        for row in get_builtin_oauth_provider_rows()
        if row["provider_name"] == "xero"
    )
    provider.update(client_id="existing-client", client_secret="existing-secret")
    connection.execute(sa.insert(PublicMCPApp.__table__), app)
    connection.execute(sa.insert(OAuthProvider.__table__), provider)
    migration.upgrade()
    migration.downgrade()
    persisted_app = (
        connection.execute(sa.select(PublicMCPApp.__table__)).mappings().one()
    )
    persisted_provider = (
        connection.execute(sa.select(OAuthProvider.__table__)).mappings().one()
    )
    assert {key: persisted_app[key] for key in app} == app
    assert {key: persisted_provider[key] for key in provider} == provider
