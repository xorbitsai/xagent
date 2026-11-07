"""Fixtures shared by the ``user`` identity and CLI tests."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.services.test_mcp_connector_session import SANDBOX_MANAGER
from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
from xagent.web.models.database import Base
from xagent.web.models.user import User


@pytest.fixture()
def db_users(tmp_path):
    """A database with the seeded catalog, the probe user ``monitor`` and an
    admin, ``other``, that is not a probe user unless a test names it."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'user-identity.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        seed_builtin_oauth_and_public_mcp_apps(connection)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(
        username="monitor",
        email="monitor@example.com",
        password_hash="x",
        is_admin=False,
    )
    other_user = User(username="other", password_hash="x", is_admin=True)
    db.add_all([user, other_user])
    db.commit()
    db.refresh(user)
    db.refresh(other_user)
    yield db, user, other_user
    db.close()
    engine.dispose()


@pytest.fixture()
def probe_users(monkeypatch):
    """``monitor`` is the one probe user, and no sandbox manager is set."""
    monkeypatch.setenv("XAGENT_CONNECTOR_PROBE_USERS", "monitor")
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: None)
