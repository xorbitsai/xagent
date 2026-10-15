"""An MCP OAuth configuration error keeps its own unavailability reason.

When an ``mcp_oauth`` server's configuration cannot say which scope a grant
must cover (for example a ``required_scope`` naming a scope that ``scope``
does not request), runtime resolution fails with ``invalid_configuration``.
The unavailable server config and the public load summary carry that code,
instead of rewriting it to a generic connection or availability failure that
would send the user to reconnect when an administrator has to fix the server.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.config import (
    _PUBLIC_MCP_UNAVAILABLE_REASONS,
    MCPUnavailableSummary,
)
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User
from xagent.web.tools.config import MCP_UNAVAILABLE_REASONS, WebToolConfig


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'invalid-configuration.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield db, user
    db.close()
    engine.dispose()


def _add_owned_server(db, user: User, auth: dict) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "records-runtime",
            "managed": "external",
            "transport": "streamable_http",
            "url": "https://mcp.example.com/mcp",
            "auth": auth,
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    return server


@pytest.mark.asyncio
async def test_an_invalid_required_scope_leaves_the_server_unavailable_as_misconfigured(
    db_session,
):
    db, user = db_session
    server = _add_owned_server(
        db,
        user,
        {
            "type": "mcp_oauth",
            "resource": "https://mcp.example.com/mcp",
            "issuer": "https://auth.example.com",
            "client_id": "client-123",
            "scope": "records.read",
            "required_scope": "records.write",
        },
    )
    cfg = WebToolConfig(
        db=db,
        request=None,
        user=user,
        user_id=user.id,
        workspace_config={"base_dir": "/tmp", "task_id": "test"},
    )

    configs = await cfg.get_mcp_server_configs()

    diagnostics = cfg.get_mcp_oauth_diagnostics()
    assert [diagnostic["code"] for diagnostic in diagnostics] == [
        "invalid_configuration"
    ]
    assert diagnostics[0]["message"] == (
        "MCP OAuth required_scope must only name scopes that scope requests"
    )
    assert len(configs) == 1
    assert configs[0]["name"] == server.name
    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["unavailable"] is True
    assert configs[0]["config"]["server_id"] == server.id
    assert configs[0]["config"]["reason"] == "invalid_configuration"
    assert configs[0]["config"]["diagnostic"] == diagnostics[0]


def test_the_unavailable_config_keeps_the_invalid_configuration_reason():
    cfg = WebToolConfig(db=None, request=None, user_id=42)
    server = SimpleNamespace(id=7, name="records-runtime", description=None)

    config = cfg._build_unavailable_mcp_config(
        server=server, reason="invalid_configuration"
    )

    assert config["config"]["reason"] == "invalid_configuration"


def test_the_public_load_summary_keeps_the_invalid_configuration_reason():
    summary = MCPUnavailableSummary.from_values(
        "records-runtime", "invalid_configuration"
    )

    assert summary.reason == "invalid_configuration"


def test_every_web_runtime_reason_survives_the_public_summary():
    """The web runtime allowlist is the narrower one: a reason it keeps must
    not be rewritten again when the load summary is built."""
    assert MCP_UNAVAILABLE_REASONS <= _PUBLIC_MCP_UNAVAILABLE_REASONS
