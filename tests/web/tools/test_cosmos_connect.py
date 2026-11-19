"""How the Azure Cosmos DB catalog row is connected: three env keys, hidden until verified."""

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import decrypt_env_dict
from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User

ENV = {
    "COSMOS_ENDPOINT": "https://acct.documents.azure.com:443/",
    "COSMOS_KEY": "a2V5",
    "COSMOS_DATABASE": "appdb",
}


def _session(*, visible: bool):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    user = User(username="alice", id=1, password_hash="x")
    db.add(user)
    row = get_builtin_public_mcp_app("cosmos")
    assert row is not None
    db.add(PublicMCPApp(**{**row, "is_visible_in_connector": visible}))
    db.commit()
    return db, user


def test_connect_refuses_the_shipped_hidden_row() -> None:
    db, user = _session(visible=False)
    row = get_builtin_public_mcp_app("cosmos")
    assert row is not None and row["is_visible_in_connector"] is False
    with pytest.raises(HTTPException) as excinfo:
        connect_mcp_app(
            "cosmos", MCPAppConnectRequest(env=ENV), current_user=user, db=db
        )
    assert excinfo.value.status_code == 404
    assert db.query(MCPServer).count() == 0
    db.close()


def test_connect_stores_only_the_three_declared_keys() -> None:
    db, user = _session(visible=True)
    connect_mcp_app(
        "cosmos",
        MCPAppConnectRequest(env={**ENV, "NODE_OPTIONS": "--require /tmp/x"}),
        current_user=user,
        db=db,
    )
    assoc = db.query(UserMCPServer).one()
    assert decrypt_env_dict(assoc.env) == ENV
    server = db.query(MCPServer).one()
    assert server.name == "cosmos"
    db.close()
