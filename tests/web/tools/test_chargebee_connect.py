"""The shipped Chargebee catalog row stays hidden until it is verified live."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.mcp import MCPServer


def test_connect_refuses_the_shipped_hidden_row() -> None:
    """Until xorbitsai/xagent-saas#2467 shows it, the row cannot be connected."""
    from fastapi import HTTPException

    from xagent.web.api.mcp import MCPAppConnectRequest, connect_mcp_app
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
    from xagent.web.models.database import Base
    from xagent.web.models.public_mcp import PublicMCPApp
    from xagent.web.models.user import User

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    user = User(username="alice", id=1, password_hash="x")
    db.add(user)
    row = get_builtin_public_mcp_app("chargebee")
    assert row is not None and row["is_visible_in_connector"] is False
    db.add(PublicMCPApp(**row))
    db.commit()

    with pytest.raises(HTTPException) as excinfo:
        connect_mcp_app(
            "chargebee",
            MCPAppConnectRequest(
                env={"CHARGEBEE_SITE": "acme-test", "CHARGEBEE_API_KEY": "key"}
            ),
            current_user=user,
            db=db,
        )
    assert excinfo.value.status_code == 404
    assert db.query(MCPServer).count() == 0
    db.close()
