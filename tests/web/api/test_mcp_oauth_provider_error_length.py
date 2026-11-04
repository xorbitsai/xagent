"""A provider's ``error`` code is forwarded up to 64 characters.

``_MCP_OAUTH_PROVIDER_ERROR_RE`` accepts one to 64 code characters. The
callback tests in ``test_mcp_oauth_flow.py`` show that 65 are replaced by
``unknown_error``; these pin the other side of that boundary.
"""

from __future__ import annotations

from urllib.parse import urlencode

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.api.test_mcp_oauth_flow import (
    _add_callback_client_and_state,
    _redirect_query,
    _request,
)
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import mcp_oauth_callback
from xagent.web.models import MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.user import User

LONGEST_CODE = "access_denied." + "x" * 50


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'provider-error-length.db'}",
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


def test_the_longest_accepted_provider_error_has_64_characters():
    assert len(LONGEST_CODE) == 64

    assert mcp_api._mcp_oauth_provider_error_code(LONGEST_CODE) == LONGEST_CODE
    assert mcp_api._mcp_oauth_provider_error_code(LONGEST_CODE + "x") == (
        "unknown_error"
    )


@pytest.mark.asyncio
async def test_callback_forwards_a_64_character_provider_error(
    db_session, monkeypatch, caplog
):
    db, user = db_session
    _add_callback_client_and_state(db, user, state="provider-code-64-state")

    async def fail_exchange(**kwargs):
        pytest.fail("token exchange must not run for authorization error callbacks")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fail_exchange)

    with caplog.at_level("WARNING", logger=mcp_api.logger.name):
        response = await mcp_oauth_callback(
            _request(
                "/api/mcp/oauth/callback?"
                + urlencode({"error": LONGEST_CODE, "state": "provider-code-64-state"})
            ),
            db,
        )

    query = _redirect_query(response)
    assert query["mcp_oauth_error"] == ["provider_error"]
    assert query["mcp_oauth_provider_error"] == [LONGEST_CODE]
    assert query["mcp_oauth_error_message"] == [
        f"The provider did not authorize the connection ({LONGEST_CODE})."
    ]
    assert f"provider_error={LONGEST_CODE}" in caplog.text
    assert db.query(MCPOAuthGrant).count() == 0
