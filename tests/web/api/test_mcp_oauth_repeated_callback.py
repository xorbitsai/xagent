"""A repeated MCP OAuth callback reports whether the flow's owner is connected (#3941).

A browser can request the callback URL twice for one authorization: a
double-submitted consent, or a retry. The first request claims the flow state.
The second must neither exchange a code nor write anything; it only picks the
page the browser lands on:

* the flow's owner holds an active grant for the flow's server, client, issuer
  and resource: success;
* otherwise: ``state_already_consumed``, never the cookie error a missing
  cookie would give.

An unclaimed flow still needs its state cookie, and an expired one is treated
as before.
"""

from __future__ import annotations

from datetime import timedelta
from urllib.parse import urlparse

import pytest
from fastapi import HTTPException, status
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from tests.web.api.test_mcp_oauth_required_scope_callback import (
    ACTOR_OWNER,
    BITBUCKET_SCOPES,
    ISSUED_TOKENS,
    ISSUER,
    MCP_URL,
    _add_flow,
    _atlassian_auth,
    _callback_request,
    _redirect_query,
    _scope_without,
)
from xagent.core.utils.encryption import decrypt_value, encrypt_value
from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import mcp_oauth_callback
from xagent.web.builtin_mcp_registry import ATLASSIAN_MCP_OAUTH_SCOPE
from xagent.web.models import MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer
from xagent.web.models.user import User

STATE = "state-repeated"
SUCCESS = {"mcp_oauth_success": ["1"]}


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'repeated-callback.db'}",
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


@pytest.fixture()
def flow_db(db_session):
    """A session holding one unclaimed flow, ``STATE``, for alice."""
    db, user = db_session
    _add_flow(
        db,
        user,
        _atlassian_auth(),
        state=STATE,
        requested_scope=ATLASSIAN_MCP_OAUTH_SCOPE,
    )
    return db, user


def _cookieless_request(state: str) -> Request:
    """The repeat as the browser sends it after the first response cleared the cookie."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/mcp/oauth/callback",
            "query_string": f"code=second-code&state={state}".encode(),
            "headers": [],
        }
    )


def _repeat_request(state: str, *, with_cookie: bool) -> Request:
    # With the cookie: the two requests overlapped, so the second one left
    # before the first response cleared it.
    return _callback_request(state) if with_cookie else _cookieless_request(state)


def _record_exchanges(monkeypatch, *, fail: bool = False) -> list[str]:
    """Answer each code exchange (or refuse it); return the codes exchanged."""
    exchanged: list[str] = []

    async def fake_exchange(**kwargs):
        exchanged.append(kwargs["code"])
        if fail:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "token_exchange_failed", "message": "refused"},
            )
        return dict(ISSUED_TOKENS)

    async def refuse_revoke(_snapshot):
        raise AssertionError("no issued token is revoked here")

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)
    monkeypatch.setattr(
        mcp_api, "_revoke_mcp_oauth_issued_token_externally", refuse_revoke
    )
    return exchanged


def _flow(db) -> MCPOAuthFlowState:
    db.expire_all()
    return db.query(MCPOAuthFlowState).filter(MCPOAuthFlowState.state == STATE).one()


def _add_grant(
    db,
    flow: MCPOAuthFlowState,
    *,
    owner: str,
    scope: str = ATLASSIAN_MCP_OAUTH_SCOPE,
    server_id: int | None = None,
) -> MCPOAuthGrant:
    grant = MCPOAuthGrant(
        mcp_server_id=server_id if server_id is not None else flow.mcp_server_id,
        user_id=flow.user_id,
        mcp_oauth_client_id=flow.mcp_oauth_client_id,
        lookup_hash=f"hash-{owner}-{server_id}-{scope}",
        resource_owner_key=owner,
        issuer=ISSUER,
        resource=MCP_URL,
        scope=scope,
        access_token=encrypt_value("stored-access-token"),
        status="active",
    )
    db.add(grant)
    db.commit()
    return grant


def _mark_consumed(db, *, expired: bool = False) -> MCPOAuthFlowState:
    """A flow some earlier callback already claimed."""
    flow = _flow(db)
    now = mcp_api._utc_now()
    flow.consumed_at = now - timedelta(seconds=2)
    if expired:
        flow.expires_at = now - timedelta(seconds=1)
    db.commit()
    return _flow(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_cookie", [False, True], ids=["no-cookie", "cookie"])
async def test_a_repeat_after_a_successful_callback_lands_on_success(
    flow_db, monkeypatch, with_cookie
):
    db, user = flow_db
    exchanged = _record_exchanges(monkeypatch)
    first = await mcp_oauth_callback(_callback_request(STATE), db)
    assert _redirect_query(first) == SUCCESS
    db.expire_all()
    stored = db.query(MCPOAuthGrant).one()
    stored_id, stored_updated_at = stored.id, stored.updated_at
    consumed_at = _flow(db).consumed_at

    repeat = await mcp_oauth_callback(
        _repeat_request(STATE, with_cookie=with_cookie), db
    )

    assert repeat.status_code == 307
    assert _redirect_query(repeat) == SUCCESS
    # Only the first request exchanged a code.
    assert exchanged == ["auth-code"]
    db.expire_all()
    kept = db.query(MCPOAuthGrant).one()
    assert (kept.id, kept.updated_at) == (stored_id, stored_updated_at)
    assert decrypt_value(str(kept.access_token)) == "issued-access-token"
    assert _flow(db).consumed_at == consumed_at


@pytest.mark.asyncio
@pytest.mark.parametrize("with_cookie", [False, True], ids=["no-cookie", "cookie"])
async def test_a_repeat_after_a_failed_exchange_reports_the_state_as_used(
    flow_db, monkeypatch, with_cookie
):
    db, user = flow_db
    exchanged = _record_exchanges(monkeypatch, fail=True)
    first = await mcp_oauth_callback(_callback_request(STATE), db)
    assert _redirect_query(first)["mcp_oauth_error"] == ["token_exchange_failed"]

    repeat = await mcp_oauth_callback(
        _repeat_request(STATE, with_cookie=with_cookie), db
    )

    assert _redirect_query(repeat)["mcp_oauth_error"] == ["state_already_consumed"]
    assert exchanged == ["auth-code"]
    db.expire_all()
    assert db.query(MCPOAuthGrant).count() == 0


@pytest.mark.asyncio
async def test_a_revoked_grant_does_not_make_a_repeat_succeed(flow_db, monkeypatch):
    db, user = flow_db
    _record_exchanges(monkeypatch)
    await mcp_oauth_callback(_callback_request(STATE), db)
    db.expire_all()
    grant = db.query(MCPOAuthGrant).one()
    grant.status = "revoked"
    grant.revoked_at = mcp_api._utc_now()
    db.commit()

    repeat = await mcp_oauth_callback(_cookieless_request(STATE), db)

    assert _redirect_query(repeat)["mcp_oauth_error"] == ["state_already_consumed"]


@pytest.mark.asyncio
async def test_another_owners_grant_does_not_make_a_repeat_succeed(
    flow_db, monkeypatch
):
    db, user = flow_db
    exchanged = _record_exchanges(monkeypatch)
    flow = _mark_consumed(db)
    # Same server, client, issuer and resource, but another actor's grant.
    _add_grant(db, flow, owner="actor:workspace:bob")

    repeat = await mcp_oauth_callback(_cookieless_request(STATE), db)

    assert _redirect_query(repeat)["mcp_oauth_error"] == ["state_already_consumed"]
    assert exchanged == []


@pytest.mark.asyncio
async def test_a_grant_short_of_the_required_scope_does_not_make_a_repeat_succeed(
    flow_db, monkeypatch
):
    """A failed reconnect over an earlier grant that no longer covers the
    required scope: /oauth/status does not count that grant, so neither may the
    repeat (it would close the pop-up on a connector that is not usable)."""
    db, _user = flow_db
    exchanged = _record_exchanges(monkeypatch)
    flow = _mark_consumed(db)
    _add_grant(
        db,
        flow,
        owner=ACTOR_OWNER,
        scope=_scope_without(*BITBUCKET_SCOPES, "read:me"),
    )

    repeat = await mcp_oauth_callback(_cookieless_request(STATE), db)

    assert _redirect_query(repeat)["mcp_oauth_error"] == ["state_already_consumed"]
    assert exchanged == []


@pytest.mark.asyncio
async def test_a_grant_on_another_server_does_not_make_a_repeat_succeed(
    flow_db, monkeypatch
):
    db, _user = flow_db
    _record_exchanges(monkeypatch)
    flow = _mark_consumed(db)
    other_server = MCPServer.from_config(
        {
            "name": "atlassian-other",
            "managed": "external",
            "transport": "streamable_http",
            "url": MCP_URL,
            "auth": _atlassian_auth(),
        }
    )
    db.add(other_server)
    db.commit()
    _add_grant(db, flow, owner=ACTOR_OWNER, server_id=other_server.id)

    repeat = await mcp_oauth_callback(_cookieless_request(STATE), db)

    assert _redirect_query(repeat)["mcp_oauth_error"] == ["state_already_consumed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("connected", "expected_query"),
    [
        (True, {"mcp_oauth_success": ["1"]}),
        # That page's existing contract: one generic failure code.
        (False, {"mcp_oauth_error": ["authorization_failed"]}),
    ],
    ids=["connected", "not-connected"],
)
async def test_a_repeat_keeps_the_mcp_complete_page_contract(
    flow_db, monkeypatch, connected, expected_query
):
    db, _user = flow_db
    _record_exchanges(monkeypatch)
    flow = _flow(db)
    flow.redirect_after = "/auth/mcp-complete"
    db.commit()
    flow = _mark_consumed(db)
    if connected:
        _add_grant(db, flow, owner=ACTOR_OWNER)

    repeat = await mcp_oauth_callback(_cookieless_request(STATE), db)

    assert urlparse(repeat.headers["location"]).path.endswith("/auth/mcp-complete")
    assert _redirect_query(repeat) == expected_query


@pytest.mark.asyncio
async def test_an_expired_consumed_flow_is_treated_as_before(flow_db, monkeypatch):
    db, user = flow_db
    exchanged = _record_exchanges(monkeypatch)
    flow = _mark_consumed(db, expired=True)
    # Even with the owner connected, an expired flow reports nothing new.
    _add_grant(db, flow, owner=ACTOR_OWNER)

    repeat = await mcp_oauth_callback(_cookieless_request(STATE), db)

    assert _redirect_query(repeat)["mcp_oauth_error"] == ["invalid_state"]
    assert exchanged == []


@pytest.mark.asyncio
async def test_an_unclaimed_flow_still_needs_its_cookie(flow_db, monkeypatch):
    db, user = flow_db
    exchanged = _record_exchanges(monkeypatch)
    # An owner already connected through an earlier flow changes nothing: this
    # flow was never claimed, so without its cookie it is refused.
    _add_grant(db, _flow(db), owner=ACTOR_OWNER)

    response = await mcp_oauth_callback(_cookieless_request(STATE), db)

    assert _redirect_query(response)["mcp_oauth_error"] == ["invalid_state"]
    assert exchanged == []
    assert _flow(db).consumed_at is None


@pytest.mark.asyncio
async def test_a_repeated_team_member_callback_lands_on_success(flow_db, monkeypatch):
    db, user = flow_db
    exchanged = _record_exchanges(monkeypatch)
    owner = f"xagent:user:{user.id}"
    flow = _flow(db)
    flow.access_path = "team_member"
    flow.association_lifecycle_generation = None
    flow.resource_owner_key = owner
    db.commit()
    flow = _mark_consumed(db)
    _add_grant(db, flow, owner=owner)

    repeat = await mcp_oauth_callback(_cookieless_request(STATE), db)

    assert _redirect_query(repeat) == SUCCESS
    assert exchanged == []
