from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import String, create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer
from xagent.web.models.mcp_oauth import (
    mcp_oauth_client_lookup_hash,
    mcp_oauth_client_registration_lookup_hash,
    mcp_oauth_grant_lookup_hash,
)
from xagent.web.models.user import User
from xagent.web.services.mcp_oauth import (
    MCP_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH,
    MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _create_user_and_server(db_session):
    user = User(id=1, username="alice", password_hash="hashed")
    server = MCPServer(
        id=10,
        name="tenant_records_mcp",
        managed="external",
        transport="streamable_http",
        url="https://mcp.example.com/mcp",
    )
    db_session.add_all([user, server])
    db_session.commit()
    return user, server


def test_mcp_oauth_models_map_metadata_column_without_new_mcp_server_model(
    db_session,
):
    user, server = _create_user_and_server(db_session)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)

    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        client_id="xagent-client",
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
        metadata_json={"scopes_supported": ["records.read"]},
    )
    grant = MCPOAuthGrant(
        mcp_server_id=server.id,
        user_id=user.id,
        oauth_client=client,
        resource_owner_key="external:customer:resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        access_token="encrypted-access-token",
        refresh_token="encrypted-refresh-token",
        expires_at=expires_at,
        metadata_json={"resource": "https://mcp.example.com/mcp"},
    )
    flow_state = MCPOAuthFlowState(
        state="state-123",
        mcp_server_id=server.id,
        user_id=user.id,
        oauth_client=client,
        resource_owner_key="external:customer:resource-owner-a",
        issuer="https://auth.example.com",
        resource="https://mcp.example.com/mcp",
        scope="records.read",
        code_verifier="encrypted-code-verifier",
        expires_at=expires_at,
    )
    db_session.add_all([client, grant, flow_state])
    db_session.commit()

    inspector = inspect(db_session.bind)
    client_columns = {
        column["name"] for column in inspector.get_columns("mcp_oauth_clients")
    }
    grant_columns = {
        column["name"] for column in inspector.get_columns("mcp_oauth_grants")
    }
    assert "lookup_hash" in client_columns
    assert "registration_lookup_hash" in client_columns
    assert "metadata" in grant_columns
    assert "lookup_hash" in grant_columns
    grant_column_types = {
        column["name"]: column["type"]
        for column in inspector.get_columns("mcp_oauth_grants")
    }
    assert isinstance(grant_column_types["scope"], String)
    assert grant_column_types["scope"].length == 1000
    assert grant_column_types["resource_owner_key"].length == (
        MCP_OAUTH_RESOURCE_OWNER_KEY_MAX_LENGTH
    )
    client_column_types = {
        column["name"]: column["type"]
        for column in inspector.get_columns("mcp_oauth_clients")
    }
    assert client_column_types["token_endpoint_auth_method"].length == (
        MCP_OAUTH_TOKEN_ENDPOINT_AUTH_METHOD_MAX_LENGTH
    )
    client_indexes = {
        index["name"]: index for index in inspector.get_indexes("mcp_oauth_clients")
    }
    assert "ix_mcp_oauth_clients_issuer" not in client_indexes
    registration_index = client_indexes["ux_mcp_oauth_clients_registration_lookup_hash"]
    assert registration_index["column_names"] == ["registration_lookup_hash"]
    assert registration_index["unique"] == 1

    stored_grant = db_session.query(MCPOAuthGrant).one()
    assert stored_grant.mcp_server_id == server.id
    assert stored_grant.user_id == user.id
    assert stored_grant.mcp_oauth_client_id == client.id
    assert stored_grant.resource_owner_key == "external:customer:resource-owner-a"
    assert stored_grant.metadata_json == {"resource": "https://mcp.example.com/mcp"}
    assert stored_grant.mcp_server.id == server.id
    assert client.lookup_hash == mcp_oauth_client_lookup_hash(
        server.id,
        "https://auth.example.com",
        "xagent-client",
    )
    assert len(client.lookup_hash) == 64
    assert client.registration_lookup_hash is None
    assert stored_grant.lookup_hash == mcp_oauth_grant_lookup_hash(
        server.id,
        user.id,
        "external:customer:resource-owner-a",
        client.id,
        "https://auth.example.com",
        "https://mcp.example.com/mcp",
        "records.read",
    )
    assert len(stored_grant.lookup_hash) == 64


def test_mcp_oauth_registration_lookup_hash_identifies_public_client_profile():
    lookup_hash = mcp_oauth_client_registration_lookup_hash(
        10,
        "https://auth.example.com",
        "https://xagent.example.com/api/mcp/oauth/callback",
    )

    assert len(lookup_hash) == 64
    assert lookup_hash == mcp_oauth_client_registration_lookup_hash(
        10,
        "https://auth.example.com",
        "https://xagent.example.com/api/mcp/oauth/callback",
    )
    assert lookup_hash != mcp_oauth_client_registration_lookup_hash(
        10,
        "https://auth.example.com",
        "https://other.example.com/api/mcp/oauth/callback",
    )


def test_mcp_oauth_grant_lookup_uniqueness_includes_resource_owner_key(db_session):
    user, server = _create_user_and_server(db_session)
    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        client_id="xagent-client",
        redirect_uri="https://xagent.example.com/api/mcp/oauth/callback",
    )
    db_session.add(client)
    db_session.flush()
    base_grant = {
        "mcp_server_id": server.id,
        "user_id": user.id,
        "mcp_oauth_client_id": client.id,
        "issuer": "https://auth.example.com",
        "resource": "https://mcp.example.com/mcp",
        "scope": "records.read",
        "access_token": "encrypted-access-token",
    }

    db_session.add(
        MCPOAuthGrant(
            **base_grant,
            resource_owner_key="external:customer:resource-owner-a",
        )
    )
    db_session.add(
        MCPOAuthGrant(
            **base_grant,
            resource_owner_key="external:customer:resource-owner-b",
        )
    )
    db_session.commit()

    db_session.add(
        MCPOAuthGrant(
            **base_grant,
            resource_owner_key="external:customer:resource-owner-a",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
