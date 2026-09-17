from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    event,
)
from sqlalchemy.orm import relationship
from sqlalchemy.schema import Index, UniqueConstraint
from sqlalchemy.sql import func

from .database import Base


def _lookup_hash(parts: tuple[object, ...]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        value = "" if part is None else str(part)
        encoded = value.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b";")
    return digest.hexdigest()


def mcp_oauth_client_lookup_hash(
    mcp_server_id: object,
    issuer: object,
    client_id: object,
) -> str:
    return _lookup_hash((mcp_server_id, issuer, client_id))


def mcp_oauth_client_registration_lookup_hash(
    mcp_server_id: object,
    issuer: object,
    redirect_uri: object,
) -> str:
    """Identify one reusable public-client registration profile."""

    return _lookup_hash(
        (
            mcp_server_id,
            issuer,
            redirect_uri,
            "public-auth-code-pkce-v1",
        )
    )


def mcp_oauth_grant_lookup_hash(
    mcp_server_id: object,
    user_id: object,
    resource_owner_key: object,
    mcp_oauth_client_id: object,
    issuer: object,
    resource: object,
    scope: object,
) -> str:
    return _lookup_hash(
        (
            mcp_server_id,
            user_id,
            resource_owner_key,
            mcp_oauth_client_id,
            issuer,
            resource,
            scope,
        )
    )


class MCPOAuthClient(Base):  # type: ignore
    """OAuth client metadata for an existing HTTP MCP server."""

    __tablename__ = "mcp_oauth_clients"
    __table_args__ = (
        UniqueConstraint(
            "lookup_hash",
            name="uq_mcp_oauth_clients_server_issuer_client",
        ),
        Index(
            "ux_mcp_oauth_clients_registration_lookup_hash",
            "registration_lookup_hash",
            unique=True,
        ),
    )

    id = Column(Integer, primary_key=True)
    mcp_server_id = Column(
        Integer,
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=False,
    )
    lookup_hash = Column(String(64), nullable=False)
    registration_lookup_hash = Column(String(64), nullable=True)
    issuer = Column(String(1000), nullable=False)
    authorization_endpoint = Column(String(1000), nullable=False)
    token_endpoint = Column(String(1000), nullable=False)
    client_id = Column(String(1000), nullable=False)
    client_secret = Column(Text, nullable=True)
    token_endpoint_auth_method = Column(String(100), nullable=False, default="none")
    redirect_uri = Column(String(1000), nullable=False)
    metadata_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    mcp_server = relationship("MCPServer")

    def __repr__(self) -> str:
        return (
            f"<MCPOAuthClient(id={self.id}, mcp_server_id={self.mcp_server_id}, "
            f"issuer='{self.issuer}')>"
        )


class MCPOAuthGrant(Base):  # type: ignore
    """Encrypted OAuth grant for an MCP resource owner."""

    __tablename__ = "mcp_oauth_grants"
    __table_args__ = (
        UniqueConstraint(
            "lookup_hash",
            name="uq_mcp_oauth_grants_lookup",
        ),
    )

    id = Column(Integer, primary_key=True)
    mcp_server_id = Column(
        Integer,
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mcp_oauth_client_id = Column(
        Integer,
        ForeignKey("mcp_oauth_clients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    lookup_hash = Column(String(64), nullable=False)
    resource_owner_key = Column(String(512), nullable=False)
    issuer = Column(String(1000), nullable=False)
    resource = Column(String(1000), nullable=False)
    scope = Column(String(1000), nullable=False, default="")
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    token_type = Column(String(50), nullable=False, default="Bearer")
    status = Column(String(50), nullable=False, default="active")
    metadata_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    mcp_server = relationship("MCPServer")
    oauth_client = relationship("MCPOAuthClient")
    user = relationship("User")

    def __repr__(self) -> str:
        return (
            f"<MCPOAuthGrant(id={self.id}, mcp_server_id={self.mcp_server_id}, "
            f"user_id={self.user_id}, resource_owner_key='{self.resource_owner_key}')>"
        )


def _set_client_lookup_hash(
    _mapper: Any, _connection: Any, target: MCPOAuthClient
) -> None:
    setattr(
        target,
        "lookup_hash",
        mcp_oauth_client_lookup_hash(
            target.mcp_server_id,
            target.issuer,
            target.client_id,
        ),
    )


def _set_grant_lookup_hash(
    _mapper: Any, _connection: Any, target: MCPOAuthGrant
) -> None:
    setattr(
        target,
        "lookup_hash",
        mcp_oauth_grant_lookup_hash(
            target.mcp_server_id,
            target.user_id,
            target.resource_owner_key,
            target.mcp_oauth_client_id,
            target.issuer,
            target.resource,
            target.scope,
        ),
    )


event.listen(MCPOAuthClient, "before_insert", _set_client_lookup_hash)
event.listen(MCPOAuthClient, "before_update", _set_client_lookup_hash)
event.listen(MCPOAuthGrant, "before_insert", _set_grant_lookup_hash)
event.listen(MCPOAuthGrant, "before_update", _set_grant_lookup_hash)


class MCPOAuthFlowState(Base):  # type: ignore
    """Short-lived OAuth state for MCP Authorization Code + PKCE."""

    __tablename__ = "mcp_oauth_flow_states"

    id = Column(Integer, primary_key=True)
    state = Column(String(255), nullable=False, unique=True)
    mcp_server_id = Column(
        Integer,
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Snapshot of the exact UserMCPServer lifecycle that created this flow.
    # NULL is reserved for flows created before the lifecycle fence migration;
    # callbacks for those rows fail closed because their association identity
    # cannot be proven after a delete-and-recreate race.
    association_lifecycle_generation = Column(Uuid, nullable=True)
    mcp_oauth_client_id = Column(
        Integer,
        ForeignKey("mcp_oauth_clients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resource_owner_key = Column(String(512), nullable=False)
    issuer = Column(String(1000), nullable=False)
    resource = Column(String(1000), nullable=False)
    scope = Column(Text, nullable=False, default="")
    code_verifier = Column(Text, nullable=False)
    redirect_after = Column(String(1000), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    consumed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    mcp_server = relationship("MCPServer")
    oauth_client = relationship("MCPOAuthClient")
    user = relationship("User")

    def __repr__(self) -> str:
        return (
            f"<MCPOAuthFlowState(id={self.id}, mcp_server_id={self.mcp_server_id}, "
            f"user_id={self.user_id})>"
        )
