"""Per-user OAuth tokens for remote MCP connectors (MCP Authorization spec)."""

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .database import Base


class MCPUserOAuthToken(Base):  # type: ignore[no-any-unimported]
    """OAuth tokens a user has obtained for a specific remote MCP server.

    One row per (user, server). Sensitive fields are stored Fernet-encrypted
    by the service layer, not here.
    """

    __tablename__ = "mcp_user_oauth_tokens"
    __table_args__ = (
        UniqueConstraint("user_id", "mcpserver_id", name="uq_mcp_user_server_token"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mcpserver_id = Column(
        Integer,
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Tokens (encrypted by service layer)
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    token_type = Column(String(50), nullable=True)
    scope = Column(Text, nullable=True)

    # connected | pending | expired | error
    status = Column(String(20), nullable=False, default="pending")

    # Short-lived, only during the pending authorization handshake
    pkce_verifier = Column(Text, nullable=True)  # encrypted
    state = Column(String(255), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self) -> str:
        return (
            f"<MCPUserOAuthToken(user_id={self.user_id}, "
            f"mcpserver_id={self.mcpserver_id}, status='{self.status}')>"
        )
