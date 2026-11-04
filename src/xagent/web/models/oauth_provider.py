from sqlalchemy import JSON, Column, Integer, String

from .database import Base


class OAuthProvider(Base):  # type: ignore[no-any-unimported]
    """Configuration for OAuth providers used by public MCP apps."""

    __tablename__ = "oauth_providers"

    id = Column(Integer, primary_key=True, index=True)
    provider_name = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    client_id = Column(String(500), nullable=False)
    client_secret = Column(String(500), nullable=False)
    auth_url = Column(String(1000), nullable=False)
    token_url = Column(String(1000), nullable=False)
    redirect_uri = Column(String(1000), nullable=True)
    userinfo_url = Column(String(1000), nullable=True)
    user_id_path = Column(String(100), default="id")
    email_path = Column(String(100), default="email")
    default_scopes = Column(JSON, nullable=True)  # List[str]
