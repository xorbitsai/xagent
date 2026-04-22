from sqlalchemy import JSON, Column, Integer, String, Text

from .database import Base


class PublicMCPApp(Base):  # type: ignore[no-any-unimported]
    """Registry of official MCP apps available for users to connect to."""

    __tablename__ = "public_mcp_apps"

    id = Column(Integer, primary_key=True, index=True)
    app_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    icon = Column(String(1000), nullable=True)
    transport = Column(String(50), default="oauth", nullable=False)

    # Optional FK to OAuthProvider
    provider_name = Column(String(50), nullable=True)

    category = Column(String(100), nullable=True)
    oauth_scopes = Column(JSON, nullable=True)  # List[str]
    launch_config = Column(
        JSON, nullable=True
    )  # Dict e.g., {"command": "npx", "args": ["..."]}
