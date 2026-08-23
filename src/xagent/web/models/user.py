from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class User(Base):  # type: ignore
    """User model"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)  # Admin role flag
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    refresh_token = Column(String(255), nullable=True)
    refresh_token_expires_at = Column(DateTime(timezone=True), nullable=True)
    password_reset_token_hash = Column(String(64), index=True, nullable=True)
    password_reset_expires_at = Column(DateTime(timezone=True), nullable=True)
    # Onboarding-collected settings: {onboarded, department, industry, voice,
    # goals}. Written incrementally (one PATCH per onboarding step merges
    # into this dict, not replaces it) - see update_current_user_preferences
    # in api/auth.py. `voice` feeds _apply_user_voice's system-prompt
    # injection in api/chat.py.
    preferences = Column(JSON, nullable=True, default=dict)

    # Relationships
    tasks = relationship("Task", back_populates="user")
    agents = relationship("Agent", back_populates="user")
    mcp_servers = relationship(
        "MCPServer",
        secondary="user_mcpservers",
        primaryjoin="User.id==UserMCPServer.user_id",
        secondaryjoin="MCPServer.id==UserMCPServer.mcpserver_id",
        viewonly=True,
    )
    user_mcpservers = relationship(
        "UserMCPServer", back_populates="user", cascade="all, delete-orphan"
    )
    user_custom_apis = relationship(
        "UserCustomApi", back_populates="user", cascade="all, delete-orphan"
    )
    user_models = relationship(
        "UserModel", back_populates="user", cascade="all, delete-orphan"
    )
    uploaded_files = relationship(
        "UploadedFile", back_populates="user", cascade="all, delete-orphan"
    )
    chat_messages = relationship(
        "TaskChatMessage", back_populates="user", cascade="all, delete-orphan"
    )
    user_default_models = relationship(
        "UserDefaultModel", back_populates="user", cascade="all, delete-orphan"
    )
    # This historical relationship represents the user's ordinary OAuth
    # accounts. Actor-owned credentials share the table for storage only and
    # must be loaded through explicit owner-scoped service queries. Their user
    # deletion depends on the database ``ON DELETE CASCADE``; SQLite engine
    # initialization enables and monitors the required foreign-key pragma.
    oauth_accounts = relationship(
        "UserOAuth",
        primaryjoin=(
            "and_(User.id == UserOAuth.user_id, UserOAuth.resource_owner_key.is_(None))"
        ),
        back_populates="user",
        cascade="all, delete-orphan",
        # SQL predicates do not filter Python-side back-reference updates.
        # Disable synchronization so assigning an actor row's ``user`` cannot
        # inject it into this ordinary-only delete-orphan collection.
        sync_backref=False,
    )
    identities = relationship(
        "UserIdentity", back_populates="user", cascade="all, delete-orphan"
    )
    channels = relationship(
        "UserChannel", back_populates="user", cascade="all, delete-orphan"
    )
    background_jobs = relationship(
        "BackgroundJob", back_populates="user", cascade="all, delete-orphan"
    )
    agent_triggers = relationship(
        "AgentTrigger", back_populates="user", cascade="all, delete-orphan"
    )
    tool_configs = relationship(
        "UserToolConfig", back_populates="user", cascade="all, delete-orphan"
    )
    template_relations = relationship(
        "UserTemplateRelation", back_populates="user", cascade="all, delete-orphan"
    )
    api_keys = relationship(
        "UserApiKey", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', is_admin={self.is_admin})>"


class UserModel(Base):  # type: ignore
    """User-Model relationship table for model ownership and sharing"""

    __tablename__ = "user_models"
    __table_args__ = (UniqueConstraint("user_id", "model_id", name="uq_user_model"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    model_id = Column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    is_owner = Column(
        Boolean, default=False, nullable=False
    )  # True if user created the model
    can_edit = Column(
        Boolean, default=False, nullable=False
    )  # True if user can edit the model
    can_delete = Column(
        Boolean, default=False, nullable=False
    )  # True if user can delete the model
    is_shared = Column(
        Boolean, default=False, nullable=False
    )  # True if model is shared by admin
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="user_models")
    model = relationship("Model", back_populates="user_models")

    def __repr__(self) -> str:
        return f"<UserModel(user_id={self.user_id}, model_id={self.model_id}, is_owner={self.is_owner})>"


class UserDefaultModel(Base):  # type: ignore
    """User default model configurations"""

    __tablename__ = "user_default_models"
    __table_args__ = (
        UniqueConstraint("user_id", "config_type", name="uq_user_default_model"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    model_id = Column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    config_type = Column(
        String(50), nullable=False
    )  # 'general', 'small_fast', 'visual', 'compact', 'embedding'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="user_default_models")
    model = relationship("Model", back_populates="user_default_models")

    def __repr__(self) -> str:
        return f"<UserDefaultModel(user_id={self.user_id}, config_type='{self.config_type}', model_id={self.model_id})>"
