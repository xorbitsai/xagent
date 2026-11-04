from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


class UserSkill(Base):  # type: ignore
    """Personal database-backed skill owned by one xagent user."""

    __tablename__ = "user_skills"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_skill_name"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name = Column(String(100), nullable=False)
    origin = Column(String(50), nullable=False, default="custom")
    clawhub_slug = Column(String(128), nullable=True)
    clawhub_version = Column(String(64), nullable=True)
    skill_metadata = Column(JSON, nullable=True, default=dict)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    files = relationship(
        "UserSkillFile",
        back_populates="skill",
        cascade="all, delete-orphan",
        order_by="UserSkillFile.path",
    )


class UserSkillFile(Base):  # type: ignore
    """One file blob belonging to a personal skill."""

    __tablename__ = "user_skill_files"
    __table_args__ = (
        UniqueConstraint("skill_id", "path", name="uq_user_skill_file_path"),
    )

    id = Column(Integer, primary_key=True, index=True)
    skill_id = Column(
        Integer, ForeignKey("user_skills.id", ondelete="CASCADE"), nullable=False
    )
    path = Column(String(500), nullable=False)
    content = Column(LargeBinary, nullable=False)
    size_bytes = Column(Integer, nullable=False)
    sha256 = Column(String(64), nullable=False)
    media_type = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    skill = relationship("UserSkill", back_populates="files")
