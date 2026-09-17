from sqlalchemy import (
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


class AutoModelConfig(Base):  # type: ignore
    """The single configured Auto router owned by one user."""

    __tablename__ = "auto_model_configs"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_auto_model_config_user"),
        UniqueConstraint("router_model_id", name="uq_auto_model_config_router_model"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    router_model_id = Column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    strategy = Column(String(20), nullable=False, default="balanced")
    fallback_model_id = Column(
        Integer, ForeignKey("models.id", ondelete="RESTRICT"), nullable=True
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    router_model = relationship("Model", foreign_keys=[router_model_id])
    fallback_model = relationship("Model", foreign_keys=[fallback_model_id])
    candidates = relationship(
        "AutoModelCandidate",
        back_populates="config",
        cascade="all, delete-orphan",
        order_by="AutoModelCandidate.id",
    )


class AutoModelCandidate(Base):  # type: ignore
    """Bind one xrouter profile to one concrete saved model config."""

    __tablename__ = "auto_model_candidates"
    __table_args__ = (
        UniqueConstraint(
            "config_id",
            "routing_model_id",
            name="uq_auto_candidate_routing_model",
        ),
        UniqueConstraint(
            "config_id",
            "target_model_id",
            name="uq_auto_candidate_target_model",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(
        Integer,
        ForeignKey("auto_model_configs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    routing_model_id = Column(String(200), nullable=False)
    target_model_id = Column(
        Integer, ForeignKey("models.id", ondelete="RESTRICT"), nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    config = relationship("AutoModelConfig", back_populates="candidates")
    target_model = relationship("Model", foreign_keys=[target_model_id])
