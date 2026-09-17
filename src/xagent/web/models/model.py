from __future__ import annotations

from typing import TYPE_CHECKING, Any, Type

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from xagent.core.model.storage.db.db_models import create_model_table

if TYPE_CHECKING:
    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()

    class Model(Base):  # type: ignore[valid-type, misc]
        id = Column(Integer, primary_key=True, index=True)
        model_id = Column(String(100), unique=True, index=True, nullable=False)
        category = Column(String(20), nullable=False, default="llm")
        model_provider = Column(String(50), nullable=False)
        model_name = Column(String(100), nullable=False)
        api_key = Column(String(500), nullable=False)
        base_url = Column(String(500), nullable=True)
        temperature = Column(Float, nullable=True)
        dimension = Column(Integer, nullable=True)
        abilities = Column(JSON, nullable=True)
        description = Column(Text, nullable=True)
        managed_by = Column(String(50), nullable=True)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        updated_at = Column(DateTime(timezone=True), onupdate=func.now())
        is_active = Column(Boolean, default=True)
        user_models = relationship(
            "UserModel",
            back_populates="model",
            cascade="all, delete-orphan",
        )
        user_default_models = relationship(
            "UserDefaultModel",
            back_populates="model",
            cascade="all, delete-orphan",
        )

        @property
        def is_visual(self) -> bool:
            """Check if this model has vision ability"""
            if not self.abilities:
                return False
            return "vision" in self.abilities

        def __repr__(self) -> str:
            return f"<Model(id={self.id}, model_id='{self.model_id}', model_name='{self.model_name}')>"


else:
    from .database import Base

    Model: Type[Any] = create_model_table(Base)
    # Relationships
    Model.user_models = relationship(
        "UserModel",
        back_populates="model",
        cascade="all, delete-orphan",
    )
    Model.user_default_models = relationship(
        "UserDefaultModel",
        back_populates="model",
        cascade="all, delete-orphan",
    )
