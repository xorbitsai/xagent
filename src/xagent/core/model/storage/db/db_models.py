from __future__ import annotations

import os
from typing import Any, Optional, Type

from cryptography.fernet import Fernet
from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func


def create_model_table(Base: Type[Any]) -> Type[Any]:
    """
    Factory function to create model table with any SQLAlchemy Base class.

    Args:
        Base: SQLAlchemy declarative base class

    Returns:
        Model class
    """

    class Model(Base):
        """Model configuration table for storing AI model settings"""

        __tablename__ = "models"

        id = Column(Integer, primary_key=True, index=True)
        model_id = Column(String(100), unique=True, index=True, nullable=False)
        category = Column(
            String(20), nullable=False, default="llm"
        )  # llm, image, embedding, etc.
        model_provider = Column(
            String(50), nullable=False
        )  # openai, zhipu, dashscope, etc.
        model_name = Column(String(100), nullable=False)  # gpt-4, glm-4, etc.
        base_url = Column(String(500), nullable=True)
        temperature = Column(Float, nullable=True)
        max_tokens = Column(Integer, nullable=True)
        context_window = Column(
            Integer, nullable=True
        )  # Total context window in tokens
        dimension = Column(
            Integer, nullable=True
        )  # Vector dimension for embedding models
        abilities = Column(
            JSON, nullable=True
        )  # Model abilities: ["chat", "vision", etc.]
        description = Column(Text, nullable=True)
        max_retries = Column(Integer, nullable=True, default=10)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        updated_at = Column(DateTime(timezone=True), onupdate=func.now())
        is_active = Column(Boolean, default=True)
        _api_key_encrypted = Column(String(500), nullable=False)

        # Properties
        @property
        def is_visual(self) -> bool:
            """Check if this model has vision ability"""
            if not self.abilities:
                return False
            return "vision" in self.abilities

        def __repr__(self) -> str:
            return f"<Model(id={self.id}, model_id='{self.model_id}', model_name='{self.model_name}')>"

        def _get_encryption_key(self) -> str:
            encryption_key = os.getenv("ENCRYPTION_KEY")
            if not encryption_key:
                # FIXME: For dev only
                return "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="
            return encryption_key

        @property
        def api_key(self) -> Optional[str]:
            encryption_key = self._get_encryption_key()
            cipher = Fernet(
                encryption_key.encode()
                if isinstance(encryption_key, str)
                else encryption_key
            )
            return cipher.decrypt(self._api_key_encrypted.encode()).decode()

        @api_key.setter
        def api_key(self, value: Optional[str]) -> None:
            if value is None:
                return
            encryption_key = self._get_encryption_key()
            cipher = Fernet(
                encryption_key.encode()
                if isinstance(encryption_key, str)
                else encryption_key
            )
            self._api_key_encrypted = cipher.encrypt(value.encode()).decode()  # type: ignore[assignment]

    return Model
