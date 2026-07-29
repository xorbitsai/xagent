import copy
import os
from typing import Any, cast

from cryptography.fernet import Fernet
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base


def _get_cipher() -> Fernet:
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        # FIXME: For dev only
        encryption_key = "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="
    return Fernet(
        encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
    )


class UserChannel(Base):  # type: ignore[no-any-unimported]
    """User Channels configurations (e.g. Telegram, Feishu, Slack)."""

    __tablename__ = "user_channels"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    channel_type = Column(String(50), nullable=False)  # e.g. "telegram"
    channel_name = Column(String(100), nullable=False)  # User-friendly name
    _config = Column("config", JSON, nullable=False)  # e.g. {"bot_token": "..."}
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="channels")

    @property
    def config(self) -> dict:
        if not self._config:
            return {}
        cipher = _get_cipher()
        raw_config = cast(dict[str, Any], self._config)
        config_copy = copy.deepcopy(raw_config)

        # Decrypt sensitive fields
        for field in ("bot_token", "app_secret", "app_token"):
            if not config_copy.get(field):
                continue
            try:
                config_copy[field] = cipher.decrypt(
                    config_copy[field].encode()
                ).decode()
            except Exception:
                pass  # Fallback to plaintext if not encrypted

        return config_copy

    @config.setter
    def config(self, value: dict) -> None:
        if not value:
            self._config = value  # type: ignore[assignment]
            return
        cipher = _get_cipher()
        config_copy = copy.deepcopy(value)

        # Encrypt sensitive fields
        for field in ("bot_token", "app_secret", "app_token"):
            if not config_copy.get(field):
                continue
            try:
                cipher.decrypt(config_copy[field].encode())
            except Exception:
                config_copy[field] = cipher.encrypt(
                    config_copy[field].encode()
                ).decode()

        self._config = config_copy  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"<UserChannel(user_id={self.user_id}, type='{self.channel_type}', name='{self.channel_name}')>"
