import copy
import os

from cryptography.fernet import Fernet
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.ext.hybrid import hybrid_property
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
    """User Channels configurations (e.g. Telegram Bot, Feishu)"""

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

    @hybrid_property
    def config(self) -> dict:
        if not self._config:
            return {}
        cipher = _get_cipher()
        config_copy = copy.deepcopy(self._config)

        # Decrypt sensitive fields
        if config_copy.get("bot_token"):
            try:
                config_copy["bot_token"] = cipher.decrypt(
                    config_copy["bot_token"].encode()
                ).decode()
            except Exception:
                pass  # Fallback to plaintext if not encrypted

        if config_copy.get("app_secret"):
            try:
                config_copy["app_secret"] = cipher.decrypt(
                    config_copy["app_secret"].encode()
                ).decode()
            except Exception:
                pass  # Fallback to plaintext if not encrypted

        return config_copy

    @config.setter  # type: ignore[no-redef]
    def config(self, value: dict) -> None:
        if not value:
            self._config = value
            return
        cipher = _get_cipher()
        config_copy = copy.deepcopy(value)

        # Encrypt sensitive fields
        if config_copy.get("bot_token"):
            try:
                cipher.decrypt(config_copy["bot_token"].encode())
            except Exception:
                config_copy["bot_token"] = cipher.encrypt(
                    config_copy["bot_token"].encode()
                ).decode()

        if config_copy.get("app_secret"):
            try:
                cipher.decrypt(config_copy["app_secret"].encode())
            except Exception:
                config_copy["app_secret"] = cipher.encrypt(
                    config_copy["app_secret"].encode()
                ).decode()

        self._config = config_copy

    def __repr__(self) -> str:
        return f"<UserChannel(user_id={self.user_id}, type='{self.channel_type}', name='{self.channel_name}')>"
