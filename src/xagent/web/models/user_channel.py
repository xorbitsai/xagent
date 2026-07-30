import copy
import os
from typing import Any, cast

from cryptography.fernet import Fernet
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base

# Dev-only fallback so local setups work without configuration. Production
# deployments must set ENCRYPTION_KEY; has_production_channel_encryption_key()
# gates features (such as Slack workspace OAuth) that persist third-party
# tokens on a real key being configured.
_DEV_FALLBACK_ENCRYPTION_KEY = "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="


def has_production_channel_encryption_key() -> bool:
    """Report whether a non-default channel-config encryption key is set."""
    encryption_key = os.getenv("ENCRYPTION_KEY")
    return bool(encryption_key) and encryption_key != _DEV_FALLBACK_ENCRYPTION_KEY


def _get_cipher() -> Fernet:
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        # FIXME: For dev only
        encryption_key = _DEV_FALLBACK_ENCRYPTION_KEY
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


class SlackOAuthFlowState(Base):  # type: ignore[no-any-unimported]
    """Single-use nonce ledger for the Slack workspace OAuth flow.

    The authorization URL carries a signed JWT state; this row makes that
    state single-use — the callback atomically claims the nonce, so a
    replayed or attacker-forwarded state is rejected after first use.
    """

    __tablename__ = "slack_oauth_flow_states"

    id = Column(Integer, primary_key=True)
    nonce = Column(String(64), nullable=False, unique=True, index=True)
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    consumed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<SlackOAuthFlowState(id={self.id}, user_id={self.user_id})>"
