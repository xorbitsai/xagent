"""Encryption utilities for sensitive data."""

import logging
import os
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


def _get_encryption_key() -> str:
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        env = os.getenv("ENVIRONMENT", "development")
        if env != "development":
            raise ValueError(
                "ENCRYPTION_KEY environment variable is not set in non-development environment"
            )
        # FIXME: For dev only, same as in db_models.py
        return "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="
    return encryption_key


@lru_cache()
def get_cipher() -> Fernet:
    encryption_key = _get_encryption_key()
    return Fernet(
        encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
    )


def encrypt_value(value: str) -> str:
    """Encrypt a string value."""
    if not value:
        return value
    cipher = get_cipher()
    return cipher.encrypt(value.encode()).decode()


def decrypt_value(encrypted_value: str) -> str:
    """Decrypt an encrypted string value. If it is not encrypted or invalid, return the original value."""
    if not encrypted_value:
        return encrypted_value
    try:
        cipher = get_cipher()
        return cipher.decrypt(encrypted_value.encode()).decode()
    except InvalidToken:
        logger.debug("Failed to decrypt value: Invalid token (might be plain text)")
        return encrypted_value
    except Exception as e:
        logger.debug(f"Failed to decrypt value: {e} (might be plain text)")
        return encrypted_value
