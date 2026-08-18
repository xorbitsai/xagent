"""Encryption utilities for sensitive data."""

import base64
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


def _is_encrypted(value: str) -> bool:
    """True if value is one of our Fernet tokens (decrypts cleanly).

    Makes encryption idempotent without a brittle prefix check: a plaintext
    value that merely looks like a token fails the HMAC and is treated as
    plaintext, so it still gets encrypted at rest.
    """
    try:
        get_cipher().decrypt(value.encode())
        return True
    except Exception:
        return False


def encrypt_value(value: str) -> str:
    """Encrypt a string value. Idempotent: an already-encrypted value is returned as-is."""
    if not value or _is_encrypted(value):
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


def encrypt_env_dict(env: dict | None) -> dict | None:
    """Encrypt env var values at rest (encrypt_value skips empty/already-encrypted)."""
    if not env:
        return env
    return {k: (encrypt_value(v) if isinstance(v, str) else v) for k, v in env.items()}


def decrypt_env_dict(env: dict | None) -> dict | None:
    """Decrypt env var values for runtime/consumption."""
    if not env:
        return env
    return {k: (decrypt_value(v) if isinstance(v, str) else v) for k, v in env.items()}


class EncryptionDecodeError(ValueError):
    """A Fernet-shaped value could not be decrypted with the configured key.

    Raised only by the strict decryption helpers. It never carries the value
    or any part of it: these are credentials.
    """


# A Fernet token is url-safe base64 over: 1 version byte (0x80) + 8 timestamp
# bytes + 16 IV bytes + at least one 16-byte AES-CBC block + 32 HMAC bytes,
# so the shortest possible token decodes to 73 bytes.
_FERNET_VERSION_BYTE = 0x80
_FERNET_MIN_TOKEN_BYTES = 73


def _looks_like_fernet_token(value: str) -> bool:
    """True if value has the structure of a Fernet token, ignoring the key.

    Deliberately not `_is_encrypted`: that one answers "does the configured
    key open this", which is exactly the question the strict helpers must
    not conflate with "is this a token at all". A token produced under a
    different key fails `_is_encrypted` while still being a token.

    The check is the structural half of what Fernet itself does before it
    touches the key: url-safe base64, a leading version byte of 0x80, and
    enough bytes to hold version, timestamp, IV, one ciphertext block and
    the HMAC. A value that fails any of these cannot be opened by any key,
    so treating it as plaintext loses nothing.
    """
    try:
        data = base64.urlsafe_b64decode(value)
    except (ValueError, TypeError):
        return False
    return len(data) >= _FERNET_MIN_TOKEN_BYTES and data[0] == _FERNET_VERSION_BYTE


def decrypt_value_strict(encrypted_value: str) -> str:
    """Decrypt a value, raising when a token cannot be opened.

    Same result as `decrypt_value` for every input that one handles
    successfully, and the same pass-through for values that are not tokens
    (empty, None, plaintext). The one difference is the case `decrypt_value`
    cannot report: a value that is structurally a Fernet token but does not
    decrypt under the configured key raises `EncryptionDecodeError` instead
    of being returned unchanged.

    Use this where returning ciphertext to the caller would be worse than
    failing -- for example before handing a value to a subprocess
    environment, or when deciding whether stored data is still readable.
    """
    if not encrypted_value:
        return encrypted_value
    # Shape first, key second. A value that is not token-shaped is passed
    # through without ever asking for a key, so a missing or malformed
    # ENCRYPTION_KEY cannot turn plaintext into an error; and because the
    # shape check never encodes the value itself, a string that cannot be
    # UTF-8 encoded is simply "not a token" here rather than a raw
    # UnicodeEncodeError. Only a token-shaped value needs the cipher, and for
    # it a key configuration fault is exactly what should surface.
    if not _looks_like_fernet_token(encrypted_value):
        logger.debug("Value is not a Fernet token; returning it unchanged")
        return encrypted_value
    cipher = get_cipher()
    try:
        plaintext = cipher.decrypt(encrypted_value.encode())
    except InvalidToken as exc:
        raise EncryptionDecodeError(
            "Value is a Fernet token but could not be decrypted with the "
            "configured encryption key"
        ) from exc
    # Decode outside any except block so the error carries no reference to
    # the decrypted bytes: UnicodeDecodeError keeps them in .object/.args, and
    # `raise ... from None` would still leave it reachable via __context__.
    text: str | None
    try:
        text = plaintext.decode()
    except UnicodeDecodeError:
        text = None
    if text is None:
        raise EncryptionDecodeError(
            "Value is a Fernet token that decrypted, but its content is not "
            "valid UTF-8 text"
        )
    return text


def decrypt_env_dict_strict(env: dict | None) -> dict | None:
    """Decrypt env var values, raising on the first undecryptable token.

    All-or-nothing on purpose: it does not drop or replace bad entries.
    Deciding what a partially readable env map means -- skip the server,
    surface an error, fall back to another layer -- belongs to the caller,
    so this only reports that the map is not fully readable.
    """
    if not env:
        return env
    return {
        k: (decrypt_value_strict(v) if isinstance(v, str) else v)
        for k, v in env.items()
    }
