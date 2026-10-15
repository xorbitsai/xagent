"""Encrypt existing user OAuth access and refresh tokens at rest."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.core.utils.encryption import (
    EncryptionDecodeError,
    decrypt_value_strict,
    encrypt_value,
)

revision: str = "20261014_encrypt_user_oauth_tokens"
down_revision: Union[str, None] = "20261015_sync_xero_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _encrypt_legacy_token(value: str | None) -> str | None:
    """Encrypt plaintext while preserving ciphertext from another key.

    ``encrypt_value`` is idempotent only for the currently configured key. A
    retry during a key rotation must not wrap an already encrypted token with
    the new key, because the original ciphertext is the only recoverable copy
    until the deployment restores the old key.
    """
    if not value:
        return value
    try:
        decrypted = decrypt_value_strict(value)
    except EncryptionDecodeError:
        # This is ciphertext from another key. Leave it untouched so the
        # owning key can still recover it after configuration is corrected.
        return value
    if decrypted != value:
        return value
    return encrypt_value(value)


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if "user_oauth" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("user_oauth")}
    if not {"id", "access_token"}.issubset(columns):
        return

    user_oauth = sa.table(
        "user_oauth",
        sa.column("id", sa.Integer),
        sa.column("access_token", sa.String),
        sa.column("refresh_token", sa.String),
    )
    selected_columns = [user_oauth.c.id, user_oauth.c.access_token]
    if "refresh_token" in columns:
        selected_columns.append(user_oauth.c.refresh_token)
    result = connection.execute(sa.select(*selected_columns)).mappings().all()
    for row in result:
        values: dict[str, str | None] = {}
        access_token = row["access_token"]
        encrypted_access_token = _encrypt_legacy_token(access_token)
        if encrypted_access_token != access_token:
            values["access_token"] = encrypted_access_token
        if "refresh_token" in columns:
            refresh_token = row["refresh_token"]
            encrypted_refresh_token = _encrypt_legacy_token(refresh_token)
            if encrypted_refresh_token != refresh_token:
                values["refresh_token"] = encrypted_refresh_token
        if values:
            connection.execute(
                user_oauth.update().where(user_oauth.c.id == row["id"]).values(**values)
            )


def downgrade() -> None:
    """Restore plaintext tokens for the pre-encryption ORM implementation."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if "user_oauth" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("user_oauth")}
    if not {"id", "access_token"}.issubset(columns):
        return

    user_oauth = sa.table(
        "user_oauth",
        sa.column("id", sa.Integer),
        sa.column("access_token", sa.String),
        sa.column("refresh_token", sa.String),
    )
    selected_columns = [user_oauth.c.id, user_oauth.c.access_token]
    if "refresh_token" in columns:
        selected_columns.append(user_oauth.c.refresh_token)
    result = connection.execute(sa.select(*selected_columns)).mappings().all()
    for row in result:
        values: dict[str, str | None] = {}
        access_token = row["access_token"]
        plaintext_access_token = (
            decrypt_value_strict(access_token) if access_token else access_token
        )
        if plaintext_access_token != access_token:
            values["access_token"] = plaintext_access_token
        if "refresh_token" in columns:
            refresh_token = row["refresh_token"]
            plaintext_refresh_token = (
                decrypt_value_strict(refresh_token) if refresh_token else refresh_token
            )
            if plaintext_refresh_token != refresh_token:
                values["refresh_token"] = plaintext_refresh_token
        if values:
            connection.execute(
                user_oauth.update().where(user_oauth.c.id == row["id"]).values(**values)
            )
