"""rename model_type to model_provider and encrypt api_key

Revision ID: 441d4f5d399c
Revises:
Create Date: 2025-10-27 16:48:36.971068

"""

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from cryptography.fernet import Fernet
from sqlalchemy import text

revision: str = "441d4f5d399c"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def get_cipher() -> Fernet:
    key = os.environ.get("ENCRYPTION_KEY")
    if not key:
        # FIXME: For dev only
        key = "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="
    return Fernet(key.encode())


def upgrade() -> None:
    from sqlalchemy.engine.reflection import Inspector

    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if models table exists
    tables = inspector.get_table_names()
    if "models" not in tables:
        # Table doesn't exist yet, will be created by SQLAlchemy
        return

    # Get current columns in models table
    columns = [col["name"] for col in inspector.get_columns("models")]

    cipher = get_cipher()

    # Check if migration is already applied
    if "model_provider" in columns and "_api_key_encrypted" in columns:
        # Already migrated, skip
        return

    # Check if we need to do the migration
    if "model_type" in columns and "model_provider" not in columns:
        # Old schema: need to migrate
        with op.batch_alter_table("models") as batch_op:
            # Add encrypted column
            if "_api_key_encrypted" not in columns:
                batch_op.add_column(
                    sa.Column(
                        "_api_key_encrypted",
                        sa.String(500),
                        nullable=False,
                        server_default="",
                    )
                )
            # Rename column
            batch_op.alter_column("model_type", new_column_name="model_provider")

        # Encrypt existing keys
        result = bind.execute(text("SELECT id, api_key FROM models"))
        for row in result.fetchall():
            if row.api_key:
                encrypted = cipher.encrypt(row.api_key.encode()).decode()
                bind.execute(
                    text("UPDATE models SET _api_key_encrypted = :enc WHERE id = :id"),
                    {"enc": encrypted, "id": row.id},
                )

        # Drop old column
        with op.batch_alter_table("models") as batch_op:
            batch_op.drop_column("api_key")

    elif "api_key" in columns and "_api_key_encrypted" not in columns:
        # Has api_key but not encrypted yet, just add encryption
        with op.batch_alter_table("models") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "_api_key_encrypted",
                    sa.String(500),
                    nullable=False,
                    server_default="",
                )
            )

        # Encrypt existing keys
        result = bind.execute(text("SELECT id, api_key FROM models"))
        for row in result.fetchall():
            if row.api_key:
                encrypted = cipher.encrypt(row.api_key.encode()).decode()
                bind.execute(
                    text("UPDATE models SET _api_key_encrypted = :enc WHERE id = :id"),
                    {"enc": encrypted, "id": row.id},
                )

        # Drop old column
        with op.batch_alter_table("models") as batch_op:
            batch_op.drop_column("api_key")


def downgrade() -> None:
    from sqlalchemy.engine.reflection import Inspector

    cipher = get_cipher()
    bind = op.get_bind()
    inspector = Inspector.from_engine(bind)

    # Check if models table exists
    tables = inspector.get_table_names()
    if "models" not in tables:
        return

    with op.batch_alter_table("models") as batch_op:
        # Add back plain column
        batch_op.add_column(
            sa.Column("api_key", sa.String(500), nullable=False, server_default="")
        )
        # Rename back
        batch_op.alter_column("model_provider", new_column_name="model_type")

    # Decrypt keys
    result = bind.execute(text("SELECT id, _api_key_encrypted FROM models"))
    for row in result.fetchall():
        if row._api_key_encrypted:
            decrypted = cipher.decrypt(row._api_key_encrypted.encode()).decode()
            bind.execute(
                text("UPDATE models SET api_key = :dec WHERE id = :id"),
                {"dec": decrypted, "id": row.id},
            )

    # Drop encrypted column
    with op.batch_alter_table("models") as batch_op:
        batch_op.drop_column("_api_key_encrypted")
