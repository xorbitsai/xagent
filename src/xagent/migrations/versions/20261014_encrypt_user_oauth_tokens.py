"""Encrypt existing user OAuth access and refresh tokens at rest.

The ORM token type encrypts all future writes. This migration covers rows
created before that type was introduced; ``encrypt_value`` is idempotent so a
retry is safe after a partially completed deployment.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.core.utils.encryption import encrypt_value

revision: str = "20261014_encrypt_user_oauth_tokens"
down_revision: Union[str, None] = "20261012_zendesk_oauth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


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
    result = connection.execute(sa.select(*selected_columns)).mappings()
    for row in result:
        values: dict[str, str | None] = {
            "access_token": encrypt_value(str(row["access_token"] or "")),
        }
        if "refresh_token" in columns:
            values["refresh_token"] = (
                encrypt_value(str(row["refresh_token"]))
                if row["refresh_token"]
                else None
            )
        connection.execute(
            user_oauth.update().where(user_oauth.c.id == row["id"]).values(**values)
        )


def downgrade() -> None:
    # Token ciphertext must never be written back as plaintext during a
    # downgrade. The ORM remains able to decrypt rows from this migration.
    pass
