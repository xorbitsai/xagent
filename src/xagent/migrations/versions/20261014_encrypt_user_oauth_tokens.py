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
    user_oauth = sa.table(
        "user_oauth",
        sa.column("id", sa.Integer),
        sa.column("access_token", sa.String),
        sa.column("refresh_token", sa.String),
    )
    rows = list(
        connection.execute(
            sa.select(
                user_oauth.c.id,
                user_oauth.c.access_token,
                user_oauth.c.refresh_token,
            )
        ).mappings()
    )
    for row in rows:
        connection.execute(
            user_oauth.update()
            .where(user_oauth.c.id == row["id"])
            .values(
                access_token=encrypt_value(str(row["access_token"] or "")),
                refresh_token=(
                    encrypt_value(str(row["refresh_token"]))
                    if row["refresh_token"]
                    else None
                ),
            )
        )


def downgrade() -> None:
    # Token ciphertext must never be written back as plaintext during a
    # downgrade. The ORM remains able to decrypt rows from this migration.
    pass
