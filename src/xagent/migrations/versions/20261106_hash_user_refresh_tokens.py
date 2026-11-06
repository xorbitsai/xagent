"""Store only a SHA-256 digest of each user's refresh token.

users.refresh_token held the refresh JWT itself, so a copy of the database
yielded tokens that could be redeemed directly. Convert every stored token to
its digest in place; the application hashes the token a client presents before
comparing, so signed-in users keep their sessions.

Revision ID: 20261106_hash_user_refresh_tokens
Revises: 20261106_api_idempotency_keys
"""

import hashlib
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261106_hash_user_refresh_tokens"
down_revision: Union[str, None] = "20261106_api_idempotency_keys"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Rows are converted in keyset batches to bound memory. env.py runs the whole
# upgrade in one transaction, so the locks on converted rows are held until it
# commits; the conditional UPDATE below keeps a re-run or a concurrent writer safe.
BATCH_SIZE = 500

users = sa.table(
    "users",
    sa.column("id", sa.Integer),
    sa.column("refresh_token", sa.String),
    sa.column("refresh_token_expires_at", sa.DateTime(timezone=True)),
)

# A JWT always contains "."; a hex digest never does. This is what tells a
# plaintext token from an already converted row, so the upgrade can be re-run.
_PLAINTEXT = users.c.refresh_token.like("%.%")


def _has_refresh_token_column() -> bool:
    inspector = sa.inspect(op.get_bind())
    if "users" not in inspector.get_table_names():
        return False
    return "refresh_token" in {c["name"] for c in inspector.get_columns("users")}


def upgrade() -> None:
    if not _has_refresh_token_column():
        return
    connection = op.get_bind()
    last_id = 0
    while True:
        rows = connection.execute(
            sa.select(users.c.id, users.c.refresh_token)
            .where(users.c.id > last_id, _PLAINTEXT)
            .order_by(users.c.id)
            .limit(BATCH_SIZE)
        ).all()
        if not rows:
            return
        for user_id, token in rows:
            # Matching on the value read keeps a concurrent login or refresh,
            # which stores a new digest, from being overwritten.
            connection.execute(
                sa.update(users)
                .where(users.c.id == user_id, users.c.refresh_token == token)
                .values(refresh_token=hashlib.sha256(token.encode()).hexdigest())
            )
        last_id = rows[-1][0]


def downgrade() -> None:
    # A digest cannot be turned back into the token. Revoke the converted
    # sessions so the previous release never compares against a digest.
    if not _has_refresh_token_column():
        return
    op.get_bind().execute(
        sa.update(users)
        .where(users.c.refresh_token.is_not(None), ~_PLAINTEXT)
        .values(refresh_token=None, refresh_token_expires_at=None)
    )
