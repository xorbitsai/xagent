"""Wrap legacy unsalted SHA-256 password hashes in bcrypt.

Older releases stored ``users.password_hash`` as a bare, unsalted
``sha256(password).hexdigest()``. Those rows were only upgraded to bcrypt the
next time their owner logged in, so dormant accounts kept a fast, unsalted
digest at rest indefinitely.

The current ``hash_password`` format is
``bcrypt(sha256(password).hexdigest())``. A legacy row already holds exactly
that inner value, so it can be wrapped without knowing the password:
``bcrypt(stored_hex)`` is indistinguishable from a freshly hashed password and
verifies through the normal bcrypt branch of ``verify_password``.

Only values that are exactly 64 hex characters are rewritten. bcrypt rows
(``$2a$``/``$2b$``/``$2y$``) and unusable-password markers such as ``!`` are
left untouched, which also makes the upgrade idempotent. The hex digest is
lowercased first because ``hexdigest()`` always yields lowercase.

Downgrade is a no-op: the original digest cannot be recovered from a bcrypt
hash, and the wrapped rows remain valid for the previous release, which already
verifies bcrypt rows with the same SHA-256 pre-hash.

Revision ID: 20261027_wrap_legacy_password_hashes
Revises: 20261027_employment_hero_leave
Create Date: 2026-10-27 00:00:00.000000
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Union

import bcrypt
import sqlalchemy as sa
from alembic import op

revision: str = "20261027_wrap_legacy_password_hashes"
down_revision: Union[str, None] = "20261027_employment_hero_leave"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

BATCH_SIZE = 200
LEGACY_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


def _wrap_legacy_digest(legacy_digest: str) -> str:
    """Return the current password-hash format for a legacy SHA-256 digest.

    Uses the default ``bcrypt.gensalt()`` cost, matching ``hash_password``.
    """
    return bcrypt.hashpw(
        legacy_digest.lower().encode("ascii"), bcrypt.gensalt()
    ).decode("ascii")


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    if "users" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("users")}
    if not {"id", "password_hash"}.issubset(columns):
        return

    users = sa.table(
        "users",
        sa.column("id", sa.Integer),
        sa.column("password_hash", sa.String),
    )
    # bcrypt releases the GIL, so a small pool cuts the wall-clock cost of the
    # deliberately slow hash without changing its parameters.
    workers = max(1, min(8, os.cpu_count() or 1))
    last_id = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            query = sa.select(users.c.id, users.c.password_hash).order_by(users.c.id)
            if last_id is not None:
                query = query.where(users.c.id > last_id)
            batch = connection.execute(query.limit(BATCH_SIZE)).all()
            if not batch:
                break
            last_id = batch[-1].id

            legacy_rows = [
                (row.id, row.password_hash)
                for row in batch
                if row.password_hash and LEGACY_SHA256_HEX.fullmatch(row.password_hash)
            ]
            if not legacy_rows:
                continue
            wrapped = pool.map(
                _wrap_legacy_digest, [digest for _, digest in legacy_rows]
            )
            for (user_id, legacy_digest), new_hash in zip(legacy_rows, wrapped):
                # Match the old value so a concurrent login-time rehash wins.
                connection.execute(
                    users.update()
                    .where(users.c.id == user_id)
                    .where(users.c.password_hash == legacy_digest)
                    .values(password_hash=new_hash)
                )


def downgrade() -> None:
    """Irreversible by design; wrapped rows stay valid for older releases.

    bcrypt is one-way, so the bare digest cannot be restored, and doing so
    would reintroduce the unsalted at-rest hashes this revision removes.
    """
