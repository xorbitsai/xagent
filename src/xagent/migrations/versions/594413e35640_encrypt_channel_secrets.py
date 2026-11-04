"""encrypt channel secrets

Revision ID: 594413e35640
Revises: 7f6d2ffea948
Create Date: 2026-04-01 19:07:38.158190

"""

import json
import os
from typing import Sequence, Union

from alembic import op
from cryptography.fernet import Fernet
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "594413e35640"
down_revision: Union[str, None] = "7f6d2ffea948"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def get_cipher() -> Fernet:
    key = os.environ.get("ENCRYPTION_KEY")
    if not key:
        # FIXME: For dev only
        key = "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="
    return Fernet(key.encode() if isinstance(key, str) else key)


def upgrade() -> None:
    bind = op.get_bind()
    from sqlalchemy.engine.reflection import Inspector

    inspector = Inspector.from_engine(bind)

    if "user_channels" not in inspector.get_table_names():
        return

    cipher = get_cipher()

    result = bind.execute(text("SELECT id, config FROM user_channels"))
    for row in result.fetchall():
        config_str = row.config
        if not config_str:
            continue

        try:
            config = (
                json.loads(config_str) if isinstance(config_str, str) else config_str
            )
        except Exception:
            continue

        updated = False

        if config.get("bot_token"):
            try:
                cipher.decrypt(config["bot_token"].encode())
            except Exception:
                config["bot_token"] = cipher.encrypt(
                    config["bot_token"].encode()
                ).decode()
                updated = True

        if config.get("app_secret"):
            try:
                cipher.decrypt(config["app_secret"].encode())
            except Exception:
                config["app_secret"] = cipher.encrypt(
                    config["app_secret"].encode()
                ).decode()
                updated = True

        if updated:
            bind.execute(
                text("UPDATE user_channels SET config = :config WHERE id = :id"),
                {"config": json.dumps(config), "id": row.id},
            )


def downgrade() -> None:
    bind = op.get_bind()
    from sqlalchemy.engine.reflection import Inspector

    inspector = Inspector.from_engine(bind)

    if "user_channels" not in inspector.get_table_names():
        return

    cipher = get_cipher()

    result = bind.execute(text("SELECT id, config FROM user_channels"))
    for row in result.fetchall():
        config_str = row.config
        if not config_str:
            continue

        try:
            config = (
                json.loads(config_str) if isinstance(config_str, str) else config_str
            )
        except Exception:
            continue

        updated = False

        if config.get("bot_token"):
            try:
                decrypted = cipher.decrypt(config["bot_token"].encode()).decode()
                config["bot_token"] = decrypted
                updated = True
            except Exception:
                pass

        if config.get("app_secret"):
            try:
                decrypted = cipher.decrypt(config["app_secret"].encode()).decode()
                config["app_secret"] = decrypted
                updated = True
            except Exception:
                pass

        if updated:
            bind.execute(
                text("UPDATE user_channels SET config = :config WHERE id = :id"),
                {"config": json.dumps(config), "id": row.id},
            )
