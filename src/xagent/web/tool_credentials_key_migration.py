"""Offline, explicit re-encryption of tool credentials; never a runtime fallback."""

import base64
import copy
import hashlib
import os
from typing import cast

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import TOOL_CREDENTIALS_MIGRATION_KEY, get_tool_credentials_encryption_key
from .models.database import configure_db, get_session_local
from .models.tool_config import ToolConfig, UserToolConfig


def _cipher(raw: str) -> Fernet:
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest()))


def migrate_tool_credentials(db: Session, old_key: str) -> int:
    """Own one all-or-nothing transaction. Stop writers and back up first."""
    destination = get_tool_credentials_encryption_key()
    # This offline tool must never downgrade data to a development fallback,
    # even when an operator forgot to select the production environment.
    if len(destination.strip()) < 32:
        raise RuntimeError(
            "Configure a private 32+ character destination tool key before migration"
        )
    old, new = _cipher(old_key), _cipher(destination)
    count = 0
    with db.begin():
        for model in (ToolConfig, UserToolConfig):
            for record in db.query(model).all():
                row = cast(ToolConfig | UserToolConfig, record)
                payload = copy.deepcopy(row.config)
                if not isinstance(payload, dict):
                    continue
                changed = False
                for group in ("credentials", "sql_connections"):
                    fields = payload.get(group)
                    if not isinstance(fields, dict):
                        continue
                    for item in fields.values():
                        if not isinstance(item, dict) or not isinstance(
                            item.get("ciphertext"), str
                        ):
                            continue
                        encoded = item["ciphertext"].encode()
                        try:
                            new.decrypt(encoded)
                            continue
                        except InvalidToken:
                            pass
                        try:
                            plaintext = old.decrypt(encoded)
                        except InvalidToken:
                            raise RuntimeError(
                                "A stored credential cannot be decrypted; migration rolled back"
                            ) from None
                        item["ciphertext"] = new.encrypt(plaintext).decode()
                        changed = True
                        count += 1
                if changed:
                    row.config = payload
    return count


def main() -> None:
    old_key = os.getenv(TOOL_CREDENTIALS_MIGRATION_KEY)
    if not old_key:
        raise SystemExit(f"Set {TOOL_CREDENTIALS_MIGRATION_KEY} for this offline run")
    configure_db()
    try:
        with get_session_local()() as db:
            count = migrate_tool_credentials(db, old_key)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    except SQLAlchemyError as exc:
        raise SystemExit(
            f"Database migration failed ({type(exc).__name__}); transaction rolled back"
        ) from None
    print(f"Re-encrypted {count} tool credential fields.")


if __name__ == "__main__":
    main()
