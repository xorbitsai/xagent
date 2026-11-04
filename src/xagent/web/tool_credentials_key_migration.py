"""Offline, explicit re-encryption of tool credentials; never a runtime fallback."""

import copy
import os

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import (
    TOOL_CREDENTIALS_MIGRATION_KEY,
    get_tool_credentials_encryption_key,
    validate_tool_credentials_key,
)
from .models.database import configure_db, get_session_local
from .services.encryption_key_rotation import credential_rows, tool_credential_items
from .services.tool_credentials import _build_fernet_key


def _cipher(raw: str) -> Fernet:
    return Fernet(_build_fernet_key(raw))


def migrate_tool_credentials(db: Session, old_key: str) -> int:
    """Own one all-or-nothing transaction. Stop writers and back up first."""
    destination = get_tool_credentials_encryption_key()
    # This offline tool must never downgrade data to a development fallback,
    # even when an operator forgot to select the production environment.
    try:
        validate_tool_credentials_key(destination, setting="Destination tool key")
    except RuntimeError:
        raise RuntimeError(
            "Configure a private 32+ character destination tool key before migration"
        ) from None
    old, new = _cipher(old_key), _cipher(destination)
    count = 0
    with db.begin():
        connection = db.connection()
        # This command runs before schema upgrades; do not SELECT every column
        # of today's ORM model or require tables introduced by later releases.
        for table, row_id, column, value in credential_rows(
            connection, {"tool_configs": ("config",), "user_tool_configs": ("config",)}
        ):
            payload = copy.deepcopy(value)
            if not isinstance(payload, dict):
                continue
            changed = False
            for field, item in tool_credential_items(payload):
                location = f"{table.name} row {row_id} field {field}"
                try:
                    encoded = item["ciphertext"].encode()
                except UnicodeError:
                    raise RuntimeError(
                        f"Unreadable {location}; migration rolled back"
                    ) from None
                try:
                    new.decrypt(encoded)
                    continue
                except InvalidToken:
                    pass
                try:
                    plaintext = old.decrypt(encoded)
                except InvalidToken:
                    raise RuntimeError(
                        f"Unreadable {location}; migration rolled back"
                    ) from None
                item["ciphertext"] = new.encrypt(plaintext).decode()
                changed = True
                count += 1
            if changed:
                primary = next(iter(table.primary_key.columns))
                connection.execute(
                    update(table).where(primary == row_id).values(**{column: payload})
                )
    return count


def main() -> None:
    old_key = os.getenv(TOOL_CREDENTIALS_MIGRATION_KEY)
    if not old_key:
        raise SystemExit(f"Set {TOOL_CREDENTIALS_MIGRATION_KEY} for this offline run")
    try:
        configure_db()
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
