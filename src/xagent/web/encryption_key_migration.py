"""Offline rotation for persisted ENCRYPTION_KEY data (not tool credentials)."""

import os

from cryptography.fernet import Fernet
from sqlalchemy.exc import SQLAlchemyError

from ..config import (
    DEV_FALLBACK_ENCRYPTION_KEY,
    ENCRYPTION_KEY,
    ENCRYPTION_MIGRATION_KEY,
)
from .models.database import configure_db, get_engine
from .services.encryption_key_rotation import rotate_encryption_key


def main() -> None:
    source = os.getenv(ENCRYPTION_MIGRATION_KEY)
    destination = os.getenv(ENCRYPTION_KEY)
    if not source or not destination:
        raise SystemExit(
            f"Set {ENCRYPTION_MIGRATION_KEY} and an explicit private ENCRYPTION_KEY"
        )
    if destination.strip() == DEV_FALLBACK_ENCRYPTION_KEY:
        raise SystemExit("Refusing the published development destination key")
    try:
        old, new = Fernet(source.encode()), Fernet(destination.encode())
    except (ValueError, UnicodeError):
        raise SystemExit("Source and destination must be valid Fernet keys") from None
    try:
        configure_db()
        with get_engine().begin() as connection:
            count = rotate_encryption_key(connection, old, new)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    except SQLAlchemyError as exc:
        raise SystemExit(
            f"Database migration failed ({type(exc).__name__}); transaction rolled back"
        ) from None
    print(f"Re-encrypted {count} persisted credential fields.")


if __name__ == "__main__":
    main()
