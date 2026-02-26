import argparse
import getpass
import logging
import sys
from typing import cast

from dotenv import load_dotenv

from .api.auth import hash_password
from .auth_config import PASSWORD_MIN_LENGTH
from .models.database import get_db, init_db
from .models.user import User

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset password for an admin user",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m xagent.web.reset_admin_password --username admin
  python -m xagent.web.reset_admin_password --username admin --password "new-strong-password"
  python -m xagent.web.reset_admin_password --yes
        """,
    )
    _ = parser.add_argument(
        "--username",
        default=None,
        help="Admin username to reset (if omitted, will prompt; default: admin)",
    )
    _ = parser.add_argument(
        "--password",
        default=None,
        help="New password (if omitted, will prompt securely)",
    )
    _ = parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    return parser.parse_args()


def _read_username(username_from_args: str | None) -> str:
    if username_from_args and username_from_args.strip():
        return username_from_args.strip()

    entered = input("Admin username [admin]: ").strip()
    return entered or "admin"


def _read_password(password_from_args: str | None) -> str:
    if password_from_args:
        if len(password_from_args) < PASSWORD_MIN_LENGTH:
            raise ValueError(
                f"Password must be at least {PASSWORD_MIN_LENGTH} characters"
            )
        return password_from_args

    print(f"New password must be at least {PASSWORD_MIN_LENGTH} characters.")
    while True:
        password = getpass.getpass("New password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match. Please try again.")
            continue
        if len(password) < PASSWORD_MIN_LENGTH:
            print(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.")
            continue
        return password


def _confirm_reset(username: str, skip_confirmation: bool) -> None:
    if skip_confirmation:
        return

    confirmed = input(
        f"Reset password for admin '{username}' and revoke active refresh tokens? [y/N]: "
    ).strip()
    if confirmed.lower() not in {"y", "yes"}:
        raise ValueError("Operation cancelled")


def reset_admin_password(username: str, new_password: str) -> None:
    if len(new_password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")

    db = next(get_db())
    try:
        admin = (
            db.query(User)
            .filter(User.username == username)
            .filter(User.is_admin.is_(True))
            .first()
        )

        if admin is None:
            raise ValueError(f"Admin user '{username}' not found")

        setattr(admin, "password_hash", hash_password(new_password))
        setattr(admin, "refresh_token", None)
        setattr(admin, "refresh_token_expires_at", None)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    _ = load_dotenv()
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    username_arg = cast(str | None, args.username)
    password_arg = cast(str | None, args.password)
    yes_arg = cast(bool, args.yes)
    try:
        init_db()
        username = _read_username(username_arg)
        _confirm_reset(username, yes_arg)
        new_password = _read_password(password_arg)
        reset_admin_password(username, new_password)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to reset admin password: {exc}")
        sys.exit(1)

    print(f"Password for admin '{username}' has been reset successfully.")


if __name__ == "__main__":
    main()
