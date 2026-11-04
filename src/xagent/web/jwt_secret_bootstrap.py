"""Create or load the persistent JWT secret used by Docker Compose."""

from __future__ import annotations

import argparse
import fcntl
import os
import secrets
import sys
import tempfile
from pathlib import Path

from ..config import KNOWN_INSECURE_SECRETS

_MIN_SECRET_LENGTH = 32


def _validate_secret(secret: str, path: Path, label: str = "JWT secret") -> None:
    if len(secret) < _MIN_SECRET_LENGTH or secret.lower() in KNOWN_INSECURE_SECRETS:
        raise ValueError(f"{label} file is weak or too short: {path}")


def load_or_create_jwt_secret(
    path: Path, *, label: str = "JWT secret"
) -> tuple[str, bool]:
    """Return a strong persistent secret and whether this call created it."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # A valid secret on a read-only mounted volume is reusable without needing
    # to create or chmod a sibling lock file. The locked branch below still
    # handles first creation and races between replicas.
    if path.exists():
        secret = path.read_text(encoding="utf-8").strip()
        _validate_secret(secret, path, label)
        return secret, False

    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        if path.exists():
            secret = path.read_text(encoding="utf-8").strip()
            _validate_secret(secret, path, label)
            return secret, False

        secret = secrets.token_urlsafe(48)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                os.fchmod(temporary_file.fileno(), 0o600)
                temporary_file.write(f"{secret}\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return secret, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load or initialize a persistent JWT signing secret."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--label", default="JWT secret")
    args = parser.parse_args()
    secret, created = load_or_create_jwt_secret(args.path, label=args.label)
    action = "generated a new one" if created else "reusing the existing secret"
    print(f"{args.label} at {args.path}: {action}.", file=sys.stderr)
    print(secret)


if __name__ == "__main__":
    main()
