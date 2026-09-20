"""Create or load the Docker Compose encryption key without exposing it."""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import tempfile
from pathlib import Path

from cryptography.fernet import Fernet


def load_or_create_encryption_key(path: Path) -> tuple[str, bool]:
    """Return a valid Fernet key and whether this call created it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if not key:
                raise ValueError(f"Encryption key file is empty: {path}")
            Fernet(key.encode())
            os.chmod(path, 0o600)
            return key, False

        key = Fernet.generate_key().decode()
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
                temporary_file.write(f"{key}\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return key, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load or initialize a persistent Fernet encryption key."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    key, created = load_or_create_encryption_key(args.path)
    if created:
        print(
            f"No encryption key found at {args.path}; generated a new one.",
            file=sys.stderr,
        )
    else:
        print(f"Reusing encryption key from {args.path}.", file=sys.stderr)
    print(key)


if __name__ == "__main__":
    main()
