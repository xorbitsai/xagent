"""Secret materialization for the in-process asyncssh runner (Phase 3).

The in-process runner connects from the backend, so it materializes the
credential + known_hosts into a private temp dir on the backend host: a 0700
directory, 0600 files, created with O_EXCL|O_NOFOLLOW, and shredded + removed on
every exit path (success, error, timeout, cancellation).

This is the runner-local materializer. A sandbox materializer that writes into
an isolated sandbox tmpfs (design §15.2) is a separate adapter (P3·6); both
satisfy the ``SandboxSecretMaterializer`` seam.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from .types import MaterializedSshPaths, SensitiveSshCredential


class LocalTmpSecretMaterializer:
    """Materializes secrets to a local private temp dir. Ignores ``sandbox``."""

    @asynccontextmanager
    async def materialize_ssh(
        self,
        sandbox: object,
        credential: SensitiveSshCredential,
        known_hosts: str,
    ) -> AsyncIterator[MaterializedSshPaths]:
        directory = tempfile.mkdtemp(prefix="xagent-ssh-")
        os.chmod(directory, 0o700)
        key_path = os.path.join(directory, "id_key")
        known_hosts_path = os.path.join(directory, "known_hosts")
        try:
            _write_private(key_path, credential.private_key)
            _write_private(known_hosts_path, known_hosts.encode("utf-8"))
            yield MaterializedSshPaths(
                private_key_path=key_path, known_hosts_path=known_hosts_path
            )
        finally:
            for path in (key_path, known_hosts_path):
                _best_effort_shred(path)
            # suppress: a cleanup failure here must not replace a real exception
            # already propagating out of the yield with a confusing ENOTEMPTY.
            if os.path.isdir(directory):
                with suppress(OSError):
                    os.rmdir(directory)


def _write_private(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _best_effort_shred(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        size = os.path.getsize(path)
        fd = os.open(path, os.O_WRONLY)
        try:
            os.write(fd, b"\x00" * size)
            os.fsync(fd)  # flush zeros to disk before the unlink
        finally:
            os.close(fd)
    except OSError:
        pass
    with suppress(OSError):
        os.unlink(path)
