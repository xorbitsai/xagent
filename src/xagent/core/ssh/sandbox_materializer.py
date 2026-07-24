"""Sandbox secret materialization (design §12.3 / §15.2).

Writes exactly one credential + known_hosts into a leased sandbox's private
directory for a single call, then removes it on every exit path. The key
material travels only through ``sandbox.write_file`` (a tar/put_archive stream);
the ``exec`` calls that create the directory, tighten modes, and clean up carry
only paths — never key bytes — so the secret never reaches process argv or logs.

This is the sandbox counterpart to ``LocalTmpSecretMaterializer``; both satisfy
the ``SandboxSecretMaterializer`` seam. The runner-local one writes to the
backend host; this one writes into an agent-inaccessible sandbox instead, so
the decrypted key never touches the backend disk.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import cast

from .errors import SshError, SshErrorCode
from .interfaces import SandboxLike
from .types import MaterializedSshPaths, SensitiveSshCredential

# Root for the per-call private dir. Not /dev/shm: Docker's archive API (used by
# the sandbox write_file) cannot extract into a tmpfs mount, so we use a normal
# path on the sandbox's own ephemeral filesystem. The secret is confined by a
# 0700 random dir + 0600 files, shredded on exit, with sandbox teardown as the
# crash-cleanup backstop.
_SECRET_ROOT = "/tmp"  # noqa: S108 — sandbox-private, cleaned per call


class SandboxTmpfsSecretMaterializer:
    """Materializes secrets into a leased sandbox for one call. Requires a
    sandbox exposing ``exec(command, *args)`` and
    ``write_file(content, remote_path)``.

    ``secret_root`` is where the per-call private dir is created; overridable
    for tests."""

    def __init__(self, *, secret_root: str = _SECRET_ROOT) -> None:
        self._secret_root = secret_root

    @asynccontextmanager
    async def materialize_ssh(
        self,
        sandbox: object,
        credential: SensitiveSshCredential,
        known_hosts: str,
    ) -> AsyncIterator[MaterializedSshPaths]:
        sb = cast(SandboxLike, sandbox)
        directory = f"{self._secret_root}/xagent-ssh-{secrets.token_hex(16)}"
        key_path = f"{directory}/id_key"
        known_hosts_path = f"{directory}/known_hosts"
        # 0700 private dir before any secret lands in it.
        result = await sb.exec("mkdir", "-p", "-m", "700", directory)
        if getattr(result, "exit_code", 0) != 0:
            raise SshError(
                SshErrorCode.SANDBOX_UNAVAILABLE,
                "could not create sandbox secret directory",
            )
        try:
            # Key material only ever goes through write_file (tar stream), not argv.
            await sb.write_file(
                content=credential.private_key.decode("utf-8"),
                remote_path=key_path,
                overwrite=True,
            )
            await sb.write_file(
                content=known_hosts, remote_path=known_hosts_path, overwrite=True
            )
            await sb.exec("chmod", "600", key_path, known_hosts_path)
            yield MaterializedSshPaths(
                private_key_path=key_path, known_hosts_path=known_hosts_path
            )
        finally:
            # rm removes the private dir; sandbox destroy is the final backstop.
            with suppress(Exception):
                await sb.exec("rm", "-rf", directory)
