"""SshExecutor: orchestrates one authorized SSH command execution (Phase 3).

Ties the seams together in the design's per-call order (§11.2): resolve the
binding, enforce the requested capability, run the DNS-resolving egress
pre-flight, decrypt exactly one credential version, materialize it for this
call only, run the command with clamped limits, and cap the returned output.
Secret cleanup is the materializer's context manager (runs on success, error,
timeout, and cancellation).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .egress import EgressPolicyConfig
from .egress_io import resolve_and_authorize
from .errors import SshError, SshErrorCode
from .interfaces import SandboxSecretMaterializer, SshSecretStore, SshTargetProvider
from .runner import SshRunner
from .types import SshExecutionContext

DEFAULT_MAX_OUTPUT_BYTES = 1 << 20  # 1 MiB, combined stdout + stderr
DEFAULT_MAX_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class SshExecuteOutcome:
    """Non-secret result of a remote command."""

    exit_code: int
    stdout: str
    stderr: str
    truncated: bool
    duration_ms: int


class SshExecutor:
    def __init__(
        self,
        *,
        provider: SshTargetProvider,
        secret_store: SshSecretStore,
        materializer: SandboxSecretMaterializer,
        runner: SshRunner,
        egress_config: EgressPolicyConfig,
        sandbox: object | None = None,
        resolver: Callable[[str, int], Awaitable[list[str]]] | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_timeout_seconds: int = DEFAULT_MAX_TIMEOUT_SECONDS,
    ) -> None:
        self._provider = provider
        self._secret_store = secret_store
        self._materializer = materializer
        self._runner = runner
        self._egress_config = egress_config
        self._sandbox = sandbox
        self._resolver = resolver
        self._max_output_bytes = max_output_bytes
        self._max_timeout_seconds = max_timeout_seconds

    async def execute(
        self,
        context: SshExecutionContext,
        *,
        target_alias: str,
        command: str,
        timeout_seconds: int = 60,
    ) -> SshExecuteOutcome:
        resolved = await self._provider.resolve(context, target_alias)
        if "execute" not in resolved.capabilities:
            raise SshError(SshErrorCode.OPERATION_NOT_ALLOWED, "binding does not allow execute")

        # Early, fail-closed egress check before any secret is read.
        await resolve_and_authorize(
            resolved.hostname, resolved.port, self._egress_config, resolver=self._resolver
        )

        timeout = min(max(1, timeout_seconds), self._max_timeout_seconds)
        credential = await self._secret_store.read_version(resolved.secret_handle)

        start = time.monotonic()
        async with self._materializer.materialize_ssh(
            self._sandbox, credential, resolved.known_hosts
        ) as paths:
            run = await self._runner.execute(
                hostname=resolved.hostname,
                port=resolved.port,
                username=resolved.username,
                private_key_path=paths.private_key_path,
                known_hosts_path=paths.known_hosts_path,
                command=command,
                timeout_seconds=timeout,
                egress_config=self._egress_config,
            )
        duration_ms = int((time.monotonic() - start) * 1000)

        stdout, stderr, capped = _cap_outputs(run.stdout, run.stderr, self._max_output_bytes)
        return SshExecuteOutcome(
            exit_code=run.exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=run.truncated or capped,
            duration_ms=duration_ms,
        )


def _cap_outputs(stdout: str, stderr: str, budget: int) -> tuple[str, str, bool]:
    """Cap combined stdout+stderr to ``budget`` bytes; flag if anything was cut.
    Decodes with errors='ignore' so a byte-boundary cut can't raise."""
    out_b = stdout.encode("utf-8")
    err_b = stderr.encode("utf-8")
    truncated = False
    if len(out_b) > budget:
        out_b = out_b[:budget]
        err_b = b""
        truncated = True
    else:
        remaining = budget - len(out_b)
        if len(err_b) > remaining:
            err_b = err_b[:remaining]
            truncated = True
    return out_b.decode("utf-8", "ignore"), err_b.decode("utf-8", "ignore"), truncated
