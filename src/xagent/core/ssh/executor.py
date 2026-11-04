"""SshExecutor: orchestrates one authorized SSH command execution (Phase 3).

Ties the seams together in the design's per-call order (§11.2): resolve the
binding, enforce the requested capability, run the DNS-resolving egress
pre-flight, decrypt exactly one credential version, materialize it for this
call only, run the command with clamped limits, and cap the returned output.
Secret cleanup is the materializer's context manager (runs on success, error,
timeout, and cancellation).
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass

from .egress import EgressPolicyConfig
from .egress_io import resolve_and_authorize
from .errors import SshError, SshErrorCode
from .interfaces import (
    SandboxSecretMaterializer,
    SshAuditSink,
    SshSecretStore,
    SshTargetProvider,
)
from .runner import SshRunner
from .types import (
    MaterializedSshPaths,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshExecutionContext,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_OUTPUT_BYTES = 1 << 20  # 1 MiB, combined stdout + stderr
DEFAULT_MAX_TIMEOUT_SECONDS = 600
DEFAULT_TRANSFER_TIMEOUT_SECONDS = 300  # per-SFTP-transfer wall-clock bound


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
        sandbox_lease: Callable[[], AbstractAsyncContextManager[object]] | None = None,
        resolver: Callable[[str, int], Awaitable[list[str]]] | None = None,
        audit_sink: SshAuditSink | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_timeout_seconds: int = DEFAULT_MAX_TIMEOUT_SECONDS,
    ) -> None:
        self._provider = provider
        self._secret_store = secret_store
        self._materializer = materializer
        self._runner = runner
        self._egress_config = egress_config
        # A zero-arg callable yielding an async CM over a leased sandbox. When
        # set (xagent-cloud sandbox deploy), one lease spans materialize+run per call
        # and the runner executes inside it. When None (self-hosted, no sandbox
        # subsystem), sandbox is None and the runner runs in-process.
        self._sandbox_lease = sandbox_lease
        self._resolver = resolver
        self._audit_sink = audit_sink
        self._max_output_bytes = max_output_bytes
        self._max_timeout_seconds = max_timeout_seconds
        # SFTP transfers get their own default budget, but never more than the
        # deployment's configured max (execute() clamps the same way) so a
        # tighter max_timeout_seconds also tightens transfers (m3).
        self._transfer_timeout_seconds = min(
            DEFAULT_TRANSFER_TIMEOUT_SECONDS, max_timeout_seconds
        )

    async def execute(
        self,
        context: SshExecutionContext,
        *,
        target_alias: str,
        command: str,
        timeout_seconds: int = 60,
    ) -> SshExecuteOutcome:
        resolved: ResolvedSshTarget | None = None
        try:
            resolved = await self._provider.resolve(context, target_alias)
            if "execute" not in resolved.capabilities:
                raise SshError(
                    SshErrorCode.OPERATION_NOT_ALLOWED, "binding does not allow execute"
                )

            # Early, fail-closed egress check before any secret is read. The
            # authorized IP is pinned as connect_ip so the runner connects to
            # the vetted address (DNS-rebinding defense for the sandbox runner,
            # which cannot re-check the peer the way the in-process one does).
            addresses = await resolve_and_authorize(
                resolved.hostname,
                resolved.port,
                self._egress_config,
                resolver=self._resolver,
            )
            connect_ip = addresses[0]

            timeout = min(max(1, timeout_seconds), self._max_timeout_seconds)
            credential = await self._secret_store.read_version(resolved.secret_handle)

            start = time.monotonic()
            async with self._sandbox_and_secret(credential, resolved.known_hosts) as (
                sandbox,
                paths,
            ):
                run = await self._runner.execute(
                    sandbox=sandbox,
                    hostname=resolved.hostname,
                    connect_ip=connect_ip,
                    port=resolved.port,
                    username=resolved.username,
                    private_key_path=paths.private_key_path,
                    known_hosts_path=paths.known_hosts_path,
                    command=command,
                    timeout_seconds=timeout,
                    egress_config=self._egress_config,
                    max_output_bytes=self._max_output_bytes,
                )
            duration_ms = int((time.monotonic() - start) * 1000)
        except BaseException as exc:
            await self._audit_failure(context, "ssh_execute", resolved, exc)
            raise

        await self._audit(context, "ssh_execute", "success", resolved, None)
        stdout, stderr, capped = _cap_outputs(
            run.stdout, run.stderr, self._max_output_bytes
        )
        return SshExecuteOutcome(
            exit_code=run.exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=run.truncated or capped,
            duration_ms=duration_ms,
        )

    async def upload(
        self,
        context: SshExecutionContext,
        *,
        target_alias: str,
        local_path: str,
        remote_path: str,
        overwrite: bool = False,
    ) -> None:
        """Upload ``local_path`` (already resolved within the task workspace by
        the caller) to ``remote_path`` on a bound target."""
        resolved: ResolvedSshTarget | None = None
        try:
            resolved, connect_ip = await self._authorize_transfer(
                context, target_alias, "upload", remote_path
            )
            credential = await self._secret_store.read_version(resolved.secret_handle)
            async with self._sandbox_and_secret(credential, resolved.known_hosts) as (
                sandbox,
                paths,
            ):
                await self._runner.upload(
                    sandbox=sandbox,
                    hostname=resolved.hostname,
                    connect_ip=connect_ip,
                    port=resolved.port,
                    username=resolved.username,
                    private_key_path=paths.private_key_path,
                    known_hosts_path=paths.known_hosts_path,
                    local_path=local_path,
                    remote_path=remote_path,
                    overwrite=overwrite,
                    egress_config=self._egress_config,
                    timeout_seconds=self._transfer_timeout_seconds,
                )
        except BaseException as exc:
            await self._audit_failure(context, "ssh_upload", resolved, exc)
            raise
        await self._audit(context, "ssh_upload", "success", resolved, None)

    async def download(
        self,
        context: SshExecutionContext,
        *,
        target_alias: str,
        remote_path: str,
        local_path: str,
        overwrite: bool = False,
    ) -> None:
        """Download ``remote_path`` from a bound target to ``local_path`` (already
        resolved within the task workspace by the caller)."""
        resolved: ResolvedSshTarget | None = None
        try:
            resolved, connect_ip = await self._authorize_transfer(
                context, target_alias, "download", remote_path
            )
            credential = await self._secret_store.read_version(resolved.secret_handle)
            async with self._sandbox_and_secret(credential, resolved.known_hosts) as (
                sandbox,
                paths,
            ):
                await self._runner.download(
                    sandbox=sandbox,
                    hostname=resolved.hostname,
                    connect_ip=connect_ip,
                    port=resolved.port,
                    username=resolved.username,
                    private_key_path=paths.private_key_path,
                    known_hosts_path=paths.known_hosts_path,
                    remote_path=remote_path,
                    local_path=local_path,
                    overwrite=overwrite,
                    egress_config=self._egress_config,
                    timeout_seconds=self._transfer_timeout_seconds,
                )
        except BaseException as exc:
            await self._audit_failure(context, "ssh_download", resolved, exc)
            raise
        await self._audit(context, "ssh_download", "success", resolved, None)

    async def _audit_failure(
        self,
        context: SshExecutionContext,
        operation: str,
        target: ResolvedSshTarget | None,
        exc: BaseException,
    ) -> None:
        """Guarantee a failure audit for every escaping error — not just
        SshError. A non-SshError (unexpected bug) or a CancelledError would
        otherwise slip past ``except SshError`` and leave no audit trail,
        breaking the one-event-per-operation contract (#6)."""
        code = exc.code.value if isinstance(exc, SshError) else None
        # Shield the audit: this path runs while an exception (often a
        # CancelledError) is propagating, and a second cancellation landing
        # during the await would drop the failure audit — breaking the
        # one-event-per-operation guarantee on the cancellation path (m1).
        await asyncio.shield(self._audit(context, operation, "failure", target, code))

    async def _audit(
        self,
        context: SshExecutionContext,
        operation: str,
        status: str,
        target: ResolvedSshTarget | None,
        error_code: str | None,
    ) -> None:
        """Emit one audit event, best-effort. A sink failure is logged and
        swallowed so auditing can never fail the operation itself."""
        if self._audit_sink is None:
            return
        try:
            await self._audit_sink.record(
                context=context,
                operation=operation,
                status=status,
                target=target,
                error_code=error_code,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "ssh audit sink failed for %s (%s)", operation, status, exc_info=True
            )

    def _acquire_sandbox(self) -> AbstractAsyncContextManager[object]:
        """One sandbox for this call: a fresh lease when configured (fail-closed
        — capacity errors propagate, no host fallback), else a None sandbox for
        the in-process runner."""
        if self._sandbox_lease is None:
            return nullcontext(None)
        return self._sandbox_lease()

    @asynccontextmanager
    async def _sandbox_and_secret(
        self, credential: SensitiveSshCredential, known_hosts: str
    ) -> AsyncIterator[tuple[object, MaterializedSshPaths]]:
        """Lease one sandbox and materialize the secret into it, as a single
        scope so the lease spans materialize+run and both are cleaned up on
        every exit path. Yields the leased sandbox and the written paths."""
        async with (
            self._acquire_sandbox() as sandbox,
            self._materializer.materialize_ssh(
                sandbox, credential, known_hosts
            ) as paths,
        ):
            yield sandbox, paths

    async def _authorize_transfer(
        self,
        context: SshExecutionContext,
        target_alias: str,
        capability: str,
        remote_path: str,
    ) -> tuple[ResolvedSshTarget, str]:
        """Shared SFTP prologue: resolve, enforce the capability, run the
        fail-closed egress pre-flight, and constrain the remote path. Returns
        the resolved target and the authorized IP to connect to."""
        resolved = await self._provider.resolve(context, target_alias)
        if capability not in resolved.capabilities:
            raise SshError(
                SshErrorCode.OPERATION_NOT_ALLOWED,
                f"binding does not allow {capability}",
            )
        addresses = await resolve_and_authorize(
            resolved.hostname,
            resolved.port,
            self._egress_config,
            resolver=self._resolver,
        )
        _constrain_remote_path(remote_path, resolved.remote_root)
        return resolved, addresses[0]


def _constrain_remote_path(remote_path: str, remote_root: str | None) -> None:
    """Remote paths must be absolute; when the target pins a ``remote_root`` the
    path (with ``..`` collapsed) must stay within it. Lexical only — the remote
    filesystem is not consulted — so a symlink on the remote could still escape;
    this is the conservative first guard, not a full remote realpath check."""
    # The sandbox runner builds its SFTP batch file by wrapping each path in
    # double quotes, one command per line. A quote or newline in the path would
    # close the quoting / start a second, independent transfer command outside
    # the intended root — a confinement bypass. Reject them here, on the shared
    # transfer prologue, so both runners are covered at the root (#1).
    if any(c in remote_path for c in ('"', "\n", "\r", "\x00")):
        raise SshError(
            SshErrorCode.OPERATION_NOT_ALLOWED,
            "remote path contains forbidden characters",
        )
    if not remote_path.startswith("/"):
        raise SshError(
            SshErrorCode.OPERATION_NOT_ALLOWED, "remote path must be absolute"
        )
    if remote_root is None:
        return
    # normpath preserves a leading "//" (POSIX-defined), so "//root/x" would
    # stay "//root/x" and wrongly fail the "/root/" prefix check below; collapse
    # it to a single leading slash first (m4).
    normalized = posixpath.normpath(remote_path)
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    root = posixpath.normpath(remote_root)
    # normpath("/") == "/", so root + "/" would be "//"; guard that case.
    prefix = root if root.endswith("/") else root + "/"
    if normalized != root and not normalized.startswith(prefix):
        raise SshError(
            SshErrorCode.OPERATION_NOT_ALLOWED,
            "remote path escapes the target's remote root",
        )


def _cap_outputs(stdout: str, stderr: str, budget: int) -> tuple[str, str, bool]:
    """Cap combined stdout+stderr to ``budget`` bytes; flag if anything was cut.
    Decodes with errors='ignore' so a byte-boundary cut can't raise.

    stderr gets a reserved floor of the budget so a large stdout can't zero it
    out entirely — on a failing command the diagnostics matter more than the
    tail of stdout. stderr then also takes whatever stdout leaves unused."""
    out_b = stdout.encode("utf-8")
    err_b = stderr.encode("utf-8")
    truncated = False
    err_floor = min(len(err_b), budget // 4)
    out_budget = budget - err_floor
    if len(out_b) > out_budget:
        out_b = out_b[:out_budget]
        truncated = True
    err_budget = budget - len(out_b)
    if len(err_b) > err_budget:
        err_b = err_b[:err_budget]
        truncated = True
    return out_b.decode("utf-8", "ignore"), err_b.decode("utf-8", "ignore"), truncated
