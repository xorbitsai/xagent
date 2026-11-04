"""Immutable value objects for the SSH MCP domain layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActorType = Literal["user", "client_application", "system"]
PrincipalType = Literal["user", "team", "platform"]
SshCapability = Literal["execute", "upload", "download"]
ApprovalPolicy = Literal["always", "risk_based", "not_required"]


@dataclass(frozen=True)
class ActorRef:
    """Who initiated the operation, for audit."""

    actor_type: ActorType
    actor_id: str


@dataclass(frozen=True)
class PrincipalRef:
    """Whose permissions and resource ownership the call executes under.

    ``principal_id`` is ``None`` for platform-scoped execution.
    """

    principal_type: PrincipalType
    principal_id: str | None


@dataclass(frozen=True)
class SshExecutionContext:
    """Per-call execution context passed to the target resolver.

    The caller cannot submit execution_principal / credential / owner ids
    directly; these must be derived from the authenticated request and the
    agent scope by the xagent-cloud layer before constructing this context.

    NOTE: the design doc lists a ``sandbox`` field here. We keep it out of the
    resolver context on purpose: the materializer receives the sandbox as an
    explicit argument, so coupling this context to the sandbox type buys
    nothing.
    """

    actor: ActorRef
    execution_principal: PrincipalRef
    agent_id: int
    task_id: int | None
    turn_id: str | None
    request_id: str


@dataclass(frozen=True)
class SshSecretHandle:
    """Opaque pointer to a credential version. Carries no secret material."""

    credential_id: str
    version_id: str


@dataclass(frozen=True)
class ResolvedSshTarget:
    """Result of resolving an agent binding alias to a concrete target.

    Contains only non-secret connection info plus a handle the secret store
    can later exchange for decrypted material. ``known_hosts`` holds host
    *public* keys and is not secret.
    """

    target_public_id: str
    hostname: str
    port: int
    username: str
    remote_root: str | None
    capabilities: frozenset[SshCapability]
    approval_policy: ApprovalPolicy
    secret_handle: SshSecretHandle
    known_hosts: str
    credential_public_id: str
    credential_version_id: str
    host_key_fingerprint: str


@dataclass(frozen=True, repr=False)
class SensitiveSshCredential:
    """Decrypted client credential. MUST NOT be logged or serialized.

    ``__repr__``/``__str__`` are redacted so accidental logging / f-strings /
    trace capture never leak the private key.
    """

    private_key: bytes
    public_key: str
    key_algorithm: str

    def __repr__(self) -> str:
        return f"<SensitiveSshCredential algorithm={self.key_algorithm} redacted>"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(frozen=True)
class MaterializedSshPaths:
    """Filesystem paths a materializer exposed for a single call."""

    private_key_path: str
    known_hosts_path: str


@dataclass(frozen=True)
class BoundTargetInfo:
    """Non-secret summary of a target an agent is bound to (for list_targets)."""

    alias: str
    display_name: str | None
    capabilities: frozenset[SshCapability]
