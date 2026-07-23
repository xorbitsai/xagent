"""SSH MCP domain layer: types, errors, interfaces, egress policy.

This package is pure domain code. It contains no database access, no HTTP,
no real sandbox wiring, and no real remote execution. Concrete adapters
(SaaS storage, real sandbox materialization, real SSH executor) live in
later phases and in the closed-source SaaS layer.
"""

from .egress import EgressDecision, EgressPolicyConfig, check_ip
from .errors import SshError, SshErrorCode
from .interfaces import (
    SandboxSecretMaterializer,
    SshSecretStore,
    SshTargetProvider,
)
from .types import (
    ActorRef,
    ApprovalPolicy,
    BoundTargetInfo,
    MaterializedSshPaths,
    PrincipalRef,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshCapability,
    SshExecutionContext,
    SshSecretHandle,
)

__all__ = [
    "ActorRef",
    "ApprovalPolicy",
    "BoundTargetInfo",
    "EgressDecision",
    "EgressPolicyConfig",
    "MaterializedSshPaths",
    "PrincipalRef",
    "ResolvedSshTarget",
    "SandboxSecretMaterializer",
    "SensitiveSshCredential",
    "SshCapability",
    "SshError",
    "SshErrorCode",
    "SshExecutionContext",
    "SshSecretHandle",
    "SshSecretStore",
    "SshTargetProvider",
    "check_ip",
]
