"""SSH MCP domain layer: types, errors, interfaces, egress policy, plus the
execution engine (executor, runners, materializers).

It contains no database access, no HTTP, and no credential storage; those —
along with RBAC and audit persistence — are provided by the closed-source
xagent-cloud layer through the injected provider/secret-store/audit adapters.
"""

from .egress import EgressDecision, EgressPolicyConfig, check_ip
from .errors import SshError, SshErrorCode
from .interfaces import (
    SandboxSecretMaterializer,
    SshAuditSink,
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
    "SshAuditSink",
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
