"""
Sandbox Support.
"""

from ..config import get_sandbox_image
from .base import (
    SPEC_CONTRACT_VERSION,
    CodeType,
    ExecResult,
    ObservedRuntimeFacts,
    ResolvedSandboxRuntimeSpec,
    Sandbox,
    SandboxAlreadyExistsError,
    SandboxConfig,
    SandboxContractError,
    SandboxInfo,
    SandboxInspection,
    SandboxMountEscapeError,
    SandboxMountIntent,
    SandboxNotFoundError,
    SandboxReconcileUnsupportedError,
    SandboxRecoveryRequiredError,
    SandboxRuntimeConflictError,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
    SpecVerdict,
    TemplateType,
    canonical_sandbox_path,
    spec_matches_inspection,
)

# Use the `latest` image as a fallback
# We should pin the version at release by env "SANDBOX_IMAGE" (`latest` may lead to caching problems)
DEFAULT_SANDBOX_IMAGE = get_sandbox_image()

__all__ = [
    "DEFAULT_SANDBOX_IMAGE",
    "TemplateType",
    "CodeType",
    "SandboxTemplate",
    "SandboxConfig",
    "SandboxInfo",
    "SandboxNotFoundError",
    "SandboxSnapshot",
    "ExecResult",
    "Sandbox",
    "SandboxService",
    "SandboxContractError",
    "SandboxAlreadyExistsError",
    "SandboxRuntimeConflictError",
    "SandboxMountEscapeError",
    "SandboxRecoveryRequiredError",
    "SandboxReconcileUnsupportedError",
    "ResolvedSandboxRuntimeSpec",
    "ObservedRuntimeFacts",
    "SandboxInspection",
    "SandboxMountIntent",
    "SpecVerdict",
    "spec_matches_inspection",
    "canonical_sandbox_path",
    "SPEC_CONTRACT_VERSION",
]

try:
    from .boxlite_sandbox import (
        BoxliteSandbox,
        BoxliteSandboxService,
        BoxliteStore,
        MemBoxliteStore,
    )

    __all__ += [
        "BoxliteSandbox",
        "BoxliteStore",
        "MemBoxliteStore",
        "BoxliteSandboxService",
    ]
except ImportError:
    pass

try:
    from .docker_sandbox import (
        DockerSandbox,
        DockerSandboxService,
        DockerStore,
        MemDockerStore,
        is_docker_available,
    )

    __all__ += [
        "DockerSandbox",
        "DockerSandboxService",
        "DockerStore",
        "MemDockerStore",
        "is_docker_available",
    ]
except ImportError:
    pass
