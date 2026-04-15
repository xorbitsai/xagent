"""
Sandbox Support.
"""

from ..config import get_sandbox_image
from .base import (
    CodeType,
    ExecResult,
    Sandbox,
    SandboxConfig,
    SandboxInfo,
    SandboxNotFoundError,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
    TemplateType,
)
from .docker_sandbox import (
    DockerSandbox,
    DockerSandboxService,
    DockerStore,
    MemDockerStore,
    is_docker_available,
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
    "DockerSandbox",
    "DockerStore",
    "MemDockerStore",
    "DockerSandboxService",
    "is_docker_available",
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
