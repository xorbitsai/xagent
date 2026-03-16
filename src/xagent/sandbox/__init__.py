"""
Sandbox Support.
"""

from .base import (
    CodeType,
    ExecResult,
    Sandbox,
    SandboxConfig,
    SandboxInfo,
    SandboxService,
    SandboxSnapshot,
    SandboxTemplate,
    TemplateType,
)

__all__ = [
    "TemplateType",
    "CodeType",
    "SandboxTemplate",
    "SandboxConfig",
    "SandboxInfo",
    "SandboxSnapshot",
    "ExecResult",
    "Sandbox",
    "SandboxService",
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
