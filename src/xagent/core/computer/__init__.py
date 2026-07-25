"""Provider-neutral Computer Use contracts and runtime helpers."""

from .desktop import DesktopRelayEnvironment
from .desktop_relay import (
    DESKTOP_RELAY_PROTOCOL_VERSION,
    get_desktop_relay_registry,
)
from .environment import (
    ComputerEnvironment,
    ComputerFrameMismatchError,
    ComputerSessionMismatchError,
    ComputerTargetNotFoundError,
)
from .extension import ExtensionComputerEnvironment
from .policy import (
    ComputerActionPolicy,
    ComputerPolicyDecision,
    ComputerPolicyOutcome,
    ComputerRiskLevel,
    DefaultComputerActionPolicy,
    find_computer_target_element,
)
from .redis_relay import RedisBrowserRelayRegistry
from .relay import (
    BROWSER_RELAY_PROTOCOL_VERSION,
    BrowserRelayAuthenticationError,
    BrowserRelayCommandConnection,
    BrowserRelayConnection,
    BrowserRelayError,
    BrowserRelayInUseError,
    BrowserRelayProtocolError,
    BrowserRelayRegistry,
    BrowserRelayRegistryProtocol,
    BrowserRelayUnavailableError,
    ComputerReadinessIssue,
    ComputerTargetReadiness,
    DesktopRelayStatusMessage,
    build_computer_target_readiness,
    get_browser_relay_registry,
    require_computer_target_ready,
)
from .schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerObservation,
    ComputerTarget,
    NormalizedPoint,
    NormalizedRect,
    Viewport,
)
from .session import (
    BrowserRuntimeKind,
    ComputerSessionBinding,
    validate_browser_profile_id,
)
from .store import ObservationStore

__all__ = [
    "BrowserRuntimeKind",
    "DESKTOP_RELAY_PROTOCOL_VERSION",
    "BROWSER_RELAY_PROTOCOL_VERSION",
    "BrowserRelayAuthenticationError",
    "BrowserRelayCommandConnection",
    "BrowserRelayConnection",
    "BrowserRelayError",
    "BrowserRelayInUseError",
    "BrowserRelayProtocolError",
    "BrowserRelayRegistry",
    "BrowserRelayRegistryProtocol",
    "BrowserRelayUnavailableError",
    "ComputerAction",
    "ComputerActionBatch",
    "ComputerActionPolicy",
    "ComputerActionType",
    "ComputerElement",
    "ComputerElementSource",
    "ComputerEnvironment",
    "ComputerEnvironmentType",
    "ComputerFrameMismatchError",
    "ComputerObservation",
    "ComputerPolicyDecision",
    "ComputerPolicyOutcome",
    "ComputerReadinessIssue",
    "ComputerRiskLevel",
    "ComputerSessionMismatchError",
    "ComputerSessionBinding",
    "ComputerTarget",
    "ComputerTargetReadiness",
    "ComputerTargetNotFoundError",
    "DesktopRelayEnvironment",
    "DesktopRelayStatusMessage",
    "DefaultComputerActionPolicy",
    "ExtensionComputerEnvironment",
    "NormalizedPoint",
    "NormalizedRect",
    "ObservationStore",
    "RedisBrowserRelayRegistry",
    "Viewport",
    "get_browser_relay_registry",
    "get_desktop_relay_registry",
    "build_computer_target_readiness",
    "find_computer_target_element",
    "require_computer_target_ready",
    "validate_browser_profile_id",
]
