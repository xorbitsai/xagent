"""Provider-neutral Computer Use contracts and runtime helpers."""

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
)
from .relay import (
    BROWSER_RELAY_PROTOCOL_VERSION,
    BrowserRelayAuthenticationError,
    BrowserRelayConnection,
    BrowserRelayError,
    BrowserRelayInUseError,
    BrowserRelayProtocolError,
    BrowserRelayRegistry,
    BrowserRelayUnavailableError,
    get_browser_relay_registry,
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
    "BROWSER_RELAY_PROTOCOL_VERSION",
    "BrowserRelayAuthenticationError",
    "BrowserRelayConnection",
    "BrowserRelayError",
    "BrowserRelayInUseError",
    "BrowserRelayProtocolError",
    "BrowserRelayRegistry",
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
    "ComputerRiskLevel",
    "ComputerSessionMismatchError",
    "ComputerSessionBinding",
    "ComputerTarget",
    "ComputerTargetNotFoundError",
    "ExtensionComputerEnvironment",
    "NormalizedPoint",
    "NormalizedRect",
    "ObservationStore",
    "Viewport",
    "get_browser_relay_registry",
    "validate_browser_profile_id",
]
