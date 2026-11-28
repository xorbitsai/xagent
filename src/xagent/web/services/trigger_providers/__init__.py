"""Unified TriggerProvider framework for third-party trigger callbacks."""

from .audit import record_trigger_audit, record_trigger_audit_best_effort
from .base import (
    CallbackRequestContext,
    TriggerConfigError,
    TriggerEventParseError,
    TriggerProvider,
    TriggerProviderError,
)
from .github import GithubProvider, sign_github_payload
from .gmail import GmailProvider
from .pipeline import CallbackResult, process_trigger_callback
from .registry import (
    UnknownTriggerProviderError,
    get_trigger_provider,
    maybe_get_trigger_provider,
    register_trigger_provider,
    registered_trigger_provider_names,
    unregister_trigger_provider,
)
from .schemas import (
    AckPolicy,
    BaseTriggerConfig,
    ChallengeResponse,
    GithubTriggerConfig,
    GmailTriggerConfig,
    NormalizedEvent,
    RegistrationResult,
    ScheduledTriggerConfig,
    TriggerConfig,
    VerificationResult,
    WebhookTriggerConfig,
    dump_trigger_config,
    parse_trigger_config,
)
from .webhook import WebhookProvider, sign_webhook_payload

__all__ = [
    "AckPolicy",
    "BaseTriggerConfig",
    "CallbackRequestContext",
    "CallbackResult",
    "ChallengeResponse",
    "GithubProvider",
    "GithubTriggerConfig",
    "GmailTriggerConfig",
    "GmailProvider",
    "NormalizedEvent",
    "RegistrationResult",
    "ScheduledTriggerConfig",
    "TriggerConfig",
    "TriggerConfigError",
    "TriggerEventParseError",
    "TriggerProvider",
    "TriggerProviderError",
    "UnknownTriggerProviderError",
    "VerificationResult",
    "WebhookProvider",
    "WebhookTriggerConfig",
    "dump_trigger_config",
    "sign_github_payload",
    "sign_webhook_payload",
    "get_trigger_provider",
    "maybe_get_trigger_provider",
    "parse_trigger_config",
    "process_trigger_callback",
    "record_trigger_audit",
    "record_trigger_audit_best_effort",
    "register_trigger_provider",
    "registered_trigger_provider_names",
    "unregister_trigger_provider",
]
