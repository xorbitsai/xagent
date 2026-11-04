"""In-process registry mapping provider names to TriggerProvider instances."""

from __future__ import annotations

from .base import TriggerProvider


class UnknownTriggerProviderError(LookupError):
    """Raised when resolving a provider name that is not registered."""


_registry: dict[str, TriggerProvider] = {}


def register_trigger_provider(
    provider: TriggerProvider, *, replace: bool = False
) -> None:
    name = provider.name
    if not replace and name in _registry:
        raise ValueError(f"Trigger provider already registered: {name}")
    _registry[name] = provider


def unregister_trigger_provider(name: str) -> None:
    _registry.pop(name, None)


def get_trigger_provider(name: str) -> TriggerProvider:
    provider = _registry.get(name)
    if provider is None:
        raise UnknownTriggerProviderError(f"Unknown trigger provider: {name}")
    return provider


def maybe_get_trigger_provider(name: str) -> TriggerProvider | None:
    return _registry.get(name)


def registered_trigger_provider_names() -> tuple[str, ...]:
    return tuple(sorted(_registry))
