"""Source-platform adapters that parse an on-disk footprint into a bundle."""

from __future__ import annotations

from pathlib import Path

from ..bundle import MigrationBundle
from .base import SourceAdapter
from .hermes import HermesAdapter
from .openclaw import OpenClawAdapter

# Registry of source adapters keyed by their CLI ``--from`` value.
ADAPTERS: dict[str, type[SourceAdapter]] = {
    "openclaw": OpenClawAdapter,
    "hermes": HermesAdapter,
}


def detect_sources() -> list[SourceAdapter]:
    """Return an adapter for every source platform present on this machine."""
    found: list[SourceAdapter] = []
    for adapter_cls in ADAPTERS.values():
        adapter = adapter_cls()
        if adapter.default_root().exists():
            found.append(adapter)
    return found


def get_adapter(source: str, root: Path | None = None) -> SourceAdapter:
    """Return the adapter for an explicit ``--from`` value."""
    try:
        adapter_cls = ADAPTERS[source]
    except KeyError:
        raise ValueError(
            f"Unknown migration source {source!r}. "
            f"Known sources: {', '.join(sorted(ADAPTERS))}."
        ) from None
    return adapter_cls(root=root)


__all__ = [
    "ADAPTERS",
    "SourceAdapter",
    "OpenClawAdapter",
    "HermesAdapter",
    "MigrationBundle",
    "detect_sources",
    "get_adapter",
]
