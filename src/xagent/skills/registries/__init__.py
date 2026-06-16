"""
Skill Hub registry providers.

Following the same pattern as ``xagent/core/model/providers.py``:

- ``_REGISTRY_PROVIDERS`` — declarative tuple of all available providers.
  Each entry has ``id``, ``name``, ``description``, and ``module``
  (the dotted import path to the provider's module).
- ``get_registry(id)`` — returns the ``SkillRegistry`` instance for *id*.
- ``all_registries()`` — returns a list of ``{id, displayName, description}``
  dicts for the frontend.

**Adding a new registry provider** (3 steps):

  1. Create a new file ``skill_hub_registries/<your_provider>.py``.
  2. Implement a class that subclasses ``SkillRegistry`` from ``base``.
  3. Add an entry to ``_REGISTRY_PROVIDERS`` below.

No other file needs to be touched — routes auto-discover via
``get_registry(source)`` and the frontend dropdown is built from
``all_registries()``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List

from fastapi import HTTPException

from xagent.skills.registries.base import SkillRegistry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Declarative provider list — add new registries here
# ═══════════════════════════════════════════════════════════════════

_REGISTRY_PROVIDERS: tuple[Dict[str, str], ...] = (
    {
        "id": "clawhub",
        "name": "ClawHub",
        "description": "ClawHub public skill registry",
        "module": "xagent.skills.registries.clawhub",
    },
)


# ═══════════════════════════════════════════════════════════════════
# Lazy-loaded registry instances
# ═══════════════════════════════════════════════════════════════════

_REGISTRY_INSTANCES: Dict[str, SkillRegistry] = {}


def _load_registry(provider: Dict[str, str]) -> SkillRegistry:
    """Lazy-import a registry provider module and instantiate it.

    Each module is expected to export a top-level attribute whose
    name is ``<id>_registry`` (e.g. ``clawhub_registry``).
    """
    module_path = provider["module"]
    attr_name = f"{provider['id']}_registry"
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        logger.error(
            "Skill Hub: failed to import registry provider %r (%s): %s",
            provider["id"],
            module_path,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Registry provider {provider['id']!r} failed to load.",
        ) from exc
    instance = getattr(mod, attr_name, None)
    if not isinstance(instance, SkillRegistry):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Registry provider {provider['id']!r}: "
                f"module {module_path!r} is missing "
                f"'{attr_name}' or it is not a SkillRegistry."
            ),
        )
    return instance


def get_registry(source: str) -> SkillRegistry:
    """Look up a registry by its id (lazy-loaded). Raises 400 if unknown."""
    if source in _REGISTRY_INSTANCES:
        return _REGISTRY_INSTANCES[source]

    for p in _REGISTRY_PROVIDERS:
        if p["id"] == source:
            instance = _load_registry(p)
            _REGISTRY_INSTANCES[source] = instance
            return instance

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown skill source {source!r}. "
            f"Available: {[p['id'] for p in _REGISTRY_PROVIDERS]}"
        ),
    )


def all_registries() -> List[Dict[str, str]]:
    """Return metadata for every registered provider."""
    return [
        {"id": p["id"], "displayName": p["name"], "description": p["description"]}
        for p in _REGISTRY_PROVIDERS
    ]
