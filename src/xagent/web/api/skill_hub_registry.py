"""
Backward-compatibility shim for skill hub registry providers.

All provider logic now lives in ``skills/registries/``.

To add a new registry:
  1. Create ``skills/registries/<your_provider>.py``
  2. Subclass ``SkillRegistry`` from ``skills/registries.base``
  3. Export an instance named ``<id>_registry``
  4. Add provider metadata to ``_REGISTRY_PROVIDERS`` in
     ``skills/registries/__init__.py``

See the ClawHub provider (``skills/registries/clawhub.py``)
for a complete example.
"""

from xagent.skills.registries import (
    all_registries,
    get_registry,
)
from xagent.skills.registries.base import MAX_DOWNLOAD_BYTES as _MAX_DOWNLOAD_BYTES
from xagent.skills.registries.base import MAX_REGISTRY_BODY as _MAX_REGISTRY_BODY
from xagent.skills.registries.base import (
    SkillRegistry,
)

__all__ = [
    "SkillRegistry",
    "get_registry",
    "all_registries",
    "_MAX_REGISTRY_BODY",
    "_MAX_DOWNLOAD_BYTES",
]
