"""Public SDK ``/v1/*`` namespace.

This subpackage holds endpoints exposed to external SDK clients
(Python / TypeScript / JavaScript) that authenticate with an API key
rather than a JWT session: an ``xag_<prefix>_<secret>`` runtime key bound
to one agent or workforce, or an ``xag_personal_<prefix>_<secret>``
personal key that acts as its holder.

The split from ``/api/*`` is deliberate (see SDK design doc §3):

  - ``/api/*`` is JWT-gated, for agent owners managing their own
    resources through the web UI.
  - ``/v1/*`` is API-key-gated, for SaaS callers running an agent
    on behalf of an end user, and for users scripting their own agents
    with a personal key. Errors follow a separate stable schema
    (``{"error": {"code": ..., "message": ...}}``) and the surface
    will version independently of internal admin endpoints.

Re-exports ``v1_router`` for ``web/app.py`` to mount under ``/v1``.
"""

from fastapi import APIRouter

from .agents import router as _agents_router
from .features import router as _features_router
from .me import router as _me_router
from .skills import router as _skills_router
from .tasks import router as _tasks_router
from .templates import router as _templates_router
from .workforces import router as _workforces_router

v1_router = APIRouter(prefix="/v1", tags=["sdk-v1"])
v1_router.include_router(_me_router)
v1_router.include_router(_features_router)
v1_router.include_router(_templates_router)
v1_router.include_router(_agents_router)
v1_router.include_router(_skills_router)
v1_router.include_router(_tasks_router)
v1_router.include_router(_workforces_router)

__all__ = ["v1_router"]
