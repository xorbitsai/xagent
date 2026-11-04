"""Deployment targets used when owners publish XAgent capabilities.

The browser's current origin is not always the public target that external
clients should call. Hosting layers may replace this route to advertise an
explicit regional origin while the standalone application keeps using its
configured public URLs.

This endpoint is intentionally unauthenticated: it exposes only public URLs
that generated snippets and links already reveal. A hosting layer that uses
``region`` to construct a ``/change-region?next=...`` bootstrap must validate
``next`` as an allowlisted same-origin relative path; this generic contract
cannot enforce a downstream route that standalone XAgent does not provide.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from ...config import get_app_base_url

router = APIRouter(prefix="/api", tags=["deployment"])


class DeploymentConfigResponse(BaseModel):
    """Public origins needed to generate deployment artifacts.

    ``deployment_origin`` is a hosting-layer override for installations whose
    external API and widget assets share one ingress. Standalone XAgent leaves
    it unset so API snippets retain their configured API URL while widget
    snippets retain the owner's browser origin.

    ``region`` is deliberately optional and unset by standalone XAgent.
    Multi-region hosting layers may set it so share links can establish the
    recipient's routing state before opening the canonical application URL.
    """

    deployment_origin: str | None
    app_origin: str | None
    region: str | None


@router.get("/deployment-config", response_model=DeploymentConfigResponse)
def get_deployment_config() -> DeploymentConfigResponse:
    """Return standalone deployment targets for the owner frontend."""

    return DeploymentConfigResponse(
        deployment_origin=None,
        app_origin=get_app_base_url(),
        region=None,
    )
