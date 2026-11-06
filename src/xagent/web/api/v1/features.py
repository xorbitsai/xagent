"""GET /v1/features -- what the presented key may do."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .deps import ApiKeyPrincipal, get_principal_from_api_key

router = APIRouter()

STRUCTURED_OUTPUT_FEATURE = "structured_output"


class FeaturesResponse(BaseModel):
    """Response model for ``GET /v1/features``."""

    features: list[str] = Field(
        ...,
        description=(
            "Names of the optional features the presented key can use. "
            '"structured_output" means the key can set, change or clear the '
            "response_format of a task."
        ),
    )


@router.get("/features", response_model=FeaturesResponse)
async def get_features(
    principal: ApiKeyPrincipal = Depends(get_principal_from_api_key),
) -> FeaturesResponse:
    """List the optional features the presented key can use.

    A personal key and an agent key can use structured output. A workforce
    key cannot: an empty list means this key cannot set or clear a
    ``response_format``, so a client should not send one with it.
    """
    if principal.workforce is not None:
        return FeaturesResponse(features=[])
    return FeaturesResponse(features=[STRUCTURED_OUTPUT_FEATURE])
