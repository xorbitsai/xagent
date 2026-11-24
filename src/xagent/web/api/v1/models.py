"""SDK endpoint listing the models the key holder can use."""

from fastapi import APIRouter, Depends

from ...schemas.v1 import V1Model
from ...services.agent_management import AgentManagementRuntime, UsableModelSnapshot
from .deps import (
    PersonalApiKeySnapshot,
    UserPrincipalSnapshot,
    get_user_from_personal_key,
)

router = APIRouter(prefix="/models")


def _model(snapshot: UsableModelSnapshot) -> V1Model:
    return V1Model(
        id=snapshot.id,
        model_id=snapshot.model_id,
        model_name=snapshot.model_name,
        model_provider=snapshot.model_provider,
        category=snapshot.category,
        abilities=list(snapshot.abilities),
        is_platform=snapshot.is_platform,
        owned_by_you=snapshot.owned_by_you,
    )


@router.get("", response_model=list[V1Model])
async def list_models(
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> list[V1Model]:
    """List every model the key holder can use, of any category.

    Only models whose ``category`` is ``llm`` can be placed in an agent's
    model slots. Order is not guaranteed. The list is read from the database
    on every call.
    """
    user, _key = authed
    snapshots = await AgentManagementRuntime().list_usable_models(user_id=int(user.id))
    return [_model(snapshot) for snapshot in snapshots]
