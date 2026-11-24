"""SDK endpoint listing the models the key holder can use."""

from fastapi import APIRouter, Depends

from ...schemas.v1 import V1Model
from ...services.agent_management import AgentManagementRuntime
from .deps import (
    PersonalApiKeySnapshot,
    UserPrincipalSnapshot,
    get_user_from_personal_key,
)

router = APIRouter(prefix="/models")


@router.get("", response_model=list[V1Model])
async def list_models(
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> list[V1Model]:
    """List every model the key holder can use, of any category.

    Order is not guaranteed. The list is read from the database on every
    call.
    """
    user, _key = authed
    snapshots = await AgentManagementRuntime().list_usable_models(user_id=int(user.id))
    return [
        V1Model.model_validate(snapshot, from_attributes=True) for snapshot in snapshots
    ]
