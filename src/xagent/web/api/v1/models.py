"""SDK endpoint listing the models the key holder can use."""

from fastapi import APIRouter, Depends, Query

from ...schemas.v1 import V1Model, V1ModelForAgent
from ...services.agent_management import AgentManagementRuntime
from .deps import (
    PersonalApiKeySnapshot,
    UserPrincipalSnapshot,
    get_user_from_personal_key,
)
from .errors import V1ApiError, V1ErrorCode

router = APIRouter(prefix="/models")


@router.get("", response_model=list[V1ModelForAgent] | list[V1Model])
async def list_models(
    # ``agents.id`` is a 32-bit integer column.
    agent_id: int | None = Query(
        default=None,
        gt=0,
        le=2**31 - 1,
        description="An agent the key holder can see. Any other id from 1 to "
        "2147483647 returns 404 `agent_not_found`. When given, only models whose "
        "`category` is `llm` are listed, each with `owner_can_use`.",
    ),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> list[V1ModelForAgent] | list[V1Model]:
    """List the models the key holder can use.

    Without ``agent_id`` every category is listed; ``agent_id`` narrows the
    list as its description says. Order is not guaranteed. The list is read
    from the database on every call.
    """
    user, _key = authed
    if agent_id is not None:
        marked = await AgentManagementRuntime().list_models_for_agent(
            user_id=int(user.id), agent_id=agent_id
        )
        if marked is None:
            raise V1ApiError(V1ErrorCode.AGENT_NOT_FOUND, 404)
        return [
            V1ModelForAgent.model_validate(snapshot, from_attributes=True)
            for snapshot in marked
        ]
    snapshots = await AgentManagementRuntime().list_usable_models(user_id=int(user.id))
    return [
        V1Model.model_validate(snapshot, from_attributes=True) for snapshot in snapshots
    ]
