"""SDK endpoints for the MCP connectors the key holder can see."""

from fastapi import APIRouter, Depends, Path

from ...schemas.v1 import V1McpServer
from ...services.mcp_server_management import McpServerManagementRuntime
from .deps import (
    PersonalApiKeySnapshot,
    UserPrincipalSnapshot,
    get_user_from_personal_key,
)
from .errors import V1ApiError, V1ErrorCode

router = APIRouter(prefix="/mcp-servers")


@router.get("", response_model=list[V1McpServer])
async def list_mcp_servers(
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> list[V1McpServer]:
    """List the connectors the key holder can see.

    The holder's own connectors come first, including ones the holder switched
    off, then connectors visible only because the holder's team shares them.
    Custom APIs are not included.
    """
    user, _key = authed
    return await McpServerManagementRuntime().list_servers(user_id=int(user.id))


@router.get("/{server_id}", response_model=V1McpServer)
async def get_mcp_server(
    server_id: int = Path(gt=0, le=2**31 - 1),
    authed: tuple[UserPrincipalSnapshot, PersonalApiKeySnapshot] = Depends(
        get_user_from_personal_key
    ),
) -> V1McpServer:
    """Show one connector the key holder can see.

    A connector the holder cannot see and one that does not exist both answer
    404 ``connector_not_found``.
    """
    user, _key = authed
    server = await McpServerManagementRuntime().get_server(
        user_id=int(user.id), server_id=server_id
    )
    if server is None:
        raise V1ApiError(V1ErrorCode.CONNECTOR_NOT_FOUND, 404)
    return server
