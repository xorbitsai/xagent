"""What a repeated MCP OAuth callback reports (#3941).

A browser can request the callback URL twice for one authorization: a
double-submitted consent, or a retry. The first request claims the flow state.
The second finds the claimed row, with or without the state cookie. A claimed
row can never be claimed again, because the claim requires
``consumed_at IS NULL``. So the second request has nothing left to do except
choose the page the browser lands on. It exchanges no code and writes nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast

from sqlalchemy.orm import Session

from ..models.mcp import MCPServer
from ..models.mcp_oauth import MCPOAuthFlowState
from .mcp_oauth import select_mcp_oauth_grants
from .time_utils import coerce_utc

RepeatedCallbackOutcome = Literal["connected", "not_connected"]


def repeated_callback_outcome(
    db: Session, flow_state: MCPOAuthFlowState, *, now: datetime
) -> RepeatedCallbackOutcome | None:
    """What a callback for an already-claimed flow reports.

    * ``None``: the flow is unclaimed or has expired. The callback goes on as
      usual: cookie check, claim, exchange.
    * ``"connected"``: the flow's owner is connected to the flow's server, as
      ``/oauth/status`` and the runtime decide it (``select_mcp_oauth_grants``
      under the server's current auth configuration, its required scope
      included).
    * ``"not_connected"``: otherwise. The flow's exchange failed, its grant was
      revoked since, or the first callback is still exchanging.

    The grant does not have to be the one this flow wrote: a usable grant from
    an earlier flow makes the connector just as connected. Matching on write
    time would compare the app's clock with the database's. What matters is
    that a page saying "connected" never contradicts the status the connector
    then shows.
    """
    expires_at = coerce_utc(cast(datetime, flow_state.expires_at))
    if flow_state.consumed_at is None or expires_at is None or expires_at <= now:
        return None
    server = (
        db.query(MCPServer).filter(MCPServer.id == flow_state.mcp_server_id).first()
    )
    if server is None:
        return "not_connected"
    auth_config: Any = server.to_config_dict().get("auth")
    grants = select_mcp_oauth_grants(
        db,
        server_id=cast(int, flow_state.mcp_server_id),
        user_id=cast(int, flow_state.user_id),
        auth_config=auth_config if isinstance(auth_config, dict) else {},
        resource_owner_key=cast(str, flow_state.resource_owner_key),
        # A team member's own flow is the one path a caller without an
        # association connects through; it may count that caller's grant.
        allow_team_member_grants=str(flow_state.access_path) == "team_member",
    )
    return "connected" if grants else "not_connected"
