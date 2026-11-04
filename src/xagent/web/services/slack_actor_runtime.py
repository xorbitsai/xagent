"""Trusted fallback grant for actor execution of the built-in Slack connector."""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .mcp_runtime import MCPActorExecutionIdentity

logger = logging.getLogger(__name__)

SLACK_CHANNEL_ACCESS_POLICY_ENV = "XAGENT_SLACK_CHANNEL_ACCESS_POLICY"
SLACK_ACTOR_RUNTIME_REFRESH_KEY = "_slack_actor_runtime_refresh"
SLACK_READ_CAPABILITY = "read"
_SLACK_CONVERSATION_ID = re.compile(r"^[CGD][A-Z0-9]{5,}$")
_SLACK_RUNTIME_CAPABILITIES = frozenset({SLACK_READ_CAPABILITY})


class _SensitiveRuntimeValue(str):
    """A subprocess env value that remains redacted in container reprs."""

    def __repr__(self) -> str:
        return "<redacted>"


@dataclass(frozen=True)
class SlackChannelAccessPolicy:
    """Immutable exact channel allowlist supplied by a trusted embedder."""

    channel_ids: frozenset[str] = field(repr=False)
    expires_at: datetime = field(repr=False)
    capabilities: frozenset[str] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.channel_ids) is not frozenset or any(
            not isinstance(channel_id, str)
            or not _SLACK_CONVERSATION_ID.fullmatch(channel_id)
            for channel_id in self.channel_ids
        ):
            raise ValueError("channel_ids must be an exact frozenset of Slack IDs")
        if (
            not isinstance(self.expires_at, datetime)
            or self.expires_at.tzinfo is None
            or self.expires_at.utcoffset() is None
        ):
            raise ValueError("expires_at must be a timezone-aware datetime")
        if (
            type(self.capabilities) is not frozenset
            or self.capabilities != _SLACK_RUNTIME_CAPABILITIES
        ):
            raise ValueError("capabilities must contain only the read capability")


@dataclass(frozen=True)
class SlackActorRuntimeGrantRequest:
    """Trusted execution identity passed to the embedding grant resolver."""

    user_id: int
    resource_owner_key: str = field(repr=False)
    execution_identity: MCPActorExecutionIdentity = field(repr=False)
    scope: Any = field(default=None, repr=False)
    builtin_app_id: str = "slack"


@dataclass(frozen=True)
class SlackActorRuntimeGrant:
    """A fallback token inseparably paired with its strict channel policy."""

    access_token: str = field(repr=False)
    channel_access: SlackChannelAccessPolicy = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.access_token, str) or not self.access_token.strip():
            raise ValueError("access_token must be a non-empty string")
        if not isinstance(self.channel_access, SlackChannelAccessPolicy):
            raise ValueError("channel_access must be a SlackChannelAccessPolicy")
        object.__setattr__(
            self, "access_token", _SensitiveRuntimeValue(self.access_token)
        )


SlackActorRuntimeGrantResult = (
    SlackActorRuntimeGrant | Awaitable[SlackActorRuntimeGrant | None] | None
)
SlackActorRuntimeGrantResolver = Callable[
    [SlackActorRuntimeGrantRequest], SlackActorRuntimeGrantResult
]

_resolver: SlackActorRuntimeGrantResolver | None = None


def set_slack_actor_runtime_grant_resolver(
    resolver: SlackActorRuntimeGrantResolver | None,
) -> None:
    """Register the trusted Slack fallback resolver for embedding runtimes."""

    global _resolver
    _resolver = resolver


async def resolve_slack_actor_runtime_grant(
    *,
    user_id: int | None,
    resource_owner_key: str,
    execution_identity: MCPActorExecutionIdentity | None,
    scope: Any,
) -> SlackActorRuntimeGrant | None:
    resolver = _resolver
    if (
        resolver is None
        or isinstance(user_id, bool)
        or not isinstance(user_id, int)
        or user_id <= 0
        or execution_identity is None
    ):
        return None
    request = SlackActorRuntimeGrantRequest(
        user_id=user_id,
        resource_owner_key=resource_owner_key,
        execution_identity=execution_identity,
        scope=scope,
    )
    try:
        grant = resolver(request)
        if inspect.isawaitable(grant):
            grant = await grant
    except Exception as exc:
        logger.warning(
            "Slack actor runtime grant resolution failed (%s)",
            type(exc).__name__,
        )
        return None
    if not isinstance(grant, SlackActorRuntimeGrant):
        logger.warning("Slack actor runtime grant resolver returned no valid grant")
        return None
    if grant.channel_access.expires_at <= datetime.now(timezone.utc):
        logger.warning("Slack actor runtime grant resolver returned an expired grant")
        return None
    return grant


def serialize_slack_channel_access_policy(policy: SlackChannelAccessPolicy) -> str:
    """Serialize policy for the canonical Slack child without logging it."""

    return _SensitiveRuntimeValue(
        json.dumps(
            {
                "version": 2,
                "channel_ids": sorted(policy.channel_ids),
                "expires_at": policy.expires_at.timestamp(),
                "capabilities": sorted(policy.capabilities),
            },
            separators=(",", ":"),
        )
    )
