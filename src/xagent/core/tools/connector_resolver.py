"""Host hook behind the ``find_app_connection`` tool.

xagent knows nothing about which apps exist, how they are approved, or where
their credentials live. A host that supports "connect an app from chat"
registers one :class:`ConnectorResolver`; with none registered the
``find_app_connection`` tool is not offered at all.

Who is asking: ``user_id`` is the task *owner*. Widget, shared-link and channel
runs execute as the owner, so the resolver cannot tell the owner from a guest by
``user_id`` alone. The tool refuses the public ``widget`` and ``shared_link``
task sources itself, but a channel run reports the same ``task_source`` as the
owner's own chat. A resolver must therefore treat ``task_source`` and
``task_id`` as part of the request and refuse any run it cannot attribute to the
owner acting for themselves, both in ``resolve`` and in ``connection_state``.
No separate actor identity is available to tools.

Secret handling contract: nothing here carries a secret. A
:class:`ConnectorResolution` ``connector`` payload is display data only
(name, route, field labels, steps). The credential itself travels from the
user's browser to the host's own endpoint and is stored there; it never enters
the chat transcript or the model context. The tool settles a resumed call from
:meth:`ConnectorResolver.connection_state`, i.e. server-side state, never from
the user's reply text.

This module has no web or ORM dependency, like the MCP approval gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

ResolutionStatus = Literal["connected", "needs_connection", "not_found"]
ConnectionStatus = Literal["connected", "skipped", "pending"]


@dataclass(frozen=True)
class ConnectorResolveRequest:
    """What the model asked for, plus the identity of the run asking."""

    app_name: str
    purpose: str
    # Issued by the tool and echoed back to the user's browser; the host keys
    # its server-side connection state by it.
    interaction_id: str
    user_id: int | None = None
    task_id: str | None = None
    task_source: str | None = None


@dataclass(frozen=True)
class ConnectorResolution:
    """Outcome of resolving an app name for this user.

    ``connector`` is required for ``needs_connection`` and is forwarded as-is
    in the interaction's ``connector`` field. ``message`` is model- and
    user-facing text; ``result`` is extra model-visible data for ``connected``.
    """

    status: ResolutionStatus
    message: str | None = None
    connector: Mapping[str, Any] | None = None
    result: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ConnectorConnectionState:
    """Server-side state of one connect interaction."""

    status: ConnectionStatus
    result: Mapping[str, Any] | None = None


class ConnectorResolver(Protocol):
    async def resolve(self, request: ConnectorResolveRequest) -> ConnectorResolution:
        """Resolve ``request.app_name`` for the requesting user."""

    async def connection_state(
        self,
        *,
        interaction_id: str,
        user_id: int | None,
        task_id: str | None,
        task_source: str | None,
    ) -> ConnectorConnectionState | None:
        """Return the stored state of ``interaction_id``, or ``None`` if unknown."""


_resolver: ConnectorResolver | None = None


def set_connector_resolver_hook(resolver: ConnectorResolver | None) -> None:
    """Register (or clear with ``None``) the host's connector resolver."""

    global _resolver
    _resolver = resolver


def get_connector_resolver_hook() -> ConnectorResolver | None:
    return _resolver
