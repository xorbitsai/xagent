"""The identity adapter: how the engine runs checks as someone.

An identity is whoever the checks run as: the Toby QA member under the
workspace's actor policy (in Xagent SaaS), or a local user for verifying the
engine. The engine knows nothing about either. It asks the adapter for the
identity's connectors, opens one session per connector through it, and
resolves fixture keys through it.

The adapter decides, from its own enumeration and never from a result, which
connectors it cannot start: such a target carries a ``skip_reason`` code and
the engine reports its session as ``skip`` (``skipped_by_identity``) without
calling the adapter again. A path the adapter cannot exercise can fail exactly
like a real failure, which is why the decision is made up front.

``open_session`` starts the connector the way a task of that identity would
(``services/mcp_connector_session.py``) and returns the listing; it reports a
connector failure through the listing's outcome rather than raising. A
connector the enumeration already knows is broken (a grant that needs
reconnecting) is returned as a non-OK listing without starting anything.

The adapter is responsible for the identity being one that may be probed;
the engine trusts it as given.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from .codes import connector_label, is_code
from .fixtures import FixtureValue

if TYPE_CHECKING:
    from ..services.mcp_connector_session import ConnectorToolsResult


class SessionPurpose(str, Enum):
    """What follows the listing; it decides how a sandboxed connection runs.

    The listing's sandbox lease is released when it returns, so a tool call
    afterwards would start the server on the web host.
    """

    # Tools will be called: a sandboxed (npx/uvx) connection is refused as
    # ``unsupported``/``sandboxed`` before anything starts.
    CALL_TOOLS = "call_tools"
    # No check will call a tool (each is unprobeable, or the app has no
    # declaration): a sandboxed connection lists through a lease, as a task
    # starts it and as V1's Toby probe lists it.
    LIST_ONLY = "list_only"


@dataclass(frozen=True)
class ConnectorTarget:
    """One connector of the identity, as its enumeration saw it.

    ``app_id`` is the catalog app id, unique within one enumeration. An
    admin-created app's id is free text, so it is taken as it is and the
    result names the connector by ``connector`` instead. ``ref`` is the
    adapter's own handle, which ``open_session`` gets back: the Toby
    adapter, for one, starts a server row by id and an actor stdio
    connection by app id. Neither ``app_id`` nor ``ref`` is written to the
    result.
    """

    app_id: str
    skip_reason: str | None = None
    ref: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.app_id, str) or not self.app_id:
            raise ValueError("a connector target needs an app id")
        if self.skip_reason is not None and not is_code(self.skip_reason):
            raise ValueError("a skip reason must be a code")

    @property
    def connector(self) -> str:
        """The connector's name in the result (``codes.connector_label``)."""
        return connector_label(self.app_id)


class IdentityAdapter(Protocol):
    """What the engine needs from an identity."""

    @property
    def label(self) -> str:
        """The identity's public label in the result (``toby``, ``user``):
        never a user id, a workspace id or an email."""

    def enumerate_targets(self) -> Sequence[ConnectorTarget]:
        """The identity's connectors, in the order they should run."""

    async def open_session(
        self,
        target: ConnectorTarget,
        *,
        purpose: SessionPurpose,
    ) -> ConnectorToolsResult:
        """Start ``target`` as this identity and list its tools.

        The engine finds a check's tool by its MCP wire name, so the
        agent-facing name prefix the listing applies is the adapter's choice.
        """

    def resolve_fixture(self, key: str) -> FixtureValue | None:
        """The fixture value for ``key`` in this environment and identity,
        from server-side configuration, or None."""
