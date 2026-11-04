"""The MCP loader boundary wraps tools without changing anything yet.

Two guarantees are covered here, both of which the host wiring depends on:

* With an empty approval registry, a tool the loader wrapped is
  observationally identical to the tool it wrapped -- name, description,
  tags, metadata (including identity), schemas, sandbox flag and dispatch.
  Nothing in production changes until a host registers a gate.
* Every loader seam that materializes MCP tools carries the persisted
  connector identity the gate requires. A gated call whose wrapper has no
  ``ConnectorRef`` fails closed before dispatch, so a seam that drops it
  becomes an outage the moment a host registers a gate for that source.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolMetadata
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRef
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPLoadResult,
    load_mcp_tools_as_agent_tools,
)
from xagent.core.tools.adapters.vibe.mcp_approval_gate import (
    GateDecision,
    gate_mcp_tools,
    register_mcp_approval_gate,
    unregister_mcp_approval_gate,
)


class _Args(BaseModel):
    text: str = ""


class _State(BaseModel):
    seen: int = 0


class _Result(BaseModel):
    ok: bool = True


class _McpTarget(AbstractBaseTool):
    """A stand-in for what ``_load_direct_mcp_tools`` returns."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.sync_calls: list[dict[str, Any]] = []
        self.source_server = "linkedin"
        self._metadata = ToolMetadata(
            name="mcp_LinkedIn_create_post",
            concurrency_safe=True,
            read_only=False,
            source_server="linkedin",
            mcp_non_idempotent_write=True,
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return "mcp_LinkedIn_create_post"

    @property
    def description(self) -> str:
        return "Create a LinkedIn post."

    @property
    def tags(self) -> list[str]:
        return ["mcp", "linkedin"]

    @property
    def is_sandboxed(self) -> bool:
        return True

    def args_type(self) -> type[BaseModel]:
        return _Args

    def return_type(self) -> type[BaseModel]:
        return _Result

    def state_type(self) -> type[BaseModel] | None:
        return _State

    def return_value_as_string(self, value: Any) -> str:
        return f"target:{value}"

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        self.sync_calls.append(dict(args))
        return {"success": True, "via": "sync"}

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        self.calls.append(dict(args))
        return {"success": True, "via": "async"}


async def _load_one_wrapped(
    target: _McpTarget,
    *,
    connector_refs: dict[str, ConnectorRef] | None = None,
) -> Any:
    """Run the real loader boundary and return the single tool it produced."""

    with (
        patch(
            "xagent.core.tools.adapters.vibe.mcp_adapter._load_direct_mcp_tools",
            new=AsyncMock(
                return_value=MagicMock(tools=(target,), failures=()),
            ),
        ),
        patch(
            "xagent.core.tools.adapters.vibe.mcp_adapter.should_sandbox_mcp_connection",
            return_value=False,
        ),
    ):
        result = await load_mcp_tools_as_agent_tools(
            {"linkedin": {"transport": "sse", "url": "https://example.invalid"}},
            connector_refs=connector_refs,
        )
    assert len(result.tools) == 1
    return result.tools[0]


@pytest.mark.asyncio
async def test_loaded_tool_is_indistinguishable_from_its_target_without_a_gate() -> (
    None
):
    """No registration: the wrapper must be a pass-through in every respect."""

    target = _McpTarget()
    wrapped = await _load_one_wrapped(target)

    # It really is the wrapper, not the bare tool -- otherwise this test
    # would pass trivially and prove nothing about the wrapping.
    assert wrapped is not target
    assert wrapped.target is target

    assert wrapped.name == target.name
    assert wrapped.description == target.description
    assert wrapped.tags == target.tags
    assert wrapped.is_sandboxed == target.is_sandboxed
    assert wrapped.args_type() is target.args_type()
    assert wrapped.return_type() is target.return_type()
    assert wrapped.state_type() is target.state_type()
    assert wrapped.return_value_as_string(1) == target.return_value_as_string(1)
    # Metadata identity, not just equality: nothing downstream sees a copy
    # while no gate is registered.
    assert wrapped.metadata is target.metadata
    assert wrapped.metadata.source_server == "linkedin"
    assert wrapped.metadata.concurrency_safe is True

    assert await wrapped.run_json_async({"text": "hi"}) == {
        "success": True,
        "via": "async",
    }
    assert wrapped.run_json_sync({"text": "hi"}) == {"success": True, "via": "sync"}
    assert target.calls == [{"text": "hi"}]
    assert target.sync_calls == [{"text": "hi"}]


@pytest.mark.asyncio
async def test_registering_a_gate_is_what_changes_the_loaded_tool() -> None:
    """The mutation guard for the test above: with a registration present the
    scheduler-visible metadata does change, so an assertion of 'identical'
    is a real claim about the unregistered case rather than a tautology."""

    target = _McpTarget()
    wrapped = await _load_one_wrapped(target)

    async def _gate(**_: Any) -> GateDecision:
        return GateDecision.allow()

    async def _resume(**_: Any) -> None:
        raise AssertionError("resume must not run")

    handle = register_mcp_approval_gate(
        task_source="some-other-host", gate=_gate, resume=_resume
    )
    try:
        assert wrapped.metadata is not target.metadata
        assert wrapped.metadata.concurrency_safe is False
        assert wrapped.metadata.source_server == "linkedin"
    finally:
        unregister_mcp_approval_gate(handle)

    assert wrapped.metadata is target.metadata


@pytest.mark.asyncio
async def test_loader_carries_the_connector_ref_for_its_own_server() -> None:
    target = _McpTarget()
    wrapped = await _load_one_wrapped(
        target, connector_refs={"linkedin": ConnectorRef("mcp", 41)}
    )

    assert wrapped._connector_ref == ConnectorRef("mcp", 41)


@pytest.mark.asyncio
async def test_config_seam_passes_persisted_server_ids_as_connector_refs() -> None:
    """The WebToolConfig/config-driven seam, mirroring the database-path test.

    ``_create_mcp_tools_from_configs`` builds its own transport mapping from
    transport plus nested parameters, dropping the outer persisted ``id``.
    The gate's identity therefore has to travel beside it.
    """

    configs = [
        {
            "id": 12,
            "name": "linkedin",
            "transport": "sse",
            "config": {"url": "https://example.invalid/sse"},
        },
        {
            # No persisted id (an ad-hoc config): no ref, rather than a
            # fabricated one.
            "name": "scratch",
            "transport": "sse",
            "config": {"url": "https://example.invalid/other"},
        },
        {
            # ``True`` is an ``int`` subclass; it must not become server 1.
            "id": True,
            "name": "boolish",
            "transport": "sse",
            "config": {"url": "https://example.invalid/bool"},
        },
        {
            "id": 0,
            "name": "zero",
            "transport": "sse",
            "config": {"url": "https://example.invalid/zero"},
        },
    ]

    load = AsyncMock(
        return_value=MCPLoadResult(tools=(), loaded_servers=(), failures=())
    )
    with patch(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        new=load,
    ):
        await ToolFactory._create_mcp_tools_from_configs(configs)

    connector_refs = load.await_args.kwargs["connector_refs"]
    assert connector_refs == {"linkedin": ConnectorRef("mcp", 12)}
    # All four servers reached the loader; only the persisted one is
    # identifiable, so this is a scoping assertion, not a filtering artifact.
    assert set(load.await_args.args[0]) == {"linkedin", "scratch", "boolish", "zero"}
    # The transport mapping the loader (and, through it, a sandbox guest)
    # receives still carries no persisted identity.
    connections = load.await_args.args[0]
    assert all("id" not in connection for connection in connections.values())


def test_connector_ref_helper_rejects_non_positive_and_bool_ids() -> None:
    refs = ToolFactory._mcp_connector_refs(
        {
            "good": {"id": 7},
            "bool": {"id": True},
            "zero": {"id": 0},
            "negative": {"id": -3},
            "string": {"id": "7"},
            "missing": {},
        }
    )

    assert refs == {"good": ConnectorRef("mcp", 7)}


@pytest.mark.asyncio
async def test_actor_stdio_session_tools_are_gated_with_their_connector_ref() -> None:
    """The Chrome actor-stdio consumer bypasses the generic MCP loader.

    ``consume_chrome_actor_stdio_session`` binds a host-only execution scope
    and builds its adapters through ``load_execution_scoped_chrome_tools``,
    so the loader's wrapping never sees them. They are ordinary dispatchable
    MCP adapters on a live production path, so the factory wraps them at the
    consumption site instead -- with the persisted server id, or a gated call
    would fail closed for want of a connector ref.
    """

    target = _McpTarget()
    consumed: list[Any] = []

    async def consumer(**kwargs: Any) -> list[Any]:
        consumed.append(kwargs)
        return [target]

    configs = [
        {
            "id": 31,
            "name": "chrome",
            "transport": "stdio",
            "config": {"command": "npx", "args": ["chrome-mcp"]},
        }
    ]
    load = AsyncMock(
        return_value=MCPLoadResult(tools=(), loaded_servers=(), failures=())
    )
    with patch(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        new=load,
    ):
        tools = await ToolFactory._create_mcp_tools_from_configs(
            configs,
            actor_stdio_session_identities={"chrome": object()},
            actor_stdio_session_consumer=consumer,
        )

    assert consumed, "the consumer was never reached"
    # The consumer's own tools never go through the loader, so nothing else
    # could have wrapped them.
    load.assert_not_awaited()
    assert len(tools) == 1
    assert tools[0] is not target
    assert tools[0].target is target
    assert tools[0]._connector_ref == ConnectorRef("mcp", 31)


def test_load_summary_reads_source_server_through_metadata() -> None:
    """A wrapped MCP tool must still count toward its server's load summary.

    ``_build_mcp_load_summary`` used to read ``tool.source_server`` directly.
    ``SandboxedToolWrapper`` forwards ``metadata`` but defines neither that
    attribute nor ``__getattr__``, so every sandboxed MCP server silently
    reported zero loaded tools -- a pre-existing gap the gate wrapper would
    otherwise have widened. Reading it off ``metadata`` first fixes both
    wrappers at once; the assertions below pin the real wrapper's shape so
    this stand-in cannot drift away from it.
    """

    from xagent.core.tools.adapters.vibe.mcp_tools import _build_mcp_load_summary
    from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
        SandboxedToolWrapper,
    )

    assert not hasattr(SandboxedToolWrapper, "source_server")
    assert "__getattr__" not in vars(SandboxedToolWrapper)

    class _MetadataOnlyWrapper:
        """The sandbox wrapper's shape: metadata forwarded, nothing else."""

        def __init__(self, target: _McpTarget) -> None:
            self._target = target

        @property
        def metadata(self) -> ToolMetadata:
            return self._target.metadata

    target = _McpTarget()
    configs = [{"name": "linkedin"}]

    assert _build_mcp_load_summary(
        configs, [_MetadataOnlyWrapper(target)]
    ).loaded_servers == ("linkedin",)
    # And the gate wrapper, which reaches it either way on this base.
    assert _build_mcp_load_summary(
        configs, [gate_mcp_tools([target])[0]]
    ).loaded_servers == ("linkedin",)

    # The unwrapped adapter keeps working through the direct-attribute
    # fallback, which is what a tool with no ToolMetadata still needs.
    # ``target`` itself cannot exercise this: its own ``metadata.source_server``
    # is already "linkedin", so the metadata-first read above would satisfy
    # this assertion even if the fallback were deleted entirely. A target
    # whose metadata carries no ``source_server`` is required to actually
    # reach the ``getattr(tool, "source_server", None)`` branch.
    class _DirectAttributeOnlyTarget(_McpTarget):
        """No ``source_server`` on ``metadata`` -- only the direct attribute."""

        def __init__(self) -> None:
            super().__init__()
            self._metadata = ToolMetadata(
                name="mcp_LinkedIn_create_post",
                concurrency_safe=True,
                read_only=False,
                source_server=None,
                mcp_non_idempotent_write=True,
            )

    direct_attribute_target = _DirectAttributeOnlyTarget()
    assert direct_attribute_target.metadata.source_server is None
    assert direct_attribute_target.source_server == "linkedin"
    assert _build_mcp_load_summary(
        configs, [direct_attribute_target]
    ).loaded_servers == ("linkedin",)
