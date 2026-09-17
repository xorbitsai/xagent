"""Unit tests for the ``tools_initialized`` ternary semantics in AgentService.

Before this fix, ``AgentService.__init__`` unconditionally set
``self._tools_initialized = False`` even when the caller already provided
a fully-built ``tools`` list. ``_ensure_tools_initialized`` then ran on
the first ``execute_task`` and rebuilt tools via ``ToolFactory`` against
``tool_config`` -- producing the duplicate ToolFactory construction
observed in production logs.

The fix wires the field from an inferred-or-explicit
``tools_initialized: bool | None`` parameter:

  - ``tools_initialized=None`` (default): infer from ``tools is not None``
    (an empty list is a legitimate "no tools allowed" outcome, not a
    "tools not built" signal -- hence ``is not None`` rather than
    ``bool(tools)``).
  - ``tools_initialized=True`` (explicit): caller asserts the tool list
    is final, ``_ensure_tools_initialized`` short-circuits.
  - ``tools_initialized=False`` (explicit): caller wants the lazy
    supplement to run even if ``tools`` was pre-populated (specialized
    tests / future callers).

These tests pin the four behaviorally distinct combinations.
"""

from __future__ import annotations

from xagent.core.agent.service import AgentService


def _make_service(**kwargs):
    """Construct a minimal AgentService; only ``_tools_initialized`` is
    asserted, so we pass the bare minimum required args."""
    return AgentService(
        name="test-service",
        id="test-id",
        enable_workspace=False,
        **kwargs,
    )


def test_default_tools_none_keeps_lazy_path():
    """Caller did not pre-build tools -- lazy supplement must remain
    armed so the agent can still populate its tool set via
    ``_ensure_tools_initialized`` on first execute."""
    service = _make_service(tools=None)
    assert service.tools == []
    assert service._tools_initialized is False


def test_default_tools_empty_list_counts_as_initialized():
    """Caller explicitly passed ``tools=[]`` -- this is a legitimate
    'no tools allowed for this agent' result from the caller's build
    path. The lazy supplement must NOT run and overwrite it with
    rebuilt-from-tool_config tools."""
    service = _make_service(tools=[])
    assert service.tools == []
    assert service._tools_initialized is True


def test_default_tools_non_empty_counts_as_initialized():
    """Caller passed a fully-built tool list (the chat.py production
    path). ``_ensure_tools_initialized`` must short-circuit on first
    execute to avoid the duplicate ToolFactory build the original bug
    produced."""
    sentinel_tool = object()
    service = _make_service(tools=[sentinel_tool])
    assert service.tools == [sentinel_tool]
    assert service._tools_initialized is True


def test_explicit_false_overrides_default_inference():
    """Even when ``tools`` is pre-populated, an explicit
    ``tools_initialized=False`` keeps the lazy supplement armed. Use
    case: specialized test that wants both pre-seeded tools AND the
    lazy rebuild path to run."""
    service = _make_service(tools=[object()], tools_initialized=False)
    assert service._tools_initialized is False


def test_explicit_true_overrides_default_inference():
    """Caller didn't pre-build but explicitly asserts initialized,
    suppressing the lazy supplement. Use case: caller plans to
    populate ``service.tools`` later through a non-standard path and
    wants the lazy rebuild disabled."""
    service = _make_service(tools=None, tools_initialized=True)
    assert service.tools == []
    assert service._tools_initialized is True


def test_tools_list_is_copied_not_aliased():
    """The constructor materializes ``self.tools`` from the caller's
    list so mutating the caller-side list after construction can't
    silently change the agent's tool set."""
    caller_list = [object()]
    service = _make_service(tools=caller_list)
    caller_list.append(object())
    assert len(service.tools) == 1


def test_default_tool_configuration_can_be_disabled() -> None:
    service = _make_service(tools=[], enable_default_tools=False)

    assert service.tools == []
    assert service.tool_config is None
    assert service.enable_default_tools is False
