"""Positive tool-allowlist hook, its WebToolConfig wiring, factory filter, and
tool-policy-signature inclusion (added for xagent-saas #81 area A).

The allowlist is a positive counterpart to the disable-set override hook: the
factory keeps only tools whose name is in the list, applied to the already-built
tool list so dynamically-loaded (e.g. MCP) tools are covered.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import xagent.core.tools.adapters.vibe.factory as factory_module
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.web.services.tool_credentials import (
    get_user_tool_allowlist,
    set_user_tool_allowlist_hook,
)
from xagent.web.tools.config import WebToolConfig


@pytest.fixture
def clear_allowlist_hook():
    set_user_tool_allowlist_hook(None)
    yield
    set_user_tool_allowlist_hook(None)


# --------------------------------------------------------------------------- #
# Hook registration
# --------------------------------------------------------------------------- #
def test_no_hook_returns_none(clear_allowlist_hook):
    assert get_user_tool_allowlist(None, None) is None


def test_registered_hook_is_invoked(clear_allowlist_hook):
    set_user_tool_allowlist_hook(lambda db, user: ["a", "b"])
    assert get_user_tool_allowlist(None, None) == ["a", "b"]


# --------------------------------------------------------------------------- #
# WebToolConfig caching / refresh
# --------------------------------------------------------------------------- #
def _config() -> WebToolConfig:
    return WebToolConfig(db=None, request=SimpleNamespace(), user_id=1)


def test_config_reads_and_caches_allowlist(clear_allowlist_hook):
    holder = {"value": ["only_this"]}
    set_user_tool_allowlist_hook(lambda db, user: holder["value"])
    cfg = _config()

    assert cfg.get_user_tool_allowlist() == ["only_this"]
    # Cached: a later change to the hook result is not observed until refresh.
    holder["value"] = ["changed"]
    assert cfg.get_user_tool_allowlist() == ["only_this"]
    assert cfg.refresh_user_tool_allowlist() == ["changed"]


def test_config_caches_none_result(clear_allowlist_hook):
    calls = {"n": 0}

    def _hook(db, user):
        calls["n"] += 1
        return None

    set_user_tool_allowlist_hook(_hook)
    cfg = _config()

    assert cfg.get_user_tool_allowlist() is None
    assert cfg.get_user_tool_allowlist() is None
    # None is a real value, not "uncomputed": the hook is consulted only once.
    assert calls["n"] == 1


def test_config_fails_closed_on_hook_errors(clear_allowlist_hook):
    def _boom(db, user):
        raise RuntimeError("hook failed")

    set_user_tool_allowlist_hook(_boom)
    cfg = _config()
    # A failing hook must not break tool building, but it must not be reported
    # as "no allowlist configured" either: the application enforces
    # authorization through this hook, so an unresolved read denies every tool
    # instead of skipping the positive filter and building the full set.
    assert cfg.get_user_tool_allowlist() == []


def test_config_reports_no_allowlist_when_no_hook_registered(clear_allowlist_hook):
    # Standalone xagent registers no policy hook, so there is no policy to
    # lose: the accessor still means "no filtering", not "deny every tool".
    assert _config().get_user_tool_allowlist() is None


def test_config_denies_when_an_allowlist_read_precedes_a_failed_overrides_read(
    clear_allowlist_hook,
):
    """The deny must not depend on which policy input is read first.

    The allowlist read caches "no allowlist configured". If a later overrides
    read cannot be resolved, that cached ``None`` must not keep reporting "no
    filtering" — which is exactly the full-tool-set fail-open path.
    """
    from unittest.mock import MagicMock

    from xagent.web.services.tool_credentials import set_user_tool_overrides_hook

    set_user_tool_allowlist_hook(lambda db, user: None)
    set_user_tool_overrides_hook(lambda db, user: {"file": {"enabled": False}})
    try:
        request_without_user = MagicMock()
        del request_without_user.user
        cfg = WebToolConfig(db=MagicMock(), request=request_without_user, user_id=42)

        # Allowlist first: nothing has failed yet, so "no allowlist" is right.
        assert cfg.get_user_tool_allowlist() is None
        # The overrides read cannot reach the hook (no runtime user).
        assert cfg.get_user_tool_overrides() == {}
        # That failure has to invalidate the cached allowlist and deny.
        assert cfg.get_user_tool_allowlist() == []
    finally:
        set_user_tool_overrides_hook(None)


@pytest.mark.parametrize("accessor", ["overrides", "allowlist"])
def test_config_propagates_pool_timeouts_instead_of_denying(
    clear_allowlist_hook, accessor
):
    """A pool checkout timeout is not an unresolved policy.

    The next step in the turn needs the same pool, so the caller must get the
    timeout and retry rather than spending the turn with no tools. This matches
    ``_load_tool_runtime_policy_snapshot``, which already re-raises.
    """
    from unittest.mock import MagicMock

    from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

    from xagent.web.services.tool_credentials import set_user_tool_overrides_hook

    checkout_timeout = SQLAlchemyTimeoutError("pool checkout timed out")

    def _timeout(db, user):
        raise checkout_timeout

    set_user_tool_allowlist_hook(_timeout)
    set_user_tool_overrides_hook(_timeout)
    try:
        cfg = WebToolConfig(
            db=MagicMock(), request=MagicMock(user=MagicMock(id=42)), user_id=42
        )
        read = (
            cfg.get_user_tool_overrides
            if accessor == "overrides"
            else cfg.get_user_tool_allowlist
        )

        with pytest.raises(SQLAlchemyTimeoutError) as exc_info:
            read()
        assert exc_info.value is checkout_timeout

        # Nothing recorded and nothing cached: the retry re-reads from scratch
        # rather than inheriting a deny-all from the timed-out attempt.
        assert cfg._unresolved_tool_policy_inputs == set()
    finally:
        set_user_tool_overrides_hook(None)


def test_config_serves_real_policy_after_a_pool_timeout_retry(clear_allowlist_hook):
    """The retry a propagated timeout invites must see the real policy."""
    from unittest.mock import MagicMock

    from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

    failing = {"value": True}

    def _flaky(db, user):
        if failing["value"]:
            raise SQLAlchemyTimeoutError("pool checkout timed out")
        return ["only_this"]

    set_user_tool_allowlist_hook(_flaky)
    cfg = WebToolConfig(
        db=MagicMock(), request=MagicMock(user=MagicMock(id=42)), user_id=42
    )

    with pytest.raises(SQLAlchemyTimeoutError):
        cfg.get_user_tool_allowlist()

    failing["value"] = False
    # No refresh call needed: the timed-out read cached nothing.
    assert cfg.get_user_tool_allowlist() == ["only_this"]


def test_config_keeps_denying_when_an_allowlist_timeout_follows_a_failed_overrides_read(
    clear_allowlist_hook,
):
    """A propagated timeout must not discard an already-unresolved input.

    The allowlist read clears only its own entry before consulting the hook, so
    an overrides input that could not be resolved has to keep denying once the
    retried allowlist read succeeds.
    """
    from unittest.mock import MagicMock

    from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

    from xagent.web.services.tool_credentials import set_user_tool_overrides_hook

    failing = {"value": True}

    def _flaky_allowlist(db, user):
        if failing["value"]:
            raise SQLAlchemyTimeoutError("pool checkout timed out")
        return None  # the hook itself says "no allowlist configured"

    set_user_tool_allowlist_hook(_flaky_allowlist)
    set_user_tool_overrides_hook(lambda db, user: {"file": {"enabled": False}})
    try:
        request_without_user = MagicMock()
        del request_without_user.user
        cfg = WebToolConfig(db=MagicMock(), request=request_without_user, user_id=42)

        # The overrides read cannot reach the hook (no runtime user).
        assert cfg.get_user_tool_overrides() == {}
        with pytest.raises(SQLAlchemyTimeoutError):
            cfg.get_user_tool_allowlist()

        failing["value"] = False
        # The hook now answers "no allowlist", but the unresolved overrides read
        # still denies: a resolved allowlist does not vouch for the other input.
        assert cfg.get_user_tool_allowlist() == []
    finally:
        set_user_tool_overrides_hook(None)


def test_missing_user_does_not_deny_an_allowlist_only_deployment(clear_allowlist_hook):
    """A missing runtime user must not deny when no overrides hook is registered.

    With no overrides hook, ``get_user_tool_overrides`` ignores ``user`` and
    returns ``{}`` regardless, so that input is resolved rather than unresolved.
    Recording it would discard an allowlist the hook returned successfully.
    """
    from unittest.mock import MagicMock

    set_user_tool_allowlist_hook(lambda db, user: ["shell", "file"])
    request_without_user = MagicMock()
    del request_without_user.user
    cfg = WebToolConfig(db=MagicMock(), request=request_without_user, user_id=42)

    # Read order matches the factory's (overrides at factory.py:632, then
    # allowlist at :651) so the cross-input path is exercised.
    assert cfg.get_user_tool_overrides() == {}
    assert cfg.get_user_tool_allowlist() == ["shell", "file"]


def test_missing_user_still_denies_when_the_overrides_hook_is_registered(
    clear_allowlist_hook,
):
    """The narrower gate must not reopen the fail-open path it guards."""
    from unittest.mock import MagicMock

    from xagent.web.services.tool_credentials import set_user_tool_overrides_hook

    set_user_tool_allowlist_hook(lambda db, user: ["shell"])
    set_user_tool_overrides_hook(lambda db, user: {"file": {"enabled": False}})
    try:
        request_without_user = MagicMock()
        del request_without_user.user
        cfg = WebToolConfig(db=MagicMock(), request=request_without_user, user_id=42)

        assert cfg.get_user_tool_overrides() == {}
        assert cfg.get_user_tool_allowlist() == []
    finally:
        set_user_tool_overrides_hook(None)


def test_config_recovers_after_a_transient_hook_failure(clear_allowlist_hook):
    calls = {"n": 0}

    def _flaky(db, user):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("hook failed")
        return ["only_this"]

    set_user_tool_allowlist_hook(_flaky)
    cfg = _config()

    assert cfg.get_user_tool_allowlist() == []
    # Fail closed must not latch: once the hook answers again the turn gets the
    # real policy rather than staying denied for the life of the config.
    assert cfg.refresh_user_tool_allowlist() == ["only_this"]


@pytest.mark.parametrize(
    ("hook_result", "expected"),
    [
        ("only_this", ["only_this"]),  # bare string is one tool name, not chars
        (5, ["5"]),  # non-iterable scalar coerced to a single name
        (True, ["True"]),
        (("a", 2), ["a", "2"]),  # any iterable stringified per item
    ],
)
def test_config_normalizes_scalar_allowlist(
    clear_allowlist_hook, hook_result, expected
):
    # A hook that violates the ``Optional[list]`` contract must not iterate a
    # string character-by-character or crash on a non-iterable scalar; it is
    # coerced to a list of tool-name strings at the WebToolConfig boundary.
    set_user_tool_allowlist_hook(lambda db, user: hook_result)
    assert _config().get_user_tool_allowlist() == expected


# --------------------------------------------------------------------------- #
# Factory positive filter
# --------------------------------------------------------------------------- #
class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeConfig:
    def __init__(self, allowlist) -> None:
        self._allowlist = allowlist

    def get_tool_selection_spec(self):
        return None

    def get_allowed_tools(self):
        return None

    def get_user_tool_overrides(self):
        return {}

    def get_user_tool_allowlist(self):
        return self._allowlist

    def get_sandbox(self):
        return None

    def get_max_output_length(self):
        return 10_000

    def get_max_field_count(self):
        return 100

    def get_max_recursion_depth(self):
        return 10


_UNIVERSE = ["web_search", "python_executor", "mcp__server__do_thing"]


@pytest.fixture
def fake_registry(monkeypatch):
    async def _create_registered_tools(config):
        return [_FakeTool(name) for name in _UNIVERSE]

    monkeypatch.setattr(
        factory_module.ToolRegistry,
        "create_registered_tools",
        _create_registered_tools,
    )


def _build(config, *, apply_user_override_filter=True):
    import asyncio

    tools = asyncio.run(
        ToolFactory.create_all_tools(
            config, apply_user_override_filter=apply_user_override_filter
        )
    )
    return [t.name for t in tools]


def test_factory_keeps_only_allowlisted_including_mcp(fake_registry):
    assert _build(_FakeConfig(["web_search", "mcp__server__do_thing"])) == [
        "web_search",
        "mcp__server__do_thing",
    ]


def test_factory_none_allowlist_keeps_all(fake_registry):
    assert _build(_FakeConfig(None)) == _UNIVERSE


def test_factory_empty_allowlist_drops_all(fake_registry):
    assert _build(_FakeConfig([])) == []


def test_factory_display_layer_ignores_allowlist(fake_registry):
    assert (
        _build(_FakeConfig(["web_search"]), apply_user_override_filter=False)
        == _UNIVERSE
    )


def test_factory_normalizes_scalar_allowlist(fake_registry):
    # A non-WebToolConfig config may hand back a bare string; the factory must
    # treat it as a single tool name, not split it into characters (which would
    # match nothing and silently drop every tool).
    assert _build(_FakeConfig("web_search")) == ["web_search"]


# --------------------------------------------------------------------------- #
# Tool-policy signature includes the allowlist (cache isolation across turns)
# --------------------------------------------------------------------------- #
class _SigConfig:
    def __init__(self, allowlist) -> None:
        self._allowlist = allowlist

    def get_user_tool_allowlist(self):
        return self._allowlist

    def refresh_user_tool_allowlist(self):
        return self._allowlist


def _signature(config):
    from xagent.core.agent.service import AgentService

    svc = AgentService.__new__(AgentService)
    svc.tool_config = config
    return svc._current_tool_policy_signature()


def test_signature_differs_by_allowlist():
    sig_a = _signature(_SigConfig(["a"]))
    sig_b = _signature(_SigConfig(["a", "b"]))
    sig_none = _signature(_SigConfig(None))
    # Different allowlists must not collide, or a reused AgentService could
    # serve one client application's cached tool set to another.
    assert sig_a != sig_b
    assert sig_a != sig_none
    assert sig_b != sig_none


def test_signature_stable_for_same_allowlist():
    assert _signature(_SigConfig(["a", "b"])) == _signature(_SigConfig(["a", "b"]))


def test_signature_normalizes_scalar_allowlist():
    # A scalar allowlist and its single-element list form must hash identically,
    # so the signature does not depend on which the config happens to return.
    assert _signature(_SigConfig("a")) == _signature(_SigConfig(["a"]))


def test_signature_backward_compatible_without_allowlist_methods():
    # A legacy config lacking the allowlist methods must still produce a
    # signature (the allowlist slot is simply None).
    legacy = SimpleNamespace()
    from xagent.core.agent.service import AgentService

    svc = AgentService.__new__(AgentService)
    svc.tool_config = legacy
    # truthy config required; SimpleNamespace() is truthy.
    assert isinstance(svc._current_tool_policy_signature(), tuple)
