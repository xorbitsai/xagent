"""Privilege-isolation pins for WebToolConfig identity.

When the runtime builds a tool config for a task OWNER while an admin is the
acting principal on the request, the config must reflect the owner -- not get
silently widened to admin scope by the request.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.web import auth_dependencies
from xagent.web.tools.config import WebToolConfig


def _admin_request(actor_user_id: int = 999):
    """A request whose authenticated user is an admin (the acting principal)."""
    return SimpleNamespace(user=SimpleNamespace(id=actor_user_id, is_admin=True))


def test_explicit_is_admin_false_wins_over_admin_request() -> None:
    """Tri-state: an explicit ``is_admin=False`` (the owner's status) is
    authoritative and is NOT OR-ed with the admin request. Otherwise an admin
    acting on another user's task would get admin-scoped tools/visibility."""
    cfg = WebToolConfig(
        db=None,
        request=_admin_request(actor_user_id=999),
        user_id=42,  # the task owner
        is_admin=False,  # owner is not an admin
    )
    assert cfg.get_user_id() == 42
    assert cfg.is_admin() is False


def test_unset_is_admin_falls_back_to_request() -> None:
    """When ``is_admin`` is unset (None), fall back to the request's user --
    preserves behavior for callers that don't pass an explicit value."""
    cfg = WebToolConfig(
        db=None,
        request=_admin_request(actor_user_id=7),
        user_id=7,
    )
    assert cfg.is_admin() is True


def test_minimal_request_without_user_is_not_admin() -> None:
    """A minimal request carrying only an id (no ``user``) must resolve to
    non-admin without raising / logging a spurious warning."""
    cfg = WebToolConfig(db=None, request=SimpleNamespace(), user_id=5)
    assert cfg.is_admin() is False


def test_identity_free_config_ignores_request_authentication(monkeypatch) -> None:
    """Identity-free configuration must not derive a user from its request."""
    token_helper_calls: list[tuple[object, object]] = []

    def unexpected_token_helper(token: object, db: object) -> None:
        token_helper_calls.append((token, db))
        raise AssertionError("identity-free config must not authenticate a request")

    request = SimpleNamespace(
        headers={"authorization": "Bearer request-token"},
        query_params={"token": "request-token"},
        user=SimpleNamespace(id=77, is_admin=True),
    )
    database = object()
    monkeypatch.setattr(
        auth_dependencies, "get_user_from_websocket_token", unexpected_token_helper
    )

    cfg = WebToolConfig(db=database, request=request)

    assert cfg.get_user_id() is None
    assert cfg.is_admin() is False
    assert token_helper_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["unavailable", "executable"])
async def test_mcp_config_builders_require_an_explicit_user_identity(
    builder: str,
) -> None:
    cfg = WebToolConfig(db=None, request=None, user_id=None)
    server = SimpleNamespace(
        id=1,
        name="Example",
        description=None,
        transport="unsupported-test-transport",
        managed="external",
        concurrency_safe=False,
        concurrent_tools=[],
    )

    with pytest.raises(RuntimeError, match="require a user identity"):
        if builder == "unavailable":
            cfg._build_unavailable_mcp_config(
                server=server, reason="config_load_failed"
            )
        else:
            await cfg._build_mcp_server_config(
                server=server,
                user_env_by_id={},
                shared_env_by_id={},
                env_source_by_id={},
            )


def test_build_unavailable_mcp_config_omits_app_name_for_a_hidden_catalog_app() -> None:
    """A catalog app with is_visible_in_connector=False must not be named in
    a connect_apps pause: /api/mcp/apps (the frontend's connector catalog)
    excludes it, so naming it would leave the user staring at a dead-end
    pause card with no Connect button to act on."""
    cfg = WebToolConfig(db=None, request=None, user_id=7)
    server = SimpleNamespace(id=1, name="Hidden App", description=None)

    visible = cfg._build_unavailable_mcp_config(
        server=server,
        reason="oauth_token_required",
        failure_code="oauth_token_required",
        app_info={"name": "Hidden App", "is_visible_in_connector": True},
    )
    hidden = cfg._build_unavailable_mcp_config(
        server=server,
        reason="oauth_token_required",
        failure_code="oauth_token_required",
        app_info={"name": "Hidden App", "is_visible_in_connector": False},
    )

    assert visible["config"]["app_name"] == "Hidden App"
    assert "app_name" not in hidden["config"]


@pytest.mark.asyncio
async def test_identity_free_mcp_load_returns_before_cache_or_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = WebToolConfig(db=object(), request=None, user_id=None, include_mcp_tools=True)
    include_mcp_tools = MagicMock()
    include_mcp_tools.__bool__.side_effect = AssertionError(
        "include flag must not be read"
    )
    cfg._include_mcp_tools = include_mcp_tools
    cfg._cached_mcp_configs = [{"user_id": "stale"}]
    load = AsyncMock()
    monkeypatch.setattr(cfg, "_load_mcp_server_configs", load)
    monkeypatch.setattr(
        cfg,
        "_mcp_config_cache_is_valid",
        MagicMock(side_effect=AssertionError("cache must not be read")),
    )

    assert await cfg.get_mcp_server_configs() == []
    load.assert_not_awaited()


def test_skill_scope_context_does_not_retain_request_resources() -> None:
    from xagent.web.services.skill_runtime import build_detached_skill_scope

    db = object()
    request = _admin_request(actor_user_id=999)
    explicit_user = SimpleNamespace(id=42, is_admin=False)
    cfg = WebToolConfig(
        db=db,
        request=request,
        user_id=42,
        is_admin=False,
        user=explicit_user,
    )

    context = cfg.get_skill_scope_context()

    assert context == build_detached_skill_scope(user_id=42)
    assert not hasattr(context, "db")
    assert not hasattr(context, "request")
    assert not hasattr(context, "user")
