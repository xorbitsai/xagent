"""Running one connector's declared read-only probe (L3).

The connector is started through the connector session module with the MCP
loader replaced by a fake that hands back adapters whose calls are scripted,
so the tests cover the probe's own logic: finding the declared tool, judging
the tool's envelope, bounding the call, and never leaking the tool's text.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.web.services.test_mcp_connector_session import (
    LOADER,
    SANDBOX_MANAGER,
    _connect_github,
    _link,
)
from xagent.builtin_identity import owned_catalog_marker
from xagent.core.tools.adapters.vibe.mcp_adapter import (
    MCPFailurePhase,
    MCPLoadResult,
    MCPServerLoadFailure,
)
from xagent.core.utils.encryption import encrypt_env_dict
from xagent.web import mcp_apps
from xagent.web.api.mcp import (
    _ensure_catalog_app_server,
    _ensure_catalog_mcp_oauth_server,
)
from xagent.web.builtin_mcp_registry import seed_builtin_oauth_and_public_mcp_apps
from xagent.web.connector_probes import ConnectorProbe
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth
from xagent.web.services import connector_probe
from xagent.web.services.connector_probe import (
    ConnectorProbeBusy,
    ConnectorProbeOutcome,
    ConnectorProbeResult,
    evaluate_probe_result,
    is_connector_probe_user,
    run_connector_probe,
)

SECRET_TEXT = "Bad credentials for token ghp_secret at https://api.github.com/user"


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": is_error}


class _Adapter:
    """A loaded MCP tool adapter: ``mcp_tool.name`` is the wire name."""

    def __init__(self, name: str, result: Any = None, *, delay: float = 0.0) -> None:
        self.mcp_tool = SimpleNamespace(name=name)
        self.name = f"probe_server_{name}"
        self.result = result
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    async def run_json_async(self, args):
        self.calls.append(dict(args))
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _Loader:
    def __init__(self, *adapters: _Adapter, result: MCPLoadResult | None = None):
        self.adapters = adapters
        self.result = result
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def __call__(self, connections, **kwargs):
        self.calls.append((connections, kwargs))
        if self.result is not None:
            return self.result
        return MCPLoadResult(
            tools=tuple(self.adapters),
            loaded_servers=tuple(connections),
            failures=(),
        )


def _github_ok(**extra: Any) -> _Adapter:
    payload = {"status": "success", "user": {"login": "monitor"}, **extra}
    return _Adapter("github_get_current_user", _text_result(json.dumps(payload)))


def _link_without_owning(db, user, server: MCPServer, **fields: Any) -> None:
    """The connect routes' link: the user never owns the shared catalog row."""
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=False,
            can_edit=False,
            can_delete=True,
            is_active=True,
            **fields,
        )
    )
    db.commit()


def _connect_catalog_stripe(db, user) -> MCPServer:
    """Connect the shared catalog stripe row the way POST /apps/{id}/connect
    does: the row has no owner, and the user's link is a non-owner link."""
    server, _ = _ensure_catalog_app_server(db, "stripe")
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=False,
            can_edit=False,
            can_delete=True,
            is_active=True,
            env=encrypt_env_dict({"STRIPE_API_KEY": "user-key"}),
            env_source="own",
        )
    )
    db.commit()
    return server


def _stripe_ok() -> _Adapter:
    payload = {"status": "success", "customers": [], "truncated": False}
    return _Adapter("stripe_list_customers", _text_result(json.dumps(payload)))


class _ForbiddenSandboxManager:
    """A sandbox manager whose lease methods must never run."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_or_create_lease_provider(self, lifecycle: str, user_id: str):
        self.calls.append("get_or_create_lease_provider")
        raise AssertionError("the probe must not lease a sandbox")

    async def attach_provider(self, lifecycle: str, user_id: str, provider) -> bool:
        self.calls.append("attach_provider")
        raise AssertionError("the probe must not attach a sandbox")

    async def release(self, lifecycle: str, user_id: str) -> None:
        self.calls.append("release")


@pytest.fixture()
def db_users(tmp_path):
    """The connector session tests' database: the seeded catalog plus two
    plain users."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'connector-probe.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        seed_builtin_oauth_and_public_mcp_apps(connection)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="monitor", password_hash="x", is_admin=False)
    other_user = User(username="other", password_hash="x", is_admin=False)
    db.add_all([user, other_user])
    db.commit()
    db.refresh(user)
    db.refresh(other_user)
    yield db, user, other_user
    db.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def _no_sandbox_manager(monkeypatch):
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: None)


@pytest.fixture(autouse=True)
def _not_busy():
    connector_probe._probe_running = False
    yield
    connector_probe._probe_running = False


# --- the whole run ---------------------------------------------------------


@pytest.mark.asyncio
async def test_declared_probe_passes_on_a_success_envelope(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    adapter = _github_ok()
    loader = _Loader(_Adapter("github_create_issue"), adapter)
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result == ConnectorProbeResult(
        ConnectorProbeOutcome.OK, server.id, "github", "github_get_current_user", None
    )
    assert adapter.calls == [{}]
    # The connector was started as a task would start it, with the token.
    ((connections, _),) = loader.calls
    (connection,) = connections.values()
    assert connection["env"]["GITHUB_ACCESS_TOKEN"] == "gh-token"


@pytest.mark.asyncio
async def test_declared_arguments_are_passed_as_is(db_users, monkeypatch):
    db, user, _ = db_users
    app = mcp_apps.get_app_by_id(db, "slack")
    assert app is not None
    server = MCPServer(
        name=app["name"],
        managed="external",
        transport="oauth",
        auth={
            "app_id": "slack",
            "provider": "slack",
            "builtin_provenance": owned_catalog_marker(app),
        },
    )
    db.add(server)
    db.commit()
    _link(db, user, server)
    adapter = _Adapter(
        "slack_list_channels",
        _text_result(json.dumps({"status": "success", "channels": []})),
    )
    monkeypatch.setattr(
        "xagent.web.services.connector_probe.list_connector_tools",
        _listing_of(server.id, "slack-mcp", adapter),
    )

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.OK
    assert adapter.calls == [{"limit": 1}]


def _listing_of(server_id: int, name: str, *adapters: _Adapter):
    from xagent.web.services.mcp_connector_session import (
        ConnectorToolsOutcome,
        ConnectorToolsResult,
    )

    async def listing(db, *, user, server_id, name_prefix, **_):
        return ConnectorToolsResult(
            ConnectorToolsOutcome.OK,
            server_id,
            name,
            load_result=MCPLoadResult(tuple(adapters), (name,), ()),
        )

    return listing


@pytest.mark.asyncio
async def test_error_envelope_fails_the_probe_without_its_text(db_users, monkeypatch):
    """The in-repo servers catch vendor errors and answer success on the
    wire with ``{"status": "error"}``; the flag alone would pass a revoked
    token."""
    db, user, _ = db_users
    server = _connect_github(db, user)
    adapter = _Adapter(
        "github_get_current_user",
        _text_result(json.dumps({"status": "error", "message": SECRET_TEXT})),
    )
    monkeypatch.setattr(LOADER, _Loader(adapter))

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.PROBE_ASSERTION_FAILED
    assert result.reason == "status_not_success"
    assert "ghp_secret" not in repr(result)
    assert "ghp_secret" not in json.dumps(result.payload())


@pytest.mark.asyncio
async def test_flagged_error_fails_the_probe(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    adapter = _Adapter(
        "github_get_current_user", _text_result(SECRET_TEXT, is_error=True)
    )
    monkeypatch.setattr(LOADER, _Loader(adapter))

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.PROBE_FAILED
    assert result.reason == "tool_error"
    assert "ghp_secret" not in json.dumps(result.payload())


@pytest.mark.asyncio
async def test_missing_declared_tool_is_reported(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    monkeypatch.setattr(LOADER, _Loader(_Adapter("github_search_code")))

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.PROBE_TOOL_MISSING
    assert result.tool == "github_get_current_user"


@pytest.mark.asyncio
async def test_raising_tool_is_a_probe_failure(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    adapter = _Adapter("github_get_current_user", RuntimeError(SECRET_TEXT))
    monkeypatch.setattr(LOADER, _Loader(adapter))

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.PROBE_FAILED
    assert result.reason == "exception"
    assert "ghp_secret" not in json.dumps(result.payload())


@pytest.mark.asyncio
async def test_call_is_bounded_by_the_init_timeout(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    adapter = _Adapter("github_get_current_user", _text_result("{}"), delay=5)
    monkeypatch.setattr(LOADER, _Loader(adapter))
    monkeypatch.setattr(
        "xagent.web.services.connector_probe.get_mcp_tool_init_timeout_seconds",
        lambda: 0.05,
    )

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.TIMED_OUT
    assert result.reason == "probe_call"


@pytest.mark.asyncio
async def test_call_is_bounded_even_when_the_init_timeout_is_disabled(
    db_users, monkeypatch
):
    db, user, _ = db_users
    server = _connect_github(db, user)
    adapter = _Adapter("github_get_current_user", _text_result("{}"), delay=5)
    monkeypatch.setattr(LOADER, _Loader(adapter))
    monkeypatch.setattr(
        "xagent.web.services.connector_probe.get_mcp_tool_init_timeout_seconds",
        lambda: 0,
    )
    monkeypatch.setattr(connector_probe, "_PROBE_CALL_FALLBACK_SECONDS", 0.05)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.TIMED_OUT
    assert result.reason == "probe_call"
    assert connector_probe._probe_running is False


@pytest.mark.asyncio
async def test_not_connected_passes_the_builder_reason_through(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user, token=None)
    loader = _Loader(_github_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.NOT_CONNECTED
    assert result.reason == "oauth_token_required"
    assert result.payload()["code"] == "oauth_token_required"
    assert loader.calls == []


@pytest.mark.asyncio
async def test_start_failures_are_passed_through(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    failed = MCPLoadResult(
        tools=(),
        loaded_servers=(),
        failures=(
            MCPServerLoadFailure(
                "github", MCPFailurePhase.INITIALIZE, "ProcessLookupError"
            ),
        ),
    )
    monkeypatch.setattr(LOADER, _Loader(result=failed))

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.FAILED_TO_START
    assert result.reason == MCPFailurePhase.INITIALIZE.value
    assert result.http_status == 502
    assert result.payload()["code"] == "mcp_tools_unavailable"


@pytest.mark.asyncio
async def test_a_server_the_user_has_not_linked_is_not_found(db_users, monkeypatch):
    db, user, other_user = db_users
    server = _connect_github(db, other_user)
    monkeypatch.setattr(LOADER, _Loader(_github_ok()))

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result == ConnectorProbeResult(ConnectorProbeOutcome.NOT_FOUND, server.id)
    assert result.http_status == 404


@pytest.mark.asyncio
async def test_remote_mcp_connector_is_unprobeable(db_users, monkeypatch):
    """The shared catalog Notion row (the one catalog connect creates: the
    app's transport and URL, no owner) is covered by L2."""
    db, user, _ = db_users
    server, _ = _ensure_catalog_mcp_oauth_server(db, "notion")
    _link_without_owning(db, user, server)
    loader = _Loader(_github_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.UNPROBEABLE
    assert result.reason == "covered_by_l2"
    assert result.http_status == 400
    assert result.payload() == {
        "code": "unprobeable",
        "outcome": "unprobeable",
        "server_id": server.id,
        "app_id": "notion",
        "tool": None,
        "reason": "covered_by_l2",
    }
    assert loader.calls == []


@pytest.mark.asyncio
async def test_custom_server_row_has_no_probe(db_users, monkeypatch):
    db, user, _ = db_users
    server = MCPServer(name="mine", managed="external", transport="stdio")
    server.command = "python"
    db.add(server)
    db.commit()
    _link(db, user, server)
    loader = _Loader(_github_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.NOT_DECLARED
    assert result.http_status == 404
    assert loader.calls == []


# --- the row must be the catalog's, and must run where the probe calls it ---


def _custom_row(name: str, **fields: Any) -> MCPServer:
    return MCPServer(
        name=name,
        managed="external",
        auth={"app_id": "stripe"},
        **fields,
    )


@pytest.mark.asyncio
async def test_custom_remote_row_cannot_claim_a_catalog_identity(db_users, monkeypatch):
    """A custom row's ``auth`` is caller-authored: stamping it with a catalog
    app id must not buy that app's probe (a lookalike tool would pass it)."""
    db, user, _ = db_users
    server = _custom_row(
        "my-stripe", transport="streamable_http", url="https://evil.example.com/mcp"
    )
    db.add(server)
    db.commit()
    _link(db, user, server)
    loader = _Loader(_stripe_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result == ConnectorProbeResult(ConnectorProbeOutcome.NOT_DECLARED, server.id)
    assert result.http_status == 404
    assert result.app_id is None
    assert loader.calls == []


@pytest.mark.asyncio
async def test_custom_stdio_row_cannot_claim_a_catalog_identity(db_users, monkeypatch):
    db, user, _ = db_users
    server = _custom_row("my-stripe", transport="stdio", command="python")
    db.add(server)
    db.commit()
    _link(db, user, server)
    loader = _Loader(_stripe_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result == ConnectorProbeResult(ConnectorProbeOutcome.NOT_DECLARED, server.id)
    assert result.http_status == 404
    assert loader.calls == []


@pytest.mark.asyncio
async def test_user_owned_row_under_a_catalog_id_is_a_squat(db_users, monkeypatch):
    """Named exactly like the app, but owned by a user: a squat created
    before catalog ids were reserved, not the shared catalog row."""
    db, user, _ = db_users
    server = MCPServer(
        name="stripe", managed="external", transport="stdio", command="python"
    )
    db.add(server)
    db.commit()
    _link(db, user, server)
    loader = _Loader(_stripe_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result == ConnectorProbeResult(ConnectorProbeOutcome.NOT_DECLARED, server.id)
    assert loader.calls == []


def test_is_user_owned_server_looks_only_at_owner_links(db_users):
    db, user, other_user = db_users
    shared, _ = _ensure_catalog_app_server(db, "stripe")
    assert mcp_apps.is_user_owned_server(db, shared.id) is False
    _link_without_owning(db, user, shared)
    assert mcp_apps.is_user_owned_server(db, shared.id) is False
    _link(db, other_user, shared)
    assert mcp_apps.is_user_owned_server(db, shared.id) is True


def test_catalog_connect_still_rejects_a_user_owned_row(db_users):
    from fastapi import HTTPException

    db, user, _ = db_users
    server, _ = _ensure_catalog_app_server(db, "stripe")
    _link(db, user, server)

    with pytest.raises(HTTPException) as caught:
        _ensure_catalog_app_server(db, "stripe")

    assert caught.value.status_code == 409
    assert "user-owned" in caught.value.detail


@pytest.mark.asyncio
async def test_catalog_api_key_row_passes_its_declared_probe(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_catalog_stripe(db, user)
    adapter = _stripe_ok()
    loader = _Loader(adapter)
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result == ConnectorProbeResult(
        ConnectorProbeOutcome.OK, server.id, "stripe", "stripe_list_customers", None
    )
    assert adapter.calls == [{"limit": 1}]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["node", "npx"])
async def test_catalog_row_whose_command_drifted_is_not_the_catalog_row(
    db_users, monkeypatch, command
):
    """Catalog connect adopts an ownerless row only when it launches what the
    catalog launches; one whose own command differs is not the catalog row, so
    nothing starts (and for npx, nothing is leased or run on the host)."""
    db, user, _ = db_users
    server = _connect_catalog_stripe(db, user)
    server.command = command
    server.args = ["-y", "some-pkg"]
    db.commit()
    manager = _ForbiddenSandboxManager()
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: manager)
    loader = _Loader(_stripe_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result == ConnectorProbeResult(ConnectorProbeOutcome.NOT_DECLARED, server.id)
    assert result.http_status == 404
    assert manager.calls == []
    assert loader.calls == []


@pytest.mark.asyncio
async def test_remote_row_under_a_catalog_id_must_have_the_catalog_url(
    db_users, monkeypatch
):
    db, user, _ = db_users
    server, _ = _ensure_catalog_mcp_oauth_server(db, "notion")
    server.url = "https://evil.example.com/mcp"
    db.commit()
    _link_without_owning(db, user, server)
    loader = _Loader(_github_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result == ConnectorProbeResult(ConnectorProbeOutcome.NOT_DECLARED, server.id)
    assert loader.calls == []


def _make_catalog_stripe_launch_npx(db, monkeypatch) -> None:
    """The catalog's stripe app launches through npx (its persisted launch
    config, read as it is for a custom app); the row created from it then
    matches the catalog and is a launch a task would sandbox."""
    monkeypatch.setattr(
        "xagent.web.mcp_apps.get_builtin_execution_fields_and_optional_scopes",
        lambda app_id: (None, []),
    )
    app = db.query(PublicMCPApp).filter(PublicMCPApp.app_id == "stripe").one()
    app.launch_config = {
        **app.launch_config,
        "command": "npx",
        "args": ["-y", "stripe-mcp"],
    }
    db.commit()


@pytest.mark.asyncio
async def test_a_catalog_row_a_task_would_sandbox_fails_closed(db_users, monkeypatch):
    """Identity passes (the row matches the catalog), but the effective
    connection is one a task would sandbox, and the probe's call would open a
    session on the web host after the listing released its lease."""
    db, user, _ = db_users
    _make_catalog_stripe_launch_npx(db, monkeypatch)
    server = _connect_catalog_stripe(db, user)
    assert server.command == "npx"
    loader = _Loader(_stripe_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.UNSUPPORTED
    assert result.reason == "sandboxed"
    assert result.tool == "stripe_list_customers"
    assert loader.calls == []


@pytest.mark.asyncio
async def test_a_sandboxed_row_never_leases_a_sandbox_for_the_probe(
    db_users, monkeypatch
):
    db, user, _ = db_users
    _make_catalog_stripe_launch_npx(db, monkeypatch)
    server = _connect_catalog_stripe(db, user)
    manager = _ForbiddenSandboxManager()
    monkeypatch.setattr(SANDBOX_MANAGER, lambda: manager)
    loader = _Loader(_stripe_ok())
    monkeypatch.setattr(LOADER, loader)

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result.outcome is ConnectorProbeOutcome.UNSUPPORTED
    assert result.reason == "sandboxed"
    assert manager.calls == []
    assert loader.calls == []


@pytest.mark.asyncio
async def test_legacy_unstamped_oauth_row_named_by_display_name_still_probes(
    db_users, monkeypatch
):
    """Builtin OAuth rows are written by the platform and launch from the
    catalog; one that predates ``auth.app_id`` and carries the display name
    keeps resolving through the name."""
    db, user, _ = db_users
    app = mcp_apps.get_app_by_id(db, "github")
    assert app is not None
    server = MCPServer(name=app["name"], managed="external", transport="oauth")
    assert server.name == "GitHub"
    db.add(server)
    db.commit()
    _link(db, user, server)
    db.add(
        UserOAuth(
            user_id=user.id,
            provider="github",
            access_token="gh-token",
            refresh_token="gh-refresh",
            provider_user_id="1",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
    )
    db.commit()
    adapter = _github_ok()
    monkeypatch.setattr(LOADER, _Loader(adapter))

    result = await run_connector_probe(db, user=user, server_id=server.id)

    assert result == ConnectorProbeResult(
        ConnectorProbeOutcome.OK, server.id, "github", "github_get_current_user", None
    )
    assert adapter.calls == [{}]


@pytest.mark.asyncio
async def test_one_probe_at_a_time(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)
    adapter = _Adapter("github_get_current_user", _text_result("{}"), delay=0.2)
    monkeypatch.setattr(LOADER, _Loader(adapter))

    first = asyncio.create_task(run_connector_probe(db, user=user, server_id=server.id))
    await asyncio.sleep(0.05)
    with pytest.raises(ConnectorProbeBusy):
        await run_connector_probe(db, user=user, server_id=server.id)
    await first
    # The slot is free again once the first probe finishes.
    adapter.delay = 0
    adapter.result = _text_result(json.dumps({"status": "success", "user": {}}))
    second = await run_connector_probe(db, user=user, server_id=server.id)
    assert second.outcome is ConnectorProbeOutcome.OK


@pytest.mark.asyncio
async def test_slot_is_released_when_the_run_raises(db_users, monkeypatch):
    db, user, _ = db_users
    server = _connect_github(db, user)

    async def boom(*args, **kwargs):
        raise RuntimeError("loader broke")

    monkeypatch.setattr(
        "xagent.web.services.connector_probe.list_connector_tools", boom
    )
    with pytest.raises(RuntimeError):
        await run_connector_probe(db, user=user, server_id=server.id)
    assert connector_probe._probe_running is False


# --- judging a result ------------------------------------------------------

_PLAIN = ConnectorProbe("t")
_KEYED = ConnectorProbe("t", expect_keys=("sub",))


@pytest.mark.parametrize(
    ("raw", "probe", "expected"),
    [
        (
            _text_result('{"status": "success"}'),
            _PLAIN,
            (ConnectorProbeOutcome.OK, None),
        ),
        (
            _text_result('{"status": "success", "sub": "1"}'),
            _KEYED,
            (ConnectorProbeOutcome.OK, None),
        ),
        (
            _text_result('{"sub": "1", "name": "x"}'),
            _KEYED,
            (ConnectorProbeOutcome.OK, None),
        ),
        (
            _text_result('{"status": "error", "message": "boom"}'),
            _PLAIN,
            (ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "status_not_success"),
        ),
        (
            _text_result('{"status": "success"}'),
            _KEYED,
            (ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "missing_keys"),
        ),
        (
            _text_result('{"name": "x"}'),
            _KEYED,
            (ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "missing_keys"),
        ),
        (
            _text_result('{"user": {}}'),
            _PLAIN,
            (ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "no_status"),
        ),
        (
            _text_result(
                "Error: LINKEDIN_ACCESS_TOKEN environment variable is missing"
            ),
            _KEYED,
            (ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "not_a_json_object"),
        ),
        (
            _text_result("[1, 2]"),
            _PLAIN,
            (ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "not_a_json_object"),
        ),
        (
            {"content": [], "is_error": False},
            _PLAIN,
            (ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "not_a_json_object"),
        ),
        (
            {
                "content": [
                    {"type": "text", "text": "{}"},
                    {"type": "text", "text": "{}"},
                ],
                "is_error": False,
            },
            _PLAIN,
            (ConnectorProbeOutcome.PROBE_ASSERTION_FAILED, "not_a_json_object"),
        ),
        (
            _text_result('{"status": "success"}', is_error=True),
            _PLAIN,
            (ConnectorProbeOutcome.PROBE_FAILED, "tool_error"),
        ),
        (
            "not a mapping",
            _PLAIN,
            (ConnectorProbeOutcome.PROBE_FAILED, "invalid_result"),
        ),
        (None, _PLAIN, (ConnectorProbeOutcome.PROBE_FAILED, "invalid_result")),
    ],
)
def test_evaluate_probe_result(raw, probe, expected):
    assert evaluate_probe_result(raw, probe) == expected


# --- who may probe ---------------------------------------------------------


def _user(username: str, email: str | None) -> SimpleNamespace:
    return SimpleNamespace(username=username, email=email, is_admin=True)


def test_nobody_may_probe_when_unset(monkeypatch):
    monkeypatch.delenv("XAGENT_CONNECTOR_PROBE_USERS", raising=False)
    assert is_connector_probe_user(_user("monitor", "monitor@example.com")) is False


def test_allowlist_matches_username_exactly(monkeypatch):
    monkeypatch.setenv("XAGENT_CONNECTOR_PROBE_USERS", "monitor, ops")
    assert is_connector_probe_user(_user("monitor", None)) is True
    assert is_connector_probe_user(_user("ops", None)) is True
    assert is_connector_probe_user(_user("Monitor", None)) is False
    assert is_connector_probe_user(_user("monitor2", None)) is False


def test_allowlist_matches_email_case_insensitively(monkeypatch):
    monkeypatch.setenv("XAGENT_CONNECTOR_PROBE_USERS", "Monitor@Example.com")
    assert is_connector_probe_user(_user("u1", "monitor@example.com")) is True
    assert is_connector_probe_user(_user("u2", " MONITOR@example.com ")) is True
    assert is_connector_probe_user(_user("u3", "other@example.com")) is False
    assert is_connector_probe_user(_user("u4", None)) is False


def test_admin_flag_grants_nothing(monkeypatch):
    monkeypatch.setenv("XAGENT_CONNECTOR_PROBE_USERS", "monitor")
    assert is_connector_probe_user(_user("admin", "admin@example.com")) is False
