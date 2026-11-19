"""Task-supplied OAuth owner selectors on custom MCP servers under an actor policy.

A custom (non-catalog) remote MCP OAuth server is linked to the executing user,
and several actors can hold grants on it. Under an active actor policy a task
must not choose another actor's grant through ``mcp_auth_context`` or the
runtime ``auth_selector`` (#2155). Only the policy's own owner and the
executing user's default owner may be named.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.user import User
from xagent.web.services.mcp_runtime import MCPActorAuthorizationPolicy
from xagent.web.tools.config import (
    ResolvedToken,
    WebToolConfig,
    set_oauth_token_resolver_hook,
)

OWNER_A = "embedding:workspace:actor-a"
OWNER_B = "embedding:workspace:actor-b"
URL = "https://custom.example/mcp"


@pytest.fixture()
def seeded(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'custom-owner.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = session_factory()
    user = User(username="shared-executing-user", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    server = MCPServer(
        name="custom-remote",
        description="custom OAuth server",
        managed="external",
        transport="streamable_http",
        url=URL,
        auth={
            "type": "mcp_oauth",
            "resource": URL,
            "issuer": "https://issuer.example",
            "client_id": "custom-client",
            "scope": "read",
        },
    )
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=int(user.id),
            mcpserver_id=int(server.id),
            is_owner=True,
            is_active=True,
        )
    )
    client = MCPOAuthClient(
        mcp_server_id=int(server.id),
        issuer="https://issuer.example",
        authorization_endpoint="https://issuer.example/authorize",
        token_endpoint="https://issuer.example/token",
        client_id="custom-client",
        token_endpoint_auth_method="none",
        redirect_uri="https://xagent.example/callback",
    )
    db.add(client)
    db.flush()
    for owner, token in (
        (OWNER_A, "actor-a-token"),
        (OWNER_B, "actor-b-token"),
        (f"xagent:user:{user.id}", "workspace-token"),
    ):
        db.add(
            MCPOAuthGrant(
                mcp_server_id=int(server.id),
                user_id=int(user.id),
                mcp_oauth_client_id=int(client.id),
                resource_owner_key=owner,
                issuer="https://issuer.example",
                resource=URL,
                scope="read",
                access_token=encrypt_value(token),
                expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
                status="active",
            )
        )
    db.commit()
    try:
        yield SimpleNamespace(
            db=db,
            user=user,
            server=server,
            session_factory=session_factory,
            workspace_dir=str(tmp_path / "workspace"),
        )
    finally:
        set_oauth_token_resolver_hook(None)
        db.close()
        engine.dispose()


def _config(
    seeded,
    *,
    policy: MCPActorAuthorizationPolicy | None,
    mcp_auth_context: dict | None = None,
    auth_selector: dict | None = None,
) -> WebToolConfig:
    config = WebToolConfig(
        db=seeded.db,
        db_factory=seeded.session_factory,
        request=None,
        user=seeded.user,
        user_id=int(seeded.user.id),
        workspace_config={"base_dir": seeded.workspace_dir, "task_id": "1"},
        mcp_auth_context=mcp_auth_context,
        mcp_runtime_authorization_policy=policy,
    )
    if auth_selector is not None:
        config._connector_runtime_view = {
            f"mcp:{seeded.server.id}": {"auth_selector": auth_selector}
        }
    return config


def _policy() -> MCPActorAuthorizationPolicy:
    return MCPActorAuthorizationPolicy(resource_owner_key=OWNER_A)


def _authorization(configs: list[dict]) -> str:
    assert len(configs) == 1
    assert configs[0]["transport"] == "streamable_http"
    return configs[0]["config"]["headers"]["Authorization"]


def _assert_rejected(configs: list[dict]) -> None:
    assert len(configs) == 1
    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["reason"] == "config_load_failed"
    assert "actor-b-token" not in str(configs)
    # The rejection must not echo the selected owner key either.
    assert OWNER_B not in str(configs)


@pytest.mark.asyncio
async def test_task_auth_context_cannot_select_another_actors_grant(seeded) -> None:
    configs = await _config(
        seeded,
        policy=_policy(),
        mcp_auth_context={str(seeded.server.id): {"resource_owner_key": OWNER_B}},
    ).get_mcp_server_configs()

    _assert_rejected(configs)


@pytest.mark.asyncio
async def test_runtime_auth_selector_cannot_select_another_actors_grant(
    seeded,
) -> None:
    configs = await _config(
        seeded,
        policy=_policy(),
        auth_selector={"resource_owner_key": OWNER_B},
    ).get_mcp_server_configs()

    _assert_rejected(configs)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [f" {OWNER_A} ", 42, ["x"]])
async def test_selector_owner_must_match_exactly(seeded, value) -> None:
    configs = await _config(
        seeded,
        policy=_policy(),
        mcp_auth_context={str(seeded.server.id): {"resource_owner_key": value}},
    ).get_mcp_server_configs()

    assert configs[0]["transport"] == "unavailable"
    assert configs[0]["config"]["reason"] == "config_load_failed"


@pytest.mark.asyncio
async def test_selector_may_name_the_policy_owner(seeded) -> None:
    configs = await _config(
        seeded,
        policy=_policy(),
        auth_selector={"resource_owner_key": OWNER_A},
    ).get_mcp_server_configs()

    assert _authorization(configs) == "Bearer actor-a-token"


@pytest.mark.asyncio
async def test_selector_may_name_the_executing_users_default_owner(seeded) -> None:
    configs = await _config(
        seeded,
        policy=_policy(),
        mcp_auth_context={
            str(seeded.server.id): {
                "resource_owner_key": f"xagent:user:{seeded.user.id}"
            }
        },
    ).get_mcp_server_configs()

    assert _authorization(configs) == "Bearer workspace-token"


@pytest.mark.asyncio
async def test_no_selector_keeps_the_default_owner_under_a_policy(seeded) -> None:
    configs = await _config(seeded, policy=_policy()).get_mcp_server_configs()

    assert _authorization(configs) == "Bearer workspace-token"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", None])
async def test_an_empty_selector_owner_keeps_the_default_owner(seeded, value) -> None:
    # The native lookup treats a falsy owner as the default owner, so the
    # guard must let it through rather than reject it.
    configs = await _config(
        seeded,
        policy=_policy(),
        mcp_auth_context={str(seeded.server.id): {"resource_owner_key": value}},
    ).get_mcp_server_configs()

    assert _authorization(configs) == "Bearer workspace-token"


@pytest.mark.asyncio
async def test_without_a_policy_task_selectors_are_unchanged(seeded) -> None:
    configs = await _config(
        seeded,
        policy=None,
        mcp_auth_context={str(seeded.server.id): {"resource_owner_key": OWNER_B}},
    ).get_mcp_server_configs()

    assert _authorization(configs) == "Bearer actor-b-token"


@pytest.mark.asyncio
async def test_a_resolver_token_is_unaffected_by_a_foreign_selector(seeded) -> None:
    async def resolver(request) -> ResolvedToken | None:
        return ResolvedToken(access_token="resolver-token")

    set_oauth_token_resolver_hook(resolver)
    configs = await _config(
        seeded,
        policy=_policy(),
        mcp_auth_context={str(seeded.server.id): {"resource_owner_key": OWNER_B}},
    ).get_mcp_server_configs()

    assert _authorization(configs) == "Bearer resolver-token"
