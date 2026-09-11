from __future__ import annotations

import base64
import inspect
import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from xagent.core.tools.adapters.vibe.config import (
    ACTOR_STDIO_SESSION_RUNTIME_UNAVAILABLE_REASON,
    ACTOR_STDIO_SHADOWED_REASON,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.web.builtin_mcp_registry import get_builtin_execution_fields
from xagent.web.models.database import Base
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.services.actor_mcp_connections import (
    ActorMCPConnectionCredentialCorruptionError,
    ActorMCPConnectionMetadata,
    ActorMCPConnectionSnapshot,
    ActorMCPConnectionValidationError,
    create_actor_mcp_connection,
)
from xagent.web.services.actor_mcp_runtime import (
    ActorMCPConnectionServiceAdapter,
    ActorMCPRuntimeDefinitionError,
    ActorMCPStdioConnectionIdentity,
    ActorMCPStdioSessionIdentity,
    _cached_builtin_stdio_execution,
    _canonical_oauth_scopes,
    production_actor_mcp_stdio_connection_adapter,
    resolve_actor_mcp_stdio_configs,
)
from xagent.web.services.mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPActorExecutionIdentity,
)
from xagent.web.tools.config import WebToolConfig

USER_ID = 41
OWNER = "toby:slack:T1:U1"
APP_ID = "posthog"


@pytest.fixture()
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def _seed_app(
    db: Session, *, app_id: str = APP_ID, visible: bool = True
) -> PublicMCPApp:
    execution = get_builtin_execution_fields(app_id)
    assert execution is not None
    app = PublicMCPApp(
        app_id=app_id,
        name=execution["name"],
        transport=execution["transport"],
        provider_name=execution["provider_name"],
        oauth_scopes=execution["oauth_scopes"],
        launch_config=execution["launch_config"],
        is_visible_in_connector=visible,
    )
    db.add(app)
    db.flush()
    return app


class _FakeAdapter:
    def __init__(
        self,
        identity: ActorMCPStdioConnectionIdentity,
        credentials: Mapping[str, str] | None,
    ) -> None:
        self.identity = identity
        self.credentials = credentials
        self.list_calls: list[dict[str, object]] = []
        self.secret_calls: list[dict[str, object]] = []

    def list_connection_identities(
        self,
        db: Session,
        *,
        user_id: int,
        resource_owner_key: str,
    ) -> Sequence[ActorMCPStdioConnectionIdentity]:
        self.list_calls.append(
            {"user_id": user_id, "resource_owner_key": resource_owner_key}
        )
        return (self.identity,)

    def get_connection_credentials(
        self,
        db: Session,
        *,
        user_id: int,
        resource_owner_key: str,
        app_id: str,
        catalog_app_generation: uuid.UUID,
        expected_lifecycle_generation: uuid.UUID,
    ) -> Mapping[str, str] | None:
        self.secret_calls.append(
            {
                "user_id": user_id,
                "resource_owner_key": resource_owner_key,
                "app_id": app_id,
                "catalog_app_generation": catalog_app_generation,
                "expected_lifecycle_generation": expected_lifecycle_generation,
            }
        )
        return self.credentials


class _StableIdentityOnlyServer:
    def __init__(self, server_id: int, name: str) -> None:
        self.id = server_id
        self.name = name
        self.forbidden_accesses: list[str] = []

    def __getattr__(self, name: str) -> object:
        self.forbidden_accesses.append(name)
        raise AssertionError(f"actor resolver read forbidden MCPServer field: {name}")


def _identity(app: PublicMCPApp, **overrides: object):
    values = {
        "user_id": USER_ID,
        "resource_owner_key": OWNER,
        "app_id": app.app_id,
        "catalog_app_generation": app.generation,
        "lifecycle_generation": uuid.uuid4(),
    }
    values.update(overrides)
    return ActorMCPStdioConnectionIdentity(**values)


def _credentials() -> dict[str, str]:
    return {
        "POSTHOG_API_KEY": "actor-api-key",
        "POSTHOG_HOST": "https://actor.example.test",
    }


def _credentials_for(app_id: str, *, suffix: str = "value") -> dict[str, str]:
    execution = get_builtin_execution_fields(app_id)
    assert execution is not None
    return {
        field_name: f"{field_name.lower()}-{suffix}"
        for field_name in execution["launch_config"].get("required_env", [])
    }


def _policy(*, allow: bool = True) -> MCPActorAuthorizationPolicy:
    return MCPActorAuthorizationPolicy(
        resource_owner_key=OWNER,
        allow_builtin_stdio=allow,
    )


def _execution_identity(
    *,
    task_id: int = 91,
    run_id: str = "run-1",
    turn_id: str = "turn-1",
    lease_attempt_id: str = "attempt-1",
) -> MCPActorExecutionIdentity:
    return MCPActorExecutionIdentity(
        task_id=task_id,
        run_id=run_id,
        turn_id=turn_id,
        lease_attempt_id=lease_attempt_id,
    )


def test_synthetic_config_uses_exact_lifecycle_fenced_adapter_inputs(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    identity = _identity(app)
    adapter = _FakeAdapter(identity, _credentials())

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert adapter.list_calls == [{"user_id": USER_ID, "resource_owner_key": OWNER}]
    assert adapter.secret_calls == [
        {
            "user_id": USER_ID,
            "resource_owner_key": OWNER,
            "app_id": APP_ID,
            "catalog_app_generation": app.generation,
            "expected_lifecycle_generation": identity.lifecycle_generation,
        }
    ]
    assert result.blocked_server_ids == frozenset()
    assert len(result.configs) == 1
    config = result.configs[0]
    execution = get_builtin_execution_fields(APP_ID)
    assert execution is not None
    assert config["config"]["command"] == execution["launch_config"]["command"]
    assert config["config"]["args"] == execution["launch_config"]["args"]
    assert config["config"]["env"] == {
        **_credentials(),
        "XAGENT_MCP_CALLER_ID": str(USER_ID),
    }
    assert "id" not in config


def test_catalog_lookup_does_not_load_full_rows_for_collision_scan(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    statements: list[str] = []

    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        if "public_mcp_apps" in statement:
            statements.append(statement)

    event.listen(db.bind, "before_cursor_execute", capture_statement)
    try:
        resolve_actor_mcp_stdio_configs(
            db,
            user_id=USER_ID,
            policy=_policy(),
            adapter=_FakeAdapter(_identity(app), _credentials()),
            visible_servers=(),
        )
    finally:
        event.remove(db.bind, "before_cursor_execute", capture_statement)

    assert any(
        "WHERE public_mcp_apps.app_id =" in statement for statement in statements
    )
    assert any(
        "SELECT public_mcp_apps.app_id" in statement
        and "public_mcp_apps.launch_config" not in statement
        for statement in statements
    )
    assert not any(
        "public_mcp_apps.launch_config" in statement and "WHERE" not in statement
        for statement in statements
    )


def test_production_adapter_is_a_stateless_exact_service_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_generation = uuid.uuid4()
    lifecycle_generation = uuid.uuid4()
    metadata = ActorMCPConnectionMetadata(
        id=7,
        lifecycle_generation=lifecycle_generation,
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id=APP_ID,
        catalog_app_generation=catalog_generation,
        configured_field_names=frozenset(_credentials()),
    )
    snapshot = ActorMCPConnectionSnapshot(
        id=7,
        lifecycle_generation=lifecycle_generation,
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id=APP_ID,
        catalog_app_generation=catalog_generation,
        credentials=_credentials(),
    )
    list_calls: list[dict[str, object]] = []
    secret_calls: list[dict[str, object]] = []

    def fake_list(_db: Session, **kwargs: object) -> list[object]:
        list_calls.append(kwargs)
        return [metadata]

    def fake_get(_db: Session, **kwargs: object) -> object:
        secret_calls.append(kwargs)
        return snapshot

    monkeypatch.setattr(
        "xagent.web.services.actor_mcp_runtime.list_actor_mcp_connection_metadata",
        fake_list,
    )
    monkeypatch.setattr(
        "xagent.web.services.actor_mcp_runtime.get_actor_mcp_connection_credentials_internal",
        fake_get,
    )
    adapter = ActorMCPConnectionServiceAdapter()

    identities = adapter.list_connection_identities(
        object(),  # type: ignore[arg-type]
        user_id=USER_ID,
        resource_owner_key=OWNER,
    )
    credentials = adapter.get_connection_credentials(
        object(),  # type: ignore[arg-type]
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id=APP_ID,
        catalog_app_generation=catalog_generation,
        expected_lifecycle_generation=lifecycle_generation,
    )

    assert identities == (
        ActorMCPStdioConnectionIdentity(
            user_id=USER_ID,
            resource_owner_key=OWNER,
            app_id=APP_ID,
            catalog_app_generation=catalog_generation,
            lifecycle_generation=lifecycle_generation,
        ),
    )
    assert credentials == _credentials()
    assert list_calls == [{"user_id": USER_ID, "resource_owner_key": OWNER}]
    assert secret_calls == [
        {
            "user_id": USER_ID,
            "resource_owner_key": OWNER,
            "app_id": APP_ID,
            "catalog_app_generation": catalog_generation,
            "expected_lifecycle_generation": lifecycle_generation,
        }
    ]


def test_production_adapter_resolves_credentials_from_storage_service(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    _seed_app(db)
    create_actor_mcp_connection(
        db,
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id=APP_ID,
        credentials=_credentials(),
    )

    def forbidden_transaction_boundary() -> None:
        raise AssertionError("runtime adapter must not own commit or rollback")

    monkeypatch.setattr(db, "commit", forbidden_transaction_boundary)
    monkeypatch.setattr(db, "rollback", forbidden_transaction_boundary)

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=production_actor_mcp_stdio_connection_adapter(),
        visible_servers=(),
    )

    assert len(result.configs) == 1
    assert result.configs[0]["config"]["env"] == {
        **_credentials(),
        "XAGENT_MCP_CALLER_ID": str(USER_ID),
    }


def test_production_adapter_isolates_two_real_owner_rows(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    _seed_app(db)
    other_user_id = USER_ID
    other_owner = "toby:slack:T2:U2"
    first_credentials = _credentials_for(APP_ID, suffix="first")
    second_credentials = _credentials_for(APP_ID, suffix="second")
    create_actor_mcp_connection(
        db,
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id=APP_ID,
        credentials=first_credentials,
    )
    create_actor_mcp_connection(
        db,
        user_id=other_user_id,
        resource_owner_key=other_owner,
        app_id=APP_ID,
        credentials=second_credentials,
    )
    adapter = production_actor_mcp_stdio_connection_adapter()
    assert [
        identity.resource_owner_key
        for identity in adapter.list_connection_identities(
            db,
            user_id=USER_ID,
            resource_owner_key=OWNER,
        )
    ] == [OWNER]
    assert [
        identity.resource_owner_key
        for identity in adapter.list_connection_identities(
            db,
            user_id=other_user_id,
            resource_owner_key=other_owner,
        )
    ] == [other_owner]

    first = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )
    second = resolve_actor_mcp_stdio_configs(
        db,
        user_id=other_user_id,
        policy=MCPActorAuthorizationPolicy(
            resource_owner_key=other_owner,
            allow_builtin_stdio=True,
        ),
        adapter=adapter,
        visible_servers=(),
    )

    assert first.configs[0]["config"]["env"] == {
        **first_credentials,
        "XAGENT_MCP_CALLER_ID": str(USER_ID),
    }
    assert second.configs[0]["config"]["env"] == {
        **second_credentials,
        "XAGENT_MCP_CALLER_ID": str(other_user_id),
    }


@pytest.mark.asyncio
async def test_create_default_tools_registers_production_adapter_only_for_actor_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xagent.web.services.agent_service_manager import create_default_tools

    captured: list[dict[str, object]] = []

    class _FakeToolConfig:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

        def set_task_runtime_contribution(self, _contribution: object) -> None:
            pass

    async def create_tools(_config: object) -> list[object]:
        return []

    monkeypatch.setattr("xagent.web.tools.config.WebToolConfig", _FakeToolConfig)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local", lambda: object()
    )
    monkeypatch.setattr(ToolFactory, "create_all_tools", create_tools)

    for policy in (None, _policy()):
        await create_default_tools(
            None,
            user=SimpleNamespace(id=USER_ID, is_admin=False),
            task_id="91",
            mcp_runtime_authorization_policy=policy,
        )

    assert captured[0]["mcp_actor_stdio_connection_adapter"] is None
    assert (
        captured[1]["mcp_actor_stdio_connection_adapter"]
        is production_actor_mcp_stdio_connection_adapter()
    )


def test_web_tool_config_actor_parameters_are_appended_after_voice() -> None:
    parameters = list(inspect.signature(WebToolConfig.__init__).parameters)

    assert parameters[-3:] == [
        "voice",
        "mcp_actor_stdio_connection_adapter",
        "mcp_actor_execution_identity",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        ActorMCPConnectionCredentialCorruptionError("sensitive-value"),
        RuntimeError("sensitive-value"),
    ],
)
def test_credential_failures_remain_blocked_without_logging_values(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    caplog.set_level(logging.INFO)
    app = _seed_app(db)

    class _FailingAdapter(_FakeAdapter):
        def get_connection_credentials(self, *_args: object, **_kwargs: object):
            raise failure

    collision = _StableIdentityOnlyServer(73, "chrome-devtools")
    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=_FailingAdapter(_identity(app), _credentials()),
        visible_servers=(collision,),
    )

    assert result.configs == ()
    assert result.blocked_server_ids == frozenset({73})
    assert type(failure).__name__ in caplog.text
    assert "sensitive-value" not in caplog.text


def test_missing_adapter_remains_blocked_without_custom_fallback(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    collision = _StableIdentityOnlyServer(73, APP_ID)

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=None,
        visible_servers=(collision,),
    )

    assert result.configs == ()
    assert result.blocked_server_ids == frozenset({73})


@pytest.mark.parametrize(
    "failure",
    [
        ActorMCPConnectionValidationError("sensitive-value"),
        RuntimeError("sensitive-value"),
    ],
)
def test_list_failures_remain_blocked_without_logging_values(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    caplog.set_level(logging.INFO)

    class _FailingListAdapter:
        def list_connection_identities(self, *_args: object, **_kwargs: object):
            raise failure

    collision = _StableIdentityOnlyServer(73, APP_ID)
    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=_FailingListAdapter(),  # type: ignore[arg-type]
        visible_servers=(collision,),
    )

    assert result.configs == ()
    assert result.blocked_server_ids == frozenset({73})
    assert type(failure).__name__ in caplog.text
    assert "sensitive-value" not in caplog.text


def test_reserved_collision_reads_only_stable_server_identity(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app), _credentials())
    collision = _StableIdentityOnlyServer(73, "  PostHog  ")

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(collision,),
    )

    assert result.configs == ()
    assert result.blocked_server_ids == frozenset({73})
    assert result.blocked_server_reasons == ((73, ACTOR_STDIO_SHADOWED_REASON),)
    assert adapter.secret_calls == []
    assert collision.forbidden_accesses == []


@pytest.mark.parametrize(
    "visible_name",
    ["google-maps", "google_maps", "google maps", "  GOOGLE_maps  "],
)
def test_reserved_collision_uses_dispatch_name_normalization(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    visible_name: str,
) -> None:
    monkeypatch.delenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", raising=False)
    collision = _StableIdentityOnlyServer(73, visible_name)

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=None,
        visible_servers=(collision,),
    )

    assert result.blocked_server_ids == frozenset({73})
    assert collision.forbidden_accesses == []


def test_per_identity_collision_uses_dispatch_name_normalization(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web.services import actor_mcp_runtime

    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db, app_id="google-maps")
    adapter = _FakeAdapter(
        _identity(app),
        _credentials_for("google-maps"),
    )
    collision = _StableIdentityOnlyServer(73, "google_maps")
    # Isolate the second collision gate: even if the aggregate blocker is
    # accidentally bypassed, per-identity admission must still fail closed.
    monkeypatch.setattr(
        actor_mcp_runtime,
        "_blocked_visible_server_ids",
        lambda _visible_servers: frozenset(),
    )

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(collision,),
    )

    assert result.configs == ()
    assert adapter.secret_calls == []
    assert collision.forbidden_accesses == []


def test_policy_or_feature_off_never_falls_back_to_reserved_server(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", raising=False)
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app), _credentials())
    collision = _StableIdentityOnlyServer(73, APP_ID)

    feature_off = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(collision,),
    )
    legacy_policy = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(allow=False),
        adapter=adapter,
        visible_servers=(collision,),
    )

    assert feature_off.configs == ()
    assert feature_off.blocked_server_ids == frozenset({73})
    assert legacy_policy.blocked_server_ids == frozenset()
    assert adapter.list_calls == []


@pytest.mark.asyncio
async def test_reserved_collision_surfaces_distinct_unavailable_reason(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", raising=False)
    collision = SimpleNamespace(id=73, name=APP_ID, description=None)
    config = WebToolConfig(
        db=db,
        request=None,
        user_id=USER_ID,
        mcp_runtime_authorization_policy=_policy(),
    )
    monkeypatch.setattr(
        config,
        "_visible_mcp_server_query",
        lambda _team_ids: SimpleNamespace(all=lambda: [collision]),
    )
    monkeypatch.setattr(
        "xagent.web.mcp_apps.classify_actor_builtin_oauth_server",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "xagent.web.mcp_apps.classify_actor_remote_oauth_server",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "xagent.web.services.mcp_runtime.load_user_env_overrides",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        "xagent.web.services.mcp_runtime.load_shared_env_overrides",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        "xagent.web.services.mcp_runtime.load_user_env_sources",
        lambda *_args: {},
    )

    result = await config._load_mcp_server_configs()

    assert result[0]["config"]["reason"] == ACTOR_STDIO_SHADOWED_REASON


@pytest.mark.parametrize(
    "override",
    [
        {"user_id": 42},
        {"resource_owner_key": "toby:slack:T1:U2"},
        {"catalog_app_generation": uuid.uuid4()},
    ],
)
def test_mismatched_connection_identity_fails_closed(
    db: Session, monkeypatch: pytest.MonkeyPatch, override: dict[str, object]
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app, **override), _credentials())

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert result.configs == ()
    assert adapter.secret_calls == []


def test_incomplete_runtime_credentials_fail_closed(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(
        _identity(app),
        {"POSTHOG_API_KEY": "actor-api-key"},
    )

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert result.configs == ()


@pytest.mark.parametrize(
    ("field_name", "drifted_value"),
    [
        ("name", "Drifted PostHog"),
        ("transport", "oauth"),
        ("provider_name", "posthog"),
        ("oauth_scopes", ["unexpected"]),
        (
            "launch_config",
            {
                "command": "attacker",
                "args": ["-m", "xagent.web.tools.mcp.posthog"],
                "required_env": ["POSTHOG_API_KEY", "POSTHOG_HOST"],
            },
        ),
        (
            "launch_config",
            {
                "command": "python",
                "args": ["-m", "attacker.module"],
                "required_env": ["POSTHOG_API_KEY", "POSTHOG_HOST"],
            },
        ),
        (
            "launch_config",
            {
                "command": "python",
                "args": ["-m", "xagent.web.tools.mcp.posthog"],
                "required_env": ["ATTACKER_SECRET"],
            },
        ),
    ],
)
def test_catalog_execution_drift_fails_closed_before_secret_read(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    drifted_value: object,
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app), _credentials())
    setattr(app, field_name, drifted_value)
    db.flush()

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert result.configs == ()
    assert adapter.secret_calls == []


def test_builtin_provenance_version_drift_does_not_disable_actor_stdio(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db, app_id="shopify")
    launch = deepcopy(app.launch_config)
    launch["builtin_provenance"]["version"] = 999
    app.launch_config = launch
    db.flush()
    adapter = _FakeAdapter(_identity(app), _credentials_for("shopify"))

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert len(result.configs) == 1
    assert len(adapter.secret_calls) == 1


@pytest.mark.parametrize(
    ("field_name", "drifted_value"),
    [
        ("command", "attacker"),
        ("args", ["-m", "attacker.module"]),
        ("required_env", ["ATTACKER_SECRET"]),
        ("credential_scope", "shared"),
        ("required_admin_scopes", ["read_products"]),
        ("optional_admin_scopes", ["write_customers"]),
        (
            "builtin_provenance",
            {"registry": "attacker", "app_id": "shopify", "version": 1},
        ),
    ],
)
def test_trusted_launch_fingerprint_drift_fails_closed_before_secret_read(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    drifted_value: object,
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db, app_id="shopify")
    launch = deepcopy(app.launch_config)
    launch[field_name] = drifted_value
    app.launch_config = launch
    db.flush()
    adapter = _FakeAdapter(_identity(app), _credentials_for("shopify"))

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert result.configs == ()
    assert adapter.secret_calls == []


def test_builtin_execution_reads_return_independent_mutable_objects() -> None:
    first = _cached_builtin_stdio_execution(APP_ID)
    second = _cached_builtin_stdio_execution(APP_ID)
    assert first is not None and second is not None

    first["launch_config"]["command"] = "mutated"

    assert second["launch_config"]["command"] != "mutated"


def test_catalog_oauth_scope_order_is_ignored_but_content_drift_fails_closed(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web.services import actor_mcp_runtime

    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    execution = get_builtin_execution_fields(APP_ID)
    assert execution is not None
    execution["oauth_scopes"] = ["scope-b", "scope-a"]
    monkeypatch.setattr(
        actor_mcp_runtime,
        "_cached_builtin_stdio_execution",
        lambda _app_id: execution,
    )
    app.oauth_scopes = ["scope-a", "scope-b"]
    db.flush()
    adapter = _FakeAdapter(_identity(app), _credentials())

    reordered = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert len(reordered.configs) == 1
    assert len(adapter.secret_calls) == 1

    app.oauth_scopes = ["scope-a", "scope-c"]
    db.flush()
    adapter.secret_calls.clear()
    drifted = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert drifted.configs == ()
    assert adapter.secret_calls == []


@pytest.mark.parametrize(
    "invalid_scopes",
    [
        ["scope-a", "scope-a"],
        ["scope-a", 7],
        ["scope-a", ""],
    ],
)
def test_oauth_scope_normalization_rejects_invalid_elements_and_duplicates(
    invalid_scopes: list[object],
) -> None:
    with pytest.raises(ActorMCPRuntimeDefinitionError):
        _canonical_oauth_scopes(invalid_scopes)


@pytest.mark.parametrize(
    "invalid_scopes",
    [
        ["scope-a", "scope-a"],
        ["scope-a", 7],
        ["scope-a", ""],
    ],
)
def test_invalid_or_duplicate_oauth_scopes_fail_closed_before_secret_read(
    db: Session,
    monkeypatch: pytest.MonkeyPatch,
    invalid_scopes: list[object],
) -> None:
    from xagent.web.services import actor_mcp_runtime

    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    execution = get_builtin_execution_fields(APP_ID)
    assert execution is not None
    execution["oauth_scopes"] = invalid_scopes
    monkeypatch.setattr(
        actor_mcp_runtime,
        "_cached_builtin_stdio_execution",
        lambda _app_id: execution,
    )
    app.oauth_scopes = invalid_scopes  # type: ignore[assignment]
    db.flush()
    adapter = _FakeAdapter(_identity(app), _credentials())

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert result.configs == ()
    assert adapter.secret_calls == []


def test_hidden_execution_scoped_app_fails_closed_before_secret_read(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db, app_id="chrome-devtools", visible=False)
    adapter = _FakeAdapter(_identity(app), None)

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
        execution_identity=_execution_identity(),
    )

    assert result.configs == ()
    assert result.session_identities == ()
    assert adapter.secret_calls == []


def test_trusted_runtime_env_collision_fails_closed(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app), _credentials())
    monkeypatch.setattr(
        "xagent.web.services.actor_mcp_runtime.caller_id_env",
        lambda _user_id: {"POSTHOG_API_KEY": "fixed"},
    )

    result = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )

    assert result.configs == ()


def test_reserved_stdio_registry_keys_are_cached(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xagent.web.services import actor_mcp_runtime

    actor_mcp_runtime._builtin_stdio_rows.cache_clear()
    actor_mcp_runtime._reserved_stdio_keys.cache_clear()
    calls = 0
    original = actor_mcp_runtime.get_builtin_public_mcp_app_rows

    def counted_rows() -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(
        actor_mcp_runtime, "get_builtin_public_mcp_app_rows", counted_rows
    )
    collision = _StableIdentityOnlyServer(73, APP_ID)
    for _ in range(2):
        resolve_actor_mcp_stdio_configs(
            db,
            user_id=USER_ID,
            policy=_policy(),
            adapter=None,
            visible_servers=(collision,),
        )

    assert calls == 1
    actor_mcp_runtime._builtin_stdio_rows.cache_clear()
    actor_mcp_runtime._reserved_stdio_keys.cache_clear()


@pytest.mark.asyncio
async def test_web_loader_appends_synthetic_config_without_env_source_queries(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app), _credentials())
    config = WebToolConfig(
        db=db,
        request=None,
        user_id=USER_ID,
        task_id="execution-id-is-not-derived-here",
        mcp_runtime_authorization_policy=_policy(),
        mcp_actor_stdio_connection_adapter=adapter,
    )
    monkeypatch.setattr(
        config,
        "_visible_mcp_server_query",
        lambda _team_ids: SimpleNamespace(all=lambda: []),
    )

    def forbidden(*_args: object, **_kwargs: object):
        raise AssertionError("ordinary MCP credential source was queried")

    from xagent.web.services import mcp_runtime

    monkeypatch.setattr(mcp_runtime, "load_user_env_overrides", forbidden)
    monkeypatch.setattr(mcp_runtime, "load_shared_env_overrides", forbidden)
    monkeypatch.setattr(mcp_runtime, "load_user_env_sources", forbidden)

    result = await config._load_mcp_server_configs()

    assert [item["name"] for item in result] == [APP_ID]


@pytest.mark.asyncio
async def test_nonempty_ordinary_env_layers_never_leak_into_actor_config(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db)
    adapter = _FakeAdapter(_identity(app), _credentials())
    ordinary = _StableIdentityOnlyServer(73, "ordinary-server")
    config = WebToolConfig(
        db=db,
        request=None,
        user_id=USER_ID,
        mcp_runtime_authorization_policy=_policy(),
        mcp_actor_stdio_connection_adapter=adapter,
    )
    monkeypatch.setattr(
        config,
        "_visible_mcp_server_query",
        lambda _team_ids: SimpleNamespace(all=lambda: [ordinary]),
    )
    monkeypatch.setattr(
        "xagent.web.mcp_apps.classify_actor_builtin_oauth_server",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "xagent.web.mcp_apps.classify_actor_remote_oauth_server",
        lambda *_args, **_kwargs: None,
    )

    async def build_ordinary(**_kwargs: object) -> dict[str, object]:
        return {
            "id": 73,
            "name": "ordinary-server",
            "transport": "stdio",
            "config": {"command": "python", "env": {"ORDINARY": "value"}},
        }

    monkeypatch.setattr(config, "_load_mcp_server_config", build_ordinary)
    monkeypatch.setattr(
        "xagent.web.services.mcp_runtime.load_user_env_overrides",
        lambda *_args: {73: {"POSTHOG_API_KEY": "user-leak"}},
    )
    monkeypatch.setattr(
        "xagent.web.services.mcp_runtime.load_shared_env_overrides",
        lambda *_args: {73: {"POSTHOG_HOST": "shared-leak"}},
    )
    monkeypatch.setattr(
        "xagent.web.services.mcp_runtime.load_user_env_sources",
        lambda *_args: {73: "platform"},
    )

    result = await config._load_mcp_server_configs()

    actor_config = next(item for item in result if item["name"] == APP_ID)
    assert actor_config["config"]["env"] == {
        **_credentials(),
        "XAGENT_MCP_CALLER_ID": str(USER_ID),
    }
    assert ordinary.forbidden_accesses == []


def test_execution_scoped_chrome_requires_complete_execution_identity(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db, app_id="chrome-devtools")
    adapter = _FakeAdapter(_identity(app), None)

    missing = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
    )
    present = resolve_actor_mcp_stdio_configs(
        db,
        user_id=USER_ID,
        policy=_policy(),
        adapter=adapter,
        visible_servers=(),
        execution_identity=_execution_identity(),
    )

    assert missing.configs == ()
    assert len(present.configs) == 1
    assert "actor_stdio_session_identity" not in present.configs[0]
    assert json.loads(json.dumps(present.configs[0])) == present.configs[0]
    session_identity = dict(present.session_identities)["chrome-devtools"]
    assert isinstance(session_identity, ActorMCPStdioSessionIdentity)
    assert session_identity.key == (
        91,
        "run-1",
        "turn-1",
        "attempt-1",
        USER_ID,
        OWNER,
        "chrome-devtools",
        app.generation,
        adapter.identity.lifecycle_generation,
    )


@pytest.mark.asyncio
async def test_web_loader_keeps_chrome_identity_in_host_only_side_channel(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAGENT_TOBY_PERSONAL_STDIO_ENABLED", "true")
    app = _seed_app(db, app_id="chrome-devtools")
    adapter = _FakeAdapter(_identity(app), None)
    config = WebToolConfig(
        db=db,
        request=None,
        user_id=USER_ID,
        mcp_runtime_authorization_policy=_policy(),
        mcp_actor_stdio_connection_adapter=adapter,
        mcp_actor_execution_identity=_execution_identity(),
    )
    monkeypatch.setattr(
        config,
        "_visible_mcp_server_query",
        lambda _team_ids: SimpleNamespace(all=lambda: []),
    )

    result = await config._load_mcp_server_configs()

    assert len(result) == 1
    assert "actor_stdio_session_identity" not in result[0]
    identities = config.get_actor_mcp_stdio_session_identities()
    assert identities["chrome-devtools"].connection.app_id == "chrome-devtools"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("task_id", None),
        ("run_id", None),
        ("turn_id", None),
        ("lease_attempt_id", None),
    ],
)
def test_actor_execution_identity_rejects_each_missing_field(
    field_name: str, value: object
) -> None:
    values: dict[str, object] = {
        "task_id": 91,
        "run_id": "run-1",
        "turn_id": "turn-1",
        "lease_attempt_id": "attempt-1",
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        MCPActorExecutionIdentity(**values)  # type: ignore[arg-type]


def test_chrome_session_key_changes_for_retry_and_later_turn() -> None:
    connection = ActorMCPStdioConnectionIdentity(
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id="chrome-devtools",
        catalog_app_generation=uuid.uuid4(),
        lifecycle_generation=uuid.uuid4(),
    )
    initial = ActorMCPStdioSessionIdentity(_execution_identity(), connection)
    retry = ActorMCPStdioSessionIdentity(
        _execution_identity(lease_attempt_id="attempt-2"), connection
    )
    later_turn = ActorMCPStdioSessionIdentity(
        _execution_identity(turn_id="turn-2"), connection
    )

    assert initial.key != retry.key
    assert initial.key != later_turn.key


@pytest.mark.asyncio
async def test_tool_factory_keeps_actor_stdio_session_identity_out_of_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ActorMCPStdioConnectionIdentity(
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id="chrome-devtools",
        catalog_app_generation=uuid.uuid4(),
        lifecycle_generation=uuid.uuid4(),
    )
    session_identity = ActorMCPStdioSessionIdentity(_execution_identity(), connection)
    captured: dict[str, object] = {}
    owner_bytes = OWNER.encode()

    async def forbidden_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("session-scoped config reached the per-call loader")

    async def consume(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        connection_config = kwargs["connection"]
        serialized = json.dumps(connection_config).encode()
        assert owner_bytes not in serialized
        from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_mcp_tool_helper import (
            _serialize_connection,
        )

        argv_payload = base64.b64decode(_serialize_connection(connection_config))
        assert owner_bytes not in argv_payload
        import cloudpickle

        pickled = base64.b64decode(
            base64.b64encode(cloudpickle.dumps(connection_config))
        )
        assert owner_bytes not in pickled
        return []

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        forbidden_load,
    )

    await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "name": "chrome-devtools",
                "transport": "stdio",
                "config": {"command": "npx", "args": [], "env": {}},
            }
        ],
        actor_stdio_session_identities={"chrome-devtools": session_identity},
        actor_stdio_session_consumer=consume,
    )

    assert captured["session_identity"] is session_identity
    assert "actor_stdio_session_identity" not in captured["connection"]  # type: ignore[operator]


@pytest.mark.asyncio
async def test_session_and_per_call_tools_are_both_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = ActorMCPStdioConnectionIdentity(
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id="chrome-devtools",
        catalog_app_generation=uuid.uuid4(),
        lifecycle_generation=uuid.uuid4(),
    )
    session_tool = SimpleNamespace(name="session-tool")
    per_call_tool = SimpleNamespace(name="per-call-tool")

    async def load_per_call(
        connections: Mapping[str, object], **_kwargs: object
    ) -> object:
        assert set(connections) == {"posthog"}
        return SimpleNamespace(tools=[per_call_tool], failures=[])

    async def consume(**kwargs: object) -> list[object]:
        assert kwargs["server_name"] == "chrome-devtools"
        return [session_tool]

    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.mcp_adapter.load_mcp_tools_as_agent_tools",
        load_per_call,
    )

    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "name": "chrome-devtools",
                "transport": "stdio",
                "config": {"command": "npx", "args": [], "env": {}},
            },
            {
                "name": "posthog",
                "transport": "stdio",
                "config": {"command": "npx", "args": [], "env": {}},
            },
        ],
        actor_stdio_session_identities={
            "chrome-devtools": ActorMCPStdioSessionIdentity(
                _execution_identity(), connection
            )
        },
        actor_stdio_session_consumer=consume,
    )

    assert tools == [session_tool, per_call_tool]


@pytest.mark.asyncio
async def test_execution_scoped_config_fails_closed_without_session_consumer() -> None:
    connection = ActorMCPStdioConnectionIdentity(
        user_id=USER_ID,
        resource_owner_key=OWNER,
        app_id="chrome-devtools",
        catalog_app_generation=uuid.uuid4(),
        lifecycle_generation=uuid.uuid4(),
    )
    tools = await ToolFactory._create_mcp_tools_from_configs(
        [
            {
                "name": "chrome-devtools",
                "transport": "stdio",
                "config": {"command": "npx", "args": [], "env": {}},
            }
        ],
        actor_stdio_session_identities={
            "chrome-devtools": ActorMCPStdioSessionIdentity(
                _execution_identity(), connection
            )
        },
    )

    assert len(tools) == 1
    assert tools[0].unavailability_reason == (  # type: ignore[attr-defined]
        ACTOR_STDIO_SESSION_RUNTIME_UNAVAILABLE_REASON
    )
