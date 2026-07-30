from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.core.tools.adapters.vibe.connector_runtime import (
    ConnectorRef,
    ConnectorRuntimeError,
)
from xagent.web.models import Agent, Base, MCPServer, Task, User, UserMCPServer
from xagent.web.models.agent import AgentStatus
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.task import TaskStatus
from xagent.web.services import connector_runtime as connector_runtime_service
from xagent.web.services import connector_team_scope
from xagent.web.services.connector_runtime import (
    ConnectorRuntimeValues,
    bind_connector_runtime_selection_snapshot,
    bind_create_connector_runtime_plan,
    load_connector_runtime_view,
    prepare_append_connector_runtime,
    prepare_connector_runtime_selection_snapshot,
    prepare_create_connector_runtime,
    set_connector_runtime_resolver_for_testing,
)


@pytest.fixture(autouse=True)
def reset_connector_runtime_resolver() -> Iterator[None]:
    set_connector_runtime_resolver_for_testing(None)
    yield
    set_connector_runtime_resolver_for_testing(None)


@pytest.fixture()
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _create_user(db: Session, username: str) -> User:
    user = User(username=username, password_hash="hash")
    db.add(user)
    db.flush()
    return user


def _create_runtime_mcp(
    db: Session,
    user: User,
    name: str,
    *,
    secret_required: bool = False,
) -> MCPServer:
    runtime_input_schema = {
        "context": {"account_id": {"type": "string", "required": False}}
    }
    runtime_bindings = [
        {
            "source": {"input_type": "context", "key": "account_id"},
            "target": {"target_type": "mcp_meta", "key": "account_id"},
        }
    ]
    if secret_required:
        runtime_input_schema["secrets"] = {
            "authorization": {"type": "string", "required": True}
        }
        runtime_bindings.append(
            {
                "source": {
                    "input_type": "secrets",
                    "key": "authorization",
                },
                "target": {
                    "target_type": "transport_headers",
                    "key": "Authorization",
                },
            }
        )
    server = MCPServer(
        name=name,
        description=f"{name} description",
        managed="external",
        transport="streamable_http",
        url="https://example.com/mcp",
        runtime_input_schema=runtime_input_schema,
        runtime_bindings=runtime_bindings,
    )
    db.add(server)
    db.flush()
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.flush()
    return server


def _create_runtime_custom_api(db: Session, user: User, name: str) -> CustomApi:
    api = CustomApi(
        name=name,
        description=f"{name} description",
        url="https://example.com/api",
        method="GET",
        runtime_input_schema={
            "context": {"account_id": {"type": "string", "required": False}}
        },
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
    )
    db.add(api)
    db.flush()
    db.add(
        UserCustomApi(
            user_id=user.id,
            custom_api_id=api.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.flush()
    return api


def test_task_model_defaults_connector_runtime_selected_refs_to_empty_list(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    task = Task(user_id=user.id, title="default refs", status=TaskStatus.PENDING)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    assert task.connector_runtime_selected_refs == []

    legacy_task = Task(
        user_id=user.id,
        title="legacy null refs",
        status=TaskStatus.PENDING,
        connector_runtime_selected_refs=None,
    )
    db_session.add(legacy_task)
    db_session.commit()
    db_session.refresh(legacy_task)

    assert legacy_task.connector_runtime_selected_refs is None


def test_selection_snapshot_uses_runtime_owner_connector_visibility(
    db_session: Session,
) -> None:
    agent_owner = _create_user(db_session, "agent-owner")
    task_owner = _create_user(db_session, "task-owner")
    agent = Agent(
        user_id=agent_owner.id,
        name="Published Agent",
        description="shared",
        instructions="Use tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()

    owner_server = _create_runtime_mcp(db_session, agent_owner, "owner-server")
    task_owner_server = _create_runtime_mcp(db_session, task_owner, "task-owner-server")

    selected_refs = prepare_connector_runtime_selection_snapshot(
        db=db_session,
        agent=agent,
        connector_user_id=int(task_owner.id),
    )
    task = Task(
        user_id=task_owner.id,
        agent_id=agent.id,
        title="published agent task",
        status=TaskStatus.PENDING,
    )
    bind_connector_runtime_selection_snapshot(task=task, selected_refs=selected_refs)

    assert task.connector_runtime_selected_refs == [
        {"connector_type": "mcp", "connector_id": int(task_owner_server.id)}
    ]
    assert {"connector_type": "mcp", "connector_id": int(owner_server.id)} not in (
        task.connector_runtime_selected_refs or []
    )


def test_selection_snapshot_scopes_custom_api_by_source_server_name(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    agent = Agent(
        user_id=user.id,
        name="Scoped Connector Agent",
        description="shared",
        instructions="Use scoped tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp:Records"],
    )
    db_session.add(agent)
    db_session.flush()

    selected_mcp = _create_runtime_mcp(db_session, user, "Records")
    selected_api = _create_runtime_custom_api(db_session, user, "Records")
    unselected_api = _create_runtime_custom_api(db_session, user, "Billing")

    selected_refs = prepare_connector_runtime_selection_snapshot(
        db=db_session,
        agent=agent,
        connector_user_id=int(user.id),
    )
    task = Task(
        user_id=user.id,
        agent_id=agent.id,
        title="scoped connector task",
        status=TaskStatus.PENDING,
    )
    bind_connector_runtime_selection_snapshot(task=task, selected_refs=selected_refs)

    assert task.connector_runtime_selected_refs == [
        {"connector_type": "custom_api", "connector_id": int(selected_api.id)},
        {"connector_type": "mcp", "connector_id": int(selected_mcp.id)},
    ]
    assert {
        "connector_type": "custom_api",
        "connector_id": int(unselected_api.id),
    } not in (task.connector_runtime_selected_refs or [])


def test_selected_connector_resolver_returns_selected_mcp_without_runtime_declaration(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    agent = Agent(
        user_id=user.id,
        name="OAuth Connector Agent",
        description="shared",
        instructions="Use the selected connector.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    server = _create_runtime_mcp(db_session, user, "oauth-only")
    server.runtime_input_schema = None
    server.runtime_bindings = None
    db_session.flush()

    resolved = connector_runtime_service.resolve_agent_selected_connectors(
        db=db_session,
        agent=agent,
        connector_user_id=int(user.id),
    )

    ref = ConnectorRef("mcp", int(server.id))
    assert list(resolved) == [ref]
    assert resolved[ref] is server


@pytest.mark.parametrize(
    ("tool_categories", "selects_all"),
    [
        (None, True),
        ([], False),
    ],
)
def test_selected_connector_resolver_honors_all_and_none_sentinels(
    db_session: Session,
    tool_categories: list[str] | None,
    selects_all: bool,
) -> None:
    user = _create_user(db_session, "owner")
    agent = Agent(
        user_id=user.id,
        name="Sentinel Connector Agent",
        description="shared",
        instructions="Use selected connectors.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=tool_categories,
    )
    db_session.add(agent)
    db_session.flush()
    mcp_server = _create_runtime_mcp(db_session, user, "Records")
    custom_api = _create_runtime_custom_api(db_session, user, "Records")

    db_session.expire(agent, ["tool_categories"])
    assert agent.tool_categories == tool_categories

    resolved = connector_runtime_service.resolve_agent_selected_connectors(
        db=db_session,
        agent=agent,
        connector_user_id=int(user.id),
    )

    expected = (
        [
            ConnectorRef("custom_api", int(custom_api.id)),
            ConnectorRef("mcp", int(mcp_server.id)),
        ]
        if selects_all
        else []
    )
    assert list(resolved) == expected


def test_selected_connector_resolver_includes_owner_and_team_visible_connectors_only(
    db_session: Session,
) -> None:
    viewer = _create_user(db_session, "viewer")
    other_user = _create_user(db_session, "other-user")
    hidden_user = _create_user(db_session, "hidden-user")
    agent = Agent(
        user_id=viewer.id,
        name="Scoped Connector Agent",
        description="shared",
        instructions="Use selected tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp: Team Records"],
    )
    db_session.add(agent)
    db_session.flush()
    team_server = _create_runtime_mcp(db_session, other_user, "Team Records")
    owner_server = _create_runtime_mcp(db_session, viewer, "team-records")
    hidden_server = _create_runtime_mcp(db_session, hidden_user, "TEAM_records")
    unselected_server = _create_runtime_mcp(db_session, viewer, "Billing")

    connector_team_scope.set_connector_team_hooks(
        visibility=lambda db, user_id: {
            "mcp": {int(team_server.id)} if user_id == int(viewer.id) else set(),
            "custom_api": set(),
        }
    )
    try:
        resolved = connector_runtime_service.resolve_agent_selected_connectors(
            db=db_session,
            agent=agent,
            connector_user_id=int(viewer.id),
        )
    finally:
        connector_team_scope.set_connector_team_hooks()

    owner_ref = ConnectorRef("mcp", int(owner_server.id))
    team_ref = ConnectorRef("mcp", int(team_server.id))
    assert list(resolved) == [team_ref, owner_ref]
    assert ConnectorRef("mcp", int(hidden_server.id)) not in resolved
    assert ConnectorRef("mcp", int(unselected_server.id)) not in resolved


def test_selected_connector_resolver_skips_name_normalization_without_server_scope(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session, "owner")
    agent = Agent(
        user_id=user.id,
        name="Unscoped Connector Agent",
        description="shared",
        instructions="Use MCP tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    selected_mcp = _create_runtime_mcp(db_session, user, "Records")
    _create_runtime_custom_api(db_session, user, "Billing")

    normalization_calls: list[str] = []
    normalize_name = connector_runtime_service.normalize_mcp_server_name

    def record_normalization(name: str) -> str:
        normalization_calls.append(name)
        return normalize_name(name)

    monkeypatch.setattr(
        connector_runtime_service,
        "normalize_mcp_server_name",
        record_normalization,
    )

    resolved = connector_runtime_service.resolve_agent_selected_connectors(
        db=db_session,
        agent=agent,
        connector_user_id=int(user.id),
    )

    assert list(resolved) == [ConnectorRef("mcp", int(selected_mcp.id))]
    assert normalization_calls == []


def test_selected_connector_resolver_normalizes_names_and_orders_refs(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    agent = Agent(
        user_id=user.id,
        name="Normalized Connector Agent",
        description="shared",
        instructions="Use selected tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp: Google Drive"],
    )
    db_session.add(agent)
    db_session.flush()
    mcp_server = _create_runtime_mcp(db_session, user, "Google-Drive")
    custom_api = _create_runtime_custom_api(db_session, user, "google drive")

    resolved = connector_runtime_service.resolve_agent_selected_connectors(
        db=db_session,
        agent=agent,
        connector_user_id=int(user.id),
    )

    assert list(resolved) == [
        ConnectorRef("custom_api", int(custom_api.id)),
        ConnectorRef("mcp", int(mcp_server.id)),
    ]


def test_create_plan_preserves_canonical_cross_type_connector_order(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    agent = Agent(
        user_id=user.id,
        name="Ordered Connector Agent",
        description="shared",
        instructions="Use selected tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp: Records"],
    )
    db_session.add(agent)
    db_session.flush()
    selected_mcp = _create_runtime_mcp(db_session, user, "Records")
    selected_api = _create_runtime_custom_api(db_session, user, "records")
    unselected_mcp = _create_runtime_mcp(db_session, user, "Billing")

    plan = prepare_create_connector_runtime(
        db=db_session,
        agent=agent,
        connector_user_id=int(user.id),
        task_source="sdk",
        payload_items=None,
    )

    assert plan.selected_refs == (
        ConnectorRef("custom_api", int(selected_api.id)),
        ConnectorRef("mcp", int(selected_mcp.id)),
    )
    assert ConnectorRef("mcp", int(unselected_mcp.id)) not in plan.selected_refs


def test_selection_snapshot_uses_public_resolver_and_filters_runtime_declarations(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _create_user(db_session, "owner")
    agent = Agent(
        user_id=user.id,
        name="Snapshot Connector Agent",
        description="shared",
        instructions="Use selected tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    declared = _create_runtime_mcp(db_session, user, "declared")
    undeclared = _create_runtime_mcp(db_session, user, "undeclared")
    undeclared.runtime_input_schema = None
    undeclared.runtime_bindings = None
    db_session.flush()
    calls = []

    def resolve_selected(**kwargs):
        calls.append(kwargs)
        return {
            ConnectorRef("mcp", int(declared.id)): declared,
            ConnectorRef("mcp", int(undeclared.id)): undeclared,
        }

    monkeypatch.setattr(
        connector_runtime_service,
        "resolve_agent_selected_connectors",
        resolve_selected,
    )

    selected_refs = prepare_connector_runtime_selection_snapshot(
        db=db_session,
        agent=agent,
        connector_user_id=int(user.id),
    )

    assert selected_refs == (ConnectorRef("mcp", int(declared.id)),)
    assert calls == [
        {
            "db": db_session,
            "agent": agent,
            "connector_user_id": int(user.id),
        }
    ]


def test_create_plan_rejects_undeclared_selected_connector_payload(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    agent = Agent(
        user_id=user.id,
        name="Runtime Declaration Agent",
        description="shared",
        instructions="Use selected tools.",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    server = _create_runtime_mcp(db_session, user, "oauth-only")
    server.runtime_input_schema = None
    server.runtime_bindings = None
    db_session.flush()

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        prepare_create_connector_runtime(
            db=db_session,
            agent=agent,
            task_source="sdk",
            connector_user_id=int(user.id),
            payload_items=[
                {
                    "connector_ref": {
                        "connector_type": "mcp",
                        "connector_id": int(server.id),
                    }
                }
            ],
        )

    assert exc_info.value.code == "invalid_runtime_context"
    assert exc_info.value.details["reason"] == "connector_not_selected"


def test_scoped_resolver_applies_consistently_to_create_and_binding(
    db_session: Session,
) -> None:
    agent_owner = _create_user(db_session, "agent-owner")
    task_owner = _create_user(db_session, "task-owner")
    agent = Agent(
        user_id=agent_owner.id,
        name="External Agent",
        instructions="Use tools.",
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    server = _create_runtime_mcp(
        db_session, task_owner, "external-runtime", secret_required=True
    )
    requests = []

    def resolver(request):
        requests.append(request)
        return ConnectorRuntimeValues(
            context=request.values.context,
            secrets={"authorization": "Bearer external"},
            auth_selector=request.values.auth_selector,
        )

    set_connector_runtime_resolver_for_testing(resolver, task_sources={"external"})
    try:
        plan = prepare_create_connector_runtime(
            db=db_session,
            agent=agent,
            task_source="external",
            connector_user_id=int(task_owner.id),
            payload_items=None,
        )
        task = Task(
            user_id=task_owner.id,
            agent_id=agent.id,
            title="external task",
            source="external",
            status=TaskStatus.PENDING,
        )
        bind_create_connector_runtime_plan(task=task, plan=plan)
        db_session.add(task)
        db_session.flush()

        view = load_connector_runtime_view(
            db=db_session,
            task_id=int(task.id),
            turn_id="external-turn",
            user_id=int(task_owner.id),
        )
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert task.connector_runtime_selected_refs == [
        {"connector_type": "mcp", "connector_id": int(server.id)}
    ]
    assert view[f"mcp:{server.id}"]["secrets"] == {"authorization": "Bearer external"}
    assert len(requests) == 1
    assert requests[0].task_source == "external"
    assert requests[0].user_id == int(task_owner.id)


@pytest.mark.parametrize("task_source", ["sdk", "unknown"])
def test_scoped_resolver_does_not_bypass_other_create_sources(
    db_session: Session,
    task_source: str,
) -> None:
    owner = _create_user(db_session, "owner")
    agent = Agent(
        user_id=owner.id,
        name="SDK Agent",
        instructions="Use tools.",
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    _create_runtime_mcp(db_session, owner, "sdk-runtime", secret_required=True)

    set_connector_runtime_resolver_for_testing(
        lambda request: request.values,
        task_sources={"external"},
    )
    try:
        with pytest.raises(ConnectorRuntimeError) as exc_info:
            prepare_create_connector_runtime(
                db=db_session,
                agent=agent,
                task_source=task_source,
                connector_user_id=int(owner.id),
                payload_items=None,
            )
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert exc_info.value.code == "runtime_secret_unavailable"
    assert exc_info.value.details["reason"] == "not_provided"


@pytest.mark.parametrize(
    ("task_user", "task_source"),
    [("other", "external"), ("owner", "sdk")],
)
def test_create_plan_rejects_task_identity_mismatch(
    db_session: Session,
    task_user: str,
    task_source: str,
) -> None:
    owner = _create_user(db_session, "owner")
    other = _create_user(db_session, "other")
    agent = Agent(
        user_id=owner.id,
        name="Identity Agent",
        instructions="Use tools.",
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    _create_runtime_mcp(db_session, owner, "identity-runtime")
    plan = prepare_create_connector_runtime(
        db=db_session,
        agent=agent,
        task_source="external",
        connector_user_id=int(owner.id),
        payload_items=None,
    )
    task = Task(
        user_id=owner.id if task_user == "owner" else other.id,
        agent_id=agent.id,
        title="mismatched task",
        source=task_source,
        status=TaskStatus.PENDING,
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        bind_create_connector_runtime_plan(task=task, plan=plan)

    assert exc_info.value.code == "connector_runtime_unavailable"
    assert exc_info.value.details["reason"] == "runtime_task_identity_mismatch"
    assert task.connector_runtime_selected_refs is None


def test_append_uses_persisted_task_owner_connector_visibility(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_owner = _create_user(db_session, "agent-owner")
    task_owner = _create_user(db_session, "task-owner")
    agent = Agent(
        user_id=agent_owner.id,
        name="Historical Agent",
        instructions="Use tools.",
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    server = _create_runtime_mcp(db_session, task_owner, "task-owner-runtime")
    task = Task(
        user_id=task_owner.id,
        agent_id=agent.id,
        title="historical task",
        source="sdk",
        status=TaskStatus.COMPLETED,
        connector_runtime_selected_refs=[
            {"connector_type": "mcp", "connector_id": int(server.id)}
        ],
    )
    db_session.add(task)
    db_session.flush()

    plan = prepare_append_connector_runtime(
        db=db_session,
        agent=agent,
        task=task,
        connector_user_id=int(task_owner.id),
        payload_items=[
            {
                "connector_ref": {
                    "connector_type": "mcp",
                    "connector_id": int(server.id),
                }
            }
        ],
    )

    assert plan.ephemeral_by_ref == {}

    def reject_visibility_query(*args, **kwargs):
        raise AssertionError("owner mismatch must fail before connector visibility")

    monkeypatch.setattr(
        connector_runtime_service,
        "_load_visible_runtime_connectors",
        reject_visibility_query,
    )
    with pytest.raises(ConnectorRuntimeError) as exc_info:
        prepare_append_connector_runtime(
            db=db_session,
            agent=agent,
            task=task,
            connector_user_id=int(agent_owner.id),
            payload_items=None,
        )
    assert exc_info.value.code == "connector_runtime_unavailable"
    assert exc_info.value.details["reason"] == "runtime_owner_mismatch"


def test_runtime_binding_rejects_owner_mismatch_before_resolver(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _create_user(db_session, "owner")
    other = _create_user(db_session, "other")
    agent = Agent(
        user_id=owner.id,
        name="Binding Agent",
        instructions="Use tools.",
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    server = _create_runtime_mcp(db_session, owner, "binding-runtime")
    task = Task(
        user_id=owner.id,
        agent_id=agent.id,
        title="binding task",
        source="external",
        status=TaskStatus.PENDING,
        connector_runtime_selected_refs=[
            {"connector_type": "mcp", "connector_id": int(server.id)}
        ],
    )
    db_session.add(task)
    db_session.flush()
    view = load_connector_runtime_view(
        db=db_session,
        task_id=int(task.id),
        turn_id="binding-turn",
        user_id=None,
    )
    assert f"mcp:{server.id}" in view
    resolver_calls = 0

    def resolver(request):
        nonlocal resolver_calls
        resolver_calls += 1
        return request.values

    def reject_visibility_query(*args, **kwargs):
        raise AssertionError("owner mismatch must fail before connector visibility")

    monkeypatch.setattr(
        connector_runtime_service,
        "_load_visible_runtime_connectors",
        reject_visibility_query,
    )
    set_connector_runtime_resolver_for_testing(resolver, task_sources={"external"})
    try:
        with pytest.raises(ConnectorRuntimeError) as exc_info:
            load_connector_runtime_view(
                db=db_session,
                task_id=int(task.id),
                turn_id="binding-turn",
                user_id=int(other.id),
            )
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert exc_info.value.code == "connector_runtime_unavailable"
    assert exc_info.value.details["reason"] == "runtime_owner_mismatch"
    assert resolver_calls == 0


def test_unscoped_resolver_preserves_legacy_none_source_behavior(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    agent = Agent(
        user_id=owner.id,
        name="Legacy Agent",
        instructions="Use tools.",
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    server = _create_runtime_mcp(
        db_session, owner, "legacy-runtime", secret_required=True
    )
    task = Task(
        user_id=owner.id,
        agent_id=agent.id,
        title="legacy task",
        source=None,
        status=TaskStatus.PENDING,
        connector_runtime_selected_refs=[
            {"connector_type": "mcp", "connector_id": int(server.id)}
        ],
    )
    db_session.add(task)
    db_session.flush()

    def resolver(request):
        return ConnectorRuntimeValues(
            context=request.values.context,
            secrets={"authorization": "Bearer legacy"},
            auth_selector=request.values.auth_selector,
        )

    set_connector_runtime_resolver_for_testing(resolver)
    try:
        view = load_connector_runtime_view(
            db=db_session,
            task_id=int(task.id),
            turn_id="legacy-turn",
            user_id=int(owner.id),
        )
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert view[f"mcp:{server.id}"]["secrets"] == {"authorization": "Bearer legacy"}

    set_connector_runtime_resolver_for_testing(resolver, task_sources={"external"})
    try:
        with pytest.raises(ConnectorRuntimeError) as exc_info:
            load_connector_runtime_view(
                db=db_session,
                task_id=int(task.id),
                turn_id="legacy-turn",
                user_id=int(owner.id),
            )
    finally:
        set_connector_runtime_resolver_for_testing(None)

    assert exc_info.value.code == "runtime_secret_unavailable"


def test_resolver_registration_rejects_string_source_scope() -> None:
    with pytest.raises(TypeError):
        set_connector_runtime_resolver_for_testing(
            lambda request: request.values,
            task_sources="external",  # type: ignore[arg-type]
        )


def test_resolver_registration_rejects_empty_source_scope() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        set_connector_runtime_resolver_for_testing(
            lambda request: request.values,
            task_sources=set(),
        )


@pytest.mark.parametrize("task_source", [" external", "external ", "\texternal\n"])
def test_resolver_registration_rejects_source_scope_with_surrounding_whitespace(
    task_source: str,
) -> None:
    try:
        with pytest.raises(ValueError, match="surrounding whitespace"):
            set_connector_runtime_resolver_for_testing(
                lambda request: request.values,
                task_sources={task_source},
            )
    finally:
        set_connector_runtime_resolver_for_testing(None)


def test_resolver_registration_copies_mutable_source_scope(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "mutable-scope-owner")
    agent = Agent(
        user_id=owner.id,
        name="Mutable Scope Agent",
        instructions="Use tools.",
        tool_categories=["mcp"],
    )
    db_session.add(agent)
    db_session.flush()
    _create_runtime_mcp(
        db_session,
        owner,
        "mutable-scope-runtime",
        secret_required=True,
    )
    task_sources = {"external"}
    set_connector_runtime_resolver_for_testing(
        lambda request: request.values,
        task_sources=task_sources,
    )
    task_sources.add("sdk")

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        prepare_create_connector_runtime(
            db=db_session,
            agent=agent,
            task_source="sdk",
            connector_user_id=int(owner.id),
            payload_items=None,
        )

    assert exc_info.value.code == "runtime_secret_unavailable"
