"""Task-source gate for OAuth grants a team member made for themselves.

``team_member_mcp_grants_allowed_for_task`` decides, from the task row alone,
whether the runtime may select such a grant. These tests feed it the task as
the runtime really receives it: the detached ``_TaskFields`` that
``load_task_setup_snapshot_sync`` reads from a persisted ``tasks`` row, and the
ORM ``Task`` row itself.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import pickle
import textwrap
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.shared.auth_database import auth_db_override
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.utils.encryption import encrypt_value
from xagent.web.api.auth import auth_router
from xagent.web.api.chat import chat_router
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.auth_database import get_auth_db
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.mcp import MCPServer
from xagent.web.models.mcp_oauth import MCPOAuthClient, MCPOAuthGrant
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.services import agent_service_manager as asm_module
from xagent.web.services import connector_team_scope
from xagent.web.services.agent_service_manager import AgentServiceManager
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)
from xagent.web.services.mcp_oauth import (
    resolve_mcp_oauth_runtime_auth,
    select_mcp_oauth_grants,
)
from xagent.web.services.mcp_runtime import (
    build_mcp_runtime_connection,
    team_member_mcp_grants_allowed_for_task,
)
from xagent.web.services.task_setup_snapshot import (
    TASK_FIELD_NOT_LOADED,
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
    load_task_setup_snapshot_sync,
)
from xagent.web.tools.config import WebToolConfig

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "xagent"


@pytest.fixture()
def gate_db(tmp_path) -> Iterator[Any]:
    init_db(db_url=f"sqlite:///{tmp_path / 'team_member_gate.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


def _user(db, name: str) -> User:
    user = User(username=name, password_hash="hash", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _channel(db, user_id: int) -> UserChannel:
    channel = UserChannel(
        user_id=user_id,
        channel_type="feishu",
        channel_name="feishu-app",
        _config={},
        is_active=True,
    )
    db.add(channel)
    db.commit()
    db.refresh(channel)
    return channel


def _task(db, user_id: int, **columns: Any) -> Task:
    values: dict[str, Any] = dict(
        user_id=user_id,
        title="gate task",
        description="gate task",
        status=TaskStatus.PENDING,
        execution_mode="flash",
    )
    values.update(columns)
    explicit_null_source = "source" in values and values["source"] is None
    if explicit_null_source:
        del values["source"]
    task = Task(**values)
    db.add(task)
    db.commit()
    if explicit_null_source:
        db.query(Task).filter(Task.id == task.id).update({Task.source: None})
        db.commit()
    db.refresh(task)
    return task


def _snapshot_fields(task: Task) -> _TaskFields:
    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=int(task.user_id)
    )
    assert snapshot is not None
    assert isinstance(snapshot.task, _TaskFields)
    return snapshot.task


# (case id, task columns, allowed). ``channel`` stands for "link the task to a
# real user_channels row".
_ROW_CASES: list[tuple[str, dict[str, Any], bool]] = [
    ("internal_no_channel", {"source": "internal"}, True),
    ("none_source_no_channel", {"source": None}, True),
    ("internal_with_channel", {"source": "internal", "channel": True}, False),
    ("trigger", {"source": "trigger"}, True),
    ("trigger_with_channel", {"source": "trigger", "channel": True}, False),
    ("sdk_personal", {"source": "sdk", "sdk_credential_kind": "personal"}, True),
    ("sdk_runtime_null_kind", {"source": "sdk", "sdk_credential_kind": None}, False),
    ("a2a", {"source": "a2a"}, False),
    ("widget", {"source": "widget"}, False),
    ("shared_link", {"source": "shared_link"}, False),
    ("external", {"source": "external"}, False),
    ("unknown_source", {"source": "brand_new_source"}, False),
    (
        "unbound_then_rebound_feishu_task",
        {"source": "internal", "channel_name": "feishu-app"},
        False,
    ),
    (
        "telegram_task_channel_unbound",
        {"source": "internal", "telegram_user_id": "777001"},
        False,
    ),
]


def _make_row_task(db, user_id: int, columns: dict[str, Any]) -> Task:
    columns = dict(columns)
    if columns.pop("channel", False):
        channel = _channel(db, user_id)
        columns["channel_id"] = int(channel.id)
        columns["channel_name"] = str(channel.channel_name)
    return _task(db, user_id, **columns)


def _hand_built_task_fields() -> _TaskFields:
    task_fields = _TaskFields(
        id=1,
        user_id=1,
        status=TaskStatus.PENDING,
        agent_id=None,
        agent_config=None,
        model_name=None,
        compact_model_name=None,
        execution_mode="flash",
        agent_type="standard",
        source="internal",
    )
    for name in (
        "channel_id",
        "channel_name",
        "telegram_user_id",
        "sdk_credential_kind",
    ):
        assert getattr(task_fields, name) is TASK_FIELD_NOT_LOADED
    return task_fields


def _create_task_through_chat_api(db) -> Task:
    """Write a web chat task with the real create-task route."""

    def _override() -> Iterator[Any]:
        session = None
        try:
            session = next(get_db())
            yield session
        finally:
            if session is not None:
                session.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.dependency_overrides[get_db] = _override
    app.dependency_overrides[get_auth_db] = auth_db_override(_override)
    client = TestClient(app, raise_server_exceptions=False)

    setup = client.post(
        "/api/auth/setup-admin",
        json={
            "username": "admin",
            "email": "admin@example.com",
            "password": "admin123",
        },
    )
    assert setup.status_code == 200, setup.text
    login = client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin123"}
    )
    assert login.status_code == 200, login.text
    created = client.post(
        "/api/chat/task/create",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
        json={"title": "web chat", "description": "web chat"},
    )
    assert created.status_code == 200, created.text

    db.expire_all()
    task = db.query(Task).filter(Task.id == int(created.json()["task_id"])).one()
    assert task.channel_id is None
    assert task.channel_name is None
    assert task.telegram_user_id is None
    return task


_ALL_CASE_IDS = [case_id for case_id, _, _ in _ROW_CASES] + [
    "task_none",
    "web_task_channel_name_null",
    "hand_built_fields_without_new_attrs",
]
_ROW_CASES_BY_ID = {case_id: (cols, ok) for case_id, cols, ok in _ROW_CASES}


@pytest.mark.parametrize("case_id", _ALL_CASE_IDS)
def test_team_member_grants_allowed_for_snapshot_task(gate_db, case_id: str) -> None:
    if case_id == "task_none":
        assert team_member_mcp_grants_allowed_for_task(None) is False
        return
    if case_id == "hand_built_fields_without_new_attrs":
        assert (
            team_member_mcp_grants_allowed_for_task(_hand_built_task_fields()) is False
        )
        return

    if case_id == "web_task_channel_name_null":
        task = _create_task_through_chat_api(gate_db)
        allowed = True
    else:
        columns, allowed = _ROW_CASES_BY_ID[case_id]
        user = _user(gate_db, "gate-member")
        task = _make_row_task(gate_db, int(user.id), columns)

    assert team_member_mcp_grants_allowed_for_task(_snapshot_fields(task)) is allowed
    assert team_member_mcp_grants_allowed_for_task(task) is allowed


# --- grant-forwarding registry ------------------------------------------------

_FORWARDING_KEYWORDS = frozenset(
    {"allow_team_member_grants", "mcp_team_member_grants_allowed"}
)

# (path under src/xagent, enclosing function) of every call that passes one of
# the keywords above, whatever the value. A new call site must be added here on
# purpose.
_REGISTERED_CALLS = frozenset(
    {
        (
            "web/services/agent_service_manager.py",
            "AgentServiceManager._build_tools_for_task",
        ),
        (
            "web/services/agent_service_manager.py",
            "AgentServiceManager._get_agent_for_task_unlocked",
        ),
        ("web/services/agent_service_manager.py", "create_default_tools"),
        ("web/tools/config.py", "WebToolConfig._build_mcp_server_config"),
        ("web/services/mcp_runtime.py", "build_mcp_runtime_connection"),
        ("web/services/mcp_oauth.py", "resolve_mcp_oauth_runtime_auth"),
    }
)


class _CallScanner(ast.NodeVisitor):
    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.scope: list[str] = []
        self.found: set[tuple[str, str]] = set()

    def _enter(self, node: ast.AST) -> None:
        self.scope.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter
    visit_ClassDef = _enter

    def visit_Call(self, node: ast.Call) -> None:
        if any(kw.arg in _FORWARDING_KEYWORDS for kw in node.keywords):
            self.found.add((self.relpath, ".".join(self.scope) or "<module>"))
        self.generic_visit(node)


def _scan_forwarding_calls(root: Path) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scanner = _CallScanner(path.relative_to(root).as_posix())
        scanner.visit(tree)
        found |= scanner.found
    return found


def test_team_member_grant_kwargs_are_registered() -> None:
    assert _scan_forwarding_calls(SRC_ROOT) == set(_REGISTERED_CALLS)


def test_forwarding_scan_catches_unregistered_call_sites(tmp_path: Path) -> None:
    route = tmp_path / "web" / "api" / "new_route.py"
    route.parent.mkdir(parents=True)
    route.write_text(
        textwrap.dedent(
            """
            def list_things(db):
                return select_mcp_oauth_grants(db, allow_team_member_grants=True)
            """
        ),
        encoding="utf-8",
    )
    builder = tmp_path / "web" / "tools" / "third_builder.py"
    builder.parent.mkdir(parents=True)
    builder.write_text(
        textwrap.dedent(
            """
            class Builder:
                def build(self, flag):
                    return WebToolConfig(mcp_team_member_grants_allowed=flag)
            """
        ),
        encoding="utf-8",
    )

    found = _scan_forwarding_calls(tmp_path)

    assert found == {
        ("web/api/new_route.py", "list_things"),
        ("web/tools/third_builder.py", "Builder.build"),
    }
    assert not (found & set(_REGISTERED_CALLS))


def test_gate_reads_fields_without_getattr_defaults() -> None:
    source = textwrap.dedent(inspect.getsource(team_member_mcp_grants_allowed_for_task))
    function = ast.parse(source).body[0]
    getattr_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
    ]
    assert getattr_calls == []

    defaults = {f.name: f.default for f in fields(_TaskFields)}
    for name in (
        "channel_id",
        "channel_name",
        "telegram_user_id",
        "sdk_credential_kind",
    ):
        assert defaults[name] is TASK_FIELD_NOT_LOADED

    assert pickle.loads(pickle.dumps(TASK_FIELD_NOT_LOADED)) is TASK_FIELD_NOT_LOADED
    copied = copy.deepcopy(_hand_built_task_fields())
    assert copied.channel_id is TASK_FIELD_NOT_LOADED


def test_gate_refuses_when_only_sdk_credential_kind_is_not_loaded() -> None:
    task_fields = _TaskFields(
        id=1,
        user_id=1,
        status=TaskStatus.PENDING,
        agent_id=None,
        agent_config=None,
        model_name=None,
        compact_model_name=None,
        execution_mode="flash",
        agent_type="standard",
        source="internal",
        channel_id=None,
        channel_name=None,
        telegram_user_id=None,
    )
    assert task_fields.sdk_credential_kind is TASK_FIELD_NOT_LOADED
    assert team_member_mcp_grants_allowed_for_task(task_fields) is False


@pytest.mark.parametrize(
    ("function", "parameter"),
    [
        pytest.param(
            asm_module.create_default_tools,
            "mcp_team_member_grants_allowed",
            id="create_default_tools",
        ),
        pytest.param(
            WebToolConfig.__init__,
            "mcp_team_member_grants_allowed",
            id="WebToolConfig.__init__",
        ),
        pytest.param(
            build_mcp_runtime_connection,
            "allow_team_member_grants",
            id="build_mcp_runtime_connection",
        ),
        pytest.param(
            resolve_mcp_oauth_runtime_auth,
            "allow_team_member_grants",
            id="resolve_mcp_oauth_runtime_auth",
        ),
        pytest.param(
            select_mcp_oauth_grants,
            "allow_team_member_grants",
            id="select_mcp_oauth_grants",
        ),
    ],
)
def test_team_member_grant_parameters_default_to_false(
    function: Any, parameter: str
) -> None:
    assert inspect.signature(function).parameters[parameter].default is False


@pytest.mark.asyncio
async def test_build_tools_for_task_forwards_gate_result() -> None:
    def _fields(**channel: Any) -> _TaskFields:
        return _TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.PENDING,
            agent_id=None,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
            source="internal",
            channel_id=channel.get("channel_id"),
            channel_name=None,
            telegram_user_id=None,
            sdk_credential_kind=None,
        )

    for task, expected in ((_fields(channel_id=5), False), (_fields(), True)):
        manager = AgentServiceManager()
        snapshot = TaskSetupSnapshot(
            task=task,
            runtime_user=RuntimeUserFields(id=1, is_admin=False),
            has_reconstructable_history=False,
            task_pattern="single_call",
            task_llm=None,
            task_fast_llm=None,
            task_vision_llm=None,
            task_compact_llm=None,
            agent=None,
            agent_config=None,
            excluded_agent_id=None,
        )
        with (
            patch(
                "xagent.web.services.agent_service_manager.create_default_tools",
                new=AsyncMock(return_value=([], MagicMock())),
            ) as create_tools_mock,
            patch.object(
                manager, "_get_or_create_task_sandbox", AsyncMock(return_value=None)
            ),
        ):
            await manager._build_tools_for_task(
                task_id=42,
                task=task,
                db=None,
                user=snapshot.runtime_user,
                agent_config=None,
                task_llm=None,
                task_vision_llm=None,
                task_setup_snapshot=snapshot,
            )
        kwargs = create_tools_mock.call_args.kwargs
        assert kwargs["mcp_team_member_grants_allowed"] is expected


# --- end to end through the snapshot path -------------------------------------

_MEMBER_TOKEN = "member-path-access-token"
_MCP_URL = "https://mcp.gate.test/mcp"
_ISSUER = "https://auth.gate.test"
_TEAM_ID = 101


def _team_connector_fixture(db) -> dict[str, int]:
    owner = _user(db, "gate-owner")
    member = _user(db, "gate-member-e2e")
    server = MCPServer.from_config(
        {
            "name": "gate-team-connector",
            "managed": "external",
            "transport": "streamable_http",
            "url": _MCP_URL,
            "auth": {"type": "mcp_oauth"},
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    agent = Agent(
        user_id=owner.id,
        name="gate-team-agent",
        instructions="be terse",
        execution_mode="balanced",
        status=AgentStatus.PUBLISHED,
        tool_categories=["mcp"],
        team_id=_TEAM_ID,
        visibility="team",
        models={},
        knowledge_bases=[],
        skills=[],
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    oauth_client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer=_ISSUER,
        authorization_endpoint=f"{_ISSUER}/authorize",
        token_endpoint=f"{_ISSUER}/token",
        client_id="gate-client",
        token_endpoint_auth_method="none",
        redirect_uri="https://api.gate.test/api/mcp/oauth/callback",
    )
    db.add(oauth_client)
    db.flush()
    db.add(
        MCPOAuthGrant(
            mcp_server_id=server.id,
            user_id=member.id,
            mcp_oauth_client_id=oauth_client.id,
            resource_owner_key=f"xagent:user:{member.id}",
            issuer=_ISSUER,
            resource=_MCP_URL,
            scope="",
            access_token=encrypt_value(_MEMBER_TOKEN),
            status="active",
        )
    )
    db.commit()
    return {
        "server_id": int(server.id),
        "agent_id": int(agent.id),
        "member_id": int(member.id),
    }


async def _build_tools_and_load_mcp(
    db, *, task: Task, member_id: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the real snapshot path of ``get_agent_for_task`` and load the MCP
    server configs from the tool config it built."""
    manager = AgentServiceManager()
    manager._default_llm = MagicMock()
    snapshot = load_task_setup_snapshot_sync(
        task_id=int(task.id), task_owner_user_id=member_id
    )
    assert snapshot is not None
    assert snapshot.agent is not None and snapshot.agent.team_id == _TEAM_ID

    captured: dict[str, Any] = {}
    real_create_default_tools = asm_module.create_default_tools

    async def _spy(*args: Any, **kwargs: Any) -> Any:
        result = await real_create_default_tools(*args, **kwargs)
        captured["kwargs"] = kwargs
        captured["tool_config"] = result[1]
        return result

    async def _no_tools(config: Any) -> list:
        return []

    member = db.query(User).filter(User.id == member_id).one()
    with (
        patch.object(asm_module, "create_default_tools", new=_spy),
        patch.object(ToolFactory, "create_all_tools", _no_tools),
        patch.object(
            manager, "_get_or_create_task_sandbox", AsyncMock(return_value=None)
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.services.agent_service_manager.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch("xagent.web.services.agent_service_manager.AgentService"),
    ):
        try:
            await manager.get_agent_for_task(
                task_id=int(task.id),
                db=db,
                user=member,
                task_setup_snapshot=snapshot,
            )
        except Exception:
            # Construction after tool building runs against a mocked
            # AgentService; what is asserted was recorded before that.
            pass

    assert "tool_config" in captured, "tool building was never reached"
    tool_config = captured["tool_config"]
    configs = await tool_config.get_mcp_server_configs()
    return configs, tool_config.get_mcp_oauth_diagnostics(), captured["kwargs"]


@pytest.mark.asyncio
async def test_channel_task_through_snapshot_path_excludes_member_grant(
    gate_db,
) -> None:
    ids = _team_connector_fixture(gate_db)
    server_id = ids["server_id"]
    member_id = ids["member_id"]
    channel = _channel(gate_db, member_id)

    def _team_visibility(db: Any, *, team_id: int) -> dict[str, set[int]]:
        if team_id == _TEAM_ID:
            return {"mcp": {server_id}, "custom_api": set()}
        return {"mcp": set(), "custom_api": set()}

    set_agent_team_scope_hook(
        lambda db, user_id: (
            AgentTeamScope(team_id=_TEAM_ID, is_team_admin=False)
            if user_id == member_id
            else None
        )
    )
    connector_team_scope.set_connector_team_hooks(team_visibility=_team_visibility)
    try:
        channel_task = _task(
            gate_db,
            member_id,
            source="internal",
            agent_id=ids["agent_id"],
            channel_id=int(channel.id),
            channel_name=str(channel.channel_name),
        )
        rebound_task = _task(
            gate_db,
            member_id,
            source="internal",
            agent_id=ids["agent_id"],
            channel_id=None,
            channel_name=str(channel.channel_name),
        )
        web_task = _task(
            gate_db,
            member_id,
            source="internal",
            agent_id=ids["agent_id"],
        )

        for task in (channel_task, rebound_task):
            configs, diagnostics, kwargs = await _build_tools_and_load_mcp(
                gate_db, task=task, member_id=member_id
            )
            assert kwargs["mcp_team_member_grants_allowed"] is False
            assert [d.get("code") for d in diagnostics] == ["authorization_required"]
            assert _MEMBER_TOKEN not in json.dumps(configs, default=str)

        configs, diagnostics, kwargs = await _build_tools_and_load_mcp(
            gate_db, task=web_task, member_id=member_id
        )
        assert kwargs["mcp_team_member_grants_allowed"] is True
        assert diagnostics == []
        assert _MEMBER_TOKEN in json.dumps(configs, default=str)
    finally:
        set_agent_team_scope_hook(None)
        connector_team_scope.set_connector_team_hooks()
