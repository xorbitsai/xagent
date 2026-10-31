"""Personal keys on the ``/v1/chat/tasks`` routes.

A personal key resolves to its holder. Tasks it creates are owned by the
holder, record ``sdk_credential_kind = 'personal'``, and are reachable only
by personal keys of that holder; runtime keys never reach them, and a
personal key never reaches a task a runtime key or another surface created.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from xagent.web.api.v1 import tasks as v1_tasks
from xagent.web.api.v1.deps import (
    AgentPrincipalSnapshot,
    ApiKeyPrincipal,
    PersonalApiKeySnapshot,
    RuntimeApiKeySnapshot,
    UserPrincipalSnapshot,
    _resolve_principal_from_credentials,
)
from xagent.web.api.v1.errors import V1ApiError
from xagent.web.models.agent import Agent, AgentOrigin, AgentStatus
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.expired_task import ExpiredTaskTombstone
from xagent.web.models.task import Task, TaskConnectorRuntimeContext, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.models.workforce import WorkforceRun
from xagent.web.services import task_start
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)
from xagent.web.services.task_command_transport import COMMAND_COMPLETED
from xagent.web.services.task_runtime import PROMPT_CONTEXT_AGENT_CONFIG_KEY

from ..conftest import _admin_headers, _direct_db_session, _register_second_user, client
from .test_auth import _assert_invalid_api_key, _create_personal_key
from .test_workforces import _create_active_workforce, _create_workforce_key

pytestmark = pytest.mark.usefixtures("_test_db")

HOLDER = UserPrincipalSnapshot(id=5, username="holder", email=None, is_admin=False)
AGENT = AgentPrincipalSnapshot(
    id=3, user_id=9, execution_mode="balanced", status="draft", origin="user"
)


def _bearer(full_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {full_key}"}


def test_api_key_principal_rejects_inconsistent_kind() -> None:
    personal = PersonalApiKeySnapshot(key_prefix="abc123")
    runtime = RuntimeApiKeySnapshot(key_prefix="abc123")
    with pytest.raises(ValueError):
        ApiKeyPrincipal(key=personal, agent=AGENT)
    with pytest.raises(ValueError):
        ApiKeyPrincipal(key=runtime, agent=AGENT, user=HOLDER)

    holder = ApiKeyPrincipal(key=personal, user=HOLDER)
    agent_key = ApiKeyPrincipal(key=runtime, agent=AGENT)
    assert (holder.owner_user_id, agent_key.owner_user_id) == (5, 9)
    # Same prefix, different key tables: the stream slot identity differs,
    # and a runtime key keeps the bare prefix it has always counted under.
    assert holder.stream_principal_id == "personal:abc123"
    assert agent_key.stream_principal_id == "abc123"
    with pytest.raises(RuntimeError):
        _ = ApiKeyPrincipal(key=runtime).owner_user_id


def test_personal_key_resolves_to_its_holder() -> None:
    full_key, prefix = _create_personal_key()
    me = client.get("/v1/me", headers=_bearer(full_key)).json()

    principal = _resolve_principal_from_credentials(
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=full_key)
    )

    assert principal == ApiKeyPrincipal(
        key=PersonalApiKeySnapshot(key_prefix=prefix),
        user=UserPrincipalSnapshot(
            id=me["user_id"], username="admin", email="admin@example.com", is_admin=True
        ),
    )


def test_personal_key_rejected_on_runtime_only_routes() -> None:
    full_key, _prefix = _create_personal_key()

    run = client.post(
        "/v1/workforces/1/runs",
        headers=_bearer(full_key),
        json={"message": {"role": "user", "content": "go"}},
    )
    _assert_invalid_api_key(run)
    a2a = client.get(
        "/api/a2a/agents/1/tasks", headers={**_bearer(full_key), "A2A-Version": "1.0"}
    )
    assert a2a.status_code == 401, a2a.text
    assert a2a.json()["error"]["details"][0]["reason"] == "INVALID_API_KEY"


# ===== Task routes under a personal key =====

UNUSED_TASK_ID = 9_000_002
PERSONAL_PROMPT_CONTEXT_MESSAGE = "prompt_context is not supported for personal keys."
# Whether the holder (a plain member of team 1) can run each agent.
VISIBLE = {
    "own": True,
    "teammate": True,
    "admins_only": False,
    "admins_only_as_team_admin": True,
    "other_team": False,
    "published_elsewhere": False,
    "manager": False,
    "missing": False,
}
# Inputs that are rejected for a visible agent; an invisible one is a 404 first.
BAD_INPUTS = {
    "plain": {},
    "unknown_file": {"message": {"files": ["file-that-does-not-exist"]}},
    "unselected_connector": {
        "connector_runtime_context": [
            {
                "connector_ref": {"connector_type": "mcp", "connector_id": 424242},
                "context": {"region": "eu"},
            }
        ]
    },
}


@pytest.fixture(autouse=True)
def mock_schedule_bg():
    with patch(
        "xagent.web.services.task_orchestrator._schedule_bg", new=MagicMock()
    ) as mocked:
        yield mocked


def _update(model, row_id: int, **values) -> None:
    db = _direct_db_session()
    try:
        db.query(model).filter(model.id == row_id).update(
            values, synchronize_session=False
        )
        db.commit()
    finally:
        db.close()


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User.id).filter(User.username == username).scalar())
    finally:
        db.close()


def _agent(headers: dict[str, str], name: str, **values) -> int:
    resp = client.post(
        "/api/agents",
        headers=headers,
        json={
            "name": name,
            "description": "personal key test agent",
            "instructions": "test",
            "execution_mode": "balanced",
        },
    )
    assert resp.status_code == 200, resp.text
    agent_id = int(resp.json()["id"])
    if values:
        _update(Agent, agent_id, **values)
    return agent_id


def _agent_key(headers: dict[str, str], agent_id: int) -> str:
    resp = client.post(f"/api/agents/{agent_id}/api-key", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["full_key"]


def _task(task_id: int) -> Task:
    db = _direct_db_session()
    try:
        return db.query(Task).filter(Task.id == task_id).one()
    finally:
        db.close()


def _row_counts() -> dict[str, int]:
    db = _direct_db_session()
    try:
        return {
            model.__tablename__: db.query(model).count()
            for model in (
                Task,
                TaskChatMessage,
                TaskConnectorRuntimeContext,
                TaskExecutionCommand,
            )
        }
    finally:
        db.close()


def _create(full_key: str, agent_id: int, **body):
    message = {"role": "user", "content": "hi", **body.pop("message", {})}
    return client.post(
        "/v1/chat/tasks",
        headers=_bearer(full_key),
        json={"agent_id": agent_id, "message": message, **body},
    )


def _finish(task_id: int) -> None:
    """Settle a created task the way a finished turn leaves it."""
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        task.status = TaskStatus.COMPLETED
        task.control_state = TaskStatus.COMPLETED.value
        task.lease_expires_at = None
        db.query(TaskExecutionCommand).filter(
            TaskExecutionCommand.task_id == task_id
        ).update({"status": COMMAND_COMPLETED})
        db.commit()
    finally:
        db.close()


def _call(route: str, task_id: int, full_key: str, owner: dict):
    headers = _bearer(full_key)
    message = {"role": "user", "content": "again"}
    if route == "get":
        return client.get(f"/v1/chat/tasks/{task_id}", headers=headers)
    if route == "events":
        return client.get(f"/v1/chat/tasks/{task_id}/events", headers=headers)
    if route == "append":
        return client.post(
            f"/v1/chat/tasks/{task_id}/messages",
            headers=headers,
            json={**owner, "message": message},
        )
    if route == "reply":
        return client.post(
            f"/v1/chat/tasks/{task_id}/reply",
            headers=headers,
            json={**owner, "command_id": "cmd-personal", "message": message},
        )
    if route == "cancel":
        return client.post(f"/v1/chat/tasks/{task_id}/cancel-question", headers=headers)
    if route == "delete":
        return client.delete(f"/v1/chat/tasks/{task_id}", headers=headers)
    raise AssertionError(route)


def _error_code(resp) -> str | None:
    body = resp.json()
    return body.get("error", {}).get("code") if isinstance(body, dict) else None


@pytest.fixture
def world():
    """The holder (admin, with a personal key) and bob in team 1; carol in 2."""
    scopes: dict[int, AgentTeamScope | None] = {}
    set_agent_team_scope_hook(lambda _db, user_id: scopes.get(int(user_id)))
    headers = {
        "admin": _admin_headers(),
        "bob": _register_second_user("bob", "bobpass1"),
        "carol": _register_second_user("carol", "carolpass1"),
    }
    ids = {name: _user_id(name) for name in headers}
    key, prefix = _create_personal_key()
    manager = AgentOrigin.WORKFORCE_GENERATED_MANAGER.value
    agents = {
        "own": _agent(headers["admin"], "own"),
        "teammate": _agent(headers["bob"], "teammate", team_id=1),
        "admins_only": _agent(
            headers["bob"], "admins only", team_id=1, visibility="admins"
        ),
        "other_team": _agent(headers["carol"], "other team", team_id=2),
        "published_elsewhere": _agent(
            headers["carol"], "published", status=AgentStatus.PUBLISHED
        ),
        "manager": _agent(headers["admin"], "manager", origin=manager),
        "missing": 999_999,
    }
    agents["admins_only_as_team_admin"] = agents["admins_only"]
    team_scopes = {
        ids["admin"]: AgentTeamScope(team_id=1, is_team_admin=False),
        ids["bob"]: AgentTeamScope(team_id=1, is_team_admin=False),
        ids["carol"]: AgentTeamScope(team_id=2, is_team_admin=False),
    }
    scopes.update(team_scopes)
    try:
        yield SimpleNamespace(
            key=key,
            prefix=prefix,
            holder_id=ids["admin"],
            ids=ids,
            headers=headers,
            agents=agents,
            scopes=scopes,
            team_scopes=team_scopes,
        )
    finally:
        set_agent_team_scope_hook(None)


def _personal_task(world, agent: str) -> int:
    resp = _create(world.key, world.agents[agent])
    assert resp.status_code == 202, resp.text
    return int(resp.json()["task_id"])


@pytest.mark.parametrize("agent", ["own", "teammate"])
def test_personal_key_task_runs_as_key_holder(world, monkeypatch, agent) -> None:
    calls: list[dict] = []
    create_sdk_task = task_start.create_sdk_task

    async def recording(**kwargs):
        calls.append(kwargs)
        return await create_sdk_task(**kwargs)

    monkeypatch.setattr(task_start, "create_sdk_task", recording)
    upload = client.post(
        "/v1/chat/files",
        headers=_bearer(world.key),
        files=[("files", ("note.txt", b"for the holder", "text/plain"))],
    )
    assert upload.status_code == 200, upload.text
    file_id = upload.json()["files"][0]["file_id"]

    # The holder's upload only binds to a task the holder owns.
    resp = _create(world.key, world.agents[agent], message={"files": [file_id]})

    assert resp.status_code == 202, resp.text
    task = _task(resp.json()["task_id"])
    assert (task.user_id, task.source, task.sdk_credential_kind) == (
        world.holder_id,
        "sdk",
        "personal",
    )
    assert [(c["task_owner_user_id"], c["actor_user_id"]) for c in calls] == [
        (world.holder_id, world.holder_id)
    ]


@pytest.mark.parametrize("cell", list(VISIBLE))
def test_personal_key_create_task_agent_visibility_matrix(world, cell) -> None:
    if cell == "admins_only_as_team_admin":
        world.scopes[world.holder_id] = AgentTeamScope(team_id=1, is_team_admin=True)
    agent_id = world.agents[cell]

    if VISIBLE[cell]:
        resp = _create(world.key, agent_id)
        assert resp.status_code == 202, resp.text
        assert _task(resp.json()["task_id"]).user_id == world.holder_id
        return
    before = _row_counts()
    for label, extra in BAD_INPUTS.items():
        resp = _create(world.key, agent_id, **extra)
        assert resp.status_code == 404, (label, resp.text)
        assert _error_code(resp) == "agent_not_found", label
    assert _row_counts() == before


def test_credential_kind_column_mapping(world) -> None:
    agent_id = world.agents["own"]
    runtime = _create(_agent_key(world.headers["admin"], agent_id), agent_id)
    assert runtime.status_code == 202, runtime.text
    assert _task(runtime.json()["task_id"]).sdk_credential_kind is None


# The write routes that start work on the task re-check that the holder can
# still run its agent; reading, cancelling a question and deleting do not.
ALLOWED = {"append": 202, "reply": 409, "cancel": 409, "delete": 200, "get": 200}


@pytest.mark.parametrize("reachable", [True, False], ids=["in_team", "removed"])
@pytest.mark.parametrize("route", list(ALLOWED))
def test_personal_key_write_routes_recheck_agent_visibility(
    world, route, reachable
) -> None:
    task_id = _personal_task(world, "teammate")
    _finish(task_id)
    if not reachable:
        world.scopes[world.holder_id] = None

    resp = _call(route, task_id, world.key, {})

    if route in ("append", "reply") and not reachable:
        assert resp.status_code == 404, resp.text
        assert _error_code(resp) == "agent_not_found"
    else:
        assert resp.status_code == ALLOWED[route], resp.text


def test_personal_key_task_scope_matrix(world) -> None:
    """A personal key reaches only tasks its holder created with a personal key."""
    own = world.agents["own"]
    agent_key_task = _create(_agent_key(world.headers["admin"], own), own)
    assert agent_key_task.status_code == 202, agent_key_task.text
    bob_key = client.post("/api/me/personal-keys", headers=world.headers["bob"])
    assert bob_key.status_code == 200, bob_key.text
    bobs = _create(bob_key.json()["full_key"], world.agents["teammate"])
    assert bobs.status_code == 202, bobs.text
    world.scopes.clear()  # the workforce belongs to no team
    workforce_id = _create_active_workforce(world.headers["admin"], name="Scope WF")
    run = client.post(
        f"/v1/workforces/{workforce_id}/runs",
        headers=_bearer(_create_workforce_key(world.headers["admin"], workforce_id)),
        json={"message": {"role": "user", "content": "go"}},
    )
    assert run.status_code == 202, run.text
    world.scopes.update(world.team_scopes)
    deleted_agent = _personal_task(world, "own")
    _update(Task, deleted_agent, agent_id=None)
    db = _direct_db_session()
    try:
        web, rewritten = (
            Task(
                user_id=world.holder_id,
                title="not an SDK task",
                agent_id=own,
                source="internal",
                sdk_credential_kind=kind,
            )
            for kind in (None, "personal")
        )
        db.add_all([web, rewritten])
        db.commit()
        web_id, rewritten_id = int(web.id), int(rewritten.id)
    finally:
        db.close()

    expected = {
        "personal_key_task": (_personal_task(world, "own"), 200),
        "agent_key_task_same_owner": (agent_key_task.json()["task_id"], 404),
        "another_holders_task": (bobs.json()["task_id"], 404),
        "workforce_run_same_owner": (run.json()["task_id"], 404),
        "web_task": (web_id, 404),
        "rewritten_by_rollback": (rewritten_id, 404),
        "agent_deleted": (deleted_agent, 404),
    }
    for label, (task_id, status) in expected.items():
        resp = client.get(f"/v1/chat/tasks/{task_id}", headers=_bearer(world.key))
        assert resp.status_code == status, (label, resp.text)


@pytest.mark.parametrize("route", ["get", "append", "delete", "events"])
def test_runtime_keys_never_see_personal_key_tasks(world, route) -> None:
    agent_id = world.agents["teammate"]
    task_id = _personal_task(world, "teammate")
    _finish(task_id)
    world.scopes.clear()
    workforce_id = _create_active_workforce(world.headers["admin"], name="Bound WF")
    workforce_key = _create_workforce_key(world.headers["admin"], workforce_id)
    world.scopes.update(world.team_scopes)
    db = _direct_db_session()
    try:
        db.add(
            WorkforceRun(
                workforce_id=workforce_id,
                task_id=task_id,
                user_id=world.holder_id,
                status="completed",
                snapshot={"version": 1},
            )
        )
        db.commit()
    finally:
        db.close()

    callers = [
        # The agent owner's key, then one a teammate generated afterwards
        # (generating a key revokes the earlier one).
        ("agent_owner_key", lambda: _agent_key(world.headers["bob"], agent_id)),
        ("teammate_key", lambda: _agent_key(world.headers["admin"], agent_id)),
        ("workforce_key", lambda: workforce_key),
    ]
    for label, make_key in callers:
        owner = {} if label == "workforce_key" else {"agent_id": agent_id}
        resp = _call(route, task_id, make_key(), owner)
        assert resp.status_code == 404, (label, resp.text)
        assert _error_code(resp) == "task_not_found", label
    assert _task(task_id).id == task_id


@pytest.mark.parametrize("prompt_context", [{}, {"shift": "night"}])
def test_personal_key_prompt_context_rejected(world, prompt_context) -> None:
    before = _row_counts()
    created = _create(world.key, world.agents["own"], prompt_context=prompt_context)
    assert created.status_code == 422, created.text
    assert created.json()["error"]["message"] == PERSONAL_PROMPT_CONTEXT_MESSAGE
    assert _row_counts() == before

    task_id = _personal_task(world, "own")
    _finish(task_id)
    stored = {PROMPT_CONTEXT_AGENT_CONFIG_KEY: {"data": {"shift": "day"}}}
    _update(Task, task_id, agent_config=stored)
    before = _row_counts()

    appended = client.post(
        f"/v1/chat/tasks/{task_id}/messages",
        headers=_bearer(world.key),
        json={
            "message": {"role": "user", "content": "x"},
            "prompt_context": prompt_context,
        },
    )

    assert appended.status_code == 422, appended.text
    assert appended.json()["error"]["message"] == PERSONAL_PROMPT_CONTEXT_MESSAGE
    assert _task(task_id).agent_config == stored
    assert _row_counts() == before


def test_sdk_task_scope_without_owner_fails_closed(world) -> None:
    task_id = _personal_task(world, "own")
    empty = task_start.SdkTaskScope(agent_id=None, workforce_id=None, user_id=None)
    db = _direct_db_session()
    try:
        # A tombstone that matches every owner column a scope could name.
        db.add(
            ExpiredTaskTombstone(
                task_id=UNUSED_TASK_ID,
                user_id=world.holder_id,
                agent_id=world.agents["own"],
                workforce_id=None,
                source="sdk",
                sdk_credential_kind=None,
                is_visible=False,
                is_channel_plumbing=False,
                expired_at=datetime(2026, 9, 1, tzinfo=UTC),
            )
        )
        db.commit()
        for lookup in (task_id, UNUSED_TASK_ID):
            with pytest.raises(task_start.TaskStartRejected) as rejected:
                task_start.resolve_sdk_task(lookup, empty, db)
            assert rejected.value.reason == "task_not_found"
        assert task_start._raise_if_expired_for_scope(UNUSED_TASK_ID, empty, db) is None
        with pytest.raises(task_start.TaskStartRejected) as rejected:
            task_start.validate_sdk_owner_scope(
                empty,
                request_agent_id=None,
                request_workforce_id=None,
                task_agent_id=world.agents["own"],
            )
        assert rejected.value.reason == "task_not_found"
    finally:
        db.close()

    ownerless = ApiKeyPrincipal(key=RuntimeApiKeySnapshot(key_prefix="abc123"))
    with pytest.raises(V1ApiError) as raised:
        v1_tasks._sdk_task_scope(ownerless)
    assert raised.value.http_status == 404
