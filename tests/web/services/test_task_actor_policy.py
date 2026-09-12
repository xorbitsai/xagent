"""Fresh shared actor execution retains only server-owned credential references."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_event_bridge, task_start_consumer
from xagent.web.services.mcp_runtime import (
    MCPActorAuthorizationPolicy,
    MCPBuiltinOAuthActorPolicyRequiredError,
)
from xagent.web.services.task_command_transport import claim_task_command
from xagent.web.services.task_orchestrator import TaskTurnOrchestrator, TaskTurnPayload
from xagent.web.services.task_runtime import (
    MCP_RUNTIME_AUTHORIZATION_POLICY_IDENTITY_KEY,
    MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY,
    MCP_RUNTIME_AUTHORIZATION_POLICY_STDIO_KEY,
    sanitize_client_agent_config,
)


@pytest.fixture
def actor_task(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_TASK_EXECUTION_ROLE", "worker")
    monkeypatch.setattr(
        task_event_bridge,
        "get_task_event_bridge",
        lambda: SimpleNamespace(require_ready=Mock()),
    )
    monkeypatch.setattr(task_start_consumer, "get_runner_id", lambda: "worker")
    init_db(db_url=f"sqlite:///{tmp_path / 'actor.db'}")
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(
            user_id=user.id,
            title="actor",
            source="external",
            status=TaskStatus.PENDING,
            agent_config={MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True},
        )
        db.add(task)
        db.commit()
        yield task.id, user.id
    Base.metadata.drop_all(bind=get_engine())


@pytest.mark.parametrize("stdio", [False, True])
def test_trusted_actor_reference_is_committed_before_start_and_reconstructed(
    actor_task, stdio
):
    task_id, owner_id = actor_task
    policy = MCPActorAuthorizationPolicy(
        resource_owner_key="actor:test", allow_builtin_stdio=stdio
    )
    with get_session_local()() as db:
        accepted = TaskTurnOrchestrator.claim_created_turn_no_commit(
            db,
            task_id=task_id,
            task_owner_user_id=owner_id,
            payload=TaskTurnPayload("hello"),
            mcp_runtime_authorization_policy=policy,
        )
        db.commit()
        row = db.get(TaskExecutionCommand, accepted.command_db_id)
        assert "actor:test" not in str(row.payload)
        claim = claim_task_command(db, runner_id="worker", command_db_id=row.id)
    handoff = task_start_consumer._commit_handoff(claim)
    assert handoff.actor_policy == policy
    assert handoff.actor_policy is not policy


def test_actor_marked_acceptance_without_trusted_policy_rolls_back(actor_task):
    task_id, owner_id = actor_task
    with get_session_local()() as db:
        with pytest.raises(MCPBuiltinOAuthActorPolicyRequiredError):
            TaskTurnOrchestrator.claim_created_turn_no_commit(
                db,
                task_id=task_id,
                task_owner_user_id=owner_id,
                payload=TaskTurnPayload("hello"),
            )
        db.rollback()
        assert db.get(Task, task_id).status == TaskStatus.PENDING
        assert db.query(TaskExecutionCommand).count() == 0


def test_client_cannot_supply_shared_actor_reference():
    assert sanitize_client_agent_config(
        {
            MCP_RUNTIME_AUTHORIZATION_POLICY_REQUIRED_KEY: True,
            MCP_RUNTIME_AUTHORIZATION_POLICY_IDENTITY_KEY: "actor:another",
            MCP_RUNTIME_AUTHORIZATION_POLICY_STDIO_KEY: True,
            "custom": "preserved",
        }
    ) == {"custom": "preserved"}
