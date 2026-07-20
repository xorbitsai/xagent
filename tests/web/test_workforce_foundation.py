import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from xagent.web.models import Agent, Base, User, Workforce, WorkforceAgent, WorkforceRun
from xagent.web.models.agent import AgentStatus
from xagent.web.services.workforce_access import ensure_workforce_access
from xagent.web.services.workforce_snapshot import (
    build_agent_tool_overrides,
    build_worker_tool_name,
    build_workforce_snapshot,
    normalize_workforce_run_status,
    normalize_workforce_status,
)
from xagent.web.services.workforce_workers import (
    create_workforce_worker,
    ensure_supported_source_type,
)


@pytest.fixture()
def db_session() -> Session:
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


def _create_user(db: Session, username: str, *, is_admin: bool = False) -> User:
    user = User(
        username=username,
        password_hash="hash",
        is_admin=is_admin,
    )
    db.add(user)
    db.flush()
    return user


def _create_agent(
    db: Session,
    user: User,
    name: str,
    *,
    status: AgentStatus = AgentStatus.PUBLISHED,
) -> Agent:
    agent = Agent(
        user_id=user.id,
        name=name,
        description=f"{name} description",
        instructions=f"{name} instructions",
        execution_mode="balanced",
        models={"general": "test-model"},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
        status=status,
    )
    db.add(agent)
    db.flush()
    return agent


def _create_workforce(
    db: Session,
    user: User,
    manager: Agent,
    *,
    status: str = "active",
) -> Workforce:
    workforce = Workforce(
        owner_user_id=user.id,
        scope_type="user",
        scope_id=str(user.id),
        name="Research Team",
        description="Coordinates research tasks",
        manager_agent_id=manager.id,
        status=status,
    )
    db.add(workforce)
    db.flush()
    return workforce


def test_workforce_models_are_registered(db_session: Session) -> None:
    tables = set(inspect(db_session.bind).get_table_names())

    assert "workforces" in tables
    assert "workforce_agents" in tables
    assert "workforce_runs" in tables
    assert "workforce_builder_messages" in tables


def test_deleting_workforce_deletes_runs_with_orm_cascade(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager)
    run = WorkforceRun(
        workforce_id=workforce.id,
        user_id=user.id,
        status="pending",
        snapshot={"version": 1},
    )
    db_session.add(run)
    db_session.commit()

    workforce_id = int(workforce.id)
    run_id = int(run.id)
    db_session.delete(workforce)
    db_session.commit()

    assert db_session.get(Workforce, workforce_id) is None
    assert db_session.get(WorkforceRun, run_id) is None


def test_build_workforce_snapshot_for_active_workforce(db_session: Session) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    worker_agent = _create_agent(db_session, user, "Analyst")
    workforce = _create_workforce(db_session, user, manager)
    worker = create_workforce_worker(
        db_session,
        workforce,
        user,
        source_type="existing",
        agent_id=worker_agent.id,
        alias="Research Analyst",
        assignment_instructions="Collect evidence and cite sources.",
    )

    snapshot = build_workforce_snapshot(db_session, user, workforce)
    overrides = build_agent_tool_overrides(snapshot, workforce_run_id=123)

    assert snapshot["version"] == 1
    assert snapshot["workforce"]["status"] == "active"
    assert snapshot["manager"]["agent_id"] == manager.id
    assert "Workforce Manager" in snapshot["manager"]["runtime_prompt"]
    # Workforce-level manager instructions are removed (#800).
    assert "workforce_instructions" not in snapshot["manager"]
    assert (
        "Workforce-specific manager instructions"
        not in snapshot["manager"]["runtime_prompt"]
    )
    assert snapshot["workers"] == [
        {
            "member_id": worker.id,
            "agent_id": worker_agent.id,
            "name": "Analyst",
            "alias": "Research Analyst",
            "description": "Analyst description",
            "assignment_instructions": "Collect evidence and cite sources.",
            "execution_mode": "balanced",
            "tool_name": build_worker_tool_name(worker_agent.id, "Research Analyst"),
            "enabled": True,
        }
    ]
    assert overrides[worker_agent.id]["workforce_run_id"] == 123
    assert overrides[worker_agent.id]["tool_name"] == (
        f"worker_research_analyst__a{worker_agent.id}"
    )
    assert (
        f"Research Analyst (tool: worker_research_analyst__a{worker_agent.id})"
        in snapshot["manager"]["runtime_prompt"]
    )

    snapshot["workers"][0]["tool_name"] = f"agent_{worker_agent.id}"
    legacy_overrides = build_agent_tool_overrides(snapshot, workforce_run_id=124)
    assert legacy_overrides[worker_agent.id]["tool_name"] == f"agent_{worker_agent.id}"


def test_validate_workforce_run_allows_draft_preview_and_requires_enabled_workers(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager, status="draft")
    worker_agent = _create_agent(db_session, user, "Analyst")
    worker = create_workforce_worker(
        db_session,
        workforce,
        user,
        source_type="existing",
        agent_id=worker_agent.id,
        assignment_instructions="Collect evidence.",
        enabled=True,
    )

    with pytest.raises(HTTPException) as draft_error:
        build_workforce_snapshot(db_session, user, workforce)
    assert draft_error.value.status_code == 400
    assert draft_error.value.detail == "Workforce must be active to run"

    snapshot = build_workforce_snapshot(
        db_session,
        user,
        workforce,
        is_preview=True,
    )
    assert snapshot["workforce"]["status"] == "draft"

    worker.enabled = False

    with pytest.raises(HTTPException) as worker_error:
        build_workforce_snapshot(
            db_session,
            user,
            workforce,
            is_preview=True,
        )
    assert worker_error.value.status_code == 400
    assert worker_error.value.detail == "Workforce requires at least one enabled worker"

    worker.enabled = True
    workforce.status = "archived"
    with pytest.raises(HTTPException) as archived_error:
        build_workforce_snapshot(
            db_session,
            user,
            workforce,
            is_preview=True,
        )
    assert archived_error.value.status_code == 400
    assert archived_error.value.detail == "Archived workforce cannot run"


def test_workforce_access_allows_owner_and_admin_only(db_session: Session) -> None:
    owner = _create_user(db_session, "owner")
    admin = _create_user(db_session, "admin", is_admin=True)
    other = _create_user(db_session, "other")
    manager = _create_agent(db_session, owner, "Manager")
    workforce = _create_workforce(db_session, owner, manager)

    assert ensure_workforce_access(db_session, owner, workforce) is workforce
    assert ensure_workforce_access(db_session, admin, workforce) is workforce

    with pytest.raises(HTTPException) as denied:
        ensure_workforce_access(db_session, other, workforce)
    assert denied.value.status_code == 403
    assert denied.value.detail == "Access denied"


def test_create_workforce_worker_requires_published_existing_agent(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    draft_agent = _create_agent(
        db_session, user, "Draft Worker", status=AgentStatus.DRAFT
    )
    workforce = _create_workforce(db_session, user, manager)

    with pytest.raises(HTTPException) as unsupported:
        ensure_supported_source_type("template")
    assert unsupported.value.status_code == 400
    assert unsupported.value.detail == (
        "source_type must be existing; publish an agent before adding it to a workforce"
    )

    with pytest.raises(HTTPException) as unpublished:
        create_workforce_worker(
            db_session,
            workforce,
            user,
            source_type="existing",
            agent_id=draft_agent.id,
            assignment_instructions="Collect evidence.",
        )
    assert unpublished.value.status_code == 400
    assert unpublished.value.detail == "Workforce agents must be published"


def test_create_workforce_worker_requires_workforce_edit_access(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, "owner")
    other = _create_user(db_session, "other")
    manager = _create_agent(db_session, owner, "Manager")
    other_agent = _create_agent(db_session, other, "Other Worker")
    workforce = _create_workforce(db_session, owner, manager)

    with pytest.raises(HTTPException) as denied:
        create_workforce_worker(
            db_session,
            workforce,
            other,
            source_type="existing",
            agent_id=other_agent.id,
            assignment_instructions="Collect evidence.",
        )

    assert denied.value.status_code == 403
    assert denied.value.detail == "Access denied"
    assert db_session.query(WorkforceAgent).count() == 0


def test_create_workforce_worker_rejects_duplicate_worker(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    worker_agent = _create_agent(db_session, user, "Analyst")
    workforce = _create_workforce(db_session, user, manager)

    create_workforce_worker(
        db_session,
        workforce,
        user,
        source_type="existing",
        agent_id=worker_agent.id,
        assignment_instructions="Collect evidence.",
    )

    with pytest.raises(HTTPException) as duplicate:
        create_workforce_worker(
            db_session,
            workforce,
            user,
            source_type="existing",
            agent_id=worker_agent.id,
            assignment_instructions="Collect evidence again.",
        )
    assert duplicate.value.status_code == 409
    assert duplicate.value.detail == "Agent already added to workforce"


def test_create_workforce_worker_rejects_manager_as_worker(
    db_session: Session,
) -> None:
    user = _create_user(db_session, "owner")
    manager = _create_agent(db_session, user, "Manager")
    workforce = _create_workforce(db_session, user, manager)

    with pytest.raises(HTTPException) as manager_worker:
        create_workforce_worker(
            db_session,
            workforce,
            user,
            source_type="existing",
            agent_id=manager.id,
            assignment_instructions="Manage the work.",
        )
    assert manager_worker.value.status_code == 400
    assert manager_worker.value.detail == "Manager agent cannot also be a worker"


def test_workforce_status_and_tool_name_normalization() -> None:
    tool_name = build_worker_tool_name(
        99,
        "This Worker Alias Is Long Enough To Require Stable Truncation",
    )

    assert normalize_workforce_status(None) == "draft"
    assert normalize_workforce_status(" ACTIVE ") == "active"
    assert normalize_workforce_run_status(None) == "pending"
    assert normalize_workforce_run_status(" Paused ") == "paused"
    assert normalize_workforce_run_status(" Completed ") == "completed"
    assert tool_name == (
        "worker_this_worker_alias_is_long_enough_to_require_stable__a99"
    )
    assert len(tool_name) <= 64
    assert build_worker_tool_name(98, "Duplicate Name") != build_worker_tool_name(
        99, "Duplicate Name"
    )
    assert build_worker_tool_name(42, "重复名字") == "worker_chong_fu_ming_zi__a42"

    with pytest.raises(HTTPException) as status_error:
        normalize_workforce_status("unknown")
    assert status_error.value.status_code == 400
    assert status_error.value.detail == "Invalid workforce status"
