"""Per-share run quota at the execute_task chokepoint (#973, PR2).

The owner run gate bounds the owner's team quota, but every anonymous share
run bills the owner, so a per-link + per-guest rolling ceiling is enforced on
top at run start. This verifies the share-quota block short-circuits a share
task's run when the quota is exhausted, and is skipped for non-share tasks.
"""

from __future__ import annotations

import pytest

from xagent.web.api.chat import AgentServiceManager
from xagent.web.models.database import Base, get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.share_rate_limit import reset_share_rate_limiter


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'share_run_quota.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=get_engine())


@pytest.fixture(autouse=True)
def _reset_limiter() -> None:
    reset_share_rate_limiter()
    yield
    reset_share_rate_limiter()


class _FakeAgentService:
    async def execute_task(self, **_kwargs):
        return {"success": True}

    def set_interrupt_checker(self, _checker):
        pass


def _make_task(db_session, *, agent_config: dict) -> Task:
    user = User(username="share-quota-user", password_hash="h", is_admin=False)
    db_session.add(user)
    db_session.commit()
    task = Task(
        user_id=user.id,
        title="share run",
        description="test",
        status=TaskStatus.PENDING,
        execution_mode="auto",
        agent_config=agent_config,
    )
    db_session.add(task)
    db_session.commit()
    return task


@pytest.mark.asyncio
async def test_share_run_quota_blocks_share_task(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "0/day" leaves no room, so the very first share run is refused.
    monkeypatch.setenv("XAGENT_SHARE_RUN_QUOTA", "0/day")
    reset_share_rate_limiter()

    task = _make_task(
        db_session,
        agent_config={
            "auth_mode": "share",
            "guest_id": "guest-abc",
            "share_agent_id": 4242,
        },
    )

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    assert result["success"] is False
    assert result["status"] == "quota_exceeded"
    assert result["error_code"] == "share_run_quota_exceeded"
    assert "usage limit" in result["output"]


@pytest.mark.asyncio
async def test_share_run_quota_skips_non_share_task(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-share task must never hit the share quota, even at 0/day."""
    monkeypatch.setenv("XAGENT_SHARE_RUN_QUOTA", "0/day")
    reset_share_rate_limiter()

    # No auth_mode == "share" marker: the share-quota branch is skipped, so the
    # run is not refused with the share error code (it proceeds to execution).
    task = _make_task(db_session, agent_config={})

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    assert result.get("error_code") != "share_run_quota_exceeded"


@pytest.mark.asyncio
async def test_widget_run_quota_blocks_widget_task(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "0/day" on the widget entity quota leaves no room, so the very first
    # widget run is refused — with widget-specific copy and error_code, not
    # the share wording. The share quota stays wide open, proving the widget
    # bucket gates independently. No widget_client_ip marker: this is the
    # legacy-task path, bounded by the entity quota alone.
    monkeypatch.setenv("XAGENT_WIDGET_RUN_QUOTA", "0/day")
    monkeypatch.setenv("XAGENT_SHARE_RUN_QUOTA", "500/day")
    reset_share_rate_limiter()

    task = _make_task(
        db_session,
        agent_config={
            "auth_mode": "widget",
            # Client-supplied guest_id must NOT be what gates the run.
            "guest_id": "rotatable-guest",
            "widget_agent_id": 4242,
        },
    )

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    assert result["success"] is False
    assert result["status"] == "quota_exceeded"
    assert result["error_code"] == "widget_run_quota_exceeded"
    assert "widget has reached its usage limit" in result["output"]


@pytest.mark.asyncio
async def test_widget_run_ip_quota_blocks_one_creator(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-creating-IP sub-quota (#1108) gates a task whose agent_config
    carries the server-stamped widget_client_ip, even with the entity quota
    wide open — one caller cannot drain the widget for everyone else.

    The refusal must be reported as the per-caller sub-quota, NOT the owner's
    budget: waiting clears this one, so the visitor gets copy they can act on
    instead of being told the widget owner is out of quota (D1/F1)."""
    monkeypatch.setenv("XAGENT_WIDGET_RUN_QUOTA", "500/day")
    monkeypatch.setenv("XAGENT_WIDGET_RUN_IP_QUOTA", "0/hour")
    reset_share_rate_limiter()

    task = _make_task(
        db_session,
        agent_config={
            "auth_mode": "widget",
            "guest_id": "rotatable-guest",
            "widget_agent_id": 4242,
            "widget_client_ip": "203.0.113.9",
        },
    )

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    assert result["success"] is False
    assert result["error_code"] == "widget_run_ip_quota_exceeded"
    assert "your network" in result["output"]
    # Not the owner-budget copy, which the visitor cannot act on.
    assert "widget has reached its usage limit" not in result["output"]


@pytest.mark.asyncio
async def test_widget_run_quota_blocks_workforce_widget_task(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workforce widget marker keys the quota too (entity_rate_limit_key
    prefers workforce over agent)."""
    monkeypatch.setenv("XAGENT_WIDGET_RUN_QUOTA", "0/day")
    reset_share_rate_limiter()

    task = _make_task(
        db_session,
        agent_config={
            "auth_mode": "widget",
            "guest_id": "rotatable-guest",
            "widget_workforce_id": 77,
            "widget_client_ip": "203.0.113.10",
        },
    )

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    assert result["success"] is False
    assert result["error_code"] == "widget_run_quota_exceeded"


@pytest.mark.asyncio
async def test_widget_run_quota_fails_open_on_malformed_marker(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed (non-integer) widget entity marker must admit the run rather
    than 500 or block: _coerce_optional_entity_id catches the coercion error
    and returns None, so the entity is unkeyable and _public_run_denial_channel takes
    its "unkeyable -> admit this task" branch — without falling through to the
    chokepoint's broad except."""
    monkeypatch.setenv("XAGENT_WIDGET_RUN_QUOTA", "0/day")
    reset_share_rate_limiter()

    task = _make_task(
        db_session,
        agent_config={
            "auth_mode": "widget",
            "guest_id": "g",
            "widget_agent_id": "not-an-int",
        },
    )

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    assert result.get("error_code") not in (
        "widget_run_quota_exceeded",
        "share_run_quota_exceeded",
    )
    # Positive assertion (F4): the run was actually admitted and executed —
    # this fails if the malformed-marker handling were removed and the run
    # blocked, which the negative error_code check alone would not catch.
    assert result["success"] is True


@pytest.mark.asyncio
async def test_widget_run_quota_admits_bool_entity_marker(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``bool`` widget entity marker must be rejected by
    _coerce_optional_entity_id (``isinstance(True, int)`` is True and would
    otherwise coerce to ``agent:1``), so the entity is unkeyable and the run is
    admitted rather than billed to a wrong bucket."""
    monkeypatch.setenv("XAGENT_WIDGET_RUN_QUOTA", "0/day")
    reset_share_rate_limiter()

    task = _make_task(
        db_session,
        agent_config={
            "auth_mode": "widget",
            "guest_id": "g",
            "widget_agent_id": True,
        },
    )

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    assert result["success"] is True
    assert result.get("error_code") != "widget_run_quota_exceeded"


@pytest.mark.asyncio
async def test_widget_run_quota_admits_when_entity_unkeyable(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A widget task with no entity marker cannot be attributed, so it falls
    through rather than being blocked (matches the share fall-through)."""
    monkeypatch.setenv("XAGENT_WIDGET_RUN_QUOTA", "0/day")
    reset_share_rate_limiter()

    task = _make_task(
        db_session,
        agent_config={"auth_mode": "widget", "guest_id": "g"},
    )

    result = await AgentServiceManager().execute_task(
        agent_service=_FakeAgentService(),
        task="hello",
        tracking_task_id=str(task.id),
        db_session=db_session,
        manage_task_lease=False,
    )

    # N15: assert the run actually ran, not merely that it dodged the (widget-
    # unreachable) share error code — otherwise a wrong widget_run_quota_exceeded
    # would still pass.
    assert result["success"] is True
    assert result.get("error_code") not in (
        "widget_run_quota_exceeded",
        "share_run_quota_exceeded",
    )


def test_coerce_optional_entity_id_rejects_non_positive_int_inputs() -> None:
    """N14: entity markers are positive DB primary keys, so floats (which would
    truncate onto a real entity's bucket), 0/negatives (impossible buckets),
    inf (whose int() raises OverflowError, escaping the narrow except), and
    junk all degrade to None ("unkeyable → admit") rather than keying a wrong
    bucket or crashing. Genuine positive ints and digit strings pass through."""
    from xagent.web.api.chat import _coerce_optional_entity_id

    assert _coerce_optional_entity_id(42) == 42
    assert _coerce_optional_entity_id("42") == 42
    # Rejected: float (no silent truncation onto agent:1), zero, negatives,
    # bool, inf/nan, signed or decimal strings, and outright junk.
    for bad in (
        1.9,
        0,
        -5,
        True,
        False,
        float("inf"),
        float("nan"),
        "1.9",
        "-5",
        "0",
        "",
        "abc",
        None,
        ["1"],
    ):
        assert _coerce_optional_entity_id(bad) is None, bad


@pytest.mark.asyncio
async def test_widget_run_quota_is_charged_per_turn_not_per_task(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the accounting semantics (D1/F1): the gate sits in execute_task,
    which runs once per conversation turn, so a second turn on the SAME task
    consumes another slot. With a 1/day entity quota the first turn is admitted
    and the second is refused — this is deliberate (the quota bounds
    owner-billed runs, and every turn is one), and the per-IP sub-quota is
    sized against it accordingly."""
    monkeypatch.setenv("XAGENT_WIDGET_RUN_QUOTA", "1/day")
    monkeypatch.setenv("XAGENT_WIDGET_RUN_IP_QUOTA", "100/hour")
    reset_share_rate_limiter()

    task = _make_task(
        db_session,
        agent_config={
            "auth_mode": "widget",
            "guest_id": "g",
            "widget_agent_id": 4242,
            "widget_client_ip": "203.0.113.11",
        },
    )

    async def _run_one_turn():
        return await AgentServiceManager().execute_task(
            agent_service=_FakeAgentService(),
            task="hello",
            tracking_task_id=str(task.id),
            db_session=db_session,
            manage_task_lease=False,
        )

    first = await _run_one_turn()
    assert first["success"] is True

    second = await _run_one_turn()
    assert second["success"] is False
    assert second["error_code"] == "widget_run_quota_exceeded"
