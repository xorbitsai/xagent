"""Auto-resume policy: backoff, jitter, plan ordering and trigger supersession."""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone

import pytest

from tests.shared.db_teardown import drop_all_tables
from xagent.core.agent.interruption import InterruptionReason
from xagent.web.models.agent import Agent
from xagent.web.models.database import get_db, get_engine, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_auto_recovery import (
    TaskAutoRecovery,
    TaskAutoRecoveryState,
)
from xagent.web.models.trigger import (
    TEST_TRIGGER_RUN_KEY_PREFIX,
    AgentTrigger,
    TriggerRun,
    TriggerRunStatus,
    TriggerType,
)
from xagent.web.models.user import User
from xagent.web.services.task_auto_recovery_policy import (
    AutoResumeOutcome,
    auto_resume_limit,
    auto_resume_policy,
    auto_resume_recovery_state,
    backoff_seconds,
    jitter_seconds,
    next_auto_resume_at,
    plan_auto_resume,
    scheduled_trigger_superseded,
    staleness_window_seconds,
)

R = InterruptionReason


@pytest.fixture(autouse=True)
def _clean_auto_resume_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("XAGENT_TASK_AUTO_RESUME_"):
            monkeypatch.delenv(key)


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class _Bound(random.Random):
    """rng whose ``uniform`` returns the lower or upper bound."""

    def __init__(self, upper: bool) -> None:
        super().__init__(0)
        self._upper = upper

    def uniform(self, a: float, b: float) -> float:
        return b if self._upper else a


LO = _Bound(False)
HI = _Bound(True)


def _policy(reason):
    policy = auto_resume_policy(reason)
    assert policy is not None
    return policy


def _plan(
    reason=R.LEASE_EXPIRED,
    kind="normal",
    interrupted_ago=0,
    episode_ago=0,
    no_progress=0,
    total=0,
    rng=LO,
):
    return plan_auto_resume(
        reason=reason,
        kind=kind,
        interrupted_at=NOW - timedelta(seconds=interrupted_ago),
        episode_started_at=NOW - timedelta(seconds=episode_ago),
        no_progress_resumes=no_progress,
        total_resumes=total,
        now=NOW,
        rng=rng,
    )


# --- policy / backoff -------------------------------------------------------


@pytest.mark.parametrize("reason", [R.LEASE_EXPIRED, R.PERSISTENCE_FAILURE])
def test_short_backoff_sequence(reason) -> None:
    policy = _policy(reason)
    seq = [backoff_seconds(policy, n) for n in range(5)]
    assert seq == [10, 20, 40, 60, 60]
    assert policy.max_no_progress == 3
    assert policy.max_elapsed_seconds is None
    assert policy.jitter_cap_seconds == 30


def test_llm_backoff_sequence_caps_at_max() -> None:
    policy = _policy(R.LLM_UNAVAILABLE)
    seq = [backoff_seconds(policy, n) for n in range(7)]
    assert seq == [60, 120, 240, 480, 900, 900, 900]
    assert policy.max_no_progress is None
    assert policy.max_elapsed_seconds == 7200
    assert policy.jitter_cap_seconds == 30


def test_model_output_invalid_is_immediate() -> None:
    policy = _policy(R.MODEL_OUTPUT_INVALID)
    assert [backoff_seconds(policy, n) for n in range(4)] == [0, 0, 0, 0]
    assert policy.max_no_progress == 2
    assert policy.jitter_cap_seconds == 5


@pytest.mark.parametrize(
    "reason", [R.LEASE_EXPIRED, R.LLM_UNAVAILABLE, R.MODEL_OUTPUT_INVALID]
)
def test_backoff_overflow_safe(reason) -> None:
    policy = _policy(reason)
    assert backoff_seconds(policy, 10_000) == policy.max_backoff_seconds


@pytest.mark.parametrize(
    "reason",
    [
        R.SHUTDOWN,
        R.USER_PAUSE,
        R.UNKNOWN_TOOL_EFFECT,
        R.NOT_RECOVERABLE,
        R.INPUT_OUTCOME_UNKNOWN,
    ],
)
def test_never_reasons(reason) -> None:
    assert auto_resume_policy(reason) is None
    assert _plan(reason=reason).outcome is AutoResumeOutcome.NEVER


def test_policy_reads_env_at_call_time(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS", "5")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_SHORT_MAX_NO_PROGRESS", "7")
    policy = _policy(R.LEASE_EXPIRED)
    assert policy.initial_backoff_seconds == 5
    assert policy.max_no_progress == 7
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_LLM_MAX_BACKOFF_SECONDS", "100")
    assert _policy(R.LLM_UNAVAILABLE).max_backoff_seconds == 100


# --- jitter -----------------------------------------------------------------


def test_jitter_capped_at_half_the_cap_for_large_backoff() -> None:
    policy = _policy(R.LLM_UNAVAILABLE)
    assert jitter_seconds(policy, 900, LO) == 0
    assert jitter_seconds(policy, 900, HI) == 15


def test_jitter_half_of_small_backoff() -> None:
    policy = _policy(R.LEASE_EXPIRED)
    assert jitter_seconds(policy, 10, HI) == 5
    assert jitter_seconds(policy, 10, LO) == 0


def test_jitter_immediate_policy_spans_cap() -> None:
    policy = _policy(R.MODEL_OUTPUT_INVALID)
    assert jitter_seconds(policy, 0, LO) == 0
    assert jitter_seconds(policy, 0, HI) == 5


def test_jitter_uses_injected_rng_not_global() -> None:
    policy = _policy(R.LEASE_EXPIRED)
    random.seed(1)
    a = jitter_seconds(policy, 20, random.Random(42))
    random.seed(2)
    b = jitter_seconds(policy, 20, random.Random(42))
    assert a == b
    assert 0 <= a <= 10


# --- staleness window -------------------------------------------------------


def test_staleness_windows(monkeypatch) -> None:
    assert staleness_window_seconds("channel") == 1800
    assert staleness_window_seconds("normal") == 86400
    assert staleness_window_seconds(None) == 86400
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_CHANNEL_WINDOW_SECONDS", "60")
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_WINDOW_SECONDS", "120")
    assert staleness_window_seconds("channel") == 60
    assert staleness_window_seconds("workforce") == 120


# --- plan_auto_resume -------------------------------------------------------


def test_schedule_with_backoff_and_jitter() -> None:
    v = _plan(no_progress=1, rng=HI)
    assert v.outcome is AutoResumeOutcome.SCHEDULED
    # backoff 20 + jitter min(20, 30) * 0.5
    assert v.next_attempt_at == NOW + timedelta(seconds=30)
    assert _plan(no_progress=1, rng=LO).next_attempt_at == NOW + timedelta(seconds=20)


def test_schedule_model_output_invalid_within_five_seconds() -> None:
    v = _plan(reason=R.MODEL_OUTPUT_INVALID, rng=HI)
    assert v.next_attempt_at == NOW + timedelta(seconds=5)


def test_non_schedule_verdicts_have_no_next_attempt() -> None:
    assert _plan(reason=R.USER_PAUSE).next_attempt_at is None
    assert _plan(total=20).next_attempt_at is None


def test_never_beats_everything() -> None:
    v = _plan(
        reason=R.USER_PAUSE,
        interrupted_ago=10**7,
        total=100,
        no_progress=100,
        episode_ago=10**7,
    )
    assert v.outcome is AutoResumeOutcome.NEVER


def test_expired_beats_exhausted() -> None:
    v = _plan(interrupted_ago=86401, total=20, no_progress=3)
    assert v.outcome is AutoResumeOutcome.EXPIRED


def test_staleness_boundary_is_inclusive_of_equal() -> None:
    assert _plan(interrupted_ago=86400).outcome is AutoResumeOutcome.SCHEDULED
    assert _plan(interrupted_ago=86401).outcome is AutoResumeOutcome.EXPIRED


def test_channel_window_shorter_than_normal() -> None:
    assert _plan(kind="channel", interrupted_ago=1800).outcome is (
        AutoResumeOutcome.SCHEDULED
    )
    assert _plan(kind="channel", interrupted_ago=1801).outcome is (
        AutoResumeOutcome.EXPIRED
    )
    assert _plan(kind="normal", interrupted_ago=1801).outcome is (
        AutoResumeOutcome.SCHEDULED
    )


def test_total_cap_boundary() -> None:
    assert _plan(total=19).outcome is AutoResumeOutcome.SCHEDULED
    assert _plan(total=20).outcome is AutoResumeOutcome.EXHAUSTED
    assert _plan(total=21).outcome is AutoResumeOutcome.EXHAUSTED


def test_total_cap_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_MAX_TOTAL_PER_RUN", "2")
    assert _plan(total=1).outcome is AutoResumeOutcome.SCHEDULED
    assert _plan(total=2).outcome is AutoResumeOutcome.EXHAUSTED


def test_no_progress_boundary() -> None:
    assert _plan(no_progress=2).outcome is AutoResumeOutcome.SCHEDULED
    assert _plan(no_progress=3).outcome is AutoResumeOutcome.EXHAUSTED
    mo = R.MODEL_OUTPUT_INVALID
    assert _plan(reason=mo, no_progress=1).outcome is AutoResumeOutcome.SCHEDULED
    assert _plan(reason=mo, no_progress=2).outcome is AutoResumeOutcome.EXHAUSTED


def test_llm_has_no_no_progress_limit() -> None:
    v = _plan(reason=R.LLM_UNAVAILABLE, no_progress=500)
    assert v.outcome is AutoResumeOutcome.SCHEDULED


def test_elapsed_boundary() -> None:
    llm = R.LLM_UNAVAILABLE
    assert _plan(reason=llm, episode_ago=7199).outcome is AutoResumeOutcome.SCHEDULED
    assert _plan(reason=llm, episode_ago=7200).outcome is AutoResumeOutcome.EXHAUSTED
    # Short reasons have no elapsed limit.
    assert _plan(episode_ago=10**6).outcome is AutoResumeOutcome.SCHEDULED


def test_elapsed_limit_respects_env(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_LLM_MAX_ELAPSED_SECONDS", "100")
    v = _plan(reason=R.LLM_UNAVAILABLE, episode_ago=100)
    assert v.outcome is AutoResumeOutcome.EXHAUSTED


def test_schedule_even_when_next_attempt_lands_past_limits() -> None:
    # Interrupted just inside the window; the attempt falls beyond it.
    v = _plan(interrupted_ago=86399, rng=HI)
    assert v.outcome is AutoResumeOutcome.SCHEDULED
    # Episode 1s short of max elapsed; backoff of 60s overshoots it.
    v = _plan(reason=R.LLM_UNAVAILABLE, episode_ago=7199)
    assert v.outcome is AutoResumeOutcome.SCHEDULED
    assert v.next_attempt_at is not None
    assert v.next_attempt_at - NOW >= timedelta(seconds=60)


def test_total_cap_checked_before_no_progress_and_elapsed() -> None:
    v = _plan(reason=R.LLM_UNAVAILABLE, total=20, episode_ago=10**6)
    assert v.outcome is AutoResumeOutcome.EXHAUSTED


@pytest.mark.parametrize("field", ["interrupted_at", "episode_started_at", "now"])
def test_naive_datetime_read_as_utc(field) -> None:
    kwargs = dict(
        reason=R.LLM_UNAVAILABLE,
        kind="normal",
        interrupted_at=NOW,
        episode_started_at=NOW - timedelta(seconds=7199),
        no_progress_resumes=0,
        total_resumes=0,
        now=NOW,
        rng=LO,
    )
    aware = plan_auto_resume(**kwargs)  # type: ignore[arg-type]
    kwargs[field] = kwargs[field].replace(tzinfo=None)
    naive = plan_auto_resume(**kwargs)  # type: ignore[arg-type]
    assert naive == aware
    assert naive.next_attempt_at is not None
    assert naive.next_attempt_at.tzinfo is not None


def test_mixed_naive_and_aware_and_offset_inputs() -> None:
    plus8 = timezone(timedelta(hours=8))
    # 20:00+08:00 is 12:00 UTC == NOW; naive 11:59 is read as UTC.
    verdict = plan_auto_resume(
        reason=R.LEASE_EXPIRED,
        kind="channel",
        interrupted_at=(NOW - timedelta(seconds=1800)).replace(tzinfo=None),
        episode_started_at=NOW.astimezone(plus8),
        no_progress_resumes=0,
        total_resumes=0,
        now=NOW.astimezone(plus8),
        rng=LO,
    )
    assert verdict.outcome is AutoResumeOutcome.SCHEDULED
    assert verdict.next_attempt_at == NOW + timedelta(seconds=10)
    expired = auto_resume_limit(
        reason=R.LEASE_EXPIRED,
        kind="channel",
        interrupted_at=(NOW - timedelta(seconds=1801)).replace(tzinfo=None),
        episode_started_at=NOW,
        no_progress_resumes=0,
        total_resumes=0,
        now=NOW.astimezone(plus8),
    )
    assert expired is AutoResumeOutcome.EXPIRED


def test_row_round_trip_on_sqlite(world) -> None:
    db = world.db
    task = world.task()
    db.add(
        TaskAutoRecovery(
            task_id=task.id,
            run_id="r1",
            reason=R.LLM_UNAVAILABLE.value,
            state=TaskAutoRecoveryState.MANUAL.value,
            paused_state_version=1,
            interrupted_at=NOW,
            episode_started_at=NOW - timedelta(seconds=7200),
            no_progress_resumes=0,
            total_resumes=0,
        )
    )
    db.commit()
    db.expire_all()
    row = db.query(TaskAutoRecovery).filter_by(task_id=task.id).one()
    kwargs = dict(
        reason=row.reason,
        kind="normal",
        interrupted_at=row.interrupted_at,
        episode_started_at=row.episode_started_at,
        no_progress_resumes=row.no_progress_resumes,
        total_resumes=row.total_resumes,
        now=NOW,
    )
    # Elapsed exactly at the limit, so a wrong tz read would change the verdict.
    assert auto_resume_limit(**kwargs) is AutoResumeOutcome.EXHAUSTED
    verdict = plan_auto_resume(rng=LO, **kwargs)
    assert verdict.outcome is AutoResumeOutcome.EXHAUSTED


@pytest.mark.parametrize("reason", list(R))
def test_plain_string_reason_matches_enum(reason) -> None:
    assert auto_resume_policy(reason.value) == auto_resume_policy(reason)
    assert _plan(reason=reason.value) == _plan(reason=reason)


def test_unknown_string_reason_is_never() -> None:
    assert auto_resume_policy("bogus") is None
    assert _plan(reason="bogus").outcome is AutoResumeOutcome.NEVER


def test_short_ceiling_never_below_configured_initial(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_TASK_AUTO_RESUME_SHORT_INITIAL_BACKOFF_SECONDS", "120")
    policy = _policy(R.LEASE_EXPIRED)
    assert policy.max_backoff_seconds == 120
    assert [backoff_seconds(policy, n) for n in range(3)] == [120, 120, 120]


def test_negative_attempt_treated_as_zero() -> None:
    policy = _policy(R.LEASE_EXPIRED)
    assert backoff_seconds(policy, -5) == 10


def test_next_auto_resume_at_accepts_reason_or_policy() -> None:
    policy = _policy(R.LEASE_EXPIRED)
    expected = NOW + timedelta(seconds=40 + 15)
    assert next_auto_resume_at(policy, 2, NOW, HI) == expected
    assert next_auto_resume_at(R.LEASE_EXPIRED, 2, NOW, HI) == expected
    assert next_auto_resume_at("lease_expired", 2, NOW, LO) == NOW + timedelta(
        seconds=40
    )
    with pytest.raises(ValueError):
        next_auto_resume_at(R.USER_PAUSE, 0, NOW, LO)


def test_limit_returns_none_when_allowed_and_draws_no_rng() -> None:
    assert (
        auto_resume_limit(
            reason=R.LEASE_EXPIRED,
            kind="normal",
            interrupted_at=NOW,
            episode_started_at=NOW,
            no_progress_resumes=0,
            total_resumes=0,
            now=NOW,
        )
        is None
    )


@pytest.mark.parametrize(
    ("outcome", "state"),
    [
        (AutoResumeOutcome.SCHEDULED, TaskAutoRecoveryState.SCHEDULED),
        (AutoResumeOutcome.EXHAUSTED, TaskAutoRecoveryState.EXHAUSTED),
        (AutoResumeOutcome.EXPIRED, TaskAutoRecoveryState.EXPIRED),
        (AutoResumeOutcome.NEVER, TaskAutoRecoveryState.MANUAL),
    ],
)
def test_outcome_maps_to_recovery_state(outcome, state) -> None:
    assert auto_resume_recovery_state(outcome) is state


# --- scheduled_trigger_superseded ------------------------------------------


@pytest.fixture()
def db_session(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'policy.db'}")
    db = next(get_db())
    try:
        yield db
    finally:
        db.close()
        drop_all_tables(get_engine())


class _World:
    def __init__(self, db) -> None:
        self.db = db
        user = User(username="policy-user", password_hash="hash", is_admin=False)
        db.add(user)
        db.flush()
        agent = Agent(user_id=user.id, name="policy agent")
        db.add(agent)
        db.flush()
        self.user, self.agent = user, agent
        self._n = 0

    def trigger(self, type_: str = TriggerType.SCHEDULED.value) -> AgentTrigger:
        trigger = AgentTrigger(
            user_id=self.user.id,
            agent_id=self.agent.id,
            type=type_,
            name=f"{type_} trigger",
            config={},
        )
        self.db.add(trigger)
        self.db.flush()
        return trigger

    def task(
        self,
        *,
        source="trigger",
        trigger_type="scheduled",
        test=False,
        status=TaskStatus.PAUSED,
    ) -> Task:
        task = Task(
            user_id=self.user.id,
            title="t",
            description="t",
            status=status,
            execution_mode="auto",
            source=source,
            agent_config=(
                {
                    "trigger_type": trigger_type,
                    **({"trigger_test": True} if test else {}),
                }
                if trigger_type
                else None
            ),
        )
        self.db.add(task)
        self.db.flush()
        return task

    def run(
        self,
        trigger,
        task=None,
        *,
        started=True,
        status=TriggerRunStatus.RUNNING.value,
        test=False,
    ) -> TriggerRun:
        self._n += 1
        run = TriggerRun(
            trigger_id=trigger.id,
            task_id=task.id if task is not None else None,
            status=status,
            idempotency_key=(
                f"{TEST_TRIGGER_RUN_KEY_PREFIX}{trigger.id}:{self._n}"
                if test
                else f"policy-{self._n}"
            ),
            started_at=NOW if started else None,
        )
        self.db.add(run)
        self.db.flush()
        return run


@pytest.fixture()
def world(db_session) -> _World:
    return _World(db_session)


def test_superseded_by_later_started_run(world) -> None:
    trigger = world.trigger()
    task = world.task()
    world.run(trigger, task)
    world.run(trigger, world.task())
    assert scheduled_trigger_superseded(world.db, task) is True


def test_not_superseded_by_later_unstarted_run(world) -> None:
    trigger = world.trigger()
    task = world.task()
    world.run(trigger, task)
    world.run(trigger, world.task(), started=False)
    assert scheduled_trigger_superseded(world.db, task) is False


def test_not_superseded_when_only_earlier_runs(world) -> None:
    trigger = world.trigger()
    world.run(trigger, world.task())
    task = world.task()
    world.run(trigger, task)
    assert scheduled_trigger_superseded(world.db, task) is False


def test_latest_run_by_id_is_used_when_task_has_several(world) -> None:
    trigger = world.trigger()
    task = world.task()
    world.run(trigger, task)
    world.run(trigger, task)  # the newest run of this task
    assert scheduled_trigger_superseded(world.db, task) is False
    world.run(trigger, world.task())
    assert scheduled_trigger_superseded(world.db, task) is True


def test_webhook_trigger_never_superseded(world) -> None:
    trigger = world.trigger(TriggerType.WEBHOOK.value)
    task = world.task(trigger_type="webhook")
    world.run(trigger, task)
    world.run(trigger, world.task(trigger_type="webhook"))
    assert scheduled_trigger_superseded(world.db, task) is False


def test_gmail_trigger_never_superseded(world) -> None:
    trigger = world.trigger(TriggerType.GMAIL.value)
    task = world.task(trigger_type="gmail")
    world.run(trigger, task)
    world.run(trigger, world.task(trigger_type="gmail"))
    assert scheduled_trigger_superseded(world.db, task) is False


def test_missing_trigger_run_row(world) -> None:
    assert scheduled_trigger_superseded(world.db, world.task()) is False


def test_non_trigger_task(world) -> None:
    trigger = world.trigger()
    task = world.task(source="internal")
    world.run(trigger, task)
    world.run(trigger, world.task())
    assert scheduled_trigger_superseded(world.db, task) is False


@pytest.mark.parametrize("config", [None, "oops", ["scheduled"], {"trigger_type": 3}])
def test_malformed_agent_config(world, config) -> None:
    trigger = world.trigger()
    task = world.task()
    world.run(trigger, task)
    world.run(trigger, world.task())
    task.agent_config = config
    assert scheduled_trigger_superseded(world.db, task) is False


def test_different_triggers_later_run_does_not_supersede(world) -> None:
    task = world.task()
    world.run(world.trigger(), task)
    world.run(world.trigger(), world.task())
    assert scheduled_trigger_superseded(world.db, task) is False


def test_later_completed_run_supersedes(world) -> None:
    trigger = world.trigger()
    task = world.task()
    world.run(trigger, task)
    world.run(trigger, world.task(), status=TriggerRunStatus.COMPLETED.value)
    assert scheduled_trigger_superseded(world.db, task) is True


@pytest.mark.parametrize(
    "later_task_status", [TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER]
)
def test_later_run_waiting_on_a_person_still_supersedes(
    world, later_task_status
) -> None:
    """Only the latest tick matters: a later tick its user paused, or that
    waits for an answer, keeps its run ``running``, and resuming the earlier
    tick would still duplicate it."""
    trigger = world.trigger()
    task = world.task()
    world.run(trigger, task)
    world.run(trigger, world.task(status=later_task_status))
    assert scheduled_trigger_superseded(world.db, task) is True


def test_later_failed_run_does_not_supersede(world) -> None:
    trigger = world.trigger()
    task = world.task()
    world.run(trigger, task)
    world.run(trigger, world.task(), status=TriggerRunStatus.FAILED.value)
    assert scheduled_trigger_superseded(world.db, task) is False


def test_later_test_fire_does_not_supersede(world) -> None:
    trigger = world.trigger()
    task = world.task()
    world.run(trigger, task)
    world.run(trigger, world.task(test=True), test=True)
    assert scheduled_trigger_superseded(world.db, task) is False


def test_paused_test_fire_task_is_never_superseded(world) -> None:
    trigger = world.trigger()
    task = world.task(test=True)
    world.run(trigger, task, test=True)
    world.run(trigger, world.task())
    assert scheduled_trigger_superseded(world.db, task) is False


def test_test_key_alone_marks_own_run_as_test(world) -> None:
    trigger = world.trigger()
    task = world.task()  # config carries no trigger_test flag
    world.run(trigger, task, test=True)
    world.run(trigger, world.task())
    assert scheduled_trigger_superseded(world.db, task) is False


def test_task_flagged_trigger_test_is_never_superseded(world) -> None:
    trigger = world.trigger()
    task = world.task(test=True)
    world.run(trigger, task)  # key not marked; the task config is
    world.run(trigger, world.task())
    assert scheduled_trigger_superseded(world.db, task) is False
