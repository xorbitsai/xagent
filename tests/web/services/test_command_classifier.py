"""The acceptance-time command classifier: one call per new execution command."""

import asyncio
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution_admission as admission
from xagent.web.services import task_resume_command

KINDS = transport.TaskCommandKind
EXECUTION = (KINDS.START, KINDS.MESSAGE, KINDS.RESUME, KINDS.RESUME_INPUT)
CONTROL = (KINDS.PAUSE, KINDS.CANCEL)


@pytest.fixture
def host(tmp_path, monkeypatch):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'classifier.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    with sessions() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.commit()
        user_id = user.id

    class Host:
        def __init__(self):
            self.sessions = sessions
            self.user = user_id

        def task(self):
            with sessions() as db:
                task = Task(user_id=user_id, title="Work", status=TaskStatus.PENDING)
                db.add(task)
                db.commit()
                return task.id

        def enqueue(self, kind=KINDS.START, *, task_id=None, command_id=None):
            task_id = task_id or self.task()
            with sessions() as db:
                return transport.enqueue_task_command(
                    db,
                    task_id=task_id,
                    actor_user_id=user_id,
                    command_id=command_id or f"{kind.value}-{task_id}",
                    kind=kind,
                    payload={},
                )

        def commands(self):
            with sessions() as db:
                return db.query(TaskExecutionCommand).count()

        def tickets(self):
            with sessions() as db:
                return [
                    t.bucket_key
                    for t in db.query(TaskAdmissionTicket).order_by("command_id")
                ]

    try:
        yield Host()
    finally:
        engine.dispose()


def policy(bucket="tenant:batch", capacity=1, max_pending=20):
    return admission.AdmissionPolicy(bucket, capacity, max_pending)


def counting(result=None):
    calls = []

    def classify(db, command):
        calls.append((command.task_id, command.kind))
        return result() if result else admission.CommandClassification()

    classify.calls = calls
    return classify


def test_default_classifier_stages_no_ticket(host):
    assert admission._installed.classifier is admission.upstream_command_classifier
    assert admission.admission_enabled() is False
    host.enqueue()
    assert host.tickets() == []
    with host.sessions() as db:
        row = db.query(TaskExecutionCommand).one()
        assert (row.kind, row.status) == ("start", "pending")


def test_classifier_called_once_per_execution_command(host):
    classify = counting()
    admission.set_command_classifier(classify, admission_enabled=False)
    for kind in EXECUTION:
        host.enqueue(kind)
    assert sorted(kind for _, kind in classify.calls) == sorted(
        kind.value for kind in EXECUTION
    )
    for kind in CONTROL:
        host.enqueue(kind)
    assert len(classify.calls) == len(EXECUTION)
    task_id = host.task()
    host.enqueue(KINDS.START, task_id=task_id, command_id="same")
    seen = len(classify.calls)
    replay = host.enqueue(KINDS.START, task_id=task_id, command_id="same")
    assert replay.created is False
    assert len(classify.calls) == seen


def test_admission_refusals_still_propagate(host):
    admission.set_command_classifier(
        lambda db, command: admission.CommandClassification(
            admission=policy(max_pending=1)
        ),
        admission_enabled=True,
    )
    host.enqueue(command_id="first")
    assert host.commands() == 1
    with pytest.raises(admission.AdmissionQueueFull):
        host.enqueue(command_id="second")
    assert host.commands() == 1

    class Refused(Exception):
        pass

    def refuse(db, command):
        raise Refused

    admission.set_command_classifier(refuse, admission_enabled=True)
    with pytest.raises(Refused):
        host.enqueue(command_id="third")
    assert host.commands() == 1


@pytest.mark.parametrize(
    "install, expected",
    [
        pytest.param(lambda: None, False, id="unregistered"),
        pytest.param(
            lambda: admission.set_command_classifier(
                lambda db, command: admission.CommandClassification(),
                admission_enabled=False,
            ),
            False,
            id="declared-false",
        ),
        pytest.param(
            lambda: admission.set_command_classifier(
                lambda db, command: admission.CommandClassification(),
                admission_enabled=True,
            ),
            True,
            id="declared-true",
        ),
        pytest.param(
            lambda: admission.set_task_admission_hook(lambda db, command: None),
            True,
            id="legacy-hook",
        ),
        pytest.param(
            lambda: (
                admission.set_task_admission_hook(lambda db, command: None),
                admission.set_task_admission_hook(None),
            ),
            False,
            id="legacy-hook-cleared",
        ),
    ],
)
def test_admission_enabled_reflects_declaration(host, monkeypatch, install, expected):
    install()
    assert admission.admission_enabled() is expected

    # waiting_for_capacity: with every command treated as blocked, only a
    # declared admission host reports the wait.
    monkeypatch.setattr(admission, "admission_eligible", lambda: sa.false())
    command_id = host.enqueue().command_id
    assert host.tickets() == []
    with host.sessions() as db:
        assert admission.waiting_for_capacity(db, command_id) is expected

    # enqueue_resume_input acknowledges a queued reply under the same value.
    async def passthrough(call):
        return call()

    monkeypatch.setattr(task_resume_command, "run_db_io_cancellation_safe", passthrough)
    monkeypatch.setattr(task_resume_command, "_admit_reply", lambda *a: 1)
    monkeypatch.setattr(task_resume_command, "_read_reply_outcome", lambda _id: None)
    capacity_wait = Mock(
        return_value={"run_id": "r", "state_version": 1, "control_state": "waiting"}
    )
    monkeypatch.setattr(task_resume_command, "_read_capacity_wait", capacity_wait)
    monkeypatch.setattr(task_resume_command, "notify_task_command_dispatcher", Mock())
    monkeypatch.setattr(
        task_resume_command, "get_task_reply_wait_timeout_seconds", lambda: 0
    )
    from xagent.web.services import task_event_bridge

    monkeypatch.setattr(task_event_bridge, "_bridge", Mock())
    reply = Mock(command_id="c1")

    async def submit():
        return await task_resume_command.enqueue_resume_input(
            reply, source="sdk", message_id="m"
        )

    if expected:
        assert asyncio.run(submit()).queued is True
    else:
        with pytest.raises(task_resume_command.TaskResumeOutcomeUnknownError):
            asyncio.run(submit())
    # The acknowledge_queued gate: the capacity wait is read only when the
    # installed classifier declared admission.
    assert capacity_wait.called is expected


def test_classifier_sees_flushed_command(host):
    seen = []

    def classify(db, command):
        seen.append((command.id is not None, command.kind, command.task_id))
        return admission.CommandClassification()

    admission.set_command_classifier(classify, admission_enabled=False)
    task_id = host.task()
    host.enqueue(KINDS.MESSAGE, task_id=task_id)
    assert seen == [(True, "message", task_id)]


def test_classifier_result_shape_is_enforced(host):
    admission.set_command_classifier(
        lambda db, command: policy(), admission_enabled=True
    )
    with pytest.raises(TypeError, match="must return CommandClassification"):
        host.enqueue(command_id="legacy-shape")
    assert host.commands() == 0

    admission.set_command_classifier(
        lambda db, command: admission.CommandClassification(admission=policy()),
        admission_enabled=False,
    )
    with pytest.raises(TypeError, match="admission_enabled=False"):
        host.enqueue(command_id="undeclared")
    assert host.commands() == 0

    with pytest.raises(TypeError):
        admission.CommandClassification(policy())  # type: ignore[misc]


def test_legacy_hook_wrapper_matches_old_behavior(host):
    received = []

    def hook(db, command):
        received.append((db, command))
        return policy("legacy:bucket")

    admission.set_task_admission_hook(hook)
    assert admission.admission_enabled() is True
    described = admission._describe(admission._installed.classifier)
    assert hook.__qualname__ in described
    assert "legacy_admission_hook" not in described
    host.enqueue()
    assert host.tickets() == ["legacy:bucket"]
    assert len(received) == 1
    db, command = received[0]
    assert isinstance(db, sa.orm.Session)
    assert isinstance(command, TaskExecutionCommand)


def test_classifier_reset_fixture_isolates_tests_first():
    admission.set_command_classifier(counting(), admission_enabled=True)
    assert admission.admission_enabled() is True


def test_classifier_reset_fixture_isolates_tests():
    assert admission._installed.classifier is admission.upstream_command_classifier
    assert admission.admission_enabled() is False


@pytest.mark.parametrize("bad", [None, "classifier"])
def test_set_command_classifier_rejects_non_callable(bad):
    with pytest.raises(TypeError):
        admission.set_command_classifier(bad, admission_enabled=False)
    assert admission._installed.classifier is admission.upstream_command_classifier


@pytest.mark.parametrize("bad", ["false", 1, None])
def test_set_command_classifier_rejects_non_bool_declaration(bad):
    before = admission._installed
    with pytest.raises(TypeError, match="admission_enabled must be a bool"):
        admission.set_command_classifier(counting(), admission_enabled=bad)
    assert admission._installed is before
    assert admission._installed.classifier is admission.upstream_command_classifier


def test_legacy_clear_leaves_a_new_style_classifier_installed():
    classify = counting()
    admission.set_command_classifier(classify, admission_enabled=True)
    admission.set_task_admission_hook(None)
    assert admission._installed.classifier is classify
    assert admission.admission_enabled() is True

    admission.set_task_admission_hook(lambda db, command: None)
    admission.set_task_admission_hook(None)
    assert admission._installed.classifier is admission.upstream_command_classifier
