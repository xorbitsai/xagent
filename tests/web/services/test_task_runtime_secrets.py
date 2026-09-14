"""Single-turn runtime inputs remain encrypted, scoped and transactional."""

import pytest
from cryptography.fernet import Fernet, InvalidToken

from xagent.core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ERROR_RUNTIME_SECRET_UNAVAILABLE,
    ConnectorRef,
    ConnectorRuntimeError,
)
from xagent.web.models.database import Base, get_engine, get_session_local, init_db
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_runtime_secret import TaskRuntimeSecret
from xagent.web.models.user import User
from xagent.web.services.task_runtime_secrets import (
    bind_runtime_values_to_run,
    clean_finished_runtime_values,
    load_runtime_values,
    stage_runtime_values,
)

VALUES = {
    ConnectorRef("mcp", 1): {
        "secrets": {"token": "synthetic-secret"},
        "auth_selector": {"account": "synthetic-account"},
    }
}


@pytest.fixture(autouse=True)
def private_encryption_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("ENCRYPTION_KEY", key)
    return key


@pytest.fixture
def task_id(tmp_path):
    init_db(db_url=f"sqlite:///{tmp_path / 'secrets.db'}")
    with get_session_local()() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.flush()
        task = Task(
            user_id=user.id,
            title="Inputs",
            status=TaskStatus.RUNNING,
            run_id="run-1",
            state_version=1,
            control_state="running",
        )
        db.add(task)
        db.commit()
        yield task.id
    Base.metadata.drop_all(bind=get_engine())


def stage(db, task_id):
    stage_runtime_values(db, task_id=task_id, turn_id="turn-1", values_by_ref=VALUES)
    assert bind_runtime_values_to_run(
        db, task_id=task_id, turn_id="turn-1", run_id="run-1"
    )


def test_ciphertext_round_trip_in_independent_session(task_id):
    with get_session_local()() as db:
        stage(db, task_id)
        row = db.query(TaskRuntimeSecret).one()
        assert "synthetic" not in row.ciphertext
        db.commit()
    with get_session_local()() as db:
        task = db.get(Task, task_id)
        assert load_runtime_values(db, task=task, turn_id="turn-1", required=True) == {
            ref.storage_key: values for ref, values in VALUES.items()
        }
        assert load_runtime_values(db, task=task, turn_id="turn-1", required=True)


def test_acceptance_rollback_removes_runtime_inputs(task_id):
    with get_session_local()() as db:
        stage(db, task_id)
        db.rollback()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 0


@pytest.mark.parametrize("change", ["owner", "run", "ciphertext", "turn"])
def test_missing_or_mismatched_values_fail_closed(task_id, change):
    with get_session_local()() as db:
        stage(db, task_id)
        row = db.query(TaskRuntimeSecret).one()
        if change == "owner":
            db.get(
                User, db.get(Task, task_id).user_id
            ).actor_subject = "replacement-owner"
        elif change == "run":
            db.get(Task, task_id).run_id = "run-2"
        elif change == "ciphertext":
            row.ciphertext = "synthetic-plaintext"
        db.commit()
    with get_session_local()() as db:
        with pytest.raises(ConnectorRuntimeError) as error:
            load_runtime_values(
                db,
                task=db.get(Task, task_id),
                turn_id="other-turn" if change == "turn" else "turn-1",
                required=True,
            )
        assert "synthetic" not in str(error.value)


def test_compensation_preserves_queued_and_active_inputs(task_id):
    with get_session_local()() as db:
        stage(db, task_id)
        db.commit()
    clean_finished_runtime_values()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 1
        db.get(Task, task_id).status = TaskStatus.PAUSED
        db.commit()
    clean_finished_runtime_values()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 0


@pytest.mark.parametrize(
    "resting_status", [TaskStatus.PAUSED, TaskStatus.WAITING_FOR_USER]
)
def test_resumed_execution_requires_fresh_values_after_lease_release(
    task_id, resting_status
):
    with get_session_local()() as db:
        stage(db, task_id)
        task = db.get(Task, task_id)
        task.status = resting_status
        task.runner_id = "worker"
        db.commit()
    clean_finished_runtime_values()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 1
        db.get(Task, task_id).runner_id = None
        db.commit()
    clean_finished_runtime_values()
    with get_session_local()() as db:
        assert db.query(TaskRuntimeSecret).count() == 0
        task = db.get(Task, task_id)
        task.status = TaskStatus.RUNNING
        task.runner_id = "resumed-worker"
        with pytest.raises(ConnectorRuntimeError) as error:
            load_runtime_values(db, task=task, turn_id="turn-1", required=True)
        assert error.value.code == ERROR_RUNTIME_SECRET_UNAVAILABLE
        stage(db, task_id)
        assert load_runtime_values(db, task=task, turn_id="turn-1", required=True)


def test_cleanup_cannot_observe_inputs_before_transaction_binds_run(task_id):
    with get_session_local()() as acceptance:
        stage_runtime_values(
            acceptance, task_id=task_id, turn_id="turn-1", values_by_ref=VALUES
        )
        assert acceptance.query(TaskRuntimeSecret).one().run_id is None
        with get_session_local()() as observer:
            assert observer.query(TaskRuntimeSecret).count() == 0
        # Cleanup runs through its own session while acceptance is uncommitted.
        clean_finished_runtime_values()
        assert bind_runtime_values_to_run(
            acceptance, task_id=task_id, turn_id="turn-1", run_id="run-1"
        )
        acceptance.commit()
    clean_finished_runtime_values()
    with get_session_local()() as observer:
        row = observer.query(TaskRuntimeSecret).one()
        assert row.run_id == "run-1"
        assert load_runtime_values(
            observer, task=observer.get(Task, task_id), turn_id="turn-1", required=True
        )


@pytest.mark.parametrize(
    "key",
    [None, "", "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc=", "invalid-key", "非密钥"],
)
def test_invalid_key_refuses_staging_without_writing(task_id, monkeypatch, key):
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    if key is not None:
        monkeypatch.setenv("ENCRYPTION_KEY", key)
    with get_session_local()() as db:
        with pytest.raises(ConnectorRuntimeError) as error:
            stage(db, task_id)
        assert error.value.code == ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
        assert error.value.status_code == 503
        assert "synthetic-secret" not in str(error.value)
        db.commit()
    with get_session_local()() as observer:
        assert observer.query(TaskRuntimeSecret).count() == 0


def test_store_does_not_reuse_cached_development_cipher(
    task_id, monkeypatch, private_encryption_key
):
    from xagent.core.utils.encryption import get_cipher

    get_cipher.cache_clear()
    try:
        monkeypatch.delenv("ENCRYPTION_KEY")
        development_cipher = get_cipher()
        monkeypatch.setenv("ENCRYPTION_KEY", private_encryption_key)
        with get_session_local()() as db:
            stage(db, task_id)
            ciphertext = db.query(TaskRuntimeSecret).one().ciphertext.encode()
            assert b"synthetic-secret" in Fernet(
                private_encryption_key.encode()
            ).decrypt(ciphertext)
            with pytest.raises(InvalidToken):
                development_cipher.decrypt(ciphertext)
            assert load_runtime_values(
                db, task=db.get(Task, task_id), turn_id="turn-1", required=True
            )
    finally:
        get_cipher.cache_clear()


@pytest.mark.parametrize(
    "key_state", ["unset", "empty", "default", "malformed", "rotated"]
)
def test_read_distinguishes_key_configuration_from_decryption_failure(
    task_id, monkeypatch, key_state
):
    from xagent.config import DEV_FALLBACK_ENCRYPTION_KEY

    with get_session_local()() as db:
        stage(db, task_id)
        db.commit()
    if key_state == "unset":
        monkeypatch.delenv("ENCRYPTION_KEY")
    else:
        monkeypatch.setenv(
            "ENCRYPTION_KEY",
            {
                "empty": "",
                "default": DEV_FALLBACK_ENCRYPTION_KEY,
                "malformed": "not-a-fernet-key",
                "rotated": Fernet.generate_key().decode(),
            }[key_state],
        )
    with get_session_local()() as db:
        with pytest.raises(ConnectorRuntimeError) as error:
            load_runtime_values(
                db, task=db.get(Task, task_id), turn_id="turn-1", required=True
            )
        assert error.value.code == (
            ERROR_RUNTIME_SECRET_UNAVAILABLE
            if key_state == "rotated"
            else ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
        )
        assert error.value.status_code == 503
        assert "synthetic-secret" not in str(error.value)
        assert db.query(TaskRuntimeSecret).count() == 1
