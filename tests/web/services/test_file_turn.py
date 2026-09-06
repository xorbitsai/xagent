"""Transactional invariants for task-turn file binding."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from tests.shared.execution_scope import register_scope_resolver
from tests.shared.postgres_disposable import disposable_database_factory
from xagent.core.execution_scope import (
    DeferToSnapshot,
    ExecutionScope,
    ExecutionScopeAuthorityError,
    ExecutionScopeResolverContractError,
    set_execution_scope_snapshot_loader,
)
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import file_turn as file_turn_service
from xagent.web.services.execution_scope_snapshot import (
    load_task_execution_scope_snapshot,
)
from xagent.web.services.file_turn import (
    bind_turn_files,
    bind_turn_files_no_commit,
    resolve_turn_file_infos,
)
from xagent.web.services.managed_file_ref import ManagedFileRef


@pytest.fixture()
def db_runtime(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'file-turn.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    try:
        yield SessionLocal
    finally:
        engine.dispose()


def _seed_binding_rows(SessionLocal) -> tuple[int, int, int]:
    with SessionLocal() as db:
        user = User(username="file-owner", password_hash="hash", is_admin=False)
        db.add(user)
        db.flush()
        target = Task(
            user_id=int(user.id),
            title="target",
            description="target",
            status=TaskStatus.COMPLETED,
            source="sdk",
        )
        competing = Task(
            user_id=int(user.id),
            title="competing",
            description="competing",
            status=TaskStatus.COMPLETED,
            source="sdk",
        )
        db.add_all([target, competing])
        db.flush()
        db.add_all(
            [
                UploadedFile(
                    file_id="available",
                    user_id=int(user.id),
                    filename="available.txt",
                    storage_path="/tmp/available.txt",
                    detached_reason="task_deleted",
                    detached_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                    file_size=1,
                ),
                UploadedFile(
                    file_id="competing",
                    user_id=int(user.id),
                    task_id=int(competing.id),
                    filename="competing.txt",
                    storage_path="/tmp/competing.txt",
                    file_size=1,
                ),
            ]
        )
        db.commit()
        return int(user.id), int(target.id), int(competing.id)


def _task_id_for_file(db: Session, file_id: str) -> int | None:
    value = (
        db.query(UploadedFile.task_id).filter(UploadedFile.file_id == file_id).scalar()
    )
    return int(value) if value is not None else None


def test_bind_turn_files_stages_without_committing(db_runtime) -> None:
    user_id, target_task_id, _competing_task_id = _seed_binding_rows(db_runtime)

    with db_runtime() as db:
        missing = bind_turn_files_no_commit(
            file_ids=["available", "available"],
            task_id=target_task_id,
            owner_user_id=user_id,
            db=db,
        )
        assert missing == []
        assert _task_id_for_file(db, "available") == target_task_id
        rebound = (
            db.query(UploadedFile).filter(UploadedFile.file_id == "available").one()
        )
        assert rebound.detached_reason is None
        assert rebound.detached_at is None
        db.rollback()

    with db_runtime() as db:
        assert _task_id_for_file(db, "available") is None


def test_bind_turn_files_rejects_a_missing_task(db_runtime) -> None:
    user_id, _target_task_id, _competing_task_id = _seed_binding_rows(db_runtime)

    with db_runtime() as db:
        missing = bind_turn_files_no_commit(
            file_ids=["available"],
            task_id=987654321,
            owner_user_id=user_id,
            db=db,
        )

        assert missing == ["available"]
        assert _task_id_for_file(db, "available") is None


@pytest.mark.postgresql
def test_postgresql_bind_turn_files_does_not_deadlock_child_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binding must not take a task lock that conflicts with FK KEY SHARE."""
    with disposable_database_factory("xagent_file_turn") as make_database:
        engine = make_database("bind_child")
        Base.metadata.create_all(bind=engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        user_id, task_id, _competing_task_id = _seed_binding_rows(SessionLocal)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE file_turn_child ("
                    "id INTEGER PRIMARY KEY, "
                    "task_id INTEGER NOT NULL REFERENCES tasks(id))"
                )
            )

        file_locked = threading.Event()
        task_locked = threading.Event()
        results: dict[str, object] = {}
        real_lock_task = file_turn_service.lock_task_no_commit

        def observed_lock_task(*args, **kwargs):
            task = real_lock_task(*args, **kwargs)
            task_locked.set()
            return task

        monkeypatch.setattr(
            file_turn_service,
            "lock_task_no_commit",
            observed_lock_task,
        )

        def update_file_then_insert_child() -> None:
            with SessionLocal() as db:
                try:
                    db.execute(text("SET LOCAL deadlock_timeout = '100ms'"))
                    db.execute(text("SET LOCAL statement_timeout = '5s'"))
                    db.query(UploadedFile).filter(
                        UploadedFile.file_id == "available"
                    ).update(
                        {UploadedFile.filename: "writer-holds-file.txt"},
                        synchronize_session=False,
                    )
                    file_locked.set()
                    assert task_locked.wait(timeout=5)
                    db.execute(
                        text(
                            "INSERT INTO file_turn_child (id, task_id) "
                            "VALUES (1, :task_id)"
                        ),
                        {"task_id": task_id},
                    )
                    db.commit()
                    results["writer"] = "committed"
                except Exception as exc:  # noqa: BLE001
                    db.rollback()
                    results["writer"] = type(exc).__name__

        def bind_file() -> None:
            assert file_locked.wait(timeout=5)
            with SessionLocal() as db:
                try:
                    db.execute(text("SET LOCAL deadlock_timeout = '100ms'"))
                    db.execute(text("SET LOCAL statement_timeout = '5s'"))
                    missing = file_turn_service.bind_turn_files_no_commit(
                        file_ids=["available"],
                        task_id=task_id,
                        owner_user_id=user_id,
                        db=db,
                    )
                    db.commit()
                    results["binder"] = missing
                except Exception as exc:  # noqa: BLE001
                    db.rollback()
                    results["binder"] = type(exc).__name__

        writer = threading.Thread(target=update_file_then_insert_child)
        binder = threading.Thread(target=bind_file)
        writer.start()
        binder.start()
        writer.join(timeout=10)
        binder.join(timeout=10)

        assert not writer.is_alive(), results
        assert not binder.is_alive(), results
        assert results == {"writer": "committed", "binder": []}


def test_bind_turn_files_rechecks_after_another_task_wins(db_runtime) -> None:
    user_id, target_task_id, competing_task_id = _seed_binding_rows(db_runtime)

    # This represents the read-only preparation snapshot: the file was
    # available when the caller resolved it.
    with db_runtime() as db:
        assert _task_id_for_file(db, "available") is None

    # A different transaction wins before the turn's atomic claim.
    with db_runtime() as db:
        db.query(UploadedFile).filter(UploadedFile.file_id == "available").update(
            {UploadedFile.task_id: competing_task_id},
            synchronize_session=False,
        )
        db.commit()

    with db_runtime() as db:
        missing = bind_turn_files_no_commit(
            file_ids=["available"],
            task_id=target_task_id,
            owner_user_id=user_id,
            db=db,
        )
        assert missing == ["available"]
        assert _task_id_for_file(db, "available") == competing_task_id


def test_caller_can_roll_back_partial_file_claim_as_one_domain_transaction(
    db_runtime,
) -> None:
    user_id, target_task_id, competing_task_id = _seed_binding_rows(db_runtime)

    with db_runtime() as db:
        missing = bind_turn_files_no_commit(
            file_ids=["available", "competing"],
            task_id=target_task_id,
            owner_user_id=user_id,
            db=db,
        )
        assert missing == ["competing"]
        # The available row was staged, but the domain owner can roll the
        # whole turn claim back because the requested set was incomplete.
        assert _task_id_for_file(db, "available") == target_task_id
        db.rollback()

    with db_runtime() as db:
        assert _task_id_for_file(db, "available") is None
        assert _task_id_for_file(db, "competing") == competing_task_id


def test_committing_bind_wrapper_rolls_back_a_partial_claim(db_runtime) -> None:
    user_id, target_task_id, _competing_task_id = _seed_binding_rows(db_runtime)

    with db_runtime() as db:
        with pytest.raises(ValueError, match="competing"):
            bind_turn_files(
                file_ids=["available", "competing"],
                task_id=target_task_id,
                owner_user_id=user_id,
                db=db,
            )
        db.commit()

    with db_runtime() as db:
        assert _task_id_for_file(db, "available") is None


def test_resolve_materializes_after_releasing_outer_pool_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bound file must not self-deadlock through the scope snapshot loader."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'one-slot-file-turn.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    local_path = tmp_path / "bound.txt"
    local_path.write_text("payload")

    monkeypatch.setattr(database_module, "_engine", engine)
    monkeypatch.setattr(database_module, "_SessionLocal", SessionLocal)
    set_execution_scope_snapshot_loader(load_task_execution_scope_snapshot)
    try:
        with SessionLocal() as db:
            user = User(username="scoped-owner", password_hash="hash", is_admin=False)
            db.add(user)
            db.flush()
            task = Task(
                user_id=int(user.id),
                title="scoped task",
                description="scoped task",
                status=TaskStatus.COMPLETED,
                source="sdk",
                agent_config={
                    "execution_scope": ExecutionScope(
                        workspace_segments=("tenant-a",),
                    ).to_dict()
                },
            )
            db.add(task)
            db.flush()
            db.add(
                UploadedFile(
                    file_id="bound",
                    user_id=int(user.id),
                    task_id=int(task.id),
                    filename="bound.txt",
                    storage_path=str(local_path),
                    file_size=local_path.stat().st_size,
                )
            )
            db.commit()
            user_id = int(user.id)
            task_id = int(task.id)

        checked_out_during_materialize: list[int] = []
        original_ensure_local = ManagedFileRef.ensure_local

        def observe_pool_ownership(self: ManagedFileRef) -> Path:
            checked_out_during_materialize.append(engine.pool.checkedout())
            return original_ensure_local(self)

        monkeypatch.setattr(ManagedFileRef, "ensure_local", observe_pool_ownership)

        with SessionLocal() as db:
            file_infos, missing = resolve_turn_file_infos(
                file_ids=["bound"],
                owner_user_id=user_id,
                task_id=task_id,
                db=db,
            )

        assert missing == []
        assert [item["file_id"] for item in file_infos] == ["bound"]
        assert checked_out_during_materialize == [0]
    finally:
        set_execution_scope_snapshot_loader(None)
        engine.dispose()


def test_resolve_preserves_registered_scope_fallback_without_nested_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing persisted snapshot must still use the canonical resolver."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'one-slot-resolver-fallback.db'}",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    local_path = tmp_path / "resolver-bound.txt"
    local_path.write_text("payload")
    isolated_scope = ExecutionScope(
        workspace_segments=("clients", "3", "end_users", "7"),
        isolate_external_dirs=True,
    )

    monkeypatch.setattr(database_module, "_engine", engine)
    monkeypatch.setattr(database_module, "_SessionLocal", SessionLocal)
    set_execution_scope_snapshot_loader(load_task_execution_scope_snapshot)
    register_scope_resolver(lambda _task_id: isolated_scope)
    try:
        with SessionLocal() as db:
            user = User(
                username="resolver-owner",
                password_hash="hash",
                is_admin=False,
            )
            db.add(user)
            db.flush()
            task = Task(
                user_id=int(user.id),
                title="resolver task",
                description="resolver task",
                status=TaskStatus.COMPLETED,
                source="sdk",
                agent_config={},
            )
            db.add(task)
            db.flush()
            db.add(
                UploadedFile(
                    file_id="resolver-bound",
                    user_id=int(user.id),
                    task_id=int(task.id),
                    filename="resolver-bound.txt",
                    storage_path=str(local_path),
                    file_size=local_path.stat().st_size,
                )
            )
            db.commit()
            user_id = int(user.id)
            task_id = int(task.id)

        observed_prefixes: list[str] = []
        observed_checked_out: list[int] = []
        original_ensure_local = ManagedFileRef.ensure_local

        def observe_scope_and_pool(self: ManagedFileRef) -> Path:
            observed_prefixes.append(self.storage.prefix)
            observed_checked_out.append(engine.pool.checkedout())
            return original_ensure_local(self)

        monkeypatch.setattr(
            ManagedFileRef,
            "ensure_local",
            observe_scope_and_pool,
        )

        with SessionLocal() as db:
            file_infos, missing = resolve_turn_file_infos(
                file_ids=["resolver-bound"],
                owner_user_id=user_id,
                task_id=task_id,
                db=db,
            )

        assert missing == []
        assert [item["file_id"] for item in file_infos] == ["resolver-bound"]
        assert observed_checked_out == [0]
        assert observed_prefixes == ["users/1/clients/3/end_users/7"]
    finally:
        register_scope_resolver(None)
        set_execution_scope_snapshot_loader(None)
        engine.dispose()


def test_resolve_fails_closed_on_registered_scope_authority_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A turn face: the resolved scope selects the namespace files are read
    back from, and this call passes the persisted snapshot it already read
    in its own Session explicitly via ``persisted_agent_config``, so a
    disagreement between the registered resolver and that snapshot must
    propagate ``ExecutionScopeAuthorityError`` instead of being downgraded,
    and no file may be materialized on the mismatched path."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'authority-mismatch.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    local_path = tmp_path / "disputed.txt"
    local_path.write_text("payload")

    register_scope_resolver(
        lambda _task_id: ExecutionScope(workspace_segments=("resolver-tenant",))
    )
    try:
        with SessionLocal() as db:
            user = User(username="disputed-owner", password_hash="hash", is_admin=False)
            db.add(user)
            db.flush()
            task = Task(
                user_id=int(user.id),
                title="disputed task",
                description="disputed task",
                status=TaskStatus.COMPLETED,
                source="sdk",
                agent_config={
                    "execution_scope": ExecutionScope(
                        workspace_segments=("snapshot-tenant",),
                    ).to_dict()
                },
            )
            db.add(task)
            db.flush()
            db.add(
                UploadedFile(
                    file_id="disputed",
                    user_id=int(user.id),
                    task_id=int(task.id),
                    filename="disputed.txt",
                    storage_path=str(local_path),
                    file_size=local_path.stat().st_size,
                )
            )
            db.commit()
            user_id = int(user.id)
            task_id = int(task.id)

        materialize_calls: list[str] = []
        original_ensure_local = ManagedFileRef.ensure_local

        def track_materialize(self: ManagedFileRef) -> Path:
            materialize_calls.append(self.storage_key)
            return original_ensure_local(self)

        monkeypatch.setattr(ManagedFileRef, "ensure_local", track_materialize)

        with SessionLocal() as db:
            with pytest.raises(ExecutionScopeAuthorityError):
                resolve_turn_file_infos(
                    file_ids=["disputed"],
                    owner_user_id=user_id,
                    task_id=task_id,
                    db=db,
                )

        assert materialize_calls == []
    finally:
        register_scope_resolver(None)
        engine.dispose()


def test_malformed_snapshot_is_ignored_when_the_resolver_is_authoritative(
    tmp_path: Path,
) -> None:
    """An authoritative resolver can answer without the snapshot, so a row
    whose snapshot fails field validation is ignored and the turn proceeds.
    Tolerance is a property of this branch only -- see the two tests below for
    the branches where the same row must fail the turn."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'malformed-snapshot.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    local_path = tmp_path / "attached.txt"
    local_path.write_text("payload")

    register_scope_resolver(
        lambda _task_id: ExecutionScope(workspace_segments=("resolver-tenant",))
    )
    try:
        with SessionLocal() as db:
            user = User(username="corrupt-owner", password_hash="hash", is_admin=False)
            db.add(user)
            db.flush()
            task = Task(
                user_id=int(user.id),
                title="corrupt snapshot task",
                description="corrupt snapshot task",
                status=TaskStatus.COMPLETED,
                source="sdk",
                # workspace_segments must be a list/tuple: this row cannot be
                # decoded into a scope at all.
                agent_config={"execution_scope": {"workspace_segments": 5}},
            )
            db.add(task)
            db.flush()
            db.add(
                UploadedFile(
                    file_id="attached",
                    user_id=int(user.id),
                    task_id=int(task.id),
                    filename="attached.txt",
                    storage_path=str(local_path),
                    file_size=local_path.stat().st_size,
                )
            )
            db.commit()
            user_id = int(user.id)
            task_id = int(task.id)

        with SessionLocal() as db:
            infos, missing = resolve_turn_file_infos(
                file_ids=["attached"],
                owner_user_id=user_id,
                task_id=task_id,
                db=db,
            )

        assert missing == []
        assert [info["file_id"] for info in infos] == ["attached"]
    finally:
        register_scope_resolver(None)
        engine.dispose()


def _seed_malformed_snapshot_task(SessionLocal, local_path: Path) -> tuple[int, int]:
    with SessionLocal() as db:
        user = User(username="corrupt-owner", password_hash="hash", is_admin=False)
        db.add(user)
        db.flush()
        task = Task(
            user_id=int(user.id),
            title="corrupt snapshot task",
            description="corrupt snapshot task",
            status=TaskStatus.COMPLETED,
            source="sdk",
            # workspace_segments must be a list/tuple: this row cannot be
            # decoded into a scope at all.
            agent_config={"execution_scope": {"workspace_segments": 5}},
        )
        db.add(task)
        db.flush()
        db.add(
            UploadedFile(
                file_id="attached",
                user_id=int(user.id),
                task_id=int(task.id),
                filename="attached.txt",
                storage_path=str(local_path),
                file_size=local_path.stat().st_size,
            )
        )
        db.commit()
        return int(user.id), int(task.id)


def test_malformed_snapshot_fails_the_turn_with_no_resolver_registered(
    tmp_path: Path,
) -> None:
    """No resolver registered is the shape this repository ships in, and there
    the persisted snapshot is the only namespace authority. A row that cannot
    be decoded therefore has to fail the turn: treating it as "no candidate"
    would resolve the task to unscoped, which is a namespace decision made by
    accident rather than by an authority."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'no-resolver.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    local_path = tmp_path / "attached.txt"
    local_path.write_text("payload")
    try:
        user_id, task_id = _seed_malformed_snapshot_task(SessionLocal, local_path)

        with SessionLocal() as db:
            with pytest.raises(ExecutionScopeResolverContractError):
                resolve_turn_file_infos(
                    file_ids=["attached"],
                    owner_user_id=user_id,
                    task_id=task_id,
                    db=db,
                )
    finally:
        engine.dispose()


def test_malformed_snapshot_fails_the_turn_when_the_resolver_abstains(
    tmp_path: Path,
) -> None:
    """An abstaining resolver has just said it does not know this task's
    namespace, so a snapshot it cannot read leaves nobody who knows. The turn
    must fail rather than fall through to the abstention's fallback on the
    strength of a row that never decoded."""

    engine = create_engine(
        f"sqlite:///{tmp_path / 'abstain.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    local_path = tmp_path / "attached.txt"
    local_path.write_text("payload")

    register_scope_resolver(
        lambda _task_id: DeferToSnapshot(fallback=ExecutionScope()),
    )
    try:
        user_id, task_id = _seed_malformed_snapshot_task(SessionLocal, local_path)

        with SessionLocal() as db:
            with pytest.raises(ExecutionScopeResolverContractError):
                resolve_turn_file_infos(
                    file_ids=["attached"],
                    owner_user_id=user_id,
                    task_id=task_id,
                    db=db,
                )
    finally:
        register_scope_resolver(None)
        engine.dispose()
