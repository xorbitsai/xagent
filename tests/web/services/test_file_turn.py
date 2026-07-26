"""Transactional invariants for task-turn file binding."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.execution_scope import (
    ExecutionScope,
    set_execution_scope_resolver,
    set_execution_scope_snapshot_loader,
)
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.execution_scope_snapshot import (
    load_task_execution_scope_snapshot,
)
from xagent.web.services.file_turn import (
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
        db.rollback()

    with db_runtime() as db:
        assert _task_id_for_file(db, "available") is None


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
    set_execution_scope_resolver(lambda _task_id: isolated_scope)
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
        set_execution_scope_resolver(None)
        set_execution_scope_snapshot_loader(None)
        engine.dispose()
