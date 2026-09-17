"""Tests for stale uploaded-file compensation recovery."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier, Event, Thread

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from tests.web.pool_contention_shared import CONTENTION_POOL_TIMEOUT
from xagent.web.models.database import Base
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services import uploaded_file_recovery
from xagent.web.services.uploaded_file_recovery import (
    UploadedFileCompensationRecoveryBatch,
    get_stale_uploaded_file_compensation_candidates,
    recover_stale_uploaded_file_compensations_batch_isolated,
    run_uploaded_file_compensation_recovery_loop,
)
from xagent.web.services.uploaded_file_store import (
    RegisteredUploadCompensationClaim,
    UploadedFileStore,
    UploadedFileVersionConflict,
    compensate_registered_uploads_sync,
    take_over_uploaded_file_compensation_no_commit,
)


def _database(tmp_path):
    # WHY: the CAS test's loser must wait its turn for the winner's commit, not
    # give up. 0.1s was too tight under `pytest -n 4` (spurious QueuePool
    # TimeoutError); the CONTENTION budget is never expected to fire.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'uploaded-file-recovery.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=CONTENTION_POOL_TIMEOUT,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)


def _create_compensating_file(
    SessionLocal,
    *,
    suffix: str,
    updated_at: datetime | None,
) -> tuple[int, int, str, str]:
    with SessionLocal() as db:
        user = User(
            username=f"uploaded-file-recovery-{suffix}",
            password_hash="hash",
            is_admin=False,
        )
        db.add(user)
        db.flush()
        user_id = int(user.id)
        file_id = f"recovery-{suffix}"
        storage_key = f"users/{user_id}/uploads/{file_id}/{suffix}.txt"
        record = UploadedFile(
            file_id=file_id,
            user_id=user_id,
            filename=f"{suffix}.txt",
            storage_path=f"/tmp/{suffix}.txt",
            storage_backend="s3",
            storage_key=storage_key,
            checksum="checksum",
            storage_status="compensating",
            file_size=7,
            updated_at=updated_at,
        )
        db.add(record)
        db.commit()
        return int(record.id), user_id, file_id, storage_key


def test_recovery_grace_excludes_recent_compensation(tmp_path) -> None:
    engine, SessionLocal = _database(tmp_path)
    now = datetime.now(timezone.utc)
    old_id, *_ = _create_compensating_file(
        SessionLocal,
        suffix="old",
        updated_at=now - timedelta(minutes=10),
    )
    _create_compensating_file(
        SessionLocal,
        suffix="recent",
        updated_at=now - timedelta(seconds=10),
    )
    _create_compensating_file(
        SessionLocal,
        suffix="missing-token",
        updated_at=None,
    )

    try:
        with SessionLocal() as db:
            candidates = get_stale_uploaded_file_compensation_candidates(
                db,
                cutoff=now - timedelta(minutes=5),
                limit=10,
            )
        assert [candidate.row_id for candidate in candidates] == [old_id]
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    (
        "presence",
        "expected_status",
        "expected_deleted",
        "expected_deferred_exists",
        "expected_deferred_unknown",
    ),
    [
        ("exists", "compensating", 0, 1, 0),
        ("absent", None, 1, 0, 0),
        ("unknown", "compensating", 0, 0, 1),
    ],
)
def test_recovery_deletes_absent_and_defers_unsafe_storage_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    presence: str,
    expected_status: str | None,
    expected_deleted: int,
    expected_deferred_exists: int,
    expected_deferred_unknown: int,
) -> None:
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))
    engine, SessionLocal = _database(tmp_path)
    now = datetime.now(timezone.utc)
    row_id, _, file_id, _ = _create_compensating_file(
        SessionLocal,
        suffix=presence,
        updated_at=now - timedelta(minutes=10),
    )
    cache_dir = storage_root / "svg_png_cache"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / f"{file_id}.cached.preview.png"
    cache_path.write_bytes(b"cached preview")
    storage_deletes_checked_out: list[int] = []

    def delete_object(*, user_id: int, storage_key: str) -> str:
        del user_id, storage_key
        storage_deletes_checked_out.append(engine.pool.checkedout())
        return presence

    try:
        result = recover_stale_uploaded_file_compensations_batch_isolated(
            cutoff=now - timedelta(minutes=5),
            batch_size=10,
            session_factory=SessionLocal,
            compensation_delete=delete_object,
        )

        assert storage_deletes_checked_out == [0]
        assert result.scanned == 1
        assert result.deleted == expected_deleted
        assert result.deferred_exists == expected_deferred_exists
        assert result.deferred_unknown == expected_deferred_unknown
        with SessionLocal() as db:
            record = db.query(UploadedFile).filter(UploadedFile.id == row_id).first()
            assert (
                str(record.storage_status) if record is not None else None
            ) == expected_status
        assert cache_path.exists() is (presence != "absent")
    finally:
        engine.dispose()


def test_recovery_takeover_is_cas_safe_across_workers(tmp_path) -> None:
    engine, SessionLocal = _database(tmp_path)
    now = datetime.now(timezone.utc)
    _create_compensating_file(
        SessionLocal,
        suffix="cas",
        updated_at=now - timedelta(minutes=10),
    )
    try:
        with SessionLocal() as db:
            candidate = get_stale_uploaded_file_compensation_candidates(
                db,
                cutoff=now - timedelta(minutes=5),
                limit=1,
            )[0]

        barrier = Barrier(2)

        def take_over() -> datetime | None:
            with SessionLocal() as db:
                barrier.wait()
                result = take_over_uploaded_file_compensation_no_commit(
                    db,
                    row_id=candidate.row_id,
                    user_id=candidate.user_id,
                    file_id=candidate.file_id,
                    task_id=candidate.task_id,
                    storage_key=candidate.storage_key,
                    expected_updated_at=candidate.updated_at,
                )
                db.commit()
                return result

        with ThreadPoolExecutor(max_workers=2) as executor:
            takeovers = list(executor.map(lambda _index: take_over(), range(2)))

        assert sum(token is not None for token in takeovers) == 1
    finally:
        engine.dispose()


def test_recovery_never_restores_while_original_delete_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine, SessionLocal = _database(tmp_path)
    now = datetime.now(timezone.utc)
    row_id, user_id, file_id, storage_key = _create_compensating_file(
        SessionLocal,
        suffix="late-delete",
        updated_at=now - timedelta(minutes=10),
    )
    with SessionLocal() as db:
        record = db.query(UploadedFile).filter(UploadedFile.id == row_id).one()
        record.storage_status = "available"
        db.commit()
    claim = RegisteredUploadCompensationClaim(
        user_id=user_id,
        file_id=file_id,
        expected_task_id=None,
        expected_storage_key=storage_key,
    )
    original_delete_started = Event()
    allow_original_delete_to_return = Event()
    original_errors: list[BaseException] = []

    def original_delete(*, user_id: int, storage_key: str) -> str:
        del user_id, storage_key
        original_delete_started.set()
        assert allow_original_delete_to_return.wait(timeout=5)
        return "absent"

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.delete_uploaded_file_compensation_object",
        original_delete,
    )

    def run_original() -> None:
        try:
            compensate_registered_uploads_sync((claim,))
        except BaseException as exc:  # pragma: no cover - asserted below
            original_errors.append(exc)

    original_thread = Thread(target=run_original)
    original_thread.start()
    assert original_delete_started.wait(timeout=5)

    try:
        first_recovery = recover_stale_uploaded_file_compensations_batch_isolated(
            cutoff=now + timedelta(minutes=10),
            batch_size=10,
            session_factory=SessionLocal,
            compensation_delete=lambda **_kwargs: "exists",
        )
        assert first_recovery.deleted == 0
        assert first_recovery.deferred_exists == 1
        with SessionLocal() as db:
            claimed_record = (
                db.query(UploadedFile).filter(UploadedFile.id == row_id).one()
            )
            assert claimed_record.storage_status == "compensating"

        allow_original_delete_to_return.set()
        original_thread.join(timeout=5)
        assert not original_thread.is_alive()
        assert original_errors == []
        with SessionLocal() as db:
            claimed_record = (
                db.query(UploadedFile).filter(UploadedFile.id == row_id).one()
            )
            assert claimed_record.storage_status == "compensating"

        final_recovery = recover_stale_uploaded_file_compensations_batch_isolated(
            cutoff=now + timedelta(minutes=10),
            batch_size=10,
            session_factory=SessionLocal,
            compensation_delete=lambda **_kwargs: "absent",
        )
        assert final_recovery.deleted == 1
        with SessionLocal() as db:
            assert (
                db.query(UploadedFile).filter(UploadedFile.id == row_id).first() is None
            )
    finally:
        allow_original_delete_to_return.set()
        original_thread.join(timeout=5)
        engine.dispose()


def test_storage_path_upsert_cannot_revive_an_inflight_compensation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    engine, SessionLocal = _database(tmp_path)
    now = datetime.now(timezone.utc)
    row_id, user_id, file_id, storage_key = _create_compensating_file(
        SessionLocal,
        suffix="upsert-race",
        updated_at=now - timedelta(minutes=10),
    )
    source = tmp_path / "upsert-race.txt"
    source.write_text("replacement", encoding="utf-8")
    with SessionLocal() as db:
        record = db.query(UploadedFile).filter(UploadedFile.id == row_id).one()
        record.storage_status = "available"
        record.storage_path = str(source)
        db.commit()

    claim = RegisteredUploadCompensationClaim(
        user_id=user_id,
        file_id=file_id,
        expected_task_id=None,
        expected_storage_key=storage_key,
    )
    original_delete_started = Event()
    allow_original_delete_to_return = Event()
    original_errors: list[BaseException] = []
    sync_calls: list[str] = []

    def original_delete(*, user_id: int, storage_key: str) -> str:
        del user_id, storage_key
        original_delete_started.set()
        assert allow_original_delete_to_return.wait(timeout=5)
        return "absent"

    def unexpected_sync(self, *, storage_key=None, mime_type=None):
        del storage_key, mime_type
        sync_calls.append(str(self.record.file_id))
        self.record.storage_status = "available"
        return self.record

    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.get_session_local",
        lambda: SessionLocal,
    )
    monkeypatch.setattr(
        "xagent.web.services.uploaded_file_store.delete_uploaded_file_compensation_object",
        original_delete,
    )
    monkeypatch.setattr(
        "xagent.web.services.managed_file_ref.ManagedFileRef.sync_to_durable",
        unexpected_sync,
    )

    def run_original() -> None:
        try:
            compensate_registered_uploads_sync((claim,))
        except BaseException as exc:  # pragma: no cover - asserted below
            original_errors.append(exc)

    original_thread = Thread(target=run_original)
    original_thread.start()
    assert original_delete_started.wait(timeout=5)

    try:
        with SessionLocal() as db:
            with pytest.raises(
                UploadedFileVersionConflict,
                match="compensating",
            ):
                UploadedFileStore(db).upsert_by_storage_path(
                    user_id=user_id,
                    filename=source.name,
                    storage_path=source,
                    mime_type="text/plain",
                    file_size=source.stat().st_size,
                )
            db.rollback()
        assert sync_calls == []

        allow_original_delete_to_return.set()
        original_thread.join(timeout=5)
        assert not original_thread.is_alive()
        assert original_errors == []
        with SessionLocal() as db:
            assert (
                db.query(UploadedFile).filter(UploadedFile.id == row_id).first() is None
            )
    finally:
        allow_original_delete_to_return.set()
        original_thread.join(timeout=5)
        engine.dispose()


def test_recovery_cursor_prevents_unknown_first_page_from_starving_later_rows(
    tmp_path,
) -> None:
    engine, SessionLocal = _database(tmp_path)
    now = datetime.now(timezone.utc)
    for index in range(3):
        _create_compensating_file(
            SessionLocal,
            suffix=f"page-{index}",
            updated_at=now - timedelta(minutes=10) + timedelta(seconds=index),
        )

    def delete_object(*, user_id: int, storage_key: str) -> str:
        del user_id
        return "absent" if storage_key.endswith("page-2.txt") else "unknown"

    try:
        first = recover_stale_uploaded_file_compensations_batch_isolated(
            cutoff=now - timedelta(minutes=5),
            batch_size=2,
            session_factory=SessionLocal,
            compensation_delete=delete_object,
        )
        second = recover_stale_uploaded_file_compensations_batch_isolated(
            cutoff=now - timedelta(minutes=5),
            batch_size=2,
            after=first.next_cursor,
            session_factory=SessionLocal,
            compensation_delete=delete_object,
        )

        assert first.scanned == 2
        assert first.deferred_unknown == 2
        assert first.next_cursor is not None
        assert second.scanned == 1
        assert second.deleted == 1
        with SessionLocal() as db:
            remaining = {
                str(record.file_id): str(record.storage_status)
                for record in db.query(UploadedFile).all()
            }
        assert remaining == {
            "recovery-page-0": "compensating",
            "recovery-page-1": "compensating",
        }
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_recovery_loop_survives_pool_timeout_and_waits_for_next_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_recover(*, cutoff: datetime, batch_size: int, after):
        nonlocal calls
        calls += 1
        assert cutoff.tzinfo is not None
        assert batch_size == 7
        assert after is None
        if calls == 1:
            raise SQLAlchemyTimeoutError("pool checkout timed out")
        raise asyncio.CancelledError

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        uploaded_file_recovery,
        "recover_stale_uploaded_file_compensations_batch_isolated",
        fake_recover,
    )
    monkeypatch.setattr(uploaded_file_recovery.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await run_uploaded_file_compensation_recovery_loop(
            poll_interval_seconds=11,
            stale_after_seconds=300,
            batch_size=7,
        )

    assert calls == 2
    assert sleeps == [11]


@pytest.mark.asyncio
async def test_recovery_loop_advances_full_page_and_resets_after_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    cursors = [(now, 2), (now, 3)]
    seen_after = []
    sleeps: list[float] = []

    def fake_recover(*, cutoff: datetime, batch_size: int, after):
        del cutoff
        seen_after.append(after)
        assert batch_size == 2
        if len(seen_after) == 1:
            return UploadedFileCompensationRecoveryBatch(
                scanned=2,
                deferred_unknown=2,
                next_cursor=cursors[0],
            )
        if len(seen_after) == 2:
            return UploadedFileCompensationRecoveryBatch(
                scanned=1,
                deleted=1,
                next_cursor=cursors[1],
            )
        raise asyncio.CancelledError

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        uploaded_file_recovery,
        "recover_stale_uploaded_file_compensations_batch_isolated",
        fake_recover,
    )
    monkeypatch.setattr(uploaded_file_recovery.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await run_uploaded_file_compensation_recovery_loop(
            poll_interval_seconds=11,
            stale_after_seconds=300,
            batch_size=2,
        )

    assert seen_after == [None, cursors[0], None]
    assert sleeps == [11, 11]
