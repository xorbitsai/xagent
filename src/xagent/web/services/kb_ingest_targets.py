from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models.kb_ingest_target import KBIngestTarget


def _target_query(
    db: Session,
    *,
    user_id: int,
    collection: str,
    target_path: str,
) -> Any:
    return db.query(KBIngestTarget).filter(
        KBIngestTarget.user_id == user_id,
        KBIngestTarget.collection == collection,
        KBIngestTarget.target_path == target_path,
    )


def _target_query_for_update(
    db: Session,
    *,
    user_id: int,
    collection: str,
    target_path: str,
) -> Any:
    return _target_query(
        db,
        user_id=user_id,
        collection=collection,
        target_path=target_path,
    ).with_for_update()


def admit_kb_ingest_target(
    db: Session,
    *,
    user_id: int,
    collection: str,
    target_path: str,
    file_id: str,
    generation_id: str,
    job_id: str,
    file_sha256: str,
) -> KBIngestTarget:
    """Accept a new background-ingest generation for one canonical KB target."""

    try:
        target = cast(
            KBIngestTarget | None,
            _target_query_for_update(
                db,
                user_id=user_id,
                collection=collection,
                target_path=target_path,
            ).first(),
        )
        if target is None:
            target = KBIngestTarget(
                user_id=user_id,
                collection=collection,
                target_path=target_path,
                file_id=file_id,
                latest_generation_id=generation_id,
                latest_job_id=job_id,
                latest_file_sha256=file_sha256,
                deleted_at=None,
            )
        else:
            setattr(target, "file_id", file_id)
            setattr(target, "latest_generation_id", generation_id)
            setattr(target, "latest_job_id", job_id)
            setattr(target, "latest_file_sha256", file_sha256)
            setattr(target, "deleted_at", None)

        db.add(target)
        db.commit()
        db.refresh(target)
        return target
    except IntegrityError:
        db.rollback()
        target = cast(
            KBIngestTarget,
            _target_query_for_update(
                db,
                user_id=user_id,
                collection=collection,
                target_path=target_path,
            ).one(),
        )
        setattr(target, "file_id", file_id)
        setattr(target, "latest_generation_id", generation_id)
        setattr(target, "latest_job_id", job_id)
        setattr(target, "latest_file_sha256", file_sha256)
        setattr(target, "deleted_at", None)
        db.add(target)
        db.commit()
        db.refresh(target)
        return target


def is_latest_kb_ingest_generation(
    db: Session,
    *,
    user_id: int,
    collection: str,
    target_path: str,
    generation_id: str | None,
) -> bool:
    if not generation_id:
        return False

    target = cast(
        KBIngestTarget | None,
        _target_query_for_update(
            db,
            user_id=user_id,
            collection=collection,
            target_path=target_path,
        ).first(),
    )
    return (
        target is not None
        and target.deleted_at is None
        and str(target.latest_generation_id) == generation_id
    )


def release_kb_ingest_target_generation(
    db: Session,
    *,
    user_id: int,
    collection: str,
    target_path: str,
    generation_id: str | None,
) -> bool:
    if not generation_id:
        return False

    target = cast(
        KBIngestTarget | None,
        _target_query_for_update(
            db,
            user_id=user_id,
            collection=collection,
            target_path=target_path,
        ).first(),
    )
    if target is None or str(target.latest_generation_id) != generation_id:
        return False

    setattr(target, "deleted_at", datetime.now(timezone.utc))
    db.add(target)
    db.commit()
    return True


def tombstone_kb_ingest_target(
    db: Session,
    *,
    user_id: int,
    collection: str,
    target_path: str,
    file_id: str | None = None,
    commit: bool = True,
) -> KBIngestTarget:
    target = cast(
        KBIngestTarget | None,
        _target_query_for_update(
            db,
            user_id=user_id,
            collection=collection,
            target_path=target_path,
        ).first(),
    )
    if target is None:
        target = KBIngestTarget(
            user_id=user_id,
            collection=collection,
            target_path=target_path,
            file_id=file_id or str(uuid.uuid4()),
            latest_generation_id=str(uuid.uuid4()),
            latest_job_id=None,
            latest_file_sha256="",
        )
    else:
        if file_id:
            setattr(target, "file_id", file_id)
        setattr(target, "latest_generation_id", str(uuid.uuid4()))
        setattr(target, "latest_job_id", None)

    setattr(target, "deleted_at", datetime.now(timezone.utc))
    db.add(target)
    if commit:
        db.commit()
        db.refresh(target)
    return target


def tombstone_kb_ingest_targets_for_collection(
    db: Session,
    *,
    user_id: int,
    collection: str,
) -> int:
    targets = cast(
        list[KBIngestTarget],
        (
            db.query(KBIngestTarget)
            .filter(
                KBIngestTarget.user_id == user_id,
                KBIngestTarget.collection == collection,
            )
            .with_for_update()
            .all()
        ),
    )
    if not targets:
        return 0

    now = datetime.now(timezone.utc)
    for target in targets:
        setattr(target, "latest_generation_id", str(uuid.uuid4()))
        setattr(target, "latest_job_id", None)
        setattr(target, "deleted_at", now)
        db.add(target)
    db.commit()
    return len(targets)
