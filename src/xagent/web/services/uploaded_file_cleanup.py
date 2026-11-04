"""Complete a committed upload claim without losing unfinished obligations."""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Callable, Literal

from sqlalchemy.orm import Session

from ...core.tools.core.RAG_tools.storage.file_reference import file_cleanup_lock
from ..models.uploaded_file import UploadedFile
from .uploaded_file_cleanup_disposal import (
    CleanupBudgetExhausted,
    CleanupResourceUncertain,
    CleanupWorkBudget,
)
from .uploaded_file_cleanup_resources import (
    build_cleanup_manifest,
    capture_previews,
    dispose_resource,
    validate_configuration,
)

logger = logging.getLogger(__name__)
CleanupOutcome = Literal["deleted", "stale", "exists", "unknown", "pending", "yielded"]
CLEANUP_PHASES = ("durable", "local", "previews")


def cleanup_complete(manifest: Any) -> bool:
    return bool(
        isinstance(manifest, dict)
        and manifest.get("version") == 1
        and all(phase in manifest.get("done", []) for phase in CLEANUP_PHASES)
    )


def run_uploaded_file_cleanup(
    *,
    session_factory: Callable[[], Session],
    row_id: int,
    user_id: int,
    file_id: str,
    task_id: int | None,
    storage_key: str,
    expected_updated_at: datetime | None,
    compensation_delete: Callable[..., str],
    take_over: bool = False,
) -> CleanupOutcome:
    """Fence execution, takeover, phase receipts and settlement by exact token.

    The execution lock prevents a takeover from settling while an older worker
    still has destructive I/O in flight. It is distinct from the reference lock;
    no SQL connection or reference lock spans any storage operation.
    """
    from .uploaded_file_store import (
        settle_uploaded_file_compensation_no_commit,
        snapshot_uploaded_file_version,
        take_over_uploaded_file_compensation_no_commit,
    )

    with file_cleanup_lock(file_id):
        token = expected_updated_at
        with session_factory() as db:
            if take_over:
                token = take_over_uploaded_file_compensation_no_commit(
                    db,
                    row_id=row_id,
                    user_id=user_id,
                    file_id=file_id,
                    task_id=task_id,
                    storage_key=storage_key,
                    expected_updated_at=token,
                )
                if token is None:
                    return "stale"
                db.commit()
            record = _claim_query(
                db, row_id, user_id, file_id, task_id, storage_key, token
            ).first()
            if record is None or token is None:
                return "stale"
            manifest = copy.deepcopy(record.cleanup_manifest)
            snapshot = snapshot_uploaded_file_version(record)
        # Upgrade-era compensations still retain their source metadata. Adopt
        # only what that metadata and current resource evidence can prove.
        if manifest is None:
            manifest = build_cleanup_manifest(snapshot)
            if not _save_manifest(
                session_factory,
                row_id,
                user_id,
                file_id,
                task_id,
                storage_key,
                token,
                manifest,
            ):
                return "stale"
        budget = CleanupWorkBudget()

        def save_progress() -> bool:
            return _save_manifest(
                session_factory,
                row_id,
                user_id,
                file_id,
                task_id,
                storage_key,
                token,
                manifest,
            )

        try:
            if (
                manifest.get("version") != 1
                or manifest.get("file_id") != file_id
                or manifest.get("user_id") != user_id
                or manifest.get("storage_key") != storage_key
            ):
                raise CleanupResourceUncertain(
                    "Cleanup manifest identity does not match its claim"
                )
            validate_configuration(manifest)
            # Retain the object when captured evidence already needs reconciliation.
            if "local" not in manifest["done"]:
                uncertainty = manifest.get("uncertain_materialization") or next(
                    (
                        resource["uncertain"]
                        for resource in manifest["local"]
                        if resource.get("uncertain")
                    ),
                    None,
                )
                if uncertainty:
                    raise CleanupResourceUncertain(uncertainty)
            for phase in CLEANUP_PHASES:
                if phase in manifest["done"]:
                    continue
                if phase == "durable":
                    presence = compensation_delete(
                        user_id=user_id, storage_key=storage_key
                    )
                    if presence != "absent":
                        return "exists" if presence == "exists" else "unknown"
                elif phase == "local":
                    _dispose_phase_resources(manifest, phase, budget)
                else:
                    if manifest["previews"] is None:
                        manifest["previews"] = capture_previews(manifest)
                        if not save_progress():
                            return "stale"
                    _dispose_phase_resources(manifest, phase, budget)
                    # Producers hold the same execution guard and cannot publish
                    # another preview between this check and settlement.
                    if capture_previews(manifest):
                        raise CleanupResourceUncertain(
                            "Owned previews remain after disposal"
                        )
                manifest["done"].append(phase)
                if not save_progress():
                    return "stale"
        except CleanupBudgetExhausted:
            if not save_progress():
                return "stale"
            return "yielded"
        except (CleanupResourceUncertain, OSError):
            logger.warning(
                "Upload cleanup needs reconciliation for %s", file_id, exc_info=True
            )
            return "pending"
        with session_factory() as db:
            result = settle_uploaded_file_compensation_no_commit(
                db,
                row_id=row_id,
                user_id=user_id,
                file_id=file_id,
                task_id=task_id,
                storage_key=storage_key,
                expected_updated_at=token,
                presence="absent",
            )
            if result is None:
                return "stale"
            db.commit()
            return "deleted"


def _dispose_phase_resources(
    manifest: dict[str, Any], phase: str, budget: CleanupWorkBudget
) -> None:
    """Advance a durable prefix through the resources captured for one phase."""
    positions = manifest.setdefault("disposal_positions", {})
    position = positions.get(phase, 0)
    resources = manifest[phase] or []
    while position < len(resources):
        budget.consume_disposal()
        resource = resources[position]
        dispose_resource(resource, budget=budget)
        position += 1
        positions[phase] = position


def _claim_query(
    db: Session,
    row_id: int,
    user_id: int,
    file_id: str,
    task_id: int | None,
    storage_key: str,
    token: datetime | None,
) -> Any:
    return db.query(UploadedFile).filter(
        UploadedFile.id == row_id,
        UploadedFile.user_id == user_id,
        UploadedFile.file_id == file_id,
        UploadedFile.task_id == task_id,
        UploadedFile.storage_key == storage_key,
        UploadedFile.storage_status == "compensating",
        UploadedFile.updated_at == token,
    )


def _save_manifest(
    sessions: Callable[[], Session],
    row_id: int,
    user_id: int,
    file_id: str,
    task_id: int | None,
    storage_key: str,
    token: datetime,
    manifest: dict[str, Any],
) -> bool:
    with sessions() as db:
        changed = _claim_query(
            db, row_id, user_id, file_id, task_id, storage_key, token
        ).update(
            {UploadedFile.cleanup_manifest: manifest, UploadedFile.updated_at: token},
            synchronize_session=False,
        )
        if changed != 1:
            return False
        db.commit()
        return True
