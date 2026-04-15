"""
Database-backed SandboxStore implementation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from ..sandbox import (
    BoxliteStore,
    DockerStore,
    SandboxConfig,
    SandboxInfo,
    SandboxSnapshot,
    SandboxTemplate,
)
from .models.database import get_db
from .models.sandbox import SandboxInfo as SandboxInfoModel
from .models.sandbox import SandboxSnapshot as SandboxSnapshotModel

logger = logging.getLogger(__name__)

# Sandbox type constant
SANDBOX_TYPE_BOXLITE = "boxlite"
SANDBOX_TYPE_DOCKER = "docker"


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 text to a datetime for database persistence."""
    if value is None:
        return None

    normalized_value = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized_value)


class _DBSandboxStoreBase:
    """Shared database-backed CRUD implementation for sandbox info records."""

    sandbox_type: str

    def _get_db_session(self):  # type: ignore[no-untyped-def]
        """Get database session. Can be mocked in tests."""
        return next(get_db())

    def get_info(self, name: str) -> Optional[SandboxInfo]:
        """Get sandbox info from database."""
        db = self._get_db_session()
        try:
            model = (
                db.query(SandboxInfoModel)
                .filter(
                    SandboxInfoModel.sandbox_type == self.sandbox_type,
                    SandboxInfoModel.name == name,
                )
                .first()
            )
            if not model:
                return None

            return self._model_to_info(model)
        except Exception as e:
            logger.error(
                "Failed to get sandbox info for %s (type=%s): %s",
                name,
                self.sandbox_type,
                e,
            )
            raise
        finally:
            db.close()

    def add_info(self, name: str, info: SandboxInfo) -> None:
        """Add sandbox info to database."""
        db = self._get_db_session()
        try:
            # Check if already exists
            existing = (
                db.query(SandboxInfoModel)
                .filter(
                    SandboxInfoModel.sandbox_type == self.sandbox_type,
                    SandboxInfoModel.name == name,
                )
                .first()
            )

            if existing:
                # Update existing
                self._update_model_from_info(existing, info)
            else:
                # Create new
                model = self._info_to_model(info)
                db.add(model)

            db.commit()
        except Exception as e:
            logger.error(
                "Failed to add sandbox info for %s (type=%s): %s",
                name,
                self.sandbox_type,
                e,
            )
            db.rollback()
            raise
        finally:
            db.close()

    def update_info_state(self, name: str, state: str) -> None:
        """Update sandbox state in database."""
        db = self._get_db_session()
        try:
            model = (
                db.query(SandboxInfoModel)
                .filter(
                    SandboxInfoModel.sandbox_type == self.sandbox_type,
                    SandboxInfoModel.name == name,
                )
                .first()
            )
            if model:
                model.state = state
                db.commit()
        except Exception as e:
            logger.error(
                "Failed to update sandbox state for %s (type=%s): %s",
                name,
                self.sandbox_type,
                e,
            )
            db.rollback()
            raise
        finally:
            db.close()

    def delete_info(self, name: str) -> None:
        """Delete sandbox info from database."""
        db = self._get_db_session()
        try:
            db.query(SandboxInfoModel).filter(
                SandboxInfoModel.sandbox_type == self.sandbox_type,
                SandboxInfoModel.name == name,
            ).delete()
            db.commit()
        except Exception as e:
            logger.error(
                "Failed to delete sandbox info for %s (type=%s): %s",
                name,
                self.sandbox_type,
                e,
            )
            db.rollback()
            raise
        finally:
            db.close()

    def _model_to_info(self, model: SandboxInfoModel) -> SandboxInfo:
        """Convert database model to SandboxInfo."""
        # Parse template JSON
        template_str = str(model.template) if model.template is not None else "{}"
        template_data = json.loads(template_str)
        template = SandboxTemplate(**template_data)

        # Parse config JSON
        config_str = str(model.config) if model.config is not None else "{}"
        config_data = json.loads(config_str)
        config = SandboxConfig(**config_data)

        return SandboxInfo(
            name=str(model.name),
            state=str(model.state),
            template=template,
            config=config,
            created_at=model.created_at.isoformat()
            if model.created_at is not None
            else None,
        )

    def _info_to_model(self, info: SandboxInfo) -> SandboxInfoModel:
        """Convert SandboxInfo to database model."""

        # Convert Pydantic model to dict, then to JSON
        template_json = json.dumps(info.template.model_dump())
        config_json = json.dumps(info.config.model_dump())

        model = SandboxInfoModel(
            sandbox_type=self.sandbox_type,
            name=info.name,
            state=info.state,
            template=template_json,
            config=config_json,
        )
        return model

    def _update_model_from_info(
        self, model: SandboxInfoModel, info: SandboxInfo
    ) -> None:
        """Update database model from SandboxInfo."""
        model.state = info.state  # type: ignore[assignment]

        # Convert Pydantic model to dict, then to JSON
        model.template = json.dumps(info.template.model_dump())  # type: ignore[assignment]
        model.config = json.dumps(info.config.model_dump())  # type: ignore[assignment]


class DBBoxliteStore(_DBSandboxStoreBase, BoxliteStore):
    """Database-backed implementation of BoxliteStore."""

    sandbox_type = SANDBOX_TYPE_BOXLITE


class DBDockerStore(_DBSandboxStoreBase, DockerStore):
    """Database-backed implementation of DockerStore."""

    sandbox_type = SANDBOX_TYPE_DOCKER

    def get_snapshot(self, snapshot_id: str) -> Optional[SandboxSnapshot]:
        """Get Docker snapshot info from database."""
        db = self._get_db_session()
        try:
            model = (
                db.query(SandboxSnapshotModel)
                .filter(
                    SandboxSnapshotModel.sandbox_type == SANDBOX_TYPE_DOCKER,
                    SandboxSnapshotModel.snapshot_id == snapshot_id,
                )
                .first()
            )
            if not model:
                return None
            return self._snapshot_model_to_snapshot(model)
        except Exception as e:
            logger.error(f"Failed to get docker snapshot info for {snapshot_id}: {e}")
            raise
        finally:
            db.close()

    def add_snapshot(self, snapshot: SandboxSnapshot) -> None:
        """Add or update Docker snapshot info in database."""
        db = self._get_db_session()
        try:
            existing = (
                db.query(SandboxSnapshotModel)
                .filter(
                    SandboxSnapshotModel.sandbox_type == SANDBOX_TYPE_DOCKER,
                    SandboxSnapshotModel.snapshot_id == snapshot.snapshot_id,
                )
                .first()
            )
            if existing:
                self._update_snapshot_model(existing, snapshot)
            else:
                db.add(self._snapshot_to_model(snapshot))
            db.commit()
        except Exception as e:
            logger.error(
                f"Failed to add docker snapshot info for {snapshot.snapshot_id}: {e}"
            )
            db.rollback()
            raise
        finally:
            db.close()

    def list_snapshots(self) -> list[SandboxSnapshot]:
        """List persisted Docker snapshots."""
        db = self._get_db_session()
        try:
            models = (
                db.query(SandboxSnapshotModel)
                .filter(SandboxSnapshotModel.sandbox_type == SANDBOX_TYPE_DOCKER)
                .order_by(SandboxSnapshotModel.snapshot_id.asc())
                .all()
            )
            return [self._snapshot_model_to_snapshot(model) for model in models]
        except Exception as e:
            logger.error(f"Failed to list docker snapshots: {e}")
            raise
        finally:
            db.close()

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete Docker snapshot info from database."""
        db = self._get_db_session()
        try:
            db.query(SandboxSnapshotModel).filter(
                SandboxSnapshotModel.sandbox_type == SANDBOX_TYPE_DOCKER,
                SandboxSnapshotModel.snapshot_id == snapshot_id,
            ).delete()
            db.commit()
        except Exception as e:
            logger.error(
                f"Failed to delete docker snapshot info for {snapshot_id}: {e}"
            )
            db.rollback()
            raise
        finally:
            db.close()

    def _snapshot_model_to_snapshot(
        self, model: SandboxSnapshotModel
    ) -> SandboxSnapshot:
        """Convert database model to SandboxSnapshot."""
        metadata_str = (
            str(model.metadata_json) if model.metadata_json is not None else "{}"
        )
        return SandboxSnapshot(
            snapshot_id=str(model.snapshot_id),
            metadata=json.loads(metadata_str),
            created_at=model.created_at.isoformat()
            if model.created_at is not None
            else None,
        )

    def _snapshot_to_model(self, snapshot: SandboxSnapshot) -> SandboxSnapshotModel:
        """Convert SandboxSnapshot to database model."""
        return SandboxSnapshotModel(
            sandbox_type=SANDBOX_TYPE_DOCKER,
            snapshot_id=snapshot.snapshot_id,
            metadata_json=json.dumps(snapshot.metadata),
            created_at=_parse_iso_datetime(snapshot.created_at),
        )

    def _update_snapshot_model(
        self, model: SandboxSnapshotModel, snapshot: SandboxSnapshot
    ) -> None:
        """Update snapshot database model from SandboxSnapshot."""
        model.metadata_json = json.dumps(snapshot.metadata)  # type: ignore[assignment]
        model.created_at = _parse_iso_datetime(snapshot.created_at)  # type: ignore[assignment]
