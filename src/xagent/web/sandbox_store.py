"""
Database-backed SandboxStore implementation.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from ..sandbox import BoxliteStore, SandboxConfig, SandboxInfo, SandboxTemplate
from .models.database import get_db
from .models.sandbox import SandboxInfo as SandboxInfoModel

logger = logging.getLogger(__name__)

# Sandbox type constant
SANDBOX_TYPE_BOXLITE = "boxlite"


class DBBoxliteStore(BoxliteStore):
    """
    Database-backed implementation of SandboxStore.
    """

    def __init__(self) -> None:
        """
        Initialize SandboxStore.
        """
        pass

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
                    SandboxInfoModel.sandbox_type == SANDBOX_TYPE_BOXLITE,
                    SandboxInfoModel.name == name,
                )
                .first()
            )
            if not model:
                return None

            return self._model_to_info(model)
        except Exception as e:
            logger.error(f"Failed to get sandbox info for {name}: {e}")
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
                    SandboxInfoModel.sandbox_type == SANDBOX_TYPE_BOXLITE,
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
            logger.error(f"Failed to add sandbox info for {name}: {e}")
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
                    SandboxInfoModel.sandbox_type == SANDBOX_TYPE_BOXLITE,
                    SandboxInfoModel.name == name,
                )
                .first()
            )
            if model:
                model.state = state
                db.commit()
        except Exception as e:
            logger.error(f"Failed to update sandbox state for {name}: {e}")
            db.rollback()
            raise
        finally:
            db.close()

    def delete_info(self, name: str) -> None:
        """Delete sandbox info from database."""
        db = self._get_db_session()
        try:
            db.query(SandboxInfoModel).filter(
                SandboxInfoModel.sandbox_type == SANDBOX_TYPE_BOXLITE,
                SandboxInfoModel.name == name,
            ).delete()
            db.commit()
        except Exception as e:
            logger.error(f"Failed to delete sandbox info for {name}: {e}")
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
            sandbox_type=SANDBOX_TYPE_BOXLITE,
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
