"""Personal database-backed skill provider."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from .library import SkillRecord, SkillScopeContext

logger = logging.getLogger(__name__)


def _load_personal_skill_records_sync(
    session_factory: Any, user_id: int
) -> list[SkillRecord]:
    """Load and detach one user's personal skills in an owned DB session."""
    from sqlalchemy.orm import selectinload

    from xagent.web.models.skill import UserSkill

    with session_factory() as db:
        skills = (
            db.query(UserSkill)
            .options(selectinload(UserSkill.files))
            .filter(UserSkill.user_id == user_id)
            .order_by(UserSkill.name)
            .all()
        )
        records: list[SkillRecord] = []
        for skill in skills:
            files = {file.path: bytes(file.content) for file in skill.files}
            if "SKILL.md" not in files:
                continue
            records.append(
                SkillRecord(
                    name=str(skill.name),
                    source="personal",
                    scope="personal",
                    files=files,
                    path=f"db://personal/{skill.id}",
                    metadata=deepcopy(dict(skill.skill_metadata or {})),
                    provider_id="xagent-personal-db",
                )
            )
        return records


class XagentPersonalDbSkillProvider:
    """Load personal skills owned by the current xagent user."""

    async def list_records(self, context: SkillScopeContext) -> list[SkillRecord]:
        if context.user_id is None:
            return []

        from xagent.web.models.database import get_optional_session_local
        from xagent.web.services.db_runtime import run_db_io_cancellation_safe

        user_id = int(context.user_id)
        session_factory = get_optional_session_local()
        if session_factory is None:
            logger.warning("Personal skill database layer is unavailable")
            return []
        return await run_db_io_cancellation_safe(
            lambda: _load_personal_skill_records_sync(session_factory, user_id)
        )

    async def read_file(
        self, context: SkillScopeContext, record: SkillRecord, path: str
    ) -> bytes:
        if path in record.files:
            return record.files[path]
        raise FileNotFoundError(f"File not found: {path!r} in skill {record.name!r}")
