"""Personal database-backed skill provider."""

from __future__ import annotations

from .library import SkillRecord, SkillScopeContext


class XagentPersonalDbSkillProvider:
    """Load personal skills owned by the current xagent user."""

    async def list_records(self, context: SkillScopeContext) -> list[SkillRecord]:
        db = context.db
        user_id = context.user_id or getattr(context.user, "id", None)
        if db is None or user_id is None:
            return []

        from xagent.web.models.skill import UserSkill

        skills = (
            db.query(UserSkill)
            .filter(UserSkill.user_id == int(user_id))
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
                    metadata=dict(skill.skill_metadata or {}),
                    provider_id="xagent-personal-db",
                )
            )
        return records

    async def read_file(
        self, context: SkillScopeContext, record: SkillRecord, path: str
    ) -> bytes:
        if path in record.files:
            return record.files[path]
        raise FileNotFoundError(f"File not found: {path!r} in skill {record.name!r}")
