"""Personal skill operations for key-authenticated SDK callers.

The runtime owns no Session. Reads build a skill manager for the key holder on
the event loop, exactly as a task run does, and return detached values. Writes
run in a worker that creates and closes its own Session and call the same
helpers the web skill routes use, so both entry points share one set of
validation rules.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar

from fastapi import HTTPException

from ...skills.parser import SkillParser
from ...skills.utils import create_skill_manager
from ..api import skill_hub
from ..models.database import get_session_local
from ..models.skill import UserSkill
from ..models.user import User
from .db_runtime import run_db_io_cancellation_safe
from .skill_runtime import build_detached_skill_scope

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def is_valid_personal_skill_name(name: str) -> bool:
    """Whether ``name`` can be the name of a personal skill.

    The length is checked first, against the width of the stored column, and
    the pattern then has to match the whole string: ``re.match`` with a ``$``
    anchor would also accept one trailing newline.
    """
    max_length = UserSkill.__table__.c.name.type.length
    return (
        1 <= len(name) <= max_length and skill_hub._NAME_RE.fullmatch(name) is not None
    )


def _v1_skill_source(entry: dict[str, Any]) -> str:
    """Map a skill manager cache entry to the ``source`` an SDK caller sees.

    Filesystem skills outside the shipped set, whether they live in the user
    storage root or in an external directory, are both ``external``: SDK
    callers can only change personal skills.
    """
    if entry.get("scope") == "personal":
        return "personal"
    if entry.get("scope") == "team":
        return "team"
    if entry.get("source") == "builtin":
        return "builtin"
    return "external"


@dataclass(frozen=True)
class SkillSummarySnapshot:
    name: str
    description: str
    when_to_use: str
    tags: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class SkillDetailSnapshot(SkillSummarySnapshot):
    skill_md: str
    files: tuple[str, ...]


def _summary_from_entry(entry: dict[str, Any]) -> SkillSummarySnapshot:
    return SkillSummarySnapshot(
        name=str(entry["name"]),
        description=str(entry.get("description", "")),
        when_to_use=str(entry.get("when_to_use", "")),
        tags=tuple(str(tag) for tag in entry.get("tags", [])),
        source=_v1_skill_source(entry),
    )


def _detail_from_entry(entry: dict[str, Any]) -> SkillDetailSnapshot:
    return SkillDetailSnapshot(
        **asdict(_summary_from_entry(entry)),
        skill_md=str(entry.get("content", "")),
        files=tuple(str(path) for path in entry.get("files", [])),
    )


def _personal_summary_from_bundle(
    name: str, files: dict[str, bytes]
) -> SkillSummarySnapshot:
    """Describe a personal skill from the validated bytes that were stored."""
    parsed = SkillParser.parse_bundle(name=name, files=files)
    return _summary_from_entry({**parsed, "scope": "personal"})


def _load_holder(db: Any, user_id: int) -> User:
    user: User | None = db.get(User, user_id)
    if user is None:
        # Imported here: the ``api.v1`` package imports the service layer, so
        # a module-level import would make this module unimportable on its own.
        from ..api.v1.errors import V1ApiError, V1ErrorCode

        raise V1ApiError(V1ErrorCode.INVALID_API_KEY, 401)
    return user


def _log_write(action: str, key_prefix: str, user_id: int, name: str) -> None:
    logger.info(
        "personal key skill write: action=%s key_prefix=%s user_id=%d name=%r",
        action,
        key_prefix,
        user_id,
        name,
    )


def _write_or_accept_repeat(
    *, db: Any, user: User, name: str, files: dict[str, bytes], origin: str
) -> None:
    """Store a new personal skill, treating a repeat of the same write as done."""
    try:
        skill_hub._write_personal_skill(
            db=db, user=user, name=name, files=files, origin=origin
        )
    except HTTPException as exc:
        if exc.status_code != 409 or not skill_hub._personal_upload_matches_bundle(
            db=db, user=user, name=name, files=files, origin=origin
        ):
            raise


class SkillManagementRuntime:
    """Async owner for SDK personal skill operations."""

    @staticmethod
    async def _manager(user_id: int) -> Any:
        manager = create_skill_manager(
            context=build_detached_skill_scope(user_id=user_id)
        )
        await manager.ensure_initialized()
        return manager

    async def list_entries(self, *, user_id: int) -> tuple[SkillSummarySnapshot, ...]:
        """The skills the holder's tasks run with, personal ones first."""
        manager = await self._manager(user_id)
        names = [skill["name"] for skill in await manager.list_skills()]
        entries = [_summary_from_entry(await manager.get_skill(name)) for name in names]
        return tuple(
            sorted(entries, key=lambda e: (e.source != "personal", e.name.lower()))
        )

    async def visible_names(self, *, user_id: int) -> frozenset[str]:
        """The names of the skills the holder's tasks run with."""
        manager = await self._manager(user_id)
        return frozenset(skill["name"] for skill in await manager.list_skills())

    async def get_entry(self, *, user_id: int, name: str) -> SkillDetailSnapshot | None:
        manager = await self._manager(user_id)
        entry = await manager.get_skill(name)
        return None if entry is None else _detail_from_entry(entry)

    @staticmethod
    async def _write(
        action: str,
        user_id: int,
        key_prefix: str,
        name: str,
        operation: Callable[[Any, User], _T],
    ) -> _T:
        """Run ``operation`` in a worker with its own Session and the holder."""

        def run() -> _T:
            with get_session_local()() as db:
                result = operation(db, _load_holder(db, user_id))
            _log_write(action, key_prefix, user_id, name)
            return result

        return await run_db_io_cancellation_safe(run)

    async def create_skill(
        self, *, user_id: int, key_prefix: str, name: str, skill_md: str
    ) -> SkillSummarySnapshot:
        def operation(db: Any, user: User) -> SkillSummarySnapshot:
            files = skill_hub._normalize_skill_files(
                {"SKILL.md": skill_md.encode("utf-8")}
            )
            _write_or_accept_repeat(
                db=db, user=user, name=name, files=files, origin="custom"
            )
            return _personal_summary_from_bundle(name, files)

        return await self._write("create", user_id, key_prefix, name, operation)

    async def upload_skill(
        self, *, user_id: int, key_prefix: str, name: str, files: dict[str, bytes]
    ) -> SkillSummarySnapshot:
        def operation(db: Any, user: User) -> SkillSummarySnapshot:
            _write_or_accept_repeat(
                db=db, user=user, name=name, files=files, origin="upload"
            )
            return _personal_summary_from_bundle(name, files)

        return await self._write("upload", user_id, key_prefix, name, operation)

    async def update_skill(
        self, *, user_id: int, key_prefix: str, name: str, skill_md: str
    ) -> SkillSummarySnapshot:
        def operation(db: Any, user: User) -> SkillSummarySnapshot:
            skill_hub._update_personal_skill_md(
                db=db, user=user, name=name, skill_md=skill_md
            )
            return _personal_summary_from_bundle(
                name, {"SKILL.md": skill_md.encode("utf-8")}
            )

        return await self._write("update", user_id, key_prefix, name, operation)

    async def delete_skill(self, *, user_id: int, key_prefix: str, name: str) -> None:
        def operation(db: Any, user: User) -> None:
            skill_hub._delete_personal_skill(db=db, user=user, name=name)

        await self._write("delete", user_id, key_prefix, name, operation)
