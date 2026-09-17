"""Generic skill library providers.

The skill runtime consumes logical skill records through this module instead of
assuming every skill lives in a local directory.  Application layers can install
ordered providers to add scopes such as database-backed personal/team skills
without teaching core xagent about those policies.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

SkillScopeMetadataValue = str | int | float | bool | None
_DETACHED_SKILL_METADATA_VALUE_TYPES = {str, int, float, bool, type(None)}


def _freeze_detached_skill_metadata(
    metadata: Mapping[str, SkillScopeMetadataValue], *, context_name: str
) -> Mapping[str, SkillScopeMetadataValue]:
    """Copy and freeze scalar extension metadata at the public boundary."""
    copied_metadata = dict(metadata)
    if any(
        type(key) is not str or type(value) not in _DETACHED_SKILL_METADATA_VALUE_TYPES
        for key, value in copied_metadata.items()
    ):
        raise TypeError(f"{context_name} metadata must contain detached scalar values")
    return MappingProxyType(copied_metadata)


def _validate_detached_skill_user_id(user_id: int | None) -> None:
    """Reject identity values that could retain request-owned state."""
    if user_id is not None and type(user_id) is not int:
        raise TypeError("skill user_id must be an exact detached integer or None")


@dataclass(frozen=True)
class SkillScopeContext:
    """Detached runtime identity passed to read-only skill providers.

    Request objects, ORM entities, and database sessions must never enter this
    context because agent services may retain it across long-running waits.
    """

    user_id: int | None = None
    metadata: Mapping[str, SkillScopeMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_detached_skill_user_id(self.user_id)
        object.__setattr__(
            self,
            "metadata",
            _freeze_detached_skill_metadata(self.metadata, context_name="skill scope"),
        )


@dataclass(frozen=True)
class SkillWriteContext:
    """Detached scalar identity passed to a scoped Skill write provider.

    Providers resolve authorization from ``user_id`` and own their database
    dependency, transaction, and result materialization.  Callers must not
    place request, Session, or ORM state in this context.
    """

    user_id: int | None = None
    metadata: Mapping[str, SkillScopeMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_detached_skill_user_id(self.user_id)
        object.__setattr__(
            self,
            "metadata",
            _freeze_detached_skill_metadata(self.metadata, context_name="skill write"),
        )


class SkillWriteProviderErrorReason(str, Enum):
    """Allowlisted public failure categories for Skill write providers."""

    FORBIDDEN = "forbidden"
    INVALID_REQUEST = "invalid_request"


class SkillWriteProviderError(Exception):
    """A provider failure with an explicitly public-safe message."""

    def __init__(
        self,
        reason: SkillWriteProviderErrorReason,
        public_detail: str,
    ) -> None:
        self.reason = reason
        self.public_detail = public_detail
        super().__init__(public_detail)


@dataclass(frozen=True)
class SkillRecord:
    """Provider-neutral skill file bundle."""

    name: str
    source: str
    files: dict[str, bytes]
    scope: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    effective: bool = True
    shadowed_by: str | None = None
    provider_id: str | None = None
    _all_file_names: list[str] = field(default_factory=list)

    @property
    def file_names(self) -> list[str]:
        """All file names in this skill, including those not pre-loaded into memory."""
        if self._all_file_names:
            return sorted(self._all_file_names)
        return sorted(self.files)


class SkillLibraryProvider(Protocol):
    """Read interface implemented by filesystem and database providers.

    Metadata is non-authoritative: database providers re-resolve access from
    ``context.user_id`` in their own short-lived database session.
    """

    async def list_records(self, context: SkillScopeContext) -> list[SkillRecord]:
        """Return records in provider precedence order."""

    async def read_file(
        self, context: SkillScopeContext, record: SkillRecord, path: str
    ) -> bytes:
        """Read one file from a record."""


class SkillWriteProvider(Protocol):
    """Optional scoped write interface with provider-owned transactions.

    A database provider opens, commits or rolls back, materializes, and closes
    its own session before returning. Expected public failures use
    :class:`SkillWriteProviderError`; unknown failures are sanitized at the
    invocation boundary.
    """

    async def create_skill(
        self,
        context: SkillWriteContext,
        *,
        scope: str,
        name: str,
        files: dict[str, bytes],
        origin: str = "custom",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create one skill in an explicit writable scope."""

    async def update_skill_file(
        self,
        context: SkillWriteContext,
        *,
        scope: str,
        name: str,
        path: str,
        content: bytes,
    ) -> None:
        """Update one file in an explicit writable scope."""

    async def delete_skill(
        self,
        context: SkillWriteContext,
        *,
        scope: str,
        name: str,
    ) -> None:
        """Delete one skill in an explicit writable scope."""


class CompositeSkillLibraryProvider:
    """Ordered provider chain where later records override earlier names."""

    def __init__(self, providers: list[SkillLibraryProvider]):
        self.providers = list(providers)

    async def list_records(self, context: SkillScopeContext) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        for provider in self.providers:
            records.extend(await provider.list_records(context))
        return records

    async def list_visible_records(
        self, context: SkillScopeContext
    ) -> list[SkillRecord]:
        records = await self.list_records(context)
        winner_by_name: dict[str, SkillRecord] = {}
        for record in records:
            winner_by_name[record.name] = record

        visible: list[SkillRecord] = []
        for record in records:
            winner = winner_by_name.get(record.name)
            if winner is record:
                visible.append(replace(record, effective=True, shadowed_by=None))
            else:
                visible.append(
                    replace(
                        record,
                        effective=False,
                        shadowed_by=winner.scope or winner.source if winner else None,
                    )
                )
        return visible

    async def read_file(
        self, context: SkillScopeContext, record: SkillRecord, path: str
    ) -> bytes:
        if path in record.files:
            return record.files[path]
        if record.path:
            target = (Path(record.path) / path).resolve()
            root = Path(record.path).resolve()
            target.relative_to(root)
            return target.read_bytes()
        raise FileNotFoundError(f"File not found: {path!r} in skill {record.name!r}")


class FilesystemSkillLibraryProvider:
    """Directory-backed provider for built-in, project, and external skills."""

    def __init__(self, roots: list[Path]):
        self.roots = [Path(root) for root in roots]

    async def list_records(self, context: SkillScopeContext) -> list[SkillRecord]:
        records: list[SkillRecord] = []
        for root in self.roots:
            if not root.is_dir():
                continue
            for skill_dir in sorted(root.iterdir(), key=lambda p: p.name):
                if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
                    continue
                files: dict[str, bytes] = {}
                all_names: list[str] = []
                for file_path in sorted(skill_dir.rglob("*")):
                    if not file_path.is_file():
                        continue
                    rel = str(file_path.relative_to(skill_dir)).replace("\\", "/")
                    all_names.append(rel)
                    mt = mimetypes.guess_type(file_path.name)[0] or ""
                    if not (
                        mt.startswith("image/")
                        or mt.startswith("audio/")
                        or mt.startswith("video/")
                    ):
                        files[rel] = file_path.read_bytes()
                records.append(
                    SkillRecord(
                        name=skill_dir.name,
                        source=_source_for_root(root),
                        scope=_source_for_root(root),
                        files=files,
                        path=str(skill_dir),
                        provider_id="filesystem",
                        _all_file_names=all_names,
                    )
                )
        return records

    async def read_file(
        self, context: SkillScopeContext, record: SkillRecord, path: str
    ) -> bytes:
        if path in record.files:
            return record.files[path]
        if record.path:
            target = (Path(record.path) / path).resolve()
            root = Path(record.path).resolve()
            target.relative_to(root)
            return target.read_bytes()
        raise FileNotFoundError(f"File not found: {path!r} in skill {record.name!r}")


class StaticRecordsProvider:
    """Wraps a pre-fetched list of SkillRecords.

    Used to share a process-wide filesystem scan result across per-request
    SkillManager instances, avoiding repeated disk scans while still
    allowing a fresh DB layer to be queried each time.
    """

    def __init__(self, records: list[SkillRecord]) -> None:
        self._records = list(records)

    async def list_records(self, context: SkillScopeContext) -> list[SkillRecord]:
        return list(self._records)

    async def read_file(
        self, context: SkillScopeContext, record: SkillRecord, path: str
    ) -> bytes:
        if path in record.files:
            return record.files[path]
        if record.path:
            target = (Path(record.path) / path).resolve()
            root = Path(record.path).resolve()
            target.relative_to(root)
            return target.read_bytes()
        raise FileNotFoundError(f"File not found: {path!r} in skill {record.name!r}")


_skill_library_provider: SkillLibraryProvider | None = None
_skill_write_provider: SkillWriteProvider | None = None


def set_skill_library_provider(provider: SkillLibraryProvider | None) -> None:
    """Install the process-wide skill provider hook."""

    global _skill_library_provider
    _skill_library_provider = provider


def get_skill_library_provider() -> SkillLibraryProvider | None:
    return _skill_library_provider


def set_skill_write_provider(provider: SkillWriteProvider | None) -> None:
    """Install the process-wide scoped skill write hook."""

    global _skill_write_provider
    _skill_write_provider = provider


def get_skill_write_provider() -> SkillWriteProvider | None:
    return _skill_write_provider


def guess_media_type(path: str) -> str | None:
    return mimetypes.guess_type(path)[0]


def _source_for_root(root: Path) -> str:
    from ..core.storage.manager import get_storage_root
    from .manager import SkillManager

    try:
        resolved = root.resolve()
        if resolved == SkillManager.get_builtin_root().resolve():
            return "builtin"
        if resolved == (get_storage_root() / "skills").resolve():
            return "user"
    except OSError:
        pass
    return "external"
