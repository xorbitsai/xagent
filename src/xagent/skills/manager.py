"""
Skill Manager - Manage skill scanning and retrieval
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .library import (
    FilesystemSkillLibraryProvider,
    SkillLibraryProvider,
    SkillScopeContext,
    get_skill_library_provider,
)
from .parser import SkillParser

logger = logging.getLogger(__name__)


class SkillManager:
    """Core manager for the skill system"""

    def __init__(
        self,
        skills_roots: List[Path] | None = None,
        *,
        provider: SkillLibraryProvider | None = None,
        context: SkillScopeContext | None = None,
    ):
        """
        Args:
            skills_roots: List of skills directory paths (supports multiple directories)
                - First is the built-in skills directory (read-only)
                - Subsequent ones are user-defined skills directories (writable)
        """
        self.skills_roots = [Path(p) for p in skills_roots or []]
        self.provider = provider or get_skill_library_provider()
        if self.provider is None:
            self.provider = FilesystemSkillLibraryProvider(self.skills_roots)
        self.context = context or SkillScopeContext()

        self._skills_cache: Dict[str, Dict] = {}
        self._initialized = False
        self._init_task: Optional[Any] = None

    async def ensure_initialized(self) -> None:
        """Ensure initialization is complete (lazy loading mode)"""
        if self._initialized:
            return

        # If there's an initialization task running, wait for it to complete
        if self._init_task is not None:
            await self._init_task
            return

        # Create and execute initialization task
        self._init_task = asyncio.create_task(self._do_initialize())
        await self._init_task

    async def _do_initialize(self) -> None:
        """Actual initialization logic"""
        await self.initialize()
        self._init_task = None

    async def initialize(self) -> None:
        """Initialize: scan all skills"""
        logger.info("📂 Scanning skills...")
        if self.skills_roots:
            for root in self.skills_roots:
                logger.info(f"  from {root}...")
        await self.reload()
        self._initialized = True
        logger.info(f"✓ Loaded {len(self._skills_cache)} skills")

    async def reload(self) -> None:
        """Reload all skills"""
        self._skills_cache.clear()

        assert self.provider is not None  # set in __init__
        records = await self.provider.list_records(self.context)
        for record in records:
            try:
                skill_info = SkillParser.parse_bundle(
                    name=record.name,
                    files=record.files,
                    path=record.path or f"provider://{record.source}/{record.name}",
                )
                skill_info["source"] = record.source
                skill_info["scope"] = record.scope
                skill_info["effective"] = record.effective
                skill_info["shadowed_by"] = record.shadowed_by
                skill_info["_record"] = record
                skill_info["files"] = record.file_names
                self._skills_cache[skill_info["name"]] = skill_info
                logger.info("  ✓ Loaded: %s (%s)", record.name, record.source)
            except Exception as e:
                logger.error("  ✗ Error loading %s: %s", record.name, e, exc_info=True)

        logger.info(f"Total skills loaded: {len(self._skills_cache)}")

    async def list_skills(self) -> List[Dict]:
        """List all skills (brief information)"""
        await self.ensure_initialized()
        return [
            {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "when_to_use": skill.get("when_to_use", ""),
                "tags": skill.get("tags", []),
            }
            for skill in self._skills_cache.values()
        ]

    async def get_skill(self, name: str) -> Optional[Dict]:
        """Get single skill (full information including template)"""
        await self.ensure_initialized()
        return self._skills_cache.get(name)

    def has_skills(self) -> bool:
        """Check if there are available skills"""
        return len(self._skills_cache) > 0

    @classmethod
    def get_builtin_root(cls) -> Path:
        """Get built-in skills directory"""
        return Path(__file__).parent / "builtin"
