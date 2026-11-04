"""
Skill utilities - Utility functions for creating skill_manager
"""

import logging
from pathlib import Path
from typing import List, Optional

from ..config import get_external_skills_dirs
from ..core.storage.manager import get_storage_root
from .library import (
    CompositeSkillLibraryProvider,
    FilesystemSkillLibraryProvider,
    SkillLibraryProvider,
    SkillScopeContext,
    get_skill_library_provider,
)
from .manager import SkillManager

logger = logging.getLogger(__name__)


def create_skill_manager(
    skills_roots: Optional[List[Path]] = None,
    *,
    context: SkillScopeContext | None = None,
    provider: SkillLibraryProvider | None = None,
) -> "SkillManager":
    """
    Create skill_manager (not initialized)

    Args:
        skills_roots: Optional list of skills directories. If None, uses defaults:
                     - builtin, project (./skills/), user (~/.xagent/skills/)
                     - XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS env var is always appended if set, regardless of this parameter.

    Returns:
        SkillManager instance (not initialized)
    """

    if skills_roots is None:
        # Start with default directories if not specified
        skills_roots = _get_default_skill_dirs()

    # Always append external directories from environment variable
    external_dirs = get_external_skills_dirs()
    if external_dirs:
        skills_roots = skills_roots + external_dirs
        logger.info(f"Appended {len(external_dirs)} external skill directories")

    skill_manager = SkillManager(
        skills_roots=skills_roots,
        provider=provider
        or get_skill_library_provider()
        or _build_default_provider(skills_roots),
        context=context,
    )

    return skill_manager


def build_default_skill_library_provider(
    overlays: Optional[List[SkillLibraryProvider]] = None,
) -> SkillLibraryProvider:
    """Build xagent's default provider chain.

    Overlay providers are inserted between filesystem base skills and personal
    DB skills.  SaaS uses this to add team skills below personal skills.
    """

    return _build_default_provider(_get_default_skill_dirs(), overlays=overlays)


def _build_default_provider(
    skills_roots: List[Path],
    overlays: Optional[List[SkillLibraryProvider]] = None,
) -> SkillLibraryProvider:
    from .personal_db import XagentPersonalDbSkillProvider

    return CompositeSkillLibraryProvider(
        [
            FilesystemSkillLibraryProvider(skills_roots),
            *(overlays or []),
            XagentPersonalDbSkillProvider(),
        ]
    )


def _get_default_skill_dirs() -> List[Path]:
    """
    Get default skill directories.

    Load order (later skills override earlier ones with the same name):
    1. Built-in skills (read-only, shipped with xagent)
    2. Project skills (./skills/ in current working directory)
    3. User skills (~/.xagent/skills/, created if needed)

    Returns:
        List of default skill directory paths
    """
    builtin_skills_dir = SkillManager.get_builtin_root()
    project_skills_dir = Path("skills")
    user_skills_dir = get_storage_root() / "skills"

    return [builtin_skills_dir, project_skills_dir, user_skills_dir]
