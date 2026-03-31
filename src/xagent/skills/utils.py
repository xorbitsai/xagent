"""
Skill utilities - Utility functions for creating skill_manager
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from ..core.storage.manager import get_storage_root
from .manager import SkillManager

logger = logging.getLogger(__name__)


def create_skill_manager(
    skills_roots: Optional[List[Path]] = None,
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
    if env_dirs := os.getenv("XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS", ""):
        if external_dirs := _parse_skill_dirs(env_dirs):
            skills_roots = skills_roots + external_dirs
            logger.info(f"Appended {len(external_dirs)} external skill directories")

    # Import here to avoid circular import
    from .manager import SkillManager

    # Create skill_manager (not initialized)
    skill_manager = SkillManager(skills_roots=skills_roots)

    return skill_manager


def _parse_skill_dirs(env_value: str) -> List[Path]:
    """
    Parse XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS environment variable

    Args:
        env_value: Comma-separated directory paths

    Returns:
        List of valid Path objects
    """
    skills_roots = []

    for dir_path in env_value.split(","):
        dir_path = dir_path.strip()
        if not dir_path:
            continue

        # Check for URL-like paths before path expansion
        if "://" in dir_path:
            logger.warning(f"Skipping non-local path (not supported yet): {dir_path}")
            continue

        # Expand environment variables and user home directory
        expanded_path = os.path.expanduser(os.path.expandvars(dir_path))
        path = Path(expanded_path).expanduser()

        # Add path (directory may be created or replaced after xagent starts)
        skills_roots.append(path)
        logger.info(f"Added skills directory: {path}")

    return skills_roots


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
