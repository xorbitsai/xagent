"""
xagent Skills Module

This module provides a skill management system compatible with Claude Skills format.
Skills are directory-based modules that provide knowledge and templates for task planning.
"""

from .library import (
    CompositeSkillLibraryProvider,
    SkillLibraryProvider,
    SkillRecord,
    SkillScopeContext,
    StaticRecordsProvider,
    get_skill_library_provider,
    set_skill_library_provider,
)
from .manager import SkillManager
from .parser import SkillParser

__all__ = [
    "CompositeSkillLibraryProvider",
    "SkillLibraryProvider",
    "SkillManager",
    "SkillParser",
    "SkillRecord",
    "SkillScopeContext",
    "StaticRecordsProvider",
    "get_skill_library_provider",
    "set_skill_library_provider",
]
