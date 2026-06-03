"""
xagent Skills Module

This module provides a skill management system compatible with Claude Skills format.
Skills are directory-based modules that provide knowledge and templates for task planning.
"""

from .manager import SkillManager
from .parser import SkillParser
from .selector import SkillSelector
from .library import (
    CompositeSkillLibraryProvider,
    SkillLibraryProvider,
    SkillRecord,
    SkillScopeContext,
    get_skill_library_provider,
    set_skill_library_provider,
)

__all__ = [
    "CompositeSkillLibraryProvider",
    "SkillLibraryProvider",
    "SkillManager",
    "SkillParser",
    "SkillRecord",
    "SkillScopeContext",
    "SkillSelector",
    "get_skill_library_provider",
    "set_skill_library_provider",
]
