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
    SkillWriteContext,
    SkillWriteProvider,
    SkillWriteProviderError,
    SkillWriteProviderErrorReason,
    StaticRecordsProvider,
    get_skill_library_provider,
    get_skill_write_provider,
    set_skill_library_provider,
    set_skill_write_provider,
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
    "SkillWriteProvider",
    "SkillWriteProviderError",
    "SkillWriteProviderErrorReason",
    "SkillWriteContext",
    "StaticRecordsProvider",
    "get_skill_library_provider",
    "get_skill_write_provider",
    "set_skill_library_provider",
    "set_skill_write_provider",
]
