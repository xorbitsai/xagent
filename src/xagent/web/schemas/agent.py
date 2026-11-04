"""Shared agent request/response schema pieces."""

import unicodedata
from typing import Annotated, List, Optional

from pydantic import AfterValidator, BaseModel, Field, StrictInt

PROMPT_CONTEXT_FIELD_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$"
PROMPT_CONTEXT_MAX_FIELDS = 32
PROMPT_CONTEXT_MIN_LENGTH = 1
PROMPT_CONTEXT_MAX_LENGTH = 1024
PROMPT_CONTEXT_MAX_DESCRIPTION_LENGTH = 200
PROMPT_CONTEXT_PRIMITIVES = (str, int, float, bool)
_UNSAFE_DESCRIPTION_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


def is_valid_prompt_context_description(value: str) -> bool:
    """True when a description renders on one line of the meanings block."""
    return (
        0 < len(value) <= PROMPT_CONTEXT_MAX_DESCRIPTION_LENGTH
        and "`" not in value
        and not any(
            unicodedata.category(char) in _UNSAFE_DESCRIPTION_CATEGORIES
            for char in value
        )
    )


def _normalize_description(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > PROMPT_CONTEXT_MAX_DESCRIPTION_LENGTH:
        raise ValueError(
            "description must be at most "
            f"{PROMPT_CONTEXT_MAX_DESCRIPTION_LENGTH} characters"
        )
    # Backticks and control/line-break/bidi characters break the block format.
    if not is_valid_prompt_context_description(value):
        raise ValueError(
            "description cannot contain backticks, control, line-break or bidi "
            "characters"
        )
    return value


PromptContextFieldDescription = Annotated[
    Optional[str], AfterValidator(_normalize_description)
]


class PromptContextFieldDeclaration(BaseModel):
    """One field unsigned callers (widget entry, personal keys) may pass."""

    name: str = Field(..., pattern=PROMPT_CONTEXT_FIELD_NAME_PATTERN)
    max_length: StrictInt = Field(
        ...,
        ge=PROMPT_CONTEXT_MIN_LENGTH,
        le=PROMPT_CONTEXT_MAX_LENGTH,
    )
    description: PromptContextFieldDescription = None


def _unique_field_names(
    declarations: List[PromptContextFieldDeclaration],
) -> List[PromptContextFieldDeclaration]:
    names = [declaration.name for declaration in declarations]
    if len(names) != len(set(names)):
        raise ValueError("prompt_context_fields names must be unique")
    return declarations


PromptContextFieldList = Annotated[
    List[PromptContextFieldDeclaration],
    Field(max_length=PROMPT_CONTEXT_MAX_FIELDS),
    AfterValidator(_unique_field_names),
]
