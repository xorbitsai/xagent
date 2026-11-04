from __future__ import annotations

from enum import Enum
from typing import Annotated, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
)

from .schema import ComputerActionBatch, ComputerObservation


class ComputerRiskLevel(str, Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


class ComputerPolicyOutcome(str, Enum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    BLOCK = "block"


class ComputerPolicyDecision(BaseModel):
    """Policy outcome for the action batch's sole action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ComputerPolicyOutcome
    risk: ComputerRiskLevel
    reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=4_096),
    ]


class ComputerActionPolicy(Protocol):
    """Policy boundary implemented by an environment or product adapter."""

    async def evaluate(
        self,
        batch: ComputerActionBatch,
        observation: ComputerObservation,
    ) -> ComputerPolicyDecision: ...
