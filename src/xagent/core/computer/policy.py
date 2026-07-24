from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

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
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: ComputerPolicyOutcome
    risk: ComputerRiskLevel
    reason: str = Field(min_length=1)
    action_indexes: list[int] = Field(default_factory=list)


class ComputerActionPolicy(Protocol):
    """Policy boundary implemented by a product-specific risk evaluator."""

    async def evaluate(
        self,
        batch: ComputerActionBatch,
        observation: ComputerObservation,
    ) -> ComputerPolicyDecision: ...
