from __future__ import annotations

import pytest
from pydantic import ValidationError

from xagent.core.computer.policy import (
    ComputerPolicyDecision,
    ComputerPolicyOutcome,
    ComputerRiskLevel,
)


def test_policy_decision_normalizes_reason() -> None:
    decision = ComputerPolicyDecision(
        outcome=ComputerPolicyOutcome.REQUIRE_CONFIRMATION,
        risk=ComputerRiskLevel.ELEVATED,
        reason="  external side effect  ",
    )

    assert decision.reason == "external side effect"


def test_policy_decision_rejects_removed_action_indexes() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ComputerPolicyDecision(
            outcome=ComputerPolicyOutcome.BLOCK,
            risk=ComputerRiskLevel.HIGH,
            reason="blocked",
            action_indexes=[0],
        )
