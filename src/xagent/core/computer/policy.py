from __future__ import annotations

import re
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerObservation,
)


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


def find_computer_target_element(
    action: ComputerAction,
    observation: ComputerObservation,
) -> ComputerElement | None:
    """Resolve an element target, including the smallest element under a point."""
    target = action.target
    if target is None:
        if action.type in {
            ComputerActionType.TYPE,
            ComputerActionType.KEYPRESS,
        }:
            return next(
                (
                    element
                    for element in observation.elements
                    if element.metadata.get("focused") is True
                ),
                None,
            )
        return None
    if target.element_id is not None:
        return next(
            (
                element
                for element in observation.elements
                if element.element_id == target.element_id
            ),
            None,
        )
    point = target.point
    if point is None:
        return None
    candidates = [
        element
        for element in observation.elements
        if (
            element.bounds.x <= point.x <= element.bounds.x + element.bounds.width
            and element.bounds.y <= point.y <= element.bounds.y + element.bounds.height
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda element: element.bounds.width * element.bounds.height,
    )


_HIGH_IMPACT_LABEL = re.compile(
    r"\b("
    r"accept|agree|approve|authorize|book|buy|checkout|confirm|create\s+account|"
    r"delete|follow|install|like|log\s*out|pay|place\s+order|post|publish|"
    r"purchase|remove|save|schedule|send|share|sign\s*in|submit|subscribe|"
    r"transfer|unsubscribe|upload"
    r")\b|"
    r"(付款|购买|结账|确认|同意|授权|预订|预约|删除|退出登录|保存|创建账户|"
    r"发布|发送|分享|登录|提交|订阅|取消订阅|转账|上传)",
    re.IGNORECASE,
)


class DefaultComputerActionPolicy:
    """Conservative browser policy for actions with external side effects."""

    async def evaluate(
        self,
        batch: ComputerActionBatch,
        observation: ComputerObservation,
    ) -> ComputerPolicyDecision:
        confirmation_indexes: list[int] = []
        blocked_indexes: list[int] = []
        confirmation_reasons: list[str] = []
        blocked_reasons: list[str] = []

        for index, action in enumerate(batch.actions):
            element = find_computer_target_element(action, observation)
            if (
                action.type is ComputerActionType.TYPE
                and element is not None
                and element.metadata.get("sensitive") is True
            ):
                blocked_indexes.append(index)
                blocked_reasons.append(
                    "Sensitive credentials, payment data, or one-time codes "
                    "must be entered by the user."
                )
                continue

            if action.type is ComputerActionType.DRAG:
                confirmation_indexes.append(index)
                confirmation_reasons.append(
                    "Dragging can reorder, upload, or move user content."
                )
                continue

            if action.type is ComputerActionType.KEYPRESS and self._can_submit(action):
                confirmation_indexes.append(index)
                confirmation_reasons.append(
                    "Pressing Enter or Return can submit the current form."
                )
                continue

            if action.type in {
                ComputerActionType.CLICK,
                ComputerActionType.DOUBLE_CLICK,
            } and self._is_high_impact_element(element):
                confirmation_indexes.append(index)
                confirmation_reasons.append(
                    "The selected control may create an external or destructive "
                    "side effect."
                )

        if blocked_indexes:
            return ComputerPolicyDecision(
                outcome=ComputerPolicyOutcome.BLOCK,
                risk=ComputerRiskLevel.HIGH,
                reason=" ".join(dict.fromkeys(blocked_reasons)),
                action_indexes=blocked_indexes,
            )
        if confirmation_indexes:
            return ComputerPolicyDecision(
                outcome=ComputerPolicyOutcome.REQUIRE_CONFIRMATION,
                risk=ComputerRiskLevel.ELEVATED,
                reason=" ".join(dict.fromkeys(confirmation_reasons)),
                action_indexes=confirmation_indexes,
            )
        return ComputerPolicyDecision(
            outcome=ComputerPolicyOutcome.ALLOW,
            risk=ComputerRiskLevel.LOW,
            reason="The requested computer action is low risk.",
        )

    @staticmethod
    def _can_submit(action: ComputerAction) -> bool:
        return any(key.strip().upper() in {"ENTER", "RETURN"} for key in action.keys)

    @staticmethod
    def _is_high_impact_element(element: ComputerElement | None) -> bool:
        if element is None:
            return False
        searchable = " ".join(
            value.strip()
            for value in (element.label, element.text, element.role)
            if isinstance(value, str) and value.strip()
        )
        return bool(searchable and _HIGH_IMPACT_LABEL.search(searchable))
