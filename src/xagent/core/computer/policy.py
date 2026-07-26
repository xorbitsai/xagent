from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from .schema import (
    ELEMENT_EXTRACTION_FAILED_KEY,
    ELEMENT_EXTRACTION_INCOMPLETE_KEY,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerObservation,
)

#: Actions that only read the environment and therefore need no risk gate.
_READ_ONLY_ACTIONS = frozenset({ComputerActionType.SCREENSHOT, ComputerActionType.WAIT})

#: Actions whose effect depends on hitting the intended element.
_POINTED_ACTIONS = frozenset(
    {
        ComputerActionType.CLICK,
        ComputerActionType.DOUBLE_CLICK,
        ComputerActionType.TYPE,
        ComputerActionType.REPLACE_TEXT,
    }
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


def normalize_host_patterns(patterns: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize configured host patterns for navigation matching."""
    if not patterns:
        return ()
    normalized = []
    for pattern in patterns:
        value = str(pattern).strip().lower().lstrip(".")
        if value:
            normalized.append(value)
    return tuple(dict.fromkeys(normalized))


def host_matches(host: str | None, patterns: Sequence[str]) -> bool:
    """Whether ``host`` equals or is a subdomain of any configured pattern."""
    if not host or not patterns:
        return False
    candidate = host.strip().lower().rstrip(".")
    return any(
        candidate == pattern or candidate.endswith(f".{pattern}")
        for pattern in patterns
    )


def navigation_block_reason(
    raw_url: str,
    *,
    allowlist: Sequence[str] | None = None,
    denylist: Sequence[str] | None = None,
) -> str | None:
    """Return why a browser URL is forbidden, or ``None`` when permitted."""
    url = raw_url.strip()
    if url == "about:blank":
        return None
    host = urlsplit(url).hostname
    if host is None:
        # Workspace file navigation is contained by the workspace. Host policy
        # only governs network destinations.
        return None
    normalized_allowlist = normalize_host_patterns(allowlist)
    normalized_denylist = normalize_host_patterns(denylist)
    if host_matches(host, normalized_denylist):
        return f"Navigation to {host} is blocked by the configured policy."
    if normalized_allowlist and not host_matches(host, normalized_allowlist):
        return (
            f"Navigation to {host} is outside the configured allowlist. "
            "Ask the user to open this site or extend the allowlist."
        )
    return None


class DefaultComputerActionPolicy:
    """Conservative browser policy for actions with external side effects.

    The policy deliberately fails closed: whenever the page structure is
    unknown, or the action's target cannot be resolved in the current frame, it
    asks the user instead of assuming the action is harmless.
    """

    def __init__(
        self,
        *,
        navigation_allowlist: Sequence[str] | None = None,
        navigation_denylist: Sequence[str] | None = None,
    ) -> None:
        self.navigation_allowlist = normalize_host_patterns(navigation_allowlist)
        self.navigation_denylist = normalize_host_patterns(navigation_denylist)

    async def evaluate(
        self,
        batch: ComputerActionBatch,
        observation: ComputerObservation,
    ) -> ComputerPolicyDecision:
        confirmation_indexes: list[int] = []
        blocked_indexes: list[int] = []
        confirmation_reasons: list[str] = []
        blocked_reasons: list[str] = []
        structure_unknown = (
            observation.metadata.get(ELEMENT_EXTRACTION_FAILED_KEY) is True
        )
        structure_incomplete = (
            observation.metadata.get(ELEMENT_EXTRACTION_INCOMPLETE_KEY) is True
        )

        for index, action in enumerate(batch.actions):
            element = find_computer_target_element(action, observation)

            if action.type is ComputerActionType.CAPTURE_MEDIA:
                confirmation_indexes.append(index)
                confirmation_reasons.append(
                    "Capturing audio or video records content from the authorized "
                    "computer target and creates a downloadable file."
                )
                continue

            if action.type is ComputerActionType.NAVIGATE:
                blocked_reason = self._blocked_navigation_reason(action.url or "")
                if blocked_reason is not None:
                    blocked_indexes.append(index)
                    blocked_reasons.append(blocked_reason)
                continue

            active_url_reason = self._blocked_navigation_reason(
                observation.active_url or ""
            )
            if action.type not in _READ_ONLY_ACTIONS and active_url_reason is not None:
                blocked_indexes.append(index)
                blocked_reasons.append(
                    f"The current page is outside the allowed browser boundary. "
                    f"{active_url_reason}"
                )
                continue

            if (
                action.type
                in {
                    ComputerActionType.TYPE,
                    ComputerActionType.REPLACE_TEXT,
                }
                and element is None
                and (
                    structure_unknown
                    or structure_incomplete
                    or not observation.elements
                )
            ):
                blocked_indexes.append(index)
                blocked_reasons.append(
                    "The input target cannot be inspected, so Xagent cannot "
                    "verify that it is not a password, payment, or one-time-code "
                    "field. The user must enter the text."
                )
                continue

            if action.type not in _READ_ONLY_ACTIONS and structure_unknown:
                confirmation_indexes.append(index)
                confirmation_reasons.append(
                    "The page structure could not be read, so the effect of "
                    "this action cannot be checked in advance."
                )
                continue

            if (
                action.type
                in {
                    ComputerActionType.TYPE,
                    ComputerActionType.REPLACE_TEXT,
                }
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

            if action.type in _POINTED_ACTIONS and element is None:
                confirmation_indexes.append(index)
                if structure_incomplete:
                    confirmation_reasons.append(
                        "Some page surfaces could not be inspected and the "
                        "action matches no verified control."
                    )
                else:
                    confirmation_reasons.append(
                        "The action targets a position that matches no known "
                        "control, so what it activates cannot be verified."
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

    def _blocked_navigation_reason(self, raw_url: str) -> str | None:
        """Return why navigation is refused, or None when it is permitted."""
        return navigation_block_reason(
            raw_url,
            allowlist=self.navigation_allowlist,
            denylist=self.navigation_denylist,
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
