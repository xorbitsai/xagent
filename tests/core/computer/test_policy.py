from __future__ import annotations

import pytest

from xagent.core.computer.policy import (
    ComputerPolicyOutcome,
    ComputerRiskLevel,
    DefaultComputerActionPolicy,
)
from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerMediaKind,
    ComputerObservation,
    ComputerTarget,
    NormalizedPoint,
    NormalizedRect,
    Viewport,
)
from xagent.core.context_ref import (
    ContextReference,
    ContextReferencePurpose,
)


def _element(
    *,
    element_id: str = "dom-1",
    label: str = "Continue",
    sensitive: bool = False,
    focused: bool = False,
) -> ComputerElement:
    return ComputerElement(
        element_id=element_id,
        source=ComputerElementSource.DOM,
        bounds=NormalizedRect(x=0.1, y=0.1, width=0.3, height=0.1),
        label=label,
        role="button",
        metadata={"sensitive": sensitive, "focused": focused},
    )


def _observation(*elements: ComputerElement) -> ComputerObservation:
    return ComputerObservation(
        session_id="session-1",
        frame_id="frame-1",
        environment=ComputerEnvironmentType.BROWSER,
        viewport=Viewport(width=1280, height=720),
        screenshot=ContextReference(
            file_ref={
                "file_id": "image-1",
                "filename": "frame-1.png",
                "mime_type": "image/png",
            },
            purpose=ContextReferencePurpose.OBSERVATION,
            frame_id="frame-1",
            metadata={"sha256": "same-page"},
        ),
        elements=list(elements),
        active_url="https://example.com/settings",
    )


def _batch(action: ComputerAction) -> ComputerActionBatch:
    return ComputerActionBatch(
        session_id="session-1",
        expected_frame_id="frame-1",
        actions=[action],
    )


@pytest.mark.asyncio
async def test_policy_allows_low_risk_navigation_and_scrolling() -> None:
    policy = DefaultComputerActionPolicy()
    observation = _observation()

    navigate = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.NAVIGATE,
                url="https://example.com",
            )
        ),
        observation,
    )
    scroll = await policy.evaluate(
        _batch(ComputerAction(type=ComputerActionType.SCROLL, delta_y=0.5)),
        observation,
    )

    assert navigate.outcome is ComputerPolicyOutcome.ALLOW
    assert navigate.risk is ComputerRiskLevel.LOW
    assert scroll.outcome is ComputerPolicyOutcome.ALLOW


@pytest.mark.asyncio
async def test_policy_requires_confirmation_for_high_impact_click() -> None:
    policy = DefaultComputerActionPolicy()
    observation = _observation(_element(label="Place order"))

    decision = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(element_id="dom-1"),
            )
        ),
        observation,
    )

    assert decision.outcome is ComputerPolicyOutcome.REQUIRE_CONFIRMATION
    assert decision.risk is ComputerRiskLevel.ELEVATED
    assert decision.action_indexes == [0]


@pytest.mark.asyncio
async def test_policy_resolves_high_impact_element_under_point() -> None:
    policy = DefaultComputerActionPolicy()
    observation = _observation(_element(label="Delete account"))

    decision = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(point=NormalizedPoint(x=0.2, y=0.15)),
            )
        ),
        observation,
    )

    assert decision.outcome is ComputerPolicyOutcome.REQUIRE_CONFIRMATION


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        ComputerAction(type=ComputerActionType.KEYPRESS, keys=["ENTER"]),
        ComputerAction(
            type=ComputerActionType.DRAG,
            start=NormalizedPoint(x=0.1, y=0.1),
            end=NormalizedPoint(x=0.9, y=0.9),
        ),
    ],
)
async def test_policy_requires_confirmation_for_submission_capable_actions(
    action: ComputerAction,
) -> None:
    decision = await DefaultComputerActionPolicy().evaluate(
        _batch(action),
        _observation(),
    )

    assert decision.outcome is ComputerPolicyOutcome.REQUIRE_CONFIRMATION
    assert decision.risk is ComputerRiskLevel.ELEVATED


@pytest.mark.asyncio
async def test_policy_requires_confirmation_for_media_capture() -> None:
    decision = await DefaultComputerActionPolicy().evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.CAPTURE_MEDIA,
                media_kind=ComputerMediaKind.VIDEO,
                duration_ms=5_000,
            )
        ),
        _observation(),
    )

    assert decision.outcome is ComputerPolicyOutcome.REQUIRE_CONFIRMATION
    assert decision.risk is ComputerRiskLevel.ELEVATED
    assert "records" in decision.reason


@pytest.mark.asyncio
async def test_policy_blocks_typing_into_focused_sensitive_input() -> None:
    observation = _observation(
        _element(
            label="Sensitive input",
            sensitive=True,
            focused=True,
        )
    )

    decision = await DefaultComputerActionPolicy().evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.TYPE,
                text="secret-value",
            )
        ),
        observation,
    )

    assert decision.outcome is ComputerPolicyOutcome.BLOCK
    assert decision.risk is ComputerRiskLevel.HIGH
    assert "must be entered by the user" in decision.reason
