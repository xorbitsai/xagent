"""The action policy must fail closed when it cannot see what it is acting on."""

from __future__ import annotations

import pytest

from xagent.core.computer.policy import (
    ComputerPolicyOutcome,
    DefaultComputerActionPolicy,
    host_matches,
)
from xagent.core.computer.schema import (
    ELEMENT_EXTRACTION_FAILED_KEY,
    ELEMENT_EXTRACTION_INCOMPLETE_KEY,
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerObservation,
    ComputerTarget,
    NormalizedPoint,
    NormalizedRect,
    Viewport,
)
from xagent.core.context_ref import ContextReference, ContextReferencePurpose


def _observation(
    *,
    elements: list[ComputerElement] | None = None,
    metadata: dict | None = None,
) -> ComputerObservation:
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
        ),
        elements=list(elements or []),
        active_url="https://example.com/page",
        metadata=dict(metadata or {}),
    )


def _batch(action: ComputerAction) -> ComputerActionBatch:
    return ComputerActionBatch(
        session_id="session-1",
        expected_frame_id="frame-1",
        actions=[action],
    )


def _plain_button() -> ComputerElement:
    return ComputerElement(
        element_id="dom-1",
        source=ComputerElementSource.DOM,
        bounds=NormalizedRect(x=0.1, y=0.1, width=0.2, height=0.1),
        label="Show details",
        role="button",
    )


@pytest.mark.asyncio
async def test_unknown_page_structure_requires_confirmation() -> None:
    policy = DefaultComputerActionPolicy()
    observation = _observation(metadata={ELEMENT_EXTRACTION_FAILED_KEY: True})

    decision = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(point=NormalizedPoint(x=0.5, y=0.5)),
            )
        ),
        observation,
    )

    assert decision.outcome is ComputerPolicyOutcome.REQUIRE_CONFIRMATION
    assert "page structure" in decision.reason


@pytest.mark.asyncio
async def test_screenshot_is_allowed_even_without_page_structure() -> None:
    policy = DefaultComputerActionPolicy()
    observation = _observation(metadata={ELEMENT_EXTRACTION_FAILED_KEY: True})

    decision = await policy.evaluate(
        _batch(ComputerAction(type=ComputerActionType.SCREENSHOT)),
        observation,
    )

    assert decision.outcome is ComputerPolicyOutcome.ALLOW


@pytest.mark.asyncio
async def test_click_missing_every_known_control_requires_confirmation() -> None:
    policy = DefaultComputerActionPolicy()
    observation = _observation(elements=[_plain_button()])

    decision = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(point=NormalizedPoint(x=0.9, y=0.9)),
            )
        ),
        observation,
    )

    assert decision.outcome is ComputerPolicyOutcome.REQUIRE_CONFIRMATION
    assert "matches no known" in decision.reason


@pytest.mark.asyncio
async def test_empty_element_list_does_not_make_point_actions_low_risk() -> None:
    policy = DefaultComputerActionPolicy()

    click = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(point=NormalizedPoint(x=0.5, y=0.5)),
            )
        ),
        _observation(),
    )
    typed = await policy.evaluate(
        _batch(ComputerAction(type=ComputerActionType.TYPE, text="hello")),
        _observation(),
    )

    assert click.outcome is ComputerPolicyOutcome.REQUIRE_CONFIRMATION
    assert typed.outcome is ComputerPolicyOutcome.BLOCK
    assert "user must enter" in typed.reason


@pytest.mark.asyncio
async def test_incomplete_page_requires_confirmation_for_unresolved_target() -> None:
    policy = DefaultComputerActionPolicy()
    observation = _observation(
        elements=[_plain_button()],
        metadata={ELEMENT_EXTRACTION_INCOMPLETE_KEY: True},
    )

    decision = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(point=NormalizedPoint(x=0.9, y=0.9)),
            )
        ),
        observation,
    )

    assert decision.outcome is ComputerPolicyOutcome.REQUIRE_CONFIRMATION
    assert "could not be inspected" in decision.reason


@pytest.mark.asyncio
async def test_click_on_a_known_harmless_control_is_allowed() -> None:
    policy = DefaultComputerActionPolicy()
    observation = _observation(elements=[_plain_button()])

    decision = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(element_id="dom-1"),
            )
        ),
        observation,
    )

    assert decision.outcome is ComputerPolicyOutcome.ALLOW


@pytest.mark.asyncio
async def test_navigation_denylist_blocks_the_host() -> None:
    policy = DefaultComputerActionPolicy(navigation_denylist=["admin.example.com"])

    decision = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.NAVIGATE,
                url="https://admin.example.com/users",
            )
        ),
        _observation(),
    )

    assert decision.outcome is ComputerPolicyOutcome.BLOCK
    assert "admin.example.com" in decision.reason


@pytest.mark.asyncio
async def test_navigation_allowlist_blocks_everything_else() -> None:
    policy = DefaultComputerActionPolicy(navigation_allowlist=["example.com"])

    allowed = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.NAVIGATE,
                url="https://docs.example.com/guide",
            )
        ),
        _observation(),
    )
    blocked = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.NAVIGATE,
                url="https://elsewhere.test/",
            )
        ),
        _observation(),
    )

    assert allowed.outcome is ComputerPolicyOutcome.ALLOW
    assert blocked.outcome is ComputerPolicyOutcome.BLOCK


@pytest.mark.asyncio
async def test_workspace_file_navigation_ignores_host_policy() -> None:
    policy = DefaultComputerActionPolicy(navigation_allowlist=["example.com"])

    decision = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.NAVIGATE,
                url="report.html",
            )
        ),
        _observation(),
    )

    assert decision.outcome is ComputerPolicyOutcome.ALLOW


@pytest.mark.asyncio
async def test_mutation_on_current_disallowed_host_is_blocked() -> None:
    policy = DefaultComputerActionPolicy(navigation_allowlist=["example.com"])
    observation = _observation(elements=[_plain_button()])
    observation = observation.model_copy(
        update={"active_url": "https://outside.test/account"}
    )

    decision = await policy.evaluate(
        _batch(
            ComputerAction(
                type=ComputerActionType.CLICK,
                target=ComputerTarget(element_id="dom-1"),
            )
        ),
        observation,
    )

    assert decision.outcome is ComputerPolicyOutcome.BLOCK
    assert "current page" in decision.reason


def test_host_matching_covers_subdomains_only() -> None:
    assert host_matches("example.com", ["example.com"]) is True
    assert host_matches("docs.example.com", ["example.com"]) is True
    assert host_matches("notexample.com", ["example.com"]) is False
    assert host_matches("example.com.evil.test", ["example.com"]) is False
