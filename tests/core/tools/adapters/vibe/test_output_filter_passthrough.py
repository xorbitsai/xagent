"""Runtime capabilities that must survive the output-filter wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.tools.adapters.vibe.output_filter_wrapper import (
    OutputFilteredToolWrapper,
)


def _wrap(target: Any) -> OutputFilteredToolWrapper:
    return OutputFilteredToolWrapper(
        target_tool=target,
        max_chars=1_000,
        max_fields=50,
        max_recursion=5,
    )


def test_optional_capabilities_reach_through_the_wrapper() -> None:
    class ResumableTool:
        name = "resumable"
        description = "Accepts a user response."
        tags: list[str] = []
        decision_group = "interactive"

        def __init__(self) -> None:
            self.responses: list[dict[str, str]] = []

        def resume_user_interaction(
            self,
            *,
            interaction_id: str,
            response: str,
        ) -> None:
            self.responses.append(
                {"interaction_id": interaction_id, "response": response}
            )

    target = ResumableTool()
    wrapper = _wrap(target)

    resume = getattr(wrapper, "resume_user_interaction", None)
    assert callable(resume)
    resume(interaction_id="interaction-1", response="Continue")

    assert target.responses == [
        {"interaction_id": "interaction-1", "response": "Continue"}
    ]
    assert wrapper.decision_group == "interactive"


def test_absent_capabilities_stay_absent() -> None:
    class PlainTool:
        name = "plain"
        description = "No optional capabilities."
        tags: list[str] = []

    wrapper = _wrap(PlainTool())

    assert getattr(wrapper, "resume_user_interaction", None) is None
    with pytest.raises(AttributeError):
        wrapper.resume_user_interaction  # noqa: B018


@pytest.mark.asyncio
async def test_waiting_control_envelope_survives_field_filtering() -> None:
    class WaitingTool:
        name = "waiting"
        description = "Returns an interaction after unrelated output."
        tags: list[str] = []

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "large_unrelated_field": "x" * 100,
                "status": "waiting_for_user",
                "interaction_id": "14ee67d1-d18e-47e0-a8f8-b28ef31262f5",
                "message": "Provide a value.",
                "message_type": "question",
                "interactions": [
                    {
                        "type": "file_upload",
                        "field": "exact-routing-field",
                        "label": "Choose the desired operation",
                        "accept": ["video/mp4", "audio/mpeg"],
                        "multiple": False,
                        "multiline": False,
                        "options": [
                            {
                                "label": "Approve this operation",
                                "value": "exact-routing-option-value",
                            },
                            {
                                "label": "Reject this operation",
                                "value": "second-routing-option-value",
                            },
                        ],
                    },
                    {
                        "type": "confirm",
                        "field": "second-routing-field",
                        "label": "Continue to the next step?",
                        "default": False,
                    },
                ],
            }

    wrapper = OutputFilteredToolWrapper(
        target_tool=WaitingTool(),
        max_chars=8,
        max_fields=1,
        max_recursion=3,
    )

    result = await wrapper.run_json_async({})

    assert result["status"] == "waiting_for_user"
    assert result["interaction_id"] == "14ee67d1-d18e-47e0-a8f8-b28ef31262f5"
    assert result["message"].startswith("Provide ")
    assert result["message_type"] == "question"
    assert len(result["interactions"]) == 2
    assert result["interactions"][0]["type"] == "file_upload"
    assert result["interactions"][0]["field"] == "exact-routing-field"
    assert result["interactions"][0]["label"].startswith("Choose t")
    assert result["interactions"][0]["accept"] == ["video/mp4", "audio/mpeg"]
    assert result["interactions"][0]["multiple"] is False
    assert result["interactions"][0]["multiline"] is False
    assert len(result["interactions"][0]["options"]) == 2
    assert result["interactions"][0]["options"][0]["value"] == (
        "exact-routing-option-value"
    )
    assert result["interactions"][0]["options"][0]["label"].startswith("Approve ")
    assert result["interactions"][0]["options"][1]["value"] == (
        "second-routing-option-value"
    )
    assert result["interactions"][1]["field"] == "second-routing-field"
    assert result["interactions"][1]["default"] is False


@pytest.mark.asyncio
async def test_classified_failure_survives_field_filtering() -> None:
    """A classified tool failure must keep its classification keys.

    Mirrors the waiting-envelope restore: field-count filtering could
    otherwise drop ``failure_code``/``status``/``is_error`` behind a
    "truncated" placeholder, silently turning a classified failure back into
    an opaque result the parent classifier can no longer recognize.
    """

    class FailingTool:
        name = "classified-failure"
        description = "Returns a classified nested-wait failure."
        tags: list[str] = []

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "success": False,
                "is_error": True,
                "status": "error",
                "failure_code": "unsupported_nested_interaction",
                "error": "Nested agent calls cannot forward interactive prompts.",
                "output": "Nested agent calls cannot forward interactive prompts.",
                "response": "Nested agent calls cannot forward interactive prompts.",
            }

    wrapper = OutputFilteredToolWrapper(
        target_tool=FailingTool(),
        max_chars=1_000,
        max_fields=2,
        max_recursion=3,
    )

    result = await wrapper.run_json_async({})

    assert result["success"] is False
    assert result["is_error"] is True
    assert result["status"] == "error"
    assert result["failure_code"] == "unsupported_nested_interaction"
    assert result["error"] == "Nested agent calls cannot forward interactive prompts."
    assert result["output"] == "Nested agent calls cannot forward interactive prompts."
    assert (
        result["response"] == "Nested agent calls cannot forward interactive prompts."
    )


@pytest.mark.asyncio
async def test_classified_failure_restore_truncates_oversized_values() -> None:
    """The classified-failure restore must still enforce the char budget.

    Mirrors ``test_classified_failure_survives_field_filtering`` (same
    ``max_fields=2``, so ordinary field-count filtering already collapses
    ``error``/``output``/``response`` behind a "truncated" placeholder before
    the classified-failure restore puts them back). That test's fields are
    short enough to survive ``max_chars`` untouched, which would hide a
    restore loop that copies the raw child value straight through instead of
    re-running it through ``self._filter.filter``. This test's fields are
    each 5000 characters against a ``max_chars=1_000`` budget, so the restore
    loop's own truncation is what has to hold.
    """
    from xagent.core.tools.adapters.vibe.output_filter import (
        DEFAULT_TRUNCATION_MESSAGE,
    )

    oversized = "x" * 5_000

    class OversizedFailingTool:
        name = "oversized-classified-failure"
        description = "Returns a classified failure with oversized text fields."
        tags: list[str] = []

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "success": False,
                "is_error": True,
                "status": "error",
                "failure_code": "unsupported_nested_interaction",
                "error": oversized,
                "output": oversized,
                "response": oversized,
            }

    max_chars = 1_000
    wrapper = OutputFilteredToolWrapper(
        target_tool=OversizedFailingTool(),
        max_chars=max_chars,
        max_fields=2,
        max_recursion=3,
    )

    result = await wrapper.run_json_async({})

    assert result["success"] is False
    assert result["is_error"] is True
    assert result["status"] == "error"
    assert result["failure_code"] == "unsupported_nested_interaction"
    for key in ("error", "output", "response"):
        assert result[key] != oversized
        assert len(result[key]) <= max_chars + len(DEFAULT_TRUNCATION_MESSAGE)


@pytest.mark.asyncio
async def test_unavailable_mcp_failure_keeps_content_and_reason() -> None:
    """The classified-failure restore is additive; it strips nothing.

    ``_run_unavailable`` is the one MCP builder that carries both ``success``
    and ``is_error``, so it takes the classified branch. Its user-facing
    ``content`` list and its ``reason`` must survive that branch unchanged.
    """
    from xagent.core.tools.adapters.vibe.mcp_adapter import UnavailableMCPTool

    tool = UnavailableMCPTool(
        server_name="github",
        server_id=7,
        failure_code="oauth_token_required",
        reason="oauth_token_required",
    )
    result = await _wrap(tool).run_json_async({})

    assert result["failure_code"] == "oauth_token_required"
    assert result["reason"] == "oauth_token_required"
    assert isinstance(result["content"], list) and result["content"]
    assert "MCP server credentials are unavailable" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_unavailable_mcp_failure_restores_classification_under_truncation() -> (
    None
):
    """Aggressive field-count truncation must not cost the classification.

    At ``max_fields=4`` ordinary recursive filtering alone would already
    drop ``failure_code``/``is_error``/``status`` behind a "truncated"
    placeholder; the classified-failure restore is what keeps them, and this
    pins that under the exact shape ``_run_unavailable`` produces.
    """
    from xagent.core.tools.adapters.vibe.mcp_adapter import UnavailableMCPTool

    tool = UnavailableMCPTool(
        server_name="github",
        server_id=7,
        failure_code="oauth_token_required",
        reason="oauth_token_required",
    )
    wrapper = OutputFilteredToolWrapper(
        target_tool=tool,
        max_chars=1_000,
        max_fields=4,
        max_recursion=5,
    )

    result = await wrapper.run_json_async({})

    assert result["failure_code"] == "oauth_token_required"
    assert result["is_error"] is True
    assert result["status"] == "error"
    assert result["content"] is not None


@pytest.mark.asyncio
async def test_classified_failure_restore_rejects_malformed_envelope() -> None:
    """The classified-failure restore must not let a tool smuggle raw values.

    Any wrapped tool returning ``success=False``/``is_error=True`` takes the
    restore branch, so a misbehaving or compromised tool must not be able to
    write an oversized ``status`` or an arbitrary ``failure_code`` object
    past the filter through it. Both must fall back to whatever ordinary
    filtering already produced for them.
    """
    from xagent.core.tools.adapters.vibe.output_filter import (
        DEFAULT_TRUNCATION_MESSAGE,
    )

    class _Unserializable:
        def __str__(self) -> str:
            return "unserializable-failure-code"

    bad_failure_code = _Unserializable()
    oversized_status = "x" * 100_000

    class MalformedTool:
        name = "malformed"
        description = "Returns a malformed classified failure envelope."
        tags: list[str] = []

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "success": False,
                "is_error": True,
                "status": oversized_status,
                "failure_code": bad_failure_code,
                "error": "boom",
            }

    result = await _wrap(MalformedTool()).run_json_async({})

    assert result["success"] is False
    assert result["is_error"] is True
    assert result["status"] != oversized_status
    assert len(result["status"]) <= 1_000 + len(DEFAULT_TRUNCATION_MESSAGE)
    assert result["failure_code"] is not bad_failure_code
    assert isinstance(result["failure_code"], str)

    class _LyingStatus(str):
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return hash(str(self))

    lying_status = _LyingStatus("x" * 100_000)

    class LyingStatusTool:
        name = "lying-status"
        description = "Returns a str-subclass status that lies about equality."
        tags: list[str] = []

        async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
            return {
                "success": False,
                "is_error": True,
                "status": lying_status,
                "error": "boom",
            }

    lying_result = await _wrap(LyingStatusTool()).run_json_async({})

    assert lying_result["success"] is False
    assert lying_result["is_error"] is True
    assert lying_result["status"] is not lying_status
    assert len(lying_result["status"]) <= 1_000 + len(DEFAULT_TRUNCATION_MESSAGE)


@pytest.mark.asyncio
async def test_teardown_forwards_execution_status_when_supported() -> None:
    class StatusAwareTool:
        name = "status-aware"
        description = "Records teardown."
        tags: list[str] = []

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def teardown(
            self,
            task_id: str | None = None,
            execution_status: str | None = None,
        ) -> None:
            self.calls.append(
                {"task_id": task_id, "execution_status": execution_status}
            )

    target = StatusAwareTool()

    await _wrap(target).teardown(
        task_id="task-1",
        execution_status="waiting_for_user",
    )

    assert target.calls == [
        {"task_id": "task-1", "execution_status": "waiting_for_user"}
    ]


@pytest.mark.asyncio
async def test_teardown_omits_status_for_legacy_tool() -> None:
    class LegacyTool:
        name = "legacy"
        description = "Legacy teardown signature."
        tags: list[str] = []

        def __init__(self) -> None:
            self.calls: list[str | None] = []

        async def teardown(self, task_id: str | None = None) -> None:
            self.calls.append(task_id)

    target = LegacyTool()

    await _wrap(target).teardown(
        task_id="task-2",
        execution_status="completed",
    )

    assert target.calls == ["task-2"]
