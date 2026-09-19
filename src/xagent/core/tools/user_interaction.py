"""Generic contract for tools that need a user response before continuing.

Tools opt into the control flow by returning a mapping whose ``status`` is
``waiting_for_user``.  The execution pattern presents the returned message,
checkpoints, and stops the current run. After the user replies, tools that keep
server-owned interaction state may implement ``resume_user_interaction``; the
runtime then delivers the reply to that exact suspended interaction. Returning
``ToolInteractionSettlement`` projects the terminal result onto the original
tool call before the model continues. Returning ``None`` preserves the legacy
replan behavior.

The optional runtime capability keeps server-owned interaction state out of
model-generated tool arguments and lets those tools decide how to interpret the
user's answer. Callback-less tools use the normal ReAct context and replan path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, get_args, runtime_checkable

WAITING_FOR_USER_STATUS = "waiting_for_user"
ToolInteractionSettlementStatus = Literal[
    "succeeded",
    "rejected",
    "failed",
    "dispatch_unknown",
]
_TOOL_INTERACTION_SETTLEMENT_STATUSES = frozenset(
    get_args(ToolInteractionSettlementStatus)
)
_DEFAULT_SETTLEMENT_ERRORS = {
    "rejected": "The user rejected the tool call.",
    "failed": "The resumed tool call failed.",
    "dispatch_unknown": (
        "The tool call may have reached the external system. Automatic retry is "
        "disabled; verify the external system before trying again."
    ),
}


@dataclass(frozen=True)
class ToolInteractionSettlement:
    """Durable outcome returned when a suspended tool call is resumed.

    Returning this value asks the execution pattern to replace the original
    ``waiting_for_user`` observation with the terminal result. A callback that
    returns ``None`` keeps the legacy behavior: delivery is acknowledged and
    the model replans from the user's response.
    """

    status: ToolInteractionSettlementStatus
    result: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _TOOL_INTERACTION_SETTLEMENT_STATUSES:
            raise ValueError(f"Invalid tool interaction settlement: {self.status!r}")
        if self.status == "succeeded":
            if self.result is None:
                raise ValueError(
                    "A succeeded tool interaction settlement needs a result."
                )
            if tool_result_waits_for_user(self.result):
                raise ValueError(
                    "A succeeded tool interaction settlement cannot wait for user input."
                )
            # Deferred import avoids the agent.result -> tools import cycle.
            from ..agent.result import tool_result_succeeded

            if not tool_result_succeeded(self.result):
                raise ValueError(
                    "A succeeded tool interaction settlement cannot carry a "
                    "failed tool result."
                )
        elif tool_result_waits_for_user(self.result):
            raise ValueError(
                "A terminal tool interaction settlement cannot wait for user input."
            )

    @classmethod
    def succeeded(cls, result: Any) -> ToolInteractionSettlement:
        return cls(status="succeeded", result=result)

    @classmethod
    def rejected(
        cls,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> ToolInteractionSettlement:
        return cls(status="rejected", result=result, error=error)

    @classmethod
    def failed(
        cls,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> ToolInteractionSettlement:
        return cls(status="failed", result=result, error=error)

    @classmethod
    def dispatch_unknown(
        cls,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> ToolInteractionSettlement:
        return cls(status="dispatch_unknown", result=result, error=error)

    def projected_result(self) -> Any:
        """Return the model-visible result with an unambiguous outcome."""

        if self.status == "succeeded":
            return self.result

        result_error = (
            self.result.get("error") if isinstance(self.result, dict) else None
        )
        error = self.error or result_error or _DEFAULT_SETTLEMENT_ERRORS[self.status]
        if isinstance(self.result, dict):
            return {
                **self.result,
                "success": False,
                "settlement_status": self.status,
                "error": error,
            }
        projected = {
            "success": False,
            "settlement_status": self.status,
            "error": error,
        }
        if self.result is not None:
            projected["result"] = self.result
        return projected


@runtime_checkable
class ResumableUserInteractionTool(Protocol):
    """Optional capability implemented by tools with resumable interactions."""

    def resume_user_interaction(
        self,
        *,
        interaction_id: str,
        response: str,
    ) -> ToolInteractionSettlement | None:
        """Accept a response and optionally settle the suspended tool call.

        A host may invoke this callback again after a failed checkpoint or
        process restart. Implementations that perform external work must
        durably replay the same terminal settlement for the interaction.
        """


def tool_result_waits_for_user(result: Any) -> bool:
    """Return whether a tool result requests a user-interaction pause."""

    return (
        isinstance(result, dict)
        and str(result.get("status") or "").strip().lower() == WAITING_FOR_USER_STATUS
    )


def user_interaction_resume_callable(tool: Any) -> Callable[..., Any] | None:
    """Return a tool's optional user-interaction resume callback."""

    resume = getattr(tool, "resume_user_interaction", None)
    return resume if callable(resume) else None
