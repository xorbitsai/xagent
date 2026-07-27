"""Generic contract for tools that need a user response before continuing.

Tools opt into the control flow by returning a mapping whose ``status`` is
``waiting_for_user``.  The execution pattern presents the returned message,
checkpoints, and stops the current run. After the user replies, the runtime asks
the model to replan with the annotated response in context. Tools that keep
server-owned interaction state may additionally implement
``resume_user_interaction``; the runtime then delivers the reply to that exact
suspended interaction before replanning.

The optional runtime capability keeps server-owned interaction state out of
model-generated tool arguments and lets those tools decide how to interpret the
user's answer. Callback-less tools use the normal ReAct context and replan path.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

WAITING_FOR_USER_STATUS = "waiting_for_user"


@runtime_checkable
class ResumableUserInteractionTool(Protocol):
    """Optional capability implemented by tools with resumable interactions."""

    def resume_user_interaction(
        self,
        *,
        interaction_id: str,
        response: str,
    ) -> Any:
        """Accept one user response identified by the suspended interaction."""


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
