"""Resolved execution budgets, independent of accounts and policy storage."""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from time import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ...config import get_execution_budget_defaults


class ExecutionBudgetPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    max_tokens: int | None = Field(default=None, gt=0)
    soft_limit_percent: int = Field(default=80, ge=1, le=99)
    source: str = "system"
    soft_limit_source: str = "system"


BudgetPolicyProvider = Callable[
    [], ExecutionBudgetPolicy | Awaitable[ExecutionBudgetPolicy]
]


class ExecutionBudget(BaseModel):
    """Checkpointed allowance shared by all work in one execution turn.

    Counts reported input and output tokens, including cached input. Limits
    are checked at call boundaries, not a reservation of provider-side usage:
    already admitted concurrent calls may overshoot the limit.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    policy: ExecutionBudgetPolicy
    turn_id: str = ""
    used_tokens: int = Field(default=0, ge=0)
    soft_notified: bool = False
    closed: bool = False
    started_at: float = Field(default_factory=time)
    _warning_in_flight: bool = PrivateAttr(default=False)

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.used_tokens += max(0, input_tokens) + max(0, output_tokens)

    def checkpoint_state(self) -> dict[str, Any] | None:
        """Keep unlimited runs from adding budget-only checkpoint writes."""
        return self.model_dump() if self.policy.max_tokens is not None else None

    @property
    def exhausted(self) -> bool:
        limit = self.policy.max_tokens
        return limit is not None and self.used_tokens >= limit

    @property
    def soft_reached(self) -> bool:
        limit = self.policy.max_tokens
        return (
            limit is not None
            and self.used_tokens * 100 >= limit * self.policy.soft_limit_percent
        )

    def tighten(self, policy: ExecutionBudgetPolicy) -> None:
        """A resumed turn cannot gain allowance through a policy change."""
        limits = [
            value
            for value in (self.policy.max_tokens, policy.max_tokens)
            if value is not None
        ]
        self.policy = policy.model_copy(
            update={"max_tokens": min(limits) if limits else None}
        )

    def notice(self) -> str:
        return (
            "Execution token budget: the soft threshold has been reached. "
            "Prioritize handing over supported results and existing file links. "
            "Avoid starting optional work. State remaining gaps honestly."
        )

    async def notify(
        self, handler: Callable[["ExecutionBudget"], Awaitable[bool]]
    ) -> None:
        """Only consume the shared warning after delivery; serialize siblings."""
        if self.soft_notified or self._warning_in_flight:
            return
        self._warning_in_flight = True
        try:
            self.soft_notified = await handler(self)
        finally:
            self._warning_in_flight = False


active_execution_budget: ContextVar[ExecutionBudget | None] = ContextVar(
    "active_execution_budget", default=None
)
budget_warning_handler: ContextVar[
    Callable[[ExecutionBudget], Awaitable[bool]] | None
] = ContextVar("budget_warning_handler", default=None)


def default_execution_budget_policy() -> ExecutionBudgetPolicy:
    settings = get_execution_budget_defaults()
    limits = [
        value
        for key in ("default_tokens", "max_tokens")
        if (value := settings[key]) is not None
    ]
    return ExecutionBudgetPolicy(
        max_tokens=min(limits) if limits else None,
        soft_limit_percent=int(settings["soft_limit_percent"] or 80),
    )


def budget_llm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Give every pattern the same soft-limit instruction without editing history."""
    budget = active_execution_budget.get()
    if budget is None or not budget.soft_reached:
        return kwargs
    messages = list(kwargs.get("messages") or [])
    notice = budget.notice()
    if messages and messages[0].get("role") == "system":
        messages[0] = {
            **messages[0],
            "content": f"{messages[0].get('content', '')}\n\n{notice}",
        }
    else:
        messages.insert(0, {"role": "system", "content": notice})
    return {**kwargs, "messages": messages}
