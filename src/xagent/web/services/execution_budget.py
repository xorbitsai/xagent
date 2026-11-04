"""Default budget policy and the application policy-provider extension point."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...config import get_execution_budget_defaults
from ...core.agent.budget import ExecutionBudgetPolicy
from ...core.execution_scope import ExecutionScope
from ..models.database import get_session_local
from ..models.system_setting import SystemSetting
from ..models.user import User
from .db_runtime import run_db_io_cancellation_safe

BUDGET_SETTINGS_KEY = "execution_budget"


class ExecutionBudgetDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    default_tokens: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    soft_limit_percent: int = Field(default=80, ge=1, le=99)

    @model_validator(mode="after")
    def validate_default_within_maximum(self) -> ExecutionBudgetDefaults:
        if (
            self.default_tokens is not None
            and self.max_tokens is not None
            and self.default_tokens > self.max_tokens
        ):
            raise ValueError("default_tokens must not exceed max_tokens")
        return self


@dataclass(frozen=True)
class BudgetPolicyRequest:
    """Trusted execution identity, supplied by the host rather than task text."""

    user_id: int
    task_id: str | None = None
    scope: ExecutionScope | None = None


BudgetPolicyResolver = Callable[
    [BudgetPolicyRequest, ExecutionBudgetPolicy],
    ExecutionBudgetPolicy | Awaitable[ExecutionBudgetPolicy],
]
_policy_resolver: BudgetPolicyResolver | None = None


def set_execution_budget_policy_resolver(resolver: BudgetPolicyResolver | None) -> None:
    """Install a policy resolver; existing quota gates remain independent.

    Called once per start/resume on the event loop. Async resolvers are
    supported. A resolution failure prevents execution rather than granting
    an unlimited allowance. Register the resolver in every worker process.
    """
    global _policy_resolver
    _policy_resolver = resolver


def load_budget_defaults(db: Any) -> ExecutionBudgetDefaults:
    """Saved administrator settings replace environment defaults as a whole."""
    row = (
        db.query(SystemSetting).filter(SystemSetting.key == BUDGET_SETTINGS_KEY).first()
    )
    if row is not None:
        return ExecutionBudgetDefaults.model_validate_json(row.value)
    return ExecutionBudgetDefaults.model_validate(get_execution_budget_defaults())


def resolve_default_policy(
    defaults: ExecutionBudgetDefaults, preferences: dict[str, Any]
) -> ExecutionBudgetPolicy:
    personal_limit = preferences.get("execution_budget_tokens")
    personal_percent = preferences.get("execution_budget_soft_percent")
    limit = personal_limit if personal_limit is not None else defaults.default_tokens
    source = "personal" if personal_limit is not None else "system"
    if defaults.max_tokens is not None and (
        limit is None or limit > defaults.max_tokens
    ):
        limit = defaults.max_tokens
        source = "system_maximum"
    return ExecutionBudgetPolicy(
        max_tokens=limit,
        soft_limit_percent=(
            personal_percent
            if personal_percent is not None
            else defaults.soft_limit_percent
        ),
        source=source,
        soft_limit_source="personal" if personal_percent is not None else "system",
    )


def _load_user_budget(user_id: int) -> tuple[ExecutionBudgetDefaults, dict[str, Any]]:
    with get_session_local()() as db:
        user = db.get(User, user_id)
        if user is None:
            raise ValueError("Execution budget owner no longer exists")
        return load_budget_defaults(db), dict(user.preferences or {})


async def resolve_execution_budget_policy(
    request: BudgetPolicyRequest,
) -> ExecutionBudgetPolicy:
    defaults, preferences = await run_db_io_cancellation_safe(
        lambda: _load_user_budget(request.user_id)
    )
    policy = resolve_default_policy(defaults, preferences)
    if _policy_resolver is not None:
        resolved = _policy_resolver(request, policy)
        if inspect.isawaitable(resolved):
            resolved = await resolved
        policy = ExecutionBudgetPolicy.model_validate(resolved)
        # An extension may replace defaults but cannot relax a deployment cap.
        if defaults.max_tokens is not None and (
            policy.max_tokens is None or policy.max_tokens > defaults.max_tokens
        ):
            policy = policy.model_copy(
                update={"max_tokens": defaults.max_tokens, "source": "system_maximum"}
            )
    return policy


def budget_policy_provider(
    *, user_id: int, task_id: str | None, scope: ExecutionScope | None
) -> Callable[[], Awaitable[ExecutionBudgetPolicy]]:
    request = BudgetPolicyRequest(user_id=user_id, task_id=task_id, scope=scope)
    return lambda: resolve_execution_budget_policy(request)
